#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>  // Required for std::numeric_limits
#include <optional>
#include <queue>
#include <random>
#include <span>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"

// --- HNSW Metadata ---

typedef unsigned short int vl_type;

class VisitedList {
   public:
    vl_type curV;
    vl_type* mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList*> pool;
    std::mutex poolguard;
    int numelements;

   public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++) pool.push_front(new VisitedList(numelements));
    }

    VisitedList* getFreeVisitedList() {
        VisitedList* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList* vl) {
        std::unique_lock<std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList* rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};

struct HNSWMetadataPage {
    bool dirty;
    PID base_pid;                 // The starting Page ID for HNSW data pages.
    u64 max_elements;             // The maximum number of elements the index can hold.
    std::atomic<u64> node_count;  // The total number of nodes currently inserted.

    // Entry point information.
    u32 enter_point_node_id;     // The global entry point node ID for the graph.
    std::atomic<u32> max_level;  // The highest level currently present in the graph.
    static constexpr u32 invalid_node_id = std::numeric_limits<u32>::max();
};

// --- HNSW Main Class ---

template <size_t Dim,  // Vector dimension.
          typename DistanceMetric = hnsw_distance::SIMDAcceleratedL2>
class HNSW {
   public:
    // Fixed-size node layout (no more slots/indirection):
    // Each node has: [Vector Data][Level Counts][Neighbor IDs]
    // Node size is constant and computed at runtime

    // --- Runtime parameters (formerly template parameters) ---
    const size_t M;               // Max number of neighbors for layers > 0
    const size_t M0;              // Max number of neighbors for layer 0 (computed as 2*M)
    const size_t efConstruction;  // Number of neighbors to search during index construction
    const size_t MaxLevel;        // Pre-estimated max level for random level generation

    // --- Runtime size calculations for fixed-size nodes ---
    const size_t MaxNeighborsHeaderSize;  // MaxLevel * sizeof(u16) - level counts
    const size_t MaxNeighborsListSize;    // (M0 + (MaxLevel - 1) * M) * sizeof(u32) - neighbor IDs
    static constexpr size_t VectorSize = Dim * sizeof(float);
    const size_t FixedNodeSize;  // VectorSize + MaxNeighborsHeaderSize + MaxNeighborsListSize

    // HNSW data page: uses fixed-size node grid layout for direct access.
    struct HNSWPage {
        bool dirty;
        u16 node_count;
        u16 padding[3];  // Align to 8 bytes

        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count) + sizeof(padding);

        // Get node data area (nodes start immediately after header)
        u8* getNodeData() { return reinterpret_cast<u8*>(this) + HeaderSize; }

        const u8* getNodeData() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }
    };

    const size_t MaxNodesPerPage;  // Runtime computed: (pageSize - HNSWPage::HeaderSize) / FixedNodeSize
    const size_t NodesPerPage;     // Same as MaxNodesPerPage

   private:
    PID metadata_pid;
    PID base_pid;
    u64 max_elements_;
    double mult_factor;
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

   public:
    // Constructor with runtime parameters
    HNSW(u64 max_elements, size_t M = 16, size_t ef_construction = 200);
    ~HNSW() { visited_list_pool_.reset(nullptr); }

   private:
    // Helper function to estimate MaxLevel from max_elements
    static size_t estimateMaxLevel(u64 max_elements, size_t M);

   public:
    // Add a point to the index.
    void addPoint(const float* point, u32& node_id_out);

    // Search for the K nearest neighbors for a single query.
    std::vector<std::pair<float, u32>> searchKnn(const float* query, size_t k, size_t ef_search_param);

    // Search for the K nearest neighbors for a batch of queries in parallel.
    std::vector<std::vector<std::pair<float, u32>>> searchKnn_parallel(std::span<const float> queries, size_t k,
                                                                       size_t ef_search_param, size_t num_threads = 0);
    // Add a batch of points to the index in parallel.
    void addPoint_parallel(std::span<const float> points, size_t num_threads = 0);

    HNSW(const HNSW&) = delete;
    HNSW& operator=(const HNSW&) = delete;
    HNSW(HNSW&&) = delete;
    HNSW& operator=(HNSW&&) = delete;
    PID getMetadataPid() const { return metadata_pid; }

    // Get buffer manager statistics
    u64 getBufferManagerAllocCount() const;

    // Print comprehensive index information
    std::string getIndexInfo() const;

   private:
    // --- Internal Helper Classes & Functions ---
    inline PID getNodePID(u32 node_id) const;
    inline u32 getNodeIndexInPage(u32 node_id) const;

    class NodeAccessor {
       private:
        const HNSWPage* page;
        u32 node_index;
        const HNSW* hnsw;

       public:
        NodeAccessor(const HNSWPage* p, u32 node_idx, const HNSW* hnsw_instance)
            : page(p), node_index(node_idx), hnsw(hnsw_instance) {}

        u32 getLevel() const {
            // Level is stored as the maximum non-zero count in the level counts array
            const u16* counts = getLevelCounts();
            u32 max_level = 0;
            for (u32 i = 0; i < hnsw->MaxLevel; ++i) {
                if (counts[i] > 0) max_level = i;
            }
            return max_level;
        }

        const float* getVector() const {
            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<const float*>(node_start);
        }

        static u32 getVectorOffset(HNSW* hnsw, u32 graph_node_id) {
            return HNSWPage::HeaderSize + hnsw->getNodeIndexInPage(graph_node_id) * hnsw->FixedNodeSize;
        }

        const u16* getLevelCounts() const {
            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<const u16*>(node_start + VectorSize);
        }

        std::span<const u32> getNeighbors(u32 level, const HNSW* hnsw_instance) const {
            const u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return {};

            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            const u32* neighbor_ids_start =
                reinterpret_cast<const u32*>(node_start + VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            const u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }
    };

    // Non-const version for modification
    class MutableNodeAccessor {
       private:
        HNSWPage* page;
        u32 node_index;
        const HNSW* hnsw;

       public:
        MutableNodeAccessor(HNSWPage* p, u32 node_idx, const HNSW* hnsw_instance)
            : page(p), node_index(node_idx), hnsw(hnsw_instance) {}

        const float* getVector() const {
            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<const float*>(node_start);
        }

        float* getVector() {
            u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<float*>(node_start);
        }

        u16* getLevelCounts() {
            u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<u16*>(node_start + VectorSize);
        }

        const u16* getLevelCounts() const {
            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            return reinterpret_cast<const u16*>(node_start + VectorSize);
        }

        std::span<const u32> getNeighbors(u32 level, const HNSW* hnsw_instance) const {
            const u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return {};

            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            const u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            const u32* neighbor_ids_start =
                reinterpret_cast<const u32*>(node_start + VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            const u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }

        void setNeighbors(u32 level, const std::vector<u32>& new_neighbors, const HNSW* hnsw_instance) {
            u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return;

            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            u32* neighbor_ids_start =
                reinterpret_cast<u32*>(node_start + VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            size_t M_level = (level == 0) ? hnsw_instance->M0 : hnsw_instance->M;

            size_t count_to_copy = std::min(new_neighbors.size(), M_level);
            std::copy(new_neighbors.begin(), new_neighbors.begin() + count_to_copy, level_start_ptr);
            counts_per_level[level] = count_to_copy;
            page->dirty = true;
        }

        bool addNeighbor(u32 level, u32 new_neighbor_id, const HNSW* hnsw_instance) {
            u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return false;

            size_t M_level = (level == 0) ? hnsw_instance->M0 : hnsw_instance->M;
            u16 current_count = counts_per_level[level];

            // Check if there is space
            if (current_count >= M_level) {
                return false;
            }

            // Calculate the memory offset to the start of the neighbor list for this level
            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            u8* node_start = page->getNodeData() + node_index * hnsw->FixedNodeSize;
            u32* neighbor_ids_start =
                reinterpret_cast<u32*>(node_start + VectorSize + hnsw_instance->MaxNeighborsHeaderSize);

            // Get the pointer to the exact location to append the new neighbor
            u32* append_location = neighbor_ids_start + level_neighbor_offset_count + current_count;

            // Write the new neighbor ID directly into the page
            *append_location = new_neighbor_id;

            // Increment the count for this level
            counts_per_level[level]++;
            page->dirty = true;

            return true;
        }
    };

    u32 getRandomLevel();
    std::vector<std::pair<float, u32>> searchLayer(const float* query, u32 entry_point_id, u32 level, size_t ef,
                                                   std::optional<std::pair<float, u32>> initial_entry_dist_pair);
    std::pair<float, u32> findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level,
                                                     float entry_point_dist);
    std::pair<float, u32> searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level);

    // --- Now takes the query vector to match hnswlib's heuristic ---
    std::vector<std::pair<float, u32>> selectNeighborsHeuristic(const float* query,
                                                                const std::vector<std::pair<float, u32>>& candidates,
                                                                size_t M_limit);

    void addPoint_internal(const float* point, u32 node_id);
};

#endif  // HNSW_HPP
