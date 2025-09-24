#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>  // Required for std::numeric_limits
#include <queue>
#include <random>
#include <span>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"

// --- HNSW Metadata ---

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
          typename DistanceMetric = hnsw_distance::SIMDAcceleratedL2,
          size_t M = 16,                // Max number of neighbors for layers > 0.
          size_t M0 = 32,               // Max number of neighbors for layer 0.
          size_t efConstruction = 200,  // Number of neighbors to search during index construction.
          size_t MaxLevel = 16          // Pre-estimated max level for random level generation.
          >
class HNSW {
   public:
    // Node slot within a page: a small, fixed-size metadata structure.
    struct HNSWNodeSlot {
        u16 vector_offset;     // Offset to the vector data in the page's heap area.
        u16 neighbors_offset;  // Offset to the neighbor list in the page's heap area.
        u32 level;             // The actual highest level of this node.
    };

    // --- Worst-case size calculation for a single node ---
    static constexpr size_t MaxNeighborsHeaderSize = MaxLevel * sizeof(u16);
    static constexpr size_t MaxNeighborsListSize = (M0 + (MaxLevel - 1) * M) * sizeof(PID);
    static constexpr size_t VectorSize = Dim * sizeof(float);
    static constexpr size_t MaxNodeHeapSize = VectorSize + MaxNeighborsHeaderSize + MaxNeighborsListSize;
    static constexpr size_t MaxNodeTotalSize = sizeof(HNSWNodeSlot) + MaxNodeHeapSize;

    // HNSW data page: uses a slot/heap layout for high space efficiency.
    struct HNSWPage {
        bool dirty;
        u16 node_count;
        u16 data_offset;
        u32 padding;

        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count) + sizeof(data_offset) + sizeof(padding);
        static constexpr size_t MaxNodesPerPage = (pageSize - HeaderSize) / MaxNodeTotalSize;
        static_assert(MaxNodesPerPage > 0, "Page size is too small to fit even one node in the worst case.");
        HNSWNodeSlot slots[MaxNodesPerPage];
    };

    static constexpr size_t NodesPerPage = HNSWPage::MaxNodesPerPage;

   private:
    PID metadata_pid;
    PID base_pid;
    u64 max_elements_;
    double mult_factor;

   public:
    // Constructor
    HNSW(u64 max_elements);

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

   private:
    // --- Internal Helper Classes & Functions ---
    PID getNodePID(u32 node_id) const;
    u32 getSlotIndexInPage(u32 node_id) const;

    class NodeAccessor {
       private:
        const HNSWPage* page;
        const HNSWNodeSlot* slot;

       public:
        NodeAccessor(const HNSWPage* p, u32 slot_index) : page(p), slot(&p->slots[slot_index]) {}
        u32 getLevel() const { return slot->level; }
        const float* getVector() const {
            return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(page) + slot->vector_offset);
        }

        std::span<const PID> getNeighbors(u32 level) const {
            const u8* base_ptr = reinterpret_cast<const u8*>(page) + slot->neighbors_offset;
            const u16* counts_per_level = reinterpret_cast<const u16*>(base_ptr);
            if (level >= MaxLevel || level > slot->level) return {};

            u32 pids_start_offset = MaxLevel * sizeof(u16);
            u32 level_pid_offset_count = 0;
            if (level > 0) {
                level_pid_offset_count += M0;
                for (u32 i = 1; i < level; ++i) {
                    level_pid_offset_count += M;
                }
            }

            const PID* level_start_ptr =
                reinterpret_cast<const PID*>(base_ptr + pids_start_offset) + level_pid_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }
    };

    // Non-const version for modification
    class MutableNodeAccessor {
       private:
        HNSWPage* page;
        const HNSWNodeSlot* slot;

       public:
        MutableNodeAccessor(HNSWPage* p, u32 slot_index) : page(p), slot(&p->slots[slot_index]) {}
        const float* getVector() const {
            return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(page) + slot->vector_offset);
        }
        std::span<const PID> getNeighbors(u32 level) const {
            const u8* base_ptr = reinterpret_cast<const u8*>(page) + slot->neighbors_offset;
            const u16* counts_per_level = reinterpret_cast<const u16*>(base_ptr);
            if (level >= MaxLevel || level > slot->level) return {};

            u32 pids_start_offset = MaxLevel * sizeof(u16);
            u32 level_pid_offset_count = 0;
            if (level > 0) {
                level_pid_offset_count += M0;
                for (u32 i = 1; i < level; ++i) {
                    level_pid_offset_count += M;
                }
            }

            const PID* level_start_ptr =
                reinterpret_cast<const PID*>(base_ptr + pids_start_offset) + level_pid_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }
        void setNeighbors(u32 level, const std::vector<PID>& new_neighbors) {
            u8* base_ptr = reinterpret_cast<u8*>(page) + slot->neighbors_offset;
            u16* counts_per_level = reinterpret_cast<u16*>(base_ptr);
            if (level >= MaxLevel) return;

            u32 pids_start_offset = MaxLevel * sizeof(u16);
            u32 level_pid_offset_count = 0;
            if (level > 0) {
                level_pid_offset_count += M0;
                for (u32 i = 1; i < level; ++i) {
                    level_pid_offset_count += M;
                }
            }

            PID* level_start_ptr = reinterpret_cast<PID*>(base_ptr + pids_start_offset) + level_pid_offset_count;
            size_t M_level = (level == 0) ? M0 : M;

            size_t count_to_copy = std::min(new_neighbors.size(), M_level);
            std::copy(new_neighbors.begin(), new_neighbors.begin() + count_to_copy, level_start_ptr);
            counts_per_level[level] = count_to_copy;
            page->dirty = true;
        }
    };

    u32 getRandomLevel();
    std::vector<std::pair<float, u32>> searchLayer(const float* query, u32 entry_point_id, u32 level, size_t ef);
    u32 searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level);

    // --- Now takes the query vector to match hnswlib's heuristic ---
    std::vector<std::pair<float, u32>> selectNeighborsHeuristic(const float* query,
                                                                const std::vector<std::pair<float, u32>>& candidates,
                                                                size_t M_limit);

    void addPoint_internal(const float* point, u32 node_id);
};

#endif  // HNSW_HPP
