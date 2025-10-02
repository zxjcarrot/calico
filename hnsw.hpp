#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <span>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"

// --- HNSW Metadata & Support Structures ---

typedef unsigned char vl_type;

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

struct HNSWStats {
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> search_hops{0};
    u32 num_levels = 0;
    std::vector<u64> nodes_per_level;
    std::vector<u64> links_per_level;
    double avg_neighbors_total = 0.0;
    std::vector<double> avg_neighbors_per_level;
    HNSWStats() = default;
    HNSWStats(const HNSWStats& other) {
        dist_comps.store(other.dist_comps.load());
        search_hops.store(other.search_hops.load());
        num_levels = other.num_levels;
        nodes_per_level = other.nodes_per_level;
        links_per_level = other.links_per_level;
        avg_neighbors_total = other.avg_neighbors_total;
        avg_neighbors_per_level = other.avg_neighbors_per_level;
    }

    // Custom Copy Assignment Operator
    HNSWStats& operator=(const HNSWStats& other) {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            search_hops.store(other.search_hops.load());
            num_levels = other.num_levels;
            nodes_per_level = other.nodes_per_level;
            links_per_level = other.links_per_level;
            avg_neighbors_total = other.avg_neighbors_total;
            avg_neighbors_per_level = other.avg_neighbors_per_level;
        }
        return *this;
    }

    // We can also define move operations for efficiency, although the copy constructor will solve the immediate
    // problem. Custom Move Constructor
    HNSWStats(HNSWStats&& other) noexcept {
        dist_comps.store(other.dist_comps.load());
        search_hops.store(other.search_hops.load());
        num_levels = other.num_levels;
        nodes_per_level = std::move(other.nodes_per_level);
        links_per_level = std::move(other.links_per_level);
        avg_neighbors_total = other.avg_neighbors_total;
        avg_neighbors_per_level = std::move(other.avg_neighbors_per_level);
    }

    // Custom Move Assignment Operator
    HNSWStats& operator=(HNSWStats&& other) noexcept {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            search_hops.store(other.search_hops.load());
            num_levels = other.num_levels;
            nodes_per_level = std::move(other.nodes_per_level);
            links_per_level = std::move(other.links_per_level);
            avg_neighbors_total = other.avg_neighbors_total;
            avg_neighbors_per_level = std::move(other.avg_neighbors_per_level);
        }
        return *this;
    }

    // Reset live counters
    void reset_live_counters() {
        dist_comps.store(0);
        search_hops.store(0);
    }
    std::string toString() const {
        std::ostringstream oss;
        oss << "--- HNSW Performance Stats ---\n";
        oss << "Distance Computations: " << dist_comps.load() << "\n";
        oss << "Search Hops:           " << search_hops.load() << "\n";
        oss << "\n--- Graph Structure Stats ---\n";
        oss << "Number of Levels: " << num_levels << "\n";

        u64 total_nodes = 0;
        u64 total_links = 0;

        for (u32 i = 0; i < num_levels; ++i) {
            oss << "Level " << i << ":\n";
            oss << "  Nodes: " << nodes_per_level[i] << "\n";
            oss << "  Avg Neighbors: " << std::fixed << std::setprecision(2) << avg_neighbors_per_level[i] << "\n";
            total_nodes += nodes_per_level[i];
            total_links += links_per_level[i];
        }

        oss << "\n--- Overall Graph Stats ---\n";
        oss << "Total Nodes: " << total_nodes << "\n";
        oss << "Total Links: " << total_links << "\n";
        oss << "Avg Neighbor List Length (Total): " << std::fixed << std::setprecision(2) << avg_neighbors_total
            << "\n";

        return oss.str();
    }
};

// --- Core HNSW Structures ---

// Represents a logical pointer to a block within an OverflowPage.
struct OverflowPointer {
    PID page_id;
    u16 offset;  // Offset within the OverflowPage where the LinkListBlock starts.
    static constexpr PID INVALID_PID = std::numeric_limits<PID>::max();

    bool isValid() const { return page_id != INVALID_PID; }
};

struct HNSWMetadataPage {
    bool dirty;
    PID base_pid;
    u64 max_elements;
    std::atomic<u64> node_count;

    u32 enter_point_node_id;
    std::atomic<u32> max_level;

    // --- Added fields for managing overflow page allocations ---
    PID last_overflow_pid;     // Tracks the current overflow page for new allocations.
    u16 last_overflow_offset;  // Tracks the next free byte offset on that page.

    static constexpr u32 invalid_node_id = std::numeric_limits<u32>::max();
};

// --- HNSW Main Class ---

template <size_t Dim, typename DistanceMetric = hnsw_distance::SIMDAcceleratedL2>
class HNSW {
   public:
    // --- Runtime parameters ---
    const size_t M;
    const size_t M0;
    const size_t efConstruction;
    const size_t MaxLevel;

    // --- Runtime size and offset calculations for the new layout ---
    static constexpr size_t VectorSize = Dim * sizeof(float);
    const size_t BaseNodeSize;      // Size of a single node's data on a BaseNodePage
    const size_t LinkSizePerLevel;  // Size of neighbor list for one level on an OverflowPage (u16 count + M * u32 ids)
    const size_t MaxNodesPerBasePage;

    // --- Page Structures defined within the HNSW class scope ---

    // A page containing the primary data for nodes (vector, level 0 neighbors).
    struct BaseNodePage {
        bool dirty;
        u16 node_count;
        u16 padding[3];
        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count) + sizeof(padding);

        u8* getNodeDataArea() { return reinterpret_cast<u8*>(this) + HeaderSize; }
        const u8* getNodeDataArea() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }
    };

    // A page that acts as a heap for storing higher-level neighbor lists.
    struct OverflowPage {
        bool dirty;
        u16 free_space_offset;  // Offset to the next available byte for allocation.
        u16 padding[3];
        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(free_space_offset) + sizeof(padding);

        u8* getDataArea() { return reinterpret_cast<u8*>(this) + HeaderSize; }
        const u8* getDataArea() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }
    };

   public:
    // --- Public API ---
    HNSW(u64 max_elements, size_t M = 16, size_t ef_construction = 200, bool enable_prefetch = true);
    ~HNSW() { visited_list_pool_.reset(nullptr); }

    void addPoint(const float* point, u32& node_id_out);
    void addPoint_parallel(std::span<const float> points, size_t num_threads = 0);

    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchKnn(const float* query, size_t k, size_t ef_search_param);

    template <bool stats = false>
    std::vector<std::vector<std::pair<float, u32>>> searchKnn_parallel(std::span<const float> queries, size_t k,
                                                                       size_t ef_search_param, size_t num_threads = 0);

    HNSWStats getStats() const;
    void resetStats() { stats_.reset_live_counters(); }
    PID getMetadataPid() const { return metadata_pid; }
    u64 getBufferManagerAllocCount() const;
    std::string getIndexInfo() const;

    HNSW(const HNSW&) = delete;
    HNSW& operator=(const HNSW&) = delete;
    HNSW(HNSW&&) = delete;
    HNSW& operator=(HNSW&&) = delete;
    

   private:
    PID metadata_pid;
    PID base_pid;
    u64 max_elements_;
    double mult_factor;
    bool enable_prefetch_;
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
    mutable HNSWStats stats_;

    // --- Private helper functions for the new layout ---
    OverflowPointer allocateOverflowSpace(size_t required_size);
    static size_t calculateBaseNodeSize(size_t M0_param);

    // Internal implementation helpers (signatures might change to adapt to new layout)
    void addPoint_internal(const float* point, u32 node_id);
    u32 getRandomLevel();
    static size_t estimateMaxLevel(u64 max_elements, size_t M);

    // Search helpers
    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchLayer(const float* query, u32 entry_point_id, u32 level, size_t ef,
                                                   std::optional<std::pair<float, u32>> initial_entry_dist_pair);
    template <bool stats = false>
    std::pair<float, u32> findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level,
                                                     float entry_point_dist);
    template <bool stats = false>
    std::pair<float, u32> searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level);

    std::vector<std::pair<float, u32>> selectNeighborsHeuristic(const float* query,
                                                                const std::vector<std::pair<float, u32>>& candidates,
                                                                size_t M_limit);

    // --- Location helpers ---
    inline PID getNodePID(u32 node_id) const;
    inline u32 getNodeIndexInPage(u32 node_id) const;

    // --- Node Accessors ---

    // Accessor for reading data from a BaseNodePage. It does NOT access OverflowPages.
    // It provides the OverflowPointer, which the caller must use to fetch the correct OverflowPage.
    class NodeAccessor {
       private:
        const u8* node_start_ptr_;  // Pointer to the start of this node's slot in the page
        const HNSW* hnsw_;

       public:
        NodeAccessor(const BaseNodePage* page, u32 node_idx, const HNSW* hnsw_instance) : hnsw_(hnsw_instance) {
            node_start_ptr_ = page->getNodeDataArea() + node_idx * hnsw_->BaseNodeSize;
        }

        const float* getVector() const { return reinterpret_cast<const float*>(node_start_ptr_); }

        u16 getLevel() const { return *reinterpret_cast<const u16*>(node_start_ptr_ + VectorSize); }

        u16 getL0Count() const { return *reinterpret_cast<const u16*>(node_start_ptr_ + VectorSize + sizeof(u16)); }

        OverflowPointer getOverflowPointer() const {
            return *reinterpret_cast<const OverflowPointer*>(node_start_ptr_ + VectorSize + sizeof(u16) + sizeof(u16));
        }

        std::span<const u32> getL0Neighbors() const {
            const u32* neighbors_start = reinterpret_cast<const u32*>(node_start_ptr_ + VectorSize + sizeof(u16) +
                                                                      sizeof(u16) + sizeof(OverflowPointer));
            return {neighbors_start, getL0Count()};
        }

        // Helper to get higher-level neighbors from a provided, already-fetched OverflowPage.
        // The CALLER is responsible for fetching the page using the OverflowPointer.
        std::span<const u32> getHigherLevelNeighbors(u32 level, const OverflowPage* overflow_page) const {
            if (level == 0) return {};

            OverflowPointer ptr = getOverflowPointer();
            if (!ptr.isValid()) return {};

            // Calculate offset to the start of this level's neighbor list within the LinkListBlock
            size_t level_offset_in_block = (level - 1) * hnsw_->LinkSizePerLevel;
            const u8* list_start = overflow_page->getDataArea() + ptr.offset + level_offset_in_block;

            u16 count = *reinterpret_cast<const u16*>(list_start);
            const u32* neighbors = reinterpret_cast<const u32*>(list_start + sizeof(u16));
            return {neighbors, count};
        }

        static inline size_t getVectorOffset(const HNSW* hnsw_instance, u32 graph_node_id) {
            u32 node_idx_in_page = hnsw_instance->getNodeIndexInPage(graph_node_id);
            // The offset is the page header size plus the space for all preceding nodes in that page.
            // The vector is the very first field in the node's data slot.
            return BaseNodePage::HeaderSize + (node_idx_in_page * hnsw_instance->BaseNodeSize);
        }
    };

    // Mutable accessor for modifying a node's data.
    class MutableNodeAccessor {
       private:
        u8* node_start_ptr_;
        HNSW* hnsw_;
        BaseNodePage* page_;  // Keep a pointer to the page to mark it dirty

       public:
        MutableNodeAccessor(BaseNodePage* page, u32 node_idx, HNSW* hnsw_instance) : hnsw_(hnsw_instance), page_(page) {
            node_start_ptr_ = page->getNodeDataArea() + node_idx * hnsw_->BaseNodeSize;
        }

        float* getVector() { return reinterpret_cast<float*>(node_start_ptr_); }
        u16 getLevel() const { return *reinterpret_cast<const u16*>(node_start_ptr_ + VectorSize); }
        void setLevel(u16 level) {
            *reinterpret_cast<u16*>(node_start_ptr_ + VectorSize) = level;
            page_->dirty = true;
        }
        u16 getL0Count() const { return *reinterpret_cast<const u16*>(node_start_ptr_ + VectorSize + sizeof(u16)); }
        void setL0Count(u16 count) {
            *reinterpret_cast<u16*>(node_start_ptr_ + VectorSize + sizeof(u16)) = count;
            page_->dirty = true;
        }
        void setOverflowPointer(OverflowPointer ptr) {
            *reinterpret_cast<OverflowPointer*>(node_start_ptr_ + VectorSize + sizeof(u16) + sizeof(u16)) = ptr;
            page_->dirty = true;
        }

        OverflowPointer getOverflowPointer() const {
            return *reinterpret_cast<const OverflowPointer*>(node_start_ptr_ + VectorSize + sizeof(u16) + sizeof(u16));
        }

        void setL0Neighbors(const std::vector<u32>& new_neighbors) {
            u32* neighbors_start = reinterpret_cast<u32*>(node_start_ptr_ + VectorSize + sizeof(u16) + sizeof(u16) +
                                                          sizeof(OverflowPointer));
            size_t count_to_copy = std::min(new_neighbors.size(), hnsw_->M0);
            std::copy(new_neighbors.begin(), new_neighbors.begin() + count_to_copy, neighbors_start);
            setL0Count(count_to_copy);
        }

        std::span<const u32> getL0Neighbors() const {
            const u32* neighbors_start = reinterpret_cast<const u32*>(node_start_ptr_ + VectorSize + sizeof(u16) +
                                                                      sizeof(u16) + sizeof(OverflowPointer));
            return {neighbors_start, getL0Count()};
        }

        // Helper to modify higher-level neighbors on a provided OverflowPage.
        void setHigherLevelNeighbors(u32 level, const std::vector<u32>& new_neighbors, OverflowPage* overflow_page) {
            if (level == 0) return;

            OverflowPointer ptr =
                *reinterpret_cast<const OverflowPointer*>(node_start_ptr_ + VectorSize + sizeof(u16) + sizeof(u16));

            if (!ptr.isValid()) return;  // Should not happen if called correctly

            size_t level_offset_in_block = (level - 1) * hnsw_->LinkSizePerLevel;
            u8* list_start = overflow_page->getDataArea() + ptr.offset + level_offset_in_block;

            size_t count_to_copy = std::min(new_neighbors.size(), hnsw_->M);
            *reinterpret_cast<u16*>(list_start) = count_to_copy;
            u32* neighbors = reinterpret_cast<u32*>(list_start + sizeof(u16));
            std::copy(new_neighbors.begin(), new_neighbors.begin() + count_to_copy, neighbors);

            overflow_page->dirty = true;
        }

        std::span<const u32> getHigherLevelNeighbors(u32 level, const OverflowPage* overflow_page) const {
            if (level == 0) return {};

            OverflowPointer ptr = getOverflowPointer();
            if (!ptr.isValid()) return {};

            // Calculate offset to the start of this level's neighbor list within the LinkListBlock
            size_t level_offset_in_block = (level - 1) * hnsw_->LinkSizePerLevel;
            const u8* list_start = overflow_page->getDataArea() + ptr.offset + level_offset_in_block;

            u16 count = *reinterpret_cast<const u16*>(list_start);
            const u32* neighbors = reinterpret_cast<const u32*>(list_start + sizeof(u16));
            return {neighbors, count};
        }

        static inline size_t getVectorOffset(const HNSW* hnsw_instance, u32 graph_node_id) {
            u32 node_idx_in_page = hnsw_instance->getNodeIndexInPage(graph_node_id);
            // The offset is the page header size plus the space for all preceding nodes in that page.
            // The vector is the very first field in the node's data slot.
            return BaseNodePage::HeaderSize + (node_idx_in_page * hnsw_instance->BaseNodeSize);
        }
    };

    inline float calculateDistance(const float* v1, const float* v2, size_t dim) const {
        stats_.dist_comps.fetch_add(1, std::memory_order_relaxed);
        return DistanceMetric::compare(v1, v2, dim);
    }
};

#endif  // HNSW_HPP