#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <future>
#include <iomanip>
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

// --- A Simple Thread Pool Implementation (unchanged) ---
class ThreadPool {
   public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

   private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// --- HNSW Stats ---
struct HNSWStats {
    // Live counters (updated during operations)
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> search_hops{0};  // Total nodes visited/evaluated during search

    // Snapshot stats (calculated by iterating the graph)
    u32 num_levels = 0;
    std::vector<u64> nodes_per_level;
    std::vector<u64> links_per_level;
    double avg_neighbors_total = 0.0;
    std::vector<double> avg_neighbors_per_level;

    // Default constructor
    HNSWStats() = default;

    // Custom Copy Constructor
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

    // Format stats into a readable string
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

    // Calculates and returns a snapshot of the graph structure stats,
    // along with the current values of live counters.
    HNSWStats getStats() const;

    // Resets live counters like distance computations and search hops.
    void resetStats() { stats_.reset_live_counters(); }

   private:
    PID metadata_pid;
    PID base_pid;
    u64 max_elements_;
    double mult_factor;
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // `mutable` allows const methods like searchKnn to update stats.
    mutable HNSWStats stats_;

   public:
    // Constructor with runtime parameters
    HNSW(u64 max_elements, size_t M = 16, size_t ef_construction = 200);
    ~HNSW() { visited_list_pool_.reset(nullptr); }

   private:
    // Helper function to estimate MaxLevel from max_elements
    static size_t estimateMaxLevel(u64 max_elements, size_t M);

    inline float calculateDistance(const float* v1, const float* v2, size_t dim) const {
        // This is now the single point for distance calculation counting.
        stats_.dist_comps.fetch_add(1, std::memory_order_relaxed);
        return DistanceMetric::compare(v1, v2, dim);
    }

    int getNodeMaxLevel(const u16* level_counts) const {
        for (int i = MaxLevel - 1; i >= 0; --i) {
            if (level_counts[i] > 0) {
                return i;
            }
        }
        // This case might happen for a node being constructed or an empty one.
        // Layer 0 is the base.
        return 0;
    }

   public:
    // Add a point to the index.
    void addPoint(const float* point, u32& node_id_out);

    // Search for the K nearest neighbors for a single query.
    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchKnn(const float* query, size_t k, size_t ef_search_param);

    // Search for the K nearest neighbors for a batch of queries in parallel.
    template <bool stats = false>
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
    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchLayer(const float* query, u32 entry_point_id, u32 level, size_t ef,
                                                   const std::unordered_map<u32, float>& distance_cache);
    template <bool stats = false>
    std::pair<float, u32> findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level,
                                                     float entry_point_dist,
                                                     std::unordered_map<u32, float>& distance_cache);
    template <bool stats = false>
    std::pair<float, u32> searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level,
                                          std::unordered_map<u32, float>& distance_cache);

    // --- Now takes the query vector to match hnswlib's heuristic ---
    std::vector<std::pair<float, u32>> selectNeighborsHeuristic(const float* query,
                                                                const std::vector<std::pair<float, u32>>& candidates,
                                                                size_t M_limit);

    void addPoint_internal(const float* point, u32 node_id);
};

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchKnn(const float* query, size_t k,
                                                                        size_t ef_search_param) {
    // --- 1. Get the global entry point ---
    u32 enter_point_id;
    u32 max_l;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue;  // Retry if metadata is being updated
        }
    }

    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        return {};  // Index is empty
    }

    // --- 2. Greedily search upper layers to find the best entry point for layer 0 ---
    std::unordered_map<u32, float> distance_cache;
    std::pair<float, u32> entry_point_for_layer0 = {0.0f, enter_point_id};

    if (max_l > 0) {
        // This call will find the best entry for layer 1 and return its ID and distance.
        entry_point_for_layer0 = searchBaseLayer<stats>(query, enter_point_id, max_l, 1, distance_cache);
    } else {
        // If there are no upper layers, we still need to compute the initial distance
        try {
            GuardO<HNSWPage> page_guard(getNodePID(enter_point_id));
            NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(enter_point_id), this);
            if (stats) {
                entry_point_for_layer0.first = this->calculateDistance(query, acc.getVector(), Dim);
            } else {
                entry_point_for_layer0.first = DistanceMetric::compare(query, acc.getVector(), Dim);
            }
        } catch (const OLCRestartException&) { /* continue with dist=0, searchLayer will re-evaluate */
        }
    }

    // Ensure the final entry point is in the cache for searchLayer to use
    distance_cache[entry_point_for_layer0.second] = entry_point_for_layer0.first;

    // --- 3. Set search parameters ---
    size_t ef_search = (ef_search_param == 0) ? std::max(static_cast<size_t>(efConstruction), k) : ef_search_param;
    ef_search = std::max(ef_search, k);

    // --- 4. Perform beam search on the base layer (layer 0) ---
    // Pass the pre-calculated distance and ID to searchLayer to avoid redundant computation.
    auto candidates = searchLayer<stats>(query, entry_point_for_layer0.second, 0, ef_search, distance_cache);

    // --- 5. Finalize and return results ---
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::vector<std::pair<float, u32>>> HNSW<Dim, DistanceMetric>::searchKnn_parallel(
    std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads) {
    if (queries.empty()) return {};
    if (queries.size() % Dim != 0) {
        throw std::invalid_argument("Total number of floats in query span is not a multiple of the vector dimension.");
    }
    size_t num_queries = queries.size() / Dim;
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_queries);
    std::vector<std::vector<std::pair<float, u32>>> all_results(num_queries);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            all_results[i] = searchKnn<stats>(queries.data() + i * Dim, k, ef_search_param);
        }
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            futures.emplace_back(
                pool.enqueue([this, &all_results, queries_data = queries.data(), i, k, ef_search_param]() {
                    all_results[i] = this->searchKnn<stats>(queries_data + i * Dim, k, ef_search_param);
                }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    return all_results;
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<Dim, DistanceMetric>::searchBaseLayer(const float* query, u32 entry_point_id,
                                                                 int start_level, int end_level,
                                                                 std::unordered_map<u32, float>& distance_cache) {
    u32 current_entry_point_id = entry_point_id;
    float current_dist = 0.0;

    // Calculate the distance for the initial entry point only ONCE.
    try {
        GuardO<HNSWPage> initial_guard(getNodePID(current_entry_point_id));
        NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(current_entry_point_id), this);
        if (stats) {
            current_dist = this->calculateDistance(query, initial_acc.getVector(), Dim);
        } else {
            current_dist = DistanceMetric::compare(query, initial_acc.getVector(), Dim);
        }
    } catch (const OLCRestartException&) {
        // If the initial node cannot be read, return a sentinel value.
        // The caller (searchKnn) can then decide how to proceed.
        return {std::numeric_limits<float>::max(), entry_point_id};
    }

    for (int level = start_level; level >= end_level; --level) {
        // Pass the current best distance to the next level's search function.
        auto result =
            findBestEntryPointForLevel<stats>(query, current_entry_point_id, level, current_dist, distance_cache);
        current_dist = result.first;
        current_entry_point_id = result.second;
    }
    return {current_dist, current_entry_point_id};
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchLayer(
    const float* query, u32 entry_point_id, u32 level, size_t ef,
    const std::unordered_map<u32, float>& distance_cache) {
    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;

    std::priority_queue<std::pair<float, u32>> top_candidates;
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, std::greater<std::pair<float, u32>>>
        candidate_queue;

    // --- INITIALIZATION ---
    // Check if the entry point's distance is in our cache.
    auto cache_it = distance_cache.find(entry_point_id);
    if (cache_it != distance_cache.end()) {
        float dist = cache_it->second;
        top_candidates.push({dist, entry_point_id});
        candidate_queue.push({dist, entry_point_id});
        visited_array[entry_point_id] = visited_array_tag;
    } else {
        // Fallback: compute if not in cache (e.g., when an empty cache is passed).
        try {
            GuardO<HNSWPage> page_guard(getNodePID(entry_point_id));
            NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(entry_point_id), this);
            float dist = 0.0;
            if (stats) {
                dist = this->calculateDistance(query, acc.getVector(), Dim);
            } else {
                dist = DistanceMetric::compare(query, acc.getVector(), Dim);
            }
            top_candidates.push({dist, entry_point_id});
            candidate_queue.push({dist, entry_point_id});
            visited_array[entry_point_id] = visited_array_tag;
        } catch (const OLCRestartException&) {
            visited_list_pool_->releaseVisitedList(visited_nodes);
            return {};
        }
    }

    // --- BEAM SEARCH LOOP ---
    while (!candidate_queue.empty()) {
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        if (top_candidates.size() >= ef && current_pair.first > top_candidates.top().first) {
            break;
        }

        if (stats) {
            stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
        }

        u32 current_id = current_pair.second;
        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id));
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_id), this);
            auto neighbors = current_acc.getNeighbors(level, this);

            for (const u32& neighbor_id : neighbors) {
                if (!(visited_array[neighbor_id] == visited_array_tag)) {
                    visited_array[neighbor_id] = visited_array_tag;
                    float neighbor_dist = 0.0;

                    // --- OPTIMIZATION: USE THE CACHE ---
                    auto neighbor_cache_it = distance_cache.find(neighbor_id);
                    if (neighbor_cache_it != distance_cache.end()) {
                        neighbor_dist = neighbor_cache_it->second;
                    } else {
                        GuardO<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id));
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        if (stats) {
                            neighbor_dist = this->calculateDistance(query, neighbor_acc.getVector(), Dim);
                        } else {
                            neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                        }
                    }

                    if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                        candidate_queue.push({neighbor_dist, neighbor_id});
                        top_candidates.push({neighbor_dist, neighbor_id});
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    std::vector<std::pair<float, u32>> results;
    results.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        results.push_back(top_candidates.top());
        top_candidates.pop();
    }
    std::reverse(results.begin(), results.end());
    visited_list_pool_->releaseVisitedList(visited_nodes);
    return results;
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<Dim, DistanceMetric>::findBestEntryPointForLevel(
    const float* query, u32 entry_point_id, int level, float entry_point_dist,
    std::unordered_map<u32, float>& distance_cache) {
    u32 current_node_id = entry_point_id;
    float min_dist = entry_point_dist;  // Use the pre-calculated distance

    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    std::unique_ptr<VisitedList, std::function<void(VisitedList*)>> visited_guard(
        visited_nodes, [&](VisitedList* p) { this->visited_list_pool_->releaseVisitedList(p); });

    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;

    bool changed = true;
    while (changed) {
        changed = false;
        u32 best_neighbor_id = current_node_id;

        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_node_id));
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_node_id), this);

            // The first node is marked as visited here.
            // Note: We don't need to re-calculate its distance as it's passed in.
            visited_array[current_node_id] = visited_array_tag;

            if (stats) {
                stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
            }

            auto neighbors_span = current_acc.getNeighbors(level, this);

            // Prefetch the first neighbor's data to improve pipeline efficiency.
            if (!neighbors_span.empty()) {
                u32 first_neighbor_id = neighbors_span[0];
                _mm_prefetch(reinterpret_cast<const char*>(&visited_array[first_neighbor_id]), _MM_HINT_T0);
                PID page_to_prefetch = getNodePID(first_neighbor_id);
                bm.prefetchPages(&page_to_prefetch, 1);
            }

            for (size_t i = 0; i < neighbors_span.size(); ++i) {
                const u32& neighbor_id = neighbors_span[i];

                // In-loop prefetch for the *next* neighbor to keep the pipeline full.
                if (i + 1 < neighbors_span.size()) {
                    u32 next_neighbor_id = neighbors_span[i + 1];
                    _mm_prefetch(reinterpret_cast<const char*>(&visited_array[next_neighbor_id]), _MM_HINT_T0);
                    PID page_to_prefetch = getNodePID(next_neighbor_id);
                    bm.prefetchPages(&page_to_prefetch, 1);
                }

                if (!(visited_array[neighbor_id] == visited_array_tag)) {
                    visited_array[neighbor_id] = visited_array_tag;

                    GuardO<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id));
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    float neighbor_dist = 0.0;
                    if (stats) {
                        neighbor_dist = this->calculateDistance(query, neighbor_acc.getVector(), Dim);
                    } else {
                        neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                    }
                    distance_cache[neighbor_id] = neighbor_dist;

                    if (neighbor_dist < min_dist) {
                        min_dist = neighbor_dist;
                        best_neighbor_id = neighbor_id;
                        changed = true;
                    }
                }
            }
            current_node_id = best_neighbor_id;
        } catch (const OLCRestartException&) {
            // If a concurrent modification happens, stop the greedy search at this level
            // and proceed with the best node found so far.
            changed = false;
        }
    }

    return {min_dist, current_node_id};
}

#endif  // HNSW_HPP
