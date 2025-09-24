#include "hnsw.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <immintrin.h>

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

// --- Helper function to estimate MaxLevel ---
template <size_t Dim, typename DistanceMetric>
size_t HNSW<Dim, DistanceMetric>::estimateMaxLevel(u64 max_elements, size_t M) {
    // The random level generation uses: level = -log(uniform_random) / log(M)
    // For max_elements, the expected maximum level is approximately log(max_elements) / log(M)
    if (max_elements <= 1) return 1;
    
    // Correct formula based on the actual random level generation algorithm
    double expected_max_level = std::log(static_cast<double>(max_elements)) / std::log(static_cast<double>(M));
    size_t estimated_max_level = static_cast<size_t>(std::ceil(expected_max_level + 1)); // +2 for safety margin
    
    // Ensure reasonable bounds: at least 3 levels, at most 16 levels
    return std::max(static_cast<size_t>(3), std::min(static_cast<size_t>(16), estimated_max_level));
}

// --- HNSW Constructor Implementation ---
template <size_t Dim, typename DistanceMetric>
HNSW<Dim, DistanceMetric>::HNSW(u64 max_elements, size_t M_param, size_t ef_construction_param)
    : M(M_param),
      M0(2 * M_param),  // M0 = 2 * M as requested
      efConstruction(ef_construction_param),
      MaxLevel(estimateMaxLevel(max_elements, M_param)),
      MaxNeighborsHeaderSize(MaxLevel * sizeof(u16)),
      MaxNeighborsListSize((M0 + (MaxLevel - 1) * M) * sizeof(u32)),
      FixedNodeSize(VectorSize + MaxNeighborsHeaderSize + MaxNeighborsListSize),
      MaxNodesPerPage((pageSize - HNSWPage::HeaderSize) / FixedNodeSize),
      NodesPerPage(MaxNodesPerPage) {
      
    // Validate that we can fit at least one node per page
    if (MaxNodesPerPage == 0) {
        throw std::runtime_error("Page size is too small to fit even one node with the given parameters. "
                               "Consider reducing M or vector dimension.");
    }
    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    AllocGuard<HNSWMetadataPage> meta_guard;
    this->metadata_pid = meta_guard.pid;

    this->max_elements_ = max_elements;
    mult_factor = 1.0 / std::log(1.0 * M);

    meta_guard->dirty = false;
    meta_guard->base_pid = -1;
    meta_guard->max_elements = max_elements;
    meta_guard->node_count.store(0);
    meta_guard->enter_point_node_id = HNSWMetadataPage::invalid_node_id;
    meta_guard->max_level.store(0);

    u64 total_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
    if (total_pages > 0) {
        AllocGuard<HNSWPage> first_page_guard;
        meta_guard->base_pid = first_page_guard.pid;
        first_page_guard->dirty = false;
        first_page_guard->node_count = 0;
        first_page_guard->dirty = true;

        for (u64 i = 1; i < total_pages; ++i) {
            AllocGuard<HNSWPage> page_guard;
            page_guard->dirty = false;
            page_guard->node_count = 0;
            page_guard->dirty = true;
        }
    }

    this->base_pid = meta_guard->base_pid;
    meta_guard->dirty = true;
}

// --- Helper Functions (searchLayer, searchBaseLayer)---
template <size_t Dim, typename DistanceMetric>
u32 HNSW<Dim, DistanceMetric>::getRandomLevel() {
    thread_local static std::mt19937 level_generator(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator)) * mult_factor;
    return std::min(static_cast<u32>(r), static_cast<u32>(MaxLevel - 1));
}

template <size_t Dim, typename DistanceMetric>
u32 HNSW<Dim, DistanceMetric>::findBestEntryPointForLevel(const float* query,
                                                          u32 entry_point_id,
                                                          int level) {
    u32 current_node_id = entry_point_id;

    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    std::unique_ptr<VisitedList, std::function<void(VisitedList*)>> visited_guard(
        visited_nodes, [&](VisitedList* p) { this->visited_list_pool_->releaseVisitedList(p); });

    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;

    bool changed = true;
    while (changed) {
        changed = false;
        u32 best_neighbor_id = current_node_id;
        float min_dist;

        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_node_id));
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_node_id), this);

            visited_array[current_node_id] = visited_array_tag;
            min_dist = DistanceMetric::compare(query, current_acc.getVector(), Dim);

            auto neighbors_span = current_acc.getNeighbors(level, this);

            if (!neighbors_span.empty()) {
                u32 first_neighbor_id = neighbors_span[0];
                _mm_prefetch(reinterpret_cast<const char*>(&visited_array[first_neighbor_id]), _MM_HINT_T0);
                PID page_to_prefetch = getNodePID(first_neighbor_id);
                bm.prefetchPages(&page_to_prefetch, 1);
            }

            for (size_t i = 0; i < neighbors_span.size(); ++i) {
                const u32& neighbor_id = neighbors_span[i];
                
                // --- In-loop prefetch for the *next* neighbor to keep the pipeline full ---
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
                    float neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);

                    if (neighbor_dist < min_dist) {
                        min_dist = neighbor_dist;
                        best_neighbor_id = neighbor_id;
                        changed = true;
                    }
                }
            }
            current_node_id = best_neighbor_id;
        } catch (const OLCRestartException&) {
            changed = false;
        }
    }

    return current_node_id;
}

template <size_t Dim, typename DistanceMetric>
u32 HNSW<Dim, DistanceMetric>::searchBaseLayer(const float* query, u32 entry_point_id,
                                                                                int start_level, int end_level) {
    u32 current_entry_point = entry_point_id;
    for (int level = start_level; level >= end_level; --level) {
        current_entry_point = findBestEntryPointForLevel(query, current_entry_point, level);
    }
    return current_entry_point;
}

template <size_t Dim, typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchLayer(
    const float* query, u32 entry_point_id, u32 level, size_t ef) {
    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;
    std::priority_queue<std::pair<float, u32>> top_candidates;
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, std::greater<std::pair<float, u32>>>
        candidate_queue;

    try {
        GuardO<HNSWPage> page_guard(getNodePID(entry_point_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(entry_point_id), this);
        float dist = DistanceMetric::compare(query, acc.getVector(), Dim);
        top_candidates.push({dist, entry_point_id});
        candidate_queue.push({dist, entry_point_id});
        visited_array[entry_point_id] = visited_array_tag;
    } catch (const OLCRestartException&) {
        return {};
    }

    while (!candidate_queue.empty()) {
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        if (top_candidates.size() >= ef && current_pair.first > top_candidates.top().first) {
            break;
        }

        // Collect unique pages and prefetch them all at once
        std::vector<PID> pages_to_prefetch;
        pages_to_prefetch.reserve(24);
        u32 current_id = current_pair.second;
        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id));
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_id), this);

            if (current_acc.getLevel() < level) continue;

            auto neighbors = current_acc.getNeighbors(level, this);

            // --- Prefetch all neighbors at once before processing ---
            if (!neighbors.empty()) {
                // Prefetch visited array entries for all neighbors
                for (const u32& neighbor_id : neighbors) {
                    _mm_prefetch(reinterpret_cast<const char*>(&visited_array[neighbor_id]), _MM_HINT_T0);
                }
                pages_to_prefetch.clear();
                for (const u32& neighbor_id : neighbors) {
                    PID page_id = getNodePID(neighbor_id);
                    pages_to_prefetch.push_back(page_id);
                }
                
                if (!pages_to_prefetch.empty()) {
                    bm.prefetchPages(pages_to_prefetch.data(), pages_to_prefetch.size());
                }
            }
            
            for (size_t i = 0; i < neighbors.size(); ++i) {
                const u32& neighbor_id = neighbors[i];

                if (!(visited_array[neighbor_id] == visited_array_tag)) {
                    visited_array[neighbor_id] = visited_array_tag;
                    GuardO<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id));
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    float neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);

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

        // --- Prefetch the next best candidate from the queue (this part was correct) ---
        if (!candidate_queue.empty()) {
            PID next_candidate_pid = getNodePID(candidate_queue.top().second);
            bm.prefetchPages(&next_candidate_pid, 1);
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
inline PID HNSW<Dim, DistanceMetric>::getNodePID(u32 node_id) const {
    return base_pid + (node_id / NodesPerPage);
}

template <size_t Dim, typename DistanceMetric>
inline u32 HNSW<Dim, DistanceMetric>::getNodeIndexInPage(u32 node_id) const {
    return node_id % NodesPerPage;
}

template <size_t Dim, typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::selectNeighborsHeuristic(
    const float* query, const std::vector<std::pair<float, u32>>& candidates, size_t M_limit) {
    if (candidates.size() <= M_limit) {
        return candidates;
    }

    std::vector<std::pair<float, u32>> result;
    result.reserve(M_limit);

    // candidates vector is already sorted by distance, from closest to farthest.
    // We can iterate through it directly.
    for (const auto& current_candidate : candidates) {
        if (result.size() >= M_limit) {
            break;
        }

        bool is_good_candidate = true;
        // Check against the neighbors we have already selected for the result list.
        for (const auto& selected : result) {
            try {
                GuardO<HNSWPage> candidate_page_guard(getNodePID(current_candidate.second));
                NodeAccessor candidate_acc(candidate_page_guard.ptr, getNodeIndexInPage(current_candidate.second), this);
                const float* candidate_vector = candidate_acc.getVector();

                GuardO<HNSWPage> selected_page_guard(getNodePID(selected.second));
                NodeAccessor selected_acc(selected_page_guard.ptr, getNodeIndexInPage(selected.second), this);
                const float* selected_vector = selected_acc.getVector();

                if (DistanceMetric::compare(candidate_vector, selected_vector, Dim) < current_candidate.first) {
                    is_good_candidate = false;
                    break;
                }
            } catch (const OLCRestartException&) {
                is_good_candidate = false;
                break;
            }
        }

        if (is_good_candidate) {
            result.push_back(current_candidate);
        }
    }

    return result;
}

template <size_t Dim, typename DistanceMetric>
void HNSW<Dim, DistanceMetric>::addPoint_internal(const float* point,
                                                                                   u32 new_node_id) {
    const u32 new_node_level = getRandomLevel();
    PID new_node_pid = getNodePID(new_node_id);

    // --- 1. Allocate space for the new node on its page ---
    // Fixed-size node allocation: direct placement in node grid.
    {
        GuardX<HNSWPage> page_guard(new_node_pid);
        HNSWPage* page = page_guard.ptr;
        u32 node_idx = getNodeIndexInPage(new_node_id);

        // Check if page has room for this node
        if (node_idx >= MaxNodesPerPage) {
            throw std::runtime_error("Node index exceeds page capacity. This should not happen with pre-calculation.");
        }

        // Calculate direct offset to this node's data
        u8* node_start = page->getNodeData() + node_idx * FixedNodeSize;
        
        // Initialize node: [Vector][Level Counts][Neighbor IDs]
        float* vector_ptr = reinterpret_cast<float*>(node_start);
        u16* level_counts_ptr = reinterpret_cast<u16*>(node_start + VectorSize);
        u32* neighbor_ids_ptr = reinterpret_cast<u32*>(node_start + VectorSize + MaxNeighborsHeaderSize);

        // Copy vector data
        memcpy(vector_ptr, point, VectorSize);
        
        // Zero-initialize level counts and neighbor IDs
        memset(level_counts_ptr, 0, MaxNeighborsHeaderSize + MaxNeighborsListSize);

        // Update page metadata
        page->node_count = std::max(page->node_count, static_cast<u16>(node_idx + 1));
        page->dirty = true;
    }

    // --- 2. Find entry point for the search ---
    u32 enter_point_id;
    u32 max_l;

    for (;;) { // Add retry loop
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break; // Success
        } catch (const OLCRestartException&) {
            continue; // Retry
        }
    }

    // If this is the first node, set it as the entry point and return.
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        if (meta_guard->enter_point_node_id == HNSWMetadataPage::invalid_node_id) {
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level = new_node_level;
            meta_guard->dirty = true;
        }
        return;
    }

    // --- 3. Search from top layer down to the new node's level ---
    // Find the closest node to the query point on the upper layers.
    if (new_node_level < max_l) {
        enter_point_id = searchBaseLayer(point, enter_point_id, max_l, new_node_level + 1);
    }

    // --- 4. Connection phase: from new_node_level down to 0 ---
    for (int level = std::min(new_node_level, max_l); level >= 0; --level) {
        // Find the best neighbors for the new node at the current level.
        auto candidates = searchLayer(point, enter_point_id, level, efConstruction);
        if (candidates.empty()) {
            // This can happen in rare concurrent scenarios. If so, we can't connect
            // to anything at this level, so we just continue to the next level down.
            continue;
        }

        size_t M_level = (level == 0) ? M0 : M;
        auto neighbors_to_link_pairs = selectNeighborsHeuristic(point, candidates, M_level);

        // *** FINE-GRAINED LOCKING ***
        // We perform two separate steps to avoid deadlocks:
        // 1. Link the new node TO its neighbors (one lock on the new node's page).
        // 2. Link the neighbors BACK TO the new node (one lock per neighbor, sequentially).

        // Step 4a: Link the new node TO its chosen neighbors.
        {
            // Lock only the new node's page to update its own neighbor list.
            GuardX<HNSWPage> new_node_page_guard(new_node_pid);
            MutableNodeAccessor new_node_acc(new_node_page_guard.ptr, getNodeIndexInPage(new_node_id), this);
            std::vector<u32> neighbor_ids;
            neighbor_ids.reserve(neighbors_to_link_pairs.size());
            for (const auto& p : neighbors_to_link_pairs) {
                neighbor_ids.push_back(p.second);
            }
            new_node_acc.setNeighbors(level, neighbor_ids, this);
        }  // Lock on new_node_pid is released here.

        // Step 4b: Link neighbors BACK to the new node, one by one.
        for (const auto& neighbor_pair : neighbors_to_link_pairs) {
            u32 neighbor_id = neighbor_pair.second;
            PID neighbor_pid = getNodePID(neighbor_id);

            // This loop ensures that the entire read-and-write operation is atomic.
            // If any data is changed by another thread during our read phase, we restart.
            for (;;) {  // OLC retry loop
                try {
                    // --- PHASE 1: READ EVERYTHING NEEDED, WITHOUT ANY EXCLUSIVE LOCKS ---
                    std::vector<float> neighbor_vector_copy(Dim);
                    std::vector<u32> current_neighbors;
                    std::unordered_map<PID, std::vector<float>> connection_vectors;  // Stores vectors for pruning
                    bool needs_pruning;

                    // Step 1.1: Optimistically read the neighbor's primary data (its vector and its neighbor list).
                    {
                        GuardO<HNSWPage> neighbor_page_guard(neighbor_pid);
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);

                        const float* neighbor_vec_ptr = neighbor_acc.getVector();
                        std::copy(neighbor_vec_ptr, neighbor_vec_ptr + Dim, neighbor_vector_copy.begin());

                        auto neighbors_span = neighbor_acc.getNeighbors(level, this);
                        current_neighbors.assign(neighbors_span.begin(), neighbors_span.end());
                    }

                    needs_pruning = (current_neighbors.size() >= M_level);

                    // Step 1.2: If pruning is necessary, we must also read the vectors of all its connections.
                    // This is the crucial step to avoid needing to read them inside the write lock.
                    if (needs_pruning) {
                        for (PID conn_id : current_neighbors) {
                            GuardO<HNSWPage> conn_page_guard(getNodePID(conn_id));
                            NodeAccessor conn_acc(conn_page_guard.ptr, getNodeIndexInPage(conn_id), this);
                            const float* conn_vec_ptr = conn_acc.getVector();
                            connection_vectors[conn_id].assign(conn_vec_ptr, conn_vec_ptr + Dim);
                        }
                    }

                    // --- PHASE 2: ACQUIRE A SINGLE LOCK AND WRITE ---
                    // If Phase 1 completed without an exception, we have a consistent snapshot of all needed data.
                    // Now, we can acquire the single exclusive lock and perform our modifications.
                    {
                        GuardX<HNSWPage> neighbor_page_guard(neighbor_pid);

                        // Inside the X-Lock, we must re-read the neighbor list ONE LAST TIME to prevent a race
                        // condition. This read is safe and does not require another lock.
                        NodeAccessor locked_reader(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        auto latest_neighbors_span = locked_reader.getNeighbors(level, this);
                        std::vector<u32> latest_neighbors(latest_neighbors_span.begin(), latest_neighbors_span.end());

                        // Now, perform all logic using ONLY our local data copies. NO MORE GuardO calls!
                        if (latest_neighbors.size() >= M_level) {
                            std::priority_queue<std::pair<float, PID>> connections_to_prune;

                            // Add the new node as a candidate, using the query vector `point`.
                            float new_node_dist = DistanceMetric::compare(neighbor_vector_copy.data(), point, Dim);
                            connections_to_prune.push({new_node_dist, new_node_id});

                            // Add existing neighbors, using the vectors we copied in Phase 1.
                            for (PID conn_id : latest_neighbors) {
                                auto it = connection_vectors.find(conn_id);

                                // Check if we successfully read this connection's vector during Phase 1.
                                // If not, it means the neighbor list changed between our read and write phases.
                                // It's safest to just skip this new, unknown connection in this iteration.
                                // It will be properly linked in other nodes' update operations.
                                if (it != connection_vectors.end()) {
                                    const std::vector<float>& conn_vector = it->second;
                                    float dist =
                                        DistanceMetric::compare(neighbor_vector_copy.data(), conn_vector.data(), Dim);
                                    connections_to_prune.push({dist, conn_id});
                                }
                            }

                            while (connections_to_prune.size() > M_level) {
                                connections_to_prune.pop();
                            }

                            std::vector<u32> new_neighbor_list;
                            while (!connections_to_prune.empty()) {
                                new_neighbor_list.push_back(connections_to_prune.top().second);
                                connections_to_prune.pop();
                            }

                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                            neighbor_acc.setNeighbors(level, new_neighbor_list, this);
                        } else {
                            latest_neighbors.push_back(new_node_id);
                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                            neighbor_acc.setNeighbors(level, latest_neighbors, this);
                        }
                    }  // Single X-Lock is released here.

                    // If the entire try block completes, the operation was successful. Break the retry loop.
                    break;

                } catch (const OLCRestartException&) {
                    // An optimistic lock failed during Phase 1. Another thread modified the data.
                    // We simply `continue` to restart the entire read-then-write process.
                    continue;
                }
            }  // End of retry loop
        }  // End of for-each neighbor loop

        // Update the entry point for the next level down.
        if (!neighbors_to_link_pairs.empty()) {
            enter_point_id = neighbors_to_link_pairs.front().second;
        }
    }

    // --- 5. Update global entry point if the new node is the highest ---
    if (new_node_level > max_l) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        // Re-check inside the lock to handle race conditions.
        if (new_node_level > meta_guard->max_level.load(std::memory_order_acquire)) {
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level = new_node_level;
            meta_guard->dirty = true;
        }
    }
}

// --- Parallel and Single Add/Search Functions (unchanged) ---
template <size_t Dim, typename DistanceMetric>
void HNSW<Dim, DistanceMetric>::addPoint_parallel(std::span<const float> points,
                                                                                   size_t num_threads) {
    if (points.empty()) return;
    if (points.size() % Dim != 0) {
        throw std::invalid_argument("Total number of floats in points span is not a multiple of the vector dimension.");
    }
    size_t num_points = points.size() / Dim;
    u32 start_id;
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        start_id = meta_guard->node_count.load();
        if (start_id + num_points > max_elements_) {
            throw std::runtime_error("Cannot add items; index would exceed max_elements.");
        }
        meta_guard->node_count.store(start_id + num_points);
        meta_guard->dirty = true;
    }
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_points);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_points; ++i) {
            addPoint_internal(points.data() + i * Dim, start_id + i);
        }
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            futures.emplace_back(pool.enqueue([this, points_data = points.data(), i, start_id]() {
                const float* point_vector = points_data + i * Dim;
                u32 node_id = start_id + i;
                this->addPoint_internal(point_vector, node_id);
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
}

template <size_t Dim, typename DistanceMetric>
void HNSW<Dim, DistanceMetric>::addPoint(const float* point, u32& node_id_out) {
    u32 new_node_id;
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        new_node_id = meta_guard->node_count.fetch_add(1);
        if (new_node_id >= max_elements_) {
            meta_guard->node_count.fetch_sub(1);
            throw std::runtime_error("HNSW index is full.");
        }
        meta_guard->dirty = true;
    }
    node_id_out = new_node_id;
    addPoint_internal(point, new_node_id);
}

template <size_t Dim, typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchKnn(
    const float* query, size_t k, size_t ef_search_param) {
    u32 enter_point_id;
    u32 max_l;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break; // Success
        } catch (const OLCRestartException&) {
            continue; // Retry
        }
    }
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) return {};
    if (max_l > 0) enter_point_id = searchBaseLayer(query, enter_point_id, max_l, 1);

    // Use the new parameter for ef_search
    size_t ef_search = ef_search_param;
    if (ef_search == 0) {  // If not provided, use the old default logic
        ef_search = std::max(static_cast<size_t>(efConstruction), k);
    }
    ef_search = std::max(ef_search, k);  // ef must be at least k

    auto candidates = searchLayer(query, enter_point_id, 0, ef_search);
    if (candidates.size() > k) candidates.resize(k);
    return candidates;
}

template <size_t Dim, typename DistanceMetric>
std::vector<std::vector<std::pair<float, u32>>>
HNSW<Dim, DistanceMetric>::searchKnn_parallel(std::span<const float> queries, size_t k,
                                                                               size_t ef_search_param,
                                                                               size_t num_threads) {
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
            all_results[i] = searchKnn(queries.data() + i * Dim, k, ef_search_param);
        }
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            futures.emplace_back(
                pool.enqueue([this, &all_results, queries_data = queries.data(), i, k, ef_search_param]() {
                    all_results[i] = this->searchKnn(queries_data + i * Dim, k, ef_search_param);
                }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    return all_results;
}

template <size_t Dim, typename DistanceMetric>
u64 HNSW<Dim, DistanceMetric>::getBufferManagerAllocCount() const {
    return bm.allocCount.load();
}

template <size_t Dim, typename DistanceMetric>
std::string HNSW<Dim, DistanceMetric>::getIndexInfo() const {
    std::ostringstream info;
    
    info << "=== HNSW Index Information ===\n";
    
    // Template parameters (compile-time configuration)
    info << "Configuration Parameters:\n";
    info << "  Vector Dimension: " << Dim << "\n";
    info << "  Max Neighbors (M): " << M << " (layers > 0)\n";
    info << "  Max Neighbors (M0): " << M0 << " (layer 0)\n";
    info << "  Construction ef: " << efConstruction << "\n";
    info << "  Max Levels: " << MaxLevel << "\n";
    
    // Memory layout information
    info << "\nMemory Layout:\n";
    info << "  Node Vector Size: " << VectorSize << " bytes\n";
    info << "  Max Neighbors Header Size: " << MaxNeighborsHeaderSize << " bytes\n";
    info << "  Max Neighbors List Size: " << MaxNeighborsListSize << " bytes\n";
    info << "  Fixed Node Size: " << FixedNodeSize << " bytes\n";
    info << "  Nodes Per Page: " << NodesPerPage << "\n";
    info << "  Page Size: " << pageSize << " bytes\n";
    
    // Runtime metadata from HNSWMetadataPage
    u32 enter_point_id = HNSWMetadataPage::invalid_node_id;
    u32 max_level = 0;
    u64 node_count = 0;
    u64 max_elements = 0;
    PID base_pid_value = 0;
    
    // Read metadata with retry loop for OLC
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_level = meta_guard->max_level.load(std::memory_order_acquire);
            node_count = meta_guard->node_count.load(std::memory_order_acquire);
            max_elements = meta_guard->max_elements;
            base_pid_value = meta_guard->base_pid;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    
    info << "\nIndex Metadata:\n";
    info << "  Metadata PID: " << metadata_pid << "\n";
    info << "  Base PID: " << base_pid_value << "\n";
    info << "  Max Elements: " << max_elements << "\n";
    info << "  Current Node Count: " << node_count << "\n";
    info << "  Current Max Level: " << max_level << "\n";
    info << "  Entry Point Node ID: ";
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        info << "NONE (index is empty)\n";
    } else {
        info << enter_point_id << "\n";
    }
    
    // Entry point node information (if exists)
    if (enter_point_id != HNSWMetadataPage::invalid_node_id) {
        try {
            GuardO<HNSWPage> entry_page_guard(getNodePID(enter_point_id));
            NodeAccessor entry_acc(entry_page_guard.ptr, getNodeIndexInPage(enter_point_id), this);
            
            info << "\nEntry Point Node Details:\n";
            info << "  Node Level: " << entry_acc.getLevel() << "\n";
            info << "  Node PID: " << getNodePID(enter_point_id) << "\n";
            info << "  Node Index: " << getNodeIndexInPage(enter_point_id) << "\n";
            
            // Show neighbor counts per level
            for (u32 level = 0; level <= entry_acc.getLevel() && level < MaxLevel; ++level) {
                auto neighbors = entry_acc.getNeighbors(level, this);
                info << "  Level " << level << " neighbors: " << neighbors.size() << "\n";
            }
        } catch (const OLCRestartException&) {
            info << "\nEntry Point Node Details: (could not read due to concurrent access)\n";
        }
    }
    
    // Buffer manager statistics
    info << "\nBuffer Manager Statistics:\n";
    info << "  Total Allocations: " << bm.allocCount.load() << "\n";
    info << "  Physical Pages Used: " << bm.physUsedCount.load() << "\n";
    info << "  Read Operations: " << bm.readCount.load() << "\n";
    info << "  Write Operations: " << bm.writeCount.load() << "\n";
    
    // Calculate storage efficiency
    double storage_utilization = 0.0;
    if (max_elements > 0) {
        storage_utilization = (double)node_count / max_elements * 100.0;
    }
    info << "\nStorage Utilization:\n";
    info << "  Index Capacity Usage: " << std::fixed << std::setprecision(2) << storage_utilization << "%\n";
    
    u64 total_allocated_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
    if (total_allocated_pages > 0) {
        double page_utilization = (double)((node_count + NodesPerPage - 1) / NodesPerPage) / total_allocated_pages * 100.0;
        info << "  Page Utilization: " << std::fixed << std::setprecision(2) << page_utilization << "%\n";
    }
    
    info << "\n=== End Index Information ===\n";
    
    return info.str();
}

// --- Explicit Template Instantiation ---
template class HNSW<128, hnsw_distance::SIMDAcceleratedL2>;