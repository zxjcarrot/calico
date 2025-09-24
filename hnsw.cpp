#include "hnsw.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

// --- HNSW Constructor Implementation (unchanged) ---
template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::HNSW(u64 max_elements) {
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
        first_page_guard->data_offset = pageSize;
        first_page_guard->dirty = true;

        for (u64 i = 1; i < total_pages; ++i) {
            AllocGuard<HNSWPage> page_guard;
            page_guard->dirty = false;
            page_guard->node_count = 0;
            page_guard->data_offset = pageSize;
            page_guard->dirty = true;
        }
    }

    this->base_pid = meta_guard->base_pid;
    meta_guard->dirty = true;
}

// --- Helper Functions (searchLayer, searchBaseLayer, etc., mostly unchanged) ---
template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
u32 HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::getRandomLevel() {
    thread_local static std::mt19937 level_generator(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator)) * mult_factor;
    return std::min(static_cast<u32>(r), static_cast<u32>(MaxLevel - 1));
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
u32 HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::searchBaseLayer(const float* query, u32 entry_point_id,
                                                                                int start_level, int end_level) {
    u32 current_entry_point = entry_point_id;
    for (int level = start_level; level >= end_level; --level) {
        auto candidates = searchLayer(query, current_entry_point, level, 1);
        if (!candidates.empty()) {
            current_entry_point = candidates.front().second;
        } else {
            // This can happen in a concurrent environment. It's safer to not change the entry point.
        }
    }
    return current_entry_point;
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::searchLayer(
    const float* query, u32 entry_point_id, u32 level, size_t ef) {
    std::priority_queue<std::pair<float, u32>> top_candidates;
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, std::greater<std::pair<float, u32>>>
        candidate_queue;
    std::unordered_set<u32> visited_nodes;

    // OLC-safe reading of entry point
    try {
        GuardO<HNSWPage> page_guard(getNodePID(entry_point_id));
        NodeAccessor acc(page_guard.ptr, getSlotIndexInPage(entry_point_id));
        float dist = DistanceMetric::compare(query, acc.getVector(), Dim);
        top_candidates.push({dist, entry_point_id});
        candidate_queue.push({dist, entry_point_id});
        visited_nodes.insert(entry_point_id);
    } catch (const OLCRestartException&) {
        // If entry point is being modified, it's safer to just return an empty set and let the caller handle it.
        return {};
    }

    while (!candidate_queue.empty()) {
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        if (top_candidates.size() >= ef && current_pair.first > top_candidates.top().first) {
            break;
        }

        u32 current_id = current_pair.second;
        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id));
            NodeAccessor current_acc(current_page_guard.ptr, getSlotIndexInPage(current_id));

            if (current_acc.getLevel() < level) continue;

            auto neighbors = current_acc.getNeighbors(level);
            for (const PID& neighbor_id : neighbors) {
                if (visited_nodes.find(neighbor_id) == visited_nodes.end()) {
                    visited_nodes.insert(neighbor_id);
                    GuardO<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id));
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getSlotIndexInPage(neighbor_id));
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
            // A node we were about to visit was modified. We can safely skip it and continue the search with the
            // remaining candidates.
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
    return results;
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
PID HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::getNodePID(u32 node_id) const {
    return base_pid + (node_id / NodesPerPage);
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
u32 HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::getSlotIndexInPage(u32 node_id) const {
    return node_id % NodesPerPage;
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::selectNeighborsHeuristic(
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
                NodeAccessor candidate_acc(candidate_page_guard.ptr, getSlotIndexInPage(current_candidate.second));
                const float* candidate_vector = candidate_acc.getVector();

                GuardO<HNSWPage> selected_page_guard(getNodePID(selected.second));
                NodeAccessor selected_acc(selected_page_guard.ptr, getSlotIndexInPage(selected.second));
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

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
void HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::addPoint_internal(const float* point,
                                                                                   u32 new_node_id) {
    const u32 new_node_level = getRandomLevel();
    PID new_node_pid = getNodePID(new_node_id);

    // --- 1. Allocate space for the new node on its page ---
    // It prepares the node's storage.
    {
        GuardX<HNSWPage> page_guard(new_node_pid);
        HNSWPage* page = page_guard.ptr;
        u32 slot_idx = getSlotIndexInPage(new_node_id);
        HNSWNodeSlot* slot = &page->slots[slot_idx];

        size_t neighbors_header_size = MaxLevel * sizeof(u16);
        size_t neighbors_list_size = (M0 + (MaxLevel - 1) * M) * sizeof(PID);
        size_t required_space = VectorSize + neighbors_header_size + neighbors_list_size;

        if (page->data_offset < HNSWPage::HeaderSize + (page->node_count + 1) * sizeof(HNSWNodeSlot) + required_space) {
            throw std::runtime_error("HNSW page is full. This should not happen with pre-calculation.");
        }
        u16 new_offset = page->data_offset - required_space;
        page->data_offset = new_offset;

        u8* heap_ptr = reinterpret_cast<u8*>(page) + new_offset;
        slot->level = new_node_level;
        slot->vector_offset = new_offset;
        slot->neighbors_offset = new_offset + VectorSize;

        memcpy(heap_ptr, point, VectorSize);
        memset(heap_ptr + VectorSize, 0, neighbors_header_size + neighbors_list_size);

        page->node_count++;
        page->dirty = true;
    }

    // --- 2. Find entry point for the search ---
    u32 enter_point_id;
    u32 max_l;
    {
        GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
        enter_point_id = meta_guard->enter_point_node_id;
        max_l = meta_guard->max_level.load(std::memory_order_acquire);
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
            MutableNodeAccessor new_node_acc(new_node_page_guard.ptr, getSlotIndexInPage(new_node_id));
            std::vector<PID> neighbor_ids;
            neighbor_ids.reserve(neighbors_to_link_pairs.size());
            for (const auto& p : neighbors_to_link_pairs) {
                neighbor_ids.push_back(p.second);
            }
            new_node_acc.setNeighbors(level, neighbor_ids);
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
                    std::vector<PID> current_neighbors;
                    std::unordered_map<PID, std::vector<float>> connection_vectors;  // Stores vectors for pruning
                    bool needs_pruning;

                    // Step 1.1: Optimistically read the neighbor's primary data (its vector and its neighbor list).
                    {
                        GuardO<HNSWPage> neighbor_page_guard(neighbor_pid);
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getSlotIndexInPage(neighbor_id));

                        const float* neighbor_vec_ptr = neighbor_acc.getVector();
                        std::copy(neighbor_vec_ptr, neighbor_vec_ptr + Dim, neighbor_vector_copy.begin());

                        auto neighbors_span = neighbor_acc.getNeighbors(level);
                        current_neighbors.assign(neighbors_span.begin(), neighbors_span.end());
                    }

                    needs_pruning = (current_neighbors.size() >= M_level);

                    // Step 1.2: If pruning is necessary, we must also read the vectors of all its connections.
                    // This is the crucial step to avoid needing to read them inside the write lock.
                    if (needs_pruning) {
                        for (PID conn_id : current_neighbors) {
                            GuardO<HNSWPage> conn_page_guard(getNodePID(conn_id));
                            NodeAccessor conn_acc(conn_page_guard.ptr, getSlotIndexInPage(conn_id));
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
                        NodeAccessor locked_reader(neighbor_page_guard.ptr, getSlotIndexInPage(neighbor_id));
                        auto latest_neighbors_span = locked_reader.getNeighbors(level);
                        std::vector<PID> latest_neighbors(latest_neighbors_span.begin(), latest_neighbors_span.end());

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

                            std::vector<PID> new_neighbor_list;
                            while (!connections_to_prune.empty()) {
                                new_neighbor_list.push_back(connections_to_prune.top().second);
                                connections_to_prune.pop();
                            }

                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getSlotIndexInPage(neighbor_id));
                            neighbor_acc.setNeighbors(level, new_neighbor_list);
                        } else {
                            latest_neighbors.push_back(new_node_id);
                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getSlotIndexInPage(neighbor_id));
                            neighbor_acc.setNeighbors(level, latest_neighbors);
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
template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
void HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::addPoint_parallel(std::span<const float> points,
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

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
void HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::addPoint(const float* point, u32& node_id_out) {
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

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::searchKnn(
    const float* query, size_t k, size_t ef_search_param) {

    u32 enter_point_id;
    u32 max_l;
    {
        GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
        enter_point_id = meta_guard->enter_point_node_id;
        max_l = meta_guard->max_level.load(std::memory_order_acquire);
    }
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) return {};
    if (max_l > 0) enter_point_id = searchBaseLayer(query, enter_point_id, max_l, 1);
    
    // Use the new parameter for ef_search
    size_t ef_search = ef_search_param;
    if (ef_search == 0) { // If not provided, use the old default logic
        ef_search = std::max(static_cast<size_t>(efConstruction), k);
    }
    ef_search = std::max(ef_search, k); // ef must be at least k

    auto candidates = searchLayer(query, enter_point_id, 0, ef_search);
    if (candidates.size() > k) candidates.resize(k);
    return candidates;
}

template <size_t Dim, typename DistanceMetric, size_t M, size_t M0, size_t efConstruction, size_t MaxLevel>
std::vector<std::vector<std::pair<float, u32>>>
HNSW<Dim, DistanceMetric, M, M0, efConstruction, MaxLevel>::searchKnn_parallel(std::span<const float> queries, size_t k, size_t ef_search_param,
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
            all_results[i] = searchKnn(queries.data() + i * Dim, k);
        }
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            futures.emplace_back(pool.enqueue([this, &all_results, queries_data = queries.data(), i, k]() {
                all_results[i] = this->searchKnn(queries_data + i * Dim, k, ef_search_param);
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    return all_results;
}

// --- Explicit Template Instantiation ---
template class HNSW<128, hnsw_distance::SIMDAcceleratedL2, 16, 32, 200, 16>;