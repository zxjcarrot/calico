#include "hnsw.hpp"

#include <immintrin.h>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

// --- A Simple Thread Pool Implementation ---
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
    if (max_elements <= 1) return 1;
    double expected_max_level = std::log(static_cast<double>(max_elements)) / std::log(static_cast<double>(M));
    size_t estimated_max_level = static_cast<size_t>(std::ceil(expected_max_level + 1));
    return std::max(static_cast<size_t>(3), std::min(static_cast<size_t>(16), estimated_max_level));
}

// --- NEW: Helper to calculate BaseNodeSize ---
template <size_t Dim, typename DistanceMetric>
size_t HNSW<Dim, DistanceMetric>::calculateBaseNodeSize(size_t M0_param) {
    // Layout: [Vector][Level][L0_Count][OverflowPointer][L0_Neighbors]
    return VectorSize + sizeof(u16)     // level
           + sizeof(u16)                // l0_count
           + sizeof(OverflowPointer)    // overflow_ptr
           + (M0_param * sizeof(u32));  // l0_neighbors
}

// --- HNSW Constructor Implementation (MODIFIED) ---
template <size_t Dim, typename DistanceMetric>
HNSW<Dim, DistanceMetric>::HNSW(u64 max_elements, size_t M_param, size_t ef_construction_param,
                                bool enable_prefetch_param)
    : M(M_param),
      M0(2 * M_param),
      efConstruction(ef_construction_param),
      MaxLevel(estimateMaxLevel(max_elements, M_param)),
      BaseNodeSize(calculateBaseNodeSize(2 * M_param)),
      LinkSizePerLevel(sizeof(u16) + M_param * sizeof(u32)),  // count + neighbors
      MaxNodesPerBasePage((pageSize - BaseNodePage::HeaderSize) / BaseNodeSize),
      enable_prefetch_(enable_prefetch_param) {
    if (MaxNodesPerBasePage == 0) {
        throw std::runtime_error(
            "Page size is too small to fit even one node. Consider reducing M or vector dimension.");
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
    // NEW: Initialize overflow allocation metadata
    meta_guard->last_overflow_pid = OverflowPointer::INVALID_PID;
    meta_guard->last_overflow_offset = 0;

    u64 total_pages = (max_elements + MaxNodesPerBasePage - 1) / MaxNodesPerBasePage;
    if (total_pages > 0) {
        AllocGuard<BaseNodePage> first_page_guard;
        meta_guard->base_pid = first_page_guard.pid;
        first_page_guard->dirty = false;
        first_page_guard->node_count = 0;
        first_page_guard->dirty = true;

        for (u64 i = 1; i < total_pages; ++i) {
            AllocGuard<BaseNodePage> page_guard;
            page_guard->dirty = false;
            page_guard->node_count = 0;
            page_guard->dirty = true;
        }
    }

    this->base_pid = meta_guard->base_pid;
    meta_guard->dirty = true;
}

// --- Location helpers ---
template <size_t Dim, typename DistanceMetric>
inline PID HNSW<Dim, DistanceMetric>::getNodePID(u32 node_id) const {
    return base_pid + (node_id / MaxNodesPerBasePage);
}

template <size_t Dim, typename DistanceMetric>
inline u32 HNSW<Dim, DistanceMetric>::getNodeIndexInPage(u32 node_id) const {
    return node_id % MaxNodesPerBasePage;
}

// --- NEW: Overflow space allocator ---
template <size_t Dim, typename DistanceMetric>
OverflowPointer HNSW<Dim, DistanceMetric>::allocateOverflowSpace(size_t required_size) {
    GuardX<HNSWMetadataPage> meta_guard(metadata_pid);

    if (meta_guard->last_overflow_pid == OverflowPointer::INVALID_PID ||
        (meta_guard->last_overflow_offset + required_size) > (pageSize - OverflowPage::HeaderSize)) {
        AllocGuard<OverflowPage> new_overflow_page_guard;
        new_overflow_page_guard->dirty = true;
        new_overflow_page_guard->free_space_offset = 0;

        meta_guard->last_overflow_pid = new_overflow_page_guard.pid;
        meta_guard->last_overflow_offset = 0;
        meta_guard->dirty = true;
    }

    PID target_pid = meta_guard->last_overflow_pid;
    u16 target_offset = meta_guard->last_overflow_offset;

    meta_guard->last_overflow_offset += required_size;

    // Update the free space offset on the overflow page itself for recovery purposes.
    {
        GuardX<OverflowPage> overflow_page_guard(target_pid);
        overflow_page_guard->free_space_offset = meta_guard->last_overflow_offset;
        overflow_page_guard->dirty = true;
    }

    meta_guard->dirty = true;

    return {target_pid, target_offset};
}

template <size_t Dim, typename DistanceMetric>
u32 HNSW<Dim, DistanceMetric>::getRandomLevel() {
    thread_local static std::mt19937 level_generator(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator)) * mult_factor;
    return std::min(static_cast<u32>(r), static_cast<u32>(MaxLevel - 1));
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchLayer(
    const float* query, u32 entry_point_id, u32 level, size_t ef,
    std::optional<std::pair<float, u32>> initial_entry_dist_pair) {
    
    // OLC Restart Loop: If any GuardO_NoThrow::check() fails, we restart the whole search.
    for (;;) {
        try {
            VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
            std::unique_ptr<VisitedList, std::function<void(VisitedList*)>> visited_guard(
                visited_nodes, [&](VisitedList* p) { this->visited_list_pool_->releaseVisitedList(p); });
            vl_type* visited_array = visited_nodes->mass;
            vl_type visited_array_tag = visited_nodes->curV;

            std::priority_queue<std::pair<float, u32>> top_candidates;
            std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, std::greater<std::pair<float, u32>>>
                candidate_queue;

            if (initial_entry_dist_pair && initial_entry_dist_pair->second == entry_point_id) {
                float dist = initial_entry_dist_pair->first;
                top_candidates.push({dist, entry_point_id});
                candidate_queue.push({dist, entry_point_id});
                visited_array[entry_point_id] = visited_array_tag;
            } else {
                GuardO_NoThrow<BaseNodePage> page_guard(getNodePID(entry_point_id));
                NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(entry_point_id), this);
                float dist = stats ? this->calculateDistance(query, acc.getVector(), Dim)
                                   : DistanceMetric::compare(query, acc.getVector(), Dim);
                page_guard.check(); // Check after use
                top_candidates.push({dist, entry_point_id});
                candidate_queue.push({dist, entry_point_id});
                visited_array[entry_point_id] = visited_array_tag;
            }

            std::vector<PID> pages_to_prefetch;
            std::vector<u32> offsets_within_pages;
            pages_to_prefetch.reserve(M0);
            offsets_within_pages.reserve(M0);

            while (!candidate_queue.empty()) {
                auto current_pair = candidate_queue.top();
                candidate_queue.pop();
                if (top_candidates.size() >= ef && current_pair.first > top_candidates.top().first) {
                    break;
                }
                if (stats) stats_.search_hops.fetch_add(1, std::memory_order_relaxed);

                u32 current_id = current_pair.second;
                
                GuardO_NoThrow<BaseNodePage> current_base_page_guard(getNodePID(current_id));
                NodeAccessor current_acc(current_base_page_guard.ptr, getNodeIndexInPage(current_id), this);

                std::span<const u32> neighbors_span;
                std::optional<GuardO_NoThrow<OverflowPage>> overflow_page_guard;

                if (level == 0) {
                    neighbors_span = current_acc.getL0Neighbors();
                } else {
                    OverflowPointer ptr = current_acc.getOverflowPointer();
                    if (ptr.isValid()) {
                        overflow_page_guard.emplace(ptr.page_id);
                        neighbors_span = current_acc.getHigherLevelNeighbors(level, overflow_page_guard->ptr);
                    }
                }
                
                // Explicitly check versions after we have all the data we need from this node
                current_base_page_guard.check();
                if (overflow_page_guard.has_value()) {
                    overflow_page_guard->check();
                }
                
                if (level <= 2 && !neighbors_span.empty() && enable_prefetch_) {
                    pages_to_prefetch.clear(); offsets_within_pages.clear();
                    for (const u32& neighbor_id : neighbors_span) {
                        if (!(visited_array[neighbor_id] == visited_array_tag)) {
                            pages_to_prefetch.push_back(getNodePID(neighbor_id));
                            offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id));
                        }
                    }
                    bm.prefetchPages(pages_to_prefetch.data(), pages_to_prefetch.size(), offsets_within_pages.data());
                }

                for (const u32& neighbor_id : neighbors_span) {
                    if (!(visited_array[neighbor_id] == visited_array_tag)) {
                        visited_array[neighbor_id] = visited_array_tag;
                        GuardO_NoThrow<BaseNodePage> neighbor_page_guard(getNodePID(neighbor_id));
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        float neighbor_dist = stats ? this->calculateDistance(query, neighbor_acc.getVector(), Dim)
                                                    : DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                        neighbor_page_guard.check();

                        if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                            candidate_queue.push({neighbor_dist, neighbor_id});
                            top_candidates.push({neighbor_dist, neighbor_id});
                            if (top_candidates.size() > ef) {
                                top_candidates.pop();
                            }
                        }
                    }
                }
            }

            std::vector<std::pair<float, u32>> results;
            results.reserve(top_candidates.size());
            while (!top_candidates.empty()) {
                results.push_back(top_candidates.top());
                top_candidates.pop();
            }
            std::reverse(results.begin(), results.end());
            return results; // Success
        } catch (const OLCRestartException&) {
            continue; // Restart the entire function
        }
    }
}

// --- addPoint and Search Entry Points) ---

template <size_t Dim, typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::selectNeighborsHeuristic(
    const float* query, const std::vector<std::pair<float, u32>>& candidates, size_t M_limit) {
    if (candidates.size() <= M_limit) {
        return candidates;
    }

    for (;;) { // OLC Restart Loop
        try {
            std::vector<std::pair<float, u32>> result;
            result.reserve(M_limit);
            for (const auto& current_candidate : candidates) {
                if (result.size() >= M_limit) break;
                bool is_good_candidate = true;
                for (const auto& selected : result) {
                    GuardO_NoThrow<BaseNodePage> candidate_page_guard(getNodePID(current_candidate.second));
                    NodeAccessor candidate_acc(candidate_page_guard.ptr, getNodeIndexInPage(current_candidate.second), this);
                    
                    GuardO_NoThrow<BaseNodePage> selected_page_guard(getNodePID(selected.second));
                    NodeAccessor selected_acc(selected_page_guard.ptr, getNodeIndexInPage(selected.second), this);

                    // Read vectors first
                    const float* candidate_vec = candidate_acc.getVector();
                    const float* selected_vec = selected_acc.getVector();

                    // Then check
                    candidate_page_guard.check();
                    selected_page_guard.check();
                    
                    if (DistanceMetric::compare(candidate_vec, selected_vec, Dim) < current_candidate.first) {
                        is_good_candidate = false;
                        break;
                    }
                }
                if (is_good_candidate) {
                    result.push_back(current_candidate);
                }
            }
            return result; // Success
        } catch (const OLCRestartException&) {
            continue; // Restart the entire heuristic
        }
    }
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<Dim, DistanceMetric>::findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level, float entry_point_dist) {
    for (;;) { // OLC Restart Loop
        try {
            u32 current_node_id = entry_point_id;
            float min_dist = entry_point_dist; 
            
            VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
            std::unique_ptr<VisitedList, std::function<void(VisitedList*)>> visited_guard(
                visited_nodes, [&](VisitedList* p) { this->visited_list_pool_->releaseVisitedList(p); });
            vl_type* visited_array = visited_nodes->mass;
            vl_type visited_array_tag = visited_nodes->curV;

            std::vector<PID> pages_to_prefetch;
            std::vector<u32> offsets_within_pages;
            pages_to_prefetch.reserve(M0);
            offsets_within_pages.reserve(M0);

            bool changed = true;
            while (changed) {
                changed = false;
                u32 best_neighbor_id = current_node_id;
                
                GuardO_NoThrow<BaseNodePage> current_base_page_guard(getNodePID(current_node_id));
                NodeAccessor current_acc(current_base_page_guard.ptr, getNodeIndexInPage(current_node_id), this);

                visited_array[current_node_id] = visited_array_tag;
                if (stats) stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
                
                std::span<const u32> neighbors_span;
                std::optional<GuardO_NoThrow<OverflowPage>> overflow_page_guard;
                if (level == 0) {
                    neighbors_span = current_acc.getL0Neighbors();
                } else {
                    OverflowPointer ptr = current_acc.getOverflowPointer();
                    if (ptr.isValid()) {
                        overflow_page_guard.emplace(ptr.page_id);
                        neighbors_span = current_acc.getHigherLevelNeighbors(level, overflow_page_guard->ptr);
                    }
                }

                current_base_page_guard.check();
                if (overflow_page_guard.has_value()) {
                    overflow_page_guard->check();
                }
                
                if (!neighbors_span.empty() && enable_prefetch_) {
                    pages_to_prefetch.clear(); offsets_within_pages.clear();
                    for (const u32& neighbor_id : neighbors_span) {
                        if (!(visited_array[neighbor_id] == visited_array_tag)) {
                            pages_to_prefetch.push_back(getNodePID(neighbor_id));
                            offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id));
                        }
                    }
                    bm.prefetchPages(pages_to_prefetch.data(), pages_to_prefetch.size(), offsets_within_pages.data());
                }

                for (const u32& neighbor_id : neighbors_span) {
                    if (!(visited_array[neighbor_id] == visited_array_tag)) {
                        visited_array[neighbor_id] = visited_array_tag;
                        GuardO_NoThrow<BaseNodePage> neighbor_page_guard(getNodePID(neighbor_id));
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        float neighbor_dist = stats ? this->calculateDistance(query, neighbor_acc.getVector(), Dim)
                                                    : DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                        neighbor_page_guard.check();

                        if (neighbor_dist < min_dist) {
                            min_dist = neighbor_dist;
                            best_neighbor_id = neighbor_id;
                            changed = true;
                        }
                    }
                }
                current_node_id = best_neighbor_id;
            }
            return {min_dist, current_node_id}; // Success
        } catch (const OLCRestartException&) {
            continue; // Restart
        }
    }
}

// searchBaseLayer and searchKnn are high-level and need no changes as they rely on the refactored helpers.
template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<Dim, DistanceMetric>::searchBaseLayer(const float* query, u32 entry_point_id,
                                                                 int start_level, int end_level) {
    for (;;) { // OLC Restart Loop
        try {
            u32 current_entry_point_id = entry_point_id;
            float current_dist;

            GuardO_NoThrow<BaseNodePage> initial_guard(getNodePID(current_entry_point_id));
            NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(current_entry_point_id), this);
            current_dist = stats ? this->calculateDistance(query, initial_acc.getVector(), Dim)
                                 : DistanceMetric::compare(query, initial_acc.getVector(), Dim);
            initial_guard.check();

            for (int level = start_level; level >= end_level; --level) {
                // The callee is now robust and will retry internally until it succeeds.
                auto result = findBestEntryPointForLevel<stats>(query, current_entry_point_id, level, current_dist);
                current_dist = result.first;
                current_entry_point_id = result.second;
            }
            return {current_dist, current_entry_point_id}; // Success
        } catch (const OLCRestartException&) {
            continue; // Restart
        }
    }
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<Dim, DistanceMetric>::searchKnn(const float* query, size_t k,
                                                                        size_t ef_search_param) {
    u32 enter_point_id;
    u32 max_l;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) return {};
    std::pair<float, u32> entry_point_for_layer0 = {0.0f, enter_point_id};
    if (max_l > 0) {
        entry_point_for_layer0 = searchBaseLayer<stats>(query, enter_point_id, max_l, 1);
    }
    size_t ef_search = (ef_search_param == 0) ? std::max(static_cast<size_t>(efConstruction), k) : ef_search_param;
    ef_search = std::max(ef_search, k);
    auto candidates = searchLayer<stats>(query, entry_point_for_layer0.second, 0, ef_search, entry_point_for_layer0);
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

// --- addPoint_internal  ---
template <size_t Dim, typename DistanceMetric>
void HNSW<Dim, DistanceMetric>::addPoint_internal(const float* point, u32 new_node_id) {
    // --- OLC Transactional Loop ---
    // The entire insertion process is one transaction. If any optimistic read fails
    // at any point (throws OLCRestartException), we restart the whole process.
    // This resolves the livelock by failing fast instead of spinning.
    for (;;) {
        try {
            const u32 new_node_level = getRandomLevel();
            PID new_node_pid = getNodePID(new_node_id);

            // Phase 1: Initialize the new node's data.
            {
                GuardX<BaseNodePage> page_guard(new_node_pid);
                MutableNodeAccessor new_node_acc(page_guard.ptr, getNodeIndexInPage(new_node_id), this);
                memcpy(new_node_acc.getVector(), point, VectorSize);
                new_node_acc.setLevel(new_node_level);
                new_node_acc.setL0Count(0);
                new_node_acc.setOverflowPointer({OverflowPointer::INVALID_PID, 0});
                if (new_node_level > 0) {
                    size_t required_size = new_node_level * LinkSizePerLevel;
                    OverflowPointer ptr = allocateOverflowSpace(required_size);
                    new_node_acc.setOverflowPointer(ptr);
                    GuardX<OverflowPage> overflow_guard(ptr.page_id);
                    u8* block_start = overflow_guard.ptr->getDataArea() + ptr.offset;
                    memset(block_start, 0, required_size);
                    overflow_guard.ptr->dirty = true;
                }
                u32 node_idx_in_page = getNodeIndexInPage(new_node_id);
                page_guard.ptr->node_count = std::max(page_guard.ptr->node_count, static_cast<u16>(node_idx_in_page + 1));
                page_guard.ptr->dirty = true;
            }

            // Phase 2: Optimistically read the graph's entry point.
            u32 enter_point_id;
            u32 max_l;
            GuardO_NoThrow<HNSWMetadataPage> meta_guard_read(metadata_pid);
            enter_point_id = meta_guard_read->enter_point_node_id;
            max_l = meta_guard_read->max_level.load(std::memory_order_acquire);
            meta_guard_read.check();

            if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
                GuardX<HNSWMetadataPage> meta_guard_write(metadata_pid);
                if (meta_guard_write->enter_point_node_id == HNSWMetadataPage::invalid_node_id) {
                    meta_guard_write->enter_point_node_id = new_node_id;
                    meta_guard_write->max_level = new_node_level;
                    meta_guard_write->dirty = true;
                }
                return;
            }
            
            // Phase 3: The long search phase.
            std::optional<std::pair<float, u32>> entry_point_with_dist_opt = std::nullopt;
            if (new_node_level < max_l) {
                auto result = searchBaseLayer(point, enter_point_id, max_l, new_node_level + 1);
                enter_point_id = result.second;
                entry_point_with_dist_opt = result;
            }

            // Phase 4: Connection phase.
            for (int level = std::min(static_cast<u32>(new_node_level), max_l); level >= 0; --level) {
                auto candidates = searchLayer(point, enter_point_id, level, efConstruction, entry_point_with_dist_opt);
                if (entry_point_with_dist_opt.has_value()) entry_point_with_dist_opt = std::nullopt;
                if (candidates.empty()) continue;

                size_t M_level = (level == 0) ? M0 : M;
                auto neighbors_to_link_pairs = selectNeighborsHeuristic(point, candidates, M_level);
                
                {
                    std::vector<u32> neighbor_ids;
                    neighbor_ids.reserve(neighbors_to_link_pairs.size());
                    for (const auto& p : neighbors_to_link_pairs) neighbor_ids.push_back(p.second);
                    GuardX<BaseNodePage> new_node_base_page_guard(new_node_pid);
                    MutableNodeAccessor new_node_acc(new_node_base_page_guard.ptr, getNodeIndexInPage(new_node_id), this);
                    if (level == 0) {
                        new_node_acc.setL0Neighbors(neighbor_ids);
                    } else {
                        OverflowPointer ptr = new_node_acc.getOverflowPointer();
                        GuardX<OverflowPage> overflow_page_guard(ptr.page_id);
                        new_node_acc.setHigherLevelNeighbors(level, neighbor_ids, overflow_page_guard.ptr);
                    }
                }
                
                for (const auto& neighbor_pair : neighbors_to_link_pairs) {
                    const u32 neighbor_id = neighbor_pair.second;
                    const float dist_to_new_node = neighbor_pair.first;
                    
                    // The inner OLC loop for updating a single neighbor
                    for (;;) {
                        try {
                            std::vector<u32> current_neighbors_ids;
                            OverflowPointer neighbor_overflow_ptr;
                            std::vector<float> neighbor_vector_copy(Dim);
                            {
                                GuardO_NoThrow<BaseNodePage> neighbor_base_page_guard(getNodePID(neighbor_id));
                                NodeAccessor neighbor_acc(neighbor_base_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                                const float* vec_ptr = neighbor_acc.getVector();
                                std::copy(vec_ptr, vec_ptr + Dim, neighbor_vector_copy.begin());
                                std::optional<GuardO_NoThrow<OverflowPage>> overflow_page_guard;
                                if (level == 0) {
                                    current_neighbors_ids.assign(neighbor_acc.getL0Neighbors().begin(), neighbor_acc.getL0Neighbors().end());
                                } else {
                                    neighbor_overflow_ptr = neighbor_acc.getOverflowPointer();
                                    if (neighbor_overflow_ptr.isValid()) {
                                        overflow_page_guard.emplace(neighbor_overflow_ptr.page_id);
                                        auto span = neighbor_acc.getHigherLevelNeighbors(level, overflow_page_guard->ptr);
                                        current_neighbors_ids.assign(span.begin(), span.end());
                                    }
                                }
                                neighbor_base_page_guard.check();
                                if (overflow_page_guard.has_value()) overflow_page_guard->check();
                            }
                            
                            std::vector<u32> new_neighbor_list;
                            if (current_neighbors_ids.size() < M_level) {
                                new_neighbor_list = current_neighbors_ids;
                                new_neighbor_list.push_back(new_node_id);
                            } else {
                                std::vector<std::pair<float, u32>> candidates_prune;
                                candidates_prune.reserve(current_neighbors_ids.size() + 1);
                                candidates_prune.push_back({dist_to_new_node, new_node_id});
                                for (u32 conn_id : current_neighbors_ids) {
                                    GuardO_NoThrow<BaseNodePage> conn_page_guard(getNodePID(conn_id));
                                    NodeAccessor conn_acc(conn_page_guard.ptr, getNodeIndexInPage(conn_id), this);
                                    float dist = DistanceMetric::compare(neighbor_vector_copy.data(), conn_acc.getVector(), Dim);
                                    conn_page_guard.check();
                                    candidates_prune.push_back({dist, conn_id});
                                }
                                auto pruned_pairs = selectNeighborsHeuristic(neighbor_vector_copy.data(), candidates_prune, M_level);
                                new_neighbor_list.reserve(pruned_pairs.size());
                                for (const auto& p : pruned_pairs) new_neighbor_list.push_back(p.second);
                            }
                            
                            if (level == 0) {
                                GuardX<BaseNodePage> neighbor_base_page_guard(getNodePID(neighbor_id));
                                MutableNodeAccessor neighbor_acc(neighbor_base_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                                neighbor_acc.setL0Neighbors(new_neighbor_list);
                            } else {
                                if (neighbor_overflow_ptr.isValid()) {
                                    GuardX<OverflowPage> overflow_page_guard(neighbor_overflow_ptr.page_id);
                                    GuardO<BaseNodePage> temp_base_guard(getNodePID(neighbor_id));
                                    MutableNodeAccessor temp_acc(temp_base_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                                    temp_acc.setHigherLevelNeighbors(level, new_neighbor_list, overflow_page_guard.ptr);
                                }
                            }
                            break; 
                        } catch (const OLCRestartException&) {
                            // If this small part fails, just retry it. If it throws up to the top level,
                            // the whole transaction will restart, which is also safe.
                            continue;
                        }
                    }
                }
                if (!neighbors_to_link_pairs.empty()) {
                    enter_point_id = neighbors_to_link_pairs.front().second;
                }
            }
            
            // Phase 5: Final update of the global entry point.
            if (new_node_level > max_l) {
                GuardX<HNSWMetadataPage> meta_guard_write(metadata_pid);
                if (new_node_level > meta_guard_write->max_level.load(std::memory_order_acquire)) {
                    meta_guard_write->enter_point_node_id = new_node_id;
                    meta_guard_write->max_level = new_node_level;
                    meta_guard_write->dirty = true;
                }
            }
            
            return; // Transaction successful
        } catch (const OLCRestartException&) {
            continue; // Restart entire transaction
        }
    }
}

// --- Parallel and Single Add/Search Functions ---
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
void HNSW<Dim, DistanceMetric>::addPoint_parallel(std::span<const float> points, size_t num_threads) {
    if (points.empty()) return;
    if (points.size() % Dim != 0) throw std::invalid_argument("Invalid points span size.");
    size_t num_points = points.size() / Dim;
    u32 start_id;
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        start_id = meta_guard->node_count.load();
        if (start_id + num_points > max_elements_)
            throw std::runtime_error("Cannot add items; index would exceed max_elements.");
        meta_guard->node_count.store(start_id + num_points);
        meta_guard->dirty = true;
    }
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_points);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_points; ++i) addPoint_internal(points.data() + i * Dim, start_id + i);
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        std::atomic<size_t> point_counter{0};
        for (size_t i = 0; i < threads_to_use; ++i) {
            futures.emplace_back(
                pool.enqueue([this, points_data = points.data(), start_id, num_points, &point_counter]() {
                    size_t point_idx;
                    while ((point_idx = point_counter.fetch_add(1, std::memory_order_relaxed)) < num_points) {
                        this->addPoint_internal(points_data + point_idx * Dim, start_id + point_idx);
                        if (point_idx % (num_points / 10 + 1) == 0) {
                            std::cout << "Added point " << point_idx + 1 << " / " << num_points << std::endl;
                        }
                    }
                }));
        }
        for (auto& future : futures) future.get();
    }
}

template <size_t Dim, typename DistanceMetric>
template <bool stats>
std::vector<std::vector<std::pair<float, u32>>> HNSW<Dim, DistanceMetric>::searchKnn_parallel(
    std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads) {
    if (queries.empty()) return {};
    if (queries.size() % Dim != 0) throw std::invalid_argument("Invalid query span size.");
    size_t num_queries = queries.size() / Dim;
    std::vector<std::vector<std::pair<float, u32>>> all_results(num_queries);
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_queries);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            all_results[i] = searchKnn<stats>(queries.data() + i * Dim, k, ef_search_param);
        }
    } else {
        ThreadPool pool(threads_to_use);
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < num_queries; ++i) {
            futures.emplace_back(
                pool.enqueue([this, &all_results, queries_data = queries.data(), i, k, ef_search_param]() {
                    all_results[i] = this->searchKnn<stats>(queries_data + i * Dim, k, ef_search_param);
                }));
        }
        for (auto& future : futures) future.get();
    }
    return all_results;
}

// --- Stats and Info Functions ---
template <size_t Dim, typename DistanceMetric>
u64 HNSW<Dim, DistanceMetric>::getBufferManagerAllocCount() const {
    return bm.allocCount.load();
}

template <size_t Dim, typename DistanceMetric>
HNSWStats HNSW<Dim, DistanceMetric>::getStats() const {
    HNSWStats stats;

    // 1. Copy live performance counters (this is fast)
    stats.dist_comps.store(this->stats_.dist_comps.load());
    stats.search_hops.store(this->stats_.search_hops.load());

    // 2. Read metadata to get overall graph dimensions
    u64 node_count = 0;
    u32 max_l = 0;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            node_count = meta_guard->node_count.load(std::memory_order_acquire);
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    if (node_count == 0) {
        return stats;  // Return early for an empty index
    }

    stats.num_levels = max_l + 1;
    stats.nodes_per_level.assign(stats.num_levels, 0);
    stats.links_per_level.assign(stats.num_levels, 0);
    stats.avg_neighbors_per_level.assign(stats.num_levels, 0.0);

    // 3. Iterate through all nodes to calculate structural stats
    u64 total_links = 0;
    PID current_pid = -1;
    std::optional<GuardO<BaseNodePage>> base_page_guard;

    for (u64 node_id = 0; node_id < node_count; ++node_id) {
        try {
            PID pid = getNodePID(node_id);
            if (pid != current_pid) {
                base_page_guard.emplace(pid);
                current_pid = pid;
            }

            NodeAccessor node_acc(base_page_guard->ptr, getNodeIndexInPage(node_id), this);
            u32 node_level = node_acc.getLevel();

            for (u32 level = 0; level <= node_level; ++level) {
                stats.nodes_per_level[level]++;

                size_t neighbor_count = 0;
                if (level == 0) {
                    neighbor_count = node_acc.getL0Neighbors().size();
                } else {
                    OverflowPointer ptr = node_acc.getOverflowPointer();
                    if (ptr.isValid()) {
                        GuardO<OverflowPage> overflow_guard(ptr.page_id);
                        neighbor_count = node_acc.getHigherLevelNeighbors(level, overflow_guard.ptr).size();
                    }
                }
                stats.links_per_level[level] += neighbor_count;
                total_links += neighbor_count;
            }
        } catch (const OLCRestartException&) {
            // If a page is modified while we're reading, we might skip a node.
            // For stats, this is usually an acceptable trade-off to avoid blocking.
            continue;
        }
    }

    // 4. Calculate averages
    u64 total_nodes = node_count;
    stats.avg_neighbors_total = (total_nodes > 0) ? static_cast<double>(total_links) / total_nodes : 0.0;

    for (u32 level = 0; level < stats.num_levels; ++level) {
        if (stats.nodes_per_level[level] > 0) {
            stats.avg_neighbors_per_level[level] =
                static_cast<double>(stats.links_per_level[level]) / stats.nodes_per_level[level];
        }
    }

    return stats;
}

template <size_t Dim, typename DistanceMetric>
std::string HNSW<Dim, DistanceMetric>::getIndexInfo() const {
    std::ostringstream info;
    info << "=== HNSW Index Information ===\n";
    info << "Configuration Parameters:\n";
    info << "  Vector Dimension: " << Dim << "\n";
    info << "  Max Neighbors (M): " << M << " (layers > 0)\n";
    info << "  Max Neighbors (M0): " << M0 << " (layer 0)\n";
    info << "  Construction ef: " << efConstruction << "\n";
    info << "  Max Levels Cap: " << MaxLevel << "\n";

    info << "\nMemory Layout:\n";
    info << "  Vector Size: " << VectorSize << " bytes\n";
    info << "  Base Node Size: " << BaseNodeSize << " bytes\n";
    info << "  Link Size Per Level (Overflow): " << LinkSizePerLevel << " bytes\n";
    info << "  Nodes Per Base Page: " << MaxNodesPerBasePage << "\n";
    info << "  Page Size: " << pageSize << " bytes\n";

    u32 enter_point_id = HNSWMetadataPage::invalid_node_id;
    u32 max_level = 0;
    u64 node_count = 0;
    u64 max_elements_val = 0;
    PID base_pid_val = 0;

    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_level = meta_guard->max_level.load(std::memory_order_acquire);
            node_count = meta_guard->node_count.load(std::memory_order_acquire);
            max_elements_val = meta_guard->max_elements;
            base_pid_val = meta_guard->base_pid;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    info << "\nIndex Metadata:\n";
    info << "  Metadata PID: " << metadata_pid << "\n";
    info << "  Base PID: " << base_pid_val << "\n";
    info << "  Max Elements: " << max_elements_val << "\n";
    info << "  Current Node Count: " << node_count << "\n";
    info << "  Current Max Level: " << max_level << "\n";
    info << "  Entry Point Node ID: "
         << (enter_point_id == HNSWMetadataPage::invalid_node_id ? "NONE" : std::to_string(enter_point_id)) << "\n";

    if (enter_point_id != HNSWMetadataPage::invalid_node_id) {
        try {
            GuardO<BaseNodePage> entry_page_guard(getNodePID(enter_point_id));
            NodeAccessor entry_acc(entry_page_guard.ptr, getNodeIndexInPage(enter_point_id), this);

            info << "\nEntry Point Node Details:\n";
            u32 node_level = entry_acc.getLevel();
            info << "  Node Level: " << node_level << "\n";
            info << "  Node PID: " << getNodePID(enter_point_id) << "\n";
            info << "  Node Index in Page: " << getNodeIndexInPage(enter_point_id) << "\n";

            for (u32 level = 0; level <= node_level; ++level) {
                size_t neighbor_count = 0;
                if (level == 0) {
                    neighbor_count = entry_acc.getL0Neighbors().size();
                } else {
                    OverflowPointer ptr = entry_acc.getOverflowPointer();
                    if (ptr.isValid()) {
                        GuardO<OverflowPage> overflow_guard(ptr.page_id);
                        neighbor_count = entry_acc.getHigherLevelNeighbors(level, overflow_guard.ptr).size();
                    }
                }
                info << "  Level " << level << " neighbors: " << neighbor_count << "\n";
            }
        } catch (const OLCRestartException&) {
            info << "\nEntry Point Node Details: (could not read due to concurrent access)\n";
        }
    }

    info << "\n--- Storage Utilization ---\n";

    // 1. Logical Capacity Usage (how full the index is conceptually)
    double capacity_utilization = 0.0;
    if (max_elements_val > 0) {
        capacity_utilization = static_cast<double>(node_count) / max_elements_val * 100.0;
    }
    info << "Index Capacity Usage: " << std::fixed << std::setprecision(2) << capacity_utilization << "% ("
         << node_count << " / " << max_elements_val << " nodes)\n";

    // 2. Base Page Utilization (how many of the pre-allocated base pages are being used)
    u64 total_allocated_base_pages = (max_elements_val + MaxNodesPerBasePage - 1) / MaxNodesPerBasePage;
    if (total_allocated_base_pages > 0) {
        u64 used_base_pages = (node_count > 0) ? ((node_count - 1) / MaxNodesPerBasePage + 1) : 0;
        double page_utilization = static_cast<double>(used_base_pages) / total_allocated_base_pages * 100.0;
        info << "Base Page Utilization:  " << std::fixed << std::setprecision(2) << page_utilization << "% ("
             << used_base_pages << " / " << total_allocated_base_pages << " pages)\n";
    }

    // 3. Overflow Storage Status (information about the heap-like overflow pages)
    PID last_of_pid = OverflowPointer::INVALID_PID;
    u16 last_of_offset = 0;
    // We already read these from the metadata guard earlier, but a fresh read is safer
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            last_of_pid = meta_guard->last_overflow_pid;
            last_of_offset = meta_guard->last_overflow_offset;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    info << "\nOverflow Storage Status:\n";
    if (last_of_pid == OverflowPointer::INVALID_PID) {
        info << "  No overflow pages allocated.\n";
    } else {
        // Note: To get the total number of overflow pages, you would ideally add an
        // atomic counter to HNSWMetadataPage. Here, we report on the current one.
        info << "  Current Overflow Page ID: " << last_of_pid << "\n";

        double current_of_page_util = 0.0;
        size_t overflow_capacity = pageSize - OverflowPage::HeaderSize;
        if (overflow_capacity > 0) {
            current_of_page_util = static_cast<double>(last_of_offset) / overflow_capacity * 100.0;
        }
        info << "  Current Page Fill Level:  " << std::fixed << std::setprecision(2) << current_of_page_util << "% ("
             << last_of_offset << " / " << overflow_capacity << " bytes used)\n";
    }

    info << "\n=== End Index Information ===\n";

    return info.str();
}

// --- Explicit Template Instantiation ---
template class HNSW<128, hnsw_distance::SIMDAcceleratedL2>;

// Explicit instantiation for templated member functions
template std::vector<std::pair<float, u32>> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchKnn<true>(
    const float* query, size_t k, size_t ef_search_param);
template std::vector<std::pair<float, u32>> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchKnn<false>(
    const float* query, size_t k, size_t ef_search_param);
template std::vector<std::vector<std::pair<float, u32>>>
HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchKnn_parallel<true>(std::span<const float> queries, size_t k,
                                                                      size_t ef_search_param, size_t num_threads);
template std::vector<std::vector<std::pair<float, u32>>>
HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchKnn_parallel<false>(std::span<const float> queries, size_t k,
                                                                       size_t ef_search_param, size_t num_threads);

// Explicit instantiation for internal templated helper methods
template std::vector<std::pair<float, u32>> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchLayer<true>(
    const float* query, u32 entry_point_id, u32 level, size_t ef,
    std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::vector<std::pair<float, u32>> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchLayer<false>(
    const float* query, u32 entry_point_id, u32 level, size_t ef,
    std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::pair<float, u32> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::findBestEntryPointForLevel<true>(
    const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::findBestEntryPointForLevel<false>(
    const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchBaseLayer<true>(const float* query,
                                                                                                  u32 entry_point_id,
                                                                                                  int start_level,
                                                                                                  int end_level);
template std::pair<float, u32> HNSW<128, hnsw_distance::SIMDAcceleratedL2>::searchBaseLayer<false>(const float* query,
                                                                                                   u32 entry_point_id,
                                                                                                   int start_level,
                                                                                                   int end_level);