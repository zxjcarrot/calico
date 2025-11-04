#include "diskann.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <map>  // For PID grouping
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "calico.hpp"

static std::unique_ptr<Distance<float>> distance_metric =
    std::unique_ptr<Distance<float>>(get_distance_function<float>(Metric::L2));

// =========================================================================
// Factory Function, Accessors, and ThreadPool
// =========================================================================
std::unique_ptr<DiskANNBase> create_index(size_t dimensions, uint64_t max_elements, size_t R_max_degree,
                                          bool is_dynamic) {
    switch (dimensions) {
        case 16:
            return std::make_unique<DiskANN<float, uint32_t, 16>>(max_elements, R_max_degree, is_dynamic);
        case 32:
            return std::make_unique<DiskANN<float, uint32_t, 32>>(max_elements, R_max_degree, is_dynamic);
        case 64:
            return std::make_unique<DiskANN<float, uint32_t, 64>>(max_elements, R_max_degree, is_dynamic);
        case 128:
            return std::make_unique<DiskANN<float, uint32_t, 128>>(max_elements, R_max_degree, is_dynamic);
        case 256:
            return std::make_unique<DiskANN<float, uint32_t, 256>>(max_elements, R_max_degree, is_dynamic);
        case 384:
            return std::make_unique<DiskANN<float, uint32_t, 384>>(max_elements, R_max_degree, is_dynamic);
        case 512:
            return std::make_unique<DiskANN<float, uint32_t, 512>>(max_elements, R_max_degree, is_dynamic);
        case 768:
            return std::make_unique<DiskANN<float, uint32_t, 768>>(max_elements, R_max_degree, is_dynamic);
        case 1024:
            return std::make_unique<DiskANN<float, uint32_t, 1024>>(max_elements, R_max_degree, is_dynamic);
        default:
            throw std::runtime_error("Unsupported dimension: " + std::to_string(dimensions) + ". Please recompile.");
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::NodeAccessor(const VamanaPage* page, uint32_t node_idx,
                                                                  const DiskANN* index)
    : _node_start(page->getNodeData() + node_idx * index->_FixedNodeSize), _index(index) {}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
const T* DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getVector() const {
    return reinterpret_cast<const T*>(_node_start);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::span<const TagT> DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getTags() const {
    const auto* count_ptr = reinterpret_cast<const uint16_t*>(_node_start + VectorSize);
    const auto* tags_start = reinterpret_cast<const TagT*>(count_ptr + 1);
    return {tags_start, *count_ptr};
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::span<const uint32_t> DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getNeighbors() const {
    const auto* count_ptr = reinterpret_cast<const uint16_t*>(_node_start + VectorSize + TagsHeaderSize + TagsDataSize);
    const auto* neighbors_start = reinterpret_cast<const uint32_t*>(count_ptr + 1);
    return {neighbors_start, *count_ptr};
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::MutableNodeAccessor(VamanaPage* page, uint32_t node_idx,
                                                                                const DiskANN* index)
    : NodeAccessor(page, node_idx, index), _page(page) {}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setVector(const T* vector) {
    memcpy(const_cast<uint8_t*>(this->_node_start), vector, VectorSize);
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setTags(const std::vector<TagT>& tags) {
    auto* count_ptr = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(this->_node_start) + VectorSize);
    auto* tags_start = reinterpret_cast<TagT*>(count_ptr + 1);
    size_t count_to_copy = std::min(tags.size(), MaxTagsPerNode);
    *count_ptr = count_to_copy;
    memcpy(tags_start, tags.data(), count_to_copy * sizeof(TagT));
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setNeighbors(const std::vector<uint32_t>& neighbors) {
    auto* count_ptr = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(this->_node_start) + VectorSize +
                                                  TagsHeaderSize + TagsDataSize);
    auto* neighbors_start = reinterpret_cast<uint32_t*>(count_ptr + 1);
    size_t count_to_copy = std::min(neighbors.size(), this->_index->_R_max_degree);
    *count_ptr = count_to_copy;
    memcpy(neighbors_start, neighbors.data(), count_to_copy * sizeof(uint32_t));
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
ThreadPool* DiskANN<T, TagT, Dim, MaxTagsPerNode>::getOrCreateThreadPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(_pool_mutex);
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    if (!_thread_pool || _thread_pool->getThreadCount() != threads_to_use) {
        _thread_pool = std::make_unique<ThreadPool>(threads_to_use);
    }
    return _thread_pool.get();
}

// =========================================================================
// Constructor & Destructor
// =========================================================================

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::DiskANN(uint64_t max_elements, size_t R_max_degree, bool is_dynamic)
    : _is_dynamic(is_dynamic), _max_elements(max_elements), _R_max_degree(R_max_degree), _is_layout_optimized(false) {
    _NeighborsDataSize = _R_max_degree * sizeof(uint32_t);
    size_t calculated_size = VectorSize + TagsHeaderSize + TagsDataSize + NeighborsHeaderSize + _NeighborsDataSize;
    _FixedNodeSize = (calculated_size + 63) & ~63;

    _NodesPerPage = (pageSize - VamanaPage::HeaderSize) / _FixedNodeSize;
    if (_NodesPerPage == 0) throw std::runtime_error("Page size too small for node configuration.");

    std::cout << "Creating new index... FixedNodeSize=" << _FixedNodeSize << ", NodesPerPage=" << _NodesPerPage
              << std::endl;

    AllocGuard<DiskANNMetadataPage> meta_guard;
    _metadata_pid = meta_guard.pid;
    meta_guard->dirty = false;
    meta_guard->max_elements = _max_elements;
    meta_guard->node_count = 0;
    meta_guard->base_pid = -1;
    meta_guard->is_layout_optimized = false;
    uint64_t num_pages = (_max_elements + _NodesPerPage - 1) / _NodesPerPage;
    if (num_pages > 0) {
        AllocGuard<VamanaPage> first_page_guard;
        _base_pid = first_page_guard.pid;
        meta_guard->base_pid = _base_pid;
        first_page_guard->dirty = false;
        first_page_guard->node_count_in_page = 0;
        for (uint64_t i = 1; i < num_pages; ++i) {
            AllocGuard<VamanaPage> page_guard;
            page_guard->dirty = false;
            page_guard->node_count_in_page = 0;
        }
    }
    meta_guard->dirty = true;
    _visited_list_pool = std::make_unique<VisitedListPool>(1, _max_elements);
    if (_is_dynamic) _internal_to_external_map.reserve(_max_elements);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::~DiskANN() {}

// =========================================================================
// BATCH DATA ACCESS HELPERS (OPTIMIZED)
// =========================================================================

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::get_nodes_data_batch(const std::vector<uint32_t>& ids,
                                                                 NodeDataBatch& data_batch) const {
    if (ids.empty()) return true;
    std::map<PID, std::vector<std::pair<uint32_t, size_t>>> page_to_nodes_map;
    std::vector<PID> pids_to_prefetch;
    pids_to_prefetch.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t id = ids[i];
        PID pid = getNodePID(id);
        if (page_to_nodes_map.find(pid) == page_to_nodes_map.end()) {
            pids_to_prefetch.push_back(pid);
        }
        page_to_nodes_map[pid].emplace_back(id, i);
    }
    if (!pids_to_prefetch.empty()) {
        bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
    }
    data_batch.clear();
    data_batch.vectors.resize(ids.size() * Dim);
    data_batch.tag_counts.resize(ids.size());
    data_batch.tags.reserve(ids.size() * MaxTagsPerNode / 2);
    try {
        for (const auto& [pid, node_infos] : page_to_nodes_map) {
            GuardORelaxed<VamanaPage> page_guard(pid);
            for (const auto& [node_id, original_index] : node_infos) {
                NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
                memcpy(data_batch.vectors.data() + original_index * Dim, acc.getVector(), VectorSize);
                auto tags_span = acc.getTags();
                data_batch.tag_counts[original_index] = tags_span.size();
                data_batch.tags.insert(data_batch.tags.end(), tags_span.begin(), tags_span.end());
            }
        }
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::get_vectors_batch(const std::vector<uint32_t>& ids, T* dest_buffer) const {
    if (ids.empty()) return true;
    std::map<PID, std::vector<std::pair<uint32_t, size_t>>> page_to_nodes_map;
    std::vector<PID> pids_to_prefetch;
    pids_to_prefetch.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t id = ids[i];
        PID pid = getNodePID(id);
        if (page_to_nodes_map.find(pid) == page_to_nodes_map.end()) {
            pids_to_prefetch.push_back(pid);
        }
        page_to_nodes_map[pid].emplace_back(id, i);
    }
    if (!pids_to_prefetch.empty()) {
        bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
    }
    try {
        for (const auto& [pid, node_infos] : page_to_nodes_map) {
            GuardO<VamanaPage> page_guard(pid);
            for (const auto& [node_id, original_index] : node_infos) {
                NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
                memcpy(dest_buffer + original_index * Dim, acc.getVector(), VectorSize);
            }
        }
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}

// =========================================================================
// SEARCH ALGORITHMS (OPTIMIZED)
// =========================================================================
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::greedy_search_final(
    uint32_t start_node_id, const T* start_node_vec, const T* query, size_t K, const SearchParams& params,
    const TagT* filter_label) {
    alignas(32) T aligned_query[Dim];
    memcpy(aligned_query, query, VectorSize);

    VisitedList* visited_nodes = _visited_list_pool->getFreeVisitedList();
    auto visited_guard = std::unique_ptr<VisitedList, std::function<void(VisitedList*)>>(
        visited_nodes, [&](VisitedList* p) { _visited_list_pool->releaseVisitedList(p); });

    visited_nodes->reset();

    thread_local static NeighborPriorityQueue best_L_nodes;
    best_L_nodes.clear();
    best_L_nodes.reserve(params.L_search);

    // MODIFIED: Use the pre-fetched vector if available to avoid initial lock contention
    if (visited_nodes->try_visit(start_node_id)) {
        if (start_node_vec) {  // Use the cached vector if available
            float dist = distance_metric->compare(aligned_query, start_node_vec, Dim);
            best_L_nodes.insert(Neighbor(start_node_id, dist));
        } else {  // Fallback to fetching from disk if cache missed
            alignas(32) T vec_from_disk[Dim];
            if (getNodeVector(start_node_id, vec_from_disk)) {
                float dist = distance_metric->compare(aligned_query, vec_from_disk, Dim);
                best_L_nodes.insert(Neighbor(start_node_id, dist));
            } else {
                visited_nodes->unvisit(start_node_id);
            }
        }
    }

    // HIGH-PERFORMANCE BEAM EXPANSION LOOP
    std::vector<uint32_t> expansion_candidate_ids;
    std::vector<uint32_t> all_neighbors_from_beam;
    std::vector<uint32_t> unvisited_neighbors_ids;
    std::vector<PID> pids_to_prefetch;

    while (best_L_nodes.has_unexpanded_node()) {
        pids_to_prefetch.clear();
        size_t prefetch_count = 0;
        for (size_t i = best_L_nodes._cur; i < best_L_nodes.size() && prefetch_count < params.beam_width; ++i) {
            if (!best_L_nodes[i].expanded) {
                pids_to_prefetch.push_back(getNodePID(best_L_nodes[i].id));
                prefetch_count++;
            }
        }
        if (!pids_to_prefetch.empty()) {
            bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
        }

        expansion_candidate_ids.clear();
        for (size_t i = 0; i < params.beam_width && best_L_nodes.has_unexpanded_node(); ++i) {
            Neighbor candidate = best_L_nodes.closest_unexpanded();
            expansion_candidate_ids.push_back(candidate.id);
        }

        all_neighbors_from_beam.clear();
        for (uint32_t candidate_id : expansion_candidate_ids) {
            std::vector<uint32_t> neighbors;
            if (getNodeNeighbors(candidate_id, neighbors)) {
                all_neighbors_from_beam.insert(all_neighbors_from_beam.end(), neighbors.begin(), neighbors.end());
            }
        }

        unvisited_neighbors_ids.clear();
        unvisited_neighbors_ids.reserve(all_neighbors_from_beam.size());
        for (uint32_t neighbor_id : all_neighbors_from_beam) {
            if (!_deleted_nodes.contains(neighbor_id) && visited_nodes->try_visit(neighbor_id)) {
                unvisited_neighbors_ids.push_back(neighbor_id);
            }
        }
        if (unvisited_neighbors_ids.empty()) continue;

        if (filter_label) {
            thread_local static NodeDataBatch data_batch;
            if (!get_nodes_data_batch(unvisited_neighbors_ids, data_batch)) {
                for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
                continue;
            }
            size_t tags_offset = 0;
            for (size_t i = 0; i < unvisited_neighbors_ids.size(); ++i) {
                uint32_t id = unvisited_neighbors_ids[i];
                uint16_t tag_count = data_batch.tag_counts[i];
                bool has_label = false;
                std::span<const TagT> current_tags(data_batch.tags.data() + tags_offset, tag_count);
                for (const TagT& tag : current_tags) {
                    if (tag == *filter_label) {
                        has_label = true;
                        break;
                    }
                }
                tags_offset += tag_count;
                if (has_label) {
                    const T* vector = data_batch.vectors.data() + i * Dim;
                    float dist = distance_metric->compare(aligned_query, vector, Dim);
                    best_L_nodes.insert(Neighbor(id, dist));
                }
            }
        } else {
            thread_local static std::vector<T, AlignedAllocator<T, 32>> neighbor_vectors;
            neighbor_vectors.resize(unvisited_neighbors_ids.size() * Dim);
            if (!get_vectors_batch(unvisited_neighbors_ids, neighbor_vectors.data())) {
                for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
                continue;
            }
            for (size_t i = 0; i < unvisited_neighbors_ids.size(); ++i) {
                uint32_t id = unvisited_neighbors_ids[i];
                const T* vector = neighbor_vectors.data() + i * Dim;
                float dist = distance_metric->compare(aligned_query, vector, Dim);
                best_L_nodes.insert(Neighbor(id, dist));
            }
        }
    }

    std::vector<std::pair<float, uint32_t>> results;
    size_t result_count = std::min((size_t)K, best_L_nodes.size());
    results.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        results.emplace_back(best_L_nodes[i].distance, best_L_nodes[i].id);
    }
    return results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::greedy_search_original(
    const std::vector<uint32_t>& start_nodes, const T* query, size_t K, const SearchParams& params,
    const TagT* filter_label) {
    alignas(32) T aligned_query[Dim];
    memcpy(aligned_query, query, VectorSize);

    VisitedList* visited_nodes = _visited_list_pool->getFreeVisitedList();
    auto visited_guard = std::unique_ptr<VisitedList, std::function<void(VisitedList*)>>(
        visited_nodes, [&](VisitedList* p) { _visited_list_pool->releaseVisitedList(p); });

    visited_nodes->reset();

    thread_local static NeighborPriorityQueue best_L_nodes;
    best_L_nodes.clear();
    best_L_nodes.reserve(params.L_search);

    for (uint32_t start_node : start_nodes) {
        if (visited_nodes->try_visit(start_node)) {
            alignas(32) T start_node_vec[Dim];
            if (getNodeVector(start_node, start_node_vec)) {
                float dist = distance_metric->compare(aligned_query, start_node_vec, Dim);
                best_L_nodes.insert(Neighbor(start_node, dist));
            } else {
                visited_nodes->unvisit(start_node);
            }
        }
    }

    std::vector<uint32_t> unvisited_neighbors_ids;

    while (best_L_nodes.has_unexpanded_node()) {
        Neighbor candidate = best_L_nodes.closest_unexpanded();

        std::vector<uint32_t> neighbors;
        if (!getNodeNeighbors(candidate.id, neighbors)) continue;

        unvisited_neighbors_ids.clear();
        for (uint32_t neighbor_id : neighbors) {
            if (visited_nodes->try_visit(neighbor_id)) {
                unvisited_neighbors_ids.push_back(neighbor_id);
            }
        }
        if (unvisited_neighbors_ids.empty()) continue;

        thread_local static std::vector<T, AlignedAllocator<T, 32>> neighbor_vectors;
        neighbor_vectors.resize(unvisited_neighbors_ids.size() * Dim);
        if (!get_vectors_batch(unvisited_neighbors_ids, neighbor_vectors.data())) {
            for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
            continue;
        }

        for (size_t i = 0; i < unvisited_neighbors_ids.size(); ++i) {
            uint32_t neighbor_id = unvisited_neighbors_ids[i];
            const T* vector = neighbor_vectors.data() + i * Dim;
            float dist = distance_metric->compare(aligned_query, vector, Dim);
            best_L_nodes.insert(Neighbor(neighbor_id, dist));
        }
    }

    std::vector<std::pair<float, uint32_t>> results;
    size_t result_count = std::min((size_t)params.L_search, best_L_nodes.size());
    results.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        results.emplace_back(best_L_nodes[i].distance, best_L_nodes[i].id);
    }
    return results;
}

// =========================================================================
// BUILD PROCESS & OTHER FUNCTIONS
// =========================================================================

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::build_typed(const T* data, const std::vector<std::vector<TagT>>& tags,
                                                        uint64_t num_points, const BuildParams& params) {
    if (num_points > _max_elements) throw std::runtime_error("Number of points exceeds max_elements capacity.");
    std::cout << "Starting index build for " << num_points << " points with R=" << _R_max_degree << "..." << std::endl;
    std::cout << "Phase 1/4: Writing vector and tag data..." << std::endl;
    if (!_is_dynamic) {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        _internal_to_external_map.resize(num_points);
        std::iota(_internal_to_external_map.begin(), _internal_to_external_map.end(), 0);
        _external_to_internal_map.reserve(num_points);
        for (uint32_t i = 0; i < num_points; ++i) _external_to_internal_map[i] = i;
    }
    for (uint64_t i = 0; i < num_points; ++i) {
        PID pid = getNodePID(i);
        GuardX<VamanaPage> page_guard(pid);
        page_guard->node_count_in_page =
            std::max(page_guard->node_count_in_page, static_cast<uint16_t>(getNodeIndexInPage(i) + 1));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(i), this);
        acc.setVector(data + i * Dim);
        acc.setTags(tags[i]);
    }
    {
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        meta_guard->node_count = num_points;
        meta_guard->is_layout_optimized = false;
    }
    _is_layout_optimized = false;
    std::cout << "Phase 2/4: Computing medoids..." << std::endl;
    compute_medoids(num_points);
    std::cout << "Phase 3/4: Initializing random graph..." << std::endl;
    std::mt19937 rng(1001);
    for (uint32_t i = 0; i < num_points; ++i) {
        std::vector<uint32_t> neighbors;
        std::uniform_int_distribution<uint32_t> dist(0, num_points - 1);
        std::unordered_set<uint32_t> chosen;
        while (chosen.size() < _R_max_degree && chosen.size() < num_points - 1) {
            uint32_t neighbor = dist(rng);
            if (neighbor != i) chosen.insert(neighbor);
        }
        neighbors.assign(chosen.begin(), chosen.end());
        PID pid = getNodePID(i);
        GuardX<VamanaPage> page_guard(pid);
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(i), this);
        acc.setNeighbors(neighbors);
    }
    std::cout << "Phase 4/4: Running Vamana refinement passes..." << std::endl;
    std::vector<uint32_t> permutation(num_points);
    std::iota(permutation.begin(), permutation.end(), 0);
    auto run_pass = [&](float alpha, bool filter_aware, int pass_num) {
        std::cout << "  - Pass " << pass_num << " with alpha=" << alpha << "..." << std::endl;
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::atomic<size_t> processed_count = 0;
        size_t num_threads = (params.num_threads == 0) ? omp_get_max_threads() : params.num_threads;
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 64)
        for (uint64_t i = 0; i < num_points; ++i) {
            uint32_t node_id = permutation[i];
            for (;;) {
                try {
                    alignas(32) T query_vec[Dim];
                    std::vector<TagT> node_tags;
                    if (!getNodeVector(node_id, query_vec)) throw OLCRestartException();
                    if (filter_aware && !getNodeTags(node_id, node_tags)) throw OLCRestartException();
                    std::vector<uint32_t> start_nodes;
                    if (filter_aware) {
                        for (const TagT& tag : node_tags)
                            if (_medoids.count(tag)) start_nodes.push_back(_medoids.at(tag));
                    }
                    if (start_nodes.empty() && !_medoids.empty()) start_nodes.push_back(_medoids.begin()->second);
                    if (start_nodes.empty()) start_nodes.push_back(0);
                    SearchParams search_params{params.L_build, 4};
                    auto candidates_with_dist =
                        greedy_search_original(start_nodes, query_vec, params.L_build, search_params, nullptr);
                    robust_prune_batched(node_id, candidates_with_dist, alpha, filter_aware);
                    std::vector<uint32_t> final_neighbors;
                    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);
                    connect_neighbors_batched(node_id, final_neighbors, params, filter_aware);
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
            size_t count = processed_count.fetch_add(1);
            if ((count + 1) % (num_points / 100 + 1) == 0)
                std::cout << "\r    Processed " << count + 1 << "/" << num_points << " nodes." << std::flush;
        }
        std::cout << std::endl;
    };
    run_pass(1.0f, true, 1);
    run_pass(params.alpha, true, 2);
    std::cout << "Build complete." << std::endl;
    optimize_layout();
    std::cout << "Layout optimization finished. Index is ready for search." << std::endl;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::robust_prune_batched(uint32_t p_id,
                                                                 std::vector<std::pair<float, uint32_t>>& candidates,
                                                                 float alpha, bool filter_aware) {
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> candidate_ids;
    candidate_ids.reserve(candidates.size());
    for (const auto& pair : candidates) candidate_ids.push_back(pair.second);

    std::vector<T, AlignedAllocator<T, 32>> candidate_vectors(candidate_ids.size() * Dim);
    if (!get_vectors_batch(candidate_ids, candidate_vectors.data())) {
        if (candidates.size() > _R_max_degree) candidates.resize(_R_max_degree);
        return;
    }

    std::vector<std::pair<float, uint32_t>> new_neighbors;
    new_neighbors.reserve(_R_max_degree);
    std::vector<bool> pruned(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (new_neighbors.size() >= _R_max_degree) break;
        if (pruned[i]) continue;
        new_neighbors.push_back(candidates[i]);
        const T* p_star_vec = candidate_vectors.data() + i * Dim;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (pruned[j]) continue;
            const T* p_prime_vec = candidate_vectors.data() + j * Dim;
            float dist_p_star_p_prime = distance_metric->compare(p_star_vec, p_prime_vec, Dim);
            if (alpha * dist_p_star_p_prime < candidates[j].first) pruned[j] = true;
        }
    }
    candidates = new_neighbors;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::connect_neighbors_batched(uint32_t node_id,
                                                                      const std::vector<uint32_t>& neighbors,
                                                                      const BuildParams& params, bool filter_aware) {
    {
        GuardX<VamanaPage> page_guard(getNodePID(node_id));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        acc.setNeighbors(neighbors);
    }
    for (uint32_t neighbor_id : neighbors) {
        for (;;) {
            try {
                std::vector<uint32_t> neighbor_current_neighbors;
                if (!getNodeNeighbors(neighbor_id, neighbor_current_neighbors)) throw OLCRestartException();
                bool already_connected = false;
                for (uint32_t n : neighbor_current_neighbors)
                    if (n == node_id) {
                        already_connected = true;
                        break;
                    }
                if (already_connected) break;
                neighbor_current_neighbors.push_back(node_id);
                if (neighbor_current_neighbors.size() > _R_max_degree) {
                    std::vector<T, AlignedAllocator<T, 32>> local_vectors(neighbor_current_neighbors.size() * Dim);
                    if (!get_vectors_batch(neighbor_current_neighbors, local_vectors.data()))
                        throw OLCRestartException();
                    alignas(32) T neighbor_vec_query[Dim];
                    if (!getNodeVector(neighbor_id, neighbor_vec_query)) throw OLCRestartException();
                    std::vector<std::pair<float, uint32_t>> candidates_to_prune;
                    candidates_to_prune.reserve(neighbor_current_neighbors.size());
                    for (size_t i = 0; i < neighbor_current_neighbors.size(); ++i) {
                        candidates_to_prune.push_back(
                            {distance_metric->compare(neighbor_vec_query, local_vectors.data() + i * Dim, Dim),
                             neighbor_current_neighbors[i]});
                    }
                    robust_prune_batched(neighbor_id, candidates_to_prune, params.alpha, filter_aware);
                    neighbor_current_neighbors.clear();
                    for (const auto& p : candidates_to_prune) neighbor_current_neighbors.push_back(p.second);
                }
                {
                    GuardX<VamanaPage> x_guard(getNodePID(neighbor_id));
                    MutableNodeAccessor x_acc(x_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    x_acc.setNeighbors(neighbor_current_neighbors);
                }
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::optimize_layout() {
    if (_is_dynamic) {
        std::cerr << "Warning: Layout optimization is intended for static indices..." << std::endl;
    }
    uint64_t num_points;
    for (;;) {
        try {
            GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
            if (meta_guard->is_layout_optimized) {
                std::cout << "Layout is already optimized. Skipping." << std::endl;
                _is_layout_optimized = true;
                return;
            }
            num_points = meta_guard->node_count;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    if (num_points == 0) return;
    std::cout << "Starting graph layout optimization for " << num_points << " points..." << std::endl;
    std::cout << "Phase 1/4: Calculating BFS ordering..." << std::endl;
    std::vector<uint32_t> new_id_to_old_id(num_points);
    std::vector<int32_t> old_id_to_new_id(num_points, -1);
    std::vector<bool> visited(num_points, false);
    std::queue<uint32_t> q;
    uint32_t start_node = _medoids.empty() ? 0 : _medoids.begin()->second;
    q.push(start_node);
    visited[start_node] = true;
    uint32_t current_new_id = 0;
    while (current_new_id < num_points) {
        if (q.empty()) {
            bool found_unvisited = false;
            for (uint32_t i = 0; i < num_points; ++i) {
                if (!visited[i]) {
                    q.push(i);
                    visited[i] = true;
                    found_unvisited = true;
                    break;
                }
            }
            if (!found_unvisited) break;
        }
        uint32_t current_old_id = q.front();
        q.pop();
        new_id_to_old_id[current_new_id] = current_old_id;
        old_id_to_new_id[current_old_id] = current_new_id;
        current_new_id++;
        std::vector<uint32_t> neighbors;
        for (;;) {
            try {
                if (!getNodeNeighbors(current_old_id, neighbors)) throw OLCRestartException();
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
        for (uint32_t neighbor_old_id : neighbors) {
            if (!visited[neighbor_old_id]) {
                visited[neighbor_old_id] = true;
                q.push(neighbor_old_id);
            }
        }
    }
    std::cout << "Phase 2/4: Reading original index data into memory snapshot..." << std::endl;
    std::vector<uint8_t> original_data_buffer(num_points * _FixedNodeSize);
    for (uint32_t old_id = 0; old_id < num_points; ++old_id) {
        for (;;) {
            try {
                GuardO<VamanaPage> page_guard(getNodePID(old_id));
                const uint8_t* node_data_ptr = page_guard->getNodeData() + getNodeIndexInPage(old_id) * _FixedNodeSize;
                memcpy(original_data_buffer.data() + old_id * _FixedNodeSize, node_data_ptr, _FixedNodeSize);
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }
    std::cout << "Phase 3/4: Constructing and writing final index..." << std::endl;
    std::vector<uint8_t> temp_node_buffer(_FixedNodeSize);
    for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
        uint32_t old_id_source = new_id_to_old_id[new_id];
        const uint8_t* src_node_ptr = original_data_buffer.data() + old_id_source * _FixedNodeSize;
        memcpy(temp_node_buffer.data(), src_node_ptr, VectorSize + TagsHeaderSize + TagsDataSize);
        const uint16_t* old_count_ptr =
            reinterpret_cast<const uint16_t*>(src_node_ptr + VectorSize + TagsHeaderSize + TagsDataSize);
        const uint32_t* old_neighbors_data_ptr = reinterpret_cast<const uint32_t*>(old_count_ptr + 1);
        const uint16_t neighbor_count = *old_count_ptr;
        uint16_t* new_count_ptr =
            reinterpret_cast<uint16_t*>(temp_node_buffer.data() + VectorSize + TagsHeaderSize + TagsDataSize);
        uint32_t* new_neighbors_data_ptr = reinterpret_cast<uint32_t*>(new_count_ptr + 1);
        *new_count_ptr = neighbor_count;
        for (uint16_t i = 0; i < neighbor_count; ++i) {
            uint32_t old_neighbor_id = old_neighbors_data_ptr[i];
            new_neighbors_data_ptr[i] =
                (old_neighbor_id < num_points) ? old_id_to_new_id[old_neighbor_id] : old_neighbor_id;
        }
        GuardX<VamanaPage> page_guard(getNodePID(new_id));
        uint8_t* dest_on_disk_ptr = page_guard->getNodeData() + getNodeIndexInPage(new_id) * _FixedNodeSize;
        memcpy(dest_on_disk_ptr, temp_node_buffer.data(), _FixedNodeSize);
    }
    std::cout << "Phase 4/4: Updating internal ID maps..." << std::endl;
    {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        std::vector<uint32_t> new_internal_to_external_map(num_points);
        for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
            uint32_t old_id = new_id_to_old_id[new_id];
            new_internal_to_external_map[new_id] = _internal_to_external_map[old_id];
        }
        _internal_to_external_map = std::move(new_internal_to_external_map);
        _external_to_internal_map.clear();
        _external_to_internal_map.reserve(num_points);
        for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
            _external_to_internal_map[_internal_to_external_map[new_id]] = new_id;
        }
    }
    for (auto& pair : _medoids) {
        if (pair.second < num_points) pair.second = old_id_to_new_id[pair.second];
    }
    // After remapping IDs, the vector cache is invalid and must be repopulated.
    compute_medoids(num_points);
    {
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        meta_guard->is_layout_optimized = true;
    }
    _is_layout_optimized = true;
    std::cout << "Layout optimization complete." << std::endl;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::build(const void* data, const std::vector<std::vector<uint32_t>>& tags,
                                                  uint64_t num_points, const BuildParams& params) {
    build_typed(static_cast<const T*>(data), tags, num_points, params);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search(const void* query, size_t K,
                                                                                      const SearchParams& params) {
    return search_typed(static_cast<const T*>(query), K, params);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_typed(
    const T* query, size_t K, const SearchParams& params) {
    uint32_t start_node_id;
    const T* start_node_vec = nullptr;
    for (;;) {
        try {
            GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
            if (meta_guard->node_count == 0) return {};
            if (_medoids.empty()) compute_medoids(meta_guard->node_count);

            auto it = _medoids.find(0);
            if (it == _medoids.end() && !_medoids.empty()) it = _medoids.begin();
            if (it == _medoids.end()) return {};

            start_node_id = it->second;
            auto cache_it = _medoid_vectors_cache.find(it->first);
            if (cache_it != _medoid_vectors_cache.end()) {
                start_node_vec = cache_it->second.data();
            }
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    auto internal_results = greedy_search_final(start_node_id, start_node_vec, query, K, params, nullptr);
    std::shared_lock<std::shared_mutex> lock(_map_lock);
    for (auto& pair : internal_results) {
        if (pair.second < _internal_to_external_map.size()) {
            pair.second = _internal_to_external_map[pair.second];
        }
    }
    return internal_results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter(
    const void* query, uint32_t filter_label, size_t K, const SearchParams& params) {
    return search_with_filter_typed(static_cast<const T*>(query), filter_label, K, params);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_typed(
    const T* query, const TagT& filter_label, size_t K, const SearchParams& params) {
    uint32_t start_node_id;
    const T* start_node_vec = nullptr;

    auto it = _medoids.find(filter_label);
    if (it == _medoids.end()) {
        it = _medoids.find(0);
        if (it == _medoids.end() && !_medoids.empty()) it = _medoids.begin();
        if (it == _medoids.end()) return {};
    }
    start_node_id = it->second;
    auto cache_it = _medoid_vectors_cache.find(it->first);
    if (cache_it != _medoid_vectors_cache.end()) {
        start_node_vec = cache_it->second.data();
    }
    auto internal_results = greedy_search_final(start_node_id, start_node_vec, query, K, params, &filter_label);
    std::shared_lock<std::shared_mutex> lock(_map_lock);
    for (auto& pair : internal_results) {
        if (pair.second < _internal_to_external_map.size()) {
            pair.second = _internal_to_external_map[pair.second];
        }
    }
    return internal_results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeVector(uint32_t node_id, T* dest_vector) const {
    try {
        GuardO<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        memcpy(dest_vector, acc.getVector(), VectorSize);
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeTags(uint32_t node_id, std::vector<TagT>& dest_tags) const {
    try {
        GuardO<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        auto tags_span = acc.getTags();
        dest_tags.assign(tags_span.begin(), tags_span.end());
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeNeighbors(uint32_t node_id,
                                                             std::vector<uint32_t>& dest_neighbors) const {
    try {
        GuardORelaxed<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        auto neighbors_span = acc.getNeighbors();
        dest_neighbors.assign(neighbors_span.begin(), neighbors_span.end());
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
inline PID DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodePID(uint32_t node_id) const {
    return _base_pid + (node_id / _NodesPerPage);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
inline uint32_t DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeIndexInPage(uint32_t node_id) const {
    return node_id % _NodesPerPage;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::compute_medoids(uint64_t num_points) {
    if (num_points == 0) return;
    std::cout << "Computing medoids using centroid method..." << std::endl;
    _medoids.clear();
    _medoid_vectors_cache.clear();  // Clear the cache before re-computing
    std::unordered_map<TagT, std::vector<uint32_t>> points_by_tag;
    for (uint32_t i = 0; i < num_points; ++i) {
        std::vector<TagT> tags;
        if (getNodeTags(i, tags))
            for (const auto& tag : tags) points_by_tag[tag].push_back(i);
    }
    TagT universal_tag_key = std::numeric_limits<TagT>::max();
    points_by_tag[universal_tag_key].reserve(num_points);
    for (uint32_t i = 0; i < num_points; ++i) points_by_tag[universal_tag_key].push_back(i);

    for (auto const& [tag, points] : points_by_tag) {
        if (points.empty()) continue;
        std::vector<double> centroid(Dim, 0.0);
        size_t points_in_centroid = 0;
        const size_t BATCH_SIZE = 1024;
        std::vector<T, AlignedAllocator<T, 32>> vector_batch(BATCH_SIZE * Dim);
        std::vector<uint32_t> id_batch;
        id_batch.reserve(BATCH_SIZE);
        for (uint32_t point_id : points) {
            id_batch.push_back(point_id);
            if (id_batch.size() == BATCH_SIZE) {
                if (get_vectors_batch(id_batch, vector_batch.data())) {
                    for (size_t i = 0; i < id_batch.size(); ++i)
                        for (size_t d = 0; d < Dim; ++d) centroid[d] += vector_batch[i * Dim + d];
                    points_in_centroid += id_batch.size();
                }
                id_batch.clear();
            }
        }
        if (!id_batch.empty()) {
            std::vector<T, AlignedAllocator<T, 32>> leftover_vector_batch(id_batch.size() * Dim);
            if (get_vectors_batch(id_batch, leftover_vector_batch.data())) {
                for (size_t i = 0; i < id_batch.size(); ++i)
                    for (size_t d = 0; d < Dim; ++d) centroid[d] += leftover_vector_batch[i * Dim + d];
                points_in_centroid += id_batch.size();
            }
        }
        if (points_in_centroid == 0) continue;
        for (size_t d = 0; d < Dim; ++d) centroid[d] /= points_in_centroid;
        std::vector<T, AlignedAllocator<T, 32>> centroid_T(Dim);
        for (size_t d = 0; d < Dim; ++d) centroid_T[d] = static_cast<T>(centroid[d]);

        float min_dist_sq = std::numeric_limits<float>::max();
        uint32_t medoid_id = points[0];
        id_batch.clear();
        for (uint32_t point_id : points) {
            id_batch.push_back(point_id);
            if (id_batch.size() == BATCH_SIZE) {
                if (get_vectors_batch(id_batch, vector_batch.data())) {
                    for (size_t i = 0; i < id_batch.size(); ++i) {
                        float dist_sq = distance_metric->compare(centroid_T.data(), vector_batch.data() + i * Dim, Dim);
                        if (dist_sq < min_dist_sq) {
                            min_dist_sq = dist_sq;
                            medoid_id = id_batch[i];
                        }
                    }
                }
                id_batch.clear();
            }
        }
        if (!id_batch.empty()) {
            std::vector<T, AlignedAllocator<T, 32>> leftover_vector_batch(id_batch.size() * Dim);
            if (get_vectors_batch(id_batch, leftover_vector_batch.data())) {
                for (size_t i = 0; i < id_batch.size(); ++i) {
                    float dist_sq =
                        distance_metric->compare(centroid_T.data(), leftover_vector_batch.data() + i * Dim, Dim);
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        medoid_id = id_batch[i];
                    }
                }
            }
        }
        TagT cache_key;
        if (tag == universal_tag_key) {
            if (_medoids.find(0) == _medoids.end()) {
                _medoids[0] = medoid_id;
                cache_key = 0;
            }
        } else {
            _medoids[tag] = medoid_id;
            cache_key = tag;
        }

        if (_medoids.count(cache_key) && _medoid_vectors_cache.find(cache_key) == _medoid_vectors_cache.end()) {
            _medoid_vectors_cache[cache_key].resize(Dim);
            if (!getNodeVector(_medoids[cache_key], _medoid_vectors_cache[cache_key].data())) {
                _medoid_vectors_cache.erase(cache_key);
                std::cerr << "Warning: Failed to fetch and cache vector for medoid with tag " << cache_key << std::endl;
            }
        }
    }
    if (_medoids.empty() && num_points > 0) {
        _medoids[0] = num_points / 2;
        // Also cache this fallback medoid
        _medoid_vectors_cache[0].resize(Dim);
        if (!getNodeVector(_medoids[0], _medoid_vectors_cache[0].data())) {
            _medoid_vectors_cache.erase(0);
        }
    }
    std::cout << "Computed " << _medoids.size() << " medoids using centroid method." << std::endl;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point(const void* point, const std::vector<uint32_t>& tags,
                                                         uint32_t external_id) {
    insert_point_typed(static_cast<const T*>(point), tags, external_id);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::lazy_delete(uint32_t external_id) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call lazy_delete on a non-dynamic index.");
    uint32_t internal_id = std::numeric_limits<uint32_t>::max();
    {
        std::shared_lock<std::shared_mutex> lock(_map_lock);
        auto it = _external_to_internal_map.find(external_id);
        if (it != _external_to_internal_map.end()) internal_id = it->second;
    }
    if (internal_id != std::numeric_limits<uint32_t>::max()) _deleted_nodes.insert(internal_id);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point_typed(const T* point, const std::vector<TagT>& tags,
                                                               uint32_t external_id) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call insert_point on a non-dynamic index.");
    if (_is_layout_optimized)
        std::cerr << "Warning: Inserting into an optimized-layout index will degrade performance over time."
                  << std::endl;
    uint32_t internal_id;
    {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        internal_id = meta_guard->node_count.fetch_add(1);
        if (internal_id >= _max_elements) {
            meta_guard->node_count.fetch_sub(1);
            throw std::runtime_error("DiskANN is full.");
        }
        _external_to_internal_map[external_id] = internal_id;
        _internal_to_external_map.push_back(external_id);
        PID pid = getNodePID(internal_id);
        GuardX<VamanaPage> page_guard(pid);
        page_guard->node_count_in_page =
            std::max(page_guard->node_count_in_page, static_cast<uint16_t>(getNodeIndexInPage(internal_id) + 1));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(internal_id), this);
        acc.setVector(point);
        acc.setTags(tags);
        acc.setNeighbors({});
    }
    std::vector<uint32_t> start_nodes;
    for (const TagT& tag : tags)
        if (_medoids.count(tag)) start_nodes.push_back(_medoids.at(tag));
    if (start_nodes.empty() && !_medoids.empty()) start_nodes.push_back(_medoids.begin()->second);
    if (start_nodes.empty() && internal_id > 0) start_nodes.push_back(0);
    if (start_nodes.empty()) return;
    BuildParams insert_build_params;
    SearchParams insert_search_params{insert_build_params.L_build, 4};
    auto candidates_with_dist =
        greedy_search_original(start_nodes, point, insert_build_params.L_build, insert_search_params, nullptr);
    robust_prune_batched(internal_id, candidates_with_dist, insert_build_params.alpha, true);
    std::vector<uint32_t> final_neighbors;
    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);
    connect_neighbors_batched(internal_id, final_neighbors, insert_build_params, true);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::consolidate_deletes(const BuildParams& params) {
    throw std::runtime_error("consolidate_deletes not fully implemented in this optimized version yet.");
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel(
    const void* queries, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    return search_parallel_typed(static_cast<const T*>(queries), num_queries, K, params, num_threads);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel_typed(
    const T* queries_typed, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;
    size_t threads_to_use = (num_threads == 0) ? omp_get_max_threads() : num_threads;
#pragma omp parallel for num_threads(threads_to_use)
    for (size_t i = 0; i < num_queries; ++i) all_results[i] = search_typed(queries_typed + i * Dim, K, params);
    return all_results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel(
    const void* queries, size_t num_queries, uint32_t filter_label, size_t K, const SearchParams& params,
    size_t num_threads) {
    return search_with_filter_parallel_typed(static_cast<const T*>(queries), num_queries, filter_label, K, params,
                                             num_threads);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel_typed(const T* queries_typed, size_t num_queries,
                                                                         const TagT& filter_label, size_t K,
                                                                         const SearchParams& params,
                                                                         size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;
    size_t threads_to_use = (num_threads == 0) ? omp_get_max_threads() : num_threads;
#pragma omp parallel for num_threads(threads_to_use)
    for (size_t i = 0; i < num_queries; ++i)
        all_results[i] = search_with_filter_typed(queries_typed + i * Dim, filter_label, K, params);
    return all_results;
}

// Explicit template instantiations
template class DiskANN<float, uint32_t, 16>;
template class DiskANN<float, uint32_t, 32>;
template class DiskANN<float, uint32_t, 64>;
template class DiskANN<float, uint32_t, 128>;
template class DiskANN<float, uint32_t, 256>;
template class DiskANN<float, uint32_t, 384>;
template class DiskANN<float, uint32_t, 512>;
template class DiskANN<float, uint32_t, 768>;
template class DiskANN<float, uint32_t, 1024>;