#include "diskann.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "calico.hpp"

using DistanceMetric = hnsw_distance::SIMDAcceleratedL2;

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

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::DiskANN(uint64_t max_elements, size_t R_max_degree, bool is_dynamic)
    : _is_dynamic(is_dynamic), _max_elements(max_elements), _R_max_degree(R_max_degree) {
    _NeighborsDataSize = _R_max_degree * sizeof(uint32_t);
    _FixedNodeSize = VectorSize + TagsHeaderSize + TagsDataSize + NeighborsHeaderSize + _NeighborsDataSize;
    _NodesPerPage = (pageSize - VamanaPage::HeaderSize) / _FixedNodeSize;

    if (_NodesPerPage == 0) throw std::runtime_error("Page size too small for node configuration.");

    AllocGuard<DiskANNMetadataPage> meta_guard;
    _metadata_pid = meta_guard.pid;
    meta_guard->dirty = false;
    meta_guard->max_elements = _max_elements;
    meta_guard->node_count = 0;
    meta_guard->base_pid = -1;

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
    if (_is_dynamic) {
        _internal_to_external_map.reserve(_max_elements);
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::~DiskANN() {}

// --- OPTIMIZATION: New batch vector fetch implementation ---
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getVectorsForPruning(const std::vector<uint32_t>& ids,
                                                                 std::vector<T>& vectors) const {
    vectors.resize(ids.size() * Dim);

    std::vector<PID> pids_to_prefetch;
    pids_to_prefetch.reserve(ids.size());
    for (uint32_t id : ids) {
        pids_to_prefetch.push_back(getNodePID(id));
    }
    if (!pids_to_prefetch.empty()) {
        bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        if (!getNodeVector(ids[i], vectors.data() + i * Dim)) {
            // On OLC failure, it's safer to abort the whole operation and let the caller retry.
            return false;
        }
    }
    return true;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::build_typed(const T* data, const std::vector<std::vector<TagT>>& tags,
                                                        uint64_t num_points, const BuildParams& params) {
    if (num_points > _max_elements) throw std::runtime_error("Number of points exceeds max_elements capacity.");
    std::cout << "Starting index build for " << num_points << " points with R=" << _R_max_degree << "..." << std::endl;
    std::cout << "Phase 1/4: Writing vector and tag data..." << std::endl;
    for (uint64_t i = 0; i < num_points; ++i) {
        PID pid = getNodePID(i);
        GuardX<VamanaPage> page_guard(pid);
        page_guard->node_count_in_page =
            std::max(page_guard->node_count_in_page, static_cast<uint16_t>(getNodeIndexInPage(i) + 1));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(i), this);
        acc.setVector(data + i * Dim);
        acc.setTags(tags[i]);
    }

    GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
    meta_guard->node_count = num_points;
    meta_guard.release();

    std::cout << "Phase 2/4: Computing medoids..." << std::endl;
    compute_medoids(num_points);

    std::cout << "Phase 3/4: Initializing random graph..." << std::endl;
    std::mt19937 rng(std::random_device{}());
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

        // --- OPTIMIZATION: Switched to OpenMP for lower overhead parallel loop ---
        size_t num_threads = (params.num_threads == 0) ? omp_get_max_threads() : params.num_threads;
        omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 128)
        for (uint64_t i = 0; i < num_points; ++i) {
            uint32_t node_id = permutation[i];
            for (;;) {
                try {
                    T query_vec[Dim];
                    std::vector<TagT> node_tags;
                    if (!getNodeVector(node_id, query_vec)) throw OLCRestartException();
                    if (filter_aware && !getNodeTags(node_id, node_tags)) throw OLCRestartException();

                    std::vector<uint32_t> start_nodes;
                    if (filter_aware) {
                        for (const TagT& tag : node_tags) {
                            if (_medoids.count(tag)) start_nodes.push_back(_medoids.at(tag));
                        }
                    }
                    if (start_nodes.empty()) {
                        start_nodes.push_back(_medoids.begin()->second);
                    }

                    SearchParams search_params{params.L_build, 4};
                    auto candidates_with_dist =
                        greedy_search(start_nodes, query_vec, params.L_build, search_params, nullptr);
                    robust_prune(node_id, candidates_with_dist, alpha, filter_aware);

                    std::vector<uint32_t> final_neighbors;
                    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);
                    connect_neighbors(node_id, final_neighbors, params, filter_aware);

                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
            size_t count = processed_count.fetch_add(1);
            if ((count + 1) % (num_points / 100 + 1) == 0) {
                std::cout << "\r    Processed " << count + 1 << "/" << num_points << " nodes." << std::flush;
            }
        }
        std::cout << std::endl;
    };

    run_pass(1.0f, true, 1);
    run_pass(params.alpha, true, 2);
    std::cout << "Build complete." << std::endl;
}

// --- OPTIMIZATION: `robust_prune` rewritten to use batch data fetching ---
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::robust_prune(uint32_t p_id,
                                                         std::vector<std::pair<float, uint32_t>>& candidates,
                                                         float alpha, bool filter_aware) {
    if (candidates.empty()) return;
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> ids_to_fetch;
    ids_to_fetch.reserve(candidates.size() + 1);
    ids_to_fetch.push_back(p_id);
    for (const auto& pair : candidates) {
        ids_to_fetch.push_back(pair.second);
    }

    std::vector<T> local_vectors;
    if (!getVectorsForPruning(ids_to_fetch, local_vectors)) {
        // Fallback on contention: just keep the R closest.
        if (candidates.size() > _R_max_degree) candidates.resize(_R_max_degree);
        return;
    }

    std::vector<std::pair<float, uint32_t>> new_neighbors;
    new_neighbors.reserve(_R_max_degree);
    std::vector<bool> pruned(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (new_neighbors.size() >= _R_max_degree) break;
        if (pruned[i]) continue;

        const auto& p_star_pair = candidates[i];
        uint32_t p_star_id = p_star_pair.second;
        new_neighbors.push_back(p_star_pair);
        const T* p_star_vec = local_vectors.data() + (i + 1) * Dim;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (pruned[j]) continue;

            const auto& p_prime_pair = candidates[j];
            uint32_t p_prime_id = p_prime_pair.second;
            float dist_p_p_prime = p_prime_pair.first;
            const T* p_prime_vec = local_vectors.data() + (j + 1) * Dim;

            float dist_p_star_p_prime = DistanceMetric::compare(p_star_vec, p_prime_vec, Dim);

            if (alpha * dist_p_star_p_prime < dist_p_p_prime) {
                bool tags_covered = true;
                if (filter_aware) {
                    // This part still requires fetching tags, which is less of a bottleneck.
                    std::vector<TagT> p_star_tags, p_prime_tags;
                    if (!getNodeTags(p_star_id, p_star_tags) || !getNodeTags(p_prime_id, p_prime_tags)) {
                        continue;  // Skip check on contention
                    }
                    std::unordered_set<TagT> p_star_tags_set(p_star_tags.begin(), p_star_tags.end());
                    for (const TagT& tag : p_prime_tags) {
                        if (p_star_tags_set.find(tag) == p_star_tags_set.end()) {
                            tags_covered = false;
                            break;
                        }
                    }
                }
                if (tags_covered) {
                    pruned[j] = true;
                }
            }
        }
    }
    candidates = new_neighbors;
}

// --- OPTIMIZATION: connect_neighbors uses batch fetching for its internal prune step ---
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::connect_neighbors(uint32_t node_id, const std::vector<uint32_t>& neighbors,
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
                for (uint32_t n : neighbor_current_neighbors) {
                    if (n == node_id) {
                        already_connected = true;
                        break;
                    }
                }
                if (already_connected) break;

                neighbor_current_neighbors.push_back(node_id);

                if (neighbor_current_neighbors.size() > _R_max_degree) {
                    std::vector<T> local_vectors;
                    if (!getVectorsForPruning(neighbor_current_neighbors, local_vectors)) throw OLCRestartException();

                    // Find the vector of the neighbor itself to compare against
                    const T* neighbor_vec = nullptr;
                    for (size_t i = 0; i < neighbor_current_neighbors.size(); ++i) {
                        if (neighbor_current_neighbors[i] == neighbor_id) {
                            neighbor_vec = local_vectors.data() + i * Dim;
                            break;
                        }
                    }
                    // This should not happen if logic is correct, but as a safeguard:
                    if (!neighbor_vec) {
                        T temp_vec[Dim];
                        if (!getNodeVector(neighbor_id, temp_vec)) throw OLCRestartException();
                        neighbor_vec = temp_vec;
                    }

                    std::vector<std::pair<float, uint32_t>> candidates_to_prune;
                    candidates_to_prune.reserve(neighbor_current_neighbors.size());

                    for (size_t i = 0; i < neighbor_current_neighbors.size(); ++i) {
                        uint32_t n_id = neighbor_current_neighbors[i];
                        const T* n_vec = local_vectors.data() + i * Dim;
                        candidates_to_prune.push_back({DistanceMetric::compare(neighbor_vec, n_vec, Dim), n_id});
                    }

                    robust_prune(neighbor_id, candidates_to_prune, params.alpha, filter_aware);
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

// =========================================================================================
// Below are the remaining functions, largely unchanged but included for completeness
// =========================================================================================

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
        GuardO<VamanaPage> page_guard(getNodePID(node_id));
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
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter(
    const void* query, uint32_t filter_label, size_t K, const SearchParams& params) {
    return search_with_filter_typed(static_cast<const T*>(query), filter_label, K, params);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point(const void* point, const std::vector<uint32_t>& tags,
                                                         uint32_t external_id) {
    insert_point_typed(static_cast<const T*>(point), tags, external_id);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_typed(
    const T* query, size_t K, const SearchParams& params) {
    GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
    if (meta_guard->node_count == 0) return {};
    if (_medoids.empty()) compute_medoids(meta_guard->node_count);
    uint32_t start_node = _medoids.begin()->second;
    auto results = greedy_search({start_node}, query, K, params, nullptr);
    if (results.size() > K) results.resize(K);
    return results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_typed(
    const T* query, const TagT& filter_label, size_t K, const SearchParams& params) {
    auto it = _medoids.find(filter_label);
    if (it == _medoids.end()) return {};
    auto results = greedy_search({it->second}, query, K, params, &filter_label);
    if (results.size() > K) results.resize(K);
    return results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::greedy_search(
    const std::vector<uint32_t>& start_nodes, const T* query, size_t K, const SearchParams& params,
    const TagT* filter_label) {
    VisitedList* visited = _visited_list_pool->getFreeVisitedList();
    auto visited_guard = std::unique_ptr<VisitedList, std::function<void(VisitedList*)>>(
        visited, [&](VisitedList* p) { _visited_list_pool->releaseVisitedList(p); });
    vl_type* visited_array = visited->mass;
    vl_type visited_tag = visited->curV;
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        candidates;
    std::priority_queue<std::pair<float, uint32_t>> top_k_results;
    for (uint32_t start_node : start_nodes) {
        if (visited_array[start_node] != visited_tag) {
            visited_array[start_node] = visited_tag;
            T start_node_vec[Dim];
            if (getNodeVector(start_node, start_node_vec)) {
                float dist = DistanceMetric::compare(query, start_node_vec, Dim);
                candidates.push({dist, start_node});
                if (!filter_label && !_deleted_nodes.contains(start_node)) top_k_results.push({dist, start_node});
            }
        }
    }
    std::vector<PID> pids_to_prefetch;
    pids_to_prefetch.reserve(2 * _R_max_degree);
    while (!candidates.empty()) {
        auto [dist, id] = candidates.top();
        candidates.pop();
        if (top_k_results.size() >= params.L_search && dist > top_k_results.top().first) break;
        std::vector<uint32_t> neighbors_copy;
        if (!getNodeNeighbors(id, neighbors_copy)) continue;
        if (!neighbors_copy.empty()) {
            pids_to_prefetch.clear();
            for (const uint32_t& neighbor_id : neighbors_copy) {
                if (visited_array[neighbor_id] != visited_tag) pids_to_prefetch.push_back(getNodePID(neighbor_id));
            }
            if (!pids_to_prefetch.empty()) bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
        }
        for (uint32_t neighbor_id : neighbors_copy) {
            if (visited_array[neighbor_id] != visited_tag) {
                visited_array[neighbor_id] = visited_tag;
                if (_deleted_nodes.contains(neighbor_id)) continue;
                T neighbor_vec[Dim];
                if (!getNodeVector(neighbor_id, neighbor_vec)) continue;
                bool has_label = true;
                if (filter_label) {
                    std::vector<TagT> neighbor_tags;
                    if (!getNodeTags(neighbor_id, neighbor_tags)) continue;
                    has_label = false;
                    for (const TagT& tag : neighbor_tags) {
                        if (tag == *filter_label) {
                            has_label = true;
                            break;
                        }
                    }
                }
                if (has_label) {
                    float neighbor_dist = DistanceMetric::compare(query, neighbor_vec, Dim);
                    if (top_k_results.size() < params.L_search || neighbor_dist < top_k_results.top().first) {
                        candidates.push({neighbor_dist, neighbor_id});
                        top_k_results.push({neighbor_dist, neighbor_id});
                        if (top_k_results.size() > params.L_search) top_k_results.pop();
                    }
                }
            }
        }
    }
    std::vector<std::pair<float, uint32_t>> results;
    results.reserve(top_k_results.size());
    while (!top_k_results.empty()) {
        results.push_back(top_k_results.top());
        top_k_results.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::compute_medoids(uint64_t num_points) {
    _medoids.clear();
    std::unordered_map<TagT, std::vector<uint32_t>> points_by_tag;
    for (uint32_t i = 0; i < num_points; ++i) {
        std::vector<TagT> tags;
        if (getNodeTags(i, tags)) {
            for (const auto& tag : tags) points_by_tag[tag].push_back(i);
        }
    }
    for (auto const& [tag, points] : points_by_tag) {
        if (points.empty()) continue;
        std::vector<double> centroid(Dim, 0.0);
        for (uint32_t point_id : points) {
            T vec[Dim];
            if (getNodeVector(point_id, vec)) {
                for (size_t d = 0; d < Dim; ++d) centroid[d] += vec[d];
            }
        }
        for (size_t d = 0; d < Dim; ++d) centroid[d] /= points.size();
        float min_dist_sq = std::numeric_limits<float>::max();
        uint32_t medoid_id = points[0];
        std::vector<T> centroid_float(Dim);
        std::copy(centroid.begin(), centroid.end(), centroid_float.begin());
        for (uint32_t point_id : points) {
            T vec[Dim];
            if (getNodeVector(point_id, vec)) {
                float dist_sq = DistanceMetric::compare(centroid_float.data(), vec, Dim);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    medoid_id = point_id;
                }
            }
        }
        _medoids[tag] = medoid_id;
    }
    std::cout << "Computed " << _medoids.size() << " medoids using centroid method." << std::endl;
}

// --- Dynamic Indexing Methods (Insert, Delete, Consolidate) ---
// These are more complex and require careful handling of concurrency.
// The implementations are based on the FreshDiskANN paper.

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point_typed(const T* point, const std::vector<TagT>& tags,
                                                               uint32_t external_id) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call insert_point on a non-dynamic index.");
    uint32_t internal_id;

    // Phase 1: Atomically get a new ID and write data. This part requires a unique lock.
    {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        internal_id = meta_guard->node_count.fetch_add(1);
        if (internal_id >= _max_elements) {
            meta_guard->node_count.fetch_sub(1);
            throw std::runtime_error("DiskANN is full. Cannot insert new point.");
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

    // Phase 2: Find candidate neighbors
    std::vector<uint32_t> start_nodes;
    for (const TagT& tag : tags) {
        if (_medoids.count(tag)) start_nodes.push_back(_medoids.at(tag));
    }
    if (start_nodes.empty() && !_medoids.empty()) start_nodes.push_back(_medoids.begin()->second);
    if (start_nodes.empty() && internal_id > 0) start_nodes.push_back(0);  // Fallback for very first inserts
    if (start_nodes.empty()) return;  // Cannot connect if it's the very first point

    BuildParams insert_build_params;
    SearchParams insert_search_params{insert_build_params.L_build, 4};
    auto candidates_with_dist =
        greedy_search(start_nodes, point, insert_build_params.L_build, insert_search_params, nullptr);

    // Phase 3: Prune candidates to get final neighbors
    robust_prune(internal_id, candidates_with_dist, insert_build_params.alpha, true);
    std::vector<uint32_t> final_neighbors;
    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);

    // Phase 4: Connect new node to the graph
    connect_neighbors(internal_id, final_neighbors, insert_build_params, true);
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
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::consolidate_deletes(const BuildParams& params) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call consolidate_deletes on a non-dynamic index.");
    std::vector<uint32_t> deleted_ids = _deleted_nodes.get_elements();
    if (deleted_ids.empty()) {
        std::cout << "No deleted nodes to consolidate." << std::endl;
        return;
    }
    std::cout << "Starting consolidation for " << deleted_ids.size() << " deleted nodes..." << std::endl;
    std::unordered_set<uint32_t> affected_nodes_set;
    {
        GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
        uint64_t current_node_count = meta_guard->node_count;
        for (uint32_t i = 0; i < current_node_count; ++i) {
            if (_deleted_nodes.contains(i)) continue;
            std::vector<uint32_t> neighbors;
            if (getNodeNeighbors(i, neighbors)) {
                for (uint32_t neighbor : neighbors) {
                    if (_deleted_nodes.contains(neighbor)) {
                        affected_nodes_set.insert(i);
                        break;
                    }
                }
            }
        }
    }
    std::vector<uint32_t> affected_nodes(affected_nodes_set.begin(), affected_nodes_set.end());
    std::cout << "Found " << affected_nodes.size() << " affected nodes to repair." << std::endl;
    ThreadPool* pool = getOrCreateThreadPool(params.num_threads);
    std::atomic<size_t> repaired_count = 0;
    for (uint32_t affected_node_id : affected_nodes) {
        pool->enqueue([this, affected_node_id, &params, &repaired_count, num_affected = affected_nodes.size()]() {
            for (;;) {
                try {
                    std::unordered_set<uint32_t> candidate_set;
                    T affected_node_vec[Dim];
                    if (!getNodeVector(affected_node_id, affected_node_vec)) throw OLCRestartException();
                    std::vector<uint32_t> current_neighbors;
                    if (!getNodeNeighbors(affected_node_id, current_neighbors)) throw OLCRestartException();
                    for (uint32_t neighbor : current_neighbors) {
                        if (!_deleted_nodes.contains(neighbor))
                            candidate_set.insert(neighbor);
                        else {
                            std::vector<uint32_t> deleted_neighbors_neighbors;
                            if (getNodeNeighbors(neighbor, deleted_neighbors_neighbors)) {
                                for (uint32_t nn : deleted_neighbors_neighbors) {
                                    if (!_deleted_nodes.contains(nn) && nn != affected_node_id)
                                        candidate_set.insert(nn);
                                }
                            }
                        }
                    }
                    std::vector<std::pair<float, uint32_t>> candidates_with_dist;
                    for (uint32_t cand_id : candidate_set) {
                        T cand_vec[Dim];
                        if (getNodeVector(cand_id, cand_vec)) {
                            candidates_with_dist.push_back(
                                {DistanceMetric::compare(affected_node_vec, cand_vec, Dim), cand_id});
                        }
                    }
                    robust_prune(affected_node_id, candidates_with_dist, params.alpha, true);
                    std::vector<uint32_t> final_neighbors;
                    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);
                    {
                        GuardX<VamanaPage> page(getNodePID(affected_node_id));
                        MutableNodeAccessor acc(page.ptr, getNodeIndexInPage(affected_node_id), this);
                        acc.setNeighbors(final_neighbors);
                    }
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
            size_t count = repaired_count.fetch_add(1);
            if ((count + 1) % 100 == 0)
                std::cout << "\r    Repaired " << count + 1 << "/" << num_affected << " nodes." << std::flush;
        });
    }
    pool->waitForTasks();
    std::cout << "\nConsolidation of graph structure complete." << std::endl;
    _deleted_nodes.clear();
    std::cout << "Deleted nodes list has been cleared." << std::endl;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel_typed(
    const T* queries_typed, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;
    ThreadPool* pool = getOrCreateThreadPool(num_threads);
    std::vector<std::future<void>> futures;
    futures.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        futures.emplace_back(pool->enqueue([this, i, K, &params, &all_results, queries_typed]() {
            all_results[i] = search_typed(queries_typed + i * Dim, K, params);
        }));
    }
    for (auto& fut : futures) fut.get();
    return all_results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel_typed(const T* queries_typed, size_t num_queries,
                                                                         const TagT& filter_label, size_t K,
                                                                         const SearchParams& params,
                                                                         size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;
    ThreadPool* pool = getOrCreateThreadPool(num_threads);
    std::vector<std::future<void>> futures;
    futures.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        futures.emplace_back(pool->enqueue([this, i, K, &params, &filter_label, &all_results, queries_typed]() {
            all_results[i] = search_with_filter_typed(queries_typed + i * Dim, filter_label, K, params);
        }));
    }
    for (auto& fut : futures) fut.get();
    return all_results;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel(
    const void* queries, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    return search_parallel_typed(static_cast<const T*>(queries), num_queries, K, params, num_threads);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel(
    const void* queries, size_t num_queries, uint32_t filter_label, size_t K, const SearchParams& params,
    size_t num_threads) {
    return search_with_filter_parallel_typed(static_cast<const T*>(queries), num_queries, filter_label, K, params,
                                             num_threads);
}