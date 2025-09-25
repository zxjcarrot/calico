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

// --- Helper function to estimate MaxLevel ---
template <size_t Dim, typename DistanceMetric>
size_t HNSW<Dim, DistanceMetric>::estimateMaxLevel(u64 max_elements, size_t M) {
    // The random level generation uses: level = -log(uniform_random) / log(M)
    // For max_elements, the expected maximum level is approximately log(max_elements) / log(M)
    if (max_elements <= 1) return 1;

    // Correct formula based on the actual random level generation algorithm
    double expected_max_level = std::log(static_cast<double>(max_elements)) / std::log(static_cast<double>(M));
    size_t estimated_max_level = static_cast<size_t>(std::ceil(expected_max_level + 1));  // +2 for safety margin

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
        throw std::runtime_error(
            "Page size is too small to fit even one node with the given parameters. "
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
                NodeAccessor candidate_acc(candidate_page_guard.ptr, getNodeIndexInPage(current_candidate.second),
                                           this);
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
void HNSW<Dim, DistanceMetric>::addPoint_internal(const float* point, u32 new_node_id) {
    const u32 new_node_level = getRandomLevel();
    PID new_node_pid = getNodePID(new_node_id);

    // --- 1. Allocate space and initialize data for the new node on its page ---
    {
        GuardX<HNSWPage> page_guard(new_node_pid);
        HNSWPage* page = page_guard.ptr;
        u32 node_idx = getNodeIndexInPage(new_node_id);

        if (node_idx >= MaxNodesPerPage) {
            throw std::runtime_error("Node index exceeds page capacity. This should not happen with pre-calculation.");
        }
        u8* node_start = page->getNodeData() + node_idx * FixedNodeSize;
        float* vector_ptr = reinterpret_cast<float*>(node_start);
        u16* level_counts_ptr = reinterpret_cast<u16*>(node_start + VectorSize);
        memcpy(vector_ptr, point, VectorSize);
        memset(level_counts_ptr, 0, MaxNeighborsHeaderSize + MaxNeighborsListSize);
        page->node_count = std::max(page->node_count, static_cast<u16>(node_idx + 1));
        page->dirty = true;
    }

    // --- 2. Find the global entry point for the search ---
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

    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        if (meta_guard->enter_point_node_id == HNSWMetadataPage::invalid_node_id) {
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level = new_node_level;
            meta_guard->dirty = true;
        }
        return;
    }

    // --- 3. Search from top layers, BUILDING A DISTANCE CACHE along the way ---
    std::unordered_map<u32, float> distance_cache;
    u32 current_entry_point = enter_point_id;

    if (new_node_level < max_l) {
        // This call populates the distance_cache with distances of all nodes visited during the greedy search.
        auto result = searchBaseLayer(point, enter_point_id, max_l, new_node_level + 1, distance_cache);
        current_entry_point = result.second;
    }

    // --- 4. Connection phase: from new_node_level down to 0 ---
    bool is_first_connection_level = true;
    for (int level = std::min(static_cast<u32>(new_node_level), max_l); level >= 0; --level) {
        std::vector<std::pair<float, u32>> candidates;

        if (is_first_connection_level) {
            // For the first connection level, use the cache built by the high-level search.
            // We must ensure the final entry point's distance is in the cache, especially
            // if the high-level search was skipped (i.e., new_node_level >= max_l).
            if (distance_cache.find(current_entry_point) == distance_cache.end()) {
                try {
                    GuardO<HNSWPage> page_guard(getNodePID(current_entry_point));
                    NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(current_entry_point), this);
                    distance_cache[current_entry_point] = DistanceMetric::compare(point, acc.getVector(), Dim);
                } catch (const OLCRestartException&) {
                    // If it fails, let searchLayer compute it. The cache will be empty.
                }
            }
            candidates = searchLayer(point, current_entry_point, level, efConstruction, distance_cache);
            is_first_connection_level = false;
        } else {
            // For subsequent levels, the entry point has changed, and the old cache is irrelevant.
            // Pass an empty cache to signal that all distances must be computed.
            std::unordered_map<u32, float> distance_cache;
            candidates = searchLayer(point, current_entry_point, level, efConstruction, {});
        }

        if (candidates.empty()) {
            continue;
        }

        size_t M_level = (level == 0) ? M0 : M;
        auto neighbors_to_link_pairs = selectNeighborsHeuristic(point, candidates, M_level);

        // Step 4a: Link the new node TO its chosen neighbors.
        {
            GuardX<HNSWPage> new_node_page_guard(new_node_pid);
            MutableNodeAccessor new_node_acc(new_node_page_guard.ptr, getNodeIndexInPage(new_node_id), this);
            std::vector<u32> neighbor_ids;
            neighbor_ids.reserve(neighbors_to_link_pairs.size());
            for (const auto& p : neighbors_to_link_pairs) {
                neighbor_ids.push_back(p.second);
            }
            new_node_acc.setNeighbors(level, neighbor_ids, this);
        }

        // Step 4b: Link neighbors BACK to the new node.
        for (const std::pair<float, u32>& neighbor_pair : neighbors_to_link_pairs) {
            const float dist_point_to_neighbor = neighbor_pair.first;
            const u32 neighbor_id = neighbor_pair.second;
            PID neighbor_pid = getNodePID(neighbor_id);

            for (;;) {  // OLC retry loop for updating a single neighbor
                try {
                    // ... (PHASE 1: READ)...
                    std::vector<float> neighbor_vector_copy(Dim);
                    std::vector<u32> current_neighbors_ids;
                    std::unordered_map<u32, std::vector<float>> connection_vectors;
                    bool needs_pruning;
                    {
                        GuardO<HNSWPage> neighbor_page_guard(neighbor_pid);
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        const float* neighbor_vec_ptr = neighbor_acc.getVector();
                        std::copy(neighbor_vec_ptr, neighbor_vec_ptr + Dim, neighbor_vector_copy.begin());
                        auto neighbors_span = neighbor_acc.getNeighbors(level, this);
                        current_neighbors_ids.assign(neighbors_span.begin(), neighbors_span.end());
                    }
                    needs_pruning = (current_neighbors_ids.size() >= M_level);
                    if (needs_pruning) {
                        for (u32 conn_id : current_neighbors_ids) {
                            GuardO<HNSWPage> conn_page_guard(getNodePID(conn_id));
                            NodeAccessor conn_acc(conn_page_guard.ptr, getNodeIndexInPage(conn_id), this);
                            const float* conn_vec_ptr = conn_acc.getVector();
                            connection_vectors[conn_id].assign(conn_vec_ptr, conn_vec_ptr + Dim);
                        }
                    }

                    // --- PHASE 2: WRITE ---
                    {
                        GuardX<HNSWPage> neighbor_page_guard(neighbor_pid);
                        NodeAccessor locked_reader(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        auto latest_neighbors_span = locked_reader.getNeighbors(level, this);

                        if (latest_neighbors_span.size() >= M_level) {
                            std::priority_queue<std::pair<float, u32>> connections_to_prune;
                            connections_to_prune.push({dist_point_to_neighbor, new_node_id});
                            for (u32 conn_id : latest_neighbors_span) {
                                auto it = connection_vectors.find(conn_id);
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
                            new_neighbor_list.reserve(M_level);
                            while (!connections_to_prune.empty()) {
                                new_neighbor_list.push_back(connections_to_prune.top().second);
                                connections_to_prune.pop();
                            }
                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id),
                                                             this);
                            neighbor_acc.setNeighbors(level, new_neighbor_list, this);
                        } else {
                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id),
                                                             this);
                            neighbor_acc.addNeighbor(level, new_node_id, this);
                        }
                    }
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
        }

        // Update the entry point for the next level down
        if (!neighbors_to_link_pairs.empty()) {
            current_entry_point = neighbors_to_link_pairs.front().second;
        }
    }

    // --- 5. Update global entry point if the new node is the highest ---
    if (new_node_level > max_l) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        if (new_node_level > meta_guard->max_level.load(std::memory_order_acquire)) {
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level = new_node_level;
            meta_guard->dirty = true;
        }
    }
}

// --- Parallel and Single Add/Search Functions (unchanged) ---
template <size_t Dim, typename DistanceMetric>
void HNSW<Dim, DistanceMetric>::addPoint_parallel(std::span<const float> points, size_t num_threads) {
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
        double page_utilization =
            (double)((node_count + NodesPerPage - 1) / NodesPerPage) / total_allocated_pages * 100.0;
        info << "  Page Utilization: " << std::fixed << std::setprecision(2) << page_utilization << "%\n";
    }

    info << "\n=== End Index Information ===\n";

    return info.str();
}

// The main stats calculation function
template <size_t Dim, typename DistanceMetric>
HNSWStats HNSW<Dim, DistanceMetric>::getStats() const {
    HNSWStats result_stats;
    // Copy live counters
    result_stats.dist_comps.store(this->stats_.dist_comps.load());
    result_stats.search_hops.store(this->stats_.search_hops.load());

    u64 node_count = 0;
    u32 max_level = 0;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            node_count = meta_guard->node_count.load(std::memory_order_acquire);
            max_level = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    if (node_count == 0) {
        return result_stats;
    }

    // +1 because max_level is 0-indexed
    result_stats.num_levels = max_level + 1;
    result_stats.nodes_per_level.assign(result_stats.num_levels, 0);
    result_stats.links_per_level.assign(result_stats.num_levels, 0);

    u64 total_links = 0;

    for (u32 node_id = 0; node_id < node_count; ++node_id) {
        try {
            GuardO<HNSWPage> page_guard(getNodePID(node_id));
            NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);

            const u16* counts = acc.getLevelCounts();
            u32 node_max_level = getNodeMaxLevel(counts);

            // A node exists on all levels up to its max level
            for (u32 l = 0; l <= node_max_level; ++l) {
                if (l < result_stats.num_levels) {
                    result_stats.nodes_per_level[l]++;
                }
            }

            // Count links/neighbors at each level for this node
            for (u32 l = 0; l <= node_max_level; ++l) {
                if (l < result_stats.num_levels) {
                    auto neighbors = acc.getNeighbors(l, this);
                    result_stats.links_per_level[l] += neighbors.size();
                }
            }
        } catch (const OLCRestartException&) {
            // Skip this node if we can't read it, stats might be slightly off
            continue;
        }
    }

    result_stats.avg_neighbors_per_level.assign(result_stats.num_levels, 0.0);
    for (u32 l = 0; l < result_stats.num_levels; ++l) {
        if (result_stats.nodes_per_level[l] > 0) {
            result_stats.avg_neighbors_per_level[l] =
                static_cast<double>(result_stats.links_per_level[l]) / result_stats.nodes_per_level[l];
            total_links += result_stats.links_per_level[l];
        }
    }

    if (node_count > 0) {
        // Note: This counts each link twice (A->B and B->A), so we divide by 2 for total graph edges,
        // but for average list length, we use the sum of all list lengths.
        // We also need to sum links across all levels for a single node to get its total list length.
        // The current 'total_links' is the sum of all list lengths across all levels.
        u64 total_neighbor_list_entries = 0;
        for (u64 links : result_stats.links_per_level) {
            total_neighbor_list_entries += links;
        }

        result_stats.avg_neighbors_total = static_cast<double>(total_neighbor_list_entries) / node_count;
    }

    return result_stats;
}

// --- Explicit Template Instantiation ---
template class HNSW<128, hnsw_distance::SIMDAcceleratedL2>;