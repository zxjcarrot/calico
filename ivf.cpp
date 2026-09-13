#include "ivf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

u32 normalizeThreadCount(size_t num_threads) {
    if (num_threads == 0) {
        const u32 hw = 64;
        return std::max<u32>(1, hw == 0 ? 1 : hw);
    }
    return static_cast<u32>(std::max<size_t>(1, num_threads));
}

}  // namespace

std::string IVFStats::toString() const {
    std::ostringstream out;
    out << "--- IVF Performance Stats ---\n";
    out << "Distance Blocks:    " << dist_comps.load() << "\n";
    out << "Lists Probed:       " << lists_probed.load() << "\n";
    out << "Vectors Scanned:    " << vectors_scanned.load() << "\n";
    out << "Number of Clusters: " << num_clusters << "\n";
    out << "Average List Size:  " << std::fixed << std::setprecision(2) << avg_list_size << "\n";
    return out.str();
}

ThreadPool* IVFIndex::getOrCreateSearchPool(size_t num_threads) const {
    std::lock_guard<std::mutex> guard(search_pool_mutex_);
    const size_t resolved = normalizeThreadCount(num_threads);
    if (!search_thread_pool_ || search_pool_size_ != resolved) {
        search_thread_pool_.reset(new ThreadPool(resolved));
        search_pool_size_ = resolved;
    }
    return search_thread_pool_.get();
}

ThreadPool* IVFIndex::getOrCreateAddPool(size_t num_threads) const {
    std::lock_guard<std::mutex> guard(add_pool_mutex_);
    const size_t resolved = normalizeThreadCount(num_threads);
    if (!add_thread_pool_ || add_pool_size_ != resolved) {
        add_thread_pool_.reset(new ThreadPool(resolved));
        add_pool_size_ = resolved;
    }
    return add_thread_pool_.get();
}

IVFIndex::IVFIndex(u64 max_elements, size_t dim, u32 num_clusters, u32 retrain_interval, bool enable_prefetch,
                   bool skip_recovery)
    : dim_(static_cast<u32>(dim)),
      num_clusters_(num_clusters),
      retrain_interval_(retrain_interval),
      max_elements_(max_elements),
    enable_prefetch_(enable_prefetch),
    cluster_pid_allocators_(std::make_unique<PIDAllocator[]>(std::max<u32>(1, num_clusters))) {
    if (dim_ == 0) {
        throw std::runtime_error("IVF dimension must be greater than zero.");
    }
    if (num_clusters_ == 0) {
        throw std::runtime_error("IVF num_clusters must be greater than zero.");
    }
    if (max_elements_ == 0) {
        throw std::runtime_error("IVF max_elements must be greater than zero.");
    }

    floats_per_chunk_page_ = IVFFloatChunkPage::floatsPerPage();
    if (floats_per_chunk_page_ == 0) {
        throw std::runtime_error("IVF float chunk page cannot store any vector payload.");
    }
    chunk_pages_per_vector_ = (dim_ + floats_per_chunk_page_ - 1) / floats_per_chunk_page_;
    list_entry_size_ = static_cast<u32>(IVFListDataPage::entrySize(dim_));
    entries_per_list_page_ = IVFListDataPage::entriesPerPage(dim_);
    if (entries_per_list_page_ == 0) {
        throw std::runtime_error("IVF list pages cannot store a full inline residual entry for this dimension.");
    }
    entries_per_dir_page_ = IVFListDirPage::entriesPerPage();
    dir_pages_ = (num_clusters_ + entries_per_dir_page_ - 1) / entries_per_dir_page_;

    bool has_existing_meta = false;
    bool params_match = false;
    PID existing_metadata_pid = BufferManager::invalidPID;

    {
        GuardX<MetaDataPage> shared_meta(metadataPageId);
        IVFMetaInfo* anchor = &shared_meta->ivf_meta;
        has_existing_meta = anchor->isValid();
        params_match = has_existing_meta && anchor->dim == dim_ && anchor->num_clusters == num_clusters_ &&
                       anchor->max_elements >= max_elements_;
        if (has_existing_meta) {
            existing_metadata_pid = anchor->metadata_pid;
        }
    }

    if (params_match) {
        GuardO<IVFMetadataPage> meta_guard(existing_metadata_pid);
        params_match = meta_guard->dim == dim_ && meta_guard->num_clusters == num_clusters_ &&
                       meta_guard->retrain_interval == retrain_interval_ &&
                       meta_guard->layout_version == IVF_LAYOUT_VERSION &&
                       meta_guard->floats_per_chunk_page == floats_per_chunk_page_ &&
                       meta_guard->chunk_pages_per_vector == chunk_pages_per_vector_ &&
                       meta_guard->entries_per_list_page == entries_per_list_page_ &&
                       meta_guard->list_entry_size == list_entry_size_ &&
                       meta_guard->entries_per_dir_page == entries_per_dir_page_ &&
                       meta_guard->dir_pages == dir_pages_;
    }

    if (has_existing_meta && !params_match) {
        GuardX<MetaDataPage> shared_meta(metadataPageId);
        shared_meta->ivf_meta.clear();
        shared_meta->dirty = true;
    }

    if (!skip_recovery && params_match) {
        metadata_pid_ = existing_metadata_pid;
        GuardO<IVFMetadataPage> meta_guard(metadata_pid_);
        centroids_base_pid_ = meta_guard->centroids_base_pid;
        invlist_dir_base_pid_ = meta_guard->invlist_dir_base_pid;
        max_elements_ = meta_guard->max_elements;
        is_trained_.store(meta_guard->is_trained != 0, std::memory_order_release);
        recovered_from_disk_ = true;
    } else {
        if (has_existing_meta && params_match) {
            metadata_pid_ = existing_metadata_pid;
            GuardO<IVFMetadataPage> meta_guard(metadata_pid_);
            centroids_base_pid_ = meta_guard->centroids_base_pid;
            invlist_dir_base_pid_ = meta_guard->invlist_dir_base_pid;
            resetExistingStorage();
        } else {
            initializeFreshStorage();
        }

        publishSharedMetadata(false);
        recovered_from_disk_ = false;
    }
}

IVFIndex::~IVFIndex() {
    search_thread_pool_.reset(nullptr);
    add_thread_pool_.reset(nullptr);
}

void IVFIndex::initializeFreshStorage() {
    AllocGuard<IVFMetadataPage> meta_guard;
    metadata_pid_ = meta_guard.pid;
    meta_guard->dim = dim_;
    meta_guard->num_clusters = num_clusters_;
    meta_guard->retrain_interval = retrain_interval_;
    meta_guard->layout_version = IVF_LAYOUT_VERSION;
    meta_guard->floats_per_chunk_page = floats_per_chunk_page_;
    meta_guard->chunk_pages_per_vector = chunk_pages_per_vector_;
    meta_guard->entries_per_list_page = entries_per_list_page_;
    meta_guard->list_entry_size = list_entry_size_;
    meta_guard->entries_per_dir_page = entries_per_dir_page_;
    meta_guard->dir_pages = dir_pages_;
    meta_guard->max_elements = max_elements_;
    meta_guard->num_vectors.store(0, std::memory_order_relaxed);
    meta_guard->last_train_count.store(0, std::memory_order_relaxed);
    meta_guard->is_trained = 0;

    const u64 total_centroid_pages = static_cast<u64>(num_clusters_) * chunk_pages_per_vector_;
    if (total_centroid_pages == 0) {
        throw std::runtime_error("IVF requires at least one centroid page.");
    }

    for (u64 page_idx = 0; page_idx < total_centroid_pages; ++page_idx) {
        AllocGuard<IVFFloatChunkPage> chunk_guard;
        if (page_idx == 0) {
            centroids_base_pid_ = chunk_guard.pid;
        }
        const u32 chunk_idx = static_cast<u32>(page_idx % chunk_pages_per_vector_);
        chunk_guard->next_page = BufferManager::invalidPID;
        chunk_guard->float_count = chunkFloatCount(chunk_idx);
        std::memset(chunk_guard->getData(), 0, chunk_guard->float_count * sizeof(float));
        chunk_guard->dirty = true;
    }

    for (u32 dir_idx = 0; dir_idx < dir_pages_; ++dir_idx) {
        AllocGuard<IVFListDirPage> dir_guard;
        if (dir_idx == 0) {
            invlist_dir_base_pid_ = dir_guard.pid;
        }
        const u32 first_cluster = dir_idx * entries_per_dir_page_;
        const u32 count = std::min<u32>(entries_per_dir_page_, num_clusters_ - first_cluster);
        dir_guard->entry_count = count;
        dir_guard->first_cluster_id = first_cluster;
        for (u32 i = 0; i < count; ++i) {
            dir_guard->getEntry(i)->init();
        }
        dir_guard->dirty = true;
    }

    meta_guard->centroids_base_pid = centroids_base_pid_;
    meta_guard->invlist_dir_base_pid = invlist_dir_base_pid_;
    meta_guard->dirty = true;
    is_trained_.store(false, std::memory_order_release);
}

void IVFIndex::resetExistingStorage() {
    GuardX<IVFMetadataPage> meta_guard(metadata_pid_);
    meta_guard->dim = dim_;
    meta_guard->num_clusters = num_clusters_;
    meta_guard->retrain_interval = retrain_interval_;
    meta_guard->layout_version = IVF_LAYOUT_VERSION;
    meta_guard->floats_per_chunk_page = floats_per_chunk_page_;
    meta_guard->chunk_pages_per_vector = chunk_pages_per_vector_;
    meta_guard->entries_per_list_page = entries_per_list_page_;
    meta_guard->list_entry_size = list_entry_size_;
    meta_guard->entries_per_dir_page = entries_per_dir_page_;
    meta_guard->dir_pages = dir_pages_;
    meta_guard->max_elements = max_elements_;
    meta_guard->num_vectors.store(0, std::memory_order_relaxed);
    meta_guard->last_train_count.store(0, std::memory_order_relaxed);
    meta_guard->is_trained = 0;
    centroids_base_pid_ = meta_guard->centroids_base_pid;
    invlist_dir_base_pid_ = meta_guard->invlist_dir_base_pid;

    for (u32 dir_idx = 0; dir_idx < dir_pages_; ++dir_idx) {
        GuardX<IVFListDirPage> dir_guard(invlist_dir_base_pid_ + dir_idx);
        const u32 first_cluster = dir_idx * entries_per_dir_page_;
        const u32 count = std::min<u32>(entries_per_dir_page_, num_clusters_ - first_cluster);
        dir_guard->entry_count = count;
        dir_guard->first_cluster_id = first_cluster;
        for (u32 i = 0; i < count; ++i) {
            dir_guard->getEntry(i)->init();
        }
        dir_guard->dirty = true;
    }

    is_trained_.store(false, std::memory_order_release);
}

void IVFIndex::publishSharedMetadata(bool trained_flag) {
    GuardX<MetaDataPage> shared_meta(metadataPageId);
    IVFMetaInfo* anchor = &shared_meta->ivf_meta;
    anchor->magic_value = IVFMetaInfo::magic;
    anchor->metadata_pid = metadata_pid_;
    anchor->max_elements = max_elements_;
    anchor->dim = dim_;
    anchor->num_clusters = num_clusters_;
    anchor->retrain_interval = retrain_interval_;
    anchor->is_trained = trained_flag ? 1 : 0;
    anchor->valid = 1;
    shared_meta->dirty = true;
}

PIDAllocator* IVFIndex::clusterPidAllocator(u32 cluster_id) {
    return &cluster_pid_allocators_[cluster_id];
}

u32 IVFIndex::chunkFloatCount(u32 chunk_idx) const {
    const u32 offset = chunk_idx * floats_per_chunk_page_;
    if (offset >= dim_) {
        return 0;
    }
    return std::min<u32>(floats_per_chunk_page_, dim_ - offset);
}

PID IVFIndex::centroidChunkPid(u32 cluster_id, u32 chunk_idx) const {
    return centroids_base_pid_ + static_cast<PID>(cluster_id) * chunk_pages_per_vector_ + chunk_idx;
}

float IVFIndex::computeDistanceToCentroid(const float* vector, u32 cluster_id) const {
    float distance = 0.0f;
    u32 offset = 0;
    for (u32 chunk_idx = 0; chunk_idx < chunk_pages_per_vector_ && offset < dim_; ++chunk_idx) {
        const PID current_pid = centroidChunkPid(cluster_id, chunk_idx);
        if (enable_prefetch_ && chunk_idx + 1 < chunk_pages_per_vector_) {
            const PID next_pid = centroidChunkPid(cluster_id, chunk_idx + 1);
            u32 zero = 0;
            bm.prefetchPages(&next_pid, 1, &zero);
        }

        GuardORelaxed<IVFFloatChunkPage> chunk_guard(current_pid);
        const u32 float_count = std::min<u32>(chunk_guard->float_count, dim_ - offset);
        distance += computeDistance(vector + offset, chunk_guard->getData(), float_count);
        offset += float_count;
    }
    return distance;
}

void IVFIndex::computeResidualToCentroid(const float* vector, u32 cluster_id, float* residual) const {
    u32 offset = 0;
    for (u32 chunk_idx = 0; chunk_idx < chunk_pages_per_vector_ && offset < dim_; ++chunk_idx) {
        const PID current_pid = centroidChunkPid(cluster_id, chunk_idx);
        if (enable_prefetch_ && chunk_idx + 1 < chunk_pages_per_vector_) {
            const PID next_pid = centroidChunkPid(cluster_id, chunk_idx + 1);
            u32 zero = 0;
            bm.prefetchPages(&next_pid, 1, &zero);
        }

        GuardORelaxed<IVFFloatChunkPage> chunk_guard(current_pid);
        const u32 float_count = std::min<u32>(chunk_guard->float_count, dim_ - offset);
        const float* centroid_chunk = chunk_guard->getData();
        for (u32 idx = 0; idx < float_count; ++idx) {
            residual[offset + idx] = vector[offset + idx] - centroid_chunk[idx];
        }
        offset += float_count;
    }
}

void IVFIndex::storeCentroid(u32 cluster_id, const float* centroid) {
    if (cluster_id >= num_clusters_) {
        throw std::runtime_error("Invalid centroid cluster id.");
    }
    u32 offset = 0;
    for (u32 chunk_idx = 0; chunk_idx < chunk_pages_per_vector_; ++chunk_idx) {
        GuardX<IVFFloatChunkPage> chunk_guard(centroidChunkPid(cluster_id, chunk_idx));
        const u32 float_count = chunkFloatCount(chunk_idx);
        chunk_guard->next_page = BufferManager::invalidPID;
        chunk_guard->float_count = float_count;
        std::memcpy(chunk_guard->getData(), centroid + offset, float_count * sizeof(float));
        chunk_guard->dirty = true;
        offset += float_count;
    }
}

void IVFIndex::initializeCentroidsKMeansPlusPlus(const float* vectors, u64 n, float* centroids) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<u64> sample_dist(0, n - 1);
    const u64 first_idx = sample_dist(rng);
    std::memcpy(centroids, vectors + first_idx * dim_, dim_ * sizeof(float));

    std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    for (u32 centroid_idx = 1; centroid_idx < num_clusters_; ++centroid_idx) {
        const float* last_centroid = centroids + static_cast<size_t>(centroid_idx - 1) * dim_;
        double total_distance = 0.0;
        for (u64 i = 0; i < n; ++i) {
            const float distance = computeDistance(vectors + i * dim_, last_centroid, dim_);
            if (distance < min_distances[i]) {
                min_distances[i] = distance;
            }
            total_distance += min_distances[i];
        }

        std::uniform_real_distribution<double> choose_dist(0.0, total_distance);
        const double target = choose_dist(rng);
        double cumulative = 0.0;
        u64 chosen_idx = 0;
        for (u64 i = 0; i < n; ++i) {
            cumulative += min_distances[i];
            if (cumulative >= target) {
                chosen_idx = i;
                break;
            }
        }

        std::memcpy(centroids + static_cast<size_t>(centroid_idx) * dim_, vectors + chosen_idx * dim_,
                    dim_ * sizeof(float));
    }
}

void IVFIndex::kmeansStep(const float* vectors, u64 n, float* centroids, std::vector<u32>& assignments) {
    std::vector<float> new_centroids(static_cast<size_t>(num_clusters_) * dim_, 0.0f);
    std::vector<u64> cluster_counts(num_clusters_, 0);
    std::mt19937 rng(1337);
    std::uniform_int_distribution<u64> sample_dist(0, n - 1);

    for (u64 i = 0; i < n; ++i) {
        const float* vector = vectors + i * dim_;
        float best_distance = std::numeric_limits<float>::max();
        u32 best_cluster = 0;
        for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
            const float* centroid = centroids + static_cast<size_t>(cluster_id) * dim_;
            const float distance = computeDistance(vector, centroid, dim_);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster_id;
            }
        }
        assignments[i] = best_cluster;
        cluster_counts[best_cluster] += 1;
        float* sum = new_centroids.data() + static_cast<size_t>(best_cluster) * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            sum[d] += vector[d];
        }
    }

    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        float* centroid = centroids + static_cast<size_t>(cluster_id) * dim_;
        if (cluster_counts[cluster_id] == 0) {
            const u64 fallback_idx = sample_dist(rng);
            std::memcpy(centroid, vectors + fallback_idx * dim_, dim_ * sizeof(float));
            continue;
        }

        const float inv_count = 1.0f / static_cast<float>(cluster_counts[cluster_id]);
        const float* sum = new_centroids.data() + static_cast<size_t>(cluster_id) * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            centroid[d] = sum[d] * inv_count;
        }
    }
}

void IVFIndex::train(const float* training_vectors, u64 n_train, u32 kmeans_iters) {
    if (n_train < num_clusters_) {
        throw std::runtime_error("IVF training requires at least num_clusters vectors.");
    }
    if (kmeans_iters == 0) {
        kmeans_iters = 1;
    }

    std::vector<float> centroids(static_cast<size_t>(num_clusters_) * dim_);
    initializeCentroidsKMeansPlusPlus(training_vectors, n_train, centroids.data());

    std::vector<u32> assignments(n_train, 0);
    for (u32 iter = 0; iter < kmeans_iters; ++iter) {
        kmeansStep(training_vectors, n_train, centroids.data(), assignments);
    }

    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        storeCentroid(cluster_id, centroids.data() + static_cast<size_t>(cluster_id) * dim_);
    }

    GuardX<IVFMetadataPage> meta_guard(metadata_pid_);
    meta_guard->is_trained = 1;
    meta_guard->last_train_count.store(meta_guard->num_vectors.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
    meta_guard->dirty = true;

    is_trained_.store(true, std::memory_order_release);
    publishSharedMetadata(true);
}

u32 IVFIndex::findNearestCentroid(const float* vector) const {
    float best_distance = std::numeric_limits<float>::max();
    u32 best_cluster = 0;
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const float distance = computeDistanceToCentroid(vector, cluster_id);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster_id;
        }
    }
    return best_cluster;
}

std::vector<std::pair<float, u32>> IVFIndex::findNearestCentroids(const float* vector, size_t nprobe) const {
    const size_t probe_limit = std::min<size_t>(std::max<size_t>(1, nprobe), num_clusters_);
    std::priority_queue<std::pair<float, u32>> heap;

    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const float distance = computeDistanceToCentroid(vector, cluster_id);
        if (heap.size() < probe_limit) {
            heap.push({distance, cluster_id});
        } else if (distance < heap.top().first) {
            heap.pop();
            heap.push({distance, cluster_id});
        }
    }

    std::vector<std::pair<float, u32>> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
}

void IVFIndex::appendToInvList(u32 cluster_id, u32 vector_id, const float* residual) {
    const PID dir_page_pid = invlist_dir_base_pid_ + (cluster_id / entries_per_dir_page_);
    const u32 local_idx = cluster_id % entries_per_dir_page_;
    PIDAllocator* allocator = clusterPidAllocator(cluster_id);

    GuardX<IVFListDirPage> dir_guard(dir_page_pid);
    InvListEntry* entry = dir_guard->getEntry(local_idx);

    if (entry->first_page_pid == BufferManager::invalidPID) {
        AllocGuard<IVFListDataPage> list_guard(allocator);
        const PID list_pid = list_guard.pid;
        list_guard->next_page = BufferManager::invalidPID;
        list_guard->count = 0;
        list_guard->capacity = entries_per_list_page_;
        list_guard->dirty = true;

        entry->first_page_pid = list_pid;
        entry->last_page_pid = list_pid;
        entry->list_size.store(0, std::memory_order_release);
        entry->num_pages.store(1, std::memory_order_release);
        entry->cached_pages[0] = list_pid;
        entry->cached_page_count.store(1, std::memory_order_release);
    }

    PID current_page_pid = entry->last_page_pid;
    GuardX<IVFListDataPage> page_guard(current_page_pid);
    if (page_guard->count >= page_guard->capacity) {
        AllocGuard<IVFListDataPage> new_page_guard(allocator);
        const PID new_page_pid = new_page_guard.pid;
        new_page_guard->next_page = BufferManager::invalidPID;
        new_page_guard->count = 0;
        new_page_guard->capacity = entries_per_list_page_;
        new_page_guard->dirty = true;

        page_guard->next_page = new_page_pid;
        page_guard->dirty = true;

        const u32 old_pages = entry->num_pages.fetch_add(1, std::memory_order_relaxed);
        entry->last_page_pid = new_page_pid;
        if (old_pages < IVF_MAX_CACHED_PAGES) {
            entry->cached_pages[old_pages] = new_page_pid;
            entry->cached_page_count.store(old_pages + 1, std::memory_order_release);
        }

        IVFListVectorEntry* stored = new_page_guard->getEntry(new_page_guard->count++, list_entry_size_);
        stored->original_id = vector_id;
        stored->reserved = 0;
        std::memcpy(stored->getResidual(), residual, static_cast<size_t>(dim_) * sizeof(float));
        new_page_guard->dirty = true;
    } else {
        IVFListVectorEntry* stored = page_guard->getEntry(page_guard->count++, list_entry_size_);
        stored->original_id = vector_id;
        stored->reserved = 0;
        std::memcpy(stored->getResidual(), residual, static_cast<size_t>(dim_) * sizeof(float));
        page_guard->dirty = true;
    }

    entry->list_size.fetch_add(1, std::memory_order_relaxed);
    dir_guard->dirty = true;
}

void IVFIndex::addPoint(const float* vector, u32 vector_id) {
    if (!isTrained()) {
        throw std::runtime_error("IVF index must be trained before adding vectors.");
    }

    std::vector<float> residual(dim_);
    const u32 cluster_id = findNearestCentroid(vector);
    computeResidualToCentroid(vector, cluster_id, residual.data());

    appendToInvList(cluster_id, vector_id, residual.data());

    GuardX<IVFMetadataPage> meta_guard(metadata_pid_);
    const u64 new_count = meta_guard->num_vectors.fetch_add(1, std::memory_order_relaxed) + 1;
    if (retrain_interval_ > 0 && (new_count - meta_guard->last_train_count.load(std::memory_order_relaxed)) >=
                                     retrain_interval_) {
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
    meta_guard->dirty = true;
}

void IVFIndex::addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads) {
    if (!isTrained()) {
        throw std::runtime_error("IVF index must be trained before adding vectors.");
    }
    if (count == 0) {
        return;
    }
    if (count < 32 || normalizeThreadCount(num_threads) == 1) {
        for (u64 i = 0; i < count; ++i) {
            addPoint(vectors + i * dim_, ids[i]);
        }
        return;
    }

    std::vector<u32> cluster_ids(count, 0);
    std::vector<float> residuals(static_cast<size_t>(count) * dim_);
    for (u64 i = 0; i < count; ++i) {
        const float* vector = vectors + i * dim_;
        const u32 best_cluster = findNearestCentroid(vector);
        cluster_ids[i] = best_cluster;
        float* residual = residuals.data() + static_cast<size_t>(i) * dim_;
        computeResidualToCentroid(vector, best_cluster, residual);
    }

    std::vector<std::vector<u64>> cluster_to_indices(num_clusters_);
    for (u64 i = 0; i < count; ++i) {
        cluster_to_indices[cluster_ids[i]].push_back(i);
    }

    ThreadPool* pool = getOrCreateAddPool(num_threads);
    std::vector<std::future<void>> futures;
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        if (cluster_to_indices[cluster_id].empty()) {
            continue;
        }
        futures.push_back(pool->enqueue([this, cluster_id, &cluster_to_indices, &residuals, ids] {
            for (u64 idx : cluster_to_indices[cluster_id]) {
                const float* residual = residuals.data() + static_cast<size_t>(idx) * dim_;
                appendToInvList(cluster_id, ids[idx], residual);
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }

    GuardX<IVFMetadataPage> meta_guard(metadata_pid_);
    const u64 new_count = meta_guard->num_vectors.fetch_add(count, std::memory_order_relaxed) + count;
    if (retrain_interval_ > 0 && (new_count - meta_guard->last_train_count.load(std::memory_order_relaxed)) >=
                                     retrain_interval_) {
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
    meta_guard->dirty = true;
}

u32 IVFIndex::scanClusterList(u32 cluster_id, const float* query_residual, size_t k,
                              std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                              bool enable_pruning, bool collect_stats) const {
    const PID dir_page_pid = invlist_dir_base_pid_ + (cluster_id / entries_per_dir_page_);
    const u32 local_idx = cluster_id % entries_per_dir_page_;

    GuardORelaxed<IVFListDirPage> dir_guard(dir_page_pid);
    const InvListEntry* entry = dir_guard->getEntry(local_idx);
    if (entry->list_size.load(std::memory_order_acquire) == 0) {
        return 0;
    }

    const u32 cached_count = enable_prefetch_
                                 ? std::min<u32>(entry->cached_page_count.load(std::memory_order_acquire),
                                                 IVF_MAX_CACHED_PAGES)
                                 : 0;

    PID current_page = entry->first_page_pid;
    u32 vectors_scanned = 0;
    u64 total_dist_blocks = 0;
    u32 scanned_pages = 0;

    while (current_page != BufferManager::invalidPID) {
        if (enable_prefetch_ && scanned_pages + 1 < cached_count && (scanned_pages % IVF_PREFETCH_BATCH_PAGES) == 0) {
            PID page_pids[IVF_PREFETCH_BATCH_PAGES];
            u32 offsets[IVF_PREFETCH_BATCH_PAGES];
            u32 prefetch_count = 0;
            const u32 prefetch_limit = std::min<u32>(cached_count, scanned_pages + 1 + IVF_PREFETCH_BATCH_PAGES);
            for (u32 i = scanned_pages + 1; i < prefetch_limit; ++i) {
                const PID cached_pid = entry->cached_pages[i];
                if (cached_pid != BufferManager::invalidPID) {
                    page_pids[prefetch_count] = cached_pid;
                    offsets[prefetch_count] = 0;
                    ++prefetch_count;
                }
            }
            if (prefetch_count > 0) {
                bm.prefetchPages(page_pids, prefetch_count, offsets);
            }
        }

        GuardORelaxed<IVFListDataPage> list_guard(current_page);
        const IVFListDataPage* list_page = list_guard.ptr;
        const u32 count = std::min<u32>(list_page->count, entries_per_list_page_);
        PID next_page_id = list_page->next_page;

        if (scanned_pages >= cached_count && enable_prefetch_ && next_page_id != BufferManager::invalidPID) {
            // prefetch the next page
            u32 zero = 0;
            bm.prefetchPages(&next_page_id, 1, &zero);
        }

        for (u32 i = 0; i < count; ++i) {
            const IVFListVectorEntry* stored = list_page->getEntry(i, list_entry_size_);
            const float* residual = stored->getResidual();
            float distance = 0.0f;
            bool pruned = false;

            for (u32 offset = 0; offset < dim_; offset += IVF_PRUNE_BLOCK_FLOATS) {
                const u32 block = std::min<u32>(IVF_PRUNE_BLOCK_FLOATS, dim_ - offset);
                distance += computeDistance(query_residual + offset, residual + offset, block);
                ++total_dist_blocks;
                if (enable_pruning && result_heap.size() >= k && std::isfinite(heap_threshold) &&
                    distance > heap_threshold) {
                    pruned = true;
                    break;
                }
            }

            if (!pruned) {
                if (result_heap.size() < k) {
                    result_heap.push({distance, stored->original_id});
                    if (result_heap.size() == k) {
                        heap_threshold = result_heap.top().first;
                    }
                } else if (distance < heap_threshold) {
                    result_heap.pop();
                    result_heap.push({distance, stored->original_id});
                    heap_threshold = result_heap.top().first;
                }
            }

            ++vectors_scanned;
        }

        current_page = next_page_id;
        ++scanned_pages;
    }

    if (collect_stats) {
        stats_.dist_comps.fetch_add(total_dist_blocks, std::memory_order_relaxed);
        stats_.vectors_scanned.fetch_add(vectors_scanned, std::memory_order_relaxed);
    }
    return vectors_scanned;
}

std::vector<std::pair<float, u32>> IVFIndex::search(const float* query, size_t k, size_t nprobe, bool collect_stats) {
    if (!isTrained()) {
        throw std::runtime_error("IVF index must be trained before searching.");
    }
    if (k == 0) {
        return {};
    }

    for (u64 retry_counter = 0;; ++retry_counter) {
        try {
            auto nearest_clusters = findNearestCentroids(query, nprobe);
            if (collect_stats) {
                stats_.lists_probed.fetch_add(nearest_clusters.size(), std::memory_order_relaxed);
            }

            std::priority_queue<std::pair<float, u32>> result_heap;
            float heap_threshold = std::numeric_limits<float>::max();
            std::vector<float> query_residual(dim_);

            for (size_t cluster_idx = 0; cluster_idx < nearest_clusters.size(); ++cluster_idx) {
                const u32 cluster_id = nearest_clusters[cluster_idx].second;
                computeResidualToCentroid(query, cluster_id, query_residual.data());
                const bool enable_pruning = cluster_idx > 0;
                scanClusterList(cluster_id, query_residual.data(), k, result_heap, heap_threshold, enable_pruning,
                                collect_stats);
            }

            std::vector<std::pair<float, u32>> results;
            results.reserve(result_heap.size());
            while (!result_heap.empty()) {
                results.push_back(result_heap.top());
                result_heap.pop();
            }
            std::reverse(results.begin(), results.end());
            return results;
        } catch (const OLCRestartException&) {
            yield(retry_counter);
        }
    }
}

std::vector<std::vector<std::pair<float, u32>>> IVFIndex::searchBatch(std::span<const float> queries, size_t k,
                                                                      size_t nprobe, size_t num_threads,
                                                                      bool collect_stats) {
    const size_t num_queries = queries.size() / dim_;
    std::vector<std::vector<std::pair<float, u32>>> results(num_queries);
    const u32 resolved_threads = normalizeThreadCount(num_threads);

    if (resolved_threads == 1 || num_queries <= 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries.data() + i * dim_, k, nprobe, collect_stats);
        }
        return results;
    }

    ThreadPool* pool = getOrCreateSearchPool(resolved_threads);
    std::vector<std::future<std::vector<std::pair<float, u32>>>> futures;
    futures.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        const float* query = queries.data() + i * dim_;
        futures.push_back(pool->enqueue([this, query, k, nprobe, collect_stats] {
            return search(query, k, nprobe, collect_stats);
        }));
    }
    for (size_t i = 0; i < num_queries; ++i) {
        results[i] = futures[i].get();
    }
    return results;
}

IVFStats IVFIndex::getStats() const {
    IVFStats snapshot(stats_);
    snapshot.num_clusters = num_clusters_;
    snapshot.list_sizes.assign(num_clusters_, 0);

    u64 total_vectors = 0;
    for (u32 dir_idx = 0; dir_idx < dir_pages_; ++dir_idx) {
        GuardORelaxed<IVFListDirPage> dir_guard(invlist_dir_base_pid_ + dir_idx);
        const u32 count = std::min<u32>(dir_guard->entry_count, num_clusters_ - dir_guard->first_cluster_id);
        for (u32 i = 0; i < count; ++i) {
            const u64 list_size = dir_guard->getEntry(i)->list_size.load(std::memory_order_relaxed);
            const u32 cluster_id = dir_guard->first_cluster_id + i;
            snapshot.list_sizes[cluster_id] = list_size;
            total_vectors += list_size;
        }
    }
    snapshot.avg_list_size = num_clusters_ == 0 ? 0.0 : static_cast<double>(total_vectors) / num_clusters_;
    return snapshot;
}

void IVFIndex::flush() {
    flush_system();
}

u64 IVFIndex::size() const {
    GuardO<IVFMetadataPage> meta_guard(metadata_pid_);
    return meta_guard->num_vectors.load(std::memory_order_acquire);
}

std::string IVFIndex::getIndexInfo() const {
    std::ostringstream out;
    out << "=== IVF Index Information ===\n";
    out << "Configuration:\n";
    out << "  Vector Dimension: " << dim_ << "\n";
    out << "  Num Clusters: " << num_clusters_ << "\n";
    out << "  Max Elements: " << max_elements_ << "\n";
    out << "  Chunk Floats/Page: " << floats_per_chunk_page_ << "\n";
    out << "  Chunk Pages/Vector: " << chunk_pages_per_vector_ << "\n";
    out << "  List Entry Size: " << list_entry_size_ << "\n";
    out << "  List Entries/Page: " << entries_per_list_page_ << "\n";
    out << "  Dir Entries/Page: " << entries_per_dir_page_ << "\n";
    out << "  Prefetch Enabled: " << (enable_prefetch_ ? "true" : "false") << "\n";

    GuardO<IVFMetadataPage> meta_guard(metadata_pid_);
    out << "\nMetadata:\n";
    out << "  Metadata PID: " << metadata_pid_ << "\n";
    out << "  Centroid Base PID: " << meta_guard->centroids_base_pid << "\n";
    out << "  List Dir Base PID: " << meta_guard->invlist_dir_base_pid << "\n";
    out << "  Current Vector Count: " << meta_guard->num_vectors.load(std::memory_order_acquire) << "\n";
    out << "  Trained: " << (meta_guard->is_trained ? "true" : "false") << "\n";

    IVFStats stats = getStats();
    out << "\nRuntime Stats:\n";
    out << "  Lists Probed: " << stats.lists_probed.load() << "\n";
    out << "  Vectors Scanned: " << stats.vectors_scanned.load() << "\n";
    out << "  Distance Blocks: " << stats.dist_comps.load() << "\n";
    out << "  Avg List Size: " << std::fixed << std::setprecision(2) << stats.avg_list_size << "\n";
    out << "=== End IVF Index Information ===\n";
    return out.str();
}