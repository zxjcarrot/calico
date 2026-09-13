#include "pdx.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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

size_t resolvedFloatPageCount(const PDXFloatPage& page, bool is_last_page) {
    if (!is_last_page) {
        return static_cast<size_t>(PDXFloatPage::floatsPerPage());
    }
    return page.float_count == 0 ? static_cast<size_t>(PDXFloatPage::floatsPerPage())
                                 : static_cast<size_t>(page.float_count);
}

size_t resolvedIdPageCount(const PDXIdPage& page, bool is_last_page) {
    if (!is_last_page) {
        return static_cast<size_t>(PDXIdPage::idsPerPage());
    }
    return page.id_count == 0 ? static_cast<size_t>(PDXIdPage::idsPerPage()) : static_cast<size_t>(page.id_count);
}

void reserveContiguousAllocatorPages(PIDAllocator* allocator, u32 required_pages) {
    if (allocator == nullptr || required_pages == 0) {
        return;
    }

    allocator->lockSpin();
    if (allocator->next + required_pages > allocator->end) {
        const u64 start = bm.allocCount.fetch_add(required_pages, std::memory_order_acq_rel);
        allocator->next = start;
        allocator->end = start + required_pages;
    }
    allocator->unlockSpin();
}

u32 normalizeThreadCount(size_t num_threads) {
    if (num_threads == 0) {
        const u32 hw = std::thread::hardware_concurrency();
        return std::max<u32>(1, hw == 0 ? 1 : hw);
    }
    return static_cast<u32>(std::max<size_t>(1, num_threads));
}

float readEnvProbeMultiplier() {
    const char* raw = std::getenv("PDX_NPROBE_MULTIPLIER");
    if (raw == nullptr || raw[0] == '\0') {
        return PDX_DEFAULT_NPROBE_MULTIPLIER;
    }
    char* end = nullptr;
    const float parsed = std::strtof(raw, &end);
    if (end == raw || !std::isfinite(parsed) || parsed < 1.0f) {
        return PDX_DEFAULT_NPROBE_MULTIPLIER;
    }
    return parsed;
}

u32 readAsyncPrefetchLookaheadWindow() {
    static const u32 lookahead = []() {
        constexpr u32 default_lookahead = 32;
        const char* raw = std::getenv("PDX_ASYNC_PREFETCH_WINDOW");
        if (raw == nullptr || raw[0] == '\0') {
            return std::min<u32>(default_lookahead, PDX_MAX_ASYNC_PREFETCH_WINDOW);
        }
        char* end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed < 0) {
            return std::min<u32>(default_lookahead, PDX_MAX_ASYNC_PREFETCH_WINDOW);
        }
        return std::min<u32>(static_cast<u32>(parsed), PDX_MAX_ASYNC_PREFETCH_WINDOW);
    }();
    return lookahead;
}

constexpr size_t pdxFetchScheduleLength() {
    return sizeof(PDX_DIMENSIONS_FETCHING_SIZES) / sizeof(PDX_DIMENSIONS_FETCHING_SIZES[0]);
}

u32 nextFetchSize(size_t fetch_round) {
    if (fetch_round >= pdxFetchScheduleLength()) {
        return PDX_DIMENSIONS_FETCHING_SIZES[pdxFetchScheduleLength() - 1];
    }
    return PDX_DIMENSIONS_FETCHING_SIZES[fetch_round];
}

#if defined(__GNUC__) || defined(__clang__)
#define PDX_ALWAYS_INLINE inline __attribute__((always_inline))
#define PDX_RESTRICT_PTR __restrict__
#else
#define PDX_ALWAYS_INLINE inline
#define PDX_RESTRICT_PTR
#endif

#if defined(__clang__)
#define PDX_ENABLE_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define PDX_ENABLE_VECTORIZE_LOOP _Pragma("GCC ivdep")
#else
#define PDX_ENABLE_VECTORIZE_LOOP
#endif

PDX_ALWAYS_INLINE float pdxHorizontalDistanceScalar(const float* PDX_RESTRICT_PTR lhs,
                                                    const float* PDX_RESTRICT_PTR rhs, size_t count) {
    float distance = 0.0f;
    PDX_ENABLE_VECTORIZE_LOOP
    for (size_t idx = 0; idx < count; ++idx) {
        const float delta = lhs[idx] - rhs[idx];
        distance += delta * delta;
    }
    return distance;
}

PDX_ALWAYS_INLINE void accumulateVerticalDistancesScalar(float query_value, const float* PDX_RESTRICT_PTR src,
                                                         float* PDX_RESTRICT_PTR dst, size_t count) {
    PDX_ENABLE_VECTORIZE_LOOP
    for (size_t idx = 0; idx < count; ++idx) {
        const float delta = query_value - src[idx];
        dst[idx] += delta * delta;
    }
}

#if defined(__AVX__)
inline __m256 fusedMultiplyAdd(__m256 lhs, __m256 rhs, __m256 acc) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(lhs, rhs, acc);
#else
    return _mm256_add_ps(acc, _mm256_mul_ps(lhs, rhs));
#endif
}

inline float pdxHorizontalDistance(const float* lhs, const float* rhs, size_t count) {
    return hnsw_distance::SIMDAcceleratedL2::compare(lhs, rhs, count);
}

// inline float pdxHorizontalDistance(const float* PDX_RESTRICT_PTR lhs, const float* PDX_RESTRICT_PTR rhs,
//                                    size_t count) {
//     if (count <= 16) {
//         return pdxHorizontalDistanceScalar(lhs, rhs, count);
//     }

//     __m256 accum = _mm256_setzero_ps();
//     size_t idx = 0;
//     for (; idx + 8 <= count; idx += 8) {
//         const __m256 left = _mm256_loadu_ps(lhs + idx);
//         const __m256 right = _mm256_loadu_ps(rhs + idx);
//         const __m256 delta = _mm256_sub_ps(left, right);
//         accum = fusedMultiplyAdd(delta, delta, accum);
//     }

//     const __m128 low_f32 = _mm256_castps256_ps128(accum);
//     const __m128 high_f32 = _mm256_extractf128_ps(accum, 1);
//     const __m256d low_f64 = _mm256_cvtps_pd(low_f32);
//     const __m256d high_f64 = _mm256_cvtps_pd(high_f32);
//     const __m256d sum = _mm256_add_pd(low_f64, high_f64);
//     const __m128d sum_low = _mm256_castpd256_pd128(sum);
//     const __m128d sum_high = _mm256_extractf128_pd(sum, 1);
//     __m128d total = _mm_add_pd(sum_low, sum_high);
//     total = _mm_hadd_pd(total, total);

//     double distance = _mm_cvtsd_f64(total);
//     for (; idx < count; ++idx) {
//         const float delta = lhs[idx] - rhs[idx];
//         distance += static_cast<double>(delta) * static_cast<double>(delta);
//     }
//     return static_cast<float>(distance);
// }

// inline void accumulateVerticalDistances(float query_value, const float* PDX_RESTRICT_PTR src,
//                                         float* PDX_RESTRICT_PTR dst, size_t count) {
//     const __m256 query = _mm256_set1_ps(query_value);
//     size_t idx = 0;
//     for (; idx + 8 <= count; idx += 8) {
//         const __m256 values = _mm256_loadu_ps(src + idx);
//         const __m256 current = _mm256_loadu_ps(dst + idx);
//         const __m256 delta = _mm256_sub_ps(query, values);
//         const __m256 updated = fusedMultiplyAdd(delta, delta, current);
//         _mm256_storeu_ps(dst + idx, updated);
//     }
//     if (idx < count) {
//         accumulateVerticalDistancesScalar(query_value, src + idx, dst + idx, count - idx);
//     }
// }
inline void accumulateVerticalDistances(float query_value, const float* src, float* dst, size_t count) {
    accumulateVerticalDistancesScalar(query_value, src, dst, count);
}
#else
inline float pdxHorizontalDistance(const float* lhs, const float* rhs, size_t count) {
    return hnsw_distance::SIMDAcceleratedL2::compare(lhs, rhs, count);
}

inline void accumulateVerticalDistances(float query_value, const float* src, float* dst, size_t count) {
    accumulateVerticalDistancesScalar(query_value, src, dst, count);
}
#endif

inline void accumulateVerticalSelectedDistances(float query_value, const float* src, size_t row_begin, size_t row_end,
                                                size_t row_base, const u32* positions, size_t position_count,
                                                size_t& position_idx, std::vector<float>& distances) {
    while (position_idx < position_count) {
        const u32 row = positions[position_idx];
        const size_t local_row = row - row_base;
        if (local_row >= row_end) {
            break;
        }

        size_t run_length = 1;
        while (position_idx + run_length < position_count) {
            const u32 next_row = positions[position_idx + run_length];
            const size_t next_local_row = next_row - row_base;
            if (next_local_row >= row_end || next_local_row != local_row + run_length) {
                break;
            }
            ++run_length;
        }

        if (run_length > 1) {
            accumulateVerticalDistances(query_value, src + (local_row - row_begin), distances.data() + row, run_length);
            position_idx += run_length;
            continue;
        }

        const float delta = query_value - src[local_row - row_begin];
        distances[row] += delta * delta;
        ++position_idx;
    }
}

}  // namespace

std::string PDXStats::toString() const {
    std::ostringstream out;
    out << "--- PDX Performance Stats ---\n";
    out << "Distance Components: " << dist_comps.load() << "\n";
    out << "Lists Probed:         " << lists_probed.load() << "\n";
    out << "Vectors Scanned:      " << vectors_scanned.load() << "\n";
    out << "Vectors Pruned:       " << vectors_pruned.load() << "\n";
    const u64 horizontal_samples = horizontal_selected_page_samples.load();
    if (horizontal_samples > 0) {
        const u64 horizontal_transitions = horizontal_selected_page_transitions.load();
        out << "Horizontal Selected Samples:          " << horizontal_samples << "\n";
        out << "Horizontal Selected Transitions:      " << horizontal_transitions << "\n";
        out << "Horizontal Logical Adjacent Ratio:    "
            << (horizontal_transitions == 0
                    ? 0.0
                    : static_cast<double>(horizontal_selected_logical_adjacent_page.load()) / horizontal_transitions)
            << "\n";
        out << "Horizontal Payload Adjacent Ratio:    "
            << (horizontal_transitions == 0
                    ? 0.0
                    : static_cast<double>(horizontal_selected_payload_adjacent_page.load()) / horizontal_transitions)
            << "\n";
        out << "Horizontal Logical Avg Abs Delta:     "
            << (horizontal_transitions == 0
                    ? 0.0
                    : static_cast<double>(horizontal_selected_logical_abs_delta_sum.load()) / horizontal_transitions)
            << "\n";
        out << "Horizontal Payload Avg Abs Delta:     "
            << (horizontal_transitions == 0
                    ? 0.0
                    : static_cast<double>(horizontal_selected_payload_abs_delta_sum.load()) / horizontal_transitions)
            << "\n";
    }
    const u64 replay_queries = replay_pid_stream_queries.load();
    if (replay_queries > 0) {
        const u64 live_touches = replay_pid_stream_calico_live_touches.load();
        const u64 bypass_touches = replay_pid_stream_calico_bypass_touches.load();
        const u64 vmcache_touches = replay_pid_stream_vmcache_touches.load();
        out << "Replay PID Stream Queries:             " << replay_queries << "\n";
        out << "Replay PID Stream Samples:             " << replay_pid_stream_samples.load() << "\n";
        out << "Replay PID Stream Resident Samples:    " << replay_pid_stream_resident_samples.load() << "\n";
        if (live_touches > 0) {
            out << "Replay Calico Live ns/access:          "
                << static_cast<double>(replay_pid_stream_calico_live_ns.load()) / static_cast<double>(live_touches)
                << "\n";
        }
        if (bypass_touches > 0) {
            out << "Replay Calico Bypass ns/access:        "
                << static_cast<double>(replay_pid_stream_calico_bypass_ns.load()) /
                       static_cast<double>(bypass_touches)
                << "\n";
        }
        if (vmcache_touches > 0) {
            out << "Replay Vmcache ns/access:              "
                << static_cast<double>(replay_pid_stream_vmcache_ns.load()) / static_cast<double>(vmcache_touches)
                << "\n";
        }
    }
    out << "Number of Clusters:   " << num_clusters << "\n";
    out << "Average Cluster Size: " << std::fixed << std::setprecision(2) << avg_cluster_size << "\n";
    return out.str();
}

ThreadPool* PDXIndex::getOrCreateSearchPool(size_t num_threads) const {
    std::lock_guard<std::mutex> guard(search_pool_mutex_);
    const size_t resolved = normalizeThreadCount(num_threads);
    if (!search_thread_pool_ || search_pool_size_ != resolved) {
        search_thread_pool_.reset(new ThreadPool(resolved));
        search_pool_size_ = resolved;
    }
    return search_thread_pool_.get();
}

ThreadPool* PDXIndex::getOrCreateAddPool(size_t num_threads) const {
    std::lock_guard<std::mutex> guard(add_pool_mutex_);
    const size_t resolved = normalizeThreadCount(num_threads);
    if (!add_thread_pool_ || add_pool_size_ != resolved) {
        add_thread_pool_.reset(new ThreadPool(resolved));
        add_pool_size_ = resolved;
    }
    return add_thread_pool_.get();
}

PDXIndex::PDXIndex(u64 max_elements, size_t dim, u32 num_clusters, u32 retrain_interval, bool enable_prefetch,
                   bool skip_recovery, float sampling_fraction, u32 transform_seed)
    : dim_(static_cast<u32>(dim)),
      num_clusters_(num_clusters),
      retrain_interval_(retrain_interval),
      max_elements_(max_elements),
      enable_prefetch_(enable_prefetch),
      sampling_fraction_(sampling_fraction),
      pruning_aggressiveness_(PDX_DEFAULT_PRUNING_AGGRESSIVENESS),
            nprobe_multiplier_(readEnvProbeMultiplier()),
            disable_pruning_(envOr("PDX_DISABLE_PRUNING", 0) != 0),
      transform_seed_(transform_seed),
      cluster_pid_allocators_(std::make_unique<PIDAllocator[]>(std::max<u32>(1, num_clusters))) {
    if (dim_ == 0) {
        throw std::runtime_error("PDX dimension must be greater than zero.");
    }
    if (num_clusters_ == 0) {
        throw std::runtime_error("PDX num_clusters must be greater than zero.");
    }
    if (max_elements_ == 0) {
        throw std::runtime_error("PDX max_elements must be greater than zero.");
    }
    if (sampling_fraction_ < 0.0f || sampling_fraction_ > 1.0f) {
        throw std::runtime_error("PDX sampling_fraction must be in [0, 1].");
    }
    rebuildPruningRatioLut();

    const PDXDimensionSplit split = getPdxDimensionSplit(dim_);
    horizontal_dims_ = split.horizontal_dims;
    vertical_dims_ = split.vertical_dims;
    horizontal_group_count_ =
        horizontal_dims_ == 0 ? 0 : (horizontal_dims_ + PDX_HORIZONTAL_BLOCK_DIMS - 1) / PDX_HORIZONTAL_BLOCK_DIMS;
    cluster_dir_entries_per_page_ = PDXClusterDirPage::entriesPerPage();
    cluster_dir_page_count_ = (num_clusters_ + cluster_dir_entries_per_page_ - 1) / cluster_dir_entries_per_page_;
    if (dim_ > PDXFloatPage::floatsPerPage()) {
        throw std::runtime_error("PDX centroid dimension exceeds the float payload of a single page.");
    }
    centroid_page_count_ = num_clusters_;
    cluster_extent_capacity_ = computeClusterExtentCapacity();
    cluster_extent_data_page_count_ =
        static_cast<u32>((static_cast<size_t>(cluster_extent_capacity_) * dim_ + PDXFloatPage::floatsPerPage() - 1) /
                         PDXFloatPage::floatsPerPage());
    cluster_extent_id_page_count_ =
        static_cast<u32>((static_cast<size_t>(cluster_extent_capacity_) + PDXIdPage::idsPerPage() - 1) /
                         PDXIdPage::idsPerPage());
    if (centroid_page_count_ == 0) {
        throw std::runtime_error("PDX requires at least one centroid page.");
    }
    if (cluster_extent_capacity_ == 0 ||
        (1 + cluster_extent_data_page_count_ + cluster_extent_id_page_count_) > PDX_CLUSTER_EXTENT_PAGE_COUNT) {
        throw std::runtime_error("PDX cluster extent geometry is invalid for the current dimension/page size.");
    }

    std::cout << "PDX initialization: dim=" << dim_ << ", vertical_dims=" << vertical_dims_
              << ", horizontal_dims=" << horizontal_dims_
              << ", horizontal_groups=" << horizontal_group_count_ << "\n";

    initializeTransform();
    resetClusterDirectoryState();

    bool has_existing_root = false;
    bool params_match = false;
    PID existing_metadata_pid = BufferManager::invalidPID;

    {
        GuardX<MetaDataPage> shared_meta(metadataPageId);
        existing_metadata_pid = shared_meta->roots[PDX_METADATA_ROOT_SLOT];
        has_existing_root = existing_metadata_pid != BufferManager::invalidPID;
    }

    if (has_existing_root) {
        GuardO<PDXMetadataPage> meta_guard(existing_metadata_pid);
        params_match = meta_guard->layout_version == PDX_LAYOUT_VERSION && meta_guard->dim == dim_ &&
                       meta_guard->num_clusters == num_clusters_ && meta_guard->retrain_interval == retrain_interval_ &&
                       meta_guard->horizontal_dims == horizontal_dims_ && meta_guard->vertical_dims == vertical_dims_ &&
                       meta_guard->horizontal_block_dims == PDX_HORIZONTAL_BLOCK_DIMS &&
                       meta_guard->cluster_dir_entries_per_page == cluster_dir_entries_per_page_ &&
                       meta_guard->cluster_dir_page_count == cluster_dir_page_count_ &&
                       meta_guard->centroid_page_count == centroid_page_count_ &&
                       meta_guard->transform_seed == transform_seed_ && meta_guard->max_elements >= max_elements_;
    }

    if (!skip_recovery && params_match) {
        metadata_pid_ = existing_metadata_pid;
        GuardO<PDXMetadataPage> meta_guard(metadata_pid_);
        centroids_base_pid_ = meta_guard->centroids_base_pid;
        cluster_dir_base_pid_ = meta_guard->cluster_dir_base_pid;
        max_elements_ = meta_guard->max_elements;
        sampling_fraction_ = meta_guard->sampling_fraction;
        pruning_aggressiveness_ = meta_guard->pruning_aggressiveness;
        rebuildPruningRatioLut();
        is_trained_.store(meta_guard->is_trained != 0, std::memory_order_release);
        loadRecoveredClusterStates();
        restoreRecoveredAllocCount();
        recovered_from_disk_ = true;
        return;
    }

    if (has_existing_root && !params_match) {
        clearRootAnchor();
    }

    initializeFreshStorage();
    persistMetadata(false);
    publishRootAnchor();
    recovered_from_disk_ = false;
}

PDXIndex::~PDXIndex() {
    search_thread_pool_.reset(nullptr);
    add_thread_pool_.reset(nullptr);
}

void PDXIndex::initializeTransform() {
    std::mt19937 rng(transform_seed_);
    std::normal_distribution<float> normal_dist;

    rotation_matrix_.resize(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    for (u32 row = 0; row < dim_; ++row) {
        for (u32 col = 0; col < dim_; ++col) {
            rotation_matrix_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = normal_dist(rng);
        }
    }

    const Eigen::HouseholderQR<Eigen::MatrixXf> qr(rotation_matrix_);
    rotation_matrix_ = qr.householderQ() * Eigen::MatrixXf::Identity(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    rotation_matrix_transposed_ = rotation_matrix_.transpose();
}

void PDXIndex::transformVector(const float* input, float* output) const {
    Eigen::Map<const Eigen::RowVectorXf> input_vec(input, static_cast<Eigen::Index>(dim_));
    Eigen::Map<Eigen::RowVectorXf> output_vec(output, static_cast<Eigen::Index>(dim_));
    output_vec.noalias() = input_vec * rotation_matrix_transposed_;
}

void PDXIndex::transformVectors(const float* input, u64 count, float* output, size_t num_threads) const {
    if (count == 0) {
        return;
    }
    const u32 resolved_threads = normalizeThreadCount(num_threads);
    if (resolved_threads == 1 || count < 32) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_matrix(
            input, static_cast<Eigen::Index>(count), static_cast<Eigen::Index>(dim_));
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_matrix(
            output, static_cast<Eigen::Index>(count), static_cast<Eigen::Index>(dim_));
        output_matrix.noalias() = input_matrix * rotation_matrix_transposed_;
        return;
    }

    ThreadPool* pool = getOrCreateAddPool(resolved_threads);
    std::vector<std::future<void>> futures;
    const u64 block_size = (count + resolved_threads - 1) / resolved_threads;
    for (u32 thread_idx = 0; thread_idx < resolved_threads; ++thread_idx) {
        const u64 begin = static_cast<u64>(thread_idx) * block_size;
        if (begin >= count) {
            break;
        }
        const u64 end = std::min<u64>(count, begin + block_size);
        futures.push_back(pool->enqueue([this, input, output, begin, end] {
            const u64 local_count = end - begin;
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_matrix(
                input + begin * dim_, static_cast<Eigen::Index>(local_count), static_cast<Eigen::Index>(dim_));
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_matrix(
                output + begin * dim_, static_cast<Eigen::Index>(local_count), static_cast<Eigen::Index>(dim_));
            output_matrix.noalias() = input_matrix * rotation_matrix_transposed_;
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
}

float PDXIndex::resolveSamplingFraction(u64 n_train) const {
    if (sampling_fraction_ <= 0.0f) {
        return 1.0f;
    }
    const double min_fraction =
        std::min<double>(1.0, static_cast<double>(PDX_MIN_EMBEDDINGS_TO_SAMPLE) / static_cast<double>(std::max<u64>(1, n_train)));
    return static_cast<float>(std::clamp(std::max<double>(sampling_fraction_, min_fraction), 0.0, 1.0));
}

std::vector<float> PDXIndex::sampleTrainingVectors(const float* transformed_vectors, u64 n_train, u64& sampled_count) const {
    const float effective_fraction = resolveSamplingFraction(n_train);
    if (effective_fraction >= 1.0f) {
        sampled_count = n_train;
        return std::vector<float>(transformed_vectors, transformed_vectors + static_cast<size_t>(n_train) * dim_);
    }

    sampled_count = std::max<u64>(num_clusters_, static_cast<u64>(std::ceil(effective_fraction * n_train)));
    sampled_count = std::min<u64>(sampled_count, n_train);

    std::vector<u64> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(transform_seed_ ^ 0x9e3779b9U);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<float> sampled(static_cast<size_t>(sampled_count) * dim_);
    for (u64 i = 0; i < sampled_count; ++i) {
        std::memcpy(sampled.data() + i * dim_, transformed_vectors + indices[i] * dim_, static_cast<size_t>(dim_) * sizeof(float));
    }
    return sampled;
}

void PDXIndex::initializeCentroidsKMeansPlusPlus(const float* vectors, u64 n, float* centroids) {
    std::mt19937 rng(transform_seed_ ^ 0x85ebca6bU);
    std::uniform_int_distribution<u64> sample_dist(0, n - 1);
    const u64 first_idx = sample_dist(rng);
    std::memcpy(centroids, vectors + first_idx * dim_, static_cast<size_t>(dim_) * sizeof(float));

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
                    static_cast<size_t>(dim_) * sizeof(float));
    }
}

void PDXIndex::kmeansStep(const float* vectors, u64 n, float* centroids, std::vector<u32>& assignments) {
    std::vector<float> new_centroids(static_cast<size_t>(num_clusters_) * dim_, 0.0f);
    std::vector<u64> cluster_counts(num_clusters_, 0);
    std::mt19937 rng(transform_seed_ ^ 0xc2b2ae35U);
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
            std::memcpy(centroid, vectors + fallback_idx * dim_, static_cast<size_t>(dim_) * sizeof(float));
            continue;
        }

        const float inv_count = 1.0f / static_cast<float>(cluster_counts[cluster_id]);
        const float* sum = new_centroids.data() + static_cast<size_t>(cluster_id) * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            centroid[d] = sum[d] * inv_count;
        }
    }
}

void PDXIndex::resetClusterDirectoryState() {
    cluster_states_.clear();
    cluster_states_.resize(num_clusters_);
    max_cluster_capacity_ = 0;
    for (ClusterState& state : cluster_states_) {
        state.extent_capacity = cluster_extent_capacity_;
    }
}

void PDXIndex::initializeFreshStorage() {
    resetClusterDirectoryState();

    AllocGuard<PDXMetadataPage> meta_guard;
    metadata_pid_ = meta_guard.pid;
    meta_guard->num_vectors.store(0, std::memory_order_relaxed);
    meta_guard->last_train_count.store(0, std::memory_order_relaxed);

    PIDAllocator centroid_allocator;
    initializeCentroidPages(&centroid_allocator, true);

    PIDAllocator dir_allocator;
    for (u32 dir_idx = 0; dir_idx < cluster_dir_page_count_; ++dir_idx) {
        AllocGuard<PDXClusterDirPage> dir_guard(&dir_allocator);
        if (dir_idx == 0) {
            cluster_dir_base_pid_ = dir_guard.pid;
        }
        const u32 first_cluster = dir_idx * cluster_dir_entries_per_page_;
        const u32 count = std::min<u32>(cluster_dir_entries_per_page_, num_clusters_ - first_cluster);
        dir_guard->entry_count = count;
        dir_guard->first_cluster_id = first_cluster;
        for (u32 i = 0; i < count; ++i) {
            dir_guard->getEntry(i)->init();
        }
        dir_guard->dirty = true;
    }

    is_trained_.store(false, std::memory_order_release);
}

void PDXIndex::publishRootAnchor() {
    GuardX<MetaDataPage> shared_meta(metadataPageId);
    shared_meta->roots[PDX_METADATA_ROOT_SLOT] = metadata_pid_;
    shared_meta->dirty = true;
}

void PDXIndex::clearRootAnchor() {
    GuardX<MetaDataPage> shared_meta(metadataPageId);
    shared_meta->roots[PDX_METADATA_ROOT_SLOT] = BufferManager::invalidPID;
    shared_meta->dirty = true;
}

void PDXIndex::persistMetadata(bool trained_flag) {
    GuardX<PDXMetadataPage> meta_guard(metadata_pid_);
    meta_guard->dim = dim_;
    meta_guard->num_clusters = num_clusters_;
    meta_guard->retrain_interval = retrain_interval_;
    meta_guard->layout_version = PDX_LAYOUT_VERSION;
    meta_guard->vertical_dims = vertical_dims_;
    meta_guard->horizontal_dims = horizontal_dims_;
    meta_guard->horizontal_block_dims = PDX_HORIZONTAL_BLOCK_DIMS;
    meta_guard->centroid_page_count = centroid_page_count_;
    meta_guard->cluster_dir_page_count = cluster_dir_page_count_;
    meta_guard->cluster_dir_entries_per_page = cluster_dir_entries_per_page_;
    meta_guard->sampling_fraction = sampling_fraction_;
    meta_guard->pruning_aggressiveness = pruning_aggressiveness_;
    meta_guard->transform_seed = transform_seed_;
    meta_guard->is_trained = trained_flag ? 1 : 0;
    meta_guard->max_elements = max_elements_;
    meta_guard->centroids_base_pid = centroids_base_pid_;
    meta_guard->cluster_dir_base_pid = cluster_dir_base_pid_;
    meta_guard->dirty = true;
}

void PDXIndex::persistClusterState(u32 cluster_id) {
    const PID dir_pid = cluster_dir_base_pid_ + (cluster_id / cluster_dir_entries_per_page_);
    const u32 local_idx = cluster_id % cluster_dir_entries_per_page_;
    GuardX<PDXClusterDirPage> dir_guard(dir_pid);
    PDXClusterEntry* entry = dir_guard->getEntry(local_idx);
    const ClusterState& state = cluster_states_[cluster_id];
    entry->num_embeddings.store(state.num_embeddings, std::memory_order_relaxed);
    entry->extent_capacity = state.extent_capacity;
    entry->extent_head_pid = state.extent_head_pid;
    entry->extent_count = state.extent_count;
    dir_guard->dirty = true;
}

void PDXIndex::loadRecoveredClusterStates() {
    resetClusterDirectoryState();
    for (u32 dir_idx = 0; dir_idx < cluster_dir_page_count_; ++dir_idx) {
        GuardO<PDXClusterDirPage> dir_guard(cluster_dir_base_pid_ + dir_idx);
        const u32 count = std::min<u32>(dir_guard->entry_count, num_clusters_ - dir_guard->first_cluster_id);
        for (u32 i = 0; i < count; ++i) {
            const u32 cluster_id = dir_guard->first_cluster_id + i;
            const PDXClusterEntry* entry = dir_guard->getEntry(i);
            ClusterState& state = cluster_states_[cluster_id];
            state.num_embeddings = entry->num_embeddings.load(std::memory_order_relaxed);
            state.extent_capacity = entry->extent_capacity == 0 ? cluster_extent_capacity_ : entry->extent_capacity;
            state.extent_head_pid = entry->extent_head_pid;
            state.extent_count = entry->extent_count;
            state.extent_base_pids.clear();
            state.row_ids.clear();

            PID current_extent_pid = state.extent_head_pid;
            u32 remaining_rows = state.num_embeddings;
            for (u32 extent_idx = 0; extent_idx < state.extent_count; ++extent_idx) {
                if (current_extent_pid == BufferManager::invalidPID) {
                    throw std::runtime_error("PDX recovery encountered a truncated extent chain.");
                }
                state.extent_base_pids.push_back(current_extent_pid);
                const u32 extent_rows = std::min<u32>(state.extent_capacity, remaining_rows);
                if (extent_rows > 0) {
                    const std::vector<u32> recovered_ids = readIdExtent(
                        clusterExtentIdBasePid(current_extent_pid), cluster_extent_id_page_count_, extent_rows);
                    state.row_ids.insert(state.row_ids.end(), recovered_ids.begin(), recovered_ids.end());
                    remaining_rows -= extent_rows;
                }
                current_extent_pid = readClusterExtentHeader(current_extent_pid).next_extent_pid;
            }
            if (remaining_rows != 0) {
                throw std::runtime_error("PDX recovery did not reconstruct all cluster rows from the extent chain.");
            }
            max_cluster_capacity_ = std::max(max_cluster_capacity_, state.num_embeddings);
        }
    }
}

void PDXIndex::restoreRecoveredAllocCount() {
    if (bm_ptr == nullptr || metadata_pid_ == BufferManager::invalidPID) {
        return;
    }

    PID highest_pid = metadata_pid_;
    if (centroids_base_pid_ != BufferManager::invalidPID && centroid_page_count_ > 0) {
        highest_pid = std::max<PID>(highest_pid, centroids_base_pid_ + static_cast<PID>(centroid_page_count_ - 1));
    }
    if (cluster_dir_base_pid_ != BufferManager::invalidPID && cluster_dir_page_count_ > 0) {
        highest_pid = std::max<PID>(highest_pid, cluster_dir_base_pid_ + static_cast<PID>(cluster_dir_page_count_ - 1));
    }

    for (const ClusterState& state : cluster_states_) {
        for (PID extent_pid : state.extent_base_pids) {
            if (extent_pid == BufferManager::invalidPID) {
                continue;
            }
            highest_pid = std::max<PID>(
                highest_pid,
                extent_pid + static_cast<PID>(PDX_CLUSTER_EXTENT_PAGE_COUNT - 1));
        }
    }

    const u64 recovered_alloc = static_cast<u64>(highest_pid) + 1;
    u64 current_alloc = bm_ptr->allocCount.load(std::memory_order_acquire);
    while (current_alloc < recovered_alloc &&
           !bm_ptr->allocCount.compare_exchange_weak(
               current_alloc, recovered_alloc, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
}

PIDAllocator* PDXIndex::clusterPidAllocator(u32 cluster_id) {
    return &cluster_pid_allocators_[cluster_id];
}

u32 PDXIndex::computeClusterExtentCapacity() const {
    const u32 payload_pages = PDX_CLUSTER_EXTENT_PAGE_COUNT - 1;
    if (payload_pages == 0) {
        return 0;
    }

    auto pages_required = [&](u32 capacity) -> u32 {
        return dataPagesForCapacity(capacity) + idPagesForCount(capacity);
    };

    if (pages_required(1) > payload_pages) {
        return 0;
    }

    u32 low = 1;
    u32 high = std::min<u64>(
        static_cast<u64>(std::numeric_limits<u32>::max()),
        static_cast<u64>(PDXFloatPage::floatsPerPage()) * payload_pages);
    u32 best = 1;

    while (low <= high) {
        const u32 mid = low + ((high - low) / 2);
        if (pages_required(mid) <= payload_pages) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return best;
}

PID PDXIndex::allocateClusterExtent(PIDAllocator* allocator) {
    reserveContiguousAllocatorPages(allocator, PDX_CLUSTER_EXTENT_PAGE_COUNT);
    PID extent_pid = BufferManager::invalidPID;
    const size_t total_extent_floats = static_cast<size_t>(cluster_extent_capacity_) * dim_;
    const size_t total_extent_ids = cluster_extent_capacity_;

    for (u32 page_idx = 0; page_idx < PDX_CLUSTER_EXTENT_PAGE_COUNT; ++page_idx) {
        const PID pid = allocator->next;
        if (page_idx == 0) {
            AllocGuard<PDXClusterExtentPage> guard(allocator);
            extent_pid = guard.pid;
            guard->next_extent_pid = BufferManager::invalidPID;
            guard->capacity = cluster_extent_capacity_;
            guard->data_page_count = cluster_extent_data_page_count_;
            guard->id_page_count = cluster_extent_id_page_count_;
            guard->dirty = true;
            continue;
        }

        const u32 data_page_begin = 1;
        const u32 data_page_end = data_page_begin + cluster_extent_data_page_count_;
        const u32 id_page_end = data_page_end + cluster_extent_id_page_count_;
        if (page_idx < data_page_end) {
            AllocGuard<PDXFloatPage> guard(allocator);
            const size_t logical_page = page_idx - data_page_begin;
            const size_t written = logical_page * PDXFloatPage::floatsPerPage();
            const size_t remaining = total_extent_floats > written ? total_extent_floats - written : 0;
            guard->float_count = static_cast<u32>(
                std::min<size_t>(PDXFloatPage::floatsPerPage(), remaining));
            if (guard->float_count < PDXFloatPage::floatsPerPage()) {
                std::memset(guard->getData(), 0, static_cast<size_t>(PDXFloatPage::floatsPerPage()) * sizeof(float));
            }
            guard->dirty = true;
        } else if (page_idx < id_page_end) {
            AllocGuard<PDXIdPage> guard(allocator);
            const size_t logical_page = page_idx - data_page_end;
            const size_t written = logical_page * PDXIdPage::idsPerPage();
            const size_t remaining = total_extent_ids > written ? total_extent_ids - written : 0;
            guard->id_count = static_cast<u32>(std::min<size_t>(PDXIdPage::idsPerPage(), remaining));
            std::fill(guard->getData(), guard->getData() + PDXIdPage::idsPerPage(), std::numeric_limits<u32>::max());
            guard->dirty = true;
        } else {
            AllocGuard<Page> guard(allocator);
            guard->dirty = false;
            (void)pid;
        }
    }

    return extent_pid;
}

PDXClusterExtentPage PDXIndex::readClusterExtentHeader(PID extent_pid) const {
    GuardORelaxed<PDXClusterExtentPage> guard(extent_pid);
    return *guard.ptr;
}

void PDXIndex::linkClusterExtent(PID extent_pid, PID next_extent_pid) {
    GuardX<PDXClusterExtentPage> guard(extent_pid);
    guard->next_extent_pid = next_extent_pid;
    guard->dirty = true;
}

size_t PDXIndex::clusterExtentRowOffset(size_t extent_index) const {
    return extent_index * static_cast<size_t>(cluster_extent_capacity_);
}

u32 PDXIndex::clusterExtentRowCount(const ClusterState& state, size_t extent_index) const {
    const size_t row_offset = clusterExtentRowOffset(extent_index);
    if (row_offset >= state.num_embeddings) {
        return 0;
    }
    return static_cast<u32>(std::min<size_t>(state.extent_capacity, state.num_embeddings - row_offset));
}

u32 PDXIndex::computeClusterCapacity(u32 required) const {
    return std::max<u32>(required, cluster_extent_capacity_);
}

void PDXIndex::initializeCentroidPages(PIDAllocator* allocator, bool zero_fill) {
    if (centroid_page_count_ == 0) {
        centroids_base_pid_ = BufferManager::invalidPID;
        return;
    }

    if (allocator != nullptr) {
        reserveContiguousAllocatorPages(allocator, centroid_page_count_);
    }

    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        if (allocator != nullptr) {
            AllocGuard<PDXFloatPage> guard(allocator);
            if (cluster_id == 0) {
                centroids_base_pid_ = guard.pid;
            }
            guard->float_count = dim_;
            if (zero_fill) {
                std::memset(guard->getData(), 0, static_cast<size_t>(PDXFloatPage::floatsPerPage()) * sizeof(float));
            }
            guard->dirty = true;
            continue;
        }

        GuardX<PDXFloatPage> guard(centroidPid(cluster_id));
        guard->float_count = dim_;
        if (zero_fill) {
            std::memset(guard->getData(), 0, static_cast<size_t>(PDXFloatPage::floatsPerPage()) * sizeof(float));
        }
        guard->dirty = true;
    }
}

void PDXIndex::writeCentroids(const float* centroids) {
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        GuardX<PDXFloatPage> guard(centroidPid(cluster_id));
        guard->float_count = dim_;
        std::memcpy(guard->getData(), centroids + static_cast<size_t>(cluster_id) * dim_,
                    static_cast<size_t>(dim_) * sizeof(float));
        if (dim_ < PDXFloatPage::floatsPerPage()) {
            std::memset(guard->getData() + dim_, 0,
                        static_cast<size_t>(PDXFloatPage::floatsPerPage() - dim_) * sizeof(float));
        }
        guard->dirty = true;
    }
}

u32 PDXIndex::dataPagesForCapacity(u32 capacity) const {
    if (capacity == 0) {
        return 0;
    }
    const size_t total_floats = static_cast<size_t>(capacity) * dim_;
    return static_cast<u32>((total_floats + PDXFloatPage::floatsPerPage() - 1) / PDXFloatPage::floatsPerPage());
}

u32 PDXIndex::idPagesForCount(u32 count) const {
    if (count == 0) {
        return 0;
    }
    return static_cast<u32>((static_cast<size_t>(count) + PDXIdPage::idsPerPage() - 1) / PDXIdPage::idsPerPage());
}

void PDXIndex::prefetchNextFloatPage(PID base_pid, u32 page_count, size_t page_idx) const {
    if (!enable_prefetch_ || readAsyncPrefetchLookaheadWindow() == 0 || page_idx + 1 >= page_count) {
        return;
    }

    PID next_pid = base_pid + static_cast<PID>(page_idx + 1);
    u32 next_offset = 0;
    bm.submitAsyncPrefetchPages(&next_pid, 1, &next_offset);
}

size_t PDXIndex::clusterVerticalRowOffset(u32 capacity, u32 dim_idx) const {
    return static_cast<size_t>(dim_idx) * capacity;
}

size_t PDXIndex::clusterHorizontalBlockOffset(u32 capacity, u32 block_idx) const {
    return static_cast<size_t>(vertical_dims_) * capacity + static_cast<size_t>(block_idx) * PDX_HORIZONTAL_BLOCK_DIMS * capacity;
}

std::pair<PID, u32> PDXIndex::allocateFloatExtent(PIDAllocator* allocator, const float* data, size_t float_count) {
    if (float_count == 0) {
        return {BufferManager::invalidPID, 0};
    }

    const u32 page_count = static_cast<u32>((float_count + PDXFloatPage::floatsPerPage() - 1) / PDXFloatPage::floatsPerPage());
    reserveContiguousAllocatorPages(allocator, page_count);
    PID base_pid = BufferManager::invalidPID;
    size_t written = 0;
    for (u32 page_idx = 0; page_idx < page_count; ++page_idx) {
        AllocGuard<PDXFloatPage> guard(allocator);
        if (page_idx == 0) {
            base_pid = guard.pid;
        }
        const u32 chunk = static_cast<u32>(std::min<size_t>(PDXFloatPage::floatsPerPage(), float_count - written));
        guard->float_count = chunk;
        if (chunk > 0) {
            std::memcpy(guard->getData(), data + written, static_cast<size_t>(chunk) * sizeof(float));
        }
        if (chunk < PDXFloatPage::floatsPerPage()) {
            std::memset(guard->getData() + chunk, 0, static_cast<size_t>(PDXFloatPage::floatsPerPage() - chunk) * sizeof(float));
        }
        guard->dirty = true;
        written += chunk;
    }
    return {base_pid, page_count};
}

std::pair<PID, u32> PDXIndex::allocateIdExtent(PIDAllocator* allocator, const u32* ids, size_t id_count) {
    if (id_count == 0) {
        return {BufferManager::invalidPID, 0};
    }

    const u32 page_count = static_cast<u32>((id_count + PDXIdPage::idsPerPage() - 1) / PDXIdPage::idsPerPage());
    reserveContiguousAllocatorPages(allocator, page_count);
    PID base_pid = BufferManager::invalidPID;
    size_t written = 0;
    for (u32 page_idx = 0; page_idx < page_count; ++page_idx) {
        AllocGuard<PDXIdPage> guard(allocator);
        if (page_idx == 0) {
            base_pid = guard.pid;
        }
        const u32 chunk = static_cast<u32>(std::min<size_t>(PDXIdPage::idsPerPage(), id_count - written));
        guard->id_count = chunk;
        if (chunk > 0) {
            std::memcpy(guard->getData(), ids + written, static_cast<size_t>(chunk) * sizeof(u32));
        }
        if (chunk < PDXIdPage::idsPerPage()) {
            std::fill(guard->getData() + chunk, guard->getData() + PDXIdPage::idsPerPage(), std::numeric_limits<u32>::max());
        }
        guard->dirty = true;
        written += chunk;
    }
    return {base_pid, page_count};
}

void PDXIndex::writeFloatExtent(PID base_pid, u32 page_count, const float* data, size_t float_count) {
    size_t written = 0;
    for (u32 page_idx = 0; page_idx < page_count; ++page_idx) {
        GuardX<PDXFloatPage> guard(base_pid + page_idx);
        const u32 chunk = static_cast<u32>(std::min<size_t>(PDXFloatPage::floatsPerPage(), float_count - written));
        guard->float_count = chunk;
        if (chunk > 0) {
            std::memcpy(guard->getData(), data + written, static_cast<size_t>(chunk) * sizeof(float));
        }
        if (chunk < PDXFloatPage::floatsPerPage()) {
            std::memset(guard->getData() + chunk, 0, static_cast<size_t>(PDXFloatPage::floatsPerPage() - chunk) * sizeof(float));
        }
        guard->dirty = true;
        written += chunk;
    }
}

void PDXIndex::writeIdExtent(PID base_pid, u32 page_count, const u32* ids, size_t id_count) {
    size_t written = 0;
    for (u32 page_idx = 0; page_idx < page_count; ++page_idx) {
        GuardX<PDXIdPage> guard(base_pid + page_idx);
        const u32 chunk = static_cast<u32>(std::min<size_t>(PDXIdPage::idsPerPage(), id_count - written));
        guard->id_count = chunk;
        if (chunk > 0) {
            std::memcpy(guard->getData(), ids + written, static_cast<size_t>(chunk) * sizeof(u32));
        }
        if (chunk < PDXIdPage::idsPerPage()) {
            std::fill(guard->getData() + chunk, guard->getData() + PDXIdPage::idsPerPage(), std::numeric_limits<u32>::max());
        }
        guard->dirty = true;
        written += chunk;
    }
}

void PDXIndex::writeFloatRange(PID base_pid, u32 page_count, size_t logical_offset, const float* data,
                               size_t float_count) {
    size_t remaining = float_count;
    size_t offset = logical_offset;
    size_t written = 0;
    while (remaining > 0) {
        const size_t page_idx = offset / PDXFloatPage::floatsPerPage();
        if (page_idx >= page_count) {
            std::ostringstream error;
            error << "PDX float write exceeded extent: base_pid=" << base_pid << " page_idx=" << page_idx
                  << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " remaining=" << remaining;
            throw std::runtime_error(error.str());
        }
        const size_t in_page = offset % PDXFloatPage::floatsPerPage();
        GuardX<PDXFloatPage> guard(base_pid + static_cast<PID>(page_idx));
        const size_t page_float_count = resolvedFloatPageCount(*guard.ptr, page_idx + 1 == page_count);
        if (in_page >= page_float_count) {
            std::ostringstream error;
            error << "PDX float write encountered an invalid offset: base_pid=" << base_pid << " page_idx=" << page_idx
                  << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " in_page=" << in_page << " page_float_count=" << page_float_count;
            throw std::runtime_error(error.str());
        }
        const size_t available = std::min<size_t>(remaining, page_float_count - in_page);
        std::memcpy(guard->getData() + in_page, data + written, available * sizeof(float));
        guard->dirty = true;
        remaining -= available;
        written += available;
        offset += available;
    }
}

void PDXIndex::writeIdRange(PID base_pid, u32 page_count, size_t logical_offset, const u32* ids, size_t id_count) {
    size_t remaining = id_count;
    size_t offset = logical_offset;
    size_t written = 0;
    while (remaining > 0) {
        const size_t page_idx = offset / PDXIdPage::idsPerPage();
        if (page_idx >= page_count) {
            std::ostringstream error;
            error << "PDX id write exceeded extent: base_pid=" << base_pid << " page_idx=" << page_idx
                  << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " remaining=" << remaining;
            throw std::runtime_error(error.str());
        }
        const size_t in_page = offset % PDXIdPage::idsPerPage();
        GuardX<PDXIdPage> guard(base_pid + static_cast<PID>(page_idx));
        const size_t page_id_count = resolvedIdPageCount(*guard.ptr, page_idx + 1 == page_count);
        if (in_page >= page_id_count) {
            std::ostringstream error;
            error << "PDX id write encountered an invalid offset: base_pid=" << base_pid << " page_idx=" << page_idx
                  << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " in_page=" << in_page << " page_id_count=" << page_id_count;
            throw std::runtime_error(error.str());
        }
        const size_t available = std::min<size_t>(remaining, page_id_count - in_page);
        std::memcpy(guard->getData() + in_page, ids + written, available * sizeof(u32));
        guard->dirty = true;
        remaining -= available;
        written += available;
        offset += available;
    }
}

std::vector<float> PDXIndex::readFloatExtent(PID base_pid, u32 page_count, size_t float_count) const {
    std::vector<float> result(float_count, 0.0f);
    size_t read = 0;
    for (u32 page_idx = 0; page_idx < page_count && read < float_count; ++page_idx) {
        GuardORelaxed<PDXFloatPage> guard(base_pid + page_idx);
        const size_t page_float_count = resolvedFloatPageCount(*guard.ptr, page_idx + 1 == page_count);
        const u32 chunk = static_cast<u32>(std::min<size_t>(page_float_count, float_count - read));
        if (chunk > 0) {
            std::memcpy(result.data() + read, guard->getData(), static_cast<size_t>(chunk) * sizeof(float));
        }
        read += chunk;
    }
    return result;
}

std::vector<u32> PDXIndex::readIdExtent(PID base_pid, u32 page_count, size_t id_count) const {
    std::vector<u32> result(id_count, std::numeric_limits<u32>::max());
    size_t read = 0;
    for (u32 page_idx = 0; page_idx < page_count && read < id_count; ++page_idx) {
        GuardO<PDXIdPage> guard(base_pid + page_idx);
        const u32 chunk = static_cast<u32>(
            std::min<size_t>(resolvedIdPageCount(*guard.ptr, page_idx + 1 == page_count), id_count - read));
        if (chunk > 0) {
            std::memcpy(result.data() + read, guard->getData(), static_cast<size_t>(chunk) * sizeof(u32));
        }
        read += chunk;
    }
    return result;
}

template <typename Fn>
void PDXIndex::forEachFloatRange(PID base_pid, u32 page_count, size_t logical_offset, size_t float_count, Fn&& fn) const {
    size_t remaining = float_count;
    size_t offset = logical_offset;
    while (remaining > 0) {
        const size_t page_idx = offset / PDXFloatPage::floatsPerPage();
        if (page_idx >= page_count) {
            std::ostringstream error;
            error << "PDX float range exceeded extent: base_pid=" << base_pid << " page_idx=" << page_idx
                  << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " remaining=" << remaining;
            throw std::runtime_error(error.str());
        }
        const size_t in_page = offset % PDXFloatPage::floatsPerPage();
        GuardORelaxed<PDXFloatPage> guard(base_pid + static_cast<PID>(page_idx));
        //GuardO<PDXFloatPage> guard(base_pid + static_cast<PID>(page_idx));
        const size_t page_float_count = resolvedFloatPageCount(*guard.ptr, page_idx + 1 == page_count);
        if (in_page >= page_float_count) {
            std::ostringstream error;
            error << "PDX encountered a non-progress float page read: base_pid=" << base_pid
                  << " page_idx=" << page_idx << " page_count=" << page_count << " logical_offset=" << logical_offset << " offset=" << offset
                  << " in_page=" << in_page << " page_float_count=" << page_float_count
                  << " remaining=" << remaining;
            throw std::runtime_error(error.str());
        }
        const size_t available = std::min<size_t>(remaining, page_float_count - in_page);
        if (remaining > available) {
            prefetchNextFloatPage(base_pid, page_count, page_idx);
        }
        fn(guard->getData() + in_page, available);
        remaining -= available;
        offset += available;
    }
}

void PDXIndex::readFloatBlock(PID base_pid, u32 page_count, size_t logical_offset, float* output, size_t float_count) const {
    size_t copied = 0;
    forEachFloatRange(base_pid, page_count, logical_offset, float_count, [&](const float* src, size_t count) {
        std::memcpy(output + copied, src, count * sizeof(float));
        copied += count;
    });
}

float PDXIndex::readSingleFloat(PID base_pid, u32 page_count, size_t logical_offset) const {
    float value = 0.0f;
    forEachFloatRange(base_pid, page_count, logical_offset, 1, [&](const float* src, size_t count) {
        if (count > 0) {
            value = src[0];
        }
    });
    return value;
}

void PDXIndex::encodeClusterLayout(const float* row_major_vectors, u32 num_embeddings, u32 capacity,
                                   std::vector<float>& encoded) const {
    encoded.assign(static_cast<size_t>(capacity) * dim_, 0.0f);

    for (u32 dim_idx = 0; dim_idx < vertical_dims_; ++dim_idx) {
        const size_t base = clusterVerticalRowOffset(capacity, dim_idx);
        for (u32 row = 0; row < num_embeddings; ++row) {
            encoded[base + row] = row_major_vectors[static_cast<size_t>(row) * dim_ + dim_idx];
        }
    }

    for (u32 block_idx = 0; block_idx < horizontal_group_count_; ++block_idx) {
        const u32 block_offset = block_idx * PDX_HORIZONTAL_BLOCK_DIMS;
        const u32 block_width = std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_offset);
        const size_t base = clusterHorizontalBlockOffset(capacity, block_idx);
        const u32 source_dim = vertical_dims_ + block_offset;
        for (u32 row = 0; row < num_embeddings; ++row) {
            std::memcpy(encoded.data() + base + static_cast<size_t>(row) * PDX_HORIZONTAL_BLOCK_DIMS,
                        row_major_vectors + static_cast<size_t>(row) * dim_ + source_dim,
                        static_cast<size_t>(block_width) * sizeof(float));
        }
    }
}

void PDXIndex::decodeClusterLayout(const std::vector<float>& encoded, u32 num_embeddings, u32 capacity,
                                   std::vector<float>& row_major_vectors) const {
    row_major_vectors.assign(static_cast<size_t>(num_embeddings) * dim_, 0.0f);

    for (u32 dim_idx = 0; dim_idx < vertical_dims_; ++dim_idx) {
        const size_t base = clusterVerticalRowOffset(capacity, dim_idx);
        for (u32 row = 0; row < num_embeddings; ++row) {
            row_major_vectors[static_cast<size_t>(row) * dim_ + dim_idx] = encoded[base + row];
        }
    }

    for (u32 block_idx = 0; block_idx < horizontal_group_count_; ++block_idx) {
        const u32 block_offset = block_idx * PDX_HORIZONTAL_BLOCK_DIMS;
        const u32 block_width = std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_offset);
        const size_t base = clusterHorizontalBlockOffset(capacity, block_idx);
        const u32 dest_dim = vertical_dims_ + block_offset;
        for (u32 row = 0; row < num_embeddings; ++row) {
            std::memcpy(row_major_vectors.data() + static_cast<size_t>(row) * dim_ + dest_dim,
                        encoded.data() + base + static_cast<size_t>(row) * PDX_HORIZONTAL_BLOCK_DIMS,
                        static_cast<size_t>(block_width) * sizeof(float));
        }
    }
}

void PDXIndex::writeClusterExtentRows(PID extent_pid, u32 start_row, const float* appended_vectors,
                                      const u32* appended_ids, u32 append_count) {
    if (append_count == 0) {
        return;
    }

    const PID data_base_pid = clusterExtentDataBasePid(extent_pid);
    const PID ids_base_pid = clusterExtentIdBasePid(extent_pid);

    std::vector<float> vertical_buffer(append_count);
    for (u32 dim_idx = 0; dim_idx < vertical_dims_; ++dim_idx) {
        for (u32 row = 0; row < append_count; ++row) {
            vertical_buffer[row] = appended_vectors[static_cast<size_t>(row) * dim_ + dim_idx];
        }
        writeFloatRange(data_base_pid, cluster_extent_data_page_count_,
                        clusterVerticalRowOffset(cluster_extent_capacity_, dim_idx) + start_row,
                        vertical_buffer.data(), append_count);
    }

    std::vector<float> horizontal_buffer;
    for (u32 block_idx = 0; block_idx < horizontal_group_count_; ++block_idx) {
        const u32 block_offset = block_idx * PDX_HORIZONTAL_BLOCK_DIMS;
        const u32 block_width = std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_offset);
        const u32 source_dim = vertical_dims_ + block_offset;
        horizontal_buffer.assign(static_cast<size_t>(append_count) * PDX_HORIZONTAL_BLOCK_DIMS, 0.0f);
        for (u32 row = 0; row < append_count; ++row) {
            std::memcpy(horizontal_buffer.data() + static_cast<size_t>(row) * PDX_HORIZONTAL_BLOCK_DIMS,
                        appended_vectors + static_cast<size_t>(row) * dim_ + source_dim,
                        static_cast<size_t>(block_width) * sizeof(float));
        }
        writeFloatRange(data_base_pid, cluster_extent_data_page_count_,
                        clusterHorizontalBlockOffset(cluster_extent_capacity_, block_idx) +
                            static_cast<size_t>(start_row) * PDX_HORIZONTAL_BLOCK_DIMS,
                        horizontal_buffer.data(), horizontal_buffer.size());
    }

    writeIdRange(ids_base_pid, cluster_extent_id_page_count_, start_row, appended_ids, append_count);
}

void PDXIndex::appendClusterStorage(u32 cluster_id, ClusterState& state, const float* appended_vectors,
                                    const u32* appended_ids, u32 append_count) {
    if (append_count == 0) {
        return;
    }

    u32 remaining = append_count;
    u32 source_offset = 0;
    const u32 tail_used = state.num_embeddings % state.extent_capacity;
    if (!state.extent_base_pids.empty() && tail_used != 0) {
        const u32 tail_free = state.extent_capacity - tail_used;
        const u32 chunk = std::min<u32>(remaining, tail_free);
        writeClusterExtentRows(state.extent_base_pids.back(), tail_used,
                               appended_vectors + static_cast<size_t>(source_offset) * dim_, appended_ids + source_offset,
                               chunk);
        source_offset += chunk;
        remaining -= chunk;
    }

    while (remaining > 0) {
        const PID new_extent_pid = allocateClusterExtent(clusterPidAllocator(cluster_id));
        if (!state.extent_base_pids.empty()) {
            linkClusterExtent(state.extent_base_pids.back(), new_extent_pid);
        } else {
            state.extent_head_pid = new_extent_pid;
        }
        state.extent_base_pids.push_back(new_extent_pid);
        state.extent_count = static_cast<u32>(state.extent_base_pids.size());

        const u32 chunk = std::min<u32>(remaining, state.extent_capacity);
        writeClusterExtentRows(new_extent_pid, 0,
                               appended_vectors + static_cast<size_t>(source_offset) * dim_, appended_ids + source_offset,
                               chunk);
        source_offset += chunk;
        remaining -= chunk;
    }
}

float PDXIndex::horizontalBlockDistance(PID base_pid, u32 page_count, size_t logical_offset, const float* query_block,
                                        u32 block_width) const {
    float total = 0.0f;
    size_t consumed = 0;
    forEachFloatRange(base_pid, page_count, logical_offset, block_width, [&](const float* src, size_t count) {
        total += pdxHorizontalDistance(query_block + consumed, src, count);
        consumed += count;
    });
    return total;
}

void PDXIndex::rebuildClusterStorage(u32 cluster_id, const float* appended_vectors, const u32* appended_ids,
                                     u32 append_count) {
    ClusterState& state = cluster_states_[cluster_id];
    const u32 existing_count = state.num_embeddings;
    const u32 new_count = existing_count + append_count;
    if (new_count == 0) {
        return;
    }

    state.extent_capacity = cluster_extent_capacity_;
    appendClusterStorage(cluster_id, state, appended_vectors, appended_ids, append_count);

    state.num_embeddings = new_count;
    state.row_ids.insert(state.row_ids.end(), appended_ids, appended_ids + append_count);
    max_cluster_capacity_ = std::max(max_cluster_capacity_, state.num_embeddings);
    persistClusterState(cluster_id);
}

u32 PDXIndex::findNearestCentroid(const float* transformed_vector) const {
    float best_distance = std::numeric_limits<float>::max();
    u32 best_cluster = 0;
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const float distance = computeCentroidDistanceFromPages(transformed_vector, cluster_id);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster_id;
        }
    }
    return best_cluster;
}

std::vector<std::pair<float, u32>> PDXIndex::findNearestCentroids(const float* transformed_vector, size_t nprobe) const {
    const size_t probe_limit = std::min<size_t>(std::max<size_t>(1, nprobe), num_clusters_);
    std::priority_queue<std::pair<float, u32>> heap;
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const float distance = computeCentroidDistanceFromPages(transformed_vector, cluster_id);
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

float PDXIndex::computeCentroidDistanceFromPages(const float* transformed_vector, u32 cluster_id) const {
    GuardORelaxed<PDXFloatPage> guard(centroidPid(cluster_id));
    return pdxHorizontalDistance(transformed_vector, guard->getData(), dim_);
}

void PDXIndex::rebuildPruningRatioLut() {
    pruning_ratio_lut_.assign(static_cast<size_t>(dim_) + 1, 1.0f);
    for (u32 visited_dimensions = 1; visited_dimensions < dim_; ++visited_dimensions) {
        const double fraction = static_cast<double>(visited_dimensions) / static_cast<double>(dim_);
        const double boost =
            1.0 + static_cast<double>(pruning_aggressiveness_) / std::sqrt(static_cast<double>(visited_dimensions));
        pruning_ratio_lut_[visited_dimensions] = static_cast<float>(fraction * boost * boost);
    }
}

float PDXIndex::pruningRatio(u32 visited_dimensions) const {
    if (visited_dimensions >= pruning_ratio_lut_.size()) {
        return 1.0f;
    }
    return pruning_ratio_lut_[visited_dimensions];
}

float PDXIndex::pruningThreshold(float heap_threshold, u32 visited_dimensions) const {
    if (!std::isfinite(heap_threshold)) {
        return heap_threshold;
    }
    return heap_threshold * pruningRatio(visited_dimensions);
}

size_t PDXIndex::filterCandidatePositions(const std::vector<float>& distances, float threshold, u32* positions,
                                          size_t count) const {
    if (!std::isfinite(threshold)) {
        return count;
    }
    size_t kept = 0;
    for (size_t i = 0; i < count; ++i) {
        const u32 pos = positions[i];
        positions[kept] = pos;
        kept += distances[pos] <= threshold;
    }
    return kept;
}

size_t PDXIndex::countPrunedCandidates(const std::vector<float>& distances, float threshold, u32 num_embeddings) const {
    if (!std::isfinite(threshold)) {
        return 0;
    }
    size_t pruned = 0;
    for (u32 row = 0; row < num_embeddings; ++row) {
        if (distances[row] > threshold) {
            ++pruned;
        }
    }
    return pruned;
}

void PDXIndex::scanVerticalAll(const ClusterState& cluster, const float* transformed_query, u32 begin_dim, u32 end_dim,
                               std::vector<float>& distances, u64& dist_ops) const {
    constexpr size_t kPrefetchChunkPages = 8;
    constexpr size_t kPrefetchChunkFloats =
        kPrefetchChunkPages * static_cast<size_t>(PDXFloatPage::floatsPerPage());
    for (u32 dim_idx = begin_dim; dim_idx < end_dim; ++dim_idx) {
        const float q = transformed_query[dim_idx];
        for (size_t extent_idx = 0; extent_idx < cluster.extent_base_pids.size(); ++extent_idx) {
            const PID extent_pid = cluster.extent_base_pids[extent_idx];
            const PID data_base_pid = clusterExtentDataBasePid(extent_pid);
            const size_t global_row = clusterExtentRowOffset(extent_idx);
            size_t logical_offset = clusterVerticalRowOffset(cluster.extent_capacity, dim_idx);
            const size_t base_page_idx = logical_offset / PDXFloatPage::floatsPerPage();
            size_t remaining = clusterExtentRowCount(cluster, extent_idx);
            size_t local_row = 0;
            size_t consumed_chunk_boundary = 0;
            size_t next_prefetch_page_idx = kPrefetchChunkPages;
            while (remaining > 0) {
                const size_t page_idx = logical_offset / PDXFloatPage::floatsPerPage();
                const size_t in_page = logical_offset % PDXFloatPage::floatsPerPage();

                if (enable_prefetch_ && kPrefetchChunkFloats > 0) {
                    const size_t relative_page_idx = page_idx - base_page_idx;
                    while (relative_page_idx >=
                           (consumed_chunk_boundary / static_cast<size_t>(PDXFloatPage::floatsPerPage())) +
                               kPrefetchChunkPages) {
                        consumed_chunk_boundary += kPrefetchChunkFloats;

                        const size_t absolute_prefetch_page_idx = base_page_idx + next_prefetch_page_idx;
                        if (absolute_prefetch_page_idx >= cluster_extent_data_page_count_) {
                            break;
                        }

                        std::array<PID, kPrefetchChunkPages> prefetched_pids{};
                        std::array<u32, kPrefetchChunkPages> prefetched_offsets{};
                        u32 prefetch_count = 0;
                        const size_t prefetch_end = std::min<size_t>(absolute_prefetch_page_idx + kPrefetchChunkPages,
                                                                     cluster_extent_data_page_count_);
                        for (size_t pf_page_idx = absolute_prefetch_page_idx; pf_page_idx < prefetch_end;
                             ++pf_page_idx) {
                            prefetched_pids[prefetch_count] = data_base_pid + static_cast<PID>(pf_page_idx);
                            prefetched_offsets[prefetch_count] = 0;
                            ++prefetch_count;
                        }
                        if (prefetch_count > 0) {
                            bm.submitAsyncPrefetchPages(prefetched_pids.data(), static_cast<int>(prefetch_count),
                                                        prefetched_offsets.data());
                        }
                        next_prefetch_page_idx += kPrefetchChunkPages;
                    }
                }

                GuardORelaxed<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
                //GuardO<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
                const size_t page_float_count =
                    resolvedFloatPageCount(*guard.ptr, page_idx + 1 == cluster_extent_data_page_count_);
                if (in_page >= page_float_count) {
                    std::ostringstream error;
                    error << "PDX encountered a non-progress vertical-all page read: extent_pid=" << extent_pid
                          << " page_idx=" << page_idx << " dim_idx=" << dim_idx
                          << " logical_offset=" << logical_offset << " in_page=" << in_page
                          << " page_float_count=" << page_float_count << " remaining=" << remaining
                          << " num_embeddings=" << cluster.num_embeddings << " extent_capacity=" << cluster.extent_capacity;
                    throw std::runtime_error(error.str());
                }
                const size_t count = std::min<size_t>(remaining, page_float_count - in_page);
                const float* __restrict src_ptr = guard->getData() + in_page;
                float* __restrict dist_ptr = distances.data() + global_row + local_row;
                accumulateVerticalDistances(q, src_ptr, dist_ptr, count);
                local_row += count;
                remaining -= count;
                logical_offset += count;
            }
        }
        dist_ops += cluster.num_embeddings;
    }
}

void PDXIndex::scanVerticalSelected(const ClusterState& cluster, const float* transformed_query, u32 begin_dim,
                                    u32 end_dim, const u32* positions, size_t position_count,
                                    std::vector<float>& distances, u64& dist_ops) const {
    for (u32 dim_idx = begin_dim; dim_idx < end_dim; ++dim_idx) {
        const float q = transformed_query[dim_idx];
        size_t position_idx = 0;
        for (size_t extent_idx = 0; extent_idx < cluster.extent_base_pids.size() && position_idx < position_count;
             ++extent_idx) {
            const PID extent_pid = cluster.extent_base_pids[extent_idx];
            const PID data_base_pid = clusterExtentDataBasePid(extent_pid);
            const size_t row_base = clusterExtentRowOffset(extent_idx);
            const size_t extent_rows = clusterExtentRowCount(cluster, extent_idx);
            const size_t row_limit = row_base + extent_rows;
            const size_t base = clusterVerticalRowOffset(cluster.extent_capacity, dim_idx);
            size_t row_begin = 0;
            size_t logical_offset = base;
            while (row_begin < extent_rows && position_idx < position_count) {
                if (positions[position_idx] >= row_limit) {
                    break;
                }
                const size_t page_idx = logical_offset / PDXFloatPage::floatsPerPage();
                const size_t in_page = logical_offset % PDXFloatPage::floatsPerPage();
                GuardORelaxed<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
                //GuardO<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
                const size_t page_float_count =
                    resolvedFloatPageCount(*guard.ptr, page_idx + 1 == cluster_extent_data_page_count_);
                if (in_page >= page_float_count) {
                    std::ostringstream error;
                    error << "PDX encountered a non-progress vertical page read: extent_pid=" << extent_pid
                          << " page_idx=" << page_idx << " dim_idx=" << dim_idx
                          << " logical_offset=" << logical_offset << " in_page=" << in_page
                          << " page_float_count=" << page_float_count << " row_begin=" << row_begin
                          << " num_embeddings=" << cluster.num_embeddings << " extent_capacity=" << cluster.extent_capacity;
                    throw std::runtime_error(error.str());
                }
                const size_t available = std::min<size_t>(extent_rows - row_begin, page_float_count - in_page);
                if (row_begin + available < extent_rows) {
                    prefetchNextFloatPage(data_base_pid, cluster_extent_data_page_count_, page_idx);
                }
                const float* src = guard->getData() + in_page;
                const size_t row_end = row_begin + available;

                accumulateVerticalSelectedDistances(q, src, row_begin, row_end, row_base, positions, position_count,
                                                   position_idx, distances);

                row_begin = row_end;
                logical_offset += available;
            }
        }
        dist_ops += position_count;
    }
}

void PDXIndex::scanHorizontalAllBlock(const ClusterState& cluster, const float* transformed_query, u32 block_idx,
                                      std::vector<float>& distances, u64& dist_ops) const {
    const u32 block_offset = block_idx * PDX_HORIZONTAL_BLOCK_DIMS;
    const u32 block_width = std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_offset);
    const float* query_block = transformed_query + vertical_dims_ + block_offset;

    for (size_t extent_idx = 0; extent_idx < cluster.extent_base_pids.size(); ++extent_idx) {
        const PID extent_pid = cluster.extent_base_pids[extent_idx];
        const PID data_base_pid = clusterExtentDataBasePid(extent_pid);
        const size_t row_base = clusterExtentRowOffset(extent_idx);
        const size_t extent_rows = clusterExtentRowCount(cluster, extent_idx);
        const size_t base = clusterHorizontalBlockOffset(cluster.extent_capacity, block_idx);

        std::array<float, PDX_HORIZONTAL_BLOCK_DIMS> carry{};
        size_t carry_count = 0;
        size_t row = 0;
        const size_t total_floats = extent_rows * PDX_HORIZONTAL_BLOCK_DIMS;

        size_t logical_offset = base;
        size_t remaining = total_floats;
        while (remaining > 0) {
            const size_t page_idx = logical_offset / PDXFloatPage::floatsPerPage();
            const size_t in_page = logical_offset % PDXFloatPage::floatsPerPage();
            GuardORelaxed<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
            //GuardO<PDXFloatPage> guard(data_base_pid + static_cast<PID>(page_idx));
            const size_t page_float_count =
                resolvedFloatPageCount(*guard.ptr, page_idx + 1 == cluster_extent_data_page_count_);
            if (in_page >= page_float_count) {
                std::ostringstream error;
                error << "PDX encountered a non-progress horizontal-all page read: extent_pid=" << extent_pid
                      << " page_idx=" << page_idx << " block_idx=" << block_idx
                      << " logical_offset=" << logical_offset << " in_page=" << in_page
                      << " page_float_count=" << page_float_count << " remaining=" << remaining
                      << " num_embeddings=" << cluster.num_embeddings << " extent_capacity=" << cluster.extent_capacity;
                throw std::runtime_error(error.str());
            }
            const size_t count = std::min<size_t>(remaining, page_float_count - in_page);
            if (remaining > count) {
                prefetchNextFloatPage(data_base_pid, cluster_extent_data_page_count_, page_idx);
            }
            const float* src = guard->getData() + in_page;
            size_t offset = 0;

            if (carry_count > 0) {
                const size_t taken = std::min<size_t>(PDX_HORIZONTAL_BLOCK_DIMS - carry_count, count - offset);
                std::memcpy(carry.data() + carry_count, src, taken * sizeof(float));
                carry_count += taken;
                offset += taken;
                if (carry_count == PDX_HORIZONTAL_BLOCK_DIMS) {
                    distances[row_base + row++] += pdxHorizontalDistance(query_block, carry.data(), block_width);
                    carry_count = 0;
                }
            }

            while (offset + PDX_HORIZONTAL_BLOCK_DIMS <= count) {
                distances[row_base + row++] += pdxHorizontalDistance(query_block, src + offset, block_width);
                offset += PDX_HORIZONTAL_BLOCK_DIMS;
            }

            if (offset < count) {
                carry_count = count - offset;
                std::memcpy(carry.data(), src + offset, carry_count * sizeof(float));
            }
            remaining -= count;
            logical_offset += count;
        }
    }

    dist_ops += static_cast<u64>(cluster.num_embeddings) * block_width;
}

void PDXIndex::scanHorizontalSelectedBlock(const ClusterState& cluster, const float* transformed_query, u32 block_idx,
                                           const u32* positions, size_t position_count,
                                           std::vector<float>& distances, u64& dist_ops) const {
    constexpr size_t kPrefetchChunkPages = 8;
    constexpr size_t kPrefetchChunkFloats =
        kPrefetchChunkPages * static_cast<size_t>(PDXFloatPage::floatsPerPage());
    const u32 block_offset = block_idx * PDX_HORIZONTAL_BLOCK_DIMS;
    const u32 block_width = std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_offset);
    const float* query_block = transformed_query + vertical_dims_ + block_offset;
    size_t pos_idx = 0;
    for (size_t extent_idx = 0; extent_idx < cluster.extent_base_pids.size() && pos_idx < position_count; ++extent_idx) {
        const PID extent_pid = cluster.extent_base_pids[extent_idx];
        const PID data_base_pid = clusterExtentDataBasePid(extent_pid);
        const size_t row_base = clusterExtentRowOffset(extent_idx);
        const size_t extent_rows = clusterExtentRowCount(cluster, extent_idx);
        const size_t row_limit = row_base + extent_rows;
        const size_t base = clusterHorizontalBlockOffset(cluster.extent_capacity, block_idx);
        size_t consumed_chunk_boundary = 0;
        size_t next_prefetch_page_idx = kPrefetchChunkPages;

        while (pos_idx < position_count && positions[pos_idx] < row_limit) {
            const u32 row = positions[pos_idx];
            const size_t local_row = row - row_base;
            const size_t logical_offset = base + local_row * PDX_HORIZONTAL_BLOCK_DIMS;
            const size_t page_idx = logical_offset / PDXFloatPage::floatsPerPage();
            const size_t in_page = logical_offset % PDXFloatPage::floatsPerPage();

            if (enable_prefetch_ && kPrefetchChunkFloats > 0) {
                while (page_idx >= (consumed_chunk_boundary / static_cast<size_t>(PDXFloatPage::floatsPerPage())) +
                                      kPrefetchChunkPages) {
                    consumed_chunk_boundary += kPrefetchChunkFloats;

                    if (next_prefetch_page_idx >= cluster_extent_data_page_count_) {
                        break;
                    }

                    std::array<PID, kPrefetchChunkPages> prefetched_pids{};
                    std::array<u32, kPrefetchChunkPages> prefetched_offsets{};
                    u32 prefetch_count = 0;
                    const size_t prefetch_end =
                        std::min<size_t>(next_prefetch_page_idx + kPrefetchChunkPages, cluster_extent_data_page_count_);
                    for (size_t pf_page_idx = next_prefetch_page_idx; pf_page_idx < prefetch_end; ++pf_page_idx) {
                        prefetched_pids[prefetch_count] = data_base_pid + static_cast<PID>(pf_page_idx);
                        prefetched_offsets[prefetch_count] = 0;
                        ++prefetch_count;
                    }
                    if (prefetch_count > 0) {
                        bm.submitAsyncPrefetchPages(prefetched_pids.data(), static_cast<int>(prefetch_count),
                                                    prefetched_offsets.data());
                    }
                    next_prefetch_page_idx += kPrefetchChunkPages;
                }
            }

            const PID logical_pid = data_base_pid + static_cast<PID>(page_idx);
            GuardORelaxed<PDXFloatPage> guard(logical_pid);
            const size_t page_float_count =
                resolvedFloatPageCount(*guard.ptr, page_idx + 1 == cluster_extent_data_page_count_);
            const float* page_data = guard->getData();
            if (in_page >= page_float_count) {
                std::ostringstream error;
                error << "PDX encountered an invalid horizontal selected read: extent_pid=" << extent_pid
                      << " page_idx=" << page_idx << " block_idx=" << block_idx
                      << " logical_offset=" << logical_offset << " in_page=" << in_page
                      << " page_float_count=" << page_float_count << " num_embeddings=" << cluster.num_embeddings
                      << " extent_capacity=" << cluster.extent_capacity;
                throw std::runtime_error(error.str());
            }
            size_t page_run_idx = pos_idx;
            while (page_run_idx < position_count) {
                const u32 page_row = positions[page_run_idx];
                if (page_row >= row_limit) {
                    break;
                }
                const size_t page_local_row = page_row - row_base;
                const size_t page_logical_offset = base + page_local_row * PDX_HORIZONTAL_BLOCK_DIMS;
                const size_t page_run_page_idx = page_logical_offset / PDXFloatPage::floatsPerPage();
                if (page_run_page_idx != page_idx) {
                    break;
                }
                const size_t page_run_in_page = page_logical_offset % PDXFloatPage::floatsPerPage();
                if (page_run_in_page >= page_float_count) {
                    std::ostringstream error;
                    error << "PDX encountered an invalid grouped horizontal selected read: extent_pid=" << extent_pid
                          << " page_idx=" << page_idx << " block_idx=" << block_idx
                          << " logical_offset=" << page_logical_offset << " in_page=" << page_run_in_page
                          << " page_float_count=" << page_float_count << " num_embeddings=" << cluster.num_embeddings
                          << " extent_capacity=" << cluster.extent_capacity;
                    throw std::runtime_error(error.str());
                }

                ++page_run_idx;
            }

            // if (false && enable_prefetch_ && readAsyncPrefetchLookaheadWindow() != 0 && page_run_idx < position_count) {
            //     const u32 next_row = positions[page_run_idx];
            //     const size_t next_extent_idx = next_row / cluster.extent_capacity;
            //     if (next_extent_idx < cluster.extent_base_pids.size()) {
            //         const PID next_data_base_pid = clusterExtentDataBasePid(cluster.extent_base_pids[next_extent_idx]);
            //         const size_t next_row_base = clusterExtentRowOffset(next_extent_idx);
            //         const size_t next_local_row = next_row - next_row_base;
            //         const size_t next_logical_offset =
            //             clusterHorizontalBlockOffset(cluster.extent_capacity, block_idx) +
            //             next_local_row * PDX_HORIZONTAL_BLOCK_DIMS;
            //         const PID next_pid =
            //             next_data_base_pid + static_cast<PID>(next_logical_offset / PDXFloatPage::floatsPerPage());
            //         if (next_pid != logical_pid) {
            //             u32 next_offset = static_cast<u32>(PDXFloatPage::HeaderSize +
            //                                                (next_logical_offset % PDXFloatPage::floatsPerPage()) *
            //                                                    sizeof(float));
            //             bm.submitAsyncPrefetchPages(&next_pid, 1, &next_offset);
            //         }
            //     }
            // }

            for (size_t grouped_idx = pos_idx; grouped_idx < page_run_idx; ++grouped_idx) {
                const u32 page_row = positions[grouped_idx];
                const size_t page_local_row = page_row - row_base;
                const size_t page_logical_offset = base + page_local_row * PDX_HORIZONTAL_BLOCK_DIMS;
                const size_t page_run_in_page = page_logical_offset % PDXFloatPage::floatsPerPage();

                if (page_run_in_page + block_width <= page_float_count) {
                    distances[page_row] +=
                        pdxHorizontalDistance(query_block, page_data + page_run_in_page, block_width);
                } else {
                    distances[page_row] +=
                        horizontalBlockDistance(data_base_pid, cluster_extent_data_page_count_, page_logical_offset,
                                                query_block, block_width);
                }
            }

            pos_idx = page_run_idx;
        }
    }
    dist_ops += static_cast<u64>(position_count) * block_width;
}

void PDXIndex::mergeIntoHeap(const ClusterState& cluster, const std::vector<float>& distances, size_t k,
                             const u32* positions, size_t position_count,
                             std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                             u32 min_row_to_merge) const {
    if (positions == nullptr) {
        position_count = cluster.num_embeddings;
    }
    for (size_t idx = 0; idx < position_count; ++idx) {
        const u32 row = positions == nullptr ? static_cast<u32>(idx) : positions[idx];
        if (row < min_row_to_merge) {
            continue;
        }
        const float distance = distances[row];
        const u32 id = cluster.row_ids[row];
        if (result_heap.size() < k) {
            result_heap.push({distance, id});
            if (result_heap.size() == k) {
                heap_threshold = result_heap.top().first;
            }
        } else if (distance < heap_threshold) {
            result_heap.pop();
            result_heap.push({distance, id});
            heap_threshold = result_heap.top().first;
        }
    }
}

void PDXIndex::fullClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                                 std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                                 std::vector<float>& distances, bool collect_stats) const {
    if (distances.size() < cluster.num_embeddings) {
        distances.resize(cluster.num_embeddings, 0.0f);
    }
    std::fill(distances.begin(), distances.begin() + cluster.num_embeddings, 0.0f);
    u64 dist_ops = 0;
    if (vertical_dims_ > 0) {
        scanVerticalAll(cluster, transformed_query, 0, vertical_dims_, distances, dist_ops);
    }
    for (u32 block_idx = 0; block_idx < horizontal_group_count_; ++block_idx) {
        scanHorizontalAllBlock(cluster, transformed_query, block_idx, distances, dist_ops);
    }
    mergeIntoHeap(cluster, distances, k, nullptr, cluster.num_embeddings, result_heap, heap_threshold, 0);

    if (collect_stats) {
        stats_.dist_comps.fetch_add(dist_ops, std::memory_order_relaxed);
        stats_.vectors_scanned.fetch_add(cluster.num_embeddings, std::memory_order_relaxed);
    }
}

size_t PDXIndex::startClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                                    std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                                    std::vector<float>& distances, bool collect_stats) const {
    const size_t seed_count = std::min<size_t>(k, cluster.num_embeddings);
    if (seed_count == 0) {
        return 0;
    }

    if (distances.size() < cluster.num_embeddings) {
        distances.resize(cluster.num_embeddings, 0.0f);
    }
    std::fill(distances.begin(), distances.begin() + seed_count, 0.0f);

    std::vector<u32> positions(seed_count, 0u);
    std::iota(positions.begin(), positions.end(), 0u);

    u64 dist_ops = 0;
    if (vertical_dims_ > 0) {
        scanVerticalSelected(cluster, transformed_query, 0, vertical_dims_, positions.data(), seed_count, distances,
                             dist_ops);
    }
    for (u32 block_idx = 0; block_idx < horizontal_group_count_; ++block_idx) {
        scanHorizontalSelectedBlock(cluster, transformed_query, block_idx, positions.data(), seed_count, distances,
                                    dist_ops);
    }
    mergeIntoHeap(cluster, distances, k, positions.data(), seed_count, result_heap, heap_threshold, 0);

    if (collect_stats) {
        stats_.dist_comps.fetch_add(dist_ops, std::memory_order_relaxed);
        stats_.vectors_scanned.fetch_add(seed_count, std::memory_order_relaxed);
    }
    return seed_count;
}

size_t PDXIndex::warmupClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                                     std::priority_queue<std::pair<float, u32>>& result_heap, float heap_threshold,
                                     std::vector<float>& distances, u32& current_dimension_idx,
                                     u32* positions, bool collect_stats) const {
    std::iota(positions, positions + cluster.num_embeddings, 0u);
    size_t active_count = cluster.num_embeddings;
    size_t fetch_round = 0;
    u64 dist_ops = 0;
    size_t pruned_count = 0;
    const size_t prune_target = static_cast<size_t>(
        std::ceil(static_cast<double>(cluster.num_embeddings) * PDX_DEFAULT_SELECTIVITY_THRESHOLD));

    while (current_dimension_idx < vertical_dims_ && pruned_count < prune_target) {
        const u32 fetch = nextFetchSize(fetch_round++);
        const u32 next_dimension = std::min<u32>(vertical_dims_, current_dimension_idx + fetch);
        scanVerticalAll(cluster, transformed_query, current_dimension_idx, next_dimension, distances, dist_ops);
        current_dimension_idx = next_dimension;

        if (result_heap.size() < k || !std::isfinite(heap_threshold)) {
            continue;
        }

        const float threshold = pruningThreshold(heap_threshold, current_dimension_idx);
        pruned_count = countPrunedCandidates(distances, threshold, cluster.num_embeddings);
    }

    if (result_heap.size() >= k && std::isfinite(heap_threshold)) {
        const float threshold = pruningThreshold(heap_threshold, current_dimension_idx);
        active_count = filterCandidatePositions(distances, threshold, positions, cluster.num_embeddings);
    }

    if (collect_stats) {
        stats_.dist_comps.fetch_add(dist_ops, std::memory_order_relaxed);
        stats_.vectors_scanned.fetch_add(cluster.num_embeddings, std::memory_order_relaxed);
        stats_.vectors_pruned.fetch_add(cluster.num_embeddings - active_count, std::memory_order_relaxed);
    }
    return active_count;
}

size_t PDXIndex::pruneClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                                    std::priority_queue<std::pair<float, u32>>& result_heap, float heap_threshold,
                                    std::vector<float>& distances, u32 current_vertical_dims,
                                    u32 current_dimension_idx, u32* positions, size_t position_count,
                                    bool collect_stats, u32 min_row_to_merge) const {
    (void)current_vertical_dims;
    size_t active_count = position_count;
    u64 dist_ops = 0;
    u32 visited_dims = current_dimension_idx;

    if (result_heap.size() >= k && std::isfinite(heap_threshold)) {
        const float threshold = pruningThreshold(heap_threshold, visited_dims);
        const size_t previous = active_count;
        active_count = filterCandidatePositions(distances, threshold, positions, active_count);
        if (collect_stats) {
            stats_.vectors_pruned.fetch_add(previous - active_count, std::memory_order_relaxed);
        }
    }

    for (u32 block_idx = 0; block_idx < horizontal_group_count_ && active_count > 0; ++block_idx) {
        scanHorizontalSelectedBlock(cluster, transformed_query, block_idx, positions, active_count, distances,
                                    dist_ops);
        visited_dims += std::min<u32>(PDX_HORIZONTAL_BLOCK_DIMS, horizontal_dims_ - block_idx * PDX_HORIZONTAL_BLOCK_DIMS);
        if (result_heap.size() >= k && std::isfinite(heap_threshold)) {
            const float threshold = pruningThreshold(heap_threshold, visited_dims);
            const size_t previous = active_count;
            active_count = filterCandidatePositions(distances, threshold, positions, active_count);
            if (collect_stats) {
                stats_.vectors_pruned.fetch_add(previous - active_count, std::memory_order_relaxed);
            }
        }
    }

    while (current_dimension_idx < vertical_dims_ && active_count > 0) {
        const u32 next_dimension = std::min<u32>(vertical_dims_, current_dimension_idx + PDX_HORIZONTAL_BLOCK_DIMS);
        scanVerticalSelected(cluster, transformed_query, current_dimension_idx, next_dimension, positions, active_count,
                             distances, dist_ops);
        current_dimension_idx = next_dimension;
        visited_dims = horizontal_dims_ + current_dimension_idx;
        if (result_heap.size() >= k && std::isfinite(heap_threshold)) {
            const float threshold = pruningThreshold(heap_threshold, visited_dims);
            const size_t previous = active_count;
            active_count = filterCandidatePositions(distances, threshold, positions, active_count);
            if (collect_stats) {
                stats_.vectors_pruned.fetch_add(previous - active_count, std::memory_order_relaxed);
            }
        }
    }

    mergeIntoHeap(cluster, distances, k, positions, active_count, result_heap, heap_threshold, min_row_to_merge);

    if (collect_stats) {
        stats_.dist_comps.fetch_add(dist_ops, std::memory_order_relaxed);
    }
    return active_count;
}

void PDXIndex::train(const float* training_vectors, u64 n_train, u32 kmeans_iters) {
    if (n_train < num_clusters_) {
        throw std::runtime_error("PDX training requires at least num_clusters vectors.");
    }
    if (size() != 0) {
        throw std::runtime_error("PDX training must happen before vectors are added.");
    }
    if (kmeans_iters == 0) {
        kmeans_iters = 1;
    }

    std::vector<float> transformed(static_cast<size_t>(n_train) * dim_);
    transformVectors(training_vectors, n_train, transformed.data(), 1);

    u64 sampled_count = 0;
    std::vector<float> sampled = sampleTrainingVectors(transformed.data(), n_train, sampled_count);
    if (sampled_count < num_clusters_) {
        throw std::runtime_error("PDX sampled training set is smaller than num_clusters.");
    }

    std::vector<float> centroids(static_cast<size_t>(num_clusters_) * dim_);
    initializeCentroidsKMeansPlusPlus(sampled.data(), sampled_count, centroids.data());

    std::vector<u32> assignments(sampled_count, 0);
    for (u32 iter = 0; iter < kmeans_iters; ++iter) {
        kmeansStep(sampled.data(), sampled_count, centroids.data(), assignments);
    }

    writeCentroids(centroids.data());

    {
        GuardX<PDXMetadataPage> meta_guard(metadata_pid_);
        meta_guard->is_trained = 1;
        meta_guard->last_train_count.store(meta_guard->num_vectors.load(std::memory_order_relaxed),
                                           std::memory_order_relaxed);
        meta_guard->dirty = true;
    }

    is_trained_.store(true, std::memory_order_release);
    persistMetadata(true);
    publishRootAnchor();
}

void PDXIndex::addPoint(const float* vector, u32 vector_id) {
    if (!isTrained()) {
        throw std::runtime_error("PDX index must be trained before adding vectors.");
    }
    if (size() >= max_elements_) {
        throw std::runtime_error("PDX index is full.");
    }

    std::vector<float> transformed(dim_);
    transformVector(vector, transformed.data());
    const u32 cluster_id = findNearestCentroid(transformed.data());
    rebuildClusterStorage(cluster_id, transformed.data(), &vector_id, 1);

    GuardX<PDXMetadataPage> meta_guard(metadata_pid_);
    const u64 new_count = meta_guard->num_vectors.fetch_add(1, std::memory_order_relaxed) + 1;
    if (retrain_interval_ > 0 &&
        (new_count - meta_guard->last_train_count.load(std::memory_order_relaxed)) >= retrain_interval_) {
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
    meta_guard->dirty = true;
}

void PDXIndex::addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads) {
    if (!isTrained()) {
        throw std::runtime_error("PDX index must be trained before adding vectors.");
    }
    if (count == 0) {
        return;
    }
    if (size() + count > max_elements_) {
        throw std::runtime_error("PDX addPoints would exceed max_elements.");
    }

    std::vector<float> transformed(static_cast<size_t>(count) * dim_);
    transformVectors(vectors, count, transformed.data(), num_threads);

    std::vector<u32> cluster_ids(count, 0);
    for (u64 i = 0; i < count; ++i) {
        cluster_ids[i] = findNearestCentroid(transformed.data() + i * dim_);
    }

    std::vector<std::vector<u64>> cluster_to_indices(num_clusters_);
    for (u64 i = 0; i < count; ++i) {
        cluster_to_indices[cluster_ids[i]].push_back(i);
    }

    const u32 resolved_threads = normalizeThreadCount(num_threads);
    if (resolved_threads == 1 || count < 32) {
        for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
            const auto& indices = cluster_to_indices[cluster_id];
            if (indices.empty()) {
                continue;
            }
            std::vector<float> cluster_vectors(static_cast<size_t>(indices.size()) * dim_);
            std::vector<u32> cluster_row_ids(indices.size());
            for (size_t local_idx = 0; local_idx < indices.size(); ++local_idx) {
                const u64 source_idx = indices[local_idx];
                std::memcpy(cluster_vectors.data() + local_idx * dim_, transformed.data() + source_idx * dim_,
                            static_cast<size_t>(dim_) * sizeof(float));
                cluster_row_ids[local_idx] = ids[source_idx];
            }
            rebuildClusterStorage(cluster_id, cluster_vectors.data(), cluster_row_ids.data(), cluster_row_ids.size());
        }
    } else {
        ThreadPool* pool = getOrCreateAddPool(resolved_threads);
        std::vector<std::future<void>> futures;
        for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
            if (cluster_to_indices[cluster_id].empty()) {
                continue;
            }
            futures.push_back(pool->enqueue([this, cluster_id, &cluster_to_indices, &transformed, ids] {
                const auto& indices = cluster_to_indices[cluster_id];
                std::vector<float> cluster_vectors(static_cast<size_t>(indices.size()) * dim_);
                std::vector<u32> cluster_row_ids(indices.size());
                for (size_t local_idx = 0; local_idx < indices.size(); ++local_idx) {
                    const u64 source_idx = indices[local_idx];
                    std::memcpy(cluster_vectors.data() + local_idx * dim_, transformed.data() + source_idx * dim_,
                                static_cast<size_t>(dim_) * sizeof(float));
                    cluster_row_ids[local_idx] = ids[source_idx];
                }
                rebuildClusterStorage(cluster_id, cluster_vectors.data(), cluster_row_ids.data(), cluster_row_ids.size());
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }

    GuardX<PDXMetadataPage> meta_guard(metadata_pid_);
    const u64 new_count = meta_guard->num_vectors.fetch_add(count, std::memory_order_relaxed) + count;
    if (retrain_interval_ > 0 &&
        (new_count - meta_guard->last_train_count.load(std::memory_order_relaxed)) >= retrain_interval_) {
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
    meta_guard->dirty = true;
}

std::vector<std::pair<float, u32>> PDXIndex::search(const float* query, size_t k, size_t nprobe, bool collect_stats) {
    if (!isTrained()) {
        throw std::runtime_error("PDX index must be trained before searching.");
    }
    if (k == 0) {
        return {};
    }

    std::vector<float> transformed_query(dim_);
    transformVector(query, transformed_query.data());

    for (u64 retry_counter = 0;; ++retry_counter) {
        try {
            const size_t effective_nprobe =
                (nprobe == 0)
                    ? static_cast<size_t>(num_clusters_)
                    : std::min<size_t>(num_clusters_, std::max<size_t>(
                                                     1,
                                                     static_cast<size_t>(std::ceil(
                                                         static_cast<double>(nprobe) * nprobe_multiplier_))));
            const auto nearest_clusters = findNearestCentroids(transformed_query.data(), effective_nprobe);
            if (collect_stats) {
                stats_.lists_probed.fetch_add(nearest_clusters.size(), std::memory_order_relaxed);
            }

            std::priority_queue<std::pair<float, u32>> result_heap;
            float heap_threshold = std::numeric_limits<float>::max();
            std::vector<float> distances_buffer(max_cluster_capacity_, 0.0f);
            std::vector<u32> positions_buffer(max_cluster_capacity_, 0u);
            size_t seeded_cluster_idx = nearest_clusters.size();
            u32 seeded_row_count = 0;

            if (!disable_pruning_) {
                for (size_t cluster_idx = 0; cluster_idx < nearest_clusters.size(); ++cluster_idx) {
                    const ClusterState& cluster = cluster_states_[nearest_clusters[cluster_idx].second];
                    if (cluster.num_embeddings == 0) {
                        continue;
                    }

                    seeded_cluster_idx = cluster_idx;
                    seeded_row_count = static_cast<u32>(startClusterSearch(
                        cluster, transformed_query.data(), k, result_heap, heap_threshold, distances_buffer,
                        collect_stats));
                    if (result_heap.size() == k) {
                        heap_threshold = result_heap.top().first;
                    }
                    break;
                }
            }

            for (size_t cluster_idx = 0; cluster_idx < nearest_clusters.size(); ++cluster_idx) {
                const ClusterState& cluster = cluster_states_[nearest_clusters[cluster_idx].second];
                if (cluster.num_embeddings == 0) {
                    continue;
                }

                if (disable_pruning_) {
                    fullClusterSearch(cluster, transformed_query.data(), k, result_heap, heap_threshold,
                                      distances_buffer, collect_stats);
                    if (result_heap.size() == k) {
                        heap_threshold = result_heap.top().first;
                    }
                    continue;
                }

                const u32 min_row_to_merge = (cluster_idx == seeded_cluster_idx) ? seeded_row_count : 0;

                std::fill(distances_buffer.begin(), distances_buffer.begin() + cluster.num_embeddings, 0.0f);
                u32 current_dimension_idx = 0;
                const size_t active_count = warmupClusterSearch(cluster, transformed_query.data(), k, result_heap,
                                                                heap_threshold, distances_buffer, current_dimension_idx,
                                                                positions_buffer.data(), collect_stats);
                pruneClusterSearch(cluster, transformed_query.data(), k, result_heap, heap_threshold, distances_buffer,
                                   current_dimension_idx, current_dimension_idx, positions_buffer.data(),
                                   active_count, collect_stats, min_row_to_merge);
                if (result_heap.size() == k) {
                    heap_threshold = result_heap.top().first;
                }
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

std::vector<std::vector<std::pair<float, u32>>> PDXIndex::searchBatch(std::span<const float> queries, size_t k,
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
    const size_t task_count = std::min<size_t>(resolved_threads, num_queries);
    constexpr size_t query_chunk_size = 8;
    std::atomic<size_t> next_query{0};
    std::vector<std::future<void>> futures;
    futures.reserve(task_count);
    for (size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        futures.push_back(pool->enqueue([this, &results, &next_query, queries, num_queries, k, nprobe, collect_stats] {
            for (;;) {
                const size_t begin = next_query.fetch_add(query_chunk_size, std::memory_order_relaxed);
                if (begin >= num_queries) {
                    break;
                }
                const size_t end = std::min(num_queries, begin + query_chunk_size);
                for (size_t i = begin; i < end; ++i) {
                    results[i] = search(queries.data() + i * dim_, k, nprobe, collect_stats);
                }
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    return results;
}

PDXStats PDXIndex::getStats() const {
    PDXStats snapshot(stats_);
    snapshot.num_clusters = num_clusters_;
    snapshot.cluster_sizes.assign(num_clusters_, 0);

    u64 total_vectors = 0;
    for (u32 cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const u64 cluster_size = cluster_states_[cluster_id].num_embeddings;
        snapshot.cluster_sizes[cluster_id] = cluster_size;
        total_vectors += cluster_size;
    }
    snapshot.avg_cluster_size = num_clusters_ == 0 ? 0.0 : static_cast<double>(total_vectors) / num_clusters_;
    return snapshot;
}

void PDXIndex::flush() {
    flush_system();
}

u64 PDXIndex::size() const {
    GuardORelaxed<PDXMetadataPage> meta_guard(metadata_pid_);
    return meta_guard->num_vectors.load(std::memory_order_acquire);
}

std::string PDXIndex::getIndexInfo() const {
    std::ostringstream out;
    out << "=== PDX Index Information ===\n";
    out << "Configuration:\n";
    out << "  Vector Dimension: " << dim_ << "\n";
    out << "  Num Clusters: " << num_clusters_ << "\n";
    out << "  Max Elements: " << max_elements_ << "\n";
    out << "  Vertical Dims: " << vertical_dims_ << "\n";
    out << "  Horizontal Dims: " << horizontal_dims_ << "\n";
    out << "  Horizontal Groups: " << horizontal_group_count_ << "\n";
    out << "  Sampling Fraction: " << sampling_fraction_ << "\n";
    out << "  NProbe Multiplier: " << nprobe_multiplier_ << "\n";
    out << "  Transform Seed: " << transform_seed_ << "\n";
    out << "  Prefetch Enabled: " << (enable_prefetch_ ? "true" : "false") << "\n";

    GuardO<PDXMetadataPage> meta_guard(metadata_pid_);
    out << "\nMetadata:\n";
    out << "  Metadata PID: " << metadata_pid_ << "\n";
    out << "  Centroid Base PID: " << meta_guard->centroids_base_pid << "\n";
    out << "  Cluster Dir Base PID: " << meta_guard->cluster_dir_base_pid << "\n";
    out << "  Current Vector Count: " << meta_guard->num_vectors.load(std::memory_order_acquire) << "\n";
    out << "  Trained: " << (meta_guard->is_trained ? "true" : "false") << "\n";

    PDXStats stats = getStats();
    out << "\nRuntime Stats:\n";
    out << "  Lists Probed: " << stats.lists_probed.load() << "\n";
    out << "  Vectors Scanned: " << stats.vectors_scanned.load() << "\n";
    out << "  Vectors Pruned: " << stats.vectors_pruned.load() << "\n";
    out << "  Distance Components: " << stats.dist_comps.load() << "\n";
    out << "  Avg Cluster Size: " << std::fixed << std::setprecision(2) << stats.avg_cluster_size << "\n";
    out << "=== End PDX Index Information ===\n";
    return out.str();
}
