#ifndef PDX_HPP
#define PDX_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "calico.hpp"
#include "distance.hpp"
#include "utils.hpp"

static constexpr u32 PDX_DEFAULT_NUM_CLUSTERS = 256;
static constexpr u32 PDX_DEFAULT_KMEANS_ITERS = 10;
static constexpr u32 PDX_LAYOUT_VERSION = 3;
static constexpr u32 PDX_HORIZONTAL_BLOCK_DIMS = 64;
static constexpr u32 PDX_MIN_CLUSTER_CAPACITY = 256;
static constexpr float PDX_CLUSTER_CAPACITY_GROWTH = 1.3f;
static constexpr float PDX_DEFAULT_PRUNING_AGGRESSIVENESS = 1.5f;
static constexpr float PDX_DEFAULT_NPROBE_MULTIPLIER = 1.5f;
static constexpr float PDX_DEFAULT_SELECTIVITY_THRESHOLD = 0.80f;
static constexpr float PDX_DEFAULT_HORIZONTAL_DIM_PROPORTION = 0.80f;
static constexpr u32 PDX_MIN_EMBEDDINGS_TO_SAMPLE = 30720;
static constexpr u32 PDX_CLUSTER_EXTENT_PAGE_COUNT = 512;
static constexpr u32 PDX_MAX_ASYNC_PREFETCH_WINDOW = 32;
static constexpr size_t PDX_PREFETCH_MAINTENANCE_STRIDE = 4;

static constexpr u32 PDX_DIMENSIONS_FETCHING_SIZES[] = {
    16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 512, 1024, 2048, 16384,
};

static constexpr u32 PDX_METADATA_ROOT_SLOT =
    ((pageSize - sizeof(bool) - (7 * sizeof(u8)) - sizeof(HNSWMetaInfo) - sizeof(IVFMetaInfo) - sizeof(u64)) /
     sizeof(PID)) -
    1;

struct PDXDimensionSplit {
    u32 horizontal_dims;
    u32 vertical_dims;
};

constexpr PDXDimensionSplit getPdxDimensionSplit(u32 dim) {
    if (dim <= PDX_HORIZONTAL_BLOCK_DIMS) {
        return {0, dim};
    }

    float horizontal_proportion = PDX_DEFAULT_HORIZONTAL_DIM_PROPORTION;
    if (dim <= 128) {
        horizontal_proportion = 0.25f;
    }

    u32 horizontal = static_cast<u32>(static_cast<float>(dim) * horizontal_proportion);
    if (horizontal % PDX_HORIZONTAL_BLOCK_DIMS != 0) {
        horizontal = ((horizontal + (PDX_HORIZONTAL_BLOCK_DIMS / 2)) / PDX_HORIZONTAL_BLOCK_DIMS) *
                     PDX_HORIZONTAL_BLOCK_DIMS;
    }
    if (horizontal >= dim) {
        horizontal = PDX_HORIZONTAL_BLOCK_DIMS;
    }

    const u32 vertical = dim - horizontal;
    return {horizontal, vertical};
}

struct PDXStats {
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> lists_probed{0};
    std::atomic<u64> vectors_scanned{0};
    std::atomic<u64> vectors_pruned{0};
    std::atomic<u64> horizontal_selected_page_samples{0};
    std::atomic<u64> horizontal_selected_page_transitions{0};
    std::atomic<u64> horizontal_selected_logical_same_page{0};
    std::atomic<u64> horizontal_selected_payload_same_page{0};
    std::atomic<u64> horizontal_selected_logical_adjacent_page{0};
    std::atomic<u64> horizontal_selected_payload_adjacent_page{0};
    std::atomic<u64> horizontal_selected_logical_same_hugepage{0};
    std::atomic<u64> horizontal_selected_payload_same_hugepage{0};
    std::atomic<u64> horizontal_selected_logical_abs_delta_sum{0};
    std::atomic<u64> horizontal_selected_payload_abs_delta_sum{0};
    std::atomic<u64> horizontal_selected_payload_forward_transitions{0};
    std::atomic<u64> horizontal_selected_payload_backward_transitions{0};
    std::atomic<u64> replay_pid_stream_queries{0};
    std::atomic<u64> replay_pid_stream_samples{0};
    std::atomic<u64> replay_pid_stream_resident_samples{0};
    std::atomic<u64> replay_pid_stream_calico_live_ns{0};
    std::atomic<u64> replay_pid_stream_calico_live_touches{0};
    std::atomic<u64> replay_pid_stream_calico_live_checksum{0};
    std::atomic<u64> replay_pid_stream_calico_bypass_ns{0};
    std::atomic<u64> replay_pid_stream_calico_bypass_touches{0};
    std::atomic<u64> replay_pid_stream_calico_bypass_checksum{0};
    std::atomic<u64> replay_pid_stream_vmcache_ns{0};
    std::atomic<u64> replay_pid_stream_vmcache_touches{0};
    std::atomic<u64> replay_pid_stream_vmcache_checksum{0};

    u32 num_clusters = 0;
    std::vector<u64> cluster_sizes;
    double avg_cluster_size = 0.0;

    PDXStats() = default;

    PDXStats(const PDXStats& other) {
        dist_comps.store(other.dist_comps.load());
        lists_probed.store(other.lists_probed.load());
        vectors_scanned.store(other.vectors_scanned.load());
        vectors_pruned.store(other.vectors_pruned.load());
        horizontal_selected_page_samples.store(other.horizontal_selected_page_samples.load());
        horizontal_selected_page_transitions.store(other.horizontal_selected_page_transitions.load());
        horizontal_selected_logical_same_page.store(other.horizontal_selected_logical_same_page.load());
        horizontal_selected_payload_same_page.store(other.horizontal_selected_payload_same_page.load());
        horizontal_selected_logical_adjacent_page.store(other.horizontal_selected_logical_adjacent_page.load());
        horizontal_selected_payload_adjacent_page.store(other.horizontal_selected_payload_adjacent_page.load());
        horizontal_selected_logical_same_hugepage.store(other.horizontal_selected_logical_same_hugepage.load());
        horizontal_selected_payload_same_hugepage.store(other.horizontal_selected_payload_same_hugepage.load());
        horizontal_selected_logical_abs_delta_sum.store(other.horizontal_selected_logical_abs_delta_sum.load());
        horizontal_selected_payload_abs_delta_sum.store(other.horizontal_selected_payload_abs_delta_sum.load());
        horizontal_selected_payload_forward_transitions.store(other.horizontal_selected_payload_forward_transitions.load());
        horizontal_selected_payload_backward_transitions.store(other.horizontal_selected_payload_backward_transitions.load());
        replay_pid_stream_queries.store(other.replay_pid_stream_queries.load());
        replay_pid_stream_samples.store(other.replay_pid_stream_samples.load());
        replay_pid_stream_resident_samples.store(other.replay_pid_stream_resident_samples.load());
        replay_pid_stream_calico_live_ns.store(other.replay_pid_stream_calico_live_ns.load());
        replay_pid_stream_calico_live_touches.store(other.replay_pid_stream_calico_live_touches.load());
        replay_pid_stream_calico_live_checksum.store(other.replay_pid_stream_calico_live_checksum.load());
        replay_pid_stream_calico_bypass_ns.store(other.replay_pid_stream_calico_bypass_ns.load());
        replay_pid_stream_calico_bypass_touches.store(other.replay_pid_stream_calico_bypass_touches.load());
        replay_pid_stream_calico_bypass_checksum.store(other.replay_pid_stream_calico_bypass_checksum.load());
        replay_pid_stream_vmcache_ns.store(other.replay_pid_stream_vmcache_ns.load());
        replay_pid_stream_vmcache_touches.store(other.replay_pid_stream_vmcache_touches.load());
        replay_pid_stream_vmcache_checksum.store(other.replay_pid_stream_vmcache_checksum.load());
        num_clusters = other.num_clusters;
        cluster_sizes = other.cluster_sizes;
        avg_cluster_size = other.avg_cluster_size;
    }

    PDXStats& operator=(const PDXStats& other) {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            lists_probed.store(other.lists_probed.load());
            vectors_scanned.store(other.vectors_scanned.load());
            vectors_pruned.store(other.vectors_pruned.load());
            horizontal_selected_page_samples.store(other.horizontal_selected_page_samples.load());
            horizontal_selected_page_transitions.store(other.horizontal_selected_page_transitions.load());
            horizontal_selected_logical_same_page.store(other.horizontal_selected_logical_same_page.load());
            horizontal_selected_payload_same_page.store(other.horizontal_selected_payload_same_page.load());
            horizontal_selected_logical_adjacent_page.store(other.horizontal_selected_logical_adjacent_page.load());
            horizontal_selected_payload_adjacent_page.store(other.horizontal_selected_payload_adjacent_page.load());
            horizontal_selected_logical_same_hugepage.store(other.horizontal_selected_logical_same_hugepage.load());
            horizontal_selected_payload_same_hugepage.store(other.horizontal_selected_payload_same_hugepage.load());
            horizontal_selected_logical_abs_delta_sum.store(other.horizontal_selected_logical_abs_delta_sum.load());
            horizontal_selected_payload_abs_delta_sum.store(other.horizontal_selected_payload_abs_delta_sum.load());
            horizontal_selected_payload_forward_transitions.store(other.horizontal_selected_payload_forward_transitions.load());
            horizontal_selected_payload_backward_transitions.store(other.horizontal_selected_payload_backward_transitions.load());
            replay_pid_stream_queries.store(other.replay_pid_stream_queries.load());
            replay_pid_stream_samples.store(other.replay_pid_stream_samples.load());
            replay_pid_stream_resident_samples.store(other.replay_pid_stream_resident_samples.load());
            replay_pid_stream_calico_live_ns.store(other.replay_pid_stream_calico_live_ns.load());
            replay_pid_stream_calico_live_touches.store(other.replay_pid_stream_calico_live_touches.load());
            replay_pid_stream_calico_live_checksum.store(other.replay_pid_stream_calico_live_checksum.load());
            replay_pid_stream_calico_bypass_ns.store(other.replay_pid_stream_calico_bypass_ns.load());
            replay_pid_stream_calico_bypass_touches.store(other.replay_pid_stream_calico_bypass_touches.load());
            replay_pid_stream_calico_bypass_checksum.store(other.replay_pid_stream_calico_bypass_checksum.load());
            replay_pid_stream_vmcache_ns.store(other.replay_pid_stream_vmcache_ns.load());
            replay_pid_stream_vmcache_touches.store(other.replay_pid_stream_vmcache_touches.load());
            replay_pid_stream_vmcache_checksum.store(other.replay_pid_stream_vmcache_checksum.load());
            num_clusters = other.num_clusters;
            cluster_sizes = other.cluster_sizes;
            avg_cluster_size = other.avg_cluster_size;
        }
        return *this;
    }

    void reset_live_counters() {
        dist_comps.store(0);
        lists_probed.store(0);
        vectors_scanned.store(0);
        vectors_pruned.store(0);
        horizontal_selected_page_samples.store(0);
        horizontal_selected_page_transitions.store(0);
        horizontal_selected_logical_same_page.store(0);
        horizontal_selected_payload_same_page.store(0);
        horizontal_selected_logical_adjacent_page.store(0);
        horizontal_selected_payload_adjacent_page.store(0);
        horizontal_selected_logical_same_hugepage.store(0);
        horizontal_selected_payload_same_hugepage.store(0);
        horizontal_selected_logical_abs_delta_sum.store(0);
        horizontal_selected_payload_abs_delta_sum.store(0);
        horizontal_selected_payload_forward_transitions.store(0);
        horizontal_selected_payload_backward_transitions.store(0);
        replay_pid_stream_queries.store(0);
        replay_pid_stream_samples.store(0);
        replay_pid_stream_resident_samples.store(0);
        replay_pid_stream_calico_live_ns.store(0);
        replay_pid_stream_calico_live_touches.store(0);
        replay_pid_stream_calico_live_checksum.store(0);
        replay_pid_stream_calico_bypass_ns.store(0);
        replay_pid_stream_calico_bypass_touches.store(0);
        replay_pid_stream_calico_bypass_checksum.store(0);
        replay_pid_stream_vmcache_ns.store(0);
        replay_pid_stream_vmcache_touches.store(0);
        replay_pid_stream_vmcache_checksum.store(0);
    }

    std::string toString() const;
};

struct PDXFloatPage {
    bool dirty;
    u8 pad0[7];
    u32 float_count;
    u32 reserved;

    PDXFloatPage() : dirty(false), float_count(0), reserved(0) { memset(pad0, 0, sizeof(pad0)); }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(float_count) + sizeof(reserved);

    static constexpr u32 floatsPerPage() {
        return static_cast<u32>((pageSize - HeaderSize) / sizeof(float));
    }

    float* getData() { return reinterpret_cast<float*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const float* getData() const {
        return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(this) + HeaderSize);
    }
};

struct PDXIdPage {
    bool dirty;
    u8 pad0[7];
    u32 id_count;
    u32 reserved;

    PDXIdPage() : dirty(false), id_count(0), reserved(0) { memset(pad0, 0, sizeof(pad0)); }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(id_count) + sizeof(reserved);

    static constexpr u32 idsPerPage() {
        return static_cast<u32>((pageSize - HeaderSize) / sizeof(u32));
    }

    u32* getData() { return reinterpret_cast<u32*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const u32* getData() const {
        return reinterpret_cast<const u32*>(reinterpret_cast<const u8*>(this) + HeaderSize);
    }
};

struct PDXClusterExtentPage {
    bool dirty;
    u8 pad0[7];
    PID next_extent_pid;
    u32 capacity;
    u32 data_page_count;
    u32 id_page_count;
    u32 reserved;

    PDXClusterExtentPage()
        : dirty(false), next_extent_pid(BufferManager::invalidPID), capacity(0), data_page_count(0),
          id_page_count(0), reserved(0) {
        memset(pad0, 0, sizeof(pad0));
    }
};

struct PDXClusterEntry {
    std::atomic<u32> num_embeddings;
    u32 extent_capacity;
    PID extent_head_pid;
    u32 extent_count;
    u32 reserved0;
    u32 reserved1;
    u32 reserved2;
    u32 reserved3;

    void init() {
        num_embeddings.store(0, std::memory_order_relaxed);
        extent_capacity = 0;
        extent_head_pid = BufferManager::invalidPID;
        extent_count = 0;
        reserved0 = 0;
        reserved1 = 0;
        reserved2 = 0;
        reserved3 = 0;
    }
};

struct PDXClusterDirPage {
    bool dirty;
    u8 pad0[7];
    u32 entry_count;
    u32 first_cluster_id;
    u8 pad1[8];

    PDXClusterDirPage() : dirty(false), entry_count(0), first_cluster_id(0) {
        memset(pad0, 0, sizeof(pad0));
        memset(pad1, 0, sizeof(pad1));
    }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(entry_count) +
                                         sizeof(first_cluster_id) + sizeof(pad1);

    static constexpr u32 entriesPerPage() {
        return static_cast<u32>((pageSize - HeaderSize) / sizeof(PDXClusterEntry));
    }

    PDXClusterEntry* getEntries() {
        return reinterpret_cast<PDXClusterEntry*>(reinterpret_cast<u8*>(this) + HeaderSize);
    }
    const PDXClusterEntry* getEntries() const {
        return reinterpret_cast<const PDXClusterEntry*>(reinterpret_cast<const u8*>(this) + HeaderSize);
    }

    PDXClusterEntry* getEntry(u32 local_idx) { return getEntries() + local_idx; }
    const PDXClusterEntry* getEntry(u32 local_idx) const { return getEntries() + local_idx; }
};

struct PDXMetadataPage {
    bool dirty;
    u8 pad0[7];
    u32 dim;
    u32 num_clusters;
    u32 retrain_interval;
    u32 layout_version;
    u32 vertical_dims;
    u32 horizontal_dims;
    u32 horizontal_block_dims;
    u32 centroid_page_count;
    u32 cluster_dir_page_count;
    u32 cluster_dir_entries_per_page;
    float sampling_fraction;
    float pruning_aggressiveness;
    u32 transform_seed;
    u8 is_trained;
    u8 pad1[3];
    u64 max_elements;
    std::atomic<u64> num_vectors;
    std::atomic<u64> last_train_count;
    PID centroids_base_pid;
    PID cluster_dir_base_pid;

    PDXMetadataPage()
        : dirty(false),
          dim(0),
          num_clusters(0),
          retrain_interval(0),
          layout_version(0),
          vertical_dims(0),
          horizontal_dims(0),
          horizontal_block_dims(PDX_HORIZONTAL_BLOCK_DIMS),
          centroid_page_count(0),
          cluster_dir_page_count(0),
          cluster_dir_entries_per_page(0),
          sampling_fraction(0.0f),
          pruning_aggressiveness(PDX_DEFAULT_PRUNING_AGGRESSIVENESS),
          transform_seed(42),
          is_trained(0),
          max_elements(0),
          num_vectors(0),
          last_train_count(0),
          centroids_base_pid(BufferManager::invalidPID),
          cluster_dir_base_pid(BufferManager::invalidPID) {
        memset(pad0, 0, sizeof(pad0));
        memset(pad1, 0, sizeof(pad1));
    }
};

static_assert(sizeof(PDXMetadataPage) <= pageSize, "PDX metadata page must fit inside one page");

class PDXIndex {
   public:
    PDXIndex(u64 max_elements, size_t dim, u32 num_clusters = PDX_DEFAULT_NUM_CLUSTERS,
             u32 retrain_interval = 0, bool enable_prefetch = true, bool skip_recovery = false,
             float sampling_fraction = 0.0f, u32 transform_seed = 42);
    ~PDXIndex();

    void train(const float* training_vectors, u64 n_train, u32 kmeans_iters = PDX_DEFAULT_KMEANS_ITERS);
    bool isTrained() const { return is_trained_.load(std::memory_order_acquire); }

    void addPoint(const float* vector, u32 vector_id);
    void addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads = 1);

    std::vector<std::pair<float, u32>> search(const float* query, size_t k, size_t nprobe, bool collect_stats = false);
    std::vector<std::vector<std::pair<float, u32>>> searchBatch(std::span<const float> queries, size_t k,
                                                                size_t nprobe, size_t num_threads = 1,
                                                                bool collect_stats = false);

    PDXStats getStats() const;
    void resetStats() { stats_.reset_live_counters(); }

    void flush();
    bool wasRecovered() const { return recovered_from_disk_; }

    u32 getDim() const { return dim_; }
    u32 getNumClusters() const { return num_clusters_; }
    u64 size() const;
    float getSamplingFraction() const { return sampling_fraction_; }
    u32 getClusterExtentCapacity() const { return cluster_extent_capacity_; }
    u32 getClusterExtentDataPageCount() const { return cluster_extent_data_page_count_; }
    u32 getClusterExtentIdPageCount() const { return cluster_extent_id_page_count_; }
    u32 getClusterExtentCount(u32 cluster_id) const { return cluster_states_[cluster_id].extent_count; }
    std::string getIndexInfo() const;

   private:
    struct ClusterState {
        u32 num_embeddings = 0;
        u32 extent_capacity = 0;
        PID extent_head_pid = BufferManager::invalidPID;
        u32 extent_count = 0;
        std::vector<PID> extent_base_pids;
        std::vector<u32> row_ids;
    };

    struct PrefetchPageRequest {
        PID pid = BufferManager::invalidPID;
        u32 offset = 0;
    };

    struct PrefetchWindowState {
        std::array<size_t, PDX_MAX_ASYNC_PREFETCH_WINDOW> queued_access_indices{};
        size_t head = 0;
        size_t count = 0;
        size_t next_access_idx = 0;
        u32 lookahead = 0;
        u32 refill_threshold = 0;

        void reset(u32 window_size, size_t next_idx) {
            head = 0;
            count = 0;
            next_access_idx = next_idx;
            lookahead = std::min<u32>(window_size, static_cast<u32>(queued_access_indices.size()));
            refill_threshold = lookahead == 0 ? 0 : std::max<u32>(1u, lookahead / 2u);
        }
    };

    u32 dim_;
    u32 num_clusters_;
    u32 retrain_interval_;
    u64 max_elements_;
    bool enable_prefetch_;
    float sampling_fraction_;
    float pruning_aggressiveness_;
    float nprobe_multiplier_ = PDX_DEFAULT_NPROBE_MULTIPLIER;
    bool disable_pruning_ = false;
    std::vector<float> pruning_ratio_lut_;
    u32 transform_seed_;
    u32 vertical_dims_;
    u32 horizontal_dims_;
    u32 horizontal_group_count_;
    u32 cluster_dir_entries_per_page_ = 0;
    u32 cluster_dir_page_count_ = 0;
    u32 centroid_page_count_ = 0;
    u32 cluster_extent_capacity_ = 0;
    u32 cluster_extent_data_page_count_ = 0;
    u32 cluster_extent_id_page_count_ = 0;
    u32 max_cluster_capacity_ = 0;

    std::unique_ptr<PIDAllocator[]> cluster_pid_allocators_;
    PID metadata_pid_ = BufferManager::invalidPID;
    PID centroids_base_pid_ = BufferManager::invalidPID;
    PID cluster_dir_base_pid_ = BufferManager::invalidPID;

    std::atomic<bool> is_trained_{false};
    bool recovered_from_disk_ = false;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rotation_matrix_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rotation_matrix_transposed_;
    std::vector<ClusterState> cluster_states_;
    mutable PDXStats stats_;

    mutable std::unique_ptr<ThreadPool> search_thread_pool_{nullptr};
    mutable size_t search_pool_size_{0};
    mutable std::mutex search_pool_mutex_;

    mutable std::unique_ptr<ThreadPool> add_thread_pool_{nullptr};
    mutable size_t add_pool_size_{0};
    mutable std::mutex add_pool_mutex_;

    ThreadPool* getOrCreateSearchPool(size_t num_threads) const;
    ThreadPool* getOrCreateAddPool(size_t num_threads) const;

    inline float computeDistance(const float* lhs, const float* rhs, u32 count) const {
        return hnsw_distance::SIMDAcceleratedL2::compare(lhs, rhs, count);
    }

    void initializeTransform();
    void transformVector(const float* input, float* output) const;
    void transformVectors(const float* input, u64 count, float* output, size_t num_threads) const;

    float resolveSamplingFraction(u64 n_train) const;
    std::vector<float> sampleTrainingVectors(const float* transformed_vectors, u64 n_train, u64& sampled_count) const;
    void initializeCentroidsKMeansPlusPlus(const float* vectors, u64 n, float* centroids);
    void kmeansStep(const float* vectors, u64 n, float* centroids, std::vector<u32>& assignments);
    void resetClusterDirectoryState();

    void initializeFreshStorage();
    void resetExistingStorage();
    void publishRootAnchor();
    void clearRootAnchor();
    void persistMetadata(bool trained_flag);
    void persistClusterState(u32 cluster_id);
    void loadRecoveredClusterStates();
    void restoreRecoveredAllocCount();
    PIDAllocator* clusterPidAllocator(u32 cluster_id);
    u32 computeClusterExtentCapacity() const;
    PID allocateClusterExtent(PIDAllocator* allocator);
    PDXClusterExtentPage readClusterExtentHeader(PID extent_pid) const;
    void linkClusterExtent(PID extent_pid, PID next_extent_pid);
    size_t clusterExtentRowOffset(size_t extent_index) const;
    u32 clusterExtentRowCount(const ClusterState& state, size_t extent_index) const;
    PID clusterExtentDataBasePid(PID extent_pid) const { return extent_pid + 1; }
    PID clusterExtentIdBasePid(PID extent_pid) const {
        return extent_pid + 1 + static_cast<PID>(cluster_extent_data_page_count_);
    }

    u32 computeClusterCapacity(u32 required) const;
    u32 centroidFloatCount() const { return dim_; }
    u32 centroidFloatsPerPage() const { return PDXFloatPage::floatsPerPage(); }
    PID centroidPid(u32 cluster_id) const { return centroids_base_pid_ + static_cast<PID>(cluster_id); }
    u32 idsPerPage() const { return PDXIdPage::idsPerPage(); }
    u32 dataFloatsPerClusterCapacity(u32 capacity) const { return capacity * dim_; }
    u32 dataPagesForCapacity(u32 capacity) const;
    u32 idPagesForCount(u32 count) const;
    void initializeCentroidPages(PIDAllocator* allocator, bool zero_fill);
    void writeCentroids(const float* centroids);
    template <typename RequestBuilder>
    void maintainPrefetchWindow(PrefetchWindowState& state, size_t current_access_idx, size_t total_accesses,
                                RequestBuilder&& build_request) const {
        if (!enable_prefetch_ || state.lookahead == 0 || total_accesses == 0 || current_access_idx + 1 >= total_accesses) {
            return;
        }

        while (state.count > 0 && state.queued_access_indices[state.head] < current_access_idx) {
            state.head = (state.head + 1) % state.queued_access_indices.size();
            --state.count;
        }
        if (state.next_access_idx <= current_access_idx) {
            state.next_access_idx = current_access_idx + 1;
        }
        if ((current_access_idx % PDX_PREFETCH_MAINTENANCE_STRIDE) != 0) {
            return;
        }
        if (state.count >= state.refill_threshold) {
            return;
        }

        std::array<PID, PDX_MAX_ASYNC_PREFETCH_WINDOW> prefetched_pids{};
        std::array<u32, PDX_MAX_ASYNC_PREFETCH_WINDOW> prefetched_offsets{};
        u32 submit_count = 0;
        while (state.next_access_idx < total_accesses && state.count < state.lookahead) {
            const PrefetchPageRequest request = build_request(state.next_access_idx);
            prefetched_pids[submit_count] = request.pid;
            prefetched_offsets[submit_count] = request.offset;
            const size_t tail = (state.head + state.count) % state.queued_access_indices.size();
            state.queued_access_indices[tail] = state.next_access_idx;
            ++state.count;
            ++state.next_access_idx;
            ++submit_count;
        }

        if (submit_count > 0) {
            bm.submitAsyncPrefetchPages(prefetched_pids.data(), static_cast<int>(submit_count),
                                        prefetched_offsets.data());
        }
    }
    void prefetchNextFloatPage(PID base_pid, u32 page_count, size_t page_idx) const;
    size_t clusterVerticalRowOffset(u32 capacity, u32 dim_idx) const;
    size_t clusterHorizontalBlockOffset(u32 capacity, u32 block_idx) const;

    std::pair<PID, u32> allocateFloatExtent(PIDAllocator* allocator, const float* data, size_t float_count);
    std::pair<PID, u32> allocateIdExtent(PIDAllocator* allocator, const u32* ids, size_t id_count);
    void writeFloatExtent(PID base_pid, u32 page_count, const float* data, size_t float_count);
    void writeIdExtent(PID base_pid, u32 page_count, const u32* ids, size_t id_count);
    std::vector<float> readFloatExtent(PID base_pid, u32 page_count, size_t float_count) const;
    std::vector<u32> readIdExtent(PID base_pid, u32 page_count, size_t id_count) const;
    void writeFloatRange(PID base_pid, u32 page_count, size_t logical_offset, const float* data, size_t float_count);
    void writeIdRange(PID base_pid, u32 page_count, size_t logical_offset, const u32* ids, size_t id_count);

    template <typename Fn>
    void forEachFloatRange(PID base_pid, u32 page_count, size_t logical_offset, size_t float_count, Fn&& fn) const;

    void readFloatBlock(PID base_pid, u32 page_count, size_t logical_offset, float* output, size_t float_count) const;
    float readSingleFloat(PID base_pid, u32 page_count, size_t logical_offset) const;

    void encodeClusterLayout(const float* row_major_vectors, u32 num_embeddings, u32 capacity,
                             std::vector<float>& encoded) const;
    void decodeClusterLayout(const std::vector<float>& encoded, u32 num_embeddings, u32 capacity,
                             std::vector<float>& row_major_vectors) const;
    void writeClusterExtentRows(PID extent_pid, u32 start_row, const float* appended_vectors, const u32* appended_ids,
                                u32 append_count);
    void appendClusterStorage(u32 cluster_id, ClusterState& state, const float* appended_vectors,
                              const u32* appended_ids, u32 append_count);
    void rebuildClusterStorage(u32 cluster_id, const float* appended_vectors, const u32* appended_ids,
                               u32 append_count);

    u32 findNearestCentroid(const float* transformed_vector) const;
    std::vector<std::pair<float, u32>> findNearestCentroids(const float* transformed_vector, size_t nprobe) const;
    float computeCentroidDistanceFromPages(const float* transformed_vector, u32 cluster_id) const;
    void rebuildPruningRatioLut();

    float pruningRatio(u32 visited_dimensions) const;
    float pruningThreshold(float heap_threshold, u32 visited_dimensions) const;
    size_t filterCandidatePositions(const std::vector<float>& distances, float threshold, u32* positions,
                                    size_t count) const;
    size_t countPrunedCandidates(const std::vector<float>& distances, float threshold, u32 num_embeddings) const;
    float horizontalBlockDistance(PID base_pid, u32 page_count, size_t logical_offset, const float* query_block,
                                  u32 block_width) const;

    void scanVerticalAll(const ClusterState& cluster, const float* transformed_query, u32 begin_dim, u32 end_dim,
                         std::vector<float>& distances, u64& dist_ops) const;
    void scanVerticalSelected(const ClusterState& cluster, const float* transformed_query, u32 begin_dim,
                              u32 end_dim, const u32* positions, size_t position_count,
                              std::vector<float>& distances, u64& dist_ops) const;
    void scanHorizontalAllBlock(const ClusterState& cluster, const float* transformed_query, u32 block_idx,
                                std::vector<float>& distances, u64& dist_ops) const;
    void scanHorizontalSelectedBlock(const ClusterState& cluster, const float* transformed_query, u32 block_idx,
                                     const u32* positions, size_t position_count,
                          std::vector<float>& distances, u64& dist_ops) const;
    void mergeIntoHeap(const ClusterState& cluster, const std::vector<float>& distances, size_t k,
                       const u32* positions, size_t position_count,
                       std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                       u32 min_row_to_merge) const;
    void fullClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                           std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                           std::vector<float>& distances, bool collect_stats) const;
    size_t startClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                              std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                  std::vector<float>& distances, bool collect_stats) const;
    size_t warmupClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                               std::priority_queue<std::pair<float, u32>>& result_heap, float heap_threshold,
                               std::vector<float>& distances, u32& current_dimension_idx,
                               u32* positions, bool collect_stats) const;
    size_t pruneClusterSearch(const ClusterState& cluster, const float* transformed_query, size_t k,
                              std::priority_queue<std::pair<float, u32>>& result_heap, float heap_threshold,
                              std::vector<float>& distances, u32 current_vertical_dims,
                              u32 current_dimension_idx, u32* positions, size_t position_count,
                  bool collect_stats, u32 min_row_to_merge) const;
};

#endif
