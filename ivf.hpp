#ifndef IVF_HPP
#define IVF_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"
#include "utils.hpp"

static constexpr u32 IVF_MAX_CACHED_PAGES = 128;
static constexpr u32 IVF_PREFETCH_BATCH_PAGES = 16;
static constexpr u32 IVF_DEFAULT_NUM_CLUSTERS = 256;
static constexpr u32 IVF_DEFAULT_KMEANS_ITERS = 10;
static constexpr u32 IVF_PRUNE_BLOCK_FLOATS = 64;
static constexpr u32 IVF_LAYOUT_VERSION = 3;

struct IVFStats {
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> lists_probed{0};
    std::atomic<u64> vectors_scanned{0};

    u32 num_clusters = 0;
    std::vector<u64> list_sizes;
    double avg_list_size = 0.0;

    IVFStats() = default;

    IVFStats(const IVFStats& other) {
        dist_comps.store(other.dist_comps.load());
        lists_probed.store(other.lists_probed.load());
        vectors_scanned.store(other.vectors_scanned.load());
        num_clusters = other.num_clusters;
        list_sizes = other.list_sizes;
        avg_list_size = other.avg_list_size;
    }

    IVFStats& operator=(const IVFStats& other) {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            lists_probed.store(other.lists_probed.load());
            vectors_scanned.store(other.vectors_scanned.load());
            num_clusters = other.num_clusters;
            list_sizes = other.list_sizes;
            avg_list_size = other.avg_list_size;
        }
        return *this;
    }

    void reset_live_counters() {
        dist_comps.store(0);
        lists_probed.store(0);
        vectors_scanned.store(0);
    }

    std::string toString() const;
};

struct IVFFloatChunkPage {
    bool dirty;
    u8 pad0[7];
    PID next_page;
    u32 float_count;
    u32 reserved;

    IVFFloatChunkPage() : dirty(false), next_page(BufferManager::invalidPID), float_count(0), reserved(0) {
        memset(pad0, 0, sizeof(pad0));
    }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(next_page) + sizeof(float_count) +
                                         sizeof(reserved);

    static constexpr u32 floatsPerPage() {
        return static_cast<u32>((pageSize - HeaderSize) / sizeof(float));
    }

    float* getData() { return reinterpret_cast<float*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const float* getData() const { return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(this) + HeaderSize); }
};

struct InvListEntry {
    std::atomic<u32> list_size;
    std::atomic<u32> num_pages;
    PID first_page_pid;
    PID last_page_pid;
    PID cached_pages[IVF_MAX_CACHED_PAGES];
    std::atomic<u32> cached_page_count;
    u32 reserved;

    void init() {
        list_size.store(0, std::memory_order_relaxed);
        num_pages.store(0, std::memory_order_relaxed);
        first_page_pid = BufferManager::invalidPID;
        last_page_pid = BufferManager::invalidPID;
        cached_page_count.store(0, std::memory_order_relaxed);
        reserved = 0;
        for (u32 i = 0; i < IVF_MAX_CACHED_PAGES; ++i) {
            cached_pages[i] = BufferManager::invalidPID;
        }
    }
};

struct IVFListDirPage {
    bool dirty;
    u8 pad0[7];
    u32 entry_count;
    u32 first_cluster_id;
    u8 pad1[8];

    IVFListDirPage() : dirty(false), entry_count(0), first_cluster_id(0) {
        memset(pad0, 0, sizeof(pad0));
        memset(pad1, 0, sizeof(pad1));
    }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(entry_count) +
                                         sizeof(first_cluster_id) + sizeof(pad1);

    static constexpr u32 entriesPerPage() {
        return static_cast<u32>((pageSize - HeaderSize) / sizeof(InvListEntry));
    }

    InvListEntry* getEntries() { return reinterpret_cast<InvListEntry*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const InvListEntry* getEntries() const {
        return reinterpret_cast<const InvListEntry*>(reinterpret_cast<const u8*>(this) + HeaderSize);
    }

    InvListEntry* getEntry(u32 local_idx) { return getEntries() + local_idx; }
    const InvListEntry* getEntry(u32 local_idx) const { return getEntries() + local_idx; }
};

struct IVFListVectorEntry {
    u32 original_id;
    u32 reserved;

    static constexpr size_t HeaderSize = sizeof(original_id) + sizeof(reserved);

    float* getResidual() { return reinterpret_cast<float*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const float* getResidual() const {
        return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(this) + HeaderSize);
    }
};

struct IVFListDataPage {
    bool dirty;
    u8 pad0[7];
    PID next_page;
    u32 count;
    u32 capacity;
    u8 pad1[8];

    IVFListDataPage() : dirty(false), next_page(BufferManager::invalidPID), count(0), capacity(0) {
        memset(pad0, 0, sizeof(pad0));
        memset(pad1, 0, sizeof(pad1));
    }

    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(pad0) + sizeof(next_page) + sizeof(count) +
                                         sizeof(capacity) + sizeof(pad1);

    static constexpr size_t entrySize(u32 dim) {
        return IVFListVectorEntry::HeaderSize + static_cast<size_t>(dim) * sizeof(float);
    }

    static constexpr u32 entriesPerPage(u32 dim) {
        return static_cast<u32>((pageSize - HeaderSize) / entrySize(dim));
    }

    u8* getEntryData() { return reinterpret_cast<u8*>(this) + HeaderSize; }
    const u8* getEntryData() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }

    IVFListVectorEntry* getEntry(u32 idx, u32 entry_size) {
        return reinterpret_cast<IVFListVectorEntry*>(getEntryData() + static_cast<size_t>(idx) * entry_size);
    }
    const IVFListVectorEntry* getEntry(u32 idx, u32 entry_size) const {
        return reinterpret_cast<const IVFListVectorEntry*>(getEntryData() + static_cast<size_t>(idx) * entry_size);
    }
};

struct IVFMetadataPage {
    bool dirty;
    u8 pad0[7];
    u32 dim;
    u32 num_clusters;
    u32 retrain_interval;
    u32 layout_version;
    u32 floats_per_chunk_page;
    u32 chunk_pages_per_vector;
    u32 entries_per_list_page;
    u32 list_entry_size;
    u32 entries_per_dir_page;
    u32 dir_pages;
    u32 reserved0;
    u8 is_trained;
    u8 pad1[7];
    u64 max_elements;
    std::atomic<u64> num_vectors;
    std::atomic<u64> last_train_count;
    PID centroids_base_pid;
    PID invlist_dir_base_pid;

    IVFMetadataPage()
        : dirty(false),
          dim(0),
          num_clusters(0),
          retrain_interval(0),
          layout_version(0),
          floats_per_chunk_page(0),
          chunk_pages_per_vector(0),
          entries_per_list_page(0),
          list_entry_size(0),
          entries_per_dir_page(0),
          dir_pages(0),
          reserved0(0),
          is_trained(0),
          max_elements(0),
          num_vectors(0),
          last_train_count(0),
          centroids_base_pid(BufferManager::invalidPID),
                    invlist_dir_base_pid(BufferManager::invalidPID) {
                memset(pad0, 0, sizeof(pad0));
                memset(pad1, 0, sizeof(pad1));
        }
};

static_assert(sizeof(IVFMetadataPage) <= pageSize, "IVF metadata page must fit inside one page");

class IVFIndex {
   public:
    IVFIndex(u64 max_elements, size_t dim, u32 num_clusters = IVF_DEFAULT_NUM_CLUSTERS,
             u32 retrain_interval = 0, bool enable_prefetch = true, bool skip_recovery = false);
    ~IVFIndex();

    void train(const float* training_vectors, u64 n_train, u32 kmeans_iters = IVF_DEFAULT_KMEANS_ITERS);
    bool isTrained() const { return is_trained_.load(std::memory_order_acquire); }

    void addPoint(const float* vector, u32 vector_id);
    void addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads = 1);

    std::vector<std::pair<float, u32>> search(const float* query, size_t k, size_t nprobe, bool collect_stats = false);
    std::vector<std::vector<std::pair<float, u32>>> searchBatch(std::span<const float> queries, size_t k,
                                                                size_t nprobe, size_t num_threads = 1,
                                                                bool collect_stats = false);

    IVFStats getStats() const;
    void resetStats() { stats_.reset_live_counters(); }

    void flush();
    bool wasRecovered() const { return recovered_from_disk_; }

    u32 getDim() const { return dim_; }
    u32 getNumClusters() const { return num_clusters_; }
    u64 size() const;
    std::string getIndexInfo() const;

   private:
    u32 dim_;
    u32 num_clusters_;
    u32 retrain_interval_;
    u64 max_elements_;
    bool enable_prefetch_;
    std::unique_ptr<PIDAllocator[]> cluster_pid_allocators_;

    PID metadata_pid_ = BufferManager::invalidPID;
    PID centroids_base_pid_ = BufferManager::invalidPID;
    PID invlist_dir_base_pid_ = BufferManager::invalidPID;

    u32 floats_per_chunk_page_ = 0;
    u32 chunk_pages_per_vector_ = 0;
    u32 entries_per_dir_page_ = 0;
    u32 entries_per_list_page_ = 0;
    u32 list_entry_size_ = 0;
    u32 dir_pages_ = 0;

    std::atomic<bool> is_trained_{false};
    bool recovered_from_disk_ = false;
    mutable IVFStats stats_;

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

    void initializeFreshStorage();
    void resetExistingStorage();
    void publishSharedMetadata(bool trained_flag);
    PIDAllocator* clusterPidAllocator(u32 cluster_id);

    u32 chunkFloatCount(u32 chunk_idx) const;
    PID centroidChunkPid(u32 cluster_id, u32 chunk_idx) const;

    float computeDistanceToCentroid(const float* vector, u32 cluster_id) const;
    void computeResidualToCentroid(const float* vector, u32 cluster_id, float* residual) const;
    void storeCentroid(u32 cluster_id, const float* centroid);
    void initializeCentroidsKMeansPlusPlus(const float* vectors, u64 n, float* centroids);
    void kmeansStep(const float* vectors, u64 n, float* centroids, std::vector<u32>& assignments);

    u32 findNearestCentroid(const float* vector) const;
    std::vector<std::pair<float, u32>> findNearestCentroids(const float* vector, size_t nprobe) const;

    void appendToInvList(u32 cluster_id, u32 vector_id, const float* residual);
    u32 scanClusterList(u32 cluster_id, const float* query_residual, size_t k,
                        std::priority_queue<std::pair<float, u32>>& result_heap, float& heap_threshold,
                        bool enable_pruning, bool collect_stats) const;
};

#endif