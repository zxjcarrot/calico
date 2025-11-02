#ifndef DISKANN_HPP
#define DISKANN_HPP

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"
#include "utils.hpp"

// Forward declaration of the ThreadPool class
class ThreadPool;

// --- Abstract Base Class for Python Binding (Type Erasure) ---
class DiskANNBase {
   public:
    virtual ~DiskANNBase() = default;

    struct BuildParams {
        size_t L_build = 100;
        float alpha = 1.2f;
        size_t num_threads = 0;
    };
    struct SearchParams {
        size_t L_search;
        size_t beam_width = 4;
    };

    virtual void build(const void* data, const std::vector<std::vector<uint32_t>>& tags, uint64_t num_points,
                       const BuildParams& params) = 0;
    virtual std::vector<std::pair<float, uint32_t>> search(const void* query, size_t K, const SearchParams& params) = 0;
    virtual std::vector<std::pair<float, uint32_t>> search_with_filter(const void* query, uint32_t filter_label,
                                                                       size_t K, const SearchParams& params) = 0;
    virtual void insert_point(const void* point, const std::vector<uint32_t>& tags, uint32_t external_id) = 0;
    virtual void lazy_delete(uint32_t external_id) = 0;
    virtual void consolidate_deletes(const BuildParams& params) = 0;
    virtual size_t get_dimensions() const = 0;
    virtual size_t get_R() const = 0;
    virtual std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel(const void* queries,
                                                                                 size_t num_queries, size_t K,
                                                                                 const SearchParams& params,
                                                                                 size_t num_threads) = 0;
    virtual std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel(
        const void* queries, size_t num_queries, uint32_t filter_label, size_t K, const SearchParams& params,
        size_t num_threads) = 0;
};

std::unique_ptr<DiskANNBase> create_index(size_t dimensions, uint64_t max_elements, size_t R_max_degree,
                                          bool is_dynamic);

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode = 8>
class DiskANN : public DiskANNBase {
    static_assert(std::is_same_v<TagT, uint32_t>, "TagT must be uint32_t to implement the DiskANNBase interface.");

   public:
    DiskANN(uint64_t max_elements, size_t R_max_degree, bool is_dynamic);
    ~DiskANN();

    void build(const void* data, const std::vector<std::vector<uint32_t>>& tags, uint64_t num_points,
               const BuildParams& params) override;
    std::vector<std::pair<float, uint32_t>> search(const void* query, size_t K, const SearchParams& params) override;
    std::vector<std::pair<float, uint32_t>> search_with_filter(const void* query, uint32_t filter_label, size_t K,
                                                               const SearchParams& params) override;
    void insert_point(const void* point, const std::vector<uint32_t>& tags, uint32_t external_id) override;
    void lazy_delete(uint32_t external_id) override;
    void consolidate_deletes(const BuildParams& params) override;
    size_t get_dimensions() const override { return Dim; }
    size_t get_R() const override { return _R_max_degree; }
    std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel(const void* queries, size_t num_queries,
                                                                         size_t K, const SearchParams& params,
                                                                         size_t num_threads) override;
    std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel(const void* queries,
                                                                                     size_t num_queries,
                                                                                     uint32_t filter_label, size_t K,
                                                                                     const SearchParams& params,
                                                                                     size_t num_threads) override;

    void build_typed(const T* data, const std::vector<std::vector<TagT>>& tags, uint64_t num_points,
                     const BuildParams& params);
    std::vector<std::pair<float, uint32_t>> search_typed(const T* query, size_t K, const SearchParams& params);
    std::vector<std::pair<float, uint32_t>> search_with_filter_typed(const T* query, const TagT& filter_label, size_t K,
                                                                     const SearchParams& params);
    void insert_point_typed(const T* point, const std::vector<TagT>& tags, uint32_t external_id);
    std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel_typed(const T* queries, size_t num_queries,
                                                                               size_t K, const SearchParams& params,
                                                                               size_t num_threads);
    std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel_typed(
        const T* queries, size_t num_queries, const TagT& filter_label, size_t K, const SearchParams& params,
        size_t num_threads);

   private:
    struct DiskANNMetadataPage {
        bool dirty;
        PID base_pid;
        uint64_t max_elements;
        std::atomic<uint64_t> node_count;
    };

    struct VamanaPage {
        bool dirty;
        uint16_t node_count_in_page;
        uint16_t padding[3];
        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count_in_page) + sizeof(padding);
        uint8_t* getNodeData() { return reinterpret_cast<uint8_t*>(this) + HeaderSize; }
        const uint8_t* getNodeData() const { return reinterpret_cast<const uint8_t*>(this) + HeaderSize; }
    };

    static constexpr size_t VectorSize = Dim * sizeof(T);
    static constexpr size_t TagsHeaderSize = sizeof(uint16_t);
    static constexpr size_t TagsDataSize = MaxTagsPerNode * sizeof(TagT);
    static constexpr size_t NeighborsHeaderSize = sizeof(uint16_t);

    size_t _NeighborsDataSize;
    size_t _FixedNodeSize;
    size_t _NodesPerPage;

    class NodeAccessor {
       protected:
        const uint8_t* _node_start;
        const DiskANN* _index;

       public:
        NodeAccessor(const VamanaPage* page, uint32_t node_idx, const DiskANN* index);
        const T* getVector() const;
        std::span<const TagT> getTags() const;
        std::span<const uint32_t> getNeighbors() const;
    };

    class MutableNodeAccessor : public NodeAccessor {
       private:
        VamanaPage* _page;

       public:
        MutableNodeAccessor(VamanaPage* page, uint32_t node_idx, const DiskANN* index);
        void setVector(const T* vector);
        void setTags(const std::vector<TagT>& tags);
        void setNeighbors(const std::vector<uint32_t>& neighbors);
    };

    const bool _is_dynamic;
    PID _metadata_pid;
    PID _base_pid;
    uint64_t _max_elements;
    size_t _R_max_degree;

    std::unordered_map<TagT, uint32_t> _medoids;
    std::unordered_map<uint32_t, uint32_t> _external_to_internal_map;
    std::vector<uint32_t> _internal_to_external_map;
    std::shared_mutex _map_lock;
    ConcurrentSet<uint32_t> _deleted_nodes;
    std::unique_ptr<VisitedListPool> _visited_list_pool;

    ThreadPool* getOrCreateThreadPool(size_t num_threads) const;
    mutable std::unique_ptr<ThreadPool> _thread_pool;
    mutable std::mutex _pool_mutex;

   private:
    inline PID getNodePID(uint32_t node_id) const;
    inline uint32_t getNodeIndexInPage(uint32_t node_id) const;

    bool getNodeVector(uint32_t node_id, T* dest_vector) const;
    bool getNodeTags(uint32_t node_id, std::vector<TagT>& dest_tags) const;
    bool getNodeNeighbors(uint32_t node_id, std::vector<uint32_t>& dest_neighbors) const;

    // --- OPTIMIZATION: New helper for batch vector fetching ---
    bool getVectorsForPruning(const std::vector<uint32_t>& ids, std::vector<T>& vectors) const;

    void compute_medoids(uint64_t num_points);
    std::vector<std::pair<float, uint32_t>> greedy_search(const std::vector<uint32_t>& start_nodes, const T* query,
                                                          size_t result_size, const SearchParams& params,
                                                          const TagT* filter_label);
    void robust_prune(uint32_t p_id, std::vector<std::pair<float, uint32_t>>& candidates, float alpha,
                      bool filter_aware);
    void connect_neighbors(uint32_t node_id, const std::vector<uint32_t>& neighbors, const BuildParams& params,
                           bool filter_aware);
};

#endif  // DISKANN_HPP