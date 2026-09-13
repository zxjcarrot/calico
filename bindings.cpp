#include <pybind11/functional.h>  // Needed for py::function
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "calico.hpp"
#include "diskann.hpp"
#include "distance.hpp"
#include "hnsw.hpp"
#include "ivf.hpp"
#include "pdx.hpp"

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

size_t getCurrentRSS() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
}

#elif defined(__APPLE__)
#include <mach/mach.h>

size_t getCurrentRSS() {
    mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}

#else  // Assuming Linux
#include <unistd.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <string>

size_t getCurrentRSS() {
    std::ifstream status_file("/proc/self/status");
    if (!status_file) {
        return 0;
    }
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            try {
                // Format is "VmRSS:     12345 kB"
                size_t rss_kb = std::stoul(line.substr(line.find_first_of("0123456789")));
                return rss_kb * 1024;  // Convert kB to bytes
            } catch (...) {
                return 0;
            }
        }
    }
    return 0;
}
#endif

namespace py = pybind11;

// --- Forward declare init/shutdown functions from calico.cpp ---
void initialize_system();
void shutdown_system();
void flush_system();
void warmup_storage_to_alloc_count();
std::vector<u64> snapshot_pid_to_frame_range(u64 start_pid, u64 count);

static void ensure_system_initialized() {
    initialize_system();
}

// --- Define the specific HNSW instance we will bind ---
// For this binding, we will use a fixed dimension of 128 to match the explicit
// template instantiation in hnsw.cpp. This is a common approach for high-performance bindings.
constexpr size_t HNSW_DIM = 128;
using HnswIndexType = HNSW<hnsw_distance::SIMDAcceleratedL2>;
using IvfIndexType = IVFIndex;
using PdxIndexType = PDXIndex;

// --- Python Module Definition ---
PYBIND11_MODULE(calico_py, m) {
    m.doc() = "Python bindings for the Calico Index(B-Tree, HNSW)";

    // This line is the key. It registers a translator that catches
    // standard C++ exceptions (like std::runtime_error, std::invalid_argument)
    // and converts them into corresponding Python exceptions (RuntimeError, ValueError).
    // The error message from C++ will be preserved.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            // py::exception is a Pybind11 helper to create a Python exception object.
            // PyExc_RuntimeError is the Python C-API equivalent for RuntimeError.
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // py::class_<HnswIndexType>(m, "HnswIndex")
    //     .def(py::init<u64, size_t, size_t, bool>(), py::arg("max_elements"), py::arg("M") = 16,
    //          py::arg("ef_construction") = 200, py::arg("enable_prefetch") = true,
    //          "Initializes a new, empty HNSW index with runtime parameters.")
        m.def("initialize_storage", []() { ensure_system_initialized(); },
            "Initializes the Calico storage subsystem on demand.");
        m.def("shutdown_storage", []() { shutdown_system(); },
            "Flushes and shuts down the Calico storage subsystem.");
        m.def("flush_storage", []() { flush_system(); },
          "Flushes all dirty pages managed by the Calico buffer pool to persistent storage.");
                m.def("warmup_storage_to_alloc_count", []() { warmup_storage_to_alloc_count(); },
                    "Sequentially accesses pid 1 through allocCount-1 to fault recovered pages in ascending pid order.");
                m.def("snapshot_pid_to_frame_range", [](u64 start_pid, u64 count) {
                        return snapshot_pid_to_frame_range(start_pid, count);
                }, py::arg("start_pid"), py::arg("count"),
                    "Returns pid-to-frame mappings for a contiguous pid range. Evicted or unmapped pids are returned as invalidFrame.");

        py::class_<HnswIndexType>(m, "HnswIndex")
                .def(py::init([](u64 max_elements, size_t dim, size_t M, size_t ef_construction,
                           bool enable_prefetch, bool skip_recovery) {
                     ensure_system_initialized();
                 return new HnswIndexType(max_elements, dim, M, ef_construction,
                              enable_prefetch, skip_recovery);
                   }),
                         py::arg("max_elements"),
                         py::arg("dim") = HNSW_DIM,
                         py::arg("M") = 16,
                         py::arg("ef_construction") = 200,
                         py::arg("enable_prefetch") = true,
                         py::arg("skip_recovery") = false,
                         "Initializes a new, empty HNSW index with runtime parameters. Set skip_recovery to True to rebuild the index from scratch.")
        .def(
            "flush",
            [](HnswIndexType&) { flush_system(); },
            "Flushes all dirty pages managed by Calico to persistent storage.")
        .def("was_recovered", &HnswIndexType::wasRecovered,
             "Returns True if the underlying index state was recovered from existing storage.")

        .def(
            "add_points",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items) {
                // Check input array dimensions
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, n_features)");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(items.shape(1)));
                }

                size_t num_points = items.shape(0);
                py::buffer_info buf = items.request();
                float* data_ptr = static_cast<float*>(buf.ptr);

                // Add points one by one
                for (size_t i = 0; i < num_points; ++i) {
                    u32 node_id_out;
                    // Get a pointer to the start of the i-th vector
                    const float* point = data_ptr + i * dim;
                    self.addPoint(point, node_id_out);
                    // For simplicity, we assume the returned node_id matches the item index i.
                    // A more complex implementation could store a mapping if needed.
                }
            },
            py::arg("items"), "Adds a batch of points from a NumPy array to the index.")

        .def(
            "search_knn",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t k,
               size_t ef_search_param, bool stats = false) -> py::object {
                // Check input query vector dimensions
                if (query.ndim() != 1) {
                    throw std::runtime_error("Query must be a 1-dimensional NumPy array");
                }
                const size_t dim = self.getDim();
                if (query.shape(0) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(query.shape(0)));
                }

                const float* query_ptr = query.data();
                std::vector<std::pair<float, u32>> result_vec;
                if (stats) {
                    result_vec = self.searchKnn<true>(query_ptr, k, ef_search_param);
                } else {
                    result_vec = self.searchKnn<false>(query_ptr, k, ef_search_param);
                }

                // Create two Python lists to hold labels and distances
                py::list labels;
                py::list distances;

                for (const auto& pair : result_vec) {
                    distances.append(pair.first);  // distance
                    labels.append(pair.second);    // label (node_id)
                }

                // Return a tuple of two NumPy arrays, similar to hnswlib
                return py::make_tuple(py::array(labels), py::array(distances));
            },
            py::arg("query"), py::arg("k"), py::arg("ef_search"), py::arg("stats") = false,
            "Searches for the k-nearest neighbors for a given query vector. Returns (labels, distances).")

        .def(
            "search_knn_parallel",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
               size_t ef_search_param, size_t num_threads = 0, bool stats = false) -> py::object {
                // --- 1. Validate Input ---
                if (queries.ndim() != 2) {
                    throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim)");
                }
                const size_t dim = self.getDim();
                if (queries.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(queries.shape(1)));
                }
                if (k == 0) {
                    throw std::runtime_error("k must be > 0");
                }

                size_t num_queries = queries.shape(0);
                if (num_queries == 0) {
                    // Explicitly create shape vectors to resolve ambiguity.
                    // We use ssize_t for dimensions, as is common in pybind11/NumPy.
                    std::vector<ssize_t> shape = {0, static_cast<ssize_t>(k)};
                    py::array_t<int64_t> empty_labels(shape);  // Use int64_t for labels
                    py::array_t<float> empty_distances(shape);
                    return py::make_tuple(empty_labels, empty_distances);
                }

                // --- 2. Prepare C++ data and call the parallel search function ---
                std::span<const float> query_span(queries.data(), queries.size());
                std::vector<std::vector<std::pair<float, u32>>> all_results_vec;
                if (stats) {
                    all_results_vec = self.searchKnn_parallel<true>(query_span, k, ef_search_param, num_threads);
                } else {
                    all_results_vec = self.searchKnn_parallel(query_span, k, ef_search_param, num_threads);
                }

                // --- 3. Create output NumPy arrays ---
                // Use int64_t for labels to be consistent with hnswlib and allow -1 as a sentinel
                // value.
                std::vector<ssize_t> result_shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
                py::array_t<int64_t> labels_arr(result_shape);
                py::array_t<float> distances_arr(result_shape);

                auto labels_ptr = labels_arr.mutable_data();
                auto distances_ptr = distances_arr.mutable_data();

                // --- 4. Populate the output arrays from the C++ result vector ---
                for (size_t i = 0; i < num_queries; ++i) {
                    const auto& results_for_one_query = all_results_vec[i];
                    for (size_t j = 0; j < k; ++j) {
                        if (j < results_for_one_query.size()) {
                            distances_ptr[i * k + j] = results_for_one_query[j].first;
                            labels_ptr[i * k + j] = static_cast<int64_t>(results_for_one_query[j].second);
                        } else {
                            // Fill remaining spots if fewer than k results were returned.
                            // This matches hnswlib's behavior.
                            distances_ptr[i * k + j] = std::numeric_limits<float>::infinity();
                            labels_ptr[i * k + j] = -1;  // Use -1 as the invalid label marker
                        }
                    }
                }

                // --- 5. Return a tuple of the two NumPy arrays ---
                return py::make_tuple(labels_arr, distances_arr);
            },
            py::arg("queries"), py::arg("k"), py::arg("ef_search"), py::arg("num_threads") = 0,
            py::arg("stats") = false,
            R"doc(
    Searches for the k-nearest neighbors for a batch of query vectors in parallel.

    Parameters:
    -----------
    queries : numpy.ndarray
        A 2D NumPy array of shape (num_queries, dim) containing the query vectors.
    k : int
        The number of nearest neighbors to search for.
    
    ef_search_param: int
        The ef_search num.

    num_threads : int, optional
        The number of threads to use for the search. If 0, it defaults to the
        number of hardware cores. (default: 0).
    
    stats: bool, optional

    Returns:
    --------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing two 2D NumPy arrays:
        - labels: An array of shape (num_queries, k) and dtype int64 with the IDs of the nearest neighbors.
        - distances: An array of shape (num_queries, k) and dtype float32 with the corresponding distances.
    )doc")

        .def(
            "add_items",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               size_t num_threads = 0) {
                // --- 1. Validate Input ---
                if (data.ndim() != 2) {
                    throw std::runtime_error("Data must be a 2-dimensional NumPy array of shape (num_items, dim)");
                }
                const size_t dim = self.getDim();
                if (data.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Data vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " + std::to_string(data.shape(1)));
                }

                if (data.shape(0) == 0) {
                    return;  // Nothing to add
                }

                // --- 2. Call the C++ parallel implementation ---
                std::span<const float> data_span(data.data(), data.size());
                self.addPoint_parallel(data_span, num_threads);
            },
            py::arg("data"), py::arg("num_threads") = 0,
            R"doc(
    Adds a batch of items to the index in parallel.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D NumPy array of shape (num_items, dim) containing the vectors to be added.
    num_threads : int, optional
        The number of threads to use for adding items. If 0, it defaults to the
        number of hardware cores. (default: 0)
    )doc")

        .def("get_buffer_manager_alloc_count", &HnswIndexType::getBufferManagerAllocCount,
             "Returns the current allocation count from the buffer manager.")

        .def(
            "get_index_info", &HnswIndexType::getIndexInfo,
            "Returns comprehensive information about the HNSW index including configuration, metadata, and statistics.")

        .def("get_dim", &HnswIndexType::getDim, "Returns the dimensionality of vectors managed by the index.")

        .def("reset_stats", &HnswIndexType::resetStats,
             "Resets live statistics counters like distance computations and search hops.")

        .def(
            "get_stats_string", [](HnswIndexType& self) { return self.getStats().toString(); },
            "Returns a formatted string of the current performance and graph structure statistics.")

        .def(
            "get_stats",
            [](HnswIndexType& self) -> py::dict {
                HNSWStats stats = self.getStats();
                py::dict result;

                result["dist_comps"] = stats.dist_comps.load();
                result["search_hops"] = stats.search_hops.load();
                result["num_levels"] = stats.num_levels;
                result["nodes_per_level"] = py::cast(stats.nodes_per_level);
                result["links_per_level"] = py::cast(stats.links_per_level);
                result["avg_neighbors_per_level"] = py::cast(stats.avg_neighbors_per_level);
                result["avg_neighbors_total"] = stats.avg_neighbors_total;

                return result;
            },
            R"doc(
    Returns a dictionary containing detailed performance and graph statistics.

    Returns:
    --------
    dict:
        A dictionary with the following keys:
        - 'dist_comps': Total number of distance computations.
        - 'search_hops': Total number of nodes visited during searches.
        - 'num_levels': The number of levels in the graph.
        - 'nodes_per_level': A list with the count of nodes on each level.
        - 'links_per_level': A list with the total count of links on each level.
        - 'avg_neighbors_per_level': A list with the average number of neighbors per node for each level.
        - 'avg_neighbors_total': The average number of neighbors per node across the entire graph.
    )doc")
        .def(
            "get_memory_usage",
            [](HnswIndexType& self) -> py::dict {
                py::dict result;
                result["process_rss_bytes"] = getCurrentRSS();
                return result;
            },
            R"doc(
    Returns a dictionary with current memory usage statistics.

    Returns:
    --------
    dict:
        A dictionary with the following keys:
        - 'process_rss_bytes': Total physical memory (RSS) used by the Python process.
    )doc");

    py::class_<IvfIndexType>(m, "IVFIndex")
        .def(py::init([](u64 max_elements, size_t dim, u32 num_clusters, u32 retrain_interval,
                 bool enable_prefetch, bool skip_recovery) {
             ensure_system_initialized();
                 return new IvfIndexType(max_elements, dim, num_clusters, retrain_interval,
                                         enable_prefetch, skip_recovery);
             }),
             py::arg("max_elements"),
             py::arg("dim"),
             py::arg("num_clusters") = IVF_DEFAULT_NUM_CLUSTERS,
             py::arg("retrain_interval") = 0,
             py::arg("enable_prefetch") = true,
             py::arg("skip_recovery") = false,
             "Initializes a flat IVF index with raw residual storage.")
        .def("flush", [](IvfIndexType& self) { self.flush(); }, "Flushes all dirty pages managed by Calico.")
        .def("was_recovered", &IvfIndexType::wasRecovered,
             "Returns True if the IVF index state was recovered from existing storage.")
        .def("is_trained", &IvfIndexType::isTrained, "Returns True if IVF centroids have been trained.")
        .def("get_dim", &IvfIndexType::getDim, "Returns the vector dimension.")
        .def("get_num_clusters", &IvfIndexType::getNumClusters, "Returns the IVF cluster count.")
        .def("get_count", &IvfIndexType::size, "Returns the current number of stored vectors.")
        .def("get_index_info", &IvfIndexType::getIndexInfo,
             "Returns detailed IVF metadata and runtime statistics.")
        .def(
            "train",
            [](IvfIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> training_data,
               u32 kmeans_iters) {
                if (training_data.ndim() != 2) {
                    throw std::runtime_error("Training data must be a 2-dimensional NumPy array.");
                }
                const size_t dim = self.getDim();
                if (training_data.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Training data has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(training_data.shape(1)));
                }
                self.train(training_data.data(), training_data.shape(0), kmeans_iters);
            },
            py::arg("training_data"), py::arg("kmeans_iters") = IVF_DEFAULT_KMEANS_ITERS,
            "Trains IVF centroids on a representative sample of vectors.")
        .def(
            "add_points",
            [](IvfIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items,
               size_t num_threads) {
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, dim).");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(items.shape(1)));
                }
                const size_t count = items.shape(0);
                std::vector<u32> ids(count);
                const u64 start_id = self.size();
                for (size_t i = 0; i < count; ++i) {
                    ids[i] = static_cast<u32>(start_id + i);
                }
                self.addPoints(items.data(), ids.data(), count, num_threads);
            },
            py::arg("items"), py::arg("num_threads") = 0,
            "Adds a batch of vectors to the IVF index using sequential IDs.")
        .def(
            "add_items",
            [](IvfIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items,
               size_t num_threads) {
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, dim).");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(items.shape(1)));
                }
                const size_t count = items.shape(0);
                std::vector<u32> ids(count);
                const u64 start_id = self.size();
                for (size_t i = 0; i < count; ++i) {
                    ids[i] = static_cast<u32>(start_id + i);
                }
                self.addPoints(items.data(), ids.data(), count, num_threads);
            },
            py::arg("items"), py::arg("num_threads") = 0,
            "Alias for add_points().")
        .def(
            "search_knn",
            [](IvfIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t k,
               size_t nprobe, bool stats) -> py::object {
                if (query.ndim() != 1) {
                    throw std::runtime_error("Query must be a 1-dimensional NumPy array.");
                }
                const size_t dim = self.getDim();
                if (query.shape(0) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(query.shape(0)));
                }
                auto result_vec = self.search(query.data(), k, nprobe, stats);
                py::list labels;
                py::list distances;
                for (const auto& pair : result_vec) {
                    distances.append(pair.first);
                    labels.append(pair.second);
                }
                return py::make_tuple(py::array(labels), py::array(distances));
            },
            py::arg("query"), py::arg("k"), py::arg("nprobe"), py::arg("stats") = false,
            "Searches the IVF index for the k nearest neighbors.")
        .def(
            "search_knn_parallel",
            [](IvfIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries,
               size_t k, size_t nprobe, size_t num_threads, bool stats) -> py::object {
                if (queries.ndim() != 2) {
                    throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim).");
                }
                const size_t dim = self.getDim();
                if (queries.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(queries.shape(1)));
                }
                if (k == 0) {
                    throw std::runtime_error("k must be > 0");
                }
                const size_t num_queries = queries.shape(0);
                std::span<const float> query_span(queries.data(), queries.size());
                auto all_results_vec = self.searchBatch(query_span, k, nprobe, num_threads, stats);

                std::vector<ssize_t> result_shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
                py::array_t<int64_t> labels_arr(result_shape);
                py::array_t<float> distances_arr(result_shape);
                auto* labels_ptr = labels_arr.mutable_data();
                auto* distances_ptr = distances_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    const auto& per_query = all_results_vec[i];
                    for (size_t j = 0; j < k; ++j) {
                        if (j < per_query.size()) {
                            distances_ptr[i * k + j] = per_query[j].first;
                            labels_ptr[i * k + j] = static_cast<int64_t>(per_query[j].second);
                        } else {
                            distances_ptr[i * k + j] = std::numeric_limits<float>::infinity();
                            labels_ptr[i * k + j] = -1;
                        }
                    }
                }
                return py::make_tuple(labels_arr, distances_arr);
            },
            py::arg("queries"), py::arg("k"), py::arg("nprobe"), py::arg("num_threads") = 0,
            py::arg("stats") = false,
            "Searches a batch of query vectors in parallel.")
        .def("reset_stats", &IvfIndexType::resetStats, "Resets IVF runtime statistics.")
        .def(
            "get_stats",
            [](IvfIndexType& self) -> py::dict {
                IVFStats stats = self.getStats();
                py::dict result;
                result["dist_comps"] = stats.dist_comps.load();
                result["lists_probed"] = stats.lists_probed.load();
                result["vectors_scanned"] = stats.vectors_scanned.load();
                result["num_clusters"] = stats.num_clusters;
                result["list_sizes"] = py::cast(stats.list_sizes);
                result["avg_list_size"] = stats.avg_list_size;
                return result;
            },
            "Returns IVF runtime and structural statistics.")
        .def(
            "get_stats_string", [](IvfIndexType& self) { return self.getStats().toString(); },
            "Returns a formatted IVF statistics summary.")
        .def(
            "get_memory_usage",
            [](IvfIndexType& self) -> py::dict {
                py::dict result;
                result["process_rss_bytes"] = getCurrentRSS();
                return result;
            },
            "Returns process RSS memory usage.");

    py::class_<PdxIndexType>(m, "PDXIndex")
        .def(py::init([](u64 max_elements, size_t dim, u32 num_clusters, u32 retrain_interval,
                 bool enable_prefetch, bool skip_recovery, float sampling_fraction,
                 u32 transform_seed) {
             ensure_system_initialized();
                 return new PdxIndexType(max_elements, dim, num_clusters, retrain_interval,
                                         enable_prefetch, skip_recovery, sampling_fraction,
                                         transform_seed);
             }),
             py::arg("max_elements"),
             py::arg("dim"),
             py::arg("num_clusters") = PDX_DEFAULT_NUM_CLUSTERS,
             py::arg("retrain_interval") = 0,
             py::arg("enable_prefetch") = true,
             py::arg("skip_recovery") = false,
             py::arg("sampling_fraction") = 0.0f,
             py::arg("transform_seed") = 42,
             "Initializes a Calico-backed PDX index with transposed per-cluster storage.")
        .def("flush", [](PdxIndexType& self) { self.flush(); }, "Flushes all dirty pages managed by Calico.")
        .def("was_recovered", &PdxIndexType::wasRecovered,
             "Returns True if the PDX index state was recovered from existing storage.")
        .def("is_trained", &PdxIndexType::isTrained, "Returns True if PDX centroids have been trained.")
        .def("get_dim", &PdxIndexType::getDim, "Returns the vector dimension.")
        .def("get_num_clusters", &PdxIndexType::getNumClusters, "Returns the PDX cluster count.")
        .def("get_count", &PdxIndexType::size, "Returns the current number of stored vectors.")
        .def("get_sampling_fraction", &PdxIndexType::getSamplingFraction,
             "Returns the effective build-time clustering sampling fraction.")
        .def("get_index_info", &PdxIndexType::getIndexInfo,
             "Returns detailed PDX metadata and runtime statistics.")
        .def(
            "train",
            [](PdxIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> training_data,
               u32 kmeans_iters) {
                if (training_data.ndim() != 2) {
                    throw std::runtime_error("Training data must be a 2-dimensional NumPy array.");
                }
                const size_t dim = self.getDim();
                if (training_data.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Training data has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(training_data.shape(1)));
                }
                self.train(training_data.data(), training_data.shape(0), kmeans_iters);
            },
            py::arg("training_data"), py::arg("kmeans_iters") = PDX_DEFAULT_KMEANS_ITERS,
            "Trains PDX centroids on transformed vectors, optionally with clustering-time sampling.")
        .def(
            "add_points",
            [](PdxIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items,
               size_t num_threads) {
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, dim).");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(items.shape(1)));
                }
                const size_t count = items.shape(0);
                std::vector<u32> ids(count);
                const u64 start_id = self.size();
                for (size_t i = 0; i < count; ++i) {
                    ids[i] = static_cast<u32>(start_id + i);
                }
                self.addPoints(items.data(), ids.data(), count, num_threads);
            },
            py::arg("items"), py::arg("num_threads") = 0,
            "Adds a batch of vectors to the PDX index using sequential IDs.")
        .def(
            "add_items",
            [](PdxIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items,
               size_t num_threads) {
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, dim).");
                }
                const size_t dim = self.getDim();
                if (items.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(items.shape(1)));
                }
                const size_t count = items.shape(0);
                std::vector<u32> ids(count);
                const u64 start_id = self.size();
                for (size_t i = 0; i < count; ++i) {
                    ids[i] = static_cast<u32>(start_id + i);
                }
                self.addPoints(items.data(), ids.data(), count, num_threads);
            },
            py::arg("items"), py::arg("num_threads") = 0,
            "Alias for add_points().")
        .def(
            "search_knn",
            [](PdxIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t k,
               size_t nprobe, bool stats) -> py::object {
                if (query.ndim() != 1) {
                    throw std::runtime_error("Query must be a 1-dimensional NumPy array.");
                }
                const size_t dim = self.getDim();
                if (query.shape(0) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(query.shape(0)));
                }
                auto result_vec = self.search(query.data(), k, nprobe, stats);
                py::list labels;
                py::list distances;
                for (const auto& pair : result_vec) {
                    distances.append(pair.first);
                    labels.append(pair.second);
                }
                return py::make_tuple(py::array(labels), py::array(distances));
            },
            py::arg("query"), py::arg("k"), py::arg("nprobe"), py::arg("stats") = false,
            "Searches the PDX index for the k nearest neighbors.")
        .def(
            "search_knn_parallel",
            [](PdxIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries,
               size_t k, size_t nprobe, size_t num_threads, bool stats) -> py::object {
                if (queries.ndim() != 2) {
                    throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim).");
                }
                const size_t dim = self.getDim();
                if (queries.shape(1) != static_cast<py::ssize_t>(dim)) {
                    throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                             std::to_string(dim) + ", but got " +
                                             std::to_string(queries.shape(1)));
                }
                if (k == 0) {
                    throw std::runtime_error("k must be > 0");
                }
                const size_t num_queries = queries.shape(0);
                std::span<const float> query_span(queries.data(), queries.size());
                auto all_results_vec = self.searchBatch(query_span, k, nprobe, num_threads, stats);

                std::vector<ssize_t> result_shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
                py::array_t<int64_t> labels_arr(result_shape);
                py::array_t<float> distances_arr(result_shape);
                auto* labels_ptr = labels_arr.mutable_data();
                auto* distances_ptr = distances_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    const auto& per_query = all_results_vec[i];
                    for (size_t j = 0; j < k; ++j) {
                        if (j < per_query.size()) {
                            distances_ptr[i * k + j] = per_query[j].first;
                            labels_ptr[i * k + j] = static_cast<int64_t>(per_query[j].second);
                        } else {
                            distances_ptr[i * k + j] = std::numeric_limits<float>::infinity();
                            labels_ptr[i * k + j] = -1;
                        }
                    }
                }
                return py::make_tuple(labels_arr, distances_arr);
            },
            py::arg("queries"), py::arg("k"), py::arg("nprobe"), py::arg("num_threads") = 0,
            py::arg("stats") = false,
            "Searches a batch of query vectors in parallel.")
        .def("reset_stats", &PdxIndexType::resetStats, "Resets PDX runtime statistics.")
        .def(
            "get_stats",
            [](PdxIndexType& self) -> py::dict {
                PDXStats stats = self.getStats();
                py::dict result;
                result["dist_comps"] = stats.dist_comps.load();
                result["lists_probed"] = stats.lists_probed.load();
                result["vectors_scanned"] = stats.vectors_scanned.load();
                result["vectors_pruned"] = stats.vectors_pruned.load();
                const u64 horizontal_selected_page_samples = stats.horizontal_selected_page_samples.load();
                if (horizontal_selected_page_samples > 0) {
                    result["horizontal_selected_page_samples"] = horizontal_selected_page_samples;
                    result["horizontal_selected_page_transitions"] = stats.horizontal_selected_page_transitions.load();
                    result["horizontal_selected_logical_same_page"] = stats.horizontal_selected_logical_same_page.load();
                    result["horizontal_selected_payload_same_page"] = stats.horizontal_selected_payload_same_page.load();
                    result["horizontal_selected_logical_adjacent_page"] = stats.horizontal_selected_logical_adjacent_page.load();
                    result["horizontal_selected_payload_adjacent_page"] = stats.horizontal_selected_payload_adjacent_page.load();
                    result["horizontal_selected_logical_same_hugepage"] = stats.horizontal_selected_logical_same_hugepage.load();
                    result["horizontal_selected_payload_same_hugepage"] = stats.horizontal_selected_payload_same_hugepage.load();
                    result["horizontal_selected_logical_abs_delta_sum"] = stats.horizontal_selected_logical_abs_delta_sum.load();
                    result["horizontal_selected_payload_abs_delta_sum"] = stats.horizontal_selected_payload_abs_delta_sum.load();
                    result["horizontal_selected_payload_forward_transitions"] = stats.horizontal_selected_payload_forward_transitions.load();
                    result["horizontal_selected_payload_backward_transitions"] = stats.horizontal_selected_payload_backward_transitions.load();
                }
                const u64 replay_pid_stream_queries = stats.replay_pid_stream_queries.load();
                if (replay_pid_stream_queries > 0) {
                    result["replay_pid_stream_queries"] = replay_pid_stream_queries;
                    result["replay_pid_stream_samples"] = stats.replay_pid_stream_samples.load();
                    result["replay_pid_stream_resident_samples"] = stats.replay_pid_stream_resident_samples.load();
                    result["replay_pid_stream_calico_live_ns"] = stats.replay_pid_stream_calico_live_ns.load();
                    result["replay_pid_stream_calico_live_touches"] = stats.replay_pid_stream_calico_live_touches.load();
                    result["replay_pid_stream_calico_live_checksum"] = stats.replay_pid_stream_calico_live_checksum.load();
                    result["replay_pid_stream_calico_bypass_ns"] = stats.replay_pid_stream_calico_bypass_ns.load();
                    result["replay_pid_stream_calico_bypass_touches"] = stats.replay_pid_stream_calico_bypass_touches.load();
                    result["replay_pid_stream_calico_bypass_checksum"] = stats.replay_pid_stream_calico_bypass_checksum.load();
                    result["replay_pid_stream_vmcache_ns"] = stats.replay_pid_stream_vmcache_ns.load();
                    result["replay_pid_stream_vmcache_touches"] = stats.replay_pid_stream_vmcache_touches.load();
                    result["replay_pid_stream_vmcache_checksum"] = stats.replay_pid_stream_vmcache_checksum.load();
                }
                result["num_clusters"] = stats.num_clusters;
                result["cluster_sizes"] = py::cast(stats.cluster_sizes);
                result["avg_cluster_size"] = stats.avg_cluster_size;
                return result;
            },
            "Returns PDX runtime and structural statistics.")
        .def(
            "get_stats_string", [](PdxIndexType& self) { return self.getStats().toString(); },
            "Returns a formatted PDX statistics summary.")
        .def(
            "get_memory_usage",
            [](PdxIndexType& self) -> py::dict {
                py::dict result;
                result["process_rss_bytes"] = getCurrentRSS();
                return result;
            },
            "Returns process RSS memory usage.");

    // =========================================================================================
    // DiskANN Bindings Start
    // =========================================================================================

    py::class_<DiskANNBase::BuildParams>(m, "BuildParams")
        .def(py::init<>())
        // R_max_degree is now configured at index creation, so it's removed from BuildParams.
        // It's still a member of the C++ struct, but we don't expose it here to avoid confusion.
        .def_readwrite("L_build", &DiskANNBase::BuildParams::L_build)
        .def_readwrite("alpha", &DiskANNBase::BuildParams::alpha)
        .def_readwrite("num_threads", &DiskANNBase::BuildParams::num_threads);

    py::class_<DiskANNBase::SearchParams>(m, "SearchParams")
        .def(py::init<size_t>(), py::arg("L_search"))
        .def_readwrite("L_search", &DiskANNBase::SearchParams::L_search)
        .def_readwrite("beam_width", &DiskANNBase::SearchParams::beam_width);

    py::class_<DiskANNBase, std::unique_ptr<DiskANNBase>>(m, "DiskANN")
        .def(py::init([](size_t dimensions, uint64_t max_elements, size_t R_max_degree, bool is_dynamic) {
                 ensure_system_initialized();
                 // *** MODIFIED HERE ***
                 // Pass the R_max_degree to the factory function.
                 return create_index(dimensions, max_elements, R_max_degree, is_dynamic);
             }),
             py::arg("dimensions"), py::arg("max_elements"), py::arg("R_max_degree") = 64,
             py::arg("is_dynamic") = false,
             "Initializes a new, empty Vamana index with a specific dimension, capacity, and max graph degree (R).")

        .def_property_readonly("dimensions", &DiskANNBase::get_dimensions, "Returns the dimension of the index.")
        // *** ADDED HERE ***
        .def_property_readonly("R", &DiskANNBase::get_R, "Returns the configured max graph degree (R) of the index.")

        .def(
            "build",  // Renamed from "add_items" to "build" for clarity, as it's a bulk-load operation.
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               const std::vector<std::vector<uint32_t>>& tags, const DiskANNBase::BuildParams& params) {
                if (data.ndim() != 2 || (size_t)data.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Data has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()) + ", but got " +
                                                std::to_string(data.shape(1)));
                }
                size_t num_points = data.shape(0);
                if (tags.size() != num_points) {
                    throw std::invalid_argument("The number of tag lists must match the number of data points.");
                }
                self.build(data.data(), tags, num_points, params);
            },
            py::arg("data"), py::arg("tags"), py::arg("params"),
            "Builds the index from a NumPy array of vectors and a list of tag lists. This is a destructive operation "
            "and can only be called once on a static index.")

        .def(
            "search",  // Renamed for consistency
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t K,
               const DiskANNBase::SearchParams& params) -> py::tuple {
                if (query.ndim() != 1 || (size_t)query.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Query has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }

                auto result_vec = self.search(query.data(), K, params);

                py::array_t<uint32_t> labels(result_vec.size());
                py::array_t<float> distances(result_vec.size());

                auto labels_ptr = labels.mutable_data();
                auto dists_ptr = distances.mutable_data();

                for (size_t i = 0; i < result_vec.size(); ++i) {
                    labels_ptr[i] = result_vec[i].second;
                    dists_ptr[i] = result_vec[i].first;
                }

                return py::make_tuple(labels, distances);
            },
            py::arg("query"), py::arg("K"), py::arg("params"),
            "Performs a K-Nearest Neighbor search. Returns a tuple of (labels, distances).")

        .def(
            "search_with_filter",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> query,
               uint32_t filter_label, size_t K, const DiskANNBase::SearchParams& params) -> py::tuple {
                if (query.ndim() != 1 || (size_t)query.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Query has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }

                auto result_vec = self.search_with_filter(query.data(), filter_label, K, params);

                py::array_t<uint32_t> labels(result_vec.size());
                py::array_t<float> distances(result_vec.size());

                auto labels_ptr = labels.mutable_data();
                auto dists_ptr = distances.mutable_data();

                for (size_t i = 0; i < result_vec.size(); ++i) {
                    labels_ptr[i] = result_vec[i].second;
                    dists_ptr[i] = result_vec[i].first;
                }

                return py::make_tuple(labels, distances);
            },
            py::arg("query"), py::arg("filter_label"), py::arg("K"), py::arg("params"),
            "Performs a filtered K-Nearest Neighbor search. Returns a tuple of (labels, distances).")

        .def(
            "search_knn_parallel",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t K,
               const DiskANNBase::SearchParams& params, size_t num_threads) -> py::tuple {
                if (queries.ndim() != 2 || (size_t)queries.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Queries must be a 2D NumPy array with shape (num_queries, dim)");
                }
                size_t num_queries = queries.shape(0);
                if (K == 0) throw std::invalid_argument("K must be > 0");

                auto all_results_vec = self.search_parallel(queries.data(), num_queries, K, params, num_threads);

                // --- Convert result to a pair of NumPy arrays ---
                py::array_t<uint32_t> labels_arr({num_queries, K});
                py::array_t<float> dists_arr({num_queries, K});

                auto labels_ptr = labels_arr.mutable_data();
                auto dists_ptr = dists_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        if (j < all_results_vec[i].size()) {
                            labels_ptr[i * K + j] = all_results_vec[i][j].second;
                            dists_ptr[i * K + j] = all_results_vec[i][j].first;
                        } else {
                            labels_ptr[i * K + j] = -1;  // Sentinel for invalid
                            dists_ptr[i * K + j] = std::numeric_limits<float>::infinity();
                        }
                    }
                }
                return py::make_tuple(labels_arr, dists_arr);
            },
            py::arg("queries"), py::arg("K"), py::arg("params"), py::arg("num_threads") = 0,
            "Performs a parallel K-NN search for a batch of queries.")

        .def(
            "search_with_filter_parallel",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries,
               uint32_t filter_label, size_t K, const DiskANNBase::SearchParams& params,
               size_t num_threads) -> py::tuple {
                if (queries.ndim() != 2 || (size_t)queries.shape(1) != self.get_dimensions()) {
                    throw std::invalid_argument("Queries must be a 2D NumPy array with shape (num_queries, dim)");
                }
                size_t num_queries = queries.shape(0);
                if (K == 0) throw std::invalid_argument("K must be > 0");

                auto all_results_vec =
                    self.search_with_filter_parallel(queries.data(), num_queries, filter_label, K, params, num_threads);

                // --- Convert result to a pair of NumPy arrays (same logic as above) ---
                py::array_t<uint32_t> labels_arr({num_queries, K});
                py::array_t<float> dists_arr({num_queries, K});

                auto labels_ptr = labels_arr.mutable_data();
                auto dists_ptr = dists_arr.mutable_data();

                for (size_t i = 0; i < num_queries; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        if (j < all_results_vec[i].size()) {
                            labels_ptr[i * K + j] = all_results_vec[i][j].second;
                            dists_ptr[i * K + j] = all_results_vec[i][j].first;
                        } else {
                            labels_ptr[i * K + j] = -1;  // Sentinel for invalid
                            dists_ptr[i * K + j] = std::numeric_limits<float>::infinity();
                        }
                    }
                }
                return py::make_tuple(labels_arr, dists_arr);
            },
            py::arg("queries"), py::arg("filter_label"), py::arg("K"), py::arg("params"), py::arg("num_threads") = 0,
            "Performs a parallel filtered K-NN search for a batch of queries.")

        .def(
            "insert_point",
            [](DiskANNBase& self, py::array_t<float, py::array::c_style | py::array::forcecast> point,
               const std::vector<uint32_t>& tags, uint32_t external_id) {
                if (point.ndim() != 1 || (size_t)point.shape(0) != self.get_dimensions()) {
                    throw std::invalid_argument("Point has incorrect dimension. Expected " +
                                                std::to_string(self.get_dimensions()));
                }
                self.insert_point(point.data(), tags, external_id);
            },
            py::arg("point"), py::arg("tags"), py::arg("external_id"), "Inserts a single point into a dynamic index.")

        .def("lazy_delete", &DiskANNBase::lazy_delete, py::arg("external_id"),
             "Marks a point for deletion in a dynamic index. The point is removed from search results immediately.")

        .def("consolidate_deletes", &DiskANNBase::consolidate_deletes, py::arg("params"),
             "Repairs the graph structure around deleted nodes. This is a potentially long-running operation.");
}
