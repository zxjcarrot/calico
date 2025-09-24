#include <pybind11/functional.h>  // Needed for py::function
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"
#include "hnsw.hpp"

namespace py = pybind11;

// --- Forward declare init/shutdown functions from calico.cpp ---
void initialize_system();
void shutdown_system();

// --- Automatic Initializer ---
struct SystemInitializer {
    SystemInitializer() {
        std::cout << "Calico module loading: Initializing system..." << std::endl;
        initialize_system();
        std::cout << "Calico system initialized." << std::endl;
    }
    ~SystemInitializer() {
        std::cout << "Calico module unloading: Shutting down system..." << std::endl;
        shutdown_system();
    }
};
static SystemInitializer initializer;

// --- Define the specific HNSW instance we will bind ---
// For this binding, we will use a fixed dimension of 128 to match the explicit
// template instantiation in hnsw.cpp. This is a common approach for high-performance bindings.
constexpr size_t HNSW_DIM = 128;
using HnswIndexType = HNSW<HNSW_DIM, hnsw_distance::SIMDAcceleratedL2>;

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

    py::class_<HnswIndexType>(m, "HnswIndex")
        .def(py::init<u64>(), py::arg("max_elements"),
             "Initializes a new, empty HNSW index with a given maximum capacity.")

        .def(
            "add_points",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> items) {
                // Check input array dimensions
                if (items.ndim() != 2) {
                    throw std::runtime_error("Input array must be 2-dimensional (n_samples, n_features)");
                }
                if (items.shape(1) != HNSW_DIM) {
                    throw std::runtime_error("Input array has incorrect dimension. Expected " +
                                             std::to_string(HNSW_DIM) + ", but got " + std::to_string(items.shape(1)));
                }

                size_t num_points = items.shape(0);
                py::buffer_info buf = items.request();
                float* data_ptr = static_cast<float*>(buf.ptr);

                // Add points one by one
                for (size_t i = 0; i < num_points; ++i) {
                    u32 node_id_out;
                    // Get a pointer to the start of the i-th vector
                    const float* point = data_ptr + i * HNSW_DIM;
                    self.addPoint(point, node_id_out);
                    // For simplicity, we assume the returned node_id matches the item index i.
                    // A more complex implementation could store a mapping if needed.
                }
            },
            py::arg("items"), "Adds a batch of points from a NumPy array to the index.")

        .def(
            "search_knn",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> query, size_t k,
               size_t ef_search_param) -> py::object {
                // Check input query vector dimensions
                if (query.ndim() != 1) {
                    throw std::runtime_error("Query must be a 1-dimensional NumPy array");
                }
                if (query.shape(0) != HNSW_DIM) {
                    throw std::runtime_error("Query vector has incorrect dimension. Expected " +
                                             std::to_string(HNSW_DIM) + ", but got " + std::to_string(query.shape(0)));
                }

                const float* query_ptr = query.data();
                auto result_vec = self.searchKnn(query_ptr, k, ef_search_param);

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
            py::arg("query"), py::arg("k"), py::arg("ef_search"),
            "Searches for the k-nearest neighbors for a given query vector. Returns (labels, distances).")

        .def(
            "search_knn_parallel",
            [](HnswIndexType& self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, size_t k,
               size_t ef_search_param, size_t num_threads = 0) -> py::object {
                // --- 1. Validate Input ---
                if (queries.ndim() != 2) {
                    throw std::runtime_error("Queries must be a 2-dimensional NumPy array of shape (num_queries, dim)");
                }
                if (queries.shape(1) != HNSW_DIM) {
                    throw std::runtime_error("Query vectors have incorrect dimension. Expected " +
                                             std::to_string(HNSW_DIM) + ", but got " +
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
                auto all_results_vec = self.searchKnn_parallel(query_span, k, ef_search_param, num_threads);

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
        number of hardware cores. (default: 0)

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
                if (data.shape(1) != HNSW_DIM) {
                    throw std::runtime_error("Data vectors have incorrect dimension. Expected " +
                                             std::to_string(HNSW_DIM) + ", but got " + std::to_string(data.shape(1)));
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
    )doc");
}