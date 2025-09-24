import numpy as np
import time
import os
import hnswlib

import calico_py


def read_fvecs(filename, c_contiguous=True):
    """
    Reads a .fvecs file.
    """
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert fv.size % (dim + 1) == 0
    fv = fv.reshape(-1, dim + 1)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector dimensions in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_ivecs(filename):
    """
    Reads a .ivecs file.
    """
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert iv.size % (dim + 1) == 0
    iv = iv.reshape(-1, dim + 1)
    if not all(iv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector dimensions in " + filename)
    iv = iv[:, 1:]
    return iv


def calculate_recall(ground_truth, result_labels, k):
    """
    Calculates recall@k.
    """
    correct = 0
    total_queries = len(ground_truth)
    for i in range(total_queries):
        gt_set = set(ground_truth[i][:k])
        res_set = set(result_labels[i])
        correct += len(gt_set.intersection(res_set))

    return correct / (total_queries * k)


def benchmark_calico(X_base, X_query, ground_truth, params):
    """
    Benchmarks your calico_py HNSW implementation.
    """
    print("\n--- Benchmarking calico_py HNSW ---")

    num_elements, dim = X_base.shape
    k = params["k"]
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)

    print(
        f"Building index with M={params['M']}, ef_construction={params['ef_construction']}..."
    )
    start_time = time.time()
    index_calico = calico_py.HnswIndex(max_elements=num_elements)
    index_calico.add_items(X_base_float32)

    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f} seconds")
    ef_search = params.get("ef_search", 50)
    print(f"Querying for k={k} nearest neighbors with ef_search={ef_search}...")

    start_time = time.time()

    labels, distances = index_calico.search_knn_parallel(X_query_float32, k, ef_search, 0)

    search_time = time.time() - start_time
    qps = len(X_query) / search_time
    print(
        f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)"
    )

    recall = calculate_recall(ground_truth, labels, k)
    print(f"Recall@{k}: {recall:.4f}")

    return build_time, search_time, recall


def benchmark_hnswlib(X_base, X_query, ground_truth, params):
    """
    Benchmarks the hnswlib library for comparison.
    """
    print("\n--- Benchmarking hnswlib ---")

    num_elements, dim = X_base.shape
    k = params["k"]

    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    print(
        f"Building index with M={params['M']}, ef_construction={params['ef_construction']}..."
    )
    start_time = time.time()

    p = hnswlib.Index(space="l2", dim=dim)  # 'l2' for Euclidean distance
    p.init_index(
        max_elements=num_elements,
        ef_construction=params["ef_construction"],
        M=params["M"],
    )

    p.add_items(X_base_float32, num_threads=-1)

    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f} seconds")

    ef_search = params.get("ef_search", 50)
    p.set_ef(ef_search)
    print(f"Querying for k={k} nearest neighbors with ef_search={ef_search}...")

    start_time = time.time()

    labels, distances = p.knn_query(X_query_float32, k=k, num_threads=-1)

    search_time = time.time() - start_time
    qps = len(X_query) / search_time
    print(
        f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)"
    )

    recall = calculate_recall(ground_truth, labels, k)
    print(f"Recall@{k}: {recall:.4f}")

    return build_time, search_time, recall


if __name__ == "__main__":
    
    dataset_name = "sift10m"
    
    if dataset_name == "sift1m":
    
        DATA_DIR = "./data/sift1m"
        BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
        QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")
        GT_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

        if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
            print("Error: SIFT1M data files not found.")
            print(f"Please download and place them in the '{DATA_DIR}' directory.")
            exit(1)

        hnsw_params = {"M": 16, "ef_construction": 200, "k": 10, "ef_search": 50}

        print("Loading SIFT1M dataset...")
        X_base = read_fvecs(BASE_FILE)
        X_query = read_fvecs(QUERY_FILE)
        ground_truth = read_ivecs(GT_FILE)
        print(f"Base vectors shape: {X_base.shape}")
        print(f"Query vectors shape: {X_query.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
    else: #sift10m
        import h5py
        mat_path = "./data/SIFT10M/SIFT10Mfeatures.mat"
        f = h5py.File(mat_path, 'r')
        vector = f['fea']
        
        num_base = 10_000_000
        num_queries = 100_000
        
        X_base = vector[:num_base, :]
    
        X_query = vector[num_base : num_base + num_queries, :]
        
        
        X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
        X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
        import faiss
        index_flat = faiss.IndexFlatL2(X_base_float32.shape[1]) 

        index_flat.add(X_base_float32)

        k = 10
        start_time = time.time()
        D_flat, ground_truth = index_flat.search(X_query_float32, k) 
        end_time = time.time()
        print(f"Flat search took: {end_time - start_time:.4f} seconds")
        

    calico_build, calico_search, calico_recall = benchmark_calico(
        X_base, X_query, ground_truth, hnsw_params
    )
    hnswlib_build, hnswlib_search, hnswlib_recall = benchmark_hnswlib(
        X_base, X_query, ground_truth, hnsw_params
    )

    print("\n\n--- Benchmark Summary ---")
    print(
        f"Parameters: M={hnsw_params['M']}, ef_construction={hnsw_params['ef_construction']}, k={hnsw_params['k']}"
    )
    print("-" * 25)
    print(
        f"| Implementation | Build Time (s) | Search Time (s) | QPS    | Recall@{hnsw_params['k']} |"
    )
    print("-" * 25)
    calico_qps = len(X_query) / calico_search
    hnswlib_qps = len(X_query) / hnswlib_search
    print(
        f"| calico_py      | {calico_build:<14.4f} | {calico_search:<15.4f} | {calico_qps:<6.2f} | {calico_recall:<8.4f} |"
    )
    print(
        f"| hnswlib        | {hnswlib_build:<14.4f} | {hnswlib_search:<15.4f} | {hnswlib_qps:<6.2f} | {hnswlib_recall:<8.4f} |"
    )
    print("-" * 25)
