# benchmark_diskann.py

import numpy as np
import time
import os
import sys
import shutil
import argparse
import signal

# --- Dependency Imports ---
import calico_py
try:
    import diskannpy
except ImportError:
    print("Error: 'diskannpy' is not installed.", file=sys.stderr)
    sys.exit(1)

# --- Utility Functions (unchanged) ---
def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert fv.size % (dim + 1) == 0
    fv = fv.reshape(-1, dim + 1)[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0: return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert iv.size % (dim + 1) == 0
    return iv.reshape(-1, dim + 1)[:, 1:]

# --- Main Profiling Functions ---
def calculate_recall(ground_truth, result_labels, k):
    correct = 0
    total_queries = len(ground_truth)
    for i in range(total_queries):
        gt_set = set(ground_truth[i][:k])
        # Ensure result_labels[i] is iterable and contains numeric types
        res_set = set(int(label) for label in result_labels[i] if label != -1)
        correct += len(gt_set.intersection(res_set))
    return correct / (total_queries * k)

def run_calico_full_profile(args, params):
    """
    Performs a full build-then-search profile run for Calico in a single process.
    Pauses before build and before search.
    """
    print("\n--- Preparing Calico Full Profile Run ---")
    
    # --- Part 1: Build Preparation ---
    print("Loading base data for build...")
    X_base = read_fvecs(args.base_file)
    num_elements, dim = X_base.shape
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    dummy_tags = [[0]] * num_elements
    
    index = calico_py.DiskANN(
        dimensions=dim,
        max_elements=num_elements,
        R_max_degree=params["R"],
        is_dynamic=False,
    )
    build_params = calico_py.BuildParams()
    build_params.L_build = params["L_build"]
    build_params.alpha = params["alpha"]
    build_params.num_threads = args.threads
    
    # --- First Pause: Before Build ---
    pid = os.getpid()
    print(f"READY FOR BUILD. Script is paused. PID: {pid}", flush=True)
    if args.pause:
        os.kill(pid, signal.SIGSTOP)
    print("Resumed. Starting build...", flush=True)

    # --- Profiled Section 1: Build ---
    start_time = time.time()
    index.build(X_base_float32, dummy_tags, build_params)
    build_time = time.time() - start_time
    print(f"Calico index build finished in {build_time:.4f}s.")

    # --- Part 2: Search Preparation ---
    print("Loading query data for search...")
    X_query = read_fvecs(args.query_file)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    # Create a dummy query and search for it. A single search will touch many pages.
    # Running it a few times with different queries can help warm up more of the graph.
    # Alternatively, you can write a dedicated C++ function to just iterate
    # through the raw buffer. But a dummy search is often easiest.
    dummy_search_params = calico_py.SearchParams(L_search=params["L_search"])
    for i in range(10): # Run a few dummy searches to touch different parts of the graph
        index.search(X_query_float32[i], 1, dummy_search_params) # Note: search_knn is the single-threaded version

    # Prepare search data
   
    search_params = calico_py.SearchParams(L_search=params["L_search"])
    k = params["k"]

    # --- Second Pause: Before Search ---
    print("BUILD COMPLETE. Pausing for search profiling.", flush=True)
    if args.pause:
        os.kill(pid, signal.SIGSTOP)
    print("Resumed. Starting search...", flush=True)

    # --- Profiled Section 2: Search ---
    start_time = time.time()
    labels, distances = index.search_knn_parallel(X_query_float32, params["k"], search_params, args.threads)
    search_time = time.time() - start_time
    ground_truth = read_ivecs(args.gt_file)
    recall = calculate_recall(ground_truth, labels, k)
    qps = len(X_query_float32) / search_time if search_time > 0 else float("inf")
    print(f"Calico search finished in {search_time:.4f}s ({qps:.2f} QPS)")
    print(f"Recall@{k}: {recall:.4f}")
    print("Profiling run complete.")


def run_official_full_profile(args, params):
    """
    Performs a full build-then-search profile run for Official DiskANN in memory.
    This keeps the workflow consistent with Calico.
    """
    print("\n--- Preparing Official DiskANN Full Profile Run ---")
    
    # --- Part 1: Build Preparation ---
    print("Loading base data for build...")
    X_base = read_fvecs(args.base_file)
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)

    # --- First Pause: Before Build ---
    pid = os.getpid()
    print(f"READY FOR BUILD. Script is paused. PID: {pid}", flush=True)
    if args.pause:
        os.kill(pid, signal.SIGSTOP)
    print("Resumed. Starting build...", flush=True)
    index_dir = './diskann_index'
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    # --- Profiled Section 1: Build (In-Memory) ---
    start_time = time.time()
    
    # Note: We build in-memory to mirror the Calico workflow
    # diskannpy.build_memory_index(
    #     data=X_base_float32,
    #     distance_metric="l2",
    #     vector_dtype=np.float32,
    #     index_directory=index_dir,
    #     index_prefix="sift_diskann",
    #     complexity=params["L_build"],
    #     graph_degree=params["R"],
    #     alpha=params["alpha"],
    #     num_threads=args.threads,
    #     use_pq_build=False,
    #     use_opq=False,
    # )
    diskannpy.build_disk_index(
        data=X_base_float32,
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        complexity=params["L_build"],
        graph_degree=params["R"],
        num_threads=args.threads,
        pq_disk_bytes=256,
        search_memory_maximum=16,
        build_memory_maximum=16,
    )
    build_time = time.time() - start_time
    print(f"Official DiskANN in-memory build finished in {build_time:.4f}s.")
    print("Loading Official DiskANN index from disk...")
    # index = diskannpy.StaticMemoryIndex(
    #     distance_metric="l2",
    #     vector_dtype=np.float32,
    #     index_directory=index_dir,
    #     index_prefix="sift_diskann",
    #     num_threads=args.threads,
    #     initial_search_complexity=params["L_search"],
    # )
    index = diskannpy.StaticDiskIndex(
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        num_threads=args.threads,
        num_nodes_to_cache=1000000000
    )
    # --- Part 2: Search Preparation ---
    print("Loading query data for search...")
    X_query = read_fvecs(args.query_file)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)

    # --- Second Pause: Before Search ---
    print("BUILD COMPLETE. Pausing for search profiling.", flush=True)
    if args.pause:
        os.kill(pid, signal.SIGSTOP)
    print("Resumed. Starting search...", flush=True)
    
    # --- Profiled Section 2: Search ---
    start_time = time.time()
    labels, distances = index.batch_search(
        queries=X_query_float32,
        k_neighbors=params["k"],
        complexity=params["L_search"],
        num_threads=args.threads,
    )
    search_time = time.time() - start_time
    ground_truth = read_ivecs(args.gt_file)
    k = params["k"]
    recall = calculate_recall(ground_truth, labels, k)
    qps = len(X_query_float32) / search_time if search_time > 0 else float("inf")
    print(f"Official DiskANN search finished in {search_time:.4f}s ({qps:.2f} QPS)")
    print(f"Recall@{k}: {recall:.4f}")
    shutil.rmtree(index_dir)
    print("Profiling run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Tool for profiling DiskANN build and search phases in a single run."
    )
    parser.add_argument(
        "--profile",
        choices=["calico", "official"],
        required=True,
        help="Which implementation to profile.",
    )
    parser.add_argument("--base_file", default="./data/sift1m/sift_base.fvecs")
    parser.add_argument("--query_file", default="./data/sift1m/sift_query.fvecs")
    parser.add_argument("--gt_file", default="./data/sift1m/sift_groundtruth.ivecs")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--pause", action='store_true', help="Enable pauses for profiler attachment.")
    args = parser.parse_args()

    diskann_params = {"R": 64, "L_build": 100, "alpha": 1.2, "k": 10, "L_search": 100}

    if args.profile == "calico":
        run_calico_full_profile(args, diskann_params)
    elif args.profile == "official":
        run_official_full_profile(args, diskann_params)

if __name__ == "__main__":
    main()