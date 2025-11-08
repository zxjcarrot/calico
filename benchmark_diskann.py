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


# --- Core Profiling Functions ---


def run_calico_profile(X_base, X_query, params, num_threads, pause):
    """Builds Calico index in memory, pauses, then searches."""
    print("\n--- Preparing Calico Profile Run ---")

    # 1. Build index in memory
    print("Building Calico index in memory...")
    num_elements, dim = X_base.shape
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    dummy_tags = [[0]] * num_elements
    start_time = time.time()

    index = calico_py.DiskANN(
        dimensions=dim,
        max_elements=num_elements,
        R_max_degree=params["R"],
        is_dynamic=False,
    )
    build_params = calico_py.BuildParams()
    build_params.L_build = params["L_build"]
    build_params.alpha = params["alpha"]
    build_params.num_threads = num_threads
    index.build(X_base_float32, dummy_tags, build_params)
    print("Calico index build complete.")
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f} seconds")

    # Prepare search data
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    search_params = calico_py.SearchParams(L_search=params["L_search"])
    k = params["k"]

    # 2. Pause the script for the profiler to attach
    pid = os.getpid()
    print(f"SETUP COMPLETE. Script is now paused. PID: {pid}", flush=True)
    print(f"To continue, run: kill -SIGCONT {pid}", flush=True)
    if pause:
        os.kill(pid, signal.SIGSTOP)
    print("Script resumed by signal. Starting search now...", flush=True)

    # 3. Execute the search (this is the part that gets profiled)
    start_time = time.time()
    index.search_knn_parallel(X_query_float32, k, search_params, num_threads)
    search_time = time.time() - start_time
    qps = len(X_query_float32) / search_time if search_time > 0 else float("inf")
    print(f"Calico search finished in {search_time:.4f}s ({qps:.2f} QPS)")


def run_official_profile(X_base, X_query, params, num_threads, pause):
    """Builds official DiskANN index to disk, loads it, pauses, then searches."""
    print("\n--- Preparing Official DiskANN Profile Run ---")
    index_dir = "./diskannpy_index"

    # 1. Build index to disk
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    print("Building Official DiskANN index to disk...")
    start_time = time.time()
    diskannpy.build_memory_index(
        data=X_base_float32,
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        complexity=params["L_build"],
        graph_degree=params["R"],
        alpha=params["alpha"],
        num_threads=num_threads,
        use_pq_build=False,
        use_opq=False,
    )
    print("Official DiskANN index build complete.")
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f} seconds")

    # Load index from disk
    print("Loading Official DiskANN index from disk...")
    index = diskannpy.StaticMemoryIndex(
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        num_threads=num_threads,
        initial_search_complexity=params["L_search"],
    )

    # Prepare search data
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    k = params["k"]

    # 2. Pause the script
    pid = os.getpid()
    print(f"SETUP COMPLETE. Script is now paused. PID: {pid}", flush=True)
    print(f"To continue, run: kill -SIGCONT {pid}", flush=True)
    if pause:
        os.kill(pid, signal.SIGSTOP)
    print("Script resumed by signal. Starting search now...", flush=True)

    # 3. Execute the search
    start_time = time.time()
    index.batch_search(
        queries=X_query_float32,
        k_neighbors=k,
        complexity=params["L_search"],
        num_threads=num_threads,
    )
    search_time = time.time() - start_time
    qps = len(X_query_float32) / search_time if search_time > 0 else float("inf")
    print(f"Official DiskANN search finished in {search_time:.4f}s ({qps:.2f} QPS)")

    # Clean up
    shutil.rmtree(index_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Tool for building or searching DiskANN indexes for profiling."
    )
    parser.add_argument(
        "--profile",
        choices=["calico", "official"],
        required=True,
        help="Run a specific benchmark profile.",
    )

    # Common args (can be adjusted if needed)
    parser.add_argument("--base_file", default="./data/sift1m/sift_base.fvecs")
    parser.add_argument("--query_file", default="./data/sift1m/sift_query.fvecs")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--pause", type=bool, default=False)

    args = parser.parse_args()

    # These params are hardcoded for simplicity in the shell script
    diskann_params = {"R": 64, "L_build": 100, "alpha": 1.2, "k": 10, "L_search": 100}

    print(f"Loading base data from {args.base_file}...")
    X_base = read_fvecs(args.base_file)
    print(f"Loading query data from {args.query_file}...")
    X_query = read_fvecs(args.query_file)

    if args.profile == "calico":
        run_calico_profile(X_base, X_query, diskann_params, args.threads, args.pause)
    elif args.profile == "official":
        run_official_profile(X_base, X_query, diskann_params, args.threads, args.pause)


if __name__ == "__main__":
    main()
