import numpy as np
import time
import os
import sys
import re
from itertools import zip_longest
import signal
import subprocess
import shutil

# --- Dependency Imports ---
# Import your own Calico implementation
import calico_py 

# Import the official Microsoft DiskANN implementation
try:
    import diskannpy
except ImportError:
    print("Error: 'diskannpy' is not installed. Please run 'pip install diskannpy' to benchmark the official implementation.")
    sys.exit(1)


# Global cache events to profile
CACHE_EVENTS = ['dTLB-load-misses', 'dTLB-loads', 'instructions', 'cycles', 'cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses']

class CacheProfiler:
    """
    A context manager to profile a block of Python code for CPU cache events
    using the Linux `perf` tool.
    """
    def __init__(self, events=None):
        if sys.platform != "linux":
            print("Warning: CacheProfiler requires Linux. Disabling cache profiling.")
            self.enabled = False
            return
        if not shutil.which("perf"):
            print("Warning: 'perf' tool not found. Disabling cache profiling.")
            self.enabled = False
            return
        
        self.events = events or CACHE_EVENTS
        self.enabled = len(self.events) > 0
        
        if self.enabled:
            # Check for perf permissions
            try:
                with open('/proc/sys/kernel/perf_event_paranoid', 'r') as f:
                    if int(f.read()) > 1:
                        print("Warning: perf permissions may be insufficient. Try 'sudo sysctl kernel.perf_event_paranoid=-1'. Disabling cache profiling.")
                        self.enabled = False
                        return
            except (IOError, ValueError):
                print("Warning: Could not check perf permissions. Disabling cache profiling.")
                self.enabled = False
                return
            print(f"CacheProfiler initialized with events: {self.events}")
        else:
            print("CacheProfiler disabled: no events to profile")
        
        self.pid = os.getpid()
        self.proc = None
        self.raw_output = ""
        self.results = {}

    def __enter__(self):
        if not self.enabled:
            return self
            
        cmd = ["perf", "stat", "-e", ",".join(self.events), "-p", str(self.pid)]
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or not self.proc:
            return False
            
        self.proc.send_signal(signal.SIGINT)
        try:
            _, stderr = self.proc.communicate(timeout=5)
            self.raw_output = stderr
            self._parse_output()
        except subprocess.TimeoutExpired:
            self.proc.kill()
            print("Warning: perf process timed out and was killed.")
        
        return False

    def _parse_output(self):
        if not self.enabled:
            return
            
        for event in self.events:
            pattern = re.compile(r"^\s*([\d,]+)\s+{}.*".format(re.escape(event)), re.MULTILINE)
            match = pattern.search(self.raw_output)
            if match:
                try:
                    self.results[event] = int(match.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    self.results[event] = "N/A"
            else:
                self.results[event] = "N/A"

    def get_results(self):
        return self.results

def format_bytes(byte_count):
    if byte_count is None or not isinstance(byte_count, (int, float)):
        return "N/A"
    if byte_count == 0:
        return "0.00 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0: return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert fv.size % (dim + 1) == 0
    fv = fv.reshape(-1, dim + 1)[:, 1:]
    if c_contiguous: fv = fv.copy()
    return fv

def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0: return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert iv.size % (dim + 1) == 0
    return iv.reshape(-1, dim + 1)[:, 1:]

def calculate_recall(ground_truth, result_labels, k):
    correct = 0
    total_queries = len(ground_truth)
    for i in range(total_queries):
        gt_set = set(ground_truth[i][:k])
        # Ensure result_labels[i] is iterable and contains numeric types
        res_set = set(int(label) for label in result_labels[i] if label != -1)
        correct += len(gt_set.intersection(res_set))
    return correct / (total_queries * k)

def benchmark_calico_diskann(X_base, X_query, ground_truth, params, num_threads):
    print("\n--- Benchmarking calico_py DiskANN ---")
    results = {}
    
    num_elements, dim = X_base.shape
    k = params["k"]
    
    # Ensure data is C-contiguous float32
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)

    # We need dummy tags for the build, as the API requires it.
    # In this benchmark, we are not testing filtering, so we can use a single dummy tag.
    dummy_tags = [[0]] * num_elements

    print(f"Building index with R={params['R']}, L_build={params['L_build']}, alpha={params['alpha']}...")
    start_time = time.time()
    
    # Initialize your Calico DiskANN index
    index_calico = calico_py.DiskANN(
        dimensions=dim, 
        max_elements=num_elements, 
        R_max_degree=params['R'], 
        is_dynamic=False
    )
    
    build_params = calico_py.BuildParams()
    build_params.L_build = params['L_build']
    build_params.alpha = params['alpha']
    build_params.num_threads = num_threads
    
    index_calico.build(X_base_float32, dummy_tags, build_params)
    
    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print(f"\nQuerying for k={k} nearest neighbors with L_search={params['L_search']}...")
    
    search_params = calico_py.SearchParams(L_search=params['L_search'])
    
    # Warm-up query to ensure pages are cached, etc.
    index_calico.search_knn_parallel(X_query_float32[:1], k, search_params, 1)

    labels, distances = None, None
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = index_calico.search_knn_parallel(X_query_float32, k, search_params, num_threads)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses, refs = cache_stats.get('cache-misses', 0), cache_stats.get('cache-references', 0)
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = index_calico.search_knn_parallel(X_query_float32, k, search_params, num_threads)
        search_time = time.time() - start_time
        
    results['search_time'] = search_time
    results['qps'] = len(X_query) / search_time
    print(f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({results['qps']:.2f} QPS)")

    recall = calculate_recall(ground_truth, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")
    
    # Calico DiskANN doesn't expose these stats yet, so we'll mark them as N/A
    results['dist_comps'] = "N/A"
    results['avg_dist_comps'] = "N/A"
    results['node_count'] = num_elements

    return results

def benchmark_diskann_official(X_base, X_query, ground_truth, params, num_threads):
    print("\n--- Benchmarking official diskannpy ---")
    results = {}
    
    num_elements, dim = X_base.shape
    k = params["k"]
    
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)

    index_dir = "./diskannpy_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "sift_diskann")

    print(f"Building index with R={params['R']}, L_build={params['L_build']}, alpha={params['alpha']}...")
    start_time = time.time()
    
    diskannpy.build_memory_index(
        data=X_base_float32,
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        complexity=params['L_build'], # L_build
        graph_degree=params['R'],   # R
        alpha=params['alpha'],
        num_threads=num_threads,
        use_pq_build=False,
        use_opq=False
    )
    
    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    # Load the index for searching
    index_official = diskannpy.StaticMemoryIndex(
        initial_search_complexity=params['L_search'],
        distance_metric="l2",
        vector_dtype=np.float32,
        index_directory=index_dir,
        index_prefix="sift_diskann",
        num_threads=num_threads
    )
    
    print(f"\nQuerying for k={k} nearest neighbors with L_search={params['L_search']}...")
    
    labels, distances = None, None
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = index_official.batch_search(
                queries=X_query_float32,
                k_neighbors=k,
                complexity=params['L_search'], # L_search
                num_threads=num_threads
            )
            search_time = time.time() - start_time

        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses, refs = cache_stats.get('cache-misses', 0), cache_stats.get('cache-references', 0)
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = index_official.batch_search(
            queries=X_query_float32, k_neighbors=k, complexity=params['L_search'], num_threads=num_threads
        )
        search_time = time.time() - start_time

    results['search_time'] = search_time
    results['qps'] = len(X_query) / search_time
    print(f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({results['qps']:.2f} QPS)")
    
    recall = calculate_recall(ground_truth, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")

    # Official DiskANN doesn't expose these stats directly in the Python API
    results['dist_comps'] = "N/A"
    results['avg_dist_comps'] = "N/A"
    results['node_count'] = num_elements

    # Clean up the index directory
    shutil.rmtree(index_dir)
    
    return results

def print_comparison_table(calico_res, official_res, params):
    """
    Prints a detailed side-by-side comparison table for DiskANN benchmarks.
    """
    METRIC_COL_WIDTH = 30
    LIB_COL_WIDTH = 35
    TOTAL_WIDTH = METRIC_COL_WIDTH + LIB_COL_WIDTH * 2 + 7
    
    def format_num(value):
        if isinstance(value, int): return f"{value:,}"
        if isinstance(value, float): return f"{value:.2f}"
        return str(value)
    
    def print_header():
        print("-" * TOTAL_WIDTH)
        print(f"| {'Metric':<{METRIC_COL_WIDTH}} | {'calico_py.DiskANN':<{LIB_COL_WIDTH}} | {'Official diskannpy':<{LIB_COL_WIDTH}} |")
        print("-" * TOTAL_WIDTH)

    def print_row(metric_name, calico_val, official_val):
        print(f"| {metric_name:<{METRIC_COL_WIDTH}} | {str(calico_val):<{LIB_COL_WIDTH}} | {str(official_val):<{LIB_COL_WIDTH}} |")

    def print_separator():
        print(f"|{'-' * (METRIC_COL_WIDTH+2)}|{'-' * (LIB_COL_WIDTH+2)}|{'-' * (LIB_COL_WIDTH+2)}|")

    print("\n\n--- DiskANN Benchmark Comparison Summary ---")
    print(f"Parameters: R={params['R']}, L_build={params['L_build']}, alpha={params['alpha']}, k={params['k']}, L_search={params['L_search']}")
    
    print_header()
    
    # --- Performance ---
    print_row("--- Performance ---", "", "")
    print_row("Build Time (s)", f"{calico_res.get('build_time', 0):.4f}", f"{official_res.get('build_time', 0):.4f}")
    print_row("Search Time (s)", f"{calico_res.get('search_time', 0):.4f}", f"{official_res.get('search_time', 0):.4f}")
    print_row("QPS (Queries/Sec)", f"{calico_res.get('qps', 0):.2f}", f"{official_res.get('qps', 0):.2f}")
    print_row(f"Recall@{params['k']}", f"{calico_res.get('recall', 0):.4f}", f"{official_res.get('recall', 0):.4f}")
    
    print_separator()

    # --- Search Internals ---
    print_row("--- Search Internals ---", "", "")
    print_row("Total Distance Computations", calico_res.get('dist_comps', 'N/A'), official_res.get('dist_comps', 'N/A'))
    print_row("Avg Dist Comps / Query", calico_res.get('avg_dist_comps', 'N/A'), official_res.get('avg_dist_comps', 'N/A'))

    print_separator()

    # --- Cache Performance ---
    print_row("--- Cache Performance (Search) ---", "", "")
    for event in CACHE_EVENTS:
        print_row(event, format_num(calico_res.get(event, 'N/A')), format_num(official_res.get(event, 'N/A')))
    
    miss_rate_calico = calico_res.get('cache_miss_rate_pct', 'N/A')
    miss_rate_official = official_res.get('cache_miss_rate_pct', 'N/A')
    print_row("Cache Miss Rate", 
              f"{miss_rate_calico:.2f}%" if isinstance(miss_rate_calico, float) else 'N/A', 
              f"{miss_rate_official:.2f}%" if isinstance(miss_rate_official, float) else 'N/A')

    print("-" * TOTAL_WIDTH)


if __name__ == "__main__":
    DATA_DIR = "./data/sift1m"
    BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
    QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")
    GT_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

    if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
        print(f"Error: SIFT1M data files not found in '{DATA_DIR}'.")
        print("Please download from http://corpus-texmex.irisa.fr/ and place them in the data/sift1m directory.")
        sys.exit(1)

    print("Loading SIFT1M dataset...")
    X_base = read_fvecs(BASE_FILE)
    X_query = read_fvecs(QUERY_FILE)
    ground_truth = read_ivecs(GT_FILE)
    
    print(f"Base vectors shape: {X_base.shape}")
    print(f"Query vectors shape: {X_query.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # DiskANN specific parameters
    # These are common parameters that provide a good balance of speed and accuracy.
    diskann_params = {
        "R": 64,             # Max degree of the graph
        "L_build": 100,      # Build complexity (similar to ef_construction)
        "alpha": 1.2,        # Pruning parameter
        "k": 10,             # Number of neighbors to search for
        "L_search": 100      # Search complexity (similar to ef_search)
    }
    
    # Set to 0 to use all available cores
    num_threads = 0 
    
    effective_threads = num_threads if num_threads > 0 else os.cpu_count()
    print(f"\nUsing {effective_threads} threads for benchmarking.\n")
    
    # Benchmark your implementation
    calico_results = benchmark_calico_diskann(
        X_base, X_query, ground_truth, diskann_params, num_threads
    )
    
    # Benchmark the official implementation
    official_results = benchmark_diskann_official(
        X_base, X_query, ground_truth, diskann_params, num_threads
    )
    
    # Print the final detailed comparison table
    print_comparison_table(calico_results, official_results, diskann_params)