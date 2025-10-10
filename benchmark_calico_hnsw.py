import numpy as np
import time
import os
import sys
import re
import subprocess
import signal
import shutil

# Conditionally import calico_py - only when needed for Calico benchmarks
# This avoids unnecessary memory allocation for hnswlib-only benchmarks
calico_py = None

# Add local hnswlib directory to Python path
sys.path.insert(0, './hnswlib')
import hnswlib

# Global cache events to profile
CACHE_EVENTS = ['dTLB-load-misses', 'dTLB-loads', 'instructions', 'cycles', 'cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses']

def get_rss_from_proc():
    """Read RSS (Resident Set Size) from /proc/self/status in bytes."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # Format is "VmRSS:     12345 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kb = int(parts[1])
                        return rss_kb * 1024  # Convert to bytes
    except Exception as e:
        print(f"Warning: Could not read RSS from /proc/self/status: {e}")
    return None

class CacheProfiler:
    """
    A context manager to profile a block of Python code for CPU cache events
    using the Linux `perf` tool.
    """
    def __init__(self, events=None):
        if sys.platform != "linux":
            raise RuntimeError("CacheProfiler requires a Linux operating system.")
        if not shutil.which("perf"):
            raise RuntimeError("The 'perf' command-line tool is not installed or not in PATH.")
        
        self.events = events or ['cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses']
        self.enabled = len(self.events) > 0
        
        if self.enabled:
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
            
        cmd = [
            "perf", "stat",
            "-e", ",".join(self.events),
            "-p", str(self.pid)
        ]
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
            
        if self.proc:
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
            
        print(self.raw_output)
        for event in self.events:
            pattern = re.compile(r"^\s*([\d,]+)\s+{}.*".format(re.escape(event)), re.MULTILINE)
            match = pattern.search(self.raw_output)
            if match:
                try:
                    value = int(match.group(1).replace(',', ''))
                    self.results[event] = value
                except (ValueError, IndexError):
                    self.results[event] = "N/A"
            else:
                cpu_pattern = re.compile(r"^\s*([\d,]+)\s+cpu_(?:core|atom)/{}/*".format(re.escape(event)), re.MULTILINE)
                matches = cpu_pattern.findall(self.raw_output)
                if matches:
                    try:
                        total_value = sum(int(match.replace(',', '')) for match in matches)
                        self.results[event] = total_value
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

def parse_calico_index_info(info_string):
    """
    Parses the detailed index info string from calico_py to extract key metrics.
    """
    parsed_data = {}
    
    patterns = {
        'page_size_bytes': r"Page Size:\s*(\d+)\s*bytes",
        'physical_pages_used': r"Physical Pages Used:\s*(\d+)",
        'node_count': r"Current Node Count:\s*(\d+)",
        'max_level': r"Current Max Level:\s*(\d+)",
        'capacity_usage_pct': r"Index Capacity Usage:\s*([\d.]+)%",
        'page_utilization_pct': r"Page Utilization:\s*([\d.]+)%",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, info_string)
        if match:
            try:
                value_str = match.group(1)
                parsed_data[key] = float(value_str) if '.' in value_str else int(value_str)
            except (ValueError, IndexError):
                pass

    if 'page_size_bytes' in parsed_data and 'physical_pages_used' in parsed_data:
        parsed_data['index_bytes'] = parsed_data['page_size_bytes'] * parsed_data['physical_pages_used']

    return parsed_data

def read_fvecs(filename, c_contiguous=True):
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

def get_fvecs_metadata(filename):
    """Get dimension and count without loading the full dataset."""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            return 0, 0
        dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        record_size = (dim + 1) * 4  # (dim + 1) float32 values
        count = file_size // record_size
        return dim, count

def read_fvecs_batch(filename, start_idx, count, dim):
    """Read a batch of vectors from an fvecs file."""
    record_size = (dim + 1) * 4  # (dim + 1) float32 values per vector
    offset = start_idx * record_size
    
    with open(filename, 'rb') as f:
        f.seek(offset)
        data = f.read(count * record_size)
        
    if len(data) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    
    fv = np.frombuffer(data, dtype=np.float32)
    fv = fv.reshape(-1, dim + 1)
    
    # Verify dimensions match
    dims = fv.view(np.int32)[:, 0]
    if not all(dims == dim):
        raise IOError(f"Non-uniform vector dimensions in {filename}")
    
    # Extract just the vector data (skip dimension prefix)
    vectors = np.ascontiguousarray(fv[:, 1:])
    return vectors

def read_ivecs(filename):
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
    correct = 0
    total_queries = len(ground_truth)
    for i in range(total_queries):
        gt_set = set(ground_truth[i][:k])
        res_set = set(result_labels[i])
        correct += len(gt_set.intersection(res_set))
    return correct / (total_queries * k)

def repeat_and_scramble_queries(X_query, ground_truth, multiplier):
    """
    Repeat the query set and ground truth by multiplier times and randomly scramble them together.
    
    Args:
        X_query: Original query vectors (N, D)
        ground_truth: Ground truth nearest neighbors (N, K)
        multiplier: How many times to repeat (typically number of threads)
    
    Returns:
        Tuple of (scrambled repeated query vectors (N*multiplier, D), 
                  scrambled repeated ground truth (N*multiplier, K))
    """
    if multiplier <= 1:
        return X_query, ground_truth
    
    # Repeat the queries and ground truth
    repeated_queries = np.tile(X_query, (multiplier, 1))
    repeated_gt = np.tile(ground_truth, (multiplier, 1))
    
    # Randomly permute with the same indices
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    indices = rng.permutation(len(repeated_queries))
    
    return repeated_queries[indices], repeated_gt[indices]

def benchmark_calico_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None):
    """
    Benchmark a specific Calico configuration.
    
    Config environment variables should be set before calling this function:
    - PHYSGB, TRADITIONAL, TRADHASH, NOHUGEPAGE_FRAME_MEM, PREFETCH
    
    Args:
        X_base: Base dataset (can be None if base_file is provided for streaming)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of threads
        config_name: Configuration name
        base_file: Optional path to base dataset file for streaming
    """
    # Import calico_py only when needed for Calico benchmarks
    global calico_py
    if calico_py is None:
        print("Importing calico_py module...")
        import calico_py as calico_module
        calico_py = calico_module
    
    print(f"\n{'='*80}")
    print(f"Benchmarking Calico Configuration: {config_name}")
    print(f"Threads: {num_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads}
    
    # Get environment configuration
    env_config = {
        'PHYSGB': os.environ.get('PHYSGB', 'N/A'),
        'TRADITIONAL': os.environ.get('TRADITIONAL', 'N/A'),
        'TRADHASH': os.environ.get('TRADHASH', 'N/A'),
        'NOHUGEPAGE_FRAME_MEM': os.environ.get('NOHUGEPAGE_FRAME_MEM', 'N/A'),
        'PREFETCH': os.environ.get('PREFETCH', 'N/A'),
    }
    results.update(env_config)
    
    print(f"Environment Configuration:")
    for key, value in env_config.items():
        print(f"  {key}={value}")

    # Always use streaming mode to minimize memory usage
    dim, num_elements = get_fvecs_metadata(base_file)
    print(f"Streaming mode: Will load {num_elements} vectors of dimension {dim} in batches")
    
    # Read RSS before building
    rss_before_build = get_rss_from_proc()
    print(f"\nRSS before build: {rss_before_build / (1024**2):.2f} MB ({rss_before_build / (1024**3):.2f} GB)")

    print(f"\nBuilding index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()
    
    enable_prefetch = os.environ.get('PREFETCH', '1') == '1'
    index_calico = calico_py.HnswIndex(max_elements=num_elements, enable_prefetch=enable_prefetch)
    print("Start building graph")
    
    # Stream dataset in 5% batches to reduce peak memory usage
    batch_size = max(1, num_elements // 20)  # 5% batches
    print(f"  Inserting {num_elements} vectors in batches of {batch_size} ({batch_size/num_elements*100:.1f}%)...")
    
    # Stream from file without loading full dataset into memory
    for i in range(0, num_elements, batch_size):
        end_idx = min(i + batch_size, num_elements)
        batch = read_fvecs_batch(base_file, i, end_idx - i, dim)
        index_calico.add_items(batch)
        print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%)")
    
    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print("\nIndex info and memory usage after build:")
    index_info_str = index_calico.get_index_info()
    print(index_info_str)
    
    parsed_info = parse_calico_index_info(index_info_str)
    results.update(parsed_info)

    mem_usage = index_calico.get_memory_usage()
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")


    k = params["k"]
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    
    # Repeat and scramble queries and ground truth together with the same permutation
    X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, num_threads)
    print(f"Query set: {len(X_query_float32)} queries repeated {num_threads}x = {len(X_query_repeated)} total queries")


    ef_search = params.get("ef_search", 50)
    print(f"\nQuerying for k={k} nearest neighbors with ef_search={ef_search}, threads={num_threads}...")
    
    index_calico.reset_stats()

    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = index_calico.search_knn_parallel(X_query_repeated, k, ef_search, num_threads, stats=False)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = index_calico.search_knn_parallel(X_query_repeated, k, ef_search, num_threads, stats=True)
        search_time = time.time() - start_time
        
    results['search_time'] = search_time

    qps = len(X_query_repeated) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")

    # Calculate recall using the scrambled ground truth (matches the query order)
    recall = calculate_recall(ground_truth_repeated, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")

    stats_string = index_calico.get_stats_string()
    try:
        dist_comps_match = re.search(r"Distance Computations:\s*(\d+)", stats_string)
        hops_match = re.search(r"Search Hops:\s*(\d+)", stats_string)
        
        if dist_comps_match:
            results['dist_comps'] = int(dist_comps_match.group(1))
            results['avg_dist_comps'] = results['dist_comps'] / len(X_query_repeated)
        if hops_match:
            results['hops'] = int(hops_match.group(1))
            results['avg_hops'] = results['hops'] / len(X_query_repeated)
    except Exception as e:
        print(f"Could not parse calico stats string: {e}")

    return results


def benchmark_hnswlib_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None):
    """
    Benchmark hnswlib with optional file-backed mmap support.
    
    Environment variable MMAP_FOR_HSNWLIB=1 enables mmap mode.
    
    Args:
        X_base: Base dataset (can be None if base_file is provided for streaming)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of threads
        config_name: Configuration name
        base_file: Optional path to base dataset file for streaming
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking hnswlib Configuration: {config_name}")
    print(f"Threads: {num_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads}
    
    # Check if mmap is enabled
    mmap_enabled = os.environ.get('MMAP_FOR_HSNWLIB', '0') == '1'
    results['MMAP_FOR_HSNWLIB'] = '1' if mmap_enabled else '0'
    print(f"Environment Configuration:")
    print(f"  MMAP_FOR_HSNWLIB={results['MMAP_FOR_HSNWLIB']}")

    # Always use streaming mode to minimize memory usage
    dim, num_elements = get_fvecs_metadata(base_file)
    print(f"Streaming mode: Will load {num_elements} vectors of dimension {dim} in batches")
    
    k = params["k"]

    # Read RSS before building
    rss_before_build = get_rss_from_proc()
    print(f"\nRSS before build: {rss_before_build / (1024**2):.2f} MB ({rss_before_build / (1024**3):.2f} GB)")

    print(f"\nBuilding index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=params["ef_construction"], M=params["M"])
    
    # Stream dataset in 5% batches to reduce peak memory usage
    batch_size = max(1, num_elements // 20)  # 5% batches
    print(f"  Inserting {num_elements} vectors in batches of {batch_size} ({batch_size/num_elements*100:.1f}%)...")
    
    # Stream from file without loading full dataset into memory
    for i in range(0, num_elements, batch_size):
        end_idx = min(i + batch_size, num_elements)
        batch = read_fvecs_batch(base_file, i, end_idx - i, dim)
        p.add_items(batch)
        print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%)")

    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print("\nMemory usage after build:")
    mem_usage = p.get_memory_usage()
    results['index_bytes'] = mem_usage.get('index_bytes')
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Index bytes: {format_bytes(results['index_bytes'])}")
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")
    results['node_count'] = num_elements

    ef_search = params.get("ef_search", 50)
    p.set_ef(ef_search)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    
    # Repeat and scramble queries and ground truth together with the same permutation
    X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, num_threads)
    print(f"Query set: {len(X_query_float32)} queries repeated {num_threads}x = {len(X_query_repeated)} total queries")

    print(f"\nQuerying for k={k} nearest neighbors with ef_search={ef_search}, threads={num_threads}...")
    
    p.reset_stats()
    search_threads = num_threads if num_threads > 0 else -1
    
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = p.knn_query(X_query_repeated, k=k, num_threads=search_threads, stats=False)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = p.knn_query(X_query_repeated, k=k, num_threads=search_threads, stats=True)
        search_time = time.time() - start_time
    
    results['search_time'] = search_time
    
    qps = len(X_query_repeated) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")
    
    # Calculate recall using the scrambled ground truth
    recall = calculate_recall(ground_truth_repeated, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")
    
    results['dist_comps'] = p.get_distance_computations()
    results['hops'] = p.get_hops()
    results['avg_dist_comps'] = results['dist_comps'] / len(X_query_repeated)
    results['avg_hops'] = results['hops'] / len(X_query_repeated)
    
    graph_stats_obj = p.get_graph_stats()
    results['max_level'] = graph_stats_obj.num_levels - 1 if graph_stats_obj.num_levels > 0 else 0

    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Calico HNSW configurations')
    parser.add_argument('--config', type=str, required=True, 
                        help='Configuration name (calico_array, calico_chained_hash, hnswlib_mmap, etc.)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of search threads')
    parser.add_argument('--dataset', type=str, default='sift1m',
                        help='Dataset to use (sift1m or sift10m)')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW M parameter')
    parser.add_argument('--ef_construction', type=int, default=200,
                        help='HNSW ef_construction parameter')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors to search')
    parser.add_argument('--ef_search', type=int, default=50,
                        help='HNSW ef_search parameter')
    parser.add_argument('--library', type=str, default='calico',
                        help='Library to use: calico or hnswlib')
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "sift1m":
        DATA_DIR = "./data/sift1m"
        BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
        QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")
        GT_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

        if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
            print(f"Error: SIFT1M data files not found in '{DATA_DIR}'.")
            sys.exit(1)

        # Always use streaming mode to minimize memory usage
        print(f"Loading SIFT1M dataset (streaming mode for {args.library})...")
        dim, count = get_fvecs_metadata(BASE_FILE)
        print(f"Base vectors: {count} vectors of dimension {dim} (will be streamed in 5% batches)")
        
        X_query = read_fvecs(QUERY_FILE)
        ground_truth = read_ivecs(GT_FILE)
        
        print(f"Query vectors shape: {X_query.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
    else:
        print(f"Dataset {args.dataset} not supported yet")
        sys.exit(1)
    
    hnsw_params = {
        "M": args.M, 
        "ef_construction": args.ef_construction, 
        "k": args.k, 
        "ef_search": args.ef_search
    }
    
    # Run benchmark based on library selection (always streaming mode)
    if args.library == 'hnswlib':
        results = benchmark_hnswlib_config(
            None, X_query, ground_truth, hnsw_params, args.threads, args.config,
            base_file=BASE_FILE
        )
    else:
        results = benchmark_calico_config(
            None, X_query, ground_truth, hnsw_params, args.threads, args.config,
            base_file=BASE_FILE
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Benchmark Summary for {args.config}")
    print(f"{'='*80}")
    print(f"Build Time: {results.get('build_time', 0):.4f}s")
    print(f"Search Time: {results.get('search_time', 0):.4f}s")
    print(f"QPS: {results.get('qps', 0):.2f}")
    print(f"Recall@{args.k}: {results.get('recall', 0):.4f}")
    print(f"Avg Dist Comps/Query: {results.get('avg_dist_comps', 0):.2f}")
    print(f"Index Size: {format_bytes(results.get('index_bytes'))}")
    
    print("\nCalico module unloading: Shutting down system...")
