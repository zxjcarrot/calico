import numpy as np
import time
import os
import sys

# Add local hnswlib directory to Python path
sys.path.insert(0, './hnswlib')
import hnswlib

import re
from itertools import zip_longest
import signal
import subprocess
import sys
import shutil

import calico_py

# Global cache events to profile
CACHE_EVENTS = ['dTLB-load-misses', 'dTLB-loads', 'instructions', 'cycles', 'cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses']

class CacheProfiler:
    """
    A context manager to profile a block of Python code for CPU cache events
    using the Linux `perf` tool.
    
    Usage:
        events = ['cache-misses', 'cache-references', 'L1-dcache-load-misses']
        with CacheProfiler(events=events) as profiler:
            # Code to be profiled goes here
            ...
        
        results = profiler.get_results()
        print(results)
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
            
        # Note: perf may require specific permissions. You might need to run as root
        # or adjust /proc/sys/kernel/perf_event_paranoid
        # e.g., sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
        cmd = [
            "perf", "stat",
            "-e", ",".join(self.events),
            "-p", str(self.pid)
        ]
        # perf stat prints its report to stderr
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        # Give perf a moment to attach to the process
        time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
            
        if self.proc:
            # Send SIGINT (Ctrl+C) to perf to make it stop and print the report
            self.proc.send_signal(signal.SIGINT)
            try:
                # Wait for the process to terminate and capture its output
                _, stderr = self.proc.communicate(timeout=5)
                self.raw_output = stderr
                self._parse_output()
            except subprocess.TimeoutExpired:
                self.proc.kill()
                print("Warning: perf process timed out and was killed.")
        
        # Don't suppress exceptions
        return False

    def _parse_output(self):
        if not self.enabled:
            return
            
        print(self.raw_output)
        # Regex to find a number (with optional commas) followed by an event name
        # e.g., "  1,234,567   cache-misses"
        for event in self.events:
            # We escape the event name to handle special characters if any
            pattern = re.compile(r"^\s*([\d,]+)\s+{}.*".format(re.escape(event)), re.MULTILINE)
            match = pattern.search(self.raw_output)
            if match:
                try:
                    # Remove commas and convert to integer
                    value = int(match.group(1).replace(',', ''))
                    self.results[event] = value
                except (ValueError, IndexError):
                    self.results[event] = "N/A"
            else:
                # Try alternative pattern for CPU core/atom format
                # e.g., "151,666,428      cpu_core/cache-misses/"
                cpu_pattern = re.compile(r"^\s*([\d,]+)\s+cpu_(?:core|atom)/{}/*".format(re.escape(event)), re.MULTILINE)
                matches = cpu_pattern.findall(self.raw_output)
                if matches:
                    try:
                        # Sum up values from all CPU cores/atoms
                        total_value = sum(int(match.replace(',', '')) for match in matches)
                        self.results[event] = total_value
                    except (ValueError, IndexError):
                        self.results[event] = "N/A"
                else:
                    self.results[event] = "N/A"

    def get_results(self):
        return self.results

# Helper function to format bytes
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
    
    # Regex patterns to find specific values
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
                # Convert matched string to float or int
                value_str = match.group(1)
                parsed_data[key] = float(value_str) if '.' in value_str else int(value_str)
            except (ValueError, IndexError):
                pass # Ignore if conversion fails

    # Calculate index size as requested
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


def benchmark_calico(X_base, X_query, ground_truth, params, num_threads):
    print("\n--- Benchmarking calico_py HNSW ---")
    results = {}

    num_elements, dim = X_base.shape
    k = params["k"]
    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)

    print(f"Building index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()
    
    index_calico = calico_py.HnswIndex(max_elements=num_elements, enable_prefetch=False)
    print("Start building graph")
    index_calico.add_items(X_base_float32, 0)
    
    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print("\nIndex info and memory usage after build:")
    index_info_str = index_calico.get_index_info()
    print(index_info_str)
    
    # Parse the detailed info string and update results
    parsed_info = parse_calico_index_info(index_info_str)
    results.update(parsed_info)

    mem_usage = index_calico.get_memory_usage()
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")

    ef_search = params.get("ef_search", 50)
    print(f"Querying for k={k} nearest neighbors with ef_search={ef_search}...")
    
    index_calico.reset_stats()
    start_time = time.time()

    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = index_calico.search_knn_parallel(X_query_float32, k, 46, num_threads, stats=True)
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
        
    search_time = time.time() - start_time
    results['search_time'] = search_time

    qps = len(X_query) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")

    recall = calculate_recall(ground_truth, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")

    stats_string = index_calico.get_stats_string()
    try:
        dist_comps_match = re.search(r"Distance Computations:\s*(\d+)", stats_string)
        hops_match = re.search(r"Search Hops:\s*(\d+)", stats_string)
        
        if dist_comps_match:
            results['dist_comps'] = int(dist_comps_match.group(1))
            results['avg_dist_comps'] = results['dist_comps'] / len(X_query)
        if hops_match:
            results['hops'] = int(hops_match.group(1))
            results['avg_hops'] = results['hops'] / len(X_query)
        
        graph_stats_match = re.search(r"--- Graph Structure Stats ---(.*)", stats_string, re.DOTALL)
        if graph_stats_match:
            results['graph_stats'] = graph_stats_match.group(1).strip()
    except Exception as e:
        print(f"Could not parse calico stats string: {e}")

    return results


def benchmark_hnswlib(X_base, X_query, ground_truth, params, num_threads):
    print("\n--- Benchmarking hnswlib ---")
    results = {}

    num_elements, dim = X_base.shape
    k = params["k"]
    results['node_count'] = num_elements

    X_base_float32 = np.ascontiguousarray(X_base, dtype=np.float32)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    print(f"Building index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()

    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=params["ef_construction"], M=params["M"])
    
    p.add_items(X_base_float32)

    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print("\nMemory usage after build:")
    mem_usage = p.get_memory_usage()
    results['index_bytes'] = mem_usage.get('index_bytes')
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Index bytes: {format_bytes(results['index_bytes'])}")
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")

    ef_search = params.get("ef_search", 50)
    p.set_ef(ef_search)
    print(f"Querying for k={k} nearest neighbors with ef_search={ef_search}...")
    
    p.reset_stats()
    start_time = time.time()
    search_threads = num_threads if num_threads > 0 else -1
    
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = p.knn_query(X_query_float32, k=k, num_threads=search_threads, stats=True)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        # miss rate
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
    
    
    search_time = time.time() - start_time
    results['search_time'] = search_time
    
    qps = len(X_query) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")
    
    recall = calculate_recall(ground_truth, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")
    
    results['dist_comps'] = p.get_distance_computations()
    results['hops'] = p.get_hops()
    results['avg_dist_comps'] = results['dist_comps'] / len(X_query_float32)
    results['avg_hops'] = results['hops'] / len(X_query_float32)
    
    graph_stats_obj = p.get_graph_stats()
    results['graph_stats'] = str(graph_stats_obj)
    # The number of levels is N, but they are 0-indexed, so max level is N-1
    results['max_level'] = graph_stats_obj.num_levels - 1 if graph_stats_obj.num_levels > 0 else 0

    return results


def print_comparison_table(calico_res, hnswlib_res, params):
    """
    Prints a detailed side-by-side comparison table of the benchmark results.
    """
    k = params['k']
    
    METRIC_COL_WIDTH = 35
    CALICO_COL_WIDTH = 40
    HNSWLIB_COL_WIDTH = 40
    TOTAL_WIDTH = METRIC_COL_WIDTH + CALICO_COL_WIDTH + HNSWLIB_COL_WIDTH + 7
    
    def format_num(value):
        # Only apply comma formatting if the value is an integer
        if isinstance(value, int):
            return f"{value:,}"
        # For floats, format them as floats
        if isinstance(value, float):
            return f"{value:.2f}"
        # Otherwise, return the value as a string (e.g., "N/A")
        return str(value)
    
    def print_header():
        print("-" * TOTAL_WIDTH)
        print(f"| {'Metric':<{METRIC_COL_WIDTH}} | {'calico_py':<{CALICO_COL_WIDTH}} | {'hnswlib':<{HNSWLIB_COL_WIDTH}} |")
        print("-" * TOTAL_WIDTH)

    def print_row(metric_name, calico_val, hnswlib_val, is_formatted=False):
        c_val, h_val = (calico_val, hnswlib_val) if is_formatted else (str(calico_val), str(hnswlib_val))
        print(f"| {metric_name:<{METRIC_COL_WIDTH}} | {c_val:<{CALICO_COL_WIDTH}} | {h_val:<{HNSWLIB_COL_WIDTH}} |")

    def print_separator():
        print(f"|{'-' * (METRIC_COL_WIDTH+2)}|{'-' * (CALICO_COL_WIDTH+2)}|{'-' * (HNSWLIB_COL_WIDTH+2)}|")

    print("\n\n--- Benchmark Comparison Summary ---")
    print(f"Parameters: M={params['M']}, ef_construction={params['ef_construction']}, k={k}, ef_search={params['ef_search']}")
    
    print_header()
    
    # --- Performance ---
    print_row("--- Performance ---", "", "", is_formatted=True)
    print_row("Build Time (s)", f"{calico_res.get('build_time', 0):.4f}", f"{hnswlib_res.get('build_time', 0):.4f}", is_formatted=True)
    print_row("Search Time (s)", f"{calico_res.get('search_time', 0):.4f}", f"{hnswlib_res.get('search_time', 0):.4f}", is_formatted=True)
    print_row("QPS (Queries/Sec)", f"{calico_res.get('qps', 0):.2f}", f"{hnswlib_res.get('qps', 0):.2f}", is_formatted=True)
    print_row(f"Recall@{k}", f"{calico_res.get('recall', 0):.4f}", f"{hnswlib_res.get('recall', 0):.4f}", is_formatted=True)
    
    print_separator()

    # --- Memory ---
    print_row("--- Memory Usage ---", "", "", is_formatted=True)
    print_row("Index Size", 
              format_bytes(calico_res.get('index_bytes')), 
              format_bytes(hnswlib_res.get('index_bytes')),
              is_formatted=True)
    print_row("Process RSS (after build)", 
              format_bytes(calico_res.get('rss_after_build')), 
              format_bytes(hnswlib_res.get('rss_after_build')),
              is_formatted=True)
    
    print_separator()
    
    # --- Index Structure ---
    print_row("--- Index Structure ---", "", "", is_formatted=True)
    print_row("Node Count", 
              f"{calico_res.get('node_count', 'N/A'):,}", 
              f"{hnswlib_res.get('node_count', 'N/A'):,}",
              is_formatted=True)
    print_row("Max Level", f"{calico_res.get('max_level', 'N/A')}", f"{hnswlib_res.get('max_level', 'N/A')}", is_formatted=True)
    print_row("Index Capacity Usage", f"{calico_res.get('capacity_usage_pct', 'N/A'):.2f}%" if 'capacity_usage_pct' in calico_res else 'N/A', "N/A", is_formatted=True)
    print_row("Page Utilization", f"{calico_res.get('page_utilization_pct', 'N/A'):.2f}%" if 'page_utilization_pct' in calico_res else 'N/A', "N/A", is_formatted=True)


    print_separator()

    # --- Search Internals ---
    print_row("--- Search Internals ---", "", "", is_formatted=True)
    print_row("Total Distance Computations", f"{calico_res.get('dist_comps', 0):,}", f"{hnswlib_res.get('dist_comps', 0):,}", is_formatted=True)
    print_row("Avg Dist Comps / Query", f"{calico_res.get('avg_dist_comps', 0):.2f}", f"{hnswlib_res.get('avg_dist_comps', 0):.2f}", is_formatted=True)
    print_row("Total Hops", f"{calico_res.get('hops', 0):,}", f"{hnswlib_res.get('hops', 0):,}", is_formatted=True)
    print_row("Avg Hops / Query", f"{calico_res.get('avg_hops', 0):.2f}", f"{hnswlib_res.get('avg_hops', 0):.2f}", is_formatted=True)
    
    print_separator()

    # --- Cache Performance ---
    print_row("--- Cache Performance (Search) ---", "", "", is_formatted=True)
    
    # Print all cache events dynamically
    for event in CACHE_EVENTS:
        print_row(event, 
                  format_num(calico_res.get(event, 'N/A')), 
                  format_num(hnswlib_res.get(event, 'N/A')), 
                  is_formatted=True)
    
    # Print cache miss rate if available
    print_row("Cache Miss Rate", 
              f"{calico_res.get('cache_miss_rate_pct', 'N/A'):.2f}%" if isinstance(calico_res.get('cache_miss_rate_pct'), float) else 'N/A', 
              f"{hnswlib_res.get('cache_miss_rate_pct', 'N/A'):.2f}%" if isinstance(hnswlib_res.get('cache_miss_rate_pct'), float) else 'N/A', 
              is_formatted=True)

    print_separator()

    # --- Graph Structure per Level ---
    calico_graph_lines = calico_res.get('graph_stats', 'N/A').split('\n')
    hnswlib_graph_lines = hnswlib_res.get('graph_stats', 'N/A').split('\n')
    
    print_row("--- Graph Structure ---", "", "", is_formatted=True)
    for calico_line, hnswlib_line in zip_longest(calico_graph_lines, hnswlib_graph_lines, fillvalue=""):
        print(f"| {'':<{METRIC_COL_WIDTH}} | {calico_line.strip():<{CALICO_COL_WIDTH}} | {hnswlib_line.strip():<{HNSWLIB_COL_WIDTH}} |")

    print("-" * TOTAL_WIDTH)

if __name__ == "__main__":
    dataset_name = "sift1m"
    
    if dataset_name == "sift1m":
        DATA_DIR = "./data/sift1m"
        BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
        QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")
        GT_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

        if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
            print(f"Error: SIFT1M data files not found in '{DATA_DIR}'.")
            exit(1)

        print("Loading SIFT1M dataset...")
        X_base = read_fvecs(BASE_FILE)
        X_query = read_fvecs(QUERY_FILE)
        ground_truth = read_ivecs(GT_FILE)
        
        print(f"Base vectors shape: {X_base.shape}")
        print(f"Query vectors shape: {X_query.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
    else:
        #sift10m
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
        
        ground_truth = np.load("./data/SIFT10M/ground_truth_indices.npy")
        # import faiss
        # index_flat = faiss.IndexFlatL2(X_base_float32.shape[1]) 

        # index_flat.add(X_base_float32)

        # k = 10
        # start_time = time.time()
        # D_flat, ground_truth = index_flat.search(X_query_float32, k) 
        # end_time = time.time()
        # print(f"Flat search took: {end_time - start_time:.4f} seconds")

    # You can add the sift10m loading logic here if needed.
    
    hnsw_params = {"M": 16, "ef_construction": 200, "k": 10, "ef_search": 50}
    num_threads = 1 # Set to 0 to let hnswlib auto-detect
    
    print(f"Using {num_threads} threads for benchmarking.\n")
    
    calico_results = benchmark_calico(
        X_base, X_query, ground_truth, hnsw_params, num_threads
    )
    hnswlib_results = benchmark_hnswlib(
        X_base, X_query, ground_truth, hnsw_params, num_threads
    )
    
    # Print the final detailed comparison table
    print_comparison_table(calico_results, hnswlib_results, hnsw_params)
    
    print("\nCalico module unloading: Shutting down system...")
