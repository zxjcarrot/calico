import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

for path_entry in (
    str(SCRIPT_DIR),
    str(SCRIPT_DIR / 'predictivetranslation-sigmod2026'),
):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

if 'BLOCK' not in os.environ:
    default_block = SCRIPT_DIR / 'heapfile'
    if default_block.exists():
        os.environ['BLOCK'] = str(default_block)

try:
    import h5py
except ImportError:
    h5py = None

import benchmark_calico_hnsw as common

calico_py = None
predicache_py = None

DEFAULT_DATASET_URL = 'https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz'
HF_DEFAULT_LABEL = 'dbpedia_openai_1m'
MAX_BENCHMARK_THREADS = 64


class LocalANNBenchmarkDataset:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path).expanduser()
        self.handle = None
        self.base = None
        self.queries = None
        self.neighbors = None

    def __enter__(self):
        if h5py is None:
            raise RuntimeError(
                'h5py is required to read the local IVF benchmark dataset. '
                'Install it with: python3 -m pip install --user h5py'
            )
        if not self.file_path.exists():
            raise RuntimeError(f'Local dataset file not found: {self.file_path}')

        self.handle = h5py.File(self.file_path, 'r')
        missing = [name for name in ('train', 'test') if name not in self.handle]
        if missing:
            raise RuntimeError(
                f'Local dataset file {self.file_path} is missing required datasets: {missing}'
            )

        self.base = self.handle['train']
        self.queries = self.handle['test']
        self.neighbors = self.handle['neighbors'] if 'neighbors' in self.handle else None

        if len(self.base.shape) != 2 or len(self.queries.shape) != 2:
            raise RuntimeError('Expected 2D train/test datasets in the local ANN benchmark file.')
        if self.base.shape[1] != self.queries.shape[1]:
            raise RuntimeError(
                f'Base/query dimension mismatch: {self.base.shape[1]} != {self.queries.shape[1]}'
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.base = None
        self.queries = None
        self.neighbors = None

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])

    @property
    def total_base(self) -> int:
        return int(self.base.shape[0])

    @property
    def total_queries(self) -> int:
        return int(self.queries.shape[0])

    def get_base_slice(self, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        stop = min(start + count, self.total_base)
        return np.ascontiguousarray(np.asarray(self.base[start:stop], dtype=np.float32))

    def get_query_slice(self, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        stop = min(start + count, self.total_queries)
        return np.ascontiguousarray(np.asarray(self.queries[start:stop], dtype=np.float32))

    def get_ground_truth_slice(self, count: int, k: int) -> Optional[np.ndarray]:
        if self.neighbors is None:
            return None
        if len(self.neighbors.shape) != 2:
            return None
        if self.neighbors.shape[1] < k:
            raise RuntimeError(
                f'Ground-truth dataset width {self.neighbors.shape[1]} is smaller than requested k={k}.'
            )
        stop = min(count, int(self.neighbors.shape[0]))
        return np.ascontiguousarray(np.asarray(self.neighbors[:stop, :k], dtype=np.int32))


def repeat_queries(queries: np.ndarray, multiplier: int) -> np.ndarray:
    if multiplier <= 1:
        return queries
    repeated = np.tile(queries, (multiplier, 1))
    rng = np.random.default_rng(seed=42)
    return repeated[rng.permutation(len(repeated))]


def resolve_library(config_name: str, explicit_library: Optional[str]) -> str:
    if explicit_library:
        return explicit_library
    if config_name.startswith('predicache'):
        return 'predicache'
    if config_name.startswith('pdx'):
        return 'pdx'
    return 'calico'


def load_local_embeddings(dataset: LocalANNBenchmarkDataset, args):
    if dataset.total_base < 1:
        raise RuntimeError('The local dataset does not contain any base vectors.')
    if dataset.total_queries < 1:
        raise RuntimeError('The local dataset does not contain any query vectors.')

    base_limit = args.base_limit if args.base_limit > 0 else dataset.total_base
    base_count = min(base_limit, dataset.total_base)
    query_count = args.query_count if args.query_count > 0 else dataset.total_queries
    query_count = min(query_count, dataset.total_queries)
    if base_count <= 0:
        raise RuntimeError('Base vector count is invalid. Increase --base-limit.')
    if query_count <= 0:
        raise RuntimeError('Query vector count is invalid. Increase --query-count.')

    print(f'Loaded local dataset metadata from {dataset.file_path}')
    print(f'Available base rows: {dataset.total_base}')
    print(f'Available query rows: {dataset.total_queries}')
    print(f'Using base_count={base_count}, query_count={query_count}, dim={dataset.dim}')

    return base_count, query_count, dataset.dim


def resolve_ivf_parameters(base_count: int, args) -> tuple[int, int, int]:
    num_clusters = args.num_clusters if args.num_clusters > 0 else max(1, math.isqrt(base_count))
    nprobe = args.nprobe if args.nprobe > 0 else 16
    nprobe = min(nprobe, num_clusters)

    default_train_count = max(num_clusters, math.ceil(base_count * 0.10))
    requested_train_count = args.train_count if args.train_count > 0 else default_train_count
    train_count = min(requested_train_count, base_count)

    if train_count < num_clusters:
        raise RuntimeError('--train_count must be at least as large as --num_clusters for IVF training.')

    return num_clusters, nprobe, train_count


def compute_exact_ground_truth(dataset: LocalANNBenchmarkDataset, base_count: int, queries: np.ndarray, k: int,
                               chunk_size: int) -> np.ndarray:
    num_queries = len(queries)
    best_distances = np.full((num_queries, k), np.inf, dtype=np.float32)
    best_indices = np.full((num_queries, k), -1, dtype=np.int64)

    for start in range(0, base_count, chunk_size):
        chunk = dataset.get_base_slice(start, min(chunk_size, base_count - start))
        if chunk.size == 0:
            continue

        diff = queries[:, None, :] - chunk[None, :, :]
        distances = np.sum(diff * diff, axis=2, dtype=np.float32)
        candidate_indices = np.arange(start, start + chunk.shape[0], dtype=np.int64)
        tiled_indices = np.broadcast_to(candidate_indices, distances.shape)

        combined_distances = np.concatenate((best_distances, distances), axis=1)
        combined_indices = np.concatenate((best_indices, tiled_indices), axis=1)
        topk_partition = np.argpartition(combined_distances, kth=min(k - 1, combined_distances.shape[1] - 1), axis=1)[:, :k]
        best_distances = np.take_along_axis(combined_distances, topk_partition, axis=1)
        best_indices = np.take_along_axis(combined_indices, topk_partition, axis=1)

        order = np.argsort(best_distances, axis=1)
        best_distances = np.take_along_axis(best_distances, order, axis=1)
        best_indices = np.take_along_axis(best_indices, order, axis=1)

    return best_indices.astype(np.int32)


def get_module(library: str, thread_count: int):
    os.environ['THREADS'] = str(thread_count)
    global calico_py, predicache_py
    if library in ('calico', 'pdx'):
        if calico_py is None:
            print('Importing calico_py module...')
            import calico_py as calico_module
            calico_py = calico_module
        return calico_py

    if predicache_py is None:
        print('Importing predicache_py module...')
        import predicache_py as predicache_module
        predicache_py = predicache_module
    return predicache_py


def instantiate_index(module, library: str, config_name: str, base_count: int, dim: int,
                      params: dict, skip_recovery: bool):
    kwargs = dict(
        max_elements=base_count,
        dim=dim,
        num_clusters=params['num_clusters'],
        retrain_interval=params['retrain_interval'],
        enable_prefetch=params['enable_prefetch'],
        skip_recovery=skip_recovery,
    )
    if library == 'pdx' or config_name in ('pdx_predicache', 'pdx_predicache_no_pt'):
        return module.PDXIndex(**kwargs)
    return module.IVFIndex(**kwargs)


def should_warmup_recovered_index() -> bool:
    run_mode = os.environ.get('CALICO_IVF_EXPERIMENT_MODE', '').strip().lower()
    if run_mode != 'in-memory':
        return False

    requested = os.environ.get('CALICO_WARMUP_RECOVERED_INDEX', '1').strip().lower()
    return requested not in ('0', 'false', 'no', '')


def validate_recovered_index(index, base_count: int, dim: int, params: dict) -> list[str]:
    rebuild_reasons = []
    if not hasattr(index, 'is_trained') or not index.is_trained():
        rebuild_reasons.append('index is not trained')
    if hasattr(index, 'get_dim') and index.get_dim() != dim:
        rebuild_reasons.append(f'dimension {index.get_dim()} != expected {dim}')
    elif hasattr(index, 'dim') and int(index.dim) != dim:
        rebuild_reasons.append(f'dimension {int(index.dim)} != expected {dim}')
    if hasattr(index, 'get_num_clusters') and index.get_num_clusters() != params['num_clusters']:
        rebuild_reasons.append(
            f'num_clusters {index.get_num_clusters()} != expected {params["num_clusters"]}'
        )
    elif hasattr(index, 'num_clusters') and int(index.num_clusters) != params['num_clusters']:
        rebuild_reasons.append(
            f'num_clusters {int(index.num_clusters)} != expected {params["num_clusters"]}'
        )
    if hasattr(index, 'get_count') and index.get_count() != base_count:
        rebuild_reasons.append(f'count {index.get_count()} != expected {base_count}')
    return rebuild_reasons


def benchmark_ivf_config(dataset: LocalANNBenchmarkDataset, base_count: int, query_vectors: np.ndarray, params: dict,
                         num_threads: int, config_name: str, library: str,
                         ground_truth: Optional[np.ndarray] = None, build_threads: Optional[int] = None):
    num_threads = max(1, min(MAX_BENCHMARK_THREADS, int(num_threads)))
    requested_build_threads = build_threads if build_threads is not None else num_threads
    effective_build_threads = max(1, min(MAX_BENCHMARK_THREADS, int(requested_build_threads)))
    runtime_threads = max(num_threads, effective_build_threads)
    module = get_module(library, runtime_threads)

    env_config = {
        'TRADITIONAL': os.environ.get('TRADITIONAL', 'unset'),
        'TRADHASH': os.environ.get('TRADHASH', 'unset'),
        'NOHUGEPAGE_FRAME_MEM': os.environ.get('NOHUGEPAGE_FRAME_MEM', 'unset'),
        'PREFETCH': os.environ.get('PREFETCH', '1'),
        'BLOCK': os.environ.get('BLOCK', 'unset'),
        'IVF_PAGE_SIZE_BYTES': os.environ.get('IVF_PAGE_SIZE_BYTES', 'unset'),
    }
    print(f'Benchmarking {library} IVF configuration: {config_name}')
    print('Environment configuration:')
    for key, value in env_config.items():
        print(f'  {key}={value}')

    results = {
        'config': config_name,
        'library': library,
        'threads': num_threads,
        'build_threads': effective_build_threads,
        'dataset': params['dataset_label'],
        'dataset_url': params['dataset_url'],
        'hf_dataset': params['dataset_url'],
        'data_file': str(dataset.file_path),
        'num_clusters': params['num_clusters'],
        'nprobe': params['nprobe'],
        'k': params['k'],
        'kmeans_iters': params['kmeans_iters'],
        'train_count': params['train_count'],
        'base_count': base_count,
        'query_count': len(query_vectors),
        'page_size_bytes': env_config['IVF_PAGE_SIZE_BYTES'],
    }
    results.update(env_config)

    rss_before_build = common.get_rss_from_proc()
    if rss_before_build is not None:
        print(f'RSS before build: {common.format_bytes(rss_before_build)}')
        results['rss_before_build'] = rss_before_build

    start_time = time.time()
    try:
        index = instantiate_index(module, library, config_name, base_count, query_vectors.shape[1], params,
                                  params['skip_recovery'])
    except RuntimeError as exc:
        if params['skip_recovery']:
            raise
        reason = str(exc)
        print(f'Recovery-time index initialization failed ({reason}); rebuilding from scratch.')
        results['recovery_rebuild_reason'] = reason
        start_time = time.time()
        index = instantiate_index(module, library, config_name, base_count, query_vectors.shape[1], params, True)
    recovered = bool(index.was_recovered()) if hasattr(index, 'was_recovered') else False
    results['skip_recovery_requested'] = params['skip_recovery']
    results['recovered'] = recovered

    if recovered and not params['skip_recovery']:
        rebuild_reasons = validate_recovered_index(index, base_count, query_vectors.shape[1], params)
        if rebuild_reasons:
            reason = '; '.join(rebuild_reasons)
            print(f'Recovered IVF metadata mismatch ({reason}); rebuilding from scratch.')
            results['recovery_rebuild_reason'] = reason
            del index
            start_time = time.time()
            index = instantiate_index(module, library, config_name, base_count, query_vectors.shape[1], params, True)
            recovered = False
            results['recovered'] = False
        else:
            print('Recovered existing IVF index from disk; skipping training and ingest.')
            if should_warmup_recovered_index() and hasattr(module, 'warmup_storage_to_alloc_count'):
                warmup_start = time.time()
                print('Warming recovered storage sequentially through allocCount for in-memory mode...')
                module.warmup_storage_to_alloc_count()
                warmup_time = time.time() - warmup_start
                results['recovery_warmup_time'] = warmup_time
                print(f'Recovery warmup completed in {warmup_time:.4f} seconds')

    if not recovered:
        training_vectors = dataset.get_base_slice(0, params['train_count'])
        print(f'Training IVF centroids on {len(training_vectors)} vectors...')
        index.train(training_vectors, params['kmeans_iters'])

        batch_size = params['batch_size']
        if batch_size <= 0:
            target_batch_bytes = 256 * 1024 * 1024
            batch_size = max(1, min(base_count // 20 if base_count >= 20 else base_count,
                                    target_batch_bytes // (query_vectors.shape[1] * 4)))

        print(f'Adding {base_count} base vectors in batches of {batch_size}...')
        processed = 0
        ingest_start = time.time()
        for start_idx in range(0, base_count, batch_size):
            loop_start = time.time()
            batch = dataset.get_base_slice(start_idx, min(batch_size, base_count - start_idx))
            index.add_items(batch, num_threads=effective_build_threads)
            processed += len(batch)
            elapsed = time.time() - ingest_start
            throughput = processed / elapsed if elapsed > 0 else float('inf')
            mbps = (processed * query_vectors.shape[1] * 4) / elapsed / (1024 ** 2) if elapsed > 0 else 0.0
            print(
                f'  Inserted {processed}/{base_count} vectors '
                f'({processed / base_count * 100:.1f}%) in {time.time() - loop_start:.3f}s; '
                f'throughput={throughput:.2f} vec/s ({mbps:.2f} MiB/s)'
            )
        if hasattr(index, 'flush'):
            print('Flushing index pages to storage...')
            index.flush()

    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f'Build/recovery time: {build_time:.4f} seconds')

    if hasattr(index, 'get_index_info'):
        index_info = index.get_index_info()
        results['index_info'] = index_info
        print(index_info)

    rss_after_build = common.get_rss_from_proc()
    if rss_after_build is not None:
        print(f'RSS after build: {common.format_bytes(rss_after_build)}')
        results['rss_after_build'] = rss_after_build

    query_repeat_multiplier = 1
    repeated_queries = repeat_queries(np.ascontiguousarray(query_vectors, dtype=np.float32), query_repeat_multiplier)
    print(
        f'Query set: {len(query_vectors)} queries repeated {query_repeat_multiplier}x = '
        f'{len(repeated_queries)} total queries'
    )

    warmup_count = max(len(repeated_queries), max(1, params['warmup_queries']))
    if warmup_count > 0:
        warmup_batch_size = max(1, int(os.environ.get('IVF_WARMUP_BATCH_SIZE', '32')))
        warmup_batch_size = min(warmup_batch_size, warmup_count)

        print(f'Warming up with {warmup_count} queries...')
        for start_idx in range(0, warmup_count, warmup_batch_size):
            end_idx = min(start_idx + warmup_batch_size, warmup_count)
            index.search_knn_parallel(
                repeated_queries[start_idx:end_idx],
                params['k'],
                params['nprobe'],
                num_threads,
                False,
            )
            if warmup_count > warmup_batch_size:
                print(f'  Warmup progress: {end_idx}/{warmup_count}')
        procfs_snapshot = common.capture_procfs_snapshot('after_warmup', os.environ.get('BLOCK'))
        if procfs_snapshot is not None:
            results['procfs_snapshot_after_warmup'] = str(procfs_snapshot)
        print(f'Warmup completed.')
    collect_search_stats = common._read_bool_env('CALICO_ENABLE_SEARCH_STATS', False)
    perf_enabled = not common._read_bool_env('CALICO_DISABLE_PERF', False)
    perf_event_groups = common.parse_perf_event_groups(common.CACHE_EVENTS) if perf_enabled else []
    if perf_event_groups:
        group_names = ', '.join(group_name for group_name, _events in perf_event_groups)
        print(f'Perf stat groups: {group_names}')
        results['perf_stat_groups'] = [group_name for group_name, _events in perf_event_groups]

    def run_search_with_events(event_group_name: Optional[str], event_names: Optional[list[str]],
                               include_search_stats: bool):
        profiler = None
        if perf_enabled and event_names:
            profiler = common.CacheProfiler(events=event_names)

        with profiler if profiler is not None else contextlib.nullcontext():
            search_start = time.time()
            local_labels, local_distances = index.search_knn_parallel(
                repeated_queries,
                params['k'],
                params['nprobe'],
                num_threads,
                include_search_stats,
            )
            local_search_time = time.time() - search_start

        local_stats = profiler.get_results() if profiler is not None else {}
        if event_group_name and local_stats:
            print(f'Perf stat group {event_group_name}:')
            for event_name, value in local_stats.items():
                print(f'  {event_name}: {value}')
        return local_labels, local_distances, local_search_time, local_stats

    perf_session = None
    if perf_enabled:
        perf_session = common.start_perf_profiling(
            delay=common.DEFAULT_PERF_DELAY_SECONDS,
            duration=common.DEFAULT_PERF_DURATION_SECONDS,
            label_suffix='ivf_search',
            target_pid=os.getpid(),
        )

    labels = None
    distances = None
    try:
        primary_group_name = None
        primary_group_events = None
        if perf_event_groups:
            primary_group_name, primary_group_events = perf_event_groups[0]
        labels, distances, search_time, primary_stats = run_search_with_events(
            primary_group_name,
            primary_group_events,
            collect_search_stats,
        )
        if primary_group_name and primary_stats:
            common.add_perf_group_results(results, primary_group_name, primary_stats, include_legacy_keys=True)

        for extra_group_name, extra_group_events in perf_event_groups[1:]:
            _, _, extra_search_time, extra_stats = run_search_with_events(extra_group_name, extra_group_events, False)
            results[f'{common.sanitize_perf_counter_name(extra_group_name)}_search_time'] = extra_search_time
            if extra_stats:
                common.add_perf_group_results(results, extra_group_name, extra_stats, include_legacy_keys=False)
    except RuntimeError as exc:
        print(f'Warning: cache profiling unavailable: {exc}')
        search_start = time.time()
        labels, distances = index.search_knn_parallel(
            repeated_queries,
            params['k'],
            params['nprobe'],
            num_threads,
            collect_search_stats,
        )
        search_time = time.time() - search_start
    finally:
        common.wait_for_perf(perf_session)

    results['search_time'] = search_time
    results['qps'] = len(repeated_queries) / search_time if search_time > 0 else float('inf')
    results['avg_query_latency_ms'] = (search_time * 1000.0 / len(repeated_queries)) if repeated_queries.size else 0.0
    print(f'Search time: {search_time:.4f} seconds')
    print(f'QPS: {results["qps"]:.2f}')
    print(f'Average query latency: {results["avg_query_latency_ms"]:.4f} ms')

    if collect_search_stats:
        stats_dict = index.get_stats() if hasattr(index, 'get_stats') else index.stats()
        if isinstance(stats_dict, dict):
            results.update({
                'dist_comps': stats_dict.get('dist_comps'),
                'lists_probed': stats_dict.get('lists_probed'),
                'vectors_scanned': stats_dict.get('vectors_scanned'),
                'avg_list_size': stats_dict.get('avg_list_size'),
            })
            for key, value in stats_dict.items():
                if key.startswith('horizontal_selected_') or key.startswith('replay_pid_stream_'):
                    results[key] = value

    if ground_truth is not None:
        recall_labels, _ = index.search_knn_parallel(
            np.ascontiguousarray(query_vectors[:len(ground_truth)], dtype=np.float32),
            params['k'],
            params['nprobe'],
            min(num_threads, len(ground_truth)),
            False,
        )
        recall = common.calculate_recall(ground_truth, recall_labels, params['k'])
        results['recall'] = recall
        print(f'Recall@{params["k"]}: {recall:.4f}')
    else:
        results['recall'] = 'N/A'
        print(f'Recall@{params["k"]}: N/A (ground truth not computed)')

    # try:
    #     if hasattr(index, 'flush'):
    #         index.flush()
    # except Exception as exc:
    #     print(f'Warning: failed to flush IVF index: {exc}')

    # if library == 'predicache' and predicache_py is not None and hasattr(predicache_py, 'flush_buffer_manager'):
    #     try:
    #         predicache_py.flush_buffer_manager()
    #     except Exception as exc:
    #         print(f'Warning: failed to flush PrediCache buffer manager: {exc}')

    print('\nIVF Benchmark Summary')
    print(f'  Config: {config_name}')
    print(f'  Library: {library}')
    print(f'  Build Time: {results["build_time"]:.4f} s')
    print(f'  Search Time: {results["search_time"]:.4f} s')
    print(f'  QPS: {results["qps"]:.2f}')
    print(f'  Avg Query Latency (ms): {results["avg_query_latency_ms"]:.4f}')
    print(f'  Recall@{params["k"]}: {results["recall"]}')
    print('RESULT_JSON: ' + json.dumps(results, sort_keys=True, default=str))

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Calico, PDX, and PrediCache IVF-family configurations')
    parser.add_argument('--config', type=str, required=True, help='Configuration name')
    parser.add_argument('--library', type=str, choices=('calico', 'predicache', 'pdx'), default=None,
                        help='Override library selection; defaults to config-name inference')
    parser.add_argument('--threads', type=int, default=1, help='Search thread count')
    parser.add_argument('--build_threads', type=int, default=None, help='Index build thread count')
    parser.add_argument('--dataset', type=str, default=HF_DEFAULT_LABEL,
                        help='Dataset label for reporting.')
    parser.add_argument('--data_file', type=str, default=os.environ.get('IVF_DATA_FILE'),
                        help='Path to the local ANN benchmark HDF5 dataset file')
    parser.add_argument('--dataset_url', type=str, default=os.environ.get('IVF_DATASET_URL', DEFAULT_DATASET_URL),
                        help='Source URL recorded in benchmark metadata')
    parser.add_argument('--base_limit', type=int, default=1_000_000,
                        help='Maximum number of base vectors to consume from the local dataset; 0 uses the full base split')
    parser.add_argument('--query_count', type=int, default=0,
                        help='Number of query vectors to use from the local dataset; 0 uses the full query set')
    parser.add_argument('--train_count', type=int, default=0,
                        help='Training vectors sampled from the beginning of the base split; 0 derives as 10%% of the effective dataset size')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Build batch size; 0 chooses an automatic batch size')
    parser.add_argument('--kmeans_iters', type=int, default=8, help='K-means iterations used for IVF training')
    parser.add_argument('--num_clusters', type=int, default=0,
                        help='Number of IVF clusters; 0 derives as floor(sqrt(effective dataset size))')
    parser.add_argument('--nprobe', type=int, default=16,
                        help='Number of IVF clusters to probe during search; clamped to num_clusters')
    parser.add_argument('--k', type=int, default=10, help='Top-k search result count')
    parser.add_argument('--warmup_queries', type=int, default=256, help='Warmup queries before timed search')
    parser.add_argument('--skip_recovery', action='store_true', help='Force a rebuild instead of reusing on-disk state')
    parser.add_argument('--compute_recall', action='store_true', help='Compute brute-force recall on a small query subset')
    parser.add_argument('--recall_queries', type=int, default=0,
                        help='Number of queries for recall evaluation; 0 uses the full active query set')
    parser.add_argument('--gt_batch_size', type=int, default=4096, help='Base chunk size for brute-force recall')
    args = parser.parse_args()

    if not args.data_file:
        parser.error('--data_file is required. Use exp_ivf.sh to download the dataset or pass a local HDF5 file path.')

    library = resolve_library(args.config, args.library)
    with LocalANNBenchmarkDataset(args.data_file) as dataset:
        base_count, query_count, dim = load_local_embeddings(dataset, args)
        query_vectors = dataset.get_query_slice(0, query_count)

        num_clusters, nprobe, train_count = resolve_ivf_parameters(base_count, args)
        print(
            f'Resolved IVF parameters: num_clusters={num_clusters}, '
            f'nprobe={nprobe}, train_count={train_count}'
        )

        ground_truth = None
        recall_queries = args.recall_queries if args.recall_queries > 0 else len(query_vectors)
        recall_queries = min(recall_queries, len(query_vectors))
        if recall_queries > 0:
            can_use_precomputed_ground_truth = base_count >= dataset.total_base
            if can_use_precomputed_ground_truth:
                ground_truth = dataset.get_ground_truth_slice(recall_queries, args.k)
                if ground_truth is not None:
                    print(f'Loaded precomputed ground truth for {recall_queries} queries from {dataset.file_path}')
            elif args.compute_recall:
                print(
                    'Base set is truncated; recomputing exact ground truth '
                    f'over the first {base_count} base vectors.'
                )
            else:
                print(
                    'Recall ground truth unavailable: the indexed base set is truncated '
                    f'({base_count}/{dataset.total_base}). Precomputed neighbors only match the full dataset; '
                    'pass --compute_recall to brute-force over the truncated base set.'
                )

            if ground_truth is None and args.compute_recall:
                print(f'Computing brute-force ground truth for {recall_queries} queries...')
                ground_truth = compute_exact_ground_truth(
                    dataset,
                    base_count,
                    np.ascontiguousarray(query_vectors[:recall_queries], dtype=np.float32),
                    args.k,
                    args.gt_batch_size,
                )

        params = {
            'dataset_label': args.dataset,
            'dataset_url': args.dataset_url,
            'num_clusters': num_clusters,
            'nprobe': nprobe,
            'k': args.k,
            'train_count': train_count,
            'kmeans_iters': args.kmeans_iters,
            'batch_size': args.batch_size,
            'skip_recovery': args.skip_recovery,
            'enable_prefetch': os.environ.get('PREFETCH', '1') == '1',
            'retrain_interval': 0,
            'warmup_queries': args.warmup_queries,
        }

        benchmark_ivf_config(
            dataset,
            base_count,
            query_vectors,
            params,
            args.threads,
            args.config,
            library,
            ground_truth=ground_truth,
            build_threads=args.build_threads,
        )


if __name__ == '__main__':
    main()