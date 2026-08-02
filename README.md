# Calico
Calico is a high-performance buffer pool that combines the best aspects of different memory management approaches:

* In-memory performance that matches or exceeds mmap's direct virtual memory access
* Out-of-memory efficiency that rivals specialized kernel solutions like exmap
* User-space control without requiring custom kernel modules or complex deployment

The key insight is that user-space indirection with huge page optimization can eliminate the traditional performance trade-offs between direct memory mapping and flexible buffer pool management.

## Automated Experiments

The repository includes a comprehensive experiment suite for evaluating different buffer management strategies:

```bash
# Run full multi-threaded experiment suite
./exp_in_memory.sh
```

This script tests 5 configurations across 5 thread counts (1, 4, 8, 16, 32 threads):
- **calico_mmap**: Virtual memory cache using mmap (`TRADITIONAL=0`)
- **calico_array**: Traditional buffer pool with array translator (`TRADITIONAL=1 TRADHASH=0`)
- **calico_array_nohuge**: Array translator without huge pages for frame memory
- **calico_unordered_hash**: Traditional buffer pool with std::unordered_map (`TRADHASH=1`)
- **calico_openaddr_hash**: Traditional buffer pool with open addressing hash (`TRADHASH=2`)

Results are automatically organized in `exps/in-memory/` with subdirectories for each thread count and a comprehensive summary in `results_summary.txt`.

### Graph BFS Benchmark Suite

For graph workloads, a specialized benchmark is available:

```bash
# Run graph BFS experiment suite
./exp_in_memory_graph_bfs.sh
```

This script evaluates all buffer management strategies using a breadth-first search workload on a large graph:
- **calico_mmap**: Virtual memory cache using mmap
- **calico_array**: Traditional buffer pool with array translator  
- **calico_array_nohuge**: Array translator without huge pages
- **calico_unordered_hash**: Traditional buffer pool with std::unordered_map
- **calico_openaddr_hash**: Traditional buffer pool with open addressing hash
- **calico_lockfree_hash**: Traditional buffer pool with lock-free hash table

The graph BFS benchmark creates a synthetic graph with configurable size and performs breadth-first traversal, which exercises random memory access patterns typical of graph algorithms. Results include traversal time and detailed performance counters (LLC misses, DTLB misses, IPC).

Results are stored in `exps/in-memory/graph_bfs/` with automatic extraction to `graph_results_summary.csv` and detailed analysis in `graph_analysis.txt`.

### Focused Graph Prefetch Microbench

To isolate whether group prefetch helps array translation more than hash-table translation and PrediCache, run:

```bash
./exp_graph_prefetch_microbench.sh
```

This microbench runs graph BFS with `PREFETCH=0/1` for:
- `calico_array` (`TRADITIONAL=1 TRADHASH=0`)
- `calico_hash_open` (`TRADITIONAL=1 TRADHASH=2`)
- `predicache` (`predictivetranslation-sigmod2026/predicache_oltp_query`)

By default it executes multiple start nodes and repeats, then generates:
- `graph_prefetch_microbench_summary.csv` (per-run raw rows)
- `graph_prefetch_microbench_aggregate.csv` (avg/std/min/max by mechanism + prefetch)
- `graph_prefetch_effectiveness.csv` (per-mechanism prefetch gain)
- `array_prefetch_advantage.csv` (array gain vs hash and PrediCache)

Useful knobs:
- `GRAPH_NODES`, `GRAPH_STARTS` (comma-separated), `REPEATS`
- `EXP_BASE_DIR` to redirect outputs
- `PREDICACHE_IO_MODE=uring|libaio` (default `uring`)

## Pybind
```bash
git submodule update --init --recursive
python setup.py build_ext --inplace
python test.py
```

### HNSW sift1m and sift10m benchmark
We compare HNSW implemented in calico with HNSWLib.

First you need to download sift1m and sift10m datasets. You can choose to download from the official website, but I failed so I chose to download from the other websites.

sift1m: Download from here: https://huggingface.co/datasets/qbo-odp/sift1m
sift10m: Download from here: https://www.kaggle.com/datasets/sheikhomar/sift10m Please note that you only need the .mat file, so ignore the image subfolders.

Then, config the dataset path in the python file and run:
```bash
TRADITIONAL=1 TRADHASH=0 python benchmark_hnsw.py
```

## Environment Variables

* BLOCK: storage block device (e.g. /dev/nvme0n1 or /dev/md0); default=/tmp/bm
* VIRTGB: virtual memory allocation in GB (e.g., 1024), should be at least device size; default=16
* PHYSGB: physical memory allocation in GB = buffer pool size, should be less than available RAM; default=4
* EXMAP: if non-zero, use exmap interface, requires exmap kernel module; default=0
* BATCH: batch size for eviction in pages; default=64
* RUNFOR: benchmark run duration in seconds; default=30
* RNDREAD: if non-zero, run random read benchmark, otherwise TPC-C; default=0
* THREADS: number of threads; default=1
* DATASIZE: number of warehouses for TPC-C, number of tuples for random read benchmark; default=10
* **PERF_CSV**: if set to a filename, outputs performance counters to CSV file
* **TX_CSV**: if set to a filename, outputs transaction details to CSV file

## More buffer pool knobs

* `TRADITIONAL=1` turns on the in-process buffer pool instead of the default vmcache/exmap fault-driven path. Pages are tracked with the lock-free frame array.
* `TRADHASH=1` (only relevant together with `TRADITIONAL=1`) uses the sharded `std::unordered_map` translator, retaining per-shard read/write locks.
* `TRADHASH=2` enables the new open-addressing translator, which keeps the same sharding layout but uses probing tables to avoid chaining overhead.
* `TRADHASH=3` selects the single-access array translator, embedding the frame id directly into the page state to minimize metadata fetches.
* Leave both knobs unset to keep the original vmcache behaviour (`TRADITIONAL=0`, `EXMAP=0`), or combine `EXMAP=1` with `TRADITIONAL=0` to exercise the kernel-assisted variant.

calico mmap: `TRADITIONAL=0 TRADHASH=0 RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile THREADS=1 DATASIZE=2e7 ./calico`

calico array (single access): `TRADITIONAL=1 TRADHASH=0 RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile THREADS=1 DATASIZE=2e7 ./calico`

calico hash (unordered): `TRADITIONAL=1 TRADHASH=1 RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile THREADS=1 DATASIZE=2e7 ./calico`

calico hash (open addressing): `TRADITIONAL=1 TRADHASH=2 RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile THREADS=1 DATASIZE=2e7 ./calico`

calico array (double access): `TRADITIONAL=1 TRADHASH=3 RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile THREADS=1 DATASIZE=2e7 ./calico`

Graph BFS benchmark: `GRAPH=1 GRAPHN=10000 GRAPHSTART=0 ./calico`

* `GRAPHN` controls the number of nodes/pages in the graph (default 1024).
* `GRAPHSTART` selects the starting node index for the breadth-first search (default 0).

## Example Command Lines

* TPC-C, 4 threads, 2 warehouses: `BLOCK=/dev/nvme0n1 THREADS=4 DATASIZE=2 ./calico`
* random read, 10 threads, 1 million tuples: `RNDREAD=1 BLOCK=/dev/md0 THREADS=10 DATASIZE=1e6 ./calico`

## Dependencies and Configuration

We need the libaoi library. On Ubuntu: `sudo apt install libaio-dev`.
You will probably also need to set `vm.overcommit_memory = 1` in `/etc/sysctl.conf`. Otherwise larger values of VIRTGB will not work.

If you want to use EXMAP, you need the [exmap kernel module](https://github.com/tuhhosg/exmap).

## Citation

Please cite the paper as follows:

```
@inproceedings{vmcache,
  author    = {Viktor Leis and Adnan Alhomssi and Tobias Ziegler and Yannick Loeck and Christian Dietrich},
  title     = {Virtual-Memory Assisted Buffer Management},
  booktitle = {SIGMOD},
  year      = {2023},
}
```

## Low-Hanging Fruit (TODO)

* use C++ wait/notify to handle lock contention instead of spinning
* implement free space management for storage
