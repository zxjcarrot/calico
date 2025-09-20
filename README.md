# vmcache

Code repository for SIGMOD'23 paper [Virtual-Memory Assisted Buffer Management](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/_my_direct_uploads/vmcache.pdf). This is the buffer manager implementation. [exmap](https://github.com/tuhhosg/exmap) is in a separate repository.

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

## More buffer pool knobs

* `TRADITIONAL=1` turns on the in-process buffer pool instead of the default vmcache/exmap fault-driven path. Pages are tracked with the lock-free frame array.
* `TRADHASH=1` (only relevant together with `TRADITIONAL=1`) uses the sharded `std::unordered_map` translator, retaining per-shard read/write locks.
* `TRADHASH=2` enables the new open-addressing translator, which keeps the same sharding layout but uses probing tables to avoid chaining overhead.
* Leave both knobs unset to keep the original vmcache behaviour (`TRADITIONAL=0`, `EXMAP=0`), or combine `EXMAP=1` with `TRADITIONAL=0` to exercise the kernel-assisted variant.

vmcache mmap: `TRADITIONAL=0 TRADHASH=0 RNDREAD=1 BLOCK=/mnt/data/vmcache/heapfile THREADS=1 DATASIZE=2e7 ./vmcache`

vmcache array: `TRADITIONAL=1 TRADHASH=0 RNDREAD=1 BLOCK=/mnt/data/vmcache/heapfile THREADS=1 DATASIZE=2e7 ./vmcache`

vmcache hash (unordered): `TRADITIONAL=1 TRADHASH=1 RNDREAD=1 BLOCK=/mnt/data/vmcache/heapfile THREADS=1 DATASIZE=2e7 ./vmcache`

vmcache hash (open addressing): `TRADITIONAL=1 TRADHASH=2 RNDREAD=1 BLOCK=/mnt/data/vmcache/heapfile THREADS=1 DATASIZE=2e7 ./vmcache`

## Example Command Lines

* TPC-C, 4 threads, 2 warehouses: `BLOCK=/dev/nvme0n1 THREADS=4 DATASIZE=2 ./vmcache`
* random read, 10 threads, 1 million tuples: `RNDREAD=1 BLOCK=/dev/md0 THREADS=10 DATASIZE=1e6 ./vmcache`

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
