# Calico Buffer Pool Design

## Goal
Calico aims to match the in-memory efficiency of `vmcache`’s `mmap` mode while
retaining user-space control over replacement and I/O—without relying on kernel
modules such as `exmap`.

`mmap`-based buffer pools defer all translation and replacement decisions to the
kernel: evicting a 4 KB page requires `madvise(DONTNEED)`. This triggers a TLB
shootdown for every victim, which caps I/O throughput on fast devices. The
`exmap` kernel module batches shootdowns, but deploying it is intrusive and
raises security concerns.

Calico keeps the entire buffer pool in user space and eliminates the
per-translation overhead normally associated with hash tables. The result is a
buffer manager with predictable eviction, excellent I/O scheduling, and a TLB
profile comparable to `mmap`.

## Indirection design
Calico treats the translation from page id to physical frame as a single index
into a preallocated array: one 8-byte entry per 4 KB page of storage. The array
never changes size, so lookups translate a PID with a simple pointer arithmetic
and no rehashing. Concurrency metadata lives in the same word, so threads avoid
the heavyweight synchronization that hash-table growth would impose.

1. **Fix**: `residentPtr(pid)` uses the packed state to decode the frame and
   return `frameMem + frame`. TLB hits come from the huge-page `frameMem`
   mapping, not the virtual file mapping.
2. **Acquire**: `acquireFrameForPid()` pops a frame, stores it in the packed
   state, and remembers the reverse mapping in `frameToPidArray` so faults can
   translate back to PIDs.
3. **Release**: `releaseFrame()` reads and clears the packed frame id, then
   pushes the frame back onto the partitioned freelist.

### Packed `PageState`
Every entry is a 64-bit `PageState` storing:

```
63           56 55               32 31                     0
+--------------+------------------+------------------------+
| lock state   | 24-bit version   | 32-bit frame id        |
+--------------+------------------+------------------------+
```

- **Lock state (8 bits)** preserves `Unlocked`, `MaxShared`, `Locked`, `Marked`,
  and `Evicted` semantics.
- **Version (24 bits)** enables optimistic reads.
- **Frame id (32 bits)** identifies the physical frame (up to 16 TB).

`PageState::setFrameValue`, `getFrameValue`, and `clearFrameValue` maintain the
low bits; `sameVersion` and `nextVersion` preserve the packed data when lock
state changes. Hash-table modes continue to operate unchanged.

## Huge page optimisation
Because every access first consults the indirection array, translation adds a
dependent memory read before touching the page data. Without mitigation this can
limit CPU parallelism—the processor must fetch the `PageState` entry and then
the frame, consuming extra TLB entries. Calico addresses this by placing both
`frameMem` and the `PageState` array on regions that request huge pages. The
much larger `frameMem` mapping benefits most, leaving more room in the TLB for
the comparatively small `PageState` array and keeping translation fast.

Applying this optimisation is substantially easier than in the original
`mmap`-based design. Promoting the virtual region to huge pages in `mmap`
prevents precise 4 KB eviction: `madvise(DONTNEED)` would either fail or drop
neighbouring hot pages that share the same 2 MB mapping. Calico performs the
translation entirely in user space, so the physical memory backing can safely
use huge pages while eviction still targets individual 4 KB frames. The result
is `mmap`-like locality without the TLB shootdowns that limit I/O throughput.

## Comparison
- **vs. `mmap`**: Calico avoids per-page TLB shootdowns because it doesn’t call
  `madvise(DONTNEED)`. Replacement remains under user control, and no kernel
  module is required.
- **vs. hash-table translators**: The packed array translator touches fewer
  cache lines and avoids probing overhead, bringing lookup cost close to direct
  pointer arithmetic.

## Usage
- `TRADITIONAL=1 TRADHASH=3` enables the new array single-access translator.
- `TRADHASH=1` and `TRADHASH=2` continue to select the sharded `std::unordered_map`
  and open-address translators respectively.
- `frameMem` and `pageState` can be allocated with huge-page advice to reduce
  TLB misses.

## Summary
Calico delivers a buffer pool that is as cache/TLB-friendly as `mmap`, yet keeps
full control over which pages remain resident and when I/O happens. The ability
to keep both `frameMem` and `PageState` on huge pages is the key to matching
`mmap`’s in-memory performance—translation stays in user space, but the TLB
footprint mirrors the direct-mapped approach. By embedding frame ids directly in
`PageState`, metadata lookups touch a single word, and the indirection array
remains the hot path. This combination gives Calico `mmap` latency traits
without kernel dependencies or hash-table complexity.
