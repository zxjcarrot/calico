# Calico vs Mmap: Understanding the TLB Advantage

## Summary

**Mmap:** 
- Direct translation: Page ID (virtual address) → Physical Address
- Forced to use 4KB pages due to eviction granularity constraints
- Poor TLB coverage: 0.006% of a 64GB buffer pool (constant TLB misses)

**Calico:**
- Indirect translation: Page ID → Virtual Address (array lookup) → Physical Address  
- Uses 2MB huge pages (eviction decoupled from physical memory)
- Superior TLB coverage: 3.1% of buffer pool (**512× improvement**)

**The Bottom Line:** Calico's huge page benefits outweigh array indirection costs. The key is decoupling logical page eviction from physical memory management.

## The TLB Bottleneck

### TLB Fundamentals

The Translation Lookaside Buffer (TLB) is a small CPU cache storing virtual-to-physical address translations. Modern CPUs have severely limited TLB capacity:
- Intel Xeon: ~64 L1 entries, ~1536 L2 entries (4KB pages)
- AMD EPYC: ~72 L1 entries, ~2048 L2 entries (4KB pages)

**TLB capacity is measured in entries, not bytes.** Each entry caches one page translation regardless of page size:
- **4KB pages**: 1000 entries × 4KB = 4MB coverage (0.006% of 64GB buffer)
- **2MB huge pages**: 1000 entries × 2MB = 2GB coverage (3.1% of 64GB buffer)

**Result: 512× better TLB coverage with huge pages.**

### The Cost of TLB Misses

A TLB miss triggers a hardware page table walk traversing 4 levels of dependent pointer dereferences:
- **Cold page tables (DRAM)**: ~400 cycles (4 levels × 100 cycles each)
- **Warm page tables (L3 cache)**: ~100-120 cycles (serialized pointer chasing)
- **Compare to L1 cache hit**: ~4 cycles

For database workloads with random access patterns (B-tree lookups, hash probes, graph traversals), TLB misses impacts performance on traditional 4KB page systems.

## Why Mmap Cannot Use Huge Pages

Mmap-based buffer pools use direct virtual memory mapping where page IDs are virtual addresses:

```
Page ID (virtual address) → Physical Address (via OS page table)
```

**The Eviction Granularity Problem:**

Database pages are 4KB, but huge pages are 2MB (512× larger). To evict a single 4KB page from mmap:
- `munmap()` or `madvise(MADV_DONTNEED)` operates at OS page granularity
- **Cannot partially unmap a 2MB huge page** — must unmap entirely
- Unmapping the full 2MB region evicts all 512 database pages (losing hot data)
- Therefore, mmap is forced to use 4KB pages

**TLB Impact:**
- 4KB pages → only 0.006% TLB coverage → constant TLB misses
- Each miss costs 100+ cycles in page table walks

## Calico's Solution: Decoupled Translation

Calico adds an indirection layer that separates logical page eviction from physical memory mapping:

```
Page ID → Virtual Address → Physical Address(via OS Page Table)
```

**Two-stage translation:**
```cpp
PageState* state = &pageStateArray[pageId];   // User-space array lookup
Frame* frame = frameMem + state->getFrameId(); // Frame pointer arithmetic
// Then OS translates virtual address to physical via huge page table
```

**Key Innovation — Eviction Granularity Decoupling:**

- **Logical eviction** (4KB granularity): Atomic update to `PageState[pageId]` marks page as evicted
- **Physical memory** (2MB granularity): Huge pages remain mapped — no `munmap()` needed
- Result: Individual 4KB pages can be evicted without fragmenting huge page mappings

**TLB Benefits:**
- 2MB huge pages → 3.1% TLB coverage (512× better than mmap)
- No TLB shootdowns on eviction (physical pages stay mapped)
- No inter-processor interrupts or cross-core stalls

**Trade-off:** Array indirection adds ~8 cycles per access, but eliminates 100+ cycle TLB misses.

## How Calico Matches Mmap Performance

**The Apparent Paradox:** Calico adds an extra indirection layer (array lookup) before memory access. Shouldn't this make it slower than mmap's direct translation?

**The Resolution:** The indirection cost is negligible compared to TLB miss savings.

**Mmap Access Path (with 4KB pages):**
```
1. Access page ID (virtual address)
2. TLB lookup → Frequent MISS (only 0.006% coverage = high miss rate)
3. Page table walk (4 levels) = 100-400 cycles
4. Access data
```

**Calico Access Path (with 2MB huge pages):**
```
1. Array lookup: pageStateArray[pageId] = ~4-30 cycles (cache-dependent)
2. Frame lookup: frameMem + frameId = pointer arithmetic (< 1 cycle)
3. TLB lookup → Frequent HIT (3.1% coverage = much better hit rate)
4. Access data
```

**Why Calico Wins:**
- Mmap: 0.006% TLB coverage means most accesses trigger 100+ cycle page table walks
- Calico: 3.1% TLB coverage (512× better) dramatically reduces page table walks
- Array indirection (4-30 cycles) is cheaper than the TLB misses it helps avoid
- The huge pages on data frames eliminate the dominant performance bottleneck

**Key Insight:** The pageStateArray benefits from huge page backing:
- **Minimal TLB pressure**: Array itself uses huge pages → few TLB entries needed for metadata
- **No hash collisions**: Direct array indexing (vs. hash table pointer chasing)
- **Compact layout**: 8-byte entries improve cache density compared to hash table nodes

**Why random access is still fast:** Yes, random page IDs cause random array access with potential cache misses. But even a cache miss (~30 cycles) is cheaper than mmap's guaranteed TLB miss (100+ cycles). The huge pages on the data itself eliminate the dominant cost.
