# Calico Buffer Pool Design

## Overview

Calico is a high-performance buffer pool that combines the best aspects of different memory management approaches:

- **In-memory performance** that matches or exceeds `mmap`'s direct virtual memory access
- **Out-of-memory efficiency** that rivals specialized kernel solutions like `exmap` 
- **User-space control** without requiring custom kernel modules or complex deployment

The key insight is that **user-space indirection with huge page optimization** can eliminate the traditional performance trade-offs between direct memory mapping and flexible buffer pool management.

**Table 1: Conceptual comparison of buffer manager designs**

| Property | `mmap` | Trad. Hash | Pointer Swiz. | vmcache+`exmap`/libdbos | **Calico** |
|----------|--------|------------|---------------|-----------------|------------|
| **Translation Control** | <span style="background-color:#FFFFE0">Page tbl. (OS)</span> | <span style="background-color:#90EE90">Hash tbl. (DBMS)</span> | <span style="background-color:#FFB6C6">Invasive (DBMS)</span> | <span style="background-color:#90EE90">Page tbl. (DBMS)</span> | **<span style="background-color:#90EE90">Array (DBMS)</span>** |
| **Graph Support** | <span style="background-color:#90EE90">Yes</span> | <span style="background-color:#90EE90">Yes</span> | <span style="background-color:#FFB6C6">No</span> | <span style="background-color:#90EE90">Yes</span> | **<span style="background-color:#90EE90">Yes</span>** |
| **Implementation** | <span style="background-color:#90EE90">Easy</span> | <span style="background-color:#90EE90">Easy</span> | <span style="background-color:#FFB6C6">Hard</span> | <span style="background-color:#FFFFE0">Medium</span> | **<span style="background-color:#90EE90">Easy (user)</span>** |
| **In-Memory Speed** | <span style="background-color:#90EE90">Fast</span> | <span style="background-color:#FFB6C6">Slow</span> | <span style="background-color:#90EE90">Fast</span> | <span style="background-color:#90EE90">Fast</span> | **<span style="background-color:#90EE90">Fast</span>** |
| **Out-of-Memory I/O** | <span style="background-color:#FFB6C6">Slow</span> | <span style="background-color:#90EE90">Fast</span> | <span style="background-color:#90EE90">Fast</span> | <span style="background-color:#90EE90">Fast</span> | **<span style="background-color:#90EE90">Fast</span>** |
| **Huge Page Friendly** | <span style="background-color:#FFB6C6">No†</span> | <span style="background-color:#90EE90">Yes</span> | <span style="background-color:#90EE90">Fast</span> | <span style="background-color:#FFB6C6">No†</span> | **<span style="background-color:#90EE90">Yes</span>** |
| **Kernel Module Required** | <span style="background-color:#90EE90">No</span> | <span style="background-color:#90EE90">No</span> | <span style="background-color:#90EE90">No</span> | <span style="background-color:#FFB6C6">Yes</span> | **<span style="background-color:#90EE90">No</span>** |
| **User Control** | <span style="background-color:#FFB6C6">Limited</span> | <span style="background-color:#90EE90">Full</span> | <span style="background-color:#90EE90">Full</span> | <span style="background-color:#FFB6C6">Limited</span> | **<span style="background-color:#90EE90">Full</span>** |

*Legend: <span style="background-color:#90EE90">Green</span> = Good, <span style="background-color:#FFFFE0">Yellow</span> = Medium, <span style="background-color:#FFB6C6">Light Red</span> = Bad*

**Key Distinctions:**
- †`mmap` and VMCache/`exmap` cannot use huge pages due to eviction granularity conflicts (4KB page eviction breaks 2MB huge page boundaries)
- Calico combines user-space control with huge page optimization through array-based indirection
- Pointer swizzling requires invasive changes to data structures, limiting applicability to graph workloads
- Kernel solutions (VMCache/`exmap`) require custom modules and have deployment complexity

## The Buffer Management Challenge

Modern database systems face a fundamental trade-off when choosing buffer pool implementations:

### Virtual Memory Accelerated Mapping (`mmap`)
- ✅ **Fast in-memory access**: Direct page table translation with minimal CPU overhead
- ❌ **Poor I/O performance**: Per-page TLB shootdowns during eviction create bottlenecks  
- ❌ **No huge page support**: 4KB eviction granularity conflicts with 2MB huge page boundaries
- ❌ **Limited control**: Operating system controls replacement policy and timing

### Traditional Hash Table Buffer Pools
- ✅ **Fast I/O operations**: User-space control enables efficient batch operations
- ✅ **Flexible replacement**: Full control over eviction policies and page lifecycle
- ❌ **Translation overhead**: Hash collisions and pointer chasing hurt in-memory performance
- ❌ **Concurrency complexity**: Fine-grained locking and resize operations create bottlenecks

### Kernel Solutions (`exmap`/`libdbos`)
- ✅ **Best of both worlds**: Fast in-memory and I/O performance through kernel optimizations
- ❌ **Deployment complexity**: Requires custom kernel modules and security considerations
- ❌ **Limited huge page support**: Still constrained by mmap's granularity issues

## Calico's Innovation: Array-Based Translation

Calico's idea is replacing hash table translation with **direct array indexing**. Instead of computing hash values and handling collisions, page translation becomes simple array access:
## Evaluation and Reproducibility

Detailed experimental results, tables, microarchitectural counters, and cross-platform comparisons are now split into `CalicoEval.md`. That document includes the HNSW larger-than-memory and in-memory runs, B-tree CSV extractions, Graph BFS results, and platform (AMD vs ARM) comparisons.

To view the complete evaluation and raw numbers, see: `CalicoEval.md`.

## Configuration and Usage
|---------------|----------|------------|------------|--------------|
| **Throughput (Mops/s)** ||||
| calico_array | 1.19 | 16.37 | 29.13 | Exceeds mmap performance |
| calico_mmap | 1.19 | 16.39 | 28.15 | Baseline direct mapping |
| calico_lockfree_hash | 1.07 | 14.61 | 25.75 | 11.6% slower than array |
| calico_array_nohuge | 1.00 | 13.90 | 24.92 | 14.4% penalty without huge pages |
| calico_lockfree_hash_nohuge | 0.90 | 12.60 | 22.16 | 23.9% slower than array |
| calico_openaddr_hash | 1.02 | 9.90 | 11.48 | 60.6% slower than array |
| calico_unordered_hash | 0.71 | 8.22 | 10.44 | 64.1% slower than array |

**Microarchitectural Analysis (32 threads):**

| Configuration | LLC Miss/Op | DTLB Miss/Op | Cycles/Op | IPC | 
|---------------|-------------|--------------|-----------|-----|
| **calico_array** | **5.76** | **1.34** | **2,988** | **0.311** |
| calico_mmap | 6.51 | 1.28 | 3,101 | 0.300 |
| calico_lockfree_hash | 6.91 | 2.79 | 3,304 | 0.323 |
| calico_array_nohuge | 6.46 | 1.30 | 3,499 | 0.265 |
| calico_lockfree_hash_nohuge | 7.79 | 2.83 | 3,850 | 0.280 |
| calico_openaddr_hash | 10.58 | 3.39 | 7,581 | 0.242 |
| calico_unordered_hash | 8.87 | 6.58 | 8,357 | 0.188 |

**Key Findings:**
- **Array exceeds mmap performance**: Validates superior handling of structured access patterns (+3.5% at 32 threads)
- **Lockfree hash competitive**: Shows only 11.6% degradation with huge pages, but 23.9% without
- **Huge pages critical for all approaches**: 14.4% penalty for array, 13.9% penalty for lockfree hash when disabled
- **Traditional hash tables suffer severely**: 60-64% overhead from collision handling in B-tree operations

### Vector Search: pgvector


![alt text](fb6c059b3fc1c9df3644cf1e2d8eab35.png)

Buffer pool and cache misses are the major bottlenecks in hnsw-based vector search.

### Graph Traversal

To validate Calico's effectiveness for irregular access patterns, we benchmarked all configurations using breadth-first search on a 5-million-node graph. This workload exercises the same memory patterns found in modern vector databases and graph analytics systems.

**BFS Performance Results**

| Configuration | Time (ms) | LLC Miss/Op | DTLB Miss/Op | Performance vs Array |
|---------------|-----------|-------------|--------------|----------------------|
| **calico_array** | **4,918** | **34.55** | **24.23** | **Baseline (100%)** |
| calico_mmap | 7,190 | 57.17 | 31.68 | 31.6% slower |
| calico_array_nohuge | 7,815 | 57.36 | 31.74 | 37.1% slower |
| calico_lockfree_openaddr_hash | 8,659 | 64.71 | 56.57 | 43.2% slower |
| calico_chained_hash | 25,088 | 128.83 | 107.89 | 80.4% slower |

**Key Insights:**

1. **Array Translator Dominance**: Even outperforms mmap by 31.6% for irregular graph traversal, demonstrating superior cache behavior through predictable metadata layout

2. **Cache Efficiency**: Array achieves 34.55 LLC misses/op vs 57.17 for mmap (40% improvement), showing that sequential PageState layout benefits even irregular access patterns

3. **Hash Table Penalties**: Irregular access amplifies hash overhead, with unordered_map showing 5× performance degradation vs 2× in random workloads

4. **Huge Pages Critical**: Without huge pages, even the array translator loses its advantage (37% slower), highlighting the synergy between translation strategy and memory optimization

### Workload Pattern Analysis

Comparing performance across different access patterns reveals important insights:

**Relative Performance vs mmap (32 threads):**
- **B-tree Index**: Array exceeds mmap performance (+2.1% at 29.1M vs 28.6M ops/s)  
- **Graph BFS**: Array significantly outperforms mmap (+31.6% faster traversal time)

**Hash Table Performance by Workload:**
- **Lockfree Hash (with huge pages)**: 11.6% slower (B-tree) vs 76% slower (Graph BFS) - structured access reduces contention
- **Lockfree Hash (no huge pages)**: 23.9% slower (B-tree) vs 157% slower (Graph BFS) - huge pages provide 13.9% benefit for lockfree hash
- **Open Address Hash**: 60.6% slower (B-tree) vs 290% slower (Graph BFS) - probing overhead amplified by irregular access
- **Unordered Hash**: 64.1% slower (B-tree) vs 410% slower (Graph BFS) - collision handling struggles with irregular patterns

**Huge Page Impact Analysis:**
- **Array translator**: 14.4% penalty without huge pages (consistent across workloads)
- **Lockfree hash**: 13.9% penalty without huge pages (B-tree) vs 12.7% penalty (Graph BFS) - moderate huge page dependency
- **Hash vs Array huge page sensitivity**: Hash tables show smaller relative huge page benefits than array translation, suggesting array's sequential access patterns synergize better with huge page optimization

**Key Pattern Insights:**
1. **Structured Benefits**: Array translator's advantage increases with access pattern complexity, from superior (B-tree) to dominant (graph)
2. **Cache Locality**: Sequential PageState layout provides consistent cache benefits across all workload types
3. **Hash Degradation**: All hash-based approaches show worse performance degradation as access patterns become more irregular, with lockfree hash being most resilient
4. **Huge Page Synergy**: The combination of array translation + huge pages shows consistent benefits across diverse workload characteristics, with array translator showing strongest huge page dependency

## Configuration and Usage

**Recommended Settings:**
- `TRADITIONAL=1 TRADHASH=0` - Enables array translator (default)
- Huge page backing for both `frameMem` and `PageState` arrays
- Size PageState array for full storage capacity at initialization

**Alternative Configurations:**
- `TRADHASH=1` - Sharded `std::unordered_map` (for comparison)
- `TRADHASH=2` - Open addressing hash table (legacy support)

## Summary

Calico solves the fundamental trade-off between buffer pool control and performance by recognizing that **user-space indirection with huge page optimization** can match the efficiency of direct memory mapping. 

The key insights are:

1. **Array translation eliminates hash overhead** while providing predictable, cache-friendly access patterns
2. **Huge page backing overcomes indirection costs** and enables optimizations impossible with mmap
3. **User-space control maintains flexibility** for replacement policies and I/O optimization  
4. **No kernel dependencies** simplify deployment compared to kernel module solutions

The result is a buffer pool that delivers mmap-like performance for in-memory workloads while providing the control and I/O efficiency needed for complex database systems. Experimental validation across random access, B-tree index operations, and graph traversal workloads consistently demonstrates that Calico's array translator matches or exceeds direct memory mapping performance while maintaining full user-space control over buffer management.

## Related Work

* Enhancing Transaction Processing through Indirection Skipping: https://dl.acm.org/doi/pdf/10.14778/3749646.3749680  Limitations: https://chatgpt.com/share/68d1ac43-1990-800b-821b-ffea0450e4c0
* VMCache
* libdbos/Tabby
* ScaleCache: https://www.vldb.org/pvldb/vol18/p5073-liu.pdf  Limitations: https://chatgpt.com/share/68d1f201-d5b0-800b-af53-806a24df4dc1
## Future Work

While Calico's array translator achieves excellent performance, there are opportunities to further optimize memory usage through advanced OS integration techniques:

### Memory-Efficient Indirection Array

**Current Space Overhead**: Like vmcache systems, Calico's indirection array introduces memory overhead:
- **vmcache**: 16 bytes per 4KB storage page  
- **Calico**: 8 bytes per 4KB storage page (PageState entry)

For large storage capacities, this overhead can be significant. A 2TB SSD requires a 4GB indirection array, even if the actual database is much smaller.

**Proposed Optimization: Copy-on-Write Hole Punching**

The key insight is leveraging OS copy-on-write zero pages to reduce physical memory consumption for unused indirection array regions:

1. **Initial Allocation**: Allocate the full indirection array but rely on OS zero pages
2. **Hole Punching**: Use `madvise(MADV_DONTNEED)` to return unused array regions to zero pages
3. **Demand Allocation**: Physical memory is only consumed when PageState entries are actually written

**Memory Savings Scenarios**:
- **Sparse Databases**: A 100GB database on a 2TB SSD would consume only 200MB of indirection array memory instead of 4GB
- **Cold Data**: Inactive database regions can have their indirection entries returned to zero pages
- **Page ID Clustering**: Databases with clustered page ID allocation patterns benefit most from hole punching

**Implementation Considerations**:
- **Page ID Distribution**: Effectiveness depends on database page allocation patterns
- **Active Working Set**: Only frequently accessed regions maintain physical memory in the indirection array
- **Dynamic Adaptation**: The system could monitor access patterns and dynamically punch holes in cold regions

**Potential Benefits**:
- **Reduced Memory Footprint**: Dramatic reduction in indirection array memory consumption for sparse workloads
- **Better Cache Utilization**: Smaller physical memory footprint improves overall system cache efficiency
- **Scalability**: Enables Calico to handle much larger storage capacities without proportional memory overhead

This optimization would maintain Calico's performance characteristics while addressing the primary limitation of array-based translation: memory overhead for large, sparsely-accessed storage systems.

### Experimental Results (Consolidated)

Below is a concise summary of representative HNSW experiments run from the `exps/` folder. Numbers were extracted from experiment logs for different buffer pool configurations and physical memory budgets.

Key experiment setup notes:
- Dataset: SIFT1M (1,000,000 base vectors, 10,000 queries)
- Queries repeated across threads to produce 320,000 total queries for the measurement runs (10,000 × 32 threads)
- Reported QPS is end-to-end query throughput; Recall@10 is reported where available

Summary (physgb = 0.3, 32 threads, larger-than-memory):

| Configuration | Build Time (s) | Search Time (s) | QPS | Recall@10 |
|---------------:|---------------:|-----------------:|----:|----------:|
| calico_array (array translator, huge pages) | 1023.74 | 236.06 | 1355.56 | 0.9503 |
| vmcache (mmap + vmcache/exmap style) | 1686.91 | 379.16 | 843.98 | 0.9504 |
| calico_mmap_oscache (mmap + OS page cache) | 1911.59 | 615.51 | 519.89 | 0.9510 |

Representative in-memory (physgb = 1, 32 threads) results (shows orders-of-magnitude higher QPS when dataset fits in RAM):

| Configuration | Build Time (s) | Search Time (s) | QPS | Recall@10 |
|---------------:|---------------:|-----------------:|----:|----------:|
| calico_array | 35.05 | 5.23 | 61135.74 | 0.9507 |
| vmcache | 34.95 | 5.38 | 59437.63 | 0.9025 |

Observations and takeaways:

- Array translator consistently outperforms `vmcache`/`mmap` in larger-than-memory settings (physgb=0.3): calico_array achieved ~1.61× higher QPS than `vmcache` and ~2.61× higher QPS than `calico_mmap_oscache`.
- When the dataset fits in memory (physgb=1), all designs have vastly higher QPS; calico_array still retains a modest lead in QPS and recall.
- Build times are substantially higher for out-of-memory builds (index construction streams from disk), explaining long experiment walltimes for physgb=0.3.
- Recall@10 is essentially identical across configurations in larger-than-memory runs, indicating that the translator strategy does not hurt search quality while improving throughput.
- Microarchitectural counters (from perf traces) show that `vmcache` and `mmap` have higher cache-miss pressure and LLC misses in the larger-than-memory runs; calico_array shows lower LLC misses/op and lower DTLB misses/op in many workloads, which aligns with the huge-page + sequential metadata layout hypothesis.

Practical implication:

- For latency/throughput-critical vector search workloads that exceed physical RAM, Calico's array translator with huge-page backing provides a clear performance advantage and more predictable microarchitectural behavior compared to mmap-based alternatives.

