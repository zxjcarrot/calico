#include "calico.hpp"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cctype>
#include <cassert>
#include <chrono>
#include <cmath>

// Global configuration for row size (initialized from environment variable)
static size_t g_numColumns = 10;  // Default: 10 columns × 8 bytes = 80 bytes per row
static size_t g_rowSize = 80;     // Default row size in bytes
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "perf_util.hpp"

// Thread-local variables
extern __thread uint16_t workerThreadId;

using namespace std;

// Global BufferManager instance
extern BufferManager* bm_ptr;

// Benchmark configuration
struct OLAPConfig {
    enum class OpType {
        FilteredCount,       // Count with filter (read payload + check condition)
        NonFilteredCount,    // Count using metadata only (page header)
        Sum,         // Sum aggregation with filter
    };
    
    enum class LayoutType {
        Columnar,            // Columnar layout (Structure of Arrays)
        Row,                 // Row-oriented layout (Array of Structures)
    };

    u64 numPages;           // Total number of pages available
    u64 scanPages;          // Number of pages to scan per query (default 25% of total)
    u64 scanStartPage;      // Starting page for scan (randomized)
    u64 runForSec;          // Duration in seconds
    unsigned nthreads;      // Number of worker threads
    OpType opType;          // Operation type
    LayoutType layoutType;  // Page layout type
    double selectivity;     // Filter selectivity (0.0 to 1.0)
    bool enablePerf;        // Enable performance counters
    
    OLAPConfig() : numPages(1000000), scanPages(0), scanStartPage(0), 
                   runForSec(30), nthreads(1), 
                   opType(OpType::FilteredCount), layoutType(LayoutType::Columnar),
                   selectivity(0.1), enablePerf(false) {
        // Default: scan 25% of total pages
        scanPages = numPages / 4;
    }
};

// Page layout for OLAP workload - COLUMNAR layout (Structure of Arrays)
struct OLAPPageColumnar {
    // Page header with metadata for non-filtered operations
    u64 recordCount;        // Number of records in this page
    u64 sumValue;           // Pre-computed sum for non-filtered aggregation
    PID nextPage;           // Link to next page (for sequential scan)
    u64 pageId;             // Page identifier
    bool dirty;
    u8  padding1[7];
    u64 padding2[3];         // Padding to align data
    
    // Payload data - COLUMNAR layout (Structure of Arrays)
    // Header is 64 bytes, so data size is pageSize - 64
    static constexpr size_t headerSize = 64;
    static constexpr size_t dataSize = pageSize - headerSize;  
    // numColumns and recordSize are now configurable via g_numColumns and g_rowSize
    static size_t numColumns() { return g_numColumns; }
    static size_t recordSize() { return g_rowSize; }
    static size_t maxRecords() { return dataSize / g_rowSize; }

    u8 data[dataSize];
    
    // Columnar accessors - each column stored contiguously
    // All 10 columns are u64 values
    u64* column(size_t colIdx) { 
        return reinterpret_cast<u64*>(data + colIdx * maxRecords() * sizeof(u64)); 
    }
    const u64* column(size_t colIdx) const { 
        return reinterpret_cast<const u64*>(data + colIdx * maxRecords() * sizeof(u64)); 
    }
    
    // Convenience accessors for specific columns
    u64* col0() { return column(0); }  // Often used as key
    const u64* col0() const { return column(0); }
    
    u64* col1() { return column(1); }  // Often used for filtering/aggregation
    const u64* col1() const { return column(1); }
    u64* col5() { return column(5); }  // Often used for filtering/aggregation
    const u64* col5() const { return column(5); }
};

static_assert(sizeof(OLAPPageColumnar) == pageSize, "OLAPPageColumnar must be pageSize bytes");

// Page layout for OLAP workload - ROW-ORIENTED layout (Array of Structures)
struct OLAPPageRow {
    // Page header with metadata for non-filtered operations
    u64 recordCount;        // Number of records in this page
    u64 sumValue;           // Pre-computed sum for non-filtered aggregation
    PID nextPage;           // Link to next page (for sequential scan)
    u64 pageId;             // Page identifier
    bool dirty;
    u8  padding1[7];
    u64 padding2[3];         // Padding to align data
    
    // Header is 64 bytes
    static constexpr size_t headerSize = 64;
    static constexpr size_t dataSize = pageSize - headerSize;
    
    // Row-oriented tuple structure
    struct Tuple {
        u64 columns[128];  // Maximum: 4096 bytes / 8 bytes per u64 = 512 u64s, use 128 for safety
        
        // Access column by index (only first g_numColumns are valid)
        u64& operator[](size_t idx) { return columns[idx]; }
        const u64& operator[](size_t idx) const { return columns[idx]; }
    };
    
    // numColumns and tupleSize are now configurable
    static size_t numColumns() { return g_numColumns; }
    static size_t tupleSize() { return g_rowSize; }
    static size_t maxRecords() { return dataSize / g_rowSize; }
    
    u8 data[dataSize];
    
    // Row accessor - get tuple at index
    Tuple* tuple(size_t idx) {
        return reinterpret_cast<Tuple*>(data + idx * tupleSize());
    }
    const Tuple* tuple(size_t idx) const {
        return reinterpret_cast<const Tuple*>(data + idx * tupleSize());
    }
};

static_assert(sizeof(OLAPPageRow) == pageSize, "OLAPPageRow must be pageSize bytes");

// Type alias for backward compatibility
using OLAPPage = OLAPPageColumnar;

struct QueryStats {
    u64 pagesScanned = 0;
    u64 recordsCounted = 0;
    u64 sum = 0;
};

// Environment variable helper with default value
template<typename T>
static T envOr(const char* name, T defaultValue) {
    const char* env = getenv(name);
    if (!env) return defaultValue;
    
    if constexpr (std::is_same_v<T, bool>) {
        return atoi(env) != 0;
    } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, u64> || std::is_same_v<T, unsigned>) {
        return static_cast<T>(atoll(env));
    } else if constexpr (std::is_same_v<T, double>) {
        return atof(env);
    }
    return defaultValue;
}

// Initialize pages with linked structure (mimics B-tree leaf level)
PID initializePagesColumnar(u64 numPages, unsigned nthreads) {
    cerr << "Initializing " << numPages << " pages (columnar layout)..." << endl;
    
    std::vector<PID> pagePids;
    pagePids.reserve(numPages);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<u64> dist(1, 1000);
    
    // Allocate all pages first using bm.allocPage()
    for (u64 i = 0; i < numPages; i++) {
        Page* rawPage = bm.allocPage();
        PID pid = bm.toPID(rawPage);
        pagePids.push_back(pid);
        bm.unfixX(pid);
    }
    // Initialize pages sequentially (no parallelization)
    for (u64 i = 0; i < numPages; i++) {
        PID pid = pagePids[i];
        Page* rawPage = bm.fixX(pid);
        OLAPPageColumnar* page = reinterpret_cast<OLAPPageColumnar*>(rawPage);

        // Initialize header
        page->recordCount = OLAPPageColumnar::maxRecords();
        page->nextPage = (i + 1 < numPages) ? pagePids[i + 1] : BufferManager::invalidPID;
        page->pageId = i;
        page->sumValue = 0;

        // Initialize records in columnar layout - configurable number of columns per tuple
        for (size_t colIdx = 0; colIdx < OLAPPageColumnar::numColumns(); colIdx++) {
            u64* col = page->column(colIdx);
            for (size_t j = 0; j < OLAPPageColumnar::maxRecords(); j++) {
                if (colIdx == 0) {
                    // Column 0: unique key
                    col[j] = i * OLAPPageColumnar::maxRecords() + j;
                } else if (colIdx == 1) {
                    // Column 1: random value (used for filtering/aggregation)
                    col[j] = dist(rng);
                    page->sumValue += col[j];
                } else {
                    // Other columns: various data
                    col[j] = dist(rng);
                }
            }
        }

        rawPage->dirty = true;
        bm.unfixX(pid);
    }
    cout << "First page PID: " << pagePids.front() << endl;
    
    cerr << "Initialization complete. Space: " 
         << (bm.allocCount.load() * pageSize) / (float)BufferManager::gb << " GB" << endl;
    return pagePids.front();
}

// Initialize pages with row-oriented layout
PID initializePagesRow(u64 numPages, unsigned nthreads) {
    cerr << "Initializing " << numPages << " pages (row-oriented layout)..." << endl;
    
    std::vector<PID> pagePids;
    pagePids.reserve(numPages);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<u64> dist(1, 1000);
    
    // Allocate all pages first using bm.allocPage()
    for (u64 i = 0; i < numPages; i++) {
        Page* rawPage = bm.allocPage();
        PID pid = bm.toPID(rawPage);
        pagePids.push_back(pid);
        bm.unfixX(pid);
    }
    // Initialize pages sequentially (no parallelization)
    for (u64 i = 0; i < numPages; i++) {
        PID pid = pagePids[i];
        Page* rawPage = bm.fixX(pid);
        OLAPPageRow* page = reinterpret_cast<OLAPPageRow*>(rawPage);

        // Initialize header
        page->recordCount = OLAPPageRow::maxRecords();
        page->nextPage = (i + 1 < numPages) ? pagePids[i + 1] : BufferManager::invalidPID;
        page->pageId = i;
        page->sumValue = 0;

        // Initialize records in row-oriented layout - configurable number of columns per tuple
        for (size_t j = 0; j < OLAPPageRow::maxRecords(); j++) {
            OLAPPageRow::Tuple* t = page->tuple(j);
            for (size_t colIdx = 0; colIdx < OLAPPageRow::numColumns(); colIdx++) {
                if (colIdx == 0) {
                    // Column 0: unique key
                    t->columns[colIdx] = i * OLAPPageRow::maxRecords() + j;
                } else if (colIdx == 1) {
                    // Column 1: random value (used for filtering/aggregation)
                    t->columns[colIdx] = dist(rng);
                    page->sumValue += t->columns[colIdx];
                } else {
                    // Other columns: various data
                    t->columns[colIdx] = dist(rng);
                }
            }
        }

        rawPage->dirty = true;
        bm.unfixX(pid);
    }
    cout << "First page PID: " << pagePids.front() << endl;
    
    cerr << "Initialization complete. Space: " 
         << (bm.allocCount.load() * pageSize) / (float)BufferManager::gb << " GB" << endl;
    return pagePids.front();
}

// Worker function for filtered count - executes ONE query and returns
QueryStats workerFilteredCount(u64 startPid, u64 scanPages, double selectivity) {
    QueryStats stats;
    u64 localCount = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPage> guard(currentPid);
        OLAPPage* page = guard.ptr;
        // Scan records with filter - columnar access (using column 5)
        const u64* values = page->col5();
        u64 threshold = static_cast<u64>(1000 * (1.0 - selectivity));
        
        for (size_t i = 0; i < page->recordCount; i++) {
            // Simple filter: value > threshold
            if (values[i] > threshold) {
                localCount++;
            }
        }
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.recordsCounted = localCount;
    return stats;
}

// Worker function for non-filtered count (metadata only) - executes ONE query and returns
QueryStats workerNonFilteredCount(u64 startPid, u64 scanPages) {
    QueryStats stats;
    u64 localCount = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPage> guard(currentPid);
        OLAPPage* page = guard.ptr;
        
        // Only read metadata from header - no payload access
        localCount += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.recordsCounted = localCount;
    return stats;
}

// Worker function for sum - executes ONE query and returns
QueryStats workerSum(u64 startPid, u64 scanPages, double selectivity) {
    QueryStats stats;
    u64 localSum = 0;
    u64 localRecords = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPage> guard(currentPid);
        OLAPPage* page = guard.ptr;
        
        // Scan records with filter - columnar access (using column 5)
        const u64* values = page->col5();
        (void)selectivity;
        for (size_t i = 0; i < page->recordCount; i++) {
            // if (values[i] > threshold) {
                localSum += values[i];
            // }
        }

        localRecords += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.sum = localSum;
    stats.recordsCounted = localRecords;
    return stats;
}

// Worker function for non-filtered sum (metadata only) - executes ONE query and returns
QueryStats workerNonFilteredSum(u64 startPid, u64 scanPages) {
    QueryStats stats;
    u64 localSum = 0;
    u64 localRecords = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPage> guard(currentPid);
        OLAPPage* page = guard.ptr;
        
        // Only read pre-computed sum from header
        localSum += page->sumValue;
        localRecords += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.sum = localSum;
    stats.recordsCounted = localRecords;
    return stats;
}

// ========== ROW-ORIENTED WORKER FUNCTIONS ==========

// Worker function for filtered count (row format) - executes ONE query and returns
QueryStats workerFilteredCountRowFormat(u64 startPid, u64 scanPages, double selectivity) {
    QueryStats stats;
    u64 localCount = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPageRow> guard(currentPid);
        OLAPPageRow* page = guard.ptr;
        
        // Scan records with filter - row-oriented access
        u64 threshold = static_cast<u64>(1000 * (1.0 - selectivity));
        
        for (size_t i = 0; i < page->recordCount; i++) {
            const OLAPPageRow::Tuple* t = page->tuple(i);
            // Simple filter: column 5 > threshold
            if (t->columns[5] > threshold) {
                localCount++;
            }
        }
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.recordsCounted = localCount;
    return stats;
}

// Worker function for non-filtered count (row format, metadata only) - executes ONE query and returns
QueryStats workerNonFilteredCountRowFormat(u64 startPid, u64 scanPages) {
    QueryStats stats;
    u64 localCount = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPageRow> guard(currentPid);
        OLAPPageRow* page = guard.ptr;
        
        // Only read metadata from header - no payload access
        localCount += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.recordsCounted = localCount;
    return stats;
}

// Worker function for sum (row format) - executes ONE query and returns
QueryStats workerSumRowFormat(u64 startPid, u64 scanPages, double selectivity) {
    QueryStats stats;
    u64 localSum = 0;
    u64 localRecords = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPageRow> guard(currentPid);
        OLAPPageRow* page = guard.ptr;
        
        // Scan records with filter - row-oriented access
        (void)selectivity;
        for (size_t i = 0; i < page->recordCount; i++) {
            const OLAPPageRow::Tuple* t = page->tuple(i);
            // Sum column 5 values (with optional filter)
            // if (t->columns[5] > threshold) {
                localSum += t->columns[5];
            // }
        }

        localRecords += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.sum = localSum;
    stats.recordsCounted = localRecords;
    return stats;
}

// Worker function for non-filtered sum (row format, metadata only) - executes ONE query and returns
QueryStats workerNonFilteredSumRowFormat(u64 startPid, u64 scanPages) {
    QueryStats stats;
    u64 localSum = 0;
    u64 localRecords = 0;
    PID currentPid = startPid;
    
    // Execute a single query - scan scanPages pages
    while (stats.pagesScanned < scanPages) {
        // Use GuardO for RAII-style page fixing
        GuardO<OLAPPageRow> guard(currentPid);
        OLAPPageRow* page = guard.ptr;
        
        // Only read pre-computed sum from header
        localSum += page->sumValue;
        localRecords += page->recordCount;
        
        PID nextPid = page->nextPage;
        // GuardO automatically unfixes page when it goes out of scope
        
        stats.pagesScanned++;
        
        // Move to next page, or stop if reached end
        if (nextPid == BufferManager::invalidPID) {
            break;  // End of chain, query complete
        }
        currentPid = nextPid;
    }

    stats.sum = localSum;
    stats.recordsCounted = localRecords;
    return stats;
}

// Run OLAP benchmark
void runOLAPBenchmark(const OLAPConfig& config, PID firstPageId) {
    atomic<u64> txProgress(0);
    atomic<u64> txTotal(0);
    atomic<u64> totalPagesScanned(0);
    atomic<u64> totalRecordsCounted(0);
    atomic<u64> totalSum(0);
    atomic<bool> keepRunning(true);
    
    // Get system name for reporting
    const char* systemName = "traditional-array";
    if (bm.hashMode == BufferManager::HashMode::Array1Access)
        systemName = "traditional-array1";
    else if (bm.hashMode == BufferManager::HashMode::Unordered)
        systemName = "traditional-hash-unordered";
    else if (bm.hashMode == BufferManager::HashMode::OpenAddress)
        systemName = "traditional-hash-open";
    else if (bm.hashMode == BufferManager::HashMode::RobinHood)
        systemName = "traditional-hash-robinhood";
    else if (bm.hashMode == BufferManager::HashMode::Lockfree)
        systemName = "traditional-hash-lockfree";
    else if (bm.useExmap)
        systemName = "exmap";
    else if (!bm.useTraditional)
        systemName = "vmcache";
    
    // Get operation type name
    const char* opName = "unknown";
    switch (config.opType) {
        case OLAPConfig::OpType::FilteredCount: opName = "filtered_count"; break;
        case OLAPConfig::OpType::NonFilteredCount: opName = "nonfiltered_count"; break;
        case OLAPConfig::OpType::Sum: opName = "sum"; break;
    }
    
    // Get layout type name
    const char* layoutName = (config.layoutType == OLAPConfig::LayoutType::Columnar) ? "columnar" : "row";
    
    cerr << "Running OLAP benchmark:" << endl;
    cerr << "  System: " << systemName << endl;
    cerr << "  Operation: " << opName << endl;
    cerr << "  Layout: " << layoutName << endl;
    cerr << "  Pages: " << config.numPages << endl;
    cerr << "  Threads: " << config.nthreads << endl;
    cerr << "  Selectivity: " << config.selectivity << endl;
    cerr << "  Duration: " << config.runForSec << " seconds" << endl;
    
    // Statistics thread
    auto statFn = [&]() {
        const char* csvPath = getenv("OLAP_CSV");
        std::ofstream file;
        std::ostream* out = &cout;
        
        if (csvPath) {
            file.open(csvPath);
            if (file.is_open()) {
                out = &file;
            } else {
                cerr << "Error: Could not open OLAP_CSV file: " << csvPath << endl;
            }
        }
        
        *out << "ts,queries_per_sec,rmb,wmb,system,threads,total_pages,operation,layout,selectivity" << endl;
        u64 cnt = 0;
        u64 totalQueries = 0;
        u32 sleepSeconds = 5;
        for (uint64_t i = 0; i < config.runForSec; i+=sleepSeconds) {
            sleep(sleepSeconds);
            float rmb = (bm.readCount.exchange(0) * pageSize) / (1024.0 * 1024);
            float wmb = (bm.writeCount.exchange(0) * pageSize) / (1024.0 * 1024);
            u64 queries = txProgress.exchange(0);  // queries completed in this second
            totalQueries += queries;
            
            *out << cnt << "," << queries << "," << rmb << "," << wmb << "," 
                 << systemName << "," << config.nthreads << "," << config.numPages << "," 
                 << opName << "," << layoutName << "," << config.selectivity << endl;
            cnt += sleepSeconds;  // Adjust timestamp for multi-second sleep
        }
        
        // Record average throughput
        double avgQueriesPerSec = (config.runForSec > 0) ? (double)totalQueries / config.runForSec : 0.0;
        *out << "avg_queries_per_sec," << avgQueriesPerSec << ",0,0," 
             << systemName << "," << config.nthreads << "," << config.numPages << "," 
             << opName << "," << layoutName << "," << config.selectivity << endl;
        
        keepRunning = false;
    };
    
    // Reset counters
    bm.readCount = 0;
    bm.writeCount = 0;
    
    thread statThread(statFn);
    
    // Launch worker threads - each continuously loops and executes queries
    vector<thread> workers;
    std::unique_ptr<perf::Scope> perfScope;
    
    if (config.enablePerf) {
        perfScope = std::make_unique<perf::Scope>();
    }
    
    for (unsigned i = 0; i < config.nthreads; i++) {
        workers.emplace_back([&, i]() {
            workerThreadId = i;
            
            // Random number generator for selecting random starting PIDs
            std::mt19937_64 rng(42 + i);  // Seed with thread ID for reproducibility
            std::uniform_int_distribution<u64> dist(0, config.numPages - config.scanPages);
            
            // Continuously execute queries until time is up
            while (keepRunning.load()) {
                // Randomly select starting PID for this query
                u64 startOffset = dist(rng);
                u64 startPid = firstPageId + startOffset;
                
                // Execute one query - dispatch based on layout type
                QueryStats stats;
                if (config.layoutType == OLAPConfig::LayoutType::Columnar) {
                    // Columnar layout workers
                    switch (config.opType) {
                        case OLAPConfig::OpType::FilteredCount:
                            stats = workerFilteredCount(startPid, config.scanPages, config.selectivity);
                            break;
                        case OLAPConfig::OpType::NonFilteredCount:
                            stats = workerNonFilteredCount(startPid, config.scanPages);
                            break;
                        case OLAPConfig::OpType::Sum:
                            stats = workerSum(startPid, config.scanPages, config.selectivity);
                            break;
                    }
                } else {
                    // Row-oriented layout workers
                    switch (config.opType) {
                        case OLAPConfig::OpType::FilteredCount:
                            stats = workerFilteredCountRowFormat(startPid, config.scanPages, config.selectivity);
                            break;
                        case OLAPConfig::OpType::NonFilteredCount:
                            stats = workerNonFilteredCountRowFormat(startPid, config.scanPages);
                            break;
                        case OLAPConfig::OpType::Sum:
                            stats = workerSumRowFormat(startPid, config.scanPages, config.selectivity);
                            break;
                    }
                }

                totalPagesScanned.fetch_add(stats.pagesScanned, std::memory_order_relaxed);
                totalRecordsCounted.fetch_add(stats.recordsCounted, std::memory_order_relaxed);
                totalSum.fetch_add(stats.sum, std::memory_order_relaxed);
                
                // Query completed, increment progress
                txProgress.fetch_add(1);
            }
        });
    }
    
    statThread.join();
    
    for (auto& w : workers) {
        w.join();
    }
    
    if (config.enablePerf) {
        perf::Aggregator::instance().report(txProgress.load());
    }
    
        const u64 totalQueries = txProgress.load();
        const double avgPages = totalQueries ? static_cast<double>(totalPagesScanned.load()) / totalQueries : 0.0;
        const double avgRecords = totalQueries ? static_cast<double>(totalRecordsCounted.load()) / totalQueries : 0.0;
        const double avgSum = totalQueries ? static_cast<double>(totalSum.load()) / totalQueries : 0.0;

        cerr << "Benchmark complete. Total queries: " << totalQueries << endl;
        cerr << "Totals: pages_scanned=" << totalPagesScanned.load()
            << " records_counted=" << totalRecordsCounted.load()
            << " sum=" << totalSum.load() << endl;
        cerr << "Averages per query: pages_scanned=" << avgPages
            << " records_counted=" << avgRecords
            << " sum=" << avgSum << endl;
}

int main(int argc, char** argv) {
    // Initialize row size configuration from environment variable
    g_rowSize = envOr("ROWSIZE", 80ul);  // Default: 80 bytes per row
    
    // Validate row size
    if (g_rowSize < 8 || g_rowSize > 4096) {
        cerr << "Error: ROWSIZE must be between 8 and 4096 bytes" << endl;
        return 1;
    }
    if (g_rowSize % 8 != 0) {
        cerr << "Error: ROWSIZE must be a multiple of 8 bytes" << endl;
        return 1;
    }
    
    // Calculate number of columns based on row size
    g_numColumns = g_rowSize / sizeof(u64);
    
    cerr << "Row configuration: size=" << g_rowSize << "B, columns=" << g_numColumns << endl;
    
    // Read configuration from environment variables
    OLAPConfig config;
    config.numPages = envOr("NUMPAGES", 1000000ul);
    config.runForSec = envOr("RUNFOR", 30ul);
    config.nthreads = envOr("THREADS", 1u);
    config.selectivity = envOrDouble("SELECTIVITY", 0.1);
    config.enablePerf = envOr("PERF", false);
    
    // Configure scan pages - can be overridden by SCANPAGES environment variable
    config.scanPages = config.numPages / 10;  // Default: 10% of total pages
    u64 userScanPages = envOr("SCANPAGES", 0ul);
    if (userScanPages > 0) {
        config.scanPages = userScanPages;
    }
    
    cerr << "Scan configuration: total_pages=" << config.numPages 
         << " scan_pages=" << config.scanPages << endl;
    
    // Parse operation type
    const char* opTypeStr = getenv("OPTYPE");
    cout << "OPTYPE: " << opTypeStr << endl;
    if (opTypeStr) {
        string op(opTypeStr);
        op.erase(std::remove_if(op.begin(), op.end(), [](unsigned char c) { return std::isspace(c); }), op.end());
        std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c) { return std::tolower(c); });
        if (op == "filtered_count") {
            config.opType = OLAPConfig::OpType::FilteredCount;
        } else if (op == "nonfiltered_count") {
            config.opType = OLAPConfig::OpType::NonFilteredCount;
        } else if (op == "sum") {
            config.opType = OLAPConfig::OpType::Sum;
        } else {
            cerr << "Unknown OPTYPE: " << op << endl;
            cerr << "Valid options: filtered_count, nonfiltered_count, sum" << endl;
            return 1;
        }
        cout << "OPTYPE: " << (int)config.opType << endl;
    }
    
    // Parse layout type
    const char* layoutTypeStr = getenv("LAYOUT");
    if (layoutTypeStr) {
        string layout(layoutTypeStr);
        if (layout == "columnar" || layout == "column") {
            config.layoutType = OLAPConfig::LayoutType::Columnar;
        } else if (layout == "row") {
            config.layoutType = OLAPConfig::LayoutType::Row;
        } else {
            cerr << "Unknown LAYOUT: " << layout << endl;
            cerr << "Valid options: columnar, row" << endl;
            return 1;
        }
    }
    
    cerr << "Configuration:" << endl;
    cerr << "  Layout: " << (config.layoutType == OLAPConfig::LayoutType::Columnar ? "columnar" : "row") << endl;
    cerr << "  Operation: " << (opTypeStr ? opTypeStr : "filtered_count") << endl;
    
    // Initialize BufferManager (same as calico.cpp)
    // BufferManager constructor reads these environment variables:
    //   BLOCK - data file path (default: /tmp/calico.dat)
    //   TRADITIONAL - translation mode (0=vmcache, 1=traditional, 2=mmap+OS page cache)
    //   TRADHASH - translation mode (0=Array1Access, 1=Unordered, 2=OpenAddress,
    //              3=Array, 4=Lockfree, 5=Array2Level, 6=Array3Level, 7=RobinHood)
    //   EXMAP - use exmap (0=no, 1=yes)
    //   VIRTGB - virtual memory size in GB
    //   PHYSGB - physical memory size in GB
    //   USE_IOURING - I/O interface (0=libaio, 1=io_uring)
    bm_ptr = new BufferManager(config.nthreads);
    
    // Initialize pages with appropriate layout
    PID firstPageId;
    if (config.layoutType == OLAPConfig::LayoutType::Columnar) {
        firstPageId = initializePagesColumnar(config.numPages, config.nthreads);
    } else {
        firstPageId = initializePagesRow(config.numPages, config.nthreads);
    }
    
    // Run benchmark
    runOLAPBenchmark(config, firstPageId);
    
    // Cleanup
    delete bm_ptr;
    
    return 0;
}
