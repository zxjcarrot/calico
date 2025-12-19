#include "calico.hpp"
#include "perf_util.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>

using namespace std;

// Helper to get RSS in KB
long getProcessRSS() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    while (std::getline(stat_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::stringstream ss(line);
            std::string label;
            long rss;
            ss >> label >> rss;
            return rss;
        }
    }
    return 0;
}

static void runGraphBenchmark(u64 nodeCount, u64 startIndex) {
    if (nodeCount == 0) {
        std::cout << "graph_bfs nodes=0" << std::endl;
        return;
    }

    u32 desiredNeighbors = 2 * static_cast<u32>(std::log(static_cast<double>(std::max<u64>(2, nodeCount))) / std::log(2.0));
    desiredNeighbors = std::max<u32>(1, desiredNeighbors);
    u32 maxNeighbors = GraphNodePage::maxNeighbors();
    desiredNeighbors = std::min(desiredNeighbors, maxNeighbors);

    std::vector<PID> nodePids;
    nodePids.reserve(nodeCount);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<u64> distNode(0, nodeCount - 1);
    std::uniform_int_distribution<u64> valueDist(1, 1000);

    for (u64 i = 0; i < nodeCount; i++) {
        auto* page = reinterpret_cast<GraphNodePage*>(bm.allocPage());
        PID pid = bm.toPID(page);
        page->neighborCount = 0;
        page->padding = 0;
        page->value = valueDist(rng);
        nodePids.push_back(pid);
        page->dirty = true;
        bm.unfixX(pid);
    }

    for (u64 i = 0; i < nodeCount; i++) {
        GraphNodePage* node = reinterpret_cast<GraphNodePage*>(bm.fixX(nodePids[i]));
        node->neighborCount = desiredNeighbors;
        PID* neighbors = node->neighborData();
        for (u32 j = 0; j < desiredNeighbors; j++) neighbors[j] = nodePids[distNode(rng)];
        node->dirty = true;
        bm.unfixX(nodePids[i]);
    }

    cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;
    PID basePid = nodePids.front();
    PID maxPid = nodePids.back();

    std::vector<uint8_t> visited(nodeCount, 0);
    u64 clampedStart = std::min<u64>(startIndex, nodeCount - 1);
    PID startPid = nodePids[clampedStart];
    cerr << "bfs nodes=" << nodeCount << " neighbors_per_node=" << desiredNeighbors << " start_index=" << clampedStart
         << " start_pid=" << startPid << endl;
    visited[clampedStart] = 1;
    bool enablePerf = envOr("PERF", 0);
    bool enablePrefetch = envOr("PREFETCH", 1);
    u64 visitCount = 0;
    u64 totalValue = 0;
    auto begin = std::chrono::steady_clock::now();
    {
        std::unique_ptr<perf::Scope> perfScope;
        if (enablePerf) perfScope = std::make_unique<perf::Scope>();
        {
            GuardS<GraphNodePage> startNode(startPid);
            totalValue += startNode->value;
        }

        std::queue<PID> frontier;
        frontier.push(startPid);

        while (!frontier.empty()) {
            PID current = frontier.front();
            frontier.pop();
            visitCount++;

            GuardO<GraphNodePage> node(current);
            GraphNodePage* page = node.ptr;
            const PID* neighbors = page->neighborData();
            if (enablePrefetch) bm.prefetchPages(neighbors, page->neighborCount);
            for (u32 i = 0; i < page->neighborCount; i++) {
                // if (enablePrefetch) bm.prefetchPages(&neighbors[i], 1);
                PID neighborPid = neighbors[i];
                u64 idx64 = neighborPid - basePid;
                u32 idx = static_cast<u32>(idx64);
                GuardO<GraphNodePage> neighborNode(neighborPid);
                // compute something
                totalValue += neighborNode->value;
                if (!visited[idx]) {
                    visited[idx] = 1;
                    frontier.push(neighborPid);
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    if (enablePerf) perf::Aggregator::instance().report(visitCount);
    const char* txCsvPath = getenv("TX_CSV");
    std::ofstream file;
    std::ostream* out = &cout;

    if (txCsvPath) {
        file.open(txCsvPath);
        if (file.is_open()) {
            out = &file;
        } else {
            cerr << "Error: Could not open TX_CSV file: " << txCsvPath << endl;
        }
    }
    *out << "graph_bfs nodes=" << nodeCount << " neighbors_per_node=" << desiredNeighbors << " visited=" << visitCount
         << " value_sum=" << totalValue << " time_ms=" << durationMs << std::endl;
}


void handleSEGFAULT(int signo, siginfo_t* info, void* extra) {
    void* page = info->si_addr;
    if (bm.isValidPtr(page)) {
        cerr << "segfault restart " << bm.toPID(page) << endl;
        throw OLCRestartException();
    } else {
        cerr << "segfault " << page << endl;
        _exit(1);
    }
}

int main(int argc, char** argv) {
    bm_ptr = new BufferManager();
    if (bm.useExmap) {
        struct sigaction action;
        action.sa_flags = SA_SIGINFO;
        action.sa_sigaction = handleSEGFAULT;
        if (sigaction(SIGSEGV, &action, NULL) == -1) {
            perror("sigusr: sigaction");
            exit(1);
        }
    }

    unsigned nthreads = envOr("THREADS", 1);
    u64 n = envOr("DATASIZE", 10);
    u64 runForSec = envOr("RUNFOR", 30);
    bool isRndread = envOr("RNDREAD", 0);
    bool isYcsbD = (envOr("RNDREAD", 0) == 2);
    u64 payloadSize = envOr("PAYLOAD_SIZE", 120);
    
    // Parse skew factors
    std::vector<double> skewFactors;
    std::string skewStr = getenv("SKEW_FACTORS") ? getenv("SKEW_FACTORS") : "";
    if (!skewStr.empty()) {
        std::stringstream ss(skewStr);
        std::string segment;
        while (std::getline(ss, segment, ',')) {
            skewFactors.push_back(std::stod(segment));
        }
    } else {
        skewFactors.push_back(envOrDouble("SKEW", 0.0));
    }
    
    double skewFactor = skewFactors[0]; // Default for loading/initialization

    if (envOr("GRAPH", 0)) {
        u64 nodeCount = n;
        u64 startIndex = envOr("GRAPHSTART", 0);
        runGraphBenchmark(nodeCount, startIndex);
        return 0;
    }
    u64 statDiff = 1e8;
    atomic<u64> txProgress(0);
    atomic<u64> txTotal(0);
    atomic<bool> keepRunning(true);
    const char* tradMode = "traditional-array";
    if (bm.hashMode == BufferManager::HashMode::Array1Access)
        tradMode = "traditional-array1";
    else if (bm.hashMode == BufferManager::HashMode::Unordered)
        tradMode = "traditional-hash-unordered";
    else if (bm.hashMode == BufferManager::HashMode::OpenAddress)
        tradMode = "traditional-hash-open";
    else if (bm.hashMode == BufferManager::HashMode::Lockfree)
        tradMode = "traditional-hash-lockfree";
    const char* systemName = bm.useTraditional ? tradMode : (bm.useExmap ? "exmap" : "vmcache");

    auto statFn = [&]() {
        const char* txCsvPath = getenv("TX_CSV");
        std::ofstream file;
        std::ostream* out = &cout;

        if (txCsvPath) {
            file.open(txCsvPath);
            if (file.is_open()) {
                out = &file;
            } else {
                cerr << "Error: Could not open TX_CSV file: " << txCsvPath << endl;
            }
        }

        *out << "ts,tx,rmb,wmb,space_gb,rss_kb,zero_ref_groups,hole_punches,system,threads,datasize,workload,batch,skew" << endl;
        u64 cnt = 0;
        u64 totalTx = 0;
        for (uint64_t i = 0; i < runForSec; i++) {
            sleep(1);
            float rmb = (bm.readCount.exchange(0) * pageSize) / (1024.0 * 1024);
            float wmb = (bm.writeCount.exchange(0) * pageSize) / (1024.0 * 1024);
            float space_gb = (bm.allocCount.load() * pageSize) / (float)bm.gb;
            long rss_kb = getProcessRSS();
            u64 zero_ref_groups = bm.countZeroRefCountGroups();
            u64 prog = txProgress.exchange(0);
            u64 hole_punches = bm.holePunchCount.load();
            totalTx += prog;
            *out << cnt++ << "," << prog << "," << rmb << "," << wmb << "," << space_gb << "," << rss_kb << "," << zero_ref_groups << "," << hole_punches << "," << systemName << "," << nthreads << "," << n
                 << "," << (isRndread ? "rndread" : "tpcc") << "," << bm.batch << "," << skewFactor << endl;
        }
        // Record average throughput
        double avgThroughput = (runForSec > 0) ? (double)totalTx / runForSec : 0.0;
        *out << "avg_throughput," << avgThroughput << ",0,0,0,0,0," << systemName << "," << nthreads << "," << n
             << "," << (isRndread ? "rndread" : "tpcc") << "," << bm.batch << "," << skewFactor << endl;
        keepRunning = false;
        cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;
        cerr << "[BufferManager] Total hole punches (madvise count) " << bm.holePunchCount.load() << endl;
    };

    if (isRndread) {
        bool enablePerf = envOr("PERF", 0);

        // Print information about the key distribution being used
        if (skewFactor > 0.0) {
            cerr << "Using Zipfian key distribution with skew factor: " << skewFactor << endl;
        } else {
            cerr << "Using uniform random key distribution" << endl;
        }

        BTree bt;
        bt.splitOrdered = true;
        
        // Pre-allocate payload buffer for initial load
        std::vector<u8> initialPayload(payloadSize);

        {
            // insert
            parallel_for(0, n, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
                workerThreadId = worker;
                // Use thread-local payload buffer to avoid contention if vector allocation is slow, 
                // but here we just copy from the pre-allocated one or create a local one.
                // Creating a local vector of size ~1KB is fast enough.
                std::vector<u8> payload(payloadSize);
                
                // std::vector<u64> keys;
                // keys.reserve(end-begin);
                // for (u64 i=begin; i<end; i++)
                //    keys.push_back(i);
                // std::random_shuffle(keys.begin(), keys.end());
                for (u64 i = begin; i < end; i++) {
                    union {
                        u64 v1;
                        u8 k1[sizeof(u64)];
                    };
                    // v1 = __builtin_bswap64(keys[i-begin]);
                    v1 = __builtin_bswap64(i);
                    memcpy(payload.data(), k1, sizeof(u64));
                    bt.insert({k1, sizeof(KeyType)}, {payload.data(), payload.size()});
                }
            });
        }
        cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;

        atomic<u64> currentMaxKey(n);
        
        bool firstPhase = true;
        for (double currentSkew : skewFactors) {
            cerr << "Starting benchmark phase with skew: " << currentSkew << endl;
            
            // Reset stats for this phase
            bm.readCount = 0;
            bm.writeCount = 0;
            txProgress = 0;
            txTotal = 0;
            keepRunning = true;
            
            // Define statFn for this phase
            auto phaseStatFn = [&](double skew, bool printHeader) {
                const char* txCsvPath = getenv("TX_CSV");
                std::ofstream file;
                std::ostream* out = &cout;

                if (txCsvPath) {
                    // Append if not the first phase or if file exists and we want to append
                    // But usually we want to overwrite on first run.
                    // Here we assume firstPhase means overwrite, subsequent append.
                    if (printHeader) {
                         file.open(txCsvPath); // Truncate
                    } else {
                         file.open(txCsvPath, std::ios::app); // Append
                    }
                    
                    if (file.is_open()) {
                        out = &file;
                    } else {
                        cerr << "Error: Could not open TX_CSV file: " << txCsvPath << endl;
                    }
                }

                if (printHeader) {
                    *out << "ts,tx,rmb,wmb,space_gb,rss_kb,zero_ref_groups,hole_punches,system,threads,datasize,workload,batch,skew" << endl;
                }
                
                u64 cnt = 0;
                u64 totalTx = 0;
                for (uint64_t i = 0; i < runForSec; i++) {
                    sleep(1);
                    float rmb = (bm.readCount.exchange(0) * pageSize) / (1024.0 * 1024);
                    float wmb = (bm.writeCount.exchange(0) * pageSize) / (1024.0 * 1024);
                    float space_gb = (bm.allocCount.load() * pageSize) / (float)bm.gb;
                    long rss_kb = getProcessRSS();
                    u64 zero_ref_groups = bm.countZeroRefCountGroups();
                    u64 prog = txProgress.exchange(0);
                    u64 hole_punches = bm.holePunchCount.load();
                    totalTx += prog;
                    *out << cnt++ << "," << prog << "," << rmb << "," << wmb << "," << space_gb << "," << rss_kb << "," << zero_ref_groups << "," << hole_punches << "," << systemName << "," << nthreads << "," << n
                         << "," << (isRndread ? "rndread" : "tpcc") << "," << bm.batch << "," << skew << endl;
                }
                // Record average throughput
                double avgThroughput = (runForSec > 0) ? (double)totalTx / runForSec : 0.0;
                *out << "avg_throughput," << avgThroughput << ",0,0,0,0,0," << systemName << "," << nthreads << "," << n
                     << "," << (isRndread ? "rndread" : "tpcc") << "," << bm.batch << "," << skew << endl;
                keepRunning = false;
                cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;
                cerr << "[BufferManager] Total hole punches (madvise count) " << bm.holePunchCount.load() << endl;
            };

            // Initialize key generator based on skew factor (once per phase)
            std::unique_ptr<ScrambledZipfGenerator> sharedZipfGen;
            if (currentSkew > 0.0) {
                sharedZipfGen = std::make_unique<ScrambledZipfGenerator>(0, n, currentSkew);
            }


            thread statThread(phaseStatFn, currentSkew, firstPhase);

            parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
                workerThreadId = worker;
                u64 cnt = 0;
                u64 start = rdtsc();

                std::unique_ptr<perf::Scope> perfScope;
                if (enablePerf) perfScope = std::make_unique<perf::Scope>();

                // Use shared generator
                ScrambledZipfGenerator* zipfGen = sharedZipfGen.get();
                
                // Thread-local payload buffer
                std::vector<u8> payload(payloadSize);

                while (keepRunning.load()) {
                    union {
                        u64 v1;
                        u8 k1[sizeof(u64)];
                    };
                    
                    bool doInsert = false;
                    if (isYcsbD) {
                        // YCSB D: 95% Read Latest, 5% Insert
                        // We use a simple random check for operation type
                        if (RandomGenerator::getRand<u64>(0, 100) < 5) {
                            doInsert = true;
                        }
                    }

                    if (doInsert) {
                        // Insert new record
                        u64 newKey = currentMaxKey.fetch_add(1);
                        v1 = __builtin_bswap64(newKey);
                        memcpy(payload.data(), k1, sizeof(u64));
                        bt.insert({k1, sizeof(KeyType)}, {payload.data(), payload.size()});
                    } else {
                        // Read
                        u64 keyIndex;
                        if (isYcsbD) {
                            // Read Latest: skewed towards recently inserted
                            // We use the Zipfian generator on the initial range [0, n] 
                            // and subtract from currentMaxKey to get "latest" behavior.
                            // Note: This is a windowed approximation if currentMaxKey >> n.
                            u64 offset = 0;
                            if (currentSkew > 0.0) {
                                offset = zipfGen->rand();
                            } else {
                                offset = RandomGenerator::getRand<u64>(0, n);
                            }
                            
                            u64 maxK = currentMaxKey.load();
                            if (maxK > offset) {
                                keyIndex = maxK - 1 - offset;
                            } else {
                                keyIndex = 0; // Should not happen if offset < n and maxK >= n
                            }
                        } else {
                            // Standard YCSB C (Read Only)
                            // Generate key using either uniform random or Zipfian distribution
                            if (currentSkew > 0.0) {
                                keyIndex = zipfGen->rand();
                            } else {
                                keyIndex = RandomGenerator::getRand<u64>(0, n);
                            }
                        }
                        
                        v1 = __builtin_bswap64(keyIndex);
                        
                        bool succ =
                            bt.lookup({k1, sizeof(u64)}, [&](span<u8> p) { 
                                if (p.size() <= payload.size()) {
                                    memcpy(payload.data(), p.data(), p.size()); 
                                }
                            });
                        assert(succ);
                        // assert(memcmp(k1, payload.data(), sizeof(u64)) == 0); // Only valid if payload contains key
                    }

                    cnt++;
                    u64 stop = rdtsc();
                    if ((stop - start) > statDiff) {
                        txProgress += cnt;
                        txTotal += cnt;
                        start = stop;
                        cnt = 0;
                    }
                }
                txTotal += cnt;
                txProgress += cnt;
            });
            
            statThread.join();
            if (enablePerf) perf::Aggregator::instance().report(txTotal.load());
            
            firstPhase = false;
        }
        return 0;
    }

    // TPC-C
    Integer warehouseCount = n;

    vmcacheAdapter<warehouse_t> warehouse;
    vmcacheAdapter<district_t> district;
    vmcacheAdapter<customer_t> customer;
    vmcacheAdapter<customer_wdl_t> customerwdl;
    vmcacheAdapter<history_t> history;
    vmcacheAdapter<neworder_t> neworder;
    vmcacheAdapter<order_t> order;
    vmcacheAdapter<order_wdc_t> order_wdc;
    vmcacheAdapter<orderline_t> orderline;
    vmcacheAdapter<item_t> item;
    vmcacheAdapter<stock_t> stock;

    TPCCWorkload<vmcacheAdapter> tpcc(warehouse, district, customer, customerwdl, history, neworder, order, order_wdc,
                                      orderline, item, stock, true, warehouseCount, true);

    {
        tpcc.loadItem();
        tpcc.loadWarehouse();

        parallel_for(1, warehouseCount + 1, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
            workerThreadId = worker;
            // for (Integer w_id = begin; w_id < end; w_id++) {
            //     tpcc.loadStock(w_id);
            //     tpcc.loadDistrinct(w_id);
            //     for (Integer d_id = 1; d_id <= 10; d_id++) {
            //         tpcc.loadCustomer(w_id, d_id);
            //         tpcc.loadOrders(w_id, d_id);
            //     }
            // }

            for (Integer w_id = begin; w_id < end; w_id++) {
                tpcc.loadStock(w_id);
            }
            for (Integer w_id = begin; w_id < end; w_id++) {
                tpcc.loadDistrinct(w_id);
            }

            for (Integer w_id = begin; w_id < end; w_id++) {
                for (Integer d_id = 1; d_id <= 10; d_id++) {
                    tpcc.loadCustomer(w_id, d_id);
                }
            }
            for (Integer w_id = begin; w_id < end; w_id++) {
                for (Integer d_id = 1; d_id <= 10; d_id++) {
                    tpcc.loadOrders(w_id, d_id);
                }
            }
        });
    }
    cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;

    bm.readCount = 0;
    bm.writeCount = 0;
    thread statThread(statFn);

    parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
        workerThreadId = worker;
        u64 cnt = 0;
        u64 start = rdtsc();
        while (keepRunning.load()) {
            int w_id = tpcc.urand(1, warehouseCount);  // wh crossing
            tpcc.tx(w_id);
            cnt++;
            u64 stop = rdtsc();
            if ((stop - start) > statDiff) {
                txProgress += cnt;
                start = stop;
                cnt = 0;
            }
        }
        txProgress += cnt;
    });

    statThread.join();

    cerr << "space: " << (bm.allocCount.load() * pageSize) / (float)bm.gb << " GB " << endl;
    cerr << "[BufferManager] Total hole punches (madvise count) " << bm.holePunchCount.load() << endl;
    delete bm_ptr;
    return 0;
}