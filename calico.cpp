#include "calico.hpp"

#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// --- Intrinsics Headers ---
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "perf_util.hpp"
#include "tpcc/ZipfianGenerator.hpp"

// Definitions for thread-local variables
__thread uint16_t workerThreadId = 0;
__thread int32_t tpcchistorycounter = 0;

using namespace std;
using namespace tpcc;

// Definition for global BufferManager instance
BufferManager* bm_ptr = nullptr;

// Definition for PageState static member
bool PageState::packedMode = false;

// Definition for LibaioInterface static member
const u64 LibaioInterface::maxIOs;

__attribute__((target("sse4.2"))) uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;
    uint64_t crc = _mm_crc32_u64(seed, key);
    return crc * ((k << 32) + 1);
}

uint64_t rdtsc() {
    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// exmap helper function
static int exmapAction(int exmapfd, exmap_opcode op, u16 len) {
    struct exmap_action_params params_free = {
        .interface = workerThreadId,
        .iov_len = len,
        .opcode = (u16)op,
    };
    return ioctl(exmapfd, EXMAP_IOCTL_ACTION, &params_free);
}

// Calculate minimal number of bits needed to represent n
static u32 bitsNeeded(u64 n) {
    if (n == 0) return 1;
    return 64 - __builtin_clzll(n - 1);
}

// TwoLevelPageStateArray implementation
TwoLevelPageStateArray::TwoLevelPageStateArray(u64 virtCount) 
    : total_entries(virtCount) {
    
    // Calculate minimal bits needed for total entries
    u32 totalBits = bitsNeeded(virtCount);
    
    // Top bits = total - bottom bits
    topBits = (totalBits > bottomBits) ? (totalBits - bottomBits) : 0;
    num_top_slots = 1U << topBits;
    entries_per_slot = 1U << bottomBits;
    
    std::cerr << "[TwoLevelPageStateArray] virtCount=" << virtCount 
              << " totalBits=" << totalBits
              << " topBits=" << topBits << " (" << num_top_slots << " slots)"
              << " bottomBits=" << bottomBits << " (" << entries_per_slot << " entries/slot)"
              << std::endl;
    
    // Allocate top-level array
    size_t top_size = num_top_slots * sizeof(PageState*);
    second_level_arrays = (PageState**)allocHuge(top_size);
    
    // Allocate each second-level array
    size_t second_level_size = entries_per_slot * sizeof(PageState);
    for (u32 i = 0; i < num_top_slots; i++) {
        second_level_arrays[i] = (PageState*)allocHuge(second_level_size);
        
        // Initialize all entries in this second-level array
        for (u32 j = 0; j < entries_per_slot; j++) {
            second_level_arrays[i][j].init();
        }
    }
}

TwoLevelPageStateArray::~TwoLevelPageStateArray() {
    if (second_level_arrays) {
        size_t second_level_size = entries_per_slot * sizeof(PageState);
        for (u32 i = 0; i < num_top_slots; i++) {
            if (second_level_arrays[i]) {
                munmap(second_level_arrays[i], second_level_size);
            }
        }
        size_t top_size = num_top_slots * sizeof(PageState*);
        munmap(second_level_arrays, top_size);
    }
}

// ThreeLevelPageStateArray implementation
ThreeLevelPageStateArray::ThreeLevelPageStateArray(u64 virtCount) 
    : total_entries(virtCount) {
    
    // Calculate minimal bits needed for total entries
    u32 totalBits = bitsNeeded(virtCount);
    
    // Top bits = total - middle - bottom bits
    u32 usedBits = bottomBits + middleBits;
    topBits = (totalBits > usedBits) ? (totalBits - usedBits) : 0;
    num_top_slots = 1U << topBits;
    num_middle_slots = 1U << middleBits;
    entries_per_slot = 1U << bottomBits;
    
    std::cerr << "[ThreeLevelPageStateArray] virtCount=" << virtCount 
              << " totalBits=" << totalBits
              << " topBits=" << topBits << " (" << num_top_slots << " slots)"
              << " middleBits=" << middleBits << " (" << num_middle_slots << " slots)"
              << " bottomBits=" << bottomBits << " (" << entries_per_slot << " entries/slot)"
              << std::endl;
    
    // Allocate top-level array
    size_t top_size = num_top_slots * sizeof(PageState**);
    top_level_arrays = (PageState***)allocHuge(top_size);
    
    // Allocate each middle-level array
    size_t middle_size = num_middle_slots * sizeof(PageState*);
    for (u32 i = 0; i < num_top_slots; i++) {
        top_level_arrays[i] = (PageState**)allocHuge(middle_size);
        
        // Allocate each bottom-level array
        size_t bottom_level_size = entries_per_slot * sizeof(PageState);
        for (u32 j = 0; j < num_middle_slots; j++) {
            top_level_arrays[i][j] = (PageState*)allocHuge(bottom_level_size);
            
            // Initialize all entries in this bottom-level array
            for (u32 k = 0; k < entries_per_slot; k++) {
                top_level_arrays[i][j][k].init();
            }
        }
    }
}

ThreeLevelPageStateArray::~ThreeLevelPageStateArray() {
    if (top_level_arrays) {
        size_t bottom_level_size = entries_per_slot * sizeof(PageState);
        size_t middle_size = num_middle_slots * sizeof(PageState*);
        
        for (u32 i = 0; i < num_top_slots; i++) {
            if (top_level_arrays[i]) {
                for (u32 j = 0; j < num_middle_slots; j++) {
                    if (top_level_arrays[i][j]) {
                        munmap(top_level_arrays[i][j], bottom_level_size);
                    }
                }
                munmap(top_level_arrays[i], middle_size);
            }
        }
        size_t top_size = num_top_slots * sizeof(PageState**);
        munmap(top_level_arrays, top_size);
    }
}

// allocate memory using huge pages
void* allocHuge(size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(p, size, MADV_HUGEPAGE);
    return p;
}

// use when lock is not free
void yield(u64 counter) { 
    //_mm_pause(); 
}

ResidentPageSet::ResidentPageSet(u64 maxCount) : count(next_pow2(maxCount * 1.5)), mask(count - 1), clockPos(0) {
    ht = (Entry*)allocHuge(count * sizeof(Entry));
    memset((void*)ht, 0xFF, count * sizeof(Entry));
}

ResidentPageSet::~ResidentPageSet() { munmap(ht, count * sizeof(u64)); }

u64 ResidentPageSet::next_pow2(u64 x) { return 1 << (64 - __builtin_clzl(x - 1)); }

u64 ResidentPageSet::hash(u64 k) {
    const u64 m = 0xc6a4a7935bd1e995;
    const int r = 47;
    u64 h = 0x8445d61a4e774912 ^ (8 * m);
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

void ResidentPageSet::insert(u64 pid) {
    u64 pos = hash(pid) & mask;
    while (true) {
        u64 curr = ht[pos].pid.load();
        assert(curr != pid);
        if ((curr == empty) || (curr == tombstone))
            if (ht[pos].pid.compare_exchange_strong(curr, pid)) return;

        pos = (pos + 1) & mask;
    }
}

bool ResidentPageSet::remove(u64 pid) {
    u64 pos = hash(pid) & mask;
    while (true) {
        u64 curr = ht[pos].pid.load();
        if (curr == empty) return false;

        if (curr == pid)
            if (ht[pos].pid.compare_exchange_strong(curr, tombstone)) return true;

        pos = (pos + 1) & mask;
    }
}

LibaioInterface::LibaioInterface(int blockfd, BufferManager* bm_ptr) : blockfd(blockfd), bm_ptr(bm_ptr) {
    memset(&ctx, 0, sizeof(io_context_t));
    int ret = io_setup(maxIOs, &ctx);
    if (ret != 0) {
        std::cerr << "libaio io_setup error: " << ret << " ";
        switch (-ret) {
            case EAGAIN:
                std::cerr << "EAGAIN";
                break;
            case EFAULT:
                std::cerr << "EFAULT";
                break;
            case EINVAL:
                std::cerr << "EINVAL";
                break;
            case ENOMEM:
                std::cerr << "ENOMEM";
                break;
            case ENOSYS:
                std::cerr << "ENOSYS";
                break;
        };
        exit(EXIT_FAILURE);
    }
}

void LibaioInterface::writePages(const vector<PID>& pages) {
    assert(pages.size() < maxIOs);
    for (u64 i = 0; i < pages.size(); i++) {
        PID pid = pages[i];
        Page* page = bm_ptr->preparePageForWrite(pid);
        cbPtr[i] = &cb[i];
        io_prep_pwrite(cb + i, blockfd, page, pageSize, pageSize * pid);
    }
    int cnt = io_submit(ctx, pages.size(), cbPtr);
    assert(cnt == (int)pages.size());
    cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
    assert(cnt == (int)pages.size());
}

void LibaioInterface::readPages(const vector<PID>& pages, const vector<Page*>& destinations) {
    assert(pages.size() == destinations.size());
    assert(pages.size() < maxIOs);

    for (u64 i = 0; i < pages.size(); i++) {
        PID pid = pages[i];
        Page* dest = destinations[i];
        cbPtr[i] = &cb[i];
        io_prep_pread(cb + i, blockfd, dest, pageSize, pageSize * pid);
    }

    int cnt = io_submit(ctx, pages.size(), cbPtr);
    assert(cnt == (int)pages.size());
    cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
    assert(cnt == (int)pages.size());

    // Mark pages as clean after successful read
    for (u64 i = 0; i < pages.size(); i++) {
        destinations[i]->dirty = false;
        bm_ptr->readCount++;
    }
}

u64 envOr(const char* env, u64 value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

float envOrF(const char* env, float value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

double envOrDouble(const char* env, double value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

BufferManager::BufferManager(unsigned nthreads)
    : virtSize(envOr("VIRTGB", 16) * gb),
      physSize(envOrF("PHYSGB", 4) * gb),
      virtCount(virtSize / pageSize),
      physCount(physSize / pageSize),
      residentSet(physCount) {
    numThreads = nthreads;
    assert(virtSize >= physSize);
    const char* path = getenv("BLOCK") ? getenv("BLOCK") : "./heapfile";
    blockfd = open(path, O_RDWR | O_DIRECT, S_IRWXU);
    if (blockfd == -1) {
        cerr << "cannot open BLOCK device '" << path << "'" << endl;
        exit(EXIT_FAILURE);
    }

    useTraditional = envOr("TRADITIONAL", 0);
    useExmap = (!useTraditional) && envOr("EXMAP", 0);
    u64 tradHashSetting = useTraditional ? envOr("TRADHASH", 0) : 0;
    if (!useTraditional)
        hashMode = HashMode::Array1Access;
    else if (tradHashSetting == 1)
        hashMode = HashMode::Unordered;
    else if (tradHashSetting == 2)
        hashMode = HashMode::OpenAddress;
    else if (tradHashSetting == 3)
        hashMode = HashMode::Array;
    else if (tradHashSetting == 4)
        hashMode = HashMode::Lockfree;
    else if (tradHashSetting == 5)
        hashMode = HashMode::Array2Level;
    else if (tradHashSetting == 6)
        hashMode = HashMode::Array3Level;
    else
        hashMode = HashMode::Array1Access;
    PageState::setPackedMode(hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level);
    virtMem = nullptr;
    frameMem = nullptr;
    bool disableHugePageForFrameMem = envOr("NOHUGEPAGE_FRAME_MEM", 0);
    bool disableHugePageForTranslationArray = envOr("NOHUGEPAGE_TRANSLATION_ARRAY", 0);

    u64 virtAllocSize = virtSize + (1 << 16);  // guard space for optimistic reads

    if (useTraditional) {
        frameMem = static_cast<Page*>(allocHuge(physCount * sizeof(Page)));
        if (frameMem == MAP_FAILED) die("allocHuge frameMem");
        if (disableHugePageForFrameMem) {
            madvise(frameMem, physCount * sizeof(Page), MADV_NOHUGEPAGE);
        }
        partitionFrameSpan =
            std::max<u32>(1, static_cast<u32>((physCount + freePartitionCount - 1) / freePartitionCount));
        for (u32 i = 0; i < freePartitionCount; i++) {
            auto& partition = freePartitions[i];
            partition.lock.clear(std::memory_order_relaxed);
            partition.frames.clear();
            u32 start = i * partitionFrameSpan;
            if (start >= physCount) continue;
            u32 end = std::min<u32>(start + partitionFrameSpan, physCount);
            partition.frames.reserve(end - start);
            for (u32 frame = start; frame < end; frame++) partition.frames.push_back(frame);
            //if (!partition.frames.empty()) std::random_shuffle(partition.frames.begin(), partition.frames.end());
            std::reverse(partition.frames.begin(), partition.frames.end());
            partition.top = partition.frames.size();
        }
        frameToPidArray.reset(new std::atomic<PID>[physCount]);
        for (u64 i = 0; i < physCount; i++) frameToPidArray[i].store(invalidPID, std::memory_order_relaxed);

        if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access) {
            pid_locks = std::make_unique<pthread_rwlock_t[]>(NUM_LOCK_SHARDS);
            for (u64 i = 0; i < NUM_LOCK_SHARDS; ++i) {
                pthread_rwlock_init(&pid_locks[i], nullptr);
            }
        }

        if (hashMode == HashMode::Array) {
            pidToFrameArray.reset(new std::atomic<u64>[virtCount]);
            if (virtCount * sizeof(std::atomic<u64>) >= 2 * mb)
                madvise(pidToFrameArray.get(), virtCount * sizeof(std::atomic<u64>), MADV_HUGEPAGE);
            for (u64 i = 0; i < virtCount; i++) pidToFrameArray[i].store(invalidFrame, std::memory_order_relaxed);
        } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
            pidToFrameArray.reset();
            pidToFrameHash.reset();
            frameToPidHash.reset();
        } else {
            if (hashMode == HashMode::Unordered) {
                if (numThreads == -1) {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::OpenAddress) {
                if (numThreads == -1) {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::Lockfree) {
                pidToFrameHash = std::make_unique<LockFreeHashMap<PID, u64>>();
                frameToPidHash = std::make_unique<LockFreeHashMap<u64, PID>>();
            } else {
                die("invalid config for hashmap");
            }
            pidToFrameHash->reserve(virtCount);
            frameToPidHash->reserve(physCount);
        }
        // round robin cursors will skip empty partitions automatically
    } else {
        if (useExmap) {
            exmapfd = open("/dev/exmap", O_RDWR);
            if (exmapfd < 0) die("open exmap");

            struct exmap_ioctl_setup buffer;
            buffer.fd = blockfd;
            buffer.max_interfaces = maxWorkerThreads;
            buffer.buffer_size = physCount;
            buffer.flags = 0;
            if (ioctl(exmapfd, EXMAP_IOCTL_SETUP, &buffer) < 0) die("ioctl: exmap_setup");

            for (unsigned i = 0; i < maxWorkerThreads; i++) {
                exmapInterface[i] = (struct exmap_user_interface*)mmap(NULL, pageSize, PROT_READ | PROT_WRITE,
                                                                       MAP_SHARED, exmapfd, EXMAP_OFF_INTERFACE(i));
                if (exmapInterface[i] == MAP_FAILED) die("setup exmapInterface");
            }

            virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_SHARED, exmapfd, 0);
        } else {
            virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (disableHugePageForFrameMem) {
                madvise(virtMem, virtAllocSize, MADV_NOHUGEPAGE);
            } else {
                madvise(virtMem, virtAllocSize, MADV_HUGEPAGE);
            }
        }
        if (virtMem == MAP_FAILED) die("mmap failed");
    }

    // Allocate PageState array (single-level or two-level)
    if (hashMode == HashMode::Array2Level) {
        pageState = nullptr;
        pageState2Level = std::make_unique<TwoLevelPageStateArray>(virtCount);
        pageState3Level.reset();
    } else if (hashMode == HashMode::Array3Level) {
        pageState = nullptr;
        pageState2Level.reset();
        pageState3Level = std::make_unique<ThreeLevelPageStateArray>(virtCount);
    } else {
        pageState = (PageState*)allocHuge(virtCount * sizeof(PageState));
        for (u64 i = 0; i < virtCount; i++) pageState[i].init();
        pageState2Level.reset();
        pageState3Level.reset();
    }
    if (disableHugePageForTranslationArray && useTraditional) {
        if (hashMode == HashMode::Array1Access) {
            madvise(pageState, virtCount * sizeof(PageState), MADV_NOHUGEPAGE);
        }
    }

    physUsedCount = 0;
    allocCount = 1;  // pid 0 reserved for meta data
    readCount = 0;
    writeCount = 0;
    batch = envOr("BATCH", 64);

    cerr << "calico BufferManager initialized: "
         << "blk:" << path << " virtgb:" << virtSize / gb << " physgb:" << (float)physSize / gb
         << " traditional:" << useTraditional << " mmap_os_pagecache:" << useMmapOSPageacche
         << " trad_hash:" << static_cast<int>(hashMode) << " exmap:" << useExmap << " hugepage: " << (disableHugePageForFrameMem == 0)
         << " num_threads:" << numThreads
         << endl;
}

BufferManager::~BufferManager() {
    try {
        flushAll();
    } catch (const std::exception& ex) {
        std::cerr << "BufferManager::~BufferManager flush error: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "BufferManager::~BufferManager encountered an unknown error during flush." << std::endl;
    }
}

IOInterface& BufferManager::getIOInterface() {
    static bool useIOUring = envOr("USE_IOURING", 0);
    // if (useIOUring) {
    //     thread_local IOUringInterface instance(blockfd, this);
    //     return instance;
    // } else {
    //     thread_local LibaioInterface instance(blockfd, this);
    //     return instance;
    // }
    thread_local LibaioInterface instance(blockfd, this);
    return instance;
}

bool BufferManager::isValidPtr(void* page) {
    if (useTraditional) return (page >= frameMem) && (page < (frameMem + physCount));
    if (!virtMem) return false;
    return (page >= virtMem) && (page < (virtMem + virtCount + 16));
}

PID BufferManager::toPID(void* page) {
    Page* ptr = reinterpret_cast<Page*>(page);
    if (useTraditional) {
        auto idx = ptr - frameMem;
        assert(idx >= 0 && static_cast<u64>(idx) < physCount);
        if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
            PID pid = frameToPidArray[static_cast<size_t>(idx)].load(std::memory_order_acquire);
            assert(pid != invalidPID);
            return pid;
        } else {
            PID pid;
            bool ok = frameToPidHash->tryGet(static_cast<u64>(idx), pid);
            assert(ok);
            return pid;
        }
    }
    return ptr - virtMem;
}

inline Page* BufferManager::residentPtr(PID pid) {
    assert(pid < virtCount);
    if (!useTraditional) return virtMem + pid;
    PageState& ps = getPageState(pid);
    u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
    return residentPtr(pid, stateAndVersion);
}

Page* BufferManager::residentPtr(PID pid, u64 stateAndVersion) {
    assert(pid < virtCount);
    if (!useTraditional) return virtMem + pid;

    if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
        //pthread_rwlock_rdlock(&pid_locks[pid_lock_shard(pid)]);
        u64 frame = PageState::extractFrame(stateAndVersion);
        // u64 v = ps.getFrameValue();
        // u64 frame = PageState::extractFrame(v);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        assert(frame != PageState::packedInvalidFrame);
        assert(frame < physCount);
        return frameMem + static_cast<size_t>(frame);
    }


    if (hashMode == HashMode::Array) {
        //pthread_rwlock_rdlock(&pid_locks[pid_lock_shard(pid)]);
        u64 frame = pidToFrameArray[pid].load(std::memory_order_acquire);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        assert(frame != invalidFrame);
        assert(frame < physCount);
        return frameMem + static_cast<size_t>(frame);
    }

    u64 frame;
    bool ok = pidToFrameHash->tryGet(pid, frame);
    assert(ok);
    assert(frame < physCount);
    return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::toPtr(PID pid) { return residentPtr(pid); }

Page* BufferManager::preparePageForWrite(PID pid) {
    Page* page = residentPtr(pid);
    page->dirty = false;
    return page;
}

static thread_local u32 freePartitionCursor = []() -> u32 {
    auto h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return static_cast<u32>(h % BufferManager::freePartitionCount);
}();

u32 BufferManager::popFreeFrame() {
    for (u32 attempt = 0; attempt < freePartitionCount; ++attempt) {
        u32 partition = (freePartitionCursor + attempt) % freePartitionCount;
        auto& part = freePartitions[partition];
        if (part.frames.empty()) continue;

        lockPartition(partition);
        if (part.top > 0) {
            u32 frame = part.frames[--part.top];
            unlockPartition(partition);
            freePartitionCursor = (partition + 1) % freePartitionCount;
            return frame;
        }
        unlockPartition(partition);
    }
    throw std::runtime_error("no free frames available");
}

Page* BufferManager::acquireFrameForPid(PID pid) {
    if (!useTraditional) return virtMem + pid;
    u32 frame = popFreeFrame();
    if (hashMode == HashMode::Array) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        pidToFrameArray[pid].store(static_cast<u64>(frame), std::memory_order_release);
        frameToPidArray[frame].store(pid, std::memory_order_release);
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        getPageState(pid).setFrameValue(frame);
        frameToPidArray[frame].store(pid, std::memory_order_release);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
    } else {
        u64 frameId = static_cast<u64>(frame);
        pidToFrameHash->insertOrAssign(pid, frameId);
        frameToPidHash->insertOrAssign(frameId, pid);
    }
    return frameMem + frame;
}

void BufferManager::releaseFrame(PID pid) {
    if (!useTraditional) return;
    u64 frameId;
    if (hashMode == HashMode::Array) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        frameId = pidToFrameArray[pid].exchange(invalidFrame, std::memory_order_acq_rel);
        assert(frameId != invalidFrame);
        frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        frameId = pageState[pid].getFrameValue();
        assert(frameId != PageState::packedInvalidFrame);
        getPageState(pid).clearFrameValue();
        frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
    } else {
        bool ok = pidToFrameHash->erase(pid, frameId);
        assert(ok);
        PID ignored;
        bool ok2 = frameToPidHash->erase(frameId, ignored);
        assert(ok2);
    }

    assert(frameId < physCount);
    u32 frameIdx = static_cast<u32>(frameId);
    u32 partition = partitionForFrame(frameId);
    auto& part = freePartitions[partition];
    lockPartition(partition);
    assert(part.top < part.frames.size());
    part.frames[part.top++] = frameIdx;
    unlockPartition(partition);
}

u32 BufferManager::partitionForFrame(u64 frame) const {
    if (partitionFrameSpan == 0) return 0;
    u32 partition = static_cast<u32>(frame / partitionFrameSpan);
    return (partition >= freePartitionCount) ? (freePartitionCount - 1) : partition;
}

void BufferManager::lockPartition(u32 partition) {
    while (freePartitions[partition].lock.test_and_set(std::memory_order_acquire)) _mm_pause();
}

void BufferManager::unlockPartition(u32 partition) { freePartitions[partition].lock.clear(std::memory_order_release); }

void BufferManager::ensureFreePages() {
    if (physUsedCount >= physCount * 0.95) evict();
}

// allocated new page and fix it
Page* BufferManager::allocPage() {
    physUsedCount++;
    ensureFreePages();
    u64 pid = allocCount++;
    if (pid >= virtCount) {
        cerr << "VIRTGB is too low" << endl;
        exit(EXIT_FAILURE);
    }
    u64 stateAndVersion = getPageState(pid).stateAndVersion;
    bool succ = getPageState(pid).tryLockX(stateAndVersion);
    assert(succ);
    Page* page = acquireFrameForPid(pid);

    if (useExmap) {
        exmapInterface[workerThreadId]->iov[0].page = pid;
        exmapInterface[workerThreadId]->iov[0].len = 1;
        while (exmapAction(exmapfd, EXMAP_OP_ALLOC, 1) < 0) {
            cerr << "allocPage errno: " << errno << " pid: " << pid << " workerId: " << workerThreadId << endl;
            ensureFreePages();
        }
    }

    residentSet.insert(pid);
    page->dirty = true;

    updateAllocCountSnapshot(pid + 1);

    return page;
}

void BufferManager::updateAllocCountSnapshot(u64 latest_alloc) {
    while (true) {
        try {
            GuardO<MetaDataPage> meta_guard(metadataPageId);
            if (meta_guard->getAllocCountSnapshot() >= latest_alloc) {
                return;
            }

            GuardX<MetaDataPage> meta_write(std::move(meta_guard));
            if (meta_write->getAllocCountSnapshot() < latest_alloc) {
                meta_write->setAllocCountSnapshot(latest_alloc);
                meta_write->dirty = true;
            }
            return;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
}

Page* BufferManager::handleFault(PID pid) {
    physUsedCount++;
    //ensureFreePages();
    Page* page = acquireFrameForPid(pid);
    readPage(pid, page);
    residentSet.insert(pid);
    return page;
}

Page* BufferManager::fixX(PID pid) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; repeatCounter++) {
        u64 stateAndVersion = ps.stateAndVersion.load();
        u64 state = PageState::getState(stateAndVersion);
        
        if (useMmapOSPageacche) {
            // In mmap+OS page cache mode, pages are never evicted by DB
            // Only need to handle: Locked, Marked, Unlocked states
            if (state == PageState::Marked || state == PageState::Unlocked) {
                if (ps.tryLockX(stateAndVersion)) {
                    return residentPtr(pid, stateAndVersion);
                }
            }
            // For Locked state, just retry
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        Page* page = handleFault(pid);
                        ensureFreePages();
                        return page;
                    }
                    break;
                }
                case PageState::Marked:
                case PageState::Unlocked: {
                    if (ps.tryLockX(stateAndVersion)) {
                        // u64 lockedState = PageState::sameVersion(stateAndVersion, PageState::Locked);
                        return residentPtr(pid, stateAndVersion);
                    }
                    break;
                }
            }
        }
        yield(repeatCounter);
    }
}

Page* BufferManager::fixS(PID pid) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; repeatCounter++) {
        u64 stateAndVersion = ps.stateAndVersion;
        u64 state = PageState::getState(stateAndVersion);
        
        if (useMmapOSPageacche) {
            // In mmap+OS page cache mode, pages are never evicted by DB
            // Only need to handle: Locked, Marked, Unlocked, and shared states
            if (state == PageState::Locked) {
                // Just retry
            } else {
                // Try to acquire shared lock (state can be Marked, Unlocked, or already shared)
                if (ps.tryLockS(stateAndVersion)) {
                    u64 s = PageState::getState(stateAndVersion);
                    u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
                    u64 newState = PageState::sameVersion(stateAndVersion, nextState);
                    return residentPtr(pid, newState);
                }
            }
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::Locked: {
                    break;
                }
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        handleFault(pid);
                        ps.unlockX();
                        ensureFreePages();
                    }
                    break;
                }
                default: {
                    if (ps.tryLockS(stateAndVersion)) {
                        u64 s = PageState::getState(stateAndVersion);
                        u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
                        u64 newState = PageState::sameVersion(stateAndVersion, nextState);
                        return residentPtr(pid, newState);
                    }
                }
            }
        }
        yield(repeatCounter);
    }
}

void BufferManager::unfixS(PID pid) { getPageState(pid).unlockS(); }

void BufferManager::unfixX(PID pid) { getPageState(pid).unlockX(); }

void BufferManager::readPage(PID pid, Page* dest) {
    if (useExmap) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            int ret = pread(exmapfd, dest, pageSize, workerThreadId);
            if (ret == pageSize) {
                assert(ret == pageSize);
                dest->dirty = false;
                readCount++;
                return;
            }
            cerr << "readPage errno: " << errno << " pid: " << pid << " workerId: " << workerThreadId << endl;
            ensureFreePages();
        }
    } else {
        int ret = pread(blockfd, dest, pageSize, pid * pageSize);
        assert(ret == pageSize);
        dest->dirty = false;
        readCount++;
    }
}

void BufferManager::flushAll() {
    std::vector<PID> batch;
    batch.reserve(LibaioInterface::maxIOs);
    std::vector<PageState*> locked_states;
    locked_states.reserve(LibaioInterface::maxIOs);
    int flushed_pages = 0;
    auto flush_batch = [&]() {
        if (batch.empty()) return;
        try {
            getIOInterface().writePages(batch);
            writeCount += batch.size();
            flushed_pages += batch.size();
        } catch (...) {
            for (PageState* ps : locked_states) {
                ps->unlockS();
            }
            batch.clear();
            locked_states.clear();
            throw;
        }

        for (PageState* ps : locked_states) {
            ps->unlockS();
        }
        batch.clear();
        locked_states.clear();
    };

    u64 max_pid = allocCount.load(std::memory_order_acquire);
    for (PID pid = 0; pid < max_pid; ++pid) {
        PageState& ps = getPageState(pid);

        for (u64 repeatCounter = 0;; ++repeatCounter) {
            u64 state_and_version = ps.stateAndVersion.load(std::memory_order_acquire);
            u64 state = PageState::getState(state_and_version);

            if (state == PageState::Evicted) {
                break;
            }
            if (state == PageState::Locked) {
                yield(repeatCounter);
                continue;
            }

            if (ps.tryLockS(state_and_version)) {
                Page* page_ptr = residentPtr(pid, state_and_version);
                if (page_ptr->dirty) {
                    batch.push_back(pid);
                    locked_states.push_back(&ps);
                } else {
                    ps.unlockS();
                }
                break;
            }
            yield(repeatCounter);
        }

        if (batch.size() >= LibaioInterface::maxIOs) {
            flush_batch();
        }
    }

    flush_batch();
    cerr << "BufferManager flushed " << flushed_pages << " pages." << endl;
}

void BufferManager::evict() {
    vector<PID> toEvict;
    toEvict.reserve(batch);
    vector<PID> toWrite;
    toWrite.reserve(batch);

    // 0. find candidates, lock dirty ones in shared mode
    while (toEvict.size() + toWrite.size() < batch) {
        residentSet.iterateClockBatch(batch, [&](PID pid) {
            PageState& ps = getPageState(pid);
            u64 v = ps.stateAndVersion;
            switch (PageState::getState(v)) {
                case PageState::Marked:
                    if (residentPtr(pid)->dirty) {
                        if (ps.tryLockS(v)) toWrite.push_back(pid);
                    } else {
                        toEvict.push_back(pid);
                    }
                    break;
                case PageState::Unlocked:
                    ps.tryMark(v);
                    break;
                default:
                    break;  // skip
            };
        });
    }

    // 1. write dirty pages
    if (toWrite.size() > 0) {
        getIOInterface().writePages(toWrite);
        writeCount += toWrite.size();
    }

    // 2. try to lock clean page candidates
    toEvict.erase(std::remove_if(toEvict.begin(), toEvict.end(),
                                 [&](PID pid) {
                                     PageState& ps = getPageState(pid);
                                     u64 v = ps.stateAndVersion;
                                     return (PageState::getState(v) != PageState::Marked) || !ps.tryLockX(v);
                                 }),
                  toEvict.end());

    // 3. try to upgrade lock for dirty page candidates
    for (auto& pid : toWrite) {
        PageState& ps = getPageState(pid);
        u64 v = ps.stateAndVersion;
        if ((PageState::getState(v) == 1) &&
            ps.stateAndVersion.compare_exchange_weak(v, PageState::sameVersion(v, PageState::Locked)))
            toEvict.push_back(pid); // there were no new readers joined during the write-back, let's evict the page
        else
            ps.unlockS();
    }

    // 4. remove from page table
    if (useExmap) {
        for (u64 i = 0; i < toEvict.size(); i++) {
            exmapInterface[workerThreadId]->iov[i].page = toEvict[i];
            exmapInterface[workerThreadId]->iov[i].len = 1;
        }
        if (exmapAction(exmapfd, EXMAP_OP_FREE, toEvict.size()) < 0) die("ioctl: EXMAP_OP_FREE");
    } else if (!useTraditional) {
        for (u64& pid : toEvict) madvise(virtMem + pid, pageSize, MADV_DONTNEED);
    }

    // 5. remove from hash table and unlock
    for (u64& pid : toEvict) {
        if (useTraditional) releaseFrame(pid);
        bool succ = residentSet.remove(pid);
        assert(succ);
        getPageState(pid).unlockXEvicted();
    }

    physUsedCount -= toEvict.size();
}

void BufferManager::prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    //if (!useTraditional || hashMode == HashMode::Array1Access || hashMode == HashMode::Array || hashMode == HashMode::Array2Level) {
    // if (!useTraditional || hashMode == HashMode::Array1Access) {
    //     for (int i = 0; i < n_pages; i++) {
    //         PageState& ps = getPageState(pages[i]);
    //         char * tmp = (char*)&ps;
    //         _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
    //     }
    // }
    
    // if (!useTraditional) {
    //    // For mmap mode, all pages are always "resident"
    //    return;
    // }
    // Use unique_ptr to defer allocation until we know we have pages to read
    std::unique_ptr<vector<PID>> pidsToRead;
    std::unique_ptr<vector<Page*>> destinations;

    // Check which pages are not in memory and need to be loaded
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        assert(pid < virtCount);
        if (i + 1 < n_pages) {
            // Prefetch next PageState
            PageState& next_ps = getPageState(pages[i + 1]);
            char* tmp = (char*)&next_ps;
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        }

        PageState& ps = getPageState(pid);
        u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        u64 state = PageState::getState(stateAndVersion);

        if (hashMode == HashMode::Lockfree || hashMode == HashMode::Unordered || hashMode == HashMode::OpenAddress) {
            // Double check that the page is still mapped
            u64 frame;
            if (pidToFrameHash->tryGet(pid, frame)) {
                // Page is already mapped, skip
                continue;
            }
        }

        // Skip pages that are already in memory (not evicted)
        if (state != PageState::Evicted) {
            if (offsets_within_pages != nullptr) {
                  // Touch the page to bring it into cache
                //   char *tmp = (((char*)residentPtr(pid, stateAndVersion)) + offsets_within_pages[i]);
                //   _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T2);
                //   _mm_prefetch(reinterpret_cast<const char*>(tmp) + 64, _MM_HINT_T2);
            }
            continue;
        }

        // Lock the page first before allocating frame
        if (ps.tryLockX(stateAndVersion)) {
            ensureFreePages();
            // Try to acquire a frame for this page
            Page* frame = acquireFrameForPid(pid);
            if (frame != nullptr) {
                // Allocate vectors only when we have the first page to read
                if (!pidsToRead) {
                    pidsToRead = std::make_unique<vector<PID>>();
                    destinations = std::make_unique<vector<Page*>>();
                    pidsToRead->reserve(n_pages);
                    destinations->reserve(n_pages);
                }

                pidsToRead->push_back(pid);
                destinations->push_back(frame);
                physUsedCount++;
            } else {
                // Failed to acquire frame, unlock the page
                ps.unlockX();
            }
        }
    }

    // Batch read all pages that need to be loaded
    if (pidsToRead && !pidsToRead->empty()) {
        if (pidsToRead->size() >= LibaioInterface::maxIOs) {
            // Split into multiple batches if too many pages
            for (size_t i = 0; i < pidsToRead->size(); i += LibaioInterface::maxIOs) {
                size_t batchSize = std::min(LibaioInterface::maxIOs, pidsToRead->size() - i);
                vector<PID> batch(pidsToRead->begin() + i, pidsToRead->begin() + i + batchSize);
                vector<Page*> batchDest(destinations->begin() + i, destinations->begin() + i + batchSize);
                getIOInterface().readPages(batch, batchDest);
            }
        } else {
            getIOInterface().readPages(*pidsToRead, *destinations);
        }

        // Unlock all successfully loaded pages
        for (PID pid : *pidsToRead) {
            // Add to resident set
            residentSet.insert(pid);
            PageState& ps = getPageState(pid);
            ps.unlockX();
        }
    }
}

// BTreeNode Method Implementations

static unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }

template <class T>
static T loadUnaligned(void* p) {
    T x;
    memcpy(&x, p, sizeof(T));
    return x;
}

// Get order-preserving head of key (assuming little endian)
static u32 head(u8* key, unsigned keyLen) {
    switch (keyLen) {
        case 0:
            return 0;
        case 1:
            return static_cast<u32>(key[0]) << 24;
        case 2:
            return static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16;
        case 3:
            return (static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16) |
                   (static_cast<u32>(key[2]) << 8);
        default:
            return __builtin_bswap32(loadUnaligned<u32>(key));
    }
}

bool BTreeNode::hasSpaceFor(unsigned keyLen, unsigned payloadLen) {
    return spaceNeeded(keyLen, payloadLen) <= freeSpaceAfterCompaction();
}

PID BTreeNode::getChild(unsigned slotId) { return loadUnaligned<PID>(getPayload(slotId).data()); }

void BTreeNode::makeHint() {
    unsigned dist = count / (hintCount + 1);
    for (unsigned i = 0; i < hintCount; i++) hint[i] = slot[dist * (i + 1)].head;
}

void BTreeNode::updateHint(unsigned slotId) {
    unsigned dist = count / (hintCount + 1);
    unsigned begin = 0;
    if ((count > hintCount * 2 + 1) && (((count - 1) / (hintCount + 1)) == dist) && ((slotId / dist) > 1))
        begin = (slotId / dist) - 1;
    for (unsigned i = begin; i < hintCount; i++) hint[i] = slot[dist * (i + 1)].head;
}

void BTreeNode::searchHint(u32 keyHead, u16& lowerOut, u16& upperOut) {
    if (count > hintCount * 2) {
        u16 dist = upperOut / (hintCount + 1);
        u16 pos, pos2;
        for (pos = 0; pos < hintCount; pos++)
            if (hint[pos] >= keyHead) break;
        for (pos2 = pos; pos2 < hintCount; pos2++)
            if (hint[pos2] != keyHead) break;
        lowerOut = pos * dist;
        if (pos2 < hintCount) upperOut = (pos2 + 1) * dist;
    }
}

u16 BTreeNode::lowerBound(span<u8> skey, bool& foundExactOut) {
    foundExactOut = false;

    // check prefix
    int cmp = memcmp(skey.data(), getPrefix(), min(skey.size(), prefixLen));
    if (cmp < 0)  // key is less than prefix
        return 0;
    if (cmp > 0)  // key is greater than prefix
        return count;
    if (skey.size() < prefixLen)  // key is equal but shorter than prefix
        return 0;
    u8* key = skey.data() + prefixLen;
    unsigned keyLen = skey.size() - prefixLen;

    // check hint
    u16 lower = 0;
    u16 upper = count;
    u32 keyHead = head(key, keyLen);
    searchHint(keyHead, lower, upper);

    // binary search on remaining range
    while (lower < upper) {
        u16 mid = ((upper - lower) / 2) + lower;
        if (keyHead < slot[mid].head) {
            upper = mid;
        } else if (keyHead > slot[mid].head) {
            lower = mid + 1;
        } else {  // head is equal, check full key
            int cmp = memcmp(key, getKey(mid), min(keyLen, slot[mid].keyLen));
            if (cmp < 0) {
                upper = mid;
            } else if (cmp > 0) {
                lower = mid + 1;
            } else {
                if (keyLen < slot[mid].keyLen) {  // key is shorter
                    upper = mid;
                } else if (keyLen > slot[mid].keyLen) {  // key is longer
                    lower = mid + 1;
                } else {
                    foundExactOut = true;
                    return mid;
                }
            }
        }
    }
    return lower;
}

u16 BTreeNode::lowerBound(span<u8> key) {
    bool ignore;
    return lowerBound(key, ignore);
}

void BTreeNode::insertInPage(span<u8> key, span<u8> payload) {
    unsigned needed = spaceNeeded(key.size(), payload.size());
    if (needed > freeSpace()) {
        assert(needed <= freeSpaceAfterCompaction());
        compactify();
    }
    unsigned slotId = lowerBound(key);
    memmove(slot + slotId + 1, slot + slotId, sizeof(Slot) * (count - slotId));
    storeKeyValue(slotId, key, payload);
    count++;
    updateHint(slotId);
}

bool BTreeNode::removeSlot(unsigned slotId) {
    spaceUsed -= slot[slotId].keyLen;
    spaceUsed -= slot[slotId].payloadLen;
    memmove(slot + slotId, slot + slotId + 1, sizeof(Slot) * (count - slotId - 1));
    count--;
    makeHint();
    return true;
}

bool BTreeNode::removeInPage(span<u8> key) {
    bool found;
    unsigned slotId = lowerBound(key, found);
    if (!found) return false;
    return removeSlot(slotId);
}

void BTreeNode::copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src) {
    u64 ofs = offsetof(BTreeNodeHeader, upperInnerNode);
    memcpy(reinterpret_cast<u8*>(dst) + ofs, reinterpret_cast<u8*>(src) + ofs, sizeof(BTreeNode) - ofs);
}

void BTreeNode::compactify() {
    unsigned should = freeSpaceAfterCompaction();
    static_cast<void>(should);
    BTreeNode tmp(isLeaf);
    tmp.setFences(getLowerFence(), getUpperFence());
    copyKeyValueRange(&tmp, 0, 0, count);
    tmp.upperInnerNode = upperInnerNode;
    copyNode(this, &tmp);
    makeHint();
    assert(freeSpace() == should);
}

bool BTreeNode::mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right) {
    if (!isLeaf)
        // TODO: implement inner merge
        return true;

    assert(right->isLeaf);
    assert(parent->isInner());
    BTreeNode tmp(isLeaf);
    tmp.setFences(getLowerFence(), right->getUpperFence());
    unsigned leftGrow = (prefixLen - tmp.prefixLen) * count;
    unsigned rightGrow = (right->prefixLen - tmp.prefixLen) * right->count;
    unsigned spaceUpperBound = spaceUsed + right->spaceUsed +
                               (reinterpret_cast<u8*>(slot + count + right->count) - ptr()) + leftGrow + rightGrow;
    if (spaceUpperBound > pageSize) return false;
    copyKeyValueRange(&tmp, 0, 0, count);
    right->copyKeyValueRange(&tmp, count, 0, right->count);
    PID pid = bm.toPID(this);
    memcpy(parent->getPayload(slotId + 1).data(), &pid, sizeof(PID));
    parent->removeSlot(slotId);
    tmp.makeHint();
    tmp.nextLeafNode = right->nextLeafNode;

    copyNode(this, &tmp);
    return true;
}

void BTreeNode::storeKeyValue(u16 slotId, span<u8> skey, span<u8> payload) {
    // slot
    u8* key = skey.data() + prefixLen;
    unsigned keyLen = skey.size() - prefixLen;
    slot[slotId].head = head(key, keyLen);
    slot[slotId].keyLen = keyLen;
    slot[slotId].payloadLen = payload.size();
    // key
    unsigned space = keyLen + payload.size();
    dataOffset -= space;
    spaceUsed += space;
    slot[slotId].offset = dataOffset;
    assert(getKey(slotId) >= reinterpret_cast<u8*>(&slot[slotId]));
    memcpy(getKey(slotId), key, keyLen);
    memcpy(getPayload(slotId).data(), payload.data(), payload.size());
}

void BTreeNode::copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount) {
    if (prefixLen <= dst->prefixLen) {  // prefix grows
        unsigned diff = dst->prefixLen - prefixLen;
        for (unsigned i = 0; i < srcCount; i++) {
            unsigned newKeyLen = slot[srcSlot + i].keyLen - diff;
            unsigned space = newKeyLen + slot[srcSlot + i].payloadLen;
            dst->dataOffset -= space;
            dst->spaceUsed += space;
            dst->slot[dstSlot + i].offset = dst->dataOffset;
            u8* key = getKey(srcSlot + i) + diff;
            memcpy(dst->getKey(dstSlot + i), key, space);
            dst->slot[dstSlot + i].head = head(key, newKeyLen);
            dst->slot[dstSlot + i].keyLen = newKeyLen;
            dst->slot[dstSlot + i].payloadLen = slot[srcSlot + i].payloadLen;
        }
    } else {
        for (unsigned i = 0; i < srcCount; i++) copyKeyValue(srcSlot + i, dst, dstSlot + i);
    }
    dst->count += srcCount;
    assert((dst->ptr() + dst->dataOffset) >= reinterpret_cast<u8*>(dst->slot + dst->count));
}

void BTreeNode::copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot) {
    unsigned fullLen = slot[srcSlot].keyLen + prefixLen;
    u8 key[fullLen];
    memcpy(key, getPrefix(), prefixLen);
    memcpy(key + prefixLen, getKey(srcSlot), slot[srcSlot].keyLen);
    dst->storeKeyValue(dstSlot, {key, fullLen}, getPayload(srcSlot));
}

void BTreeNode::insertFence(FenceKeySlot& fk, span<u8> key) {
    assert(freeSpace() >= key.size());
    dataOffset -= key.size();
    spaceUsed += key.size();
    fk.offset = dataOffset;
    fk.len = key.size();
    memcpy(ptr() + dataOffset, key.data(), key.size());
}

void BTreeNode::setFences(span<u8> lower, span<u8> upper) {
    insertFence(lowerFence, lower);
    insertFence(upperFence, upper);
    for (prefixLen = 0; (prefixLen < min(lower.size(), upper.size())) && (lower[prefixLen] == upper[prefixLen]);
         prefixLen++);
}

void BTreeNode::splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep) {
    assert(sepSlot > 0);
    assert(sepSlot < (pageSize / sizeof(PID)));

    BTreeNode tmp(isLeaf);
    BTreeNode* nodeLeft = &tmp;

    AllocGuard<BTreeNode> newNode(isLeaf);
    BTreeNode* nodeRight = newNode.ptr;

    nodeLeft->setFences(getLowerFence(), sep);
    nodeRight->setFences(sep, getUpperFence());

    PID leftPID = bm.toPID(this);
    u16 oldParentSlot = parent->lowerBound(sep);
    if (oldParentSlot == parent->count) {
        assert(parent->upperInnerNode == leftPID);
        parent->upperInnerNode = newNode.pid;
    } else {
        assert(parent->getChild(oldParentSlot) == leftPID);
        memcpy(parent->getPayload(oldParentSlot).data(), &newNode.pid, sizeof(PID));
    }
    parent->insertInPage(sep, {reinterpret_cast<u8*>(&leftPID), sizeof(PID)});

    if (isLeaf) {
        copyKeyValueRange(nodeLeft, 0, 0, sepSlot + 1);
        copyKeyValueRange(nodeRight, 0, nodeLeft->count, count - nodeLeft->count);
        nodeLeft->nextLeafNode = newNode.pid;
        nodeRight->nextLeafNode = this->nextLeafNode;
    } else {
        // in inner node split, separator moves to parent (count == 1 + nodeLeft->count + nodeRight->count)
        copyKeyValueRange(nodeLeft, 0, 0, sepSlot);
        copyKeyValueRange(nodeRight, 0, nodeLeft->count + 1, count - nodeLeft->count - 1);
        nodeLeft->upperInnerNode = getChild(nodeLeft->count);
        nodeRight->upperInnerNode = upperInnerNode;
    }
    nodeLeft->makeHint();
    nodeRight->makeHint();
    copyNode(this, nodeLeft);
}

unsigned BTreeNode::commonPrefix(unsigned slotA, unsigned slotB) {
    assert(slotA < count);
    unsigned limit = min(slot[slotA].keyLen, slot[slotB].keyLen);
    u8 *a = getKey(slotA), *b = getKey(slotB);
    unsigned i;
    for (i = 0; i < limit; i++)
        if (a[i] != b[i]) break;
    return i;
}

BTreeNode::SeparatorInfo BTreeNode::findSeparator(bool splitOrdered) {
    assert(count > 1);
    if (isInner()) {
        // inner nodes are split in the middle
        unsigned slotId = count / 2;
        return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[slotId].keyLen), slotId, false};
    }

    // find good separator slot
    unsigned bestPrefixLen, bestSlot;

    if (splitOrdered) {
        bestSlot = count - 2;
    } else if (count > 16) {
        unsigned lower = (count / 2) - (count / 16);
        unsigned upper = (count / 2);

        bestPrefixLen = commonPrefix(lower, 0);
        bestSlot = lower;

        if (bestPrefixLen != commonPrefix(upper - 1, 0))
            for (bestSlot = lower + 1; (bestSlot < upper) && (commonPrefix(bestSlot, 0) == bestPrefixLen); bestSlot++);
    } else {
        bestSlot = (count - 1) / 2;
    }

    // try to truncate separator
    unsigned common = commonPrefix(bestSlot, bestSlot + 1);
    if ((bestSlot + 1 < count) && (slot[bestSlot].keyLen > common) && (slot[bestSlot + 1].keyLen > (common + 1)))
        return SeparatorInfo{prefixLen + common + 1, bestSlot, true};

    return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[bestSlot].keyLen), bestSlot, false};
}

void BTreeNode::getSep(u8* sepKeyOut, SeparatorInfo info) {
    memcpy(sepKeyOut, getPrefix(), prefixLen);
    memcpy(sepKeyOut + prefixLen, getKey(info.slot + info.isTruncated), info.len - prefixLen);
}

PID BTreeNode::lookupInner(span<u8> key) {
    unsigned pos = lowerBound(key);
    if (pos == count) return upperInnerNode;
    return getChild(pos);
}

// BTree Method Implementations

static unsigned btreeslotcounter = 0;

BTree::BTree() : splitOrdered(false) {
    GuardX<MetaDataPage> page(metadataPageId);
    AllocGuard<BTreeNode> rootNode(true);
    {
        slotId = btreeslotcounter++;
        page->roots[slotId] = rootNode.pid;
        page->dirty = true;
    }
}

BTree::~BTree() {}

GuardO<BTreeNode> BTree::findLeafO(span<u8> key) {
    GuardO<MetaDataPage> meta(metadataPageId);
    GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
    meta.release();

    while (node->isInner()) node = GuardO<BTreeNode>(node->lookupInner(key), node);
    return node;
}

GuardS<BTreeNode> BTree::findLeafS(span<u8> key) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<MetaDataPage> meta(metadataPageId);
            GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
            meta.release();

            while (node->isInner()) node = GuardO<BTreeNode>(node->lookupInner(key), node);

            return GuardS<BTreeNode>(std::move(node));
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

void BTree::trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen) {
    // create new root if necessary
    if (parent.pid == metadataPageId) {
        GuardX<MetaDataPage> meta_guard(metadataPageId);

        AllocGuard<BTreeNode> newRoot(false);
        newRoot->upperInnerNode = node.pid;
        {
            meta_guard->roots[slotId] = newRoot.pid;
            meta_guard->dirty = true;
        }
        parent = std::move(newRoot);
    }

    // split
    BTreeNode::SeparatorInfo sepInfo = node->findSeparator(splitOrdered.load());
    u8 sepKey[sepInfo.len];
    node->getSep(sepKey, sepInfo);

    if (parent->hasSpaceFor(sepInfo.len, sizeof(PID))) {  // is there enough space in the parent for the separator?
        node->splitNode(parent.ptr, sepInfo.slot, {sepKey, sepInfo.len});
        return;
    }

    // must split parent to make space for separator, restart from root to do this
    node.release();
    parent.release();
    ensureSpace(parent.ptr, {sepKey, sepInfo.len}, sizeof(PID));
}

void BTree::ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            while (node->isInner() && (node.ptr != toSplit)) {
                parent = std::move(node);
                node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
            }
            if (node.ptr == toSplit) {
                if (node->hasSpaceFor(key.size(), payloadLen)) return;  // someone else did split concurrently
                GuardX<BTreeNode> parentLocked(std::move(parent));
                GuardX<BTreeNode> nodeLocked(std::move(node));
                trySplit(std::move(nodeLocked), std::move(parentLocked), key, payloadLen);
            }
            return;
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

void BTree::insert(span<u8> key, span<u8> payload) {
    assert((key.size() + payload.size()) <= BTreeNode::maxKVSize);

    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            while (node->isInner()) {
                parent = std::move(node);
                node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
            }

            if (node->hasSpaceFor(key.size(), payload.size())) {
                // only lock leaf
                GuardX<BTreeNode> nodeLocked(std::move(node));
                parent.release();
                nodeLocked->insertInPage(key, payload);
                return;  // success
            }

            // lock parent and leaf
            GuardX<BTreeNode> parentLocked(std::move(parent));
            GuardX<BTreeNode> nodeLocked(std::move(node));
            trySplit(std::move(nodeLocked), std::move(parentLocked), key, payload.size());
            // insert hasn't happened, restart from root
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

bool BTree::remove(span<u8> key) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            u16 pos;
            while (node->isInner()) {
                pos = node->lowerBound(key);
                PID nextPage = (pos == node->count) ? node->upperInnerNode : node->getChild(pos);
                parent = std::move(node);
                node = GuardO<BTreeNode>(nextPage, parent);
            }

            bool found;
            unsigned slotId = node->lowerBound(key, found);
            if (!found) return false;

            unsigned sizeEntry = node->slot[slotId].keyLen + node->slot[slotId].payloadLen;
            if ((node->freeSpaceAfterCompaction() + sizeEntry >= BTreeNodeHeader::underFullSize) &&
                (parent.pid != metadataPageId) && (parent->count >= 2) && ((pos + 1) < parent->count)) {
                // underfull
                GuardX<BTreeNode> parentLocked(std::move(parent));
                GuardX<BTreeNode> nodeLocked(std::move(node));
                GuardX<BTreeNode> rightLocked(parentLocked->getChild(pos + 1));
                nodeLocked->removeSlot(slotId);
                if (rightLocked->freeSpaceAfterCompaction() >= (pageSize - BTreeNodeHeader::underFullSize)) {
                    if (nodeLocked->mergeNodes(pos, parentLocked.ptr, rightLocked.ptr)) {
                        // XXX: should reuse page Id
                    }
                }
            } else {
                GuardX<BTreeNode> nodeLocked(std::move(node));
                parent.release();
                nodeLocked->removeSlot(slotId);
            }
            return true;
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}



void initialize_system() {
    if (bm_ptr == nullptr) {
        // Set environment variables if they are not set, for default behavior
        // This is good practice when wrapping a library that depends on them.
        setenv("BLOCK", "./heapfile", 0); // Use a different file for python tests
        setenv("VIRTGB", "24", 0);
        setenv("PHYSGB", "16", 0);
        
        bm_ptr = new BufferManager();

        // Restore allocCount from persisted metadata snapshot if available.
        bool can_attempt_restore = true;
        if (const char* block_path = getenv("BLOCK")) {
            struct stat st {};
            if (stat(block_path, &st) == 0) {
                if (S_ISREG(st.st_mode) && st.st_size < static_cast<off_t>(pageSize)) {
                    can_attempt_restore = false;
                }
            }
        }

        if (can_attempt_restore) {
            for (;;) {
                try {
                    GuardO<MetaDataPage> meta_guard(metadataPageId);
                    u64 stored_alloc = meta_guard->getAllocCountSnapshot();
                    u64 current_alloc = bm_ptr->allocCount.load(std::memory_order_relaxed);
                    if (stored_alloc > current_alloc) {
                        bm_ptr->allocCount.store(stored_alloc, std::memory_order_relaxed);
                        std::cerr << "[Calico Init] Restored allocCount from metadata: " << stored_alloc
                                  << std::endl;
                    }
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
        }
    }
}

void flush_system() {
    if (bm_ptr != nullptr) {
        bool can_attempt_update = true;
        if (const char* block_path = getenv("BLOCK")) {
            struct stat st {};
            if (stat(block_path, &st) == 0) {
                if (S_ISREG(st.st_mode) && st.st_size < static_cast<off_t>(pageSize)) {
                    can_attempt_update = false;
                }
            }
        }

        bm_ptr->flushAll();
    }
}

// This function will be called from Python to shut down the system
void shutdown_system() {
    if (bm_ptr != nullptr) {
        flush_system();
        delete bm_ptr;
        bm_ptr = nullptr;
    }
}