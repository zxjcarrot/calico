#ifndef CALICO_HPP
#define CALICO_HPP

#include <libaio.h>
#include <liburing.h>
#include <nmmintrin.h>  // For CRC32 and prefetch
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <span>
#include "hash_tables.hpp"
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#include "predictivetranslation-sigmod2026/third_party/concurrentqueue.hpp"

// Extern declarations must come BEFORE headers that use them
extern __thread uint16_t workerThreadId;
extern __thread int32_t tpcchistorycounter;

#include "exmap.h"
#include "tpcc/TPCCWorkload.hpp"
#include "tpcc/ZipfianGenerator.hpp"

using namespace tpcc;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64 PID;  // page id type

#ifndef CALICO_PAGE_SIZE
#define CALICO_PAGE_SIZE 4096
#endif

static constexpr u64 pageSize = CALICO_PAGE_SIZE;

__attribute__((target("sse4.2"))) uint64_t hash64(uint64_t key, uint32_t seed);

template <typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

struct alignas(pageSize) Page {
    bool dirty;
};

struct GraphNodePage {
    bool dirty;
    u32 neighborCount;
    u32 padding;
    u64 value;
    PID neighbors[1];

    static constexpr size_t headerSize() { return offsetof(GraphNodePage, neighbors); }
    static constexpr u32 maxNeighbors() { return static_cast<u32>((pageSize - headerSize()) / sizeof(PID)); }
    PID* neighborData() { return neighbors; }
    const PID* neighborData() const { return neighbors; }
};

static const int16_t maxWorkerThreads = 128;

#define die(msg)            \
    do {                    \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

uint64_t rdtsc();

void* allocHuge(size_t size);

void yield(u64 counter);

struct PageState {
    std::atomic<u64> stateAndVersion;

#if defined(CALICO_PAGESTATE_ALIGN64)
    u8 cachelinePadding[64 - sizeof(std::atomic<u64>)] = {};
#endif

    static const u64 Evicted = 0;
    static const u64 Unlocked = 1;
    static const u64 SharedBase = 2;
    static const u64 MaxShared = 252;
    static const u64 ReadInFlight = 253;
    static const u64 Locked = 254;
    static const u64 Marked = 255;

    static constexpr u64 stateMask = 0xFF00000000000000ULL;
    static constexpr u64 lowMask = 0x00FFFFFFFFFFFFFFULL;
    static constexpr u64 frameMask = 0x00000000FFFFFFFFULL;
    static constexpr u64 versionMaskPacked = 0x00FFFFFF00000000ULL;
    static constexpr u32 versionBitsPacked = 24;
    static constexpr u64 versionMaxPacked = (1ULL << versionBitsPacked) - 1;
    static constexpr u32 packedInvalidFrame = 0;  // Frame 0 is reserved and never allocated

    static bool packedMode;
    static void setPackedMode(bool value) { packedMode = value; }
    static bool isPackedMode() { return packedMode; }

    PageState() {}

    static inline u64 extractLow(u64 v) { return v & lowMask; }
    static inline u64 extractFrame(u64 v) { return v & frameMask; }

    static inline u64 compose(u64 state, u64 low) { return (state << 56) | (low & lowMask); }

    static inline u64 incrementLow(u64 oldStateAndVersion) {
        if (!packedMode) return (extractLow(oldStateAndVersion) + 1) & lowMask;
        u64 framePart = oldStateAndVersion & frameMask;
        u64 versionPart = (oldStateAndVersion & versionMaskPacked) >> 32;
        versionPart = (versionPart + 1) & versionMaxPacked;
        return ((versionPart << 32) & versionMaskPacked) | framePart;
    }

    void init() {
        // Zero-value entry design: all-zero represents evicted state
        stateAndVersion.store(0, std::memory_order_release);
    }

    static inline u64 sameVersion(u64 oldStateAndVersion, u64 newState) {
        return compose(newState, extractLow(oldStateAndVersion));
    }

    static inline u64 nextVersion(u64 oldStateAndVersion, u64 newState) {
        return compose(newState, incrementLow(oldStateAndVersion));
    }

    bool tryLockX(u64 oldStateAndVersion) {
        return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Locked));
    }

    void unlockX() {
        assert(getState() == Locked);
        stateAndVersion.store(nextVersion(stateAndVersion.load(), Unlocked), std::memory_order_release);
    }

    void unlockXEvicted() {
        assert(getState() == Locked);
        // Write zero for evicted state (zero-value entry design)
        stateAndVersion.store(0, std::memory_order_release);
    }

    void downgradeLock() {
        assert(getState() == Locked);
        stateAndVersion.store(nextVersion(stateAndVersion.load(), 1), std::memory_order_release);
    }

    bool tryLockS(u64 oldStateAndVersion) {
        u64 s = getState(oldStateAndVersion);
        if (s >= Unlocked && s < MaxShared)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, s + 1));
        if (s == Marked)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, SharedBase));
        return false;
    }

    void unlockS() {
        while (true) {
            u64 oldStateAndVersion = stateAndVersion.load();
            u64 state = getState(oldStateAndVersion);
            assert(state > Unlocked && state <= MaxShared);
            if (stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, state - 1)))
                return;
        }
    }

    bool tryMark(u64 oldStateAndVersion) {
        assert(getState(oldStateAndVersion) == Unlocked);
        return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Marked));
    }

    static u64 getState(u64 v) { return v >> 56; };
    u64 getState() { return getState(stateAndVersion.load()); }

    static inline bool isResidentState(u64 v) {
        const u64 state = getState(v);
        return state != Evicted && state != ReadInFlight;
    }

    static u64 getFrameValue(u64 v) {
        if (!packedMode) return 0;
        return v & frameMask;
    }

    u64 getFrameValue() const {
        if (!packedMode) return 0;
        return extractFrame(stateAndVersion.load(std::memory_order_relaxed));
    }

    void setFrameValue(u64 frame) {
        if (!packedMode) return;
        frame &= frameMask;
        u64 expected = stateAndVersion.load(std::memory_order_relaxed);
        assert(getState(expected) == Locked);
        u64 desired = (expected & ~frameMask) | frame;
        stateAndVersion.store(desired, std::memory_order_release);
        return;
    }

    void clearFrameValue() {
        if (packedMode) setFrameValue(packedInvalidFrame);
    }

    void operator=(PageState&) = delete;

    static u64 extractVersion(u64 v) {
        if (!packedMode) return 0;
        return (v & versionMaskPacked) >> 32;
    }
};

#if defined(CALICO_PAGESTATE_ALIGN64)
static_assert(sizeof(PageState) == 64, "PageState must occupy one cache line when CALICO_PAGESTATE_ALIGN64 is enabled");
#endif

struct AsyncReadRequest {
    enum Status : u32 { Idle = 0, Submitted = 1, Completed = 2, Failed = 3, Rescuing = 4 };

    std::atomic<PID> pid{std::numeric_limits<PID>::max()};
    std::atomic<u32> frame{std::numeric_limits<u32>::max()};
    std::atomic<uintptr_t> ownerInterface{0};
    std::atomic<u64> token{0};
    std::atomic<u64> submitMicros{0};
    std::atomic<u64> completionMicros{0};
    std::atomic<u64> lastWaitLogMicros{0};
    std::atomic<u32> submitWorker{std::numeric_limits<u32>::max()};
    std::atomic<u32> completionWorker{std::numeric_limits<u32>::max()};
    std::atomic<u32> waiterCount{0};
    std::atomic<int> result{0};
    std::atomic<u32> status{Idle};

    void reset() {
        pid.store(std::numeric_limits<PID>::max(), std::memory_order_release);
        frame.store(std::numeric_limits<u32>::max(), std::memory_order_release);
        ownerInterface.store(0, std::memory_order_release);
        token.store(0, std::memory_order_release);
        submitMicros.store(0, std::memory_order_release);
        completionMicros.store(0, std::memory_order_release);
        lastWaitLogMicros.store(0, std::memory_order_release);
        submitWorker.store(std::numeric_limits<u32>::max(), std::memory_order_release);
        completionWorker.store(std::numeric_limits<u32>::max(), std::memory_order_release);
        waiterCount.store(0, std::memory_order_release);
        result.store(0, std::memory_order_release);
        status.store(Idle, std::memory_order_release);
    }
};

struct ResidentPageSet {
    static const u64 empty = ~0ull;
    static const u64 tombstone = (~0ull) - 1;

    struct Entry {
        std::atomic<u64> pid;
    };

    Entry* ht;
    u64 count;
    u64 mask;
    std::atomic<u64> clockPos;

    ResidentPageSet(u64 maxCount);
    ~ResidentPageSet();

    u64 next_pow2(u64 x);
    u64 hash(u64 k);
    void insert(u64 pid);
    bool remove(u64 pid);

    template <class Fn>
    void iterateClockBatch(u64 batch, Fn fn) {
        u64 pos, newPos;
        do {
            pos = clockPos.load();
            newPos = (pos + batch) % count;
        } while (!clockPos.compare_exchange_strong(pos, newPos));

        for (u64 i = 0; i < batch; i++) {
            u64 curr = ht[pos].pid.load();
            if ((curr != tombstone) && (curr != empty)) fn(curr);
            pos = (pos + 1) & mask;
        }
    }
};

struct BufferManager;

struct IOInterface {
    virtual void writePages(const std::vector<PID>& pages) = 0;
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations) = 0;
    virtual bool supportsAsyncReads() const { return false; }
    virtual void pollAsyncCompletions() {}
    virtual bool submitAsyncRead(PID pid, Page* destination, AsyncReadRequest* request) {
        (void)pid;
        (void)destination;
        (void)request;
        return false;
    }
    virtual size_t submitAsyncReads(const PID* pids, Page* const* destinations, AsyncReadRequest* const* requests,
                                    size_t count) {
        size_t submitted = 0;
        for (; submitted < count; ++submitted) {
            if (!submitAsyncRead(pids[submitted], destinations[submitted], requests[submitted])) {
                break;
            }
        }
        return submitted;
    }
    virtual void waitForAsyncRead(AsyncReadRequest& request) { (void)request; }
    virtual ~IOInterface() {}
};

struct LibaioInterface: public IOInterface {
    static const u64 maxIOs = 256;

    int blockfd;
    BufferManager* bm_ptr;
    io_context_t ctx;
    iocb cb[maxIOs];
    iocb* cbPtr[maxIOs];
    io_event events[maxIOs];

    LibaioInterface(int blockfd, BufferManager* bm_ptr);
    ~LibaioInterface() { io_destroy(ctx); }
    virtual void writePages(const std::vector<PID>& pages);
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations);
};

struct IOUringInterface: public IOInterface {
    static const u64 maxIOs = 256;
    static const u64 queueDepth = 512;

    int blockfd;
    BufferManager* bm_ptr;
    struct io_uring ring;
    std::atomic_flag submitLock = ATOMIC_FLAG_INIT;
    std::atomic_flag completionLock = ATOMIC_FLAG_INIT;

    IOUringInterface(int blockfd, BufferManager* bm_ptr);
    ~IOUringInterface();
    virtual void writePages(const std::vector<PID>& pages);
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations);
    virtual bool supportsAsyncReads() const override { return true; }
    virtual void pollAsyncCompletions() override;
    virtual bool submitAsyncRead(PID pid, Page* destination, AsyncReadRequest* request) override;
    virtual size_t submitAsyncReads(const PID* pids, Page* const* destinations, AsyncReadRequest* const* requests,
                                    size_t count) override;
    virtual void waitForAsyncRead(AsyncReadRequest& request) override;

    void reapCompletions(bool wait_for_one);
};

// Two-level indirection for PageState array
struct TwoLevelPageStateArray {
    static constexpr u32 bottomBits = 19;  // Fixed at 23 bits for bottom-level index
    
    PageState** second_level_arrays;  // Top-level array of pointers to second-level arrays
    u32 num_top_slots;                // Number of top-level slots (2^topBits)
    u32 entries_per_slot;             // Number of entries per second-level array (2^bottomBits)
    u32 topBits;                      // Number of bits for top-level index
    u64 total_entries;                // Total number of entries
    
    TwoLevelPageStateArray(u64 virtCount);
    ~TwoLevelPageStateArray();
    
    inline PageState& get(PID pid) {
        u32 top_idx = static_cast<u32>(pid >> bottomBits);
        u32 bottom_idx = static_cast<u32>(pid & ((1ULL << bottomBits) - 1));
        return second_level_arrays[top_idx][bottom_idx];
    }
};

// Three-level indirection for PageState array
struct ThreeLevelPageStateArray {
    static constexpr u32 bottomBits = 19;   // Fixed at 19 bits for bottom-level index
    static constexpr u32 middleBits = 2;   // Fixed at 2 bits for middle-level index
    
    PageState*** top_level_arrays;    // Top-level array of pointers to middle-level arrays
    u32 num_top_slots;                // Number of top-level slots (2^topBits)
    u32 num_middle_slots;             // Number of middle-level slots per top slot (2^middleBits)
    u32 entries_per_slot;             // Number of entries per bottom-level array (2^bottomBits)
    u32 topBits;                      // Number of bits for top-level index
    u64 total_entries;                // Total number of entries
    
    ThreeLevelPageStateArray(u64 virtCount);
    ~ThreeLevelPageStateArray();
    
    inline PageState& get(PID pid) {
        u32 top_idx = static_cast<u32>(pid >> (bottomBits + middleBits));
        u32 middle_idx = static_cast<u32>((pid >> bottomBits) & ((1ULL << middleBits) - 1));
        u32 bottom_idx = static_cast<u32>(pid & ((1ULL << bottomBits) - 1));
        return top_level_arrays[top_idx][middle_idx][bottom_idx];
    }
};

struct PIDAllocator {
    u64 next{0};
    u64 end{0};
    std::atomic_flag lock = ATOMIC_FLAG_INIT;

    inline void lockSpin() {
        while (lock.test_and_set(std::memory_order_acquire)) _mm_pause();
    }

    inline void unlockSpin() { lock.clear(std::memory_order_release); }
};

struct BufferManager {
    static const u64 mb = 1024ull * 1024;
    static const u64 gb = 1024ull * 1024 * 1024;
    u64 virtSize;
    u64 physSize;
    u64 virtCount;
    u64 physCount;
    struct exmap_user_interface* exmapInterface[maxWorkerThreads];

    enum class HashMode {
        Array1Access = 0,
        Unordered = 1,
        OpenAddress = 2,
        Array = 3,
        Lockfree = 4,
        Array2Level = 5,
        Array3Level = 6,
        RobinHood = 7
    };

    bool useTraditional;
    bool useMmapOSPageacche = false;
    bool useExmap;
    HashMode hashMode;
    unsigned numThreads;
    int blockfd;
    int exmapfd;

    static constexpr u64 invalidFrame = std::numeric_limits<u64>::max();
    static constexpr PID invalidPID = std::numeric_limits<PID>::max();
    static constexpr u32 invalidAsyncRequestSlot = std::numeric_limits<u32>::max();
    char cachline_pad0[64];
    ResidentPageSet residentSet;

    std::atomic<u64> physUsedCount;
    char cachline_pad1[64];
    std::atomic<u64> allocCount;
    char cachline_pad2[64];
    std::atomic<u64> readCount;
    char cachline_pad3[64];
    std::atomic<u64> writeCount;
    char cachline_pad4[64];
    std::atomic<u64> holePunchCount;  // Counter for translation table hole-punching operations
    char cachline_pad5[64];
    std::atomic<u64> asyncSubmitCount;
    std::atomic<u64> asyncCoalescedCount;
    std::atomic<u64> asyncGuardWaitCount;
    std::atomic<u64> asyncSyncFaultCount;
    std::atomic<u64> asyncFailedCompletionCount;
    std::atomic<u64> asyncCompletionCount;
    std::atomic<u64> asyncSubmitBackpressureCount;
    std::atomic<u64> asyncSlowWaitCount;
    std::atomic<u64> asyncInFlightCurrent;
    std::atomic<u64> asyncInFlightPeak;
    std::atomic<u64> asyncInFlightWeightedMicros;
    std::atomic<u64> asyncInFlightStartMicros;
    std::atomic<u64> asyncInFlightLastTransitionMicros;
    mutable std::atomic_flag asyncInFlightStatsLock = ATOMIC_FLAG_INIT;

    Page* virtMem;
    Page* frameMem;
    PageState* pageState;                                    // Single-level array (for non-Array2Level/Array3Level modes)
    std::unique_ptr<TwoLevelPageStateArray> pageState2Level; // Two-level array (for Array2Level mode)
    std::unique_ptr<ThreeLevelPageStateArray> pageState3Level; // Three-level array (for Array3Level mode)
    std::unique_ptr<std::atomic<u64>[]> pidToFrameArray;
    bool enableTranslationRefCount;
    std::atomic<u32>* translationRefCounts;  // Reference counts for OS page groups (lazily allocated via mmap)
    u64 numRefCountGroups;  // Number of OS page groups in translation array
    u64 translationOSPageSize;  // OS page size used for translation array (4KB or 2MB based on huge pages)
    static constexpr u32 REF_COUNT_LOCK_BIT = 0x80000000u;  // Highest bit used as lock
    static constexpr u32 REF_COUNT_MASK = 0x7FFFFFFFu;      // Lower 31 bits for count
    static constexpr u64 NUM_LOCK_SHARDS = 64;
    std::unique_ptr<pthread_rwlock_t[]> pid_locks;
    std::unique_ptr<std::atomic<PID>[]> frameToPidArray;
    std::unique_ptr<PartitionedMap<PID, u64>> pidToFrameHash;
    std::unique_ptr<PartitionedMap<u64, PID>> frameToPidHash;
    moodycamel::ConcurrentQueue<u32> freeFrameQueue;
    std::unique_ptr<AsyncReadRequest[]> asyncReadRequests;
    std::unique_ptr<std::atomic<u32>[]> pidToAsyncRequestSlot;
    moodycamel::ConcurrentQueue<u32> freeAsyncRequestSlotQueue;
    std::vector<std::unique_ptr<IOUringInterface>> ioUringInterfaces;
    std::atomic<u64> nextAsyncReadToken;
    u64 batch;
    bool useIOUring;

    PageState& getPageState(PID pid) {
        // if (hashMode == HashMode::Array2Level) {
        //     return pageState2Level->get(pid);
        // }
        // if (hashMode == HashMode::Array3Level) {
        //     return pageState3Level->get(pid);
        // }
        return pageState[pid];
    }
    u64 pid_lock_shard(PID pid) {
        return pid % NUM_LOCK_SHARDS;
    }

    void flushAll();

    BufferManager(unsigned nthreads = 1);
    ~BufferManager();

    Page* fixX(PID pid);
    void unfixX(PID pid);
    Page* fixS(PID pid);
    void unfixS(PID pid);

    bool isValidPtr(void* page);
    PID toPID(void* page);
    Page* toPtr(PID pid);
    Page* residentPtr(PID pid);
    Page* residentPtr(PID pid, u64 stateAndVersion);
    Page* residentPtrCalico(PID pid, u64 stateAndVersion);
    Page* residentPtrHash(PID pid, u64 stateAndVersion);
    Page* preparePageForWrite(PID pid);

    void ensureFreePages();
    Page* allocPage(PIDAllocator* allocator = nullptr);
    void updateAllocCountSnapshot(u64 value);
    Page* handleFault(PID pid);
    Page* resolveOptimisticRead(PID pid, u64& version);
    Page* waitForPendingRead(PID pid, u64 stateAndVersion, u64* resolvedVersion = nullptr);
    void readPage(PID pid, Page* dest);
    void evict();
    bool submitAsyncPrefetch(PID pid);
    void submitAsyncPrefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);
    void prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);
    void prefetchResidentPages(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);

    Page* acquireFrameForPid(PID pid, bool allow_eviction = true);
    void releaseFrame(PID pid);
    u32 popFreeFrame(bool allow_eviction = true);
    u64 resolveMappedFrame(PID pid, u64 stateAndVersion);
    u32 resolveAsyncRequestSlot(PID pid, u64 stateAndVersion);
    void releaseAsyncRequestSlot(PID pid, u32 slot);
    bool finalizeCompletedAsyncRead(PID pid, u32 slot);
    void recordAsyncInFlightTransition(u64 now_micros, int delta);
    double averageAsyncInFlightIO(u64 now_micros) const;
    void rollbackFailedAsyncRead(PID pid, u64 stateAndVersion);
    bool asyncReadEnabled() const;
    
    IOInterface& getIOInterface();

    u64 countZeroRefCountGroups();
};
typedef u64 KeyType;

extern BufferManager* bm_ptr;
// Define a convenience reference to avoid changing all code that uses bm
#define bm (*bm_ptr)

struct OLCRestartException {};

template <class T>
struct GuardO {
    PID pid;
    T* ptr;
    u64 version;
    static const u64 moved = ~0ull;

    // constructor
    explicit GuardO(u64 pid) : pid(pid), ptr(nullptr) { init(); }

    template <class T2>
    GuardO(u64 pid, GuardO<T2>& parent) {
        parent.checkVersionAndRestart();
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    GuardO(GuardO&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
    }

    void init() {
        assert(pid != moved);
        ptr = reinterpret_cast<T*>(bm.resolveOptimisticRead(pid, version));
    }

    // move assignment operator
    GuardO& operator=(GuardO&& other) {
        if (pid != moved) checkVersionAndRestart();
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // assignment operator
    GuardO& operator=(const GuardO&) = delete;

    // copy constructor
    GuardO(const GuardO&) = delete;

    void checkVersionAndRestart() {
        if (pid != moved) {
            PageState& ps = bm.getPageState(pid);
            u64 stateAndVersion = ps.stateAndVersion.load();
            if (version == stateAndVersion)  // fast path, nothing changed
                return;
            if ((stateAndVersion << 8) == (version << 8)) {  // same version
                u64 state = PageState::getState(stateAndVersion);
                if (state > PageState::Unlocked && state <= PageState::MaxShared) return;  // ignore shared locks
                if (state == PageState::Marked)
                    if (ps.stateAndVersion.compare_exchange_weak(
                            stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                        return;  // mark cleared
            }
            if (std::uncaught_exceptions() == 0) throw OLCRestartException();
        }
    }

    // destructor
    ~GuardO() noexcept(false) { checkVersionAndRestart(); }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        checkVersionAndRestart();
        pid = moved;
        ptr = nullptr;
    }
};

// Guard that allows relaxed checking of version changes.
// Useful the data structures can tolerate inconsistent reads, e.g. graph nodes in ANNS index.
template <class T>
struct GuardORelaxed {
    PID pid;
    T* ptr;
    u64 version;
    static const u64 moved = ~0ull;

    // constructor
    explicit GuardORelaxed(u64 pid) : pid(pid), ptr(nullptr) { init(); }

    template <class T2>
    GuardORelaxed(u64 pid, GuardO<T2>& parent) {
        // we do not need to check the parent's version here, as we are relaxed
        //parent.checkVersionAndRestart();
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    template <class T2>
    GuardORelaxed(GuardO<T2>&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
    }


    template <class T2>
    GuardORelaxed(u64 pid, GuardORelaxed<T2>& parent) {
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    GuardORelaxed(GuardORelaxed&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
    }


    void init() {
        assert(pid != moved);
        ptr = reinterpret_cast<T*>(bm.resolveOptimisticRead(pid, version));
    }

    // move assignment operator
    GuardORelaxed& operator=(GuardORelaxed&& other) {
        if (pid != moved) checkVersionAndRestart();
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // assignment operator
    GuardORelaxed& operator=(const GuardORelaxed&) = delete;

    // copy constructor
    GuardORelaxed(const GuardORelaxed&) = delete;

    void checkVersionAndRestart() {
        if (pid != moved) {
            PageState& ps = bm.getPageState(pid);
            u64 stateAndVersion = ps.stateAndVersion.load();
            if (version == stateAndVersion)  // fast path, nothing changed
                return;
            u64 oldOCCVersion = PageState::extractVersion(version);
            u64 currentOCCVersion = PageState::extractVersion(stateAndVersion);
            u64 oldFrameValue = PageState::getFrameValue(version);
            u64 currentFrameValue = PageState::getFrameValue(stateAndVersion);
            if (oldFrameValue != currentFrameValue) {
                // frame changed, must restart. 
                // This is because we might be reading data that is from a different pages which suggests a significant change.
                if (std::uncaught_exceptions() == 0) throw OLCRestartException();
            }
            if (oldOCCVersion == currentOCCVersion) {  // same version
                u64 state = PageState::getState(stateAndVersion);
                if (state > PageState::Unlocked && state <= PageState::MaxShared) return;  // ignore shared locks
                if (state == PageState::Marked)
                    if (ps.stateAndVersion.compare_exchange_weak(
                            stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                        return;  // mark cleared
            }
            // it is fine if the version changed, we are relaxed
        }
    }

    // destructor
    ~GuardORelaxed() noexcept(false) { 
        //checkVersionAndRestart(); 
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        checkVersionAndRestart();
        pid = moved;
        ptr = nullptr;
    }
};


template <class T>
struct GuardX {
    PID pid;
    T* ptr;
    static const u64 moved = ~0ull;

    // constructor
    GuardX() : pid(moved), ptr(nullptr) {}

    // constructor
    explicit GuardX(u64 pid) : pid(pid) {
        ptr = reinterpret_cast<T*>(bm.fixX(pid));
        ptr->dirty = true;
    }

    explicit GuardX(GuardO<T>&& other) {
        assert(other.pid != moved);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            PageState& ps = bm.getPageState(other.pid);
            u64 stateAndVersion = ps.stateAndVersion;
            if ((stateAndVersion << 8) != (other.version << 8)) throw OLCRestartException();
            u64 state = PageState::getState(stateAndVersion);
            if ((state == PageState::Unlocked) || (state == PageState::Marked)) {
                if (ps.tryLockX(stateAndVersion)) {
                    pid = other.pid;
                    ptr = other.ptr;
                    ptr->dirty = true;
                    other.pid = moved;
                    other.ptr = nullptr;
                    return;
                }
            }
            yield(repeatCounter);
        }
    }

    // assignment operator
    GuardX& operator=(const GuardX&) = delete;

    // move assignment operator
    GuardX& operator=(GuardX&& other) {
        if (pid != moved) {
            bm.unfixX(pid);
        }
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // copy constructor
    GuardX(const GuardX&) = delete;

    // destructor
    ~GuardX() {
        if (pid != moved) bm.unfixX(pid);
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        if (pid != moved) {
            bm.unfixX(pid);
            pid = moved;
        }
    }
};

template <class T>
struct AllocGuard : public GuardX<T> {
    template <typename... Params>
    AllocGuard(PIDAllocator* allocator, Params&&... params) {
        GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage(allocator));
        new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
        GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
    }

    template <typename... Params>
    AllocGuard(Params&&... params) : AllocGuard(static_cast<PIDAllocator*>(nullptr), std::forward<Params>(params)...) {}
};

template <class T>
struct GuardS {
    PID pid;
    T* ptr;
    static const u64 moved = ~0ull;

    // constructor
    explicit GuardS(u64 pid) : pid(pid) { ptr = reinterpret_cast<T*>(bm.fixS(pid)); }

    GuardS(GuardO<T>&& other) {
        assert(other.pid != moved);
        if (bm.getPageState(other.pid).tryLockS(other.version)) {  // XXX: optimize?
            pid = other.pid;
            ptr = other.ptr;
            other.pid = moved;
            other.ptr = nullptr;
        } else {
            throw OLCRestartException();
        }
    }

    GuardS(GuardS&& other) {
        if (pid != moved) bm.unfixS(pid);
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
    }

    // assignment operator
    GuardS& operator=(const GuardS&) = delete;

    // move assignment operator
    GuardS& operator=(GuardS&& other) {
        if (pid != moved) bm.unfixS(pid);
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // copy constructor
    GuardS(const GuardS&) = delete;

    // destructor
    ~GuardS() {
        if (pid != moved) bm.unfixS(pid);
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        if (pid != moved) {
            bm.unfixS(pid);
            pid = moved;
        }
    }
};

u64 envOr(const char* env, u64 value);
double envOrDouble(const char* env, double value);

//---------------------------------------------------------------------------

struct BTreeNode;

template <class T>
static inline T loadUnaligned(void* p) {
    T x;
    memcpy(&x, p, sizeof(T));
    return x;
}

// Keep the B-tree comparison fast path inline so the compiler can fold it into
// lookup/update/scan call sites.
static inline int cust_memcmp(const void* ptr1, const void* ptr2, size_t num) {
    const unsigned char* p1 = static_cast<const unsigned char*>(ptr1);
    const unsigned char* p2 = static_cast<const unsigned char*>(ptr2);
    while (num-- > 0) {
        if (*p1 != *p2) return static_cast<int>(*p1) - static_cast<int>(*p2);
        ++p1;
        ++p2;
    }
    return 0;
}

// Get order-preserving head of key (assuming little endian)
static inline u32 head(u8* key, unsigned keyLen) {
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

struct BTreeNodeHeader {
    static const unsigned underFullSize = (pageSize / 2) + (pageSize / 4);  // merge nodes more empty
    static const u64 noNeighbour = ~0ull;

    struct FenceKeySlot {
        u16 offset;
        u16 len;
    };

    bool dirty;
    union {
        PID upperInnerNode;              // inner
        PID nextLeafNode = noNeighbour;  // leaf
    };

    bool hasRightNeighbour() { return nextLeafNode != noNeighbour; }

    FenceKeySlot lowerFence = {0, 0};  // exclusive
    FenceKeySlot upperFence = {0, 0};  // inclusive

    bool hasLowerFence() { return !!lowerFence.len; };

    u16 count = 0;
    bool isLeaf;
    u16 spaceUsed = 0;
    u16 dataOffset = static_cast<u16>(pageSize);
    u16 prefixLen = 0;

    static const unsigned hintCount = 16;
    u32 hint[hintCount];
    u32 padding;

    BTreeNodeHeader(bool isLeaf) : isLeaf(isLeaf) {}
    ~BTreeNodeHeader() {}
};

struct BTreeNode : public BTreeNodeHeader {
    struct Slot {
        u16 offset;
        u16 keyLen;
        u16 payloadLen;
        union {
            u32 head;
            u8 headBytes[4];
        };
    } __attribute__((packed));
    union {
        Slot slot[(pageSize - sizeof(BTreeNodeHeader)) / sizeof(Slot)];  // grows from front
        u8 heap[pageSize - sizeof(BTreeNodeHeader)];                     // grows from back
    };

    static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 2;

    BTreeNode(bool isLeaf) : BTreeNodeHeader(isLeaf) { dirty = true; }

    u8* ptr() { return reinterpret_cast<u8*>(this); }
    bool isInner() { return !isLeaf; }
    std::span<u8> getLowerFence() { return {ptr() + lowerFence.offset, lowerFence.len}; }
    std::span<u8> getUpperFence() { return {ptr() + upperFence.offset, upperFence.len}; }
    u8* getPrefix() { return ptr() + lowerFence.offset; }  // any key on page is ok

    unsigned freeSpace() { return dataOffset - (reinterpret_cast<u8*>(slot + count) - ptr()); }
    unsigned freeSpaceAfterCompaction() { return pageSize - (reinterpret_cast<u8*>(slot + count) - ptr()) - spaceUsed; }

    bool hasSpaceFor(unsigned keyLen, unsigned payloadLen);

    u8* getKey(unsigned slotId) { return ptr() + slot[slotId].offset; }
    std::span<u8> getPayload(unsigned slotId) {
        return {ptr() + slot[slotId].offset + slot[slotId].keyLen, slot[slotId].payloadLen};
    }
    PID getChild(unsigned slotId);

    unsigned spaceNeeded(unsigned keyLen, unsigned payloadLen) {
        return sizeof(Slot) + (keyLen - prefixLen) + payloadLen;
    }

    void makeHint();
    void updateHint(unsigned slotId);
    void searchHint(u32 keyHead, u16& lowerOut, u16& upperOut);
    u16 lowerBound(std::span<u8> skey, bool& foundExactOut) {
        foundExactOut = false;

        int cmp = cust_memcmp(skey.data(), getPrefix(), std::min(skey.size(), static_cast<size_t>(prefixLen)));
        if (cmp < 0) return 0;
        if (cmp > 0) return count;
        if (skey.size() < prefixLen) return 0;

        u8* key = skey.data() + prefixLen;
        unsigned keyLen = skey.size() - prefixLen;

        u16 lower = 0;
        u16 upper = count;
        u32 keyHead = head(key, keyLen);
        searchHint(keyHead, lower, upper);

        while (lower < upper) {
            u16 mid = ((upper - lower) / 2) + lower;
            if (keyHead < slot[mid].head) {
                upper = mid;
            } else if (keyHead > slot[mid].head) {
                lower = mid + 1;
            } else {
                int cmp = cust_memcmp(key, getKey(mid), std::min(static_cast<size_t>(keyLen), static_cast<size_t>(slot[mid].keyLen)));
                if (cmp < 0) {
                    upper = mid;
                } else if (cmp > 0) {
                    lower = mid + 1;
                } else if (keyLen < slot[mid].keyLen) {
                    upper = mid;
                } else if (keyLen > slot[mid].keyLen) {
                    lower = mid + 1;
                } else {
                    foundExactOut = true;
                    return mid;
                }
            }
        }
        return lower;
    }
    u16 lowerBound(std::span<u8> key) {
        bool ignore;
        return lowerBound(key, ignore);
    }
    void insertInPage(std::span<u8> key, std::span<u8> payload);
    bool removeSlot(unsigned slotId);
    bool removeInPage(std::span<u8> key);
    void copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src);
    void compactify();
    bool mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right);
    void storeKeyValue(u16 slotId, std::span<u8> skey, std::span<u8> payload);
    void copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount);
    void copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot);
    void insertFence(FenceKeySlot& fk, std::span<u8> key);
    void setFences(std::span<u8> lower, std::span<u8> upper);
    void splitNode(BTreeNode* parent, unsigned sepSlot, std::span<u8> sep, PIDAllocator* allocator);

    struct SeparatorInfo {
        unsigned len;      // len of new separator
        unsigned slot;     // slot at which we split
        bool isTruncated;  // if true, we truncate the separator taking len bytes from slot+1
    };

    unsigned commonPrefix(unsigned slotA, unsigned slotB);
    SeparatorInfo findSeparator(bool splitOrdered);
    void getSep(u8* sepKeyOut, SeparatorInfo info);
    PID lookupInner(std::span<u8> key);
};

static_assert(sizeof(BTreeNode) == pageSize, "btree node size problem");

static const u64 metadataPageId = 0;

struct HNSWMetaInfo {
    static constexpr u64 magic = 0x484E53574D455441ULL;  // "HNSWMETA"
    u64 magic_value = 0;
    PID metadata_pid = BufferManager::invalidPID;
    PID base_pid = BufferManager::invalidPID;
    u64 max_elements = 0;
    u64 dim = 0;
    u64 M = 0;
    u64 ef_construction = 0;
    u64 max_level = 0;
    u8 valid = 0;
    u8 reserved[6] = {0};

    bool isValid() const {
        return valid != 0 && magic_value == magic && metadata_pid != BufferManager::invalidPID &&
               base_pid != BufferManager::invalidPID;
    }

    void clear() {
        magic_value = 0;
        metadata_pid = BufferManager::invalidPID;
        base_pid = BufferManager::invalidPID;
        max_elements = 0;
        dim = 0;
        M = 0;
        ef_construction = 0;
        max_level = 0;
        valid = 0;
        memset(reserved, 0, sizeof(reserved));
    }
};

struct IVFMetaInfo {
    static constexpr u64 magic = 0x4956464D45544131ULL;  // "IVFMETA1"
    u64 magic_value = 0;
    PID metadata_pid = BufferManager::invalidPID;
    u64 max_elements = 0;
    u32 dim = 0;
    u32 num_clusters = 0;
    u32 retrain_interval = 0;
    u8 is_trained = 0;
    u8 valid = 0;
    u8 reserved[6] = {0};

    bool isValid() const {
        return valid != 0 && magic_value == magic && metadata_pid != BufferManager::invalidPID;
    }

    void clear() {
        magic_value = 0;
        metadata_pid = BufferManager::invalidPID;
        max_elements = 0;
        dim = 0;
        num_clusters = 0;
        retrain_interval = 0;
        is_trained = 0;
        valid = 0;
        memset(reserved, 0, sizeof(reserved));
    }
};

struct MetaDataPage {
    bool dirty;
    u8 padding[7] = {0};
    HNSWMetaInfo hnsw_meta;
    IVFMetaInfo ivf_meta;
    u64 alloc_count_snapshot = 0;
    PID roots[(pageSize - sizeof(dirty) - sizeof(padding) - sizeof(HNSWMetaInfo) - sizeof(IVFMetaInfo) - sizeof(alloc_count_snapshot)) /
              sizeof(PID)];

    PID getRoot(unsigned slot) { return roots[slot]; }

    u64 getAllocCountSnapshot() const { return alloc_count_snapshot; }

    void setAllocCountSnapshot(u64 value) { alloc_count_snapshot = value; }
};

void flush_system();
void warmup_storage_to_alloc_count();
std::vector<u64> snapshot_pid_to_frame_range(u64 start_pid, u64 count);

static_assert(sizeof(MetaDataPage) <= pageSize, "MetaDataPage must fit within a single page");

struct BTree {
   private:
    void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, std::span<u8> key, unsigned payloadLen);
    void ensureSpace(BTreeNode* toSplit, std::span<u8> key, unsigned payloadLen);

   public:
    unsigned slotId;
    std::atomic<bool> splitOrdered;
    PIDAllocator pidAllocator;

    BTree();
    ~BTree();

    GuardO<BTreeNode> findLeafO(std::span<u8> key);
    GuardS<BTreeNode> findLeafS(std::span<u8> key);

    int lookup(std::span<u8> key, u8* payloadOut, unsigned payloadOutSize) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return -1;

                // key found, copy payload
                 memcpy(payloadOut, node->getPayload(pos).data(), std::min(static_cast<unsigned>(node->slot[pos].payloadLen), payloadOutSize));
                return node->slot[pos].payloadLen;
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    template <class Fn>
    bool lookup(std::span<u8> key, Fn fn) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return false;

                // key found
                fn(node->getPayload(pos));
                return true;
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    void insert(std::span<u8> key, std::span<u8> payload);
    bool remove(std::span<u8> key);

    template <class Fn>
    bool updateInPlace(std::span<u8> key, Fn fn) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return false;

                {
                    GuardX<BTreeNode> nodeLocked(std::move(node));
                    fn(nodeLocked->getPayload(pos));
                    return true;
                }
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    template <class Fn>
    void scanAsc(std::span<u8> key, Fn fn) {
        GuardS<BTreeNode> node = findLeafS(key);
        bool found;
        unsigned pos = node->lowerBound(key, found);
        bool next_prefetched = false;
        for (u64 repeatCounter = 0;; repeatCounter++) {  // XXX
            // if (!next_prefetched) {
            //     next_prefetched = true;
            //     u32 offset = 0;
            //     if (node->hasRightNeighbour()) {
            //         bm.prefetchResidentPages(&node->nextLeafNode, 1, &offset);
            //     }
            // }
            if (pos < node->count) {
                if (!fn(*node.ptr, pos)) return;
                pos++;
            } else {
                if (!node->hasRightNeighbour()) return;
                pos = 0;
                node = GuardS<BTreeNode>(node->nextLeafNode);
                next_prefetched = false;
            }
        }
    }

    template <class Fn>
    void scanDesc(std::span<u8> key, Fn fn) {
        GuardS<BTreeNode> node = findLeafS(key);
        bool exactMatch;
        int pos = node->lowerBound(key, exactMatch);
        if (pos == node->count) {
            pos--;
            exactMatch = true;  // XXX:
        }
        for (u64 repeatCounter = 0;; repeatCounter++) {  // XXX
            while (pos >= 0) {
                if (!fn(*node.ptr, pos, exactMatch)) return;
                pos--;
            }
            if (!node->hasLowerFence()) return;
            node = findLeafS(node->getLowerFence());
            pos = node->count - 1;
        }
    }
};

template <class Record>
struct vmcacheAdapter {
    BTree tree;

   public:
    void scan(const typename Record::Key& key,
              const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb,
              std::function<void()> reset_if_scan_failed_cb) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        u8 kk[Record::maxFoldLength()];
        tree.scanAsc({k, l}, [&](BTreeNode& node, unsigned slot) {
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            typename Record::Key typedKey;
            Record::unfoldKey(kk, typedKey);
            auto payload = node.getPayload(slot);
            assert(payload.size() == sizeof(Record));
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(payload.data()));
        });
    }
    // -------------------------------------------------------------------------------------
    void scanDesc(const typename Record::Key& key,
                  const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb,
                  std::function<void()> reset_if_scan_failed_cb) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        u8 kk[Record::maxFoldLength()];
        bool first = true;
        tree.scanDesc({k, l}, [&](BTreeNode& node, unsigned slot, bool exactMatch) {
            if (first) {  // XXX: hack
                first = false;
                if (!exactMatch) return true;
            }
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            typename Record::Key typedKey;
            Record::unfoldKey(kk, typedKey);
            auto payload = node.getPayload(slot);
            assert(payload.size() == sizeof(Record));
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(payload.data()));
        });
    }
    // -------------------------------------------------------------------------------------
    void insert(const typename Record::Key& key, const Record& record) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        tree.insert({k, l}, {(u8*)(&record), sizeof(Record)});
    }
    // -------------------------------------------------------------------------------------
    template <class Fn>
    void lookup1(const typename Record::Key& key, Fn fn) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        bool succ = tree.lookup({k, l}, [&](std::span<u8> payload) {
            assert(payload.size() == sizeof(Record));
            fn(*reinterpret_cast<const Record*>(payload.data()));
        });
        assert(succ);
    }
    // -------------------------------------------------------------------------------------
    template <class Fn>
    void update1(const typename Record::Key& key, Fn fn) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        tree.updateInPlace({k, l}, [&](std::span<u8> payload) {
            assert(payload.size() == sizeof(Record));
            fn(*reinterpret_cast<Record*>(payload.data()));
        });
    }
    // -------------------------------------------------------------------------------------
    // Returns false if the record was not found
    bool erase(const typename Record::Key& key) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        return tree.remove({k, l});
    }
    // -------------------------------------------------------------------------------------
    template <class Field>
    Field lookupField(const typename Record::Key& key, Field Record::* f) {
        Field value;
        lookup1(key, [&](const Record& r) { value = r.*f; });
        return value;
    }

    u64 count() {
        u64 cnt = 0;
        tree.scanAsc({(u8*)nullptr, 0}, [&](BTreeNode& node, unsigned slot) {
            cnt++;
            return true;
        });
        return cnt;
    }

    u64 countw(Integer w_id) {
        u8 k[sizeof(Integer)];
        fold(k, w_id);
        u64 cnt = 0;
        u8 kk[Record::maxFoldLength()];
        tree.scanAsc({k, sizeof(Integer)}, [&](BTreeNode& node, unsigned slot) {
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            if (memcmp(k, kk, sizeof(Integer)) != 0) return false;
            cnt++;
            return true;
        });
        return cnt;
    }
};

template <class Fn>
void parallel_for(uint64_t begin, uint64_t end, uint64_t nthreads, Fn fn) {
    std::vector<std::thread> threads;
    uint64_t n = end - begin;
    if (n < nthreads) nthreads = n;
    uint64_t perThread = n / nthreads;
    for (unsigned i = 0; i < nthreads; i++) {
        threads.emplace_back([&, i]() {
            uint64_t b = (perThread * i) + begin;
            uint64_t e = (i == (nthreads - 1)) ? end : (b + perThread);
            fn(i, b, e);
        });
    }
    for (auto& t : threads) t.join();
}

#endif  // CALICO_HPP
