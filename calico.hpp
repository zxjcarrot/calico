#ifndef CALICO_HPP
#define CALICO_HPP

#include <libaio.h>
#include <nmmintrin.h>  // For CRC32 and prefetch
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "hash_tables.hpp"

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

static const u64 pageSize = 4096;

__attribute__((target("sse4.2"))) uint64_t hash64(uint64_t key, uint32_t seed);

template <typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

struct alignas(4096) Page {
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

    static const u64 Unlocked = 0;
    static const u64 MaxShared = 252;
    static const u64 Locked = 253;
    static const u64 Marked = 254;
    static const u64 Evicted = 255;

    static constexpr u64 stateMask = 0xFF00000000000000ULL;
    static constexpr u64 lowMask = 0x00FFFFFFFFFFFFFFULL;
    static constexpr u64 frameMask = 0x00000000FFFFFFFFULL;
    static constexpr u64 versionMaskPacked = 0x00FFFFFF00000000ULL;
    static constexpr u32 versionBitsPacked = 24;
    static constexpr u64 versionMaxPacked = (1ULL << versionBitsPacked) - 1;
    static constexpr u32 packedInvalidFrame = 0xFFFFFFFFu;

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
        u64 low = 0;
        if (packedMode) low = (0ULL << 32) | static_cast<u64>(packedInvalidFrame);
        stateAndVersion.store(compose(Evicted, low), std::memory_order_release);
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
        stateAndVersion.store(nextVersion(stateAndVersion.load(), Evicted), std::memory_order_release);
    }

    void downgradeLock() {
        assert(getState() == Locked);
        stateAndVersion.store(nextVersion(stateAndVersion.load(), 1), std::memory_order_release);
    }

    bool tryLockS(u64 oldStateAndVersion) {
        u64 s = getState(oldStateAndVersion);
        if (s < MaxShared)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, s + 1));
        if (s == Marked)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, 1));
        return false;
    }

    void unlockS() {
        while (true) {
            u64 oldStateAndVersion = stateAndVersion.load();
            u64 state = getState(oldStateAndVersion);
            assert(state > 0 && state <= MaxShared);
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

    static u64 getFrameValue(u64 v) {
        if (!packedMode) return 0;
        return v & frameMask;
    }

    u64 getFrameValue() const {
        if (!packedMode) return 0;
        return extractFrame(stateAndVersion.load(std::memory_order_acquire));
    }

    void setFrameValue(u64 frame) {
        if (!packedMode) return;
        frame &= frameMask;
        u64 expected = stateAndVersion.load(std::memory_order_relaxed);
        while (true) {
            u64 desired = (expected & ~frameMask) | frame;
            if (stateAndVersion.compare_exchange_weak(expected, desired, std::memory_order_release,
                                                      std::memory_order_relaxed))
                return;
        }
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

struct LibaioInterface {
    static const u64 maxIOs = 256;

    int blockfd;
    BufferManager* bm_ptr;
    io_context_t ctx;
    iocb cb[maxIOs];
    iocb* cbPtr[maxIOs];
    io_event events[maxIOs];

    LibaioInterface(int blockfd, BufferManager* bm_ptr);
    void writePages(const std::vector<PID>& pages);
    void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations);
};

struct FreePartition {
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::vector<u32> frames;
    u32 top = 0;
};

struct BufferManager {
    static const u64 mb = 1024ull * 1024;
    static const u64 gb = 1024ull * 1024 * 1024;
    u64 virtSize;
    u64 physSize;
    u64 virtCount;
    u64 physCount;
    struct exmap_user_interface* exmapInterface[maxWorkerThreads];
    std::vector<LibaioInterface> libaioInterface;

    enum class HashMode { Array1Access = 0, Unordered = 1, OpenAddress = 2, Array = 3, Lockfree = 4 };

    bool useTraditional;
    bool useExmap;
    HashMode hashMode;
    int blockfd;
    int exmapfd;

    static constexpr u64 invalidFrame = std::numeric_limits<u64>::max();
    static constexpr PID invalidPID = std::numeric_limits<PID>::max();
    static constexpr u32 freePartitionCount = 64;

    std::atomic<u64> physUsedCount;
    ResidentPageSet residentSet;
    std::atomic<u64> allocCount;

    std::atomic<u64> readCount;
    std::atomic<u64> writeCount;

    Page* virtMem;
    Page* frameMem;
    PageState* pageState;
    std::unique_ptr<std::atomic<u64>[]> pidToFrameArray;
    std::unique_ptr<std::atomic<PID>[]> frameToPidArray;
    std::unique_ptr<PartitionedMap<PID, u64>> pidToFrameHash;
    std::unique_ptr<PartitionedMap<u64, PID>> frameToPidHash;
    std::array<FreePartition, freePartitionCount> freePartitions;
    u32 partitionFrameSpan;
    u64 batch;

    PageState& getPageState(PID pid) { return pageState[pid]; }

    BufferManager();
    ~BufferManager();

    Page* fixX(PID pid);
    void unfixX(PID pid);
    Page* fixS(PID pid);
    void unfixS(PID pid);

    bool isValidPtr(void* page);
    PID toPID(void* page);
    Page* toPtr(PID pid);
    Page* residentPtr(PID pid);
    Page* residentPtr(PID pid, PageState& ps, u64 stateAndVersion);
    Page* preparePageForWrite(PID pid);

    void ensureFreePages();
    Page* allocPage();
    Page* handleFault(PID pid);
    void readPage(PID pid, Page* dest);
    void evict();
    void prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);

    Page* acquireFrameForPid(PID pid);
    void releaseFrame(PID pid);
    u32 popFreeFrame();

    u32 partitionForFrame(u64 frame) const;
    void lockPartition(u32 partition);
    void unlockPartition(u32 partition);
};

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
        PageState& ps = bm.getPageState(pid);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                case PageState::Marked: {
                    u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                    if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                        ptr = reinterpret_cast<T*>(bm.residentPtr(pid, ps, newV));
                        version = newV;
                        return;
                    }
                    break;
                }
                case PageState::Locked:
                    break;
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                        bm.unfixX(pid);
                    }
                    break;
                default:
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, ps, v));
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
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
                if (state <= PageState::MaxShared) return;  // ignore shared locks
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
        PageState& ps = bm.getPageState(pid);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                case PageState::Marked: {
                    u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                    if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                        ptr = reinterpret_cast<T*>(bm.residentPtr(pid, ps, newV));
                        version = newV;
                        return;
                    }
                    break;
                }
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                        bm.unfixX(pid);
                    }
                    break;
                case PageState::Locked:// we can live with stale data here, fallthrough
                default:
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, ps, v));
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
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
                if (state <= PageState::MaxShared) return;  // ignore shared locks
                if (state == PageState::Marked)
                    if (ps.stateAndVersion.compare_exchange_weak(
                            stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                        return;  // mark cleared
            }
            // it is fine if the version changed, we are relaxed
        }
    }

    // destructor
    ~GuardORelaxed() noexcept(false) { checkVersionAndRestart(); }

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
    AllocGuard(Params&&... params) {
        GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage());
        new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
        GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
    }
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

    static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 4;

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
    u16 lowerBound(std::span<u8> skey, bool& foundExactOut);
    u16 lowerBound(std::span<u8> key);
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
    void splitNode(BTreeNode* parent, unsigned sepSlot, std::span<u8> sep);

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

struct MetaDataPage {
    bool dirty;
    PID roots[(pageSize - 8) / 8];

    PID getRoot(unsigned slot) { return roots[slot]; }
};

struct BTree {
   private:
    void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, std::span<u8> key, unsigned payloadLen);
    void ensureSpace(BTreeNode* toSplit, std::span<u8> key, unsigned payloadLen);

   public:
    unsigned slotId;
    std::atomic<bool> splitOrdered;

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
        for (u64 repeatCounter = 0;; repeatCounter++) {  // XXX
            if (pos < node->count) {
                if (!fn(*node.ptr, pos)) return;
                pos++;
            } else {
                if (!node->hasRightNeighbour()) return;
                pos = 0;
                node = GuardS<BTreeNode>(node->nextLeafNode);
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
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
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
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
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
        bool succ =
            tree.lookup({k, l}, [&](std::span<u8> payload) { fn(*reinterpret_cast<const Record*>(payload.data())); });
        assert(succ);
    }
    // -------------------------------------------------------------------------------------
    template <class Fn>
    void update1(const typename Record::Key& key, Fn fn) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        tree.updateInPlace({k, l}, [&](std::span<u8> payload) { fn(*reinterpret_cast<Record*>(payload.data())); });
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