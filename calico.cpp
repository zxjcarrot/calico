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
#include <cstdlib>
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
const u64 IOUringInterface::maxIOs;
const u64 IOUringInterface::queueDepth;

static u64 monotonicMicros() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count());
}

static constexpr bool kEnableAsyncPrefetchStats = false;

static bool asyncDiagnosticsEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("CALICO_ASYNC_DIAGNOSTICS");
        if (raw == nullptr || raw[0] == '\0') {
            return false;
        }
        char* end = nullptr;
        return std::strtol(raw, &end, 10) != 0 && end != raw;
    }();
    return enabled;
}

static u64 asyncWaitWarnMicros() {
    static const u64 warn_us = []() {
        const char* raw = std::getenv("CALICO_ASYNC_WAIT_WARN_US");
        if (raw == nullptr || raw[0] == '\0') {
            return static_cast<u64>(50000);
        }
        char* end = nullptr;
        const long long parsed = std::strtoll(raw, &end, 10);
        if (end == raw || parsed <= 0) {
            return static_cast<u64>(50000);
        }
        return static_cast<u64>(parsed);
    }();
    return warn_us;
}

static u64 asyncWaitLogEveryMicros() {
    static const u64 period_us = []() {
        const char* raw = std::getenv("CALICO_ASYNC_WAIT_LOG_EVERY_US");
        if (raw == nullptr || raw[0] == '\0') {
            return static_cast<u64>(250000);
        }
        char* end = nullptr;
        const long long parsed = std::strtoll(raw, &end, 10);
        if (end == raw || parsed <= 0) {
            return static_cast<u64>(250000);
        }
        return static_cast<u64>(parsed);
    }();
    return period_us;
}

static u64 asyncWaitRescueMicros() {
    static const u64 rescue_us = []() {
        const char* raw = std::getenv("CALICO_ASYNC_WAIT_RESCUE_US");
        if (raw == nullptr || raw[0] == '\0') {
            return static_cast<u64>(500000);
        }
        char* end = nullptr;
        const long long parsed = std::strtoll(raw, &end, 10);
        if (end == raw || parsed <= 0) {
            return static_cast<u64>(500000);
        }
        return static_cast<u64>(parsed);
    }();
    return rescue_us;
}

static u64 asyncWaitPollMicros() {
    static const u64 poll_us = []() {
        const char* raw = std::getenv("CALICO_ASYNC_WAIT_POLL_US");
        if (raw == nullptr || raw[0] == '\0') {
            return static_cast<u64>(1000);
        }
        char* end = nullptr;
        const long long parsed = std::strtoll(raw, &end, 10);
        if (end == raw || parsed <= 0) {
            return static_cast<u64>(1000);
        }
        return static_cast<u64>(parsed);
    }();
    return poll_us;
}

static u64 packAsyncRequestUserData(u32 slot, u64 token) {
    return (static_cast<u64>(slot) << 32) | (token & 0xffffffffULL);
}

static u32 unpackAsyncRequestSlot(u64 user_data) {
    return static_cast<u32>(user_data >> 32);
}

static u32 unpackAsyncRequestTokenLow(u64 user_data) {
    return static_cast<u32>(user_data & 0xffffffffULL);
}

static u64 packSyncReadUserData(u32 batch_index) {
    return packAsyncRequestUserData(BufferManager::invalidAsyncRequestSlot, batch_index);
}

static void logAsyncRequestSnapshot(const char* event, const AsyncReadRequest& request,
                                    const IOUringInterface* io_interface = nullptr, int submit_ret = 0);

static bool processAsyncReadCompletion(IOUringInterface* io_interface, io_uring_cqe* cqe) {
    const u64 user_data = io_uring_cqe_get_data64(cqe);
    const u32 slot = unpackAsyncRequestSlot(user_data);
    if (slot == BufferManager::invalidAsyncRequestSlot) {
        return false;
    }

    BufferManager* bm_ptr = io_interface->bm_ptr;
    AsyncReadRequest* request = bm_ptr->asyncReadRequests.get() + slot;
    const u32 token_low = unpackAsyncRequestTokenLow(user_data);
    const u64 live_token = request->token.load(std::memory_order_acquire);
    const u32 live_status = request->status.load(std::memory_order_acquire);
    if ((live_token & 0xffffffffULL) != token_low || live_status != AsyncReadRequest::Submitted) {
        return true;
    }

    const int result = cqe->res;
    request->result.store(result, std::memory_order_release);
    request->completionMicros.store(monotonicMicros(), std::memory_order_release);
    request->completionWorker.store(workerThreadId, std::memory_order_release);
    bm_ptr->recordAsyncInFlightTransition(request->completionMicros.load(std::memory_order_acquire), -1);
    if (result == static_cast<int>(pageSize)) {
        const PID pid = request->pid.load(std::memory_order_acquire);
        const u32 frame = request->frame.load(std::memory_order_acquire);
        if (frame != std::numeric_limits<u32>::max() && frame < bm_ptr->physCount) {
            bm_ptr->frameMem[frame].dirty = false;
        }
        bm_ptr->readCount.fetch_add(1, std::memory_order_relaxed);
        if constexpr (kEnableAsyncPrefetchStats) {
            bm_ptr->asyncCompletionCount.fetch_add(1, std::memory_order_relaxed);
        }
        request->status.store(AsyncReadRequest::Completed, std::memory_order_release);
        logAsyncRequestSnapshot("completion_ok", *request, io_interface);
        bm_ptr->finalizeCompletedAsyncRead(pid, slot);
    } else {
        request->status.store(AsyncReadRequest::Failed, std::memory_order_release);
        logAsyncRequestSnapshot("completion_failed", *request, io_interface);
    }

    return true;
}

[[maybe_unused]] static const char* asyncReadStatusName(u32 status) {
    switch (status) {
        case AsyncReadRequest::Idle:
            return "Idle";
        case AsyncReadRequest::Submitted:
            return "Submitted";
        case AsyncReadRequest::Completed:
            return "Completed";
        case AsyncReadRequest::Failed:
            return "Failed";
        case AsyncReadRequest::Rescuing:
            return "Rescuing";
        default:
            return "Unknown";
    }
}

static void logAsyncRequestSnapshot(const char* event, const AsyncReadRequest& request,
                                    const IOUringInterface* io_interface, int submit_ret) {
    // if (!asyncDiagnosticsEnabled()) {
    //     return;
    // }

    // const PID pid = request.pid.load(std::memory_order_acquire);
    // const u32 frame = request.frame.load(std::memory_order_acquire);
    // const u64 token = request.token.load(std::memory_order_acquire);
    // const u64 submit_micros = request.submitMicros.load(std::memory_order_acquire);
    // const u64 completion_micros = request.completionMicros.load(std::memory_order_acquire);
    // const u32 submit_worker = request.submitWorker.load(std::memory_order_acquire);
    // const u32 completion_worker = request.completionWorker.load(std::memory_order_acquire);
    // const u32 waiter_count = request.waiterCount.load(std::memory_order_acquire);
    // const int result = request.result.load(std::memory_order_acquire);
    // const u32 status = request.status.load(std::memory_order_acquire);
    // const u64 now_micros = monotonicMicros();

    // std::cerr << "[AsyncRead] event=" << event << " token=" << token << " pid=" << pid << " frame=" << frame
    //           << " status=" << asyncReadStatusName(status) << " result=" << result << " submit_worker="
    //           << submit_worker << " completion_worker=" << completion_worker << " waiter_count=" << waiter_count
    //           << " age_us=" << (submit_micros == 0 ? 0 : (now_micros - submit_micros)) << " completion_age_us="
    //           << (completion_micros == 0 ? 0 : (now_micros - completion_micros));

    // if (bm_ptr != nullptr && frame != std::numeric_limits<u32>::max() && frame < bm_ptr->physCount) {
    //     const PID mapped_pid = bm_ptr->frameToPidArray ? bm_ptr->frameToPidArray[frame].load(std::memory_order_acquire)
    //                                                    : BufferManager::invalidPID;
    //     std::cerr << " mapped_pid=" << mapped_pid;
    // }
    // if (io_interface != nullptr) {
    //     std::cerr << " sq_ready=" << io_uring_sq_ready(const_cast<io_uring*>(&io_interface->ring))
    //               << " cq_ready=" << io_uring_cq_ready(const_cast<io_uring*>(&io_interface->ring));
    // }
    // if (submit_ret != 0) {
    //     std::cerr << " submit_ret=" << submit_ret;
    // }
    // std::cerr << std::endl;
}

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

// HashThenArrayPageStateArray implementation
HashThenArrayPageStateArray::HashThenArrayPageStateArray(u64 virtCount)
    : second_level_arrays(nullptr), total_entries(virtCount) {
    u32 totalBits = bitsNeeded(virtCount);
    topBits = (totalBits > bottomBits) ? (totalBits - bottomBits) : 0;
    num_top_slots = 1U << topBits;
    entries_per_slot = 1U << bottomBits;

    std::cerr << "[HashThenArrayPageStateArray] virtCount=" << virtCount
              << " totalBits=" << totalBits
              << " topBits=" << topBits << " (" << num_top_slots << " slots)"
              << " bottomBits=" << bottomBits << " (" << entries_per_slot << " entries/slot)"
              << std::endl;

    const size_t top_size = num_top_slots * sizeof(PageState*);
    second_level_arrays = static_cast<PageState**>(allocHuge(top_size));
    if (second_level_arrays == MAP_FAILED) {
        die("allocHuge HashThenArray top level");
    }

    const size_t second_level_size = entries_per_slot * sizeof(PageState);
    top_level.reserve(num_top_slots);
    for (u32 i = 0; i < num_top_slots; ++i) {
        second_level_arrays[i] = static_cast<PageState*>(allocHuge(second_level_size));
        if (second_level_arrays[i] == MAP_FAILED) {
            die("allocHuge HashThenArray second level");
        }
        for (u32 j = 0; j < entries_per_slot; ++j) {
            second_level_arrays[i][j].init();
        }
        top_level.insertOrAssign(i, second_level_arrays[i]);
    }
}

HashThenArrayPageStateArray::~HashThenArrayPageStateArray() {
    if (second_level_arrays) {
        const size_t second_level_size = entries_per_slot * sizeof(PageState);
        for (u32 i = 0; i < num_top_slots; ++i) {
            if (second_level_arrays[i]) {
                munmap(second_level_arrays[i], second_level_size);
            }
        }
        const size_t top_size = num_top_slots * sizeof(PageState*);
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
    _mm_pause();
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
    if (cnt != (int)pages.size()) {
        std::cerr << "io_submit failed: " << cnt << " expected: " << pages.size() << " errno: " << -cnt << std::endl;
        exit(EXIT_FAILURE);
    }
    cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
    if (cnt != (int)pages.size()) {
        std::cerr << "io_getevents failed: " << cnt << " expected: " << pages.size() << " errno: " << -cnt << std::endl;
        exit(EXIT_FAILURE);
    }
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
    if (cnt != (int)pages.size()) {
        std::cerr << "io_submit read failed: " << cnt << " expected: " << pages.size() << " errno: " << -cnt << std::endl;
        exit(EXIT_FAILURE);
    }
    cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
    if (cnt != (int)pages.size()) {
        std::cerr << "io_getevents read failed: " << cnt << " expected: " << pages.size() << " errno: " << -cnt << std::endl;
        exit(EXIT_FAILURE);
    }

    // Mark pages as clean after successful read
    for (u64 i = 0; i < pages.size(); i++) {
        destinations[i]->dirty = false;
        // bm_ptr->readCount++;
    }
}

IOUringInterface::IOUringInterface(int blockfd, BufferManager* bm_ptr) : blockfd(blockfd), bm_ptr(bm_ptr) {
    const int ret = io_uring_queue_init(queueDepth, &ring, 0);
    if (ret < 0) {
        std::cerr << "io_uring_queue_init failed: " << strerror(-ret) << std::endl;
        exit(EXIT_FAILURE);
    }
}

IOUringInterface::~IOUringInterface() { io_uring_queue_exit(&ring); }

namespace {
inline void forceAsyncReadSQE(io_uring_sqe* sqe) {
    sqe->flags |= IOSQE_ASYNC;
}

struct AtomicFlagGuard {
    std::atomic_flag& lock;

    explicit AtomicFlagGuard(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            _mm_pause();
        }
    }

    ~AtomicFlagGuard() { lock.clear(std::memory_order_release); }
};

struct AtomicFlagTryGuard {
    std::atomic_flag* lock;
    bool owns;

    explicit AtomicFlagTryGuard(std::atomic_flag& lock) : lock(&lock), owns(!lock.test_and_set(std::memory_order_acquire)) {}

    ~AtomicFlagTryGuard() {
        if (owns) {
            lock->clear(std::memory_order_release);
        }
    }

    bool owns_lock() const { return owns; }
};
}  // namespace

void IOUringInterface::writePages(const vector<PID>& pages) {
    AtomicFlagGuard guard(submitLock);
    for (PID pid : pages) {
        Page* page = bm_ptr->preparePageForWrite(pid);
        const ssize_t ret = pwrite(blockfd, page, pageSize, pageSize * pid);
        if (ret != static_cast<ssize_t>(pageSize)) {
            std::cerr << "pwrite failed in IOUringInterface::writePages for pid " << pid << " errno=" << errno
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void IOUringInterface::readPages(const vector<PID>& pages, const vector<Page*>& destinations) {
    assert(pages.size() == destinations.size());
    if (pages.empty()) {
        return;
    }

    AtomicFlagGuard submit_guard(submitLock);
    AtomicFlagGuard completion_guard(completionLock);
    reapCompletions(false);

    for (size_t batch_begin = 0; batch_begin < pages.size(); batch_begin += maxIOs) {
        const size_t batch_count = std::min<size_t>(maxIOs, pages.size() - batch_begin);
        for (size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (sqe == nullptr) {
                std::cerr << "io_uring_get_sqe failed in IOUringInterface::readPages" << std::endl;
                exit(EXIT_FAILURE);
            }

            const size_t request_idx = batch_begin + batch_idx;
            io_uring_prep_read(sqe, blockfd, destinations[request_idx], pageSize, pageSize * pages[request_idx]);
            io_uring_sqe_set_data64(sqe, packSyncReadUserData(static_cast<u32>(batch_idx)));
        }

        size_t submitted = 0;
        while (submitted < batch_count) {
            const int ret = io_uring_submit(&ring);
            if (ret < 0) {
                std::cerr << "io_uring_submit failed in IOUringInterface::readPages: " << strerror(-ret)
                          << std::endl;
                exit(EXIT_FAILURE);
            }
            if (ret == 0) {
                std::cerr << "io_uring_submit made no progress in IOUringInterface::readPages" << std::endl;
                exit(EXIT_FAILURE);
            }
            submitted += static_cast<size_t>(ret);
        }

        std::vector<bool> completed(batch_count, false);
        size_t remaining = batch_count;
        while (remaining > 0) {
            io_uring_cqe* cqe = nullptr;
            const int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                std::cerr << "io_uring_wait_cqe failed in IOUringInterface::readPages: " << strerror(-ret)
                          << std::endl;
                exit(EXIT_FAILURE);
            }

            const u64 user_data = io_uring_cqe_get_data64(cqe);
            const u32 slot = unpackAsyncRequestSlot(user_data);
            if (slot == BufferManager::invalidAsyncRequestSlot) {
                const u32 batch_idx = unpackAsyncRequestTokenLow(user_data);
                if (batch_idx >= batch_count || completed[batch_idx]) {
                    std::cerr << "invalid sync read completion in IOUringInterface::readPages" << std::endl;
                    exit(EXIT_FAILURE);
                }

                const size_t request_idx = batch_begin + batch_idx;
                if (cqe->res != static_cast<int>(pageSize)) {
                    std::cerr << "io_uring read failed in IOUringInterface::readPages for pid " << pages[request_idx]
                              << " res=" << cqe->res << std::endl;
                    exit(EXIT_FAILURE);
                }

                completed[batch_idx] = true;
                destinations[request_idx]->dirty = false;
                bm_ptr->readCount.fetch_add(1, std::memory_order_relaxed);
                --remaining;
            } else {
                processAsyncReadCompletion(this, cqe);
            }

            io_uring_cqe_seen(&ring, cqe);
        }
    }
}

void BufferManager::recordAsyncInFlightTransition(u64 now_micros, int delta) {
    if constexpr (!kEnableAsyncPrefetchStats) {
        return;
    }

    AtomicFlagGuard guard(asyncInFlightStatsLock);

    const u64 current = asyncInFlightCurrent.load(std::memory_order_relaxed);
    const u64 last_transition = asyncInFlightLastTransitionMicros.load(std::memory_order_relaxed);
    if (last_transition != 0 && now_micros > last_transition && current > 0) {
        asyncInFlightWeightedMicros.fetch_add(current * (now_micros - last_transition), std::memory_order_relaxed);
    }

    if (current == 0 && delta > 0 && asyncInFlightStartMicros.load(std::memory_order_relaxed) == 0) {
        asyncInFlightStartMicros.store(now_micros, std::memory_order_relaxed);
    }

    const long long next_signed = static_cast<long long>(current) + static_cast<long long>(delta);
    const u64 next = next_signed > 0 ? static_cast<u64>(next_signed) : 0;
    asyncInFlightCurrent.store(next, std::memory_order_relaxed);
    asyncInFlightLastTransitionMicros.store(now_micros, std::memory_order_relaxed);

    u64 peak = asyncInFlightPeak.load(std::memory_order_relaxed);
    while (next > peak && !asyncInFlightPeak.compare_exchange_weak(peak, next, std::memory_order_relaxed,
                                                                   std::memory_order_relaxed)) {
    }
}

double BufferManager::averageAsyncInFlightIO(u64 now_micros) const {
    if constexpr (!kEnableAsyncPrefetchStats) {
        return 0.0;
    }

    AtomicFlagGuard guard(asyncInFlightStatsLock);

    const u64 start = asyncInFlightStartMicros.load(std::memory_order_relaxed);
    if (start == 0 || now_micros <= start) {
        return 0.0;
    }

    u64 weighted = asyncInFlightWeightedMicros.load(std::memory_order_relaxed);
    const u64 current = asyncInFlightCurrent.load(std::memory_order_relaxed);
    const u64 last_transition = asyncInFlightLastTransitionMicros.load(std::memory_order_relaxed);
    if (current > 0 && last_transition != 0 && now_micros > last_transition) {
        weighted += current * (now_micros - last_transition);
    }

    const u64 active_window = now_micros - start;
    if (active_window == 0) {
        return 0.0;
    }
    return static_cast<double>(weighted) / static_cast<double>(active_window);
}

bool IOUringInterface::submitAsyncRead(PID pid, Page* destination, AsyncReadRequest* request) {
    const PID pids[] = {pid};
    Page* const destinations[] = {destination};
    AsyncReadRequest* const requests[] = {request};
    return submitAsyncReads(pids, destinations, requests, 1) == 1;
}

size_t IOUringInterface::submitAsyncReads(const PID* pids, Page* const* destinations,
                                          AsyncReadRequest* const* requests, size_t count) {
    auto try_reap_completions = [&]() {
        AtomicFlagTryGuard completion_guard(completionLock);
        if (completion_guard.owns_lock()) {
            reapCompletions(false);
        }
    };

    auto try_submit = [&]() -> size_t {
        AtomicFlagGuard submit_guard(submitLock);
        size_t prepared = 0;
        for (; prepared < count; ++prepared) {
            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (sqe == nullptr) {
                break;
            }

            AsyncReadRequest* request = requests[prepared];
            const u32 slot = static_cast<u32>(request - bm_ptr->asyncReadRequests.get());
            const u64 token = request->token.load(std::memory_order_acquire);
            io_uring_prep_read(sqe, blockfd, destinations[prepared], pageSize, pageSize * pids[prepared]);
            //forceAsyncReadSQE(sqe);
            io_uring_sqe_set_data64(sqe, packAsyncRequestUserData(slot, token));
        }

        if (prepared == 0) {
            return 0;
        }

        const int ret = io_uring_submit(&ring);
        if (ret < 0) {
            if constexpr (kEnableAsyncPrefetchStats) {
                bm_ptr->asyncSubmitBackpressureCount.fetch_add(1, std::memory_order_relaxed);
            }
            for (size_t i = 0; i < prepared; ++i) {
                logAsyncRequestSnapshot("submit_error", *requests[i], this, ret);
            }
            std::cerr << "io_uring_submit failed: " << strerror(-ret) << std::endl;
            return 0;
        }
        if (ret == 0) {
            if constexpr (kEnableAsyncPrefetchStats) {
                bm_ptr->asyncSubmitBackpressureCount.fetch_add(1, std::memory_order_relaxed);
            }
            for (size_t i = 0; i < prepared; ++i) {
                logAsyncRequestSnapshot("submit_zero", *requests[i], this, ret);
            }
            return 0;
        }

        const size_t submitted = std::min<size_t>(prepared, static_cast<size_t>(ret));
        const u64 now_micros = monotonicMicros();
        for (size_t i = 0; i < submitted; ++i) {
            bm_ptr->recordAsyncInFlightTransition(now_micros, 1);
            logAsyncRequestSnapshot("submit_ok", *requests[i], this, ret);
        }
        if (submitted < prepared) {
            if constexpr (kEnableAsyncPrefetchStats) {
                bm_ptr->asyncSubmitBackpressureCount.fetch_add(prepared - submitted, std::memory_order_relaxed);
            }
            for (size_t i = submitted; i < prepared; ++i) {
                logAsyncRequestSnapshot("submit_partial", *requests[i], this, ret);
            }
        }
        return submitted;
    };

    try_reap_completions();
    const size_t first_attempt = try_submit();
    if (first_attempt > 0 || count == 0) {
        return first_attempt;
    }

    try_reap_completions();
    const size_t second_attempt = try_submit();
    if (second_attempt > 0 || count == 0) {
        return second_attempt;
    }

    if constexpr (kEnableAsyncPrefetchStats) {
        bm_ptr->asyncSubmitBackpressureCount.fetch_add(count, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < count; ++i) {
        logAsyncRequestSnapshot("submit_no_sqe", *requests[i], this);
    }
    return 0;
}

void IOUringInterface::pollAsyncCompletions() {
    AtomicFlagTryGuard completion_guard(completionLock);
    if (completion_guard.owns_lock()) {
        reapCompletions(false);
    }
}

void IOUringInterface::waitForAsyncRead(AsyncReadRequest& request) {
    struct WaiterScope {
        AsyncReadRequest& request;
        explicit WaiterScope(AsyncReadRequest& req) : request(req) {
            request.waiterCount.fetch_add(1, std::memory_order_acq_rel);
        }
        ~WaiterScope() { request.waiterCount.fetch_sub(1, std::memory_order_acq_rel); }
    } waiter_scope(request);

    const u64 wait_start_micros = monotonicMicros();
    bool slow_wait_counted = false;
    for (;;) {
        const u32 status = request.status.load(std::memory_order_acquire);
        if (status != AsyncReadRequest::Submitted) {
            if (status == AsyncReadRequest::Rescuing) {
                std::this_thread::yield();
                continue;
            }
            return;
        }

        const u64 now_micros = monotonicMicros();
        const u64 elapsed_micros = now_micros - wait_start_micros;
        if (elapsed_micros >= asyncWaitRescueMicros()) {
            u32 expected = AsyncReadRequest::Submitted;
            if (request.status.compare_exchange_strong(expected, AsyncReadRequest::Rescuing,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
                const PID pid = request.pid.load(std::memory_order_acquire);
                Page* destination = nullptr;
                if (bm_ptr->useTraditional) {
                    const u32 frame = request.frame.load(std::memory_order_acquire);
                    if (frame != std::numeric_limits<u32>::max() && frame < bm_ptr->physCount) {
                        destination = bm_ptr->frameMem + frame;
                    }
                } else if (pid != std::numeric_limits<PID>::max() && pid < bm_ptr->virtCount) {
                    destination = bm_ptr->virtMem + pid;
                }

                int rescue_result = -EIO;
                if (destination != nullptr && pid != std::numeric_limits<PID>::max()) {
                    rescue_result = pread(blockfd, destination, pageSize, pageSize * pid);
                }

                request.result.store(rescue_result, std::memory_order_release);
                request.completionMicros.store(monotonicMicros(), std::memory_order_release);
                request.completionWorker.store(workerThreadId, std::memory_order_release);
                bm_ptr->recordAsyncInFlightTransition(request.completionMicros.load(std::memory_order_acquire), -1);

                if (rescue_result == static_cast<int>(pageSize)) {
                    if (destination != nullptr) {
                        destination->dirty = false;
                    }
                    bm_ptr->readCount.fetch_add(1, std::memory_order_relaxed);
                    request.status.store(AsyncReadRequest::Completed, std::memory_order_release);
                    logAsyncRequestSnapshot("wait_rescue_ok", request, this);
                    bm_ptr->finalizeCompletedAsyncRead(pid, static_cast<u32>(&request - bm_ptr->asyncReadRequests.get()));
                } else {
                    request.status.store(AsyncReadRequest::Failed, std::memory_order_release);
                    logAsyncRequestSnapshot("wait_rescue_failed", request, this, rescue_result);
                }
                return;
            }
            continue;
        }

        {
            AtomicFlagTryGuard guard(completionLock);
            if (!guard.owns_lock()) {
                std::this_thread::yield();
                continue;
            }
            reapCompletions(false);
            if (request.status.load(std::memory_order_acquire) != AsyncReadRequest::Submitted) {
                return;
            }

            if (asyncDiagnosticsEnabled()) {
                if (elapsed_micros >= asyncWaitWarnMicros()) {
                    if constexpr (kEnableAsyncPrefetchStats) {
                        if (!slow_wait_counted) {
                            bm_ptr->asyncSlowWaitCount.fetch_add(1, std::memory_order_relaxed);
                            slow_wait_counted = true;
                        }
                    }
                    u64 previous_log = request.lastWaitLogMicros.load(std::memory_order_acquire);
                    if ((previous_log == 0 || (now_micros - previous_log) >= asyncWaitLogEveryMicros()) &&
                        request.lastWaitLogMicros.compare_exchange_strong(previous_log, now_micros,
                                                                          std::memory_order_acq_rel,
                                                                          std::memory_order_acquire)) {
                        logAsyncRequestSnapshot("wait_slow", request, this);
                    }
                }
            }

            reapCompletions(true);
        }
    }
}

void IOUringInterface::reapCompletions(bool wait_for_one) {
    io_uring_cqe* cqe = nullptr;
    int ret = 0;
    if (wait_for_one) {
        __kernel_timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = static_cast<long>(asyncWaitPollMicros() * 1000);
        ret = io_uring_wait_cqe_timeout(&ring, &cqe, &timeout);
        if (ret == -ETIME) {
            return;
        }
    } else {
        ret = io_uring_peek_cqe(&ring, &cqe);
    }
    while (ret == 0) {
        processAsyncReadCompletion(this, cqe);

        io_uring_cqe_seen(&ring, cqe);
        ret = io_uring_peek_cqe(&ring, &cqe);
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

static bool ignorePidAllocator() {
    static const bool enabled = envOr("IGNORE_PID_ALLOCATOR", 0) != 0;
    return enabled;
}

BufferManager::BufferManager(unsigned nthreads)
    : virtSize(envOr("VIRTGB", 16) * gb),
      physSize(envOrF("PHYSGB", 4) * gb),
      virtCount(virtSize / pageSize),
      physCount(physSize / pageSize),
      residentSet(physCount * 2) {
    // `calico_oltp_query` constructs BufferManager with the default constructor.
    // Honor THREADS from env here so hash table specialization matches runtime worker count.
    numThreads = std::max(1u, static_cast<unsigned>(envOr("THREADS", nthreads)));
    assert(virtSize >= physSize);
    const char* path = getenv("BLOCK") ? getenv("BLOCK") : "./heapfile";
    blockfd = open(path, O_RDWR | O_DIRECT, S_IRWXU);
    if (blockfd == -1) {
        cerr << "cannot open BLOCK device '" << path << "'" << endl;
        exit(EXIT_FAILURE);
    }

    useTraditional = envOr("TRADITIONAL", 0);
    useExmap = (!useTraditional) && envOr("EXMAP", 0);
    useIOUring = !useExmap && envOr("USE_IOURING", 1);
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
    else if (tradHashSetting == 7)
        hashMode = HashMode::RobinHood;
    else if (tradHashSetting == 8)
        hashMode = HashMode::HashThenArray;
    else
        hashMode = HashMode::Array1Access;
    enablePathCache = envOr("PATH_CACHE", 0) != 0;
    PageState::setPackedMode(hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level ||
                             hashMode == HashMode::Array3Level || hashMode == HashMode::HashThenArray);
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
        std::vector<u32> initial_free_frames;
        initial_free_frames.reserve(physCount > 0 ? physCount - 1 : 0);
        for (u32 frame = 1; frame < physCount; ++frame) {
            initial_free_frames.push_back(frame);
        }
        if (!initial_free_frames.empty()) {
            freeFrameQueue.enqueue_bulk(initial_free_frames.data(), initial_free_frames.size());
            memset(frameMem, 0, physCount * sizeof(Page));
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
        } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level ||
                   hashMode == HashMode::Array3Level || hashMode == HashMode::HashThenArray) {
            pidToFrameArray.reset();
            pidToFrameHash.reset();
            frameToPidHash.reset();
        } else {
            if (hashMode == HashMode::Unordered) {
                if (numThreads > 1) {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::OpenAddress) {
                if (numThreads > 1) {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::RobinHood) {
                if (numThreads > 1) {
                    pidToFrameHash = std::make_unique<ShardedRobinHoodMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedRobinHoodMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedRobinHoodMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedRobinHoodMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::Lockfree) {
                pidToFrameHash = std::make_unique<LockFreeHashMap<PID, u64>>();
                frameToPidHash = std::make_unique<LockFreeHashMap<u64, PID>>();
            } else {
                die("invalid config for hashmap");
            }
            // 
            pidToFrameHash->reserve(physCount * 1.5);
            frameToPidHash->reserve(physCount);
        }
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

    // Allocate the PageState translation structure.
    if (hashMode == HashMode::Array2Level) {
        pageState = nullptr;
        pageState2Level = std::make_unique<TwoLevelPageStateArray>(virtCount);
        pageState3Level.reset();
        pageStateHashThenArray.reset();
    } else if (hashMode == HashMode::Array3Level) {
        pageState = nullptr;
        pageState2Level.reset();
        pageState3Level = std::make_unique<ThreeLevelPageStateArray>(virtCount);
        pageStateHashThenArray.reset();
    } else if (hashMode == HashMode::HashThenArray) {
        pageState = nullptr;
        pageState2Level.reset();
        pageState3Level.reset();
        pageStateHashThenArray = std::make_unique<HashThenArrayPageStateArray>(virtCount);
    } else {
        if (disableHugePageForTranslationArray) {
            pageState = (PageState*)mmap(NULL, virtCount * sizeof(PageState), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (pageState == MAP_FAILED) die("mmap pageState");
            madvise(pageState, virtCount * sizeof(PageState), MADV_NOHUGEPAGE);
        } else {
            pageState = (PageState*)allocHuge(virtCount * sizeof(PageState));
        }
        //for (u64 i = 0; i < virtCount; i++) pageState[i].init();
        pageState2Level.reset();
        pageState3Level.reset();
        // print pageState address
        std::cerr << "pageState: " << (void*)pageState  << " length " << virtCount * sizeof(PageState) << std::endl;
        pageStateHashThenArray.reset();
    }
    // Determine OS page size for translation array based on huge page usage
    // Must align with the actual allocation: 4KB if huge pages disabled, 2MB otherwise
    // Set NOHUGEPAGE_TRANSLATION_ARRAY=1 to disable huge pages and enable fine-grained hole-punching
    translationOSPageSize = (disableHugePageForTranslationArray && useTraditional) ? 4096 : (2 * 1024 * 1024);
    
    // Translation refcount/hole-punching can add noticeable overhead on OOM miss paths.
    // Keep it opt-in for throughput experiments.
    enableTranslationRefCount = envOr("TRANSLATION_REFCOUNT", 0);
    numRefCountGroups = 0;
    // Initialize reference counting for hole-punching (Array1Access mode only)
    translationRefCounts = nullptr;
    if (enableTranslationRefCount && hashMode == HashMode::Array1Access && useTraditional) {
        // Each OS page holds translationOSPageSize/sizeof(PageState) translation entries
        u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
        numRefCountGroups = (virtCount + entriesPerOSPage - 1) / entriesPerOSPage;
        
        // Lazily allocate reference counts using mmap with MAP_PRIVATE | MAP_ANONYMOUS
        // This ensures physical memory is only allocated on first write (zero-page COW)
        size_t refCountSize = numRefCountGroups * sizeof(std::atomic<u32>);
        translationRefCounts = (std::atomic<u32>*)mmap(NULL, refCountSize, PROT_READ | PROT_WRITE,
                                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (translationRefCounts == MAP_FAILED) {
            die("mmap translationRefCounts failed");
        }
        // No need to initialize - mmap with MAP_ANONYMOUS gives zero-filled pages
    }

    physUsedCount = 0;
    allocCount = 1;  // pid 0 reserved for meta data
    readCount = 0;
    writeCount = 0;
    holePunchCount = 0;
    asyncSubmitCount = 0;
    asyncCoalescedCount = 0;
    asyncGuardWaitCount = 0;
    asyncSyncFaultCount = 0;
    asyncFailedCompletionCount = 0;
    asyncCompletionCount = 0;
    asyncSubmitBackpressureCount = 0;
    asyncSlowWaitCount = 0;
    asyncInFlightCurrent = 0;
    asyncInFlightPeak = 0;
    asyncInFlightWeightedMicros = 0;
    asyncInFlightStartMicros = 0;
    asyncInFlightLastTransitionMicros = 0;
    nextAsyncReadToken = 1;
    batch = envOr("BATCH", 64);

    //if (useIOUring) {
        const u64 async_request_capacity = useTraditional ? physCount : IOUringInterface::queueDepth;
        asyncReadRequests = std::make_unique<AsyncReadRequest[]>(async_request_capacity);
        for (u64 i = 0; i < async_request_capacity; ++i) {
            asyncReadRequests[i].reset();
        }
        if (!useTraditional) {
            pidToAsyncRequestSlot = std::make_unique<std::atomic<u32>[]>(virtCount);
            std::vector<u32> initial_async_slots;
            initial_async_slots.reserve(async_request_capacity);
            for (u64 i = 0; i < virtCount; ++i) {
                pidToAsyncRequestSlot[i].store(invalidAsyncRequestSlot, std::memory_order_relaxed);
            }
            for (u32 slot = 0; slot < async_request_capacity; ++slot) {
                initial_async_slots.push_back(slot);
            }
            if (!initial_async_slots.empty()) {
                freeAsyncRequestSlotQueue.enqueue_bulk(initial_async_slots.data(), initial_async_slots.size());
            }
        }

        const u64 configured_interfaces = envOr("IOURING_INTERFACES", numThreads);
        const u64 interface_count = std::max<u64>(1, configured_interfaces);
        ioUringInterfaces.reserve(interface_count);
        for (u64 i = 0; i < interface_count; ++i) {
            ioUringInterfaces.push_back(std::make_unique<IOUringInterface>(blockfd, this));
        }
    // }

    std::string translation_specialization = "general";
    #ifdef CALICO_SPECIALIZATION_CALICO
    translation_specialization = "calico";
    #elif defined(CALICO_SPECIALIZATION_HASH)
    translation_specialization = "hash";
    #else
    #endif
    cerr << "calico BufferManager initialized: "
         << "blk:" << path << " virtgb:" << virtSize / gb << " physgb:" << (float)physSize / gb
         << " traditional:" << useTraditional << " mmap_os_pagecache:" << useMmapOSPageacche
         << " trad_hash:" << static_cast<int>(hashMode) << " exmap:" << useExmap << " hugepage: " << (disableHugePageForFrameMem == 0)
         << " hugepage for translation array: " << (disableHugePageForTranslationArray == 0)
         << " translation_refcount:" << enableTranslationRefCount
         << " path_cache:" << enablePathCache
            << " use_io_uring:" << useIOUring
         << " num_threads:" << numThreads 
            << " ignore_pid_allocator:" << ignorePidAllocator()
            << " page_state_size:" << sizeof(PageState)
         << " translation_specialization:" << translation_specialization
         << endl;
}

BufferManager::~BufferManager() {
    try {
        //flushAll();
    } catch (const std::exception& ex) {
        std::cerr << "BufferManager::~BufferManager flush error: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "BufferManager::~BufferManager encountered an unknown error during flush." << std::endl;
    }
    
    // Cleanup lazily allocated reference counts
    if (translationRefCounts != nullptr) {
        // print ref count stats using a histogram style
        // dont use numRefCountGroups as there could be unallocated
        // use allocCount as the maximum page id allocated to estimated the number of groups
        u64 estimatedNumRefCountGroups = (allocCount + (translationOSPageSize / sizeof(PageState)) - 1) / (translationOSPageSize / sizeof(PageState));
        std::unordered_map<u32, u64> histogram;
        for (u64 i = 0; i < estimatedNumRefCountGroups; i++)
        {
            u32 count = translationRefCounts[i].load(std::memory_order_relaxed);
            histogram[count]++;
        }
        std::cerr << "[BufferManager] Translation Reference Count Histogram:" << std::endl;
        // sort the histogram by <ref count, num groups> and print them
        std::vector<std::pair<u32, u64>> sorted_histogram(histogram.begin(), histogram.end());
        std::sort(sorted_histogram.begin(), sorted_histogram.end());
        for (const auto& entry : sorted_histogram)
        {
            std::cerr << "  Ref Count " << entry.first << ": " << entry.second << " groups" << std::endl;
        }

        size_t refCountSize = numRefCountGroups * sizeof(std::atomic<u32>);
        munmap(translationRefCounts, refCountSize);
        translationRefCounts = nullptr;
    }

    // print hole punch count
    std::cerr << "[BufferManager] Total hole punches (madvise count): " << holePunchCount << std::endl;
    if constexpr (kEnableAsyncPrefetchStats) {
        const u64 async_summary_micros = monotonicMicros();
        std::cerr << "[BufferManager] Async reads submitted=" << asyncSubmitCount.load(std::memory_order_relaxed)
                  << " coalesced=" << asyncCoalescedCount.load(std::memory_order_relaxed)
                  << " guard_waits=" << asyncGuardWaitCount.load(std::memory_order_relaxed)
                  << " completions=" << asyncCompletionCount.load(std::memory_order_relaxed)
                  << " submit_backpressure=" << asyncSubmitBackpressureCount.load(std::memory_order_relaxed)
                  << " slow_waits=" << asyncSlowWaitCount.load(std::memory_order_relaxed)
                  << " avg_inflight=" << averageAsyncInFlightIO(async_summary_micros)
                  << " peak_inflight=" << asyncInFlightPeak.load(std::memory_order_relaxed)
                  << " sync_fallback_faults=" << asyncSyncFaultCount.load(std::memory_order_relaxed)
                  << " failed_completions=" << asyncFailedCompletionCount.load(std::memory_order_relaxed)
                  << std::endl;
    }
    // 
}

bool BufferManager::asyncReadEnabled() const { return useIOUring; }

IOInterface& BufferManager::getIOInterface() {
        // const size_t interface_count = ioUringInterfaces.size();
        // const size_t interface_idx = interface_count == 1
        //                                  ? 0
        //                                  : (std::hash<std::thread::id>{}(std::this_thread::get_id()) % interface_count);
        
        // return *ioUringInterfaces[interface_idx];
    if (asyncReadEnabled()) {
        const size_t interface_count = ioUringInterfaces.size();
        const size_t interface_idx = interface_count == 1
                                         ? 0
                                         : (std::hash<std::thread::id>{}(std::this_thread::get_id()) % interface_count);
        return *ioUringInterfaces[interface_idx];
    }
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
        if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level ||
            hashMode == HashMode::Array3Level || hashMode == HashMode::HashThenArray) {
            PID pid = frameToPidArray[static_cast<size_t>(idx)].load(std::memory_order_acquire);
            assert(pid != invalidPID);
            return pid;
        } else {
            // Keep retrying until hash translation entry is visible.
            PID pid;
            for (u64 repeatCounter = 0;; ++repeatCounter) {
                if (frameToPidHash->tryGet(static_cast<u64>(idx), pid)) return pid;
                yield(repeatCounter);
            }
        }
    }
    return ptr - virtMem;
}

inline Page* BufferManager::residentPtr(PID pid) {
    assert(pid < virtCount);
    if (!useTraditional) return virtMem + pid;
    PageState& ps = getPageState(pid);
    u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
#if defined(CALICO_SPECIALIZATION_CALICO)
    return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
    return residentPtrHash(pid, stateAndVersion);
#else
    return residentPtr(pid, stateAndVersion);
#endif
}

Page* BufferManager::residentPtrCalico(PID pid, u64 stateAndVersion) {
    // Specialized for Calico array indexing - extracts frame from stateAndVersion
    u64 frame = PageState::extractFrame(stateAndVersion);
    assert(frame != PageState::packedInvalidFrame);
    assert(frame < physCount);
    return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::residentPtrHash(PID pid, u64 stateAndVersion) {
    // Specialized for hash table lookup.
    // IMPORTANT: in release builds assert() is compiled out, so never rely on
    // uninitialized `frame` after a failed lookup.
    u64 frame;
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        if (pidToFrameHash->tryGet(pid, frame)) {
            if (frame < physCount) return frameMem + static_cast<size_t>(frame);
        }
        yield(repeatCounter);
    }
}

Page* BufferManager::residentPtr(PID pid, u64 stateAndVersion) {
    assert(pid < virtCount);
    if (!useTraditional) return virtMem + pid;

    if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level ||
        hashMode == HashMode::HashThenArray) {
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

    return residentPtrHash(pid, stateAndVersion);
}

Page* BufferManager::toPtr(PID pid) { return residentPtr(pid); }

Page* BufferManager::preparePageForWrite(PID pid) {
    Page* page = residentPtr(pid);
    page->dirty = false;
    return page;
}

u64 BufferManager::resolveMappedFrame(PID pid, u64 stateAndVersion) {
    if (!useTraditional) {
        return invalidFrame;
    }

    if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level ||
        hashMode == HashMode::HashThenArray) {
        return PageState::extractFrame(stateAndVersion);
    }
    if (hashMode == HashMode::Array) {
        return pidToFrameArray[pid].load(std::memory_order_acquire);
    }

    u64 frame = invalidFrame;
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        if (pidToFrameHash->tryGet(pid, frame)) {
            return frame;
        }
        yield(repeatCounter);
    }
}

u32 BufferManager::resolveAsyncRequestSlot(PID pid, u64 stateAndVersion) {
    if (useTraditional) {
        const u64 frame = resolveMappedFrame(pid, stateAndVersion);
        return frame == invalidFrame ? invalidAsyncRequestSlot : static_cast<u32>(frame);
    }
    return pidToAsyncRequestSlot[pid].load(std::memory_order_acquire);
}

void BufferManager::releaseAsyncRequestSlot(PID pid, u32 slot) {
    if (slot == invalidAsyncRequestSlot) {
        return;
    }

    if (!useTraditional) {
        const u32 previous = pidToAsyncRequestSlot[pid].exchange(invalidAsyncRequestSlot, std::memory_order_acq_rel);
        if (previous != slot) {
            throw std::runtime_error("async request slot mismatch");
        }
        asyncReadRequests[slot].reset();
        const bool enqueued = freeAsyncRequestSlotQueue.enqueue(slot);
        if (!enqueued) {
            throw std::runtime_error("failed to enqueue async request slot");
        }
        return;
    }

    asyncReadRequests[slot].reset();
}

bool BufferManager::finalizeCompletedAsyncRead(PID pid, u32 slot) {
    if (slot == invalidAsyncRequestSlot) {
        return false;
    }

    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        u64 current = ps.stateAndVersion.load(std::memory_order_acquire);
        const u64 state = PageState::getState(current);
        if (state == PageState::ReadInFlight) {
            const u64 unlocked = PageState::sameVersion(current, PageState::Unlocked);
            if (ps.stateAndVersion.compare_exchange_weak(current, unlocked, std::memory_order_acq_rel,
                                                         std::memory_order_acquire)) {
                residentSet.insert(pid);
                releaseAsyncRequestSlot(pid, slot);
                return true;
            }
            yield(repeatCounter);
            continue;
        }

        return PageState::isResidentState(current);
    }
}

void BufferManager::rollbackFailedAsyncRead(PID pid, u64 stateAndVersion) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        u64 current = ps.stateAndVersion.load(std::memory_order_acquire);
        const u64 state = PageState::getState(current);
        if (state == PageState::Evicted) {
            return;
        }
        if (state != PageState::ReadInFlight) {
            return;
        }

        const u64 locked = PageState::sameVersion(current, PageState::Locked);
        if (ps.stateAndVersion.compare_exchange_weak(current, locked, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
            const u32 request_slot = resolveAsyncRequestSlot(pid, locked);
            if (request_slot != invalidAsyncRequestSlot) {
                auto* owner_interface = reinterpret_cast<IOUringInterface*>(
                    asyncReadRequests[request_slot].ownerInterface.load(std::memory_order_acquire));
                logAsyncRequestSnapshot("rollback_failed", asyncReadRequests[request_slot], owner_interface);
            }
            releaseFrame(pid);
            releaseAsyncRequestSlot(pid, request_slot);
            if (!(useTraditional && hashMode == HashMode::Array1Access && translationRefCounts)) {
                ps.unlockXEvicted();
            }
            physUsedCount.fetch_sub(1, std::memory_order_relaxed);
            if constexpr (kEnableAsyncPrefetchStats) {
                asyncFailedCompletionCount.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        yield(repeatCounter);
    }
}

Page* BufferManager::waitForPendingRead(PID pid, u64 stateAndVersion, u64* resolvedVersion) {
    if (PageState::getState(stateAndVersion) != PageState::ReadInFlight) {
        if (resolvedVersion != nullptr) {
            *resolvedVersion = stateAndVersion;
        }
        return nullptr;
    }

    PageState& ps = getPageState(pid);
    const u64 currentStateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
    if (PageState::getState(currentStateAndVersion) != PageState::ReadInFlight) {
        if (resolvedVersion != nullptr) {
            *resolvedVersion = currentStateAndVersion;
        }
        return nullptr;
    }

    const u32 request_slot = resolveAsyncRequestSlot(pid, currentStateAndVersion);
    if (request_slot == invalidAsyncRequestSlot) {
        if (resolvedVersion != nullptr) {
            *resolvedVersion = currentStateAndVersion;
        }
        return nullptr;
    }

    AsyncReadRequest& request = asyncReadRequests[request_slot];
    if (request.pid.load(std::memory_order_acquire) != pid ||
        request.status.load(std::memory_order_acquire) == AsyncReadRequest::Idle) {
        if (resolvedVersion != nullptr) {
            *resolvedVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        }
        return nullptr;
    }

    if constexpr (kEnableAsyncPrefetchStats) {
        asyncGuardWaitCount.fetch_add(1, std::memory_order_relaxed);
    }
    IOInterface* owner_interface = reinterpret_cast<IOInterface*>(
        request.ownerInterface.load(std::memory_order_acquire));
    if (owner_interface != nullptr) {
        owner_interface->waitForAsyncRead(request);
    } else {
        getIOInterface().waitForAsyncRead(request);
    }

    for (u64 repeatCounter = 0;; ++repeatCounter) {
        u64 current = ps.stateAndVersion.load(std::memory_order_acquire);
        const u64 state = PageState::getState(current);
        if (state == PageState::ReadInFlight) {
            if (request.status.load(std::memory_order_acquire) == AsyncReadRequest::Failed ||
                request.result.load(std::memory_order_acquire) != static_cast<int>(pageSize)) {
                rollbackFailedAsyncRead(pid, current);
                const u64 reloaded = ps.stateAndVersion.load(std::memory_order_acquire);
                if (resolvedVersion != nullptr) {
                    *resolvedVersion = reloaded;
                }
                return nullptr;
            }
            const u64 unlocked = PageState::sameVersion(current, PageState::Unlocked);
            if (ps.stateAndVersion.compare_exchange_weak(current, unlocked, std::memory_order_acq_rel,
                                                         std::memory_order_acquire)) {
                residentSet.insert(pid);
                releaseAsyncRequestSlot(pid, request_slot);
                if (resolvedVersion != nullptr) {
                    *resolvedVersion = unlocked;
                }
#if defined(CALICO_SPECIALIZATION_CALICO)
                return residentPtrCalico(pid, unlocked);
#elif defined(CALICO_SPECIALIZATION_HASH)
                return residentPtrHash(pid, unlocked);
#else
                return residentPtr(pid, unlocked);
#endif
            }
            continue;
        }

        if (resolvedVersion != nullptr) {
            *resolvedVersion = current;
        }
        if (!PageState::isResidentState(current)) {
            return nullptr;
        }

#if defined(CALICO_SPECIALIZATION_CALICO)
        return residentPtrCalico(pid, current);
#elif defined(CALICO_SPECIALIZATION_HASH)
        return residentPtrHash(pid, current);
#else
        return residentPtr(pid, current);
#endif
        yield(repeatCounter);
    }
}

Page* BufferManager::resolveOptimisticRead(PID pid, u64& version) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        u64 v = ps.stateAndVersion.load(std::memory_order_acquire);
        switch (PageState::getState(v)) {
            case PageState::Marked: {
                const u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                if (ps.stateAndVersion.compare_exchange_weak(v, newV, std::memory_order_acq_rel,
                                                             std::memory_order_acquire)) {
                    version = newV;
#if defined(CALICO_SPECIALIZATION_CALICO)
                    return residentPtrCalico(pid, newV);
#elif defined(CALICO_SPECIALIZATION_HASH)
                    return residentPtrHash(pid, newV);
#else
                    return residentPtr(pid, newV);
#endif
                }
                break;
            }
            case PageState::Locked:
                break;
            case PageState::ReadInFlight: {
                if (Page* page = waitForPendingRead(pid, v, &version)) {
                    return page;
                }
                break;
            }
            case PageState::Evicted:
                if (ps.tryLockX(v)) {
                    try {
                        Page* page = handleFault(pid);
                        unfixX(pid);
                        version = ps.stateAndVersion.load(std::memory_order_acquire);
                        (void)page;
#if defined(CALICO_SPECIALIZATION_CALICO)
                        return residentPtrCalico(pid, version);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        return residentPtrHash(pid, version);
#else
                        return residentPtr(pid, version);
#endif
                    } catch (...) {
                        ps.unlockXEvicted();
                        throw;
                    }
                }
                break;
            default:
                version = v;
#if defined(CALICO_SPECIALIZATION_CALICO)
                return residentPtrCalico(pid, v);
#elif defined(CALICO_SPECIALIZATION_HASH)
                return residentPtrHash(pid, v);
#else
                return residentPtr(pid, v);
#endif
        }
        yield(repeatCounter);
    }
}

u32 BufferManager::popFreeFrame(bool allow_eviction) {
    u32 frame = 0;
    for (u64 repeatCounter = 0;; ++repeatCounter) {
        if (freeFrameQueue.try_dequeue(frame)) return frame;

        // No free frame available right now; force an eviction attempt and retry.
        if (allow_eviction) {
            evict();
        } else {
            return 0;
        }
        // yield(repeatCounter);
    }
}

Page* BufferManager::acquireFrameForPid(PID pid, bool allow_eviction) {
    if (!useTraditional) return virtMem + pid;
    u32 frame = popFreeFrame(allow_eviction);
    if (frame == 0) return nullptr;
    if (hashMode == HashMode::Array) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        pidToFrameArray[pid].store(static_cast<u64>(frame), std::memory_order_release);
        frameToPidArray[frame].store(pid, std::memory_order_release);
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level ||
               hashMode == HashMode::Array3Level || hashMode == HashMode::HashThenArray) {
        
        // Increment reference counter for this OS page group (Array1Access only)
        if (hashMode == HashMode::Array1Access && translationRefCounts) {
            u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
            u64 refCountIdx = pid / entriesPerOSPage;
            
            // Atomically increment reference count with lock bit handling
            while (true) {
                u32 oldVal = translationRefCounts[refCountIdx].load(std::memory_order_acquire);
                
                // Check if locked (highest bit set)
                if (oldVal & REF_COUNT_LOCK_BIT) {
                    _mm_pause();  // Spin-wait during hole-punching
                    continue;
                }
                
                // Try to increment count (lower 31 bits)
                u32 count = oldVal & REF_COUNT_MASK;
                u32 newVal = (count + 1) & REF_COUNT_MASK;
                
                if (translationRefCounts[refCountIdx].compare_exchange_weak(oldVal, newVal, 
                                                                            std::memory_order_release,
                                                                            std::memory_order_acquire)) {
                    break;  // Successfully incremented
                }
            }
        }
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
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level ||
               hashMode == HashMode::Array3Level || hashMode == HashMode::HashThenArray) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        frameId = getPageState(pid).getFrameValue();
        assert(frameId != PageState::packedInvalidFrame);
        getPageState(pid).clearFrameValue();
        frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        
        // Reference counting and hole-punching for Array1Access only
        if (hashMode == HashMode::Array1Access && translationRefCounts) {
            u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
            u64 refCountIdx = pid / entriesPerOSPage;
            
            // Atomically decrement reference count with lock bit handling
            while (true) {
                u32 oldVal = translationRefCounts[refCountIdx].load(std::memory_order_acquire);
                
                // Check if locked (highest bit set)
                if (oldVal & REF_COUNT_LOCK_BIT) {
                    _mm_pause();
                    continue;
                }
                
                // Try to acquire lock first
                u32 lockedVal = oldVal | REF_COUNT_LOCK_BIT;
                if (translationRefCounts[refCountIdx].compare_exchange_weak(oldVal, lockedVal,
                                                                            std::memory_order_acquire,
                                                                            std::memory_order_acquire)) {
                    // We have the lock.
                    u32 count = oldVal & REF_COUNT_MASK;
                    if (count > 0) {
                        count--;
                    }
                    // must obtain the counter lock first before unlocking
                    getPageState(pid).unlockXEvicted();  // Unlock the page state and set it to 0
                    if (count == 0) {
                         // Perform hole-punching via madvise
                        void* osPageStart = reinterpret_cast<void*>(
                            reinterpret_cast<uintptr_t>(pageState) + 
                            (refCountIdx * entriesPerOSPage * sizeof(PageState))
                        );
                        
                        int result = madvise(osPageStart, translationOSPageSize, MADV_DONTNEED);
                        if (result == 0) {
                            holePunchCount.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            std::cerr << "madvise MADV_DONTNEED failed for pid " << pid
                                      << " errno: " << errno << std::endl;
                        }
                    }
                    
                    // Release lock and store new count
                    translationRefCounts[refCountIdx].store(count, std::memory_order_release);
                    break;
                }
            }
        }
    } else {
        bool ok = pidToFrameHash->erase(pid, frameId);
        assert(ok);
        static_cast<void>(ok);
        PID ignored;
        bool ok2 = frameToPidHash->erase(frameId, ignored);
        assert(ok2);
        static_cast<void>(ok2);
    }

    assert(frameId < physCount);
    u32 frameIdx = static_cast<u32>(frameId);
    if (!freeFrameQueue.enqueue(frameIdx)) {
        throw std::runtime_error("failed to enqueue free frame");
    }
}

void BufferManager::ensureFreePages() {
    if (physUsedCount * 100 >= physCount * 99.99) evict();
}

// allocated new page and fix it
Page* BufferManager::allocPage(PIDAllocator* allocator) {
    physUsedCount++;
    ensureFreePages();
    u64 pid;
    if (allocator && !ignorePidAllocator()) {
        allocator->lockSpin();
        if (allocator->next >= allocator->end) {
            static const u64 pidAllocChunkPages = std::max<u64>(1, envOr("PID_ALLOC_CHUNK_PAGES", 256));
            for (;;) {
                u64 start = allocCount.load(std::memory_order_relaxed);
                u64 next_aligned = ((start + pidAllocChunkPages - 1) / pidAllocChunkPages) * pidAllocChunkPages;
                u64 next_reserved = next_aligned + pidAllocChunkPages;
                if (allocCount.compare_exchange_weak(start, next_reserved, std::memory_order_acq_rel, std::memory_order_relaxed) == false) continue;
                allocator->next = start;
                allocator->end = next_reserved;
                break;
            }
        }
        pid = allocator->next++;
        allocator->unlockSpin();
    } else {
        pid = allocCount++;
    }

    if (pid >= virtCount) {
        cerr << "VIRTGB is too low" << endl;
        exit(EXIT_FAILURE);
    }
    u64 stateAndVersion = getPageState(pid).stateAndVersion;
    bool succ = getPageState(pid).tryLockX(stateAndVersion);
    assert(succ);
    static_cast<void>(succ);
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

    //updateAllocCountSnapshot(pid + 1);

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
    if constexpr (kEnableAsyncPrefetchStats) {
        asyncSyncFaultCount.fetch_add(1, std::memory_order_relaxed);
    }
    physUsedCount++;
    ensureFreePages();
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
#if defined(CALICO_SPECIALIZATION_CALICO)
                    return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
                    return residentPtrHash(pid, stateAndVersion);
#else
                    return residentPtr(pid, stateAndVersion);
#endif
                }
            }
            // For Locked state, just retry
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::ReadInFlight: {
                    waitForPendingRead(pid, stateAndVersion, nullptr);
                    break;
                }
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        try {
                            Page* page = handleFault(pid);
                            return page;
                        } catch (...) {
                            // Unlock page on failure to prevent livelock
                            ps.unlockXEvicted();
                            throw;
                        }
                    }
                    break;
                }
                case PageState::Marked:
                case PageState::Unlocked: {
                    if (ps.tryLockX(stateAndVersion)) {
                        // u64 lockedState = PageState::sameVersion(stateAndVersion, PageState::Locked);
#if defined(CALICO_SPECIALIZATION_CALICO)
                        return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        return residentPtrHash(pid, stateAndVersion);
#else
                        return residentPtr(pid, stateAndVersion);
#endif
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
#if defined(CALICO_SPECIALIZATION_CALICO)
                    return residentPtrCalico(pid, newState);
#elif defined(CALICO_SPECIALIZATION_HASH)
                    return residentPtrHash(pid, newState);
#else
                    return residentPtr(pid, newState);
#endif
                }
            }
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::ReadInFlight: {
                    waitForPendingRead(pid, stateAndVersion, nullptr);
                    break;
                }
                case PageState::Locked: {
                    break;
                }
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        try {
                            handleFault(pid);
                            ps.unlockX();
                        } catch (...) {
                            // Unlock page on failure to prevent livelock
                            ps.unlockXEvicted();
                            throw;
                        }
                    }
                    break;
                }
                default: {
                    if (ps.tryLockS(stateAndVersion)) {
                        u64 s = PageState::getState(stateAndVersion);
                        u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
                        u64 newState = PageState::sameVersion(stateAndVersion, nextState);
#if defined(CALICO_SPECIALIZATION_CALICO)
                        return residentPtrCalico(pid, newState);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        return residentPtrHash(pid, newState);
#else
                        return residentPtr(pid, newState);
#endif
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
        static_cast<void>(ret);
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
            if (state == PageState::ReadInFlight) {
                waitForPendingRead(pid, state_and_version, nullptr);
                continue;
            }
            if (state == PageState::Locked) {
                yield(repeatCounter);
                continue;
            }

            if (ps.tryLockS(state_and_version)) {
#if defined(CALICO_SPECIALIZATION_CALICO)
                Page* page_ptr = residentPtrCalico(pid, state_and_version);
#elif defined(CALICO_SPECIALIZATION_HASH)
                Page* page_ptr = residentPtrHash(pid, state_and_version);
#else
                Page* page_ptr = residentPtr(pid, state_and_version);
#endif
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

    u64 tries = 0;
    u64 selected = 0;
    // 0. find candidates, lock dirty ones in shared mode
    //while (toEvict.size() + toWrite.size() < batch) {
    while (selected < batch && tries < 10000){
        tries += 1;
        residentSet.iterateClockBatch(batch, [&](PID pid) {
            if (selected >= batch) {
                return;
            }
            PageState& ps = getPageState(pid);
            u64 v = ps.stateAndVersion;
            switch (PageState::getState(v)) {
                case PageState::Marked: {
                    // Acquire S-lock before touching page contents. Without this,
                    // concurrent eviction can clear hash translation and make
                    // residentPtrHash observe a transient miss.
                    if (ps.tryLockS(v)) {
                        u64 sharedStateAndVersion = PageState::sameVersion(v, PageState::SharedBase);
#if defined(CALICO_SPECIALIZATION_CALICO)
                        Page* page_ptr = residentPtrCalico(pid, sharedStateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        Page* page_ptr = residentPtrHash(pid, sharedStateAndVersion);
#else
                        Page* page_ptr = residentPtr(pid, sharedStateAndVersion);
#endif
                        if (page_ptr->dirty) {
                            toWrite.push_back(pid);
                            selected++;
                        } else {
                            // Upgrade single-reader shared lock to X lock in place.
                            u64 expectedShared = sharedStateAndVersion;
                            if (ps.stateAndVersion.compare_exchange_weak(
                                    expectedShared, PageState::sameVersion(sharedStateAndVersion, PageState::Locked))) {
                                toEvict.push_back(pid);
                                selected++;
                            } else {
                                ps.unlockS();
                            }
                        }
                    }
                    break;
                }
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
                                     u64 state = PageState::getState(v);
                                     if (state == PageState::Locked) return false;  // Already locked in phase 0.
                                     if (state != PageState::Marked) return true;
                                     return !ps.tryLockX(v);
                                 }),
                  toEvict.end());

    // 3. try to upgrade lock for dirty page candidates
    for (auto& pid : toWrite) {
        PageState& ps = getPageState(pid);
        u64 v = ps.stateAndVersion;
        if ((PageState::getState(v) == PageState::SharedBase) && // single shared reader
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
        if (useTraditional) {
            releaseFrame(pid);
        }
        bool succ = residentSet.remove(pid);
        assert(succ);
        static_cast<void>(succ);
        if (useTraditional && hashMode == HashMode::Array1Access && translationRefCounts) {
            // unlockXEvicted has been handled in releaseFrame for hole-punching case
        } else {
            getPageState(pid).unlockXEvicted();
        }
        
    }

    physUsedCount -= toEvict.size();
}

bool BufferManager::submitAsyncPrefetch(PID pid) {
    submitAsyncPrefetchPages(&pid, 1, nullptr);
    const u64 stateAndVersion = getPageState(pid).stateAndVersion.load(std::memory_order_acquire);
    return PageState::getState(stateAndVersion) == PageState::ReadInFlight;
}

void BufferManager::submitAsyncPrefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    //if (!asyncReadEnabled()) {
        prefetchPages(pages, n_pages, offsets_within_pages);
        return;
    //}

    if (!useTraditional || hashMode == HashMode::Array1Access || hashMode == HashMode::HashThenArray) {
        for (int i = 0; i < n_pages; ++i) {
            PageState& ps = getPageState(pages[i]);
            _mm_prefetch(reinterpret_cast<const char*>(&ps), _MM_HINT_T0);
        }
    }

    std::vector<PID> prefetch_pids;
    std::vector<Page*> prefetch_frames;
    std::vector<AsyncReadRequest*> prefetch_requests;
    std::vector<u32> prefetch_request_slots;
    prefetch_pids.reserve(n_pages);
    prefetch_frames.reserve(n_pages);
    prefetch_requests.reserve(n_pages);
    prefetch_request_slots.reserve(n_pages);

    auto rollback_prefetch = [&](size_t begin_idx) {
        for (size_t idx = begin_idx; idx < prefetch_pids.size(); ++idx) {
            const PID pid = prefetch_pids[idx];
            releaseFrame(pid);
            releaseAsyncRequestSlot(pid, prefetch_request_slots[idx]);
            if (!(useTraditional && hashMode == HashMode::Array1Access && translationRefCounts)) {
                getPageState(pid).unlockXEvicted();
            }
            physUsedCount.fetch_sub(1, std::memory_order_relaxed);
        }
    };

    for (int i = 0; i < n_pages; ++i) {
        const PID pid = pages[i];
        assert(pid < virtCount);

        PageState& ps = getPageState(pid);
        const u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        const u64 state = PageState::getState(stateAndVersion);


        if (PageState::isResidentState(stateAndVersion)) {
            if (state == PageState::Locked) {
                continue;
            }
            char* tmp = !useTraditional ? reinterpret_cast<char*>(virtMem + static_cast<size_t>(pid))
                     : reinterpret_cast<char*>(frameMem + static_cast<size_t>(PageState::extractFrame(stateAndVersion)));
            tmp += (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
            continue;
        }

        if (state == PageState::ReadInFlight) {
            if constexpr (kEnableAsyncPrefetchStats) {
                asyncCoalescedCount.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }

        if (state != PageState::Evicted || !ps.tryLockX(stateAndVersion)) {
            continue;
        }

        physUsedCount.fetch_add(1, std::memory_order_relaxed);
        ensureFreePages();
        Page* frame = acquireFrameForPid(pid);
        u32 request_slot = invalidAsyncRequestSlot;
        if (useTraditional) {
            request_slot = static_cast<u32>(frame - frameMem);
        } else if (!freeAsyncRequestSlotQueue.try_dequeue(request_slot)) {
            releaseFrame(pid);
            ps.unlockXEvicted();
            physUsedCount.fetch_sub(1, std::memory_order_relaxed);
            if constexpr (kEnableAsyncPrefetchStats) {
                asyncSubmitBackpressureCount.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        } else {
            pidToAsyncRequestSlot[pid].store(request_slot, std::memory_order_release);
        }

        AsyncReadRequest& request = asyncReadRequests[request_slot];
        prefetch_pids.push_back(pid);
        prefetch_frames.push_back(frame);
        prefetch_requests.push_back(&request);
        prefetch_request_slots.push_back(request_slot);
    }

    if (prefetch_pids.empty()) {
        return;
    }

    IOInterface& io_interface = getIOInterface();
    for (size_t idx = 0; idx < prefetch_pids.size(); ++idx) {
        AsyncReadRequest& request = *prefetch_requests[idx];
        request.pid.store(prefetch_pids[idx], std::memory_order_release);
        request.frame.store(useTraditional ? static_cast<u32>(prefetch_frames[idx] - frameMem) : invalidAsyncRequestSlot,
                            std::memory_order_release);
        request.ownerInterface.store(reinterpret_cast<uintptr_t>(&io_interface), std::memory_order_release);
        request.token.store(nextAsyncReadToken.fetch_add(1, std::memory_order_relaxed), std::memory_order_release);
        request.submitMicros.store(monotonicMicros(), std::memory_order_release);
        request.completionMicros.store(0, std::memory_order_release);
        request.lastWaitLogMicros.store(0, std::memory_order_release);
        request.submitWorker.store(workerThreadId, std::memory_order_release);
        request.completionWorker.store(std::numeric_limits<u32>::max(), std::memory_order_release);
        request.waiterCount.store(0, std::memory_order_release);
        request.result.store(0, std::memory_order_release);
        request.status.store(AsyncReadRequest::Submitted, std::memory_order_release);
    }

    size_t submitted_begin = 0;
    while (submitted_begin < prefetch_pids.size()) {
        const size_t submitted =
            io_interface.submitAsyncReads(prefetch_pids.data() + submitted_begin, prefetch_frames.data() + submitted_begin,
                                          prefetch_requests.data() + submitted_begin,
                                          prefetch_pids.size() - submitted_begin);
        if (submitted == 0) {
            break;
        }

        for (size_t idx = submitted_begin; idx < submitted_begin + submitted; ++idx) {
            PageState& ps = getPageState(prefetch_pids[idx]);
            const u64 lockedStateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
            ps.stateAndVersion.store(PageState::sameVersion(lockedStateAndVersion, PageState::ReadInFlight),
                                     std::memory_order_release);
            if constexpr (kEnableAsyncPrefetchStats) {
                asyncSubmitCount.fetch_add(1, std::memory_order_relaxed);
            }
        }
        submitted_begin += submitted;
    }

    if (submitted_begin < prefetch_pids.size()) {
        rollback_prefetch(submitted_begin);
    }
}

void BufferManager::prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    //if (!useTraditional || hashMode == HashMode::Array1Access || hashMode == HashMode::Array || hashMode == HashMode::Array2Level) {
    if (!useTraditional || hashMode == HashMode::Array1Access || hashMode == HashMode::HashThenArray) {
        for (int i = 0; i < n_pages; i++) {
            PageState& ps = getPageState(pages[i]);
            char * tmp = (char*)&ps;
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        }
    }

    // if (!useTraditional) {
    //     // In vmcache mode pages are addressed directly via virtMem, not frameMem.
    //     for (int i = 0; i < n_pages; i++) {
    //         PID pid = pages[i];
    //         assert(pid < virtCount);
    //         char* tmp = reinterpret_cast<char*>(virtMem + static_cast<size_t>(pid)) +
    //                     (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
    //         _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
    //     }
    //     return;
    // }

    // Use unique_ptr to defer allocation until we know we have pages to read
    std::unique_ptr<vector<PID>> pidsToRead;
    std::unique_ptr<vector<Page*>> destinations;

    // Check which pages are not in memory and need to be loaded
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        assert(pid < virtCount);
        // if (i + 1 < n_pages) {
        //     // Prefetch next PageState
        //     PageState& next_ps = getPageState(pages[i + 1]);
        //     char* tmp = (char*)&next_ps;
        //     _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        // }

        PageState& ps = getPageState(pid);
        u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        u64 state = PageState::getState(stateAndVersion);

        if (state == PageState::ReadInFlight) {
            continue;
        }

        if (hashMode == HashMode::Lockfree || hashMode == HashMode::Unordered ||
            hashMode == HashMode::OpenAddress || hashMode == HashMode::RobinHood) {
            // Double check that the page is still mapped
            u64 frame;
            if (pidToFrameHash->tryGet(pid, frame)) {
                // Page is already mapped, skip
                continue;
            }
        }

        // Skip pages that are already in memory (not evicted)
        if (PageState::isResidentState(stateAndVersion)) {
            char* tmp = !useTraditional ? reinterpret_cast<char*>(virtMem + static_cast<size_t>(pid))
                     : reinterpret_cast<char*>(frameMem + static_cast<size_t>(PageState::extractFrame(stateAndVersion)));
            tmp += (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
            // if (offsets_within_pages != nullptr) {
            //       // Touch the page to bring it into cache
            //     //   char *tmp = (((char*)residentPtr(pid, stateAndVersion)) + offsets_within_pages[i]);
            //     //   _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T2);
            //     //   _mm_prefetch(reinterpret_cast<const char*>(tmp) + 64, _MM_HINT_T2);
            // }
            continue;
        }

        // Lock the page first before allocating frame
        if (ps.tryLockX(stateAndVersion)) {
            if (!useTraditional)
                ensureFreePages();
            // Try to acquire a frame for this page
            Page* frame = acquireFrameForPid(pid, false);
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
                // Failed to acquire frame, return to evicted state
                ps.unlockXEvicted();
            }
        }
    }

    // Batch read all pages that need to be loaded
    if (pidsToRead && !pidsToRead->empty()) {
        getIOInterface().readPages(*pidsToRead, *destinations);

        // Unlock all successfully loaded pages
        for (PID pid : *pidsToRead) {
            // Add to resident set
            residentSet.insert(pid);
            PageState& ps = getPageState(pid);
            ps.unlockX();
        }
    }
}

void BufferManager::prefetchResidentPages(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    if (!useTraditional || hashMode == HashMode::Array1Access || hashMode == HashMode::HashThenArray) {
        for (int i = 0; i < n_pages; i++) {
            PageState& ps = getPageState(pages[i]);
            char* tmp = reinterpret_cast<char*>(&ps);
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        }
    }

    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        assert(pid < virtCount);

        u64 stateAndVersion = getPageState(pid).stateAndVersion.load(std::memory_order_acquire);
        u64 state = PageState::getState(stateAndVersion);
        if (state == PageState::Evicted || state == PageState::ReadInFlight || state == PageState::Locked) continue;

        u64 frame = PageState::extractFrame(stateAndVersion);
        char* tmp = reinterpret_cast<char*>(frameMem + static_cast<size_t>(frame)) +
                    (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
        _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
    }
}

// BTreeNode Method Implementations

[[maybe_unused]] static unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }

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

void BTreeNode::splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep, PIDAllocator* allocator) {
    assert(sepSlot > 0);
    assert(sepSlot < (pageSize / sizeof(PID)));

    BTreeNode tmp(isLeaf);
    BTreeNode* nodeLeft = &tmp;

    AllocGuard<BTreeNode> newNode(allocator, isLeaf);
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
    PID rootNodePid;
    {
        AllocGuard<BTreeNode> rootNode(&pidAllocator, true);
        rootNodePid = rootNode.pid;
    }
    
    {
        GuardX<MetaDataPage> page(metadataPageId);
        slotId = btreeslotcounter++;
        page->roots[slotId] = rootNodePid;
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
        // parent already holds the lock on metadataPageId, so reuse it
        MetaDataPage* meta_page = reinterpret_cast<MetaDataPage*>(parent.ptr);

        AllocGuard<BTreeNode> newRoot(&pidAllocator, false);
        newRoot->upperInnerNode = node.pid;
        {
            meta_page->roots[slotId] = newRoot.pid;
            meta_page->dirty = true;
        }
        parent = std::move(newRoot);
    }

    // split
    BTreeNode::SeparatorInfo sepInfo = node->findSeparator(splitOrdered.load());
    u8 sepKey[sepInfo.len];
    node->getSep(sepKey, sepInfo);

    if (parent->hasSpaceFor(sepInfo.len, sizeof(PID))) {  // is there enough space in the parent for the separator?
        node->splitNode(parent.ptr, sepInfo.slot, {sepKey, sepInfo.len}, &pidAllocator);
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

            u16 pos = 0;
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

u64 BufferManager::countZeroRefCountGroups() {
    if (!translationRefCounts) return 0;
    u64 count = 0;
    u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
    u64 currentAllocCount = allocCount.load(std::memory_order_relaxed);
    u64 estimatedNumRefCountGroups = (currentAllocCount + entriesPerOSPage - 1) / entriesPerOSPage;
    
    for (u64 i = 0; i < estimatedNumRefCountGroups; i++) {
        u32 val = translationRefCounts[i].load(std::memory_order_relaxed);
        if ((val & REF_COUNT_MASK) == 0) {
            count++;
        }
    }
    return count;
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
        bm_ptr->updateAllocCountSnapshot(bm_ptr->allocCount.load(std::memory_order_acquire));
        bm_ptr->flushAll();
    }
}

void warmup_storage_to_alloc_count() {
    if (bm_ptr == nullptr || !bm_ptr->useTraditional) {
        return;
    }

    const u64 alloc_limit = bm_ptr->allocCount.load(std::memory_order_acquire);
    if (alloc_limit <= 1) {
        return;
    }

    std::cerr << "[Calico Warmup] Sequentially faulting pid range [1, " << (alloc_limit - 1)
              << "]" << std::endl;

    for (PID pid = 1; pid < alloc_limit; ++pid) {
        Page* page = bm_ptr->fixS(pid);
        (void)page;
        bm_ptr->unfixS(pid);
    }

    std::cerr << "[Calico Warmup] Completed sequential warmup through allocCount=" << alloc_limit
              << std::endl;
}

std::vector<u64> snapshot_pid_to_frame_range(u64 start_pid, u64 count) {
    std::vector<u64> frames;
    if (bm_ptr == nullptr || !bm_ptr->useTraditional || count == 0) {
        return frames;
    }

    const u64 alloc_limit = bm_ptr->allocCount.load(std::memory_order_acquire);
    if (start_pid >= alloc_limit) {
        return frames;
    }

    const u64 end_pid = std::min<u64>(alloc_limit, start_pid + count);
    frames.reserve(end_pid - start_pid);
    for (u64 pid = start_pid; pid < end_pid; ++pid) {
        const u64 state_and_version = bm_ptr->getPageState(static_cast<PID>(pid)).stateAndVersion.load(std::memory_order_acquire);
        if (PageState::getState(state_and_version) == PageState::Evicted) {
            frames.push_back(BufferManager::invalidFrame);
            continue;
        }
        frames.push_back(bm_ptr->resolveMappedFrame(static_cast<PID>(pid), state_and_version));
    }
    return frames;
}

// This function will be called from Python to shut down the system
void shutdown_system() {
    if (bm_ptr != nullptr) {
        flush_system();
        delete bm_ptr;
        bm_ptr = nullptr;
    }
}
