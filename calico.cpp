#include <atomic>
#include <algorithm>
#include <cassert>
#include <csignal>
#include <exception>
#include <stdexcept>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>
#include <limits>
#include <memory>
#include <unordered_map>
#include <array>
#include <shared_mutex>
#include <unordered_set>
#include <queue>
#include <random>
#include <chrono>
#include <cmath>
#include <span>
#include <cstddef>

#include <errno.h>
#include <libaio.h>
#include <sys/mman.h>

#include "hash_tables.hpp"
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <immintrin.h>

// --- Intrinsics Headers ---
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <nmmintrin.h> // For CRC32 and prefetch


#include "exmap.h"

__thread uint16_t workerThreadId = 0;
__thread int32_t tpcchistorycounter = 0;
#include "tpcc/TPCCWorkload.hpp"
#include "tpcc/ZipfianGenerator.hpp"
#include "perf_util.hpp"

using namespace std;
using namespace tpcc;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64 PID; // page id type

static const u64 pageSize = 4096;


__attribute__((target("sse4.2")))
uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;
    uint64_t crc = _mm_crc32_u64(seed, key);
    return crc * ((k << 32) + 1);
}

template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

struct alignas(4096) Page {
   bool dirty;
};

struct GraphNodePage: public Page {
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

#define die(msg) do { perror(msg); exit(EXIT_FAILURE); } while(0)

uint64_t rdtsc() {
   uint32_t hi, lo;
   __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
   return static_cast<uint64_t>(lo)|(static_cast<uint64_t>(hi)<<32);
}

// exmap helper function
static int exmapAction(int exmapfd, exmap_opcode op, u16 len) {
   struct exmap_action_params params_free = { .interface = workerThreadId, .iov_len = len, .opcode = (u16)op, };
   return ioctl(exmapfd, EXMAP_IOCTL_ACTION, &params_free);
}

// allocate memory using huge pages
void* allocHuge(size_t size) {
   void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
   madvise(p, size, MADV_HUGEPAGE);
   return p;
}

// use when lock is not free
void yield(u64 counter) {
   _mm_pause();
}

struct PageState {
   atomic<u64> stateAndVersion;

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
      if (!packedMode)
         return (extractLow(oldStateAndVersion) + 1) & lowMask;
      u64 framePart = oldStateAndVersion & frameMask;
      u64 versionPart = (oldStateAndVersion & versionMaskPacked) >> 32;
      versionPart = (versionPart + 1) & versionMaxPacked;
      return ((versionPart << 32) & versionMaskPacked) | framePart;
   }

   void init() {
      u64 low = 0;
      if (packedMode)
         low = (0ULL << 32) | static_cast<u64>(packedInvalidFrame);
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
      if (s<MaxShared)
         return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, s+1));
      if (s==Marked)
         return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, 1));
      return false;
   }

   void unlockS() {
      while (true) {
         u64 oldStateAndVersion = stateAndVersion.load();
         u64 state = getState(oldStateAndVersion);
         assert(state>0 && state<=MaxShared);
         if (stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, state-1)))
            return;
      }
   }

   bool tryMark(u64 oldStateAndVersion) {
      assert(getState(oldStateAndVersion)==Unlocked);
      return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Marked));
   }

   static u64 getState(u64 v) { return v >> 56; };
   u64 getState() { return getState(stateAndVersion.load()); }

   static u64 getFrameValue(u64 v) {
      if (!packedMode)
         return 0;
      return v & frameMask;
   }

   u64 getFrameValue() const {
      if (!packedMode)
         return 0;
      return extractFrame(stateAndVersion.load(std::memory_order_acquire));
   }

   void setFrameValue(u64 frame) {
      if (!packedMode)
         return;
      frame &= frameMask;
      u64 expected = stateAndVersion.load(std::memory_order_relaxed);
      while (true) {
         u64 desired = (expected & ~frameMask) | frame;
         if (stateAndVersion.compare_exchange_weak(expected, desired, std::memory_order_release, std::memory_order_relaxed))
            return;
      }
   }

   void clearFrameValue() {
      if (packedMode)
         setFrameValue(packedInvalidFrame);
   }

   void operator=(PageState&) = delete;
};

bool PageState::packedMode = false;

// open addressing hash table used for second chance replacement to keep track of currently-cached pages
struct ResidentPageSet {
   static const u64 empty = ~0ull;
   static const u64 tombstone = (~0ull)-1;

   struct Entry {
      atomic<u64> pid;
   };

   Entry* ht;
   u64 count;
   u64 mask;
   atomic<u64> clockPos;

   ResidentPageSet(u64 maxCount) : count(next_pow2(maxCount * 1.5)), mask(count - 1), clockPos(0) {
      ht = (Entry*)allocHuge(count * sizeof(Entry));
      memset((void*)ht, 0xFF, count * sizeof(Entry));
   }

   ~ResidentPageSet() {
      munmap(ht, count * sizeof(u64));
   }

   u64 next_pow2(u64 x) {
      return 1<<(64-__builtin_clzl(x-1));
   }

   u64 hash(u64 k) {
      const u64 m = 0xc6a4a7935bd1e995;
      const int r = 47;
      u64 h = 0x8445d61a4e774912 ^ (8*m);
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

   void insert(u64 pid) {
      u64 pos = hash(pid) & mask;
      while (true) {
         u64 curr = ht[pos].pid.load();
         assert(curr != pid);
         if ((curr == empty) || (curr == tombstone))
            if (ht[pos].pid.compare_exchange_strong(curr, pid))
               return;

         pos = (pos + 1) & mask;
      }
   }

   bool remove(u64 pid) {
      u64 pos = hash(pid) & mask;
      while (true) {
         u64 curr = ht[pos].pid.load();
         if (curr == empty)
            return false;

         if (curr == pid)
            if (ht[pos].pid.compare_exchange_strong(curr, tombstone))
               return true;

         pos = (pos + 1) & mask;
      }
   }

   template<class Fn>
   void iterateClockBatch(u64 batch, Fn fn) {
      u64 pos, newPos;
      do {
         pos = clockPos.load();
         newPos = (pos+batch) % count;
      } while (!clockPos.compare_exchange_strong(pos, newPos));

      for (u64 i=0; i<batch; i++) {
         u64 curr = ht[pos].pid.load();
         if ((curr != tombstone) && (curr != empty))
            fn(curr);
         pos = (pos + 1) & mask;
      }
   }
};

struct BufferManager;

// libaio interface used to write batches of pages
struct LibaioInterface {
   static const u64 maxIOs = 256;

   int blockfd;
   BufferManager* bm;
   io_context_t ctx;
   iocb cb[maxIOs];
   iocb* cbPtr[maxIOs];
   io_event events[maxIOs];

   LibaioInterface(int blockfd, BufferManager* bm);

   void writePages(const vector<PID>& pages);
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
   vector<LibaioInterface> libaioInterface;

   enum class HashMode { Array1Access = 0, Unordered = 1, OpenAddress = 2, Array = 3, Lockfree = 4 };

   bool useTraditional;
   bool useExmap;
   HashMode hashMode;
   int blockfd;
   int exmapfd;

   static constexpr u64 invalidFrame = std::numeric_limits<u64>::max();
   static constexpr PID invalidPID = std::numeric_limits<PID>::max();
   static constexpr u32 freePartitionCount = 64;

   atomic<u64> physUsedCount;
   ResidentPageSet residentSet;
   atomic<u64> allocCount;

   atomic<u64> readCount;
   atomic<u64> writeCount;

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

   PageState& getPageState(PID pid) {
      return pageState[pid];
   }

   BufferManager();
   ~BufferManager() {}

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

   Page* acquireFrameForPid(PID pid);
   void releaseFrame(PID pid);
   u32 popFreeFrame();

   u32 partitionForFrame(u64 frame) const;
   void lockPartition(u32 partition);
   void unlockPartition(u32 partition);
};


LibaioInterface::LibaioInterface(int blockfd, BufferManager* bm) : blockfd(blockfd), bm(bm) {
   memset(&ctx, 0, sizeof(io_context_t));
   int ret = io_setup(maxIOs, &ctx);
   if (ret != 0) {
      std::cerr << "libaio io_setup error: " << ret << " ";
      switch (-ret) {
         case EAGAIN: std::cerr << "EAGAIN"; break;
         case EFAULT: std::cerr << "EFAULT"; break;
         case EINVAL: std::cerr << "EINVAL"; break;
         case ENOMEM: std::cerr << "ENOMEM"; break;
         case ENOSYS: std::cerr << "ENOSYS"; break;
      };
      exit(EXIT_FAILURE);
   }
}

void LibaioInterface::writePages(const vector<PID>& pages) {
   assert(pages.size() < maxIOs);
   for (u64 i=0; i<pages.size(); i++) {
      PID pid = pages[i];
      Page* page = bm->preparePageForWrite(pid);
      cbPtr[i] = &cb[i];
      io_prep_pwrite(cb+i, blockfd, page, pageSize, pageSize*pid);
   }
   int cnt = io_submit(ctx, pages.size(), cbPtr);
   assert(cnt == pages.size());
   cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
   assert(cnt == pages.size());
}


BufferManager bm;

struct OLCRestartException {};

template<class T>
struct GuardO {
   PID pid;
   T* ptr;
   u64 version;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardO(u64 pid) : pid(pid), ptr(nullptr) {
      init();
   }

   template<class T2>
   GuardO(u64 pid, GuardO<T2>& parent)  {
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
      for (u64 repeatCounter=0; ; repeatCounter++) {
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
      if (pid != moved)
         checkVersionAndRestart();
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
         if (version == stateAndVersion) // fast path, nothing changed
            return;
         if ((stateAndVersion<<8) == (version<<8)) { // same version
            u64 state = PageState::getState(stateAndVersion);
            if (state <= PageState::MaxShared)
               return; // ignore shared locks
            if (state == PageState::Marked)
               if (ps.stateAndVersion.compare_exchange_weak(stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                  return; // mark cleared
         }
         if (std::uncaught_exceptions()==0)
            throw OLCRestartException();
      }
   }

   // destructor
   ~GuardO() noexcept(false) {
      checkVersionAndRestart();
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

template<class T>
struct GuardX {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   GuardX(): pid(moved), ptr(nullptr) {}

   // constructor
   explicit GuardX(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixX(pid));
      ptr->dirty = true;
   }

   explicit GuardX(GuardO<T>&& other) {
      assert(other.pid != moved);
      for (u64 repeatCounter=0; ; repeatCounter++) {
         PageState& ps = bm.getPageState(other.pid);
         u64 stateAndVersion = ps.stateAndVersion;
         if ((stateAndVersion<<8) != (other.version<<8))
            throw OLCRestartException();
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
      if (pid != moved)
         bm.unfixX(pid);
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

template<class T>
struct AllocGuard : public GuardX<T> {
   template <typename ...Params>
   AllocGuard(Params&&... params) {
      GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage());
      new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
      GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
   }
};

template<class T>
struct GuardS {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardS(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixS(pid));
   }

   GuardS(GuardO<T>&& other) {
      assert(other.pid != moved);
      if (bm.getPageState(other.pid).tryLockS(other.version)) { // XXX: optimize?
         pid = other.pid;
         ptr = other.ptr;
         other.pid = moved;
         other.ptr = nullptr;
      } else {
         throw OLCRestartException();
      }
   }

   GuardS(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
   }

   // assignment operator
   GuardS& operator=(const GuardS&) = delete;

   // move assignment operator
   GuardS& operator=(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
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
      if (pid != moved)
         bm.unfixS(pid);
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

u64 envOr(const char* env, u64 value) {
   if (getenv(env))
      return atof(getenv(env));
   return value;
}

double envOrDouble(const char* env, double value) {
   if (getenv(env))
      return atof(getenv(env));
   return value;
}

BufferManager::BufferManager()
   : virtSize(envOr("VIRTGB", 16) * gb),
     physSize(envOr("PHYSGB", 4) * gb),
     virtCount(virtSize / pageSize),
     physCount(physSize / pageSize),
     residentSet(physCount) {
   assert(virtSize >= physSize);
   const char* path = getenv("BLOCK") ? getenv("BLOCK") : "/tmp/bm";
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
   else
      hashMode = HashMode::Array1Access;
   PageState::setPackedMode(hashMode == HashMode::Array1Access);
   virtMem = nullptr;
   frameMem = nullptr;
   bool disableHugePageForFrameMem = envOr("NOHUGEPAGE_FRAME_MEM", 0);


   u64 virtAllocSize = virtSize + (1 << 16); // guard space for optimistic reads

   if (useTraditional) {
      frameMem = static_cast<Page*>(allocHuge(physCount * sizeof(Page)));
      if (frameMem == MAP_FAILED)
         die("allocHuge frameMem");
      if (disableHugePageForFrameMem) {
         madvise(frameMem, physCount * sizeof(Page), MADV_NOHUGEPAGE);
      }
      partitionFrameSpan = std::max<u32>(1, static_cast<u32>((physCount + freePartitionCount - 1) / freePartitionCount));
      for (u32 i = 0; i < freePartitionCount; i++) {
         auto& partition = freePartitions[i];
         partition.lock.clear(std::memory_order_relaxed);
         partition.frames.clear();
         u32 start = i * partitionFrameSpan;
         if (start >= physCount)
            continue;
         u32 end = std::min<u32>(start + partitionFrameSpan, physCount);
         partition.frames.reserve(end - start);
         for (u32 frame = start; frame < end; frame++)
            partition.frames.push_back(frame);
         if (!partition.frames.empty())
            std::random_shuffle(partition.frames.begin(), partition.frames.end());
         partition.top = partition.frames.size();
      }
      frameToPidArray.reset(new std::atomic<PID>[physCount]);
      for (u64 i = 0; i < physCount; i++)
         frameToPidArray[i].store(invalidPID, std::memory_order_relaxed);

      if (hashMode == HashMode::Array) {
         pidToFrameArray.reset(new std::atomic<u64>[virtCount]);
         if (virtCount * sizeof(std::atomic<u64>) >= 2 * mb)
            madvise(pidToFrameArray.get(), virtCount * sizeof(std::atomic<u64>), MADV_HUGEPAGE);
         for (u64 i = 0; i < virtCount; i++)
            pidToFrameArray[i].store(invalidFrame, std::memory_order_relaxed);
      } else if (hashMode == HashMode::Array1Access) {
         pidToFrameArray.reset();
         pidToFrameHash.reset();
         frameToPidHash.reset();
      } else {
         if (hashMode == HashMode::Unordered) {
            pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64>>();
            frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID>>();
         } else if (hashMode == HashMode::OpenAddress) {
            pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64>>();
            frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID>>();
         } else if (hashMode == HashMode::Lockfree) {
            pidToFrameHash = std::make_unique<LockFreeHashMap<PID, u64>>();
            frameToPidHash = std::make_unique<LockFreeHashMap<u64, PID>>();
         }
         else {
            die("invalid config for hashmap");
         }
         pidToFrameHash->reserve(virtCount);
         frameToPidHash->reserve(physCount);
      }
      // round robin cursors will skip empty partitions automatically
   } else {
      if (useExmap) {
         exmapfd = open("/dev/exmap", O_RDWR);
         if (exmapfd < 0)
            die("open exmap");

         struct exmap_ioctl_setup buffer;
         buffer.fd             = blockfd;
         buffer.max_interfaces = maxWorkerThreads;
         buffer.buffer_size    = physCount;
         buffer.flags          = 0;
         if (ioctl(exmapfd, EXMAP_IOCTL_SETUP, &buffer) < 0)
            die("ioctl: exmap_setup");

         for (unsigned i = 0; i < maxWorkerThreads; i++) {
            exmapInterface[i] = (struct exmap_user_interface*) mmap(NULL, pageSize, PROT_READ | PROT_WRITE, MAP_SHARED, exmapfd, EXMAP_OFF_INTERFACE(i));
            if (exmapInterface[i] == MAP_FAILED)
               die("setup exmapInterface");
         }

         virtMem = (Page*) mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_SHARED, exmapfd, 0);
      } else {
         virtMem = (Page*) mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
         madvise(virtMem, virtAllocSize, MADV_NOHUGEPAGE);
      }
      if (virtMem == MAP_FAILED)
         die("mmap failed");
   }

   pageState = (PageState*) allocHuge(virtCount * sizeof(PageState));
   for (u64 i = 0; i < virtCount; i++)
      pageState[i].init();

   libaioInterface.reserve(maxWorkerThreads);
   for (unsigned i = 0; i < maxWorkerThreads; i++)
      libaioInterface.emplace_back(LibaioInterface(blockfd, this));

   physUsedCount = 0;
   allocCount = 1; // pid 0 reserved for meta data
   readCount = 0;
   writeCount = 0;
   batch = envOr("BATCH", 64);

   cerr << "vmcache "
        << "blk:" << path
        << " virtgb:" << virtSize / gb
        << " physgb:" << physSize / gb
        << " traditional:" << useTraditional
        << " trad_hash:" << static_cast<int>(hashMode)
        << " exmap:" << useExmap
        << endl;
}

bool BufferManager::isValidPtr(void* page) {
   if (useTraditional)
      return (page >= frameMem) && (page < (frameMem + physCount));
   if (!virtMem)
      return false;
   return (page >= virtMem) && (page < (virtMem + virtCount + 16));
}

PID BufferManager::toPID(void* page) {
   Page* ptr = reinterpret_cast<Page*>(page);
   if (useTraditional) {
      auto idx = ptr - frameMem;
      assert(idx >= 0 && static_cast<u64>(idx) < physCount);
      if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access) {
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

Page* BufferManager::residentPtr(PID pid) {
   assert(pid < virtCount);
   if (!useTraditional)
      return virtMem + pid;
   PageState& ps = getPageState(pid);
   u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
   return residentPtr(pid, ps, stateAndVersion);
}

Page* BufferManager::residentPtr(PID pid, PageState& ps, u64 stateAndVersion) {
   assert(pid < virtCount);
   if (!useTraditional)
      return virtMem + pid;

   if (hashMode == HashMode::Array) {
      u64 frame = pidToFrameArray[pid].load(std::memory_order_acquire);
      assert(frame != invalidFrame);
      assert(frame < physCount);
      return frameMem + static_cast<size_t>(frame);
   }

   if (hashMode == HashMode::Array1Access) {
      u64 frame = PageState::extractFrame(stateAndVersion);
      assert(frame != PageState::packedInvalidFrame);
      assert(frame < physCount);
      return frameMem + static_cast<size_t>(frame);
   }

   u64 frame;
   bool ok = pidToFrameHash->tryGet(pid, frame);
   assert(ok);
   assert(frame < physCount);
   return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::toPtr(PID pid) {
   return residentPtr(pid);
}

Page* BufferManager::preparePageForWrite(PID pid) {
   Page* page = residentPtr(pid);
   page->dirty = false;
   return page;
}

static thread_local u32 freePartitionCursor = 0;

u32 BufferManager::popFreeFrame() {
   for (u32 attempt = 0; attempt < freePartitionCount; ++attempt) {
      u32 partition = (freePartitionCursor + attempt) % freePartitionCount;
      auto& part = freePartitions[partition];
      if (part.frames.empty())
         continue;

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
   if (!useTraditional)
      return virtMem + pid;
   u32 frame = popFreeFrame();
   if (hashMode == HashMode::Array) {
      pidToFrameArray[pid].store(static_cast<u64>(frame), std::memory_order_release);
      frameToPidArray[frame].store(pid, std::memory_order_release);
   } else if (hashMode == HashMode::Array1Access) {
      pageState[pid].setFrameValue(frame);
      frameToPidArray[frame].store(pid, std::memory_order_release);
   } else {
      u64 frameId = static_cast<u64>(frame);
      pidToFrameHash->insertOrAssign(pid, frameId);
      frameToPidHash->insertOrAssign(frameId, pid);
   }
   return frameMem + frame;
}

void BufferManager::releaseFrame(PID pid) {
   if (!useTraditional)
      return;
   u64 frameId;
   if (hashMode == HashMode::Array) {
      frameId = pidToFrameArray[pid].exchange(invalidFrame, std::memory_order_acq_rel);
      assert(frameId != invalidFrame);
      frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
   } else if (hashMode == HashMode::Array1Access) {
      frameId = pageState[pid].getFrameValue();
      assert(frameId != PageState::packedInvalidFrame);
      pageState[pid].clearFrameValue();
      frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
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
   if (partitionFrameSpan == 0)
      return 0;
   u32 partition = static_cast<u32>(frame / partitionFrameSpan);
   return (partition >= freePartitionCount) ? (freePartitionCount - 1) : partition;
}

void BufferManager::lockPartition(u32 partition) {
   while (freePartitions[partition].lock.test_and_set(std::memory_order_acquire))
      _mm_pause();
}

void BufferManager::unlockPartition(u32 partition) {
   freePartitions[partition].lock.clear(std::memory_order_release);
}

static void runGraphBenchmark(u64 nodeCount, u64 startIndex) {
   if (nodeCount == 0) {
      std::cout << "graph_bfs nodes=0" << std::endl;
      return;
   }

   u32 desiredNeighbors = static_cast<u32>(std::log(static_cast<double>(std::max<u64>(2, nodeCount))) / std::log(2.0));
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
      bm.unfixX(pid);
   }


   for (u64 i = 0; i < nodeCount; i++) {
      GraphNodePage* node = reinterpret_cast<GraphNodePage*>(bm.fixX(nodePids[i]));
      node->neighborCount = desiredNeighbors;
      PID* neighbors = node->neighborData();
      for (u32 j = 0; j < desiredNeighbors; j++)
         neighbors[j] = nodePids[distNode(rng)];
      bm.unfixX(nodePids[i]);
   }


   cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;
   PID basePid = nodePids.front();
   PID maxPid = nodePids.back();

   std::vector<uint8_t> visited(nodeCount, 0);
   u64 clampedStart = std::min<u64>(startIndex, nodeCount - 1);
   PID startPid = nodePids[clampedStart];
   visited[clampedStart] = 1;
   bool enablePerf = envOr("PERF", 0);
   u64 visitCount = 0;
   u64 totalValue = 0;
   auto begin = std::chrono::steady_clock::now();
   {
      
      std::unique_ptr<perf::Scope> perfScope;
      if (enablePerf)
         perfScope = std::make_unique<perf::Scope>();
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

         GuardS<GraphNodePage> node(current);
         GraphNodePage* page = node.ptr;
         const PID* neighbors = page->neighborData();
         for (u32 i = 0; i < page->neighborCount; i++) {
            PID neighborPid = neighbors[i];
            // if (neighborPid < basePid || neighborPid > maxPid)
            //    continue;
            u64 idx64 = neighborPid - basePid;
            // if (idx64 >= nodeCount)
            //    continue;
            u32 idx = static_cast<u32>(idx64);
            // if (!visited[idx]) {
            //    visited[idx] = 1;
               {
                  GuardS<GraphNodePage> neighborNode(neighborPid);
                  // compute something
                  totalValue += neighborNode->value;
                  if (!visited[idx]) {
                     visited[idx] = 1;
                     frontier.push(neighborPid);      
                  }
               }
               
            // }
         }
      }

   }

   auto end = std::chrono::steady_clock::now();
   auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
   if (enablePerf)
      perf::Aggregator::instance().report(visitCount);
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
   *out << "graph_bfs nodes=" << nodeCount
        << " neighbors_per_node=" << desiredNeighbors
        << " visited=" << visitCount
        << " value_sum=" << totalValue
        << " time_ms=" << durationMs
        << std::endl;
}

void BufferManager::ensureFreePages() {
   if (physUsedCount >= physCount*0.95)
      evict();
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

   return page;
}

Page* BufferManager::handleFault(PID pid) {
   physUsedCount++;
   ensureFreePages();
   Page* page = acquireFrameForPid(pid);
   readPage(pid, page);
   residentSet.insert(pid);
   return page;
}

Page* BufferManager::fixX(PID pid) {
   PageState& ps = getPageState(pid);
   for (u64 repeatCounter=0; ; repeatCounter++) {
      u64 stateAndVersion = ps.stateAndVersion.load();
      switch (PageState::getState(stateAndVersion)) {
         case PageState::Evicted: {
            if (ps.tryLockX(stateAndVersion)) {
               Page* page = handleFault(pid);
               return page;
            }
            break;
         }
         case PageState::Marked: case PageState::Unlocked: {
            if (ps.tryLockX(stateAndVersion)) {
               u64 lockedState = PageState::sameVersion(stateAndVersion, PageState::Locked);
               return residentPtr(pid, ps, stateAndVersion);
            }
            break;
         }
      }
      yield(repeatCounter);
   }
}

Page* BufferManager::fixS(PID pid) {
   PageState& ps = getPageState(pid);
   for (u64 repeatCounter=0; ; repeatCounter++) {
      u64 stateAndVersion = ps.stateAndVersion;
      switch (PageState::getState(stateAndVersion)) {
         case PageState::Locked: {
            break;
         } case PageState::Evicted: {
            if (ps.tryLockX(stateAndVersion)) {
               handleFault(pid);
               ps.unlockX();
            }
            break;
         }
         default: {
            if (ps.tryLockS(stateAndVersion)) {
               u64 s = PageState::getState(stateAndVersion);
               u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
               u64 newState = PageState::sameVersion(stateAndVersion, nextState);
               return residentPtr(pid, ps, newState);
            }
         }
      }
      yield(repeatCounter);
   }
}

void BufferManager::unfixS(PID pid) {
   getPageState(pid).unlockS();
}

void BufferManager::unfixX(PID pid) {
   getPageState(pid).unlockX();
}

void BufferManager::readPage(PID pid, Page* dest) {
   if (useExmap) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
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
      int ret = pread(blockfd, dest, pageSize, pid*pageSize);
      assert(ret==pageSize);
      dest->dirty = false;
      readCount++;
   }
}

void BufferManager::evict() {
   vector<PID> toEvict;
   toEvict.reserve(batch);
   vector<PID> toWrite;
   toWrite.reserve(batch);

   // 0. find candidates, lock dirty ones in shared mode
   while (toEvict.size()+toWrite.size() < batch) {
      residentSet.iterateClockBatch(batch, [&](PID pid) {
         PageState& ps = getPageState(pid);
         u64 v = ps.stateAndVersion;
         switch (PageState::getState(v)) {
            case PageState::Marked:
               if (residentPtr(pid)->dirty) {
                  if (ps.tryLockS(v))
                     toWrite.push_back(pid);
               } else {
                  toEvict.push_back(pid);
               }
               break;
            case PageState::Unlocked:
               ps.tryMark(v);
               break;
            default:
               break; // skip
         };
      });
   }

   // 1. write dirty pages
   libaioInterface[workerThreadId].writePages(toWrite);
   writeCount += toWrite.size();

   // 2. try to lock clean page candidates
   toEvict.erase(std::remove_if(toEvict.begin(), toEvict.end(), [&](PID pid) {
      PageState& ps = getPageState(pid);
      u64 v = ps.stateAndVersion;
      return (PageState::getState(v) != PageState::Marked) || !ps.tryLockX(v);
   }), toEvict.end());

   // 3. try to upgrade lock for dirty page candidates
   for (auto& pid : toWrite) {
      PageState& ps = getPageState(pid);
      u64 v = ps.stateAndVersion;
      if ((PageState::getState(v) == 1) && ps.stateAndVersion.compare_exchange_weak(v, PageState::sameVersion(v, PageState::Locked)))
         toEvict.push_back(pid);
      else
         ps.unlockS();
   }

   // 4. remove from page table
   if (useExmap) {
      for (u64 i=0; i<toEvict.size(); i++) {
         exmapInterface[workerThreadId]->iov[i].page = toEvict[i];
         exmapInterface[workerThreadId]->iov[i].len = 1;
      }
      if (exmapAction(exmapfd, EXMAP_OP_FREE, toEvict.size()) < 0)
         die("ioctl: EXMAP_OP_FREE");
   } else if (!useTraditional) {
      for (u64& pid : toEvict)
         madvise(virtMem + pid, pageSize, MADV_DONTNEED);
   }

   // 5. remove from hash table and unlock
   for (u64& pid : toEvict) {
      if (useTraditional)
         releaseFrame(pid);
      bool succ = residentSet.remove(pid);
      assert(succ);
      getPageState(pid).unlockXEvicted();
   }

   physUsedCount -= toEvict.size();
}

//---------------------------------------------------------------------------

struct BTreeNode;

struct BTreeNodeHeader {
   static const unsigned underFullSize = (pageSize/2) + (pageSize/4);  // merge nodes more empty
   static const u64 noNeighbour = ~0ull;

   struct FenceKeySlot {
      u16 offset;
      u16 len;
   };

   bool dirty;
   union {
      PID upperInnerNode; // inner
      PID nextLeafNode = noNeighbour; // leaf
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

static unsigned min(unsigned a, unsigned b)
{
   return a < b ? a : b;
}

template <class T>
static T loadUnaligned(void* p)
{
   T x;
   memcpy(&x, p, sizeof(T));
   return x;
}

// Get order-preserving head of key (assuming little endian)
static u32 head(u8* key, unsigned keyLen)
{
   switch (keyLen) {
      case 0:
         return 0;
      case 1:
         return static_cast<u32>(key[0]) << 24;
      case 2:
         return static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16;
      case 3:
         return (static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16) | (static_cast<u32>(key[2]) << 8);
      default:
         return __builtin_bswap32(loadUnaligned<u32>(key));
   }
}

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
      u8 heap[pageSize - sizeof(BTreeNodeHeader)];                // grows from back
   };

   static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 4;

   BTreeNode(bool isLeaf) : BTreeNodeHeader(isLeaf) { dirty = true; }

   u8* ptr() { return reinterpret_cast<u8*>(this); }
   bool isInner() { return !isLeaf; }
   span<u8> getLowerFence() { return { ptr() + lowerFence.offset, lowerFence.len}; }
   span<u8> getUpperFence() { return { ptr() + upperFence.offset, upperFence.len}; }
   u8* getPrefix() { return ptr() + lowerFence.offset; } // any key on page is ok

   unsigned freeSpace() { return dataOffset - (reinterpret_cast<u8*>(slot + count) - ptr()); }
   unsigned freeSpaceAfterCompaction() { return pageSize - (reinterpret_cast<u8*>(slot + count) - ptr()) - spaceUsed; }

   bool hasSpaceFor(unsigned keyLen, unsigned payloadLen)
   {
      return spaceNeeded(keyLen, payloadLen) <= freeSpaceAfterCompaction();
   }

   u8* getKey(unsigned slotId) { return ptr() + slot[slotId].offset; }
   span<u8> getPayload(unsigned slotId) { return {ptr() + slot[slotId].offset + slot[slotId].keyLen, slot[slotId].payloadLen}; }

   PID getChild(unsigned slotId) { return loadUnaligned<PID>(getPayload(slotId).data()); }

   // How much space would inserting a new key of len "keyLen" require?
   unsigned spaceNeeded(unsigned keyLen, unsigned payloadLen) {
      return sizeof(Slot) + (keyLen - prefixLen) + payloadLen;
   }

   void makeHint()
   {
      unsigned dist = count / (hintCount + 1);
      for (unsigned i = 0; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void updateHint(unsigned slotId)
   {
      unsigned dist = count / (hintCount + 1);
      unsigned begin = 0;
      if ((count > hintCount * 2 + 1) && (((count - 1) / (hintCount + 1)) == dist) && ((slotId / dist) > 1))
         begin = (slotId / dist) - 1;
      for (unsigned i = begin; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void searchHint(u32 keyHead, u16& lowerOut, u16& upperOut)
   {
      if (count > hintCount * 2) {
         u16 dist = upperOut / (hintCount + 1);
         u16 pos, pos2;
         for (pos = 0; pos < hintCount; pos++)
            if (hint[pos] >= keyHead)
               break;
         for (pos2 = pos; pos2 < hintCount; pos2++)
            if (hint[pos2] != keyHead)
               break;
         lowerOut = pos * dist;
         if (pos2 < hintCount)
            upperOut = (pos2 + 1) * dist;
      }
   }

   // lower bound search, foundExactOut indicates if there is an exact match, returns slotId
   u16 lowerBound(span<u8> skey, bool& foundExactOut)
   {
      foundExactOut = false;

      // check prefix
      int cmp = memcmp(skey.data(), getPrefix(), min(skey.size(), prefixLen));
      if (cmp < 0) // key is less than prefix
         return 0;
      if (cmp > 0) // key is greater than prefix
         return count;
      if (skey.size() < prefixLen) // key is equal but shorter than prefix
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
         } else { // head is equal, check full key
            int cmp = memcmp(key, getKey(mid), min(keyLen, slot[mid].keyLen));
            if (cmp < 0) {
               upper = mid;
            } else if (cmp > 0) {
               lower = mid + 1;
            } else {
               if (keyLen < slot[mid].keyLen) { // key is shorter
                  upper = mid;
               } else if (keyLen > slot[mid].keyLen) { // key is longer
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

   // lowerBound wrapper ignoring exact match argument (for convenience)
   u16 lowerBound(span<u8> key)
   {
      bool ignore;
      return lowerBound(key, ignore);
   }

   // insert key/value pair
   void insertInPage(span<u8> key, span<u8> payload)
   {
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

   bool removeSlot(unsigned slotId)
   {
      spaceUsed -= slot[slotId].keyLen;
      spaceUsed -= slot[slotId].payloadLen;
      memmove(slot + slotId, slot + slotId + 1, sizeof(Slot) * (count - slotId - 1));
      count--;
      makeHint();
      return true;
   }

   bool removeInPage(span<u8> key)
   {
      bool found;
      unsigned slotId = lowerBound(key, found);
      if (!found)
         return false;
      return removeSlot(slotId);
   }

   void copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src) {
      u64 ofs = offsetof(BTreeNodeHeader, upperInnerNode);
      memcpy(reinterpret_cast<u8*>(dst)+ofs, reinterpret_cast<u8*>(src)+ofs, sizeof(BTreeNode)-ofs);
   }

   void compactify()
   {
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

   // merge right node into this node
   bool mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right)
   {
      if (!isLeaf)
         // TODO: implement inner merge
         return true;

      assert(right->isLeaf);
      assert(parent->isInner());
      BTreeNode tmp(isLeaf);
      tmp.setFences(getLowerFence(), right->getUpperFence());
      unsigned leftGrow = (prefixLen - tmp.prefixLen) * count;
      unsigned rightGrow = (right->prefixLen - tmp.prefixLen) * right->count;
      unsigned spaceUpperBound =
         spaceUsed + right->spaceUsed + (reinterpret_cast<u8*>(slot + count + right->count) - ptr()) + leftGrow + rightGrow;
      if (spaceUpperBound > pageSize)
         return false;
      copyKeyValueRange(&tmp, 0, 0, count);
      right->copyKeyValueRange(&tmp, count, 0, right->count);
      PID pid = bm.toPID(this);
      memcpy(parent->getPayload(slotId+1).data(), &pid, sizeof(PID));
      parent->removeSlot(slotId);
      tmp.makeHint();
      tmp.nextLeafNode = right->nextLeafNode;

      copyNode(this, &tmp);
      return true;
   }

   // store key/value pair at slotId
   void storeKeyValue(u16 slotId, span<u8> skey, span<u8> payload)
   {
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

   void copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount)
   {
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
         for (unsigned i = 0; i < srcCount; i++)
            copyKeyValue(srcSlot + i, dst, dstSlot + i);
      }
      dst->count += srcCount;
      assert((dst->ptr() + dst->dataOffset) >= reinterpret_cast<u8*>(dst->slot + dst->count));
   }

   void copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot)
   {
      unsigned fullLen = slot[srcSlot].keyLen + prefixLen;
      u8 key[fullLen];
      memcpy(key, getPrefix(), prefixLen);
      memcpy(key+prefixLen, getKey(srcSlot), slot[srcSlot].keyLen);
      dst->storeKeyValue(dstSlot, {key, fullLen}, getPayload(srcSlot));
   }

   void insertFence(FenceKeySlot& fk, span<u8> key)
   {
      assert(freeSpace() >= key.size());
      dataOffset -= key.size();
      spaceUsed += key.size();
      fk.offset = dataOffset;
      fk.len = key.size();
      memcpy(ptr() + dataOffset, key.data(), key.size());
   }

   void setFences(span<u8> lower, span<u8> upper)
   {
      insertFence(lowerFence, lower);
      insertFence(upperFence, upper);
      for (prefixLen = 0; (prefixLen < min(lower.size(), upper.size())) && (lower[prefixLen] == upper[prefixLen]); prefixLen++)
         ;
   }

   void splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep)
   {
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

   struct SeparatorInfo {
      unsigned len;      // len of new separator
      unsigned slot;     // slot at which we split
      bool isTruncated;  // if true, we truncate the separator taking len bytes from slot+1
   };

   unsigned commonPrefix(unsigned slotA, unsigned slotB)
   {
      assert(slotA < count);
      unsigned limit = min(slot[slotA].keyLen, slot[slotB].keyLen);
      u8 *a = getKey(slotA), *b = getKey(slotB);
      unsigned i;
      for (i = 0; i < limit; i++)
         if (a[i] != b[i])
            break;
      return i;
   }

   SeparatorInfo findSeparator(bool splitOrdered)
   {
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
            for (bestSlot = lower + 1; (bestSlot < upper) && (commonPrefix(bestSlot, 0) == bestPrefixLen); bestSlot++)
               ;
      } else {
         bestSlot = (count-1) / 2;
      }


      // try to truncate separator
      unsigned common = commonPrefix(bestSlot, bestSlot + 1);
      if ((bestSlot + 1 < count) && (slot[bestSlot].keyLen > common) && (slot[bestSlot + 1].keyLen > (common + 1)))
         return SeparatorInfo{prefixLen + common + 1, bestSlot, true};

      return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[bestSlot].keyLen), bestSlot, false};
   }

   void getSep(u8* sepKeyOut, SeparatorInfo info)
   {
      memcpy(sepKeyOut, getPrefix(), prefixLen);
      memcpy(sepKeyOut + prefixLen, getKey(info.slot + info.isTruncated), info.len - prefixLen);
   }

   PID lookupInner(span<u8> key)
   {
      unsigned pos = lowerBound(key);
      if (pos == count)
         return upperInnerNode;
      return getChild(pos);
   }
};

static_assert(sizeof(BTreeNode) == pageSize, "btree node size problem");

static const u64 metadataPageId = 0;

struct MetaDataPage {
   bool dirty;
   PID roots[(pageSize-8)/8];

   PID getRoot(unsigned slot) { return roots[slot]; }
};

struct BTree {
   private:

   void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen);
   void ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen);

   public:
   unsigned slotId;
   atomic<bool> splitOrdered;

   BTree();
   ~BTree();

   GuardO<BTreeNode> findLeafO(span<u8> key) {
      GuardO<MetaDataPage> meta(metadataPageId);
      GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
      meta.release();

      while (node->isInner())
         node = GuardO<BTreeNode>(node->lookupInner(key), node);
      return node;
   }

   // point lookup, returns payload len on success, or -1 on failure
   int lookup(span<u8> key, u8* payloadOut, unsigned payloadOutSize) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return -1;

            // key found, copy payload
            memcpy(payloadOut, node->getPayload(pos).data(), min(node->slot[pos].payloadLen, payloadOutSize));
            return node->slot[pos].payloadLen;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   bool lookup(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            // key found
            fn(node->getPayload(pos));
            return true;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   void insert(span<u8> key, span<u8> payload);
   bool remove(span<u8> key);

   template<class Fn>
   bool updateInPlace(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            {
               GuardX<BTreeNode> nodeLocked(move(node));
               fn(nodeLocked->getPayload(pos));
               return true;
            }
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   GuardS<BTreeNode> findLeafS(span<u8> key) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<MetaDataPage> meta(metadataPageId);
            GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
            meta.release();

            while (node->isInner())
               node = GuardO<BTreeNode>(node->lookupInner(key), node);

            return GuardS<BTreeNode>(move(node));
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   void scanAsc(span<u8> key, Fn fn) {
      GuardS<BTreeNode> node = findLeafS(key);
      bool found;
      unsigned pos = node->lowerBound(key, found);
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         if (pos<node->count) {
            if (!fn(*node.ptr, pos))
               return;
            pos++;
         } else {
            if (!node->hasRightNeighbour())
               return;
            pos = 0;
            node = GuardS<BTreeNode>(node->nextLeafNode);
         }
      }
   }

   template<class Fn>
   void scanDesc(span<u8> key, Fn fn) {
      GuardS<BTreeNode> node = findLeafS(key);
      bool exactMatch;
      int pos = node->lowerBound(key, exactMatch);
      if (pos == node->count) {
         pos--;
         exactMatch = true; // XXX:
      }
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         while (pos>=0) {
            if (!fn(*node.ptr, pos, exactMatch))
               return;
            pos--;
         }
         if (!node->hasLowerFence())
            return;
         node = findLeafS(node->getLowerFence());
         pos = node->count-1;
      }
   }
};

static unsigned btreeslotcounter = 0;

BTree::BTree() : splitOrdered(false) {
   GuardX<MetaDataPage> page(metadataPageId);
   AllocGuard<BTreeNode> rootNode(true);
   slotId = btreeslotcounter++;
   page->roots[slotId] = rootNode.pid;
}

BTree::~BTree() {}

void BTree::trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen)
{

   // create new root if necessary
   if (parent.pid == metadataPageId) {
      MetaDataPage* metaData = reinterpret_cast<MetaDataPage*>(parent.ptr);
      AllocGuard<BTreeNode> newRoot(false);
      newRoot->upperInnerNode = node.pid;
      metaData->roots[slotId] = newRoot.pid;
      parent = move(newRoot);
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

void BTree::ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner() && (node.ptr != toSplit)) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }
         if (node.ptr == toSplit) {
            if (node->hasSpaceFor(key.size(), payloadLen))
               return; // someone else did split concurrently
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            trySplit(move(nodeLocked), move(parentLocked), key, payloadLen);
         }
         return;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

void BTree::insert(span<u8> key, span<u8> payload)
{
   assert((key.size()+payload.size()) <= BTreeNode::maxKVSize);

   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner()) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }

         if (node->hasSpaceFor(key.size(), payload.size())) {
            // only lock leaf
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->insertInPage(key, payload);
            return; // success
         }

         // lock parent and leaf
         GuardX<BTreeNode> parentLocked(move(parent));
         GuardX<BTreeNode> nodeLocked(move(node));
         trySplit(move(nodeLocked), move(parentLocked), key, payload.size());
         // insert hasn't happened, restart from root
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

bool BTree::remove(span<u8> key)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         u16 pos;
         while (node->isInner()) {
            pos = node->lowerBound(key);
            PID nextPage = (pos == node->count) ? node->upperInnerNode : node->getChild(pos);
            parent = move(node);
            node = GuardO<BTreeNode>(nextPage, parent);
         }

         bool found;
         unsigned slotId = node->lowerBound(key, found);
         if (!found)
            return false;

         unsigned sizeEntry = node->slot[slotId].keyLen + node->slot[slotId].payloadLen;
         if ((node->freeSpaceAfterCompaction()+sizeEntry >= BTreeNodeHeader::underFullSize) && (parent.pid != metadataPageId) && (parent->count >= 2) && ((pos + 1) < parent->count)) {
            // underfull
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            GuardX<BTreeNode> rightLocked(parentLocked->getChild(pos + 1));
            nodeLocked->removeSlot(slotId);
            if (rightLocked->freeSpaceAfterCompaction() >= (pageSize-BTreeNodeHeader::underFullSize)) {
               if (nodeLocked->mergeNodes(pos, parentLocked.ptr, rightLocked.ptr)) {
                  // XXX: should reuse page Id
               }
            }
         } else {
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->removeSlot(slotId);
         }
         return true;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

typedef u64 KeyType;

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

template <class Record>
struct vmcacheAdapter
{
   BTree tree;

   public:
   void scan(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, l}, [&](BTreeNode& node, unsigned slot) {
         memcpy(kk, node.getPrefix(), node.prefixLen);
         memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         typename Record::Key typedKey;
         Record::unfoldKey(kk, typedKey);
         return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
      });
   }
   // -------------------------------------------------------------------------------------
   void scanDesc(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      bool first = true;
      tree.scanDesc({k, l}, [&](BTreeNode& node, unsigned slot, bool exactMatch) {
         if (first) { // XXX: hack
            first = false;
            if (!exactMatch)
               return true;
         }
         memcpy(kk, node.getPrefix(), node.prefixLen);
         memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
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
   template<class Fn>
   void lookup1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      bool succ = tree.lookup({k, l}, [&](span<u8> payload) {
         fn(*reinterpret_cast<const Record*>(payload.data()));
      });
      assert(succ);
   }
   // -------------------------------------------------------------------------------------
   template<class Fn>
   void update1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      tree.updateInPlace({k, l}, [&](span<u8> payload) {
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
   Field lookupField(const typename Record::Key& key, Field Record::*f) {
      Field value;
      lookup1(key, [&](const Record& r) { value = r.*f; });
      return value;
   }

   u64 count() {
      u64 cnt = 0;
      tree.scanAsc({(u8*)nullptr, 0}, [&](BTreeNode& node, unsigned slot) { cnt++; return true; } );
      return cnt;
   }

   u64 countw(Integer w_id) {
      u8 k[sizeof(Integer)];
      fold(k, w_id);
      u64 cnt = 0;
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, sizeof(Integer)}, [&](BTreeNode& node, unsigned slot) {
         memcpy(kk, node.getPrefix(), node.prefixLen);
         memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         if (memcmp(k, kk, sizeof(Integer))!=0)
            return false;
         cnt++;
         return true;
      });
      return cnt;
   }
};

template<class Fn>
void parallel_for(uint64_t begin, uint64_t end, uint64_t nthreads, Fn fn) {
   std::vector<std::thread> threads;
   uint64_t n = end-begin;
   if (n<nthreads)
      nthreads = n;
   uint64_t perThread = n/nthreads;
   for (unsigned i=0; i<nthreads; i++) {
      threads.emplace_back([&,i]() {
         uint64_t b = (perThread*i) + begin;
         uint64_t e = (i==(nthreads-1)) ? end : (b+perThread);
         fn(i, b, e);
      });
   }
   for (auto& t : threads)
      t.join();
}

int main(int argc, char** argv) {
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
   double skewFactor = envOrDouble("SKEW", 0.0); // Zipfian skew factor, 0.0 = uniform


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
   const char* systemName = bm.useTraditional
      ? tradMode
      : (bm.useExmap ? "exmap" : "vmcache");

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
      
      *out << "ts,tx,rmb,wmb,system,threads,datasize,workload,batch,skew" << endl;
      u64 cnt = 0;
      u64 totalTx = 0;
      for (uint64_t i=0; i<runForSec; i++) {
         sleep(1);
         float rmb = (bm.readCount.exchange(0)*pageSize)/(1024.0*1024);
         float wmb = (bm.writeCount.exchange(0)*pageSize)/(1024.0*1024);
         u64 prog = txProgress.exchange(0);
         totalTx += prog;
         *out << cnt++ << "," << prog << "," << rmb << "," << wmb << "," << systemName << "," << nthreads << "," << n << "," << (isRndread?"rndread":"tpcc") << "," << bm.batch << "," << skewFactor << endl;
      }
      // Record average throughput
      double avgThroughput = (runForSec > 0) ? (double)totalTx / runForSec : 0.0;
      *out << "avg_throughput," << avgThroughput << ",0,0," << systemName << "," << nthreads << "," << n << "," << (isRndread?"rndread":"tpcc") << "," << bm.batch << "," << skewFactor << endl;
      keepRunning = false;
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

      {
         // insert
         parallel_for(0, n, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
            workerThreadId = worker;
            array<u8, 120> payload;
            // std::vector<u64> keys;
            // keys.reserve(end-begin);
            // for (u64 i=begin; i<end; i++)
            //    keys.push_back(i);
            // std::random_shuffle(keys.begin(), keys.end());
            for (u64 i=begin; i<end; i++) {
               union { u64 v1; u8 k1[sizeof(u64)]; };
               //v1 = __builtin_bswap64(keys[i-begin]);
               v1 = __builtin_bswap64(i);
               memcpy(payload.data(), k1, sizeof(u64));
               bt.insert({k1, sizeof(KeyType)}, payload);
            }
         });
      }
      cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;

      bm.readCount = 0;
      bm.writeCount = 0;
      thread statThread(statFn);

      parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         u64 cnt = 0;
         u64 start = rdtsc();

         std::unique_ptr<perf::Scope> perfScope;
         if (enablePerf)
            perfScope = std::make_unique<perf::Scope>();
         
         // Initialize key generator based on skew factor
         std::unique_ptr<ScrambledZipfGenerator> zipfGen;
         if (skewFactor > 0.0) {
            zipfGen = std::make_unique<ScrambledZipfGenerator>(0, n, skewFactor);
         }
         
         while (keepRunning.load()) {
            union { u64 v1; u8 k1[sizeof(u64)]; };
            
            // Generate key using either uniform random or Zipfian distribution
            if (skewFactor > 0.0) {
               v1 = __builtin_bswap64(zipfGen->rand());
            } else {
               v1 = __builtin_bswap64(RandomGenerator::getRand<u64>(0, n));
            }

            array<u8, 120> payload;
            bool succ = bt.lookup({k1, sizeof(u64)}, [&](span<u8> p) {
               memcpy(payload.data(), p.data(), p.size());
            });
            assert(succ);
            assert(memcmp(k1, payload.data(), sizeof(u64))==0);

            cnt++;
            u64 stop = rdtsc();
            if ((stop-start) > statDiff) {
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
      if (enablePerf)
         perf::Aggregator::instance().report(txTotal.load());
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

   TPCCWorkload<vmcacheAdapter> tpcc(warehouse, district, customer, customerwdl, history, neworder, order, order_wdc, orderline, item, stock, true, warehouseCount, true);

   {
      tpcc.loadItem();
      tpcc.loadWarehouse();

      parallel_for(1, warehouseCount+1, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         for (Integer w_id=begin; w_id<end; w_id++) {
            tpcc.loadStock(w_id);
            tpcc.loadDistrinct(w_id);
            for (Integer d_id = 1; d_id <= 10; d_id++) {
               tpcc.loadCustomer(w_id, d_id);
               tpcc.loadOrders(w_id, d_id);
            }
         }
      });
   }
   cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;

   bm.readCount = 0;
   bm.writeCount = 0;
   thread statThread(statFn);

   parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
      workerThreadId = worker;
      u64 cnt = 0;
      u64 start = rdtsc();
      while (keepRunning.load()) {
         int w_id = tpcc.urand(1, warehouseCount); // wh crossing
         tpcc.tx(w_id);
         cnt++;
         u64 stop = rdtsc();
         if ((stop-start) > statDiff) {
            txProgress += cnt;
            start = stop;
            cnt = 0;
         }
      }
      txProgress += cnt;
   });

   statThread.join();
   cerr << "space: " << (bm.allocCount.load()*pageSize)/(float)bm.gb << " GB " << endl;

   return 0;
}
