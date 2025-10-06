#pragma once

#include <unordered_map>
#include <vector>
#include <array>
#include <shared_mutex>
#include <atomic>
#include <sys/mman.h>
#include <mutex>

// Forward declaration of Hasher template
template<typename K>
struct Hasher;

template <typename K, typename V>
class PartitionedMap {
public:
   virtual ~PartitionedMap() = default;
   virtual void reserve(size_t entries) = 0;
   virtual void insertOrAssign(const K& key, const V& value) = 0;
   virtual bool tryGet(const K& key, V& value) const = 0;
   virtual bool erase(const K& key, V& value) = 0;
};

template <typename K, typename V, size_t ShardCount = 64>
class ShardedUnorderedMap : public PartitionedMap<K, V> {
   static_assert((ShardCount & (ShardCount - 1)) == 0, "ShardCount must be a power of two");

   struct Shard {
      std::unordered_map<K, V, Hasher<K>> map;
      mutable std::shared_mutex mutex;
   };

   std::array<Shard, ShardCount> shards;
   Hasher<K> hasher;

   static constexpr size_t shardMask = ShardCount - 1;

   size_t shardFor(const K& key) const {
      return hasher(key) & shardMask;
   }

public:
   void reserve(size_t entries) override {
      size_t perShard = (entries + ShardCount - 1) / ShardCount;
      for (auto& shard : shards)
         shard.map.reserve(perShard);
   }

   void insertOrAssign(const K& key, const V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      shard.map[key] = value;
   }

   bool tryGet(const K& key, V& value) const override {
      auto& shard = shards[shardFor(key)];
      std::shared_lock lock(shard.mutex);
      auto it = shard.map.find(key);
      if (it == shard.map.end())
         return false;
      value = it->second;
      return true;
   }

   bool erase(const K& key, V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      auto it = shard.map.find(key);
      if (it == shard.map.end())
         return false;
      value = it->second;
      shard.map.erase(it);
      return true;
   }
};

// Single-threaded specialization with no mutex overhead
template <typename K, typename V>
class ShardedUnorderedMap<K, V, 1> : public PartitionedMap<K, V> {
   std::unordered_map<K, V, Hasher<K>> map;

public:
   void reserve(size_t entries) override {
      map.reserve(entries);
   }

   void insertOrAssign(const K& key, const V& value) override {
      map[key] = value;
   }

   bool tryGet(const K& key, V& value) const override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      return true;
   }

   bool erase(const K& key, V& value) override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      map.erase(it);
      return true;
   }
};

template <typename K, typename V>
struct OAEntry {
   enum class State : uint8_t { Empty, Occupied, Deleted };
   K key{};
   V value{};
   State state = State::Empty;
};

template <typename K, typename V, size_t ShardCount = 64>
class ShardedOpenAddressMap : public PartitionedMap<K, V> {
   static_assert((ShardCount & (ShardCount - 1)) == 0, "ShardCount must be a power of two");

   struct Shard {
      mutable std::shared_mutex mutex;
      std::vector<OAEntry<K, V>> table;
      size_t size = 0;
      Hasher<K> hasher;

      static size_t nextPow2(size_t v) {
         if (v < 8)
            return 8;
         v--;
         v |= v >> 1;
         v |= v >> 2;
         v |= v >> 4;
         v |= v >> 8;
         v |= v >> 16;
#if SIZE_MAX > UINT32_MAX
         v |= v >> 32;
#endif
         return v + 1;
      }

      void ensureCapacity() {
         if (table.empty())
            rehash(8);
         else if ((size + 1) * 10 > table.size() * 7)
            rehash(table.size() * 2);
      }

      void rehash(size_t newCap) {
         newCap = nextPow2(newCap);
         std::vector<OAEntry<K, V>> newTable(newCap);
         size_t oldSize = size;
         size = 0;
         for (auto& entry : table) {
            if (entry.state == OAEntry<K, V>::State::Occupied)
               insertInto(newTable, entry.key, entry.value, hasher);
         }
         table.swap(newTable);
         size = oldSize;
         madvise(table.data(), table.size() * sizeof(OAEntry<K, V>), MADV_HUGEPAGE);
      }

      static void insertInto(std::vector<OAEntry<K, V>>& tbl, const K& key, const V& value, const Hasher<K>& hasher) {
         size_t mask = tbl.size() - 1;
         size_t idx = hasher(key) & mask;
         size_t firstDeleted = tbl.size();
         for (;;) {
            auto& slot = tbl[idx];
            if (slot.state == OAEntry<K, V>::State::Empty) {
               size_t target = (firstDeleted != tbl.size()) ? firstDeleted : idx;
               auto& dest = tbl[target];
               dest.key = key;
               dest.value = value;
               dest.state = OAEntry<K, V>::State::Occupied;
               return;
            } else if (slot.state == OAEntry<K, V>::State::Deleted) {
               if (firstDeleted == tbl.size())
                  firstDeleted = idx;
            } else if (slot.key == key) {
               slot.value = value;
               return;
            }
            idx = (idx + 1) & mask;
         }
      }

      void insertOrAssign(const K& key, const V& value) {
         ensureCapacity();
         size_t mask = table.size() - 1;
         size_t idx = hasher(key) & mask;
         size_t firstDeleted = table.size();
         for (;;) {
            auto& slot = table[idx];
            if (slot.state == OAEntry<K, V>::State::Empty) {
               size_t target = (firstDeleted != table.size()) ? firstDeleted : idx;
               auto& dest = table[target];
               dest.key = key;
               dest.value = value;
               if (dest.state != OAEntry<K, V>::State::Occupied) {
                  dest.state = OAEntry<K, V>::State::Occupied;
                  size++;
               }
               return;
            } else if (slot.state == OAEntry<K, V>::State::Deleted) {
               if (firstDeleted == table.size())
                  firstDeleted = idx;
            } else if (slot.key == key) {
               slot.value = value;
               return;
            }
            idx = (idx + 1) & mask;
            if (idx == (hasher(key) & mask)) {
               rehash(table.size() * 2);
               insertOrAssign(key, value);
               return;
            }
         }
      }

      bool tryGet(const K& key, V& value, size_t hash) const {
         if (table.empty())
            return false;
         size_t mask = table.size() - 1;
         size_t idx = hash & mask;
         for (;;) {
            const auto& slot = table[idx];
            if (slot.state == OAEntry<K, V>::State::Empty)
               return false;
            if (slot.state == OAEntry<K, V>::State::Occupied && slot.key == key) {
               value = slot.value;
               return true;
            }
            idx = (idx + 1) & mask;
         }
      }

      bool tryGet(const K& key, V& value) const {
         return tryGet(key, value, hasher(key));
      }

      bool erase(const K& key, V& value) {
         if (table.empty())
            return false;
         size_t mask = table.size() - 1;
         size_t idx = hasher(key) & mask;
         for (;;) {
            auto& slot = table[idx];
            if (slot.state == OAEntry<K, V>::State::Empty)
               return false;
            if (slot.state == OAEntry<K, V>::State::Occupied && slot.key == key) {
               value = slot.value;
               slot.state = OAEntry<K, V>::State::Deleted;
               size--;
               return true;
            }
            idx = (idx + 1) & mask;
         }
      }
   };

   std::array<Shard, ShardCount> shards;
   Hasher<K> hasher;

   static constexpr size_t shardMask = ShardCount - 1;

   size_t shardFor(const K& key) const {
      return hasher(key) & shardMask;
   }

public:
   void reserve(size_t entries) override {
      size_t perShard = (entries + ShardCount - 1) / ShardCount;
      for (auto& shard : shards) {
         std::unique_lock lock(shard.mutex);
         shard.rehash(Shard::nextPow2(perShard * 2));
      }
   }

   void insertOrAssign(const K& key, const V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      shard.insertOrAssign(key, value);
   }

   bool tryGet(const K& key, V& value) const override {
      size_t hash = hasher(key);
      auto& shard = shards[hash & shardMask];
      std::shared_lock lock(shard.mutex);
      return shard.tryGet(key, value, hash);
   }

   bool erase(const K& key, V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      return shard.erase(key, value);
   }
};

// Single-threaded specialization with no mutex overhead
template <typename K, typename V>
class ShardedOpenAddressMap<K, V, 1> : public PartitionedMap<K, V> {
   std::vector<OAEntry<K, V>> table;
   size_t size = 0;
   Hasher<K> hasher;

   static size_t nextPow2(size_t v) {
      if (v < 8)
         return 8;
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
#if SIZE_MAX > UINT32_MAX
      v |= v >> 32;
#endif
      return v + 1;
   }

   void ensureCapacity() {
      if (table.empty())
         rehash(8);
      else if ((size + 1) * 10 > table.size() * 7)
         rehash(table.size() * 2);
   }

   void rehash(size_t newCap) {
      newCap = nextPow2(newCap);
      std::vector<OAEntry<K, V>> newTable(newCap);
      size_t oldSize = size;
      size = 0;
      for (auto& entry : table) {
         if (entry.state == OAEntry<K, V>::State::Occupied)
            insertInto(newTable, entry.key, entry.value);
      }
      table.swap(newTable);
      size = oldSize;
      madvise(table.data(), table.size() * sizeof(OAEntry<K, V>), MADV_HUGEPAGE);
   }

   void insertInto(std::vector<OAEntry<K, V>>& tbl, const K& key, const V& value) {
      size_t mask = tbl.size() - 1;
      size_t idx = hasher(key) & mask;
      size_t firstDeleted = tbl.size();
      for (;;) {
         auto& slot = tbl[idx];
         if (slot.state == OAEntry<K, V>::State::Empty) {
            size_t target = (firstDeleted != tbl.size()) ? firstDeleted : idx;
            auto& dest = tbl[target];
            dest.key = key;
            dest.value = value;
            dest.state = OAEntry<K, V>::State::Occupied;
            return;
         } else if (slot.state == OAEntry<K, V>::State::Deleted) {
            if (firstDeleted == tbl.size())
               firstDeleted = idx;
         } else if (slot.key == key) {
            slot.value = value;
            return;
         }
         idx = (idx + 1) & mask;
      }
   }

public:
   void reserve(size_t entries) override {
      rehash(nextPow2(entries * 2));
   }

   void insertOrAssign(const K& key, const V& value) override {
      ensureCapacity();
      size_t mask = table.size() - 1;
      size_t idx = hasher(key) & mask;
      size_t firstDeleted = table.size();
      for (;;) {
         auto& slot = table[idx];
         if (slot.state == OAEntry<K, V>::State::Empty) {
            size_t target = (firstDeleted != table.size()) ? firstDeleted : idx;
            auto& dest = table[target];
            dest.key = key;
            dest.value = value;
            if (dest.state != OAEntry<K, V>::State::Occupied) {
               dest.state = OAEntry<K, V>::State::Occupied;
               size++;
            }
            return;
         } else if (slot.state == OAEntry<K, V>::State::Deleted) {
            if (firstDeleted == table.size())
               firstDeleted = idx;
         } else if (slot.key == key) {
            slot.value = value;
            return;
         }
         idx = (idx + 1) & mask;
         if (idx == (hasher(key) & mask)) {
            rehash(table.size() * 2);
            insertOrAssign(key, value);
            return;
         }
      }
   }

   bool tryGet(const K& key, V& value) const override {
      if (table.empty())
         return false;
      size_t mask = table.size() - 1;
      size_t idx = hasher(key) & mask;
      for (;;) {
         const auto& slot = table[idx];
         if (slot.state == OAEntry<K, V>::State::Empty)
            return false;
         if (slot.state == OAEntry<K, V>::State::Occupied && slot.key == key) {
            value = slot.value;
            return true;
         }
         idx = (idx + 1) & mask;
      }
   }

   bool erase(const K& key, V& value) override {
      if (table.empty())
         return false;
      size_t mask = table.size() - 1;
      size_t idx = hasher(key) & mask;
      for (;;) {
         auto& slot = table[idx];
         if (slot.state == OAEntry<K, V>::State::Empty)
            return false;
         if (slot.state == OAEntry<K, V>::State::Occupied && slot.key == key) {
            value = slot.value;
            slot.state = OAEntry<K, V>::State::Deleted;
            size--;
            return true;
         }
         idx = (idx + 1) & mask;
      }
   }
};

// Epoch-Based Reclamation (EBR) for safe memory management in lock-free data structures
// Implementation follows Xenium's approach: https://github.com/mpoeter/xenium
class EpochManager {
public:
    struct alignas(64) ThreadState {
        std::atomic<bool> is_in_critical_region{false};
        std::atomic<uint64_t> local_epoch{0};
        std::vector<void*> retired[3];  // Circular buffer indexed by epoch % 3
        // No padding needed - alignment is sufficient
    };
    
private:
    std::atomic<uint64_t> global_epoch_{1};
    std::vector<ThreadState*> thread_states_;
    std::mutex registration_mutex_;
    
public:
    class Guard {
        ThreadState* state_;
        uint64_t epoch_;
    public:
        Guard(ThreadState* state, uint64_t epoch) : state_(state), epoch_(epoch) {
            // Mark thread as in critical region
            state_->is_in_critical_region.store(true, std::memory_order_relaxed);
            
            // (1) - seq_cst fence enforces total order and synchronizes-with acquire fence (3)
            std::atomic_thread_fence(std::memory_order_seq_cst);
            
            // Store the epoch (relaxed is safe due to fence)
            state_->local_epoch.store(epoch, std::memory_order_relaxed);
        }
        
        ~Guard() {
            // (2) - release-store synchronizes-with acquire fence (3)
            state_->is_in_critical_region.store(false, std::memory_order_release);
        }
        
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };
    
    ~EpochManager() {
        for (auto* state : thread_states_) {
            for (auto& retired_list : state->retired) {
                for (auto* ptr : retired_list) {
                    ::operator delete(ptr);
                }
            }
            delete state;
        }
    }
    
    ThreadState* registerThread() {
        std::lock_guard<std::mutex> lock(registration_mutex_);
        auto* state = new ThreadState();
        thread_states_.push_back(state);
        return state;
    }
    
    Guard enterCriticalSection(ThreadState* state) {
        // Load global epoch with acquire to synchronize-with release CAS (4)
        uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
        return Guard(state, epoch);
    }
    
    void retire(ThreadState* state, void* ptr) {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_relaxed);
        state->retired[current_epoch % 3].push_back(ptr);
        tryAdvanceEpoch(state);
    }
    
private:
    void tryAdvanceEpoch(ThreadState* state) {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_relaxed);
        
        // Check if all threads have moved past the epoch from 2 epochs ago
        bool can_advance = true;
        uint64_t min_active_epoch = current_epoch;
        
        {
            std::lock_guard<std::mutex> lock(registration_mutex_);
            
            // (3) - acquire fence synchronizes-with seq_cst fence (1) and release-store (2)
            std::atomic_thread_fence(std::memory_order_acquire);
            
            for (auto* ts : thread_states_) {
                // Load operations can be relaxed due to acquire fence above
                bool in_critical = ts->is_in_critical_region.load(std::memory_order_relaxed);
                uint64_t local = ts->local_epoch.load(std::memory_order_relaxed);
                
                if (in_critical && local != current_epoch) {
                    can_advance = false;
                    break;
                }
                
                if (in_critical) {
                    min_active_epoch = std::min(min_active_epoch, local);
                }
            }
        }
        
        if (can_advance && min_active_epoch == current_epoch) {
            // (4) - release CAS synchronizes-with acquire load in enterCriticalSection
            if (global_epoch_.compare_exchange_strong(current_epoch, current_epoch + 1, 
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {
                // Safe to free memory from 2 epochs ago
                size_t old_epoch_idx = current_epoch % 3;
                for (auto* ptr : state->retired[old_epoch_idx]) {
                    ::operator delete(ptr);
                }
                state->retired[old_epoch_idx].clear();
            }
        }
    }
};

// A lock-free hash map with automatic resizing and epoch-based memory reclamation
template <typename K, typename V>
class LockFreeHashMap : public PartitionedMap<K, V> {
private:
    struct Slot {
        std::atomic<uint8_t> tag;
        K key;
        V value;
    };
    
    struct Table {
        Slot* slots;
        size_t capacity;
        size_t capacity_mask;
        std::atomic<size_t> size{0};
        
        Table(size_t cap) : capacity(cap), capacity_mask(cap - 1) {
            slots = static_cast<Slot*>(::operator new(cap * sizeof(Slot)));
            for (size_t i = 0; i < cap; ++i) {
                new (&slots[i]) Slot();
                slots[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
            }
            madvise(slots, cap * sizeof(Slot), MADV_HUGEPAGE);
        }
        
        ~Table() {
            if (slots) {
                for (size_t i = 0; i < capacity; ++i) {
                    slots[i].~Slot();
                }
                ::operator delete(slots);
            }
        }
    };

public:
    static constexpr uint8_t EMPTY_TAG = 0xFF;
    static constexpr uint8_t LOCKED_TAG = 0xFE;
    static constexpr uint8_t DELETED_TAG = 0xFD;

private:
    std::atomic<Table*> current_table_;
    std::atomic<Table*> next_table_{nullptr};
    std::atomic<bool> resizing_{false};
    std::atomic<size_t> resize_progress_{0};
    
    EpochManager epoch_manager_;
    Hasher<K> hasher_;
    
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;
    
    // Thread-local state for automatic registration
    struct ThreadLocalState {
        EpochManager::ThreadState* ebr_state = nullptr;
        LockFreeHashMap* map = nullptr;
        
        ~ThreadLocalState() {
            // Cleanup happens in EpochManager destructor
        }
    };
    
    EpochManager::ThreadState* getThreadState() {
        // Use function-local static thread_local to avoid guard name collision
        static thread_local ThreadLocalState tls_{};
        if (tls_.ebr_state == nullptr || tls_.map != this) {
            tls_.ebr_state = epoch_manager_.registerThread();
            tls_.map = this;
        }
        return tls_.ebr_state;
    }

    static size_t nextPow2(size_t n) {
        if (n == 0) return 1;
        #if defined(__GNUC__) || defined(__clang__)
            return 1UL << (64 - __builtin_clzll(n - 1));
        #else
            size_t p = 1;
            while (p < n) p <<= 1;
            return p;
        #endif
    }
    
    // Insert into new table during migration (lock-free)
    void insertIntoNewTable(Table* new_table, const K& key, const V& value) {
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;
        
        size_t pos = hash & new_table->capacity_mask;
        const size_t initial_pos = pos;
        
        while (true) {
            uint8_t current_tag = new_table->slots[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == EMPTY_TAG || current_tag == DELETED_TAG) {
                if (new_table->slots[pos].tag.compare_exchange_strong(
                        current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                    new_table->slots[pos].key = key;
                    new_table->slots[pos].value = value;
                    new_table->slots[pos].tag.store(tag, std::memory_order_release);
                    return;
                }
                continue;
            }
            
            // Check if key already exists (from another migration thread)
            if (current_tag == tag) {
                // Lock the slot to safely check the key
                if (new_table->slots[pos].tag.compare_exchange_strong(
                        current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                    bool is_same_key = (new_table->slots[pos].key == key);
                    // Restore the original tag
                    new_table->slots[pos].tag.store(tag, std::memory_order_release);
                    if (is_same_key) {
                        return;  // Already migrated
                    }
                    // Different key with same tag, continue probing
                } else {
                    // CAS failed, retry this slot
                    continue;
                }
            }
            
            pos = (pos + 1) & new_table->capacity_mask;
            if (pos == initial_pos) {
                throw std::runtime_error("New table is full during migration");
            }
        }
    }
    
    // Help migrate buckets from old to new table
    void helpMigrate() {
        Table* old_table = current_table_.load(std::memory_order_acquire);
        Table* new_table = next_table_.load(std::memory_order_acquire);
        
        if (!new_table || !resizing_.load(std::memory_order_acquire)) {
            return;
        }
        
        // Each thread claims buckets to migrate
        size_t batch_size = 64;  // Migrate in batches for better cache locality
        size_t start_idx = resize_progress_.fetch_add(batch_size, std::memory_order_relaxed);
        size_t end_idx = std::min(start_idx + batch_size, old_table->capacity);
        
        if (start_idx >= old_table->capacity) {
            return;  // All buckets claimed
        }
        
        // Migrate claimed buckets
        for (size_t i = start_idx; i < end_idx; ++i) {
            uint8_t tag = old_table->slots[i].tag.load(std::memory_order_acquire);
            // Skip locked, deleted, or empty slots
            if (tag >= DELETED_TAG) {
                continue;
            }
            // Read key and value, then double-check tag wasn't modified
            K key = old_table->slots[i].key;
            V value = old_table->slots[i].value;
            // Verify slot wasn't modified during read
            if (old_table->slots[i].tag.load(std::memory_order_acquire) == tag) {
                insertIntoNewTable(new_table, key, value);
            }
            // If tag changed, slot was modified - skip it, will be handled by insertOrAssign
        }
        
        // Check if migration is complete
        if (resize_progress_.load(std::memory_order_acquire) >= old_table->capacity) {
            // Try to finalize migration
            if (current_table_.compare_exchange_strong(old_table, new_table, std::memory_order_acq_rel)) {
                resizing_.store(false, std::memory_order_release);
                next_table_.store(nullptr, std::memory_order_release);
                
                // Retire old table for later reclamation
                auto* state = getThreadState();
                epoch_manager_.retire(state, old_table);
            }
        }
    }
    
    // Try to start a resize operation
    void tryResize(Table* table) {
        if (resizing_.load(std::memory_order_acquire)) {
            helpMigrate();  // Already resizing, help out
            return;
        }
        
        size_t current_size = table->size.load(std::memory_order_relaxed);
        if (current_size < table->capacity * LOAD_FACTOR_THRESHOLD) {
            return;  // No resize needed
        }
        
        bool expected = false;
        if (!resizing_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            helpMigrate();  // Another thread started resize
            return;
        }
        
        // We won the race to start resizing
        size_t new_capacity = nextPow2(table->capacity * 2);
        Table* new_table = new Table(new_capacity);
        
        next_table_.store(new_table, std::memory_order_release);
        resize_progress_.store(0, std::memory_order_release);
        
        // Help with migration
        helpMigrate();
    }

public:
    explicit LockFreeHashMap(size_t initial_capacity = 8192) {  // Start smaller, let it resize properly
        size_t capacity = nextPow2(initial_capacity);
        current_table_.store(new Table(capacity), std::memory_order_relaxed);
    }

    ~LockFreeHashMap() {
        Table* table = current_table_.load(std::memory_order_acquire);
        delete table;
        
        Table* next = next_table_.load(std::memory_order_acquire);
        if (next) {
            delete next;
        }
    }

    LockFreeHashMap(const LockFreeHashMap&) = delete;
    LockFreeHashMap& operator=(const LockFreeHashMap&) = delete;

    // Pre-allocate capacity (thread-safe, triggers migration if needed)
    void reserve(size_t entries) override {
        Table* table = current_table_.load(std::memory_order_acquire);
        size_t target_capacity = nextPow2(entries * 2);
        
        if (table->capacity >= target_capacity) {
            return;  // Already large enough
        }
        
        // Trigger resize by artificially increasing size
        size_t old_size = table->size.load(std::memory_order_relaxed);
        table->size.store(table->capacity, std::memory_order_relaxed);
        tryResize(table);
        
        // Wait for resize to complete
        while (resizing_.load(std::memory_order_acquire)) {
            helpMigrate();
        }
        
        // Restore actual size
        Table* new_table = current_table_.load(std::memory_order_acquire);
        new_table->size.store(old_size, std::memory_order_relaxed);
    }

    void insertOrAssign(const K& key, const V& value) override {
        auto* state = getThreadState();
        auto guard = epoch_manager_.enterCriticalSection(state);
        
        Table* table = current_table_.load(std::memory_order_acquire);
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        bool is_new_insert = false;

        while (true) {
            size_t pos = hash & table->capacity_mask;
            const size_t initial_pos = pos;
            
            while (true) {
                uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);

                if (current_tag == tag && table->slots[pos].key == key) {
                    if (table->slots[pos].tag.compare_exchange_strong(
                            current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                        table->slots[pos].value = value;
                        table->slots[pos].tag.store(tag, std::memory_order_release);
                        return;
                    }
                    continue;
                }

                if (current_tag == EMPTY_TAG || current_tag == DELETED_TAG) {
                    if (table->slots[pos].tag.compare_exchange_strong(
                            current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                        table->slots[pos].key = key;
                        table->slots[pos].value = value;
                        table->slots[pos].tag.store(tag, std::memory_order_release);
                        
                        if (current_tag == EMPTY_TAG) {
                            is_new_insert = true;
                        }
                        goto insertion_complete;
                    }
                    continue;
                }

                pos = (pos + 1) & table->capacity_mask;
                if (pos == initial_pos) {
                    // Table is full, trigger resize and wait for completion
                    tryResize(table);
                    
                    // Wait for resize to complete by helping with migration
                    while (resizing_.load(std::memory_order_acquire)) {
                        helpMigrate();
                    }
                    
                    // Double-check resizing flag and reload table to avoid stale pointer
                    // (another resize could have started and completed)
                    while (resizing_.load(std::memory_order_acquire)) {
                        helpMigrate();
                    }
                    table = current_table_.load(std::memory_order_acquire);
                    // Break inner loop to restart with new table and reset initial_pos
                    break;
                }
            }
        }
        
insertion_complete:
        
        // Update size and check if resize is needed
        if (is_new_insert) {
            size_t new_size = table->size.fetch_add(1, std::memory_order_relaxed) + 1;
            if (new_size > table->capacity * LOAD_FACTOR_THRESHOLD) {
                tryResize(table);
            }
        }
    }

    bool tryGet(const K& key, V& value) const override {
        // Note: const_cast is safe here as we're only modifying thread_local state
        auto* state = const_cast<LockFreeHashMap*>(this)->getThreadState();
        auto guard = const_cast<LockFreeHashMap*>(this)->epoch_manager_.enterCriticalSection(state);
        
        // Help with any ongoing migration first
        const_cast<LockFreeHashMap*>(this)->helpMigrate();
        
        // During migration, new inserts go to new table, so check new table first
        Table* new_table = const_cast<LockFreeHashMap*>(this)->next_table_.load(std::memory_order_acquire);
        if (new_table != nullptr) {
            // Migration in progress - search new table first
            if (tryGetFromTable(new_table, key, value)) {
                return true;
            }
        }
        
        // Search current (or old) table
        Table* table = current_table_.load(std::memory_order_acquire);
        return tryGetFromTable(table, key, value);
    }

private:
    bool tryGetFromTable(Table* table, const K& key, V& value) const {
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & table->capacity_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && table->slots[pos].key == key) {
                value = table->slots[pos].value;
                // Double-check tag to ensure slot wasn't modified during read
                if (table->slots[pos].tag.load(std::memory_order_acquire) == tag) {
                    return true;
                }
                // Tag changed (slot was being modified), continue probing
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & table->capacity_mask;

        } while (pos != initial_pos);
        
        return false;
    }

public:

    bool erase(const K& key, V& value) override {
        auto* state = getThreadState();
        auto guard = epoch_manager_.enterCriticalSection(state);
        
        // Help with any ongoing migration first
        helpMigrate();
        
        Table* table = current_table_.load(std::memory_order_acquire);
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & table->capacity_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && table->slots[pos].key == key) {
                if (table->slots[pos].tag.compare_exchange_strong(
                        current_tag, DELETED_TAG, std::memory_order_acq_rel)) {
                    value = table->slots[pos].value;
                    table->size.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
                continue;
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & table->capacity_mask;
        } while (pos != initial_pos);

        return false;
    }
};
