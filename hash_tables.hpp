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

// A lock-free hash map that supports non-concurrent resizing via reserve().
template <typename K, typename V>
class LockFreeHashMap : public PartitionedMap<K, V> {
private:
    // The internal representation for each slot in the hash table.
    // We use a pointer to the slots array to allow resizing.
    struct Slot {
        std::atomic<uint8_t> tag;
        K key;
        V value;
    };

public:
    // --- Tag constants for slot state management ---
    static constexpr uint8_t EMPTY_TAG = 0xFF;
    static constexpr uint8_t LOCKED_TAG = 0xFE;
    static constexpr uint8_t DELETED_TAG = 0xFD;

private:
    // These must be atomic because they are read by concurrent threads,
    // even though they are only written by the single-threaded reserve().
    size_t capacity_;
    size_t capacity_mask_;
    // The pointer to the slot array itself doesn't need to be atomic if we assume
    // that no operations are running *during* the swap in reserve(). The guarantee
    // that reserve() is not called concurrently with other operations is key.
    std::vector<Slot> slots_;

    Hasher<K> hasher_;

    // Calculates the next power of 2 for a given number.
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
    
    // Non-atomic insert helper used only during rehash.
    void insertInto(std::vector<Slot>& new_slots, size_t new_mask, const K& key, const V& value) {
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;
        
        size_t pos = hash & new_mask;
        
        while (true) {
            // During rehash, we know we are the only writer, so we can use relaxed memory order.
            uint8_t current_tag = new_slots[pos].tag.load(std::memory_order_relaxed);
            if (current_tag == EMPTY_TAG) {
                new_slots[pos].key = key;
                new_slots[pos].value = value;
                new_slots[pos].tag.store(tag, std::memory_order_relaxed);
                return;
            }
            pos = (pos + 1) & new_mask;
        }
    }

public:
    explicit LockFreeHashMap()
        : capacity_(0), capacity_mask_(0) {
    }

    LockFreeHashMap(const LockFreeHashMap&) = delete;
    LockFreeHashMap& operator=(const LockFreeHashMap&) = delete;

    // Resizes the hash map to accommodate at least `entries` elements.
    // IMPORTANT: This method is NOT thread-safe and must NOT be called
    // concurrently with any other method on this map.
    void reserve(size_t entries) override {
        size_t old_capacity = capacity_;
        // Target a load factor of ~50% to leave space for probing
        size_t new_capacity = nextPow2(entries * 2);
        if (new_capacity <= old_capacity) {
            return;
        }

        auto new_slots = std::vector<Slot>(new_capacity);
        for (size_t i = 0; i < new_capacity; ++i) {
            new_slots[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
        }
        madvise(new_slots.data(), new_capacity * sizeof(Slot), MADV_HUGEPAGE);
        size_t new_mask = new_capacity - 1;

        // Move elements from the old table to the new one
        if (!slots_.empty()) {
            for (size_t i = 0; i < old_capacity; ++i) {
                uint8_t tag = slots_[i].tag.load(std::memory_order_relaxed);
                if (tag < DELETED_TAG) { // If it's a valid, occupied slot
                    insertInto(new_slots, new_mask, slots_[i].key, slots_[i].value);
                }
            }
        }
        
        // Atomically update the state for other threads to see
        slots_ = std::move(new_slots);
        capacity_mask_ = new_mask;
        capacity_ = new_capacity;
    }

    void insertOrAssign(const K& key, const V& value) override {
        // Load capacity info at the start of the operation.
        // It's guaranteed not to change during this call because reserve() is not concurrent.
        size_t current_mask = capacity_mask_;
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & current_mask;
        const size_t initial_pos = pos;

        while (true) {
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);

            if (current_tag == tag && slots_[pos].key == key) {
                if (slots_[pos].tag.compare_exchange_strong(current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                    slots_[pos].value = value;
                    slots_[pos].tag.store(tag, std::memory_order_release);
                    return;
                }
                continue;
            }

            if (current_tag == EMPTY_TAG || current_tag == DELETED_TAG) {
                if (slots_[pos].tag.compare_exchange_strong(current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                    slots_[pos].key = key;
                    slots_[pos].value = value;
                    slots_[pos].tag.store(tag, std::memory_order_release);
                    return;
                }
                continue;
            }

            pos = (pos + 1) & current_mask;
            if (pos == initial_pos) {
                throw std::runtime_error("LockFreeHashMap is full. Consider calling reserve().");
            }
        }
    }

    bool tryGet(const K& key, V& value) const override {
        size_t current_mask = capacity_mask_;
        if (current_mask == 0 && capacity_ == 0) return false;

        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & current_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && slots_[pos].key == key) {
                value = slots_[pos].value;
                if (slots_[pos].tag.load(std::memory_order_acquire) == tag) {
                    return true;
                }
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & current_mask;

        } while (pos != initial_pos);
        
        return false;
    }

    bool erase(const K& key, V& value) override {
        size_t current_mask = capacity_mask_;
        if (current_mask == 0 && capacity_ == 0) return false;
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & current_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && slots_[pos].key == key) {
                if (slots_[pos].tag.compare_exchange_strong(current_tag, DELETED_TAG, std::memory_order_acq_rel)) {
                    value = slots_[pos].value;
                    return true;
                }
                continue;
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & current_mask;
        } while (pos != initial_pos);

        return false;
    }
};
