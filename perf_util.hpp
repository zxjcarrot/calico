#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace perf
{

constexpr std::size_t kNumEvents = 6;

struct EventConfig {
   perf_type_id type;
   std::uint64_t config;
   const char* name;
};

inline const std::array<EventConfig, kNumEvents>& eventConfigs()
{
   static const std::array<EventConfig, kNumEvents> events = {{
       {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
       {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
       {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "llc_refs"},
       {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "llc_misses"},
       {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "dtlb_refs"},
       {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "dtlb_misses"},
   }};
   return events;
}

class Aggregator {
public:
   static Aggregator& instance()
   {
      static Aggregator inst;
      return inst;
   }

   void add(const std::array<std::uint64_t, kNumEvents>& values)
   {
      for (std::size_t i = 0; i < kNumEvents; i++)
         totals[i].fetch_add(values[i], std::memory_order_relaxed);
   }

   void report(std::ostream& os, bool resetTotals = true)
   {
      std::array<std::uint64_t, kNumEvents> snapshot{};
      for (std::size_t i = 0; i < kNumEvents; i++)
         snapshot[i] = totals[i].load(std::memory_order_relaxed);

      os << "perf_summary";
      for (std::size_t i = 0; i < kNumEvents; i++) {
         os << ' ' << eventConfigs()[i].name << '=' << snapshot[i];
      }
      if (snapshot[0] && snapshot[1]) {
         double ipc = static_cast<double>(snapshot[0]) / static_cast<double>(snapshot[1]);
         os << " ipc=" << ipc;
      }
      if (snapshot[2]) {
         double miss_rate = snapshot[3] / static_cast<double>(snapshot[2]);
         os << " llc_miss_rate=" << miss_rate;
      }
      if (snapshot[4]) {
         double dtlb_miss_rate = snapshot[5] / static_cast<double>(snapshot[4]);
         os << " dtlb_miss_rate=" << dtlb_miss_rate;
      }
      os << std::endl;

      if (resetTotals)
         reset();
   }

   void reset()
   {
      for (auto& v : totals)
         v.store(0, std::memory_order_relaxed);
   }

private:
   Aggregator()
   {
      for (auto& v : totals)
         v.store(0, std::memory_order_relaxed);
   }

   std::array<std::atomic<std::uint64_t>, kNumEvents> totals;
};

class Scope
{
public:
   Scope() { enabled = openEvents(); }
   ~Scope()
   {
      if (!enabled)
         return;
      stop();
      std::array<std::uint64_t, kNumEvents> snapshot{};
      for (std::size_t i = 0; i < kNumEvents; i++) {
         std::uint64_t value = 0;
         if (read(fds[i], &value, sizeof(value)) == sizeof(value))
            snapshot[i] = value;
         ::close(fds[i]);
      }
      Aggregator::instance().add(snapshot);
   }

   Scope(const Scope&) = delete;
   Scope& operator=(const Scope&) = delete;

private:
   bool openEvents()
   {
      const auto& configs = eventConfigs();
      for (std::size_t i = 0; i < kNumEvents; i++) {
         perf_event_attr attr{};
         attr.type = configs[i].type;
         attr.size = sizeof(perf_event_attr);
         attr.config = configs[i].config;
         attr.disabled = 1;
         attr.exclude_kernel = 1;
         attr.exclude_hv = 1;
         attr.inherit = 0;
         attr.read_format = 0;

         int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
         if (fd == -1) {
            for (std::size_t j = 0; j < i; j++)
               ::close(fds[j]);
            return false;
         }
         fds[i] = fd;
      }
      for (auto fd : fds) {
         ioctl(fd, PERF_EVENT_IOC_RESET, 0);
         ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
      }
      return true;
   }

   void stop()
   {
      if (enabled)
         for (auto fd : fds)
            ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
   }

   bool enabled = false;
   std::array<int, kNumEvents> fds{};
};

}  // namespace perf

