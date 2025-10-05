#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <errno.h>
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

   void setAvailability(const std::array<bool, kNumEvents>& avail)
   {
      bool anyNew = false;
      for (std::size_t i = 0; i < kNumEvents; i++) {
         bool prev = availability[i].load(std::memory_order_relaxed);
         if (!prev && avail[i]) {
            availability[i].store(true, std::memory_order_relaxed);
            anyNew = true;
         }
      }
      // if (anyNew)
      //    printConfiguredEvents();
   }

   void add(const std::array<std::uint64_t, kNumEvents>& values)
   {
      for (std::size_t i = 0; i < kNumEvents; i++)
         if (availability[i].load(std::memory_order_relaxed))
            totals[i].fetch_add(values[i], std::memory_order_relaxed);
   }

   void report(std::uint64_t opsSampled, bool resetTotals = true)
   {
      const char* perfCsvPath = std::getenv("PERF_CSV");
      if (perfCsvPath) {
         std::ofstream file(perfCsvPath);
         if (file.is_open()) {
            report(file, opsSampled, resetTotals);
            file.close();
         } else {
            std::cerr << "Error: Could not open PERF_CSV file: " << perfCsvPath << std::endl;
            report(std::cout, opsSampled, resetTotals);
         }
      } else {
         report(std::cout, opsSampled, resetTotals);
      }
   }

   void report(std::ostream& os, std::uint64_t opsSampled, bool resetTotals = true)
   {
      std::array<std::uint64_t, kNumEvents> snapshot{};
      for (std::size_t i = 0; i < kNumEvents; i++)
         snapshot[i] = totals[i].load(std::memory_order_relaxed);

      os << "label,total,per_op\n";
      for (std::size_t i = 0; i < kNumEvents; i++) {
         os << eventConfigs()[i].name << ',';
         if (availability[i].load(std::memory_order_relaxed)) {
            os << snapshot[i] << ',';
            if (opsSampled)
               os << static_cast<double>(snapshot[i]) / static_cast<double>(opsSampled);
            else
               os << "n/a";
         } else {
            os << "n/a,n/a";
         }
         os << '\n';
      }
      os << "derived_ipc,";
      if (availability[0].load(std::memory_order_relaxed) && availability[1].load(std::memory_order_relaxed) && snapshot[1])
         os << static_cast<double>(snapshot[0]) / static_cast<double>(snapshot[1]) << ",n/a\n";
      else
         os << "n/a,n/a\n";
      os << "derived_llc_miss_rate,";
      if (availability[2].load(std::memory_order_relaxed) && availability[3].load(std::memory_order_relaxed) && snapshot[2])
         os << snapshot[3] / static_cast<double>(snapshot[2]) << ",n/a\n";
      else
         os << "n/a,n/a\n";
      os << "derived_dtlb_miss_rate,";
      if (availability[4].load(std::memory_order_relaxed) && availability[5].load(std::memory_order_relaxed) && snapshot[4])
         os << snapshot[5] / static_cast<double>(snapshot[4]) << ",n/a\n";
      else
         os << "n/a,n/a\n";

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
      for (auto& v : availability)
         v.store(false, std::memory_order_relaxed);
   }

   void printConfiguredEvents()
   {
      if (printed.test_and_set())
         return;
      std::cerr << "perf_events_configured";
      const auto& configs = eventConfigs();
      for (std::size_t i = 0; i < kNumEvents; i++)
         if (availability[i].load(std::memory_order_relaxed))
            std::cerr << ' ' << configs[i].name;
      std::cerr << std::endl;
   }

   std::array<std::atomic<std::uint64_t>, kNumEvents> totals;
   std::array<std::atomic<bool>, kNumEvents> availability;
   std::atomic_flag printed = ATOMIC_FLAG_INIT;
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
      available.fill(false);
      bool any = false;
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
            fds[i] = -1;
            continue;
         }
         fds[i] = fd;
          available[i] = true;
          any = true;
      }
      if (!any) {
         for (auto fd : fds)
            if (fd != -1)
               ::close(fd);
         return false;
      }
      for (std::size_t i = 0; i < kNumEvents; i++) {
         if (available[i]) {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
         }
      }
      Aggregator::instance().setAvailability(available);
      return true;
   }

   void stop()
   {
      if (enabled)
         for (std::size_t i = 0; i < kNumEvents; i++)
            if (available[i])
               ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
   }

   bool enabled = false;
   std::array<int, kNumEvents> fds{};
   std::array<bool, kNumEvents> available{};
};

}  // namespace perf
