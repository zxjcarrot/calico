#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <errno.h>
#include <memory>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

// Simple debug logging controlled by PERF_DEBUG env var.
namespace perf_debug {
inline bool enabled() {
   static int val = [](){ const char* e = std::getenv("PERF_DEBUG"); return (e && std::string(e) == "1") ? 1 : 0; }();
   return val == 1;
}
inline void log(const std::string& msg) {
   if (!enabled()) return;
   std::cerr << "[perf_debug] " << msg << std::endl;
}
}

namespace perf
{

constexpr std::size_t kNumEvents = 8;

struct EventConfig {
   perf_type_id type;
   std::uint64_t config;
   const char* name;
};

// Helper function to get the PMU type from sysfs
inline int get_pmu_type(const std::string& pmu_name) {
    std::string path = "/sys/bus/event_source/devices/" + pmu_name + "/type";
    std::ifstream file(path);
    if (!file.is_open()) {
        return -1;
    }
    int type = -1;
    file >> type;
    return type;
}

inline const std::array<EventConfig, kNumEvents>& eventConfigs()
{
    static std::array<EventConfig, kNumEvents> events;
    static bool initialized = false;

    if (!initialized) {
        int amd_l3_type = get_pmu_type("amd_l3");

        if (amd_l3_type != -1) {
            // AMD L3 PMU is available, use raw events for LLC
         events = {{
                {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
                {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
                {(perf_type_id)amd_l3_type, 0xFF04ULL | (1ULL << 46) | (1ULL << 47), "llc_refs"}, // AMD L3 Accesses (L3LookupState, all states)
                {(perf_type_id)amd_l3_type, 0x0601ULL | (1ULL << 46) | (1ULL << 47), "llc_misses"}, // AMD L3 Misses (L3CombClstrState, miss state)
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "llc_store_refs"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "llc_store_misses"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "dtlb_refs"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "dtlb_misses"},
            }};
        } else {
            // Fallback to generic hardware events
            events = {{
                {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
                {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "llc_refs"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "llc_misses"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "llc_store_refs"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "llc_store_misses"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), "dtlb_refs"},
                {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), "dtlb_misses"},
            }};
        }
        initialized = true;
    }
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
      os << "derived_total_llc_refs,";
      bool has_refs = availability[2].load(std::memory_order_relaxed) || availability[4].load(std::memory_order_relaxed);
      if (has_refs) {
         double total_refs = (availability[2].load(std::memory_order_relaxed) ? snapshot[2] : 0) + (availability[4].load(std::memory_order_relaxed) ? snapshot[4] : 0);
         os << total_refs << ",n/a\n";
      } else os << "n/a,n/a\n";
      os << "derived_total_llc_misses,";
      bool has_misses = availability[3].load(std::memory_order_relaxed) || availability[5].load(std::memory_order_relaxed);
      if (has_misses) {
         double total_misses = (availability[3].load(std::memory_order_relaxed) ? snapshot[3] : 0) + (availability[5].load(std::memory_order_relaxed) ? snapshot[5] : 0);
         os << total_misses << ",n/a\n";
      } else os << "n/a,n/a\n";
      os << "derived_total_llc_miss_rate,";
      if (has_refs && has_misses) {
         double total_refs = (availability[2].load(std::memory_order_relaxed) ? snapshot[2] : 0) + (availability[4].load(std::memory_order_relaxed) ? snapshot[4] : 0);
         double total_misses = (availability[3].load(std::memory_order_relaxed) ? snapshot[3] : 0) + (availability[5].load(std::memory_order_relaxed) ? snapshot[5] : 0);
         if (total_refs > 0) os << total_misses / total_refs << ",n/a\n";
         else os << "n/a,n/a\n";
      } else os << "n/a,n/a\n";
      os << "derived_dtlb_miss_rate,";
      if (availability[6].load(std::memory_order_relaxed) && availability[7].load(std::memory_order_relaxed) && snapshot[6])
         os << snapshot[7] / static_cast<double>(snapshot[6]) << ",n/a\n";
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
   Scope() {
      enabled = openEvents();
      // If AMD L3 events aren't available via perf_event, start a per-scope perf stat runner
      // without interval, and collect summary in destructor.
      maybeStartPerfL3Scope();
   }
   ~Scope()
   {
      // Stop perf_event counters if they were enabled and accumulate
      if (enabled) {
         stop();
         std::array<std::uint64_t, kNumEvents> snapshot{};
         for (std::size_t i = 0; i < kNumEvents; i++) {
            std::uint64_t value = 0;
            if (fds[i] != -1 && read(fds[i], &value, sizeof(value)) == sizeof(value))
               snapshot[i] = value;
            if (fds[i] != -1) ::close(fds[i]);
         }
         Aggregator::instance().add(snapshot);
      }
      // Stop the per-scope perf runner and merge results
      collectPerfL3Scope();
   }

   Scope(const Scope&) = delete;
   Scope& operator=(const Scope&) = delete;

private:
   // Per-scope perf stat runner (no interval). We start at scope begin and stop in destructor.
   class PerfL3ScopeProc {
   public:
      void start() {
         if (active_) return;
         int fds[2];
         if (pipe2(fds, O_CLOEXEC) != 0) return;
         pid_t pid = fork();
         if (pid == 0) {
            dup2(fds[1], STDERR_FILENO);
            close(fds[0]);
            close(fds[1]);
            std::string pid_str = std::to_string(getppid());
            const char* mode = std::getenv("PERF_L3_MODE");
            bool systemWideMode = false;
            if (mode) {
               std::string m(mode);
               if (m == "system" || m == "system-wide" || m == "a") systemWideMode = true;
            }
            const char* argv_proc[] = {"perf","stat","-x",",","-p", pid_str.c_str(), "-e","amd_l3/l3_accesses/,amd_l3/l3_misses/", nullptr};
            const char* argv_sys[]  = {"perf","stat","-x",",","-a", "-e","amd_l3/l3_accesses/,amd_l3/l3_misses/", nullptr};
            const char* const* argv = systemWideMode ? argv_sys : argv_proc;
            execvp(argv[0], const_cast<char* const*>(argv));
            _exit(127);
         } else if (pid > 0) {
            close(fds[1]);
            pipefd_ = fds[0];
            child_ = pid;
            active_ = true;
            perf_debug::log(std::string("Started per-scope perf child PID=") + std::to_string(pid));
         } else {
            close(fds[0]); close(fds[1]);
         }
      }
      // Stops the child and parses summary results. Returns whether values are valid.
      bool stopAndCollect(uint64_t& refs, uint64_t& misses) {
         if (!active_) return false;
         // Ask perf to print summary
         if (child_ > 0) kill(child_, SIGINT);
         // Wait with timeout
         bool done = false;
         for (int i=0; i<40; ++i) {
            if (waitpid(child_, nullptr, WNOHANG) > 0) { done = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
         }
         if (!done && child_ > 0) {
            kill(child_, SIGTERM);
            for (int i=0; i<40; ++i) {
               if (waitpid(child_, nullptr, WNOHANG) > 0) { done = true; break; }
               std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
         }
         if (!done && child_ > 0) {
            kill(child_, SIGKILL);
            waitpid(child_, nullptr, 0);
         }
         child_ = -1;
         // Read remaining output
         std::string out;
         char buf[4096];
         ssize_t n;
         while ((n = ::read(pipefd_, buf, sizeof(buf))) > 0) out.append(buf, buf + n);
         if (pipefd_ != -1) { close(pipefd_); pipefd_ = -1; }
         active_ = false;
         perf_debug::log(std::string("perf_summary:\n") + out);
         // Parse lines
         std::istringstream iss(out);
         std::string line;
         bool haveRefs=false, haveMisses=false;
         while (std::getline(iss, line)) {
            if (line.empty()) continue;
            std::vector<std::string> toks; std::string tok; std::istringstream ls(line);
            while (std::getline(ls, tok, ',')) toks.push_back(tok);
            size_t ev_idx = std::string::npos;
            for (size_t i=0;i<toks.size();++i) if (toks[i].find("amd_l3/") != std::string::npos) { ev_idx = i; break; }
            if (ev_idx == std::string::npos) continue;
            // Value is typically at index 0 when no interval
            std::string valtok = toks.size() > 0 ? toks[0] : std::string();
            if (valtok.find("<not") != std::string::npos) continue;
            uint64_t v=0; try { v = static_cast<uint64_t>(std::stoll(valtok)); } catch (...) { continue; }
            const std::string& ev = toks[ev_idx];
            if (ev.find("amd_l3/l3_accesses/") != std::string::npos) { refs = v; haveRefs = true; }
            else if (ev.find("amd_l3/l3_misses/") != std::string::npos) { misses = v; haveMisses = true; }
         }
         return haveRefs || haveMisses;
      }
      ~PerfL3ScopeProc() {
         if (active_) {
            uint64_t r=0,m=0; stopAndCollect(r,m);
         }
      }
    private:
      pid_t child_ = -1;
      int pipefd_ = -1;
      bool active_ = false;
   };

   void maybeStartPerfL3Scope() {
      // Start perf-based L3 collection only if amd_l3 PMU exists and our llc events are unavailable
      int amd_l3_type = get_pmu_type("amd_l3");
      if (amd_l3_type == -1) return;
      // Check current availability indices (2: llc_refs, 3: llc_misses)
      // If either fd for llc refs/misses failed, start per-scope perf.
      if (fds[2] == -1 || fds[3] == -1) {
         l3scope = std::make_unique<PerfL3ScopeProc>();
         l3scope->start();
      }
   }
   void collectPerfL3Scope() {
      if (!l3scope) return;
      uint64_t refs = 0, misses = 0;
      bool ok = l3scope->stopAndCollect(refs, misses);
      if (!ok) return;
      std::array<bool, kNumEvents> avail{}; avail.fill(false);
      if (refs) avail[2] = true; if (misses || refs) avail[3] = true; // mark if any present
      Aggregator::instance().setAvailability(avail);
      std::array<std::uint64_t, kNumEvents> add{}; add.fill(0);
      add[2] = refs; add[3] = misses;
      Aggregator::instance().add(add);
      perf_debug::log(std::string("Per-scope perf add llc_refs=") + std::to_string(refs) + " llc_misses=" + std::to_string(misses));
      l3scope.reset();
   }

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

         // For AMD L3 events, set config and config1
         if (std::string(configs[i].name) == "llc_refs") {
            attr.config = 0x04;
            attr.config1 = 0xFF;
         } else if (std::string(configs[i].name) == "llc_misses") {
            attr.config = 0x01;
            attr.config1 = 0x06;
         }

         int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
         if (fd == -1) {
            std::cerr << "Failed to open perf event: " << configs[i].name
                      << " (type: " << attr.type << ", config: 0x" << std::hex << attr.config << std::dec << ")"
                      << " - " << strerror(errno) << " (errno: " << errno << ")" << std::endl;
            perf_debug::log(std::string("openEvents failure for ") + configs[i].name + ": errno=" + std::to_string(errno));
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
   std::unique_ptr<PerfL3ScopeProc> l3scope;
};

}  // namespace perf
