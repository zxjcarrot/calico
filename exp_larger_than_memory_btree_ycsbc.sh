#!/bin/bash

# exp_larger_than_memory_btree_ycsbc.sh - Larger-than-memory experiments with varying data sizes
# This script runs experiments with data sizes ranging from 50M to 400M records with 32 threads and 16GB memory


# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXP_BASE_DIR="${EXP_BASE_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/larger-than-memory/btree_ycsbc}"

# Data sizes to test (doubling from 50M to 400M)
DATA_SIZES=(50000000 100000000 200000000 400000000)
# Fixed thread count and memory
THREAD_COUNT=64
PHYSMEM=4
RUN_DURATION=240

# Common settings for all experiments
export RNDREAD=1
export BLOCK="${BLOCK:-${SCRIPT_DIR}/heapfile}"
export VIRTGB=800G
export PHYSGB=${PHYSMEM}G
export THREADS=${THREAD_COUNT}
export RUNFOR=${RUN_DURATION}
export PERF=0
NUMA_CPU_BIND=${NUMA_CPU_BIND:-0-127}
NUMA_MEM_INTERLEAVE=${NUMA_MEM_INTERLEAVE:-0,1}
USE_NUMACTL_BINDING=${USE_NUMACTL_BINDING:-1}
export CALICO_DISABLE_PERF=${CALICO_DISABLE_PERF:-1}
unset CALICO_ENABLE_SEARCH_STATS

numa_command_prefix() {
    if [ "$USE_NUMACTL_BINDING" != "1" ]; then
        return 0
    fi

    if ! command -v numactl >/dev/null 2>&1; then
        echo "Warning: numactl not found; running without NUMA binding." >&2
        return 0
    fi

    printf '%s\n' "numactl" "--physcpubind=${NUMA_CPU_BIND}" "--interleave=${NUMA_MEM_INTERLEAVE}"
}

lmdb_frontend_available() {
    local binary_path="$1"

    if [ ! -x "$binary_path" ]; then
        echo "Skipping LMDB experiment: binary not found at $binary_path" >&2
        return 1
    fi

    if command -v ldd >/dev/null 2>&1; then
        local missing_libs
        missing_libs=$(ldd "$binary_path" 2>/dev/null | grep 'not found' || true)
        if [ -n "$missing_libs" ]; then
            echo "Skipping LMDB experiment: unresolved shared libraries for $binary_path" >&2
            echo "$missing_libs" >&2
            echo "Install the LMDB development package so liblmdb.so is present, e.g. liblmdb-dev." >&2
            return 1
        fi
    fi

    return 0
}

wiredtiger_frontend_available() {
    local binary_path="$1"
    local target_name="$2"
    local status_var

    case "$target_name" in
        wiredtiger_tpcc)
            status_var="WIREDTIGER_TPCC_AVAILABLE"
            ;;
        wiredtiger_ycsb)
            status_var="WIREDTIGER_YCSB_AVAILABLE"
            ;;
        *)
            status_var="WIREDTIGER_FRONTEND_AVAILABLE"
            ;;
    esac

    if [ "${!status_var:-}" = "1" ]; then
        return 0
    fi
    if [ "${!status_var:-}" = "0" ]; then
        return 1
    fi

    if [ ! -x "$binary_path" ]; then
        local build_log
        build_log=$(mktemp)
        echo "Building missing WiredTiger frontend ${target_name}..." >&2
        if ! make -C "${SCRIPT_DIR}/leanstore/build" "$target_name" -j"$(nproc)" >"${build_log}" 2>&1; then
            echo "Skipping WiredTiger experiment: failed to build ${target_name}." >&2
            if grep -q 'wiredtiger.h: No such file or directory' "${build_log}"; then
                echo "Install the WiredTiger development package so wiredtiger.h is present, e.g. libwiredtiger-dev." >&2
            fi
            tail -n 20 "${build_log}" >&2
            rm -f "${build_log}"
            printf -v "$status_var" '%s' "0"
            return 1
        fi
        rm -f "${build_log}"
    fi

    if [ ! -x "$binary_path" ]; then
        echo "Skipping WiredTiger experiment: binary not found at $binary_path" >&2
        printf -v "$status_var" '%s' "0"
        return 1
    fi

    if command -v ldd >/dev/null 2>&1; then
        local missing_libs
        missing_libs=$(ldd "$binary_path" 2>/dev/null | grep 'not found' || true)
        if [ -n "$missing_libs" ]; then
            echo "Skipping WiredTiger experiment: unresolved shared libraries for $binary_path" >&2
            echo "$missing_libs" >&2
            echo "Install the WiredTiger runtime/development packages so libwiredtiger is present, e.g. libwiredtiger-dev." >&2
            printf -v "$status_var" '%s' "0"
            return 1
        fi
    fi

    printf -v "$status_var" '%s' "1"
    return 0
}

build_required_leanstore_frontends() {
    echo "Building required LeanStore frontends..."
    make -C "${SCRIPT_DIR}/leanstore/build" -j"$(nproc)" tpcc ycsb lmdb_tpcc lmdb_ycsb wiredtiger_tpcc wiredtiger_ycsb
}

log_rss_for_pid() {
    local pid="$1"
    local outfile="$2"
    echo "timestamp_sec rss_kb" > "${outfile}"
    while kill -0 "${pid}" 2>/dev/null; do
        if [ -r "/proc/${pid}/status" ]; then
            local rss_kb
            rss_kb=$(grep -E '^VmRSS:' "/proc/${pid}/status" | awk '{print $2}')
            if [ -n "${rss_kb}" ]; then
                printf "%s %s\n" "$(date +%s)" "${rss_kb}" >> "${outfile}"
            fi
        fi
        sleep 1
    done
}

log_progress_for_pid() {
    local pid="$1"
    local label="$2"
    local interval_sec="${3:-10}"
    local start_ts
    start_ts=$(date +%s)
    while kill -0 "${pid}" 2>/dev/null; do
        sleep "${interval_sec}"
        if ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        local now_ts elapsed rss_kb
        now_ts=$(date +%s)
        elapsed=$((now_ts - start_ts))
        rss_kb="n/a"
        if [ -r "/proc/${pid}/status" ]; then
            rss_kb=$(grep -E '^VmRSS:' "/proc/${pid}/status" | awk '{print $2}')
            rss_kb=${rss_kb:-"n/a"}
        fi
        echo "[progress] ${label}: still running (elapsed=${elapsed}s, rss_kb=${rss_kb})"
    done
}

calico_executable_for_config() {
    local config_name="$1"

    if [[ "$config_name" == calico_array* ]]; then
        printf '%s\n' "./calico_oltp_query_special"
        return 0
    fi

    printf '%s\n' "./calico_oltp_query"
}

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

echo "Building predicache..."
make -C "${SCRIPT_DIR}/predictivetranslation-sigmod2026" clean
make -C "${SCRIPT_DIR}/predictivetranslation-sigmod2026" predicache -j$(nproc)

build_required_leanstore_frontends



# Function to run lmdb experiment
run_lmdb_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="lmdb"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    
    rm ${tx_file}
    echo "=================================================================="
    echo "Running lmdb experiment with datasize ${datasize}"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"
    
    echo "Running lmdb ycsb..."
    echo "Output files:"
    rm -rf lmdb_db
    mkdir lmdb_db
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"

    if ! lmdb_frontend_available "build/frontend/lmdb_ycsb"; then
        printf '%s\n' 'LMDB experiment skipped: build/frontend/lmdb_ycsb cannot load liblmdb.so. Install liblmdb-dev and rebuild leanstore frontends.' > "${tx_file}"
        touch "${perf_file}"
        cd "${SCRIPT_DIR}"
        echo "lmdb experiment with datasize ${datasize} skipped."
        echo
        return 0
    fi
    # Run lmdb and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local attempt=1
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        # Create cgroup for memory limitation
        CGROUP_NAME="lmdb_bench"
        sudo cgcreate -g memory:${CGROUP_NAME}
        # Set memory limit to PHYSMEM (in bytes)
        MEM_LIMIT_BYTES=$((${PHYSMEM} * 1024 * 1024 * 1024))
        sudo cgset -r memory.limit_in_bytes=${MEM_LIMIT_BYTES} ${CGROUP_NAME}
        # Enable swap (set memsw limit higher than mem limit to allow swapping)
        # If swap is enabled on system, this allows swapping. If memsw is not set or equal to limit, swap might be restricted depending on config.
        # Usually setting limit_in_bytes is enough to force swapping if the process exceeds it.
        
        # Run in cgroup
        sudo cgexec -g memory:${CGROUP_NAME} build/frontend/lmdb_ycsb \
            --ycsb_read_ratio=100 \
            --ycsb_tuple_count=${datasize} \
            --ssd_path=./lmdb_db \
            --worker_threads=${THREAD_COUNT} \
            --pp_threads=1 \
            --dram_gib=100 \
            --csv_path=./log \
            --free_pct=1 \
            --print_tx_console \
            --run_for_seconds=${RUN_DURATION} > "${tx_file}" 2>&1 &
        bench_pid=$!
        
        # We can't easily get the PID of the process started by sudo cgexec for rss logging in the same way
        # But since we are limiting memory via cgroup, RSS logging might be less relevant or needs to track the cgroup usage.
        # For now, we skip specific PID RSS logging or we'd need to find the PID of lmdb_ycsb.
        # Let's try to find it:
        sleep 1
        real_pid=$(pgrep -f "lmdb_ycsb")
        if [ -n "$real_pid" ]; then
             log_rss_for_pid "${real_pid}" "${rss_file}" &
             rss_pid=$!
        fi

        # Wait for the benchmark process (which is the sudo cgexec command in this shell context? No, bench_pid is the sudo command)
        # Actually, bench_pid is the PID of the sudo command.
        wait "${bench_pid}"
        exit_code=$?
        
        # Clean up cgroup
        sudo cgdelete memory:${CGROUP_NAME}

        kill "${rss_pid}" 2>/dev/null || true
        wait "${rss_pid}" 2>/dev/null || true
        kill "${iostat_pid}" 2>/dev/null || true
        wait "${iostat_pid}" 2>/dev/null || true

        if [ ${exit_code} -eq 0 ]; then
            break
        fi
        echo "ycsb failed with exit code ${exit_code}, retrying in 5 seconds (attempt ${attempt})..."
        attempt=$((attempt + 1))
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "lmdb experiment with datasize ${datasize} completed."
    echo
    
    rm -rf lmdb_db
}

# Function to run leanstore experiment
run_leanstore_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="leanstore"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    rm ${tx_file}

    echo "=================================================================="
    echo "Running leanstore experiment with datasize ${datasize}"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"
    
    # Ensure heapfile exists (allocate enough space for largest dataset)
    if [ ! -f "build/heapfile" ]; then
        echo "Creating leanstore heapfile..."
        fallocate -l 100G build/heapfile
    fi
    
    echo "Running leanstore ycsb..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"
    # Run leanstore and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local attempt=1
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        build/frontend/ycsb \
            --ycsb_read_ratio=100 \
            --ycsb_tuple_count=${datasize} \
            --ssd_path=build/heapfile \
            --worker_threads=${THREAD_COUNT} \
            --pp_threads=4 \
            --dram_gib=${PHYSMEM} \
            --csv_path=./log \
            --free_pct=1 \
            --wal=false \
            --vi=false \
            --print_tx_console \
            --run_for_seconds=${RUN_DURATION} \
            --isolation_level=ru > "${tx_file}" 2>&1 &
        bench_pid=$!

        log_rss_for_pid "${bench_pid}" "${rss_file}" &
        rss_pid=$!

        wait "${bench_pid}"
        exit_code=$?

        kill "${rss_pid}" 2>/dev/null || true
        wait "${rss_pid}" 2>/dev/null || true
        kill "${iostat_pid}" 2>/dev/null || true
        wait "${iostat_pid}" 2>/dev/null || true

        if [ ${exit_code} -eq 0 ]; then
            break
        fi
        echo "ycsb failed with exit code ${exit_code}, retrying in 5 seconds (attempt ${attempt})..."
        attempt=$((attempt + 1))
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "Leanstore experiment with datasize ${datasize} completed."
    echo

    rm -rf build/heapfile
}

# Function to run WiredTiger experiment
run_wiredtiger_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="wiredtiger"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    rm ${tx_file}
    iostat -x 1 nvme1n1 > "${iostat_file}" &
    iostat_pid=$!

    echo "=================================================================="
    echo "Running WiredTiger experiment with datasize ${datasize}"
    echo "=================================================================="

    
    cd "${SCRIPT_DIR}/leanstore"

    # Remove old WiredTiger database directory
    rm -rf build/wt_db
    mkdir build/wt_db
    echo "Running WiredTiger ycsb..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"

    if ! wiredtiger_frontend_available "build/frontend/wiredtiger_ycsb" "wiredtiger_ycsb"; then
        printf '%s\n' 'WiredTiger experiment skipped: build/frontend/wiredtiger_ycsb is unavailable. Install libwiredtiger-dev and rebuild leanstore frontends.' > "${tx_file}"
        touch "${perf_file}" "${iostat_file}"
        kill "${iostat_pid}" 2>/dev/null || true
        wait "${iostat_pid}" 2>/dev/null || true
        cd "${SCRIPT_DIR}"
        echo "WiredTiger experiment with datasize ${datasize} skipped."
        return 0
    fi

    # Run WiredTiger and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    while true; do
        build/frontend/wiredtiger_ycsb \
            --ycsb_read_ratio=100 \
            --ycsb_tuple_count=${datasize} \
            --ssd_path=build/wt_db \
            --worker_threads=${THREAD_COUNT} \
            --pp_threads=1 \
            --dram_gib=${PHYSMEM} \
            --csv_path=./log \
            --free_pct=1 \
            --wal=false\
            --vi=false\
            --contention_split \
            --xmerge \
            --print_tx_console \
            --run_for_seconds=${RUN_DURATION} \
            --isolation_level=si > "${tx_file}" 2>&1 &
        bench_pid=$!

        log_rss_for_pid "${bench_pid}" "${rss_file}" &
        rss_pid=$!

        wait "${bench_pid}"
        exit_code=$?

        kill "${rss_pid}" 2>/dev/null || true

        if [ ${exit_code} -eq 0 ]; then
            break
        fi
        echo "wiredtiger_ycsb failed with exit code ${exit_code}, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "WiredTiger experiment with datasize ${datasize} completed."
    echo
    kill "${iostat_pid}"
    rm -rf build/wt_db
}

# Function to run a single experiment
run_experiment() {
    local datasize="$1"
    local config_name="$2"
    local description="$3"
    shift 3
    local env_vars=("$@")
    
    # Create datasize-specific directory
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    echo "=================================================================="
    echo "Running experiment: ${config_name} with datasize ${datasize}"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Set datasize
    export DATASIZE=${datasize}
    
    # Set environment variables for this configuration
    for env_var in "${env_vars[@]}"; do
        export ${env_var}
        echo "Set ${env_var}"
    done
    
    # Set output file paths in datasize-specific directory
    export TX_CSV="${datasize_dir}/${config_name}_tx.csv"
    export PERF_CSV="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    
    rm ${TX_CSV}
    fallocate -l 100G heapfile
    echo "Data size: ${datasize}"
    echo "Thread count: ${THREAD_COUNT}"
    echo "Memory: ${PHYSMEM}G"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo
    
    # Run the experiment
    echo "Starting experiment..."
    
    # repeat until ./calico_oltp_query returns 0 return code
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!
        local -a CMD_PREFIX=()
        local executable
        mapfile -t CMD_PREFIX < <(numa_command_prefix)
        executable="$(calico_executable_for_config "$config_name")"
        "${CMD_PREFIX[@]}" "$executable" \
            > >(tee "${TX_CSV}") \
            2> >(tee "${datasize_dir}/${config_name}_stderr.log" >&2) &
        bench_pid=$!
        log_rss_for_pid "${bench_pid}" "${datasize_dir}/${config_name}_rss.log" &
        rss_pid=$!
        wait "${bench_pid}"
        exit_code=$?
        kill "${rss_pid}" 2>/dev/null || true

        # Check line count
        line_count=0
        if [ -f "${TX_CSV}" ]; then
            line_count=$(wc -l < "${TX_CSV}")
        fi

        if [ ${exit_code} -eq 0 ] || [ ${line_count} -gt 60 ]; then
            break
        fi
        echo "calico_oltp_query failed with exit code ${exit_code} (lines: ${line_count}), retrying in 5 seconds..."
        sleep 5
    done

    kill "${rss_pid}" 2>/dev/null || true
    kill "${iostat_pid}"
    echo "Experiment ${config_name} with datasize ${datasize} completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
    rm -rf heapfile
}

# Function to run PrediCache random-read (YCSB-C style) experiment
run_predicache_ycsbc_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"

    local config_name="predicache_ycsbc"
    export TX_CSV="${datasize_dir}/${config_name}_tx.csv"
    export PERF_CSV="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    local rss_file="${datasize_dir}/${config_name}_rss.log"

    echo "=================================================================="
    echo "Running PrediCache YCSB-C style experiment with datasize ${datasize}"
    echo "=================================================================="
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo "  IOSTAT LOG: ${iostat_file}"

    export DATASIZE=${datasize}
    # PrediCache requires BLOCK to exist before startup.
    if [ ! -f "${BLOCK}" ]; then
        fallocate -l 100G "${BLOCK}"
    fi
    local load_threads="${PREDICACHE_LOAD_THREADS:-2}"
    echo "  PrediCache load threads: ${load_threads}"
    local attempt=1
    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        POOLGB=${PHYSMEM} WORKLOAD=rndread RNDREAD=1 RNDREAD_LOAD_THREADS="${load_threads}" \
            "${CMD_PREFIX[@]}" "${SCRIPT_DIR}/predictivetranslation-sigmod2026/predicache" \
            > >(tee "${TX_CSV}") \
            2> >(tee "${datasize_dir}/${config_name}_stderr.log" >&2) &
        bench_pid=$!
        log_progress_for_pid "${bench_pid}" "predicache_ycsbc datasize=${datasize}" 10 &
        progress_pid=$!

        log_rss_for_pid "${bench_pid}" "${rss_file}" &
        rss_pid=$!

        wait "${bench_pid}"
        exit_code=$?

        kill "${progress_pid}" 2>/dev/null || true
        wait "${progress_pid}" 2>/dev/null || true
        kill "${rss_pid}" 2>/dev/null || true
        wait "${rss_pid}" 2>/dev/null || true
        kill "${iostat_pid}" 2>/dev/null || true
        wait "${iostat_pid}" 2>/dev/null || true

        line_count=0
        if [ -f "${TX_CSV}" ]; then
            line_count=$(wc -l < "${TX_CSV}")
        fi

        if [ ${exit_code} -eq 0 ] || [ ${line_count} -gt 10 ]; then
            break
        fi

        if [ ${exit_code} -eq 139 ] && [ "${load_threads}" -gt 1 ]; then
            echo "PrediCache segfaulted during multi-thread load; retrying with load_threads=1"
            load_threads=1
        fi

        echo "PrediCache run failed with exit code ${exit_code} (lines: ${line_count}), retrying in 5 seconds..."
        attempt=$((attempt + 1))
        sleep 5
    done

    touch "${PERF_CSV}"
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "Larger-than-Memory Experiment Summary" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Data sizes tested: ${DATA_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Fixed settings: THREADS=${THREAD_COUNT} PHYSGB=${PHYSMEM}G VIRTGB=128G RNDREAD=1 PERF=1" >> "${GLOBAL_SUMMARY_FILE}"
echo "Configurations: calico_mmap_nohuge, calico_array_huge, predicache_ycsbc, calico_lockfree_hash_huge, calico_robinhood_hash_huge, leanstore, wiredtiger" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

# Run experiments for each data size
for datasize in "${DATA_SIZES[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH DATASIZE ${datasize}"
    echo "######################################################################"
    
    # Create datasize-specific config file
    datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    cat > "${datasize_dir}/config.json" <<EOF
{
  "description": "Configuration for datasize ${datasize} experiments",
  "datasize": ${datasize},
  "threads": ${THREAD_COUNT},
  "memory_gb": ${PHYSMEM},
  "common_settings": {
    "RNDREAD": 1,
    "BLOCK": "${BLOCK}",
    "VIRTGB": "128G",
    "PHYSGB": "${PHYSMEM}G",
    "THREADS": ${THREAD_COUNT},
    "DATASIZE": ${datasize},
    "PERF": 1
  }
}
EOF
    
    echo "Running experiments with datasize ${datasize}..." >> "${GLOBAL_SUMMARY_FILE}"

    # PrediCache random-read baseline
    # run_predicache_ycsbc_experiment ${datasize}

    # run_lmdb_experiment ${datasize}


    # # # Configuration 5: WiredTiger
    # run_wiredtiger_experiment ${datasize}

    # Configuration 4: Leanstore
    run_leanstore_experiment ${datasize}


    #     # Configuration 1: Calico mmap without huge pages (vmcache)
    # run_experiment ${datasize} "calico_mmap_nohuge" \
    #     "Calico with mmap (virtual memory cache) without huge pages" \
    #     "TRADITIONAL=0" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=1"

 
    # # Configuration 3: Calico openaddr_hash with huge pages
    # run_experiment ${datasize} "calico_openaddr_hash_huge" \
    #     "Calico with traditional buffer pool using lock free open addressing hash translator with huge pages" \
    #     "TRADITIONAL=1" "TRADHASH=2" "NOHUGEPAGE_FRAME_MEM=0"

    # Robin Hood hash baseline with huge-page-backed buffer frames
    run_experiment ${datasize} "calico_robinhood_hash_huge" \
        "Calico with traditional buffer pool using Robin Hood hash translator with huge pages" \
        "TRADITIONAL=1" "TRADHASH=7" "NOHUGEPAGE_FRAME_MEM=0"

    #Configuration 2: Calico array with huge pages
    # run_experiment ${datasize} "calico_array_huge" \
    #     "Calico with traditional buffer pool using array translator with huge pages" \
    #     "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=0" "TRANSLATION_REFCOUNT=1"


    echo "Completed experiments with datasize ${datasize}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH DATASIZE ${datasize}"
    echo "######################################################################"
    echo
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Configurations tested per data size:"
echo "  - calico_mmap_nohuge: without huge pages"
echo "  - calico_array_huge: with huge pages"
echo "  - predicache_ycsbc: PrediCache random-read"
echo "  - calico_lockfree_hash_huge: with huge pages"
echo "  - calico_robinhood_hash_huge: Robin Hood hash with huge pages"
echo "  - leanstore"
echo "  - wiredtiger"
echo "  Total per data size: 5 experiment configurations"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by data size:"
for datasize in "${DATA_SIZES[@]}"; do
    echo "--- ${datasize} records ---"
    find "${EXP_BASE_DIR}/${datasize}" -name "*.csv" -o -name "*.json" 2>/dev/null | sort
done

echo
echo "Running results extraction for all data sizes..."
./extract_results_btree_oom_ycsbc.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
