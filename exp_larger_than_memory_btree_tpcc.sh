#!/bin/bash

# exp_larger_than_memory_btree_tpcc.sh - Larger-than-memory experiments with varying data sizes
# This script runs TPCC experiments with data sizes ranging from 10 to 400 warehouses

# set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/btree_tpcc"

# Data sizes to test (in warehouses)
DATA_SIZES=(20 40 80 100 200 400 800)
#DATA_SIZES=(200 400 800)
# Fixed thread count and memory
THREAD_COUNT=64
PHYSMEM=8
RUN_DURATION=180
LMDB_MAP_GB=${LMDB_MAP_GB:-400}

# Common settings for all experiments
export RNDREAD=0 # 0 for TPC-C
export BLOCK="${BLOCK:-${SCRIPT_DIR}/heapfile}"
export VIRTGB=800G
export PHYSGB=${PHYSMEM}G
export THREADS=${THREAD_COUNT}
export RUNFOR=${RUN_DURATION}
export PERF=0
NUMA_CPU_BIND=${NUMA_CPU_BIND:-0-127}
NUMA_MEM_BIND=${NUMA_MEM_BIND:-0}
USE_NUMACTL_BINDING=${USE_NUMACTL_BINDING:-1}
export CALICO_DISABLE_PERF=${CALICO_DISABLE_PERF:-1}
unset CALICO_ENABLE_SEARCH_STATS

is_cgroup_v2() {
    [ -f "/sys/fs/cgroup/cgroup.controllers" ]
}

setup_memory_cgroup() {
    local cgroup_name="$1"
    local memory_limit_bytes="$2"
    local cgroup_path

    if ! command -v sudo >/dev/null 2>&1; then
        echo "Warning: sudo not found; running without cgroup limits." >&2
        return 1
    fi

    if is_cgroup_v2; then
        cgroup_path="/sys/fs/cgroup/${cgroup_name}"
        sudo mkdir -p "$cgroup_path" || return 1
        if [ -w "/sys/fs/cgroup/cgroup.subtree_control" ] || sudo test -w "/sys/fs/cgroup/cgroup.subtree_control"; then
            echo "+memory" | sudo tee /sys/fs/cgroup/cgroup.subtree_control >/dev/null 2>&1 || true
        fi
        echo "$memory_limit_bytes" | sudo tee "${cgroup_path}/memory.max" >/dev/null || return 1
    else
        cgroup_path="/sys/fs/cgroup/memory/${cgroup_name}"
        sudo mkdir -p "$cgroup_path" || return 1
        echo "$memory_limit_bytes" | sudo tee "${cgroup_path}/memory.limit_in_bytes" >/dev/null || return 1
    fi

    printf '%s\n' "$cgroup_path"
}

run_command_in_cgroup() {
    local cgroup_path="$1"
    shift

    if [ -z "$cgroup_path" ]; then
        "$@"
        return
    fi

    local procs_file="${cgroup_path}/tasks"
    if is_cgroup_v2; then
        procs_file="${cgroup_path}/cgroup.procs"
    fi

    if [ ! -e "$procs_file" ]; then
        echo "Warning: cgroup file $procs_file is missing; running without cgroup limits." >&2
        "$@"
        return
    fi

    sudo env PATH="$PATH" bash -lc 'echo $$ > "$1" && shift && exec "$@"' bash "$procs_file" "$@"
}

cleanup_memory_cgroup() {
    local cgroup_path="$1"

    if [ -z "$cgroup_path" ] || [ ! -d "$cgroup_path" ]; then
        return 0
    fi

    sudo rmdir "$cgroup_path" 2>/dev/null || true
}

numa_command_prefix() {
    if [ "$USE_NUMACTL_BINDING" != "1" ]; then
        return 0
    fi

    if ! command -v numactl >/dev/null 2>&1; then
        echo "Warning: numactl not found; running without NUMA binding." >&2
        return 0
    fi

    printf '%s\n' "numactl" "--physcpubind=${NUMA_CPU_BIND}" "--membind=${NUMA_MEM_BIND}"
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
make -C "${SCRIPT_DIR}/predictivetranslation-sigmod2026" predicache_no_pt -j$(nproc)

build_required_leanstore_frontends

# Function to run leanstore experiment
run_leanstore_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="leanstore"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    local stdout_file="${datasize_dir}/${config_name}_stdout.log"
    
    echo "=================================================================="
    echo "Running leanstore experiment with datasize ${datasize} warehouses"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"
    
    # Ensure heapfile exists (allocate enough space for largest dataset)
    if [ ! -f "build/heapfile" ]; then
        echo "Creating leanstore heapfile..."
        fallocate -l 100G build/heapfile
    fi
    
    echo "Running leanstore tpcc..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo "  STDOUT LOG: ${stdout_file}"
    # Run leanstore and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local attempt=1
    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        "${CMD_PREFIX[@]}" build/frontend/tpcc \
            --tpcc_warehouse_count=${datasize} \
            --ssd_path=build/heapfile \
            --worker_threads=$THREAD_COUNT \
            --pp_threads=8 \
            --dram_gib=${PHYSMEM} \
            --csv_path=./log \
            --wal=0 \
            --vi=0 \
            --tpcc_warehouse_affinity=0 \
            --free_pct=1 \
            --print_tx_console \
            --run_for_seconds=${RUN_DURATION} \
            --isolation_level=rc > >(tee "${stdout_file}" > "${tx_file}") 2>&1 &
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
        echo "tpcc failed with exit code ${exit_code}, retrying in 5 seconds (attempt ${attempt})..."
        attempt=$((attempt + 1))
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "Leanstore experiment with datasize ${datasize} completed."
    echo
}


# Function to run lmdb experiment
run_lmdb_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="lmdb"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    local stdout_file="${datasize_dir}/${config_name}_stdout.log"
    
    echo "=================================================================="
    echo "Running lmdb experiment with datasize ${datasize} warehouses"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"

    rm -rf lmdb_db 2>/dev/null || sudo rm -rf lmdb_db
    mkdir -p lmdb_db
    echo "Running lmdb tpcc..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo "  STDOUT LOG: ${stdout_file}"

    if ! lmdb_frontend_available "build/frontend/lmdb_tpcc"; then
        printf '%s\n' 'LMDB experiment skipped: build/frontend/lmdb_tpcc cannot load liblmdb.so. Install liblmdb-dev and rebuild leanstore frontends.' > "${tx_file}"
        printf '%s\n' 'LMDB experiment skipped because liblmdb.so is unavailable.' > "${stdout_file}"
        touch "${perf_file}"
        cd "${SCRIPT_DIR}"
        echo "lmdb experiment with datasize ${datasize} skipped."
        echo
        return 0
    fi
    # Run lmdb and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local attempt=1
    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)

    local cgroup_name="lmdb_bench_${datasize}"
    local mem_limit_bytes=$((${PHYSMEM} * 1024 * 1024 * 1024))

    while true; do
        local cgroup_path=""
        cgroup_path=$(setup_memory_cgroup "$cgroup_name" "$mem_limit_bytes") || {
            cgroup_path=""
            echo "Failed to create cgroup ${cgroup_name}. Running without cgroup limits."
        }

        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        local -a BENCH_CMD=(
            "${CMD_PREFIX[@]}"
            build/frontend/lmdb_tpcc
            --tpcc_warehouse_count=${datasize}
            --ssd_path=./lmdb_db
            --worker_threads=$THREAD_COUNT
            --pp_threads=4
            --dram_gib=${LMDB_MAP_GB}
            --csv_path=./log
            --wal=0
            --vi=0
            --tpcc_warehouse_affinity=0
            --free_pct=1
            --print_tx_console
            --run_for_seconds=${RUN_DURATION}
            --isolation_level=rc
        )

        run_command_in_cgroup "$cgroup_path" "${BENCH_CMD[@]}" > >(tee "${stdout_file}" > "${tx_file}") 2>&1 &
        bench_pid=$!

        sleep 1
        real_pid=$(pgrep -f "lmdb_tpcc" | head -n 1)
        if [ -n "$real_pid" ]; then
             log_rss_for_pid "${real_pid}" "${rss_file}" &
             rss_pid=$!
        fi

        wait "${bench_pid}"
        exit_code=$?

        kill "${rss_pid}" 2>/dev/null || true
        wait "${rss_pid}" 2>/dev/null || true
        kill "${iostat_pid}" 2>/dev/null || true
        wait "${iostat_pid}" 2>/dev/null || true
        
        cleanup_memory_cgroup "$cgroup_path"
        cgroup_path=""

        if [ ${exit_code} -eq 0 ]; then
            break
        fi
        echo "tpcc failed with exit code ${exit_code}, retrying in 5 seconds (attempt ${attempt})..."
        attempt=$((attempt + 1))
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "lmdb experiment with datasize ${datasize} completed."
    echo
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
    local stdout_file="${datasize_dir}/${config_name}_stdout.log"
    
    echo "=================================================================="
    echo "Running WiredTiger experiment with datasize ${datasize} warehouses"
    echo "=================================================================="

    
    cd "${SCRIPT_DIR}/leanstore"

    # Remove old WiredTiger database directory
    rm -rf build/wt_db
    mkdir build/wt_db
    echo "Running WiredTiger tpcc..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo "  STDOUT LOG: ${stdout_file}"

    if ! wiredtiger_frontend_available "build/frontend/wiredtiger_tpcc" "wiredtiger_tpcc"; then
        printf '%s\n' 'WiredTiger experiment skipped: build/frontend/wiredtiger_tpcc is unavailable. Install libwiredtiger-dev and rebuild leanstore frontends.' > "${tx_file}"
        printf '%s\n' 'WiredTiger experiment skipped before execution because the frontend is unavailable.' > "${stdout_file}"
        touch "${perf_file}" "${iostat_file}"
        cd "${SCRIPT_DIR}"
        echo "WiredTiger experiment with datasize ${datasize} skipped."
        return 0
    fi

    # Run WiredTiger and capture output, rerun if failed
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local attempt=1
    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        "${CMD_PREFIX[@]}" build/frontend/wiredtiger_tpcc \
            --tpcc_warehouse_count=${datasize} \
            --ssd_path=build/wt_db \
            --worker_threads=${THREAD_COUNT} \
            --pp_threads=1 \
            --dram_gib=${PHYSMEM} \
            --csv_path=./log \
            --free_pct=1 \
            --tpcc_warehouse_affinity=0 \
            --print_tx_console \
            --run_for_seconds=${RUN_DURATION} \
            --isolation_level=si > >(tee "${stdout_file}" > "${tx_file}") 2>&1 &
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
        echo "wiredtiger_tpcc failed with exit code ${exit_code}, retrying in 5 seconds (attempt ${attempt})..."
        attempt=$((attempt + 1))
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "WiredTiger experiment with datasize ${datasize} completed."
    echo
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
    echo "Running experiment: ${config_name} with datasize ${datasize} warehouses"
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
    local stdout_file="${datasize_dir}/${config_name}_stdout.log"
    
    echo "Data size: ${datasize} warehouses"
    echo "Thread count: ${THREAD_COUNT}"
    echo "Memory: ${PHYSMEM}G"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo "  STDOUT LOG: ${stdout_file}"
    echo
    
    # Run the experiment
    echo "Starting experiment..."
    iostat -x 1 nvme1n1 > "${iostat_file}" &
    iostat_pid=$!

    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local -a CMD_PREFIX=()
    local executable
    mapfile -t CMD_PREFIX < <(numa_command_prefix)
    executable="$(calico_executable_for_config "$config_name")"
    "${CMD_PREFIX[@]}" "$executable" > >(tee "${stdout_file}") 2>&1 &
    bench_pid=$!
    log_rss_for_pid "${bench_pid}" "${rss_file}" &
    rss_pid=$!
    wait "${bench_pid}"
    kill "${rss_pid}" 2>/dev/null || true
    kill "${iostat_pid}"
    echo "Experiment ${config_name} with datasize ${datasize} completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
}

# Function to run PrediCache TPC-C experiment
run_predicache_tpcc_experiment() {
    local datasize="$1"
    local config_name="$2"
    local binary_name="$3"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"

    export TX_CSV="${datasize_dir}/${config_name}_tx.csv"
    export PERF_CSV="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    local rss_file="${datasize_dir}/${config_name}_rss.log"
    local binary_path="${SCRIPT_DIR}/predictivetranslation-sigmod2026/${binary_name}"
    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)

    echo "=================================================================="
    echo "Running ${config_name} experiment with datasize ${datasize} warehouses"
    echo "=================================================================="
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo "  IOSTAT LOG: ${iostat_file}"

    export DATASIZE=${datasize}
    # PrediCache requires BLOCK to exist before startup.
    if [ ! -f "${BLOCK}" ]; then
        fallocate -l 100G "${BLOCK}"
    fi
    if [ ! -x "${binary_path}" ]; then
        echo "Missing PrediCache binary: ${binary_path}" >&2
        return 1
    fi
    local attempt=1
    while true; do
        iostat -x 1 nvme1n1 > "${iostat_file}" &
        iostat_pid=$!

        POOLGB=${PHYSMEM} WORKLOAD=tpcc \
            "${CMD_PREFIX[@]}" "${binary_path}" > "${TX_CSV}" 2>&1 &
        bench_pid=$!

        log_rss_for_pid "${bench_pid}" "${rss_file}" &
        rss_pid=$!

        wait "${bench_pid}"
        exit_code=$?

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

        echo "${config_name} run failed with exit code ${exit_code} (lines: ${line_count}), retrying in 5 seconds..."
        attempt=$((attempt + 1))
        sleep 5
    done

    touch "${PERF_CSV}"
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "Larger-than-Memory TPCC Experiment Summary" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Data sizes tested (warehouses): ${DATA_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Fixed settings: THREADS=${THREAD_COUNT} PHYSGB=${PHYSMEM}G VIRTGB=128G RNDREAD=0 PERF=0" >> "${GLOBAL_SUMMARY_FILE}"
echo "Configurations: calico_mmap_nohuge, calico_array_huge, calico_openaddr_hash_huge, calico_robinhood_hash_huge, predicache_tpcc_pt_on, predicache_tpcc_pt_off, leanstore, wiredtiger" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

# Run experiments for each data size
for datasize in "${DATA_SIZES[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH DATASIZE ${datasize} WAREHOUSES"
    echo "######################################################################"
    
    # Create datasize-specific config file
    datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    cat > "${datasize_dir}/config.json" <<EOF
{
  "description": "Configuration for datasize ${datasize} experiments",
  "datasize_warehouses": ${datasize},
  "threads": ${THREAD_COUNT},
  "memory_gb": ${PHYSMEM},
  "common_settings": {
    "RNDREAD": 0,
    "BLOCK": "${BLOCK}",
    "VIRTGB": "128G",
    "PHYSGB": "${PHYSMEM}G",
    "THREADS": ${THREAD_COUNT},
    "DATASIZE": ${datasize},
    "PERF": 0
  }
}
EOF
    
    echo "Running experiments with datasize ${datasize} warehouses..." >> "${GLOBAL_SUMMARY_FILE}"


    # run_predicache_tpcc_experiment ${datasize} predicache_tpcc_pt_on predicache
    # run_predicache_tpcc_experiment ${datasize} predicache_tpcc_pt_off predicache_no_pt
    
    # run_lmdb_experiment ${datasize}


# #    Configuration 2: Calico array with huge pages
    # run_experiment ${datasize} "calico_array_huge" \
    #     "Calico with traditional buffer pool using array translator with huge pages" \
    #     "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=0" # "NOHUGEPAGE_TRANSLATION_ARRAY=0" "TRANSLATION_REFCOUNT=0" "IGNORE_PID_ALLOCATOR=1"



    # # # # Configuration 4: Leanstore
    run_leanstore_experiment ${datasize}


    # # Configuration 1: Calico mmap without huge pages
    # run_experiment ${datasize} "calico_mmap_nohuge" \
    #     "Calico with mmap (virtual memory cache) without huge pages" \
    #     "TRADITIONAL=0" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=1"






    # Configuration 3: Calico openaddr_hash with huge pages
    # run_experiment ${datasize} "calico_openaddr_hash_huge" \
    #     "Calico with traditional buffer pool using lock free open addressing hash translator with huge pages" \
    #     "TRADITIONAL=1" "TRADHASH=2" "NOHUGEPAGE_FRAME_MEM=0"

    # Robin Hood hash baseline with huge-page-backed buffer frames
    run_experiment ${datasize} "calico_robinhood_hash_huge" \
        "Calico with traditional buffer pool using Robin Hood hash translator with huge pages" \
        "TRADITIONAL=1" "TRADHASH=7" "NOHUGEPAGE_FRAME_MEM=0"




    # # Configuration 5: WiredTiger
    # run_wiredtiger_experiment ${datasize}
    echo "Completed experiments with datasize ${datasize}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH DATASIZE ${datasize} WAREHOUSES"
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
echo "  - predicache_tpcc_pt_on: PrediCache TPCC with predictive translation"
echo "  - predicache_tpcc_pt_off: PrediCache TPCC without predictive translation"
echo "  - calico_lockfree_hash_huge: with huge pages"
echo "  - calico_robinhood_hash_huge: Robin Hood hash with huge pages"
echo "  - leanstore"
echo "  - wiredtiger"
echo "  Total per data size: 6 experiment configurations"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by data size:"
for datasize in "${DATA_SIZES[@]}"; do
    echo "--- ${datasize} warehouses ---"
    find "${EXP_BASE_DIR}/${datasize}" -name "*.csv" -o -name "*.json" 2>/dev/null | sort
done

echo
echo "Running results extraction for all data sizes..."
./extract_results_btree_oom_tpcc.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
