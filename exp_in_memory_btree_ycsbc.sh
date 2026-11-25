#!/bin/bash

# exp_in_memory.sh - In-memory experiments for different Calico configurations
# This script runs experiments for various buffer management strategies with different thread counts

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
EXP_BASE_DIR="${EXP_BASE_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/in-memory/btree_ycsbc}"

default_block_path() {
    if [ -n "${BLOCK:-}" ]; then
        printf '%s\n' "$BLOCK"
        return 0
    fi

    if [ -f "${SCRIPT_DIR}/heapfile" ]; then
        printf '%s\n' "${SCRIPT_DIR}/heapfile"
        return 0
    fi

    printf '%s\n' "${SCRIPT_DIR}/heapfile.predicache_bench"
}

ensure_block_file() {
    local block_path="$1"
    local block_size="${2:-20G}"

    if [ -f "$block_path" ]; then
        return 0
    fi

    echo "Creating sparse block file at ${block_path} (${block_size})..."
    if ! fallocate -l "$block_size" "$block_path"; then
        truncate -s "$block_size" "$block_path"
    fi
}

# Thread counts to test
#THREAD_COUNTS=(64 128)
THREAD_COUNTS=(128 64 32 16 8 4 1)
#THREAD_COUNTS=(128)
##THREAD_COUNTS=(1 4 8)
PHYSMEM=32
SMALL_POOLGB=16
RUN_DURATION=60

# Common settings for all experiments (except THREADS which varies)
export RNDREAD=1
export BLOCK="$(default_block_path)"
ensure_block_file "$BLOCK" "20G"
export VIRTGB=32G
export POOLGB=${PHYSMEM}G
export PHYSGB=${PHYSMEM}G
export DATASIZE=1e8
export RUNFOR=${RUN_DURATION}
export PERF=1
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

run_with_numa_prefix() {
    local -a cmd_prefix=()
    mapfile -t cmd_prefix < <(numa_command_prefix)
    "${cmd_prefix[@]}"  "$@"
    # "$@"
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
make clean
make -j$(nproc)

echo "Building predicache..."
make -C "${SCRIPT_DIR}/predictivetranslation-sigmod2026" clean
make -C "${SCRIPT_DIR}/predictivetranslation-sigmod2026" predicache_oltp_query -j$(nproc)

build_required_leanstore_frontends

# Function to run leanstore experiment
run_leanstore_experiment() {
    local thread_count="$1"
    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"
    
    local config_name="leanstore"
    local tx_file="${thread_dir}/${config_name}_tx.csv"
    local perf_file="${thread_dir}/${config_name}_perf.csv"
    
    echo "=================================================================="
    echo "Running leanstore experiment with ${thread_count} threads"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"
    
    # Ensure heapfile exists
    if [ ! -f "build/heapfile" ]; then
        echo "Creating leanstore heapfile..."
        fallocate -l 20G build/heapfile
    fi
    
    echo "Running leanstore ycsb..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"
    
    # Run leanstore and capture output, rerun if failed
    until run_with_numa_prefix build/frontend/ycsb \
        --ycsb_read_ratio=100 \
        --ycsb_tuple_count=100000000 \
        --ssd_path=build/heapfile \
        --worker_threads=${thread_count} \
        --pp_threads=1 \
        --dram_gib=${PHYSMEM} \
        --csv_path=./log \
        --free_pct=1 \
        --contention_split \
        --xmerge \
        --print_tx_console \
        --run_for_seconds=${RUN_DURATION} \
        --isolation_level=ru > "${tx_file}" 2>&1
    do
        echo "leanstore_ycsb failed, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "Leanstore experiment with ${thread_count} threads completed."
    echo
}

# Function to run WiredTiger experiment
run_wiredtiger_experiment() {
    local thread_count="$1"
    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"
    
    local config_name="wiredtiger"
    local tx_file="${thread_dir}/${config_name}_tx.csv"
    local perf_file="${thread_dir}/${config_name}_perf.csv"

    echo "=================================================================="
    echo "Running WiredTiger experiment with ${thread_count} threads"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"

    # Remove old WiredTiger database directory
    rm -rf build/wt_db
    mkdir build/wt_db
    echo "Running WiredTiger ycsb..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"

    if ! wiredtiger_frontend_available "build/frontend/wiredtiger_ycsb" "wiredtiger_ycsb"; then
        printf '%s\n' 'WiredTiger experiment skipped: build/frontend/wiredtiger_ycsb is unavailable. Install libwiredtiger-dev and rebuild leanstore frontends.' > "${tx_file}"
        touch "${perf_file}"
        cd "${SCRIPT_DIR}"
        echo "WiredTiger experiment with ${thread_count} threads skipped."
        echo
        return 0
    fi

    # Run WiredTiger and capture output, rerun if failed
    until run_with_numa_prefix build/frontend/wiredtiger_ycsb \
        --ycsb_read_ratio=100 \
        --ycsb_tuple_count=100000000 \
        --ssd_path=build/wt_db \
        --worker_threads=${thread_count} \
        --pp_threads=1 \
        --dram_gib=${PHYSMEM} \
        --csv_path=./log \
        --free_pct=1 \
        --wal=false \
        --contention_split \
        --xmerge \
        --print_tx_console \
        --run_for_seconds=${RUN_DURATION} \
        --isolation_level=si > "${tx_file}" 2>&1 # must run in si mode
    do
        echo "wiredtiger_ycsb failed, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "WiredTiger experiment with ${thread_count} threads completed."
    echo
}



# Function to run WiredTiger experiment
run_lmdb_experiment() {
    local thread_count="$1"
    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"
    
    local config_name="lmdb"
    local tx_file="${thread_dir}/${config_name}_tx.csv"
    local perf_file="${thread_dir}/${config_name}_perf.csv"

    echo "=================================================================="
    echo "Running LMDB experiment with ${thread_count} threads"
    echo "=================================================================="
    
    cd "${SCRIPT_DIR}/leanstore"

    # Remove old LMDB database directory
    rm -rf lmdb_db
    mkdir lmdb_db
    echo "Running LMDB ycsb..."
    echo "Output files:"
    echo "  TX CSV: ${tx_file}"
    echo "  PERF CSV: ${perf_file}"

    if ! lmdb_frontend_available "build/frontend/lmdb_ycsb"; then
        printf '%s\n' 'LMDB experiment skipped: build/frontend/lmdb_ycsb cannot load liblmdb.so. Install liblmdb-dev and rebuild leanstore frontends.' > "${tx_file}"
        touch "${perf_file}"
        cd "${SCRIPT_DIR}"
        echo "LMDB experiment with ${thread_count} threads skipped."
        echo
        return 0
    fi

    # Run WiredTiger and capture output, rerun if failed
    until run_with_numa_prefix build/frontend/lmdb_ycsb \
        --ycsb_read_ratio=100 \
        --ycsb_tuple_count=100000000 \
        --ssd_path=./lmdb_db \
        --worker_threads=${thread_count} \
        --pp_threads=1 \
        --dram_gib=${PHYSMEM} \
        --csv_path=./log \
        --free_pct=1 \
        --print_tx_console \
        --run_for_seconds=${RUN_DURATION} > "${tx_file}" 2>&1
    do
        echo "lmdb_ycsb failed, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "WiredTiger experiment with ${thread_count} threads completed."
    echo
}

# Function to run a single experiment
run_experiment() {
    local thread_count="$1"
    local config_name="$2"
    local description="$3"
    shift 3
    local env_vars=("$@")
    local -a previous_env_names=()
    local -a previous_env_values=()
    local -a previous_env_was_set=()
    
    # Create thread-specific directory
    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"
    
    echo "=================================================================="
    echo "Running experiment: ${config_name} with ${thread_count} threads"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Set thread count
    export THREADS=${thread_count}
    
    # Set environment variables for this configuration
    for env_var in "${env_vars[@]}"; do
        local var_name="${env_var%%=*}"
        if [[ -v $var_name ]]; then
            previous_env_was_set+=("1")
            previous_env_values+=("${!var_name}")
        else
            previous_env_was_set+=("0")
            previous_env_values+=("")
        fi
        previous_env_names+=("${var_name}")
        export "${env_var}"
        echo "Set ${env_var}"
    done
    
    # Set output file paths in thread-specific directory
    export TX_CSV="${thread_dir}/${config_name}_tx.csv"
    export PERF_CSV="${thread_dir}/${config_name}_perf.csv"
    
    echo "Thread count: ${thread_count}"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo
    
    # Run the experiment
    local calico_executable
    calico_executable="$(calico_executable_for_config "${config_name}")"
    echo "Starting experiment..."
    echo "Executable: ${calico_executable}"
    run_with_numa_prefix "${calico_executable}"
    
    echo "Experiment ${config_name} with ${thread_count} threads completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    local restore_idx
    for restore_idx in "${!previous_env_names[@]}"; do
        local restore_name="${previous_env_names[$restore_idx]}"
        if [ "${previous_env_was_set[$restore_idx]}" = "1" ]; then
            export "${restore_name}=${previous_env_values[$restore_idx]}"
        else
            unset "${restore_name}"
        fi
    done
}

# Function to run a single predicache experiment
run_predicache_experiment() {
    local thread_count="$1"
    local config_name="$2"
    local description="$3"
    local executable="$4"
    shift 4
    local env_vars=("$@")
    local -a previous_env_names=()
    local -a previous_env_values=()
    local -a previous_env_was_set=()

    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"

    echo "=================================================================="
    echo "Running predicache experiment: ${config_name} with ${thread_count} threads"
    echo "Description: ${description}"
    echo "Executable: ${executable}"
    echo "=================================================================="

    export THREADS=${thread_count}

    for env_var in "${env_vars[@]}"; do
        local var_name="${env_var%%=*}"
        if [[ -v $var_name ]]; then
            previous_env_was_set+=("1")
            previous_env_values+=("${!var_name}")
        else
            previous_env_was_set+=("0")
            previous_env_values+=("")
        fi
        previous_env_names+=("${var_name}")
        export "${env_var}"
        echo "Set ${env_var}"
    done

    export TX_CSV="${thread_dir}/${config_name}_tx.csv"
    export PERF_CSV="${thread_dir}/${config_name}_perf.csv"

    echo "Thread count: ${thread_count}"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo

    echo "Starting predicache experiment..."
    run_with_numa_prefix "${executable}"

    # Create empty perf file placeholder
    touch "${PERF_CSV}"

    echo "Predicache experiment ${config_name} with ${thread_count} threads completed successfully."
    echo

    local restore_idx
    for restore_idx in "${!previous_env_names[@]}"; do
        local restore_name="${previous_env_names[$restore_idx]}"
        if [ "${previous_env_was_set[$restore_idx]}" = "1" ]; then
            export "${restore_name}=${previous_env_values[$restore_idx]}"
        else
            unset "${restore_name}"
        fi
    done
}

# Function to run experiments for a specific base configuration with huge page variations
run_config_combinations() {
    local thread_count="$1"
    local base_name="$2"
    local base_description="$3"
    shift 3
    local base_env_vars=("$@")
    
    echo "=================================================================="
    echo "Running huge page combinations for: ${base_name} with ${thread_count} threads"
    echo "=================================================================="
    
    # Combination 1: No huge pages
    # run_experiment ${thread_count} "${base_name}_nohuge" \
    #     "${base_description} (no huge pages)" \
    #     "${base_env_vars[@]}" "NOHUGEPAGE_FRAME_MEM=1"
    
    # Combination 2: With huge pages
    run_experiment ${thread_count} "${base_name}_huge" \
        "${base_description} (with huge pages)" \
        "${base_env_vars[@]}" "NOHUGEPAGE_FRAME_MEM=0"
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "In-Memory Experiment Summary - Multi-threaded" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
#echo "Thread counts tested: ${THREAD_COUNTS[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Common settings: RNDREAD=1 BLOCK=${BLOCK} VIRTGB=32G PHYSGB=24G DATASIZE=1e8 PERF=1" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

# Run experiments for each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH ${threads} THREADS"
    echo "######################################################################"
    
    # Create thread-specific config file
    thread_dir="${EXP_BASE_DIR}/${threads}"
    mkdir -p "${thread_dir}"
    
    cat > "${thread_dir}/config.json" <<EOF
{
  "description": "Configuration for ${threads} threads experiments",
  "threads": ${threads},
  "common_settings": {
    "RNDREAD": 1,
    "BLOCK": "${BLOCK}",
    "VIRTGB": "32G",
    "PHYSGB": "24G", 
    "THREADS": ${threads},
    "DATASIZE": "1e8",
    "PERF": 1
  }
}
EOF
    
    echo "Running experiments with ${threads} threads..." >> "${GLOBAL_SUMMARY_FILE}"

    # Configuration 6: Leanstore
    run_leanstore_experiment ${threads}

    # # Configuration 7: Wiredtiger
    run_wiredtiger_experiment ${threads}

    # Configuration 0: PrediCache (random read baseline)
    run_predicache_experiment ${threads} "predicache" "PrediCache random read" \
        "${SCRIPT_DIR}/predictivetranslation-sigmod2026/predicache_oltp_query" \
        "POOLGB=${SMALL_POOLGB}G"


    run_lmdb_experiment ${threads}

    #Configuration 1: Calico array (traditional buffer pool with array translator)
    run_config_combinations ${threads} "calico_array" "Calico with traditional buffer pool using array translator" \
        "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_TRANSLATION_ARRAY=0" "POOLGB=${SMALL_POOLGB}G"

    # # Configuration 1: Calico array (traditional buffer pool with 2-level array translator)
    # run_config_combinations ${threads} "calico_array_2level" "Calico with traditional buffer pool using array translator" \
    #     "TRADITIONAL=1" "TRADHASH=5" "NOHUGEPAGE_TRANSLATION_ARRAY=0"

    # # Configuration 1: Calico array (traditional buffer pool with 3-level array translator)
    # run_config_combinations ${threads} "calico_array_3level" "Calico with traditional buffer pool using array translator" \
    #     "TRADITIONAL=1" "TRADHASH=6" "NOHUGEPAGE_TRANSLATION_ARRAY=0"


    # Configuration 2: Calico mmap (virtual memory cache)
    run_config_combinations ${threads} "calico_mmap" "Calico with mmap (virtual memory cache)" \
        "TRADITIONAL=0" "TRADHASH=0" "POOLGB=${SMALL_POOLGB}G"

    # # # Configuration 3: Calico chained_hash (traditional buffer pool with unordered_map)
    # run_config_combinations ${threads} "calico_chained_hash" "Calico with traditional buffer pool using chained hash (unordered_map) translator" \
    #     "TRADITIONAL=1" "TRADHASH=1"

    # # Configuration 4: Calico openaddr_hash (traditional buffer pool with open addressing)
    # run_config_combinations ${threads} "calico_openaddr_hash" "Calico with traditional buffer pool using open addressing hash translator" \
    #     "TRADITIONAL=1" "TRADHASH=2"

    # Configuration 5: Robin Hood hash (martinus/robin-hood-hashing)
    run_config_combinations ${threads} "calico_robinhood_hash" "Calico with traditional buffer pool using Robin Hood hash translator" \
        "TRADITIONAL=1" "TRADHASH=7"
    
    # # # Configuration 6: Calico lockfree_hash (traditional buffer pool with lock free open addressing)
    # run_config_combinations ${threads} "calico_lockfree_hash" "Calico with traditional buffer pool using lock free open addressing hash translator" \
    #     "TRADITIONAL=1" "TRADHASH=4"


    echo "Completed experiments with ${threads} threads" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH ${threads} THREADS"
    echo "######################################################################"
    echo
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Configuration combinations tested per thread count:"
echo "  - calico_array: 2 combinations (nohuge, huge)"
echo "  - calico_mmap: 2 combinations (nohuge, huge)"
echo "  - calico_chained_hash: 2 combinations (nohuge, huge)"
echo "  - calico_openaddr_hash: 2 combinations (nohuge, huge)"
echo "  - calico_robinhood_hash: 2 combinations (nohuge, huge)"
echo "  - calico_lockfree_hash: 2 combinations (nohuge, huge)"
echo "  Total per thread count: 10 experiment configurations"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by thread count:"
for threads in "${THREAD_COUNTS[@]}"; do
    echo "--- ${threads} threads ---"
    find "${EXP_BASE_DIR}/${threads}" -name "*.csv" -o -name "*.json" | sort
done

echo
echo "Running results extraction for all thread counts..."
./extract_results_btree_ycsbc.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
