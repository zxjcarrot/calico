#!/bin/bash

# exp_in_memory.sh - In-memory experiments for different Calico configurations
# This script runs experiments for various buffer management strategies with different thread counts

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/btree_ycsbc"

# Thread counts to test
THREAD_COUNTS=(1 4 8 16 32)

PHYSMEM=24
RUN_DURATION=60

# Common settings for all experiments (except THREADS which varies)
export RNDREAD=1
export BLOCK=/home/zxjcarrot/Workspace/calico/heapfile
export VIRTGB=32G
export PHYSGB=${PHYSMEM}G
export DATASIZE=1e8
export RUNFOR=${RUN_DURATION}
export PERF=1

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

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
    
    # Run leanstore and capture output
    build/frontend/ycsb \
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
    
    # Parse output to extract perf-like metrics
    # This will be done in extract script
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "Leanstore experiment with ${thread_count} threads completed."
    echo
}

# Function to run a single experiment
run_experiment() {
    local thread_count="$1"
    local config_name="$2"
    local description="$3"
    shift 3
    local env_vars=("$@")
    
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
        export ${env_var}
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
    echo "Starting experiment..."
    ./calico
    
    echo "Experiment ${config_name} with ${thread_count} threads completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
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
    run_experiment ${thread_count} "${base_name}_nohuge" \
        "${base_description} (no huge pages)" \
        "${base_env_vars[@]}" "NOHUGEPAGE_FRAME_MEM=1"
    
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
echo "Thread counts tested: ${THREAD_COUNTS[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Common settings: RNDREAD=1 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile VIRTGB=32G PHYSGB=24G DATASIZE=1e8 PERF=1" >> "${GLOBAL_SUMMARY_FILE}"
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
    "BLOCK": "/home/zxjcarrot/Workspace/calico",
    "VIRTGB": "32G",
    "PHYSGB": "24G", 
    "THREADS": ${threads},
    "DATASIZE": "1e8",
    "PERF": 1
  }
}
EOF
    
    echo "Running experiments with ${threads} threads..." >> "${GLOBAL_SUMMARY_FILE}"
    
    # Configuration 1: Calico array (traditional buffer pool with array translator)
    run_config_combinations ${threads} "calico_array" "Calico with traditional buffer pool using array translator" \
        "TRADITIONAL=1" "TRADHASH=0"

    # Configuration 2: Calico mmap (virtual memory cache)
    run_config_combinations ${threads} "calico_mmap" "Calico with mmap (virtual memory cache)" \
        "TRADITIONAL=0" "TRADHASH=0"

    # Configuration 3: Calico chained_hash (traditional buffer pool with unordered_map)
    run_config_combinations ${threads} "calico_chained_hash" "Calico with traditional buffer pool using chained hash (unordered_map) translator" \
        "TRADITIONAL=1" "TRADHASH=1"

    # Configuration 4: Calico openaddr_hash (traditional buffer pool with open addressing)
    run_config_combinations ${threads} "calico_openaddr_hash" "Calico with traditional buffer pool using open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=2"
    
    # Configuration 5: Calico lockfree_hash (traditional buffer pool with lock free open addressing)
    run_config_combinations ${threads} "calico_lockfree_hash" "Calico with traditional buffer pool using lock free open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=4"

    # Configuration 6: Leanstore
    run_leanstore_experiment ${threads}

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
