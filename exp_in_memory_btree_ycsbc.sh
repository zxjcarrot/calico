#!/bin/bash

# exp_in_memory.sh - In-memory experiments for different Calico configurations
# This script runs experiments for various buffer management strategies with different thread counts

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/btree_ycsbc"

# Thread counts to test
THREAD_COUNTS=(1 4 8 16 32)

# Common settings for all experiments (except THREADS which varies)
export RNDREAD=1
export BLOCK=/home/zxjcarrot/Workspace/calico/heapfile
export VIRTGB=32G
export PHYSGB=24G
export DATASIZE=1e8
export PERF=1

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

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
    
    # Experiment 1: Calico mmap (virtual memory cache)
    run_experiment ${threads} "calico_mmap" "Calico with mmap (virtual memory cache)" \
        "TRADITIONAL=0" "TRADHASH=0"

    # Experiment 2: Calico array (traditional buffer pool with array translator)
    run_experiment ${threads} "calico_array" "Calico with traditional buffer pool using array translator" \
        "TRADITIONAL=1" "TRADHASH=0"

    # Experiment 3: Calico array + nohuge (array translator without huge pages)
    run_experiment ${threads} "calico_array_nohuge" "Calico array translator without huge pages for frame memory" \
        "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=1"

    # Experiment 4: Calico unordered_hash (traditional buffer pool with unordered_map)
    run_experiment ${threads} "calico_unordered_hash" "Calico with traditional buffer pool using unordered_map translator" \
        "TRADITIONAL=1" "TRADHASH=1"

    # Experiment 5: Calico openaddr_hash (traditional buffer pool with open addressing)
    run_experiment ${threads} "calico_openaddr_hash" "Calico with traditional buffer pool using open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=2"
    
    # Experiment 6: Calico lockfree_hash (traditional buffer pool with lock free open addressing)
    run_experiment ${threads} "calico_lockfree_hash" "Calico with traditional buffer pool using lock free open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=4"

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
