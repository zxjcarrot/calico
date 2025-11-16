#!/bin/bash

# exp_larger_than_memory_btree_tpcc.sh - Larger-than-memory experiments with varying data sizes
# This script runs TPCC experiments with data sizes ranging from 10 to 400 warehouses

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/btree_tpcc"

# Data sizes to test (in warehouses)
DATA_SIZES=(20 40 80 100 200 400)

# Fixed thread count and memory
THREAD_COUNT=96
PHYSMEM=8
RUN_DURATION=360

# Common settings for all experiments
export RNDREAD=0 # 0 for TPC-C
export BLOCK=/ssd/ssd0/xj/calico/heapfile
export VIRTGB=128G
export PHYSGB=${PHYSMEM}G
export THREADS=${THREAD_COUNT}
export RUNFOR=${RUN_DURATION}
export PERF=0

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

# Function to run leanstore experiment
run_leanstore_experiment() {
    local datasize="$1"
    local datasize_dir="${EXP_BASE_DIR}/${datasize}"
    mkdir -p "${datasize_dir}"
    
    local config_name="leanstore"
    local tx_file="${datasize_dir}/${config_name}_tx.csv"
    local perf_file="${datasize_dir}/${config_name}_perf.csv"
    local iostat_file="${datasize_dir}/${config_name}_iostat.log"
    
    iostat -x 1 nvme1n1 > "${iostat_file}" &
    iostat_pid=$!
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
    # Run leanstore and capture output, rerun if failed
    until build/frontend/tpcc \
        --tpcc_warehouse_count=${datasize} \
        --ssd_path=build/heapfile \
        --worker_threads=${THREAD_COUNT} \
        --pp_threads=4 \
        --dram_gib=${PHYSMEM} \
        --csv_path=./log \
        --wal=0 \
        --vi=0 \
        --tpcc_warehouse_affinity=0 \
        --free_pct=1 \
        --print_tx_console \
        --run_for_seconds=${RUN_DURATION} \
        --isolation_level=rc > "${tx_file}" 2>&1
    do
        echo "tpcc failed, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "Leanstore experiment with datasize ${datasize} completed."
    echo
    kill "${iostat_pid}"
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
    
    iostat -x 1 nvme1n1 > "${iostat_file}" &
    iostat_pid=$!

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

    # Run WiredTiger and capture output, rerun if failed
    until build/frontend/wiredtiger_tpcc \
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
        --isolation_level=si > "${tx_file}" 2>&1 # must run in si mode
    do
        echo "wiredtiger_tpcc failed, retrying in 5 seconds..."
        sleep 5
    done
    
    # Create empty perf file placeholder
    touch "${perf_file}"
    
    cd "${SCRIPT_DIR}"
    echo "WiredTiger experiment with datasize ${datasize} completed."
    echo
    kill "${iostat_pid}"
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
    
    echo "Data size: ${datasize} warehouses"
    echo "Thread count: ${THREAD_COUNT}"
    echo "Memory: ${PHYSMEM}G"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo "  IOSTAT LOG: ${iostat_file}"
    echo
    
    # Run the experiment
    echo "Starting experiment..."
    iostat -x 1 nvme1n1 > "${iostat_file}" &
    iostat_pid=$!

    ./calico
    
    kill "${iostat_pid}"
    echo "Experiment ${config_name} with datasize ${datasize} completed successfully."
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
echo "Larger-than-Memory TPCC Experiment Summary" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Data sizes tested (warehouses): ${DATA_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Fixed settings: THREADS=${THREAD_COUNT} PHYSGB=${PHYSMEM}G VIRTGB=128G RNDREAD=0 PERF=0" >> "${GLOBAL_SUMMARY_FILE}"
echo "Configurations: calico_mmap_nohuge, calico_array_huge, calico_lockfree_hash_huge, leanstore, wiredtiger" >> "${GLOBAL_SUMMARY_FILE}"
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
    "BLOCK": "/ssd/ssd0/xj/calico/heapfile",
    "VIRTGB": "128G",
    "PHYSGB": "${PHYSMEM}G",
    "THREADS": ${THREAD_COUNT},
    "DATASIZE": ${datasize},
    "PERF": 0
  }
}
EOF
    
    echo "Running experiments with datasize ${datasize} warehouses..." >> "${GLOBAL_SUMMARY_FILE}"


    # Configuration 5: WiredTiger
    run_wiredtiger_experiment ${datasize}

    # Configuration 4: Leanstore
    run_leanstore_experiment ${datasize}


    # Configuration 1: Calico mmap without huge pages
    run_experiment ${datasize} "calico_mmap_nohuge" \
        "Calico with mmap (virtual memory cache) without huge pages" \
        "TRADITIONAL=0" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=1"

 
    # Configuration 3: Calico openaddr_hash with huge pages
    run_experiment ${datasize} "calico_openaddr_hash_huge" \
        "Calico with traditional buffer pool using lock free open addressing hash translator with huge pages" \
        "TRADITIONAL=1" "TRADHASH=2" "NOHUGEPAGE_FRAME_MEM=0"

    # Configuration 2: Calico array with huge pages
    run_experiment ${datasize} "calico_array_huge" \
        "Calico with traditional buffer pool using array translator with huge pages" \
        "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=0"


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
echo "  - calico_lockfree_hash_huge: with huge pages"
echo "  - leanstore"
echo "  - wiredtiger"
echo "  Total per data size: 5 experiment configurations"
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
# This needs to be adapted for TPCC results
# ./extract_results_btree_oom_ycsbc.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
