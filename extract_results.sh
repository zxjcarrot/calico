#!/bin/bash

# extract_results.sh - Extract and summarize experiment results
# This script processes the TX and PERF CSV files to create a summary table for multi-threaded experiments

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory"

# Thread counts to process
THREAD_COUNTS=(1 4 8 16 32)

# Check if experiment directory exists
if [ ! -d "${EXP_BASE_DIR}" ]; then
    echo "Error: Experiment directory ${EXP_BASE_DIR} does not exist"
    exit 1
fi

# Function to extract value from perf CSV by label
extract_perf_value() {
    local perf_file="$1"
    local label="$2"
    
    if [ ! -f "$perf_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the per_op value for the given label
    awk -F',' -v label="$label" '$1 == label { print $3 }' "$perf_file"
}

# Function to extract IPC from perf CSV (uses total column, not per_op)
extract_ipc_value() {
    local perf_file="$1"
    
    if [ ! -f "$perf_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the total value for derived_ipc (column 2)
    awk -F',' '$1 == "derived_ipc" { print $2 }' "$perf_file"
}

# Function to extract avg_throughput from tx CSV
extract_avg_throughput() {
    local tx_file="$1"
    
    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the throughput value from the avg_throughput row
    awk -F',' '$1 == "avg_throughput" { print $2 }' "$tx_file"
}

# Create summary CSV header
echo "thread_count,config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process
configs=("calico_mmap" "calico_array" "calico_array_nohuge" "calico_unordered_hash" "calico_openaddr_hash")

echo "Extracting results from experiment files for all thread counts..."
echo "=================================================================="

# Process each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    thread_dir="${EXP_BASE_DIR}/${threads}"
    
    if [ ! -d "$thread_dir" ]; then
        echo "Warning: Thread directory $thread_dir does not exist, skipping..."
        continue
    fi
    
    echo "Processing results for ${threads} threads..."
    
    # Create thread-specific summary
    thread_summary_csv="${thread_dir}/results_summary.csv"
    echo "config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "$thread_summary_csv"
    
    # Process each configuration for this thread count
    for config in "${configs[@]}"; do
        echo "  Processing $config..."
        
        tx_file="${thread_dir}/${config}_tx.csv"
        perf_file="${thread_dir}/${config}_perf.csv"
        
        # Extract values
        avg_throughput=$(extract_avg_throughput "$tx_file")
        llc_miss_per_op=$(extract_perf_value "$perf_file" "llc_misses")
        dtlb_miss_per_op=$(extract_perf_value "$perf_file" "dtlb_misses")
        cycles_per_op=$(extract_perf_value "$perf_file" "cycles")
        insts_per_op=$(extract_perf_value "$perf_file" "instructions")
        ipc=$(extract_ipc_value "$perf_file")
        
        # Handle empty values
        avg_throughput=${avg_throughput:-"n/a"}
        llc_miss_per_op=${llc_miss_per_op:-"n/a"}
        dtlb_miss_per_op=${dtlb_miss_per_op:-"n/a"}
        cycles_per_op=${cycles_per_op:-"n/a"}
        insts_per_op=${insts_per_op:-"n/a"}
        ipc=${ipc:-"n/a"}
        
        # Write to thread-specific summary CSV
        echo "$config,$avg_throughput,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc" >> "$thread_summary_csv"
        
        # Write to global summary CSV with thread count
        echo "$threads,$config,$avg_throughput,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc" >> "${EXP_BASE_DIR}/results_summary_all.csv"
    done
    
    echo "  Thread-specific summary: $thread_summary_csv"
    echo
done

echo "=================================================================="
echo "Summary tables created:"
echo "  Global summary: ${EXP_BASE_DIR}/results_summary_all.csv"
for threads in "${THREAD_COUNTS[@]}"; do
    thread_summary="${EXP_BASE_DIR}/${threads}/results_summary.csv"
    if [ -f "$thread_summary" ]; then
        echo "  ${threads} threads: $thread_summary"
    fi
done
echo

echo "Global Results Summary (all thread counts):"
echo "--------------------------------------------"
if [ -f "${EXP_BASE_DIR}/results_summary_all.csv" ]; then
    column -t -s',' "${EXP_BASE_DIR}/results_summary_all.csv"
fi
echo

# Display individual thread summaries
for threads in "${THREAD_COUNTS[@]}"; do
    thread_summary="${EXP_BASE_DIR}/${threads}/results_summary.csv"
    if [ -f "$thread_summary" ]; then
        echo "Results for ${threads} threads:"
        echo "-----------------------------"
        column -t -s',' "$thread_summary"
        echo
    fi
done
