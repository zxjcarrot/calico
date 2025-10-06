#!/bin/bash

# extract_results.sh - Extract and summarize experiment results
# This script processes the TX and PERF CSV files to create a summary table for multi-threaded experiments

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/btree_ycsbc"

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

# Function to extract leanstore metrics from tx CSV (which contains console output)
extract_leanstore_metrics() {
    local tx_file="$1"
    local metric="$2"
    
    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi
    
    case "$metric" in
        "throughput")
            # Extract OLTP TX column, skip header and summary rows, compute average
            # Skip lines with "~Transactions", "+---", "| t ", and lines with 0 throughput
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    tx = fields[3]
                    gsub(/^[ \t]+|[ \t]+$/, "", tx)
                    if (tx ~ /^[0-9]+$/ && tx > 0) {
                        sum += tx
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        # Return average transactions per second (same format as Calico)
                        print sum/count
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "l1_per_tx")
            # Extract L1/TX column average
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    l1 = fields[11]
                    gsub(/^[ \t]+|[ \t]+$/, "", l1)
                    if (l1 ~ /^[0-9.]+/) {
                        sum += l1
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            ' "$tx_file"
            ;;
        "llc_per_tx")
            # Extract LLC/TX column average
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    llc = fields[12]
                    gsub(/^[ \t]+|[ \t]+$/, "", llc)
                    if (llc ~ /^[0-9.]+/) {
                        sum += llc
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            ' "$tx_file"
            ;;
        "cycles_per_tx")
            # Extract Cycles/TX column average
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    cycles = fields[9]
                    gsub(/^[ \t]+|[ \t]+$/, "", cycles)
                    if (cycles ~ /^[0-9.]+/) {
                        sum += cycles
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            ' "$tx_file"
            ;;
        "insts_per_tx")
            # Extract Instrs/TX column average
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    insts = fields[8]
                    gsub(/^[ \t]+|[ \t]+$/, "", insts)
                    if (insts ~ /^[0-9.]+/) {
                        sum += insts
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            ' "$tx_file"
            ;;
        "ipc")
            # IPC = instructions / cycles
            local insts=$(extract_leanstore_metrics "$tx_file" "insts_per_tx")
            local cycles=$(extract_leanstore_metrics "$tx_file" "cycles_per_tx")
            if [ "$insts" != "n/a" ] && [ "$cycles" != "n/a" ] && [ "$cycles" != "0" ]; then
                echo "scale=4; $insts / $cycles" | bc
            else
                echo "n/a"
            fi
            ;;
    esac
}

# Create summary CSV header
echo "thread_count,config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process (huge page variations)
configs=(
    "calico_array_nohuge" "calico_array_huge"
    "calico_mmap_nohuge" "calico_mmap_huge"
    "calico_chained_hash_nohuge" "calico_chained_hash_huge"
    "calico_openaddr_hash_nohuge" "calico_openaddr_hash_huge"
    "calico_lockfree_hash_nohuge" "calico_lockfree_hash_huge"
    "leanstore"
)

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
        
        # Extract values based on configuration type
        if [ "$config" == "leanstore" ]; then
            # Leanstore uses custom parser for console table output
            avg_throughput=$(extract_leanstore_metrics "$tx_file" "throughput")
            llc_miss_per_op=$(extract_leanstore_metrics "$tx_file" "llc_per_tx")
            dtlb_miss_per_op="n/a"  # Not available in leanstore output
            cycles_per_op=$(extract_leanstore_metrics "$tx_file" "cycles_per_tx")
            insts_per_op=$(extract_leanstore_metrics "$tx_file" "insts_per_tx")
            ipc=$(extract_leanstore_metrics "$tx_file" "ipc")
        else
            # Calico configurations use standard CSV parsers
            avg_throughput=$(extract_avg_throughput "$tx_file")
            llc_miss_per_op=$(extract_perf_value "$perf_file" "llc_misses")
            dtlb_miss_per_op=$(extract_perf_value "$perf_file" "dtlb_misses")
            cycles_per_op=$(extract_perf_value "$perf_file" "cycles")
            insts_per_op=$(extract_perf_value "$perf_file" "instructions")
            ipc=$(extract_ipc_value "$perf_file")
        fi
        
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
echo "Configuration combinations processed:"
echo "  - calico_array: 2 combinations (nohuge, huge)"
echo "  - calico_mmap: 2 combinations (nohuge, huge)"
echo "  - calico_chained_hash: 2 combinations (nohuge, huge)"
echo "  - calico_openaddr_hash: 2 combinations (nohuge, huge)"
echo "  - calico_lockfree_hash: 2 combinations (nohuge, huge)"
echo "  - leanstore: 1 configuration"
echo "  Total per thread count: 10 configurations"
echo

echo "Global Results Summary (all thread counts):"
echo "--------------------------------------------"
if [ -f "${EXP_BASE_DIR}/results_summary_all.csv" ]; then
    column -t -s',' "${EXP_BASE_DIR}/results_summary_all.csv" | head -100
    total_lines=$(wc -l < "${EXP_BASE_DIR}/results_summary_all.csv")
    if [ "$total_lines" -gt 100 ]; then
        echo "... (showing first 100 lines, total ${total_lines} lines)"
        echo "View complete results: ${EXP_BASE_DIR}/results_summary_all.csv"
    fi
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
