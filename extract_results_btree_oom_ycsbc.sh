#!/bin/bash

# extract_results_btree_oom_ycsbc.sh - Extract and summarize larger-than-memory experiment results
# This script processes the TX and PERF CSV files to create a summary table for varying data sizes

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/larger-than-memory/btree_ycsbc"

# Data sizes to process
DATA_SIZES=(50000000 100000000 200000000 400000000)

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

# Function to extract leanstore/wiredtiger metrics from tx CSV (which contains console output)
extract_leanstore_metrics() {
    local tx_file="$1"
    local metric="$2"
    
    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi
    
    case "$metric" in
        "throughput")
            # Extract OLTP TX column for LAST 60 SECONDS ONLY
            # Skip lines with "~Transactions", "+---", "| t ", and lines with 0 throughput
            # Only consider the last 60 data points (1 per second)
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    tx = fields[3]
                    gsub(/^[ \t]+|[ \t]+$/, "", tx)
                    if (tx ~ /^[0-9]+$/ && tx > 0) {
                        values[count] = tx
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        # Only use last 60 seconds
                        start_idx = (count > 60) ? count - 60 : 0
                        sum = 0
                        samples = 0
                        for (i = start_idx; i < count; i++) {
                            sum += values[i]
                            samples++
                        }
                        if (samples > 0) {
                            print sum/samples
                        } else {
                            print "n/a"
                        }
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "l1_per_tx")
            # Extract L1/TX column average (LAST 60 SECONDS)
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    l1 = fields[11]
                    gsub(/^[ \t]+|[ \t]+$/, "", l1)
                    if (l1 ~ /^[0-9.]+/) {
                        values[count] = l1
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0
                        sum = 0
                        samples = 0
                        for (i = start_idx; i < count; i++) {
                            sum += values[i]
                            samples++
                        }
                        if (samples > 0) print sum/samples
                        else print "n/a"
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "llc_per_tx")
            # Extract LLC/TX column average (LAST 60 SECONDS)
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    llc = fields[12]
                    gsub(/^[ \t]+|[ \t]+$/, "", llc)
                    if (llc ~ /^[0-9.]+/) {
                        values[count] = llc
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0
                        sum = 0
                        samples = 0
                        for (i = start_idx; i < count; i++) {
                            sum += values[i]
                            samples++
                        }
                        if (samples > 0) print sum/samples
                        else print "n/a"
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "cycles_per_tx")
            # Extract Cycles/TX column average (LAST 60 SECONDS)
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    cycles = fields[9]
                    gsub(/^[ \t]+|[ \t]+$/, "", cycles)
                    if (cycles ~ /^[0-9.]+/) {
                        values[count] = cycles
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0
                        sum = 0
                        samples = 0
                        for (i = start_idx; i < count; i++) {
                            sum += values[i]
                            samples++
                        }
                        if (samples > 0) print sum/samples
                        else print "n/a"
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "insts_per_tx")
            # Extract Instrs/TX column average (LAST 60 SECONDS)
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    insts = fields[8]
                    gsub(/^[ \t]+|[ \t]+$/, "", insts)
                    if (insts ~ /^[0-9.]+/) {
                        values[count] = insts
                        count++
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0
                        sum = 0
                        samples = 0
                        for (i = start_idx; i < count; i++) {
                            sum += values[i]
                            samples++
                        }
                        if (samples > 0) print sum/samples
                        else print "n/a"
                    } else {
                        print "n/a"
                    }
                }
            ' "$tx_file"
            ;;
        "ipc")
            # IPC = instructions / cycles
            local insts=$(extract_leanstore_metrics "$tx_file" "insts_per_tx")
            local cycles=$(extract_leanstore_metrics "$tx_file" "cycles_per_tx")
            if [ "$insts" != "n/a" ] && [ "$cycles" != "n/a" ] && [ "$cycles" != "0" ]; then
                echo "scale=6; $insts / $cycles" | bc
            else
                echo "n/a"
            fi
            ;;
    esac
}

# Create summary CSV header
echo "datasize,config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process
configs=(
    "calico_mmap_nohuge"
    "calico_array_huge"
    "calico_lockfree_hash_huge"
    "leanstore"
    "wiredtiger"
)

echo "Extracting results from experiment files for all data sizes..."
echo "=================================================================="

# Process each data size
for datasize in "${DATA_SIZES[@]}"; do
    datasize_dir="${EXP_BASE_DIR}/${datasize}"
    
    if [ ! -d "$datasize_dir" ]; then
        echo "Warning: Data size directory $datasize_dir does not exist, skipping..."
        continue
    fi
    
    echo "Processing results for datasize ${datasize}..."
    
    # Create datasize-specific summary
    datasize_summary_csv="${datasize_dir}/results_summary.csv"
    echo "config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "$datasize_summary_csv"
    
    # Process each configuration for this data size
    for config in "${configs[@]}"; do
        echo "  Processing $config..."
        
        tx_file="${datasize_dir}/${config}_tx.csv"
        perf_file="${datasize_dir}/${config}_perf.csv"
        
        # Extract values based on configuration type
        if [ "$config" == "leanstore" ] || [ "$config" == "wiredtiger" ]; then
            # Leanstore and WiredTiger use custom parser for console table output
            avg_throughput=$(extract_leanstore_metrics "$tx_file" "throughput")
            llc_miss_per_op=$(extract_leanstore_metrics "$tx_file" "llc_per_tx")
            dtlb_miss_per_op="n/a"  # Not available in leanstore/wiredtiger output
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
        
        # Write to datasize-specific summary CSV
        echo "$config,$avg_throughput,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc" >> "$datasize_summary_csv"
        
        # Write to global summary CSV with data size
        echo "$datasize,$config,$avg_throughput,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc" >> "${EXP_BASE_DIR}/results_summary_all.csv"
    done
    
    echo "  Data size-specific summary: $datasize_summary_csv"
    echo
done

echo "=================================================================="
echo "Summary tables created:"
echo "  Global summary: ${EXP_BASE_DIR}/results_summary_all.csv"
for datasize in "${DATA_SIZES[@]}"; do
    datasize_summary="${EXP_BASE_DIR}/${datasize}/results_summary.csv"
    if [ -f "$datasize_summary" ]; then
        echo "  ${datasize} records: $datasize_summary"
    fi
done
echo
echo "Configurations processed:"
echo "  - calico_mmap_nohuge: without huge pages"
echo "  - calico_array_huge: with huge pages"
echo "  - calico_lockfree_hash_huge: with huge pages"
echo "  - leanstore"
echo "  - wiredtiger"
echo "  Total per data size: 5 configurations"
echo

echo "Global Results Summary (all data sizes):"
echo "--------------------------------------------"
if [ -f "${EXP_BASE_DIR}/results_summary_all.csv" ]; then
    column -t -s',' "${EXP_BASE_DIR}/results_summary_all.csv"
fi
echo

# Display individual data size summaries
for datasize in "${DATA_SIZES[@]}"; do
    datasize_summary="${EXP_BASE_DIR}/${datasize}/results_summary.csv"
    if [ -f "$datasize_summary" ]; then
        echo "Results for ${datasize} records:"
        echo "-----------------------------"
        column -t -s',' "$datasize_summary"
        echo
    fi
done
