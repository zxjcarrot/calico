#!/bin/bash

# extract_results.sh - Extract and summarize experiment results
# This script processes the TX and PERF CSV files to create a summary table for multi-threaded experiments

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${EXP_BASE_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/in-memory/tpcc}"

# Thread counts to process
THREAD_COUNTS=(1 4 8 16 32 64 128)

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

# Function to extract avg_throughput from tx CSV, considering only the last 30 seconds
extract_avg_throughput() {
    local tx_file="$1"
    
    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi
    
    # Try extracting avg_throughput key first (for Calico)
    local avg=$(awk -F',' '$1 == "avg_throughput" { print $2 }' "$tx_file")
    if [ ! -z "$avg" ]; then
        echo "$avg"
        return
    fi

    # Fallback to time-series averaging (for Predicache)
    max_t=$(awk -F',' '/^[0-9]/ {print $1}' "$tx_file" | sort -nr | head -n1)
    if [ -z "$max_t" ]; then
        echo "n/a"
        return
    fi

    # Calculate the start time for the last 30 seconds
    start_t=$((max_t - 30))

    # Extract the throughput value, averaging over the last 30 seconds
    # Assumes throughput is the 2nd column for Calico TPCC results
    awk -F',' -v start_t="$start_t" '/^[0-9]/ && $1 > start_t { sum += $2; count++ } END { if (count > 0) print sum/count; else print "n/a" }' "$tx_file"
}

# Function to extract leanstore metrics from tx CSV (which contains console output)
extract_leanstore_metrics() {
    local tx_file="$1"
    local metric="$2"
    
    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi
    
    local csv_column
    local require_positive="0"
    local treat_all_zero_as_na="1"

    case "$metric" in
        "throughput")
            csv_column=2
            require_positive="1"
            treat_all_zero_as_na="0"
            ;;
        "insts_per_tx")
            csv_column=8
            ;;
        "cycles_per_tx")
            csv_column=9
            ;;
        "l1_per_tx")
            csv_column=11
            ;;
        "llc_per_tx")
            csv_column=12
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
        *)
            echo "n/a"
            ;;
    esac

    if [ -n "${csv_column:-}" ]; then
        local csv_value
        csv_value=$(awk -F',' -v column="$csv_column" -v require_positive="$require_positive" -v treat_all_zero_as_na="$treat_all_zero_as_na" '
            NF == 17 && $1 ~ /^[0-9]+$/ && $2 ~ /^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ && $3 == 100 && $4 == 0 {
                t = $1 + 0
                value = $column + 0
                if (t > max_t) {
                    max_t = t
                }
                timestamps[t] = 1
                values[t] = value
            }
            END {
                if (max_t == "") {
                    print "n/a"
                    exit
                }
                start_t = max_t - 30
                for (t in timestamps) {
                    if ((t + 0) > start_t) {
                        value = values[t]
                        if (require_positive == 1) {
                            if (value > 0) {
                                sum += value
                                count++
                            }
                        } else {
                            sum += value
                            count++
                            if (value != 0) {
                                non_zero_count++
                            }
                        }
                    }
                }
                if (count == 0) {
                    print "n/a"
                } else if (treat_all_zero_as_na == 1 && non_zero_count == 0) {
                    print "n/a"
                } else {
                    print sum / count
                }
            }
        ' "$tx_file")
        if [ "$csv_value" != "n/a" ]; then
            echo "$csv_value"
            return
        fi
    fi

    case "$metric" in
        "throughput")
            tail -n 30 "$tx_file" | awk '
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
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            '
            ;;
        "l1_per_tx")
            tail -n 30 "$tx_file" | awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    value = fields[11]
                    gsub(/^[ \t]+|[ \t]+$/, "", value)
                    if (value ~ /^[0-9.]+/) {
                        sum += value
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            '
            ;;
        "llc_per_tx")
            tail -n 30 "$tx_file" | awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    value = fields[12]
                    gsub(/^[ \t]+|[ \t]+$/, "", value)
                    if (value ~ /^[0-9.]+/) {
                        sum += value
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            '
            ;;
        "cycles_per_tx")
            tail -n 30 "$tx_file" | awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    value = fields[9]
                    gsub(/^[ \t]+|[ \t]+$/, "", value)
                    if (value ~ /^[0-9.]+/) {
                        sum += value
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            '
            ;;
        "insts_per_tx")
            tail -n 30 "$tx_file" | awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|")
                    value = fields[8]
                    gsub(/^[ \t]+|[ \t]+$/, "", value)
                    if (value ~ /^[0-9.]+/) {
                        sum += value
                        count++
                    }
                }
                END {
                    if (count > 0) print sum/count
                    else print "n/a"
                }
            '
            ;;
    esac
}

# Function to extract wiredtiger metrics from tx CSV
extract_wiredtiger_metrics() {
    local tx_file="$1"
    local metric="$2"

    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi

    case "$metric" in
        "throughput")
            # Sum of oltp_committed / number of seconds, considering only the last 30 seconds
            max_t=$(awk -F',' 'NR > 1 {print $1}' "$tx_file" | sort -nr | head -n1)
            if [ -z "$max_t" ]; then
                echo "n/a"
                return
            fi
            start_t=$((max_t - 30))
            awk -F',' -v start_t="$start_t" 'NR > 1 && $1 > start_t { sum += $5; count++ } END { if (count > 0) print sum/count; else print "n/a" }' "$tx_file"
            ;;
        *)
            echo "n/a"
            ;;
    esac
}

# Create summary CSV header
echo "thread_count,config_name,avg_throughput,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process (huge page variations)
configs=(
    "predicache_tpcc"
    "calico_array_nohuge" "calico_array_huge"
    "calico_mmap_nohuge" "calico_mmap_huge"
    "calico_chained_hash_nohuge" "calico_chained_hash_huge"
    "calico_openaddr_hash_nohuge" "calico_openaddr_hash_huge"
    "calico_robinhood_hash_nohuge" "calico_robinhood_hash_huge"
    "calico_lockfree_hash_nohuge" "calico_lockfree_hash_huge"
    "leanstore" "wiredtiger" "lmdb"
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
        elif [ "$config" == "wiredtiger" ] || [ "$config" == "lmdb" ]; then
            # WiredTiger and LMDB use a simple CSV output
            avg_throughput=$(extract_wiredtiger_metrics "$tx_file" "throughput")
            # For perf metrics, assume it uses the same format as Calico
            llc_miss_per_op=$(extract_perf_value "$perf_file" "llc_misses")
            dtlb_miss_per_op=$(extract_perf_value "$perf_file" "dtlb_misses")
            cycles_per_op=$(extract_perf_value "$perf_file" "cycles")
            insts_per_op=$(extract_perf_value "$perf_file" "instructions")
            ipc=$(extract_ipc_value "$perf_file")
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
echo "  - calico_robinhood_hash: 2 combinations (nohuge, huge)"
echo "  - calico_lockfree_hash: 2 combinations (nohuge, huge)"
echo "  - leanstore: 1 configuration"
echo "  - wiredtiger: 1 configuration"
echo "  Total per thread count: 12 configurations"
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
