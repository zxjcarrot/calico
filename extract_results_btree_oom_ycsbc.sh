#!/bin/bash

# extract_results_btree_oom_ycsbc.sh - Extract and summarize larger-than-memory experiment results
# This script processes the TX and PERF CSV files to create a summary table for varying data sizes

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${EXP_BASE_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/larger-than-memory/btree_ycsbc}"

# Data sizes to process
DATA_SIZES=(25000000 50000000 100000000 200000000 400000000)

# Get total number of cores from lscpu
TOTAL_CORES=$(lscpu -p | grep -v '^#' | wc -l)
if [ -z "$TOTAL_CORES" ] || [ "$TOTAL_CORES" -eq 0 ]; then
    echo "Warning: Could not determine total number of cores. Throughput-per-core will be n/a."
    TOTAL_CORES=0
fi
echo "Detected ${TOTAL_CORES} total cores on the system."
echo

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
    
    # Try to find the summary line first
    local summary_val=$(awk -F',' '$1 == "avg_throughput" { print $2 }' "$tx_file")
    
    if [ -n "$summary_val" ]; then
        echo "$summary_val"
    else
        # Calculate average from data rows if summary is missing
        # Skip header (NR>1) and ensure $2 is a number
        awk -F',' '
            NR > 1 && $2 ~ /^[0-9]+(\.[0-9]+)?$/ { 
                sum += $2; 
                count++; 
            } 
            END { 
                if (count > 0) print sum / count; 
                else print "n/a"; 
            }
        ' "$tx_file"
    fi
}

# Function to extract average CPU utilization from iostat log
extract_avg_cpu_util() {
    local iostat_file="$1"
    
    if [ ! -f "$iostat_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the %user and %system columns (avg-cpu block), average (user+system)
    # Skip lines until we see "avg-cpu" header, then read the numeric line(s) that follow
    awk '
        BEGIN { in_cpu_section = 0; count = 0; }
        /avg-cpu/ { in_cpu_section = 1; next; }
        /Device/ { in_cpu_section = 0; }
        in_cpu_section && NF >= 3 {
            # Expect the numeric line(s) after the avg-cpu header. Typical columns:
            # %user %nice %system %iowait %steal %idle
            # We take $1 (%user) + $3 (%system) as active CPU utilization.
            if ($1 ~ /^[0-9.]+$/ && $3 ~ /^[0-9.]+$/) {
                util = ($1 + $3);
                values[count] = util;
                count++;
            }
        }
        END {
            if (count > 0) {
                # Only use last 40 seconds
                start_idx = (count > 40) ? count - 40 : 0;
                sum = 0;
                samples = 0;
                for (i = start_idx; i < count; i++) {
                    sum += values[i];
                    samples++;
                }
                if (samples > 0) {
                    print sum / samples;
                } else {
                    print "n/a";
                }
            } else {
                print "n/a";
            }
        }
    ' "$iostat_file"
}

# Function to extract average SSD bandwidth from iostat log
extract_avg_ssd_bandwidth() {
    local iostat_file="$1"
    
    if [ ! -f "$iostat_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the MB/s read and write columns for the device, average them
    # Skip header lines, then average the r/s and w/s columns
    awk '
        BEGIN { count = 0; total_bw = 0; }
        # Match lines with numeric data for the nvme device
        /nvme1n1/ && $2 ~ /^[0-9]/ {
            # iostat -x output has rMB/s and wMB/s as columns 6 and 7
            read_bw = $6
            write_bw = $7
            if (read_bw ~ /^[0-9.]+$/ && write_bw ~ /^[0-9.]+$/) {
                total_bw += read_bw + write_bw;
                count++;
            }
        }
        END {
            if (count > 0) {
                avg_bw = total_bw / count;
                print avg_bw;
            } else {
                print "n/a";
            }
        }
    ' "$iostat_file"
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
            # Extract OLTP TX column for LAST 40 SECONDS ONLY
            # Skip lines with "~Transactions", "+---", "| t ", and lines with 0 throughput
            # Only consider the last 40 data points (1 per second)
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
                        # Only use last 40 seconds
                        start_idx = (count > 40) ? count - 40 : 0
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
            # Extract L1/TX column average (LAST 40 SECONDS)
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
                        start_idx = (count > 40) ? count - 40 : 0
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
            # Extract LLC/TX column average (LAST 40 SECONDS)
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
                        start_idx = (count > 40) ? count - 40 : 0
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
            # Extract Cycles/TX column average (LAST 40 SECONDS)
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
                        start_idx = (count > 40) ? count - 40 : 0
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
            # Extract Instrs/TX column average (LAST 40 SECONDS)
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
                        start_idx = (count > 40) ? count - 40 : 0
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
            if [ "$insts" != "n/a" ] && [ "$cycles" != "n/a" ] && [ "$cycles" != "0" ] && [ $(echo "$cycles > 0" | bc -l) -eq 1 ]; then
                echo "scale=6; $insts / $cycles" | bc
            else
                echo "n/a"
            fi
            ;;
    esac
}

extract_wiredtiger_metrics() {
    local tx_file="$1"
    local metric="$2"

    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi

    case "$metric" in
        "throughput")
            # For wiredtiger, the format is a simple CSV: t,tag,oltp_committed,oltp_aborted
            # We average the oltp_committed column, skipping the header and the first data point (t=0)
            awk -F',' '
                NR > 2 { # Skip header and t=0
                    sum += $3;
                    count++;
                }
                END {
                    if (count > 0) {
                        print sum/count;
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        *)
            # Other metrics are not available for wiredtiger in this experiment setup
            echo "n/a"
            ;;
    esac
}

# Create summary CSV header
echo "datasize,config_name,avg_throughput,avg_throughput_per_core,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc,avg_ssd_bandwidth_mb_s" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process
configs=(
    "calico_mmap_nohuge"
    "calico_array_huge"
    "predicache_ycsbc"
    "calico_lockfree_hash_huge"
    "calico_openaddr_hash_huge"
    "calico_robinhood_hash_huge"
    "leanstore"
    "wiredtiger"
    "lmdb"
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
    echo "config_name,avg_throughput,avg_throughput_per_core,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc,avg_ssd_bandwidth_mb_s" > "$datasize_summary_csv"
    
    # Process each configuration for this data size
    for config in "${configs[@]}"; do
        echo "  Processing $config..."
        
        tx_file="${datasize_dir}/${config}_tx.csv"
        perf_file="${datasize_dir}/${config}_perf.csv"
        iostat_file="${datasize_dir}/${config}_iostat.log"
        
        # Determine active cores. Prefer explicit thread count from config.json when available.
        active_cores="n/a"
        config_json="${datasize_dir}/config.json"

        
        # If we don't have threads from config.json, fall back to iostat-based estimate
        if [ "$active_cores" = "n/a" ]; then
            avg_cpu_util=$(extract_avg_cpu_util "$iostat_file")
            if [ "$avg_cpu_util" != "n/a" ] && [ "$TOTAL_CORES" -gt 0 ]; then
                # Calculate active cores based on utilization percentage and cap to TOTAL_CORES
                est_active=$(echo "scale=4; ($avg_cpu_util / 100) * $TOTAL_CORES" | bc)
                # Round up to nearest integer, but not more than TOTAL_CORES
                est_active_ceil=$(printf "%0.f" "$(echo "($est_active+0.5)/1" | bc -l)")
                if [ -z "$est_active_ceil" ] || [ "$est_active_ceil" -le 0 ]; then
                    active_cores="n/a"
                else
                    if [ "$est_active_ceil" -gt "$TOTAL_CORES" ]; then
                        active_cores="$TOTAL_CORES"
                    else
                        active_cores="$est_active_ceil"
                    fi
                fi
            fi
        fi

        echo "$active_cores for $config"

        
        # Extract values based on configuration type
        if [ "$config" == "leanstore" ]; then
            # Leanstore and WiredTiger use custom parser for console table output
            avg_throughput=$(extract_leanstore_metrics "$tx_file" "throughput")
            llc_miss_per_op=$(extract_leanstore_metrics "$tx_file" "llc_per_tx")
            dtlb_miss_per_op="n/a"  # Not available in leanstore/wiredtiger output
            cycles_per_op=$(extract_leanstore_metrics "$tx_file" "cycles_per_tx")
            insts_per_op=$(extract_leanstore_metrics "$tx_file" "insts_per_tx")
            ipc=$(extract_leanstore_metrics "$tx_file" "ipc")
        elif [ "$config" == "wiredtiger" ] || [ "$config" == "lmdb" ]; then
            avg_throughput=$(extract_wiredtiger_metrics "$tx_file" "throughput")
            llc_miss_per_op=$(extract_wiredtiger_metrics "$tx_file" "llc_per_tx")
            dtlb_miss_per_op=$(extract_wiredtiger_metrics "$tx_file" "dtlb_per_tx")
            cycles_per_op=$(extract_wiredtiger_metrics "$tx_file" "cycles_per_tx")
            insts_per_op=$(extract_wiredtiger_metrics "$tx_file" "insts_per_tx")
            ipc=$(extract_wiredtiger_metrics "$tx_file" "ipc")
        else
            # Calico configurations use standard CSV parsers
            avg_throughput=$(extract_avg_throughput "$tx_file")
            llc_miss_per_op=$(extract_perf_value "$perf_file" "llc_misses")
            dtlb_miss_per_op=$(extract_perf_value "$perf_file" "dtlb_misses")
            cycles_per_op=$(extract_perf_value "$perf_file" "cycles")
            insts_per_op=$(extract_perf_value "$perf_file" "instructions")
            ipc=$(extract_ipc_value "$perf_file")
        fi
        
        # Extract SSD bandwidth
        avg_ssd_bandwidth=$(extract_avg_ssd_bandwidth "$iostat_file")

        # Calculate throughput per core using integer active_cores when available
        avg_throughput_per_core="n/a"
        if [ "$avg_throughput" != "n/a" ]; then
            if [ "$active_cores" != "n/a" ] && [[ "$active_cores" =~ ^[0-9]+$ ]] && [ "$active_cores" -gt 0 ]; then
                # Use awk for division because it supports scientific notation (e.g., 1.51835e+06)
                avg_throughput_per_core=$(awk -v a="$avg_throughput" -v b="$active_cores" 'BEGIN { if (a=="" || a=="n/a" || b==0) { print "n/a"; } else { printf "%.2f", (a / b) } }')
            else
                # No valid active core count; leave as n/a
                avg_throughput_per_core="n/a"
            fi
        fi
        
        # Handle empty values
        avg_throughput=${avg_throughput:-"n/a"}
        avg_throughput_per_core=${avg_throughput_per_core:-"n/a"}
        llc_miss_per_op=${llc_miss_per_op:-"n/a"}
        dtlb_miss_per_op=${dtlb_miss_per_op:-"n/a"}
        cycles_per_op=${cycles_per_op:-"n/a"}
        insts_per_op=${insts_per_op:-"n/a"}
        ipc=${ipc:-"n/a"}
        avg_ssd_bandwidth=${avg_ssd_bandwidth:-"n/a"}
        
        # Write to datasize-specific summary CSV
        echo "$config,$avg_throughput,$avg_throughput_per_core,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc,$avg_ssd_bandwidth" >> "$datasize_summary_csv"
        
        # Write to global summary CSV with data size
        echo "$datasize,$config,$avg_throughput,$avg_throughput_per_core,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc,$avg_ssd_bandwidth" >> "${EXP_BASE_DIR}/results_summary_all.csv"
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
echo "  - predicache_ycsbc: PrediCache random-read"
echo "  - calico_lockfree_hash_huge: with huge pages"
echo "  - calico_robinhood_hash_huge: Robin Hood hash with huge pages"
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
