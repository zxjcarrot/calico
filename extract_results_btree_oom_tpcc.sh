#!/bin/bash

# extract_results_btree_oom_tpcc.sh - Summarize larger-than-memory TPCC experiment results
# Processes TX, PERF, and IOSTAT logs for each warehouse scale and configuration

set -e

# Base directory for experiments
detect_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

SCRIPT_DIR="$(detect_script_dir)"
EXP_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/btree_tpcc"

# Warehouse scales to process (must match exp_larger_than_memory_btree_tpcc.sh)
DATA_SIZES=(20 40 80 100 200 400)

# Detect total core count once for throughput-per-core estimation
TOTAL_CORES=$(lscpu -p | grep -v '^#' | wc -l)
if [ -z "$TOTAL_CORES" ] || [ "$TOTAL_CORES" -eq 0 ]; then
    echo "Warning: Could not determine total number of cores. Throughput-per-core will be n/a."
    TOTAL_CORES=0
fi
echo "Detected ${TOTAL_CORES} total cores on the system."
echo

# Ensure experiment directory exists
if [ ! -d "${EXP_BASE_DIR}" ]; then
    echo "Error: Experiment directory ${EXP_BASE_DIR} does not exist"
    exit 1
fi

# Function helpers shared with YCSB extractor ---------------------------------

extract_perf_value() {
    local perf_file="$1"
    local label="$2"

    if [ ! -f "$perf_file" ]; then
        echo "n/a"
        return
    fi

    awk -F',' -v label="$label" '$1 == label { print $3 }' "$perf_file"
}

extract_ipc_value() {
    local perf_file="$1"

    if [ ! -f "$perf_file" ]; then
        echo "n/a"
        return
    fi

    awk -F',' '$1 == "derived_ipc" { print $2 }' "$perf_file"
}

extract_avg_throughput() {
    local tx_file="$1"

    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi

    awk -F',' '$1 == "avg_throughput" { print $2 }' "$tx_file"
}

extract_avg_cpu_util() {
    local iostat_file="$1"

    if [ ! -f "$iostat_file" ]; then
        echo "n/a"
        return
    fi

    awk '
        BEGIN { in_cpu_section = 0; count = 0; total_util = 0; }
        /avg-cpu/ { in_cpu_section = 1; next; }
        /Device/ { in_cpu_section = 0; }
        in_cpu_section && NF >= 3 {
            if ($1 ~ /^[0-9.]+$/ && $3 ~ /^[0-9.]+$/) {
                util = ($1 + $3);
                total_util += util;
                count++;
            }
        }
        END {
            if (count > 0) {
                print total_util / count;
            } else {
                print "n/a";
            }
        }
    ' "$iostat_file"
}

extract_avg_ssd_bandwidth() {
    local iostat_file="$1"

    if [ ! -f "$iostat_file" ]; then
        echo "n/a"
        return
    fi

    awk '
        BEGIN { count = 0; total_bw = 0; }
        /nvme1n1/ && $2 ~ /^[0-9]/ {
            read_bw = $6;
            write_bw = $7;
            if (read_bw ~ /^[0-9.]+$/ && write_bw ~ /^[0-9.]+$/) {
                total_bw += read_bw + write_bw;
                count++;
            }
        }
        END {
            if (count > 0) {
                print total_bw / count;
            } else {
                print "n/a";
            }
        }
    ' "$iostat_file"
}

extract_leanstore_metrics() {
    local tx_file="$1"
    local metric="$2"

    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi

    case "$metric" in
        "throughput")
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|");
                    tx = fields[3];
                    gsub(/^[ \t]+|[ \t]+$/, "", tx);
                    if (tx ~ /^[0-9]+$/ && tx > 0) {
                        values[count] = tx;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0;
                        sum = 0;
                        for (i = start_idx; i < count; i++) {
                            sum += values[i];
                        }
                        samples = count - start_idx;
                        if (samples > 0) {
                            print sum / samples;
                        } else {
                            print "n/a";
                        }
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        "l1_per_tx")
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|");
                    val = fields[11];
                    gsub(/^[ \t]+|[ \t]+$/, "", val);
                    if (val ~ /^[0-9.]+/) {
                        values[count] = val;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0;
                        sum = 0;
                        for (i = start_idx; i < count; i++) sum += values[i];
                        samples = count - start_idx;
                        if (samples > 0) print sum / samples; else print "n/a";
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        "llc_per_tx")
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|");
                    val = fields[12];
                    gsub(/^[ \t]+|[ \t]+$/, "", val);
                    if (val ~ /^[0-9.]+/) {
                        values[count] = val;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0;
                        sum = 0;
                        for (i = start_idx; i < count; i++) sum += values[i];
                        samples = count - start_idx;
                        if (samples > 0) print sum / samples; else print "n/a";
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        "cycles_per_tx")
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|");
                    val = fields[9];
                    gsub(/^[ \t]+|[ \t]+$/, "", val);
                    if (val ~ /^[0-9.]+/) {
                        values[count] = val;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0;
                        sum = 0;
                        for (i = start_idx; i < count; i++) sum += values[i];
                        samples = count - start_idx;
                        if (samples > 0) print sum / samples; else print "n/a";
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        "insts_per_tx")
            awk '
                /^\| [0-9]+/ && !/^\| t / {
                    split($0, fields, "|");
                    val = fields[8];
                    gsub(/^[ \t]+|[ \t]+$/, "", val);
                    if (val ~ /^[0-9.]+/) {
                        values[count] = val;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        start_idx = (count > 60) ? count - 60 : 0;
                        sum = 0;
                        for (i = start_idx; i < count; i++) sum += values[i];
                        samples = count - start_idx;
                        if (samples > 0) print sum / samples; else print "n/a";
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        "ipc")
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

extract_wiredtiger_metrics() {
    local tx_file="$1"
    local metric="$2"

    if [ ! -f "$tx_file" ]; then
        echo "n/a"
        return
    fi

    case "$metric" in
        "throughput")
            awk -F',' '
                NR == 1 { next; }                # skip "loading finished"
                NR == 2 { next; }                # skip CSV header
                {
                    # CSV format: t,tag,olap_committed,olap_aborted,oltp_committed,oltp_aborted
                    # We care about oltp_committed (column 5)
                    if ($5 ~ /^[0-9]+$/) {
                        sum += $5;
                        count++;
                    }
                }
                END {
                    if (count > 0) {
                        print sum / count;
                    } else {
                        print "n/a";
                    }
                }
            ' "$tx_file"
            ;;
        *)
            echo "n/a"
            ;;
    esac
}

# Summary CSV header
echo "datasize,config_name,active_cores,avg_throughput,avg_throughput_per_core,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc,avg_ssd_bandwidth_mb_s" > "${EXP_BASE_DIR}/results_summary_all.csv"

# Configurations emitted by exp_larger_than_memory_btree_tpcc.sh
configs=(
    "calico_mmap_nohuge"
    "calico_array_huge"
    "calico_openaddr_hash_huge"
    "leanstore"
    "wiredtiger"
)

echo "Extracting results from TPCC experiment files for all warehouse scales..."
echo "=================================================================="

for datasize in "${DATA_SIZES[@]}"; do
    datasize_dir="${EXP_BASE_DIR}/${datasize}"

    if [ ! -d "$datasize_dir" ]; then
        echo "Warning: Data size directory $datasize_dir does not exist, skipping..."
        continue
    fi

    echo "Processing results for datasize ${datasize} warehouses..."

    datasize_summary_csv="${datasize_dir}/results_summary.csv"
    echo "config_name,active_cores,avg_throughput,avg_throughput_per_core,llc_miss_per_op,dtlb_miss_per_op,cycles_per_op,insts_per_op,ipc,avg_ssd_bandwidth_mb_s" > "$datasize_summary_csv"

    for config in "${configs[@]}"; do
        echo "  Processing $config..."

        tx_file="${datasize_dir}/${config}_tx.csv"
        perf_file="${datasize_dir}/${config}_perf.csv"
        iostat_file="${datasize_dir}/${config}_iostat.log"

        active_cores="n/a"
        config_json="${datasize_dir}/config.json"

        # Future: parse config_json for thread counts when available

        if [ "$active_cores" = "n/a" ]; then
            avg_cpu_util=$(extract_avg_cpu_util "$iostat_file")
            if [ "$avg_cpu_util" != "n/a" ] && [ "$TOTAL_CORES" -gt 0 ]; then
                est_active=$(echo "scale=4; ($avg_cpu_util / 100) * $TOTAL_CORES" | bc)
                est_active_ceil=$(printf "%0.f" "$(echo "($est_active+0.5)/1" | bc -l)")
                if [ -n "$est_active_ceil" ] && [ "$est_active_ceil" -gt 0 ]; then
                    if [ "$est_active_ceil" -gt "$TOTAL_CORES" ]; then
                        active_cores="$TOTAL_CORES"
                    else
                        active_cores="$est_active_ceil"
                    fi
                else
                    active_cores="n/a"
                fi
            fi
        fi

        echo "$active_cores active cores inferred for $config"

        if [ "$config" = "leanstore" ]; then
            avg_throughput=$(extract_leanstore_metrics "$tx_file" "throughput")
            llc_miss_per_op=$(extract_leanstore_metrics "$tx_file" "llc_per_tx")
            dtlb_miss_per_op="n/a"
            cycles_per_op=$(extract_leanstore_metrics "$tx_file" "cycles_per_tx")
            insts_per_op=$(extract_leanstore_metrics "$tx_file" "insts_per_tx")
            ipc=$(extract_leanstore_metrics "$tx_file" "ipc")
        elif [ "$config" = "wiredtiger" ]; then
            avg_throughput=$(extract_wiredtiger_metrics "$tx_file" "throughput")
            llc_miss_per_op="n/a"
            dtlb_miss_per_op="n/a"
            cycles_per_op="n/a"
            insts_per_op="n/a"
            ipc="n/a"
        else
            avg_throughput=$(extract_avg_throughput "$tx_file")
            llc_miss_per_op=$(extract_perf_value "$perf_file" "llc_misses")
            dtlb_miss_per_op=$(extract_perf_value "$perf_file" "dtlb_misses")
            cycles_per_op=$(extract_perf_value "$perf_file" "cycles")
            insts_per_op=$(extract_perf_value "$perf_file" "instructions")
            ipc=$(extract_ipc_value "$perf_file")
        fi

        avg_ssd_bandwidth=$(extract_avg_ssd_bandwidth "$iostat_file")

        avg_throughput_per_core="n/a"
        if [ "$avg_throughput" != "n/a" ]; then
            if [ "$active_cores" != "n/a" ] && [[ "$active_cores" =~ ^[0-9]+$ ]] && [ "$active_cores" -gt 0 ]; then
                avg_throughput_per_core=$(awk -v a="$avg_throughput" -v b="$active_cores" 'BEGIN { if (a=="" || a=="n/a" || b==0) { print "n/a"; } else { printf "%.2f", (a / b) } }')
            else
                avg_throughput_per_core="n/a"
            fi
        fi

        avg_throughput=${avg_throughput:-"n/a"}
        avg_throughput_per_core=${avg_throughput_per_core:-"n/a"}
        llc_miss_per_op=${llc_miss_per_op:-"n/a"}
        dtlb_miss_per_op=${dtlb_miss_per_op:-"n/a"}
        cycles_per_op=${cycles_per_op:-"n/a"}
        insts_per_op=${insts_per_op:-"n/a"}
        ipc=${ipc:-"n/a"}
        avg_ssd_bandwidth=${avg_ssd_bandwidth:-"n/a"}

    echo "$config,$active_cores,$avg_throughput,$avg_throughput_per_core,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc,$avg_ssd_bandwidth" >> "$datasize_summary_csv"
    echo "$datasize,$config,$active_cores,$avg_throughput,$avg_throughput_per_core,$llc_miss_per_op,$dtlb_miss_per_op,$cycles_per_op,$insts_per_op,$ipc,$avg_ssd_bandwidth" >> "${EXP_BASE_DIR}/results_summary_all.csv"
    done

    echo "  Warehouse-scale summary: $datasize_summary_csv"
    echo
done

echo "=================================================================="
echo "Summary tables created:"
echo "  Global summary: ${EXP_BASE_DIR}/results_summary_all.csv"
for datasize in "${DATA_SIZES[@]}"; do
    datasize_summary="${EXP_BASE_DIR}/${datasize}/results_summary.csv"
    if [ -f "$datasize_summary" ]; then
        echo "  ${datasize} warehouses: $datasize_summary"
    fi
done
echo
echo "Configurations processed:"
echo "  - calico_mmap_nohuge"
echo "  - calico_array_huge"
echo "  - calico_openaddr_hash_huge"
echo "  - leanstore"
echo "  - wiredtiger"
echo "  Total per warehouse scale: 5 configurations"
echo

echo "Global Results Summary (all warehouse scales):"
echo "--------------------------------------------"
if [ -f "${EXP_BASE_DIR}/results_summary_all.csv" ]; then
    column -t -s',' "${EXP_BASE_DIR}/results_summary_all.csv"
fi
echo

for datasize in "${DATA_SIZES[@]}"; do
    datasize_summary="${EXP_BASE_DIR}/${datasize}/results_summary.csv"
    if [ -f "$datasize_summary" ]; then
        echo "Results for ${datasize} warehouses:"
        echo "-----------------------------"
        column -t -s',' "$datasize_summary"
        echo
    fi
done
