#!/bin/bash

# Extract and summarize Ablation benchmark results
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_BASE_DIR="${CALICO_ABLATION_RESULTS_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/in-memory/ablation_figure11_hash_then_array}"
OUTPUT_CSV="${RESULTS_BASE_DIR}/ablation_results_summary.csv"

# Configuration names (directories)
CONFIGS=(
    "1_hash_only"
    "2_array_index"
    "3_hash_then_array"
    "4_hash_then_array_path"
    "5_hash_then_array_path_huge_frames"
    "6_hash_then_array_path_huge_trans"
    "7_hash_then_array_path_optread"
    "8_hash_then_array_path_prefetch"
)

# Function to extract a metric from log file
extract_metric() {
    local log_file=$1
    local pattern=$2
    local default=${3:-"N/A"}

    if [[ ! -f "$log_file" ]]; then
        echo "$default"
        return
    fi

    local value=$(grep -oP "$pattern" "$log_file" | tail -1)
    if [[ -z "$value" ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Function to extract from "Benchmark Summary" section
extract_summary_metric() {
    local log_file=$1
    local metric_name=$2
    local default=${3:-"N/A"}

    if [[ ! -f "$log_file" ]]; then
        echo "$default"
        return
    fi

    # Extract value after the metric name in the summary section
    # Use awk to find the line containing the metric name within the summary block
    # Then split by colon and take the second part, trimming whitespace
    local value=$(awk "/Benchmark Summary/,/^$/" "$log_file" | grep "$metric_name" | awk -F':' '{print $2}' | tr -d ' ' | head -1)

    # If value is empty, try the grep -oP method as fallback (for cases without colon or different format)
    if [[ -z "$value" ]]; then
         value=$(awk "/Benchmark Summary/,/^$/" "$log_file" | grep "$metric_name" | grep -oP '[\d.]+' | head -1)
    fi

    if [[ -z "$value" ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Function to parse a single log file
parse_log_file() {
    local log_file=$1
    local config=$2
    local filename=$(basename "$log_file")

    # Extract metadata from filename
    # Format: <dataset>_build<b_threads>_search<s_threads>_<timestamp>.log
    local dataset=$(echo "$filename" | cut -d'_' -f1)
    local build_threads=$(echo "$filename" | grep -oP "build\K\d+")
    local search_threads=$(echo "$filename" | grep -oP "search\K\d+")
    local timestamp=$(echo "$filename" | grep -oP "\d{8}_\d{6}")

    echo "Parsing: $filename (Config: $config, Dataset: $dataset, Threads: $search_threads)" >&2

    # Extract metrics
    local build_time=$(extract_summary_metric "$log_file" "Build Time" "0")
    local search_time=$(extract_summary_metric "$log_file" "Search Time" "0")
    local qps=$(extract_summary_metric "$log_file" "QPS" "0")
    local recall=$(extract_summary_metric "$log_file" "Recall" "0")
    local avg_dist_comps=$(extract_summary_metric "$log_file" "Avg Dist Comps/Query" "0")

    # Extract cache metrics
    local cache_misses=$(extract_metric "$log_file" "(?<=cache-misses:\s)[\d,]+" "N/A" | tr -d ',')
    local cache_refs=$(extract_metric "$log_file" "(?<=cache-references:\s)[\d,]+" "N/A" | tr -d ',')
    local dtlb_misses=$(extract_metric "$log_file" "(?<=dTLB-load-misses:\s)[\d,]+" "N/A" | tr -d ',')
    local llc_misses=$(extract_metric "$log_file" "(?<=LLC-load-misses:\s)[\d,]+" "N/A" | tr -d ',')

    # Calculate cache miss rate
    local cache_miss_rate="N/A"
    if [[ "$cache_refs" != "N/A" && "$cache_misses" != "N/A" && "$cache_refs" != "0" ]]; then
        cache_miss_rate=$(echo "scale=4; ($cache_misses / $cache_refs) * 100" | bc)
    fi

    # Extract index size
    local index_size=$(extract_metric "$log_file" "(?<=Index Size: )[\d.]+ [KMGT]?B" "N/A")

    # Output CSV row
    echo "$config,$dataset,$build_threads,$search_threads,$timestamp,$build_time,$search_time,$qps,$recall,$avg_dist_comps,$cache_misses,$cache_refs,$cache_miss_rate,$dtlb_misses,$llc_misses,$index_size"
}

# Main execution
main() {
    echo "Extracting Ablation benchmark results from: $RESULTS_BASE_DIR"
    echo ""

    # Create CSV header
    echo "Config,Dataset,BuildThreads,SearchThreads,Timestamp,BuildTime(s),SearchTime(s),QPS,Recall,AvgDistComps,CacheMisses,CacheRefs,CacheMissRate(%),dTLBMisses,LLCMisses,IndexSize" > "$OUTPUT_CSV"

    # Process each configuration
    for config in "${CONFIGS[@]}"; do
        config_dir="${RESULTS_BASE_DIR}/${config}"

        if [[ ! -d "$config_dir" ]]; then
            echo "  Warning: Directory not found: $config_dir"
            continue
        fi

        echo "Processing config: $config"

        # Find all log files in the config directory
        # We sort by name to have some order, but we process all of them
        find "$config_dir" -name "*.log" | sort | while read -r log_file; do
            parse_log_file "$log_file" "$config" >> "$OUTPUT_CSV"
        done

        echo ""
    done

    echo "Results summary saved to: $OUTPUT_CSV"
    echo ""

    # Display summary table
    if command -v column &> /dev/null; then
        echo "Summary Table (Top 20 rows):"
        echo "=============================================="
        column -t -s',' "$OUTPUT_CSV" | head -20
        echo "... (see $OUTPUT_CSV for full results)"
    else
        echo "Install 'column' command for pretty table display"
        echo "Raw CSV content (first 10 rows):"
        head -10 "$OUTPUT_CSV"
    fi
}

# Run main function
main "$@"
