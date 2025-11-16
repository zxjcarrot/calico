#!/bin/bash

# Extract and summarize HNSW benchmark results
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/in-memory/hnsw/sift10M"
OUTPUT_CSV="${RESULTS_BASE_DIR}/hnsw_results_summary.csv"

# Thread counts
THREAD_COUNTS=(1 4 8 16 32 64 128)

# Configuration names
#CONFIGS=("chained_hash" "openaddr_hash" "lockfree_hash" "vmcache" "calico" "calico-2level" "calico-3level" "mmap_oscache")
CONFIGS=("chained_hash" "openaddr_hash" "lockfree_hash" "vmcache" "calico" "usearch")
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
    local value=$(awk "/Benchmark Summary/,/^$/" "$log_file" | grep "$metric_name:" | grep -oP '[\d.]+' | head -1)
    
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
    local threads=$3
    
    echo "Parsing: $log_file" >&2  # Send to stderr, not stdout
    
    # Extract metrics
    local build_time=$(extract_summary_metric "$log_file" "Build Time" "0")
    local search_time=$(extract_summary_metric "$log_file" "Search Time" "0")
    local qps=$(extract_summary_metric "$log_file" "QPS" "0")
    local recall=$(extract_summary_metric "$log_file" "Recall@" "0")
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
    
    # Extract environment config
    local traditional=$(extract_metric "$log_file" "(?<=TRADITIONAL=)\d+" "N/A")
    local tradhash=$(extract_metric "$log_file" "(?<=TRADHASH=)\d+" "N/A")
    
    # Output CSV row
    echo "$config,$threads,$build_time,$search_time,$qps,$recall,$avg_dist_comps,$cache_misses,$cache_refs,$cache_miss_rate,$dtlb_misses,$llc_misses,$index_size,$traditional,$tradhash"
}

# Main execution
main() {
    echo "Extracting HNSW benchmark results from: $RESULTS_BASE_DIR"
    echo ""
    
    # Create CSV header
    echo "Config,Threads,BuildTime(s),SearchTime(s),QPS,Recall,AvgDistComps,CacheMisses,CacheRefs,CacheMissRate(%),dTLBMisses,LLCMisses,IndexSize,TRADITIONAL,TRADHASH" > "$OUTPUT_CSV"
    
    # Process each thread count
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "Processing results for $threads threads..."
        
        results_dir="${RESULTS_BASE_DIR}/${threads}"
        
        if [[ ! -d "$results_dir" ]]; then
            echo "  Warning: Directory not found: $results_dir"
            continue
        fi
        
        # Process each configuration
        for config in "${CONFIGS[@]}"; do
            # Find the most recent log file for this config
            log_file=$(ls -t "${results_dir}/${config}_threads${threads}_"*.log 2>/dev/null | head -1)
            
            if [[ -z "$log_file" ]]; then
                echo "  Warning: No log file found for $config with $threads threads"
                # Add placeholder row
                echo "$config,$threads,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> "$OUTPUT_CSV"
            else
                parse_log_file "$log_file" "$config" "$threads" >> "$OUTPUT_CSV"
            fi
        done
        
        echo ""
    done
    
    echo "Results summary saved to: $OUTPUT_CSV"
    echo ""
    
    # Display summary table
    if command -v column &> /dev/null; then
        echo "Summary Table:"
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
