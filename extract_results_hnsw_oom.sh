#!/bin/bash

# Extract and summarize HNSW OOM benchmark results
# Parses log files and generates CSV summary
# Base directory containing results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/hnsw-deep10M"
#RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/hnsw-sift10M"

# Output CSV file
OUTPUT_CSV="${RESULTS_BASE_DIR}/hnsw_oom_results_summary.csv"

# Helper: check if a value is a positive number (int or float)
is_number() {
    echo "$1" | grep -E '^[0-9]+(\.[0-9]+)?$' >/dev/null 2>&1
}

# Function to extract a metric from log file
extract_metric() {
    local log_file=$1
    local pattern=$2
    local default_value=${3:-"N/A"}
    
    if [ ! -f "$log_file" ]; then
        echo "$default_value"
        return
    fi
    
    local value=$(grep -oP "$pattern" "$log_file" | head -1 || echo "")
    if [ -z "$value" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

# Function to extract summary metrics (like from "Benchmark Summary")
extract_summary_metric() {
    local log_file=$1
    local metric_name=$2
    local default_value=${3:-"N/A"}
    
    if [ ! -f "$log_file" ]; then
        echo "$default_value"
        return
    fi
    
    # Use cut instead of sed to avoid issues with special characters
    local value=$(grep "^${metric_name}:" "$log_file" | head -1 | cut -d':' -f2- | sed 's/^ *//' || echo "")
    if [ -z "$value" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

# Function to parse iostat file and compute averages for a time range
# Args: iostat_file, start_sample_num (1-indexed), duration_seconds
parse_iostat_range() {
    local iostat_file=$1
    local start_sample=$2
    local duration_secs=$3
    
    if [ ! -f "$iostat_file" ]; then
        echo "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        return
    fi
    
    # iostat outputs every 1 second, so num_samples = duration_secs
    local end_sample=$((start_sample + duration_secs))
    
    # Parse CPU metrics for the specified range
    local cpu_user=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /avg-cpu/ {sample++; getline; if (sample >= start && sample <= end) {sum+=$1; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local cpu_system=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /avg-cpu/ {sample++; getline; if (sample >= start && sample <= end) {sum+=$3; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local cpu_iowait=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /avg-cpu/ {sample++; getline; if (sample >= start && sample <= end) {sum+=$4; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local cpu_idle=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /avg-cpu/ {sample++; getline; if (sample >= start && sample <= end) {sum+=$6; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    # Parse device metrics for the specified range
    local read_iops=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$2; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local read_kbps=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$3; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local write_iops=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$8; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local write_kbps=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$9; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local util_pct=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$NF; count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    local await_ms=$(awk -v start="$start_sample" -v end="$end_sample" '
        BEGIN {sample=0; sum=0; count=0}
        /nvme0n1/ {sample++; if (sample >= start && sample <= end) {sum+=$(NF-2); count++}}
        END {if (count>0) printf "%.2f", sum/count; else print "N/A"}' "$iostat_file")
    
    echo "$cpu_user,$cpu_system,$cpu_iowait,$cpu_idle,$read_iops,$read_kbps,$write_iops,$write_kbps,$util_pct,$await_ms"
}

# Function to parse a single log file
parse_log_file() {
    local log_file=$1
    
    >&2 echo "Parsing: $log_file"
    
    # Extract configuration from filename
    # Format: configname_physgbX_threadsY_timestamp.log
    local filename=$(basename "$log_file")
    local config=$(echo "$filename" | sed -E 's/_physgb.*//')
    local physgb=$(echo "$filename" | grep -oE 'physgb[0-9.]+' | sed 's/physgb//')
    local threads=$(echo "$filename" | grep -oE 'threads[0-9]+' | sed 's/threads//')
    
    # Extract metrics from log
    local build_time=$(extract_summary_metric "$log_file" "Build Time" "N/A")
    build_time=$(echo "$build_time" | sed 's/s$//')  # Remove trailing 's'
    
    local search_time=$(extract_summary_metric "$log_file" "Search Time" "N/A")
    search_time=$(echo "$search_time" | sed 's/s$//')
    # Fallback: parse from inline "Search time (N queries): X seconds (...)"
    if [ -z "$search_time" ] || [ "$search_time" = "N/A" ]; then
        local st_inline=$(grep -E "^Search time \([0-9,]+ queries\): [0-9.]+ seconds" "$log_file" | head -1 | awk -F":" '{print $2}' | awk '{print $1}')
        if [ -n "$st_inline" ]; then
            search_time="$st_inline"
        fi
    fi
    
    local qps=$(extract_summary_metric "$log_file" "QPS")
    # Fallback: parse QPS from inline line
    if [ -z "$qps" ] || [ "$qps" = "N/A" ]; then
        qps=$(grep -E "^Search time \([0-9,]+ queries\): [0-9.]+ seconds \([0-9.]+ QPS\)" "$log_file" | head -1 | grep -oE "\([0-9.]+ QPS\)" | tr -d '()' | awk '{print $1}')
        [ -z "$qps" ] && qps="N/A"
    fi
    
    local recall=$(extract_summary_metric "$log_file" "Recall@10")
    # Fallback: early line "Recall@10: X" if summary missing
    if [ -z "$recall" ] || [ "$recall" = "N/A" ]; then
        recall=$(grep -E "^Recall@10:" "$log_file" | head -1 | awk -F":" '{print $2}' | sed 's/^ *//')
        [ -z "$recall" ] && recall="N/A"
    fi
    
    local avg_dist_comps=$(extract_summary_metric "$log_file" "Avg Dist Comps/Query")
    
    local index_size=$(extract_summary_metric "$log_file" "Index Size")
    # Fallback: for usearch logs which print "Index file size: X"
    if [ -z "$index_size" ] || [ "$index_size" = "N/A" ]; then
        index_size=$(grep -E "^Index file size:" "$log_file" | head -1 | cut -d':' -f2- | sed 's/^ *//')
        [ -z "$index_size" ] && index_size="N/A"
    fi
    
    # Extract cache metrics using simpler patterns
    local cache_misses=$(grep "cache-misses" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$cache_misses" ] && cache_misses="N/A"
    
    local cache_refs=$(grep "cache-references" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$cache_refs" ] && cache_refs="N/A"
    
    local l1_misses=$(grep "L1-dcache-load-misses" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$l1_misses" ] && l1_misses="N/A"
    
    local llc_misses=$(grep "LLC-load-misses" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$llc_misses" ] && llc_misses="N/A"
    
    local dtlb_misses=$(grep "dTLB-load-misses" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$dtlb_misses" ] && dtlb_misses="N/A"
    
    local dtlb_loads=$(grep "dTLB-loads" "$log_file" | grep -oE "[0-9,]+" | head -1 | tr -d ',')
    [ -z "$dtlb_loads" ] && dtlb_loads="N/A"
    
    # Extract environment config
    local traditional=$(grep "TRADITIONAL=" "$log_file" | head -1 | grep -oE "TRADITIONAL=[0-9]+" | cut -d'=' -f2)
    [ -z "$traditional" ] && traditional="N/A"
    
    local tradhash=$(grep "TRADHASH=" "$log_file" | head -1 | grep -oE "TRADHASH=[0-9]+" | cut -d'=' -f2)
    [ -z "$tradhash" ] && tradhash="N/A"
    
    local mmap_enabled=$(grep "MMAP_FOR_HSNWLIB=" "$log_file" | head -1 | grep -oE "MMAP_FOR_HSNWLIB=[0-1]" | cut -d'=' -f2)
    [ -z "$mmap_enabled" ] && mmap_enabled="N/A"
    
    # Find corresponding iostat file
    local iostat_file="${log_file%.log}.iostat"
    
    # Parse iostat for build and search phases separately
    # iostat starts at sample 1, we skip first 2 samples to stabilize
    local start_sample=3
    
    # Extract build and search durations from log
    local build_duration="N/A"
    local search_duration="N/A"
    
    if is_number "$build_time"; then
        build_duration=$(awk -v t="$build_time" 'BEGIN {if (t > 0) printf "%d", int(t + 0.5); else print "N/A"}')
    fi
    
    if is_number "$search_time"; then
        search_duration=$(awk -v t="$search_time" 'BEGIN {if (t > 0) printf "%d", int(t + 0.5); else print "N/A"}')
    fi
    
    # Parse iostat for build phase (starts at sample 3)
    local build_iostat="N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
    if [ -n "$build_duration" ] && [ "$build_duration" != "N/A" ] && [ -f "$iostat_file" ]; then
        build_iostat=$(parse_iostat_range "$iostat_file" "$start_sample" "$build_duration")
    fi
    
    # Parse iostat for search phase (starts after build phase + 2 sample buffer)
    local search_start=""
    if [ -n "$build_duration" ] && [ "$build_duration" != "N/A" ]; then
        search_start=$((start_sample + build_duration + 2))
    fi
    local search_iostat="N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
    if [ -n "$search_start" ] && [ -n "$search_duration" ] && [ "$search_duration" != "N/A" ] && [ -f "$iostat_file" ]; then
        search_iostat=$(parse_iostat_range "$iostat_file" "$search_start" "$search_duration")
    fi
    
    # Compute IO per query and KB per query for search phase
    local io_per_query="N/A"
    local kb_per_query="N/A"
    if is_number "$qps" && awk -v q="$qps" 'BEGIN{exit !(q>0)}'; then
        local search_read_iops=$(echo "$search_iostat" | cut -d',' -f5)
        local search_write_iops=$(echo "$search_iostat" | cut -d',' -f7)
        local search_read_kbps=$(echo "$search_iostat" | cut -d',' -f6)
        local search_write_kbps=$(echo "$search_iostat" | cut -d',' -f8)
        
        if is_number "$search_read_iops" && is_number "$search_write_iops"; then
            io_per_query=$(awk -v r="$search_read_iops" -v w="$search_write_iops" -v qps="$qps" 'BEGIN {printf "%.4f", (r+w)/qps}')
        fi
        
        if is_number "$search_read_kbps" && is_number "$search_write_kbps"; then
            kb_per_query=$(awk -v kr="$search_read_kbps" -v kw="$search_write_kbps" -v qps="$qps" 'BEGIN {printf "%.2f", (kr+kw)/qps}')
        fi
    fi
    
    # Output CSV row with separate build and search iostat metrics
    echo "$config,$physgb,$threads,$build_time,$search_time,$qps,$recall,$avg_dist_comps,$index_size,$cache_misses,$cache_refs,$l1_misses,$llc_misses,$dtlb_misses,$dtlb_loads,$traditional,$tradhash,$mmap_enabled,$build_iostat,$search_iostat,$io_per_query,$kb_per_query"
}

# Main function
main() {
    echo "Extracting HNSW OOM benchmark results..."
    echo "Results directory: $RESULTS_BASE_DIR"
    
    # Create CSV header with separate build and search metrics
    echo "config,physgb,threads,build_time_s,search_time_s,qps,recall,avg_dist_comps,index_size,cache_misses,cache_refs,l1_misses,llc_misses,dtlb_misses,dtlb_loads,traditional,tradhash,mmap_enabled,build_cpu_user_pct,build_cpu_system_pct,build_cpu_iowait_pct,build_cpu_idle_pct,build_read_iops,build_read_kbps,build_write_iops,build_write_kbps,build_util_pct,build_await_ms,search_cpu_user_pct,search_cpu_system_pct,search_cpu_iowait_pct,search_cpu_idle_pct,search_read_iops,search_read_kbps,search_write_iops,search_write_kbps,search_util_pct,search_await_ms,io_per_query,kb_per_query" > "$OUTPUT_CSV"
    
    # Find all log files and parse them
    if [ ! -d "$RESULTS_BASE_DIR" ]; then
        echo "Error: Results directory not found: $RESULTS_BASE_DIR"
        exit 1
    fi
    
    log_count=0
    while IFS= read -r -d '' log_file; do
        result=$(parse_log_file "$log_file")
        echo "$result" >> "$OUTPUT_CSV"
        ((log_count++))
    done < <(find "$RESULTS_BASE_DIR" -name "*.log" -type f -print0 | sort -z)
    
    echo ""
    echo "Extraction complete!"
    echo "Processed $log_count log files"
    echo "Summary CSV: $OUTPUT_CSV"
    echo ""
    
    # Display summary
    if [ -f "$OUTPUT_CSV" ]; then
        echo "First few lines of summary:"
        head -5 "$OUTPUT_CSV" | column -t -s ','
        echo ""
        echo "Total rows: $(wc -l < "$OUTPUT_CSV")"
    fi
}

# Run main function
main "$@"
