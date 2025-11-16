#!/bin/bash

# extract_results_olap.sh - Extract and summarize OLAP experiment results
# This script processes the OLAP CSV files to create a summary table for varying row sizes

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/olap_row"

# Fixed page size (KB)
FIXED_PAGE_SIZE_KB=8

# Row sizes to process (in bytes)
ROW_SIZES=(128 256 512 1024 2048 4096)

# Operation types to process
OP_TYPES=("nonfiltered_count" "sum")

# Layout types to process
LAYOUT_TYPES=("columnar" "row")

# Check if experiment directory exists
if [ ! -d "${EXP_BASE_DIR}" ]; then
    echo "Error: Experiment directory ${EXP_BASE_DIR} does not exist"
    exit 1
fi

# Function to extract average queries per second from OLAP CSV
extract_avg_queries_per_sec() {
    local olap_file="$1"
    
    if [ ! -f "$olap_file" ]; then
        echo "n/a"
        return
    fi
    
    # Extract the avg_queries_per_sec value from the row starting with "avg_queries_per_sec"
    awk -F',' '$1 == "avg_queries_per_sec" { print $2 }' "$olap_file"
}

# Function to extract average read MB from OLAP CSV
extract_avg_read_mb() {
    local olap_file="$1"
    
    if [ ! -f "$olap_file" ]; then
        echo "n/a"
        return
    fi
    
    # Calculate average from rmb column (column 3), excluding the header and avg row
    awk -F',' '
        NR > 1 && $1 != "avg_queries_per_sec" && $3 ~ /^[0-9.]+$/ {
            sum += $3
            count++
        }
        END {
            if (count > 0) print sum/count
            else print "n/a"
        }
    ' "$olap_file"
}

# Function to extract average write MB from OLAP CSV
extract_avg_write_mb() {
    local olap_file="$1"
    
    if [ ! -f "$olap_file" ]; then
        echo "n/a"
        return
    fi
    
    # Calculate average from wmb column (column 4), excluding the header and avg row
    awk -F',' '
        NR > 1 && $1 != "avg_queries_per_sec" && $4 ~ /^[0-9.]+$/ {
            sum += $4
            count++
        }
        END {
            if (count > 0) print sum/count
            else print "n/a"
        }
    ' "$olap_file"
}

# Function to calculate throughput (GB/s) based on queries/sec, pages/query, and page size
calculate_throughput_gbps() {
    local queries_per_sec="$1"
    local pages_per_query="$2"
    
    if [ "$queries_per_sec" == "n/a" ] || [ -z "$queries_per_sec" ]; then
        echo "n/a"
        return
    fi
    
    # Throughput (GB/s) = queries/sec × pages/query × page_size_KB / (1024 * 1024)
    # Using fixed 8KB page size
    # Using bc for floating point calculation
    echo "scale=4; $queries_per_sec * $pages_per_query * ${FIXED_PAGE_SIZE_KB} / (1024 * 1024)" | bc
}

# Create summary CSV header
echo "row_size_bytes,config_name,op_type,layout,avg_queries_per_sec,throughput_gbps,avg_read_mb,avg_write_mb" > "${EXP_BASE_DIR}/results_summary_all.csv"

# List of configurations to process
configs=(
    "mmap"
    "mmap_huge"
    "calico"
    "chained_hash"
    "openaddr_hash"
    "lockfree_hash"
)

echo "Extracting OLAP results from experiment files for all row sizes..."
echo "=================================================================="

# Process each row size
for row_size in "${ROW_SIZES[@]}"; do
    row_dir="${EXP_BASE_DIR}/${row_size}bytes"
    
    if [ ! -d "$row_dir" ]; then
        echo "Warning: Row directory $row_dir does not exist, skipping..."
        continue
    fi
    
    echo "Processing results for ${row_size}B rows (${FIXED_PAGE_SIZE_KB}KB pages)..."
    
    # Read num_pages from config.json if available
    config_file="${row_dir}/config.json"
    if [ -f "$config_file" ]; then
        num_pages=$(grep -oP '"num_pages":\s*\K[0-9]+' "$config_file" || echo "0")
        scan_pages=$(grep -oP '"SCANPAGES":\s*\K[0-9]+' "$config_file" || echo "0")
    else
        # Fallback calculation if config.json doesn't exist
        data_size_kb=$((50 * 1024 * 1024))  # 50GB in KB
        num_pages=$((data_size_kb / FIXED_PAGE_SIZE_KB))
        scan_pages=$((num_pages / 10))
    fi
    
    echo "  Row size: ${row_size}B"
    echo "  Page size: ${FIXED_PAGE_SIZE_KB}KB (fixed)"
    echo "  Number of pages: ${num_pages}"
    echo "  Pages per query: ${scan_pages}"
    
    # Create row-size-specific summary
    row_summary_csv="${row_dir}/results_summary.csv"
    echo "config_name,op_type,layout,avg_queries_per_sec,throughput_gbps,avg_read_mb,avg_write_mb" > "$row_summary_csv"
    
    # Process each layout type
    for layout_type in "${LAYOUT_TYPES[@]}"; do
        layout_dir="${row_dir}/${layout_type}"
        
        if [ ! -d "$layout_dir" ]; then
            echo "  Warning: Layout directory $layout_dir does not exist, skipping..."
            continue
        fi
        
        echo "  Processing layout type: ${layout_type}..."
        
        # Process each operation type
        for op_type in "${OP_TYPES[@]}"; do
            echo "    Processing operation type: ${op_type}..."
            
            # Process each configuration for this row size, layout, and operation type
            for config in "${configs[@]}"; do
                echo "      Processing ${config} with ${op_type} and ${layout_type}..."
                
                olap_file="${layout_dir}/${config}_${op_type}_${layout_type}_olap.csv"
                
                # Extract values
                avg_queries_per_sec=$(extract_avg_queries_per_sec "$olap_file")
                avg_read_mb=$(extract_avg_read_mb "$olap_file")
                avg_write_mb=$(extract_avg_write_mb "$olap_file")
                
                # Calculate throughput in GB/s
                throughput_gbps=$(calculate_throughput_gbps "$avg_queries_per_sec" "$scan_pages")
                
                # Handle empty values
                avg_queries_per_sec=${avg_queries_per_sec:-"n/a"}
                throughput_gbps=${throughput_gbps:-"n/a"}
                avg_read_mb=${avg_read_mb:-"n/a"}
                avg_write_mb=${avg_write_mb:-"n/a"}
                
                # Write to row-size-specific summary CSV
                echo "$config,$op_type,$layout_type,$avg_queries_per_sec,$throughput_gbps,$avg_read_mb,$avg_write_mb" >> "$row_summary_csv"
                
                # Write to global summary CSV with row size, op_type, and layout
                echo "$row_size,$config,$op_type,$layout_type,$avg_queries_per_sec,$throughput_gbps,$avg_read_mb,$avg_write_mb" >> "${EXP_BASE_DIR}/results_summary_all.csv"
            done
        done
    done
    
    echo "  Row-size-specific summary: $row_summary_csv"
    echo
done

echo "=================================================================="
echo "Summary tables created:"
echo "  Global summary: ${EXP_BASE_DIR}/results_summary_all.csv"
for row_size in "${ROW_SIZES[@]}"; do
    row_summary="${EXP_BASE_DIR}/${row_size}bytes/results_summary.csv"
    if [ -f "$row_summary" ]; then
        echo "  ${row_size}B rows: $row_summary"
    fi
done
echo
echo "Fixed page size: ${FIXED_PAGE_SIZE_KB}KB"
echo "Row sizes processed: ${ROW_SIZES[*]} bytes"
echo
echo "Configurations processed per row size:"
echo "  - mmap: VMCache without huge pages"
echo "  - mmap_huge: VMCache with huge pages"
echo "  - calico: Traditional buffer pool with array translator (huge pages)"
echo "  - chained_hash: Traditional buffer pool with chained hash (huge pages)"
echo "  - lockfree_hash: Traditional buffer pool with lockfree hash (huge pages)"
echo "  Total per row size: 5 configurations × 2 operation types × 2 layout types = 20 experiments"
echo

echo "Global Results Summary (all row sizes):"
echo "--------------------------------------------"
if [ -f "${EXP_BASE_DIR}/results_summary_all.csv" ]; then
    column -t -s',' "${EXP_BASE_DIR}/results_summary_all.csv"
fi
echo

# Display individual row size summaries
for row_size in "${ROW_SIZES[@]}"; do
    row_summary="${EXP_BASE_DIR}/${row_size}bytes/results_summary.csv"
    if [ -f "$row_summary" ]; then
        echo "Results for ${row_size}B rows (${FIXED_PAGE_SIZE_KB}KB pages):"
        echo "-----------------------------------"
        column -t -s',' "$row_summary"
        echo
    fi
done

echo "=================================================================="
echo "Results extraction complete!"
echo "=================================================================="
