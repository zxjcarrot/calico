#!/bin/bash

# exp_in_memory_olap.sh - In-memory OLAP experiments with varying row sizes
# This script runs OLAP benchmarks for different buffer management strategies 
# with varying row sizes from 128B to 4096B (fixed 8KB page size)

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/olap_row"

# Fixed page size: 8KB
FIXED_PAGE_SIZE_KB=8
FIXED_PAGE_SIZE_BYTES=$((FIXED_PAGE_SIZE_KB * 1024))

# Row sizes to test (in bytes) - exponential growth from 128B to 4096B
ROW_SIZES=(128 256 512 1024 2048 4096)

# Fixed settings
PHYSMEM=54
VIRTMEM=54
DATA_SIZE_GB=50  # 50GB of data
RUN_DURATION=60
THREADS=1

# Common settings for all experiments
export BLOCK=/ssd/ssd0/xj/calico/heapfile
export RUNFOR=${RUN_DURATION}
export THREADS=${THREADS}

# Operation types to test
OP_TYPES=("nonfiltered_count" "sum")

# Layout types to test
LAYOUT_TYPES=("columnar" "row")

# Function to calculate number of pages based on data size (fixed 8KB page size)
calculate_num_pages() {
    local data_size_gb="$1"
    
    # Convert GB to KB: 50GB = 50 * 1024 * 1024 KB
    local data_size_kb=$((data_size_gb * 1024 * 1024))
    
    # Calculate number of pages (fixed 8KB page size)
    local num_pages=$((data_size_kb / FIXED_PAGE_SIZE_KB))
    
    echo "$num_pages"
}

# Function to ensure page size is set to 8KB and compile
ensure_page_size_and_compile() {
    echo "============================================"
    echo "Ensuring page size is set to ${FIXED_PAGE_SIZE_KB}KB (${FIXED_PAGE_SIZE_BYTES} bytes)"
    echo "============================================"
    
    # Check current page size in calico.hpp
    local current_page_size=$(grep "static constexpr u64 pageSize" "${SCRIPT_DIR}/calico.hpp" | \
                             grep -oP 'pageSize = \K[0-9]+')
    
    if [[ "$current_page_size" != "$FIXED_PAGE_SIZE_BYTES" ]]; then
        echo "Updating page size to ${FIXED_PAGE_SIZE_KB}KB..."
        sed -i "s/static constexpr u64 pageSize = [0-9]*;/static constexpr u64 pageSize = ${FIXED_PAGE_SIZE_BYTES};/" \
            "${SCRIPT_DIR}/calico.hpp"
        
        # Recompile
        echo "Recompiling with fixed page size..."
        cd "${SCRIPT_DIR}"
        make clean > /dev/null 2>&1
        make -j$(nproc) > /dev/null 2>&1
        echo "Successfully compiled with page size: ${FIXED_PAGE_SIZE_KB}KB"
    else
        echo "Page size already set to ${FIXED_PAGE_SIZE_KB}KB"
        # Still compile to ensure everything is up to date
        cd "${SCRIPT_DIR}"
        make -j$(nproc) > /dev/null 2>&1
    fi
    echo ""
}

# Function to run a single OLAP experiment
run_olap_experiment() {
    local row_size="$1"
    local config_name="$2"
    local op_type="$3"
    local layout_type="$4"
    local description="$5"
    shift 5
    local env_vars=("$@")
    
    # Create row-size-specific directory with layout subdirectory
    local row_dir="${EXP_BASE_DIR}/${row_size}bytes/${layout_type}"
    mkdir -p "${row_dir}"
    
    echo "=================================================================="
    echo "Running OLAP experiment: ${config_name} with ${row_size}B rows (${FIXED_PAGE_SIZE_KB}KB pages), op=${op_type}, layout=${layout_type}"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Calculate number of pages (fixed 8KB page size)
    local num_pages=$(calculate_num_pages ${DATA_SIZE_GB})
    
    echo "Row size: ${row_size}B"
    echo "Page size: ${FIXED_PAGE_SIZE_KB}KB (fixed)"
    echo "Data size: ${DATA_SIZE_GB}GB"
    echo "Number of pages: ${num_pages}"
    echo "Operation type: ${op_type}"
    echo "Layout type: ${layout_type}"
    
    # Set environment variables for this configuration
    export ROWSIZE=${row_size}
    export NUMPAGES=${num_pages}
    export SCANPAGES=$((num_pages / 10))  # Scan 10% of pages per query
    export VIRTGB=${VIRTMEM}G
    export PHYSGB=${PHYSMEM}G
    export OPTYPE=${op_type}
    export LAYOUT=${layout_type}
    
    for env_var in "${env_vars[@]}"; do
        export ${env_var}
        echo "Set ${env_var}"
    done
    
    # Set output file paths in row-size-specific directory - include op_type and layout in filename
    export OLAP_CSV="${row_dir}/${config_name}_${op_type}_${layout_type}_olap.csv"
    
    echo "Output file: ${OLAP_CSV}"
    echo
    
    # Run the experiment
    echo "Starting OLAP experiment..."
    ./calico_olap_query
    
    echo "Experiment ${config_name} with ${row_size}B rows (${FIXED_PAGE_SIZE_KB}KB pages), op=${op_type}, layout=${layout_type} completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
    unset OPTYPE
    unset LAYOUT
    unset ROWSIZE
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Ensure page size is set to 8KB and compile once
ensure_page_size_and_compile

# Calculate number of pages (fixed 8KB page size)
num_pages=$(calculate_num_pages ${DATA_SIZE_GB})

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "In-Memory OLAP Experiment Summary - Varying Row Sizes" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Page size (fixed): ${FIXED_PAGE_SIZE_KB}KB (${FIXED_PAGE_SIZE_BYTES} bytes)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Row sizes tested (bytes): ${ROW_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Fixed settings: THREADS=${THREADS}, VIRTGB=${VIRTMEM}G, PHYSGB=${PHYSMEM}G, DATA_SIZE=${DATA_SIZE_GB}GB" >> "${GLOBAL_SUMMARY_FILE}"
echo "Number of pages: ${num_pages}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Using huge pages: NOHUGEPAGE_FRAME_MEM=0" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

# Run experiments for each row size
for row_size in "${ROW_SIZES[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH ${row_size}B ROW SIZE (${FIXED_PAGE_SIZE_KB}KB PAGE SIZE)"
    echo "######################################################################"
    
    # Create row-size-specific directory
    row_dir="${EXP_BASE_DIR}/${row_size}bytes"
    mkdir -p "${row_dir}"
    
    # Create config file
    cat > "${row_dir}/config.json" <<EOF
{
  "description": "Configuration for ${row_size}B row size experiments",
  "row_size_bytes": ${row_size},
  "page_size_kb": ${FIXED_PAGE_SIZE_KB},
  "page_size_bytes": ${FIXED_PAGE_SIZE_BYTES},
  "data_size_gb": ${DATA_SIZE_GB},
  "num_pages": ${num_pages},
  "op_types": ["nonfiltered_count", "sum"],
  "layout_types": ["columnar", "row"],
  "common_settings": {
    "THREADS": ${THREADS},
    "VIRTGB": "${VIRTMEM}G",
    "PHYSGB": "${PHYSMEM}G",
    "RUNFOR": ${RUN_DURATION},
    "NUMPAGES": ${num_pages},
    "SCANPAGES": $((num_pages / 4))
  }
}
EOF
    
    echo "Running experiments with ${row_size}B row size..." >> "${GLOBAL_SUMMARY_FILE}"
    
    # Run experiments for each layout type
    for layout_type in "${LAYOUT_TYPES[@]}"; do
        echo "  Testing layout type: ${layout_type}" >> "${GLOBAL_SUMMARY_FILE}"
        
        # Run experiments for each operation type
        for op_type in "${OP_TYPES[@]}"; do
            echo "    Testing operation type: ${op_type}" >> "${GLOBAL_SUMMARY_FILE}"
            
            # Configuration 1: mmap (without huge pages)
            run_olap_experiment ${row_size} "mmap" "${op_type}" "${layout_type}" "Calico mmap without huge pages" \
                "TRADITIONAL=0" "NOHUGEPAGE_FRAME_MEM=1"
            
            # Configuration 2: mmap_huge (using huge pages)
            run_olap_experiment ${row_size} "mmap_huge" "${op_type}" "${layout_type}" "Calico mmap with huge pages" \
                "TRADITIONAL=0" "NOHUGEPAGE_FRAME_MEM=0"
            
            # Configuration 3: calico (traditional buffer pool with array translator, huge pages)
            run_olap_experiment ${row_size} "calico" "${op_type}" "${layout_type}" "Calico traditional with array translator and huge pages" \
                "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=0"
            
            # Configuration 4: chained_hash (traditional buffer pool with chained hash, huge pages)
            run_olap_experiment ${row_size} "chained_hash" "${op_type}" "${layout_type}" "Calico traditional with chained hash and huge pages" \
                "TRADITIONAL=1" "TRADHASH=1" "NOHUGEPAGE_FRAME_MEM=0"

            # Configuration 4: openaddr (traditional buffer pool with open addressing, huge pages)
            run_olap_experiment ${row_size} "openaddr_hash" "${op_type}" "${layout_type}" "Calico traditional with openaddr hash and huge pages" \
                "TRADITIONAL=1" "TRADHASH=2" "NOHUGEPAGE_FRAME_MEM=0"
            
            # # Configuration 5: lockfree_hash (traditional buffer pool with lockfree hash, huge pages)
            # run_olap_experiment ${row_size} "lockfree_hash" "${op_type}" "${layout_type}" "Calico traditional with lockfree hash and huge pages" \
            #     "TRADITIONAL=1" "TRADHASH=4" "NOHUGEPAGE_FRAME_MEM=0"
        done
    done
    
    echo "Completed experiments with ${row_size}B row size" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH ${row_size}B ROW SIZE"
    echo "######################################################################"
    echo
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Fixed page size: ${FIXED_PAGE_SIZE_KB}KB"
echo "Row sizes tested: ${ROW_SIZES[*]} bytes"
echo
echo "Configurations tested per row size:"
echo "  - mmap: VMCache without huge pages"
echo "  - mmap_huge: VMCache with huge pages"
echo "  - calico: Traditional buffer pool with array translator (huge pages)"
echo "  - chained_hash: Traditional buffer pool with chained hash (huge pages)"
echo "  - lockfree_hash: Traditional buffer pool with lockfree hash (huge pages)"
echo "  Total per row size: 5 configurations × 2 operation types × 2 layout types = 20 experiments"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by row size:"
for row_size in "${ROW_SIZES[@]}"; do
    echo "--- ${row_size}B rows ---"
    find "${EXP_BASE_DIR}/${row_size}bytes" -name "*.csv" -o -name "*.json" 2>/dev/null | sort || true
done

echo
echo "Running results extraction..."
./extract_results_olap.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
