#!/bin/bash

# exp_in_memory_olap.sh - In-memory OLAP experiments with varying page sizes
# This script runs OLAP benchmarks for different buffer management strategies 
# with varying page sizes from 4KB to 512KB

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/olap"

# Page sizes to test (in KB) - exponential growth from 4KB to 512KB
PAGE_SIZES=(4 8 16 32 64 128 256 512)
# Fixed settings
PHYSMEM=54
VIRTMEM=54
DATA_SIZE_GB=50  # 50GB of data
RUN_DURATION=60
THREADS=1

# Common settings for all experiments
export BLOCK=/home/zxjcarrot/Workspace/calico/heapfile
export RUNFOR=${RUN_DURATION}
export THREADS=${THREADS}

# Operation types to test
OP_TYPES=("nonfiltered_count" "sum")

# Layout types to test
LAYOUT_TYPES=("columnar" "row")

# Function to calculate page size in bytes
calculate_page_size_bytes() {
    local page_size_kb="$1"
    echo $((page_size_kb * 1024))
}

# Function to calculate number of pages based on data size and page size
calculate_num_pages() {
    local data_size_gb="$1"
    local page_size_kb="$2"
    
    # Convert GB to KB: 30GB = 30 * 1024 * 1024 KB
    local data_size_kb=$((data_size_gb * 1024 * 1024))
    
    # Calculate number of pages
    local num_pages=$((data_size_kb / page_size_kb))
    
    echo "$num_pages"
}

# Function to update pageSize in calico.hpp and recompile
update_page_size_and_compile() {
    local page_size_bytes="$1"
    
    echo "=================================================================="
    echo "Updating pageSize to ${page_size_bytes} bytes in calico.hpp"
    echo "=================================================================="
    
    # Backup original calico.hpp if not already backed up
    if [ ! -f "${SCRIPT_DIR}/calico.hpp.backup" ]; then
        cp "${SCRIPT_DIR}/calico.hpp" "${SCRIPT_DIR}/calico.hpp.backup"
        echo "Created backup: calico.hpp.backup"
    fi
    
    # Update pageSize using sed - note that it's "static constexpr" in calico.hpp
    sed -i "s/^static constexpr u64 pageSize = [0-9]*;/static constexpr u64 pageSize = ${page_size_bytes};/" "${SCRIPT_DIR}/calico.hpp"
    
    # Verify the update
    local actual_page_size=$(grep "^static constexpr u64 pageSize" "${SCRIPT_DIR}/calico.hpp" | sed -n 's/.*= \([0-9]*\);/\1/p')
    echo "Updated pageSize to ${page_size_bytes} bytes (verified: ${actual_page_size} bytes)"
    
    if [ "$actual_page_size" != "$page_size_bytes" ]; then
        echo "ERROR: Failed to update pageSize! Expected ${page_size_bytes} but found ${actual_page_size}"
        exit 1
    fi
    
    # Recompile
    echo "Recompiling calico_olap_query..."
    cd "${SCRIPT_DIR}"
    make clean 
    make all
    
    echo "Compilation complete."
    echo
}

# Function to restore original calico.hpp
restore_calico_hpp() {
    if [ -f "${SCRIPT_DIR}/calico.hpp.backup" ]; then
        echo "Restoring original calico.hpp..."
        mv "${SCRIPT_DIR}/calico.hpp.backup" "${SCRIPT_DIR}/calico.hpp"
        echo "Restored calico.hpp"
        
        # Recompile with original settings
        echo "Recompiling with original settings..."
        cd "${SCRIPT_DIR}"
        make clean && make -j$(nproc)
    fi
}

# Function to run a single OLAP experiment
run_olap_experiment() {
    local page_size_kb="$1"
    local config_name="$2"
    local op_type="$3"
    local layout_type="$4"
    local description="$5"
    shift 5
    local env_vars=("$@")
    
    # Create page-size-specific directory with layout subdirectory
    local page_dir="${EXP_BASE_DIR}/${page_size_kb}kb/${layout_type}"
    mkdir -p "${page_dir}"
    
    echo "=================================================================="
    echo "Running OLAP experiment: ${config_name} with ${page_size_kb}KB pages, op=${op_type}, layout=${layout_type}"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Calculate number of pages for this page size
    local num_pages=$(calculate_num_pages ${DATA_SIZE_GB} ${page_size_kb})
    
    echo "Page size: ${page_size_kb}KB"
    echo "Data size: ${DATA_SIZE_GB}GB"
    echo "Number of pages: ${num_pages}"
    echo "Operation type: ${op_type}"
    echo "Layout type: ${layout_type}"
    
    # Set environment variables for this configuration
    export NUMPAGES=${num_pages}
    export SCANPAGES=$((num_pages / 10))  # Scan 25% of pages per query
    export VIRTGB=${VIRTMEM}G
    export PHYSGB=${PHYSMEM}G
    export OPTYPE=${op_type}
    export LAYOUT=${layout_type}
    
    for env_var in "${env_vars[@]}"; do
        export ${env_var}
        echo "Set ${env_var}"
    done
    
    # Set output file paths in page-size-specific directory - include op_type and layout in filename
    export OLAP_CSV="${page_dir}/${config_name}_${op_type}_${layout_type}_olap.csv"
    
    echo "Output file: ${OLAP_CSV}"
    echo
    
    # Run the experiment
    echo "Starting OLAP experiment..."
    ./calico_olap_query
    
    echo "Experiment ${config_name} with ${page_size_kb}KB pages, op=${op_type}, layout=${layout_type} completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
    unset OPTYPE
    unset LAYOUT
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "In-Memory OLAP Experiment Summary - Varying Page Sizes" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Page sizes tested (KB): ${PAGE_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Fixed settings: THREADS=${THREADS}, VIRTGB=${VIRTMEM}G, PHYSGB=${PHYSMEM}G, DATA_SIZE=${DATA_SIZE_GB}GB" >> "${GLOBAL_SUMMARY_FILE}"
echo "Using huge pages: NOHUGEPAGE_FRAME_MEM=0" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

# Trap to ensure we restore calico.hpp on exit
trap restore_calico_hpp EXIT

# Run experiments for each page size
for page_size_kb in "${PAGE_SIZES[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH ${page_size_kb}KB PAGE SIZE"
    echo "######################################################################"
    
    # Calculate page size in bytes
    page_size_bytes=$(calculate_page_size_bytes ${page_size_kb})
    
    # Update pageSize in calico.hpp and recompile
    update_page_size_and_compile ${page_size_bytes}
    
    # Create page-size-specific directory
    page_dir="${EXP_BASE_DIR}/${page_size_kb}kb"
    mkdir -p "${page_dir}"
    
    # Calculate number of pages
    num_pages=$(calculate_num_pages ${DATA_SIZE_GB} ${page_size_kb})
    
    # Create config file
    cat > "${page_dir}/config.json" <<EOF
{
  "description": "Configuration for ${page_size_kb}KB page size experiments",
  "page_size_kb": ${page_size_kb},
  "page_size_bytes": ${page_size_bytes},
  "data_size_gb": ${DATA_SIZE_GB},
  "num_pages": ${num_pages},
  "op_types": ["nonfiltered_count", "nonfiltered_sum"],
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
    
    echo "Running experiments with ${page_size_kb}KB pages..." >> "${GLOBAL_SUMMARY_FILE}"
    
    # Run experiments for each layout type
    for layout_type in "${LAYOUT_TYPES[@]}"; do
        echo "  Testing layout type: ${layout_type}" >> "${GLOBAL_SUMMARY_FILE}"
        
        # Run experiments for each operation type
        for op_type in "${OP_TYPES[@]}"; do
            echo "    Testing operation type: ${op_type}" >> "${GLOBAL_SUMMARY_FILE}"
            
            # Configuration 1: mmap (without huge pages)
            run_olap_experiment ${page_size_kb} "mmap" "${op_type}" "${layout_type}" "Calico mmap without huge pages" \
                "TRADITIONAL=0" "NOHUGEPAGE_FRAME_MEM=1"
            
            # Configuration 2: mmap_huge (using huge pages)
            run_olap_experiment ${page_size_kb} "mmap_huge" "${op_type}" "${layout_type}" "Calico mmap with huge pages" \
                "TRADITIONAL=0" "NOHUGEPAGE_FRAME_MEM=0"
            
            # Configuration 3: calico (traditional buffer pool with array translator, huge pages)
            run_olap_experiment ${page_size_kb} "calico" "${op_type}" "${layout_type}" "Calico traditional with array translator and huge pages" \
                "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=0"
            
            # Configuration 4: chained_hash (traditional buffer pool with chained hash, huge pages)
            run_olap_experiment ${page_size_kb} "chained_hash" "${op_type}" "${layout_type}" "Calico traditional with chained hash and huge pages" \
                "TRADITIONAL=1" "TRADHASH=1" "NOHUGEPAGE_FRAME_MEM=0"
            
            # Configuration 5: lockfree_hash (traditional buffer pool with lockfree hash, huge pages)
            run_olap_experiment ${page_size_kb} "lockfree_hash" "${op_type}" "${layout_type}" "Calico traditional with lockfree hash and huge pages" \
                "TRADITIONAL=1" "TRADHASH=4" "NOHUGEPAGE_FRAME_MEM=0"
        done
    done
    
    echo "Completed experiments with ${page_size_kb}KB pages" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH ${page_size_kb}KB PAGE SIZE"
    echo "######################################################################"
    echo

    restore_calico_hpp
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Configurations tested per page size:"
echo "  - mmap: VMCache without huge pages"
echo "  - mmap_huge: VMCache with huge pages"
echo "  - calico: Traditional buffer pool with array translator (huge pages)"
echo "  - chained_hash: Traditional buffer pool with chained hash (huge pages)"
echo "  - lockfree_hash: Traditional buffer pool with lockfree hash (huge pages)"
echo "  Total per page size: 5 configurations × 2 operation types × 2 layout types = 20 experiments"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by page size:"
for page_size_kb in "${PAGE_SIZES[@]}"; do
    echo "--- ${page_size_kb}KB pages ---"
    find "${EXP_BASE_DIR}/${page_size_kb}kb" -name "*.csv" -o -name "*.json" 2>/dev/null | sort || true
done

echo
echo "Running results extraction..."
./extract_results_olap.sh

echo
echo "Experiment completed! Check ${EXP_BASE_DIR} for all results."
echo "Original calico.hpp has been restored."
