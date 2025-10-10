#!/bin/bash

# HNSW Benchmark Experiment Script
# Tests different Calico configurations with varying thread counts

set -e  # Exit on error

# Configuration
DATASET="sift1m"
M=16
EF_CONSTRUCTION=200
K=10
EF_SEARCH=50

# Physical memory size (GB) - make sure this is large enough to hold all data
# For SIFT1M: ~1GB base vectors + index overhead, 24GB should be plenty
export PHYSGB=4G
export VIRTGB=8G
# Thread counts to test
THREAD_COUNTS=(1 4 8 16 32)

# Base directory for results
RESULTS_BASE_DIR="./exps/in-memory/hnsw"

# Calico configurations to test
# Format: config_name:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH
CONFIGS=(
    "chained_hash:1:1:0:1"
    "openaddr_hash:1:2:0:1"
    "lockfree_hash:1:4:0:1"
    "vmcache:0:0:1:1"
    "calico:1:0:0:1"
    "mmap_oscache:2:0:1:0"
)

# Function to run a single benchmark
run_benchmark() {
    local config_name=$1
    local traditional=$2
    local tradhash=$3
    local nohugepage=$4
    local prefetch=$5
    local threads=$6
    local output_dir=$7
    
    echo "=========================================="
    echo "Running: $config_name with $threads threads"
    echo "TRADITIONAL=$traditional, TRADHASH=$tradhash"
    echo "NOHUGEPAGE_FRAME_MEM=$nohugepage, PREFETCH=$prefetch"
    echo "=========================================="
    
    # Set environment variables
    export TRADITIONAL=$traditional
    export TRADHASH=$tradhash
    export NOHUGEPAGE_FRAME_MEM=$nohugepage
    export PREFETCH=$prefetch
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Output file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${output_dir}/${config_name}_threads${threads}_${timestamp}.log"
    
    # Run benchmark
    python3 benchmark_calico_hnsw.py \
        --config "$config_name" \
        --threads "$threads" \
        --dataset "$DATASET" \
        --M "$M" \
        --ef_construction "$EF_CONSTRUCTION" \
        --k "$K" \
        --ef_search "$EF_SEARCH" \
        2>&1 | tee "$output_file"
    
    echo "Results saved to: $output_file"
    echo ""
}

# Main execution
main() {
    echo "========================================"
    echo "HNSW Benchmark Experiment"
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "HNSW Parameters: M=$M, ef_construction=$EF_CONSTRUCTION"
    echo "Search Parameters: k=$K, ef_search=$EF_SEARCH"
    echo "Physical Memory: ${PHYSGB}"
    echo "Thread Counts: ${THREAD_COUNTS[*]}"
    echo "========================================"
    echo ""
    
    # Build the project first
    echo "Building project..."
    make clean && make -j$(nproc)
    echo "Build complete"
    echo ""
    
    # Run benchmarks for each thread count
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "========================================"
        echo "Testing with $threads threads"
        echo "========================================"
        
        # Create results directory for this thread count
        results_dir="${RESULTS_BASE_DIR}/${threads}"
        mkdir -p "$results_dir"
        
        # Test each configuration
        for config_spec in "${CONFIGS[@]}"; do
            IFS=':' read -r config_name traditional tradhash nohugepage prefetch <<< "$config_spec"
            
            run_benchmark \
                "$config_name" \
                "$traditional" \
                "$tradhash" \
                "$nohugepage" \
                "$prefetch" \
                "$threads" \
                "$results_dir"
            
            # Small delay between runs
            sleep 2
        done
        
        echo ""
    done
    
    echo "========================================"
    echo "All experiments completed!"
    echo "Results directory: $RESULTS_BASE_DIR"
    echo "========================================"
    echo ""
    
    # Extract and summarize results
    echo "Extracting results..."
    ./extract_results_hnsw.sh
}

# Run main function
main "$@"
