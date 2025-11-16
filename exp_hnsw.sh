#!/bin/bash

# HNSW Benchmark Experiment Script
# Tests different Calico configurations with varying thread counts

set -e  # Exit on error

# Configuration
#DATASET="sift1m"
#DATASET="deep10M"
DATASET="sift10M"
M=16
EF_CONSTRUCTION=100
K=10
EF_SEARCH=50

# Physical memory size (GB) - make sure this is large enough to hold all data
# For SIFT1M: ~1GB base vectors + index overhead, 24GB should be plenty
export PHYSGB=16G
export VIRTGB=24G
# Thread counts to test
#THREAD_COUNTS=(64 32 16 8 4 1)
THREAD_COUNTS=(64)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory for results
RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/in-memory/hnsw/${DATASET}/"

# Usearch index file location (built once, used by all search runs)
USEARCH_INDEX_FILE="${SCRIPT_DIR}/exps/usearch_index.bin"

# Calico configurations to test
# Format: config_name:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH
CONFIGS=(
    # "usearch:0:0:0:0"
    # "chained_hash:1:1:0:1"
    # "openaddr_hash:1:2:0:1"
    # "lockfree_hash:1:4:0:1"
    "vmcache:0:0:1:1"
    "calico:1:0:0:1"
    # "mmap_oscache:2:0:1:0"
    # "calico-2level:1:5:0:1"
    # "calico-3level:1:6:0:1"
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
    
    # Construct base python args
    PY_ARGS=(
        -u benchmark_calico_hnsw.py
        --config "$config_name"
        --threads "$threads"
        --dataset "$DATASET"
        --M "$M"
        --ef_construction "$EF_CONSTRUCTION"
        --k "$K"
        --ef_search "$EF_SEARCH"
    )

    # For usearch, pass the library and mode if provided via RUN_MODE env
    if [ "$config_name" = "usearch" ]; then
        PY_ARGS+=(--library usearch)
        if [ -n "$RUN_MODE" ]; then
            PY_ARGS+=(--mode "$RUN_MODE")
        fi
        if [ "$RUN_MODE" = "index" ] || [ "$RUN_MODE" = "search" ]; then
            PY_ARGS+=(--index_file "$USEARCH_INDEX_FILE")
        fi
    fi

    # Run benchmark
    python3 "${PY_ARGS[@]}" 2>&1 | tee "$output_file"
    
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
    
    # Build the project first
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
            
            # Build usearch index once before the loop (on first iteration over threads)
            if [ "$config_name" = "usearch" ] && [ "$threads" = "${THREAD_COUNTS[0]}" ]; then
                mkdir -p "$(dirname "$USEARCH_INDEX_FILE")"
                RUN_MODE="index" \
                run_benchmark \
                    "$config_name" \
                    "$traditional" \
                    "$tradhash" \
                    "$nohugepage" \
                    "$prefetch" \
                    "$threads" \
                    "$results_dir"
            fi

            # For usearch, run search mode for each thread count
            if [ "$config_name" = "usearch" ]; then
                RUN_MODE="search" \
                run_benchmark \
                    "$config_name" \
                    "$traditional" \
                    "$tradhash" \
                    "$nohugepage" \
                    "$prefetch" \
                    "$threads" \
                    "$results_dir"
            else
                # Non-usearch configs use default library (calico) and no mode
                unset RUN_MODE
                run_benchmark \
                    "$config_name" \
                    "$traditional" \
                    "$tradhash" \
                    "$nohugepage" \
                    "$prefetch" \
                    "$threads" \
                    "$results_dir"
            fi
            
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
