#!/bin/bash

# exp_in_memory_graph.sh - In-memory graph benchmark experiments for different Calico configurations
# This script runs graph BFS traversal experiments for various buffer management strategies

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/graph_bfs"

# Thread counts to test
THREAD_COUNTS=(1 4 8 16 32)

# Common settings for all graph experiments
export GRAPH=1
export RNDREAD=0
export BLOCK=/home/zxjcarrot/Workspace/calico/heapfile
export VIRTGB=48G
export PHYSGB=24G
export DATASIZE=5e6
export PERF=1

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

# Function to run a single graph experiment
run_graph_experiment() {
    local thread_count="$1"
    local config_name="$2"
    local description="$3"
    shift 3
    local env_vars=("$@")
    
    # Create thread-specific directory
    local thread_dir="${EXP_BASE_DIR}/${thread_count}"
    mkdir -p "${thread_dir}"
    
    echo "=================================================================="
    echo "Running graph experiment: ${config_name} with ${thread_count} threads"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Set thread count
    export THREADS=${thread_count}
    
    # Set environment variables for this configuration
    for env_var in "${env_vars[@]}"; do
        export ${env_var}
        echo "Set ${env_var}"
    done
    
    # Set output file paths in thread-specific directory
    export TX_CSV="${thread_dir}/${config_name}_tx.csv"
    export PERF_CSV="${thread_dir}/${config_name}_perf.csv"
    
    echo "Thread count: ${thread_count}"
    echo "Configuration: ${config_name}"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo
    
    # Run the experiment
    echo "Starting graph BFS experiment..."
    ./calico
    
    echo "Graph experiment ${config_name} with ${thread_count} threads completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
}

# Function to run experiments for a specific base configuration with all combinations
run_config_combinations() {
    local thread_count="$1"
    local base_name="$2"
    local base_description="$3"
    shift 3
    local base_env_vars=("$@")
    
    echo "=================================================================="
    echo "Running all combinations for: ${base_name} with ${thread_count} threads"
    echo "=================================================================="
    
    # Combination 1: No prefetch, No huge pages
    run_graph_experiment ${thread_count} "${base_name}_noprefetch_nohuge" \
        "${base_description} (no prefetch, no huge pages)" \
        "${base_env_vars[@]}" "PREFETCH=0" "NOHUGEPAGE_FRAME_MEM=1"
    
    # Combination 2: No prefetch, With huge pages
    run_graph_experiment ${thread_count} "${base_name}_noprefetch_huge" \
        "${base_description} (no prefetch, with huge pages)" \
        "${base_env_vars[@]}" "PREFETCH=0" "NOHUGEPAGE_FRAME_MEM=0"
    
    # Combination 3: With prefetch, No huge pages
    run_graph_experiment ${thread_count} "${base_name}_prefetch_nohuge" \
        "${base_description} (with prefetch, no huge pages)" \
        "${base_env_vars[@]}" "PREFETCH=1" "NOHUGEPAGE_FRAME_MEM=1"
    
    # Combination 4: With prefetch, With huge pages
    run_graph_experiment ${thread_count} "${base_name}_prefetch_huge" \
        "${base_description} (with prefetch, with huge pages)" \
        "${base_env_vars[@]}" "PREFETCH=1" "NOHUGEPAGE_FRAME_MEM=0"
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "Graph BFS Experiment Summary - Multi-threaded" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Thread counts tested: ${THREAD_COUNTS[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "Common settings: GRAPH=1 RNDREAD=0 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile VIRTGB=48G PHYSGB=24G DATASIZE=5e6 PERF=1" >> "${GLOBAL_SUMMARY_FILE}"
echo "Testing combinations: PREFETCH=0|1, NOHUGEPAGE_FRAME_MEM=0|1 for each configuration" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

echo "######################################################################"
echo "STARTING GRAPH BFS EXPERIMENTS"
echo "######################################################################"

# Run experiments for each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    echo "######################################################################"
    echo "STARTING EXPERIMENTS WITH ${threads} THREADS"
    echo "######################################################################"
    
    echo "Running graph BFS experiments with ${threads} threads..." >> "${GLOBAL_SUMMARY_FILE}"
    
    # Configuration 1: Calico array (traditional buffer pool with array translator)
    run_config_combinations ${threads} "calico_array" "Calico with traditional buffer pool using array translator" \
        "TRADITIONAL=1" "TRADHASH=0"

    # Configuration 2: Calico mmap (virtual memory cache)
    run_config_combinations ${threads} "calico_mmap" "Calico with mmap (virtual memory cache)" \
        "TRADITIONAL=0" "TRADHASH=0"

    # Configuration 3: Calico chained_hash (traditional buffer pool with unordered_map)
    run_config_combinations ${threads} "calico_chained_hash" "Calico with traditional buffer pool using chained hash (unordered_map) translator" \
        "TRADITIONAL=1" "TRADHASH=1"

    # Configuration 4: Calico openaddr_hash (traditional buffer pool with open addressing)
    run_config_combinations ${threads} "calico_openaddr_hash" "Calico with traditional buffer pool using open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=2"
    
    # Configuration 5: Calico lockfree_hash (traditional buffer pool with lock free open addressing)
    run_config_combinations ${threads} "calico_lockfree_hash" "Calico with traditional buffer pool using lock free open addressing hash translator" \
        "TRADITIONAL=1" "TRADHASH=4"

    echo "Completed experiments with ${threads} threads" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"
    
    echo "######################################################################"
    echo "COMPLETED EXPERIMENTS WITH ${threads} THREADS"
    echo "######################################################################"
    echo
done

echo "######################################################################"
echo "COMPLETED ALL GRAPH BFS EXPERIMENTS"
echo "######################################################################"

echo "=================================================================="
echo "ALL GRAPH EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Configuration combinations tested per thread count:"
echo "  - calico_array: 4 combinations (noprefetch_nohuge, noprefetch_huge, prefetch_nohuge, prefetch_huge)"
echo "  - calico_mmap: 4 combinations (noprefetch_nohuge, noprefetch_huge, prefetch_nohuge, prefetch_huge)"
echo "  - calico_chained_hash: 4 combinations (noprefetch_nohuge, noprefetch_huge, prefetch_nohuge, prefetch_huge)"
echo "  - calico_openaddr_hash: 4 combinations (noprefetch_nohuge, noprefetch_huge, prefetch_nohuge, prefetch_huge)"
echo "  - calico_lockfree_hash: 4 combinations (noprefetch_nohuge, noprefetch_huge, prefetch_nohuge, prefetch_huge)"
echo "  Total per thread count: 20 experiment configurations"
echo "  Thread counts: ${THREAD_COUNTS[*]}"
echo "  Total experiments: $((20 * ${#THREAD_COUNTS[@]}))"
echo
echo "Directory structure:"
find "${EXP_BASE_DIR}" -type d | sort
echo
echo "Generated files by thread count:"
for threads in "${THREAD_COUNTS[@]}"; do
    echo "--- ${threads} threads ---"
    find "${EXP_BASE_DIR}/${threads}" -name "*.csv" 2>/dev/null | sort | head -5
    file_count=$(find "${EXP_BASE_DIR}/${threads}" -name "*.csv" 2>/dev/null | wc -l)
    echo "  (${file_count} files total)"
done

echo
echo "Running results extraction for graph experiments..."
./extract_graph_bfs_results.sh

echo
echo "Graph experiment completed! Check ${EXP_BASE_DIR} for all results."
