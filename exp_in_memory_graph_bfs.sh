#!/bin/bash

# exp_in_memory_graph.sh - In-memory graph benchmark experiments for different Calico configurations
# This script runs graph BFS traversal experiments for various buffer management strategies

set -e

# Base directory for experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_BASE_DIR="${SCRIPT_DIR}/exps/in-memory/graph_bfs"

# Common settings for all graph experiments
export GRAPH=1
export RNDREAD=0
export BLOCK=/home/zxjcarrot/Workspace/calico/heapfile
export VIRTGB=36G
export PHYSGB=24G
export DATASIZE=5e6
export PERF=1
export THREADS=1

# Build the project first
echo "Building calico..."
make clean && make -j$(nproc)

# Function to run a single graph experiment
run_graph_experiment() {
    local config_name="$1"
    local description="$2"
    shift 2
    local env_vars=("$@")
    
    echo "=================================================================="
    echo "Running graph experiment: ${config_name}"
    echo "Description: ${description}"
    echo "=================================================================="
    
    # Set environment variables for this configuration
    for env_var in "${env_vars[@]}"; do
        export ${env_var}
        echo "Set ${env_var}"
    done
    
    # Set output file paths directly in base directory
    export TX_CSV="${EXP_BASE_DIR}/${config_name}_tx.csv"
    export PERF_CSV="${EXP_BASE_DIR}/${config_name}_perf.csv"
    
    echo "Configuration: ${config_name}"
    echo "Output files:"
    echo "  TX CSV: ${TX_CSV}"
    echo "  PERF CSV: ${PERF_CSV}"
    echo
    
    # Run the experiment
    echo "Starting graph BFS experiment..."
    ./calico
    
    echo "Graph experiment ${config_name} completed successfully."
    echo
    
    # Unset configuration-specific environment variables
    for env_var in "${env_vars[@]}"; do
        var_name=$(echo "${env_var}" | cut -d'=' -f1)
        unset ${var_name}
    done
}

# Create main experiment directory
mkdir -p "${EXP_BASE_DIR}"

# Create a global summary file
GLOBAL_SUMMARY_FILE="${EXP_BASE_DIR}/experiment_summary.txt"
echo "Graph BFS Experiment Summary" > "${GLOBAL_SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${GLOBAL_SUMMARY_FILE}"
echo "Common settings: GRAPH=1 RNDREAD=0 BLOCK=/home/zxjcarrot/Workspace/calico/heapfile VIRTGB=28G PHYSGB=24G DATASIZE=5e6 PERF=1 THREADS=1" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

echo "######################################################################"
echo "STARTING GRAPH BFS EXPERIMENTS"
echo "######################################################################"

echo "Running graph BFS experiments..." >> "${GLOBAL_SUMMARY_FILE}"

# Experiment 1: Calico mmap (virtual memory cache)
run_graph_experiment "calico_mmap" "Calico with mmap (virtual memory cache)" \
    "TRADITIONAL=0" "TRADHASH=0"

# Experiment 2: Calico array (traditional buffer pool with array translator)
run_graph_experiment "calico_array" "Calico with traditional buffer pool using array translator" \
    "TRADITIONAL=1" "TRADHASH=0"

# Experiment 3: Calico array + nohuge (array translator without huge pages)
run_graph_experiment "calico_array_nohuge" "Calico array translator without huge pages for frame memory" \
    "TRADITIONAL=1" "TRADHASH=0" "NOHUGEPAGE_FRAME_MEM=1"

# Experiment 4: Calico unordered_hash (traditional buffer pool with unordered_map)
run_graph_experiment "calico_unordered_hash" "Calico with traditional buffer pool using unordered_map translator" \
    "TRADITIONAL=1" "TRADHASH=1"

# Experiment 5: Calico openaddr_hash (traditional buffer pool with open addressing)
run_graph_experiment "calico_openaddr_hash" "Calico with traditional buffer pool using open addressing hash translator" \
    "TRADITIONAL=1" "TRADHASH=2"

# Experiment 6: Calico lockfree_hash (traditional buffer pool with lock free open addressing)
run_graph_experiment "calico_lockfree_hash" "Calico with traditional buffer pool using lock free open addressing hash translator" \
    "TRADITIONAL=1" "TRADHASH=4"

echo "Completed all graph experiments" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

echo "######################################################################"
echo "COMPLETED ALL GRAPH BFS EXPERIMENTS"
echo "######################################################################"

echo "=================================================================="
echo "ALL GRAPH EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Results saved in: ${EXP_BASE_DIR}"
echo "Global summary file: ${GLOBAL_SUMMARY_FILE}"
echo
echo "Generated CSV files:"
ls -la "${EXP_BASE_DIR}"/*.csv 2>/dev/null || echo "No CSV files found"

echo
echo "Running results extraction for graph experiments..."
./extract_graph_bfs_results.sh

echo
echo "Graph experiment completed! Check ${EXP_BASE_DIR} for all results."
