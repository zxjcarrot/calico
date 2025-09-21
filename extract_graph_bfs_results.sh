#!/bin/bash

# extract_graph_results.sh - Extract and summarize graph BFS experiment results
# This script processes performance counter data and graph traversal metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_EXP_DIR="${SCRIPT_DIR}/exps/in-memory/graph_bfs"

# Check if graph experiment directory exists
if [ ! -d "${GRAPH_EXP_DIR}" ]; then
    echo "Error: Graph experiment directory not found: ${GRAPH_EXP_DIR}"
    echo "Please run ./exp_in_memory_graph_bfs.sh first"
    exit 1
fi

echo "Extracting graph BFS experiment results..."
echo "Graph experiment directory: ${GRAPH_EXP_DIR}"

# Configuration names
CONFIGS=("calico_mmap" "calico_array" "calico_array_nohuge" "calico_unordered_hash" "calico_openaddr_hash" "calico_lockfree_hash" "calico_lockfree_hash_nohuge")

# Create summary CSV with headers
SUMMARY_FILE="${GRAPH_EXP_DIR}/graph_results_summary.csv"
echo "Config,Nodes,TraversalTime_ms,LLC_misses_per_op,DTLB_misses_per_op,Cycles_per_op,Instructions_per_op,IPC" > "${SUMMARY_FILE}"

echo "Processing results for each configuration..."

for config in "${CONFIGS[@]}"; do
    config_dir="${GRAPH_EXP_DIR}/"
    
    
    perf_csv="${config_dir}/${config}_perf.csv"
    tx_csv="${config_dir}/${config}_tx.csv"
    
    if [ ! -f "${perf_csv}" ] || [ ! -f "${tx_csv}" ]; then
        echo "Warning: Missing CSV files for ${config}"
        echo "  Expected: ${perf_csv}"
        echo "  Expected: ${tx_csv}"
        continue
    fi
    
    echo "Processing ${config}..."
    
    # Extract performance counters from label-based CSV format
    if [ -s "${perf_csv}" ]; then
        # Extract specific metrics by label
        llc_misses=$(awk -F',' '/^llc_misses/ {print $3}' "${perf_csv}" | head -1)
        dtlb_misses=$(awk -F',' '/^dtlb_misses/ {print $3}' "${perf_csv}" | head -1)
        cycles_per_op=$(awk -F',' '/^cycles/ {print $3}' "${perf_csv}" | head -1)
        instructions_per_op=$(awk -F',' '/^instructions/ {print $3}' "${perf_csv}" | head -1)
        ipc=$(awk -F',' '/^derived_ipc/ {print $2}' "${perf_csv}" | head -1)
        
        # Handle empty values
        [ -z "$llc_misses" ] && llc_misses="N/A"
        [ -z "$dtlb_misses" ] && dtlb_misses="N/A"
        [ -z "$cycles_per_op" ] && cycles_per_op="N/A"
        [ -z "$instructions_per_op" ] && instructions_per_op="N/A"
        [ -z "$ipc" ] && ipc="N/A"
    else
        llc_misses="N/A"
        dtlb_misses="N/A"
        cycles_per_op="N/A"
        instructions_per_op="N/A"
        ipc="N/A"
    fi
    
    # Extract graph BFS metrics from TX_CSV
    if [ -s "${tx_csv}" ]; then
        # Look for line containing "graph_bfs nodes="
        graph_line=$(grep "graph_bfs nodes=" "${tx_csv}" 2>/dev/null || echo "")
        
        if [ -n "${graph_line}" ]; then
            # Extract nodes and time_ms from format: graph_bfs nodes=xxx neighbors_per_node=xxx visited=xxx value_sum=xxx time_ms=xxx
            nodes=$(echo "${graph_line}" | sed -n 's/.*nodes=\([0-9]*\).*/\1/p')
            time_ms=$(echo "${graph_line}" | sed -n 's/.*time_ms=\([0-9.]*\).*/\1/p')
        else
            nodes="N/A"
            time_ms="N/A"
        fi
    else
        nodes="N/A"
        time_ms="N/A"
    fi
    
    # Add row to summary
    echo "${config},${nodes},${time_ms},${llc_misses},${dtlb_misses},${cycles_per_op},${instructions_per_op},${ipc}" >> "${SUMMARY_FILE}"
    
    echo "  Nodes: ${nodes}"
    echo "  Traversal Time: ${time_ms} ms" 
    echo "  LLC misses/op: ${llc_misses}"
    echo "  DTLB misses/op: ${dtlb_misses}"
    echo "  Cycles/op: ${cycles_per_op}"
    echo "  Instructions/op: ${instructions_per_op}"
    echo "  IPC: ${ipc}"
    echo
done

echo "========================================================"
echo "Graph BFS Results Summary Generated!"
echo "========================================================"
echo "Summary file: ${SUMMARY_FILE}"
echo
echo "Results table:"
echo "=============="
column -t -s',' "${SUMMARY_FILE}"

echo
echo "Detailed analysis:"
echo "=================="

# Create a more detailed analysis
ANALYSIS_FILE="${GRAPH_EXP_DIR}/graph_analysis.txt"
cat > "${ANALYSIS_FILE}" <<EOF
Graph BFS Experiment Analysis
Generated on: $(date)

Configuration Comparison:
========================

EOF

# Add performance ranking
echo "Performance Ranking (by traversal time, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by traversal time (column 3), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k3 -n | while IFS=',' read -r config nodes time_ms rest; do
        if [ "${time_ms}" != "N/A" ]; then
            echo "  ${config}: ${time_ms} ms (${nodes} nodes)" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "Memory Efficiency Ranking (by LLC misses/op, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by LLC misses per op (column 4), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k4 -n | while IFS=',' read -r config nodes time_ms llc_misses rest; do
        if [ "${llc_misses}" != "N/A" ]; then
            echo "  ${config}: ${llc_misses} LLC misses/op" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "TLB Efficiency Ranking (by DTLB misses/op, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by DTLB misses per op (column 5), skip header  
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k5 -n | while IFS=',' read -r config nodes time_ms llc_misses dtlb_misses rest; do
        if [ "${dtlb_misses}" != "N/A" ]; then
            echo "  ${config}: ${dtlb_misses} DTLB misses/op" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "Analysis file: ${ANALYSIS_FILE}"
echo
echo "To view the complete analysis:"
echo "cat ${ANALYSIS_FILE}"
