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

# Base configuration names
BASE_CONFIGS=("calico_array" "calico_mmap" "calico_chained_hash" "calico_openaddr_hash" "calico_lockfree_hash")

# All configuration combinations
CONFIGS=()
for base in "${BASE_CONFIGS[@]}"; do
    CONFIGS+=("${base}_noprefetch_nohuge")
    CONFIGS+=("${base}_noprefetch_huge")
    CONFIGS+=("${base}_prefetch_nohuge")
    CONFIGS+=("${base}_prefetch_huge")
done

# Create summary CSV with headers
SUMMARY_FILE="${GRAPH_EXP_DIR}/graph_results_summary.csv"
echo "Config,BaseConfig,Prefetch,HugePages,Nodes,TraversalTime_ms,LLC_misses_per_op,DTLB_misses_per_op,Cycles_per_op,Instructions_per_op,IPC" > "${SUMMARY_FILE}"

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
    
    # Parse config components
    if [[ $config =~ ^(.*)_(noprefetch|prefetch)_(nohuge|huge)$ ]]; then
        base_config="${BASH_REMATCH[1]}"
        prefetch_setting="${BASH_REMATCH[2]}"
        huge_setting="${BASH_REMATCH[3]}"
        
        # Convert to Yes/No format
        prefetch_yn=$([ "$prefetch_setting" = "prefetch" ] && echo "Yes" || echo "No")
        huge_yn=$([ "$huge_setting" = "huge" ] && echo "Yes" || echo "No")
    else
        base_config="$config"
        prefetch_yn="N/A"
        huge_yn="N/A"
    fi
    
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
    echo "${config},${base_config},${prefetch_yn},${huge_yn},${nodes},${time_ms},${llc_misses},${dtlb_misses},${cycles_per_op},${instructions_per_op},${ipc}" >> "${SUMMARY_FILE}"
    
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

This analysis covers 5 base configurations with 4 combinations each (20 total):
- Base Configs: calico_array, calico_mmap, calico_chained_hash, calico_openaddr_hash, calico_lockfree_hash
- Combinations: PREFETCH={0,1} × NOHUGEPAGE_FRAME_MEM={0,1}

EOF

# Overall performance ranking
echo "Overall Performance Ranking (by traversal time, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by traversal time (column 6), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k6 -n | head -10 | while IFS=',' read -r config base_config prefetch huge nodes time_ms rest; do
        if [ "${time_ms}" != "N/A" ]; then
            echo "  ${config}: ${time_ms} ms (Base: ${base_config}, Prefetch: ${prefetch}, Huge: ${huge})" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "Base Configuration Comparison (best time for each base config):" >> "${ANALYSIS_FILE}"
for base in "${BASE_CONFIGS[@]}"; do
    best_config=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1)
    if [ -n "$best_config" ]; then
        time_ms=$(echo "$best_config" | cut -d',' -f1)
        config_name=$(echo "$best_config" | cut -d',' -f2)
        echo "  ${base}: ${time_ms} ms (${config_name})" >> "${ANALYSIS_FILE}"
    fi
done

echo "" >> "${ANALYSIS_FILE}"
echo "Prefetch Impact Analysis:" >> "${ANALYSIS_FILE}"
echo "Average performance improvement with prefetch enabled:" >> "${ANALYSIS_FILE}"
for base in "${BASE_CONFIGS[@]}"; do
    # Get average time for prefetch vs no-prefetch
    prefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $3=="Yes" && $6!="N/A" {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}')
    noprefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $3=="No" && $6!="N/A" {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
    if [ "$prefetch_avg" != "N/A" ] && [ "$noprefetch_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($noprefetch_avg - $prefetch_avg) / $noprefetch_avg * 100" | bc -l 2>/dev/null || echo "N/A")
        if [ "$improvement" != "N/A" ]; then
            echo "  ${base}: ${improvement}% improvement (${noprefetch_avg}ms → ${prefetch_avg}ms)" >> "${ANALYSIS_FILE}"
        fi
    fi
done

echo "" >> "${ANALYSIS_FILE}"
echo "Huge Pages Impact Analysis:" >> "${ANALYSIS_FILE}"
echo "Average performance improvement with huge pages enabled:" >> "${ANALYSIS_FILE}"
for base in "${BASE_CONFIGS[@]}"; do
    # Get average time for huge vs no-huge
    huge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $4=="Yes" && $6!="N/A" {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}')
    nohuge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $4=="No" && $6!="N/A" {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
    if [ "$huge_avg" != "N/A" ] && [ "$nohuge_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($nohuge_avg - $huge_avg) / $nohuge_avg * 100" | bc -l 2>/dev/null || echo "N/A")
        if [ "$improvement" != "N/A" ]; then
            echo "  ${base}: ${improvement}% improvement (${nohuge_avg}ms → ${huge_avg}ms)" >> "${ANALYSIS_FILE}"
        fi
    fi
done

echo "" >> "${ANALYSIS_FILE}"
echo "Memory Efficiency Ranking (by LLC misses/op, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by LLC misses per op (column 7), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k7 -n | head -10 | while IFS=',' read -r config base_config prefetch huge nodes time_ms llc_misses rest; do
        if [ "${llc_misses}" != "N/A" ]; then
            echo "  ${config}: ${llc_misses} LLC misses/op (Base: ${base_config})" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "TLB Efficiency Ranking (by DTLB misses/op, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by DTLB misses per op (column 8), skip header  
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k8 -n | head -10 | while IFS=',' read -r config base_config prefetch huge nodes time_ms llc_misses dtlb_misses rest; do
        if [ "${dtlb_misses}" != "N/A" ]; then
            echo "  ${config}: ${dtlb_misses} DTLB misses/op (Base: ${base_config})" >> "${ANALYSIS_FILE}"
        fi
    done
fi

# Create technique-specific summaries
echo "" >> "${ANALYSIS_FILE}"
echo "Technique-Specific Best Performers:" >> "${ANALYSIS_FILE}"
echo "===================================" >> "${ANALYSIS_FILE}"

# Best performer for each technique combination
echo "Best with Prefetch + Huge Pages:" >> "${ANALYSIS_FILE}"
tail -n +2 "${SUMMARY_FILE}" | awk -F',' '$3=="Yes" && $4=="Yes" && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1 | while IFS=',' read -r time config; do
    echo "  ${config}: ${time} ms" >> "${ANALYSIS_FILE}"
done

echo "Best with Prefetch + No Huge Pages:" >> "${ANALYSIS_FILE}"
tail -n +2 "${SUMMARY_FILE}" | awk -F',' '$3=="Yes" && $4=="No" && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1 | while IFS=',' read -r time config; do
    echo "  ${config}: ${time} ms" >> "${ANALYSIS_FILE}"
done

echo "Best with No Prefetch + Huge Pages:" >> "${ANALYSIS_FILE}"
tail -n +2 "${SUMMARY_FILE}" | awk -F',' '$3=="No" && $4=="Yes" && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1 | while IFS=',' read -r time config; do
    echo "  ${config}: ${time} ms" >> "${ANALYSIS_FILE}"
done

echo "Best with No Prefetch + No Huge Pages:" >> "${ANALYSIS_FILE}"
tail -n +2 "${SUMMARY_FILE}" | awk -F',' '$3=="No" && $4=="No" && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1 | while IFS=',' read -r time config; do
    echo "  ${config}: ${time} ms" >> "${ANALYSIS_FILE}"
done

# Create pivot table summaries
PIVOT_PREFETCH="${GRAPH_EXP_DIR}/graph_prefetch_analysis.csv"
PIVOT_HUGEPAGES="${GRAPH_EXP_DIR}/graph_hugepages_analysis.csv"
PIVOT_BASECONFIG="${GRAPH_EXP_DIR}/graph_baseconfig_analysis.csv"

echo "BaseConfig,Prefetch_Avg_ms,NoPrefetch_Avg_ms,Improvement_%" > "${PIVOT_PREFETCH}"
for base in "${BASE_CONFIGS[@]}"; do
    prefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $3=="Yes" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    noprefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $3=="No" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    if [ "$prefetch_avg" != "N/A" ] && [ "$noprefetch_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($noprefetch_avg - $prefetch_avg) / $noprefetch_avg * 100" | bc -l 2>/dev/null || echo "N/A")
    else
        improvement="N/A"
    fi
    
    echo "${base},${prefetch_avg},${noprefetch_avg},${improvement}" >> "${PIVOT_PREFETCH}"
done

echo "BaseConfig,HugePages_Avg_ms,NoHugePages_Avg_ms,Improvement_%" > "${PIVOT_HUGEPAGES}"
for base in "${BASE_CONFIGS[@]}"; do
    huge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $4=="Yes" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    nohuge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $4=="No" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    if [ "$huge_avg" != "N/A" ] && [ "$nohuge_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($nohuge_avg - $huge_avg) / $nohuge_avg * 100" | bc -l 2>/dev/null || echo "N/A")
    else
        improvement="N/A"
    fi
    
    echo "${base},${huge_avg},${nohuge_avg},${improvement}" >> "${PIVOT_HUGEPAGES}"
done

echo "BaseConfig,Best_Time_ms,Best_Config,Avg_Time_ms" > "${PIVOT_BASECONFIG}"
for base in "${BASE_CONFIGS[@]}"; do
    best_result=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $6!="N/A" {print $6","$1}' | sort -t',' -k1 -n | head -1)
    avg_time=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$2==base && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    if [ -n "$best_result" ]; then
        best_time=$(echo "$best_result" | cut -d',' -f1)
        best_config=$(echo "$best_result" | cut -d',' -f2)
        echo "${base},${best_time},${best_config},${avg_time}" >> "${PIVOT_BASECONFIG}"
    else
        echo "${base},N/A,N/A,N/A" >> "${PIVOT_BASECONFIG}"
    fi
done

echo "Analysis file: ${ANALYSIS_FILE}"
echo "Summary files generated:"
echo "  Main summary: ${SUMMARY_FILE}"
echo "  Prefetch analysis: ${PIVOT_PREFETCH}"
echo "  Huge pages analysis: ${PIVOT_HUGEPAGES}"
echo "  Base config analysis: ${PIVOT_BASECONFIG}"
echo
echo "To view the complete analysis:"
echo "cat ${ANALYSIS_FILE}"
echo
echo "To view pivot tables:"
echo "echo 'Prefetch Impact:' && column -t -s',' ${PIVOT_PREFETCH}"
echo "echo 'Huge Pages Impact:' && column -t -s',' ${PIVOT_HUGEPAGES}"
echo "echo 'Base Config Comparison:' && column -t -s',' ${PIVOT_BASECONFIG}"
