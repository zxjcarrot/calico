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

# Thread counts to process
THREAD_COUNTS=(1 4 8 16 32)

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
echo "ThreadCount,Config,BaseConfig,Prefetch,HugePages,Nodes,AvgTraversalTime_ms,OverallTime_ms,LLC_misses_per_op,DTLB_misses_per_op,Cycles_per_op,Instructions_per_op,IPC" > "${SUMMARY_FILE}"

echo "Processing results for each thread count and configuration..."

for threads in "${THREAD_COUNTS[@]}"; do
    thread_dir="${GRAPH_EXP_DIR}/${threads}"
    
    if [ ! -d "$thread_dir" ]; then
        echo "Warning: Thread directory $thread_dir does not exist, skipping..."
        continue
    fi
    
    echo "Processing results for ${threads} threads..."
    
    # Create thread-specific summary
    thread_summary_csv="${thread_dir}/results_summary.csv"
    echo "Config,BaseConfig,Prefetch,HugePages,Nodes,AvgTraversalTime_ms,OverallTime_ms,LLC_misses_per_op,DTLB_misses_per_op,Cycles_per_op,Instructions_per_op,IPC" > "$thread_summary_csv"

for config in "${CONFIGS[@]}"; do
    echo "  Processing $config..."
    
    perf_csv="${thread_dir}/${config}_perf.csv"
    tx_csv="${thread_dir}/${config}_tx.csv"
    
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
        # Look for line containing "bfs nodes=" (configuration line)
        config_line=$(grep "^bfs nodes=" "${tx_csv}" 2>/dev/null || echo "")
        # Look for line containing "BFS completed:" (results line)
        results_line=$(grep "BFS completed:" "${tx_csv}" 2>/dev/null || echo "")
        
        if [ -n "${config_line}" ]; then
            # Extract nodes from config line: bfs nodes=5000000 neighbors_per_node=22 nthreads=1 start_indices=[0]
            nodes=$(echo "${config_line}" | sed -n 's/.*nodes=\([0-9]*\).*/\1/p')
        else
            nodes="N/A"
        fi
        
        if [ -n "${results_line}" ]; then
            # Extract metrics from results line: BFS completed: nthreads=1 avg_traversal_time=12172ms overall_time=12175ms total_visited=5000000
            avg_time_ms=$(echo "${results_line}" | sed -n 's/.*avg_traversal_time=\([0-9.]*\)ms.*/\1/p')
            overall_time_ms=$(echo "${results_line}" | sed -n 's/.*overall_time=\([0-9.]*\)ms.*/\1/p')
            
            # Handle empty values
            [ -z "$avg_time_ms" ] && avg_time_ms="N/A"
            [ -z "$overall_time_ms" ] && overall_time_ms="N/A"
        else
            avg_time_ms="N/A"
            overall_time_ms="N/A"
        fi
    else
        nodes="N/A"
        avg_time_ms="N/A"
        overall_time_ms="N/A"
    fi
    
    # Add row to thread-specific summary
    echo "${config},${base_config},${prefetch_yn},${huge_yn},${nodes},${avg_time_ms},${overall_time_ms},${llc_misses},${dtlb_misses},${cycles_per_op},${instructions_per_op},${ipc}" >> "$thread_summary_csv"
    
    # Add row to global summary with thread count
    echo "${threads},${config},${base_config},${prefetch_yn},${huge_yn},${nodes},${avg_time_ms},${overall_time_ms},${llc_misses},${dtlb_misses},${cycles_per_op},${instructions_per_op},${ipc}" >> "${SUMMARY_FILE}"
    
    echo "  Nodes: ${nodes}"
    echo "  Avg Traversal Time: ${avg_time_ms} ms"
    echo "  Overall Time: ${overall_time_ms} ms"
    echo "  LLC misses/op: ${llc_misses}"
    echo "  DTLB misses/op: ${dtlb_misses}"
    echo "  Cycles/op: ${cycles_per_op}"
    echo "  Instructions/op: ${instructions_per_op}"
    echo "  IPC: ${ipc}"
    echo
done

    echo "  Thread-specific summary: $thread_summary_csv"
    echo
done

echo "========================================================"
echo "Graph BFS Results Summary Generated!"
echo "========================================================"
echo "Summary files created:"
echo "  Global summary: ${SUMMARY_FILE}"
for threads in "${THREAD_COUNTS[@]}"; do
    thread_summary="${GRAPH_EXP_DIR}/${threads}/results_summary.csv"
    if [ -f "$thread_summary" ]; then
        echo "  ${threads} threads: $thread_summary"
    fi
done
echo
echo "Global Results Summary (all thread counts):"
echo "============================================"
column -t -s',' "${SUMMARY_FILE}" | head -50
total_lines=$(wc -l < "${SUMMARY_FILE}")
if [ "$total_lines" -gt 50 ]; then
    echo "... (showing first 50 lines, total ${total_lines} lines)"
    echo "View complete results: ${SUMMARY_FILE}"
fi
echo

# Display individual thread summaries
for threads in "${THREAD_COUNTS[@]}"; do
    thread_summary="${GRAPH_EXP_DIR}/${threads}/results_summary.csv"
    if [ -f "$thread_summary" ]; then
        echo "Results for ${threads} threads:"
        echo "==============================="
        column -t -s',' "$thread_summary"
        echo
    fi
done

echo
echo "Detailed analysis:"
echo "=================="

# Create a more detailed analysis
ANALYSIS_FILE="${GRAPH_EXP_DIR}/graph_analysis.txt"
cat > "${ANALYSIS_FILE}" <<EOF
Graph BFS Experiment Analysis - Multi-threaded
Generated on: $(date)

Configuration Comparison:
========================

Thread counts tested: ${THREAD_COUNTS[*]}
Base Configs: calico_array, calico_mmap, calico_chained_hash, calico_openaddr_hash, calico_lockfree_hash
Combinations per base config: PREFETCH={0,1} × NOHUGEPAGE_FRAME_MEM={0,1} = 4 combinations
Total configurations: 5 base configs × 4 combinations × ${#THREAD_COUNTS[@]} thread counts = $((5 * 4 * ${#THREAD_COUNTS[@]}))

EOF

# Overall performance ranking
echo "Overall Performance Ranking (by avg traversal time, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by avg traversal time (column 7), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k7 -n | head -10 | while IFS=',' read -r threads config base_config prefetch huge nodes time_ms overall rest; do
        if [ "${time_ms}" != "N/A" ]; then
            echo "  ${config} (${threads} threads): ${time_ms} ms (Base: ${base_config}, Prefetch: ${prefetch}, Huge: ${huge})" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "Best Configuration per Thread Count:" >> "${ANALYSIS_FILE}"
for threads in "${THREAD_COUNTS[@]}"; do
    best_config=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v t="$threads" '$1==t && $7!="N/A" {print $7","$2","$3}' | sort -t',' -k1 -n | head -1)
    if [ -n "$best_config" ]; then
        time_ms=$(echo "$best_config" | cut -d',' -f1)
        config_name=$(echo "$best_config" | cut -d',' -f2)
        base_name=$(echo "$best_config" | cut -d',' -f3)
        echo "  ${threads} threads: ${time_ms} ms (${config_name}, base: ${base_name})" >> "${ANALYSIS_FILE}"
    fi
done

echo "" >> "${ANALYSIS_FILE}"
echo "Scalability Analysis (Average traversal time by thread count):" >> "${ANALYSIS_FILE}"
for threads in "${THREAD_COUNTS[@]}"; do
    avg_time=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v t="$threads" '$1==t && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    echo "  ${threads} threads: ${avg_time} ms average" >> "${ANALYSIS_FILE}"
done

echo "" >> "${ANALYSIS_FILE}"
echo "Prefetch Impact Analysis (across all thread counts):" >> "${ANALYSIS_FILE}"
echo "Average performance improvement with prefetch enabled:" >> "${ANALYSIS_FILE}"
for base in "${BASE_CONFIGS[@]}"; do
    # Get average time for prefetch vs no-prefetch (column 7 is avg_traversal_time_ms)
    prefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $4=="Yes" && $7!="N/A" {sum+=$7; count++} END {if(count>0) print sum/count; else print "N/A"}')
    noprefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $4=="No" && $7!="N/A" {sum+=$7; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
    if [ "$prefetch_avg" != "N/A" ] && [ "$noprefetch_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($noprefetch_avg - $prefetch_avg) / $noprefetch_avg * 100" | bc -l 2>/dev/null || echo "N/A")
        if [ "$improvement" != "N/A" ]; then
            echo "  ${base}: ${improvement}% improvement (${noprefetch_avg}ms → ${prefetch_avg}ms)" >> "${ANALYSIS_FILE}"
        fi
    fi
done

echo "" >> "${ANALYSIS_FILE}"
echo "Huge Pages Impact Analysis (across all thread counts):" >> "${ANALYSIS_FILE}"
echo "Average performance improvement with huge pages enabled:" >> "${ANALYSIS_FILE}"
for base in "${BASE_CONFIGS[@]}"; do
    # Get average time for huge vs no-huge (column 7 is avg_traversal_time_ms)
    huge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $5=="Yes" && $7!="N/A" {sum+=$7; count++} END {if(count>0) print sum/count; else print "N/A"}')
    nohuge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $5=="No" && $7!="N/A" {sum+=$7; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
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
    # Sort by LLC misses per op (column 9), skip header
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k9 -n | head -10 | while IFS=',' read -r threads config base_config prefetch huge nodes time_ms overall llc_misses rest; do
        if [ "${llc_misses}" != "N/A" ]; then
            echo "  ${config} (${threads} threads): ${llc_misses} LLC misses/op" >> "${ANALYSIS_FILE}"
        fi
    done
fi

echo "" >> "${ANALYSIS_FILE}"
echo "TLB Efficiency Ranking (by DTLB misses/op, lower is better):" >> "${ANALYSIS_FILE}"
if [ -s "${SUMMARY_FILE}" ]; then
    # Sort by DTLB misses per op (column 10), skip header  
    tail -n +2 "${SUMMARY_FILE}" | sort -t',' -k10 -n | head -10 | while IFS=',' read -r threads config base_config prefetch huge nodes time_ms overall llc_misses dtlb_misses rest; do
        if [ "${dtlb_misses}" != "N/A" ]; then
            echo "  ${config} (${threads} threads): ${dtlb_misses} DTLB misses/op" >> "${ANALYSIS_FILE}"
        fi
    done
fi

# Create pivot table summaries
PIVOT_THREADS="${GRAPH_EXP_DIR}/graph_thread_scaling.csv"
PIVOT_PREFETCH="${GRAPH_EXP_DIR}/graph_prefetch_analysis.csv"
PIVOT_HUGEPAGES="${GRAPH_EXP_DIR}/graph_hugepages_analysis.csv"

echo "ThreadCount,Avg_Time_ms,Best_Config,Best_Time_ms" > "${PIVOT_THREADS}"
for threads in "${THREAD_COUNTS[@]}"; do
    avg_time=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v t="$threads" '$1==t && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    best_result=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v t="$threads" '$1==t && $7!="N/A" {print $7","$2}' | sort -t',' -k1 -n | head -1)
    
    if [ -n "$best_result" ]; then
        best_time=$(echo "$best_result" | cut -d',' -f1)
        best_config=$(echo "$best_result" | cut -d',' -f2)
        echo "${threads},${avg_time},${best_config},${best_time}" >> "${PIVOT_THREADS}"
    else
        echo "${threads},${avg_time},N/A,N/A" >> "${PIVOT_THREADS}"
    fi
done

echo "BaseConfig,Prefetch_Avg_ms,NoPrefetch_Avg_ms,Improvement_%" > "${PIVOT_PREFETCH}"
for base in "${BASE_CONFIGS[@]}"; do
    prefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $4=="Yes" && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    noprefetch_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $4=="No" && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    if [ "$prefetch_avg" != "N/A" ] && [ "$noprefetch_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($noprefetch_avg - $prefetch_avg) / $noprefetch_avg * 100" | bc -l 2>/dev/null || echo "N/A")
    else
        improvement="N/A"
    fi
    
    echo "${base},${prefetch_avg},${noprefetch_avg},${improvement}" >> "${PIVOT_PREFETCH}"
done

echo "BaseConfig,HugePages_Avg_ms,NoHugePages_Avg_ms,Improvement_%" > "${PIVOT_HUGEPAGES}"
for base in "${BASE_CONFIGS[@]}"; do
    huge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $5=="Yes" && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    nohuge_avg=$(tail -n +2 "${SUMMARY_FILE}" | awk -F',' -v base="$base" '$3==base && $5=="No" && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    
    if [ "$huge_avg" != "N/A" ] && [ "$nohuge_avg" != "N/A" ]; then
        improvement=$(echo "scale=2; ($nohuge_avg - $huge_avg) / $nohuge_avg * 100" | bc -l 2>/dev/null || echo "N/A")
    else
        improvement="N/A"
    fi
    
    echo "${base},${huge_avg},${nohuge_avg},${improvement}" >> "${PIVOT_HUGEPAGES}"
done

echo "Analysis file: ${ANALYSIS_FILE}"
echo "Summary files generated:"
echo "  Main summary: ${SUMMARY_FILE}"
echo "  Thread scaling: ${PIVOT_THREADS}"
echo "  Prefetch analysis: ${PIVOT_PREFETCH}"
echo "  Huge pages analysis: ${PIVOT_HUGEPAGES}"
echo
echo "To view the complete analysis:"
echo "cat ${ANALYSIS_FILE}"
echo
echo "To view pivot tables:"
echo "echo 'Thread Scaling:' && column -t -s',' ${PIVOT_THREADS}"
echo "echo 'Prefetch Impact:' && column -t -s',' ${PIVOT_PREFETCH}"
echo "echo 'Huge Pages Impact:' && column -t -s',' ${PIVOT_HUGEPAGES}"
