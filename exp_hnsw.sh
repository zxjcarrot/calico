#!/bin/bash

# HNSW Benchmark Experiment Script
# Tests different Calico configurations with varying thread counts

set -e  # Exit on error

# Configuration
#DATASET="sift1m"
#DATASET="deep10M"
DATASET="${DATASET:-sift10M}"
M=16
EF_CONSTRUCTION=100
K=10
EF_SEARCH=50

# Physical memory size (GB) - make sure this is large enough to hold all data
# For SIFT1M: ~1GB base vectors + index overhead, 24GB should be plenty
export PHYSGB=14G
export VIRTGB=24G
export POOLGB=14G

# Thread counts to test
#THREAD_COUNTS=(64 32 16 8 4 1)
THREAD_COUNTS=(1 4 8 16 32 64)
#THREAD_COUNTS=(128)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_BLOCK_OVERRIDE="${BLOCK:-}"
cd "$SCRIPT_DIR"

default_block_path() {
    if [ -n "$USER_BLOCK_OVERRIDE" ]; then
        printf '%s\n' "$USER_BLOCK_OVERRIDE"
        return 0
    fi

    if [ -f "${SCRIPT_DIR}/heapfile" ]; then
        printf '%s\n' "${SCRIPT_DIR}/heapfile"
        return 0
    fi

    printf '%s\n' "${SCRIPT_DIR}/heapfile.predicache_bench"
}

predicache_block_path() {
    local target_config_name="${1:-$config_name}"

    if [ -n "$USER_BLOCK_OVERRIDE" ]; then
        printf '%s\n' "$USER_BLOCK_OVERRIDE"
        return 0
    fi

    printf '%s\n' "${SCRIPT_DIR}/heapfile.${target_config_name}.${DATASET}"
}

ensure_block_file() {
    local block_path="$1"
    local block_size="${2:-20G}"

    if [ -f "$block_path" ]; then
        return 0
    fi

    echo "Creating sparse block file at ${block_path} (${block_size})..."
    if ! fallocate -l "$block_size" "$block_path"; then
        truncate -s "$block_size" "$block_path"
    fi
}

cleanup_index_file() {
    local path="$1"
    local description="$2"

    if [ -z "$path" ]; then
        return 0
    fi

    if [ -e "$path" ]; then
        echo "Removing stale ${description}: ${path}"
        rm -f "$path"
    fi
}

cleanup_reusable_index_files() {
    local cleaned_predicache_block=0
    local cleaned_usearch_index=0

    for config_spec in "${CONFIGS[@]}"; do
        IFS=':' read -r config_name traditional tradhash nohugepage prefetch <<< "$config_spec"

        if [ "$config_name" = "usearch" ] && [ "$cleaned_usearch_index" -eq 0 ]; then
            cleanup_index_file "$USEARCH_INDEX_FILE" "Usearch index file"
            cleaned_usearch_index=1
        fi

        if [[ "$config_name" == predicache* ]] && [ "$cleaned_predicache_block" -eq 0 ]; then
            cleanup_index_file "$(predicache_block_path "$config_name")" "PrediCache heapfile"
            cleaned_predicache_block=1
        fi
    done
}

# Add predicache python module to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}"
DEFAULT_BLOCK_PATH="$(default_block_path)"
export BLOCK="$DEFAULT_BLOCK_PATH"
ensure_block_file "$BLOCK" "20G"

PYTHON_EXECUTABLE=$(command -v python3 || true)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "Error: python3 not found in PATH."
    exit 1
fi

MEMORY_BW_METRIC=${MEMORY_BW_METRIC:-nps1_die_to_dram}
NUMA_CPU_BIND=${NUMA_CPU_BIND:-0-127}
NUMA_MEM_INTERLEAVE=${NUMA_MEM_INTERLEAVE:-0,1}
USE_NUMACTL_BINDING=${USE_NUMACTL_BINDING:-1}
export CALICO_DISABLE_PERF=${CALICO_DISABLE_PERF:-1}
unset CALICO_ENABLE_SEARCH_STATS

has_passwordless_sudo() {
    command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1
}

memory_bandwidth_monitor_available() {
    command -v perf >/dev/null 2>&1 && perf list 2>/dev/null | grep -q -F "$MEMORY_BW_METRIC"
}

probe_memory_bandwidth_metric() {
    local probe_output

    if ! has_passwordless_sudo; then
        printf '%s\n' "Passwordless sudo is unavailable, so perf cannot be started non-interactively."
        return 1
    fi

    probe_output=$(sudo -n perf stat -a --metric-no-group -x, -M "$MEMORY_BW_METRIC" -- sleep 0.1 2>&1 >/dev/null || true)
    if [ -n "$probe_output" ]; then
        printf '%s\n' "$probe_output"
        return 1
    fi

    return 0
}

start_memory_bandwidth_monitor() {
    local output_file=$1
    local probe_output

    if ! memory_bandwidth_monitor_available; then
        cat > "$output_file" <<EOF
# Memory bandwidth monitoring unavailable.
# Expected perf metric: ${MEMORY_BW_METRIC}
# Ensure perf is installed and the metric is exposed by this kernel/PMU.
EOF
        echo "Memory bandwidth metric ${MEMORY_BW_METRIC} unavailable; wrote diagnostic to: $output_file" >&2
        return 1
    fi

    if ! probe_output=$(probe_memory_bandwidth_metric); then
        cat > "$output_file" <<EOF
# Memory bandwidth monitoring unavailable.
# Expected perf metric: ${MEMORY_BW_METRIC}
# perf rejected the metric on this machine.
${probe_output}
EOF
        echo "Memory bandwidth metric ${MEMORY_BW_METRIC} is not runnable on this host; wrote diagnostic to: $output_file" >&2
        return 1
    fi

    echo "Starting memory bandwidth monitoring to: $output_file" >&2
    (
        echo "# system-wide DRAM bandwidth sampled once per second"
        echo "# perf metric: ${MEMORY_BW_METRIC}"
        echo "# output format: perf stat CSV (-x,)"
        echo "# started_at: $(date --iso-8601=seconds)"
        sudo -n perf stat -a --metric-no-group -I 1000 -x, -M "$MEMORY_BW_METRIC" -- sleep 1000000 2>&1
    ) > "$output_file" &
    printf '%s\n' "$!"
    return 0
}

stop_background_monitor() {
    local pid=$1
    local monitor_name=$2
    local output_file=$3

    if [ -n "$pid" ]; then
        echo "Stopping ${monitor_name} monitoring (PID: $pid)"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    if [ -n "$output_file" ]; then
        echo "${monitor_name} data saved to: $output_file"
    fi
}

python_has_module() {
    local module_name="$1"
    "$PYTHON_EXECUTABLE" -c "import ${module_name}" >/dev/null 2>&1
}

check_python_prereqs() {
    local missing_modules=()
    for module_name in setuptools pybind11 numpy; do
        if ! python_has_module "$module_name"; then
            missing_modules+=("$module_name")
        fi
    done
    if [ "$DATASET" = "sift10M" ] && ! python_has_module "h5py"; then
        missing_modules+=("h5py")
    fi
    if [ "${#missing_modules[@]}" -gt 0 ]; then
        echo "Error: Missing Python modules for $PYTHON_EXECUTABLE: ${missing_modules[*]}"
        echo "Install them with:"
        echo "  $PYTHON_EXECUTABLE -m pip install --user ${missing_modules[*]}"
        exit 1
    fi
}

check_python_prereqs

numa_command_prefix() {
    if [ "$USE_NUMACTL_BINDING" != "1" ]; then
        return 0
    fi

    if ! command -v numactl >/dev/null 2>&1; then
        echo "Warning: numactl not found; running without NUMA binding." >&2
        return 0
    fi

    printf '%s\n' "numactl" "--physcpubind=${NUMA_CPU_BIND}" "--interleave=${NUMA_MEM_INTERLEAVE}"
}

# Build predicache python library
echo "Building predicache python library..."
pushd "${SCRIPT_DIR}/predictivetranslation-sigmod2026" > /dev/null
"$PYTHON_EXECUTABLE" setup.py build_ext --inplace
popd > /dev/null
if ! BLOCK="$BLOCK" PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
    "$PYTHON_EXECUTABLE" -c "import predicache_py; print(predicache_py.__name__)" >/dev/null 2>&1; then
    echo "Error: Failed to import predicache_py after build."
    echo "Check build logs in ${SCRIPT_DIR}/predictivetranslation-sigmod2026, PYTHONPATH, and BLOCK=${BLOCK}."
    exit 1
fi

echo "Building calico python library..."
pushd "${SCRIPT_DIR}" > /dev/null
"$PYTHON_EXECUTABLE" setup.py build_ext --inplace
popd > /dev/null
if ! PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
    "$PYTHON_EXECUTABLE" -c "import calico_py; print(calico_py.__name__)" >/dev/null 2>&1; then
    echo "Error: Failed to import calico_py after build."
    exit 1
fi

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
    "robinhood_hash:1:7:0:1"
    # "lockfree_hash:1:4:0:1"
    # "predicache_inmem:0:0:0:1"
    # "vmcache:0:0:1:1"
    #"calico:1:0:0:1"

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
    export NOHUGEPAGE_TRANSLATION_ARRAY=0
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Output file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file_suffix="${config_name}_threads${threads}_${timestamp}"
    local output_file="${output_dir}/${output_file_suffix}.log"
    local membw_file="${output_dir}/${output_file_suffix}.membw"
    local -a CMD_PREFIX=()
    local run_block="$DEFAULT_BLOCK_PATH"

    if [[ "$config_name" == predicache* ]]; then
        run_block="$(predicache_block_path "$config_name")"
    fi
    ensure_block_file "$run_block" "20G"

    local membw_pid=""
    membw_pid=$(start_memory_bandwidth_monitor "$membw_file" || true)

    mapfile -t CMD_PREFIX < <(numa_command_prefix)
    
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
    # For predicache configs, use predicache library
    elif [[ "$config_name" == predicache* ]]; then
        PY_ARGS+=(--library predicache)
    fi

    # Run benchmark
    set +e
    set -o pipefail
    BLOCK="$run_block" "${CMD_PREFIX[@]}" "$PYTHON_EXECUTABLE" "${PY_ARGS[@]}" 2>&1 | tee "$output_file"
    local run_status=${PIPESTATUS[0]}
    set +o pipefail
    set -e

    stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"

    if [ "$run_status" -ne 0 ]; then
        echo "Error: benchmark failed for $config_name with $threads threads (exit=$run_status)."
        return "$run_status"
    fi
    
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
    if [ "$USE_NUMACTL_BINDING" = "1" ]; then
        echo "NUMA Policy: physcpubind=${NUMA_CPU_BIND}, interleave=${NUMA_MEM_INTERLEAVE}"
    else
        echo "NUMA Policy: disabled"
    fi
    echo "========================================"
    echo ""
    
    # Build the project first
    echo "Building project..."
    make clean && make -j$(nproc)
    echo "Build complete"
    echo ""

    if ! PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
        "$PYTHON_EXECUTABLE" -c "import calico_py; print(calico_py.__name__)" >/dev/null 2>&1; then
        echo "Error: calico_py is unavailable after the native project build."
        echo "The native Makefile clean target must not remove Python extension modules."
        exit 1
    fi

    echo "Cleaning reusable index artifacts before the first run..."
    cleanup_reusable_index_files
    echo "Cleanup complete"
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
    ./extract_results_hnsw.sh "$DATASET"
}

# Run main function
main "$@"
