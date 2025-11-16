#!/bin/bash

# HNSW Out-of-Memory (OOM) Benchmark Experiment Script
# Tests Calico configurations and hnswlib with constrained physical memory

set -e  # Exit on error
set -o pipefail  # Propagate failures through pipelines

# Configuration
#DATASET="sift1m"
DATASET="deep10M"
#DATASET="sift10M"
#DATASET="sift1m"
M=16
EF_CONSTRUCTION=100
K=10
EF_SEARCH=50

# Fixed thread count for all experiments
THREADS=64

# Virtual memory size (generous for swapping/mmap)
export VIRTGB=16G

# Physical memory sizes to test (in GB)
#PHYSGB_VALUES=(0.3 0.5 1 2)
#PHYSGB_VALUES=(0.5 1 2)
# PHYSGB_VALUES=(0.3)
# Base directory for results
#PHYSGB_VALUES=(10 5 2.5)
PHYSGB_VALUES=(10 1.5)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/larger-than-memory/hnsw-${DATASET}"

# Calico configurations to test
# Format: config_name:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH:LIBRARY
CALICO_CONFIGS=(
    "calico_array:1:0:0:1:calico"
    "calico_openaddr:1:2:0:1:calico"
    "vmcache:0:0:1:1:calico"
)
#"calico_lockfree:1:4:0:1:calico"
# Calico mmap + OS page cache configuration (needs cgroup)
# CALICO_MMAP_OSCACHE_CONFIG="calico_mmap_oscache:2:0:1:0:calico"

# hnswlib with mmap configuration
# HNSWLIB_MMAP_CONFIG="hnswlib_mmap:0:0:0:0:hnswlib"

# usearch with mmap configuration
USEARCH_MMAP_CONFIG="usearch_mmap:0:0:0:0:usearch"

# Global cgroup name (v1 memory controller)
CGROUP_NAME="hnsw_benchmark"

# Detect cgroup v2 (unified) availability
is_cgroup_v2() {
    [ -f "/sys/fs/cgroup/cgroup.controllers" ]
}

# Swap-related knobs (can be overridden via env)
# Enable swap usage within the cgroup (v1) by setting memory.swappiness>0
# Also used to set MemorySwapMax in systemd-run (v2)
SWAP_ENABLE=${SWAP_ENABLE:-1}
CGROUP_SWAPPINESS=${CGROUP_SWAPPINESS:-60}   # only used when SWAP_ENABLE=1
CGROUP_MEMSW_FACTOR=${CGROUP_MEMSW_FACTOR:-4} # mem+swap limit factor vs memory.limit

# Helper function to create and configure cgroup for memory limit (cgroup v1)
setup_cgroup() {
    local memory_limit_mb=$1
    local cgroup_name="$CGROUP_NAME"
    # For cgroup v1, we create a directory under the memory controller
    local cgroup_path="/sys/fs/cgroup/memory/${cgroup_name}"
    
    echo "Setting up cgroup v1 at ${cgroup_path} with ${memory_limit_mb}MB memory limit..." >&2
    
    # Create cgroup directory
    sudo mkdir -p "$cgroup_path"
    
    # Compute limits in bytes
    local memory_limit_bytes=$((memory_limit_mb * 1024 * 1024))
    local memsw_limit_bytes=$((memory_limit_bytes * CGROUP_MEMSW_FACTOR))

    # For v1, set mem+swap limit before memory.limit when enabling swap
    if [ -f "${cgroup_path}/memory.memsw.limit_in_bytes" ]; then
        echo "$memsw_limit_bytes" | sudo tee "${cgroup_path}/memory.memsw.limit_in_bytes" >/dev/null 2>&1 || true
    fi

    # Set memory limit (in bytes)
    echo "$memory_limit_bytes" | sudo tee "${cgroup_path}/memory.limit_in_bytes" >/dev/null

    # Reduce swapping to avoid hiding page cache growth
    if [ -f "${cgroup_path}/memory.swappiness" ]; then
        if [ "$SWAP_ENABLE" = "1" ]; then
            echo "$CGROUP_SWAPPINESS" | sudo tee "${cgroup_path}/memory.swappiness" >/dev/null 2>&1 || true
        else
            echo 0 | sudo tee "${cgroup_path}/memory.swappiness" >/dev/null 2>&1 || true
        fi
    fi

    # Enable hierarchical accounting if supported
    if [ -f "${cgroup_path}/memory.use_hierarchy" ]; then
        echo 1 | sudo tee "${cgroup_path}/memory.use_hierarchy" >/dev/null 2>&1 || true
    fi

    # Reset max usage counter for clean measurement
    if [ -f "${cgroup_path}/memory.max_usage_in_bytes" ]; then
        echo 0 | sudo tee "${cgroup_path}/memory.max_usage_in_bytes" >/dev/null 2>&1 || true
    fi

    # Diagnostics: show configured limits and swappiness (to stderr)
    {
        echo "Configured limits for ${cgroup_path}:"
        sudo bash -c "
            echo -n ' memory.limit_in_bytes='; cat '${cgroup_path}/memory.limit_in_bytes' 2>/dev/null || true
            if [ -f '${cgroup_path}/memory.memsw.limit_in_bytes' ]; then
                echo -n ' memory.memsw.limit_in_bytes='; cat '${cgroup_path}/memory.memsw.limit_in_bytes' 2>/dev/null || true
            fi
            if [ -f '${cgroup_path}/memory.swappiness' ]; then
                echo -n ' memory.swappiness='; cat '${cgroup_path}/memory.swappiness' 2>/dev/null || true
            fi
        "
    } 1>&2
    
    echo "$cgroup_path"
}

# Helper function to run process in cgroup
run_in_cgroup() {
    local cgroup_path=$1
    shift
    local cmd="$@"
    
    # Move current shell to cgroup, run command, then cleanup
    # For cgroup v1, the file is 'tasks'
    sudo sh -c "echo $$ > ${cgroup_path}/tasks && $cmd"
}

# Helper function to cleanup cgroup
cleanup_cgroup() {
    local cgroup_path=$1
    
    if [ -d "$cgroup_path" ]; then
        echo "Cleaning up cgroup: $cgroup_path"
        # On cgroup v1, we may need to kill processes before removing the directory
        # but systemd should handle this. If not, 'sudo killall -s 9 -g $(cat ${cgroup_path}/tasks)'
        sudo rmdir "$cgroup_path" 2>/dev/null || true
    fi
}

# Function to run a benchmark
run_benchmark() {
    local config_name=$1
    local traditional=$2
    local tradhash=$3
    local nohugepage=$4
    local prefetch=$5
    local library=$6
    local physgb=$7
    local output_dir=$8
    local mode=${9:-search}      # Default to 'search'
    local index_file=${10:-./hnsw_index.bin} # Default index file path
    local python_exec=${11} # Get the python executable path
    local python_path=${12} # Get the python path

    sudo vmtouch -e heapfile
    echo "=========================================="
    echo "Running: $config_name with PHYSGB=${physgb}GB, Mode: $mode"
    echo "Library: $library, Threads: $THREADS"
    echo "TRADITIONAL=$traditional, TRADHASH=$tradhash"
    echo "NOHUGEPAGE_FRAME_MEM=$nohugepage, PREFETCH=$prefetch"
    echo "python_exec=$python_exec"
    echo "python_path=$python_path"
    echo "=========================================="
    
    # Set environment variables
    export TRADITIONAL=$traditional
    export TRADHASH=$tradhash
    export NOHUGEPAGE_FRAME_MEM=$nohugepage
    export PREFETCH=$prefetch
    export PHYSGB="${physgb}"
    
    # For hnswlib, enable mmap mode
    if [ "$library" = "hnswlib" ]; then
        export MMAP_FOR_HSNWLIB=1
        echo "  MMAP_FOR_HSNWLIB=1"
    else
        unset MMAP_FOR_HSNWLIB
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Output file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file_suffix="${config_name}_physgb${physgb}_threads${THREADS}_${timestamp}"
    if [ "$mode" = "index" ]; then
        output_file_suffix="${config_name}_build_index_${timestamp}"
    fi
    local output_file="${output_dir}/${output_file_suffix}.log"
    local iostat_file="${output_dir}/${output_file_suffix}.iostat"
    local run_label="${output_file_suffix}"
    local run_output_base="${output_dir}/${output_file_suffix}"
    
    # Start iostat monitoring in background
    echo "Starting iostat monitoring to: $iostat_file"
    iostat -x 1 nvme0n1 > "$iostat_file" 2>&1 &
    local iostat_pid=$!

    # Device to control readahead for (change if necessary)
    local READAHEAD_DEV="nvme1n1"
    local readahead_path="/sys/block/${READAHEAD_DEV}/queue/read_ahead_kb"
    local orig_readahead=""
    local orig_blockdev_readahead=""
    local USE_BLOCKDEV=0

    # Prefer blockdev if available and device node exists (uses 512-byte sectors units)
    if command -v blockdev >/dev/null 2>&1 && [ -b "/dev/${READAHEAD_DEV}" ]; then
        orig_blockdev_readahead=$(sudo blockdev --getra /dev/${READAHEAD_DEV} 2>/dev/null || true)
        echo "Original blockdev readahead for /dev/${READAHEAD_DEV}: ${orig_blockdev_readahead}"
        sudo blockdev --setra 0 /dev/${READAHEAD_DEV} >/dev/null 2>&1 || true
        echo "Disabled blockdev readahead for /dev/${READAHEAD_DEV}"
        USE_BLOCKDEV=1
    elif [ -f "$readahead_path" ]; then
        orig_readahead=$(cat "$readahead_path")
        echo "Original readahead for $READAHEAD_DEV: $orig_readahead KB"
        echo 0 | sudo tee "$readahead_path" >/dev/null || true
        echo "Disabled readahead for $READAHEAD_DEV"
        USE_BLOCKDEV=0
    else
        echo "Readahead path $readahead_path not found and /dev/${READAHEAD_DEV} block device absent; skipping readahead adjustment"
    fi
    
    # Check if we need to run in cgroup
    local needs_cgroup=0
    local memory_limit_mb=0
    
    if [ "$library" == "hnswlib" ]; then
        # Memory limit = 62% of PHYSGB (to match the index size ratio between hnswlib and Calico)
        needs_cgroup=1
        memory_limit_mb=$(printf "%.0f" $(echo "$physgb * 1024 * 0.62 + 50" | bc))
    elif { [ "$config_name" = "calico_mmap_oscache" ] || [ "$library" = "usearch" ]; } && [ "$mode" = "search" ]; then
        # calico_mmap_oscache uses TRADITIONAL=2, needs cgroup with full PHYSGB limit
        needs_cgroup=1
        memory_limit_mb=$(printf "%.0f" $(echo "$physgb * 1024 + 50" | bc))
    fi
    
    # Construct the python command arguments
    local python_args=(
        -u benchmark_calico_hnsw.py
        --config "$config_name"
        --threads "$THREADS"
        --dataset "$DATASET"
        --M "$M"
        --ef_construction "$EF_CONSTRUCTION"
        --k "$K"
        --ef_search "$EF_SEARCH"
        --library "$library"
    )
    if [ "$library" = "usearch" ]; then
        python_args+=(--mode "$mode" --index_file "$index_file")
    fi
    local python_cmd
    python_cmd=$(printf " %q" "${python_args[@]}")

    # For hnswlib, usearch (search), or calico_mmap_oscache, run inside a cgroup
    if [ "$needs_cgroup" -eq 1 ]; then
        
        echo "Running $config_name in cgroup with ${memory_limit_mb}MB memory limit..."

        local cgroup_path=""

        # Optional: drop global caches if requested (strong measure; requires root)
        if [ "${DROP_CACHES:-0}" = "1" ]; then
            echo "DROP_CACHES=1: syncing and dropping global caches (echo 3 > /proc/sys/vm/drop_caches)"
            sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
        fi

        # If running usearch search, evict the index file from OS page cache so
        # subsequent page faults are charged to this cgroup.
        if [ "$library" = "usearch" ] && [ "$mode" = "search" ] && [ -n "$index_file" ] && [ -f "$index_file" ]; then
            echo "Evicting index file from page cache: $index_file"
            sudo vmtouch -e "$index_file" || true
        fi

        # Choose method based on cgroup version and available tools
        if is_cgroup_v2 && command -v systemd-run >/dev/null 2>&1; then
            echo "Detected cgroup v2. Using systemd-run transient scope with MemoryMax=${memory_limit_mb}M"
            # Compute MemorySwapMax in MB (if SWAP_ENABLE=1); else set to 0 to effectively disable swap for the unit
            local memsw_limit_mb=$(printf "%.0f" $(echo "${memory_limit_mb} * ${CGROUP_MEMSW_FACTOR}" | bc))
            local swap_max_opt="-p MemorySwapMax=0"
            if [ "$SWAP_ENABLE" = "1" ]; then
                swap_max_opt="-p MemorySwapMax=${memsw_limit_mb}M"
            fi
            {
                sudo -E systemd-run --scope -p MemoryAccounting=yes -p MemoryMax=${memory_limit_mb}M ${swap_max_opt} -p TasksMax=infinity \
                    bash -c "
                        export CALICO_RUN_OUTPUT_BASE=\"${run_output_base}\";
                        export CALICO_RUN_LABEL=\"${run_label}\";
                        export PYTHONPATH=${python_path}:\$PYTHONPATH;
                        echo 'Swap devices:'; cat /proc/swaps || true;
                        echo 'Self cgroups before exec (v2):';
                        cat /proc/\$\$/cgroup || true;
                        exec \"$python_exec\"${python_cmd};
                    "
            } | tee "$output_file"
        elif command -v cgexec >/dev/null 2>&1; then
            echo "Using cgexec to run in v1 cgroup: ${CGROUP_NAME}"
            cgroup_path=$(setup_cgroup "$memory_limit_mb")
            if [ ! -d "$cgroup_path" ]; then
                echo "ERROR: Failed to create cgroup at $cgroup_path"
                kill $iostat_pid 2>/dev/null || true
                return 1
            fi
            {
                sudo -E cgexec -g "memory:${CGROUP_NAME}" \
                    bash -c "
                        export CALICO_RUN_OUTPUT_BASE=\"${run_output_base}\";
                        export CALICO_RUN_LABEL=\"${run_label}\";
                        export PYTHONPATH=${python_path}:\$PYTHONPATH;
                        echo 'Swap devices:'; cat /proc/swaps || true;
                        echo 'Self cgroups before exec (v1 cgexec):';
                        cat /proc/\$\$/cgroup || true;
                        exec \"$python_exec\"${python_cmd};
                    "
            } | tee "$output_file"
            echo "Final cgroup memory stats:"
            sudo bash -c "
                echo -n 'memory.usage_in_bytes: '; cat '${cgroup_path}/memory.usage_in_bytes' 2>/dev/null || true
                echo -n 'memory.max_usage_in_bytes: '; cat '${cgroup_path}/memory.max_usage_in_bytes' 2>/dev/null || true
                if [ -f '${cgroup_path}/memory.memsw.usage_in_bytes' ]; then
                    echo -n 'memory.memsw.usage_in_bytes: '; cat '${cgroup_path}/memory.memsw.usage_in_bytes' 2>/dev/null || true
                fi
                echo 'memory.stat:'; cat '${cgroup_path}/memory.stat' 2>/dev/null | sed -n '1,20p' || true
            "
            cleanup_cgroup "$cgroup_path"
        else
            cgroup_path=$(setup_cgroup "$memory_limit_mb")
            if [ ! -d "$cgroup_path" ]; then
                echo "ERROR: Failed to create cgroup at $cgroup_path"
                kill $iostat_pid 2>/dev/null || true
                return 1
            fi
            echo "Using manual cgroup v1 attach (no cgexec)."
            echo "Manual attach path: $cgroup_path"

            local combined_python_path="${python_path}:${PYTHONPATH}"
            {
                sudo CALICO_RUN_OUTPUT_BASE="$run_output_base" CALICO_RUN_LABEL="$run_label" PYTHONPATH="$combined_python_path" bash -c '
                    set -e
                    cgroup_path="$1"
                    shift
                    export CALICO_RUN_OUTPUT_BASE
                    export CALICO_RUN_LABEL
                    echo $$ > "${cgroup_path}/tasks"
                    echo "Swap devices:"; cat /proc/swaps || true
                    echo "Self cgroups before exec (manual v1 attach):"
                    cat /proc/$$/cgroup || true
                    exec "$@"
                ' manual-cgroup "$cgroup_path" "$python_exec" "${python_args[@]}"
            } | tee "$output_file"

            echo "Final cgroup memory stats:"
            sudo bash -c "
                echo -n 'memory.usage_in_bytes: '; cat '${cgroup_path}/memory.usage_in_bytes' 2>/dev/null || true
                echo -n 'memory.max_usage_in_bytes: '; cat '${cgroup_path}/memory.max_usage_in_bytes' 2>/dev/null || true
                if [ -f '${cgroup_path}/memory.memsw.usage_in_bytes' ]; then
                    echo -n 'memory.memsw.usage_in_bytes: '; cat '${cgroup_path}/memory.memsw.usage_in_bytes' 2>/dev/null || true
                fi
                echo 'memory.stat:'; cat '${cgroup_path}/memory.stat' 2>/dev/null | sed -n '1,20p' || true
            "

            cleanup_cgroup "$cgroup_path"
        fi
    else
        echo "$python_exec"
        # Regular run for Calico configurations or usearch index build
        CALICO_RUN_OUTPUT_BASE="$run_output_base" CALICO_RUN_LABEL="$run_label" \
            "$python_exec" "${python_args[@]}" 2>&1 | tee "$output_file"
    fi

    # Restore original readahead if it was changed
    if [ "$USE_BLOCKDEV" -eq 1 ]; then
        if [ -n "$orig_blockdev_readahead" ]; then
            sudo blockdev --setra "$orig_blockdev_readahead" /dev/${READAHEAD_DEV} >/dev/null 2>&1 || true
            echo "Restored blockdev readahead for /dev/${READAHEAD_DEV} to ${orig_blockdev_readahead}"
        fi
    else
        if [ -n "$orig_readahead" ] && [ -f "$readahead_path" ]; then
            echo "$orig_readahead" | sudo tee "$readahead_path" >/dev/null || true
            echo "Restored readahead for $READAHEAD_DEV to $orig_readahead KB"
        fi
    fi
    
    # Stop iostat monitoring
    echo "Stopping iostat monitoring (PID: $iostat_pid)"
    kill $iostat_pid 2>/dev/null || true
    wait $iostat_pid 2>/dev/null || true
    echo "iostat data saved to: $iostat_file"
    
    echo "Results saved to: $output_file"
    echo ""
}

# Main execution
main() {
    PYTHON_EXECUTABLE=$(which python3)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Error: python3 not found in PATH."
        exit 1
    fi
    echo "Using Python executable: $PYTHON_EXECUTABLE"
    
    PYTHON_PATH=$($PYTHON_EXECUTABLE -c "import sys; print(':'.join(sys.path))")
    echo "Using Python path: $PYTHON_PATH"
    echo ""

    echo "========================================"
    echo "HNSW OOM Benchmark Experiment"
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "HNSW Parameters: M=$M, ef_construction=$EF_CONSTRUCTION"
    echo "Search Parameters: k=$K, ef_search=$EF_SEARCH"
    echo "Thread Count: $THREADS (fixed)"
    echo "Physical Memory Sizes: ${PHYSGB_VALUES[*]} GB"
    echo "Virtual Memory: ${VIRTGB}"
    echo "========================================"
    echo ""
    
    # Check cgroup v2 delegated controllers
    if [ ! -w "/sys/fs/cgroup/cgroup.procs" ]; then
        echo "Warning: You may not have permission to create cgroups."
        echo "If hnswlib/usearch tests fail, you may need to enable cgroup delegation."
        echo ""
    fi
    
    # Build the project first
    echo "Building project..."
    make clean && make -j$(nproc)
    echo "Build complete"
    echo ""
    
    # Build hnswlib
    echo "Building hnswlib..."
    (cd hnswlib && "$PYTHON_EXECUTABLE" setup.py build_ext --inplace)
    echo "hnswlib build complete"
    echo ""

    # Step 1: Build the usearch index once
    echo "--- Building usearch index ---"
    local index_file="./exps/usearch_index.bin"
    local build_dir="${RESULTS_BASE_DIR}/physgb_build"
    mkdir -p "$build_dir"
    
    IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$USEARCH_MMAP_CONFIG"
    
    run_benchmark \
        "$config_name" \
        "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
        "N/A" \
        "$build_dir" \
        "index" \
        "$index_file" \
        "$PYTHON_EXECUTABLE" \
        "$PYTHON_PATH"

    echo "--- Usearch index build complete ---"
    sleep 5

    rm ./heapfile
    fallocate -l 20G ./heapfile

    # Run benchmarks for each physical memory size for other configs
    for physgb in "${PHYSGB_VALUES[@]}"; do
        echo "========================================"
        echo "Testing with PHYSGB=${physgb}GB"
        echo "========================================"
        
        # Create results directory for this memory size
        results_dir="${RESULTS_BASE_DIR}/physgb_${physgb}"
        mkdir -p "$results_dir"
        
        # Run usearch search phase
        echo "--- Testing usearch search with PHYSGB=${physgb}GB ---"
        IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$USEARCH_MMAP_CONFIG"
        run_benchmark \
            "$config_name" \
            "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
            "$physgb" \
            "$results_dir" \
            "search" \
            "$index_file" \
            "$PYTHON_EXECUTABLE" \
            "$PYTHON_PATH"
        sleep 2

        # # Test hnswlib with mmap
        # IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$HNSWLIB_MMAP_CONFIG"
        
        # run_benchmark \
        #     "$config_name" \
        #     "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
        #     "$physgb" \
        #     "$results_dir"
        
        # sleep 2
        
        # Test calico_mmap_oscache (TRADITIONAL=2, needs cgroup)
        # IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$CALICO_MMAP_OSCACHE_CONFIG"
        
        # run_benchmark \
        #     "$config_name" \
        #     "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
        #     "$physgb" \
        #     "$results_dir" \
        #     "search" \
        #     "" \
        #     "$PYTHON_EXECUTABLE" \
        #     "$PYTHON_PATH"
        
#        Test each Calico configuration
        for config_spec in "${CALICO_CONFIGS[@]}"; do
            IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$config_spec"
            
            run_benchmark \
                "$config_name" \
                "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
                "$physgb" \
                "$results_dir" \
                "search" \
                "" \
                "$PYTHON_EXECUTABLE" \
                "$PYTHON_PATH"

            sleep 2
        done

        # Delay between memory size tests
        sleep 5
        
        echo ""
    done
    
    echo "========================================"
    echo "All experiments completed!"
    echo "Results directory: $RESULTS_BASE_DIR"
    echo "========================================"
    echo ""
    
    # Extract and summarize results
    echo "Extracting results..."
    ./extract_results_hnsw_oom.sh
}

# Run main function
main "$@"
