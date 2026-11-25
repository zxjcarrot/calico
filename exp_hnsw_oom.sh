#!/bin/bash

# HNSW Out-of-Memory (OOM) Benchmark Experiment Script
# Tests Calico configurations and hnswlib with constrained physical memory

#set -e  # Exit on error
#set -o pipefail  # Propagate failures through pipelines

# Configuration
#DATASET="sift1m"
DATASET="${DATASET:-deep10M}"
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
PHYSGB_VALUES=(10 5 2.5 1.5)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
USER_BLOCK_OVERRIDE="${BLOCK:-}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${CALICO_RESULTS_DIR:-${SCRIPT_DIR}/exps}/larger-than-memory/hnsw-${DATASET}}"

# Calico configurations to test
# Format: config_name:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH:LIBRARY
# CALICO_CONFIGS=(
#     "calico_array:1:0:0:1:calico"
#     "vmcache:0:0:1:1:calico"
#     "predicache_oom:0:0:0:1:predicache"
#     "calico_openaddr:1:2:0:1:calico"
# )

CALICO_CONFIGS=(
    #"predicache_oom:0:0:0:1:predicache"
    #"calico_chained:1:1:0:1:calico"
    "calico_robinhood:1:7:0:1:calico"
)
#"calico_lockfree:1:4:0:1:calico"
# Calico mmap + OS page cache configuration (needs cgroup)
# CALICO_MMAP_OSCACHE_CONFIG="calico_mmap_oscache:2:0:1:0:calico"

# hnswlib with mmap configuration
# HNSWLIB_MMAP_CONFIG="hnswlib_mmap:0:0:0:0:hnswlib"

# usearch with mmap configuration
USEARCH_MMAP_CONFIG="usearch_mmap:0:0:0:0:usearch"
USEARCH_SUPPORTED=0

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
MEMORY_BW_METRIC=${MEMORY_BW_METRIC:-nps1_die_to_dram}
HAVE_PASSWORDLESS_SUDO=0
NUMA_CPU_BIND=${NUMA_CPU_BIND:-0-127}
NUMA_MEM_INTERLEAVE=${NUMA_MEM_INTERLEAVE:-0,1}
USE_NUMACTL_BINDING=${USE_NUMACTL_BINDING:-1}
export CALICO_DISABLE_PERF=${CALICO_DISABLE_PERF:-1}
unset CALICO_ENABLE_SEARCH_STATS

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

numa_command_string() {
    local -a cmd_prefix=()
    mapfile -t cmd_prefix < <(numa_command_prefix)
    if [ "${#cmd_prefix[@]}" -eq 0 ]; then
        return 0
    fi

    printf "%q " "${cmd_prefix[@]}"
}

check_passwordless_sudo() {
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        HAVE_PASSWORDLESS_SUDO=1
    else
        HAVE_PASSWORDLESS_SUDO=0
    fi
}

has_passwordless_sudo() {
    [ "$HAVE_PASSWORDLESS_SUDO" -eq 1 ]
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
    local python_exec="$1"
    local module_name="$2"
    "$python_exec" -c "import ${module_name}" >/dev/null 2>&1
}

python_env_root_for_exec() {
    local python_exec="$1"
    local resolved_exec
    resolved_exec=$(readlink -f "$python_exec" 2>/dev/null || printf '%s' "$python_exec")
    dirname "$(dirname "$resolved_exec")"
}

python_can_import_ctypes() {
    local python_exec="$1"
    local python_home="$2"

    if [ -n "$python_home" ]; then
        env -u PYTHONHOME PYTHONHOME="$python_home" "$python_exec" -c "import ctypes" >/dev/null 2>&1
    else
        env -u PYTHONHOME "$python_exec" -c "import ctypes" >/dev/null 2>&1
    fi
}

select_python_interpreter() {
    local -a candidates=()
    local seen=""
    local candidate=""

    if [ -n "${PYTHON_EXECUTABLE:-}" ]; then
        candidates+=("$PYTHON_EXECUTABLE")
    fi

    candidate=$(which python3 2>/dev/null || true)
    if [ -n "$candidate" ]; then
        candidates+=("$candidate")
    fi

    for candidate in /usr/bin/python3 /usr/local/bin/python3 "$HOME/.local/bin/python3"; do
        [ -x "$candidate" ] && candidates+=("$candidate")
    done

    if [ -f "$HOME/.conda/environments.txt" ]; then
        while IFS= read -r env_path; do
            [ -z "$env_path" ] && continue
            if [ -x "$env_path/bin/python" ]; then
                candidates+=("$env_path/bin/python")
            fi
            if [ -x "$env_path/bin/python3" ]; then
                candidates+=("$env_path/bin/python3")
            fi
        done < "$HOME/.conda/environments.txt"
    fi

    for candidate in "${candidates[@]}"; do
        [ -x "$candidate" ] || continue
        if [[ ":$seen:" == *":$candidate:"* ]]; then
            continue
        fi
        seen+="${candidate}:"

        if python_can_import_ctypes "$candidate" ""; then
            printf '%s\n' "$candidate"
            return 0
        fi

        local candidate_home
        candidate_home=$(python_env_root_for_exec "$candidate")
        if [ -n "$candidate_home" ] && [ -d "$candidate_home/lib" ] && python_can_import_ctypes "$candidate" "$candidate_home"; then
            export PYTHONHOME="$candidate_home"
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

check_python_prereqs() {
    local python_exec="$1"
    local missing_modules=()

    # `usearch` depends on Python's built-in `_ctypes` extension.
    # If `_ctypes` is missing from this interpreter build, importing usearch will fail.
    if ! "$python_exec" -c "import ctypes" >/dev/null 2>&1; then
        echo "Error: Python interpreter '$python_exec' cannot import ctypes (_ctypes missing)."
        echo "This interpreter cannot run usearch benchmarks."
        echo "Set PYTHON_EXECUTABLE to a working Python (example):"
        echo "  PYTHON_EXECUTABLE=/path/to/python3 bash exp_hnsw_oom_deep10M.sh"
        return 1
    fi

    for module_name in setuptools pybind11 numpy; do
        if ! python_has_module "$python_exec" "$module_name"; then
            missing_modules+=("$module_name")
        fi
    done

    if [ "$DATASET" = "sift10M" ] && ! python_has_module "$python_exec" "h5py"; then
        missing_modules+=("h5py")
    fi

    if [ "${#missing_modules[@]}" -gt 0 ]; then
        echo "Error: Missing Python modules for $python_exec: ${missing_modules[*]}"
        echo "Install them with:"
        echo "  $python_exec -m pip install --user ${missing_modules[*]}"
        return 1
    fi

    return 0
}

verify_predicache_import() {
    local python_exec="$1"
    if ! BLOCK="${BLOCK:-${DEFAULT_BLOCK_PATH:-${SCRIPT_DIR}/heapfile}}" \
        PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
        "$python_exec" -c "import predicache_py; print(predicache_py.__name__)" >/dev/null 2>&1; then
        echo "Error: Failed to import predicache_py after build."
        echo "Check build logs in ${SCRIPT_DIR}/predictivetranslation-sigmod2026, PYTHONPATH, and BLOCK=${BLOCK:-${DEFAULT_BLOCK_PATH:-${SCRIPT_DIR}/heapfile}}."
        return 1
    fi
    return 0
}

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
    if [ -n "$USER_BLOCK_OVERRIDE" ]; then
        printf '%s\n' "$USER_BLOCK_OVERRIDE"
        return 0
    fi

    printf '%s\n' "${SCRIPT_DIR}/heapfile.${config_name}.${DATASET}"
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
    local run_block="$DEFAULT_BLOCK_PATH"

    if [[ "$config_name" == predicache* ]] || [ "$library" = "predicache" ]; then
        run_block="$(predicache_block_path)"
    fi
    ensure_block_file "$run_block" "20G"
    local BLOCK="$run_block"

    if has_passwordless_sudo; then
        sudo -n vmtouch -e heapfile >/dev/null 2>&1 || true
    else
        echo "Skipping heapfile page-cache eviction because passwordless sudo is unavailable"
    fi
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
    export NOHUGEPAGE_TRANSLATION_ARRAY=0
    export PREFETCH=$prefetch
    export PHYSGB="${physgb}"
    export THREADS
    
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
    local output_file_suffix="${config_name}_physgb${physgb}_threads${THREADS}"
    if [ "$mode" = "index" ]; then
        output_file_suffix="${config_name}_build_index"
    fi
    local output_file="${output_dir}/${output_file_suffix}.log"
    local iostat_file="${output_dir}/${output_file_suffix}.iostat"
    local membw_file="${output_dir}/${output_file_suffix}.membw"
    local run_label="${output_file_suffix}"
    local run_output_base="${output_dir}/${output_file_suffix}"
    local run_status=0
    
    # Start iostat monitoring in background
    echo "Starting iostat monitoring to: $iostat_file"
    iostat -x 1 nvme0n1 > "$iostat_file" 2>&1 &
    local iostat_pid=$!
    local membw_pid=""
    membw_pid=$(start_memory_bandwidth_monitor "$membw_file" || true)

    # Device to control readahead for (change if necessary)
    local READAHEAD_DEV="nvme1n1"
    local readahead_path="/sys/block/${READAHEAD_DEV}/queue/read_ahead_kb"
    local orig_readahead=""
    local orig_blockdev_readahead=""
    local USE_BLOCKDEV=0

    # Prefer blockdev if available and device node exists (uses 512-byte sectors units)
    if command -v blockdev >/dev/null 2>&1 && [ -b "/dev/${READAHEAD_DEV}" ]; then
        if has_passwordless_sudo; then
            orig_blockdev_readahead=$(sudo -n blockdev --getra /dev/${READAHEAD_DEV} 2>/dev/null || true)
        fi
        echo "Original blockdev readahead for /dev/${READAHEAD_DEV}: ${orig_blockdev_readahead}"
        if has_passwordless_sudo; then
            sudo -n blockdev --setra 0 /dev/${READAHEAD_DEV} >/dev/null 2>&1 || true
            echo "Disabled blockdev readahead for /dev/${READAHEAD_DEV}"
            USE_BLOCKDEV=1
        else
            echo "Skipping blockdev readahead adjustment because passwordless sudo is unavailable"
        fi
    elif [ -f "$readahead_path" ]; then
        orig_readahead=$(cat "$readahead_path")
        echo "Original readahead for $READAHEAD_DEV: $orig_readahead KB"
        if has_passwordless_sudo; then
            echo 0 | sudo -n tee "$readahead_path" >/dev/null || true
            echo "Disabled readahead for $READAHEAD_DEV"
            USE_BLOCKDEV=0
        else
            echo "Skipping readahead adjustment because passwordless sudo is unavailable"
        fi
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
    if [ "$library" != "usearch" ] && [ "$physgb" = "10" ]; then
        # Build once at the largest pool size, then reuse/recover the index
        # for smaller PHYSGB points.
        python_args+=(--skip_recovery)
    fi
    if [ "$library" = "usearch" ]; then
        python_args+=(--mode "$mode" --index_file "$index_file")
    fi
    local python_cmd
    python_cmd=$(printf " %q" "${python_args[@]}")
    local -a NUMA_PREFIX=()
    local NUMA_PREFIX_STR=""
    mapfile -t NUMA_PREFIX < <(numa_command_prefix)
    NUMA_PREFIX_STR=$(numa_command_string)

    # For hnswlib, usearch (search), or calico_mmap_oscache, run inside a cgroup
    if [ "$needs_cgroup" -eq 1 ]; then
        if ! has_passwordless_sudo; then
            echo "Skipping $config_name because cgroup runs require passwordless sudo in unattended mode."
            stop_background_monitor "$iostat_pid" "iostat" "$iostat_file"
            stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"
            return 0
        fi
        
        echo "Running $config_name in cgroup with ${memory_limit_mb}MB memory limit..."

        local cgroup_path=""

        # Optional: drop global caches if requested (strong measure; requires root)
        if [ "${DROP_CACHES:-0}" = "1" ]; then
            echo "DROP_CACHES=1: syncing and dropping global caches (echo 3 > /proc/sys/vm/drop_caches)"
            sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
        fi

        # If running usearch search, evict the index file from OS page cache so
        # subsequent page faults are charged to this cgroup.
        if [ "$library" = "usearch" ] && [ "$mode" = "search" ] && [ -n "$index_file" ] && [ -f "$index_file" ]; then
            echo "Evicting index file from page cache: $index_file"
            sudo -n vmtouch -e "$index_file" || true
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
            set -o pipefail
            {
                sudo -n -E systemd-run --scope -p MemoryAccounting=yes -p MemoryMax=${memory_limit_mb}M ${swap_max_opt} -p TasksMax=infinity \
                    bash -c "
                        cd \"${SCRIPT_DIR}\";
                        export CALICO_RUN_OUTPUT_BASE=\"${run_output_base}\";
                        export CALICO_RUN_LABEL=\"${run_label}\";
                        export BLOCK=\"${BLOCK}\";
                        export PYTHONHOME=\"${PYTHONHOME}\";
                        export PYTHONPATH=${python_path}:\$PYTHONPATH;
                        echo 'Swap devices:'; cat /proc/swaps || true;
                        echo 'Self cgroups before exec (v2):';
                        cat /proc/\$\$/cgroup || true;
                        exec ${NUMA_PREFIX_STR}"$python_exec"${python_cmd};
                    "
            } | tee "$output_file"
            run_status=${PIPESTATUS[0]}
            set +o pipefail
        elif command -v cgexec >/dev/null 2>&1; then
            echo "Using cgexec to run in v1 cgroup: ${CGROUP_NAME}"
            cgroup_path=$(setup_cgroup "$memory_limit_mb")
            if [ ! -d "$cgroup_path" ]; then
                echo "ERROR: Failed to create cgroup at $cgroup_path"
                stop_background_monitor "$iostat_pid" "iostat" "$iostat_file"
                stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"
                return 1
            fi
            set -o pipefail
            {
                sudo -n -E cgexec -g "memory:${CGROUP_NAME}" \
                    bash -c "
                        cd \"${SCRIPT_DIR}\";
                        export CALICO_RUN_OUTPUT_BASE=\"${run_output_base}\";
                        export CALICO_RUN_LABEL=\"${run_label}\";
                        export BLOCK=\"${BLOCK}\";
                        export PYTHONHOME=\"${PYTHONHOME}\";
                        export PYTHONPATH=${python_path}:\$PYTHONPATH;
                        echo 'Swap devices:'; cat /proc/swaps || true;
                        echo 'Self cgroups before exec (v1 cgexec):';
                        cat /proc/\$\$/cgroup || true;
                        exec ${NUMA_PREFIX_STR}"$python_exec"${python_cmd};
                    "
            } | tee "$output_file"
            run_status=${PIPESTATUS[0]}
            set +o pipefail
            echo "Final cgroup memory stats:"
            sudo -n bash -c "
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
                stop_background_monitor "$iostat_pid" "iostat" "$iostat_file"
                stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"
                return 1
            fi
            echo "Using manual cgroup v1 attach (no cgexec)."
            echo "Manual attach path: $cgroup_path"

            local combined_python_path="${python_path}:${PYTHONPATH}"
            set -o pipefail
            {
                sudo -n CALICO_RUN_OUTPUT_BASE="$run_output_base" CALICO_RUN_LABEL="$run_label" BLOCK="$BLOCK" PYTHONHOME="$PYTHONHOME" PYTHONPATH="$combined_python_path" bash -c '
                    set -e
                    cgroup_path="$1"
                    shift
                    cd "'$SCRIPT_DIR'"
                    export CALICO_RUN_OUTPUT_BASE
                    export CALICO_RUN_LABEL
                    export BLOCK
                    export PYTHONHOME
                    export PYTHONPATH
                    echo $$ > "${cgroup_path}/tasks"
                    echo "Swap devices:"; cat /proc/swaps || true
                    echo "Self cgroups before exec (manual v1 attach):"
                    cat /proc/$$/cgroup || true
                    exec "$@"
                ' manual-cgroup "$cgroup_path" "${NUMA_PREFIX[@]}" "$python_exec" "${python_args[@]}"
            } | tee "$output_file"
            run_status=${PIPESTATUS[0]}
            set +o pipefail

            echo "Final cgroup memory stats:"
            sudo -n bash -c "
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
        set -o pipefail
        CALICO_RUN_OUTPUT_BASE="$run_output_base" CALICO_RUN_LABEL="$run_label" \
            "${NUMA_PREFIX[@]}" "$python_exec" "${python_args[@]}" 2>&1 | tee "$output_file"
        run_status=${PIPESTATUS[0]}
        set +o pipefail
    fi

    # Restore original readahead if it was changed
    if [ "$USE_BLOCKDEV" -eq 1 ]; then
        if [ -n "$orig_blockdev_readahead" ]; then
            sudo -n blockdev --setra "$orig_blockdev_readahead" /dev/${READAHEAD_DEV} >/dev/null 2>&1 || true
            echo "Restored blockdev readahead for /dev/${READAHEAD_DEV} to ${orig_blockdev_readahead}"
        fi
    else
        if [ -n "$orig_readahead" ] && [ -f "$readahead_path" ]; then
            echo "$orig_readahead" | sudo -n tee "$readahead_path" >/dev/null || true
            echo "Restored readahead for $READAHEAD_DEV to $orig_readahead KB"
        fi
    fi
    
    # Stop background monitors
    stop_background_monitor "$iostat_pid" "iostat" "$iostat_file"
    stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"

    if [ "$run_status" -ne 0 ]; then
        echo "ERROR: Benchmark command failed for $config_name (mode=$mode, PHYSGB=$physgb) with exit code $run_status"
        return "$run_status"
    fi
    
    echo "Results saved to: $output_file"
    echo ""
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    DEFAULT_BLOCK_PATH="$(default_block_path)"
    export BLOCK="$DEFAULT_BLOCK_PATH"
    ensure_block_file "$BLOCK" "20G"
    export PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}"
    check_passwordless_sudo

    PYTHON_EXECUTABLE=$(select_python_interpreter)
    if [ -z "$PYTHON_EXECUTABLE" ]; then
        echo "Error: Could not find a Python interpreter that can import ctypes."
        echo "Tried PYTHON_EXECUTABLE, python3 on PATH, standard system locations, and conda environments listed in ~/.conda/environments.txt."
        echo "Set PYTHON_EXECUTABLE to a working interpreter, and if needed also set PYTHONHOME."
        exit 1
    fi
    echo "Using Python executable: $PYTHON_EXECUTABLE"
    if [ -n "${PYTHONHOME:-}" ]; then
        echo "Using PYTHONHOME: ${PYTHONHOME}"
    fi

    if python_has_module "$PYTHON_EXECUTABLE" "usearch"; then
        USEARCH_SUPPORTED=1
        echo "Usearch support detected for: $PYTHON_EXECUTABLE"
    else
        USEARCH_SUPPORTED=0
        echo "Warning: usearch is unavailable for $PYTHON_EXECUTABLE; usearch OOM benchmarks will be skipped."
    fi
    
    PYTHON_PATH=$($PYTHON_EXECUTABLE -c "import sys; print(':'.join(sys.path))")
    echo "Using Python path: $PYTHON_PATH"
    echo ""

    if ! check_python_prereqs "$PYTHON_EXECUTABLE"; then
        exit 1
    fi

    echo "========================================"
    echo "HNSW OOM Benchmark Experiment"
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "HNSW Parameters: M=$M, ef_construction=$EF_CONSTRUCTION"
    echo "Search Parameters: k=$K, ef_search=$EF_SEARCH"
    echo "Thread Count: $THREADS (fixed)"
    echo "Physical Memory Sizes: ${PHYSGB_VALUES[*]} GB"
    echo "Virtual Memory: ${VIRTGB}"
    if [ "$USE_NUMACTL_BINDING" = "1" ]; then
        echo "NUMA Policy: physcpubind=${NUMA_CPU_BIND}, interleave=${NUMA_MEM_INTERLEAVE}"
    else
        echo "NUMA Policy: disabled"
    fi
    echo "========================================"
    echo ""
    
    if has_passwordless_sudo; then
        echo "Passwordless sudo detected: privileged monitoring and cgroup benchmarks are enabled."
        echo ""
    else
        echo "Passwordless sudo not available: privileged monitoring, readahead tuning, and cgroup-only benchmarks will be skipped."
        echo ""
    fi

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
    if ! (cd hnswlib && "$PYTHON_EXECUTABLE" setup.py build_ext --inplace); then
        echo "Error: hnswlib build failed."
        exit 1
    fi
    echo "hnswlib build complete"
    echo ""

    echo "Building predicache python library..."
    if ! (cd "${SCRIPT_DIR}/predictivetranslation-sigmod2026" && "$PYTHON_EXECUTABLE" setup.py build_ext --inplace); then
        echo "Error: predicache Python extension build failed."
        exit 1
    fi
    if ! verify_predicache_import "$PYTHON_EXECUTABLE"; then
        exit 1
    fi
    echo "predicache build complete"
    echo ""

    local index_file="./exps/usearch_index.bin"
    if [ "$USEARCH_SUPPORTED" -eq 1 ]; then
        # Step 1: Build the usearch index once
        echo "--- Building usearch index ---"
        local build_dir="${RESULTS_BASE_DIR}/physgb_build"
        mkdir -p "$build_dir"
        
        IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$USEARCH_MMAP_CONFIG"
        
        if ! run_benchmark \
            "$config_name" \
            "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
            "N/A" \
            "$build_dir" \
            "index" \
            "$index_file" \
            "$PYTHON_EXECUTABLE" \
            "$PYTHON_PATH"; then
            echo "Error: usearch index build benchmark failed."
            exit 1
        fi

        echo "--- Usearch index build complete ---"
        sleep 5
    fi

    rm -f "$BLOCK"
    ensure_block_file "$BLOCK" "20G"
    rm -f "${SCRIPT_DIR}/heapfile.predicache_oom.${DATASET}"

    # Run benchmarks for each physical memory size for other configs
    for physgb in "${PHYSGB_VALUES[@]}"; do
        echo "========================================"
        echo "Testing with PHYSGB=${physgb}GB"
        echo "========================================"
        
        # Create results directory for this memory size
        results_dir="${RESULTS_BASE_DIR}/physgb_${physgb}"
        mkdir -p "$results_dir"
        
        if [ "$USEARCH_SUPPORTED" -eq 1 ]; then
            # Run usearch search phase
            echo "--- Testing usearch search with PHYSGB=${physgb}GB ---"
            IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$USEARCH_MMAP_CONFIG"
            if ! run_benchmark \
                "$config_name" \
                "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
                "$physgb" \
                "$results_dir" \
                "search" \
                "$index_file" \
                "$PYTHON_EXECUTABLE" \
                "$PYTHON_PATH"; then
                echo "Error: usearch search benchmark failed for PHYSGB=${physgb}."
                exit 1
            fi
            sleep 2
        else
            echo "--- Skipping usearch search with PHYSGB=${physgb}GB (module unavailable) ---"
        fi

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
            
            if ! run_benchmark \
                "$config_name" \
                "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$library" \
                "$physgb" \
                "$results_dir" \
                "search" \
                "" \
                "$PYTHON_EXECUTABLE" \
                "$PYTHON_PATH"; then
                echo "Error: benchmark failed for ${config_name} at PHYSGB=${physgb}."
                exit 1
            fi

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
    ./extract_results_hnsw_oom.sh "$DATASET"
}

# Run main function
main "$@"
