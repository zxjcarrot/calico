#!/bin/bash

# HNSW Out-of-Memory (OOM) Benchmark Experiment Script
# Tests Calico configurations and hnswlib with constrained physical memory

set -e  # Exit on error

# Configuration
DATASET="sift1m"
M=16
EF_CONSTRUCTION=200
K=10
EF_SEARCH=50

# Fixed thread count for all experiments
THREADS=32

# Virtual memory size (generous for swapping/mmap)
export VIRTGB=8G

# Physical memory sizes to test (in GB)
PHYSGB_VALUES=(0.3 0.5 1 2)
#PHYSGB_VALUES=(0.5 1 2)
# PHYSGB_VALUES=(0.3)
# Base directory for results
RESULTS_BASE_DIR="./exps/larger-than-memory/hnsw"

# Calico configurations to test
# Format: config_name:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH:LIBRARY
CALICO_CONFIGS=(
    "calico_array:1:0:0:1:calico"
    "calico_lockfree:1:4:0:1:calico"
    "vmcache:0:0:1:1:calico"
)

# Calico mmap + OS page cache configuration (needs cgroup)
CALICO_MMAP_OSCACHE_CONFIG="calico_mmap_oscache:2:0:1:0:calico"

# hnswlib with mmap configuration
HNSWLIB_MMAP_CONFIG="hnswlib_mmap:0:0:0:0:hnswlib"

# Helper function to create and configure cgroup for memory limit
setup_cgroup() {
    local memory_limit_mb=$1
    local cgroup_name="hnsw_benchmark"
    local cgroup_path="/sys/fs/cgroup/${cgroup_name}"
    
    echo "Setting up cgroup v2 with ${memory_limit_mb}MB memory limit (swap enabled)..." >&2
    
    # Create cgroup
    sudo mkdir -p "$cgroup_path"
    
    # Enable memory controller in parent
    echo "+memory" | sudo tee "${cgroup_path}/../cgroup.subtree_control" >/dev/null 2>&1 || true
    
    # Set memory limit (in bytes)
    local memory_limit_bytes=$((memory_limit_mb * 1024 * 1024))
    echo "$memory_limit_bytes" | sudo tee "${cgroup_path}/memory.max" >/dev/null
    # set memory.high to 99% of memory.max to allow some breathing room
    local memory_high_bytes=$((memory_limit_bytes * 99 / 100))
    echo "$memory_high_bytes" | sudo tee "${cgroup_path}/memory.high" >/dev/null 2>&1 || true

    # Set swap limit to 4x memory limit (allows swapping but still constrained)
    local swap_limit_bytes=$((memory_limit_bytes * 4))
    echo "$swap_limit_bytes" | sudo tee "${cgroup_path}/memory.swap.max" >/dev/null 2>&1 || true
    
    echo "$cgroup_path"
}

# Helper function to run process in cgroup
run_in_cgroup() {
    local cgroup_path=$1
    shift
    local cmd="$@"
    
    # Move current shell to cgroup, run command, then cleanup
    sudo sh -c "echo $$ > ${cgroup_path}/cgroup.procs && $cmd"
}

# Helper function to cleanup cgroup
cleanup_cgroup() {
    local cgroup_path=$1
    
    if [ -d "$cgroup_path" ]; then
        echo "Cleaning up cgroup: $cgroup_path"
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
    
    sudo vmtouch -e heapfile
    echo "=========================================="
    echo "Running: $config_name with PHYSGB=${physgb}GB"
    echo "Library: $library, Threads: $THREADS"
    echo "TRADITIONAL=$traditional, TRADHASH=$tradhash"
    echo "NOHUGEPAGE_FRAME_MEM=$nohugepage, PREFETCH=$prefetch"
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
    local output_file="${output_dir}/${config_name}_physgb${physgb}_threads${THREADS}_${timestamp}.log"
    local iostat_file="${output_dir}/${config_name}_physgb${physgb}_threads${THREADS}_${timestamp}.iostat"
    
    # Start iostat monitoring in background
    echo "Starting iostat monitoring to: $iostat_file"
    iostat -x 1 nvme0n1 > "$iostat_file" 2>&1 &
    local iostat_pid=$!
    
    # Check if we need to run in cgroup (hnswlib or calico_mmap_oscache)
    local needs_cgroup=0
    local memory_limit_mb=0
    
    if [ "$library" = "hnswlib" ]; then
        # Memory limit = 62% of PHYSGB (to match the index size ratio between hnswlib and Calico)
        # hnswlib's mmap file is ~62% the size of Calico's index
        # Given PHYSGB for Calico, hnswlib should use 62% of that memory
        needs_cgroup=1
        memory_limit_mb=$(printf "%.0f" $(echo "$physgb * 1024 * 0.62 + 50" | bc))
    elif [ "$config_name" = "calico_mmap_oscache" ]; then
        # calico_mmap_oscache uses TRADITIONAL=2, needs cgroup with full PHYSGB limit
        # (no 62% rule - use the full physical memory limit)
        needs_cgroup=1
        memory_limit_mb=$(printf "%.0f" $(echo "$physgb * 1024 + 50" | bc))
    fi
    
    # For hnswlib or calico_mmap_oscache, we need to run inside a cgroup with memory limit
    if [ "$needs_cgroup" -eq 1 ]; then
        
        echo "Running $config_name in cgroup with ${memory_limit_mb}MB memory limit..."
        
        # Setup cgroup
        cgroup_path=$(setup_cgroup "$memory_limit_mb")
        
        # Verify cgroup was created
        if [ ! -d "$cgroup_path" ]; then
            echo "ERROR: Failed to create cgroup at $cgroup_path"
            return 1
        fi
        
        # Run benchmark in cgroup using a wrapper script that ensures Python is in the cgroup
        {
            # Create a temporary script that will run inside the cgroup
            local temp_script=$(mktemp)
            cat > "$temp_script" << 'INNERSCRIPT'
#!/bin/bash
# This script runs inside the cgroup
echo "Shell PID $$ in cgroup"
# Set environment variables
export MMAP_FOR_HSNWLIB="$9"
export PHYSGB="${10}"
export VIRTGB="${11}"
export TRADITIONAL="${12}"
export TRADHASH="${13}"
export NOHUGEPAGE_FRAME_MEM="${14}"
export PREFETCH="${15}"
exec python3 -u benchmark_calico_hnsw.py \
    --config "$1" \
    --threads "$2" \
    --dataset "$3" \
    --M "$4" \
    --ef_construction "$5" \
    --k "$6" \
    --ef_search "$7" \
    --library "$8"
INNERSCRIPT
            chmod +x "$temp_script"
            
            # Run the script via sudo bash -c, which moves the shell to cgroup and execs Python
            sudo bash -c "
                echo \$\$ > '${cgroup_path}/cgroup.procs'
                echo 'Process '\$\$' moved to cgroup: ${cgroup_path}'
                echo 'Memory limit: '
                cat '${cgroup_path}/memory.max'
                echo 'Swap limit: '
                cat '${cgroup_path}/memory.swap.max'
                echo 'Initial memory.current: '
                cat '${cgroup_path}/memory.current'
                exec '$temp_script' '$config_name' '$THREADS' '$DATASET' '$M' '$EF_CONSTRUCTION' '$K' '$EF_SEARCH' '$library' '1' '${physgb}G' '${VIRTGB}' '$traditional' '$tradhash' '$nohugepage' '$prefetch'
            " 2>&1
            
            # Show final memory usage
            echo "Final cgroup memory stats:"
            sudo cat "${cgroup_path}/memory.current" 2>/dev/null && echo " bytes (memory.current)" || true
            sudo cat "${cgroup_path}/memory.peak" 2>/dev/null && echo " bytes (memory.peak)" || true
            
            rm -f "$temp_script"
        } | tee "$output_file"
        
        cleanup_cgroup "$cgroup_path"
    else
        # Regular run for Calico configurations (use -u for unbuffered output)
        python3 -u benchmark_calico_hnsw.py \
            --config "$config_name" \
            --threads $THREADS \
            --dataset "$DATASET" \
            --M $M \
            --ef_construction $EF_CONSTRUCTION \
            --k $K \
            --ef_search $EF_SEARCH \
            --library "$library" \
            2>&1 | tee "$output_file"
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
        echo "If hnswlib tests fail, you may need to enable cgroup delegation."
        echo ""
    fi
    
    # Build the project first
    echo "Building project..."
    make clean && make -j$(nproc)
    echo "Build complete"
    echo ""
    
    # Build hnswlib
    echo "Building hnswlib..."
    cd hnswlib && python3 setup.py build_ext --inplace && cd ..
    echo "hnswlib build complete"
    echo ""
    
    # Run benchmarks for each physical memory size
    for physgb in "${PHYSGB_VALUES[@]}"; do
        echo "========================================"
        echo "Testing with PHYSGB=${physgb}GB"
        echo "========================================"
        
        # Create results directory for this memory size
        results_dir="${RESULTS_BASE_DIR}/physgb_${physgb}"
        mkdir -p "$results_dir"
        
                
        # Test hnswlib with mmap
        # IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$HNSWLIB_MMAP_CONFIG"
        
        # run_benchmark \
        #     "$config_name" \
        #     "$traditional" \
        #     "$tradhash" \
        #     "$nohugepage" \
        #     "$prefetch" \
        #     "$library" \
        #     "$physgb" \
        #     "$results_dir"
        
        # # Small delay between runs
        # sleep 2
        
        # Test calico_mmap_oscache (TRADITIONAL=2, needs cgroup)
        IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$CALICO_MMAP_OSCACHE_CONFIG"
        
        run_benchmark \
            "$config_name" \
            "$traditional" \
            "$tradhash" \
            "$nohugepage" \
            "$prefetch" \
            "$library" \
            "$physgb" \
            "$results_dir"
        
        # Test each Calico configuration
        for config_spec in "${CALICO_CONFIGS[@]}"; do
            IFS=':' read -r config_name traditional tradhash nohugepage prefetch library <<< "$config_spec"
            
            run_benchmark \
                "$config_name" \
                "$traditional" \
                "$tradhash" \
                "$nohugepage" \
                "$prefetch" \
                "$library" \
                "$physgb" \
                "$results_dir"
            
            # Small delay between runs
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
