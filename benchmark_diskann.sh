#!/bin/bash

# --- Configuration ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fl-dedup

PYTHON_CMD=$(which python3)
BENCHMARK_SCRIPT="benchmark_diskann.py"
FLAMEGRAPH_DIR="./FlameGraph"

# --- Main Script Logic ---
set -e # Exit immediately if a command fails.

echo "=================================================="
echo " Starting DiskANN Full Flow Profiling"
echo "=================================================="

# This function orchestrates the complex two-pause profiling for a single target
profile_target_full_flow() {
    local target_name=$1
    local perf_build_file="${target_name}_build.perf.data"
    local perf_search_file="${target_name}_search.perf.data"
    local log_file="${target_name}_full_run.log"

    echo ""
    echo "--------------------------------------------------"
    echo "--> Profiling Full Flow: ${target_name}"
    echo "--------------------------------------------------"

    # 1. Start Python script in background, logging to both screen and file
    echo "Starting Python script..."
    $PYTHON_CMD $BENCHMARK_SCRIPT --profile "$target_name" --pause 2>&1 | tee "$log_file" &
    PYTHON_BG_PID=$!

    # 2. Wait for the FIRST pause (before build)
    echo "Waiting for Python script to be ready for build..."
    PYTHON_PROC_PID=""
    for i in {1..600}; do
        if grep -q 'READY FOR BUILD' "$log_file"; then
            PYTHON_PROC_PID=$(grep 'PID:' "$log_file" | tail -n 1 | awk '{print $NF}')
            break
        fi
        sleep 1
    done

    if [ -z "$PYTHON_PROC_PID" ]; then
        echo "Error: Timed out waiting for Python's first pause signal." >&2
        cat "$log_file"
        kill -9 $PYTHON_BG_PID
        exit 1
    fi
    echo "Python (PID: $PYTHON_PROC_PID) is paused, ready for build profiling."

    # 3. Profile the BUILD phase
    echo "Starting perf for BUILD phase..."
    sudo perf record -o "$perf_build_file" --call-graph fp -p "$PYTHON_PROC_PID" &
    PERF_BUILD_PID=$!
    
    # Give perf a moment to attach, then resume Python
    sleep 1
    echo "Resuming Python to start build..."
    kill -SIGCONT "$PYTHON_PROC_PID"

    # 4. Wait for the SECOND pause (after build, before search)
    echo "Waiting for build to complete and script to pause again..."
    for i in {1..1200}; do # Build can take a long time
        # Check for the completion signal first
        if grep -q 'BUILD COMPLETE' "$log_file"; then
            break
        fi
        # Check if python process died unexpectedly
        if ! kill -0 $PYTHON_BG_PID 2>/dev/null; then
             echo "Error: Python process died unexpectedly during build." >&2
             # If python dies, perf might still be running. Stop it.
             sudo kill -SIGINT $PERF_BUILD_PID
             exit 1
        fi
        sleep 1
    done
    
    echo "Build complete. Stopping build profiler..."
    # Gracefully stop the build perf process
    sudo kill -SIGINT $PERF_BUILD_PID

    # --- FIX: Replace `wait` with a polling loop ---
    echo "Waiting for perf process ($PERF_BUILD_PID) to terminate..."
    while kill -0 $PERF_BUILD_PID 2>/dev/null; do
        sleep 0.5
    done
    # --- END FIX ---
    
    echo "Perf for build phase saved to '$perf_build_file'."

    # 5. Profile the SEARCH phase
    echo "Python is paused, ready for search profiling."
    echo "Starting perf for SEARCH phase..."
    # This perf can run in the foreground as it's the final step
    (sleep 1 && echo "Resuming Python to start search..." && kill -SIGCONT "$PYTHON_PROC_PID") &
    
    sudo perf record -o "$perf_search_file" --call-graph fp -p "$PYTHON_PROC_PID"
    
    # 6. Wait for the entire Python process to finish
    wait $PYTHON_BG_PID
    echo "Python process finished. Perf for search phase saved to '$perf_search_file'."
    rm "$log_file"
}

# --- STAGE 1 & 2: Run profiling for each target ---
profile_target_full_flow "calico"
profile_target_full_flow "official"

# --- STAGE 3: Generate Flame Graphs ---
echo ""
echo "=========================================================="
echo " Generating Flame Graphs"
echo "=========================================================="

if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    echo "Error: FlameGraph directory not found at '$FLAMEGRAPH_DIR'" >&2
    exit 1
fi

PERF_FILES=(
    "calico_build.perf.data" "calico_search.perf.data"
    "official_build.perf.data" "official_search.perf.data"
)

for perf_file in "${PERF_FILES[@]}"; do
    if [ -f "$perf_file" ]; then
        svg_file="${perf_file%.perf.data}.svg"
        echo "--> Processing '$perf_file' -> '$svg_file'..."
        sudo chown $(whoami):$(whoami) "$perf_file"
        perf script -i "$perf_file" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | "$FLAMEGRAPH_DIR/flamegraph.pl" > "$svg_file"
    else
        echo "Warning: Perf data file '$perf_file' not found. Skipping."
    fi
done

echo ""
echo "=================================================="
echo "  PROFILING COMPLETE!"
echo "=================================================="
echo "Flame graphs have been generated:"
ls -1 *.svg
echo "Open these files in a web browser to analyze the results."