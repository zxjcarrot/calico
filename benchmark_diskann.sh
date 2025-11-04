#!/bin/bash

# --- Configuration ---
# Activate your conda environment. Replace with the correct path if needed.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fl-dedup

# Get the full path to the conda python interpreter to avoid sudo path issues
PYTHON_CMD=$(which python3)

BENCHMARK_SCRIPT="benchmark_diskann.py"
PERF_CALICO_DATA="perf_calico.data"
PERF_OFFICIAL_DATA="perf_official.data"
FLAMEGRAPH_DIR="./FlameGraph" # Assumes it's cloned next to your project dir

# --- Main Script Logic ---
set -e # Exit immediately if a command fails.

echo "=================================================="
echo " Starting DiskANN Profiling Workflow"
echo "=================================================="

# Function to perform the profiling sequence
profile_target() {
    local target_name=$1 # 'calico' or 'official'
    local perf_output_file=$2
    local pid_capture_file="${target_name}_run.log"

    echo ""
    echo "--------------------------------------------------"
    echo "--> Profiling: ${target_name}"
    echo "--------------------------------------------------"

    # 1. Start the Python script in the background. 
    #    It will build, load, print its PID, and then pause itself.
    echo "Starting Python script in background to prepare index..."
    $PYTHON_CMD $BENCHMARK_SCRIPT --pause true --profile "$target_name" > "$pid_capture_file" 2>&1 &
    
    # Get the PID of the background Python process
    PYTHON_BG_PID=$!

    # 2. Wait for the script to print its PID and pause.
    #    We poll the output file until we find the "PID:" line.
    echo "Waiting for Python script to build index and pause..."
    SEARCH_PID=""
    for i in {1..500}; do # Timeout after 500 seconds
        if grep -q 'PID:' "$pid_capture_file"; then
            SEARCH_PID=$(grep 'PID:' "$pid_capture_file" | tail -n 1 | awk '{print $NF}')
            break
        fi
        sleep 1
    done

    if [ -z "$SEARCH_PID" ]; then
        echo "Error: Timed out waiting for Python script to pause. Log content:" >&2
        cat "$pid_capture_file"
        kill -9 $PYTHON_BG_PID
        exit 1
    fi
    echo "Python script (PID: $SEARCH_PID) is ready and paused."

    # 3. Start a sub-shell that will wait briefly and then resume the Python script.
    #    This gives perf time to attach properly before the real work begins.
    (sleep 1 && echo "Resuming Python process (PID: $SEARCH_PID)..." && kill -SIGCONT "$SEARCH_PID") &

    # 4. Run perf record in the FOREGROUND, attached to the paused process.
    #    When the process is resumed by the sub-shell, perf records its execution.
    #    When the Python process exits, perf automatically stops.
    echo "Attaching perf to PID $SEARCH_PID. Recording will start after resume."
    sudo perf record -o "$perf_output_file" --call-graph fp -p "$SEARCH_PID"

    # Wait for the main background process to finish completely
    wait $PYTHON_BG_PID
    echo "Perf recording for '$target_name' finished. Data saved to '$perf_output_file'."
    rm "$pid_capture_file"
}

# Run the profiling for both implementations
profile_target "calico" "$PERF_CALICO_DATA"
profile_target "official" "$PERF_OFFICIAL_DATA"

echo ""
echo "=========================================================="
echo " STEP 3: Generating Flame Graphs"
echo "=========================================================="

if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    echo "Error: FlameGraph directory not found at '$FLAMEGRAPH_DIR'" >&2
    exit 1
fi

echo "--> Fixing permissions on perf data files..."
sudo chown $(whoami):$(whoami) "$PERF_CALICO_DATA" "$PERF_OFFICIAL_DATA"

echo "--> Generating calico_search.svg..."
perf script -i "$PERF_CALICO_DATA" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | "$FLAMEGRAPH_DIR/flamegraph.pl" > calico_search.svg

echo "--> Generating official_search.svg..."
perf script -i "$PERF_OFFICIAL_DATA" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | "$FLAMEGRAPH_DIR/flamegraph.pl" > official_search.svg

echo ""
echo "=================================================="
echo "  PROFILING COMPLETE!"
echo "=================================================="
echo "Flame graphs have been generated:"
echo " - calico_search.svg"
echo " - official_search.svg"
echo "Open these files in a web browser to analyze the results."