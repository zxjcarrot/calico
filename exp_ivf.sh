#!/bin/bash

set -e


# Format: config_name:library:TRADITIONAL:TRADHASH:NOHUGEPAGE_FRAME_MEM:PREFETCH:PREDICTIVE_TRANSLATION
BASE_CONFIGS=(
    # "predicache_inmem:predicache:0:0:0:1:1"
    # "predicache_inmem_no_pt:predicache:0:0:0:1:0"
    # "vmcache:calico:0:0:1:1:-"
    #"calico:calico:1:0:0:1:-"
    # "pdx_calico:pdx:1:0:0:1:-"
    # "pdx_predicache:predicache:0:0:0:1:1"
    # "pdx_vmcache:pdx:0:0:1:1:-"
    #"pdx_predicache_no_pt:predicache:0:0:0:1:0"
    #"pdx_openaddr:pdx:1:2:0:1:-"
    "pdx_robinhood:pdx:1:7:0:1:-"
)

DATASET="${DATASET:-all}"
#DEFAULT_DATASETS=(dbpedia_openai_1m gist1m sift1m sift10M)
DEFAULT_DATASETS=(gist1m msong dbpedia_openai_1m sift10M)
# DEFAULT_DATASETS=(gist1m)
#DEFAULT_DATASETS=(gist1m)
#DEFAULT_DATASETS=(gist1m)
NUM_CLUSTERS="${NUM_CLUSTERS:-0}"
NPROBE="${NPROBE:-16}"
K="${K:-10}"
KMEANS_ITERS="${KMEANS_ITERS:-8}"
TRAIN_COUNT="${TRAIN_COUNT:-0}"
QUERY_COUNT="${QUERY_COUNT:-0}"
USER_BASE_LIMIT_OVERRIDE="${BASE_LIMIT:-}"
BASE_LIMIT="${BASE_LIMIT:-1000000}"
BUILD_THREADS_OVERRIDE="${BUILD_THREADS:-}"
USER_THREAD_COUNTS_OVERRIDE="${THREAD_COUNTS_OVERRIDE:-}"
FOCUS_CONFIGS_OVERRIDE="${FOCUS_CONFIGS:-}"
IVF_ENABLE_PERF="${IVF_ENABLE_PERF:-1}"
IVF_ENABLE_SEARCH_STATS="${IVF_ENABLE_SEARCH_STATS:-0}"
IVF_PERF_EVENT_GROUP_PRESET="${IVF_PERF_EVENT_GROUP_PRESET:-}"
CALICO_PERF_EVENT_GROUP_PRESET_OVERRIDE="${CALICO_PERF_EVENT_GROUP_PRESET:-}"
RESET_BLOCK_FILES_OVERRIDE="${RESET_BLOCK_FILES:-}"
PREFETCH_OVERRIDE_VALUE="${PREFETCH_OVERRIDE:-}"
PREFETCH_WINDOW_OVERRIDE_VALUE="${PDX_ASYNC_PREFETCH_WINDOW:-}"

USER_PHYSGB_OVERRIDE="${PHYSGB:-}"
USER_POOLGB_OVERRIDE="${POOLGB:-}"
USER_USE_IOURING_OVERRIDE="${USE_IOURING:-}"
USER_CALICO_WARMUP_RECOVERED_INDEX_OVERRIDE="${CALICO_WARMUP_RECOVERED_INDEX:-}"
export PHYSGB="${PHYSGB:-2G}"
export VIRTGB="${VIRTGB:-64G}"
export POOLGB="${POOLGB:-2G}"
export USE_IOURING="${USE_IOURING:-0}"
export CALICO_WARMUP_RECOVERED_INDEX="${CALICO_WARMUP_RECOVERED_INDEX:-1}"
DATASET_BUFFER_POOL_SIZE="6G"
OUT_OF_MEMORY_BUFFER_POOL_SIZE="1.8G"
IN_MEMORY_PREFETCH_WINDOW_SIZE=8
OUT_OF_MEMORY_PREFETCH_WINDOW_SIZE=32

if [ -n "$USER_THREAD_COUNTS_OVERRIDE" ]; then
    read -r -a THREAD_COUNTS <<< "$USER_THREAD_COUNTS_OVERRIDE"
else
    THREAD_COUNTS=(1 4 8 16 32 64 128)
    #THREAD_COUNTS=(128)
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATASET_URL_OVERRIDE="${DATASET_URL:-}"
USER_DATASET_ROOT_DIR_OVERRIDE="${DATASET_ROOT_DIR:-}"
USER_DATASET_ARCHIVE_PATH_OVERRIDE="${DATASET_ARCHIVE_PATH:-}"
USER_DATASET_H5_FILENAME_OVERRIDE="${DATASET_H5_FILENAME:-}"
USER_DATASET_LOCAL_FILE_OVERRIDE="${DATASET_LOCAL_FILE:-}"
DATASET_URL="$USER_DATASET_URL_OVERRIDE"
DATASET_ROOT_DIR="$USER_DATASET_ROOT_DIR_OVERRIDE"
DATASET_ARCHIVE_PATH="$USER_DATASET_ARCHIVE_PATH_OVERRIDE"
DATASET_H5_FILENAME="$USER_DATASET_H5_FILENAME_OVERRIDE"
DATASET_LOCAL_FILE="$USER_DATASET_LOCAL_FILE_OVERRIDE"
LEGACY_DATASET_LOCAL_FILE=""
EXTRACTED_VECTORS_PATH=""
EXTRACTED_TESTS_PATH=""
USER_DATASET_SOURCE_BASE_FILE_OVERRIDE="${DATASET_SOURCE_BASE_FILE:-}"
USER_DATASET_SOURCE_QUERY_FILE_OVERRIDE="${DATASET_SOURCE_QUERY_FILE:-}"
USER_DATASET_SOURCE_GT_FILE_OVERRIDE="${DATASET_SOURCE_GT_FILE:-}"
USER_DATASET_SOURCE_MAT_FILE_OVERRIDE="${DATASET_SOURCE_MAT_FILE:-}"
DATASET_SOURCE_BASE_FILE="$USER_DATASET_SOURCE_BASE_FILE_OVERRIDE"
DATASET_SOURCE_QUERY_FILE="$USER_DATASET_SOURCE_QUERY_FILE_OVERRIDE"
DATASET_SOURCE_GT_FILE="$USER_DATASET_SOURCE_GT_FILE_OVERRIDE"
DATASET_SOURCE_MAT_FILE="$USER_DATASET_SOURCE_MAT_FILE_OVERRIDE"
USER_DATASET_PAGE_SIZE_BYTES_OVERRIDE="${DATASET_PAGE_SIZE_BYTES:-}"
DATASET_PAGE_SIZE_BYTES="$USER_DATASET_PAGE_SIZE_BYTES_OVERRIDE"
DATASET_DOWNLOAD_MODE=""
USER_BLOCK_OVERRIDE="${BLOCK:-}"
RESOLVED_BASE_COUNT=""
RESOLVED_QUERY_COUNT=""
RESOLVED_NUM_CLUSTERS=""
RESOLVED_NPROBE=""
RESOLVED_TRAIN_COUNT=""
CURRENT_PREDICACHE_BUILD_SIGNATURE=""
CURRENT_CALICO_BUILD_SIGNATURE=""
BUILD_STATE_DIR="${SCRIPT_DIR}/.exp_ivf_build_state"
PREDICACHE_BUILD_SIGNATURE_FILE="${BUILD_STATE_DIR}/predicache_py.signature"
CALICO_BUILD_SIGNATURE_FILE="${BUILD_STATE_DIR}/calico_py.signature"
cd "$SCRIPT_DIR"
mkdir -p "$BUILD_STATE_DIR"

first_existing_path_or_default() {
    local default_path="$1"
    shift

    local candidate_path
    for candidate_path in "$@"; do
        if [ -f "$candidate_path" ]; then
            printf '%s\n' "$candidate_path"
            return 0
        fi
    done

    printf '%s\n' "$default_path"
}

configure_dataset() {
    DATASET_URL="$USER_DATASET_URL_OVERRIDE"
    DATASET_ROOT_DIR="$USER_DATASET_ROOT_DIR_OVERRIDE"
    DATASET_ARCHIVE_PATH="$USER_DATASET_ARCHIVE_PATH_OVERRIDE"
    DATASET_H5_FILENAME="$USER_DATASET_H5_FILENAME_OVERRIDE"
    DATASET_LOCAL_FILE="$USER_DATASET_LOCAL_FILE_OVERRIDE"
    LEGACY_DATASET_LOCAL_FILE=""
    EXTRACTED_VECTORS_PATH=""
    EXTRACTED_TESTS_PATH=""
    DATASET_SOURCE_BASE_FILE="$USER_DATASET_SOURCE_BASE_FILE_OVERRIDE"
    DATASET_SOURCE_QUERY_FILE="$USER_DATASET_SOURCE_QUERY_FILE_OVERRIDE"
    DATASET_SOURCE_GT_FILE="$USER_DATASET_SOURCE_GT_FILE_OVERRIDE"
    DATASET_SOURCE_MAT_FILE="$USER_DATASET_SOURCE_MAT_FILE_OVERRIDE"
    DATASET_PAGE_SIZE_BYTES="$USER_DATASET_PAGE_SIZE_BYTES_OVERRIDE"
    DATASET_DOWNLOAD_MODE=""
    local dataset_base_limit_default="1000000"

    case "$DATASET" in
        dbpedia|dbpedia_openai_1m)
            DATASET="dbpedia_openai_1m"
            DATASET_BUFFER_POOL_SIZE="9G"
            DATASET_URL="${DATASET_URL:-https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz}"
            DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${SCRIPT_DIR}/data/ann-filtered-benchmark}"
            DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH:-${DATASET_ROOT_DIR}/dbpedia_openai_1M.tgz}"
            DATASET_H5_FILENAME="${DATASET_H5_FILENAME:-dbpedia_openai_1M.h5}"
            DATASET_LOCAL_FILE="${DATASET_LOCAL_FILE:-${DATASET_ROOT_DIR}/${DATASET_H5_FILENAME}}"
            LEGACY_DATASET_LOCAL_FILE="${DATASET_ROOT_DIR}/dbpedia-openai-1M-1536-angular/dbpedia_openai_1M"
            EXTRACTED_VECTORS_PATH="${DATASET_ROOT_DIR}/vectors.npy"
            EXTRACTED_TESTS_PATH="${DATASET_ROOT_DIR}/tests.jsonl"
            DATASET_PAGE_SIZE_BYTES="${DATASET_PAGE_SIZE_BYTES:-8192}"
            DATASET_DOWNLOAD_MODE="dbpedia_archive"
            ;;
        gist1m|gist-960-euclidean|gist_960_euclidean)
            DATASET="gist1m"
            DATASET_BUFFER_POOL_SIZE="5G"
            DATASET_URL="${DATASET_URL:-http://ann-benchmarks.com/gist-960-euclidean.hdf5}"
            DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${SCRIPT_DIR}/data/ann-benchmarks/gist-960-euclidean}"
            DATASET_LOCAL_FILE="${DATASET_LOCAL_FILE:-${DATASET_ROOT_DIR}/gist-960-euclidean.hdf5}"
            DATASET_PAGE_SIZE_BYTES="${DATASET_PAGE_SIZE_BYTES:-4096}"
            DATASET_DOWNLOAD_MODE="direct_hdf5"
            ;;
        msong|msong994k|msong_994k)
            DATASET="msong"
            DATASET_BUFFER_POOL_SIZE="3G"
            DATASET_URL="${DATASET_URL:-https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz}"
            DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${SCRIPT_DIR}/data/gqr/msong}"
            DATASET_ARCHIVE_PATH="${DATASET_ARCHIVE_PATH:-${DATASET_ROOT_DIR}/msong.tar.gz}"
            DATASET_LOCAL_FILE="${DATASET_LOCAL_FILE:-${DATASET_ROOT_DIR}/msong-ivf.hdf5}"
            DATASET_PAGE_SIZE_BYTES="${DATASET_PAGE_SIZE_BYTES:-4096}"
            DATASET_DOWNLOAD_MODE="gqr_fvecs_archive"
            ;;
        sift1m|sift1M)
            DATASET="sift1m"
            DATASET_BUFFER_POOL_SIZE="0.5G"
            DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${SCRIPT_DIR}/data/sift1m}"
            DATASET_LOCAL_FILE="${DATASET_LOCAL_FILE:-${DATASET_ROOT_DIR}/sift1m-ivf.hdf5}"
            DATASET_SOURCE_BASE_FILE="${DATASET_SOURCE_BASE_FILE:-${DATASET_ROOT_DIR}/sift_base.fvecs}"
            DATASET_SOURCE_QUERY_FILE="${DATASET_SOURCE_QUERY_FILE:-${DATASET_ROOT_DIR}/sift_query.fvecs}"
            DATASET_SOURCE_GT_FILE="${DATASET_SOURCE_GT_FILE:-${DATASET_ROOT_DIR}/sift_groundtruth.ivecs}"
            DATASET_URL="${DATASET_URL:-${DATASET_SOURCE_BASE_FILE}}"
            DATASET_PAGE_SIZE_BYTES="${DATASET_PAGE_SIZE_BYTES:-4096}"
            DATASET_DOWNLOAD_MODE="local_sift1m"
            ;;
        sift10m|sift10M|sift_10m|sift-10m|SIFT10M)
            DATASET="sift10M"
            DATASET_BUFFER_POOL_SIZE="7G"
            dataset_base_limit_default="10000000"
            DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${SCRIPT_DIR}/data/SIFT10M}"
            DATASET_LOCAL_FILE="${DATASET_LOCAL_FILE:-${DATASET_ROOT_DIR}/sift10M-ivf.hdf5}"
            DATASET_SOURCE_MAT_FILE="${DATASET_SOURCE_MAT_FILE:-$(first_existing_path_or_default \
                "${DATASET_ROOT_DIR}/SIFT10Mfeatures.mat" \
                "${DATASET_ROOT_DIR}/SIFT10Mfeatures.mat" \
                "${DATASET_ROOT_DIR}/sift10Mfeatures.mat" \
                "${DATASET_ROOT_DIR}/sift10mfeatures.mat")}"
            DATASET_SOURCE_GT_FILE="${DATASET_SOURCE_GT_FILE:-$(first_existing_path_or_default \
                "${DATASET_ROOT_DIR}/ground_truth_indices.npy" \
                "${DATASET_ROOT_DIR}/ground_truth_indices.npy" \
                "${DATASET_ROOT_DIR}/ground_truth.npy" \
                "${DATASET_ROOT_DIR}/groundtruth.npy")}"
            DATASET_URL="${DATASET_URL:-${DATASET_SOURCE_MAT_FILE}}"
            DATASET_PAGE_SIZE_BYTES="${DATASET_PAGE_SIZE_BYTES:-4096}"
            DATASET_DOWNLOAD_MODE="local_sift10m"
            ;;
        *)
            echo "Error: Unsupported IVF dataset: $DATASET"
            echo "Supported datasets: dbpedia_openai_1m, gist1m, msong, sift1m, sift10M"
            exit 1
            ;;
    esac

    if [ -n "$USER_BASE_LIMIT_OVERRIDE" ]; then
        BASE_LIMIT="$USER_BASE_LIMIT_OVERRIDE"
    else
        BASE_LIMIT="$dataset_base_limit_default"
    fi

    OUT_OF_MEMORY_BUFFER_POOL_SIZE="$(scaled_buffer_pool_size "$DATASET_BUFFER_POOL_SIZE" 0.30)"
}

apply_dataset_buffer_pool_size() {
    apply_buffer_pool_size "$DATASET_BUFFER_POOL_SIZE"
}

scaled_buffer_pool_size() {
    local buffer_pool_size="$1"
    local scale_factor="$2"
    local numeric_size
    local unit

    numeric_size="$(printf '%s\n' "$buffer_pool_size" | sed -E 's/^([0-9]+(\.[0-9]+)?).*/\1/')"
    unit="$(printf '%s\n' "$buffer_pool_size" | sed -E 's/^[0-9]+(\.[0-9]+)?//')"

    if [ -z "$numeric_size" ] || [ -z "$unit" ]; then
        echo "Error: Unsupported buffer pool size format: $buffer_pool_size" >&2
        exit 1
    fi

    awk -v size="$numeric_size" -v factor="$scale_factor" -v unit="$unit" 'BEGIN {
        scaled = size * factor;
        if (scaled <= 0) {
            scaled = size;
        }
        printf "%.2f%s\n", scaled, unit;
    }'
}

apply_buffer_pool_size() {
    local buffer_pool_size="$1"

    if [ -n "$USER_PHYSGB_OVERRIDE" ]; then
        export PHYSGB="$USER_PHYSGB_OVERRIDE"
    else
        export PHYSGB="$buffer_pool_size"
    fi

    if [ -n "$USER_POOLGB_OVERRIDE" ]; then
        export POOLGB="$USER_POOLGB_OVERRIDE"
    else
        export POOLGB="$buffer_pool_size"
    fi
}

buffer_pool_size_dir_suffix() {
    local buffer_pool_size="${1^^}"
    buffer_pool_size="${buffer_pool_size%IB}"
    buffer_pool_size="${buffer_pool_size%I}"
    buffer_pool_size="${buffer_pool_size%GB}"
    buffer_pool_size="${buffer_pool_size%G}"
    printf '%sgb\n' "$buffer_pool_size"
}

dataset_results_dir_name() {
    local dataset_name="$1"
    local buffer_pool_size="$2"
    printf '%s_%s\n' "$dataset_name" "$(buffer_pool_size_dir_suffix "$buffer_pool_size")"
}

apply_prefetch_window_for_mode() {
    local default_prefetch_window="$1"

    if [ -n "$PREFETCH_WINDOW_OVERRIDE_VALUE" ]; then
        export PDX_ASYNC_PREFETCH_WINDOW="$PREFETCH_WINDOW_OVERRIDE_VALUE"
    else
        export PDX_ASYNC_PREFETCH_WINDOW="$default_prefetch_window"
    fi
}

configure_experiment_mode() {
    local experiment_mode="$1"

    case "$experiment_mode" in
        in-memory)
            ACTIVE_BUFFER_POOL_SIZE="$DATASET_BUFFER_POOL_SIZE"
            ACTIVE_PREFETCH_ENABLED=0
            ACTIVE_PREFETCH_WINDOW_SIZE="$IN_MEMORY_PREFETCH_WINDOW_SIZE"
            ;;
        out-of-memory)
            ACTIVE_BUFFER_POOL_SIZE="$OUT_OF_MEMORY_BUFFER_POOL_SIZE"
            ACTIVE_PREFETCH_ENABLED=1
            ACTIVE_PREFETCH_WINDOW_SIZE="$OUT_OF_MEMORY_PREFETCH_WINDOW_SIZE"
            ;;
        *)
            echo "Error: Unsupported IVF experiment mode: $experiment_mode"
            exit 1
            ;;
    esac

    ACTIVE_EXPERIMENT_MODE="$experiment_mode"
    ACTIVE_DATASET_RESULTS_DIR_NAME="$(dataset_results_dir_name "$DATASET" "$ACTIVE_BUFFER_POOL_SIZE")"
    RESULTS_BASE_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/${ACTIVE_EXPERIMENT_MODE}/ivf/${ACTIVE_DATASET_RESULTS_DIR_NAME}"

    apply_buffer_pool_size "$ACTIVE_BUFFER_POOL_SIZE"
    apply_prefetch_window_for_mode "$ACTIVE_PREFETCH_WINDOW_SIZE"
}

resolve_datasets_to_run() {
    if [ "$DATASET" = "all" ]; then
        DATASETS_TO_RUN=("${DEFAULT_DATASETS[@]}")
    else
        DATASETS_TO_RUN=("$DATASET")
    fi
}

resolve_datasets_to_run
DATASET="${DATASETS_TO_RUN[0]}"
configure_dataset
apply_dataset_buffer_pool_size

block_path_for_config() {
    local target_config_name="$1"
    local block_config_name="$target_config_name"

    case "$target_config_name" in
        pdx_calico_huge|pdx_calico_nohuge)
            block_config_name="pdx_calico"
            ;;
        pdx_vmcache_huge|pdx_vmcache_nohuge)
            block_config_name="pdx_vmcache"
            ;;
    esac

    if [ -n "$USER_BLOCK_OVERRIDE" ]; then
        printf '%s\n' "$USER_BLOCK_OVERRIDE"
        return 0
    fi
    printf '%s\n' "${SCRIPT_DIR}/heapfile.${block_config_name}.${DATASET}"
}

ensure_block_file() {
    local block_path="$1"
    local block_size="${2:-32G}"
    if [ -f "$block_path" ]; then
        return 0
    fi
    echo "Creating sparse block file at ${block_path} (${block_size})..."
    if ! fallocate -l "$block_size" "$block_path"; then
        truncate -s "$block_size" "$block_path"
    fi
}

cleanup_reusable_index_files() {
    for config_spec in "${CONFIGS[@]}"; do
        IFS=':' read -r config_name _library _traditional _tradhash _nohugepage _prefetch _pt_enabled <<< "$config_spec"
        local block_path
        block_path="$(block_path_for_config "$config_name")"
        if [ -e "$block_path" ]; then
            echo "Removing stale IVF block file: $block_path"
            rm -f "$block_path"
        fi
    done
}

export PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}"

PYTHON_EXECUTABLE=$(command -v python3 || true)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "Error: python3 not found in PATH."
    exit 1
fi

MEMORY_BW_METRIC=${MEMORY_BW_METRIC:-nps1_die_to_dram}
NUMA_CPU_NODE=${NUMA_CPU_NODE:-${NUMA_NODE:-0}}
NUMA_MEM_NODE=${NUMA_MEM_NODE:-${NUMA_NODE:-0}}
USE_NUMACTL_BINDING=${USE_NUMACTL_BINDING:-1}
if [ -n "${CALICO_DISABLE_PERF+x}" ]; then
    export CALICO_DISABLE_PERF="$CALICO_DISABLE_PERF"
elif [ "$IVF_ENABLE_PERF" = "1" ]; then
    export CALICO_DISABLE_PERF=0
else
    export CALICO_DISABLE_PERF=1
fi

if [ "$IVF_ENABLE_SEARCH_STATS" = "1" ]; then
    export CALICO_ENABLE_SEARCH_STATS=1
else
    unset CALICO_ENABLE_SEARCH_STATS
fi

if [ -n "$CALICO_PERF_EVENT_GROUP_PRESET_OVERRIDE" ]; then
    export CALICO_PERF_EVENT_GROUP_PRESET="$CALICO_PERF_EVENT_GROUP_PRESET_OVERRIDE"
elif [ -n "$IVF_PERF_EVENT_GROUP_PRESET" ]; then
    export CALICO_PERF_EVENT_GROUP_PRESET="$IVF_PERF_EVENT_GROUP_PRESET"
fi

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
    for module_name in setuptools pybind11 numpy h5py; do
        if ! python_has_module "$module_name"; then
            missing_modules+=("$module_name")
        fi
    done
    if [ "${#missing_modules[@]}" -gt 0 ]; then
        echo "Error: Missing Python modules for $PYTHON_EXECUTABLE: ${missing_modules[*]}"
        echo "Install them with:"
        echo "  $PYTHON_EXECUTABLE -m pip install --user ${missing_modules[*]}"
        exit 1
    fi
}

check_python_prereqs

check_required_commands() {
    local missing_commands=()
    if [ "$DATASET_DOWNLOAD_MODE" = "direct_hdf5" ]; then
        if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
            missing_commands+=("curl_or_wget")
        fi
    elif [ "$DATASET_DOWNLOAD_MODE" = "local_sift1m" ] || [ "$DATASET_DOWNLOAD_MODE" = "local_sift10m" ]; then
        :
    else
        for command_name in tar; do
            if ! command -v "$command_name" >/dev/null 2>&1; then
                missing_commands+=("$command_name")
            fi
        done
        if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
            missing_commands+=("curl_or_wget")
        fi
    fi
    if [ "${#missing_commands[@]}" -gt 0 ]; then
        echo "Error: Missing required commands: ${missing_commands[*]}"
        exit 1
    fi
}

download_file_with_resume() {
    local url="$1"
    local output_path="$2"
    local expected_size=""

    if command -v curl >/dev/null 2>&1; then
        expected_size=$(curl -fsSI "$url" | awk 'BEGIN{IGNORECASE=1} /^Content-Length:/ {gsub("\r", "", $2); print $2; exit}')

        while true; do
            local current_size=0
            if [ -f "$output_path" ]; then
                current_size=$(stat -c '%s' "$output_path")
            fi

            if [ -n "$expected_size" ] && [ "$current_size" -eq "$expected_size" ]; then
                break
            fi

            if [ -n "$expected_size" ] && [ "$current_size" -gt "$expected_size" ]; then
                echo "Error: Downloaded file is larger than expected: $output_path ($current_size > $expected_size)"
                exit 1
            fi

            if ! curl -fL --retry 8 --retry-all-errors --continue-at - --output "$output_path" "$url"; then
                echo "Retrying dataset download after curl interruption..."
            fi
        done
        return 0
    fi

    wget -c -O "$output_path" "$url"
}

ensure_ivf_dataset_available() {
    if [ -f "$DATASET_LOCAL_FILE" ]; then
        return 0
    fi

    mkdir -p "$DATASET_ROOT_DIR"

    if [ "$DATASET_DOWNLOAD_MODE" = "local_sift1m" ]; then
        if [ ! -f "$DATASET_SOURCE_BASE_FILE" ] || [ ! -f "$DATASET_SOURCE_QUERY_FILE" ] || [ ! -f "$DATASET_SOURCE_GT_FILE" ]; then
            echo "Error: SIFT1M source files not found. Expected:"
            echo "  $DATASET_SOURCE_BASE_FILE"
            echo "  $DATASET_SOURCE_QUERY_FILE"
            echo "  $DATASET_SOURCE_GT_FILE"
            exit 1
        fi

        echo "Converting local SIFT1M files into IVF HDF5 file: $DATASET_LOCAL_FILE"
        "$PYTHON_EXECUTABLE" - "$DATASET_SOURCE_BASE_FILE" "$DATASET_SOURCE_QUERY_FILE" "$DATASET_SOURCE_GT_FILE" "$DATASET_LOCAL_FILE" <<'PY'
import os
import sys

import h5py
import numpy as np


def fvecs_metadata(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.int32, mode='r')
    if raw.size == 0:
        raise RuntimeError(f'Empty fvecs file: {path}')
    dim = int(raw[0])
    stride = dim + 1
    if dim <= 0 or raw.size % stride != 0:
        raise RuntimeError(f'Invalid fvecs layout in {path}: dim={dim}, raw_size={raw.size}')
    return dim, raw.size // stride


def ivecs_metadata(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.int32, mode='r')
    if raw.size == 0:
        raise RuntimeError(f'Empty ivecs file: {path}')
    dim = int(raw[0])
    stride = dim + 1
    if dim <= 0 or raw.size % stride != 0:
        raise RuntimeError(f'Invalid ivecs layout in {path}: dim={dim}, raw_size={raw.size}')
    return dim, raw.size // stride


def copy_fvecs_to_dataset(path: str, dataset: h5py.Dataset, chunk_rows: int = 8192) -> None:
    dim, count = fvecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    for start in range(0, count, chunk_rows):
        stop = min(start + chunk_rows, count)
        dataset[start:stop] = raw[start:stop, 1:].view(np.float32)


def load_fvecs(path: str) -> np.ndarray:
    dim, count = fvecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    return np.ascontiguousarray(raw[:, 1:].view(np.float32))


def load_ivecs(path: str) -> np.ndarray:
    dim, count = ivecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    return np.ascontiguousarray(raw[:, 1:], dtype=np.int32)


def main() -> None:
    base_path, query_path, gt_path, output_path = sys.argv[1:5]
    base_dim, base_count = fvecs_metadata(base_path)
    query_dim, query_count = fvecs_metadata(query_path)
    if base_dim != query_dim:
        raise RuntimeError(f'Base/query dimension mismatch: {base_dim} != {query_dim}')

    neighbors = load_ivecs(gt_path)
    if neighbors.shape[0] != query_count:
        raise RuntimeError(
            f'Ground-truth/query count mismatch: {neighbors.shape[0]} != {query_count}'
        )

    queries = load_fvecs(query_path)
    temp_output_path = f'{output_path}.tmp'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    with h5py.File(temp_output_path, 'w') as handle:
        train_ds = handle.create_dataset('train', shape=(base_count, base_dim), dtype=np.float32)
        copy_fvecs_to_dataset(base_path, train_ds)
        handle.create_dataset('test', data=queries, dtype=np.float32)
        handle.create_dataset('neighbors', data=neighbors, dtype=np.int32)

    os.replace(temp_output_path, output_path)


if __name__ == '__main__':
    main()
PY

        if [ ! -f "$DATASET_LOCAL_FILE" ]; then
            echo "Error: Failed to create local HDF5 dataset file: $DATASET_LOCAL_FILE"
            exit 1
        fi
        return 0
    fi

    if [ "$DATASET_DOWNLOAD_MODE" = "local_sift10m" ]; then
        if [ ! -f "$DATASET_SOURCE_MAT_FILE" ] || [ ! -f "$DATASET_SOURCE_GT_FILE" ]; then
            echo "Error: SIFT10M source files not found. Expected:"
            echo "  $DATASET_SOURCE_MAT_FILE"
            echo "  $DATASET_SOURCE_GT_FILE"
            exit 1
        fi

        echo "Converting local SIFT10M files into IVF HDF5 file: $DATASET_LOCAL_FILE"
        "$PYTHON_EXECUTABLE" - "$DATASET_SOURCE_MAT_FILE" "$DATASET_SOURCE_GT_FILE" "$DATASET_LOCAL_FILE" <<'PY'
import os
import sys

import h5py
import numpy as np


def main() -> None:
    mat_path, gt_path, output_path = sys.argv[1:4]
    ground_truth = np.load(gt_path)
    if ground_truth.ndim != 2:
        raise RuntimeError(f'Expected 2D ground-truth array in {gt_path}, got shape {ground_truth.shape}')

    num_queries = int(ground_truth.shape[0])
    expected_num_base = 10_000_000
    temp_output_path = f'{output_path}.tmp'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    with h5py.File(mat_path, 'r') as source, h5py.File(temp_output_path, 'w') as output:
        features = source['fea']
        if len(features.shape) != 2:
            raise RuntimeError(f'Expected 2D fea dataset in {mat_path}, got shape {features.shape}')

        total_rows = int(features.shape[0])
        dim = int(features.shape[1])
        num_base = min(expected_num_base, total_rows)
        query_start = num_base
        available_queries = max(0, total_rows - query_start)
        if available_queries <= 0:
            raise RuntimeError('No query rows available in SIFT10Mfeatures.mat after the base segment.')

        actual_queries = min(num_queries, available_queries)
        if actual_queries != num_queries:
            print(
                f'Warning: truncating SIFT10M ground truth/query count from {num_queries} to {actual_queries}',
                file=sys.stderr,
            )
            ground_truth = ground_truth[:actual_queries]
            num_queries = actual_queries

        train_ds = output.create_dataset('train', shape=(num_base, dim), dtype=np.float32)
        chunk_rows = max(1, min(8192, num_base))
        for start in range(0, num_base, chunk_rows):
            stop = min(start + chunk_rows, num_base)
            train_ds[start:stop] = np.asarray(features[start:stop, :], dtype=np.float32)

        queries = np.ascontiguousarray(features[query_start:query_start + num_queries, :], dtype=np.float32)
        output.create_dataset('test', data=queries, dtype=np.float32)
        output.create_dataset('neighbors', data=np.ascontiguousarray(ground_truth, dtype=np.int32), dtype=np.int32)

    os.replace(temp_output_path, output_path)


if __name__ == '__main__':
    main()
PY

        if [ ! -f "$DATASET_LOCAL_FILE" ]; then
            echo "Error: Failed to create local HDF5 dataset file: $DATASET_LOCAL_FILE"
            exit 1
        fi
        return 0
    fi

    if [ "$DATASET_DOWNLOAD_MODE" = "gqr_fvecs_archive" ]; then
        if [ ! -s "$DATASET_ARCHIVE_PATH" ]; then
            echo "Downloading IVF dataset archive with resume support..."
            echo "  URL: $DATASET_URL"
            echo "  Archive: $DATASET_ARCHIVE_PATH"
            download_file_with_resume "$DATASET_URL" "$DATASET_ARCHIVE_PATH"
        else
            echo "Reusing existing IVF dataset archive: $DATASET_ARCHIVE_PATH"
        fi

        if ! find "$DATASET_ROOT_DIR" -type f \( -iname '*.fvecs' -o -iname '*.ivecs' \) -print -quit | grep -q .; then
            echo "Extracting IVF dataset archive..."
            tar -xzf "$DATASET_ARCHIVE_PATH" -C "$DATASET_ROOT_DIR"
        else
            echo "Reusing extracted IVF source files under: $DATASET_ROOT_DIR"
        fi

        echo "Converting extracted IVF dataset into local HDF5 file: $DATASET_LOCAL_FILE"
        "$PYTHON_EXECUTABLE" - "$DATASET_ROOT_DIR" "$DATASET_LOCAL_FILE" <<'PY'
import os
import sys

import h5py
import numpy as np


def fvecs_metadata(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.int32, mode='r')
    if raw.size == 0:
        raise RuntimeError(f'Empty fvecs file: {path}')
    dim = int(raw[0])
    stride = dim + 1
    if dim <= 0 or raw.size % stride != 0:
        raise RuntimeError(f'Invalid fvecs layout in {path}: dim={dim}, raw_size={raw.size}')
    return dim, raw.size // stride


def ivecs_metadata(path: str) -> tuple[int, int]:
    raw = np.memmap(path, dtype=np.int32, mode='r')
    if raw.size == 0:
        raise RuntimeError(f'Empty ivecs file: {path}')
    dim = int(raw[0])
    stride = dim + 1
    if dim <= 0 or raw.size % stride != 0:
        raise RuntimeError(f'Invalid ivecs layout in {path}: dim={dim}, raw_size={raw.size}')
    return dim, raw.size // stride


def copy_fvecs_to_dataset(path: str, dataset: h5py.Dataset, chunk_rows: int = 8192) -> None:
    dim, count = fvecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    for start in range(0, count, chunk_rows):
        stop = min(start + chunk_rows, count)
        dataset[start:stop] = raw[start:stop, 1:].view(np.float32)


def load_fvecs(path: str) -> np.ndarray:
    dim, count = fvecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    return np.ascontiguousarray(raw[:, 1:].view(np.float32))


def load_ivecs(path: str) -> np.ndarray:
    dim, count = ivecs_metadata(path)
    raw = np.memmap(path, dtype=np.int32, mode='r').reshape(count, dim + 1)
    return np.ascontiguousarray(raw[:, 1:], dtype=np.int32)


def has_token(path: str, tokens: tuple[str, ...]) -> bool:
    name = os.path.basename(path).lower()
    return any(token in name for token in tokens)


def pick_base_and_query(fvec_paths: list[str]) -> tuple[str, str]:
    if len(fvec_paths) < 2:
        raise RuntimeError(f'Expected at least two .fvecs files, found {len(fvec_paths)} in archive')

    fvec_paths = sorted(fvec_paths, key=os.path.getsize, reverse=True)
    base_candidates = [p for p in fvec_paths if has_token(p, ('base', 'train', 'data'))]
    base_path = max(base_candidates, key=os.path.getsize) if base_candidates else fvec_paths[0]

    remaining = [p for p in fvec_paths if p != base_path]
    query_candidates = [p for p in remaining if has_token(p, ('query', 'test'))]
    query_path = min(query_candidates, key=os.path.getsize) if query_candidates else min(remaining, key=os.path.getsize)
    return base_path, query_path


def pick_groundtruth(ivec_paths: list[str], query_count: int):
    if not ivec_paths:
        return None

    matching = []
    for path in ivec_paths:
        _, count = ivecs_metadata(path)
        if count == query_count:
            matching.append(path)

    preferred_tokens = ('groundtruth', 'ground_truth', 'truth', 'neighbor', 'gt')
    if matching:
        preferred = [p for p in matching if has_token(p, preferred_tokens)]
        return preferred[0] if preferred else matching[0]

    preferred = [p for p in ivec_paths if has_token(p, preferred_tokens)]
    if preferred:
        return preferred[0]
    return ivec_paths[0]


def collect_paths(root_dir: str) -> tuple[list[str], list[str]]:
    fvec_paths: list[str] = []
    ivec_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            lower = filename.lower()
            if lower.endswith('.fvecs'):
                fvec_paths.append(path)
            elif lower.endswith('.ivecs'):
                ivec_paths.append(path)
    return sorted(fvec_paths), sorted(ivec_paths)


def main() -> None:
    root_dir, output_path = sys.argv[1:3]
    fvec_paths, ivec_paths = collect_paths(root_dir)
    if not fvec_paths:
        raise RuntimeError(f'No .fvecs files found under {root_dir}')

    base_path, query_path = pick_base_and_query(fvec_paths)
    base_dim, base_count = fvecs_metadata(base_path)
    query_dim, query_count = fvecs_metadata(query_path)
    if base_dim != query_dim:
        raise RuntimeError(
            f'Base/query dimension mismatch: {base_dim} != {query_dim} ({base_path}, {query_path})'
        )

    gt_path = pick_groundtruth(ivec_paths, query_count)
    neighbors = None
    if gt_path is not None:
        neighbors = load_ivecs(gt_path)
        if neighbors.shape[0] != query_count:
            raise RuntimeError(
                f'Ground-truth/query count mismatch: {neighbors.shape[0]} != {query_count} ({gt_path})'
            )

    queries = load_fvecs(query_path)

    temp_output_path = f'{output_path}.tmp'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    with h5py.File(temp_output_path, 'w') as handle:
        train_ds = handle.create_dataset('train', shape=(base_count, base_dim), dtype=np.float32)
        copy_fvecs_to_dataset(base_path, train_ds)
        handle.create_dataset('test', data=queries, dtype=np.float32)
        if neighbors is not None:
            handle.create_dataset('neighbors', data=neighbors, dtype=np.int32)

    os.replace(temp_output_path, output_path)

    print(f'Selected base vectors: {base_path}')
    print(f'Selected queries: {query_path}')
    if gt_path is not None:
        print(f'Selected ground truth: {gt_path}')
    else:
        print('No .ivecs ground truth found; writing train/test only and benchmarking recall as N/A unless --compute_recall is enabled.')
    print(f'Wrote HDF5 dataset: {output_path}')


if __name__ == '__main__':
    main()
PY

        if [ ! -f "$DATASET_LOCAL_FILE" ]; then
            echo "Error: Failed to create local HDF5 dataset file: $DATASET_LOCAL_FILE"
            exit 1
        fi
        return 0
    fi

    if [ "$DATASET_DOWNLOAD_MODE" = "direct_hdf5" ]; then
        echo "Downloading IVF dataset file with resume support..."
        echo "  URL: $DATASET_URL"
        echo "  File: $DATASET_LOCAL_FILE"
        download_file_with_resume "$DATASET_URL" "$DATASET_LOCAL_FILE"
        return 0
    fi

    if [ ! -s "$DATASET_ARCHIVE_PATH" ]; then
        echo "Downloading IVF dataset archive with resume support..."
        echo "  URL: $DATASET_URL"
        echo "  Archive: $DATASET_ARCHIVE_PATH"
        download_file_with_resume "$DATASET_URL" "$DATASET_ARCHIVE_PATH"
    else
        echo "Reusing existing IVF dataset archive: $DATASET_ARCHIVE_PATH"
    fi

    if [ -f "$LEGACY_DATASET_LOCAL_FILE" ]; then
        echo "Reusing legacy extracted IVF dataset file: $LEGACY_DATASET_LOCAL_FILE"
        DATASET_LOCAL_FILE="$LEGACY_DATASET_LOCAL_FILE"
        return 0
    fi

    if [ ! -f "$EXTRACTED_VECTORS_PATH" ] || [ ! -f "$EXTRACTED_TESTS_PATH" ]; then
        echo "Extracting IVF dataset archive..."
        tar -xzf "$DATASET_ARCHIVE_PATH" -C "$DATASET_ROOT_DIR"
    else
        echo "Reusing extracted IVF dataset files:"
        echo "  $EXTRACTED_VECTORS_PATH"
        echo "  $EXTRACTED_TESTS_PATH"
    fi

    if [ ! -f "$EXTRACTED_VECTORS_PATH" ] || [ ! -f "$EXTRACTED_TESTS_PATH" ]; then
        echo "Error: Extracted dataset files not found after extraction."
        echo "Expected:"
        echo "  $EXTRACTED_VECTORS_PATH"
        echo "  $EXTRACTED_TESTS_PATH"
        exit 1
    fi

    echo "Converting extracted IVF dataset into local HDF5 file: $DATASET_LOCAL_FILE"
    "$PYTHON_EXECUTABLE" - "$EXTRACTED_VECTORS_PATH" "$EXTRACTED_TESTS_PATH" "$DATASET_LOCAL_FILE" <<'PY'
import json
import os
import sys

import h5py
import numpy as np


def main() -> None:
    vectors_path, tests_path, output_path = sys.argv[1:4]

    base = np.load(vectors_path, mmap_mode='r')
    if base.ndim != 2:
        raise RuntimeError(f'Expected 2D base vectors in {vectors_path}, got shape {base.shape}')

    queries = []
    neighbors = []

    with open(tests_path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            query = row.get('query')
            closest_ids = row.get('closest_ids')
            if query is None or closest_ids is None:
                raise RuntimeError(
                    f'Missing query/closest_ids fields in {tests_path} at line {line_number}'
                )
            queries.append(query)
            neighbors.append(closest_ids)

    if not queries:
        raise RuntimeError(f'No queries found in {tests_path}')

    queries_array = np.asarray(queries, dtype=np.float32)
    neighbors_array = np.asarray(neighbors, dtype=np.int32)

    if queries_array.ndim != 2:
        raise RuntimeError(f'Expected 2D queries array, got shape {queries_array.shape}')
    if queries_array.shape[1] != base.shape[1]:
        raise RuntimeError(
            f'Base/query dimension mismatch: {base.shape[1]} != {queries_array.shape[1]}'
        )
    if neighbors_array.ndim != 2:
        raise RuntimeError(f'Expected 2D neighbor ids array, got shape {neighbors_array.shape}')

    temp_output_path = f'{output_path}.tmp'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    with h5py.File(temp_output_path, 'w') as handle:
        train_ds = handle.create_dataset('train', shape=base.shape, dtype=np.float32)
        chunk_rows = max(1, min(8192, base.shape[0]))
        for start in range(0, base.shape[0], chunk_rows):
            stop = min(start + chunk_rows, base.shape[0])
            train_ds[start:stop] = np.asarray(base[start:stop], dtype=np.float32)
        handle.create_dataset('test', data=queries_array, dtype=np.float32)
        handle.create_dataset('neighbors', data=neighbors_array, dtype=np.int32)

    os.replace(temp_output_path, output_path)


if __name__ == '__main__':
    main()
PY

    if [ ! -f "$DATASET_LOCAL_FILE" ]; then
        echo "Error: Failed to create local HDF5 dataset file: $DATASET_LOCAL_FILE"
        exit 1
    fi
}

derive_ivf_benchmark_parameters() {
    mapfile -t derived_values < <("$PYTHON_EXECUTABLE" - "$DATASET_LOCAL_FILE" "$BASE_LIMIT" "$QUERY_COUNT" "$NUM_CLUSTERS" "$TRAIN_COUNT" "$NPROBE" <<'PY'
import math
import sys

import h5py


def main() -> None:
    dataset_path, base_limit_arg, query_count_arg, num_clusters_arg, train_count_arg, nprobe_arg = sys.argv[1:7]
    base_limit = int(base_limit_arg)
    requested_query_count = int(query_count_arg)
    requested_num_clusters = int(num_clusters_arg)
    requested_train_count = int(train_count_arg)
    requested_nprobe = int(nprobe_arg)

    with h5py.File(dataset_path, 'r') as handle:
        total_base = int(handle['train'].shape[0])
        total_queries = int(handle['test'].shape[0])

    effective_base_limit = base_limit if base_limit > 0 else total_base
    base_count = min(effective_base_limit, total_base)
    query_count = requested_query_count if requested_query_count > 0 else total_queries
    query_count = min(query_count, total_queries)
    num_clusters = requested_num_clusters if requested_num_clusters > 0 else max(1, math.isqrt(base_count))
    nprobe = requested_nprobe if requested_nprobe > 0 else 16
    nprobe = min(nprobe, num_clusters)
    default_train_count = max(num_clusters, math.ceil(base_count * 0.10))
    train_count = requested_train_count if requested_train_count > 0 else default_train_count
    train_count = min(train_count, base_count)

    print(base_count)
    print(query_count)
    print(num_clusters)
    print(nprobe)
    print(train_count)


if __name__ == '__main__':
    main()
PY
)

    RESOLVED_BASE_COUNT="${derived_values[0]}"
    RESOLVED_QUERY_COUNT="${derived_values[1]}"
    RESOLVED_NUM_CLUSTERS="${derived_values[2]}"
    RESOLVED_NPROBE="${derived_values[3]}"
    RESOLVED_TRAIN_COUNT="${derived_values[4]}"
}

check_required_commands

numa_command_prefix() {
    if [ "$USE_NUMACTL_BINDING" != "1" ]; then
        return 0
    fi
    if ! command -v numactl >/dev/null 2>&1; then
        echo "Warning: numactl not found; running without NUMA binding." >&2
        return 0
    fi
    printf '%s\n' "numactl" "--cpunodebind=${NUMA_CPU_NODE}" "--membind=${NUMA_MEM_NODE}"
}

calico_specialization_for_config() {
    local config_name="${1:-}"
    local forced_specialization="${CALICO_SPECIALIZATION_OVERRIDE:-}"
    if [ -n "$forced_specialization" ]; then
        printf '%s\n' "$forced_specialization"
        return 0
    fi
    case "$config_name" in
        calico|pdx_calico|pdx_calico_*)
            printf '%s\n' "calico"
            ;;
        *)
            printf '%s\n' ""
            ;;
    esac
}

read_cached_build_signature() {
    local signature_file="$1"
    if [ -f "$signature_file" ]; then
        tr -d '\r' < "$signature_file"
    fi
}

write_cached_build_signature() {
    local signature_file="$1"
    local build_signature="$2"
    printf '%s\n' "$build_signature" > "$signature_file"
}

predicache_extension_present() {
    compgen -G "${SCRIPT_DIR}/predictivetranslation-sigmod2026/predicache_py*.so" > /dev/null
}

calico_extension_present() {
    compgen -G "${SCRIPT_DIR}/calico_py*.so" > /dev/null
}

maybe_build_native_binaries() {
    if [ "${BUILD_PROJECT_BINARIES:-0}" != "1" ]; then
        echo "Skipping top-level make build; exp_ivf.sh uses Python extension modules directly. Set BUILD_PROJECT_BINARIES=1 to enable it."
        echo ""
        return 0
    fi

    if [ "${CLEAN_NATIVE_BUILD:-0}" = "1" ]; then
        echo "Cleaning native project binaries before build..."
        make clean
    fi

    echo "Building native project binaries..."
    make -j"$(nproc)"
    echo "Native binary build complete"
    echo ""
}

ensure_predicache_python_library() {
    local pt_enabled="${1:-1}"
    local config_name="${2:-predicache_inmem}"
    local mode_label="enabled"
    local build_signature="${pt_enabled}:${DATASET_PAGE_SIZE_BYTES}:${PREDICACHE_DEBUG_BUILD:-0}:${PREDICACHE_LOCK_DEBUG:-0}:${PREDICACHE_ENABLE_AUTOVEC:-1}:${PREDICACHE_USE_URING:-0}"
    local cached_signature=""
    if [ "$pt_enabled" = "0" ]; then
        mode_label="disabled"
    fi

    if [ "$CURRENT_PREDICACHE_BUILD_SIGNATURE" = "$build_signature" ]; then
        return 0
    fi

    cached_signature="$(read_cached_build_signature "$PREDICACHE_BUILD_SIGNATURE_FILE")"

    local import_block
    import_block="$(block_path_for_config "$config_name")"
    ensure_block_file "$import_block" "32G"

    if [ "$cached_signature" = "$build_signature" ] && predicache_extension_present; then
        if BLOCK="$import_block" PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
            "$PYTHON_EXECUTABLE" -c "import predicache_py; print(predicache_py.__name__)" >/dev/null 2>&1; then
            echo "Reusing predicache python library (predictive translation ${mode_label}, page size ${DATASET_PAGE_SIZE_BYTES})..."
            CURRENT_PREDICACHE_BUILD_SIGNATURE="$build_signature"
            return 0
        fi
        echo "Cached predicache build failed import validation; rebuilding..."
    fi

    echo "Building predicache python library (predictive translation ${mode_label}, page size ${DATASET_PAGE_SIZE_BYTES})..."
    pushd "${SCRIPT_DIR}/predictivetranslation-sigmod2026" > /dev/null
    PREDICACHE_ENABLE_PREDICTIVE_TRANSLATION="$pt_enabled" \
        PREDICACHE_PAGE_SIZE="$DATASET_PAGE_SIZE_BYTES" \
        PREDICACHE_USE_URING="${PREDICACHE_USE_URING:-0}" \
        "$PYTHON_EXECUTABLE" setup.py build_ext --inplace --force
    popd > /dev/null

    if ! BLOCK="$import_block" PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
        "$PYTHON_EXECUTABLE" -c "import predicache_py; print(predicache_py.__name__)" >/dev/null 2>&1; then
        echo "Error: Failed to import predicache_py after build."
        echo "Check build logs in ${SCRIPT_DIR}/predictivetranslation-sigmod2026, PYTHONPATH, BLOCK=${import_block}, PREDICACHE_ENABLE_PREDICTIVE_TRANSLATION=${pt_enabled}, and PREDICACHE_PAGE_SIZE=${DATASET_PAGE_SIZE_BYTES}."
        exit 1
    fi

    write_cached_build_signature "$PREDICACHE_BUILD_SIGNATURE_FILE" "$build_signature"
    CURRENT_PREDICACHE_BUILD_SIGNATURE="$build_signature"
}

ensure_calico_python_library() {
    local config_name="${1:-vmcache}"
    local specialization
    specialization="$(calico_specialization_for_config "$config_name")"
    local build_signature="${DATASET_PAGE_SIZE_BYTES}:${specialization}:${CALICO_OPT_LEVEL:-3}:${CALICO_ENABLE_AUTOVEC:-1}:${CALICO_HNSW_DISABLE_OPTIMISTIC_READ:-0}"
    local cached_signature=""

    if [ "$CURRENT_CALICO_BUILD_SIGNATURE" = "$build_signature" ]; then
        return 0
    fi

    cached_signature="$(read_cached_build_signature "$CALICO_BUILD_SIGNATURE_FILE")"

    if [ "$cached_signature" = "$build_signature" ] && calico_extension_present; then
        if PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
            "$PYTHON_EXECUTABLE" -c "import calico_py; print(calico_py.__name__)" >/dev/null 2>&1; then
            if [ -n "$specialization" ]; then
                echo "Reusing calico python library (page size ${DATASET_PAGE_SIZE_BYTES}, specialization ${specialization})..."
            else
                echo "Reusing calico python library (page size ${DATASET_PAGE_SIZE_BYTES}, specialization general)..."
            fi
            CURRENT_CALICO_BUILD_SIGNATURE="$build_signature"
            return 0
        fi
        echo "Cached calico build failed import validation; rebuilding..."
    fi

    if [ -n "$specialization" ]; then
        echo "Building calico python library (page size ${DATASET_PAGE_SIZE_BYTES}, specialization ${specialization})..."
    else
        echo "Building calico python library (page size ${DATASET_PAGE_SIZE_BYTES}, specialization general)..."
    fi
    pushd "${SCRIPT_DIR}" > /dev/null
    CALICO_PAGE_SIZE="$DATASET_PAGE_SIZE_BYTES" CALICO_SPECIALIZATION="$specialization" "$PYTHON_EXECUTABLE" setup.py build_ext --inplace --force
    popd > /dev/null
    if ! PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}:${PYTHONPATH}" \
        "$PYTHON_EXECUTABLE" -c "import calico_py; print(calico_py.__name__)" >/dev/null 2>&1; then
        echo "Error: Failed to import calico_py after build."
        echo "Check build logs in ${SCRIPT_DIR}, PYTHONPATH, CALICO_PAGE_SIZE=${DATASET_PAGE_SIZE_BYTES}, and CALICO_SPECIALIZATION=${specialization}."
        exit 1
    fi

    write_cached_build_signature "$CALICO_BUILD_SIGNATURE_FILE" "$build_signature"
    CURRENT_CALICO_BUILD_SIGNATURE="$build_signature"
}

build_signature_for_config() {
    local config_name="${1:-}"
    local library="${2:-}"
    local pt_enabled="${3:-1}"

    if [ "$library" = "predicache" ]; then
        printf '%s\n' "predicache:${pt_enabled}:${DATASET_PAGE_SIZE_BYTES}:${PREDICACHE_DEBUG_BUILD:-0}:${PREDICACHE_LOCK_DEBUG:-0}:${PREDICACHE_ENABLE_AUTOVEC:-1}"
        return 0
    fi

    local specialization
    specialization="$(calico_specialization_for_config "$config_name")"
    printf '%s\n' "calico:${DATASET_PAGE_SIZE_BYTES}:${specialization}:${CALICO_OPT_LEVEL:-3}:${CALICO_ENABLE_AUTOVEC:-1}:${CALICO_HNSW_DISABLE_OPTIMISTIC_READ:-0}"
}

collect_config_build_signatures() {
    local -a signatures=()
    local config_spec

    for config_spec in "${CONFIGS[@]}"; do
        IFS=':' read -r config_name library _traditional _tradhash _nohugepage _prefetch pt_enabled <<< "$config_spec"
        local signature
        local seen_signature=0
        signature="$(build_signature_for_config "$config_name" "$library" "$pt_enabled")"
        for existing_signature in "${signatures[@]}"; do
            if [ "$existing_signature" = "$signature" ]; then
                seen_signature=1
                break
            fi
        done
        if [ "$seen_signature" -eq 0 ]; then
            signatures+=("$signature")
        fi
    done

    printf '%s\n' "${signatures[@]}"
}

clamp_thread_count() {
    local value="${1:-1}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        value=1
    fi
    if [ "$value" -lt 1 ]; then
        value=1
    elif [ "$value" -gt 128 ]; then
        value=128
    fi
    printf '%s\n' "$value"
}

RESULTS_BASE_DIR=""

CONFIGS=()

config_name_selected() {
    local config_name="$1"
    local focus_name
    local trimmed_focus_name

    if [ -z "$FOCUS_CONFIGS_OVERRIDE" ]; then
        return 0
    fi

    IFS=',' read -r -a FOCUS_CONFIG_NAMES <<< "$FOCUS_CONFIGS_OVERRIDE"
    for focus_name in "${FOCUS_CONFIG_NAMES[@]}"; do
        trimmed_focus_name="${focus_name//[[:space:]]/}"
        if [ -n "$trimmed_focus_name" ] && [ "$config_name" = "$trimmed_focus_name" ]; then
            return 0
        fi
    done

    return 1
}

initialize_configs() {
    local config_spec
    local config_name

    CONFIGS=()
    for config_spec in "${BASE_CONFIGS[@]}"; do
        IFS=':' read -r config_name _library _traditional _tradhash _nohugepage _prefetch _pt_enabled <<< "$config_spec"
        if config_name_selected "$config_name"; then
            CONFIGS+=("$config_spec")
        fi
    done

    if [ "${#CONFIGS[@]}" -eq 0 ]; then
        echo "Error: No IVF configs selected. Check FOCUS_CONFIGS=$FOCUS_CONFIGS_OVERRIDE."
        exit 1
    fi
}

run_benchmark() {
    local config_name=$1
    local library=$2
    local traditional=$3
    local tradhash=$4
    local nohugepage=$5
    local prefetch=$6
    local pt_enabled=$7
    local threads
    threads="$(clamp_thread_count "$8")"
    local output_dir=$9

    if [ -n "$PREFETCH_OVERRIDE_VALUE" ]; then
        prefetch="$PREFETCH_OVERRIDE_VALUE"
    else
        prefetch="$ACTIVE_PREFETCH_ENABLED"
    fi

    echo "=========================================="
    echo "Running IVF: $config_name with $threads threads"
    echo "LIBRARY=$library TRADITIONAL=$traditional TRADHASH=$tradhash"
    echo "NOHUGEPAGE_FRAME_MEM=$nohugepage PREFETCH=$prefetch"
    echo "USE_IOURING=${USE_IOURING}"
    echo "PDX_ASYNC_PREFETCH_WINDOW=${PDX_ASYNC_PREFETCH_WINDOW}"
    echo "CALICO_WARMUP_RECOVERED_INDEX=${CALICO_WARMUP_RECOVERED_INDEX}"
    echo "CALICO_DISABLE_PERF=${CALICO_DISABLE_PERF} CALICO_ENABLE_SEARCH_STATS=${CALICO_ENABLE_SEARCH_STATS:-0}"
    echo "=========================================="

    export TRADITIONAL=$traditional
    export TRADHASH=$tradhash
    export NOHUGEPAGE_FRAME_MEM=$nohugepage
    export PREFETCH=$prefetch
    if [ -n "$USER_USE_IOURING_OVERRIDE" ]; then
        export USE_IOURING="$USER_USE_IOURING_OVERRIDE"
    else
        export USE_IOURING=0
    fi
    if [ -n "$USER_CALICO_WARMUP_RECOVERED_INDEX_OVERRIDE" ]; then
        export CALICO_WARMUP_RECOVERED_INDEX="$USER_CALICO_WARMUP_RECOVERED_INDEX_OVERRIDE"
    else
        export CALICO_WARMUP_RECOVERED_INDEX=1
    fi

    if [ "$library" = "predicache" ]; then
        ensure_predicache_python_library "$pt_enabled" "$config_name"
    else
        ensure_calico_python_library "$config_name"
    fi

    mkdir -p "$output_dir"

    local output_file_suffix="${config_name}_threads${threads}"
    local output_file="${output_dir}/${output_file_suffix}.log"
    local membw_file="${output_dir}/${output_file_suffix}.membw"
    local run_output_base="${output_dir}/${output_file_suffix}"
    local build_threads="${BUILD_THREADS_OVERRIDE:-$threads}"
    build_threads="$(clamp_thread_count "$build_threads")"
    local block_path
    block_path="$(block_path_for_config "$config_name")"

    # if [ -e "$block_path" ]; then
    #     echo "Removing existing IVF block file before rebuild: $block_path"
    #     rm -f "$block_path"
    # fi
    ensure_block_file "$block_path" "32G"

    local membw_pid=""
    membw_pid=$(start_memory_bandwidth_monitor "$membw_file" || true)

    local -a CMD_PREFIX=()
    mapfile -t CMD_PREFIX < <(numa_command_prefix)

    {
        echo "Buffer pool page size build: ${DATASET_PAGE_SIZE_BYTES}"
        echo "Buffer pool page size env: IVF_PAGE_SIZE_BYTES=${DATASET_PAGE_SIZE_BYTES}"
    } > "$output_file"

    if [ "$library" = "predicache" ]; then
        {
            echo "PrediCache predictive translation build mode: ${pt_enabled}"
            echo "PrediCache predictive translation macro: PREDICACHE_ENABLE_PREDICTIVE_TRANSLATION=${pt_enabled}"
        } >> "$output_file"
    else
        {
            echo "Calico specialization build mode: $(calico_specialization_for_config "$config_name")"
            echo "Calico specialization macro env: CALICO_SPECIALIZATION=$(calico_specialization_for_config "$config_name")"
        } >> "$output_file"
    fi

    set +e
    set -o pipefail
    BLOCK="$block_path" IVF_PAGE_SIZE_BYTES="$DATASET_PAGE_SIZE_BYTES" CALICO_RUN_OUTPUT_BASE="$run_output_base" CALICO_RUN_LABEL="$output_file_suffix" CALICO_IVF_EXPERIMENT_MODE="$ACTIVE_EXPERIMENT_MODE" CALICO_WARMUP_RECOVERED_INDEX="${CALICO_WARMUP_RECOVERED_INDEX:-1}" "${CMD_PREFIX[@]}" "$PYTHON_EXECUTABLE" -u benchmark_ivf.py \
        --config "$config_name" \
        --library "$library" \
        --threads "$threads" \
        --build_threads "$build_threads" \
        --dataset "$DATASET" \
        --data_file "$DATASET_LOCAL_FILE" \
        --dataset_url "$DATASET_URL" \
        --base_limit "$BASE_LIMIT" \
        --query_count "$RESOLVED_QUERY_COUNT" \
        --train_count "$RESOLVED_TRAIN_COUNT" \
        --kmeans_iters "$KMEANS_ITERS" \
        --num_clusters "$RESOLVED_NUM_CLUSTERS" \
        --nprobe "$RESOLVED_NPROBE" \
        --k "$K" \
        2>&1 | tee -a "$output_file"
    local run_status=${PIPESTATUS[0]}
    set +o pipefail
    set -e

    stop_background_monitor "$membw_pid" "memory bandwidth" "$membw_file"

    if [ "$run_status" -ne 0 ]; then
        echo "Error: IVF benchmark failed for $config_name with $threads threads (exit=$run_status)."
        return "$run_status"
    fi

    echo "Results saved to: $output_file"
    echo "Perf outputs saved to: ${output_dir}/perf"
    echo ""
}

main() {
    maybe_build_native_binaries
    initialize_configs

    local experiment_mode
    local reset_block_files
    if [ -n "$RESET_BLOCK_FILES_OVERRIDE" ]; then
        reset_block_files="$RESET_BLOCK_FILES_OVERRIDE"
    else
        reset_block_files=0
    fi


    # for experiment_mode in in-memory; do
    # for experiment_mode in out-of-memory; do
    for experiment_mode in in-memory ; do
        for dataset_name in "${DATASETS_TO_RUN[@]}"; do
            DATASET="$dataset_name"
            configure_dataset
            apply_dataset_buffer_pool_size
            check_required_commands

            echo "Ensuring local IVF dataset is available..."
            ensure_ivf_dataset_available
            derive_ivf_benchmark_parameters

            configure_experiment_mode "$experiment_mode"

            echo "========================================"
            echo "IVF Benchmark Experiment"
            echo "========================================"
            echo "Dataset Label: $DATASET"
            echo "Experiment Mode: ${ACTIVE_EXPERIMENT_MODE}"
            echo "Dataset URL: $DATASET_URL"
            echo "Local Dataset File: $DATASET_LOCAL_FILE"
            echo "Buffer Pool Page Size: ${DATASET_PAGE_SIZE_BYTES} bytes"
            echo "IVF Parameters: num_clusters=$RESOLVED_NUM_CLUSTERS nprobe=$RESOLVED_NPROBE k=$K kmeans_iters=$KMEANS_ITERS"
            echo "Base Limit: $BASE_LIMIT"
            echo "Effective Base Count: $RESOLVED_BASE_COUNT"
            echo "Query Count: $RESOLVED_QUERY_COUNT"
            echo "Training Count: $RESOLVED_TRAIN_COUNT"
            echo "Physical Memory: ${PHYSGB}"
            echo "Buffer Pool Size: ${POOLGB}"
            echo "PDX Async Prefetch Window: ${PDX_ASYNC_PREFETCH_WINDOW}"
            echo "Warmup Recovered Index: ${CALICO_WARMUP_RECOVERED_INDEX}"
            echo "Results Dataset Directory: ${ACTIVE_DATASET_RESULTS_DIR_NAME}"
            echo "Thread Counts: ${THREAD_COUNTS[*]}"
            echo "Selected Configs: ${CONFIGS[*]}"
            echo "Perf Enabled: $([ "$CALICO_DISABLE_PERF" = "0" ] && printf '%s' yes || printf '%s' no)"
            echo "Perf Event Group Preset: ${CALICO_PERF_EVENT_GROUP_PRESET:-default}"
            echo "Search Stats Enabled: $([ -n "${CALICO_ENABLE_SEARCH_STATS:-}" ] && printf '%s' yes || printf '%s' no)"
            if [ -n "$PREFETCH_OVERRIDE_VALUE" ]; then
                echo "Prefetch Override: ${PREFETCH_OVERRIDE_VALUE}"
            fi
            if [ -n "$PREFETCH_WINDOW_OVERRIDE_VALUE" ]; then
                echo "Prefetch Window Override: ${PREFETCH_WINDOW_OVERRIDE_VALUE}"
            fi
            if [ "${CALICO_CAPTURE_PFN_AFTER_WARMUP:-0}" = "1" ]; then
                echo "PFN Probe After Warmup: enabled"
            fi
            if [ "$USE_NUMACTL_BINDING" = "1" ]; then
                echo "NUMA Policy: cpunodebind=${NUMA_CPU_NODE}, membind=${NUMA_MEM_NODE}"
            else
                echo "NUMA Policy: disabled"
            fi
            echo "========================================"
            echo ""
            echo "Dataset ready: $DATASET_LOCAL_FILE"
            echo ""

            if [ "$reset_block_files" = "1" ]; then
                echo "Cleaning reusable IVF artifacts before the first run for ${DATASET} in ${ACTIVE_EXPERIMENT_MODE} mode..."
                cleanup_reusable_index_files
                echo "Cleanup complete"
                echo ""
            else
                echo "Reusing existing IVF heapfiles for ${DATASET} in ${ACTIVE_EXPERIMENT_MODE} mode; set RESET_BLOCK_FILES=1 to force a rebuild."
                echo ""
            fi

            for config_spec in "${CONFIGS[@]}"; do
                IFS=':' read -r config_name library traditional tradhash nohugepage prefetch pt_enabled <<< "$config_spec"

                echo "========================================"
                echo "Testing IVF config ${config_name} on ${DATASET}"
                echo "Experiment Mode: ${ACTIVE_EXPERIMENT_MODE}"
                echo "Thread Counts: ${THREAD_COUNTS[*]}"
                echo "========================================"

                for threads in "${THREAD_COUNTS[@]}"; do
                    results_dir="${RESULTS_BASE_DIR}/${threads}"
                    mkdir -p "$results_dir"
                    run_benchmark "$config_name" "$library" "$traditional" "$tradhash" "$nohugepage" "$prefetch" "$pt_enabled" "$threads" "$results_dir"
                done

                echo ""
            done

            echo "========================================"
            echo "Completed IVF experiments for ${DATASET} (${ACTIVE_EXPERIMENT_MODE})"
            echo "Results directory: $RESULTS_BASE_DIR"
            echo "========================================"
            echo ""

            echo "Extracting IVF results for ${DATASET} (${ACTIVE_EXPERIMENT_MODE})..."
            RESULTS_ROOT_DIR="${SCRIPT_DIR}/Calico-VLDB26/data/exps/${ACTIVE_EXPERIMENT_MODE}/ivf" ./extract_results_ivf.sh "$ACTIVE_DATASET_RESULTS_DIR_NAME"
        done
    done

    echo "========================================"
    echo "All IVF experiments completed for datasets: ${DATASETS_TO_RUN[*]}"
    echo "========================================"
}

main "$@"
