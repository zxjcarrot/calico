#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
RUN_ID="${CALICO_ABLATION_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_RESULTS_ROOT="${SCRIPT_DIR}/Calico-SIGMOD27/data/exps/in-memory/ablation_figure11_hash_then_array"
RESULTS_BASE_DIR="${CALICO_ABLATION_RESULTS_DIR:-${DEFAULT_RESULTS_ROOT}_${RUN_ID}}"
BLOCK_FILE="${BLOCK:-${TMPDIR:-/tmp}/calico-hash-then-array-figure11-${RUN_ID}.heapfile}"
HEAPFILE_SIZE="${CALICO_HEAPFILE_SIZE:-24G}"
LOCK_FILE="${CALICO_ABLATION_LOCK_FILE:-${TMPDIR:-/tmp}/calico-exp-hnsw-ablation.lock}"
SCREEN_NAME="${CALICO_ABLATION_SCREEN_NAME:-calico-figure11-${RUN_ID}}"
RUN_LOG="${CALICO_ABLATION_RUN_LOG:-${RESULTS_BASE_DIR}/ablation_runner.log}"
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: ./exp_hnsw_ablation.sh [--background|--foreground]

The default mode runs in the current shell. --background starts the same
experiment in a detached screen session and returns immediately.
EOF
}

MODE="foreground"
case "${1:-}" in
  "") ;;
  --foreground) ;;
  --background) MODE="background" ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$MODE" == "background" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "[ablation] --background requires screen." >&2
    exit 1
  fi
  if screen -list 2>/dev/null | grep -Fq ".${SCREEN_NAME}"; then
    echo "[ablation] Screen session already exists: ${SCREEN_NAME}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$RUN_LOG")"
  screen -dmS "$SCREEN_NAME" bash -c \
    "cd '$SCRIPT_DIR' && \
     env BLOCK='$BLOCK_FILE' CALICO_ABLATION_RESULTS_DIR='$RESULTS_BASE_DIR' \
     CALICO_ABLATION_RUN_ID='$RUN_ID' '$SCRIPT_DIR/exp_hnsw_ablation.sh' --foreground \
     > '$RUN_LOG' 2>&1; \
     status=\$?; printf '\\n[screen] exit=%s\\n' \"\$status\" >> '$RUN_LOG'; \
     exit \"\$status\""
  echo "[ablation] Started screen session: ${SCREEN_NAME}"
  echo "[ablation] Log: ${RUN_LOG}"
  echo "[ablation] Results: ${RESULTS_BASE_DIR}"
  exit 0
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[ablation] Another HNSW ablation run is already using the build directory." >&2
  echo "[ablation] Lock: ${LOCK_FILE}" >&2
  exit 1
fi

prepare_heapfile() {
  if [[ ! -e "$BLOCK_FILE" ]]; then
    echo "[ablation] Creating sparse heapfile: ${BLOCK_FILE} (${HEAPFILE_SIZE})"
    truncate -s "$HEAPFILE_SIZE" "$BLOCK_FILE"
  fi
  export BLOCK="$BLOCK_FILE"
}

DATASET="deep10M"
#DATASET="sift1m"
M=16
EF_CONSTRUCTION=100
K=10
EF_SEARCH=50
PHYSMEM=16G
VIRTMEM=24G
BUILD_THREADS=64
SEARCH_THREADS=1

# Ablation stages from Figure 11 in main_revised.tex.
# Fields: name:traditional:tradhash:nohuge_frame:nohuge_trans:optread_disabled:prefetch:path_cache:specialization
AB_STAGES=(
  "1_hash_only:1:2:1:1:1:0:0:hash"                              # Open-address hash baseline
  "2_array_index:1:0:1:1:1:0:0:calico"                         # + direct array indexing
  "3_hash_then_array:1:8:1:1:1:0:0:calico"                     # + Robin Hood top-level lookup
  "4_hash_then_array_path:1:8:1:1:1:0:1:calico"                # + thread-local path cache
  "5_hash_then_array_path_huge_frames:1:8:0:1:1:0:1:calico"    # + huge page frames
  "6_hash_then_array_path_huge_trans:1:8:0:0:1:0:1:calico"     # + huge translation-array pages
  "7_hash_then_array_path_optread:1:8:0:0:0:0:1:calico"       # + optimistic reads
  "8_hash_then_array_path_prefetch:1:8:0:0:0:1:1:calico"      # + group prefetch
)

run_stage() {
  local spec="$1"
  IFS=':' read -r name traditional tradhash nohuge_frame nohuge_trans optread_disabled prefetch path_cache specialization <<< "$spec"

  echo "==============================="
  echo "Stage: $name"
  echo "TRADITIONAL=$traditional TRADHASH=$tradhash"
  echo "NOHUGEPAGE_FRAME_MEM=$nohuge_frame NOHUGEPAGE_TRANSLATION_ARRAY=$nohuge_trans"
  echo "OPTREAD_DISABLED=$optread_disabled PREFETCH=$prefetch PATH_CACHE=$path_cache"
  echo "SPECIALIZATION=$specialization"
  echo "==============================="

  export TRADITIONAL="$traditional"
  export TRADHASH="$tradhash"
  export NOHUGEPAGE_FRAME_MEM="$nohuge_frame"
  export NOHUGEPAGE_TRANSLATION_ARRAY="$nohuge_trans"
  export PREFETCH="$prefetch"
  export PATH_CACHE="$path_cache"
  export PHYSGB="$PHYSMEM"
  export VIRTGB="$VIRTMEM"
  export CALICO_RUN_LABEL="$name"
  export CALICO_SPECIALIZATION="$specialization"
  # Control compile-time macro via env used by setup.py
  if [ "$optread_disabled" = "1" ]; then
    export CALICO_HNSW_DISABLE_OPTIMISTIC_READ=1
  else
    unset CALICO_HNSW_DISABLE_OPTIMISTIC_READ
  fi

  # Clean and rebuild bindings for this configuration. The lock acquired by
  # main prevents another ablation run from removing build/ concurrently.
  echo "[ablation] Cleaning previous build artifacts..."
  (cd "${SCRIPT_DIR}" && rm -rf build calico_py*.so && mkdir -p build)
  echo "[ablation] Rebuilding Python bindings..."
  if ! (cd "${SCRIPT_DIR}" && "$PYTHON" setup.py build_ext --inplace --force); then
    echo "[ablation] Build failed for ${name}; aborting." >&2
    return 1
  fi

  local out_dir="${RESULTS_BASE_DIR}/${name}"
  mkdir -p "$out_dir"
  local ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${out_dir}/${DATASET}_build${BUILD_THREADS}_search${SEARCH_THREADS}_${ts}.log"

  local -a PY_ARGS=(
    -u benchmark_calico_hnsw.py
    --config "calico"
    --threads "$SEARCH_THREADS"
    --build-threads "$BUILD_THREADS"
    --dataset "$DATASET"
    --M "$M"
    --ef_construction "$EF_CONSTRUCTION"
    --k "$K"
    --ef_search "$EF_SEARCH"
  )

  if ! (cd "${SCRIPT_DIR}" && "$PYTHON" "${PY_ARGS[@]}") 2>&1 | tee "$log_file"; then
    echo "[ablation] Benchmark failed for ${name}; aborting." >&2
    return 1
  fi
}

main() {
  echo "Ablation experiment on $DATASET (search_threads=${SEARCH_THREADS}, build_threads=${BUILD_THREADS})"
  echo "Results directory: ${RESULTS_BASE_DIR}"
  echo "Heapfile: ${BLOCK_FILE}"
  prepare_heapfile
  mkdir -p "$RESULTS_BASE_DIR"

  # Remember original CALICO_HNSW_DISABLE_OPTIMISTIC_READ
  local orig_optread="${CALICO_HNSW_DISABLE_OPTIMISTIC_READ-}"

  for s in "${AB_STAGES[@]}"; do
    run_stage "$s"
    sleep 2
  done

  # Restore original env for optimistic read (macro control)
  if [ -n "$orig_optread" ]; then
    export CALICO_HNSW_DISABLE_OPTIMISTIC_READ="$orig_optread"
  else
    unset CALICO_HNSW_DISABLE_OPTIMISTIC_READ
  fi

  if [[ "${CALICO_SKIP_EXTRACT:-0}" != "1" ]]; then
    echo "[ablation] Extracting completed results..."
    if ! CALICO_ABLATION_RESULTS_DIR="$RESULTS_BASE_DIR" \
      "$SCRIPT_DIR/extract_results_hnsw_ablation.sh"; then
      echo "[ablation] Warning: result extraction failed." >&2
    fi
  fi

  echo "Ablation experiments completed. Results in $RESULTS_BASE_DIR"
}

main "$@"
