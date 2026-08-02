#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
RESULTS_BASE_DIR="${CALICO_ABLATION_RESULTS_DIR:-${SCRIPT_DIR}/Calico-SIGMOD27/data/exps/in-memory/ablation_figure11_hash_then_array}"

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

  # Clean and rebuild bindings for this configuration
  echo "[ablation] Cleaning previous build artifacts..."
  (cd "${SCRIPT_DIR}" && rm -rf build calico_py*.so || true)
  echo "[ablation] Rebuilding Python bindings..."
  (cd "${SCRIPT_DIR}" && python3 setup.py build_ext --inplace)

  local out_dir="${RESULTS_BASE_DIR}/${name}"
  mkdir -p "$out_dir"
  local ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${out_dir}/${DATASET}_build${BUILD_THREADS}_search${SEARCH_THREADS}_${ts}.log"

  PY_ARGS=(
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

  (cd "${SCRIPT_DIR}" && python3 "${PY_ARGS[@]}") 2>&1 | tee "$log_file"
}

main() {
  echo "Ablation experiment on $DATASET (search_threads=${SEARCH_THREADS}, build_threads=${BUILD_THREADS})"
  mkdir -p "$RESULTS_BASE_DIR"

  # Remember original CALICO_HNSW_DISABLE_OPTIMISTIC_READ
  local orig_optread="$CALICO_HNSW_DISABLE_OPTIMISTIC_READ"

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

  echo "Ablation experiments completed. Results in $RESULTS_BASE_DIR"
}

main "$@"
