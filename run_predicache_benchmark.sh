#!/bin/bash
# Wrapper script to run PrediCache benchmarks with proper working directory

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/predictivetranslation-sigmod2026:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "========================================================================"
echo "SIFT1M Benchmark: PrediCache vs Calico vs hnswlib"
echo "========================================================================"

# Test with small subset first
echo ""
echo "Running PrediCache benchmark on SIFT1M..."
"${PYTHON_EXECUTABLE:-python3}" benchmark_calico_hnsw.py \
    --library predicache \
    --dataset sift1m \
    --threads 4 \
    --M 16 \
    --ef_construction 200 \
    --k 10 \
    --ef_search 50 \
    --config "PrediCache-InMemory-4T"

echo ""
echo "========================================================================"
echo "Benchmark complete!"
echo "========================================================================"
