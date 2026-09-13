#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-${DATASET:-all}}"
RESULTS_ROOT_DIR="${RESULTS_ROOT_DIR:-${SCRIPT_DIR}/Calico-VLDB26/data/exps/in-memory/ivf}"
LEGACY_OUTPUT_CSV="${RESULTS_ROOT_DIR}/ivf_results_summary.csv"
MEMORY_BW_METRIC="${MEMORY_BW_METRIC:-nps1_die_to_dram}"

DEFAULT_DATASETS=(gist1m sift1m msong dbpedia sift10M)
THREAD_COUNTS=(1 4 8 16 32 64 128)
CONFIGS=("pdx_calico" "pdx_vmcache" "pdx_openaddr" "pdx_robinhood" "pdx_predicache" "pdx_predicache_no_pt")

normalize_dataset_name() {
    case "$1" in
        dbpedia|dbpedia_openai_1m)
            printf '%s\n' "dbpedia_openai_1m"
            ;;
        gist1m|sift1m|msong|sift10m|sift10M)
            if [[ "$1" == "sift10m" ]]; then
                printf '%s\n' "sift10M"
            else
                printf '%s\n' "$1"
            fi
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

resolve_datasets_to_run() {
    if [[ "$DATASET" == "all" ]]; then
        DATASETS_TO_RUN=("${DEFAULT_DATASETS[@]}")
    else
        DATASETS_TO_RUN=("$DATASET")
    fi
}

extract_average_membw() {
    local membw_file=$1
    local metric_name=$2
    local default=${3:-"N/A"}

    if [[ ! -f "$membw_file" ]]; then
        echo "$default"
        return
    fi

    local avg
    avg=$(awk -F',' -v metric="$metric_name" '
        function trim(s) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }
        function isnum(s) { return (s ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) }
        {
            metric_idx = 0
            for (i = 1; i <= NF; i++) {
                f = trim($i)
                if (f ~ metric) {
                    metric_idx = i
                    break
                }
            }
            if (metric_idx > 0) {
                for (j = metric_idx - 1; j >= 1; j--) {
                    v = trim($j)
                    if (isnum(v)) {
                        sum += v
                        count++
                        break
                    }
                }
            }
        }
        END {
            if (count > 0) {
                printf "%.4f", sum / count
            }
        }
    ' "$membw_file")

    if [[ -z "$avg" ]]; then
        echo "$default"
    else
        echo "$avg"
    fi
}

placeholder_row() {
    local config=$1
    local threads=$2
    echo "$config,$threads,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
}

parse_log_file() {
    local log_file=$1
    local config=$2
    local threads=$3
    local membw_file="${log_file%.log}.membw"
    local avg_membw
    avg_membw=$(extract_average_membw "$membw_file" "$MEMORY_BW_METRIC" "N/A")

    python3 - "$log_file" "$config" "$threads" "$avg_membw" <<'PY'
import csv
import json
import pathlib
import sys

log_file = pathlib.Path(sys.argv[1])
config = sys.argv[2]
threads = sys.argv[3]
avg_membw = sys.argv[4]

row_defaults = {
    'Config': config,
    'Threads': threads,
    'Library': 'N/A',
    'Dataset': 'N/A',
    'BuildTime(s)': 'N/A',
    'SearchTime(s)': 'N/A',
    'QPS': 'N/A',
    'Recall': 'N/A',
    'AvgQueryLatencyMs': 'N/A',
    'DistComps': 'N/A',
    'ListsProbed': 'N/A',
    'VectorsScanned': 'N/A',
    'CacheMisses': 'N/A',
    'CacheRefs': 'N/A',
    'CacheMissRate(%)': 'N/A',
    'dTLBMisses': 'N/A',
    'LLCMisses': 'N/A',
    'Recovered': 'N/A',
    'BlockPath': 'N/A',
    'AvgMemBWMetric': avg_membw,
}

result = None
for line in log_file.read_text(encoding='utf-8', errors='replace').splitlines():
    if line.startswith('RESULT_JSON: '):
        payload = line[len('RESULT_JSON: '):]
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            result = None

if result is None:
    writer = csv.writer(sys.stdout)
    writer.writerow(row_defaults.values())
    sys.exit(0)

def value(key, default='N/A'):
    raw = result.get(key, default)
    if raw is None:
        return default
    return raw

row = {
    'Config': config,
    'Threads': threads,
    'Library': value('library'),
    'Dataset': value('dataset'),
    'BuildTime(s)': value('build_time'),
    'SearchTime(s)': value('search_time'),
    'QPS': value('qps'),
    'Recall': value('recall'),
    'AvgQueryLatencyMs': value('avg_query_latency_ms'),
    'DistComps': value('dist_comps'),
    'ListsProbed': value('lists_probed'),
    'VectorsScanned': value('vectors_scanned'),
    'CacheMisses': value('cache-misses'),
    'CacheRefs': value('cache-references'),
    'CacheMissRate(%)': value('cache_miss_rate_pct'),
    'dTLBMisses': value('dTLB-load-misses'),
    'LLCMisses': value('LLC-load-misses'),
    'Recovered': value('recovered'),
    'BlockPath': value('BLOCK'),
    'AvgMemBWMetric': avg_membw,
}

writer = csv.writer(sys.stdout)
writer.writerow(row.values())
PY
}

extract_dataset_results() {
    local dataset_input="$1"
    local dataset_name
    local results_base_dir
    local output_csv

    dataset_name="$(normalize_dataset_name "$dataset_input")"
    results_base_dir="${RESULTS_ROOT_DIR}/${dataset_name}"
    output_csv="${results_base_dir}/ivf_results_summary.csv"

    echo "Extracting IVF benchmark results from: $results_base_dir"
    echo ""

    if [[ ! -d "$results_base_dir" ]]; then
        echo "Error: Results directory not found: $results_base_dir"
        return 1
    fi

    mkdir -p "$results_base_dir"
    echo "Config,Threads,Library,Dataset,BuildTime(s),SearchTime(s),QPS,Recall,AvgQueryLatencyMs,DistComps,ListsProbed,VectorsScanned,CacheMisses,CacheRefs,CacheMissRate(%),dTLBMisses,LLCMisses,Recovered,BlockPath,AvgMemBWMetric" > "$output_csv"

    for threads in "${THREAD_COUNTS[@]}"; do
        echo "Processing results for $threads threads..."
        results_dir="${results_base_dir}/${threads}"

        if [[ ! -d "$results_dir" ]]; then
            echo "  Warning: Directory not found: $results_dir"
            continue
        fi

        for config in "${CONFIGS[@]}"; do
            log_file="${results_dir}/${config}_threads${threads}.log"

            if [[ ! -f "$log_file" ]]; then
                echo "  Warning: No log file found for $config with $threads threads"
                placeholder_row "$config" "$threads" >> "$output_csv"
            else
                parse_log_file "$log_file" "$config" "$threads" >> "$output_csv"
            fi
        done

        echo ""
    done

    echo "Results summary saved to: $output_csv"
    if [[ "$dataset_name" == "dbpedia_openai_1m" ]]; then
        cp "$output_csv" "$LEGACY_OUTPUT_CSV"
        echo "Compatibility summary updated: $LEGACY_OUTPUT_CSV"
    fi
    echo ""

    if command -v column >/dev/null 2>&1; then
        echo "Summary Table:"
        echo "=============================================="
        column -t -s',' "$output_csv" | head -20
        echo "... (see $output_csv for full results)"
    else
        echo "Install 'column' command for pretty table display"
        echo "Raw CSV content (first 10 rows):"
        head -10 "$output_csv"
    fi
}

main() {
    local dataset_name
    local status=0

    resolve_datasets_to_run

    for dataset_name in "${DATASETS_TO_RUN[@]}"; do
        if ! extract_dataset_results "$dataset_name"; then
            status=1
        fi
        echo ""
    done

    return "$status"
}

main "$@"
