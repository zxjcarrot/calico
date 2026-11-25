#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${CALICO_SECTION3_RESULTS_ROOT:-${CALICO_RESULTS_DIR:-${SOURCE_DIR}/exps}/in-memory/section3}"
BIN_DIR="${CALICO_SECTION3_BIN_DIR:-${SOURCE_DIR}/build/section3}"
BLOCK_FILE="${CALICO_SECTION3_BLOCK:-${BIN_DIR}/section3.heapfile}"
MODE="${1:-full}"
ONLY="${2:-all}"
RUN_ID="${CALICO_SECTION3_RUN_ID:-robinhood_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
    echo "Usage: $0 [full|smoke] [all|btree|graph|olap]" >&2
    exit 2
fi
if [[ ! " all btree graph olap " =~ " ${ONLY} " ]]; then
    echo "Unknown workload selection: ${ONLY}" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}" "${BIN_DIR}" "$(dirname "${BLOCK_FILE}")"
exec > >(tee -a "${RUN_DIR}/runner.log") 2>&1

echo "run_id=${RUN_ID}"
echo "mode=${MODE}"
echo "workloads=${ONLY}"
echo "source_dir=${SOURCE_DIR}"
echo "results=${RUN_DIR}"

if sudo -n true 2>/dev/null; then
    sudo sysctl -w kernel.perf_event_paranoid=-1
fi

mkdir -p "${RUN_DIR}/source_snapshot/section3_repro"
cp "${SOURCE_DIR}/Makefile" "${SOURCE_DIR}/calico.cpp" "${SOURCE_DIR}/calico.hpp" \
    "${SOURCE_DIR}/calico_oltp_query.cpp" "${SOURCE_DIR}/calico_olap_query.cpp" \
    "${SOURCE_DIR}/hash_tables.hpp" "${SOURCE_DIR}/third_party/robin_hood.h" \
    "${RUN_DIR}/source_snapshot/"
cp "${SCRIPT_DIR}/run_robinhood_section3.sh" \
    "${SCRIPT_DIR}/summarize_robinhood_section3.py" \
    "${SCRIPT_DIR}/merge_robinhood_with_retained.py" \
    "${RUN_DIR}/source_snapshot/section3_repro/"

metadata_token="$(curl -fsS -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token 2>/dev/null || true)"
if [[ -n "${metadata_token}" ]]; then
    {
        for metadata_path in instance-id instance-type placement/availability-zone; do
            printf '%s=' "${metadata_path//\//_}"
            curl -fsS -H "X-aws-ec2-metadata-token: ${metadata_token}" \
                "http://169.254.169.254/latest/meta-data/${metadata_path}"
            printf '\n'
        done
    } > "${RUN_DIR}/aws_instance_metadata.txt"
fi

{
    date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
    uname -a
    lscpu
    free -h
    df -hT / "$(dirname "${BLOCK_FILE}")"
    g++ --version
    perf --version
    printf 'perf_event_paranoid='; cat /proc/sys/kernel/perf_event_paranoid
    printf 'transparent_hugepage='; cat /sys/kernel/mm/transparent_hugepage/enabled
    sha256sum "${SOURCE_DIR}/calico.cpp" "${SOURCE_DIR}/calico.hpp" \
        "${SOURCE_DIR}/calico_oltp_query.cpp" "${SOURCE_DIR}/calico_olap_query.cpp" \
        "${SOURCE_DIR}/hash_tables.hpp" "${SOURCE_DIR}/third_party/robin_hood.h"
} > "${RUN_DIR}/machine_and_source.txt"

COMMON_CXXFLAGS="-DNDEBUG -O3 -std=c++23 -g3 -fno-omit-frame-pointer -fnon-call-exceptions -fasynchronous-unwind-tables -march=native"

build_binaries() {
    echo "Building 4KB OLTP/graph binary"
    make -C "${SOURCE_DIR}" clean
    make -C "${SOURCE_DIR}" -j"$(nproc)" calico_oltp_query \
        CXXFLAGS="${COMMON_CXXFLAGS} -DCALICO_PAGE_SIZE=4096"
    cp "${SOURCE_DIR}/calico_oltp_query" "${BIN_DIR}/calico_oltp_query_4k"

    echo "Building 8KB OLAP binary"
    make -C "${SOURCE_DIR}" clean
    make -C "${SOURCE_DIR}" -j"$(nproc)" calico_olap_query \
        CXXFLAGS="${COMMON_CXXFLAGS} -DCALICO_PAGE_SIZE=8192"
    cp "${SOURCE_DIR}/calico_olap_query" "${BIN_DIR}/calico_olap_query_8k"

    sha256sum "${BIN_DIR}/calico_oltp_query_4k" "${BIN_DIR}/calico_olap_query_8k" \
        > "${RUN_DIR}/binary_checksums.txt"
}

build_binaries

if [[ "${MODE}" == "full" ]]; then
    truncate -s 64G "${BLOCK_FILE}"
    BTREE_DATASIZE=100000000
    BTREE_RUNFOR=60
    BTREE_VIRTGB=32
    BTREE_PHYSGB=24
    GRAPH_DATASIZE=5000000
    GRAPH_VIRTGB=48
    GRAPH_PHYSGB=24
    OLAP_NUMPAGES=6553600
    OLAP_SCANPAGES="${CALICO_SECTION3_OLAP_SCANPAGES:-655360}"
    OLAP_RUNFOR=60
    OLAP_VIRTGB=54
    OLAP_PHYSGB=54
    OLAP_ROW_SIZE_SPEC="${CALICO_SECTION3_ROW_SIZES:-128,256,512,1024,2048,4096}"
    read -r -a OLAP_ROW_SIZES <<< "${OLAP_ROW_SIZE_SPEC//,/ }"
else
    truncate -s 2G "${BLOCK_FILE}"
    BTREE_DATASIZE=100000
    BTREE_RUNFOR=5
    BTREE_VIRTGB=2
    BTREE_PHYSGB=1
    GRAPH_DATASIZE=10000
    GRAPH_VIRTGB=2
    GRAPH_PHYSGB=1
    OLAP_NUMPAGES=65536
    OLAP_SCANPAGES=6553
    OLAP_RUNFOR=5
    OLAP_VIRTGB=2
    OLAP_PHYSGB=1
    OLAP_ROW_SIZE_SPEC="${CALICO_SECTION3_ROW_SIZES:-1024}"
    read -r -a OLAP_ROW_SIZES <<< "${OLAP_ROW_SIZE_SPEC//,/ }"
fi

cat > "${RUN_DIR}/configuration.txt" <<EOF
translation=Robin Hood Hash
TRADITIONAL=1
TRADHASH=7
THREADS=1
PERF=1
frame_huge_pages=enabled
btree_page_size=4096
btree_records=${BTREE_DATASIZE}
btree_payload_bytes=120
btree_virt_gb=${BTREE_VIRTGB}
btree_phys_gb=${BTREE_PHYSGB}
btree_run_seconds=${BTREE_RUNFOR}
graph_page_size=4096
graph_nodes=${GRAPH_DATASIZE}
graph_start=0
graph_prefetch=0
graph_virt_gb=${GRAPH_VIRTGB}
graph_phys_gb=${GRAPH_PHYSGB}
olap_page_size=8192
olap_row_sizes=${OLAP_ROW_SIZES[*]}
olap_total_pages=${OLAP_NUMPAGES}
olap_scan_pages=${OLAP_SCANPAGES}
olap_virt_gb=${OLAP_VIRTGB}
olap_phys_gb=${OLAP_PHYSGB}
olap_run_seconds=${OLAP_RUNFOR}
EOF

run_btree() {
    local dir="${RUN_DIR}/btree"
    mkdir -p "${dir}"
    echo "Running Robin Hood B-tree point lookup"
    /usr/bin/time -v env \
        BLOCK="${BLOCK_FILE}" VIRTGB="${BTREE_VIRTGB}" PHYSGB="${BTREE_PHYSGB}" \
        THREADS=1 DATASIZE="${BTREE_DATASIZE}" PAYLOAD_SIZE=120 RUNFOR="${BTREE_RUNFOR}" \
        RNDREAD=1 PERF=1 TRADITIONAL=1 TRADHASH=7 NOHUGEPAGE_FRAME_MEM=0 \
        TX_CSV="${dir}/calico_robinhood_hash_huge_tx.csv" \
        PERF_CSV="${dir}/calico_robinhood_hash_huge_perf.csv" \
        "${BIN_DIR}/calico_oltp_query_4k" > "${dir}/run.log" 2>&1
}

run_graph() {
    local dir="${RUN_DIR}/graph"
    mkdir -p "${dir}"
    echo "Running Robin Hood graph BFS"
    /usr/bin/time -v env \
        BLOCK="${BLOCK_FILE}" VIRTGB="${GRAPH_VIRTGB}" PHYSGB="${GRAPH_PHYSGB}" \
        THREADS=1 DATASIZE="${GRAPH_DATASIZE}" GRAPH=1 GRAPHSTART=0 RNDREAD=0 \
        PERF=1 PREFETCH=0 TRADITIONAL=1 TRADHASH=7 NOHUGEPAGE_FRAME_MEM=0 \
        TX_CSV="${dir}/calico_robinhood_hash_noprefetch_huge_tx.csv" \
        PERF_CSV="${dir}/calico_robinhood_hash_noprefetch_huge_perf.csv" \
        "${BIN_DIR}/calico_oltp_query_4k" > "${dir}/run.log" 2>&1
}

run_olap_mode() {
    local row_size="$1"
    local scan_mode="$2"
    local dir="${RUN_DIR}/olap/${row_size}bytes/${scan_mode}"
    mkdir -p "${dir}"
    echo "Running Robin Hood OLAP ${scan_mode} scan with ${row_size}-byte rows"
    /usr/bin/time -v env \
        BLOCK="${BLOCK_FILE}" VIRTGB="${OLAP_VIRTGB}" PHYSGB="${OLAP_PHYSGB}" \
        THREADS=1 NUMPAGES="${OLAP_NUMPAGES}" SCANPAGES="${OLAP_SCANPAGES}" \
        RUNFOR="${OLAP_RUNFOR}" ROWSIZE="${row_size}" OPTYPE=sum LAYOUT=row SCANMODE="${scan_mode}" \
        PERF=1 TRADITIONAL=1 TRADHASH=7 NOHUGEPAGE_FRAME_MEM=0 \
        OLAP_CSV="${dir}/robinhood_hash_sum_row_${scan_mode}_olap.csv" \
        PERF_CSV="${dir}/robinhood_hash_sum_row_${scan_mode}_perf.csv" \
        "${BIN_DIR}/calico_olap_query_8k" > "${dir}/run.log" 2>&1
}

if [[ "${ONLY}" == "all" || "${ONLY}" == "btree" ]]; then run_btree; fi
if [[ "${ONLY}" == "all" || "${ONLY}" == "graph" ]]; then run_graph; fi
if [[ "${ONLY}" == "all" || "${ONLY}" == "olap" ]]; then
    for row_size in "${OLAP_ROW_SIZES[@]}"; do
        run_olap_mode "${row_size}" sequential
        run_olap_mode "${row_size}" random
    done
fi

python3 "${SCRIPT_DIR}/summarize_robinhood_section3.py" "${RUN_DIR}"
echo "Completed ${RUN_ID}"
