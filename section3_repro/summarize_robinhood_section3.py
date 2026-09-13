#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


def perf_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if not row or row[0] == "label":
                continue
            if row[0].startswith("derived_"):
                if len(row) > 1 and row[1] not in ("", "n/a"):
                    values[row[0]] = float(row[1])
            elif len(row) > 2 and row[2] not in ("", "n/a"):
                values[row[0]] = float(row[2])
    return values


def labeled_csv_value(path: Path, label: str) -> float:
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if row and row[0] == label:
                return float(row[1])
    raise ValueError(f"{label} not found in {path}")


def write_csv(path: Path, header: list[str], row: list[object]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerow(row)


def read_configuration(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def summarize_btree(run_dir: Path) -> None:
    directory = run_dir / "btree"
    if not directory.exists():
        return
    perf = perf_values(directory / "calico_robinhood_hash_huge_perf.csv")
    throughput = labeled_csv_value(directory / "calico_robinhood_hash_huge_tx.csv", "avg_throughput")
    write_csv(
        directory / "robinhood_summary.csv",
        ["thread_count", "config_name", "avg_throughput", "llc_ref_per_op", "llc_miss_per_op",
         "dtlb_ref_per_op", "dtlb_miss_per_op", "cycles_per_op", "insts_per_op", "ipc"],
        [1, "calico_robinhood_hash_huge", throughput, perf["llc_refs"], perf["llc_misses"],
         perf["dtlb_refs"], perf["dtlb_misses"], perf["cycles"], perf["instructions"], perf["derived_ipc"]],
    )


def summarize_graph(run_dir: Path) -> None:
    directory = run_dir / "graph"
    if not directory.exists():
        return
    perf = perf_values(directory / "calico_robinhood_hash_noprefetch_huge_perf.csv")
    text = (directory / "calico_robinhood_hash_noprefetch_huge_tx.csv").read_text()
    match = re.search(r"nodes=(\d+).*visited=(\d+).*time_ms=([0-9.]+)", text)
    if not match:
        raise ValueError("Graph result line not found")
    nodes, _, time_ms = match.groups()
    write_csv(
        directory / "robinhood_summary.csv",
        ["ThreadCount", "Config", "BaseConfig", "Prefetch", "HugePages", "Nodes",
         "AvgTraversalTime_ms", "OverallTime_ms", "LLC_ref_per_op", "LLC_misses_per_op",
         "DTLB_ref_per_op", "DTLB_misses_per_op", "Cycles_per_op", "Instructions_per_op", "IPC"],
        [1, "calico_robinhood_hash_noprefetch_huge", "calico_robinhood_hash", "No", "Yes", nodes,
         time_ms, time_ms, perf["llc_refs"], perf["llc_misses"], perf["dtlb_refs"],
         perf["dtlb_misses"], perf["cycles"], perf["instructions"], perf["derived_ipc"]],
    )


def summarize_olap(run_dir: Path) -> None:
    root = run_dir / "olap"
    if not root.exists():
        return
    configuration = read_configuration(run_dir / "configuration.txt")
    scan_gib = (
        int(configuration["olap_scan_pages"])
        * int(configuration["olap_page_size"])
        / (1024 ** 3)
    )
    rows = []
    row_directories = []
    if (root / "sequential").exists():
        row_directories.append((1024, root))
    for directory in root.glob("*bytes"):
        row_directories.append((int(directory.name.removesuffix("bytes")), directory))
    for row_size, row_directory in sorted(row_directories):
        for scan_mode in ("sequential", "random"):
            directory = row_directory / scan_mode
            perf = perf_values(directory / f"robinhood_hash_sum_row_{scan_mode}_perf.csv")
            qps = labeled_csv_value(directory / f"robinhood_hash_sum_row_{scan_mode}_olap.csv", "avg_queries_per_sec")
            rows.append([
                row_size, "robinhood_hash", "sum", "row", scan_mode, qps, qps * scan_gib, 0, 0,
                perf["llc_refs"], perf["llc_misses"], perf["dtlb_refs"], perf["dtlb_misses"],
                perf["cycles"], perf["instructions"], perf["derived_ipc"],
            ])
    path = root / "robinhood_summary.csv"
    with path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["row_size_bytes", "config_name", "op_type", "layout", "scan_mode",
                         "avg_queries_per_sec", "throughput_gbps", "avg_read_mb", "avg_write_mb",
                         "llc_ref_per_op", "llc_miss_per_op", "dtlb_ref_per_op", "dtlb_miss_per_op",
                         "cycles_per_op", "insts_per_op", "ipc"])
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} RUN_DIR")
    run_dir = Path(sys.argv[1])
    summarize_btree(run_dir)
    summarize_graph(run_dir)
    summarize_olap(run_dir)


if __name__ == "__main__":
    main()
