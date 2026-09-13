#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


SUMMARY_SPECS = {
    "btree": {
        "retained": "btree_ycsbc_perf_analysis/results_summary_all.csv",
        "primary": "btree/robinhood_summary.csv",
        "keys": ("thread_count", "config_name"),
    },
    "graph": {
        "retained": "graph_bfs/graph_results_summary.csv",
        "primary": "graph/robinhood_summary.csv",
        "keys": ("ThreadCount", "Config"),
    },
    "olap": {
        "retained": "olap_row_perf_analysis/results_summary_all.csv",
        "primary": "olap/robinhood_summary.csv",
        "keys": ("row_size_bytes", "config_name", "op_type", "layout", "scan_mode"),
    },
}


def read_csv(path: Path):
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames, list(reader)


def row_key(row, columns):
    return tuple(row[column] for column in columns)


def merge_summary(name, spec, retained_root, primary_run, rows_run, include_primary_olap, output_dir):
    retained_path = retained_root / spec["retained"]
    retained_header, retained_rows = read_csv(retained_path)

    robin_paths = [primary_run / spec["primary"]]
    if name == "olap" and rows_run:
        robin_paths = [rows_run / "olap/robinhood_summary.csv"]
        if include_primary_olap:
            robin_paths.insert(0, primary_run / spec["primary"])

    robin_rows = []
    for robin_path in robin_paths:
        robin_header, rows = read_csv(robin_path)
        if robin_header != retained_header:
            raise ValueError(f"schema mismatch: {retained_path} vs. {robin_path}")
        robin_rows.extend(rows)

    keys = spec["keys"]
    existing = {row_key(row, keys) for row in retained_rows}
    added = set()
    for row in robin_rows:
        key = row_key(row, keys)
        if key in existing:
            raise ValueError(f"Robin Hood key already exists in retained data: {key}")
        if key in added:
            raise ValueError(f"duplicate Robin Hood key: {key}")
        added.add(key)

    output_path = output_dir / f"{name}_results_with_robinhood.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=retained_header)
        writer.writeheader()
        writer.writerows(retained_rows)
        writer.writerows(robin_rows)

    print(
        f"{name}: retained={len(retained_rows)} added={len(robin_rows)} "
        f"total={len(retained_rows) + len(robin_rows)} output={output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create augmented Section 3 summaries without modifying retained data."
    )
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--olap-rows-run", type=Path)
    parser.add_argument("--include-primary-olap", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name, spec in SUMMARY_SPECS.items():
        merge_summary(
            name,
            spec,
            args.retained_root,
            args.primary_run,
            args.olap_rows_run,
            args.include_primary_olap,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
