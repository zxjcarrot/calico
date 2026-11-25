import numpy as np
import time
import os
import sys
import re
import json
import struct
import subprocess
import signal
import shutil
import tempfile
import shlex
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('CALICO_DATA_DIR', SCRIPT_DIR / 'data')).expanduser().resolve()

# Resolve local modules from the repository root instead of the caller's cwd.
for path_entry in (
    str(SCRIPT_DIR),
    str(SCRIPT_DIR / 'predictivetranslation-sigmod2026'),
    str(SCRIPT_DIR / 'hnswlib'),
):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

if 'BLOCK' not in os.environ:
    default_block = SCRIPT_DIR / 'heapfile'
    if default_block.exists():
        os.environ['BLOCK'] = str(default_block)

try:
    import h5py  # For SIFT10M .mat dataset
except ImportError:
    h5py = None

# Conditionally import calico_py - only when needed for Calico benchmarks
# This avoids unnecessary memory allocation for hnswlib-only benchmarks
calico_py = None

# Conditionally import predicache_py - only when needed for PrediCache benchmarks
predicache_py = None

try:
    from usearch.index import Index, MetricKind, CompiledMetric
except ImportError:
    print("Warning: usearch library not found. Usearch benchmarks will not be available.")
    Index = None

import hnswlib

# Global cache events to profile
DEFAULT_CACHE_EVENTS = ['dTLB-load-misses', 'dTLB-loads', 'instructions', 'cycles', 'cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses']
CACHE_EVENTS = list(DEFAULT_CACHE_EVENTS)

PERF_EVENT_GROUP_PRESETS = {
    'default': [
        ('cache_tlb', list(DEFAULT_CACHE_EVENTS)),
    ],
    'amd_pdx_scaling': [
        ('cache_tlb', list(DEFAULT_CACHE_EVENTS)),
        (
            'numa_source',
            [
                'ls_dmnd_fills_from_sys.mem_io_local',
                'ls_dmnd_fills_from_sys.mem_io_remote',
                'ls_dmnd_fills_from_sys.ext_cache_local',
                'ls_dmnd_fills_from_sys.ext_cache_remote',
            ],
        ),
        (
            'dtlb_walks',
            [
                'dTLB-load-misses',
                'dTLB-loads',
                'ls_tablewalker.dside',
            ],
        ),
        (
            'load_pressure',
            [
                'ex_ret_instr',
                'ls_dispatch.ld_dispatch',
                'ls_dispatch.store_dispatch',
                'l2_cache_misses_from_dc_misses',
            ],
        ),
        (
            'sched_faults',
            [
                'context-switches',
                'cpu-migrations',
                'minor-faults',
                'major-faults',
            ],
        ),
    ],
}


def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        print(f"Warning: ignoring invalid {name}={value!r}; using {default}.")
        return default


DEFAULT_PERF_DELAY_SECONDS = _read_float_env("CALICO_PERF_DELAY", 0.0)
DEFAULT_PERF_DURATION_SECONDS = _read_float_env("CALICO_PERF_DURATION", 5.0, minimum=0.1)
_PERF_USABLE_CACHE: Optional[bool] = None
_PERF_COMMAND_PREFIX: Optional[tuple[str, ...]] = None


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    print(f"Warning: ignoring invalid {name}={value!r}; using {default}.")
    return default


def _perf_explicitly_disabled() -> bool:
    return _read_bool_env("CALICO_DISABLE_PERF", False)


def _perf_verbose_output_enabled() -> bool:
    return _read_bool_env("CALICO_VERBOSE_PERF", False)


def _perf_command_candidates() -> list[list[str]]:
    candidates = []
    if shutil.which("sudo") is not None:
        candidates.append(["sudo", "-n", "perf"])
    candidates.append(["perf"])
    return candidates


def _resolve_perf_command_prefix() -> Optional[list[str]]:
    global _PERF_USABLE_CACHE, _PERF_COMMAND_PREFIX

    if _PERF_USABLE_CACHE is not None:
        if _PERF_USABLE_CACHE and _PERF_COMMAND_PREFIX is not None:
            return list(_PERF_COMMAND_PREFIX)
        return None

    if _perf_explicitly_disabled():
        _PERF_USABLE_CACHE = False
        print("Note: perf profiling disabled by CALICO_DISABLE_PERF.")
        return None

    if sys.platform != "linux" or shutil.which("perf") is None:
        _PERF_USABLE_CACHE = False
        return None

    failure_details = []
    for candidate in _perf_command_candidates():
        try:
            probe = subprocess.run(
                candidate + ["stat", "-e", "cycles", "--", "sleep", "0.01"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            failure_details.append(f"{' '.join(candidate)}: {exc}")
            continue

        if probe.returncode == 0:
            _PERF_USABLE_CACHE = True
            _PERF_COMMAND_PREFIX = tuple(candidate)
            if candidate[0] == "sudo":
                print("Note: using passwordless sudo for perf profiling.")
            return list(candidate)

        stderr = (probe.stderr or "").strip().splitlines()
        details = stderr[0] if stderr else "perf events unavailable"
        failure_details.append(f"{' '.join(candidate)}: {details}")

    _PERF_USABLE_CACHE = False
    details = failure_details[0] if failure_details else "perf events unavailable"
    print(f"Note: perf profiling disabled on this host: {details}")
    return None


def _perf_is_usable() -> bool:
    return _resolve_perf_command_prefix() is not None


def sanitize_perf_counter_name(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_').lower()


def parse_perf_event_groups(default_events=None) -> list[tuple[str, list[str]]]:
    groups_env = os.environ.get('CALICO_PERF_EVENT_GROUPS', '').strip()
    preset_name = os.environ.get('CALICO_PERF_EVENT_GROUP_PRESET', 'default').strip().lower() or 'default'

    if groups_env:
        parsed_groups = []
        for group_index, raw_group in enumerate(groups_env.split(';'), start=1):
            raw_group = raw_group.strip()
            if not raw_group:
                continue
            if '=' in raw_group:
                group_name, raw_events = raw_group.split('=', 1)
            else:
                group_name = f'group_{group_index}'
                raw_events = raw_group
            events = [event.strip() for event in raw_events.split(',') if event.strip()]
            if events:
                parsed_groups.append((group_name.strip() or f'group_{group_index}', events))
        if parsed_groups:
            return parsed_groups

    if preset_name in PERF_EVENT_GROUP_PRESETS:
        return [(name, list(events)) for name, events in PERF_EVENT_GROUP_PRESETS[preset_name]]

    fallback_events = list(default_events or CACHE_EVENTS)
    return [('cache_tlb', fallback_events)]


def _procfs_path(pid: int, leaf: str) -> Path:
    return Path('/proc') / str(pid) / leaf


def _read_procfs_text(pid: int, leaf: str) -> str:
    path = _procfs_path(pid, leaf)
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        return f'# Failed to read {path}: {exc}\n'


def _parse_maps_line(line: str) -> dict:
    parts = line.split(maxsplit=5)
    record = {
        'raw': line.rstrip('\n'),
        'address': parts[0] if len(parts) > 0 else '',
        'perms': parts[1] if len(parts) > 1 else '',
        'offset': parts[2] if len(parts) > 2 else '',
        'dev': parts[3] if len(parts) > 3 else '',
        'inode': parts[4] if len(parts) > 4 else '',
        'pathname': parts[5] if len(parts) > 5 else '',
    }
    start_hex, end_hex = record['address'].split('-', 1)
    record['start'] = int(start_hex, 16)
    record['end'] = int(end_hex, 16)
    record['size_kb'] = (record['end'] - record['start']) // 1024
    return record


def _parse_smaps(pid: int) -> list[dict]:
    records = []
    current = None
    header_pattern = re.compile(r'^[0-9a-f]+-[0-9a-f]+\s')
    for raw_line in _read_procfs_text(pid, 'smaps').splitlines():
        if header_pattern.match(raw_line):
            current = _parse_maps_line(raw_line)
            current['metrics'] = {}
            records.append(current)
            continue
        if current is None or ':' not in raw_line:
            continue
        key, raw_value = raw_line.split(':', 1)
        value = raw_value.strip()
        match = re.match(r'^(\d+)\s+kB$', value)
        current['metrics'][key] = int(match.group(1)) if match else value
    return records


def _parse_numa_maps(pid: int) -> dict[int, str]:
    mapping = {}
    for raw_line in _read_procfs_text(pid, 'numa_maps').splitlines():
        parts = raw_line.split(maxsplit=1)
        if not parts:
            continue
        try:
            mapping[int(parts[0], 16)] = raw_line
        except ValueError:
            continue
    return mapping


def _run_numastat_for_pid(pid: int) -> str:
    if shutil.which('numastat') is None:
        return '# numastat unavailable\n'
    try:
        completed = subprocess.run(
            ['numastat', '-p', str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f'# numastat failed: {exc}\n'
    output = completed.stdout or completed.stderr or ''
    return output if output.endswith('\n') else output + '\n'


_PAGEMAP_ENTRY_SIZE = 8
_PAGEMAP_PRESENT_BIT = 1 << 63
_PAGEMAP_PFN_MASK = (1 << 55) - 1


def _read_procfs_binary_with_optional_sudo(path: Path, offset: int, length: int) -> bytes:
    force_sudo = path.name == 'pagemap'
    try:
        if force_sudo:
            raise PermissionError
        with path.open('rb', buffering=0) as handle:
            handle.seek(offset)
            data = handle.read(length)
    except PermissionError:
        sudo_binary = shutil.which('sudo')
        python_binary = sys.executable or shutil.which('python3') or 'python3'
        if sudo_binary is None:
            raise RuntimeError(f'Permission denied reading {path} and sudo is unavailable.')
        helper = (
            'import sys\n'
            'path = sys.argv[1]\n'
            'offset = int(sys.argv[2])\n'
            'length = int(sys.argv[3])\n'
            'with open(path, "rb", buffering=0) as handle:\n'
            '    handle.seek(offset)\n'
            '    data = handle.read(length)\n'
            'sys.stdout.buffer.write(data)\n'
        )
        completed = subprocess.run(
            [sudo_binary, '-n', python_binary, '-c', helper, str(path), str(offset), str(length)],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode('utf-8', errors='replace').strip()
            raise RuntimeError(
                f'Failed to read {path} via sudo (exit={completed.returncode}): {stderr or "no stderr"}'
            )
        data = completed.stdout
    except OSError as exc:
        raise RuntimeError(f'Failed to read {path}: {exc}') from exc

    if len(data) != length:
        raise RuntimeError(f'Short read from {path}: expected {length} bytes, got {len(data)} bytes.')
    return data


def _format_histogram(histogram: dict[int, int], top_n: int = 5) -> list[dict]:
    items = sorted(histogram.items(), key=lambda item: (-item[1], abs(item[0]), item[0]))
    return [
        {'delta': delta, 'count': count}
        for delta, count in items[:top_n]
    ]


def _probe_pfn_layout(pid: int, record: dict) -> dict:
    page_size = os.sysconf('SC_PAGE_SIZE')
    start = int(record.get('start', 0))
    end = int(record.get('end', 0))
    virtual_pages = max(0, (end - start) // page_size)
    if virtual_pages <= 0:
        return {
            'address': record.get('address', ''),
            'pathname': record.get('pathname', ''),
            'error': 'mapping has no pages',
        }

    pagemap_offset = (start // page_size) * _PAGEMAP_ENTRY_SIZE
    pagemap_length = virtual_pages * _PAGEMAP_ENTRY_SIZE
    pagemap_data = _read_procfs_binary_with_optional_sudo(_procfs_path(pid, 'pagemap'), pagemap_offset, pagemap_length)

    present_pages = 0
    zero_pfn_present_pages = 0
    adjacent_virtual_present_pairs = 0
    adjacent_virtual_contiguous_pairs = 0
    present_order_pairs = 0
    present_order_contiguous_pairs = 0
    present_order_abs_delta_sum = 0
    adjacent_stride_histogram: dict[int, int] = {}
    present_order_stride_histogram: dict[int, int] = {}
    contiguous_runs: list[int] = []

    previous_present_virtual_index: Optional[int] = None
    previous_present_pfn: Optional[int] = None
    current_run_pages = 0

    for virtual_index, (entry,) in enumerate(struct.iter_unpack('<Q', pagemap_data)):
        if not (entry & _PAGEMAP_PRESENT_BIT):
            if current_run_pages >= 2:
                contiguous_runs.append(current_run_pages)
            current_run_pages = 0
            continue

        present_pages += 1
        pfn = entry & _PAGEMAP_PFN_MASK
        if pfn == 0:
            zero_pfn_present_pages += 1

        if previous_present_virtual_index is not None and previous_present_virtual_index + 1 == virtual_index:
            adjacent_virtual_present_pairs += 1
            delta = pfn - previous_present_pfn
            adjacent_stride_histogram[int(delta)] = adjacent_stride_histogram.get(int(delta), 0) + 1
            if delta == 1:
                adjacent_virtual_contiguous_pairs += 1
                current_run_pages = current_run_pages + 1 if current_run_pages >= 2 else 2
            else:
                if current_run_pages >= 2:
                    contiguous_runs.append(current_run_pages)
                current_run_pages = 0
        else:
            if current_run_pages >= 2:
                contiguous_runs.append(current_run_pages)
            current_run_pages = 0

        if previous_present_pfn is not None:
            present_order_pairs += 1
            delta = pfn - previous_present_pfn
            present_order_stride_histogram[int(delta)] = present_order_stride_histogram.get(int(delta), 0) + 1
            if delta == 1:
                present_order_contiguous_pairs += 1
            present_order_abs_delta_sum += abs(int(delta))

        previous_present_virtual_index = virtual_index
        previous_present_pfn = pfn

    if current_run_pages >= 2:
        contiguous_runs.append(current_run_pages)

    if present_pages == 0:
        return {
            'address': record.get('address', ''),
            'pathname': record.get('pathname', ''),
            'virtual_pages': virtual_pages,
            'present_pages': 0,
            'error': 'mapping has no present pages',
        }

    pfn_redacted = zero_pfn_present_pages == present_pages
    avg_run_pages = (sum(contiguous_runs) / len(contiguous_runs)) if contiguous_runs else 0.0
    avg_present_order_abs_delta = (
        present_order_abs_delta_sum / present_order_pairs
        if present_order_pairs > 0 else 0.0
    )

    return {
        'address': record.get('address', ''),
        'pathname': record.get('pathname', ''),
        'virtual_pages': virtual_pages,
        'present_pages': present_pages,
        'present_density_pct': (present_pages / virtual_pages) * 100.0,
        'pfn_redacted': pfn_redacted,
        'adjacent_virtual_present_pairs': adjacent_virtual_present_pairs,
        'adjacent_virtual_contiguous_pairs': adjacent_virtual_contiguous_pairs,
        'adjacent_virtual_contiguous_ratio_pct': (
            (adjacent_virtual_contiguous_pairs / adjacent_virtual_present_pairs) * 100.0
            if adjacent_virtual_present_pairs > 0 else 0.0
        ),
        'present_order_pairs': present_order_pairs,
        'present_order_contiguous_pairs': present_order_contiguous_pairs,
        'present_order_contiguous_ratio_pct': (
            (present_order_contiguous_pairs / present_order_pairs) * 100.0
            if present_order_pairs > 0 else 0.0
        ),
        'present_order_avg_abs_pfn_delta': avg_present_order_abs_delta,
        'contiguous_run_count': len(contiguous_runs),
        'max_contiguous_run_pages': max(contiguous_runs) if contiguous_runs else 0,
        'avg_contiguous_run_pages': avg_run_pages,
        'top_adjacent_virtual_pfn_deltas': _format_histogram(adjacent_stride_histogram),
        'top_present_order_pfn_deltas': _format_histogram(present_order_stride_histogram),
    }


def _collect_pfn_probe(pid: int, summary_records: list[dict]) -> list[dict]:
    if not _read_bool_env('CALICO_CAPTURE_PFN_AFTER_WARMUP', False):
        return []

    top_n = int(os.environ.get('CALICO_PFN_PROBE_TOP_N', '2') or '2')
    eligible_records = [
        record for record in summary_records
        if int(record.get('rss_kb') or 0) >= 1024 and record.get('address')
    ]
    selected_records = eligible_records[:max(1, top_n)]
    results = []
    for record in selected_records:
        try:
            results.append(_probe_pfn_layout(pid, record))
        except Exception as exc:
            results.append({
                'address': record.get('address', ''),
                'pathname': record.get('pathname', ''),
                'error': str(exc),
            })
    return results


def capture_procfs_snapshot(tag: str, block_path: Optional[str] = None) -> Optional[Path]:
    if not _read_bool_env('CALICO_CAPTURE_PROCFS_AFTER_WARMUP', False):
        return None

    run_base = os.environ.get('CALICO_RUN_OUTPUT_BASE')
    if not run_base:
        return None

    pid = os.getpid()
    block_path = block_path or os.environ.get('BLOCK', '')
    output_base = Path(run_base)
    snapshot_dir = output_base.parent / 'procfs'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{output_base.name}_{tag}'

    maps_text = _read_procfs_text(pid, 'maps')
    numa_maps_text = _read_procfs_text(pid, 'numa_maps')
    smaps_rollup_text = _read_procfs_text(pid, 'smaps_rollup')
    status_text = _read_procfs_text(pid, 'status')
    statm_text = _read_procfs_text(pid, 'statm')
    numastat_text = _run_numastat_for_pid(pid)

    maps_records = []
    for line in maps_text.splitlines():
        try:
            maps_records.append(_parse_maps_line(line))
        except Exception:
            continue

    numa_map_lookup = _parse_numa_maps(pid)
    smaps_records = _parse_smaps(pid)
    block_name = Path(block_path).name if block_path else ''

    def is_relevant(record: dict) -> bool:
        pathname = record.get('pathname', '')
        if block_path and block_path in pathname:
            return True
        if block_name and block_name in pathname:
            return True
        if record.get('size_kb', 0) >= 64 * 1024:
            return True
        if pathname.startswith('['):
            return pathname in {'[heap]', '[stack]'}
        return False

    summary_records = []
    for record in smaps_records:
        if not is_relevant(record):
            continue
        metrics = record.get('metrics', {})
        summary = {
            'start': record.get('start'),
            'end': record.get('end'),
            'address': record.get('address'),
            'pathname': record.get('pathname'),
            'perms': record.get('perms'),
            'size_kb': record.get('size_kb'),
            'rss_kb': metrics.get('Rss'),
            'pss_kb': metrics.get('Pss'),
            'shared_clean_kb': metrics.get('Shared_Clean'),
            'shared_dirty_kb': metrics.get('Shared_Dirty'),
            'private_clean_kb': metrics.get('Private_Clean'),
            'private_dirty_kb': metrics.get('Private_Dirty'),
            'anonymous_kb': metrics.get('Anonymous'),
            'anon_huge_pages_kb': metrics.get('AnonHugePages'),
            'shmem_pmd_mapped_kb': metrics.get('ShmemPmdMapped'),
            'file_pmd_mapped_kb': metrics.get('FilePmdMapped'),
            'kernel_page_size_kb': metrics.get('KernelPageSize'),
            'mmu_page_size_kb': metrics.get('MMUPageSize'),
            'locked_kb': metrics.get('Locked'),
            'vm_flags': metrics.get('VmFlags'),
            'numa_maps': numa_map_lookup.get(record.get('start', -1), ''),
        }
        summary_records.append(summary)

    summary_records.sort(
        key=lambda item: (
            int(item.get('rss_kb') or 0),
            int(item.get('size_kb') or 0),
        ),
        reverse=True,
    )

    pfn_probe_records = _collect_pfn_probe(pid, summary_records)

    summary = {
        'pid': pid,
        'tag': tag,
        'block_path': block_path,
        'status_excerpt': [
            line for line in status_text.splitlines()
            if line.startswith(('Vm', 'Rss', 'Threads', 'HugetlbPages'))
        ],
        'statm': statm_text.strip(),
        'smaps_rollup': smaps_rollup_text.splitlines(),
        'numastat': numastat_text.splitlines(),
        'relevant_mappings': summary_records,
        'pfn_probe': pfn_probe_records,
    }

    json_path = snapshot_dir / f'{stem}.json'
    txt_path = snapshot_dir / f'{stem}.txt'
    maps_path = snapshot_dir / f'{stem}.maps'
    numa_maps_path = snapshot_dir / f'{stem}.numa_maps'

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    maps_path.write_text(maps_text, encoding='utf-8')
    numa_maps_path.write_text(numa_maps_text, encoding='utf-8')

    lines = [
        f'PID: {pid}',
        f'Tag: {tag}',
        f'Block Path: {block_path}',
        '',
        '[status excerpt]',
        *summary['status_excerpt'],
        '',
        '[smaps_rollup]',
        *summary['smaps_rollup'],
        '',
        '[numastat -p]',
        *summary['numastat'],
        '',
        '[relevant mappings]',
    ]
    for record in summary_records:
        lines.extend([
            f"- {record['address']} {record['perms']} {record['pathname'] or '[anonymous]'}",
            f"  size_kb={record['size_kb']} rss_kb={record['rss_kb']} anon_kb={record['anonymous_kb']} anon_huge_pages_kb={record['anon_huge_pages_kb']}",
            f"  kernel_page_size_kb={record['kernel_page_size_kb']} mmu_page_size_kb={record['mmu_page_size_kb']} vm_flags={record['vm_flags']}",
            f"  numa_maps={record['numa_maps']}",
        ])
    if pfn_probe_records:
        lines.extend([
            '',
            '[pfn probe]',
        ])
        for record in pfn_probe_records:
            lines.append(f"- {record.get('address', '')} {record.get('pathname', '') or '[anonymous]'}")
            if record.get('error'):
                lines.append(f"  error={record['error']}")
                continue
            lines.extend([
                f"  virtual_pages={record['virtual_pages']} present_pages={record['present_pages']} present_density_pct={record['present_density_pct']:.2f} pfn_redacted={record['pfn_redacted']}",
                f"  adjacent_virtual_pairs={record['adjacent_virtual_present_pairs']} contiguous_adjacent_pairs={record['adjacent_virtual_contiguous_pairs']} contiguous_adjacent_ratio_pct={record['adjacent_virtual_contiguous_ratio_pct']:.2f}",
                f"  present_order_pairs={record['present_order_pairs']} contiguous_present_order_pairs={record['present_order_contiguous_pairs']} contiguous_present_order_ratio_pct={record['present_order_contiguous_ratio_pct']:.2f}",
                f"  present_order_avg_abs_pfn_delta={record['present_order_avg_abs_pfn_delta']:.2f} contiguous_run_count={record['contiguous_run_count']} max_contiguous_run_pages={record['max_contiguous_run_pages']} avg_contiguous_run_pages={record['avg_contiguous_run_pages']:.2f}",
                f"  top_adjacent_virtual_pfn_deltas={record['top_adjacent_virtual_pfn_deltas']}",
                f"  top_present_order_pfn_deltas={record['top_present_order_pfn_deltas']}",
            ])
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'Captured procfs snapshot at {txt_path}')
    return txt_path


def add_perf_group_results(destination: dict, group_name: str, stats: dict, include_legacy_keys: bool = False) -> None:
    group_prefix = sanitize_perf_counter_name(group_name)
    destination[f'perf_group_{group_prefix}_events'] = ','.join(stats.keys())
    for event_name, value in stats.items():
        sanitized_name = sanitize_perf_counter_name(event_name)
        destination[f'{group_prefix}_{sanitized_name}'] = value
        if include_legacy_keys:
            destination[event_name] = value

    misses = stats.get('cache-misses')
    refs = stats.get('cache-references')
    if isinstance(misses, int) and isinstance(refs, int) and refs > 0:
        miss_rate_pct = (misses / refs) * 100.0
        destination[f'{group_prefix}_cache_miss_rate_pct'] = miss_rate_pct
        if include_legacy_keys:
            destination['cache_miss_rate_pct'] = miss_rate_pct


def get_rss_from_proc():
    """Read RSS (Resident Set Size) from /proc/self/status in bytes."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # Format is "VmRSS:     12345 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kb = int(parts[1])
                        return rss_kb * 1024  # Convert to bytes
    except Exception as e:
        print(f"Warning: Could not read RSS from /proc/self/status: {e}")
    return None

class CacheProfiler:
    """
    A context manager to profile a block of Python code for CPU cache events
    using the Linux `perf` tool.
    """
    def __init__(self, events=None):
        if sys.platform != "linux":
            raise RuntimeError("CacheProfiler requires a Linux operating system.")
        if not shutil.which("perf"):
            raise RuntimeError("The 'perf' command-line tool is not installed or not in PATH.")
        if not _perf_is_usable():
            raise RuntimeError("perf events are unavailable on this host.")
        
        self.events = events or ['cache-misses', 'cache-references', 'L1-dcache-load-misses', 'LLC-load-misses', 'dTLB-load-misses', 'dTLB-loads', 'instructions', 'cycles']
        self.enabled = len(self.events) > 0
        
        if self.enabled:
            print(f"CacheProfiler initialized with events: {self.events}")
        else:
            print("CacheProfiler disabled: no events to profile")
        
        self.pid = os.getpid()
        self.proc = None
        self.raw_output = ""
        self.results = {}

    def __enter__(self):
        if not self.enabled:
            return self

        perf_prefix = _resolve_perf_command_prefix()
        if perf_prefix is None:
            raise RuntimeError("perf events are unavailable on this host.")

        cmd = perf_prefix + [
            "stat",
            "-e", ",".join(self.events),
            "-p", str(self.pid)
        ]
        self.proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, start_new_session=True)
        time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
            
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                _, stderr = self.proc.communicate(timeout=5)
                self.raw_output = stderr
                self._parse_output()
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=5)
                print("Warning: perf process timed out and was killed.")
        
        return False

    def _parse_output(self):
        if not self.enabled:
            return

        for event in self.events:
            pattern = re.compile(r"^\s*([\d,]+)\s+{}.*".format(re.escape(event)), re.MULTILINE)
            match = pattern.search(self.raw_output)
            if match:
                try:
                    value = int(match.group(1).replace(',', ''))
                    self.results[event] = value
                except (ValueError, IndexError):
                    self.results[event] = "N/A"
            else:
                cpu_pattern = re.compile(r"^\s*([\d,]+)\s+cpu_(?:core|atom)/{}/*".format(re.escape(event)), re.MULTILINE)
                matches = cpu_pattern.findall(self.raw_output)
                if matches:
                    try:
                        total_value = sum(int(match.replace(',', '')) for match in matches)
                        self.results[event] = total_value
                    except (ValueError, IndexError):
                        self.results[event] = "N/A"
                else:
                    self.results[event] = "N/A"

        if _perf_verbose_output_enabled() and self.raw_output.strip():
            print(self.raw_output)

    def get_results(self):
        return self.results


@dataclass
class PerfSession:
    process: subprocess.Popen
    data_path: Path
    output_dir: Path
    label: str


_FLAMEGRAPH_DIR_CACHE: Optional[Path] = None
_FLAMEGRAPH_DIR_SCANNED = False


def _resolve_flamegraph_dir() -> Optional[Path]:
    """Locate the FlameGraph utilities if they are available."""
    global _FLAMEGRAPH_DIR_CACHE, _FLAMEGRAPH_DIR_SCANNED

    if _FLAMEGRAPH_DIR_SCANNED:
        return _FLAMEGRAPH_DIR_CACHE

    candidate_paths = []
    for env_var in ("CALICO_FLAMEGRAPH_DIR", "FLAMEGRAPH_DIR"):
        env_value = os.environ.get(env_var)
        if env_value:
            candidate_paths.append(Path(env_value))

    script_dir = Path(__file__).resolve().parent
    candidate_paths.extend(
        [
            script_dir / "FlameGraph",
            script_dir.parent / "FlameGraph",
            Path.cwd() / "FlameGraph",
        ]
    )

    for candidate in candidate_paths:
        if candidate and (candidate / "stackcollapse-perf.pl").is_file() and (candidate / "flamegraph.pl").is_file():
            _FLAMEGRAPH_DIR_CACHE = candidate
            break

    _FLAMEGRAPH_DIR_SCANNED = True
    return _FLAMEGRAPH_DIR_CACHE


def start_perf_profiling(
    delay: float = DEFAULT_PERF_DELAY_SECONDS,
    duration: float = DEFAULT_PERF_DURATION_SECONDS,
    label_suffix: str = "search",
    target_pid: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Optional[PerfSession]:
    """Launch `perf record` for the given PID and return a session handle."""
    if shutil.which("perf") is None:
        print("Note: perf binary not found; skipping perf profiling.")
        return None
    perf_prefix = _resolve_perf_command_prefix()
    if perf_prefix is None:
        return None

    target_pid = target_pid or os.getpid()
    run_base = os.environ.get("CALICO_RUN_OUTPUT_BASE")
    run_label = os.environ.get("CALICO_RUN_LABEL")

    if run_base:
        base_path = Path(run_base)
        if output_dir is None:
            output_dir = base_path.parent / "perf"
        if not run_label:
            run_label = base_path.name

    if output_dir is None:
        output_dir = Path.cwd() / "perf_profiles"

    # Ensure output directory exists and is writable; else fall back to /tmp
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            output_dir.chmod(0o777)
        except Exception:
            pass
        # quick writeability test
        _test = output_dir / ".perm_test"
        with open(_test, "wb") as f:
            f.write(b"ok")
        try:
            _test.unlink()
        except Exception:
            pass
    except Exception:
        tmp_dir = Path(tempfile.gettempdir()) / f"calico_perf_{os.getuid()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            tmp_dir.chmod(0o777)
        except Exception:
            pass
        output_dir = tmp_dir

    base_label = run_label or "perf"
    final_label = f"{base_label}_{label_suffix}" if label_suffix else base_label
    data_path = output_dir / f"{final_label}.data"
    if data_path.exists():
        try:
            data_path.unlink()
        except OSError:
            pass

    delay = max(0.0, float(delay))
    duration = max(0.1, float(duration))

    # Build a shell one-liner so delay doesn't block Python: sleep <delay>; perf record ...
    data_quoted = shlex.quote(str(data_path))
    cmd_parts = []
    if delay > 0:
        cmd_parts.append(f"sleep {delay}")
    perf_record_cmd = perf_prefix + [
        "record",
        "-g",
        "-p", str(int(target_pid)),
        "-o", str(data_path),
        "--",
        "sleep",
        str(duration),
    ]
    cmd_parts.append(f"exec {shlex.join(perf_record_cmd)}")
    cmd_str = "; ".join(cmd_parts)

    try:
        process = subprocess.Popen(["bash", "-lc", cmd_str], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Note: perf binary not found; skipping perf profiling.")
        return None

    return PerfSession(process=process, data_path=data_path, output_dir=output_dir, label=final_label)


def wait_for_perf(session: Optional[PerfSession]) -> None:
    """Finalize perf profiling and emit a flamegraph if possible."""
    if session is None:
        return

    try:
        _, stderr = session.process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        session.process.kill()
        print("WARNING: perf record timed out; profiling aborted.")
        return

    if session.process.returncode != 0:
        err_msg = (stderr.decode("utf-8", "ignore").strip() if stderr else "")
        if err_msg:
            print(f"WARNING: perf record failed ({session.process.returncode}): {err_msg}")
        else:
            print(f"WARNING: perf record failed with code {session.process.returncode}")
        return

    flamegraph_dir = _resolve_flamegraph_dir()
    if flamegraph_dir is None:
        print("Note: FlameGraph utilities not found; skipping flamegraph generation.")
        return

    stackcollapse = flamegraph_dir / "stackcollapse-perf.pl"
    flamegraph = flamegraph_dir / "flamegraph.pl"
    svg_path = session.output_dir / f"{session.label}.svg"

    perf_script = subprocess.run(
        _resolve_perf_command_prefix() + ["script", "-i", str(session.data_path)],
        capture_output=True,
    )
    if perf_script.returncode != 0:
        err_msg = perf_script.stderr.decode("utf-8", "ignore").strip()
        print(
            f"WARNING: perf script failed for {session.data_path}"
            f" ({perf_script.returncode}): {err_msg or 'no stderr output'}"
        )
        return

    if not perf_script.stdout.strip():
        header = subprocess.run(
            _resolve_perf_command_prefix() + ["report", "--header-only", "-i", str(session.data_path)],
            capture_output=True,
        )
        header_msg = header.stdout.decode("utf-8", "ignore").strip()
        header_path = session.output_dir / f"{session.label}.header.txt"
        if header_msg:
            try:
                header_path.write_text(header_msg + "\n", encoding="utf-8")
            except OSError:
                header_path = None
        print(
            f"WARNING: perf data {session.data_path} contains no decoded samples. "
            f"Try a longer capture window with CALICO_PERF_DURATION or a smaller CALICO_PERF_DELAY."
        )
        if _perf_verbose_output_enabled() and header_msg:
            print(header_msg)
        elif header_msg and header_path is not None:
            print(f"Perf header saved to {header_path}")
        return

    collapse_proc = subprocess.run(
        [str(stackcollapse)],
        input=perf_script.stdout,
        capture_output=True,
    )
    if collapse_proc.returncode != 0:
        err_msg = collapse_proc.stderr.decode("utf-8", "ignore").strip()
        print(
            f"WARNING: stackcollapse-perf failed for {session.data_path}"
            f" ({collapse_proc.returncode}): {err_msg or 'no stderr output'}"
        )
        return

    if not collapse_proc.stdout.strip():
        print(f"WARNING: stackcollapse-perf produced no stacks for {session.data_path}")
        return

    with open(svg_path, "wb") as svg_file:
        flamegraph_proc = subprocess.run(
            [str(flamegraph), "--title", session.label],
            input=collapse_proc.stdout,
            stdout=svg_file,
            stderr=subprocess.PIPE,
        )

    if flamegraph_proc.returncode != 0:
        err_msg = flamegraph_proc.stderr.decode("utf-8", "ignore").strip()
        print(
            f"WARNING: flamegraph generation failed for {session.data_path}"
            f" ({flamegraph_proc.returncode}): {err_msg or 'no stderr output'}"
        )
        if svg_path.exists():
            try:
                svg_path.unlink()
            except OSError:
                pass
        return

    print(f"Perf flamegraph written to {svg_path}")


def format_bytes(byte_count):
    if byte_count is None or not isinstance(byte_count, (int, float)):
        return "N/A"
    if byte_count == 0:
        return "0.00 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"

def parse_calico_index_info(info_string):
    """
    Parses the detailed index info string from calico_py to extract key metrics.
    """
    parsed_data = {}
    
    patterns = {
        'page_size_bytes': r"Page Size:\s*(\d+)\s*bytes",
        'physical_pages_used': r"Physical Pages Used:\s*(\d+)",
        'node_count': r"Current Node Count:\s*(\d+)",
        'max_level': r"Current Max Level:\s*(\d+)",
        'capacity_usage_pct': r"Index Capacity Usage:\s*([\d.]+)%",
        'page_utilization_pct': r"Page Utilization:\s*([\d.]+)%",
        'vector_dim': r"Vector Dimension:\s*(\d+)",
        'config_m': r"Max Neighbors \(M\):\s*(\d+)",
        'config_m0': r"Max Neighbors \(M0\):\s*(\d+)",
        'config_ef_construction': r"Construction ef:\s*(\d+)",
        'config_max_levels': r"Max Levels:\s*(\d+)",
        'metadata_max_elements': r"Max Elements:\s*(\d+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, info_string)
        if match:
            try:
                value_str = match.group(1)
                parsed_data[key] = float(value_str) if '.' in value_str else int(value_str)
            except (ValueError, IndexError):
                pass

    if 'page_size_bytes' in parsed_data and 'physical_pages_used' in parsed_data:
        parsed_data['index_bytes'] = parsed_data['page_size_bytes'] * parsed_data['physical_pages_used']

    return parsed_data

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert fv.size % (dim + 1) == 0
    fv = fv.reshape(-1, dim + 1)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector dimensions in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

"""
                  IO Utils
"""

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        if chunk_size is None:
            chunk_size = nvecs - start_idx
        
        # To avoid overflow on large files, use f.seek() to navigate to the
        # correct start position. The offset is calculated from the start of the file.
        # Header size = 2 * 4 bytes = 8 bytes.
        # Vector data starts after the header.
        offset = 8 + np.int64(start_idx) * np.int64(dim) * 4  # 4 bytes per float32
        f.seek(offset)
        
        arr = np.fromfile(f, count=chunk_size * dim, dtype=np.float32)
    return arr.reshape(chunk_size, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        if chunk_size is None:
            chunk_size = nvecs - start_idx

        # To avoid overflow on large files, use f.seek() to navigate to the
        # correct start position. The offset is calculated from the start of the file.
        # Header size = 2 * 4 bytes = 8 bytes.
        # Vector data starts after the header.
        offset = 8 + np.int64(start_idx) * np.int64(dim) * 4  # 4 bytes per int32
        f.seek(offset)

        arr = np.fromfile(f, count=chunk_size * dim, dtype=np.int32)
    return arr.reshape(chunk_size, dim)

def get_fbin_metadata(filename):
    """Get dimension and count from an fbin file."""
    with open(filename, 'rb') as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        return dim, nvecs

def get_fvecs_metadata(filename):
    """Get dimension and count without loading the full dataset."""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            return 0, 0
        dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        record_size = (dim + 1) * 4  # (dim + 1) float32 values
        count = file_size // record_size
        return dim, count

def read_fvecs_batch(filename, start_idx, count, dim):
    """Read a batch of vectors from an fvecs file."""
    record_size = (dim + 1) * 4  # (dim + 1) float32 values per vector
    offset = start_idx * record_size
    
    with open(filename, 'rb') as f:
        f.seek(offset)
        data = f.read(count * record_size)
        
    if len(data) == 0:
        return np.zeros((0, dim), dtype=np.float32)
    
    fv = np.frombuffer(data, dtype=np.float32)
    fv = fv.reshape(-1, dim + 1)
    
    # Verify dimensions match
    dims = fv.view(np.int32)[:, 0]
    if not all(dims == dim):
        raise IOError(f"Non-uniform vector dimensions in {filename}")
    
    # Extract just the vector data (skip dimension prefix)
    vectors = np.ascontiguousarray(fv[:, 1:])
    return vectors

def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert iv.size % (dim + 1) == 0
    iv = iv.reshape(-1, dim + 1)
    if not all(iv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector dimensions in " + filename)
    iv = iv[:, 1:]
    return iv

def calculate_recall(ground_truth, result_labels, k):
    correct = 0
    total_queries = len(ground_truth)
    for i in range(total_queries):
        gt_set = set(ground_truth[i][:k])
        res_set = set(result_labels[i])
        correct += len(gt_set.intersection(res_set))
    return correct / (total_queries * k)

def repeat_and_scramble_queries(X_query, ground_truth, multiplier):
    """
    Repeat the query set and ground truth by multiplier times and randomly scramble them together.
    
    Args:
        X_query: Original query vectors (N, D)
        ground_truth: Ground truth nearest neighbors (N, K)
        multiplier: How many times to repeat (typically number of threads)
    
    Returns:
        Tuple of (scrambled repeated query vectors (N*multiplier, D), 
                  scrambled repeated ground truth (N*multiplier, K))
    """
    if multiplier <= 1:
        return X_query, ground_truth
    
    # Repeat the queries and ground truth
    repeated_queries = np.tile(X_query, (multiplier, 1))
    repeated_gt = np.tile(ground_truth, (multiplier, 1))
    
    # Randomly permute with the same indices
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    indices = rng.permutation(len(repeated_queries))
    
    return repeated_queries[indices], repeated_gt[indices]

def benchmark_calico_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None, build_threads: Optional[int] = None):
    """
    Benchmark a specific Calico configuration.
    
    Config environment variables should be set before calling this function:
    - PHYSGB, TRADITIONAL, TRADHASH, NOHUGEPAGE_FRAME_MEM, PREFETCH, PATH_CACHE
    
    Args:
        X_base: Base dataset (can be None if base_file is provided for streaming)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of threads
        config_name: Configuration name
        base_file: Optional path to base dataset file for streaming
    """
    # Calico selects some internal data structures at module init time.
    # Set THREADS before importing calico_py so the native runtime matches the benchmark setting.
    os.environ["THREADS"] = str(max(1, num_threads))

    # Import calico_py only when needed for Calico benchmarks
    global calico_py
    if calico_py is None:
        print("Importing calico_py module...")
        import calico_py as calico_module
        calico_py = calico_module
    
    print(f"\n{'='*80}")
    effective_build_threads = max(1, build_threads if build_threads is not None else num_threads)
    print(f"Benchmarking Calico Configuration: {config_name}")
    print(f"Search threads: {num_threads}, build threads: {effective_build_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads, 'build_threads': effective_build_threads}
    results['recovery_rebuild_reason'] = None
    
    # Get environment configuration
    env_config = {
        'PHYSGB': os.environ.get('PHYSGB', 'N/A'),
        'TRADITIONAL': os.environ.get('TRADITIONAL', 'N/A'),
        'TRADHASH': os.environ.get('TRADHASH', 'N/A'),
        'NOHUGEPAGE_FRAME_MEM': os.environ.get('NOHUGEPAGE_FRAME_MEM', 'N/A'),
        'PREFETCH': os.environ.get('PREFETCH', 'N/A'),
        'PATH_CACHE': os.environ.get('PATH_CACHE', 'N/A'),
        "NOHUGEPAGE_TRASNLATION_ARRAY": os.environ.get("NOHUGEPAGE_TRANSLATION_ARRAY", "N/A"),
    }
    results.update(env_config)
    
    print(f"Environment Configuration:")
    for key, value in env_config.items():
        print(f"  {key}={value}")

    # Always use streaming mode to minimize memory usage
    if base_file.endswith('.fvecs'):
        dim, num_elements = get_fvecs_metadata(base_file)
    elif base_file.endswith('.mat'):
        if h5py is None:
            raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
        with h5py.File(base_file, 'r') as f:
            ds = f['fea']
            dim = int(ds.shape[1])
            # Only build on base portion (first 10M vectors)
            num_elements = min(int(ds.shape[0]), 10_000_000)
    else:
        dim, num_elements = get_fbin_metadata(base_file)
    print(f"Streaming mode: Will load {num_elements} vectors of dimension {dim} in batches")
    
    # Read RSS before building
    rss_before_build = get_rss_from_proc()
    print(f"\nRSS before build: {rss_before_build / (1024**2):.2f} MB ({rss_before_build / (1024**3):.2f} GB)")

    print(f"\nBuilding index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()

    enable_prefetch = os.environ.get('PREFETCH', '1') == '1'
    skip_recovery_requested = params.get('skip_recovery', False)
    skip_recovery_effective = skip_recovery_requested

    def _instantiate_index(skip_flag: bool):
        return calico_py.HnswIndex(
            max_elements=num_elements,
            dim=dim,
            M=params['M'],
            ef_construction=params['ef_construction'],
            enable_prefetch=enable_prefetch,
            skip_recovery=skip_flag,
        )

    index_calico = _instantiate_index(skip_recovery_requested)
    recovered = bool(index_calico.was_recovered())
    results['skip_recovery_requested'] = skip_recovery_requested
    results['recovered'] = recovered

    index_info_str = None
    parsed_info = {}

    if recovered and not skip_recovery_requested:
        index_info_str = index_calico.get_index_info()
        parsed_info = parse_calico_index_info(index_info_str)
        rebuild_reasons = []

        node_count = parsed_info.get('node_count')
        if node_count is None:
            rebuild_reasons.append("missing node count in recovered metadata")
        elif node_count != num_elements:
            rebuild_reasons.append(
                f"node_count {node_count} != expected {num_elements}"
            )

        stored_dim = parsed_info.get('vector_dim')
        if stored_dim is None:
            rebuild_reasons.append("missing vector dimension in recovered metadata")
        elif stored_dim != dim:
            rebuild_reasons.append(f"dimension {stored_dim} != expected {dim}")

        stored_m = parsed_info.get('config_m')
        if stored_m is None:
            rebuild_reasons.append("missing M in recovered metadata")
        elif stored_m != params['M']:
            rebuild_reasons.append(f"M {stored_m} != expected {params['M']}")

        stored_ef = parsed_info.get('config_ef_construction')
        if stored_ef is None:
            rebuild_reasons.append("missing ef_construction in recovered metadata")
        elif stored_ef != params['ef_construction']:
            rebuild_reasons.append(
                f"ef_construction {stored_ef} != expected {params['ef_construction']}"
            )

        stored_max_elements = parsed_info.get('metadata_max_elements')
        if stored_max_elements is None:
            rebuild_reasons.append("missing max_elements in recovered metadata")
        elif stored_max_elements < num_elements:
            rebuild_reasons.append(
                f"max_elements {stored_max_elements} < required {num_elements}"
            )

        if rebuild_reasons:
            reason_str = "; ".join(rebuild_reasons)
            print(f"Recovered index metadata mismatch ({reason_str}); rebuilding index from scratch.")
            results['recovery_rebuild_reason'] = reason_str
            del index_calico
            start_time = time.time()
            index_calico = _instantiate_index(True)
            skip_recovery_effective = True
            recovered = bool(index_calico.was_recovered())
            if recovered:
                print("Warning: forced rebuild still reported recovery; treating as fresh build.")
                recovered = False
            results['recovered'] = recovered
            index_info_str = None
            parsed_info = {}
        else:
            print("Recovered existing index from disk; skipping data load.")
    elif recovered:
        print("Recovered index state despite skip_recovery request; continuing without ingest.")

    build_time = 0.0
    if not recovered:
        print("Start building graph")
        batch_size = max(1, num_elements // 20)
        print(f"  Inserting {num_elements} vectors in batches of {batch_size} ({batch_size/num_elements*100:.1f}%)...")

        batch_time_total_start = time.time()
        processed = 0
        if base_file.endswith('.mat'):
            # Stream from HDF5 .mat (dataset name 'fea'): first 10M rows are base
            with h5py.File(base_file, 'r') as f:
                ds = f['fea']
                for i in range(0, num_elements, batch_size):
                    loop_start = time.time()
                    end_idx = min(i + batch_size, num_elements)
                    batch = np.ascontiguousarray(ds[i:end_idx, :], dtype=np.float32)
                    index_calico.add_items(batch, num_threads=effective_build_threads)
                    batch_time = time.time() - loop_start
                    processed += (end_idx - i)
                    elapsed_total = time.time() - batch_time_total_start
                    vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                    mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                    remaining = max(0, num_elements - end_idx)
                    eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                          f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")
        else:
            for i in range(0, num_elements, batch_size):
                loop_start = time.time()
                end_idx = min(i + batch_size, num_elements)
                if base_file.endswith('.fvecs'):
                    batch = read_fvecs_batch(base_file, i, end_idx - i, dim)
                else:
                    batch = read_fbin(base_file, start_idx=i, chunk_size=end_idx - i)
                index_calico.add_items(batch, num_threads=effective_build_threads)
                batch_time = time.time() - loop_start
                processed += (end_idx - i)
                elapsed_total = time.time() - batch_time_total_start
                vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                remaining = max(0, num_elements - end_idx)
                eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                      f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")
        build_time = time.time() - start_time
        print(f"Build time: {build_time:.4f} seconds")
        try:
            index_calico.flush()
        except Exception as e:
            print(f"Warning: Unable to flush Calico index after build: {e}")
    else:
        build_time = time.time() - start_time
        print(f"Skipped data loading; recovery completed in {build_time:.4f} seconds")

    results['recovered'] = recovered
    results['build_time'] = build_time
    results['skip_recovery_effective'] = skip_recovery_effective

    print("\nIndex info and memory usage after build:")
    if index_info_str is None:
        index_info_str = index_calico.get_index_info()
        parsed_info = parse_calico_index_info(index_info_str)
    elif not parsed_info:
        parsed_info = parse_calico_index_info(index_info_str)
    print(index_info_str)
    
    results.update(parsed_info)

    mem_usage = index_calico.get_memory_usage()
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")


    k = params["k"]
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    
    # Repeat and scramble queries and ground truth together with the same permutation
    X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, num_threads)
    print(f"Query set: {len(X_query_float32)} queries repeated {num_threads}x = {len(X_query_repeated)} total queries")

    # use 100% of the query to warmup the cache
    warmup_count = max(1, len(X_query_repeated))
    print(f"Warming up cache with {warmup_count} queries...")
    _ = index_calico.search_knn_parallel(X_query_repeated[:warmup_count], k, ef_search=100, num_threads=num_threads, stats=False)

    ef_search = params.get("ef_search", 50) + 8 
    print(f"\nQuerying for k={k} nearest neighbors with ef_search={ef_search}, threads={num_threads}...")
    if hasattr(calico_py, 'reset_path_cache'):
        calico_py.reset_path_cache()
    index_calico.reset_stats()
    collect_search_stats = _read_bool_env("CALICO_ENABLE_SEARCH_STATS", False)

    perf_session = start_perf_profiling(
        delay=DEFAULT_PERF_DELAY_SECONDS,
        duration=DEFAULT_PERF_DURATION_SECONDS,
        label_suffix="search",
        target_pid=os.getpid(),
    )
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = index_calico.search_knn_parallel(
                X_query_repeated,
                k,
                ef_search,
                num_threads,
                stats=collect_search_stats,
            )
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = index_calico.search_knn_parallel(
            X_query_repeated,
            k,
            ef_search,
            num_threads,
            stats=collect_search_stats,
        )
        search_time = time.time() - start_time
    finally:
        wait_for_perf(perf_session)
        
    results['search_time'] = search_time

    qps = len(X_query_repeated) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")

    # Calculate recall using the scrambled ground truth (matches the query order)
    recall = calculate_recall(ground_truth_repeated, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")

    if collect_search_stats:
        stats_string = index_calico.get_stats_string()
        try:
            dist_comps_match = re.search(r"Distance Computations:\s*(\d+)", stats_string)
            hops_match = re.search(r"Search Hops:\s*(\d+)", stats_string)

            if dist_comps_match:
                results['dist_comps'] = int(dist_comps_match.group(1))
                results['avg_dist_comps'] = results['dist_comps'] / len(X_query_repeated)
            if hops_match:
                results['hops'] = int(hops_match.group(1))
                results['avg_hops'] = results['hops'] / len(X_query_repeated)
        except Exception as e:
            print(f"Could not parse calico stats string: {e}")
    else:
        results['dist_comps'] = "N/A"
        results['avg_dist_comps'] = "N/A"
        results['hops'] = "N/A"
        results['avg_hops'] = "N/A"

    try:
        index_calico.flush()
    except Exception as e:
        print(f"Warning: Unable to flush Calico index to storage: {e}")

    return results


def benchmark_hnswlib_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None):
    """
    Benchmark hnswlib with optional file-backed mmap support.
    
    Environment variable MMAP_FOR_HSNWLIB=1 enables mmap mode.
    
    Args:
        X_base: Base dataset (can be None if base_file is provided for streaming)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of threads
        config_name: Configuration name
        base_file: Optional path to base dataset file for streaming
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking hnswlib Configuration: {config_name}")
    print(f"Threads: {num_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads}
    
    # Check if mmap is enabled
    mmap_enabled = os.environ.get('MMAP_FOR_HSNWLIB', '0') == '1'
    results['MMAP_FOR_HSNWLIB'] = '1' if mmap_enabled else '0'
    print(f"Environment Configuration:")
    print(f"  MMAP_FOR_HSNWLIB={results['MMAP_FOR_HSNWLIB']}")

    # Always use streaming mode to minimize memory usage
    if base_file.endswith('.fvecs'):
        dim, num_elements = get_fvecs_metadata(base_file)
    elif base_file.endswith('.mat'):
        if h5py is None:
            raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
        with h5py.File(base_file, 'r') as f:
            ds = f['fea']
            dim = int(ds.shape[1])
            num_elements = min(int(ds.shape[0]), 10_000_000)
    else:
        dim, num_elements = get_fbin_metadata(base_file)
    print(f"Streaming mode: Will load {num_elements} vectors of dimension {dim} in batches")
    
    k = params["k"]

    # Read RSS before building
    rss_before_build = get_rss_from_proc()
    print(f"\nRSS before build: {rss_before_build / (1024**2):.2f} MB ({rss_before_build / (1024**3):.2f} GB)")

    print(f"\nBuilding index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=params["ef_construction"], M=params["M"])
    
    # Stream dataset in 5% batches to reduce peak memory usage
    batch_size = max(1, num_elements // 20)  # 5% batches
    print(f"  Inserting {num_elements} vectors in batches of {batch_size} ({batch_size/num_elements*100:.1f}%)...")
    
    # Stream from file without loading full dataset into memory
    if base_file.endswith('.mat'):
        if h5py is None:
            raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
        with h5py.File(base_file, 'r') as f:
            ds = f['fea']
            for i in range(0, num_elements, batch_size):
                end_idx = min(i + batch_size, num_elements)
                batch = np.ascontiguousarray(ds[i:end_idx, :], dtype=np.float32)
                p.add_items(batch)
                print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%)")
    else:
        for i in range(0, num_elements, batch_size):
            end_idx = min(i + batch_size, num_elements)
            if base_file.endswith('.fvecs'):
                batch = read_fvecs_batch(base_file, i, end_idx - i, dim)
            else:
                batch = read_fbin(base_file, start_idx=i, chunk_size=end_idx - i)
            p.add_items(batch)
            print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%)")

    build_time = time.time() - start_time
    results['build_time'] = build_time
    print(f"Build time: {build_time:.4f} seconds")

    print("\nMemory usage after build:")
    mem_usage = p.get_memory_usage()
    results['index_bytes'] = mem_usage.get('index_bytes')
    results['rss_after_build'] = mem_usage.get('process_rss_bytes')
    print(f"  Index bytes: {format_bytes(results['index_bytes'])}")
    print(f"  Process RSS: {format_bytes(results['rss_after_build'])}")
    results['node_count'] = num_elements

    ef_search = params.get("ef_search", 50)  + 8
    p.set_ef(ef_search)
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    
    # Repeat and scramble queries and ground truth together with the same permutation
    X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, num_threads)
    print(f"Query set: {len(X_query_float32)} queries repeated {num_threads}x = {len(X_query_repeated)} total queries")

    print(f"\nQuerying for k={k} nearest neighbors with ef_search={ef_search}, threads={num_threads}...")
    
    p.reset_stats()
    search_threads = num_threads if num_threads > 0 else -1
    
    perf_session = start_perf_profiling(
        delay=DEFAULT_PERF_DELAY_SECONDS,
        duration=DEFAULT_PERF_DURATION_SECONDS,
        label_suffix="search",
        target_pid=os.getpid(),
    )
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            labels, distances = p.knn_query(X_query_repeated, k=k, num_threads=search_threads, stats=False)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        labels, distances = p.knn_query(X_query_repeated, k=k, num_threads=search_threads, stats=True)
        search_time = time.time() - start_time
    finally:
        wait_for_perf(perf_session)

    results['search_time'] = search_time
    
    qps = len(X_query_repeated) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")
    
    # Calculate recall using the scrambled ground truth
    recall = calculate_recall(ground_truth_repeated, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")
    
    results['dist_comps'] = p.get_distance_computations()
    results['hops'] = p.get_hops()
    results['avg_dist_comps'] = results['dist_comps'] / len(X_query_repeated)
    results['avg_hops'] = results['hops'] / len(X_query_repeated)
    
    graph_stats_obj = p.get_graph_stats()
    results['max_level'] = graph_stats_obj.num_levels - 1 if graph_stats_obj.num_levels > 0 else 0

    return results


def benchmark_predicache_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None, build_threads: Optional[int] = None):
    """
    Benchmark PrediCache HNSW implementation.
    
    Args:
        X_base: Base dataset (can be None if base_file is provided for streaming)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of search threads
        config_name: Configuration name
        base_file: Optional path to base dataset file for streaming
        build_threads: Number of build threads (defaults to num_threads)
    """
    poolgb = os.environ.get('POOLGB')
    physgb = os.environ.get('PHYSGB')
    effective_build_threads = max(1, build_threads if build_threads is not None else num_threads)
    # PrediCache initializes its global buffer manager on module import, and the
    # number of libaio contexts is derived from THREADS at that point. HNSW build
    # can use more workers than search, so size THREADS for the larger of the two.
    os.environ['THREADS'] = str(max(1, num_threads, effective_build_threads))
    if not poolgb and physgb:
        os.environ['POOLGB'] = physgb
        poolgb = physgb
        print(f"PrediCache env bridge: POOLGB was unset, using PHYSGB={physgb}")

    # Import predicache_py only when needed
    global predicache_py
    if predicache_py is None:
        print("Importing predicache_py module...")
        import predicache_py as predicache_module
        predicache_py = predicache_module
    
    print(f"\n{'='*80}")
    print(f"Benchmarking PrediCache Configuration: {config_name}")
    print(f"Search threads: {num_threads}, build threads: {effective_build_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads, 'build_threads': effective_build_threads}
    env_config = {
        'PHYSGB': physgb or 'N/A',
        'POOLGB': poolgb or 'N/A',
        'PREFETCH': os.environ.get('PREFETCH', 'N/A'),
        'TRADITIONAL': os.environ.get('TRADITIONAL', 'N/A'),
        'TRADHASH': os.environ.get('TRADHASH', 'N/A'),
        'NOHUGEPAGE_FRAME_MEM': os.environ.get('NOHUGEPAGE_FRAME_MEM', 'N/A'),
    }
    results.update(env_config)
    print("Environment Configuration:")
    for key, value in env_config.items():
        print(f"  {key}={value}")
    
    # Get dataset metadata
    if base_file.endswith('.fvecs'):
        dim, num_elements = get_fvecs_metadata(base_file)
    elif base_file.endswith('.mat'):
        if h5py is None:
            raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
        with h5py.File(base_file, 'r') as f:
            ds = f['fea']
            dim = int(ds.shape[1])
            num_elements = min(int(ds.shape[0]), 10_000_000)
    else:
        dim, num_elements = get_fbin_metadata(base_file)
    print(f"Streaming mode: Will load {num_elements} vectors of dimension {dim} in batches")
    
    # Read RSS before building
    rss_before_build = get_rss_from_proc()
    print(f"\nRSS before build: {rss_before_build / (1024**2):.2f} MB ({rss_before_build / (1024**3):.2f} GB)")

    print(f"\nBuilding index with M={params['M']}, ef_construction={params['ef_construction']}...")
    start_time = time.time()
    
    enable_prefetch = os.environ.get('PREFETCH', '1') == '1'
    skip_recovery_requested = params.get('skip_recovery', False)
    skip_recovery_effective = skip_recovery_requested

    def _instantiate_index(skip_flag: bool):
        return predicache_py.HNSW(
            num_elements,
            dim,
            params['M'],
            params['ef_construction'],
            enable_prefetch=enable_prefetch,
            # The current PrediCache binding names this constructor flag
            # `can_reuse_storage`, but it is wired to the underlying
            # skip-recovery boolean.
            can_reuse_storage=skip_flag,
        )

    index = _instantiate_index(skip_recovery_requested)
    results['skip_recovery_requested'] = skip_recovery_requested
    recovered = bool(index.was_recovered()) if hasattr(index, 'was_recovered') else False
    results['recovered_from_disk'] = recovered

    index_info_str = None
    parsed_info = {}
    if recovered and not skip_recovery_requested and hasattr(index, 'get_index_info'):
        index_info_str = index.get_index_info()
        parsed_info = parse_calico_index_info(index_info_str)
        rebuild_reasons = []

        node_count = parsed_info.get('node_count')
        if node_count is None:
            rebuild_reasons.append('missing node count in recovered metadata')
        elif node_count != num_elements:
            rebuild_reasons.append(f'node_count {node_count} != expected {num_elements}')

        stored_dim = parsed_info.get('vector_dim')
        if stored_dim is None:
            rebuild_reasons.append('missing vector dimension in recovered metadata')
        elif stored_dim != dim:
            rebuild_reasons.append(f'dimension {stored_dim} != expected {dim}')

        stored_m = parsed_info.get('config_m')
        if stored_m is None:
            rebuild_reasons.append('missing M in recovered metadata')
        elif stored_m != params['M']:
            rebuild_reasons.append(f"M {stored_m} != expected {params['M']}")

        stored_ef = parsed_info.get('config_ef_construction')
        if stored_ef is None:
            rebuild_reasons.append('missing ef_construction in recovered metadata')
        elif stored_ef != params['ef_construction']:
            rebuild_reasons.append(
                f"ef_construction {stored_ef} != expected {params['ef_construction']}"
            )

        stored_max_elements = parsed_info.get('metadata_max_elements')
        if stored_max_elements is None:
            rebuild_reasons.append('missing max_elements in recovered metadata')
        elif stored_max_elements < num_elements:
            rebuild_reasons.append(
                f'max_elements {stored_max_elements} < required {num_elements}'
            )

        if rebuild_reasons:
            reason_str = '; '.join(rebuild_reasons)
            print(f"Recovered PrediCache index metadata mismatch ({reason_str}); rebuilding index from scratch.")
            results['recovery_rebuild_reason'] = reason_str
            del index
            start_time = time.time()
            index = _instantiate_index(True)
            skip_recovery_effective = True
            recovered = bool(index.was_recovered()) if hasattr(index, 'was_recovered') else False
            if recovered:
                print('Warning: forced rebuild still reported recovery; treating as fresh build.')
                recovered = False
            results['recovered_from_disk'] = recovered
            index_info_str = None
            parsed_info = {}
        else:
            print('Recovered existing PrediCache index from disk; skipping data load.')
    elif recovered:
        print('Recovered PrediCache index state despite skip_recovery request; continuing without ingest.')

    if not recovered:
        batch_size = max(1, num_elements // 20)  # 5% batches
        print(f"  Inserting {num_elements} vectors in batches of {batch_size} ({batch_size/num_elements*100:.1f}%)...")

        batch_time_total_start = time.time()
        processed = 0

        if base_file.endswith('.mat'):
            with h5py.File(base_file, 'r') as f:
                ds = f['fea']
                for i in range(0, num_elements, batch_size):
                    loop_start = time.time()
                    end_idx = min(i + batch_size, num_elements)
                    batch = np.ascontiguousarray(ds[i:end_idx, :], dtype=np.float32)
                    index.add_items(batch, num_threads=effective_build_threads)
                    batch_time = time.time() - loop_start
                    processed += (end_idx - i)
                    elapsed_total = time.time() - batch_time_total_start
                    vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                    mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                    remaining = max(0, num_elements - end_idx)
                    eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                          f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")
        else:
            for i in range(0, num_elements, batch_size):
                loop_start = time.time()
                end_idx = min(i + batch_size, num_elements)
                if base_file.endswith('.fvecs'):
                    batch = read_fvecs_batch(base_file, i, end_idx - i, dim)
                else:
                    batch = read_fbin(base_file, start_idx=i, chunk_size=end_idx - i)
                index.add_items(batch, num_threads=effective_build_threads)
                batch_time = time.time() - loop_start
                processed += (end_idx - i)
                elapsed_total = time.time() - batch_time_total_start
                vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                remaining = max(0, num_elements - end_idx)
                eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                      f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")

    build_time = time.time() - start_time
    results['build_time'] = build_time
    results['skip_recovery_effective'] = skip_recovery_effective
    print(f"Build time: {build_time:.4f} seconds")
    
    # Get stats after build
    print("\nIndex info and memory usage after build:")
    if index_info_str is None and hasattr(index, 'get_index_info'):
        index_info_str = index.get_index_info()
        parsed_info = parse_calico_index_info(index_info_str)
    if index_info_str is not None:
        print(index_info_str)
        results.update(parsed_info)
    stats = index.stats()
    results['num_levels'] = stats.get('num_levels', 0)
    results['node_count'] = results.get('node_count', num_elements)
    
    bm_stats = predicache_py.get_buffer_stats()
    results['alloc_count'] = bm_stats['alloc_count']
    results['read_count'] = bm_stats['read_count']
    results['write_count'] = bm_stats['write_count']
    results['pool_size_gb'] = bm_stats['pool_size_gb']
    
    rss_after_build = get_rss_from_proc()
    results['rss_after_build'] = rss_after_build
    print(f"  Process RSS: {format_bytes(rss_after_build)}")
    print(f"  Buffer pool allocations: {results['alloc_count']}")
    print(f"  Buffer pool size: {results['pool_size_gb']:.2f} GB")
    
    k = params["k"]
    X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
    
    # Repeat and scramble queries
    X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, 10)
    print(f"Query set: {len(X_query_float32)} queries repeated 10x = {len(X_query_repeated)} total queries")
    
    # Warmup
    warmup_count = max(1, len(X_query_repeated))
    print(f"Warming up cache with {warmup_count} queries...")
    for query in X_query_repeated[:warmup_count]:
        _ = index.search_knn(query, k, 100)
    
    ef_search = params.get("ef_search", 50) + 8
    print(f"\nQuerying for k={k} nearest neighbors with ef_search={ef_search}, threads={num_threads}...")
    
    perf_session = start_perf_profiling(
        delay=DEFAULT_PERF_DELAY_SECONDS,
        duration=DEFAULT_PERF_DURATION_SECONDS,
        label_suffix="search",
        target_pid=os.getpid(),
    )
    try:
        profiler = CacheProfiler(events=CACHE_EVENTS)
        with profiler:
            start_time = time.time()
            # Use search_knn_parallel (always, matching Calico's API)
            labels, distances = index.search_knn_parallel(X_query_repeated, k, ef_search if ef_search > 0 else None, num_threads)
            search_time = time.time() - start_time
        
        cache_stats = profiler.get_results()
        results.update(cache_stats)
        if 'cache-misses' in cache_stats and 'cache-references' in cache_stats:
            misses = cache_stats['cache-misses']
            refs = cache_stats['cache-references']
            if isinstance(refs, int) and refs > 0 and isinstance(misses, int):
                results['cache_miss_rate_pct'] = (misses / refs) * 100
    except RuntimeError as e:
        print(f"Warning: Could not run CacheProfiler. Error: {e}")
        start_time = time.time()
        # Use search_knn_parallel (always, matching Calico's API)
        labels, distances = index.search_knn_parallel(X_query_repeated, k, ef_search if ef_search > 0 else None, num_threads)
        search_time = time.time() - start_time
    finally:
        wait_for_perf(perf_session)
    
    results['search_time'] = search_time
    qps = len(X_query_repeated) / search_time
    results['qps'] = qps
    print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")
    
    # Calculate recall
    recall = calculate_recall(ground_truth_repeated, labels, k)
    results['recall'] = recall
    print(f"Recall@{k}: {recall:.4f}")
    
    # Get final stats
    final_stats = index.stats()
    results['dist_comps'] = final_stats.get('dist_comps', 0)
    results['hops'] = final_stats.get('search_hops', 0)
    results['avg_dist_comps'] = results['dist_comps'] / len(X_query_repeated) if len(X_query_repeated) > 0 else 0
    results['avg_hops'] = results['hops'] / len(X_query_repeated) if len(X_query_repeated) > 0 else 0
    
    # Get final buffer manager stats
    final_bm_stats = predicache_py.get_buffer_stats()
    results['final_read_count'] = final_bm_stats['read_count']
    results['final_write_count'] = final_bm_stats['write_count']
    
    return results


def benchmark_usearch_config(X_base, X_query, ground_truth, params, num_threads, config_name, base_file=None, index_file=None, mode='search'):
    """
    Benchmark usearch with mmap support.
    
    Args:
        X_base: Base dataset (not used in streaming mode)
        X_query: Query dataset
        ground_truth: Ground truth nearest neighbors
        params: HNSW parameters
        num_threads: Number of threads
        config_name: Configuration name
        base_file: Path to base dataset file for streaming
        index_file: Path to save/load the serialized index
        mode: 'index' to build and save, 'search' to load and query
    """
    if Index is None:
        raise RuntimeError("usearch library is not installed.")

    print(f"\n{'='*80}")
    print(f"Benchmarking usearch Configuration: {config_name} (mode: {mode})")
    print(f"Threads: {num_threads}")
    print(f"{'='*80}")
    
    results = {'config': config_name, 'threads': num_threads, 'mode': mode}
    
    if base_file.endswith('.fvecs'):
        dim, num_elements = get_fvecs_metadata(base_file)
    elif base_file.endswith('.mat'):
        if h5py is None:
            raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
        with h5py.File(base_file, 'r') as f:
            ds = f['fea']
            dim = int(ds.shape[1])
            num_elements = min(int(ds.shape[0]), 10_000_000)
    else:
        dim, num_elements = get_fbin_metadata(base_file)
    
    if mode == 'index':
        print(f"Building index with M={params['M']}, ef_construction={params['ef_construction']}...")
        
        # Usearch uses native L2sq, so we can use f32
        index = Index(
            ndim=dim,
            metric='l2sq',
            dtype='f32',
            connectivity=params['M'],
            expansion_search=params['ef_search'],
            expansion_add=params['ef_construction']
        )
        
        rss_before_build = get_rss_from_proc()
        print(f"RSS before build: {format_bytes(rss_before_build)}")

        start_time = time.time()
        
        batch_size = max(1, num_elements // 20)
        print(f"  Inserting {num_elements} vectors in batches of {batch_size}...")
        
        # Generate integer labels for the vectors
        labels = np.arange(num_elements, dtype=np.int64)
        
        batch_time_total_start = time.time()
        processed = 0
        if base_file.endswith('.mat'):
            if h5py is None:
                raise RuntimeError("h5py is required to read .mat SIFT10M dataset.")
            with h5py.File(base_file, 'r') as f:
                ds = f['fea']
                for i in range(0, num_elements, batch_size):
                    loop_start = time.time()
                    end_idx = min(i + batch_size, num_elements)
                    batch_vectors = np.ascontiguousarray(ds[i:end_idx, :], dtype=np.float32)
                    batch_labels = labels[i:end_idx]
                    index.add(batch_labels, batch_vectors, threads=num_threads)
                    batch_time = time.time() - loop_start
                    processed += (end_idx - i)
                    elapsed_total = time.time() - batch_time_total_start
                    vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                    mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                    remaining = max(0, num_elements - end_idx)
                    eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                          f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")
        else:
            for i in range(0, num_elements, batch_size):
                loop_start = time.time()
                end_idx = min(i + batch_size, num_elements)
                if base_file.endswith('.fvecs'):
                    batch_vectors = read_fvecs_batch(base_file, i, end_idx - i, dim)
                else:
                    batch_vectors = read_fbin(base_file, start_idx=i, chunk_size=end_idx - i)
                batch_labels = labels[i:end_idx]
                index.add(batch_labels, batch_vectors, threads=num_threads)
                batch_time = time.time() - loop_start
                processed += (end_idx - i)
                elapsed_total = time.time() - batch_time_total_start
                vps = processed / elapsed_total if elapsed_total > 0 else float('inf')
                mbps = (float(processed) * dim * 4) / elapsed_total / (1024**2) if elapsed_total > 0 else 0.0
                remaining = max(0, num_elements - end_idx)
                eta_seconds = remaining / vps if vps > 0 and vps != float('inf') else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(f"    Inserted {end_idx}/{num_elements} vectors ({end_idx/num_elements*100:.1f}%). "
                      f"Batch time: {batch_time:.3f}s; Throughput: {vps:.2f} v/s ({mbps:.2f} MB/s); ETA: {eta_str}")
            
        build_time = time.time() - start_time
        results['build_time'] = build_time
        print(f"Build time: {build_time:.4f} seconds")
        
        rss_after_build = get_rss_from_proc()
        results['rss_after_build'] = rss_after_build
        print(f"RSS after build: {format_bytes(rss_after_build)}")
        
        print(f"\nSaving index to {index_file}")
        start_time = time.time()
        index.save(index_file)
        save_time = time.time() - start_time
        results['save_time'] = save_time
        print(f"Index saved in {save_time:.4f} seconds.")
        
        index_size = os.path.getsize(index_file)
        results['index_bytes'] = index_size
        print(f"Index file size: {format_bytes(index_size)}")

        return results

    elif mode == 'search':
        print(f"Loading index from {index_file} with mmap (view=True)...")
        
        rss_before_load = get_rss_from_proc()
        print(f"RSS before load: {format_bytes(rss_before_load)}")

        start_time = time.time()
        index = Index.restore(index_file, view=True)
        load_time = time.time() - start_time
        results['load_time'] = load_time
        print(f"Index loaded in {load_time:.4f} seconds.")
        
        rss_after_load = get_rss_from_proc()
        results['rss_after_load'] = rss_after_load
        print(f"RSS after load: {format_bytes(rss_after_load)}")
        
        k = params["k"]
        ef_search = params.get("ef_search", 50)
        
        X_query_float32 = np.ascontiguousarray(X_query, dtype=np.float32)
        X_query_repeated, ground_truth_repeated = repeat_and_scramble_queries(X_query_float32, ground_truth, num_threads)
        
        # use 100% of the query to warmup the cache
        warmup_count = max(1, len(X_query_repeated))
        print(f"Warming up cache with {warmup_count} queries...")
        _ = index.search(
            X_query_repeated[:warmup_count],
            k,
            exact=False,
            threads=64
        )


        print(f"Query set: {len(X_query_float32)} queries repeated {num_threads}x = {len(X_query_repeated)} total queries")

        print(f"\nQuerying for k={k} nearest neighbors with ef={ef_search}, threads={num_threads}...")

        perf_session = start_perf_profiling(
            delay=DEFAULT_PERF_DELAY_SECONDS,
            duration=DEFAULT_PERF_DURATION_SECONDS,
            label_suffix="search",
            target_pid=os.getpid(),
        )
        try:
            start_time = time.time()
            # Usearch's search function is `search`. It returns a complex object.
            # The `ef` parameter is passed directly to the search method.
            matches = index.search(
                X_query_repeated,
                k,
                exact=False,
                threads=num_threads
            )
            search_time = time.time() - start_time
        finally:
            wait_for_perf(perf_session)

        results['search_time'] = search_time
        qps = len(X_query_repeated) / search_time
        results['qps'] = qps
        print(f"Search time ({len(X_query_repeated)} queries): {search_time:.4f} seconds ({qps:.2f} QPS)")
        
        # Extract labels from the search results
        # The `matches` object is a `BatchMatches` object, which is iterable.
        # Each item corresponds to a query and contains a list of `Match` objects.
        # Each `Match` object has a `key` (the label) and `distance`.
        labels = np.array([[match.key for match in query_matches] for query_matches in matches])
        
        recall = calculate_recall(ground_truth_repeated, labels, k)
        results['recall'] = recall
        print(f"Recall@{k}: {recall:.4f}")
        rss_after_query = get_rss_from_proc()
        print(f"RSS after query: {format_bytes(rss_after_query)}")
        # Usearch does not expose these stats directly
        results['dist_comps'] = "N/A"
        results['hops'] = "N/A"
        results['avg_dist_comps'] = "N/A"
        results['avg_hops'] = "N/A"
        results['max_level'] = "N/A"
        results['index_bytes'] = os.path.getsize(index_file)

        return results

    else:
        raise ValueError(f"Invalid mode for usearch benchmark: {mode}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Calico HNSW configurations')
    parser.add_argument('--config', type=str, required=True, 
                        help='Configuration name (calico_array, calico_chained_hash, hnswlib_mmap, etc.)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of search threads')
    parser.add_argument('--build-threads', type=int, default=32,
                        help='Number of threads for index build (default: 32)')
    parser.add_argument('--dataset', type=str, default='sift1m',
                        help='Dataset to use (sift1m or sift10m)')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW M parameter')
    parser.add_argument('--ef_construction', type=int, default=200,
                        help='HNSW ef_construction parameter')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors to search')
    parser.add_argument('--ef_search', type=int, default=50,
                        help='HNSW ef_search parameter')
    parser.add_argument('--library', type=str, default='calico',
                        help='Library to use: calico, hnswlib, usearch, or predicache')
    parser.add_argument('--mode', type=str, default='search',
                        help="Mode for usearch: 'index' or 'search'")
    parser.add_argument('--index_file', type=str, default='hnsw_index.bin',
                        help='Path to save/load the index file for usearch')
    parser.add_argument('--skip_recovery', action='store_true',
                        help='Skip recovering an existing Calico index from disk (force rebuild)')
    
    args = parser.parse_args()
    num_threads = args.threads
    build_threads = args.build_threads if args.build_threads is not None else num_threads
    
    # Load dataset
    if args.dataset == "sift1m":
        DATA_DIR = str(DATA_ROOT / "sift1m")
        BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
        QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")
        GT_FILE = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

        if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
            print(f"Error: SIFT1M data files not found in '{DATA_DIR}'.")
            sys.exit(1)

        # Always use streaming mode to minimize memory usage
        print(f"Loading SIFT1M dataset (streaming mode for {args.library})...")
        dim, count = get_fvecs_metadata(BASE_FILE)
        print(f"Base vectors: {count} vectors of dimension {dim} (will be streamed in 5% batches)")
        
        X_query = read_fvecs(QUERY_FILE)
        ground_truth = read_ivecs(GT_FILE)
        
        print(f"Query vectors shape: {X_query.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
    elif args.dataset == "deep10M":
        DATA_DIR = str(DATA_ROOT / "deep10M")
        BASE_FILE = os.path.join(DATA_DIR, "base.10M.fbin")
        QUERY_FILE = os.path.join(DATA_DIR, "query.public.10K.fbin")
        GT_FILE = os.path.join(DATA_DIR, "groundtruth.ibin")

        if not all(os.path.exists(f) for f in [BASE_FILE, QUERY_FILE, GT_FILE]):
            print(f"Error: DEEP10M data files not found in '{DATA_DIR}'.")
            sys.exit(1)

        # Always use streaming mode to minimize memory usage
        print(f"Loading DEEP10M dataset (streaming mode for {args.library})...")
        dim, count = get_fbin_metadata(BASE_FILE)
        print(f"Base vectors: {count} vectors of dimension {dim} (will be streamed in 5% batches)")
        
        X_query = read_fbin(QUERY_FILE)
        ground_truth = read_ibin(GT_FILE)
        
        print(f"Query vectors shape: {X_query.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
    elif args.dataset == "sift10M":
        DATA_DIR = str(DATA_ROOT / "SIFT10M")
        MAT_FILE = os.path.join(DATA_DIR, "SIFT10Mfeatures.mat")
        GT_FILE = os.path.join(DATA_DIR, "ground_truth_indices.npy")

        if not all(os.path.exists(f) for f in [MAT_FILE, GT_FILE]):
            print(f"Error: SIFT10M data files not found in '{DATA_DIR}'.")
            sys.exit(1)
        if h5py is None:
            print("Error: h5py is required to read SIFT10M .mat file. Install with 'pip install h5py'.")
            sys.exit(1)

        # Load ground truth first to determine how many queries to load
        ground_truth = np.load(GT_FILE)
        num_queries = int(ground_truth.shape[0])
        num_base = 10_000_000

        print(f"Loading SIFT10M dataset (streaming base, queries from .mat) for {args.library})...")
        with h5py.File(MAT_FILE, 'r') as f:
            ds = f['fea']
            dim = int(ds.shape[1])
            total_rows = int(ds.shape[0])
            if total_rows < num_base + num_queries:
                print(f"Warning: dataset contains {total_rows} rows, expected at least {num_base+num_queries}. Proceeding with available rows.")
            start_q = num_base
            end_q = min(num_base + num_queries, total_rows)
            actual_queries = end_q - start_q
            if actual_queries != num_queries:
                if actual_queries <= 0:
                    print("Error: No query rows available in the .mat file after base segment.")
                    sys.exit(1)
                print(f"Warning: Truncating ground truth from {num_queries} to {actual_queries} to match available queries.")
                ground_truth = ground_truth[:actual_queries]
                num_queries = actual_queries
            X_query = np.ascontiguousarray(ds[start_q:end_q, :], dtype=np.float32)

        BASE_FILE = MAT_FILE

        print(f"Base vectors: first {num_base} rows (streamed). Query shape: {X_query.shape}. GT shape: {ground_truth.shape}")
    else:
        print(f"Dataset {args.dataset} not supported yet")
        sys.exit(1)
    
    hnsw_params = {
        "M": args.M, 
        "ef_construction": args.ef_construction, 
        "k": args.k, 
        "ef_search": args.ef_search,
        "skip_recovery": args.skip_recovery,
    }
    
    # Run benchmark based on library selection (always streaming mode)
    if args.library == 'hnswlib':
        results = benchmark_hnswlib_config(
            None, X_query, ground_truth, hnsw_params, args.threads, args.config,
            base_file=BASE_FILE
        )
    elif args.library == 'usearch':
        results = benchmark_usearch_config(
            None, X_query, ground_truth, hnsw_params, args.threads, args.config,
            base_file=BASE_FILE, index_file=args.index_file, mode=args.mode
        )
    elif args.library == 'predicache':
        results = benchmark_predicache_config(
            None,
            X_query,
            ground_truth,
            hnsw_params,
            num_threads,
            args.config,
            base_file=BASE_FILE,
            build_threads=build_threads,
        )
    else:
        results = benchmark_calico_config(
            None,
            X_query,
            ground_truth,
            hnsw_params,
            num_threads,
            args.config,
            base_file=BASE_FILE,
            build_threads=build_threads,
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Benchmark Summary for {args.config}")
    print(f"{'='*80}")
    # Safe formatting helpers: some backends may set stats to strings like "N/A"
    def _safe_format(val, fmt):
        try:
            return format(float(val), fmt)
        except Exception:
            return str(val)

    print(f"Build Time: {_safe_format(results.get('build_time', 0), '.4f')}s")
    print(f"Search Time: {_safe_format(results.get('search_time', 0), '.4f')}s")
    print(f"QPS: {_safe_format(results.get('qps', 0), '.2f')}")
    print(f"Recall@{args.k}: {_safe_format(results.get('recall', 0), '.4f')}")
    avg_dist = results.get('avg_dist_comps', 0)
    print(f"Avg Dist Comps/Query: {_safe_format(avg_dist, '.2f')}")
    print(f"Index Size: {format_bytes(results.get('index_bytes'))}")

    if args.library == 'predicache' and predicache_py is not None and hasattr(predicache_py, 'flush_buffer_manager'):
        print("\nFlushing PrediCache pages to storage...")
        predicache_py.flush_buffer_manager()
    
    print("\nCalico module unloading: Shutting down system...")
