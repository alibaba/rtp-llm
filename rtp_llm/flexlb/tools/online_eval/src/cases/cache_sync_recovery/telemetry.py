"""Bounded observation; key fingerprints are read only outside performance windows."""

import math
import datetime
import re
import urllib.request
from diagnostics.process_resources import process_resources

DIAGNOSTIC_METRICS = {
    'global_keys': ('master/cache_global_keys', {}),
    'heap_bytes': ('master/heap_bytes', {'area': 'heap'}),
    'direct_bytes': ('master/direct_bytes', {'id': 'direct'}),
    'diff_scanned_keys': ('master/cache_diff_scanned_keys', {'role': 'PREFILL'}),
    'worker_status_commits': ('master/worker_status_commits', {}),
    'cache_sync_commits': ('master/cache_sync_commits', {'role': 'PREFILL'}),
}


def fetch(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("observation exceeds byte budget")
    return raw


def archived_metrics(store, frames, *, epoch, maximum_gap_s):
    """Join only past, same-epoch archived samples; never fill a gap with future data."""
    from bisect import bisect_right
    result = [dict(row) for row in frames]
    for field, (identity, labels) in DIAGNOSTIC_METRICS.items():
        sources = [r for r in store.document["metrics"].get(identity, []) if r["epoch"] == str(epoch)
                   and all(r["labels"].get(k) == v for k,v in labels.items())]
        indexed = [(r["points"], [p[0] for p in r["points"]]) for r in sources]
        for frame in result:
            values = []
            for points, stamps in indexed:
                i = bisect_right(stamps, frame["epoch_s"])-1
                if i >= 0 and frame["epoch_s"]-stamps[i] <= maximum_gap_s:
                    value = points[i][1]
                    if value is not None and math.isfinite(value):
                        values.append(value)
            frame[field] = sum(values) if values else None
    return result


def resource(proc):
    return process_resources(proc.pid)


def fingerprint_matches(expected, observed):
    workers = {w["ip_port"]: w for w in observed["prefill_endpoints"]}
    matched = len(workers) == len(expected)
    for address, want in expected.items():
        got = workers.get(address, {}).get("fingerprint", {})
        matched = matched and all(got.get(k) == v for k, v in {
            "indexed_keys": want["keys"], "status_keys": want["keys"], "sum": want["sum"],
            "xor": want["xor"], "missing_status": 0, "missing_global": 0}.items())
        matched = matched and workers.get(address, {}).get("cache_version") == want["version"]
    return matched


def interval_rates(before, after, seconds):
    fields = ("responses", "full_responses", "unchanged_responses", "entries", "key_payload_bytes",
              "changed_responses", "added_keys", "removed_keys")
    totals = {key: sum(after[a][key]-before[a][key] for a in after) for key in fields}
    if seconds <= 0 or any(v < 0 for v in totals.values()):
        raise ValueError("invalid fixture accounting interval")
    return dict(duration_s=seconds, totals=totals, **{k: v/seconds for k, v in totals.items()})


def gc_events(path):
    pattern = re.compile(r'^\[([^]]+)\].*?\[gc\s*\]\s+GC\((\d+)\) (Pause .+?)\s+\S+->\S+\([^)]*\) ([\d.]+)ms$')
    result = []
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            stamp, identity, name, duration = match.groups()
            end = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
            result.append(dict(epoch_s=end, duration_ms=float(duration), full='Pause Full' in name,
                               gc_id=int(identity), name=name))
    return result
