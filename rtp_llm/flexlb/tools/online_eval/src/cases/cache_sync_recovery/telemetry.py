"""Bounded observation; key fingerprints are read only outside performance windows."""

import math
import re
import urllib.request
from diagnostics.process_resources import process_resources

DIAGNOSTIC_METRICS = {
    'global_keys': ('cache_global_total_count', {}),
    'heap_bytes': ('jvm_memory_used_bytes', {'area': 'heap'}),
    'direct_bytes': ('jvm_buffer_memory_used_bytes', {'id': 'direct'}),
    'diff_scanned_keys': ('cache_diff_scanned_keys_total', {'role': 'PREFILL'}),
    'worker_status_commits': ('engine_health_check_visitor_success_qps_total', {}),
    'cache_sync_commits': ('cache_status_check_visitor_success_qps_total', {'role': 'PREFILL'}),
}


def fetch(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("observation exceeds byte budget")
    return raw


def metric_values(raw):
    result = []
    for line in raw.decode().splitlines():
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([-+0-9.eE]+)', line)
        if not match:
            continue
        name, labels, value = match.groups()
        if not any(fragment in name for fragment, _ in DIAGNOSTIC_METRICS.values()):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(dict(name=name, labels=dict(re.findall(r'(\w+)="([^"]*)"', labels or "")), value=number))
    return result


def metric_sum(meters, fragment, labels=None):
    values = [m["value"] for m in meters if fragment in m["name"]
              and all(m["labels"].get(k) == v for k, v in (labels or {}).items())]
    return sum(values) if values else None


def diagnostic_metrics(raw):
    meters = metric_values(raw)
    return {field: metric_sum(meters, fragment, labels)
            for field, (fragment, labels) in DIAGNOSTIC_METRICS.items()}


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
