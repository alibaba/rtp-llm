"""Bounded observation; key fingerprints are read only outside performance windows."""

import math
import re
import urllib.request
from pathlib import Path


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
        if not any(x in name for x in ("jvm_memory_used_bytes", "jvm_buffer_memory_used_bytes",
                "cache_global_total_count", "cache_diff_scanned_keys_total", "engine_health_check_visitor_success_qps_total",
                "cache_status_check_visitor_success_qps_total")):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(dict(name=name, labels=dict(re.findall(r'(\w+)="([^"]*)"', labels or "")), value=number))
    return result


def metric_sum(meters, fragment, labels=None):
    values = [m["value"] for m in meters if fragment in m["name"]
              and all(m["labels"].get(k) == v for k, v in (labels or {}).items())]
    return sum(values) if values else None


def resource(proc):
    pid = proc.pid
    if not Path(f"/proc/{pid}/cgroup").exists():
        return dict(cpu_usec=None, memory_bytes=None, rss_bytes=None, cpu_max=None, memory_max=None,
                    unavailable="per-process Linux cgroup v2 evidence unavailable")
    path = next(line.split("::", 1)[1] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / path.lstrip("/")
    cpu = {k: int(v) for k, v in (line.split() for line in (group/"cpu.stat").read_text().splitlines())}
    rss = next(int(line.split()[1])*1024 for line in Path(f"/proc/{pid}/status").read_text().splitlines()
               if line.startswith("VmRSS:"))
    return dict(cpu_usec=cpu["usage_usec"], throttled_usec=cpu.get("throttled_usec"),
                memory_bytes=int((group/"memory.current").read_text()), rss_bytes=rss,
                cpu_max=(group/"cpu.max").read_text().strip(), memory_max=(group/"memory.max").read_text().strip())


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
