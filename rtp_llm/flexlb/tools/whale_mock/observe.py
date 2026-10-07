"""Read existing bundled traffic counters without sending requests or Fetch RPCs."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

def read_metrics(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
        raw = response.read().decode()
    values = {}
    for line in raw.splitlines():
        match = re.fullmatch(
            r'([a-zA-Z_:][a-zA-Z0-9_:]*)\{role="(prefill|decode)"\} (\S+)', line
        )
        if match:
            values[match[2] + "." + match[1]] = float(match[3])
    # Fetch is a protocol invariant, not a dashboard series. Read the existing
    # diagnostic snapshot; absent role/counter data must never mean zero Fetches.
    with urllib.request.urlopen(url.rstrip("/") + "/snapshot", timeout=5) as response:
        snapshot = json.load(response)
    for engine in snapshot["engines"]:
        role = engine["role"]
        if role not in {"prefill", "decode"}:
            continue
        count = engine["rpc_counts"]["fetch_response"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("invalid Fetch counter in snapshot")
        key = role + ".fetch_response"
        values[key] = values.get(key, 0) + count
    return values


def observe(mock, windows, interval, out):
    before = read_metrics(mock)
    required = {"decode.mock_engine_completed_total", "decode.rtp_llm_generate_tps",
                "prefill.fetch_response", "decode.fetch_response"}
    if not required.issubset(before):
        raise ValueError("missing completion/token/Fetch counters")
    counters = required - {"decode.rtp_llm_generate_tps"}
    records = []
    previous, sampled_at = before, time.monotonic()
    for _ in range(windows):
        time.sleep(interval)
        current, now = read_metrics(mock), time.monotonic()
        delta = now - sampled_at
        if not required.issubset(current):
            raise ValueError("missing counters during sampling")
        if any(current[k] < previous[k] for k in counters):
            raise ValueError("counters reset during sampling; retry after restart")
        records.append({"window_s": delta, "metrics": current,
                        "engine_completed_qps": (current["decode.mock_engine_completed_total"] - previous["decode.mock_engine_completed_total"]) / delta,
                        "generate_tps": current["decode.rtp_llm_generate_tps"]})
        previous, sampled_at = current, now
    report = {"baseline": before, "windows": records,
              "zero_fetch_rpc": all(previous[k] == before[k] == 0 for k in counters if k.endswith("fetch_response")),
              "note": "Window QPS is not a matched-request success rate. generate_tps is the engine-reported rate, not a counter delta."}
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.windows < 1 or args.interval <= 0:
        parser.error("positive windows and interval required")
    observe(args.mock, args.windows, args.interval, args.out)
