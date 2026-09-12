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
        match = re.fullmatch(
            r'mock_engine_rpc_total\{role="(prefill|decode)",rpc_method="fetch_response"\} (\S+)',
            line,
        )
        if match:
            values[match[1] + ".fetch_response"] = float(match[2])
    return values


def observe(mock, windows, interval, out):
    before = read_metrics(mock)
    required = {"decode.mock_engine_completed_total", "decode.mock_engine_generate_tokens_total",
                "prefill.fetch_response", "decode.fetch_response"}
    if not required.issubset(before):
        raise ValueError("missing completion/token/Fetch counters")
    records = []
    previous, sampled_at = before, time.monotonic()
    for _ in range(windows):
        time.sleep(interval)
        current, now = read_metrics(mock), time.monotonic()
        delta = now - sampled_at
        if not required.issubset(current):
            raise ValueError("missing counters during sampling")
        if any(current[k] < previous[k] for k in required):
            raise ValueError("counters reset during sampling; retry after restart")
        records.append({"window_s": delta, "metrics": current,
                        "engine_completed_qps": (current["decode.mock_engine_completed_total"] - previous["decode.mock_engine_completed_total"]) / delta,
                        "generate_tps": (current["decode.mock_engine_generate_tokens_total"] - previous["decode.mock_engine_generate_tokens_total"]) / delta})
        previous, sampled_at = current, now
    report = {"baseline": before, "windows": records,
              "zero_fetch_rpc": all(previous[k] == before[k] for k in required if k.endswith("fetch_response")),
              "note": "Window QPS is not a matched-request success rate."}
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
