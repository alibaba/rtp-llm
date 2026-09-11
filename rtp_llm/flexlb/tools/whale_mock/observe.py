"""Observe scheduling acknowledgments and independent completion without Fetch."""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path

import yaml


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


def observe(frontend, mock, cfg, out):
    out.mkdir(parents=True, exist_ok=False)
    before = read_metrics(mock)
    required = [
        "decode.mock_engine_completed_total",
        "decode.mock_engine_generate_tokens_total",
        "prefill.fetch_response",
        "decode.fetch_response",
    ]
    if any(key not in before for key in required):
        raise ValueError("mock metrics missing completion/token/Fetch evidence")
    results, windows = [], []
    start = time.monotonic()

    def send(index):
        body = {
            "input_ids": [index + 1] * cfg["input_tokens"],
            "max_new_tokens": cfg["output_tokens"],
            "timeout_ms": int(cfg["request_timeout_s"] * 1000),
        }
        began = time.monotonic()
        request = urllib.request.Request(
            frontend.rstrip("/") + "/internal/mock/schedule",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=cfg["request_timeout_s"] + 1
            ) as response:
                payload = json.load(response)
                ok = (
                    response.status == 202
                    and payload.get("enqueued_by_master") is True
                    and payload.get("fetch_output_stream") is False
                    and payload.get("inference_completed") is False
                )
                return {
                    "index": index,
                    "ok": ok,
                    "status": response.status,
                    "request_id": payload.get("request_id"),
                    "elapsed_s": time.monotonic() - began,
                }
        except Exception as exc:
            return {
                "index": index,
                "ok": False,
                "error": str(exc),
                "elapsed_s": time.monotonic() - began,
            }

    def sample():
        metrics = read_metrics(mock)
        elapsed = time.monotonic() - start
        previous = windows[-1] if windows else {"elapsed_s": 0, "metrics": before}
        duration = elapsed - previous["elapsed_s"]
        rates = {
            key: (value - previous["metrics"][key]) / duration
            for key, value in metrics.items()
            if key.endswith("_tokens_total") and key in previous["metrics"]
        }
        windows.append(
            {
                "elapsed_s": elapsed,
                "window_s": duration,
                "tokens_per_second": rates,
                "metrics": metrics,
            }
        )
        return metrics

    pending = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(cfg["qps"] * cfg["request_timeout_s"]))
        ) as pool:
            next_sample = start
            for i in range(cfg["requests"]):
                due = start + i / cfg["qps"]
                time.sleep(max(0, due - time.monotonic()))
                pending.append(pool.submit(send, i))
                if time.monotonic() >= next_sample:
                    sample()
                    next_sample = time.monotonic() + cfg["window_s"]
            results = [future.result() for future in pending]
        deadline = time.monotonic() + cfg["drain_timeout_s"]
        while True:
            after = sample()
            completed = after[required[0]] - before[required[0]]
            idle = all(
                after.get(role + ".mock_engine_" + state, -1) == 0
                for role in ("prefill", "decode")
                for state in ("waiting", "running")
            )
            if (completed >= cfg["requests"] and idle) or time.monotonic() >= deadline:
                break
            time.sleep(cfg["window_s"])
        checks = {
            "all_schedules_accepted": len(results) == cfg["requests"]
            and all(r["ok"] for r in results),
            "all_decode_completed": completed == cfg["requests"],
            "output_tokens_accounted": after[required[1]] - before[required[1]]
            == cfg["requests"] * cfg["output_tokens"],
            "zero_fetch_rpc": all(after[k] == before[k] for k in required[2:]),
            "engines_idle": idle,
        }
        report = {
            "mode": "SCHEDULE_ONLY",
            "configuration": cfg,
            "checks": checks,
            "passed": all(checks.values()),
            "baseline": before,
            "requests": results,
            "windows": windows,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "checks": checks,
                    "report": str(out / "report.json"),
                }
            )
        )
        return report["passed"]
    finally:
        (out / "samples.json").write_text(
            json.dumps(
                {"baseline": before, "requests": results, "windows": windows}, indent=2
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--mock", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("bundle.yaml")
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["observation"]
    raise SystemExit(0 if observe(args.frontend, args.mock, cfg, args.out) else 1)
