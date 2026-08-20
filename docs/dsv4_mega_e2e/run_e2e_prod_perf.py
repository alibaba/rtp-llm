#!/usr/bin/env python3
"""Production-shape latency/throughput A/B: normal vs mega (CSA+HCA).

Production shape here means: CUDA Graph decode (capture sizes cover the
tested concurrencies), fp8 KV, greedy, concurrent streams. Attention runs
TP1 per rank in production (dp/ep shard MoE, not attention), so a single-GPU
graph+batch run is representative for the mega attention path; full-machine
dp8/ep8 still needs a dedicated window.

Environment: same as run_e2e_compare.py (E2E_CKPT required, E2E_GPU, ...).
Extra: E2E_CONCURRENCIES (default "1,8,32"), E2E_NEW_TOKENS (default 128).
"""

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e2e_compare as e2e  # noqa: E402

CONCURRENCIES = [
    int(x) for x in os.environ.get("E2E_CONCURRENCIES", "1,8,32").split(",")
]
NEW_TOKENS = int(os.environ.get("E2E_NEW_TOKENS", "128"))
PROMPT = (
    "Write a detailed step-by-step explanation of how paged attention "
    "works in LLM inference engines."
)

# Override the compare script's eager server args: enable CUDA Graph and
# capture exactly the tested batch sizes; allow enough concurrency.
GRAPH_ARGS = []
_skip = 0
for _arg in e2e.SERVER_ARGS:
    if _skip:
        _skip -= 1
        continue
    if _arg in (
        "--enable_cuda_graph",
        "--concurrency_limit",
        "--max_context_batch_size",
    ):
        _skip = 1
        continue
    GRAPH_ARGS.append(_arg)
GRAPH_ARGS += [
    "--enable_cuda_graph",
    "1",
    "--decode_capture_config",
    ",".join(str(c) for c in sorted(set(CONCURRENCIES))),
    "--concurrency_limit",
    str(max(CONCURRENCIES) * 2),
    "--max_context_batch_size",
    "4",
]
e2e.SERVER_ARGS = GRAPH_ARGS


def one_request(results, index):
    body = json.dumps(
        {
            "prompt": PROMPT,
            "generate_config": {
                "max_new_tokens": NEW_TOKENS,
                "top_k": 1,
                "top_p": 0,
                "aux_info": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{e2e.PORT}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        payload = json.loads(resp.read())
    results[index] = payload.get("aux_info", {})


def bench(tag: str, concurrency: int) -> dict:
    results = [None] * concurrency
    threads = [
        threading.Thread(target=one_request, args=(results, i))
        for i in range(concurrency)
    ]
    wall = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - wall
    total_out = sum(r.get("output_len", 0) for r in results if r)
    decode_ms = [
        (r["cost_time"] - r["first_token_cost_time"]) / max(r["output_len"] - 1, 1)
        for r in results
        if r and r.get("output_len", 0) > 1
    ]
    row = {
        "concurrency": concurrency,
        "wall_s": round(wall, 2),
        "agg_tps": round(total_out / wall, 1),
        "decode_ms_per_token_avg": round(sum(decode_ms) / len(decode_ms), 2),
        "decode_ms_per_token_max": round(max(decode_ms), 2),
    }
    print(f"[{tag}] C={concurrency}: {row}", flush=True)
    return row


def run(tag: str, extra_env: dict) -> list:
    proc = e2e.start_server(tag, extra_env)
    rows = []
    try:
        if not e2e.wait_ready(proc):
            raise RuntimeError(f"{tag} not ready")
        # Warmup: exercise every captured graph size + JIT once.
        for c in sorted(set(CONCURRENCIES)):
            bench(f"{tag}/warmup", c)
        for c in CONCURRENCIES:
            rows.append(bench(tag, c))
        (e2e.OUT_DIR / f"{tag}.prod_perf.json").write_text(json.dumps(rows))
    finally:
        e2e.stop_server(proc)
    return rows


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "baseline":
        run("prod_baseline", {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "0"})
    elif mode == "mega":
        run("prod_mega", {"DSV4_MEGA_CSA": "1", "DSV4_MEGA_HCA": "1"})
    elif mode == "compare":
        base = json.loads((e2e.OUT_DIR / "prod_baseline.prod_perf.json").read_text())
        mega = json.loads((e2e.OUT_DIR / "prod_mega.prod_perf.json").read_text())
        print(
            f"{'C':>4} {'base tps':>9} {'mega tps':>9} {'speedup':>8} "
            f"{'base ms/tok':>12} {'mega ms/tok':>12} {'latency':>8}"
        )
        for b, m in zip(base, mega):
            print(
                f"{b['concurrency']:>4} {b['agg_tps']:>9} {m['agg_tps']:>9} "
                f"{m['agg_tps'] / b['agg_tps'] - 1:>+7.1%} "
                f"{b['decode_ms_per_token_avg']:>12} "
                f"{m['decode_ms_per_token_avg']:>12} "
                f"{m['decode_ms_per_token_avg'] / b['decode_ms_per_token_avg'] - 1:>+7.1%}"
            )
