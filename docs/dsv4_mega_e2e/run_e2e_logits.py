#!/usr/bin/env python3
"""Logits-level normal-vs-mega compare (steps: baseline|baseline2|mega|compare).

Shares configuration (E2E_CKPT/E2E_GPU/...) with run_e2e_compare.py.

The server returns the LAST decode step's logits, so a comparison is only
valid when both runs generated identical token prefixes before that step
(compare prints prefix_match per query; ignore rows where it is False).
baseline2 reruns the mega-off config to measure the run-to-run noise floor:
compare logits_baseline logits_baseline2.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e2e_compare as e2e  # noqa: E402

PROMPTS = [
    "What is the capital of France?",
    "2+2=",
    "Write a detailed step-by-step explanation of how paged attention "
    "works in LLM inference engines.",
    "The quick brown fox jumps over the lazy dog. Translate to French:",
]


def query_logits(tag: str) -> list:
    import urllib.request

    results = []
    for prompt in PROMPTS:
        body = json.dumps(
            {
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": 8,
                    "top_k": 1,
                    "top_p": 0,
                    "return_logits": True,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{e2e.PORT}/",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        results.append(payload)
        print(f"[{tag}] {prompt!r} keys={sorted(payload.keys())}", flush=True)
    return results


def run(tag: str, extra_env: dict) -> None:
    proc = e2e.start_server(tag, extra_env)
    try:
        if not e2e.wait_ready(proc):
            raise RuntimeError(f"{tag} not ready")
        results = query_logits(tag)
        (e2e.OUT_DIR / f"{tag}.logits.json").write_text(json.dumps(results))
        print(f"[{tag}] saved", flush=True)
    finally:
        e2e.stop_server(proc)


def find_logits(payload):
    if "logits" in payload:
        return payload["logits"]
    aux = payload.get("aux_info")
    if isinstance(aux, dict) and "logits" in aux:
        return aux["logits"]
    return None


def compare(a: str = "logits_baseline", b: str = "logits_mega") -> None:
    import numpy as np

    base = json.loads((e2e.OUT_DIR / f"{a}.logits.json").read_text())
    mega = json.loads((e2e.OUT_DIR / f"{b}.logits.json").read_text())
    for index, (x, y) in enumerate(zip(base, mega)):
        lx, ly = find_logits(x), find_logits(y)
        if lx is None or ly is None:
            print(
                f"query {index}: logits missing "
                f"(base keys={sorted(x)}, mega keys={sorted(y)})"
            )
            continue
        ax = np.asarray(lx, dtype=np.float32).reshape(-1)
        ay = np.asarray(ly, dtype=np.float32).reshape(-1)
        vocab = 129280
        ax = ax.reshape(-1, vocab)
        ay = ay.reshape(-1, vocab)
        steps = min(len(ax), len(ay))
        rx = str(x.get("response"))
        ry = str(y.get("response"))
        print(
            f"query {index} ({PROMPTS[index]!r}): {steps} steps; "
            f"prefix_match={rx[:-8] == ry[:-8]} "
            f"resp_a={rx[:48]!r} resp_b={ry[:48]!r}"
        )
        for t in range(steps):
            xa, ya = ax[t], ay[t]
            denom = float((xa * xa + ya * ya).sum())
            diff = 1.0 - 2.0 * float((xa * ya).sum()) / denom
            sx = np.sort(xa)
            margin = float(sx[-1] - sx[-2])
            same = int(xa.argmax() == ya.argmax())
            print(
                f"  step{t}: calc_diff={diff:.3e} max_abs="
                f"{float(np.abs(xa - ya).max()):.4f} top1_same={same} "
                f"base_top1_margin={margin:.4f}"
            )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "baseline":
        run("logits_baseline", {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "0"})
    elif mode == "baseline2":  # run-to-run noise floor probe
        run("logits_baseline2", {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "0"})
    elif mode == "mega":
        run("logits_mega", {"DSV4_MEGA_CSA": "1", "DSV4_MEGA_HCA": "1"})
    elif mode == "compare":
        compare(*sys.argv[2:4])
