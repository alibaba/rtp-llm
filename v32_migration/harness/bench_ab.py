#!/usr/bin/env python3
"""Minimal TPOT probe for the A/B lossy-offload iteration.

Usage: bench_ab.py [--port 26100] [--ctx 63000] [--out 128] [--reps 2] [--tag ...]
Prints per-rep aux_info-derived TPOT and the mean; appends jsonl.
"""
import argparse
import json
import statistics as st
import time
import urllib.request

HEAD = "请把下面的文本原样复述一遍并总结要点。"


def make_prompt(nchars):
    return HEAD + ("深度学习模型推理性能分析。" * (nchars // 13 + 1))[:nchars]


def post(port, prompt, max_new, timeout=2400):
    body = json.dumps(
        {
            "prompt": prompt,
            "generate_config": {"max_new_tokens": max_new, "top_k": 1},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=26100)
    ap.add_argument("--ctx", type=int, default=63000)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="/home/admin/rtp-hol/logs/ab_bench.jsonl")
    a = ap.parse_args()

    probe, _ = post(a.port, make_prompt(4000), 1)
    il = (probe.get("aux_info") or {}).get("input_len") or 0
    ratio = il / 4019 if il else 0.463
    print(f"calib ratio={ratio:.3f}")

    prompt = make_prompt(int(a.ctx / ratio))
    rows = []
    for r in range(a.reps):
        out, wall = post(a.port, prompt, a.out)
        aux = out.get("aux_info") or {}
        ol = aux.get("output_len") or a.out
        cost, pre = aux.get("cost_time"), aux.get("prefill_time") or aux.get(
            "first_token_cost_time"
        )
        tpot = (cost - pre) / (ol - 1) if (cost and pre and ol > 1) else None
        rows.append(
            {
                "il": aux.get("input_len"),
                "ol": ol,
                "wall_s": round(wall, 2),
                "prefill_ms": pre and round(pre, 1),
                "cost_ms": cost and round(cost, 1),
                "tpot_ms": tpot and round(tpot, 2),
                "iter": aux.get("iter_count"),
            }
        )
        print(" ", rows[-1])
    tp = [r["tpot_ms"] for r in rows if r["tpot_ms"]]
    mean = st.mean(tp) if tp else None
    print(f"TAG={a.tag} ctx={a.ctx} TPOT_mean={mean} ms (n={len(tp)})")
    with open(a.jsonl, "a") as f:
        f.write(
            json.dumps(
                {
                    "tag": a.tag,
                    "ctx": a.ctx,
                    "tpot_mean": mean,
                    "rows": rows,
                    "ts": time.time(),
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
