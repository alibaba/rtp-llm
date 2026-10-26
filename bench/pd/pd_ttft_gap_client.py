#!/usr/bin/env python3
"""TTFT / gap12 / steady-step client for PD-sep KV-transfer accounting.

Each request uses a distinct ~32K prompt from --prompts-file (no prefix reuse).
Per request records every SSE content-chunk arrival (monotonic, relative to
send). Reports TTFT, gap12 (tok1->tok2 = D-side KV-load tail + first decode
step), gap23 and the steady inter-chunk gap (median of chunks 5..end).

Usage:
  pd_ttft_gap_client.py --endpoint 127.0.0.1:14000 --prompts-file F \
      --concurrency 1 --rounds 6 --max-tokens 64 --offset 0 --tag tcp-c1 --out out.json
With --concurrency N each round fires N distinct prompts simultaneously.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def one(ep, prompt, max_tokens, barrier, out, idx):
    body = json.dumps(
        {
            "model": "deepseek_v4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{ep}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    t_send_wall = time.time()
    t0 = time.monotonic()
    chunks = []
    usage = {}
    err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
                except Exception:
                    continue
                if "error_code" in d:
                    raise RuntimeError(str(d)[:200])
                if d.get("usage"):
                    usage = d["usage"]
                delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get(
                    "content"
                )
                if delta:
                    chunks.append(time.monotonic() - t0)
    except Exception as e:
        err = repr(e)[:200]
    out[idx] = {
        "t_send_wall": t_send_wall,
        "chunks": chunks,
        "err": err,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def summarize(r):
    c = r["chunks"]
    if r["err"] or len(c) < 3:
        return None
    gaps = [b - a for a, b in zip(c, c[1:])]
    steady = gaps[4:] if len(gaps) > 6 else gaps[1:]
    return {
        "ttft_ms": c[0] * 1000,
        "gap12_ms": gaps[0] * 1000,
        "gap23_ms": gaps[1] * 1000,
        "step_ms": statistics.median(steady) * 1000,
        "prompt_tokens": r["prompt_tokens"],
    }


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="first prompt index (keeps prompts unique across runs)",
    )
    ap.add_argument(
        "--nonce",
        default="",
        help="prefix appended to every prompt to defeat any prefix reuse",
    )
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    prompts = json.load(open(a.prompts_file))
    need = a.offset + a.concurrency * a.rounds
    all_rows = []
    for rd in range(a.rounds):
        barrier = threading.Barrier(a.concurrency)
        res = [None] * a.concurrency
        ths = []
        for i in range(a.concurrency):
            k = a.offset + rd * a.concurrency + i
            p = prompts[k % len(prompts)]
            if a.nonce or k >= len(prompts):
                p = f"[{a.nonce}-{k}] " + p
            ths.append(
                threading.Thread(
                    target=one, args=(a.endpoint, p, a.max_tokens, barrier, res, i)
                )
            )
        [t.start() for t in ths]
        [t.join() for t in ths]
        for i, r in enumerate(res):
            r["round"] = rd
            r["slot"] = i
            s = summarize(r)
            r["summary"] = s
            all_rows.append(r)
            if s:
                print(
                    f"{a.tag} r{rd} s{i} ISL={s['prompt_tokens']} TTFT={s['ttft_ms']:.0f}ms "
                    f"gap12={s['gap12_ms']:.1f} gap23={s['gap23_ms']:.1f} step={s['step_ms']:.1f}",
                    flush=True,
                )
            else:
                print(
                    f"{a.tag} r{rd} s{i} FAIL err={r['err']} chunks={len(r['chunks'])}",
                    flush=True,
                )
    ok = [r["summary"] for r in all_rows if r["summary"]]
    print(
        f"== {a.tag}: ok={len(ok)}/{len(all_rows)} (need prompts {need}, have {len(prompts)})"
    )
    if ok:
        for k in ("ttft_ms", "gap12_ms", "gap23_ms", "step_ms"):
            xs = [s[k] for s in ok]
            print(
                f"   {k:9s} p50={statistics.median(xs):8.1f}  mean={statistics.mean(xs):8.1f}  "
                f"min={min(xs):8.1f}  max={max(xs):8.1f}  p90={pct(xs, 0.9):8.1f}"
            )
        exposed = [s["gap12_ms"] - s["step_ms"] for s in ok]
        print(
            f"   exposed_kv(gap12-step) p50={statistics.median(exposed):.1f} mean={statistics.mean(exposed):.1f} "
            f"min={min(exposed):.1f} max={max(exposed):.1f}"
        )
    if a.out:
        json.dump({"tag": a.tag, "args": vars(a), "rows": all_rows}, open(a.out, "w"))
        print("saved:", a.out)


if __name__ == "__main__":
    main()
