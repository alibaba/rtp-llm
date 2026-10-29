#!/usr/bin/env python3
"""Concurrent prefill-only throughput against a standalone RTP-LLM prefill engine.

Fires --n requests (generate_config.max_new_tokens=0, or --max-new-tokens) with at most --cc in flight. Reports per-request wall time
and the completion rate in the steady part of the run: (completed - 1) / (t_last_done - t_first_done).

Prompt files: a JSON list of prompts, or the {"prompts": [...]} output of make_prefix_prompts.py.
--nonce-pos prefix (default) puts the per-request nonce first so prefix reuse never hits; --nonce-pos suffix
appends it so a shared-prefix set keeps its cache hit. Prefill-only requests (max_new_tokens=0) never reuse
the prefix cache (StreamCacheResource::reuseCache), so prefix-hit runs need --max-new-tokens 1. --warmup N first sends N requests one at a time (not
counted; with a shared-prefix set this also puts the prefix in the cache). The server's reuse_len is recorded.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:23000/")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cc", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--nonce", default=str(int(time.time())))
    ap.add_argument("--nonce-pos", choices=("prefix", "suffix"), default="prefix")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 = prefill-only; use 1 for prefix-hit runs (prefill-only requests never reuse the prefix cache)",
    )
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    prompts = json.load(open(a.prompts_file))
    if isinstance(prompts, dict):
        prompts = prompts["prompts"]

    def body_for(i, tag):
        p = prompts[(a.offset + i) % len(prompts)]
        p = (
            ("[%s-%d] " % (tag, i) + p)
            if a.nonce_pos == "prefix"
            else (p + " [%s-%d]" % (tag, i))
        )
        return {
            "prompt": p,
            "generate_config": {
                "max_new_tokens": a.max_new_tokens,
                "temperature": 0,
                "top_k": 1,
            },
        }

    def aux_of(raw):
        info = json.loads(raw)
        return info.get("aux_info", {}) if isinstance(info, dict) else {}

    for w in range(a.warmup):
        ts = time.time()
        aux = aux_of(
            post(a.url, body_for(len(prompts) - 1 - w, a.nonce + "w"), a.timeout)
        )
        print(
            json.dumps(
                {
                    "warmup": w,
                    "wall_ms": round((time.time() - ts) * 1000, 1),
                    "reuse_len": aux.get("reuse_len"),
                }
            ),
            flush=True,
        )

    sem = threading.Semaphore(a.cc)
    recs = []
    lock = threading.Lock()
    t0 = time.time()

    def one(i):
        with sem:
            ts = time.time()
            try:
                aux = aux_of(post(a.url, body_for(i, a.nonce), a.timeout))
                rec = {
                    "i": i,
                    "start_s": round(ts - t0, 3),
                    "done_s": round(time.time() - t0, 3),
                    "wall_ms": round((time.time() - ts) * 1000, 1),
                    "input_len": aux.get("input_len"),
                    "cost_time_ms": aux.get("cost_time"),
                    "reuse_len": aux.get("reuse_len"),
                }
            except Exception as e:
                rec = {"i": i, "error": repr(e)[:300]}
        with lock:
            recs.append(rec)
            print(json.dumps(rec), flush=True)

    th = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    ok = sorted((r for r in recs if "done_s" in r), key=lambda r: r["done_s"])
    out = {"summary": True, "n": a.n, "cc": a.cc, "n_ok": len(ok)}
    if ok:
        out["reuse_len_set"] = sorted(
            {r["reuse_len"] for r in ok if r.get("reuse_len") is not None}
        )
    if len(ok) >= 2:
        span = ok[-1]["done_s"] - ok[0]["done_s"]
        out.update(
            {
                "wall_p50_ms": round(statistics.median(r["wall_ms"] for r in ok), 1),
                "first_done_s": ok[0]["done_s"],
                "last_done_s": ok[-1]["done_s"],
                "steady_req_per_s": (
                    round((len(ok) - 1) / span, 3) if span > 0 else None
                ),
                "overall_req_per_s": round(len(ok) / ok[-1]["done_s"], 3),
                "done_gap_p50_ms": round(
                    1000
                    * statistics.median(
                        b["done_s"] - x["done_s"] for x, b in zip(ok, ok[1:])
                    ),
                    1,
                ),
            }
        )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
