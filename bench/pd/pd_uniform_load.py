#!/usr/bin/env python3
"""Open-loop, evenly spaced (constant-rate) load test for a PD (prefill/decode separated) endpoint.

  pd_uniform_load.py --endpoint 127.0.0.1:14000 --prompts-file prompts_32k_x64_distinct.json \
      --qps 0.5 --duration 120 --max-tokens 1024 --nonce run1 --out load.jsonl [--warmup 2]

Arrival process: deterministic, evenly spaced. Request i is sent at exactly
t0 + i / qps (a fixed inter-arrival gap of 1/qps seconds; no Poisson jitter, no
bursts), independent of whether earlier requests have finished (open loop), for
--duration seconds of arrivals, i.e. round(qps * duration) requests. One
dispatcher coroutine sleeps until each slot and then starts that request, so
sends cannot reorder or bunch; request bodies are JSON-encoded before the run
starts so encoding a 32K prompt never delays a later slot.

Each request takes the next prompt from the file (cycling) with a per-request
nonce: at the start by default, so no two requests share a cache prefix, or at
the end with --nonce-pos suffix, so a shared-prefix prompt set keeps its
prefix-cache hit. Every request decodes exactly
--max-tokens tokens (ignore_eos). --warmup N first sends N requests one at a
time (not counted) so first-request JIT compilation does not land in the run.

Per request the output records the scheduled time, the time the request was
handed to the connection, every SSE chunk's arrival and the server
completion-token counter, so TTFT, time to the second token and the steady TPOT
can be derived.

The summary reports the arrival check (achieved rate, inter-send gap p50 / p99 /
max deviation from 1/qps, worst lateness against the schedule), TTFT / tok2 /
TPOT percentiles over all requests and over the second half (steady state), and
the TTFT / TPOT trend per quarter of the run (a growing TTFT means prefill
cannot keep up with the arrival rate; a growing TPOT means decode cannot).
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def build_body(args, prompt, idx):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_configs": {"ignore_eos": True},
    }
    if args.d_addrs:
        ip, port = args.d_addrs[idx % len(args.d_addrs)].rsplit(":", 1)
        payload["extra_configs"]["role_addrs"] = [
            {
                "role": "DECODE",
                "ip": ip,
                "http_port": int(port),
                "grpc_port": int(port) + 1,
            }
        ]
    return json.dumps(payload).encode()


async def one(session, url, body, idx, t_target):
    t_send = time.time()
    t0 = time.monotonic()
    chunks, toks, err, finish = [], [], None, None
    try:
        async with session.post(
            url, data=body, headers={"Content-Type": "application/json"}
        ) as r:
            if r.status != 200:
                err = "HTTP %d %s" % (r.status, (await r.text())[:200])
            else:
                async for raw in r.content:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        d = json.loads(data)
                    except Exception:
                        continue
                    if "error_code" in d or "error" in d:
                        err = str(d)[:200]
                        break
                    ch = (d.get("choices") or [{}])[0]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    u = (d.get("usage") or {}).get("completion_tokens")
                    delta = (ch.get("delta") or {}).get("content")
                    if delta or (u is not None and (not toks or u > toks[-1])):
                        chunks.append(time.monotonic() - t0)
                        toks.append(
                            u if u is not None else (toks[-1] + 1 if toks else 1)
                        )
    except Exception as e:
        err = repr(e)[:200]
    return {
        "idx": idx,
        "t_target": t_target,
        "t_send": t_send,
        "chunks": chunks,
        "toks": toks,
        "finish": finish,
        "err": err,
    }


def arrival_check(args, rows):
    sends = sorted(r["t_send"] for r in rows if r)
    gap = 1.0 / args.qps
    gaps = [b - a for a, b in zip(sends, sends[1:])]
    late = [r["t_send"] - r["t_target"] for r in rows if r]
    span = sends[-1] - sends[0] if len(sends) > 1 else 0.0
    return {
        "target_qps": args.qps,
        "target_gap_ms": gap * 1000,
        "achieved_send_qps": (len(sends) - 1) / span if span > 0 else float("nan"),
        "send_gap_p50_ms": statistics.median(gaps) * 1000 if gaps else float("nan"),
        "send_gap_p99_ms": pct(gaps, 0.99) * 1000 if gaps else float("nan"),
        "send_gap_max_dev_ms": (
            max(abs(g - gap) for g in gaps) * 1000 if gaps else float("nan")
        ),
        "send_late_max_ms": max(late) * 1000 if late else float("nan"),
    }


def summarize(args, rows, t0):
    ok = [r for r in rows if r and not r["err"] and len(r["chunks"]) >= 3]
    bad = [r for r in rows if not r or r["err"] or len(r["chunks"]) < 3]
    res = {"requests": len(rows), "ok": len(ok), "failed": len(bad)}
    res.update(arrival_check(args, rows))

    def stats(rs, tag):
        ttft = [r["chunks"][0] * 1000 for r in rs]
        tok2 = [r["chunks"][1] * 1000 for r in rs]
        tpot = []
        for r in rs:
            n = r["toks"][-1] - r["toks"][1]
            if n > 0:
                tpot.append((r["chunks"][-1] - r["chunks"][1]) * 1000 / n)
        for name, xs in (("ttft", ttft), ("tok2", tok2), ("tpot", tpot)):
            if xs:
                res["%s_%s_p50" % (tag, name)] = statistics.median(xs)
                res["%s_%s_p90" % (tag, name)] = pct(xs, 0.9)
                res["%s_%s_max" % (tag, name)] = max(xs)
                res["%s_%s_mean" % (tag, name)] = statistics.mean(xs)

    stats(ok, "all")
    half = [r for r in ok if r["t_target"] - t0 >= args.duration / 2]
    stats(half, "late")
    q = max(1, len(ok) // 4)
    by = sorted(ok, key=lambda r: r["idx"])
    res["ttft_p50_by_quarter"] = [
        statistics.median([r["chunks"][0] * 1000 for r in by[i : i + q]])
        for i in range(0, len(by), q)
    ][:4]
    tq = []
    for i in range(0, len(by), q):
        xs = []
        for r in by[i : i + q]:
            n = r["toks"][-1] - r["toks"][1]
            if n > 0:
                xs.append((r["chunks"][-1] - r["chunks"][1]) * 1000 / n)
        if xs:
            tq.append(statistics.median(xs))
    res["tpot_p50_by_quarter"] = tq[:4]
    res["output_tokens"] = sum(r["toks"][-1] for r in ok)
    end = max(r["t_send"] + r["chunks"][-1] for r in ok) if ok else t0
    res["wall_s"] = end - t0
    res["output_tok_per_s"] = (
        res["output_tokens"] / res["wall_s"] if res["wall_s"] > 0 else float("nan")
    )
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--model", default="deepseek_v4")
    ap.add_argument(
        "--prompts-file",
        required=True,
        help="JSON list of prompts (or {'prompts': [...]})",
    )
    ap.add_argument(
        "--qps",
        type=float,
        required=True,
        help="arrival rate; requests are sent every 1/qps s",
    )
    ap.add_argument("--duration", type=float, default=120, help="seconds of arrivals")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--nonce", default="u", help="per-run tag added to every prompt")
    ap.add_argument(
        "--nonce-pos",
        choices=("prefix", "suffix"),
        default="prefix",
        help="prefix: tag at the start, no two requests share a cached prefix (cold prefill); "
        "suffix: tag at the end, so shared-prefix prompt sets keep their prefix-cache hit",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="requests sent one at a time before the run, not counted",
    )
    ap.add_argument("--start-delay", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--d-addrs",
        default="",
        help="comma list of decode DP-rank http ip:port; request i is pinned to entry i %% n "
        "(stands in for an external balancer when the prefill frontend routes every request to one rank)",
    )
    args = ap.parse_args()
    args.d_addrs = [a for a in args.d_addrs.split(",") if a]

    prompts = json.load(open(args.prompts_file))
    if isinstance(prompts, dict):
        prompts = prompts["prompts"]
    url = "http://%s/v1/chat/completions" % args.endpoint
    n = int(round(args.qps * args.duration))

    def tagged(tag, text):
        if args.nonce_pos == "suffix":
            return "%s [%s]" % (text, tag)
        return "[%s] %s" % (tag, text)

    bodies = [
        build_body(
            args,
            tagged(
                "%s-%d" % (args.nonce, i), prompts[(args.offset + i) % len(prompts)]
            ),
            i,
        )
        for i in range(n)
    ]
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        for w in range(args.warmup):
            body = build_body(
                args,
                tagged(
                    "%s-warm%d" % (args.nonce, w),
                    prompts[(args.offset + n + w) % len(prompts)],
                ),
                w,
            )
            r = await one(session, url, body, -1 - w, time.time())
            print(
                "warmup %d: ttft=%s err=%s"
                % (
                    w,
                    "%.0f" % (r["chunks"][0] * 1000) if r["chunks"] else "-",
                    r["err"],
                ),
                flush=True,
            )
        t0 = time.time() + args.start_delay
        tasks = []
        for i in range(n):
            t_target = t0 + i / args.qps
            delay = t_target - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(session, url, bodies[i], i, t_target)))
        rows = await asyncio.gather(*tasks)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = summarize(args, rows, t0)
    print(json.dumps(res, ensure_ascii=False))
    for r in sorted((r for r in rows if r and r["err"]), key=lambda r: r["idx"])[:5]:
        print("ERR idx=%d %s" % (r["idx"], r["err"]))


if __name__ == "__main__":
    asyncio.run(main())
