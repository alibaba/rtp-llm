#!/usr/bin/env python3
"""Decode-only LargeEP benchmark through PD (P -> D EP16), SGLang-style slow-down gate.

1. Create the hold file on both D nodes (D runs only 1-row fake decode rounds; real streams queue).
2. Send N requests to P, request i pinned to D DP leader i % 4 (role_addrs), so every DP group gets N/4.
3. Wait until every request has its first token (prefill + KV handoff done) and D has them all queued.
4. Remove the hold file on both D nodes at the same time -> all groups start decoding the same batch.
5. Measure decode plateau from the token stream (step = SSE chunk interval with --force-accept; real TPOT
   otherwise), same analysis as decode_plateau_bench.

usage: pd_decode_gate.py --p-url http://<prefill host>:<port> --d-leaders ip:port,... --cc 64 \
         --prompts-file prompts.json --max-tokens 2048 [--force-accept --gen 2]
The hold file is toggled through ssh-less HTTP: each D node runs gate_agent.py (tiny HTTP server) on --gate-port.
"""
import argparse
import asyncio
import json
import sys
import time

import aiohttp


async def gate(session, hosts, port, on):
    async def one(h):
        async with session.post(
            f"http://{h}:{port}/{'hold' if on else 'release'}"
        ) as r:
            return await r.text()

    return await asyncio.gather(*[one(h) for h in hosts])


async def queued(session, hosts, port):
    out = {}
    for h in hosts:
        try:
            async with session.get(f"http://{h}:{port}/queued") as r:
                for k, v in json.loads(await r.text()).items():
                    out[f"{h}/dp{k}"] = v
        except Exception:
            pass
    return out


async def one_request(session, args, prompt, rid, t_ref, first_evt):
    ip, port = args.d_leaders[rid % len(args.d_leaders)].rsplit(":", 1)
    payload = {
        "model": "deepseek_v4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "extra_configs": dict(
            {"ignore_eos": True},
            **({"force_sp_accept": True} if args.force_accept else {}),
            role_addrs=[
                {
                    "role": "DECODE",
                    "ip": ip,
                    "http_port": int(port),
                    "grpc_port": int(port) + 1,
                }
            ],
        ),
    }
    token_ts, err, finish = [], None, None
    try:
        async with session.post(args.p_url + "/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    d = json.loads(data)
                except Exception:
                    continue
                ch = (d.get("choices") or [{}])[0]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if "error_code" in d or "error" in d:
                    err = str(d)[:200]
                u = (d.get("usage") or {}).get("completion_tokens")
                if u is not None and (not token_ts or u > token_ts[-1][1]):
                    token_ts.append((time.monotonic() - t_ref, u))
                    if len(token_ts) == 1:
                        first_evt.set()
    except Exception as e:
        err = repr(e)[:200]
    first_evt.set()
    return {"rid": rid, "token_ts": token_ts, "finish": finish, "err": err}


def analyze(args, results, t_release):
    tps = args.gen + 1 if args.force_accept else 1
    ok = [r for r in results if len(r["token_ts"]) > 3 and not r["err"]]
    for r in sorted(results, key=lambda r: r["rid"]):
        ts = r["token_ts"]
        print(
            f"rid{r['rid']} grp{r['rid'] % len(args.d_leaders)} chunks={len(ts)} toks={ts[-1][1] if ts else 0} "
            f"first={ts[0][0] if ts else -1:.1f} last={ts[-1][0] if ts else -1:.1f} finish={r['finish']} err={r['err']}",
            file=sys.stderr,
        )
    if not ok:
        return {"cc": args.cc, "error": "no responses"}
    # decode window: from release + skip to the first request end - 0.5s
    t_end = min(r["token_ts"][-1][0] for r in ok)
    w0, w1 = t_release + args.skip_s, min(
        t_release + args.skip_s + args.window_s, t_end - 0.5
    )
    if w1 - w0 < 2:
        return {"cc": args.cc, "error": f"window too short {w1 - w0:.1f}s"}

    def toks_at(r, t):
        best = 0
        for ts, u in r["token_ts"]:
            if ts <= t:
                best = u
            else:
                break
        return best

    # Rates only over dense streams: a forced-accept stream whose text the detokenizer holds back emits a
    # handful of SSE chunks, so its in-window rate reads ~0 and would bias the mean (same filter as
    # decode_plateau_bench).
    dense = [
        r
        for r in ok
        if len(r["token_ts"]) > 0.5 * r["token_ts"][-1][1] / (args.gen + 1)
    ]
    rates = [(toks_at(r, w1) - toks_at(r, w0)) / (w1 - w0) for r in dense]
    chunks = []
    for r in ok:
        c = [ts for ts, _ in r["token_ts"] if w0 <= ts <= w1]
        if len(c) > 2:
            chunks.append((c[-1] - c[0]) / (len(c) - 1))
    chunks.sort()
    accept = sorted(r["token_ts"][-1][1] / len(r["token_ts"]) for r in ok)
    out = {
        "cc": args.cc,
        "ok_reqs": len(ok),
        "dense_reqs": len(dense),
        "window_s": round(w1 - w0, 1),
        "t_release_s": round(t_release, 1),
        "agg_tok_per_s": round(sum(rates), 1),
        "tpot_ms": round(1000 * len(rates) / sum(rates), 2) if sum(rates) > 0 else None,
        "step_ms_p50": round(1000 * chunks[len(chunks) // 2], 2) if chunks else None,
        "step_ms_mean": round(1000 * sum(chunks) / len(chunks), 2) if chunks else None,
        "tokens_per_step_p50": round(accept[len(accept) // 2], 3),
        "forced": bool(args.force_accept),
    }
    if args.force_accept:
        out["step_ms_from_rate"] = round(1000 * tps * len(rates) / sum(rates), 2)
    return out


async def run(args):
    prompts = json.load(open(args.prompts_file))
    assert len(prompts) >= args.prompt_offset + args.cc
    conn = aiohttp.TCPConnector(limit=512)
    async with aiohttp.ClientSession(
        connector=conn, timeout=aiohttp.ClientTimeout(total=14400)
    ) as s:
        print(
            "hold:",
            await gate(s, args.gate_hosts, args.gate_port, True),
            file=sys.stderr,
        )
        await asyncio.sleep(1.0)
        t_ref = time.monotonic()
        evts = [asyncio.Event() for _ in range(args.cc)]
        tasks = [
            asyncio.create_task(
                one_request(s, args, prompts[args.prompt_offset + i], i, t_ref, evts[i])
            )
            for i in range(args.cc)
        ]
        # first token of every request = prefill + KV handoff done (D emits P's first token on admission)
        deadline = time.monotonic() + args.prefill_timeout_s
        while not all(e.is_set() for e in evts) and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        n_ready = sum(e.is_set() for e in evts)
        # D side: every DP group must report its full share held before release
        want = args.cc // len(args.d_leaders)
        q = {}
        while time.monotonic() < deadline:
            q = await queued(s, args.gate_hosts, args.gate_port)
            if len(q) >= len(args.d_leaders) and all(v >= want for v in q.values()):
                break
            await asyncio.sleep(0.5)
        else:
            print(f"WARNING: releasing before every group held {want}", file=sys.stderr)
        print(f"D held per group: {q} (want {want})", file=sys.stderr)
        await asyncio.sleep(args.settle_s)
        t_release = time.monotonic() - t_ref
        print(
            f"release at {t_release:.1f}s, ready {n_ready}/{args.cc}", file=sys.stderr
        )
        print(
            "release:",
            await gate(s, args.gate_hosts, args.gate_port, False),
            file=sys.stderr,
        )
        results = await asyncio.gather(*tasks)
    print(json.dumps(analyze(args, results, t_release), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-url", required=True)
    ap.add_argument(
        "--d-leaders", required=True, help="comma list ip:http_port of the D DP leaders"
    )
    ap.add_argument(
        "--gate-hosts",
        required=True,
        help="comma list of the D hosts running gate_agent.py",
    )
    ap.add_argument("--gate-port", type=int, default=31999)
    ap.add_argument("--cc", type=int, default=64)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--force-accept", action="store_true")
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--prefill-timeout-s", type=float, default=600)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--skip-s", type=float, default=3.0)
    ap.add_argument("--window-s", type=float, default=25.0)
    args = ap.parse_args()
    args.d_leaders = [a for a in args.d_leaders.split(",") if a]
    args.gate_hosts = [a for a in args.gate_hosts.split(",") if a]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
