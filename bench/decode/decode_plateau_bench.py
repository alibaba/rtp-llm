#!/usr/bin/env python3
"""Decode-only plateau benchmark for an OpenAI-compatible chat endpoint.

All ``--cc`` requests are sent at once with ``ignore_eos`` so each decodes
exactly ``--max-tokens``. Once the last request has its first token the whole
batch decodes together; the step time is measured inside that plateau from the
``usage.completion_tokens`` counter of every SSE chunk (the frontend may merge
several engine steps into one chunk, so chunk intervals alone are not used).
The concurrency reconstructed from token timestamps is printed as proof that
the batch was full during the window.

Output (one JSON line on stdout):
  measured_tpot_ms / measured_tok_per_s   from the token counters in the window
                                          (what the client actually sees);
  observed_tokens_per_step, measured_step_ms   tokens per SSE chunk (= per
                                          engine step when the stream is dense)
                                          and the implied step time;
  step_mean_ms, tpot_eff_ms, agg_tok_per_s   the simulated numbers for
                                          ``--force-accept``: every step emits
                                          ``--tokens-per-step`` tokens and is
                                          rescaled to ``--accept-len``.

For RTP-LLM with MTP, ``--force-accept`` sends ``force_sp_accept`` so every
draft token is accepted; set ``--tokens-per-step`` to gen_num_per_cycle + 1
and ``--accept-len`` to the acceptance length you want to report. When the
server itself forces a fractional acceptance (PPU decode with
``DSV4_SP_FORCE_ACCEPT_LEN``), do not pass ``--force-accept``; use the
``measured_*`` fields (``observed_tokens_per_step`` shows the acceptance).
"""
import argparse
import asyncio
import json
import sys
import time

import aiohttp

SEG = (
    "杭州是一座融合千年宋韵文化与现代数字经济的城市，西湖、钱塘江与数字经济产业共同构成了它的城市名片。"
    "请从地理、历史、经济、文化、科技五个角度详细分析这座城市的独特魅力与发展潜力。"
)


def build_prompt(target_tokens, salt, unique_tokens=512):
    """Shared prefix + per-request unique suffix (used without --prompts-file).

    Decode still attends over the full ~target_tokens context; with prefix
    reuse on, only the unique suffix is prefilled."""
    common = SEG * max(1, int((target_tokens - unique_tokens) * 1.83 / len(SEG)))
    uniq = (f"第{salt}号补充材料：" + SEG) * max(
        1, int(unique_tokens * 1.83 / (len(SEG) + 8))
    )
    return common + uniq + f"本次分析编号{salt}。请开始你的回答。"


async def one_request(session, args, prompt, rid, t_ref):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if args.vllm:
        payload.update(
            {
                "ignore_eos": True,
                "stream_options": {
                    "include_usage": True,
                    "continuous_usage_stats": True,
                },
            }
        )
    else:
        payload["extra_configs"] = dict(
            {"ignore_eos": True},
            **({"force_sp_accept": True} if args.force_accept else {}),
        )
    token_ts = []
    finish = None
    err = None
    try:
        async with session.post(args.url + "/v1/chat/completions", json=payload) as r:
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
    except Exception as e:
        err = repr(e)[:150]
    return {"rid": rid, "token_ts": token_ts, "finish": finish, "err": err}


def analyze(args, results):
    for r in sorted(results, key=lambda r: r["rid"]):
        ts = r["token_ts"]
        print(
            f"cc{args.cc} rid{r['rid']} chunks={len(ts)} toks={ts[-1][1] if ts else 0} "
            f"first={ts[0][0] if ts else -1:.1f} last={ts[-1][0] if ts else -1:.1f} "
            f"finish={r.get('finish')} err={r.get('err')}",
            file=sys.stderr,
        )
    # Requests that stop early (EOS edge cases, stop strings) do not cover the
    # plateau; measure on the ones that ran to max_tokens when enough did.
    full = [
        r
        for r in results
        if len(r["token_ts"]) > 1 and r["token_ts"][-1][1] >= args.max_tokens
    ]
    if len(full) >= max(1, int(0.75 * len(results))):
        results = full
    else:
        results = [r for r in results if len(r["token_ts"]) > 1]
    if not results:
        return {"cc": args.cc, "error": "no responses"}
    t_all = max(r["token_ts"][0][0] for r in results)
    t_first_end = min(r["token_ts"][-1][0] for r in results)
    win_start = t_all + args.plateau_skip_s
    win_end = min(win_start + args.plateau_s, t_first_end - 0.5)

    def toks_at(r, t):
        best = 0
        for ts, u in r["token_ts"]:
            if ts <= t:
                best = u
            else:
                break
        return best

    def conc(t):
        return sum(
            1 for r in results if r["token_ts"][0][0] <= t <= r["token_ts"][-1][0]
        )

    out = {
        "cc": args.cc,
        "ok_reqs": len(results),
        "window_s": round(win_end - win_start, 1),
        "t_all_prefilled_s": round(t_all, 1),
    }
    if win_end - win_start < 2:
        out["error"] = "no plateau window"
        return out
    probes = [win_start + k * (win_end - win_start) / 10 for k in range(11)]
    # Per-request first->last token span, used when SSE chunks are too sparse
    # for the window method; valid because all requests decode concurrently.
    spans = []
    for r in results:
        (t0, u0), (t1, u1) = r["token_ts"][0], r["token_ts"][-1]
        if u1 > u0 and t1 > t0:
            spans.append((t1 - t0) / ((u1 - u0) / args.tokens_per_step))
    dense = [
        r
        for r in results
        if len(r["token_ts"]) > 0.5 * r["token_ts"][-1][1] / args.tokens_per_step
    ]
    rates = [
        (toks_at(r, win_end) - toks_at(r, win_start)) / (win_end - win_start)
        for r in dense
    ]
    if dense and min(rates) > 0:
        step_ms = 1000 * args.tokens_per_step * len(rates) / sum(rates)
        method = f"window({len(dense)}dense)"
    else:
        step_ms = 1000 * sum(spans) / len(spans)
        method = "span"
    overlap = (min(r["token_ts"][-1][0] for r in results) - t_all) / max(
        1e-6,
        max(r["token_ts"][-1][0] for r in results)
        - min(r["token_ts"][0][0] for r in results),
    )
    # Tokens actually emitted per engine step, from the SSE chunk stream (one
    # chunk per step when the stream is dense).  With PPU
    # DSV4_SP_FORCE_ACCEPT_LEN the engine already emits accept_len tokens per
    # step, so --tokens-per-step (gen+1, the force_sp_accept assumption) would
    # overstate the step and understate the throughput by (gen+1)/accept_len.
    toks_per_chunk = sorted(
        r["token_ts"][-1][1] / len(r["token_ts"]) for r in dense if r["token_ts"]
    )
    observed_tps = toks_per_chunk[len(toks_per_chunk) // 2] if toks_per_chunk else 0.0
    if dense and min(rates) > 0:
        tpot_ms = 1000 * len(rates) / sum(rates)
        agg_tok_s = sum(rates)
    else:
        tpot_ms = step_ms / args.tokens_per_step
        agg_tok_s = len(results) * 1000 / tpot_ms
    out.update(
        {
            "observed_tokens_per_step": round(observed_tps, 3),
            "measured_tpot_ms": round(tpot_ms, 2),
            "measured_tok_per_s": round(agg_tok_s, 1),
            "measured_step_ms": round(tpot_ms * observed_tps, 2),
            "step_mean_ms": round(step_ms, 2),
            "step_min_max_ms": [
                round(1000 * min(spans), 2),
                round(1000 * max(spans), 2),
            ],
            "tpot_eff_ms": round(step_ms / args.accept_len, 2),
            "batch_proof_concurrency": [conc(t) for t in probes],
            "full_batch_fraction": round(overlap, 3),
            "method": method,
            "agg_tok_per_s": round(
                len(results) * args.accept_len / (step_ms / 1000), 1
            ),
        }
    )
    return out


async def run(args):
    file_prompts = None
    if args.prompts_file:
        file_prompts = json.load(open(args.prompts_file))
        assert (
            len(file_prompts) >= args.prompt_offset + args.cc
        ), "not enough prompts in file"
    t_ref = time.monotonic()
    conn = aiohttp.TCPConnector(limit=256)
    timeout = aiohttp.ClientTimeout(total=14400)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:

        async def delayed(i):
            await asyncio.sleep(i * args.stagger)
            if file_prompts is not None:
                prompt = file_prompts[args.prompt_offset + i]
            else:
                prompt = build_prompt(args.prompt_tokens, i * 7919 + 13)
            return await one_request(s, args, prompt, i, t_ref)

        results = await asyncio.gather(*[delayed(i) for i in range(args.cc)])
    print(json.dumps(analyze(args, results), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--cc", type=int, required=True, help="requests sent at once")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--prompts-file",
        default="",
        help="JSON list of prompts (e.g. from make_distinct_prompts.py); "
        "without it a shared-prefix prompt of --prompt-tokens is built",
    )
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--stagger", type=float, default=0.0, help="s between requests")
    ap.add_argument("--plateau-s", type=float, default=15.0)
    ap.add_argument("--plateau-skip-s", type=float, default=3.0)
    ap.add_argument("--accept-len", type=float, default=1.0)
    ap.add_argument("--tokens-per-step", type=float, default=1.0)
    ap.add_argument(
        "--force-accept", action="store_true", help="RTP-LLM force_sp_accept"
    )
    ap.add_argument("--vllm", action="store_true", help="vLLM request format")
    ap.add_argument("--model", default="", help="model field of the request")
    args = ap.parse_args()
    if not args.model:
        args.model = "DeepSeek-V4-Flash" if args.vllm else "deepseek_v4"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
