#!/usr/bin/env python3
"""PD perf client v2: TTFT / TPOT under concurrency, SSE timing.
v2: per-request unique nonce (defeats D-side prefix reuse), early chunk
timestamps (tok1->tok2 gap = KV load + first decode step window)."""
import argparse, json, threading, time, urllib.request, statistics

def build_prompt(target_tokens, salt):
    seg = ("杭州是一座融合千年宋韵文化与现代数字经济的城市，西湖、钱塘江与数字经济产业共同构成了它的城市名片。"
           "请从地理、历史、经济、文化、科技五个角度详细分析这座城市的独特魅力与发展潜力。")
    reps = max(1, int(target_tokens * 1.83 / len(seg)))
    return f"本次分析编号{salt}。" + seg * reps + "请开始你的回答。"

def one_request(ep, prompt, max_tokens, barrier, results, idx, tag):
    body = json.dumps({
        "model": "deepseek_v4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": 42,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://{ep}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.monotonic()
    chunk_ts = []; usage = {}; finish = None; err = None; text_len = 0
    nchunks = 0; dt_first = None; dt_last = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.monotonic()
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if "error_code" in d: raise RuntimeError(str(d)[:200])
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if len(chunk_ts) < 8: chunk_ts.append(now - t0)
                    nchunks += 1
                    if dt_first is None: dt_first = now - t0
                    dt_last = now - t0
                    text_len += len(delta)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as e:
        err = repr(e)
    t_end = time.monotonic()
    comp = usage.get("completion_tokens", nchunks)
    results[idx] = {
        "tpot_ms": ((dt_last - dt_first) / max(comp - 1, 1)) * 1000 if dt_first is not None and comp > 1 else None,
        "err": err, "finish": finish,
        "ttft_ms": chunk_ts[0]*1000 if chunk_ts else None,
        "gap12_ms": (chunk_ts[1]-chunk_ts[0])*1000 if len(chunk_ts) > 1 else None,
        "gap23_ms": (chunk_ts[2]-chunk_ts[1])*1000 if len(chunk_ts) > 2 else None,
        "early_gaps_ms": [round((chunk_ts[i]-chunk_ts[i-1])*1000,2) for i in range(1,len(chunk_ts))],
        "e2e_ms": (t_end - t0) * 1000,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp, "text_len": text_len,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=1)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    all_results = []
    for round_i in range(args.requests):
        n = args.concurrency
        results = [None] * n
        barrier = threading.Barrier(n)
        ths = []
        for i in range(n):
            salt = f"{args.tag}-r{round_i}-w{i}-{int(time.time())}"
            prompt = build_prompt(args.prompt_tokens, salt)
            ths.append(threading.Thread(target=one_request, args=(args.endpoint, prompt, args.max_tokens, barrier, results, i, args.tag)))
        for t in ths: t.start()
        for t in ths: t.join()
        all_results.extend(results)
        print(f"== round {round_i} done ==", flush=True)
    ok = [r for r in all_results if r and not r["err"] and r["ttft_ms"] is not None]
    bad = [r for r in all_results if not r or r["err"]]
    print(f"TAG={args.tag} conc={args.concurrency} ok={len(ok)} fail={len(bad)}")
    for r in bad[:3]: print("  FAIL:", r and r["err"])
    if ok:
        pt = ok[0]["prompt_tokens"]
        ttfts = sorted(r["ttft_ms"] for r in ok)
        gaps = sorted(r["gap12_ms"] for r in ok if r["gap12_ms"] is not None)
        e2es = sorted(r["e2e_ms"] for r in ok)
        def stats(v):
            return f"mean={statistics.mean(v):.1f} p50={v[len(v)//2]:.1f} min={v[0]:.1f} max={v[-1]:.1f}"
        print(f"prompt_tokens={pt} completion={ok[0]['completion_tokens']}")
        print(f"TTFT_ms: {stats(ttfts)}")
        if gaps: print(f"GAP12_ms(KVload+step): {stats(gaps)}")
        tpots = sorted(r["tpot_ms"] for r in ok if r["tpot_ms"] is not None)
        if tpots: print(f"TPOT_ms: {stats(tpots)}")
        print(f"E2E_ms:  {stats(e2es)}")
        # throughput estimate: total completion tokens / wall of last finisher batch
        tot_comp = sum(r["completion_tokens"] for r in ok)
        print(f"total_completion_tokens={tot_comp}")
    print("RESULT_JSON:" + json.dumps(all_results))

if __name__ == "__main__":
    main()
