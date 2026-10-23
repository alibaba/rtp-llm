#!/usr/bin/env python3
"""Distinct-prompt concurrent client: 32 genuinely different 32K prompts
(unique lead-in from token 0 + city-specific long text), warm (max_tokens=1)
or fire (max_tokens=N) mode. Deterministic prompt set across invocations."""
import argparse, json, threading, time, urllib.request, statistics

CITIES = ["杭州","北京","上海","深圳","成都","西安","南京","苏州","重庆","武汉",
          "青岛","厦门","长沙","昆明","哈尔滨","沈阳","天津","宁波","无锡","佛山",
          "东莞","合肥","福州","济南","大连","珠海","洛阳","桂林","三亚","拉萨",
          "乌鲁木齐","敦煌"]
ASPECTS = ["地理格局","历史沿革","经济产业","文化底蕴","科技创新","交通枢纽","教育资源","生态宜居"]

def build_prompt(i, target_tokens):
    city = CITIES[i % len(CITIES)]
    lead = f"专项城市研究报告（编号 ZS-{1000+i}）：本报告独立分析{city}，请勿与其他城市混淆。"
    paras = []
    for k, asp in enumerate(ASPECTS):
        paras.append(f"第{k+1}章「{asp}」：{city}在{asp}方面具有独特地位，"
                     f"从{asp}视角看，{city}展现出与众不同的发展路径与区域影响力，"
                     f"其{asp}特征深刻塑造了城市气质与未来潜力，值得系统梳理与深入解读。")
    body = "".join(paras)
    reps = max(1, int(target_tokens * 1.83 / len(body)))
    return lead + body * reps + "请开始你的回答。"

def one(ep, prompt, max_tokens, barrier, results, idx):
    body = json.dumps({"model": "deepseek_v4", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": 42,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://{ep}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.monotonic(); usage = {}; err = None; nchunks = 0; tf = tl = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"): continue
                p = line[5:].strip()
                if p == "[DONE]": break
                try: d = json.loads(p)
                except Exception: continue
                if "error_code" in d: raise RuntimeError(str(d)[:160])
                if d.get("usage"): usage = d["usage"]
                delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    now = time.monotonic()
                    if tf is None: tf = now
                    tl = now; nchunks += 1
    except Exception as e:
        err = repr(e)[:160]
    comp = usage.get("completion_tokens", nchunks)
    results[idx] = {"ttft_ms": (tf - t0)*1000 if tf else None,
                    "tpot_ms": ((tl - tf)/max(comp-1,1))*1000 if tf and comp > 1 else None,
                    "e2e_ms": (time.monotonic()-t0)*1000, "comp": comp, "chunks": nchunks,
                    "prompt_tokens": usage.get("prompt_tokens"), "err": err}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tag", default="dist")
    a = ap.parse_args()
    prompts = [build_prompt(i, a.prompt_tokens) for i in range(a.concurrency)]
    barrier = threading.Barrier(a.concurrency)
    results = [None]*a.concurrency
    ts = [threading.Thread(target=one, args=(a.endpoint, prompts[i], a.max_tokens, barrier, results, i))
          for i in range(a.concurrency)]
    t0 = time.monotonic()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.monotonic() - t0
    ok = [r for r in results if r and not r["err"]]
    fail = len(results) - len(ok)
    ttfts = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"] is not None)
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    toks = sum(r["comp"] for r in ok)
    print(f"TAG={a.tag} ok={len(ok)} fail={fail} wall_s={wall:.1f}")
    if ttfts:
        print(f"TTFT_ms: mean={statistics.mean(ttfts):.1f} p50={statistics.median(ttfts):.1f} min={min(ttfts):.1f} max={max(ttfts):.1f}")
        if tpots:
            print(f"TPOT_ms: mean={statistics.mean(tpots):.2f} p50={statistics.median(tpots):.2f} min={min(tpots):.2f} max={max(tpots):.2f}")
        print(f"total_completion={toks} agg_tok_per_s={toks/wall:.0f}")
        print("ttft_sorted:", [round(x) for x in ttfts])
        if fail: print("errs:", [r["err"] for r in results if r and r["err"]][:3])
main()
