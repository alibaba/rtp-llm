"""PD TTFT/TPOT benchmark on the 8 frozen ShareGPT 32K prompts (seed 42, vLLM builder).

Sends each prompt to the prefill server's native HTTP endpoint ("/"), streaming, greedy,
fixed output length, and records per-token arrival times on the client.

  TTFT (client)   = first streamed token - request send   (includes tokenize, prefill,
                    KV publish + eRDMA transfer to decode, first decode step, network)
  TPOT (client)   = (last token - first token) / (n_out - 1)
  server fields   = aux_info.first_token_cost_time / wait_time / pd_sep / input_len

Usage:
  python pd_bench.py --endpoint 127.0.0.1:14000 --prompts sharegpt32k.jsonl --out res.json \
      [--max-new 4096] [--warmup 2] [--n 8] [--concurrency 1]
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="127.0.0.1:14000")
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new", type=int, default=4096)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--timeout", type=int, default=1800)
ap.add_argument(
    "--nonce",
    action="store_true",
    help="prefix each request with a unique tag so decode-side prefix reuse cannot skip the KV load",
)
a = ap.parse_args()

prompts = [json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_seq = [0]


def one(idx, prompt, tag):
    if a.nonce:
        _seq[0] += 1
        prompt = "[req %d-%d-%d] " % (int(time.time()), idx, _seq[0]) + prompt
    body = {
        "prompt": prompt,
        "yield_generator": True,
        "generate_config": {
            "max_new_tokens": a.max_new,
            "min_new_tokens": a.max_new,
            "top_k": 1,
            "is_streaming": True,
            "return_output_ids": True,
            "aux_info": True,
            "timeout_ms": a.timeout * 1000,
        },
    }
    req = urllib.request.Request(
        "http://%s/" % a.endpoint,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    arrivals, n_out, aux, err = [], 0, {}, None
    out_ids = []
    try:
        with opener.open(req, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in ("[done]", "[DONE]"):
                    break
                now = time.perf_counter()
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("error_code") or d.get("error"):
                    err = str(d)[:400]
                    break
                a_ = d.get("aux_info") or {}
                if a_:
                    aux = a_
                ids = d.get("output_ids") or []
                if ids:
                    out_ids.extend(ids[0] if isinstance(ids[0], list) else ids)
                out_len = a_.get("output_len")
                if out_len is None:
                    out_len = n_out + (
                        len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
                    )
                if out_len > n_out:
                    arrivals.extend([now] * (out_len - n_out))
                    n_out = out_len
    except Exception as e:
        err = repr(e)[:400]
    t_end = time.perf_counter()
    ttft = (arrivals[0] - t0) * 1000 if arrivals else None
    tpot = (
        ((arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) * 1000)
        if len(arrivals) > 1
        else None
    )
    gaps = [(arrivals[i + 1] - arrivals[i]) * 1000 for i in range(len(arrivals) - 1)]
    return {
        "tag": tag,
        "prompt": idx,
        "error": err,
        "client_ttft_ms": ttft,
        "client_tpot_ms": tpot,
        "e2e_ms": (t_end - t0) * 1000,
        "n_out": n_out,
        "gap_p50_ms": statistics.median(gaps) if gaps else None,
        "gap12_ms": gaps[0] if gaps else None,
        "gap_p99_ms": sorted(gaps)[int(0.99 * (len(gaps) - 1))] if gaps else None,
        "gap_max_ms": max(gaps) if gaps else None,
        "server_first_token_ms": aux.get("first_token_cost_time"),
        "server_wait_ms": aux.get("wait_time"),
        "server_cost_ms": aux.get("cost_time"),
        "input_len": aux.get("input_len"),
        "output_len": aux.get("output_len"),
        "pd_sep": aux.get("pd_sep"),
        "reuse_len": aux.get("reuse_len"),
        "output_ids": out_ids,
    }


def show(r):
    f = lambda v: "-" if v is None else "%.1f" % v
    print(
        "%-8s p%d in=%s out=%s pd=%s ttft=%s srv_ft=%s tpot=%s gap_p99=%s e2e=%s err=%s"
        % (
            r["tag"],
            r["prompt"],
            r["input_len"],
            r["n_out"],
            r["pd_sep"],
            f(r["client_ttft_ms"]),
            f(r["server_first_token_ms"]),
            f(r["client_tpot_ms"]),
            f(r["gap_p99_ms"]),
            f(r["e2e_ms"]),
            r["error"],
        ),
        flush=True,
    )


rows = []
for i in range(a.warmup):
    r = one(i % len(prompts), prompts[i % len(prompts)], "warmup")
    show(r)
    rows.append(r)

jobs = [(i % len(prompts), prompts[i % len(prompts)], "measure") for i in range(a.n)]
with ThreadPoolExecutor(a.concurrency) as ex:
    for r in ex.map(lambda j: one(*j), jobs):
        show(r)
        rows.append(r)

m = [
    r
    for r in rows
    if r["tag"] == "measure" and not r["error"] and r["client_ttft_ms"] is not None
]


def stat(key):
    v = [r[key] for r in m if r[key] is not None]
    if not v:
        return None
    return {
        "mean": statistics.mean(v),
        "median": statistics.median(v),
        "min": min(v),
        "max": max(v),
        "n": len(v),
    }


summary = {
    "endpoint": a.endpoint,
    "max_new": a.max_new,
    "concurrency": a.concurrency,
    "ok": len(m),
    "failed": len([r for r in rows if r["tag"] == "measure"]) - len(m),
    "client_ttft_ms": stat("client_ttft_ms"),
    "server_first_token_ms": stat("server_first_token_ms"),
    "client_tpot_ms": stat("client_tpot_ms"),
    "gap12_ms": stat("gap12_ms"),
    "gap_p99_ms": stat("gap_p99_ms"),
}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
