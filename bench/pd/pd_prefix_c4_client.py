#!/usr/bin/env python3
"""c4 TTFT / gap12 client for the shared-prefix scenario.

  pd_prefix_c4_client.py --prompts prompts_32k_prefix24k.json --rounds 5 --conc 4 \
      --tag rdma-pfx --out-dir .

prompts file: {"prefix_tokens": N, "prompts": [...]}; every prompt shares the
same first N tokens (prefix) and has a distinct suffix. Round 0 additionally
sends one warm-up request (the prefix + a distinct suffix) so the prefix is in
the prefill-side cache. Each round fires --conc distinct prompts at once and
writes one pd_ttft_gap_client-compatible json per request
(len-<tag>-32k-<i>.json) plus the server-reported aux_info (reuse lengths).
"""
import argparse
import json
import threading
import time
import urllib.request


def fire(ep, prompt, max_tokens, barrier, out, idx):
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
            "aux_info": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{ep}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait()
    t_send = time.time()
    t0 = time.monotonic()
    chunks, usage, aux, err, text = [], {}, {}, None, []
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
                    raise RuntimeError(str(d)[:300])
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("aux_info"):
                    aux = d["aux_info"]
                delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get(
                    "content"
                )
                if delta:
                    chunks.append(time.monotonic() - t0)
                    text.append(delta)
    except Exception as e:
        err = repr(e)[:300]
    out[idx] = {
        "t_send_wall": t_send,
        "chunks": chunks,
        "err": err,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "aux": aux,
        "text": "".join(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--first", type=int, default=1, help="index of the first prompt to use"
    )
    ap.add_argument(
        "--warm", type=int, default=1, help="send one warm-up request before round 0"
    )
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between rounds")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    pf = json.load(open(a.prompts))
    P = pf["prompts"]
    cities = pf.get("suffix_cities")
    k = a.first
    if a.warm:
        res = [None]
        fire(a.endpoint, P[0], a.max_tokens, None, res, 0)
        r = res[0]
        print(
            "warm ttft=%.0f err=%s aux_reuse=%s"
            % (
                r["chunks"][0] * 1000 if r["chunks"] else -1,
                r["err"],
                {
                    x: r["aux"].get(x)
                    for x in (
                        "reuse_len",
                        "prefill_total_reuse_len",
                        "decode_total_reuse_len",
                    )
                },
            ),
            flush=True,
        )
        time.sleep(a.gap)
    n = 0
    for rd in range(a.rounds):
        barrier = threading.Barrier(a.conc)
        res = [None] * a.conc
        idxs = [k + i for i in range(a.conc)]
        ths = [
            threading.Thread(
                target=fire,
                args=(a.endpoint, P[idxs[i]], a.max_tokens, barrier, res, i),
            )
            for i in range(a.conc)
        ]
        [t.start() for t in ths]
        [t.join() for t in ths]
        k += a.conc
        order = sorted(
            range(a.conc),
            key=lambda i: res[i]["chunks"][0] if res[i]["chunks"] else 1e9,
        )
        for rank_in_round, i in enumerate(order):
            r = res[i]
            r["round"], r["slot"], r["order_in_round"], r["prompt_index"] = (
                rd,
                i,
                rank_in_round,
                idxs[i],
            )
            if cities:
                r["expected_city"] = cities[idxs[i]]
                r["city_ok"] = cities[idxs[i]] in r["text"]
            c = r["chunks"]
            line = (
                "%s r%d s%d ord%d ISL=%s TTFT=%.0f gap12=%.1f p_reuse=%s d_reuse=%s city_ok=%s err=%s"
                % (
                    a.tag,
                    rd,
                    i,
                    rank_in_round,
                    r["prompt_tokens"],
                    c[0] * 1000 if c else -1,
                    (c[1] - c[0]) * 1000 if len(c) > 1 else -1,
                    r["aux"].get("prefill_total_reuse_len"),
                    r["aux"].get("decode_total_reuse_len"),
                    r.get("city_ok"),
                    r["err"],
                )
            )
            print(line, flush=True)
            json.dump(
                {"tag": "%s-32k-%d" % (a.tag, n), "args": vars(a), "rows": [r]},
                open("%s/len-%s-32k-%d.json" % (a.out_dir, a.tag, n), "w"),
            )
            n += 1
        time.sleep(a.gap)


if __name__ == "__main__":
    main()
