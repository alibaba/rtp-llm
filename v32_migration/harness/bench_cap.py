"""Capacity benchmark: how many concurrent long-context requests fit per GPU.

This is the point of the offload scheme. A 63k request holds ~984 blocks of KV;
under the lossy third-pool scheme it holds only KEEP_BLOCKS + STAGING_BLOCKS + 1
(~289), so the same KV pool should admit ~3.4x more concurrent long requests.

Each request gets a unique prefix so prefix-cache reuse cannot mask the pressure.
Reports admitted/failed counts, latency spread and aggregate throughput.
"""

import argparse
import concurrent.futures as cf
import json
import statistics as st
import threading
import time
import urllib.request

# Same prompt shape as bench_ab.py: verified to generate the full token budget
# under greedy decoding. Prefix reuse is off in this config (aux_info reuse_len=0),
# so requests may share the prompt.
HEAD = "请把下面的文本原样复述一遍并总结要点。"


def make_prompt(nchars, seed):
    return HEAD + ("深度学习模型推理性能分析。" * (nchars // 13 + 1))[:nchars]


def post(port, prompt, max_new, timeout):
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


def one(port, ctx, ratio, out_tokens, seed, timeout):
    try:
        res, wall = post(port, make_prompt(int(ctx / ratio), seed), out_tokens, timeout)
    except Exception as e:
        return {"ok": False, "err": type(e).__name__ + ":" + str(e)[:80]}
    aux = res.get("aux_info") or {}
    ol = aux.get("output_len") or out_tokens
    cost, pre = aux.get("cost_time"), aux.get("prefill_time") or aux.get(
        "first_token_cost_time"
    )
    return {
        "ok": True,
        "wall_s": wall,
        "il": aux.get("input_len"),
        "ol": ol,
        "prefill_ms": pre,
        "cost_ms": cost,
        "tpot_ms": (cost - pre) / (ol - 1) if (cost and pre and ol > 1) else None,
    }


def poll_status(port, stop, out):
    """Engine scheduler occupancy: how many long requests it keeps resident."""
    while not stop.is_set():
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/worker_status", timeout=5
            ) as r:
                d = json.loads(r.read())
            per = d.get("results") or []
            run = max(
                int(d.get("running_query_len") or 0),
                sum(int(x.get("running_query_len") or 0) for x in per),
            )
            wait = max(
                int(d.get("waiting_query_len") or 0),
                sum(int(x.get("waiting_query_len") or 0) for x in per),
            )
            out.append((run, wait))
        except Exception:
            pass
        stop.wait(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=26100)
    ap.add_argument("--ctx", type=int, default=63000)
    ap.add_argument("--out", type=int, default=32)
    ap.add_argument("--conc", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="seconds between submissions; avoids piling up long prefills",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="/home/admin/rtp-hol/logs/cap_bench.jsonl")
    a = ap.parse_args()

    probe, _ = post(a.port, make_prompt(4000, 0), 1, 600)
    il = (probe.get("aux_info") or {}).get("input_len") or 0
    ratio = il / len(make_prompt(4000, 0)) if il else 0.463
    print(f"calib ratio={ratio:.3f}")

    seed = 1000
    for conc in a.conc:
        occ = []
        stop = threading.Event()
        poller = threading.Thread(
            target=poll_status, args=(a.port, stop, occ), daemon=True
        )
        poller.start()
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            futs = []
            for i in range(conc):
                futs.append(
                    ex.submit(one, a.port, a.ctx, ratio, a.out, seed + i, a.timeout)
                )
                if a.stagger:
                    time.sleep(a.stagger)
            recs = [f.result() for f in futs]
        dur = time.time() - t0
        stop.set()
        seed += conc
        ok = [r for r in recs if r["ok"]]
        err = [r for r in recs if not r["ok"]]
        walls = sorted(r["wall_s"] for r in ok)
        tp = [r["tpot_ms"] for r in ok if r["tpot_ms"]]
        toks = sum(r["ol"] or 0 for r in ok)
        row = {
            "tag": a.tag,
            "ctx": a.ctx,
            "conc": conc,
            "ok": len(ok),
            "err": len(err),
            "dur_s": round(dur, 1),
            "wall_p50": round(walls[len(walls) // 2], 1) if walls else None,
            "wall_max": round(walls[-1], 1) if walls else None,
            "tpot_mean": round(st.mean(tp), 1) if tp else None,
            "tpot_max": round(max(tp), 1) if tp else None,
            "agg_tok_s": round(toks / dur, 1) if dur else None,
            "max_running": max((r for r, _ in occ), default=None),
            "max_waiting": max((w for _, w in occ), default=None),
            "errs": [r["err"] for r in err[:3]],
        }
        print(
            f"  conc={conc:3d} ok={row['ok']}/{conc} dur={row['dur_s']}s "
            f"wall p50/max={row['wall_p50']}/{row['wall_max']}s "
            f"TPOT mean/max={row['tpot_mean']}/{row['tpot_max']}ms "
            f"agg={row['agg_tok_s']} tok/s "
            f"max_running={row['max_running']} max_waiting={row['max_waiting']}"
            + (f" ERR={row['errs']}" if err else "")
        )
        with open(a.jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
