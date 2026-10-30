#!/usr/bin/env python3
"""Summary of pd_uniform_load.py end-to-end runs.

  an_uniform_load.py <run.jsonl> ...

Per run: achieved send rate, first-token completion rate (2nd quartile to the
end; equals the send rate when the system keeps up), TTFT p50 / p90 and the p50
of each quarter of the run in send order (flat quarters = no queue build-up),
time to the 2nd token, TPOT p50 / p90 ((t_last - t_2nd) / (tokens - 2)),
output length and finish reasons (check that ignore_eos held), decode-side
concurrency (send rate x mean decode time, Little's law) and tokens per SSE
chunk (~ MTP acceptance per step). The sustainable QPS of a configuration is
the highest rate with flat TTFT quarters and TPOT p50 within the SLA.
"""
import json
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main():
    for f in sys.argv[1:]:
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        rows = [r for r in rows if r["idx"] >= 0]
        ok = [r for r in rows if r.get("chunks") and not r.get("err")]
        err = len(rows) - len(ok)
        name = f.split("/")[-1]
        if len(ok) < 8:
            print("%-28s ok=%d err=%d" % (name, len(ok), err))
            continue
        ok.sort(key=lambda r: r["t_send"])
        send = [r["t_send"] for r in ok]
        qps_send = (len(send) - 1) / (send[-1] - send[0])
        ft = sorted(r["t_send"] + r["chunks"][0] for r in ok)
        n = len(ft)
        frate = (n - n // 4 - 1) / (ft[-1] - ft[n // 4])
        ttft = [1000 * r["chunks"][0] for r in ok]
        quarters = [
            statistics.median(ttft[i * n // 4 : (i + 1) * n // 4]) for i in range(4)
        ]
        tok2 = [1000 * r["chunks"][1] for r in ok if len(r["chunks"]) > 1]
        tpot = [
            1000 * (r["chunks"][-1] - r["chunks"][1]) / (r["toks"][-1] - r["toks"][1])
            for r in ok
            if len(r["toks"]) > 2 and r["toks"][-1] > r["toks"][1]
        ]
        out_tok = [r["toks"][-1] for r in ok]
        fin = {}
        for r in ok:
            fin[r.get("finish")] = fin.get(r.get("finish"), 0) + 1
        dec = [r["chunks"][-1] - r["chunks"][0] for r in ok]
        tpc = [
            (r["toks"][-1] - r["toks"][1]) / (len(r["toks"]) - 2)
            for r in ok
            if len(r["toks"]) > 3
        ]
        print(
            "%-28s n=%d err=%d send=%.2f ft_rate=%.2f TTFT p50=%.0f p90=%.0f q=[%s] tok2=%.0f "
            "TPOT p50=%.2f p90=%.2f out p50=%d fin=%s Dconc=%.0f tok/chunk=%.2f"
            % (
                name,
                n,
                err,
                qps_send,
                frate,
                pct(ttft, 0.5),
                pct(ttft, 0.9),
                ",".join("%.0f" % x for x in quarters),
                pct(tok2, 0.5),
                pct(tpot, 0.5),
                pct(tpot, 0.9),
                pct(out_tok, 0.5),
                fin,
                qps_send * statistics.mean(dec),
                statistics.median(tpc),
            )
        )


if __name__ == "__main__":
    main()
