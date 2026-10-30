#!/usr/bin/env python3
"""Shared-prefix client that makes N requests enter P's scheduler together.

  pd_prefix_gated_client.py --prompts prompts_32k_hit31k1k.json --rounds 6 --conc 8 \
      --tag g31-c8-rdma [--gate 1.0]

Same request/row format as pd_prefix_c4_client.py. Without --gate the round is
fired at once (natural arrival: P starts prefilling the first arrival alone).
With --gate S, each round pauses P's scheduler (RpcService/SetPause on P rank 0),
fires the N requests, waits S seconds so all of them are queued, then resumes
(SetRestart); the row gets "t_release_wall" and latencies are reported from it.
An idle P engine loop is already blocked inside schedule() waiting for work when
the pause is set, so a short placeholder request is sent first to consume that
schedule() call; the loop then parks on the pause flag (polled every 50ms, so
gated latencies include 0-50ms of release jitter).
"""
import argparse
import json
import threading
import time

import grpc
from pd_prefix_c4_client import fire

PLACEHOLDER = "hi"


def rpc(channel, method):
    call = channel.unary_unary(
        "/RpcService/" + method,
        request_serializer=lambda _: b"",
        response_deserializer=lambda b: b,
    )
    call(None, timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="127.0.0.1:14000")
    ap.add_argument(
        "--rpc",
        default="127.0.0.1:14001",
        help="P rank-0 RpcService (SetPause/SetRestart)",
    )
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--gap", type=float, default=6.0)
    ap.add_argument(
        "--gate",
        type=float,
        default=0.0,
        help="seconds to hold P paused after firing (0 = no gate)",
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    pf = json.load(open(a.prompts))
    P = pf["prompts"]
    cities = pf.get("suffix_cities")
    channel = grpc.insecure_channel(a.rpc) if a.gate > 0 else None
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
        barrier = threading.Barrier(a.conc + (1 if a.gate > 0 else 0))
        res = [None] * a.conc
        idxs = [k + i for i in range(a.conc)]
        ths = [
            threading.Thread(
                target=fire,
                args=(a.endpoint, P[idxs[i]], a.max_tokens, barrier, res, i),
            )
            for i in range(a.conc)
        ]
        t_release = None
        if a.gate > 0:
            rpc(channel, "SetPause")
            placeholder = threading.Thread(
                target=fire, args=(a.endpoint, PLACEHOLDER, 2, None, [None], 0)
            )
            try:
                placeholder.start()
                time.sleep(0.5)
                [t.start() for t in ths]
                barrier.wait()
                time.sleep(a.gate)
                t_release = time.time()
            finally:
                rpc(channel, "SetRestart")
            placeholder.join()
        else:
            [t.start() for t in ths]
        [t.join() for t in ths]
        k += a.conc

        def t1(i):
            r = res[i]
            return r["t_send_wall"] + r["chunks"][0] if r["chunks"] else 1e18

        order = sorted(range(a.conc), key=t1)
        for rank_in_round, i in enumerate(order):
            r = res[i]
            r["round"], r["slot"], r["order_in_round"], r["prompt_index"] = (
                rd,
                i,
                rank_in_round,
                idxs[i],
            )
            r["t_release_wall"] = t_release
            if cities:
                r["expected_city"] = cities[idxs[i]]
                r["city_ok"] = cities[idxs[i]] in r["text"]
            c = r["chunks"]
            t0 = t_release if t_release else r["t_send_wall"]
            line = (
                "%s r%d s%d ord%d ISL=%s TTFT=%.0f gap12=%.1f p_reuse=%s d_reuse=%s city_ok=%s err=%s"
                % (
                    a.tag,
                    rd,
                    i,
                    rank_in_round,
                    r["prompt_tokens"],
                    (r["t_send_wall"] + c[0] - t0) * 1000 if c else -1,
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
