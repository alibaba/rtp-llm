#!/usr/bin/env python3
"""Summary of pd_prefix_gated_client.py runs without a wire sampler.

  an_gated_rounds.py <out_dir> [tag_prefix ...]

Groups the len-<tag>-32k-<i>.json rows by tag and prints, per tag: TTFT, time to
the 2nd token, gap12 (tok1 -> tok2), TPOT ((t_last - t_2nd) / (tokens - 2)) and
correct answers. Latencies are from the gate release (t_release_wall) when the
run was gated, else from each request's send time. Round 0 (the first batched
forward after a prefill restart, JIT warm-up) is skipped; SKIP_ROUNDS=0 keeps
it. For the KV wire timeline use an_prefix_gated.py with pd_wire_sampler.py.
"""
import collections
import glob
import json
import os
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main():
    out_dir, prefixes = sys.argv[1], sys.argv[2:]
    skip = int(os.environ.get("SKIP_ROUNDS", "1"))
    groups = collections.defaultdict(list)
    for f in glob.glob(os.path.join(out_dir, "len-*-32k-*.json")):
        j = json.load(open(f))
        tag = j["tag"].rsplit("-32k-", 1)[0]
        if prefixes and not any(tag.startswith(p) for p in prefixes):
            continue
        groups[tag] += j["rows"]
    for tag in sorted(groups):
        errs = sum(1 for r in groups[tag] if r.get("err") or not r.get("chunks"))
        rows = [
            r
            for r in groups[tag]
            if r.get("chunks") and not r.get("err") and r.get("round", 0) >= skip
        ]
        if not rows:
            print("%-26s no rows" % tag)
            continue
        ttft, tok2, gap, tpot, ok = [], [], [], [], 0
        for r in rows:
            t0 = r.get("t_release_wall") or r["t_send_wall"]
            c = r["chunks"]
            base = r["t_send_wall"] - t0
            ttft.append(1000 * (base + c[0]))
            if len(c) > 1:
                tok2.append(1000 * (base + c[1]))
                gap.append(1000 * (c[1] - c[0]))
            n = r.get("completion_tokens") or 0
            if len(c) > 2 and n > 2:
                tpot.append(1000 * (c[-1] - c[1]) / (n - 2))
            ok += 1 if r.get("city_ok") else 0
        print(
            "%-26s n=%d err=%d TTFT p50=%.0f mean=%.0f | tok2 p50=%.0f | gap12 p50=%.1f mean=%.1f "
            "| TPOT p50=%.2f | city %d/%d"
            % (
                tag,
                len(rows),
                errs,
                pct(ttft, 0.5),
                statistics.mean(ttft),
                pct(tok2, 0.5),
                pct(gap, 0.5),
                statistics.mean(gap),
                pct(tpot, 0.5),
                ok,
                len(rows),
            )
        )


if __name__ == "__main__":
    main()
