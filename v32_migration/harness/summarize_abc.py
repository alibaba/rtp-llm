#!/usr/bin/env python3
"""Summarise an A/B/C sweep and check the runs are actually comparable.

Refuses to draw conclusions if the schemes did not see the same request: a
differing prompt digest or input_len makes the TPOT numbers meaningless. Reports
the lossy schemes' output divergence against A, which is the quality cost that a
latency table on its own hides.
"""
import argparse
import json
from collections import OrderedDict


def load(path):
    runs = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            runs[r["tag"]] = r  # later runs win
    return runs


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/admin/rtp-hol/logs/abc_bench.jsonl")
    a = ap.parse_args()
    runs = load(a.jsonl)
    if not runs:
        print("no runs")
        return

    digests = {r["prompt_sha"] for r in runs.values()}
    ils = {row["il"] for r in runs.values() for row in r["rows"]}
    print(f"prompt sha={digests} input_len={ils}")
    if len(digests) > 1 or len(ils) > 1:
        print("!! runs did not see the same request - TPOT is NOT comparable")

    base = runs.get("A")
    base_tpot = base and base["tpot_mean"]
    base_text = None
    if base:
        keep = [r for r in base["rows"] if not r.get("warmup")]
        base_text = keep[0]["out_text"] if keep else None

    print(f"\n{'tag':<6} {'TPOT ms':>9} {'vs A':>8}   output")
    for tag, r in runs.items():
        t = r["tpot_mean"]
        rel = f"{(t / base_tpot - 1) * 100:+.2f}%" if (t and base_tpot) else "-"
        keep = [row for row in r["rows"] if not row.get("warmup")]
        txt = keep[0]["out_text"] if keep else ""
        if base_text is None or tag == "A":
            note = "baseline" if tag == "A" else "no baseline"
        elif txt == base_text:
            note = "identical to A"
        else:
            d = first_divergence(base_text, txt)
            note = f"diverges from A at char {d}"
        shas = {row["out_sha"] for row in keep}
        stable = "" if len(shas) <= 1 else "  [UNSTABLE across reps]"
        # A wrong token can make the model stop early; TPOT is averaged over the
        # tokens actually produced, so such a rep is also a contaminated sample.
        ols = [row["ol"] for row in keep]
        short = "" if len(set(ols)) <= 1 else f"  [EARLY STOP ol={ols}]"
        print(f"{tag:<6} {t if t else 0:9.2f} {rel:>8}   {note}{stable}{short}")

    all_ols = {row["ol"] for r in runs.values() for row in r["rows"]}
    if len(all_ols) > 1:
        print(f"\n!! output lengths differ across runs {sorted(all_ols)}: any rep that")
        print("   stopped early produced a wrong token and its TPOT is not comparable")


if __name__ == "__main__":
    main()
