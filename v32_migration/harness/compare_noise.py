#!/usr/bin/env python3
"""Compare a scheme against the baseline's own run-to-run spread.

Bit-identity is not a usable criterion here: on real input the engine is not
reproducible at all (six runs of the untouched baseline gave six different
outputs), because the MoE all-to-all sums in arrival order and float addition is
not associative. So instead of asking "does the scheme match the baseline", ask
"is the scheme further from the baseline than the baseline is from itself".

Reports, over all run pairs, the position at which two outputs first differ and
their token-level agreement. If the cross-tag numbers sit inside the within-tag
spread, the scheme is indistinguishable from the baseline's own noise.
"""
import argparse
import json
import statistics as st
from collections import OrderedDict
from itertools import combinations, product


def load(path, tags):
    runs = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if tags and r["tag"] not in tags:
                continue
            texts = [
                x["out_text"]
                for x in r["rows"]
                if not x.get("warmup") and x["out_text"]
            ]
            runs.setdefault(r["tag"], {"texts": [], "prompt": r["prompt_sha"]})
            runs[r["tag"]]["texts"].extend(texts)
    return runs


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def agreement(a, b):
    """Fraction of leading characters shared, relative to the shorter output."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    d = first_diff(a, b)
    return 1.0 if d < 0 else d / n


def describe(pairs, label):
    if not pairs:
        print(f"  {label:<18} (no pairs)")
        return None
    ds = [first_diff(a, b) for a, b in pairs]
    ags = [agreement(a, b) for a, b in pairs]
    ident = sum(1 for d in ds if d < 0)
    med = st.median([d if d >= 0 else max(ds) for d in ds])
    print(
        f"  {label:<18} pairs={len(pairs):<3} identical={ident:<3} "
        f"first-diff median={med:<7.0f} min={min(ds):<7} "
        f"agreement mean={st.mean(ags):.4f}"
    )
    return st.mean(ags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base", default="Acorpus", help="baseline tag")
    ap.add_argument("--scheme", action="append", default=[], help="scheme tag(s)")
    a = ap.parse_args()

    tags = [a.base] + a.scheme
    runs = load(a.jsonl, tags)
    if a.base not in runs:
        raise SystemExit(f"baseline tag {a.base} not in {a.jsonl}")
    shas = {t: d["prompt"] for t, d in runs.items()}
    if len(set(shas.values())) > 1:
        raise SystemExit(f"tags saw different prompts {shas} - not comparable")
    print(f"prompt={next(iter(shas.values()))}")
    for t, d in runs.items():
        print(f"  {t}: {len(d['texts'])} runs")

    print("\nwithin-baseline spread (this is the noise floor):")
    base_ag = describe(list(combinations(runs[a.base]["texts"], 2)), a.base)

    for t in a.scheme:
        if t not in runs:
            continue
        print(f"\n{t} vs {a.base}:")
        cross = describe(
            list(product(runs[a.base]["texts"], runs[t]["texts"])), f"{a.base} x {t}"
        )
        describe(list(combinations(runs[t]["texts"], 2)), f"{t} x {t}")
        if base_ag is not None and cross is not None:
            verdict = (
                "within the baseline's own spread"
                if cross >= base_ag * 0.9
                else "WORSE than the baseline's own spread"
            )
            print(f"  -> agreement {cross:.4f} vs floor {base_ag:.4f}: {verdict}")


if __name__ == "__main__":
    main()
