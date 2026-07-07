#!/usr/bin/env python3
"""Compare old-loader vs new-loader DUMP_WEIGHTS json dumps.

Both loaders share a symmetric DUMP_WEIGHTS hook (set env DUMP_WEIGHTS=/dir):
  - old loader: rtp_llm/model_loader/loader.py  -> keys: "global.X" / "layer{i}.X"
  - new loader: rtp_llm/models_py/model_loader.py -> keys: flat named_parameters()
Each tensor entry records: src/shape/dtype/mean/std/absmax/md5 (md5 over float32 bytes).

Because the two key-naming schemes are unrelated, this tool does NOT match by name.
It matches by *content fingerprint* (md5) first, then by shape-bucketed statistics
for tensors whose layout differs (e.g. fused QKV vs split q/k/v).

This script is PURE stdlib -> runs on the Mac side directly (no torch needed).

Usage:
  # single rank
  python compare_dumps.py --old /tmp/dump_old/rank0.json --new /tmp/dump_new/rank0.json
  # whole dir (all ranks), with tolerance
  python compare_dumps.py --old-dir /tmp/dump_old --new-dir /tmp/dump_new --rtol 1e-3
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def _load(path):
    with open(path) as f:
        return json.load(f)


def _load_many(paths):
    merged = {}
    for p in sorted(paths):
        d = _load(p)
        rank = os.path.splitext(os.path.basename(p))[0]  # rank0
        for k, v in d.items():
            merged[f"{rank}/{k}"] = v
    return merged


def _collect(args):
    if args.old_dir:
        old = _load_many(glob.glob(os.path.join(args.old_dir, "rank*.json")))
        new = _load_many(glob.glob(os.path.join(args.new_dir, "rank*.json")))
    else:
        old = _load(args.old)
        new = _load(args.new)
    return old, new


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _stat_close(a, b, rtol, atol):
    def close(x, y):
        return abs(x - y) <= atol + rtol * max(abs(x), abs(y))

    return (
        close(a["mean"], b["mean"])
        and close(a["std"], b["std"])
        and close(a["absmax"], b["absmax"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--old-dir")
    ap.add_argument("--new-dir")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument(
        "--show", type=int, default=30, help="max rows to print per section"
    )
    args = ap.parse_args()

    old, new = _collect(args)

    # ---- Tier 0: coverage ------------------------------------------------
    old_elems = sum(_numel(v["shape"]) for v in old.values())
    new_elems = sum(_numel(v["shape"]) for v in new.values())
    print("=" * 72)
    print("Tier 0  COVERAGE")
    print(f"  old: {len(old):>5} tensors, {old_elems:>14,} elements")
    print(f"  new: {len(new):>5} tensors, {new_elems:>14,} elements")
    elem_delta = abs(old_elems - new_elems) / max(old_elems, 1)
    print(f"  element-count rel diff: {elem_delta:.4%}")

    # ---- Tier 1: exact md5 multiset --------------------------------------
    old_md5 = defaultdict(list)
    new_md5 = defaultdict(list)
    for k, v in old.items():
        old_md5[v["md5"]].append(k)
    for k, v in new.items():
        new_md5[v["md5"]].append(k)

    exact = set(old_md5) & set(new_md5)
    old_only_md5 = set(old_md5) - set(new_md5)
    new_only_md5 = set(new_md5) - set(old_md5)
    matched_old = sum(len(old_md5[m]) for m in exact)
    matched_new = sum(len(new_md5[m]) for m in exact)
    print("=" * 72)
    print("Tier 1  EXACT md5 MATCH (byte-identical float32 content)")
    print(f"  distinct md5 matched : {len(exact)}")
    print(f"  old tensors covered  : {matched_old}/{len(old)}")
    print(f"  new tensors covered  : {matched_new}/{len(new)}")

    # ---- Tier 2: shape-bucketed statistical pairing for the remainder ----
    old_rem = {k: v for k, v in old.items() if v["md5"] in old_only_md5}
    new_rem = {k: v for k, v in new.items() if v["md5"] in new_only_md5}

    old_by_shape = defaultdict(list)
    new_by_shape = defaultdict(list)
    for k, v in old_rem.items():
        old_by_shape[tuple(v["shape"])].append((k, v))
    for k, v in new_rem.items():
        new_by_shape[tuple(v["shape"])].append((k, v))

    stat_matched = []
    stat_old_unmatched = []
    for shape, items in old_by_shape.items():
        cands = list(new_by_shape.get(shape, []))
        for ok, ov in items:
            hit = None
            for i, (nk, nv) in enumerate(cands):
                if _stat_close(ov, nv, args.rtol, args.atol):
                    hit = (nk, nv)
                    cands.pop(i)
                    break
            if hit:
                stat_matched.append((ok, hit[0], shape))
            else:
                stat_old_unmatched.append((ok, ov))
    new_by_shape_left = []
    for shape, cands in new_by_shape.items():
        for nk, nv in cands:
            # only those not consumed above (re-derive: anything left is unmatched)
            new_by_shape_left.append((nk, nv))
    # recompute new unmatched precisely
    consumed_new = {nk for _, nk, _ in stat_matched}
    new_unmatched = [(k, v) for k, v in new_rem.items() if k not in consumed_new]

    print("=" * 72)
    print("Tier 2  STAT MATCH on remainder (same shape, mean/std/absmax within tol)")
    print(f"  rtol={args.rtol}  atol={args.atol}")
    print(f"  stat-paired tensors  : {len(stat_matched)}")
    print(f"  OLD-only (unmatched) : {len(stat_old_unmatched)}")
    print(f"  NEW-only (unmatched) : {len(new_unmatched)}")

    if stat_old_unmatched:
        print(
            "\n  -- OLD tensors with NO new counterpart (potential MISSING in new): --"
        )
        for k, v in stat_old_unmatched[: args.show]:
            print(
                f"    {k:60s} shape={v['shape']} mean={v['mean']:.4g} std={v['std']:.4g}"
            )
        if len(stat_old_unmatched) > args.show:
            print(f"    ... +{len(stat_old_unmatched) - args.show} more")

    if new_unmatched:
        print("\n  -- NEW tensors with NO old counterpart (extra / re-laid-out): --")
        for k, v in new_unmatched[: args.show]:
            print(
                f"    {k:60s} shape={v['shape']} mean={v['mean']:.4g} std={v['std']:.4g}"
            )
        if len(new_unmatched) > args.show:
            print(f"    ... +{len(new_unmatched) - args.show} more")

    # ---- Verdict ---------------------------------------------------------
    print("=" * 72)
    ok = (
        len(stat_old_unmatched) == 0
        and elem_delta < 0.01
        and (matched_old + len(stat_matched)) == len(old)
    )
    if ok:
        print(
            "VERDICT: PASS  — every old tensor has an exact or statistical match in new."
        )
        print(
            "         (Remaining NEW-only tensors are usually fused/split re-layouts;"
        )
        print("          confirm with an end-to-end logits/token equivalence run.)")
    else:
        print(
            "VERDICT: FAIL  — some old weights are missing or numerically different in new."
        )
        print(
            "         Inspect the OLD-only list above; those are the loading bugs to fix."
        )
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
