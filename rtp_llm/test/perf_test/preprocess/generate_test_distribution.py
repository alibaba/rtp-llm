#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal distribution sampling for perf tests.

Reads a distribution.csv (ODPS bucket histogram) and generates
{batch_size: [seq_len_list]} test configs via empirical-CDF stratified
quantile sampling. No parametric fitting required — works with any
distribution shape (unimodal, bimodal, heavy-tail, etc.).

Usage:
    python generate_test_distribution.py \
        --csv  ../test_data/qwen35_tokenhub_distribution/distribution.csv \
        --out  ../test_data/qwen35_tokenhub_distribution/ \
        --max_seq_len 262144 \
        --bs_min 1 --bs_max 2048 --bs_step 8
"""

import argparse
import csv
import json
import os
import warnings

warnings.filterwarnings("ignore")

BUCKET_WIDTH = 1024


# ──────────────────────────────────────────────────────
#  Step 1: Load histogram from distribution.csv
# ──────────────────────────────────────────────────────
def load_histogram(csv_path):
    r"""Read distribution.csv -> (bucket_upper_bounds, counts).
    Skips \N (NULL) rows. Returns lists aligned by bucket."""
    uppers, counts = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row["bucket_id"].strip('"')
            if bid == "\\N" or bid == "":
                continue
            _, hi = row["length_range"].strip('"').split("-")
            uppers.append(int(hi))
            counts.append(int(row["cnt"].strip('"')))
    return uppers, counts


# ──────────────────────────────────────────────────────
#  Step 2: Empirical CDF + Stratified Quantile Sampling
# ──────────────────────────────────────────────────────
def build_empirical_cdf(uppers, counts, max_seq_len):
    """Build cumulative distribution truncated at max_seq_len.
    Returns (truncated_uppers, cdf_values)."""
    trunc_uppers, trunc_counts = [], []
    for u, c in zip(uppers, counts):
        if u > max_seq_len:
            break
        trunc_uppers.append(u)
        trunc_counts.append(c)
    total = sum(trunc_counts)
    cdf = []
    cumsum = 0
    for c in trunc_counts:
        cumsum += c
        cdf.append(cumsum / total)
    return trunc_uppers, cdf


def stratified_sample(bs, trunc_uppers, cdf):
    """Stratified quantile sampling from empirical CDF.

    Divide [0, 1] into *bs* equal strata, pick the bucket whose CDF
    first reaches each stratum midpoint. Returns bucket upper bounds
    (multiples of BUCKET_WIDTH), deterministic for fixed inputs.
    """
    samples = []
    for j in range(bs):
        q = (j + 0.5) / bs
        for i, c in enumerate(cdf):
            if c >= q:
                samples.append(trunc_uppers[i])
                break
        else:
            samples.append(trunc_uppers[-1])
    return samples


# ──────────────────────────────────────────────────────
#  Step 3: Visualization
# ──────────────────────────────────────────────────────
def plot_distribution(uppers, counts, trunc_uppers, cdf, batch_seq_len_map,
                      max_seq_len, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    total = sum(counts)
    sorted_bs = sorted(batch_seq_len_map.keys(), key=lambda k: int(k))

    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE = (
        "#3a7ca5", "#d64045", "#4a9c6d", "#e8a838", "#8172b3"
    )
    sample_colors = [C_ORANGE, C_GREEN, C_PURPLE, C_RED, "#6dbbb0"]
    markers = ["o", "s", "D", "^", "v"]

    def _fmt_k(x, _=None):
        if x >= 1000:
            return f"{x / 1000:.0f}k" if x % 1000 == 0 else f"{x / 1000:.1f}k"
        return f"{int(x)}"

    cum = 0
    p50_val = p90_val = p99_val = uppers[-1]
    p50_done = p90_done = p99_done = False
    for u, c in zip(uppers, counts):
        cum += c
        frac = cum / total
        if not p50_done and frac >= 0.5:
            p50_val, p50_done = u, True
        if not p90_done and frac >= 0.9:
            p90_val, p90_done = u, True
        if not p99_done and frac >= 0.99:
            p99_val, p99_done = u, True
    within = sum(c for u, c in zip(uppers, counts) if u <= max_seq_len)

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    fig.suptitle(
        f"Distribution & Stratified Quantile Sampling  "
        f"(max_seq_len={max_seq_len:,})",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.06, right=0.97, top=0.91, bottom=0.06)

    # ── top-left: distribution area chart ──
    ax0 = fig.add_subplot(gs[0, 0])
    midpoints = [u - BUCKET_WIDTH / 2 for u in uppers]
    pcts = [c / total * 100 for c in counts]
    x_upper = max(max_seq_len, p99_val) * 1.15
    vis_mid = [m for m in midpoints if m <= x_upper]
    vis_pct = pcts[:len(vis_mid)]
    beyond = total - sum(c for u, c in zip(uppers, counts)
                         if u - BUCKET_WIDTH / 2 <= x_upper)

    ax0.fill_between(vis_mid, vis_pct, alpha=0.35, color=C_BLUE, step="mid")
    ax0.step(vis_mid, vis_pct, where="mid", lw=1.2, color=C_BLUE)
    ax0.axvline(max_seq_len, color=C_RED, ls="--", lw=1.8,
                label=f"max_seq_len = {_fmt_k(max_seq_len)}")
    for pval, plabel, clr in [(p50_val, "P50", C_GREEN),
                               (p90_val, "P90", C_ORANGE),
                               (p99_val, "P99", C_PURPLE)]:
        if pval <= x_upper:
            ax0.axvline(pval, color=clr, ls=":", lw=1.3, alpha=0.8)
            y_top = max(vis_pct) if vis_pct else 1
            ax0.text(pval + x_upper * 0.008, y_top * 0.92,
                     f"{plabel}={_fmt_k(pval)}", color=clr, fontsize=7.5,
                     va="top", ha="left")
    if beyond > 0:
        ax0.annotate(f"+{beyond:,} beyond {_fmt_k(x_upper)}", xy=(1, 0.02),
                     xycoords="axes fraction", fontsize=7.5, color="#888",
                     ha="right", style="italic")
    ax0.set_xlabel("Sequence Length", fontsize=9)
    ax0.set_ylabel("Percentage (%)", fontsize=9)
    ax0.set_title(f"Distribution  (n={total:,}, {len(uppers)} buckets)",
                  fontsize=11, pad=8)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(axis="y", alpha=0.25, ls="--")
    ax0.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax0.set_xlim(left=0, right=x_upper)

    # ── top-right: zoomed histogram (≤ max_seq_len) ──
    ax1 = fig.add_subplot(gs[0, 1])
    zoom_uppers = [u for u in uppers if u <= max_seq_len]
    zoom_counts = [c for u, c in zip(uppers, counts) if u <= max_seq_len]
    zoom_pcts = [c / total * 100 for c in zoom_counts]
    zoom_mid = [u - BUCKET_WIDTH / 2 for u in zoom_uppers]
    bars = ax1.bar(zoom_mid, zoom_pcts, width=BUCKET_WIDTH * 0.85,
                   color=C_BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)

    if sorted_bs:
        largest_key = sorted_bs[-1]
        for s in batch_seq_len_map[largest_key]:
            ax1.axvline(s, color=C_GREEN, alpha=0.5, lw=0.7)
        ax1.plot([], [], color=C_GREEN, lw=1.5,
                 label=f"sample points (bs={largest_key})")
    ax1.set_xlabel("Sequence Length", fontsize=9)
    ax1.set_ylabel("Percentage (%)", fontsize=9)
    ax1.set_title(f"Zoomed  (0 \u2013 {_fmt_k(max_seq_len)})", fontsize=11, pad=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25, ls="--")
    if zoom_pcts:
        ax1.set_ylim(top=max(zoom_pcts) * 1.30)
    ax1.set_xlim(left=0, right=max_seq_len + BUCKET_WIDTH / 2)
    ax1.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    # ── bottom-left: original vs sampled distribution ──
    ax2 = fig.add_subplot(gs[1, 0])
    trunc_total = sum(zoom_counts) if zoom_counts else 1
    orig_pcts = [c / trunc_total * 100 for c in zoom_counts]
    ax2.bar(zoom_mid, orig_pcts, width=BUCKET_WIDTH * 0.85, color=C_BLUE,
            alpha=0.35, label="Original (within cutoff)")

    rep_bs = []
    if sorted_bs:
        if len(sorted_bs) >= 3:
            rep_bs = [sorted_bs[0], sorted_bs[len(sorted_bs) // 2], sorted_bs[-1]]
        else:
            rep_bs = list(sorted_bs)

    from collections import Counter
    for idx, bs_key in enumerate(rep_bs):
        samples = batch_seq_len_map[bs_key]
        sample_counts = Counter(samples)
        s_uppers = sorted(sample_counts.keys())
        s_pcts = [sample_counts[u] / len(samples) * 100 for u in s_uppers]
        clr = sample_colors[idx % len(sample_colors)]
        ax2.plot(s_uppers, s_pcts, color=clr, lw=1.8,
                 marker=markers[idx % len(markers)], markersize=4, alpha=0.85,
                 label=f"Sampled bs={bs_key}", zorder=5)
    ax2.set_xlabel("Sequence Length", fontsize=9)
    ax2.set_ylabel("Percentage (%)", fontsize=9)
    ax2.set_title("Original vs Sampled Distribution", fontsize=11, pad=8)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25, ls="--")
    ax2.set_xlim(left=0, right=max_seq_len + BUCKET_WIDTH / 2)
    ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    # ── bottom-right: summary table ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    stats = [
        ("Total prompts", f"{total:,}"),
        ("Buckets", f"{len(uppers)}"),
        ("Within cutoff", f"{within:,}  ({within / total * 100:.1f}%)"),
        ("Beyond cutoff",
         f"{total - within:,}  ({(total - within) / total * 100:.1f}%)"),
        ("P50", f"\u2264 {p50_val:,}  ({_fmt_k(p50_val)})"),
        ("P90", f"\u2264 {p90_val:,}  ({_fmt_k(p90_val)})"),
        ("P99", f"\u2264 {p99_val:,}  ({_fmt_k(p99_val)})"),
        ("Max bucket", f"\u2264 {uppers[-1]:,}  ({_fmt_k(uppers[-1])})"),
        ("Batch sizes",
         f"{len(sorted_bs)}  ({sorted_bs[0]}\u2013{sorted_bs[-1]})"
         if sorted_bs else "0"),
    ]
    table = ax3.table(cellText=[[k, v] for k, v in stats],
                      colLabels=["Metric", "Value"], cellLoc="left",
                      colWidths=[0.48, 0.52], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_facecolor(C_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        else:
            cell.set_facecolor("white")
    ax3.set_title("Summary Statistics", fontsize=11, pad=8)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "distribution_sampling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> saved: {path}")


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Universal distribution sampling for perf tests. "
        "Reads a distribution.csv and generates {batch_size: [seq_len_list]} JSON."
    )
    parser.add_argument("--csv", required=True, help="distribution.csv path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Truncate distribution at this value (0 = no truncation)")
    parser.add_argument("--bs_min", type=int, default=1)
    parser.add_argument("--bs_max", type=int, default=2048)
    parser.add_argument("--bs_step", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    batch_sizes = sorted(set(
        [args.bs_min] + list(range(args.bs_step, args.bs_max + 1, args.bs_step))
    ))

    print("[1/3] Loading distribution.csv ...")
    uppers, counts = load_histogram(args.csv)
    total = sum(counts)
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else uppers[-1]
    print(f"       {len(uppers)} buckets, {total:,} total samples, "
          f"range: [{uppers[0]:,} .. {uppers[-1]:,}]")
    print(f"       max_seq_len = {max_seq_len:,}")

    print("[2/3] Empirical CDF + stratified quantile sampling ...")
    trunc_uppers, cdf = build_empirical_cdf(uppers, counts, max_seq_len)
    within = sum(c for u, c in zip(uppers, counts) if u <= max_seq_len)
    print(f"       Truncated to {len(trunc_uppers)} buckets, "
          f"{within:,}/{total:,} samples ({within/total*100:.1f}%)")

    batch_seq_len_map = {}
    for bs in batch_sizes:
        samples = stratified_sample(bs, trunc_uppers, cdf)
        batch_seq_len_map[str(bs)] = samples

    print(f"       Generated {len(batch_sizes)} batch configs "
          f"(bs {batch_sizes[0]}..{batch_sizes[-1]})")

    unique_lens = sorted(set(sl for sls in batch_seq_len_map.values() for sl in sls))
    print(f"       {len(unique_lens)} unique seq_len values, "
          f"all {BUCKET_WIDTH}-aligned: "
          f"[{unique_lens[0]:,} .. {unique_lens[-1]:,}]")

    output = {
        "description": "Perf test seq_len_list via empirical CDF stratified sampling",
        "source_csv": os.path.basename(args.csv),
        "sampling_params": {
            "max_seq_len": max_seq_len,
            "bucket_width": BUCKET_WIDTH,
            "total_samples": total,
            "within_cutoff": within,
        },
        "batch_seq_len_map": batch_seq_len_map,
    }
    json_path = os.path.join(args.out, "batch_seq_len_config.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  -> saved: {json_path}")

    print("[3/3] Generating distribution plots ...")
    plot_distribution(uppers, counts, trunc_uppers, cdf, batch_seq_len_map,
                      max_seq_len, args.out)
    print("\nDone!")


if __name__ == "__main__":
    main()
