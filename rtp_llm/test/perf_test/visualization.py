"""Generate a single PNG plot from any combination of perf_test result JSONs.

All result modes (grid / distribution / tps) reduce to the same shape:
(input_len_label, batch_size) -> avg_decode_time_ms. The plotter scans
``result_dir/*.json``, groups by label, draws one line per label, and
overlays target_tpot + best_bs annotations when present.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

# (label, batch_size, tpot_ms)
DataPoint = Tuple[str, int, float]

# Filenames in result_dir that are not result JSONs.
_NON_RESULT_JSONS = {"test_info.json", "test_meta.json"}


def _extract_points(
    json_path: str,
) -> Tuple[List[DataPoint], float, Dict[str, int]]:
    """Return (points, target_tpot, best_bs_per_label).

    target_tpot=0 if not TPS mode; best_bs_per_label empty if not TPS mode.
    """
    with open(json_path) as f:
        data = json.load(f)
    mode = data.get("mode", "")
    pts: List[DataPoint] = []
    target_tpot = 0.0
    best_bs: Dict[str, int] = {}

    if mode == "tps":
        target_tpot = data.get("target_tpot", 0.0)
        for r in data.get("results", []):
            label = (
                f"seq{r['input_len']}" if r.get("input_len", 0) > 0 else "distribution"
            )
            for s in r.get("search_steps", []):
                pts.append((label, s["batch_size"], s["avg_decode_time"]))
            if r.get("best_bs", 0) > 0:
                best_bs[label] = r["best_bs"]
    elif mode == "grid":
        for m in data.get("metrics", []):
            label = f"seq{m['input_len']}"
            tpot = m.get("avg_decode_time") or m.get("avg_prefill_time", 0.0)
            pts.append((label, m["batch_size"], tpot))
    elif mode == "distribution":
        for c in data.get("test_cases", []):
            pts.append(
                ("distribution", c["batch_size"], c["avg_decode_time_per_token"])
            )
    return pts, target_tpot, best_bs


def _label_sort_key(label: str) -> Tuple[int, int]:
    """Sort labels by numeric input_len ascending; ``distribution`` last."""
    if label == "distribution":
        return (1, 0)
    try:
        return (0, int(label.replace("seq", "")))
    except ValueError:
        return (0, 0)


def plot_decode_results(result_dir: str) -> None:
    """Scan result_dir/*.json -> aggregate (label, BS, TPOT) -> one PNG.

    Failures only warn; never raise. PNG is written to
    ``{result_dir}/plot_decode.png``.
    """
    if not os.path.isdir(result_dir):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available, skipping decode plot")
        return

    all_points: List[DataPoint] = []
    target_tpot = 0.0
    best_bs: Dict[str, int] = {}
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".json") or fname in _NON_RESULT_JSONS:
            continue
        try:
            pts, tt, bb = _extract_points(os.path.join(result_dir, fname))
        except Exception as e:
            logging.warning(f"plot_decode_results: skip {fname}: {e}")
            continue
        all_points.extend(pts)
        if tt > 0:
            target_tpot = tt
        best_bs.update(bb)

    if not all_points:
        logging.info("plot_decode_results: no data points to plot")
        return

    by_label: Dict[str, List[Tuple[int, float]]] = {}
    for label, bs, tpot in all_points:
        by_label.setdefault(label, []).append((bs, tpot))

    colors = ["#3a7ca5", "#d64045", "#4a9c6d", "#e8a838", "#8172b3", "#6dbbb0"]
    markers = ["o", "s", "D", "^", "v", "P"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    for i, label in enumerate(sorted(by_label, key=_label_sort_key)):
        pts = sorted(by_label[label])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        legend = label
        if label in best_bs:
            legend += f" (best_bs={best_bs[label]})"
        color = colors[i % len(colors)]
        ax.plot(
            xs,
            ys,
            marker=markers[i % len(markers)],
            color=color,
            linewidth=2,
            markersize=7,
            label=legend,
        )
        # Annotate each point with TPS = BS / TPOT_seconds
        for bs, tpot in pts:
            if tpot <= 0:
                continue
            tps = bs * 1000.0 / tpot
            ax.annotate(
                f"{tps:.0f}",
                xy=(bs, tpot),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                alpha=0.85,
            )

    # Find best TPS across all points; in TPS mode restrict to feasible set
    best_tps = 0.0
    best_pt: Optional[Tuple[str, int, float]] = None
    for label, pts in by_label.items():
        for bs, tpot in pts:
            if tpot <= 0:
                continue
            if target_tpot > 0 and tpot > target_tpot:
                continue
            tps = bs * 1000.0 / tpot
            if tps > best_tps:
                best_tps = tps
                best_pt = (label, bs, tpot)
    if best_pt is not None:
        b_label, b_bs, b_tpot = best_pt
        ax.scatter(
            [b_bs],
            [b_tpot],
            s=320,
            marker="*",
            color="#ffd700",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label=f"BEST TPS={best_tps:.0f} ({b_label}, BS={b_bs}, TPOT={b_tpot:.1f}ms)",
        )

    if target_tpot > 0:
        ax.axhline(
            y=target_tpot,
            color="#888",
            linestyle="--",
            linewidth=1.5,
            label=f"target_tpot={target_tpot:.1f}ms",
        )

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("TPOT (ms)", fontsize=12)
    title = "Decode TPOT vs Batch Size  (numbers above points = TPS = BS/TPOT_s)"
    if target_tpot > 0:
        title += f"\nTPS mode, target_tpot={target_tpot:.1f}ms"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    out_path = os.path.join(result_dir, "plot_decode.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info(f"Decode plot saved to {out_path}")
