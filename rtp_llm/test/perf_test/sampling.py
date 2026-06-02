import json
import logging
import os
from typing import Any, Dict, List, Tuple

from rtp_llm.test.perf_test.dataset import BUCKET_WIDTH, DatasetLoader

__all__ = ["DistributionSampler", "prepare_distribution_config"]


class DistributionSampler:
    """Build empirical CDF from histogram and do stratified quantile sampling."""

    def __init__(
        self,
        uppers: List[int],
        counts: List[int],
        max_seq_len: int,
        max_concurrency: int,
        description: str = "",
    ):
        self._uppers = uppers
        self._counts = counts
        self._max_seq_len = max_seq_len
        self._max_concurrency = max_concurrency
        self._description = description or (
            f"Empirical CDF sampling "
            f"(max_seq_len={max_seq_len}, max_concurrency={max_concurrency})"
        )
        self._trunc_uppers, self._cdf = self._build_empirical_cdf(
            uppers, counts, max_seq_len
        )
        self._test_config: Dict[str, Any] = {}

    def sample(self) -> Dict[str, Any]:
        """Return test_config dict containing batch_seq_len_map."""
        batch_sizes = sorted(set([1] + list(range(8, self._max_concurrency + 1, 8))))
        batch_seq_len_map: Dict[str, List[int]] = {}
        for bs in batch_sizes:
            batch_seq_len_map[str(bs)] = self._stratified_sample(
                bs, self._trunc_uppers, self._cdf
            )

        logging.info(
            f"Sampled {len(batch_sizes)} batch sizes "
            f"(max_seq_len={self._max_seq_len}, "
            f"max_concurrency={self._max_concurrency}): "
            + ", ".join(str(bs) for bs in batch_sizes)
        )

        self._test_config = {
            "description": self._description,
            "sampling_params": {
                "max_seq_len": self._max_seq_len,
                "max_concurrency": self._max_concurrency,
            },
            "batch_seq_len_map": batch_seq_len_map,
        }
        return self._test_config

    def plot(self, output_dir: str) -> None:
        """Generate PDF + CDF sampling overview plot to output_dir."""
        if not self._test_config:
            return
        batch_seq_len_map = self._test_config["batch_seq_len_map"]
        self._plot_sampling_overview(
            self._uppers,
            self._counts,
            self._trunc_uppers,
            self._cdf,
            batch_seq_len_map,
            self._max_seq_len,
            output_dir,
        )

    def save(self, output_dir: str) -> None:
        """Save sampled_test_config.json to output_dir."""
        if not self._test_config:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "sampled_test_config.json")
        with open(path, "w") as f:
            json.dump(self._test_config, f, indent=2)
        logging.info(f"Saved sampled test config to {path}")

    @staticmethod
    def from_csv(
        csv_path: str,
        max_seq_len: int,
        max_concurrency: int,
        description: str = "",
    ) -> "DistributionSampler":
        """Create sampler from an existing distribution.csv."""
        uppers, counts = DatasetLoader.load_histogram_csv(csv_path)
        return DistributionSampler(
            uppers, counts, max_seq_len, max_concurrency, description
        )

    @staticmethod
    def _build_empirical_cdf(
        uppers: List[int], counts: List[int], max_seq_len: int
    ) -> Tuple[List[int], List[float]]:
        """Build empirical CDF truncated at max_seq_len."""
        trunc_uppers: List[int] = []
        trunc_counts: List[int] = []
        for u, c in zip(uppers, counts):
            if u > max_seq_len:
                break
            trunc_uppers.append(u)
            trunc_counts.append(c)
        total = sum(trunc_counts)
        if total == 0:
            raise ValueError(
                f"No histogram buckets within max_seq_len={max_seq_len}. "
                f"Bucket uppers start at {uppers[0] if uppers else '(empty)'}."
            )
        cdf: List[float] = []
        cumsum = 0
        for c in trunc_counts:
            cumsum += c
            cdf.append(cumsum / total)
        return trunc_uppers, cdf

    @staticmethod
    def _stratified_sample(bs: int, uppers: List[int], cdf: List[float]) -> List[int]:
        """Stratified quantile sampling from empirical CDF.
        Divide [0, 1] into bs equal strata, pick the bucket at each stratum midpoint.
        Returns bucket upper bounds (multiples of BUCKET_WIDTH)."""
        samples: List[int] = []
        for j in range(bs):
            q = (j + 0.5) / bs
            for i, c in enumerate(cdf):
                if c >= q:
                    samples.append(uppers[i])
                    break
            else:
                samples.append(uppers[-1])
        return samples

    @staticmethod
    def _plot_sampling_overview(
        uppers: List[int],
        counts: List[int],
        trunc_uppers: List[int],
        cdf: List[float],
        batch_seq_len_map: Dict[str, List[int]],
        max_seq_len: int,
        output_dir: str,
    ) -> None:
        """Generate distribution + sampling overview plot to output_dir."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            logging.warning("matplotlib not available, skipping sampling plots")
            return

        total = sum(counts)
        sorted_bs = sorted(batch_seq_len_map.keys(), key=lambda k: int(k))
        C_BLUE = "#3a7ca5"
        C_RED = "#d64045"
        C_GREEN = "#4a9c6d"
        C_ORANGE = "#e8a838"
        C_PURPLE = "#8172b3"
        sample_colors = [C_ORANGE, C_GREEN, C_PURPLE, C_RED, "#6dbbb0"]
        markers = ["o", "s", "D", "^", "v"]

        def _fmt_k(x: float, _: object = None) -> str:
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
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        gs = fig.add_gridspec(
            2,
            2,
            hspace=0.38,
            wspace=0.30,
            left=0.06,
            right=0.97,
            top=0.91,
            bottom=0.06,
        )

        # ── top-left: distribution area chart (dynamic x range) ──
        ax0 = fig.add_subplot(gs[0, 0])
        midpoints = [u - BUCKET_WIDTH / 2 for u in uppers]
        pcts = [c / total * 100 for c in counts]

        x_upper = max(max_seq_len, p99_val) * 1.15
        vis_mid = [m for m in midpoints if m <= x_upper]
        vis_pct = pcts[: len(vis_mid)]
        beyond_count = total - sum(
            c for u, c in zip(uppers, counts) if u - BUCKET_WIDTH / 2 <= x_upper
        )

        ax0.fill_between(vis_mid, vis_pct, alpha=0.35, color=C_BLUE, step="mid")
        ax0.step(vis_mid, vis_pct, where="mid", lw=1.2, color=C_BLUE)
        ax0.axvline(
            max_seq_len,
            color=C_RED,
            ls="--",
            lw=1.8,
            label=f"max_seq_len = {_fmt_k(max_seq_len)}",
        )
        pct_lines = [
            (p50_val, "P50", C_GREEN),
            (p90_val, "P90", C_ORANGE),
            (p99_val, "P99", C_PURPLE),
        ]
        seen_vals: set = set()
        for pval, plabel, clr in pct_lines:
            if pval in seen_vals:
                continue
            seen_vals.add(pval)
            if pval <= x_upper:
                ax0.axvline(pval, color=clr, ls=":", lw=1.3, alpha=0.8)
                y_top = max(vis_pct) if vis_pct else 1
                ax0.text(
                    pval + x_upper * 0.008,
                    y_top * 0.92,
                    f"{plabel}={_fmt_k(pval)}",
                    color=clr,
                    fontsize=7.5,
                    va="top",
                    ha="left",
                )
        if beyond_count > 0:
            ax0.annotate(
                f"+{beyond_count:,} beyond {_fmt_k(x_upper)}",
                xy=(1, 0.02),
                xycoords="axes fraction",
                fontsize=7.5,
                color="#888",
                ha="right",
                style="italic",
            )
        ax0.set_xlabel("Sequence Length", fontsize=9)
        ax0.set_ylabel("Percentage (%)", fontsize=9)
        ax0.set_title(
            f"Distribution  (n={total:,}, {len(uppers)} buckets)", fontsize=11, pad=8
        )
        ax0.legend(fontsize=8, loc="upper right")
        ax0.grid(axis="y", alpha=0.25, ls="--")
        ax0.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
        ax0.set_xlim(left=0, right=x_upper)

        # ── top-right: zoomed histogram (<= max_seq_len) ──
        ax1 = fig.add_subplot(gs[0, 1])
        zoom_idx = [(i, u) for i, u in enumerate(uppers) if u <= max_seq_len]
        zoom_uppers = [u for _, u in zoom_idx]
        zoom_counts = [counts[i] for i, _ in zoom_idx]
        zoom_pcts = [c / total * 100 for c in zoom_counts]
        zoom_mid = [u - BUCKET_WIDTH / 2 for u in zoom_uppers]

        bars = ax1.bar(
            zoom_mid,
            zoom_pcts,
            width=BUCKET_WIDTH * 0.85,
            color=C_BLUE,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        max_pct = max(zoom_pcts) if zoom_pcts else 1
        for bar, pct, cnt in zip(bars, zoom_pcts, zoom_counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_pct * 0.015,
                f"{pct:.1f}%\n({cnt:,})",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        if sorted_bs:
            largest_key = sorted_bs[-1]
            for s in batch_seq_len_map[largest_key]:
                ax1.axvline(s, color=C_GREEN, alpha=0.5, lw=0.7)
            ax1.plot(
                [], [], color=C_GREEN, lw=1.5, label=f"sample points (bs={largest_key})"
            )
        ax1.set_xlabel("Sequence Length", fontsize=9)
        ax1.set_ylabel("Percentage (%)", fontsize=9)
        ax1.set_title(f"Zoomed  (0 \u2013 {_fmt_k(max_seq_len)})", fontsize=11, pad=8)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.25, ls="--")
        ax1.set_ylim(top=max_pct * 1.30)
        ax1.set_xlim(left=0, right=max_seq_len + BUCKET_WIDTH / 2)
        ax1.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

        # ── bottom-left: original vs sampled distribution ──
        ax2 = fig.add_subplot(gs[1, 0])
        trunc_total = sum(c for u, c in zip(uppers, counts) if u <= max_seq_len)
        if trunc_total > 0:
            orig_pcts = [
                c / trunc_total * 100
                for u, c in zip(uppers, counts)
                if u <= max_seq_len
            ]
            orig_mids = [u - BUCKET_WIDTH / 2 for u in uppers if u <= max_seq_len]
            ax2.bar(
                orig_mids,
                orig_pcts,
                width=BUCKET_WIDTH * 0.85,
                color=C_BLUE,
                alpha=0.35,
                label="Original (within cutoff)",
            )

        rep_bs = []
        if sorted_bs:
            if len(sorted_bs) >= 3:
                rep_bs = [sorted_bs[0], sorted_bs[len(sorted_bs) // 2], sorted_bs[-1]]
            else:
                rep_bs = list(sorted_bs)
        from collections import Counter as _Counter

        for idx, bs_key in enumerate(rep_bs):
            samples = batch_seq_len_map[bs_key]
            sample_counts = _Counter(samples)
            s_uppers = sorted(sample_counts.keys())
            s_pcts = [sample_counts[u] / len(samples) * 100 for u in s_uppers]
            clr = sample_colors[idx % len(sample_colors)]
            ax2.plot(
                s_uppers,
                s_pcts,
                color=clr,
                lw=1.8,
                marker=markers[idx % len(markers)],
                markersize=4,
                alpha=0.85,
                label=f"Sampled bs={bs_key}",
                zorder=5,
            )
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
            (
                "Beyond cutoff",
                f"{total - within:,}  ({(total - within) / total * 100:.1f}%)",
            ),
            ("P50", f"\u2264 {p50_val:,}  ({_fmt_k(p50_val)})"),
            ("P90", f"\u2264 {p90_val:,}  ({_fmt_k(p90_val)})"),
            ("P99", f"\u2264 {p99_val:,}  ({_fmt_k(p99_val)})"),
            ("Max bucket", f"\u2264 {uppers[-1]:,}  ({_fmt_k(uppers[-1])})"),
            (
                "Batch sizes",
                (
                    f"{len(sorted_bs)}  ({sorted_bs[0]}\u2013{sorted_bs[-1]})"
                    if sorted_bs
                    else "0"
                ),
            ),
        ]
        table = ax3.table(
            cellText=[[k, v] for k, v in stats],
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colWidths=[0.48, 0.52],
            loc="center",
        )
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

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "distribution_sampling.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logging.info(f"Saved sampling overview to {path}")


def prepare_distribution_config(
    tokenizer_path: str,
    max_seq_len: int,
    max_concurrency: int,
    result_dir: str,
    dataset_name: str = "",
    dataset_path: str = "",
    dataset_csv: str = "",
    test_json: str = "",
) -> Dict[str, Any]:
    """Prepare distribution test_config from one of the supported sources.

    Handles dataset JSON (with tokenization + histogram + sampling),
    pre-built distribution CSV, or a previously saved test_config JSON.
    """
    if dataset_name or dataset_path:
        loader = DatasetLoader(
            tokenizer_path,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
        uppers, counts = loader.load_histogram()

        csv_path = os.path.join(result_dir, "distribution.csv")
        DatasetLoader.save_histogram_csv(uppers, counts, csv_path)

        source = dataset_name or os.path.basename(dataset_path)
        sampler = DistributionSampler(
            uppers,
            counts,
            max_seq_len,
            max_concurrency,
            description=f"{source} dataset sampling "
            f"(max_seq_len={max_seq_len}, max_concurrency={max_concurrency})",
        )
        test_config = sampler.sample()
        sampler.plot(result_dir)
        sampler.save(result_dir)
        return test_config

    if dataset_csv:
        sampler = DistributionSampler.from_csv(
            dataset_csv, max_seq_len, max_concurrency
        )
        test_config = sampler.sample()
        sampler.plot(result_dir)
        sampler.save(result_dir)
        return test_config

    if test_json:
        with open(test_json) as f:
            return json.load(f)

    raise ValueError(
        "Must provide one of: dataset_name, dataset_path, dataset_csv, test_json"
    )
