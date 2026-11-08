"""Qualification and charts for the DSV4.1 PPU 128K step benchmark."""

import csv
import json
from pathlib import Path

WORLD = 8
CONTEXT = 128 * 1024
FRESH = 4096
PREFIX = CONTEXT - FRESH
OUTPUT = 512
MAX_SEQUENCE = 131840
TAU = 3.25
WARMUP = {"prefill": 5, "decode": 50}
MEASURE = {"prefill": 10, "decode": 60}
POINTS = {
    "prefill": (4096, 8192, 16384, 32768, 65536),
    "decode": (8, 16, 32, 64, 128, 256),
}


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_steps(rows):
    previous_step, previous_end = 0, 0
    for row in rows:
        if not previous_step < row["step"] or not previous_end <= row["start_ns"] < row["end_ns"]:
            raise ValueError("non-monotone native step receipts")
        previous_step, previous_end = row["step"], row["end_ns"]


def matches_context(row):
    return row["input_min"] == row["input_max"] == CONTEXT


def prefill_result(path, chunk):
    rows = read_rows(path / "step-rank0.jsonl")
    validate_steps(rows)
    batch = chunk // FRESH
    selected = [row for row in rows if matches_context(row)]
    required = WARMUP["prefill"] + MEASURE["prefill"]
    if len(selected) != required:
        raise ValueError(f"expected {required} full prefill steps, observed {len(selected)}")
    for row in selected:
        if (
            row["batch"] != batch or row["context_batch"] != batch
            or row["prefix_min"] != PREFIX or row["prefix_max"] != PREFIX
            or row["fresh_tokens"] != chunk or row["emitted_tokens"] != batch
        ):
            raise ValueError(f"prefill step has wrong cache hit or fresh batch: {row}")
    measured = selected[WARMUP["prefill"]:]
    seconds = sum(row["end_ns"] - row["start_ns"] for row in measured) / 1e9
    return {
        "steps": len(measured),
        "step_ms": 1000 * seconds / len(measured),
        "throughput_per_ppu": sum(row["fresh_tokens"] for row in measured) / seconds / WORLD,
        "sample_steps": [row["step"] for row in measured],
    }


def qualify_prefill_optimizations(path, chunk):
    batch = chunk // FRESH
    expected = WARMUP["prefill"] + MEASURE["prefill"]
    for rank in range(WORLD):
        rows = [row for row in read_rows(path / f"prefill-rank{rank}.jsonl")
                if row["lengths"] == [FRESH] * batch and row["prefixes"] == [PREFIX] * batch]
        if len(rows) != expected:
            raise ValueError(f"rank {rank}: observed {len(rows)} true suffix forwards, expected {expected}")
        for row in rows:
            if (row["plan"] != "compact140" or not row["compacted"]
                    or row["dense_fp4_layers"] != [2, 8, 14, 20]
                    or row["sparse_fp4_layers"] != [24, 28, 32, 36]):
                raise ValueError(f"rank {rank}: missing compact140 or native FP4 execution: {row}")


def decode_result(path, batch):
    ranks, replays, output_counts = [], [], []
    for rank in range(WORLD):
        rows = read_rows(path / f"step-rank{rank}.jsonl")
        validate_steps(rows)
        output_counts.append(sum(row["emitted_tokens"] for row in rows if matches_context(row)))
        ranks.append({
            row["step"]: row for row in rows
            if matches_context(row) and row["batch"] == batch and row["context_batch"] == 0
            and row["output_max"] < OUTPUT
        })
        replays.append(read_rows(path / f"replay-rank{rank}.jsonl"))

    full_steps = sorted(set.intersection(*(set(rows) for rows in ranks)))
    required = WARMUP["decode"] + MEASURE["decode"]
    window = []
    for step in full_steps:
        rows = [rank[step] for rank in ranks]
        # All owners run on one host, with a common monotonic clock. The overlap
        # check rejects shifted step IDs and late owners during KV admission.
        if max(row["start_ns"] for row in rows) >= min(row["end_ns"] for row in rows):
            window.clear()
            continue
        identity = [row["batch_identity"] for row in rows]
        if window and (step != window[-1][0] + 1 or identity != window[-1][1]):
            window.clear()
        window.append((step, identity, rows))
        if len(window) == required:
            break
    if len(window) != required:
        raise ValueError(f"no {required}-step stable window across all eight owners")

    measured = window[WARMUP["decode"]:]
    for _, _, rows in measured:
        for rank, row in enumerate(rows):
            modes = {
                mode for stamp, mode, bucket, visible_indexer in replays[rank]
                if row["start_ns"] <= stamp <= row["end_ns"] and bucket == batch
                and (mode != "dense" or visible_indexer)
            }
            if not {"proposal", "dense", "commit"} <= modes:
                raise ValueError(f"rank {rank} step {row['step']} lacks full DSpARK Graph replay")

    window_start = max(row["end_ns"] for row in window[WARMUP["decode"] - 1][2])
    window_end = max(row["end_ns"] for row in measured[-1][2])
    seconds = (window_end - window_start) / 1e9
    steps = len(measured)
    users = batch * WORLD
    normalized = TAU * steps / seconds
    # Async dispatch may publish between engine intervals or in bursts. The
    # full HTTP wave returns OUTPUT tokens per request, including the anchor
    # from Prefill. Publish an observed rate only if every D token was counted.
    complete_counter = all(count == batch * (OUTPUT - 1) for count in output_counts)
    metrics = {
        "steps": steps,
        "step_ms": 1000 * seconds / steps,
        "normalized_user_tps": normalized,
        "throughput_per_ppu": batch * normalized,
        "decode_output_counter_complete": complete_counter,
        "decode_output_tokens_counted": sum(output_counts),
        "decode_output_tokens_expected": users * (OUTPUT - 1),
        "sample_steps": [step for step, _, _ in measured],
    }
    if complete_counter:
        emitted = sum(row["emitted_tokens"] for _, _, rows in measured for row in rows)
        metrics.update(actual_user_tps=emitted / seconds / users,
                       actual_tps_per_ppu=emitted / seconds / WORLD, emitted_tokens=emitted)
    return metrics


def plot_results(directory, phases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    rows = []
    for phase in phases:
        results = sorted(
            (json.loads(path.read_text()) for path in directory.glob(f"{phase}-*/result.json")),
            key=lambda row: row["point"],
        )
        rows.extend(results)
        good = [row for row in results if row["status"] == "ok"]
        decode = phase == "decode"
        x = "normalized_user_tps" if decode else "step_ms"
        title = "Decode · DP8 + EP8" if decode else "Prefill · CP8 + EP8"
        description = (
            "128K initial KV  |  DSpARK γ=3  |  fixed 3.25 tokens per step"
            if decode else "128K context  |  126,976 cached + 4,096 new tokens per request  |  YoCo compact140"
        )
        fig, ax = plt.subplots(figsize=(12, 7.2))
        fig.subplots_adjust(top=0.82, bottom=0.23, left=0.12, right=0.95)
        fig.text(0.12, 0.94, f"RTP-LLM DSV4.1 Flash PPU · {title}", fontsize=19, weight="bold")
        fig.text(0.12, 0.89, description, fontsize=11, color="#46566a")
        for index, row in enumerate(results):
            if row["status"] != "ok":
                continue
            label = f"B{row['point']}" if decode else f"{row['point']:,}"
            ax.annotate(
                f"{label}\n({row[x]:,.1f}, {row['throughput_per_ppu']:,.0f})",
                (row[x], row["throughput_per_ppu"]), xytext=(0, 10 + (index % 2 * 12 if decode else 0)),
                textcoords="offset points", ha="center", fontsize=9, color="#006d60",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
        for left, right in zip(results, results[1:]):
            if left["status"] == right["status"] == "ok" and (
                POINTS[phase].index(right["point"]) - POINTS[phase].index(left["point"]) == 1
            ):
                ax.plot([left[x], right[x]], [left["throughput_per_ppu"], right["throughput_per_ppu"]], color="#108672", lw=2)
        ax.scatter([row[x] for row in good], [row["throughput_per_ppu"] for row in good], color="#108672", zorder=3)
        ax.set_xlabel("Per-user speed (tokens/s/user)" if decode else "Latency (ms/prefill step)")
        ax.set_ylabel(("Output" if decode else "Non-cached input") + " throughput (tokens/s/PPU)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#e2e6e8")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if good:
            ax.set_xlim(0, max(row[x] for row in good) * 1.15)
            ax.set_ylim(0, max(row["throughput_per_ppu"] for row in good) * 1.24)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No qualified samples", transform=ax.transAxes, ha="center")
        labels = {"oom": "OOM", "capacity": "KV capacity"}
        missing = [f"{'B' if decode else ''}{row['point']:,} {labels.get(row['status'], row['status'])}"
                   for row in results if row["status"] != "ok"]
        if missing:
            fig.text(0.12, 0.115, "Missing points: " + ", ".join(missing), fontsize=10, color="#8a4b25")
        note = (
            "B is requests per DP rank. Both axes use the fixed 3.25-token convention."
            if decode else "Point labels count new tokens across the CP8 group. Throughput is divided by eight PPUs."
        )
        fig.text(0.12, 0.067, note, fontsize=10, color="#46566a")
        for extension in ("png", "svg"):
            fig.savefig(directory / f"{phase}.{extension}", dpi=180)
        plt.close(fig)

    columns = sorted({key for row in rows for key in row if key != "sample_steps"})
    with (directory / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
