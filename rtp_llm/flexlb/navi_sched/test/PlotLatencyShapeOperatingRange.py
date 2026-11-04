#!/usr/bin/env python3
"""Plot fitted inner-input ranges and representative shape curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host, shape
from CompareLatencyAugmentedSquareShape import inner_input


MODELS = ("original", "square")
GROUPS = (("A", "MI"), ("A", "RTX"), ("B", "MI"), ("B", "RTX"))
COLORS = {"low": "#26734d", "high": "#c34436"}


def load_fitted(source, results):
    fitted = {}
    for label, directory in source.items():
        for model in MODELS:
            rows = json.loads(results[(label, model)].read_text())["hosts"]
            for host in rows:
                inputs, _, _, _, _, _, _ = load_host(directory / (host["host"] + ".jsonl"))
                parameters = np.asarray(host["fit_parameters"])
                u = inner_input(inputs, parameters)
                fitted[(label, host["architecture"], model, host["host"])] = {
                    "u": u, "parameters": parameters,
                    "low": (inputs[:, 1] <= 4) & (inputs[:, 7] <= 4),
                    "high": inputs[:, 7] >= 33,
                }
    return fitted


def draw_ranges(fitted, model, output):
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    for axis, (label, architecture) in zip(axes.flat, GROUPS):
        hosts = sorted((host, row) for (window, arch, kind, host), row in fitted.items()
                       if (window, arch, kind) == (label, architecture, model))
        for index, (host, row) in enumerate(hosts):
            for pressure, offset in (("low", -0.13), ("high", 0.13)):
                values = row["u"][row[pressure]]
                if not len(values):
                    continue
                left, middle, right = np.percentile(values, [5, 50, 95])
                y = index + offset
                axis.plot([left, right], [y, y], color=COLORS[pressure], lw=2)
                axis.plot(middle, y, "o", color=COLORS[pressure], ms=4)
        axis.set_xscale("symlog", linthresh=1.0)
        axis.axvline(0, color="#777777", lw=0.8)
        axis.set_yticks(range(len(hosts)), [host.rsplit("_", 1)[-1] for host, _ in hosts])
        axis.set_title(f"{label} / {architecture} / {model}")
        axis.set_xlabel("Fitted inner u (5th-95th percentile, dot=median)")
        axis.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=color, lw=2, marker="o", label=pressure)
               for pressure, color in COLORS.items()]
    figure.legend(handles=handles, loc="upper center", ncol=2)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def draw_curves(fitted, output):
    figure, axes = plt.subplots(4, 2, figsize=(16, 16), layout="constrained")
    for row_index, (label, architecture) in enumerate(GROUPS):
        original = [(host, row) for (window, arch, model, host), row in fitted.items()
                    if (window, arch, model) == (label, architecture, "original")]
        median_k = np.median([row["parameters"][8] for _, row in original])
        host, _ = min(original, key=lambda item: abs(item[1]["parameters"][8] - median_k))
        for column, model in enumerate(MODELS):
            axis = axes[row_index, column]
            row = fitted[(label, architecture, model, host)]
            u = row["u"]
            parameters = row["parameters"] * PARAMETER_SCALES[:17]
            zero_u = parameters[0] / 320.0
            grid = np.unique(np.percentile(u, np.linspace(1, 99, 501)))
            if model == "original":
                curvature = parameters[8]
                contribution = 1500.0 * (shape(grid, curvature) -
                                         shape(np.asarray([zero_u]), curvature)[0])
            else:
                contribution = 1500.0 * (grid * grid - zero_u * zero_u)
            axis.plot(grid, contribution, color="#244466", lw=2)
            for pressure in ("low", "high"):
                values = u[row[pressure]]
                if len(values):
                    low, high = np.percentile(values, [10, 90])
                    axis.axvspan(low, high, color=COLORS[pressure], alpha=0.12)
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_yscale("symlog", linthresh=100.0)
            axis.set_title(f"{label} / {architecture} / {model} / node {host}")
            axis.set_xlabel("Fitted inner u")
            axis.set_ylabel("Shape contribution (ms)")
            axis.grid(alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-source", type=Path, required=True)
    parser.add_argument("--b-source", type=Path, required=True)
    for label in ("a", "b"):
        for model in MODELS:
            parser.add_argument(f"--{label}-{model}", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    source = {"A": args.a_source, "B": args.b_source}
    results = {(label, model): getattr(args, f"{label.lower()}_{model}")
               for label in source for model in MODELS}
    fitted = load_fitted(source, results)
    for model in MODELS:
        draw_ranges(fitted, model,
                    args.output_prefix.with_name(args.output_prefix.name + f"-{model}-u.png"))
    draw_curves(fitted, args.output_prefix.with_name(args.output_prefix.name + "-curves.png"))


if __name__ == "__main__":
    main()
