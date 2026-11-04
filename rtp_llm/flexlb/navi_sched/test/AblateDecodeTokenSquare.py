#!/usr/bin/env python3
"""Test an admission-time decode-token square term against later completions."""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from CompareLatencyOnlineRls import PARAMETER_SCALES


FIELD = re.compile(r"\b([a-z_]+)=([^ ]+)")
PARAMETERS = re.compile(
    r"revision_before=(\d+) revision_after=(\d+).*?"
    r"parameters_before=\[([^]]+)\] parameters_after=\[([^]]+)\]")
MASTER_MARKER = "[navi-master-admission-prediction]"
SAMPLE_MARKER = "[navi-latency-training-sample]"
UPDATE_MARKER = "[navi-latency-training-update]"
FEATURE_FIELDS = (
    "batch_count", "running_batch_size", "running_hit_tokens",
    "running_miss_tokens", "running_miss_square_sum", "running_hit_miss_sum",
    "waiting_batch_size", "waiting_hit_tokens", "waiting_miss_tokens",
    "waiting_miss_square_sum", "waiting_hit_miss_sum",
    "preceding_request_count", "running_elapsed_time_ms",
)
MODELS = {
    "constant": (0,),
    "linear": (0, 1),
    "square_only": (0, 2),
    "linear_and_square": (0, 1, 2),
}


def fields(line, marker):
    return dict(FIELD.findall(line.split(marker, 1)[1]))


def shape(value, curvature):
    root = np.sqrt(value * value + curvature)
    return np.where(value >= 0, value + root, curvature / (root - value))


def model_state(master, agent, parameters):
    values = [float(master[name]) for name in FEATURE_FIELDS]
    values.extend((float(agent["running_decode_request_count"]),
                   float(agent["running_decode_generated_tokens"])))
    x = np.asarray(values, dtype=np.float64)
    x[[2, 3, 7, 8, 14]] /= 1024.0
    x[[4, 5, 9, 10]] /= 1024.0 ** 2
    x[12] /= 1_000_000.0
    effective = parameters * PARAMETER_SCALES
    u = (effective[0] + x[1:6] @ effective[1:6] + x[0] * effective[6]
         + x[6:11] @ effective[10:15]
         + x[12] * effective[16] + x[13] * effective[17]
         + x[14] * effective[18]) / 320.0
    curvature = effective[8]
    zero_u = effective[0] / 320.0
    prediction = max(0.0, effective[7] + 1500.0 * (
        shape(u, curvature) - shape(zero_u, curvature))
        + effective[9] * (x[2] + x[3] + x[7] + x[8])
        + effective[15] * x[11])
    return float(u), float(curvature), float(prediction)


def load_rows(master_paths, agent_paths, start_us, end_us):
    master = {}
    for path in master_paths:
        with gzip.open(path, "rt", errors="replace") as stream:
            for line in stream:
                if MASTER_MARKER not in line:
                    continue
                item = fields(line, MASTER_MARKER)
                time_us = int(item["schedule_time_us"])
                if start_us <= time_us < end_us:
                    key = item["schedule_batch_id"]
                    if key in master and master[key] != item:
                        raise ValueError("conflicting Master admission rows")
                    master[key] = item
    revisions = {}
    samples = {}
    for path in agent_paths:
        host = ".".join(path.name[:-7].split("_")[-4:])
        with gzip.open(path, "rt", errors="replace") as stream:
            for line in stream:
                if UPDATE_MARKER in line:
                    match = PARAMETERS.search(line)
                    if match:
                        for revision, values in ((match.group(1), match.group(3)),
                                                 (match.group(2), match.group(4))):
                            key = (host, int(revision))
                            parameters = np.fromstring(values, sep=",")
                            if key in revisions and not np.array_equal(
                                    revisions[key], parameters):
                                raise ValueError("conflicting Agent model revisions")
                            revisions[key] = parameters
                elif SAMPLE_MARKER in line:
                    item = fields(line, SAMPLE_MARKER)
                    key = (host, item["schedule_batch_id"])
                    if key in samples and samples[key] != item:
                        raise ValueError("conflicting Agent training samples")
                    samples[key] = item
    rows = defaultdict(list)
    counts = defaultdict(int)
    for batch_id, admission in master.items():
        arch = "MI308" if admission["agent"].endswith(":27344") else (
            "RTX5000" if admission["agent"].endswith(":22914") else None)
        if arch is None:
            continue
        host = admission["agent"].split(":")[0]
        sample = samples.get((host, batch_id))
        if sample is None:
            counts[(arch, "missing_sample")] += 1
            continue
        parameters = revisions.get((host, int(admission["model_revision"])))
        if parameters is None:
            counts[(arch, "missing_revision")] += 1
            continue
        if len(parameters) != 19:
            raise ValueError("expected 19 model parameters")
        if abs(float(sample["master_prediction_ms"])
               - float(admission["final_execution_ms"])) > 0.001:
            raise ValueError("Master and Agent final predictions differ")
        u, curvature, recomputed = model_state(admission, sample, parameters)
        deviation = recomputed - float(admission["with_base_ms"])
        if abs(deviation) > 0.001:
            raise ValueError("Master base prediction reconstruction failed: "
                             + host + " deviation=" + str(deviation))
        agent_parameters = revisions.get(
            (host, int(sample["revision_before"].rstrip("]"))))
        if agent_parameters is None:
            counts[(arch, "missing_agent_revision")] += 1
            continue
        agent_u, agent_curvature, agent_recomputed = model_state(
            sample, sample, agent_parameters)
        if abs(agent_u - float(sample["nonlinear_u_before"])) > 0.001 or (
                abs(agent_recomputed - float(sample["base_prediction_before_ms"]))
                > 0.001):
            raise ValueError("Agent sample prediction reconstruction failed: " + host)
        feature_only_u, _, feature_only_base = model_state(
            sample, sample, parameters)
        parameter_only_u, _, parameter_only_base = model_state(
            admission, sample, agent_parameters)
        rows[arch].append({
            "host": host,
            "schedule_time_us": int(admission["schedule_time_us"]),
            "completion_time_us": int(sample["event_time_us"]),
            "master_revision": int(admission["model_revision"]),
            "agent_revision": int(sample["revision_before"].rstrip("]")),
            "actual": float(sample["target_ms"]),
            "baseline": float(admission["final_execution_ms"]),
            "master_baseline": float(admission["final_execution_ms"]),
            "master_base": recomputed,
            "master_with_final": float(admission["with_final_ms"]),
            "master_residual_ms": (float(admission["with_final_ms"])
                                   - recomputed),
            "master_fixed_ms": float(admission["fixed_execution_ms"]),
            "master_floor_base": float(admission["floor_base_ms"]),
            "agent_base": agent_recomputed,
            "agent_residual_ms": (float(sample["prediction_before_ms"])
                                  - agent_recomputed),
            "feature_only_base": feature_only_base,
            "parameter_only_base": parameter_only_base,
            "u": u,
            "curvature": curvature,
            "agent_u": float(sample["nonlinear_u_before"]),
            "agent_curvature": float(sample["nonlinear_curvature_before"]),
            "agent_baseline": float(sample["prediction_before_ms"]),
            "agent_u_recomputed": agent_u,
            "feature_only_u": feature_only_u,
            "parameter_only_u": parameter_only_u,
            "decode_tokens": float(sample["running_decode_generated_tokens"]),
            "decode_count": float(sample["running_decode_request_count"]),
            "waiting": float(admission["waiting_batch_size"]),
        })
    return rows, {"%s_%s" % key: value for key, value in counts.items()}


def score(rows, prediction):
    if not rows:
        return {"count": 0}
    actual = np.asarray([row["actual"] for row in rows])
    error = np.asarray(prediction) - actual
    return {
        "count": len(rows),
        "mae_ms": float(np.mean(np.abs(error))),
        "bias_ms": float(np.mean(error)),
        "p95_abs_ms": float(np.quantile(np.abs(error), 0.95)),
        "max_under_ms": float(np.max(-error)),
        "wape_pct": float(np.sum(np.abs(error)) / np.sum(actual) * 100),
    }


def difference_stats(rows, left, right):
    values = np.asarray([row[left] - row[right] for row in rows])
    return {"mean_ms": float(np.mean(values)),
            "mean_abs_ms": float(np.mean(np.abs(values))),
            "p95_abs_ms": float(np.quantile(np.abs(values), 0.95))}


def predictions(rows, center, powers, coefficients, normalizer):
    x = np.asarray([row["decode_tokens"] / normalizer - center for row in rows])
    basis = np.stack([x ** power for power in powers], axis=1)
    u = np.asarray([row["u"] for row in rows])
    curvature = np.asarray([row["curvature"] for row in rows])
    baseline = np.asarray([row["baseline"] for row in rows])
    delta = 1500.0 * (shape(u + basis @ coefficients, curvature)
                      - shape(u, curvature))
    return np.maximum(0.0, baseline + delta)


def evaluate(rows, test_start_us, test_end_us, normalizer=100000.0):
    train = [row for row in rows if row["schedule_time_us"] < test_start_us
             and row["completion_time_us"] < test_start_us]
    held = [row for row in rows if test_start_us <= row["schedule_time_us"]
            < test_end_us]
    if len(train) < 10 or len(held) < 10:
        raise ValueError("insufficient causally available training or held samples")
    center = float(np.mean([row["decode_tokens"] for row in train]) / normalizer)
    result = {"train_count": len(train), "held_count": len(held),
              "train_decode_tokens_mean": center * normalizer,
              "train_decode_tokens_range": [min(row["decode_tokens"] for row in train),
                                            max(row["decode_tokens"] for row in train)],
              "held_decode_tokens_range": [min(row["decode_tokens"] for row in held),
                                           max(row["decode_tokens"] for row in held)],
              "held_master_agent_final_difference": difference_stats(
                  held, "agent_baseline", "master_baseline"),
              "held_master_agent_base_difference": difference_stats(
                  held, "agent_base", "master_base"),
              "held_feature_effect_on_base": difference_stats(
                  held, "feature_only_base", "master_base"),
              "held_parameter_effect_on_base": difference_stats(
                  held, "agent_base", "feature_only_base"),
              "models": {}}
    result["models"]["original"] = {
        "train": score(train, [row["baseline"] for row in train]),
        "held": score(held, [row["baseline"] for row in held]),
    }
    target = np.asarray([row["actual"] for row in train])
    for name, powers in MODELS.items():
        fitted = least_squares(
            lambda coefficients: predictions(train, center, powers, coefficients,
                                             normalizer)
            - target,
            np.zeros(len(powers)), bounds=(-100.0, 100.0),
            max_nfev=1000,
        )
        if not fitted.success:
            raise RuntimeError(name + " fit did not converge")
        result["models"][name] = {
            "powers": powers, "coefficients": fitted.x.tolist(),
            "train": score(train, predictions(train, center, powers, fitted.x,
                                                normalizer)),
            "held": score(held, predictions(held, center, powers, fitted.x,
                                              normalizer)),
            "held_by_host": {
                host: score(
                    [row for row in held if row["host"] == host],
                    predictions([row for row in held if row["host"] == host],
                                center, powers, fitted.x, normalizer))
                for host in sorted({row["host"] for row in held})
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, nargs="+", required=True)
    parser.add_argument("--agent", type=Path, nargs="+", required=True)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--test-start-us", type=int, required=True)
    parser.add_argument("--test-end-us", type=int, required=True)
    parser.add_argument("--per-request", action="store_true",
                        help="use mean generated tokens per running decode request")
    parser.add_argument("--agent-aligned", action="store_true",
                        help="use the Agent prediction computed from the logged sample features")
    parser.add_argument("--base-only", action="store_true",
                        help="compare the pure formula without residual or Master floor")
    args = parser.parse_args()
    if not args.start_us < args.test_start_us < args.test_end_us:
        parser.error("expected start < test start < test end")
    rows, unmatched = load_rows(args.master, args.agent, args.start_us,
                                args.test_end_us)
    if args.agent_aligned:
        for items in rows.values():
            for row in items:
                row["u"] = row["agent_u"]
                row["curvature"] = row["agent_curvature"]
                row["baseline"] = row["agent_baseline"]
    if args.base_only:
        for items in rows.values():
            for row in items:
                row["baseline"] = (row["agent_base"] if args.agent_aligned
                                   else row["master_base"])
    if args.per_request:
        for items in rows.values():
            for row in items:
                row["decode_tokens"] /= max(1.0, row["decode_count"])
    report = {"prediction_basis": ("agent" if args.agent_aligned else "master")
              + ("_base" if args.base_only else "_final"),
              "decode_feature": "per_request" if args.per_request else "total",
              "train_label_available_before_us": args.test_start_us,
              "held_admission_start_us": args.test_start_us,
              "held_admission_end_us": args.test_end_us,
              "unmatched": unmatched,
              "architectures": {arch: evaluate(items, args.test_start_us,
                                                 args.test_end_us,
                                                 1000.0 if args.per_request
                                                 else 100000.0)
                                for arch, items in rows.items()}}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
