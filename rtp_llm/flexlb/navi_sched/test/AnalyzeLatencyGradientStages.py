#!/usr/bin/env python3
"""Audit the original 16-parameter replay's gradient normalization stages."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from CompareLatencyOnlineRls import PARAMETER_SCALES, prediction_and_jacobian


ACTIVE = np.array([index for index in range(17) if index != 6])
ONES = np.ones(19, dtype=np.float64)


def load_history(directory):
    history = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        samples = {}
        updates = {}
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] == "sample":
                    if "history_sequence" in event:
                        samples[event["history_sequence"]] = event
                elif event["type"] == "update":
                    updates[event["sequence"]] = event["fit_batches"]
        history[path.stem] = (samples, updates)
    return history


def audit(history, trace, hardware, coordinate_scales, limit=None,
          verify_original_scale=False, calibration_output=None):
    sums = defaultdict(lambda: np.zeros((5, 19), dtype=np.float64))
    counts = defaultdict(int)
    largest_gradient_error = 0.0
    largest_relative_error = 0.0
    worst = None
    total_samples = 0
    first_errors = []
    with Path(trace).open() as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "D":
                continue
            host = fields[1]
            sequence = int(fields[4])
            fit_index = int(fields[5])
            samples, updates = history[host]
            batch_ids = updates[sequence][fit_index]
            batch = [samples[index] for index in batch_ids]
            if (batch[-1]["event_time_us"] != int(fields[6]) or
                    not math.isclose(batch[-1]["target_ms"], float(fields[7]),
                                     rel_tol=0.0, abs_tol=1e-9)):
                raise RuntimeError("trace sample does not match fit batch")
            parameters = np.array(
                [float(fields[14 + 15 * index + 8]) for index in range(19)],
                dtype=np.float64,
            )
            prediction = []
            jacobian = []
            targets = []
            for sample in batch:
                value, derivative = prediction_and_jacobian(
                    sample["features"], parameters, coordinate_scales
                )
                prediction.append(value)
                jacobian.append(derivative)
                targets.append(sample["target_ms"])
            prediction = np.array(prediction)
            jacobian = np.array(jacobian)
            targets = np.array(targets)
            weights = np.ones(len(batch))
            weights[-1] = float(fields[2])
            weights /= weights.sum()
            absolute = np.abs(jacobian)
            log_derivative = jacobian / (1.0 + prediction[:, None])
            log_error = np.log1p(prediction) - np.log1p(targets)
            loss_derivative = log_error[:, None] * log_derivative
            mse_derivative = (prediction - targets)[:, None] * jacobian
            stages = np.stack([
                weights @ absolute,
                weights @ np.abs(log_derivative),
                weights @ np.abs(loss_derivative),
                np.abs(weights @ loss_derivative),
                np.abs(weights @ mse_derivative),
            ])
            native_gradient = np.array(
                [float(fields[14 + 15 * index]) for index in range(19)]
            )
            signed_gradient = weights @ loss_derivative
            error = np.abs(native_gradient[ACTIVE] - signed_gradient[ACTIVE])
            relative = error / np.maximum(1e-12, np.abs(native_gradient[ACTIVE]))
            if np.max(error) > largest_gradient_error:
                largest_gradient_error = float(np.max(error))
                worst = (host, sequence, fit_index, int(ACTIVE[np.argmax(error)]))
            largest_relative_error = max(largest_relative_error, float(np.max(relative)))
            if len(first_errors) < 5 and np.max(error) > 1e-8:
                first_errors.append((host, sequence, fit_index,
                                     int(ACTIVE[np.argmax(error)]),
                                     float(native_gradient[ACTIVE[np.argmax(error)]]),
                                     float(signed_gradient[ACTIVE[np.argmax(error)]]),
                                     float(prediction[-1]), float(targets[-1])))
            group = hardware[host]
            sums[group] += stages
            sums["all"] += stages
            counts[group] += 1
            counts["all"] += 1
            total_samples += len(batch)
            if limit is not None and counts["all"] >= limit:
                break
    print("updates", counts["all"], "batch_sample_uses", total_samples)
    print("native_gradient_max_abs_error", largest_gradient_error, "at", worst)
    print("native_gradient_max_rel_error", largest_relative_error)
    print("first_errors", first_errors)
    if largest_gradient_error > 1e-8:
        raise RuntimeError("Python batch gradient disagrees with native trace")
    for group in sorted(counts):
        stages = sums[group] / counts[group]
        medians = np.median(stages[:, ACTIVE], axis=1)
        if group == "all" and calibration_output is not None:
            prediction_scales = coordinate_scales.copy()
            loss_scales = coordinate_scales.copy()
            log_abs_scales = coordinate_scales.copy()
            mse_loss_scales = coordinate_scales.copy()
            prediction_scales[ACTIVE] *= medians[0] / stages[0, ACTIVE]
            loss_scales[ACTIVE] *= medians[3] / stages[3, ACTIVE]
            log_abs_scales[ACTIVE] *= medians[2] / stages[2, ACTIVE]
            mse_loss_scales[ACTIVE] *= medians[4] / stages[4, ACTIVE]
            result = {
                "updates": counts[group],
                "current_scales": coordinate_scales.tolist(),
                "stage_means": stages.tolist(),
                "prediction_scales": prediction_scales.tolist(),
                "loss_scales": loss_scales.tolist(),
                "log_abs_scales": log_abs_scales.tolist(),
                "mse_loss_scales": mse_loss_scales.tolist(),
            }
            Path(calibration_output).write_text(json.dumps(result, indent=2))
        print("group", group, "updates", counts[group])
        print("stage_medians", *medians)
        print("prediction_derivative_scales")
        for index in ACTIVE:
            print(index, f"{medians[0] / stages[0, index]:.12g}")
        if group == "all" and limit is None and verify_original_scale:
            reconstructed = medians[3] / stages[3, ACTIVE]
            relative_error = np.max(np.abs(
                reconstructed / PARAMETER_SCALES[ACTIVE] - 1.0
            ))
            print("original_scale_max_relative_error", relative_error)
            if relative_error > 1e-8:
                raise RuntimeError("old scale does not match original replay")
        print("index prediction log_chain error_weighted batch_signed mse_batch log_to_prediction batch_to_abs_error")
        for index in ACTIVE:
            values = stages[:, index]
            log_ratio = values[1] / values[0] if values[0] else math.nan
            cancellation = values[3] / values[2] if values[2] else math.nan
            print(index, *(f"{value:.9g}" for value in values),
                  f"{log_ratio:.9g}", f"{cancellation:.9g}")
        print("relative_to_median")
        for index in ACTIVE:
            print(index, *(f"{value:.6g}" for value in stages[:, index] / medians))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_dir")
    parser.add_argument("trace")
    parser.add_argument("manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scales", default=",".join(["1"] * 19))
    parser.add_argument("--use-model-scales", action="store_true")
    parser.add_argument("--verify-original-scale", action="store_true")
    parser.add_argument("--calibration-output")
    args = parser.parse_args()
    hardware = {}
    with Path(args.manifest).open() as stream:
        manifest = json.load(stream)
    for node in manifest["nodes"]:
        hardware[node["ip"].replace(".", "_")] = (
            "MI308" if "MI308" in node["deployment"] else "RTX5000"
        )
    coordinate_scales = (PARAMETER_SCALES if args.use_model_scales else
                         np.fromstring(args.scales, sep=","))
    if coordinate_scales.shape != ONES.shape or np.any(coordinate_scales <= 0):
        parser.error("--scales requires 19 positive values")
    audit(load_history(args.history_dir), args.trace, hardware,
          coordinate_scales, args.limit, args.verify_original_scale,
          args.calibration_output)


if __name__ == "__main__":
    main()
