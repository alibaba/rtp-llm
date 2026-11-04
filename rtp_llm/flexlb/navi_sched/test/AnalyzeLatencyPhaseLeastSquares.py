#!/usr/bin/env python3
"""Fit the deployed v21 base formula by node and observed queue-pressure phase."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize


PARAMETER_SCALES = np.asarray([
    0.878862443703,
    0.119054644545,
    0.0881092493705,
    2.45290259606,
    0.918359382716,
    19.5646708383,
    1.0,
    1.23972355201,
    0.00229934153041,
    0.211011741932,
    1.09757235126,
    0.878792874576,
    193.929503215,
    222.305482002,
    207.484099949,
    0.0242867645882,
    1892.09311928,
    0.0299401936353,
    0.403880546944,
])
ACTIVE = np.array([index for index in range(17) if index != 6])
LOWER = np.full(len(ACTIVE), -np.inf)
LOWER[np.where(ACTIVE == 7)[0][0]] = 0.0
LOWER[np.where(ACTIVE == 8)[0][0]] = 1e-8 / PARAMETER_SCALES[8]
LOWER[np.where(ACTIVE == 9)[0][0]] = 0.0
LOWER[np.where(ACTIVE == 15)[0][0]] = 1.0 / PARAMETER_SCALES[15]


def input_matrix(features):
    raw = np.asarray(features, dtype=np.float64)
    result = np.column_stack((
        np.ones(len(raw)), raw[:, 1], raw[:, 2:4] / 1024.0,
        raw[:, 4:6] / (1024.0 ** 2), np.zeros(len(raw)), raw[:, 6],
        raw[:, 7:9] / 1024.0, raw[:, 9:11] / (1024.0 ** 2),
        raw[:, 11], raw[:, 12] / 1_000_000.0))
    return result


def unpack(active):
    parameters = np.zeros(17)
    parameters[ACTIVE] = active[:len(ACTIVE)]
    return parameters


def interaction_features(inputs, kind):
    running = inputs[:, 1]
    waiting = inputs[:, 7]
    if kind == "decode":
        decode_only = np.maximum(0.0, inputs[:, 12] - running - waiting)
        return np.column_stack((decode_only / 100.0,
                                decode_only * waiting / 1000.0,
                                decode_only * running / 100.0))
    if kind == "decode_wait":
        decode_only = np.maximum(0.0, inputs[:, 12] - running - waiting)
        return (decode_only * waiting / 1000.0)[:, None]
    elapsed_seconds = inputs[:, 13] * 1000.0
    return np.column_stack((waiting * waiting / 100.0,
                            running * waiting / 10.0,
                            elapsed_seconds * waiting))


def shape(value, curvature):
    root = np.sqrt(value * value + curvature)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = value[positive] + root[positive]
    output[~positive] = curvature / (root[~positive] - value[~positive])
    return output


def prediction_and_jacobian(inputs, active, interaction_kind="queue"):
    p = unpack(active)
    effective = p * PARAMETER_SCALES[:17]
    linear = (inputs[:, :7] @ effective[:7] +
              inputs[:, 7:12] @ effective[10:15] +
              inputs[:, 13] * effective[16]) / 320.0
    extras = (interaction_features(inputs, interaction_kind)
              if len(active) > len(ACTIVE) else None)
    if extras is not None:
        linear += (extras @ active[len(ACTIVE):]) / 320.0
    zero_linear = effective[0] / 320.0
    curvature = effective[8]
    root = np.sqrt(linear * linear + curvature)
    zero_root = np.sqrt(zero_linear * zero_linear + curvature)
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (effective[7] + 1500.0 * (shape(linear, curvature) -
                                     shape(zero_linear, curvature)) +
           effective[9] * total_prompt + effective[15] * inputs[:, 12])
    jac = np.zeros((len(inputs), 17))
    slope = 1500.0 * (1.0 + linear / root) / 320.0
    jac[:, :7] = slope[:, None] * inputs[:, :7]
    jac[:, 0] -= 1500.0 * (1.0 + zero_linear / zero_root) / 320.0
    jac[:, 7] = 1.0
    jac[:, 8] = 750.0 * (1.0 / root - 1.0 / zero_root)
    jac[:, 9] = total_prompt
    jac[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    jac[:, 15] = inputs[:, 12]
    jac[:, 16] = slope * inputs[:, 13]
    jac *= PARAMETER_SCALES[:17]
    if extras is not None:
        jac = np.column_stack((jac, slope[:, None] * extras))
    jac[raw < 0.0] = 0.0
    indices = list(ACTIVE) + list(range(17, jac.shape[1]))
    return np.maximum(raw, 0.0), jac[:, indices]


def load_host(path, fit_start_count=3):
    if fit_start_count < 2:
        raise ValueError("fit_start_count must be at least two")
    rows = []
    updates = []
    initial = None
    current = None
    checked = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["type"] == "update":
                if initial is None:
                    initial = event["parameters_before"]
                current = event["parameters_after"]
                updates.append(current)
            elif event["type"] == "sample":
                rows.append(event)
                if current is not None and len(rows) % 25 == 0:
                    checked.append((event, current))
    if not rows or not updates:
        raise ValueError(f"no fitting samples or parameters: {path}")
    starts = [initial]
    starts.extend(
        updates[round((len(updates) - 1) * index / (fit_start_count - 1))]
        for index in range(1, fit_start_count)
    )
    inputs = input_matrix([row["features"] for row in rows])
    target = np.asarray([row["target_ms"] for row in rows])
    event_time = np.asarray([row["event_time_us"] for row in rows], dtype=np.int64)
    logged_base = np.asarray([row["base_prediction_before_ms"] for row in rows])
    maximum_difference = None
    if checked:
        check_input = input_matrix([row["features"] for row, _ in checked])
        maximum_difference = max(
            abs(prediction_and_jacobian(check_input[index:index + 1],
                                        np.asarray(parameters)[ACTIVE])[0][0] -
                row["base_prediction_before_ms"])
            for index, (row, parameters) in enumerate(checked))
    logged_final = np.asarray([row["prediction_before_ms"] for row in rows])
    return inputs, target, event_time, logged_base, logged_final, starts, maximum_difference


def rising_bins(paths):
    waiting = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event["type"] == "sample":
                    bin_number = event["event_time_us"] // 30_000_000
                    waiting[bin_number].append(event["features"][6])
    means = {key: np.mean(value) for key, value in waiting.items()}
    smoothed = {key: np.mean([means[other] for other in range(key - 2, key + 1)
                              if other in means]) for key in means}
    return {key for key, current in smoothed.items()
            if key - 2 in smoothed and
            current - smoothed[key - 2] > max(4.0, 0.1 * smoothed[key - 2])}


def metrics(target, predicted):
    if not len(target):
        return None
    error = np.abs(predicted - target)
    log_error = np.log1p(predicted) - np.log1p(target)
    return {"n": int(len(target)), "mae_ms": float(np.mean(error)),
            "rmse_ms": float(np.sqrt(np.mean(error * error))),
            "rmsle": float(np.sqrt(np.mean(log_error * log_error))),
            "p95_abs_ms": float(np.percentile(error, 95)),
            "mean_relative_error": float(np.mean(error / np.maximum(target, 100.0))),
            "bias_ms": float(np.mean(target - predicted))}


def pressure_masks(inputs):
    low = (inputs[:, 1] <= 4) & (inputs[:, 7] <= 4)
    high = inputs[:, 7] >= 33
    return low, ~(low | high), high


def waiting_constraint_matrix(inputs, parameter_count):
    matrix = np.zeros((len(inputs), parameter_count))
    for parameter_index in range(10, 15):
        active_index = int(np.where(ACTIVE == parameter_index)[0][0])
        matrix[:, active_index] = (inputs[:, parameter_index - 3] *
                                   PARAMETER_SCALES[parameter_index])
    return matrix


def fit(inputs, target, starts, loss, interaction_kind,
        enforce_nonnegative_waiting=False):
    best = None
    for initial in starts:
        active = np.asarray(initial, dtype=np.float64)[ACTIVE].copy()
        lower = LOWER
        if interaction_kind != "none":
            count = interaction_features(inputs[:1], interaction_kind).shape[1]
            active = np.concatenate((active, np.zeros(count)))
            lower = np.concatenate((LOWER, np.full(count, -np.inf)))
        active = np.maximum(active, lower + 1e-9)
        cache = [None, None, None]

        def evaluate(values):
            if cache[0] is None or not np.array_equal(values, cache[0]):
                prediction, jac = prediction_and_jacobian(
                    inputs, values, interaction_kind)
                cache[:] = [values.copy(), prediction, jac]
            return cache[1], cache[2]

        def residual(values):
            predicted = evaluate(values)[0]
            if loss == "log1p":
                return np.log1p(predicted) - np.log1p(target)
            return predicted - target

        def jacobian(values):
            predicted, jac = evaluate(values)
            return jac / (1.0 + predicted[:, None]) if loss == "log1p" else jac

        result = least_squares(
            residual, active, jac=jacobian,
            bounds=(lower, np.full(len(active), np.inf)),
            x_scale="jac", max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if enforce_nonnegative_waiting:
            constraint_matrix = waiting_constraint_matrix(inputs, len(active))
            feasible = result.x.copy()
            if np.min(constraint_matrix @ feasible) < 0.0:
                for parameter_index in range(10, 15):
                    feasible[int(np.where(ACTIVE == parameter_index)[0][0])] = 0.0
            residual_scale = (max(float(np.sqrt(np.mean(target * target))), 1.0)
                              if loss == "raw" else 1.0)

            def objective(values):
                value = residual(values) / residual_scale
                return 0.5 * float(value @ value)

            def objective_jacobian(values):
                value = residual(values) / residual_scale
                return (jacobian(values).T @ value) / residual_scale

            result = minimize(
                objective, feasible, jac=objective_jacobian, method="SLSQP",
                bounds=Bounds(lower, np.full(len(active), np.inf)),
                constraints=LinearConstraint(constraint_matrix, 0.0, np.inf),
                options={"maxiter": 1200, "ftol": 1e-10, "disp": False})
            final_residual = residual(result.x)
            result.cost = 0.5 * float(final_residual @ final_residual)
            result.optimality = float(np.max(np.abs(objective_jacobian(result.x))))
            result.constraint_violation = max(
                0.0, -float(np.min(constraint_matrix @ result.x)))
        else:
            result.constraint_violation = max(
                0.0, -float(np.min(waiting_constraint_matrix(
                    inputs, len(result.x)) @ result.x)))
        if best is None or result.cost < best.cost:
            best = result
    return best


def evaluate_host(path, rising, loss, interaction_kind="none"):
    inputs, target, event_time, logged_base, logged_final, starts, check = load_host(path)
    phase = np.isin(event_time // 30_000_000, list(rising))
    # Keep complete time blocks out of fitting to test transfer across cycles.
    held = (event_time // 120_000_000) % 4 == 3
    train = ~held
    masks = {"all": train, "rising": train & phase,
             "other": train & ~phase}
    if any(np.sum(mask) < 40 for mask in masks.values()):
        raise ValueError(f"insufficient phase samples: {path}")
    segments = {"train_all": train,
                "train_rising": train & phase,
                "train_other": train & ~phase,
                "held_all": held,
                "held_rising": held & phase,
                "held_other": held & ~phase,
                "held_low": held & (target < 256),
                "held_running_0": held & (inputs[:, 1] == 0),
                "held_running_1": held & (inputs[:, 1] == 1),
                "held_running_2plus": held & (inputs[:, 1] >= 2),
                "held_waiting_0": held & (inputs[:, 7] == 0),
                "held_waiting_33plus": held & (inputs[:, 7] >= 33),
                "held_elapsed_positive": held & (inputs[:, 13] > 0)}
    models = {}
    for name, mask in masks.items():
        result = fit(inputs[mask], target[mask], starts, loss, interaction_kind)
        models[name] = {"parameters": unpack(result.x).tolist(),
                        "interaction_parameters": result.x[len(ACTIVE):].tolist(),
                        "cost": float(result.cost), "success": bool(result.success),
                        "optimality": float(result.optimality),
                        "nfev": int(result.nfev), "evaluations": {}}
        predicted = prediction_and_jacobian(inputs, result.x, interaction_kind)[0]
        for segment, selected in segments.items():
            models[name]["evaluations"][segment] = metrics(
                target[selected], predicted[selected])
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "loss": loss, "interaction_kind": interaction_kind,
            "sample_count": int(len(target)), "rising_count": int(np.sum(phase)),
            "held_count": int(np.sum(held)), "formula_check_max_ms": check,
            "online_base": metrics(target[held], logged_base[held]),
            "online_final": metrics(target[held], logged_final[held]),
            "online_segments": {
                segment: {"base": metrics(target[selected], logged_base[selected]),
                          "final": metrics(target[selected], logged_final[selected])}
                for segment, selected in segments.items()},
            "models": models}


def evaluate_pressure_host(path, loss, interaction_kind="none"):
    inputs, target, event_time, logged_base, logged_final, starts, check = load_host(path)
    held = (event_time // 120_000_000) % 4 == 3
    low, middle, high = pressure_masks(inputs)
    masks = {"all": ~held, "low": ~held & low, "high": ~held & high}
    if any(np.sum(mask) < 40 for mask in masks.values()):
        raise ValueError(f"insufficient pressure samples: {path}")
    segments = {"held_all": held, "held_low": held & low,
                "held_middle": held & middle, "held_high": held & high,
                "train_low": ~held & low, "train_high": ~held & high,
                "held_short": held & (target < 256)}
    models = {}
    for name, mask in masks.items():
        # Initial state predates the held-out time blocks; later online states do not.
        result = fit(inputs[mask], target[mask], starts[:1], loss, interaction_kind)
        prediction, jac = prediction_and_jacobian(
            inputs[mask], result.x, interaction_kind)
        if loss == "log1p":
            jac = jac / (1.0 + prediction[:, None])
        column_norm = np.linalg.norm(jac, axis=0)
        nonzero = column_norm > 1e-10
        normalized = jac[:, nonzero] / column_norm[nonzero]
        singular = np.linalg.svd(normalized, compute_uv=False)
        models[name] = {
            "parameters": unpack(result.x).tolist(),
            "interaction_parameters": result.x[len(ACTIVE):].tolist(),
            "success": bool(result.success), "nfev": int(result.nfev),
            "optimality": float(result.optimality),
            "active_columns": int(np.sum(nonzero)),
            "normalized_jacobian_rank": int(np.sum(singular > 1e-8)),
            "normalized_min_singular": float(singular[-1]) if len(singular) else 0.0,
            "evaluations": {segment: metrics(
                target[selected], prediction_and_jacobian(
                    inputs[selected], result.x, interaction_kind)[0])
                for segment, selected in segments.items()},
        }
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "loss": loss, "interaction_kind": interaction_kind,
            "sample_count": int(len(target)),
            "train_low_count": int(np.sum(masks["low"])),
            "train_high_count": int(np.sum(masks["high"])),
            "held_low_count": int(np.sum(held & low)),
            "held_high_count": int(np.sum(held & high)),
            "formula_check_max_ms": check,
            "online_base": {segment: metrics(target[selected], logged_base[selected])
                            for segment, selected in segments.items()},
            "online_final": {segment: metrics(target[selected], logged_final[selected])
                             for segment, selected in segments.items()},
            "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--loss", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--phase", choices=("queue_trend", "pressure"),
                        default="queue_trend")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--interactions", action="store_true")
    variants.add_argument("--decode-interactions", action="store_true")
    variants.add_argument("--decode-wait-only", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.jsonl"))
    selected = [path for path in paths if not args.hosts or path.stem in args.hosts]
    rising = rising_bins(paths) if args.phase == "queue_trend" else None
    interaction_kind = ("decode_wait" if args.decode_wait_only else
                        "decode" if args.decode_interactions else
                        "queue" if args.interactions else "none")
    for path in selected:
        result = (evaluate_pressure_host(path, args.loss, interaction_kind)
                  if args.phase == "pressure" else
                  evaluate_host(path, rising, args.loss, interaction_kind))
        print(json.dumps(result,
                         separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
