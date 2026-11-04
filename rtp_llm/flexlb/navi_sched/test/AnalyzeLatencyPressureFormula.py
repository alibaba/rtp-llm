#!/usr/bin/env python3
"""Ablate direct count and pressure-dependent curvature in the v21 formula."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import spearmanr

from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE, LOWER, PARAMETER_SCALES, load_host, metrics, pressure_masks,
)


CURVATURE_INDEX = int(np.flatnonzero(ACTIVE == 8)[0])
RUNNING_SCALE = 10.0
WAITING_SCALE = 100.0
CURVATURE_EPSILON = 1e-8
VARIANTS = ("base", "direct", "curvature", "both",
            "curvature_waiting", "direct_curvature_waiting")


def has_direct(variant):
    return variant in ("direct", "both", "direct_curvature_waiting")


def has_curvature(variant):
    return variant in ("curvature", "both", "curvature_waiting",
                       "direct_curvature_waiting")


def has_running_curvature(variant):
    return variant in ("curvature", "both")


def inverse_softplus(value):
    return value + np.log(-np.expm1(-value))


def pressure_shape(value, curvature):
    curvature = np.broadcast_to(curvature, value.shape)
    root = np.sqrt(value * value + curvature)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = value[positive] + root[positive]
    output[~positive] = (curvature[~positive] /
                         (root[~positive] - value[~positive]))
    return output


def initial_values(parameters, variant):
    values = np.asarray(parameters, dtype=np.float64)[ACTIVE].copy()
    if has_curvature(variant):
        curvature = values[CURVATURE_INDEX] * PARAMETER_SCALES[8]
        root = np.sqrt(max(curvature - CURVATURE_EPSILON, 1e-12))
        values[CURVATURE_INDEX] = inverse_softplus(root) / PARAMETER_SCALES[8]
    if has_direct(variant):
        values = np.concatenate((values, np.zeros(2)))
    if has_curvature(variant):
        values = np.concatenate((values, np.zeros(2 if has_running_curvature(variant) else 1)))
    return values


def forward_and_jacobian(inputs, values, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    p = np.zeros(17)
    p[ACTIVE] = values[:len(ACTIVE)] * PARAMETER_SCALES[ACTIVE]
    running = inputs[:, 1]
    waiting = inputs[:, 7]
    linear = (inputs[:, :7] @ p[:7] + inputs[:, 7:12] @ p[10:15] +
              inputs[:, 13] * p[16]) / 320.0
    zero_linear = p[0] / 320.0
    extra_index = len(ACTIVE)
    direct = np.zeros(len(inputs))
    if has_direct(variant):
        direct = values[extra_index] * running + values[extra_index + 1] * waiting
        extra_index += 2
    if has_curvature(variant):
        zero_soft = np.logaddexp(0.0, p[8])
        zero_curvature = zero_soft * zero_soft + CURVATURE_EPSILON
        curved_input = p[8]
        if has_running_curvature(variant):
            curved_input = curved_input + values[extra_index] * running / RUNNING_SCALE
            extra_index += 1
        curved_input = curved_input + values[extra_index] * waiting / WAITING_SCALE
        curved_soft = np.logaddexp(0.0, curved_input)
        curvature = curved_soft * curved_soft + CURVATURE_EPSILON
        curvature_slope = 2.0 * curved_soft * expit(curved_input)
        zero_curvature_slope = 2.0 * zero_soft * expit(p[8])
    else:
        curvature = p[8]
        zero_curvature = p[8]
    root = np.sqrt(linear * linear + curvature)
    zero_root = np.sqrt(zero_linear * zero_linear + zero_curvature)
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (p[7] + 1500.0 * (pressure_shape(linear, curvature) -
                              pressure_shape(np.full(len(inputs), zero_linear),
                                             zero_curvature)) +
           p[9] * total_prompt + p[15] * inputs[:, 12] + direct)
    jac = np.zeros((len(inputs), len(values)))
    slope = 1500.0 * (1.0 + linear / root) / 320.0
    jac[:, :7] = slope[:, None] * inputs[:, :7]
    jac[:, 0] -= 1500.0 * (1.0 + zero_linear / zero_root) / 320.0
    jac[:, CURVATURE_INDEX] = 750.0 * (1.0 / root - 1.0 / zero_root)
    jac[:, np.flatnonzero(ACTIVE == 7)[0]] = 1.0
    jac[:, np.flatnonzero(ACTIVE == 9)[0]] = total_prompt
    jac[:, np.flatnonzero(ACTIVE == 15)[0]] = inputs[:, 12]
    jac[:, np.flatnonzero(ACTIVE == 16)[0]] = slope * inputs[:, 13]
    for position, parameter in enumerate(range(10, 15)):
        jac[:, np.flatnonzero(ACTIVE == parameter)[0]] = slope * inputs[:, 7 + position]
    extra_index = len(ACTIVE)
    if has_direct(variant):
        jac[:, extra_index] = running
        jac[:, extra_index + 1] = waiting
        extra_index += 2
    if has_curvature(variant):
        jac[:, CURVATURE_INDEX] = 750.0 * (
            curvature_slope / root - zero_curvature_slope / zero_root)
        if has_running_curvature(variant):
            jac[:, extra_index] = 750.0 * curvature_slope * running / (RUNNING_SCALE * root)
            extra_index += 1
        jac[:, extra_index] = 750.0 * curvature_slope * waiting / (WAITING_SCALE * root)
    jac[:, :len(ACTIVE)] *= PARAMETER_SCALES[ACTIVE]
    jac[raw < 0.0] = 0.0
    return np.maximum(raw, 0.0), jac


def fit(inputs, target, initial, variant):
    lower = LOWER.copy()
    if has_curvature(variant):
        lower[CURVATURE_INDEX] = -np.inf
    lower = np.concatenate((lower, np.full(len(initial) - len(lower), -np.inf)))
    initial = np.maximum(initial, lower + 1e-9)
    cached = [None, None, None]

    def evaluate(values):
        if cached[0] is None or not np.array_equal(cached[0], values):
            prediction, jacobian = forward_and_jacobian(inputs, values, variant)
            cached[:] = [values.copy(), prediction, jacobian]
        return cached[1], cached[2]

    result = least_squares(
        lambda values: np.log1p(evaluate(values)[0]) - np.log1p(target),
        initial,
        jac=lambda values: evaluate(values)[1] / (1.0 + evaluate(values)[0][:, None]),
        bounds=(lower, np.full(len(initial), np.inf)), x_scale="jac",
        max_nfev=1200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return result


def correlation(first, second):
    if len(first) < 3 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def pressure_correlations(inputs, target, logged_base, logged_final, mask):
    running = inputs[mask, 1]
    waiting = inputs[mask, 7]
    preceding = inputs[mask, 12]
    base_error = np.log1p(target[mask]) - np.log1p(logged_base[mask])
    final_error = np.log1p(target[mask]) - np.log1p(logged_final[mask])
    columns = {"running": running, "waiting": waiting,
               "preceding": preceding, "base_error": base_error,
               "final_error": final_error}
    pairs = (("running", "waiting"), ("running", "preceding"),
             ("waiting", "preceding"), ("base_error", "running"),
             ("base_error", "waiting"), ("base_error", "preceding"),
             ("final_error", "running"), ("final_error", "waiting"))
    return {f"{left}:{right}": {
        "pearson": correlation(columns[left], columns[right]),
        "spearman": (float(spearmanr(columns[left], columns[right]).statistic)
                     if len(columns[left]) >= 3 and
                     np.std(columns[left]) > 0 and
                     np.std(columns[right]) > 0 else None),
    } for left, right in pairs}


def jacobian_diagnostics(inputs, target, values, variant):
    prediction, jacobian = forward_and_jacobian(inputs, values, variant)
    jacobian /= 1.0 + prediction[:, None]
    norms = np.linalg.norm(jacobian, axis=0)
    active = norms > 1e-10
    normalized = jacobian[:, active] / norms[active]
    singular = np.linalg.svd(normalized, compute_uv=False)
    positions = {"inside_running": int(np.flatnonzero(ACTIVE == 1)[0]),
                 "inside_waiting": int(np.flatnonzero(ACTIVE == 10)[0]),
                 "preceding": int(np.flatnonzero(ACTIVE == 15)[0])}
    index = len(ACTIVE)
    if has_direct(variant):
        positions.update(direct_running=index, direct_waiting=index + 1)
        index += 2
    if has_curvature(variant):
        if has_running_curvature(variant):
            positions["curvature_running"] = index
            index += 1
        positions["curvature_waiting"] = index
    pairs = (("inside_running", "direct_running"),
             ("inside_waiting", "direct_waiting"),
             ("inside_running", "curvature_running"),
             ("inside_waiting", "curvature_waiting"),
             ("direct_running", "curvature_running"),
             ("direct_waiting", "curvature_waiting"),
             ("preceding", "direct_waiting"))
    return {
        "active_columns": int(np.sum(active)),
        "rank": int(np.sum(singular > 1e-8)),
        "min_singular": float(singular[-1]) if len(singular) else 0.0,
        "gradient_correlations": {
            f"{left}:{right}": correlation(jacobian[:, positions[left]],
                                           jacobian[:, positions[right]])
            for left, right in pairs if left in positions and right in positions},
    }


def evaluate_host(path, variant):
    inputs, target, event_time, logged_base, logged_final, starts, check = load_host(path)
    held = (event_time // 120_000_000) % 4 == 3
    low, middle, high = pressure_masks(inputs)
    segments = {"held_all": held, "held_low": held & low,
                "held_middle": held & middle, "held_high": held & high,
                "held_short": held & (target < 256),
                "train_low": ~held & low, "train_high": ~held & high}
    models = {}
    for name, mask in (("all", ~held), ("low", ~held & low),
                       ("high", ~held & high)):
        result = fit(inputs[mask], target[mask],
                     initial_values(starts[0], variant), variant)
        models[name] = {
            "values": result.x.tolist(), "success": bool(result.success),
            "nfev": int(result.nfev), "cost": float(result.cost),
            "jacobian": jacobian_diagnostics(
                inputs[mask], target[mask], result.x, variant),
            "evaluations": {
                segment: metrics(target[selected], forward_and_jacobian(
                    inputs[selected], result.x, variant)[0])
                for segment, selected in segments.items()},
        }
    train = ~held
    prediction_log_error = (np.log1p(target[train]) -
                            np.log1p(logged_base[train]))
    correlations = {
        "running_waiting": correlation(inputs[train, 1], inputs[train, 7]),
        "running_preceding": correlation(inputs[train, 1], inputs[train, 12]),
        "waiting_preceding": correlation(inputs[train, 7], inputs[train, 12]),
        "error_running": correlation(prediction_log_error, inputs[train, 1]),
        "error_waiting": correlation(prediction_log_error, inputs[train, 7]),
        "error_preceding": correlation(prediction_log_error, inputs[train, 12]),
    }
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "variant": variant, "formula_check_max_ms": check,
            "correlations": correlations,
            "pressure_correlations": {
                name: pressure_correlations(inputs, target, logged_base,
                                            logged_final, mask)
                for name, mask in (("all", train), ("low", train & low),
                                   ("high", train & high))},
            "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--hosts", nargs="*")
    args = parser.parse_args()
    for path in sorted(args.input_dir.glob("*.jsonl")):
        if args.hosts and path.stem not in args.hosts:
            continue
        print(json.dumps(evaluate_host(path, args.variant),
                         separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
