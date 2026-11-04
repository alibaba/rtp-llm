#!/usr/bin/env python3
"""Compare production Adam with diagonal/full extended RLS on exact replays."""

import argparse
import json
import math
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


PARAMETER_COUNT = 19
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
], dtype=np.float64)
INITIAL_SCALED_PARAMETERS = np.asarray([
    -36.56241493030776,
    2.7998347700525095,
    0.5296454912518096,
    0.27178684866553887,
    0.003629661106608585,
    0.00015333761680912973,
    0.0,
    250.19523675146382,
    1.9329205277529811,
    0.0,
    0.30370055600496004,
    0.053103146391784761,
    0.0034376753181673777,
    1.4994382069729369e-05,
    1.4458939266851801e-05,
    205.87344937782723,
    0.0,
    0.0,
    0.0,
], dtype=np.float64)
INACTIVE_BASE = (6,)
INACTIVE_DECODE = (17, 18)
WAITING_PARAMETER_INDICES = np.arange(10, 15, dtype=np.int64)
TOKEN_SCALE = 1024.0
LINEAR_SCALE = 320.0
NONLINEAR_SCALE = 1500.0
MIN_CURVATURE = 1e-8
MIN_PRECEDING_COEFFICIENT_MS = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPSILON = 1e-20


def make_input(features):
    if len(features) < 13:
        raise ValueError("latency sample is missing capacity-split features")
    decode_count = features[13] if len(features) >= 15 else 0
    decode_tokens = features[14] if len(features) >= 15 else 0
    return np.asarray([
        1.0,
        features[1],
        features[2] / TOKEN_SCALE,
        features[3] / TOKEN_SCALE,
        features[4] / (TOKEN_SCALE * TOKEN_SCALE),
        features[5] / (TOKEN_SCALE * TOKEN_SCALE),
        features[0],
        features[6],
        features[7] / TOKEN_SCALE,
        features[8] / TOKEN_SCALE,
        features[9] / (TOKEN_SCALE * TOKEN_SCALE),
        features[10] / (TOKEN_SCALE * TOKEN_SCALE),
        features[11],
        features[12] / 1_000_000.0,
        decode_count,
        decode_tokens / TOKEN_SCALE,
    ], dtype=np.float64)


def _shape(value, curvature):
    root = math.sqrt(value * value + curvature)
    return value + root if value >= 0.0 else curvature / (root - value)


def prediction_and_jacobian(features, parameters, coordinate_scales):
    """Return latency and d(latency)/d(parameters) in the selected coordinates."""
    inputs = make_input(features)
    effective = parameters * coordinate_scales
    linear = (
        float(inputs[:7] @ effective[:7])
        + float(inputs[7:12] @ effective[10:15])
        + inputs[13] * effective[16]
        + inputs[14] * effective[17]
        + inputs[15] * effective[18]
    ) / LINEAR_SCALE
    zero_linear = effective[0] / LINEAR_SCALE
    curvature = effective[8]
    root = math.sqrt(linear * linear + curvature)
    zero_root = math.sqrt(zero_linear * zero_linear + curvature)
    total_prompt = inputs[2] + inputs[3] + inputs[8] + inputs[9]
    raw = (
        effective[7]
        + NONLINEAR_SCALE
        * (_shape(linear, curvature) - _shape(zero_linear, curvature))
        + effective[9] * total_prompt
        + effective[15] * inputs[12]
    )
    jacobian = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    if raw < 0.0:
        return 0.0, jacobian
    slope = NONLINEAR_SCALE * (1.0 + linear / root) / LINEAR_SCALE
    zero_slope = NONLINEAR_SCALE * (1.0 + zero_linear / zero_root) / LINEAR_SCALE
    jacobian[:7] = slope * inputs[:7]
    jacobian[0] -= zero_slope
    jacobian[7] = 1.0
    jacobian[8] = 0.5 * NONLINEAR_SCALE * (1.0 / root - 1.0 / zero_root)
    jacobian[9] = total_prompt
    jacobian[10:15] = slope * inputs[7:12]
    jacobian[15] = inputs[12]
    jacobian[16] = slope * inputs[13]
    jacobian[17] = slope * inputs[14]
    jacobian[18] = slope * inputs[15]
    return raw, jacobian * coordinate_scales


def predict(features, parameters, coordinate_scales):
    return prediction_and_jacobian(features, parameters, coordinate_scales)[0]


def has_non_negative_waiting_cost(features, parameters, coordinate_scales):
    inputs = make_input(features)
    effective = parameters * coordinate_scales
    terms = inputs[7:12] * effective[WAITING_PARAMETER_INDICES]
    cost = float(np.sum(terms))
    tolerance = 1e-12 * max(1.0, float(np.sum(np.abs(terms))))
    return math.isfinite(cost) and cost >= -tolerance


def project_parameters(parameters, coordinate_scales, previous=None):
    parameters = parameters.copy()
    parameters[6] = 0.0
    effective = parameters * coordinate_scales
    updated = effective.copy()
    effective[7] = max(0.0, effective[7])
    effective[8] = max(MIN_CURVATURE, effective[8])
    effective[9] = max(0.0, effective[9])
    effective[15] = max(MIN_PRECEDING_COEFFICIENT_MS, effective[15])
    if previous is not None:
        previous_effective = previous * coordinate_scales
        for index, minimum in ((7, 0.0), (8, MIN_CURVATURE)):
            if previous_effective[index] > minimum and updated[index] <= minimum:
                effective[index] = previous_effective[index]
    return effective / coordinate_scales


def active_indices(enable_decode):
    inactive = set(INACTIVE_BASE)
    if not enable_decode:
        inactive.update(INACTIVE_DECODE)
    return np.asarray(
        [index for index in range(PARAMETER_COUNT) if index not in inactive],
        dtype=np.int64,
    )


def batch_observations(batch, parameters, coordinate_scales,
                       current_sample_weight, active):
    weights = np.ones(len(batch), dtype=np.float64)
    weights[-1] = current_sample_weight
    weights /= np.sum(weights)
    residuals = []
    jacobians = []
    for sample in batch:
        prediction, jacobian = prediction_and_jacobian(
            sample["features"], parameters, coordinate_scales
        )
        residuals.append(
            math.log1p(prediction) - math.log1p(float(sample["target_ms"]))
        )
        jacobians.append(jacobian[active] / (1.0 + prediction))
    return weights, np.asarray(residuals), np.asarray(jacobians)


class AdamOptimizer:
    def __init__(self, parameter_count, learning_rate,
                 update_multipliers=None, epsilon_multipliers=None):
        self._moment1 = np.zeros(parameter_count, dtype=np.float64)
        self._moment2 = np.zeros(parameter_count, dtype=np.float64)
        self._step = 1
        self._learning_rate = learning_rate
        self.waiting_candidate_recovery_count = 0
        self.waiting_state_reset_count = 0
        if update_multipliers is None:
            update_multipliers = np.ones(parameter_count, dtype=np.float64)
        self._update_multipliers = np.asarray(
            update_multipliers, dtype=np.float64
        )
        if (self._update_multipliers.shape != (parameter_count,)
                or not np.all(np.isfinite(self._update_multipliers))
                or np.any(self._update_multipliers <= 0.0)):
            raise ValueError("invalid Adam update multipliers")
        if epsilon_multipliers is None:
            epsilon_multipliers = np.ones(parameter_count, dtype=np.float64)
        self._epsilon_multipliers = np.asarray(
            epsilon_multipliers, dtype=np.float64
        )
        if (self._epsilon_multipliers.shape != (parameter_count,)
                or not np.all(np.isfinite(self._epsilon_multipliers))
                or np.any(self._epsilon_multipliers <= 0.0)):
            raise ValueError("invalid Adam epsilon multipliers")

    def restore(self, moment1, moment2, step):
        moment1 = np.asarray(moment1, dtype=np.float64)
        moment2 = np.asarray(moment2, dtype=np.float64)
        if (moment1.shape != self._moment1.shape
                or moment2.shape != self._moment2.shape
                or not np.all(np.isfinite(moment1))
                or not np.all(np.isfinite(moment2))
                or np.any(moment2 < 0.0) or step < 1):
            raise ValueError("invalid Adam replay anchor")
        self._moment1 = moment1.copy()
        self._moment2 = moment2.copy()
        self._step = int(step)

    def update(self, parameters, batch, coordinate_scales,
               current_sample_weight, active):
        weights, residuals, jacobians = batch_observations(
            batch, parameters, coordinate_scales, current_sample_weight, active
        )
        gradient = np.sum(weights[:, None] * residuals[:, None] * jacobians,
                          axis=0)
        previous_moment1 = self._moment1.copy()
        previous_moment2 = self._moment2.copy()
        self._moment1 = (
            ADAM_BETA1 * self._moment1 + (1.0 - ADAM_BETA1) * gradient
        )
        self._moment2 = (
            ADAM_BETA2 * self._moment2
            + (1.0 - ADAM_BETA2) * gradient * gradient
        )
        correction = self._learning_rate
        correction *= math.sqrt(1.0 - ADAM_BETA2 ** self._step)
        correction /= 1.0 - ADAM_BETA1 ** self._step
        previous = parameters.copy()
        parameters = parameters.copy()
        parameters[active] -= (
            correction * self._moment1
            / np.sqrt(
                self._moment2
                + ADAM_EPSILON * self._epsilon_multipliers
            )
            * self._update_multipliers
        )
        updated_effective = parameters * coordinate_scales
        effective = updated_effective.copy()
        active_positions = {index: position for position, index in enumerate(active)}
        for index in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17):
            if index not in active_positions:
                continue
            minimum = (MIN_CURVATURE if index == 8 else
                       MIN_PRECEDING_COEFFICIENT_MS if index == 15 else 0.0)
            if effective[index] >= minimum:
                continue
            effective[index] = minimum
            position = active_positions[index]
            if self._moment1[position] > 0.0:
                self._moment1[position] = 0.0
        previous_effective = previous * coordinate_scales
        for index, minimum in ((7, 0.0), (8, MIN_CURVATURE)):
            if previous_effective[index] <= minimum or updated_effective[index] > minimum:
                continue
            effective[index] = previous_effective[index]
            position = active_positions[index]
            self._moment1[position] = previous_moment1[position]
            self._moment2[position] = previous_moment2[position]
        parameters = effective / coordinate_scales
        parameters[6] = 0.0
        active_positions = {index: position for position, index in enumerate(active)}
        initial = INITIAL_SCALED_PARAMETERS * PARAMETER_SCALES / coordinate_scales
        for sample in batch:
            if has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                continue
            self.waiting_candidate_recovery_count += 1
            current_safe = has_non_negative_waiting_cost(
                sample["features"], previous, coordinate_scales
            )
            replacement = previous if current_safe else initial
            parameters[WAITING_PARAMETER_INDICES] = replacement[
                WAITING_PARAMETER_INDICES
            ]
            for index in WAITING_PARAMETER_INDICES:
                position = active_positions[int(index)]
                self._moment1[position] = (
                    previous_moment1[position] if current_safe else 0.0
                )
                self._moment2[position] = (
                    previous_moment2[position] if current_safe else 0.0
                )
            if not current_safe:
                self.waiting_state_reset_count += 1
            if not has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                raise FloatingPointError("waiting safety recovery failed")
        self._step += 1
        return parameters


class RlsOptimizer:
    """Batch extended RLS using rank-one Sherman-Morrison updates."""

    def __init__(self, parameter_count, forgetting, ridge, step_size,
                 diagonal=False, initial_covariance=None,
                 reset_interval=None):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must be in (0, 1]")
        if ridge <= 0.0 or step_size <= 0.0:
            raise ValueError("ridge and step size must be positive")
        self._forgetting = forgetting
        self._step_size = step_size
        self._diagonal = diagonal
        self._reset_interval = reset_interval
        self._update_count = 0
        self.reset_count = 0
        if reset_interval is not None and reset_interval <= 0:
            raise ValueError("reset interval must be positive")
        if initial_covariance is None:
            initial_covariance = np.full(parameter_count, 1.0 / ridge)
        initial_covariance = np.asarray(initial_covariance, dtype=np.float64)
        if initial_covariance.shape != (parameter_count,) or np.any(
                initial_covariance <= 0.0):
            raise ValueError("invalid initial covariance")
        self._initial_covariance = (
            initial_covariance.copy()
            if diagonal
            else np.diag(initial_covariance)
        )
        self._covariance = self._initial_covariance.copy()
        self.maximum_asymmetry = 0.0
        self.minimum_denominator = math.inf
        self.waiting_candidate_recovery_count = 0
        self.waiting_state_reset_count = 0

    @property
    def covariance(self):
        return self._covariance.copy()

    def _observe(self, jacobian):
        if self._diagonal:
            projected = self._covariance * jacobian
            denominator = 1.0 + float(jacobian @ projected)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise FloatingPointError("non-positive RLS denominator")
            self._covariance -= projected * projected / denominator
        else:
            projected = self._covariance @ jacobian
            denominator = 1.0 + float(jacobian @ projected)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise FloatingPointError("non-positive RLS denominator")
            self._covariance -= np.outer(projected, projected) / denominator
            asymmetry = float(np.max(np.abs(
                self._covariance - self._covariance.T
            )))
            self.maximum_asymmetry = max(self.maximum_asymmetry, asymmetry)
            self._covariance = (
                self._covariance + self._covariance.T
            ) * 0.5
        self.minimum_denominator = min(self.minimum_denominator, denominator)

    def update(self, parameters, batch, coordinate_scales,
               current_sample_weight, active):
        previous_covariance = self._covariance.copy()
        weights, residuals, jacobians = batch_observations(
            batch, parameters, coordinate_scales, current_sample_weight, active
        )
        if (self._reset_interval is not None
                and self._update_count > 0
                and self._update_count % self._reset_interval == 0):
            self._covariance = self._initial_covariance.copy()
            self.reset_count += 1
        self._covariance /= self._forgetting
        for weight, jacobian in zip(weights, jacobians):
            self._observe(math.sqrt(weight) * jacobian)
        gradient = np.sum(weights[:, None] * residuals[:, None] * jacobians,
                          axis=0)
        direction = (
            self._covariance * gradient
            if self._diagonal
            else self._covariance @ gradient
        )
        previous = parameters.copy()
        parameters = parameters.copy()
        parameters[active] -= self._step_size * direction
        if not np.all(np.isfinite(parameters)):
            raise FloatingPointError("non-finite RLS parameter update")
        self._update_count += 1
        parameters = project_parameters(
            parameters, coordinate_scales, previous
        )
        initial = INITIAL_SCALED_PARAMETERS * PARAMETER_SCALES / coordinate_scales
        recovered = False
        for sample in batch:
            if has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                continue
            self.waiting_candidate_recovery_count += 1
            recovered = True
            current_safe = has_non_negative_waiting_cost(
                sample["features"], previous, coordinate_scales
            )
            replacement = previous if current_safe else initial
            parameters[WAITING_PARAMETER_INDICES] = replacement[
                WAITING_PARAMETER_INDICES
            ]
            if not current_safe:
                self.waiting_state_reset_count += 1
            if not has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                raise FloatingPointError("waiting safety recovery failed")
        if recovered:
            # The full covariance couples all coordinates. Rolling it back is
            # the conservative analogue of restoring Adam's waiting moments.
            self._covariance = previous_covariance
        return parameters


class LeakyPriorOptimizer:
    """Online Gauss-Newton that retains the ridge prior while forgetting."""

    def __init__(self, parameter_count, forgetting, ridge, step_size):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must be in (0, 1]")
        if ridge <= 0.0 or step_size <= 0.0:
            raise ValueError("ridge and step size must be positive")
        self._forgetting = forgetting
        self._ridge = ridge
        self._step_size = step_size
        self._information = np.eye(parameter_count, dtype=np.float64) * ridge
        self.minimum_information_eigenvalue = ridge
        self.maximum_information_condition = 1.0
        self.waiting_candidate_recovery_count = 0
        self.waiting_state_reset_count = 0

    @property
    def covariance(self):
        identity = np.eye(self._information.shape[0], dtype=np.float64)
        return np.linalg.solve(self._information, identity)

    def update(self, parameters, batch, coordinate_scales,
               current_sample_weight, active):
        previous_information = self._information.copy()
        weights, residuals, jacobians = batch_observations(
            batch, parameters, coordinate_scales, current_sample_weight, active
        )
        self._information *= self._forgetting
        diagonal_loading = (1.0 - self._forgetting) * self._ridge
        self._information.flat[::self._information.shape[0] + 1] += (
            diagonal_loading
        )
        for weight, jacobian in zip(weights, jacobians):
            self._information += weight * np.outer(jacobian, jacobian)
        self._information = (self._information + self._information.T) * 0.5
        eigenvalues = np.linalg.eigvalsh(self._information)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise FloatingPointError("non-positive information matrix")
        self.minimum_information_eigenvalue = min(
            self.minimum_information_eigenvalue, minimum
        )
        self.maximum_information_condition = max(
            self.maximum_information_condition, maximum / minimum
        )
        gradient = np.sum(weights[:, None] * residuals[:, None] * jacobians,
                          axis=0)
        direction = np.linalg.solve(self._information, gradient)
        previous = parameters.copy()
        parameters = parameters.copy()
        parameters[active] -= self._step_size * direction
        if not np.all(np.isfinite(parameters)):
            raise FloatingPointError("non-finite leaky-prior parameter update")
        parameters = project_parameters(
            parameters, coordinate_scales, previous
        )
        initial = INITIAL_SCALED_PARAMETERS * PARAMETER_SCALES / coordinate_scales
        recovered = False
        for sample in batch:
            if has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                continue
            self.waiting_candidate_recovery_count += 1
            recovered = True
            current_safe = has_non_negative_waiting_cost(
                sample["features"], previous, coordinate_scales
            )
            replacement = previous if current_safe else initial
            parameters[WAITING_PARAMETER_INDICES] = replacement[
                WAITING_PARAMETER_INDICES
            ]
            if not current_safe:
                self.waiting_state_reset_count += 1
            if not has_non_negative_waiting_cost(
                    sample["features"], parameters, coordinate_scales):
                raise FloatingPointError("waiting safety recovery failed")
        if recovered:
            self._information = previous_information
        return parameters


@lru_cache(maxsize=None)
def load_replay(path):
    events = []
    samples_by_history = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            events.append(event)
            if event["type"] == "sample" and "history_sequence" in event:
                samples_by_history[int(event["history_sequence"])] = event
    return events, samples_by_history


def restore_logged_anchor(event, coordinate_scales, active, optimizer):
    """Restore the state after production cold-start candidate selection."""
    logged = np.asarray(event["parameters_after"], dtype=np.float64)
    if logged.shape != (PARAMETER_COUNT,):
        raise ValueError("replay does not contain 19 parameters")
    parameters = logged * PARAMETER_SCALES / coordinate_scales
    if isinstance(optimizer, AdamOptimizer):
        ratio = coordinate_scales[active] / PARAMETER_SCALES[active]
        logged_moment1 = np.asarray(event["adam_m1_after"], dtype=np.float64)
        logged_moment2 = np.asarray(event["adam_m2_after"], dtype=np.float64)
        optimizer.restore(
            logged_moment1[active] * ratio,
            logged_moment2[active] * ratio * ratio,
            int(event["step_after"]),
        )
    return project_parameters(parameters, coordinate_scales)


def run_host(path, optimizer_factory, coordinate_scales, enable_decode=False,
             maximum_updates=None, initial_state="anchor"):
    events, samples_by_history = load_replay(path)
    active = active_indices(enable_decode)
    optimizer = optimizer_factory(len(active))
    parameters = None
    if initial_state == "default":
        effective = INITIAL_SCALED_PARAMETERS * PARAMETER_SCALES
        parameters = project_parameters(
            effective / coordinate_scales, coordinate_scales
        )
    anchored = initial_state == "default"
    records = []
    zero_points = []
    update_times_ns = []
    update_count = 0
    initial_parameters = parameters.copy() if parameters is not None else None
    for event in events:
        if maximum_updates is not None and update_count >= maximum_updates:
            break
        if event["type"] == "sample":
            if anchored:
                event_time_us = int(event["event_time_us"])
                actual = float(event["target_ms"])
                records.append({
                    "event_time_us": event_time_us,
                    "ready_time_us": event_time_us - round(actual * 1000.0),
                    "actual": actual,
                    "prediction": predict(
                        event["features"], parameters, coordinate_scales
                    ),
                    "live_base": float(event["base_prediction_before_ms"]),
                    "live_final": float(event["prediction_before_ms"]),
                    "running": float(event["features"][1]),
                    "waiting": float(event["features"][6]),
                    "optimizer_updates": update_count,
                })
            continue
        if event["type"] != "update":
            continue
        if not anchored:
            parameters = restore_logged_anchor(
                event, coordinate_scales, active, optimizer
            )
            initial_parameters = parameters.copy()
            anchored = True
            continue
        current_weight = float(event.get("current_sample_weight", 4.0))
        for sequence_ids in event["fit_batches"]:
            try:
                batch = [samples_by_history[int(sequence)]
                         for sequence in sequence_ids]
            except KeyError as error:
                raise ValueError(
                    f"{path}: unknown history sequence {error.args[0]}"
                ) from error
            begin = time.perf_counter()
            parameters = optimizer.update(
                parameters, batch, coordinate_scales, current_weight, active
            )
            update_times_ns.append((time.perf_counter() - begin) * 1e9)
            update_count += 1
            if maximum_updates is not None and update_count >= maximum_updates:
                break
        zero_points.append(parameters[7] * coordinate_scales[7])
    if not records or not update_count or initial_parameters is None:
        raise ValueError(f"{path}: no replay records or optimizer updates")
    return {
        "host": path.stem,
        "records": records,
        "zero_points": zero_points,
        "update_times_ns": update_times_ns,
        "update_count": update_count,
        "initial_effective_parameters": (
            initial_parameters * coordinate_scales
        ).tolist(),
        "final_effective_parameters": (parameters * coordinate_scales).tolist(),
        "optimizer": optimizer,
    }


def load_architectures(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for node in manifest.get("nodes", []):
        deployment = node["deployment"].upper()
        if "MI308" in deployment:
            architecture = "MI308"
        elif "RTX_PRO_5000" in deployment:
            architecture = "RTX5000"
        else:
            continue
        host = node["ip"].replace(".", "_")
        result[host] = architecture
        result[f"{node['deployment']}_{host}"] = architecture
    return result


def percentile(values, quantile):
    return float(np.quantile(values, quantile)) if len(values) else None


def metric(records, field="prediction"):
    if not records:
        return {"n": 0}
    actual = np.asarray([record["actual"] for record in records])
    predicted = np.asarray([record[field] for record in records])
    error = predicted - actual
    absolute = np.abs(error)
    denominator = np.maximum(actual, 100.0)
    return {
        "n": len(records),
        "mae_ms": float(np.mean(absolute)),
        "wmape_pct": float(np.sum(absolute) / np.sum(actual) * 100.0),
        "mean_relative_error_pct": float(np.mean(absolute / denominator) * 100.0),
        "bias_ms": float(np.mean(error)),
        "p95_abs_ms": percentile(absolute, 0.95),
        "underprediction_mae_ms": float(np.mean(np.maximum(-error, 0.0))),
        "overprediction_mae_ms": float(np.mean(np.maximum(error, 0.0))),
    }


def prediction_difference(records, left, right):
    difference = np.asarray([
        abs(record[left] - record[right]) for record in records
    ], dtype=np.float64)
    return {
        "n": len(difference),
        "mae_ms": float(np.mean(difference)) if len(difference) else None,
        "maximum_ms": float(np.max(difference)) if len(difference) else None,
    }


def compare_variant_results(left_results, right_results):
    left_by_host = {result["host"]: result for result in left_results}
    right_by_host = {result["host"]: result for result in right_results}
    if set(left_by_host) != set(right_by_host):
        raise ValueError("variant host sets differ")
    differences = []
    initial_differences = []
    final_differences = []
    for host in sorted(left_by_host):
        left = left_by_host[host]
        right = right_by_host[host]
        if len(left["records"]) != len(right["records"]):
            raise ValueError(f"variant record counts differ for {host}")
        for left_record, right_record in zip(
                left["records"], right["records"]):
            if left_record["event_time_us"] != right_record["event_time_us"]:
                raise ValueError(f"variant event order differs for {host}")
            differences.append(abs(
                left_record["prediction"] - right_record["prediction"]
            ))
        initial_differences.extend(np.abs(
            np.asarray(left["initial_effective_parameters"])
            - np.asarray(right["initial_effective_parameters"])
        ))
        final_differences.extend(np.abs(
            np.asarray(left["final_effective_parameters"])
            - np.asarray(right["final_effective_parameters"])
        ))
    return {
        "prediction_difference_count": len(differences),
        "prediction_difference_mean_ms": float(np.mean(differences)),
        "prediction_difference_p95_ms": percentile(differences, 0.95),
        "prediction_difference_maximum_ms": float(np.max(differences)),
        "initial_effective_parameter_maximum_difference": float(
            np.max(initial_differences)
        ),
        "final_effective_parameter_maximum_difference": float(
            np.max(final_differences)
        ),
    }


def progress_metrics(records):
    count = len(records)
    boundaries = (
        ("first_5pct", 0.0, 0.05),
        ("first_10pct", 0.0, 0.10),
        ("first_20pct", 0.0, 0.20),
        ("middle_20pct", 0.40, 0.60),
        ("last_20pct", 0.80, 1.00),
    )
    return {
        name: metric(records[int(count * begin):max(
            int(count * begin) + 1, int(count * end)
        )])
        for name, begin, end in boundaries
    }


def covariance_diagnostics(optimizer):
    if not isinstance(optimizer, (RlsOptimizer, LeakyPriorOptimizer)):
        return {}
    covariance = optimizer.covariance
    if covariance.ndim == 1:
        minimum = float(np.min(covariance))
        maximum = float(np.max(covariance))
        bytes_used = covariance.nbytes
    else:
        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        bytes_used = covariance.nbytes
    return {
        "rls_covariance_minimum_eigenvalue": minimum,
        "rls_covariance_maximum_eigenvalue": maximum,
        "rls_covariance_condition": maximum / minimum if minimum > 0.0 else None,
        "rls_covariance_bytes": bytes_used,
        "rls_reset_count": getattr(optimizer, "reset_count", 0),
        "information_minimum_eigenvalue": getattr(
            optimizer, "minimum_information_eigenvalue", None
        ),
        "information_maximum_condition": getattr(
            optimizer, "maximum_information_condition", None
        ),
    }


def classify_phases(host_results, architectures):
    bins = defaultdict(list)
    for result in host_results:
        architecture = architectures[result["host"]]
        for record in result["records"]:
            key = (architecture, record["ready_time_us"] // 30_000_000)
            bins[key].append(record["running"] + record["waiting"])
    means = {key: float(np.mean(values)) for key, values in bins.items()}
    for result in host_results:
        architecture = architectures[result["host"]]
        for record in result["records"]:
            current = record["ready_time_us"] // 30_000_000
            now = means.get((architecture, current), 0.0)
            previous = means.get((architecture, current - 1), now)
            record["direction"] = "rising" if now >= previous else "falling"
            record["pressure"] = (
                "low" if record["running"] <= 4 and record["waiting"] <= 4
                else "high" if record["waiting"] >= 33
                else "middle"
            )


def scored_records(records, warmup_fraction, start_time_us=None,
                   end_time_us=None, time_basis="ready"):
    begin = int(len(records) * warmup_fraction)
    selected = records[begin:]
    time_field = time_basis + "_time_us"
    if start_time_us is not None:
        selected = [
            record for record in selected
            if record[time_field] >= start_time_us
        ]
    if end_time_us is not None:
        selected = [
            record for record in selected
            if record[time_field] <= end_time_us
        ]
    return selected


def summarize(name, host_results, architectures, warmup_fraction,
              start_time_us=None, end_time_us=None, time_basis="ready"):
    classify_phases(host_results, architectures)
    selected = []
    by_architecture = {}
    host_summaries = {}
    for result in host_results:
        records = result["records"]
        scored = scored_records(
            records, warmup_fraction, start_time_us, end_time_us, time_basis
        )
        selected.extend(scored)
        zero = np.asarray(result["zero_points"], dtype=np.float64)
        times = np.asarray(result["update_times_ns"], dtype=np.float64) / 1000.0
        optimizer = result["optimizer"]
        host_summaries[result["host"]] = {
            "metrics": metric(scored),
            "progress": progress_metrics(records),
            "zero_point_mean_ms": float(np.mean(zero)),
            "zero_point_p95_ms": percentile(zero, 0.95),
            "zero_point_max_drift_ms": float(np.max(np.abs(zero - zero[0]))),
            "update_time_p50_us": percentile(times, 0.50),
            "update_time_p95_us": percentile(times, 0.95),
            "update_time_p99_us": percentile(times, 0.99),
            "update_count": result["update_count"],
            "rls_minimum_denominator": getattr(
                optimizer, "minimum_denominator", None
            ),
            "rls_maximum_asymmetry": getattr(
                optimizer, "maximum_asymmetry", None
            ),
            "waiting_candidate_recovery_count": getattr(
                optimizer, "waiting_candidate_recovery_count", 0
            ),
            "waiting_state_reset_count": getattr(
                optimizer, "waiting_state_reset_count", 0
            ),
            "final_effective_parameters": result["final_effective_parameters"],
            **covariance_diagnostics(optimizer),
        }
    for architecture in sorted(set(architectures.values())):
        records = [record for result in host_results
                   if architectures[result["host"]] == architecture
                   for record in scored_records(
                       result["records"], warmup_fraction,
                       start_time_us, end_time_us, time_basis
                   )]
        by_architecture[architecture] = {
            "all": metric(records),
            "progress": progress_metrics(records),
            "rising": metric([r for r in records if r["direction"] == "rising"]),
            "falling": metric([r for r in records if r["direction"] == "falling"]),
            "low": metric([r for r in records if r["pressure"] == "low"]),
            "middle": metric([r for r in records if r["pressure"] == "middle"]),
            "high": metric([r for r in records if r["pressure"] == "high"]),
        }
    all_times = np.asarray([
        value for result in host_results for value in result["update_times_ns"]
    ], dtype=np.float64) / 1000.0
    return {
        "name": name,
        "metrics": metric(selected),
        "live_base": metric(selected, "live_base"),
        "live_final": metric(selected, "live_final"),
        "candidate_live_base_difference": prediction_difference(
            selected, "prediction", "live_base"
        ),
        "by_architecture": by_architecture,
        "update_time_p50_us": percentile(all_times, 0.50),
        "update_time_p95_us": percentile(all_times, 0.95),
        "update_time_p99_us": percentile(all_times, 0.99),
        "hosts": host_summaries,
    }


def parse_float_list(value):
    try:
        result = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return result


def parse_int_list(value):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or not all(item > 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def variant_factories(args):
    variants = []
    active = active_indices(args.enable_decode)
    if not args.rls_only:
        for scaled in (False, True):
            learning_rate = (
                args.adam_scaled_learning_rate
                if scaled else args.adam_raw_learning_rate
            )
            variants.append((
                "adam_scaled" if scaled else "adam_raw",
                PARAMETER_SCALES if scaled else np.ones(PARAMETER_COUNT),
                lambda count, rate=learning_rate: AdamOptimizer(count, rate),
            ))
        if args.include_neutral_control:
            variants.append((
                "adam_scaled_neutralized",
                PARAMETER_SCALES,
                lambda count, rate=args.adam_raw_learning_rate,
                multipliers=1.0 / PARAMETER_SCALES[active]: AdamOptimizer(
                    count,
                    rate,
                    update_multipliers=multipliers,
                    epsilon_multipliers=PARAMETER_SCALES[active] ** 2,
                ),
            ))
    if args.adam_only:
        return variants
    for forgetting in args.forgetting:
        for ridge in args.ridge:
            for step_size in args.step_size:
                for diagonal in ((True, False) if args.include_diagonal else (False,)):
                    for scaled in (False, True):
                        label = "diagonal_rls" if diagonal else "full_rls"
                        label += "_scaled" if scaled else "_raw"
                        label += f"_f{forgetting:g}_r{ridge:g}_s{step_size:g}"
                        scales = PARAMETER_SCALES if scaled else np.ones(PARAMETER_COUNT)
                        variants.append((
                            label,
                            scales,
                            lambda count, f=forgetting, r=ridge, s=step_size,
                            diagonal=diagonal: RlsOptimizer(
                                count, f, r, s, diagonal=diagonal
                            ),
                        ))
                    if not diagonal:
                        variants.append((
                            "full_rls_scaled_matched"
                            f"_f{forgetting:g}_r{ridge:g}_s{step_size:g}",
                            PARAMETER_SCALES,
                            lambda count, f=forgetting, r=ridge, s=step_size,
                            covariance=1.0 / (
                                ridge * PARAMETER_SCALES[active]
                                * PARAMETER_SCALES[active]
                            ): RlsOptimizer(
                                count, f, r, s,
                                initial_covariance=covariance,
                            ),
                        ))
                        for reset_interval in args.reset_interval:
                            variants.append((
                                "full_rls_reset"
                                f"{reset_interval}_raw_f{forgetting:g}"
                                f"_r{ridge:g}_s{step_size:g}",
                                np.ones(PARAMETER_COUNT),
                                lambda count, f=forgetting, r=ridge,
                                s=step_size, interval=reset_interval: RlsOptimizer(
                                    count, f, r, s, reset_interval=interval
                                ),
                            ))
                        if args.include_leaky_prior:
                            variants.append((
                                "leaky_prior_raw"
                                f"_f{forgetting:g}_r{ridge:g}_s{step_size:g}",
                                np.ones(PARAMETER_COUNT),
                                lambda count, f=forgetting, r=ridge,
                                s=step_size: LeakyPriorOptimizer(count, f, r, s),
                            ))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--forgetting", type=parse_float_list,
                        default=[0.999, 0.995, 0.99])
    parser.add_argument("--ridge", type=parse_float_list,
                        default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--step-size", type=parse_float_list,
                        default=[0.25, 0.5, 1.0])
    parser.add_argument("--adam-raw-learning-rate", type=float, default=0.176)
    parser.add_argument("--adam-scaled-learning-rate", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--enable-decode", action="store_true")
    parser.add_argument("--include-diagonal", action="store_true")
    parser.add_argument("--include-leaky-prior", action="store_true")
    parser.add_argument("--reset-interval", type=parse_int_list, default=[])
    parser.add_argument("--rls-only", action="store_true")
    parser.add_argument("--adam-only", action="store_true")
    parser.add_argument("--include-neutral-control", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument("--score-start-time-us", type=int)
    parser.add_argument("--score-end-time-us", type=int)
    parser.add_argument("--score-time-basis", choices=("ready", "event"),
                        default="ready")
    parser.add_argument("--initial-state", choices=("default", "anchor"),
                        default="anchor")
    args = parser.parse_args()
    if args.rls_only and args.adam_only:
        parser.error("--rls-only and --adam-only are mutually exclusive")
    if not 0.0 <= args.warmup_fraction < 1.0:
        parser.error("warmup fraction must be in [0, 1)")
    if args.maximum_updates is not None and args.maximum_updates < 1:
        parser.error("maximum updates must be positive")
    if (args.score_start_time_us is not None
            and args.score_end_time_us is not None
            and args.score_start_time_us > args.score_end_time_us):
        parser.error("score start time must not exceed score end time")
    architectures = load_architectures(args.manifest)
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.hosts:
        selected = set(host.replace(".", "_") for host in args.hosts)
        paths = [path for path in paths if path.stem in selected]
    paths = [path for path in paths if path.stem in architectures]
    if not paths:
        parser.error("no replay hosts match the manifest")
    summaries = []
    results_by_variant = {}
    for name, scales, factory in variant_factories(args):
        host_results = [
            run_host(path, factory, scales, args.enable_decode,
                     args.maximum_updates, args.initial_state) for path in paths
        ]
        summary = summarize(name, host_results, architectures,
                            args.warmup_fraction,
                            args.score_start_time_us,
                            args.score_end_time_us,
                            args.score_time_basis)
        summaries.append(summary)
        results_by_variant[name] = host_results
        if not args.quiet:
            print(json.dumps({
                "variant": name,
                "metrics": summary["metrics"],
                "by_architecture": summary["by_architecture"],
                "update_time_p50_us": summary["update_time_p50_us"],
                "update_time_p95_us": summary["update_time_p95_us"],
                "update_time_p99_us": summary["update_time_p99_us"],
            }, separators=(",", ":")), flush=True)
    comparisons = {}
    if "adam_raw" in results_by_variant and "adam_scaled" in results_by_variant:
        comparisons["adam_raw_vs_scaled"] = compare_variant_results(
            results_by_variant["adam_raw"], results_by_variant["adam_scaled"]
        )
    if ("adam_raw" in results_by_variant
            and "adam_scaled_neutralized" in results_by_variant):
        comparisons["adam_raw_vs_scaled_neutralized"] = (
            compare_variant_results(
                results_by_variant["adam_raw"],
                results_by_variant["adam_scaled_neutralized"],
            )
        )
    report = {
        "input_dir": str(args.input_dir),
        "manifest": str(args.manifest),
        "host_count": len(paths),
        "enable_decode": args.enable_decode,
        "initial_state": args.initial_state,
        "warmup_fraction": args.warmup_fraction,
        "score_start_time_us": args.score_start_time_us,
        "score_end_time_us": args.score_end_time_us,
        "score_time_basis": args.score_time_basis,
        "variants": summaries,
        "cross_variant_comparisons": comparisons,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
