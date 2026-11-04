#!/usr/bin/env python3
"""Evaluate causal decode-cohort features with whole replay cycles held out.

The lifecycle audit reconstructs per-request state at the admission snapshot.
P17/P18 select the async callback commit point, while target latency and future
release events are excluded from causal model inputs. Retrospective releases
are scored separately as an oracle diagnostic.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from AnalyzeLatencyFeatureNeighbors import pressure_phases, read_samples
from AnalyzeVllmDecodeSchedulerState import (
    CAUSAL_CANDIDATES,
    EXISTING_FEATURES,
    RETROSPECTIVE_FIELDS,
    join_samples,
    reconstruct_sample_scheduler_state,
)
from AnalyzeVllmRequestLifecycles import read_events


CYCLE_US = 300_000_000
CONTEXT_FEATURES = ("running_decode_prompt_tokens",)
TEMPLATE_FEATURES = tuple(
    f"decode_max_{limit}_count" for limit in (32, 64, 128, 256)
)
VARIANTS = {
    "existing": EXISTING_FEATURES,
    "context": (*EXISTING_FEATURES, *CONTEXT_FEATURES),
    "configured_output_mix": (*EXISTING_FEATURES, *TEMPLATE_FEATURES),
    "all_causal": (
        *EXISTING_FEATURES,
        *CONTEXT_FEATURES,
        *TEMPLATE_FEATURES,
    ),
    "oracle_release": (*EXISTING_FEATURES, *RETROSPECTIVE_FIELDS),
}


def _add_template_counts(rows):
    for row in rows:
        counts = {limit: 0 for limit in (32, 64, 128, 256)}
        for name, count in row["decode_template_counts"].items():
            maximum = int(name.rsplit("_max_", 1)[1])
            if maximum in counts:
                counts[maximum] += count
        for maximum, count in counts.items():
            row[f"decode_max_{maximum}_count"] = count


def _score(target, prediction, mask):
    selected_target = target[mask]
    selected_prediction = prediction[mask]
    error = selected_prediction - selected_target
    return {
        "n": int(np.sum(mask)),
        "absolute_error_sum_ms": float(np.sum(np.abs(error))),
        "relative_error_sum": float(
            np.sum(np.abs(error) / np.maximum(selected_target, 100.0))
        ),
        "signed_error_sum_ms": float(np.sum(error)),
    }


def _finalize(score):
    count = score["n"]
    return {
        **score,
        "mae_ms": score["absolute_error_sum_ms"] / count if count else None,
        "mean_relative_error": (
            score["relative_error_sum"] / count if count else None
        ),
        "signed_error_ms": (
            score["signed_error_sum_ms"] / count if count else None
        ),
    }


def _fit_cycle_predictions(inputs, target, baseline, cycle):
    residual = np.log1p(target) - np.log1p(baseline)
    prediction = np.full(len(target), np.nan)
    for held_cycle in np.unique(cycle):
        held = cycle == held_cycle
        train = ~held
        if np.sum(held) < 20 or np.sum(train) < 100:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=17,
        )
        model.fit(inputs[train], residual[train])
        prediction[held] = np.maximum(
            0.0,
            np.expm1(
                np.log1p(baseline[held]) + model.predict(inputs[held])
            ),
        )
    return prediction


def _shuffle_added_features(inputs, existing_count, cycle):
    shuffled = inputs.copy()
    random = np.random.default_rng(17)
    for value in np.unique(cycle):
        selected = np.flatnonzero(cycle == value)
        shuffled[selected, existing_count:] = shuffled[
            random.permutation(selected), existing_count:
        ]
    return shuffled


def evaluate_node(path, deployment):
    events, malformed = read_events([path])
    if not events:
        return None, {
            "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
            "deployment": deployment,
            "reason": "no lifecycle audit events",
        }
    all_raw_rows = read_samples(path)
    required_snapshot_fields = (
        "running_decode_request_count",
        "running_decode_generated_tokens",
    )
    raw_rows = [
        row
        for row in all_raw_rows
        if all(field in row for field in required_snapshot_fields)
    ]
    if len(raw_rows) < 120:
        return None, {
            "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
            "deployment": deployment,
            "reason": "too few samples with decode snapshot fields",
            "sample_count": len(all_raw_rows),
            "eligible_sample_count": len(raw_rows),
        }
    states, diagnostics = reconstruct_sample_scheduler_state(events, raw_rows)
    rows = join_samples(raw_rows, states)
    _add_template_counts(rows)
    if len(rows) < 120:
        return None, {
            "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
            "deployment": deployment,
            "reason": "too few fully reconstructed decode snapshots",
            "sample_count": len(all_raw_rows),
            "eligible_sample_count": len(raw_rows),
            "reconstructed_sample_count": len(rows),
        }

    target = np.asarray([row["target_ms"] for row in rows], dtype=float)
    baseline = np.asarray(
        [row["prediction_before_ms"] for row in rows], dtype=float
    )
    ready_us = np.asarray(
        [row["event_time_us"] - row["target_ms"] * 1000.0 for row in rows]
    )
    cycle = ((ready_us - np.min(ready_us)) // CYCLE_US).astype(int)
    phases = np.asarray(
        [record["phase"] for record in pressure_phases(rows, window_s=30)]
    )
    max_num_seqs = max(
        (int(event.get("max_num_seqs", 0)) for event in events), default=0
    )
    high = np.asarray(
        [
            row["running_decode_request_count"] >= 0.8 * max_num_seqs
            for row in rows
        ]
    )
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "high_decode": high,
        "high_rising": high & (phases == "rising"),
        "high_falling": high & (phases == "falling"),
        "high_flat": high & (phases == "flat"),
    }

    predictions = {"online": baseline}
    for name, features in VARIANTS.items():
        inputs = np.asarray(
            [[row[feature] for feature in features] for row in rows],
            dtype=float,
        )
        predictions[name] = _fit_cycle_predictions(
            inputs, target, baseline, cycle
        )
        if name == "all_causal":
            predictions["all_causal_shuffled"] = _fit_cycle_predictions(
                _shuffle_added_features(inputs, len(EXISTING_FEATURES), cycle),
                target,
                baseline,
                cycle,
            )

    scores = {}
    for name, prediction in predictions.items():
        available = np.isfinite(prediction)
        scores[name] = {
            segment: _score(target, prediction, available & mask)
            for segment, mask in masks.items()
        }
    return {
        "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
        "deployment": deployment,
        "sample_count": len(rows),
        "cycle_counts": {
            str(value): int(np.sum(cycle == value)) for value in np.unique(cycle)
        },
        "malformed_audit_lines": malformed,
        "reconstruction": diagnostics,
        "scores": scores,
    }, None


def _evaluate_job(job):
    return evaluate_node(*job)


def _aggregate(nodes):
    result = {}
    for node in nodes:
        for variant, segments in node["scores"].items():
            for segment, score in segments.items():
                aggregate = result.setdefault(variant, {}).setdefault(
                    segment,
                    {
                        "n": 0,
                        "absolute_error_sum_ms": 0.0,
                        "relative_error_sum": 0.0,
                        "signed_error_sum_ms": 0.0,
                    },
                )
                for key in aggregate:
                    aggregate[key] += score[key]
    return {
        variant: {
            segment: _finalize(score) for segment, score in segments.items()
        }
        for variant, segments in result.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads((args.log_dir / "manifest.json").read_text())
    deployment_by_ip = {
        node["ip"]: node.get("deployment", "unknown")
        for node in manifest.get("nodes", ())
    }
    jobs = [
        (path, deployment_by_ip.get(path.name[10:-7].replace("_", "."), "unknown"))
        for path in sorted(args.log_dir.glob("lifecycle_*.log.gz"))
    ]
    if args.workers == 1:
        analyzed = [_evaluate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            analyzed = list(executor.map(_evaluate_job, jobs))
    nodes = [node for node, _ in analyzed if node is not None]
    skipped = [entry for _, entry in analyzed if entry is not None]
    deployments = sorted({node["deployment"] for node in nodes})
    output = {
        "definitions": {
            "cycle_seconds": CYCLE_US // 1_000_000,
            "validation": "leave one complete replay cycle out per node",
            "prediction": "tree correction of logged online prediction in log1p space",
            "causal_features": list(CAUSAL_CANDIDATES) + list(TEMPLATE_FEATURES),
            "oracle_features": list(RETROSPECTIVE_FIELDS),
            "excluded_pseudo_remaining": (
                "max_tokens - generated_tokens is a configured budget difference, "
                "not actual remaining output or a release-time observation"
            ),
            "negative_control": (
                "all added causal columns shuffled within each cycle before fit"
            ),
        },
        "by_deployment": {
            deployment: _aggregate(
                [node for node in nodes if node["deployment"] == deployment]
            )
            for deployment in deployments
        },
        "all_nodes": _aggregate(nodes),
        "skipped_nodes": skipped,
        "nodes": nodes,
    }
    print(json.dumps(output, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
