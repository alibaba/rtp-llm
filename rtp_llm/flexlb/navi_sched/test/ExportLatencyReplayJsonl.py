#!/usr/bin/env python3
"""Export exact NaviSched latency replay events from an agent log."""

import argparse
import gzip
import json
from pathlib import Path
import re


NUMBER_FIELDS_RE = re.compile(
    r"\b([a-z0-9_]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
ARRAY_RE = {
    name: re.compile(r"\b" + name + r"=\[([^]]*)\]")
    for name in (
        "parameters_before",
        "parameters_after",
        "adam_m1_before",
        "adam_m1_after",
        "adam_m2_before",
        "adam_m2_after",
        "residual_parameters_before",
        "residual_parameters_after",
    )
}
UPDATE_SEQUENCE_RE = re.compile(
    r"\[navi-latency-training-update\].*?\bsequence=(\d+)"
)
UPDATE_STEP_RE = re.compile(
    r"\boptimizer_step_before=(\d+) optimizer_step_after=(\d+)"
)
UPDATE_REVISION_RE = re.compile(
    r"\brevision_before=(\d+) revision_after=(\d+)"
)
UPDATE_TIME_RE = re.compile(
    r"\bresidual_last_update_time_us_before=(\d+) "
    r"residual_last_update_time_us_after=(\d+)"
)
HISTORY_APPEND_RE = re.compile(
    r"\[latency-history-append\] sequence=(\d+) bucket=-?\d+ "
    r"local_batch_id=(-?\d+) event_time_us=(\d+)"
)
HISTORY_FIT_RE = re.compile(
    r"\[latency-history-fit\] mode=online_batch .*?"
    r"current_sequence=(\d+).*?sequence_ids=\[([^]]*)\]"
)
CONFIG_GENERATION_RE = re.compile(r"\bgeneration \[(\d+)\]")
CONFIG_HASH_RE = re.compile(r"\bhash \[([^]]+)\]")


def _number(value):
    return float(value) if any(char in value for char in ".eE") else int(value)


def _numbers(text):
    return [] if not text else [float(value) for value in text.split(",")]


def _sequence_ids(text):
    return [] if not text else [int(value) for value in text.split(",")]


def parse_effective_config(line):
    marker = "effectiveConfig ["
    offset = line.find(marker)
    if offset < 0:
        raise ValueError("scheduler config log is missing effectiveConfig")
    config, _ = json.JSONDecoder().raw_decode(line[offset + len(marker):])
    generation = CONFIG_GENERATION_RE.search(line)
    config_hash = CONFIG_HASH_RE.search(line)
    config["generation"] = int(generation.group(1)) if generation else 0
    config["config_hash"] = config_hash.group(1) if config_hash else ""
    return config


def fit_runtime_config(config):
    adaptive = config.get("latency_adaptive_learning_rate", {})
    if adaptive.get("enabled", False):
        raise ValueError(
            "config-only export cannot reconstruct adaptive per-update rates"
        )
    return {
        "base_learning_rate": float(config["latency_learning_rate"]),
        "base_learning_rate_scale": float(
            config["latency_online_learning_rate_scale"]
        ),
        "current_sample_weight": float(config["latency_current_sample_weight"]),
        "parameter_learning_rate_scales": list(
            config["latency_parameter_learning_rate_scales"]
        ),
        "nonlinear_scale": float(config["latency_nonlinear_scale"]),
        "zero_anchor_max_ms": float(config["latency_zero_anchor_max_ms"]),
        "residual_learning_rate": float(config["residual_learning_rate"]),
        "residual_update_error_floor_ms": float(
            config["residual_update_error_floor_ms"]
        ),
        "residual_update_error_ratio": float(
            config["residual_update_error_ratio"]
        ),
        "residual_idle_decay_start_us": int(
            config["residual_idle_decay_start_ms"]
        ) * 1000,
        "residual_idle_decay_end_us": int(
            config["residual_idle_decay_end_ms"]
        ) * 1000,
        "config_generation": int(config["generation"]),
        "config_hash": config["config_hash"],
    }


def _parse_update(line):
    sequence = UPDATE_SEQUENCE_RE.search(line)
    revisions = UPDATE_REVISION_RE.search(line)
    steps = UPDATE_STEP_RE.search(line)
    times = UPDATE_TIME_RE.search(line)
    arrays = {name: pattern.search(line) for name, pattern in ARRAY_RE.items()}
    if (
        not sequence
        or not revisions
        or not steps
        or not times
        or not all(arrays.values())
    ):
        raise ValueError("incomplete latency training update")
    revision_before = int(revisions.group(1))
    revision_after = int(revisions.group(2))
    return int(sequence.group(1)), {
        "revision_before": revision_before,
        "revision_after": revision_after,
        # A revision jump means the update started from an imported cold-start
        # candidate, not from parameters_before in this audit line.
        "exact_replayable": revision_after == revision_before + 1,
        "step_before": int(steps.group(1)),
        "step_after": int(steps.group(2)),
        "residual_time_before": int(times.group(1)),
        "residual_time_after": int(times.group(2)),
        **{name: _numbers(match.group(1)) for name, match in arrays.items()},
    }


def load_log(path):
    samples = []
    samples_by_batch_id = {}
    updates = {}
    runtime_config_by_update = {}
    history_to_training = {}
    fit_by_training = {}
    current_config = None
    with gzip.open(path, "rt", errors="replace") as stream:
        for line in stream:
            if "[navi-scheduler-config-update]" in line:
                current_config = parse_effective_config(line)
                continue
            if "[navi-latency-training-sample]" in line:
                sample = {
                    key: _number(value) for key, value in NUMBER_FIELDS_RE.findall(line)
                }
                required = (
                    "sequence",
                    "batch_id",
                    "event_time_us",
                    "target_ms",
                    "base_prediction_before_ms",
                    "prediction_before_ms",
                    "batch_count",
                    "running_batch_size",
                    "running_hit_tokens",
                    "running_miss_tokens",
                    "running_miss_square_sum",
                    "running_hit_miss_sum",
                    "waiting_batch_size",
                    "waiting_hit_tokens",
                    "waiting_miss_tokens",
                    "waiting_miss_square_sum",
                    "waiting_hit_miss_sum",
                    "preceding_request_count",
                )
                if not all(name in sample for name in required):
                    raise ValueError("incomplete latency training sample")
                samples.append(sample)
                samples_by_batch_id[int(sample["batch_id"])] = sample
                continue
            append = HISTORY_APPEND_RE.search(line)
            if append:
                history_sequence = int(append.group(1))
                local_batch_id = int(append.group(2))
                event_time_us = int(append.group(3))
                sample = samples_by_batch_id.get(local_batch_id)
                if sample is None or int(sample["event_time_us"]) != event_time_us:
                    raise ValueError(
                        "history append cannot be mapped to training sample: "
                        f"history_sequence={history_sequence} "
                        f"local_batch_id={local_batch_id}"
                    )
                history_to_training[history_sequence] = int(sample["sequence"])
                continue
            fit = HISTORY_FIT_RE.search(line)
            if fit:
                current_history_sequence = int(fit.group(1))
                current_training_sequence = history_to_training.get(
                    current_history_sequence
                )
                if current_training_sequence is None:
                    raise ValueError(
                        "history fit references unknown current sequence: "
                        f"{current_history_sequence}"
                    )
                fit_by_training[current_training_sequence] = _sequence_ids(
                    fit.group(2)
                )
                continue
            if "[navi-latency-training-update]" in line:
                if current_config is None:
                    raise ValueError("training update precedes scheduler config")
                sequence, update = _parse_update(line)
                updates[sequence] = update
                runtime_config_by_update[sequence] = fit_runtime_config(
                    current_config
                )
    samples.sort(key=lambda sample: int(sample["sequence"]))
    return (
        samples,
        updates,
        history_to_training,
        fit_by_training,
        runtime_config_by_update,
    )


def _sample_event(sample, history_sequence, require_running_elapsed):
    if require_running_elapsed and "running_elapsed_time_ms" not in sample:
        raise ValueError("training sample is missing running_elapsed_time_ms")
    features = [
        int(sample["batch_count"]),
        int(sample["running_batch_size"]),
        int(sample["running_hit_tokens"]),
        int(sample["running_miss_tokens"]),
        float(sample["running_miss_square_sum"]),
        float(sample["running_hit_miss_sum"]),
        int(sample["waiting_batch_size"]),
        int(sample["waiting_hit_tokens"]),
        int(sample["waiting_miss_tokens"]),
        float(sample["waiting_miss_square_sum"]),
        float(sample["waiting_hit_miss_sum"]),
        int(sample["preceding_request_count"]),
        float(sample.get("running_elapsed_time_ms", 0.0)),
    ]
    decode_fields = (
        "running_decode_request_count",
        "running_decode_generated_tokens",
    )
    present = [name in sample for name in decode_fields]
    if any(present) and not all(present):
        raise ValueError("training sample has incomplete decode progress")
    if "running_decode_remaining_token_budget" in sample and not all(present):
        raise ValueError("training sample has decode budget without progress")
    if all(present):
        features.extend(int(sample[name]) for name in decode_fields)
        features.append(
            int(sample.get("running_decode_remaining_token_budget", 0))
        )
    event = {
        "type": "sample",
        "sequence": int(sample["sequence"]),
        "event_time_us": int(sample["event_time_us"]),
        "target_ms": float(sample["target_ms"]),
        "features": features,
        "base_prediction_before_ms": float(sample["base_prediction_before_ms"]),
        "prediction_before_ms": float(sample["prediction_before_ms"]),
    }
    if history_sequence is not None:
        event["history_sequence"] = history_sequence
    return event


def export_log(path, output, require_running_elapsed=False):
    (samples, updates, history_to_training, fit_by_training,
     runtime_config_by_update) = load_log(path)
    training_to_history = {
        training_sequence: history_sequence
        for history_sequence, training_sequence in history_to_training.items()
    }
    batches_by_update = {}
    previous_update_sequence = 0
    for update_sequence in sorted(updates):
        expected = (
            updates[update_sequence]["step_after"]
            - updates[update_sequence]["step_before"]
        )
        current_sequences = sorted(
            sequence
            for sequence in fit_by_training
            if previous_update_sequence < sequence <= update_sequence
        )
        if len(current_sequences) != expected:
            raise ValueError(
                "exact history batch count mismatch: "
                f"update={update_sequence} expected={expected} "
                f"actual={len(current_sequences)}"
            )
        batches_by_update[update_sequence] = [
            fit_by_training[sequence] for sequence in current_sequences
        ]
        previous_update_sequence = update_sequence

    with output.open("w", encoding="utf-8") as stream:
        for sample in samples:
            sequence = int(sample["sequence"])
            stream.write(json.dumps(
                _sample_event(
                    sample,
                    training_to_history.get(sequence),
                    require_running_elapsed,
                ),
                separators=(",", ":"),
            ) + "\n")
            if sequence not in updates:
                continue
            update = updates[sequence]
            event = {
                "type": "update",
                "sequence": sequence,
                **update,
                "fit_batches": batches_by_update[sequence],
                **runtime_config_by_update[sequence],
            }
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-running-elapsed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.input.glob("*.log.gz")):
        output = args.output / (path.name[:-len(".log.gz")] + ".jsonl")
        export_log(path, output, args.require_running_elapsed)
        print(path.stem, output)


if __name__ == "__main__":
    main()
