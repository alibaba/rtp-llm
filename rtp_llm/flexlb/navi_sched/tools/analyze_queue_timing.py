#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path


AGENT_MARKER = "[navi-queue-debug]"
TRAINING_MARKER = "[navi-latency-training-sample]"
RTP_MARKER = "[navi-queue-timing]"
COMPONENT_MARKER = "[navi-latency-components]"
FIELD_PATTERN = re.compile(r"([a-z][a-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?)")
HOST_PATTERN = re.compile(r"\[(\d+\.\d+\.\d+\.\d+)\]")
TIMESTAMP_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} [0-9:.]+)\]")


def parse_fields(line):
    return {
        key: float(value) if "." in value else int(value)
        for key, value in FIELD_PATTERN.findall(line)
    }


def parse_agent(path):
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if AGENT_MARKER not in line:
            continue
        fields = parse_fields(line)
        host_match = HOST_PATTERN.search(line)
        timestamp_match = TIMESTAMP_PATTERN.search(line)
        fields["host"] = host_match.group(1) if host_match else ""
        fields["timestamp"] = timestamp_match.group(1) if timestamp_match else ""
        schedule_batch_id = int(fields.get("schedule_batch_id", 0))
        if schedule_batch_id > 0:
            records[schedule_batch_id] = fields
    return records


def parse_training_samples(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if TRAINING_MARKER not in line or "type=batch_finished" not in line:
            continue
        fields = parse_fields(line)
        host_match = HOST_PATTERN.search(line)
        timestamp_match = TIMESTAMP_PATTERN.search(line)
        fields["host"] = host_match.group(1) if host_match else ""
        fields["timestamp"] = (
            timestamp_match.group(1) if timestamp_match else ""
        )
        fields["queue_features_field_present"] = int(
            "queue_features_valid" in fields
        )
        fields["queue_features_valid"] = int(
            fields.get("queue_features_valid", 0)
        )
        if fields["queue_features_valid"]:
            actual_queue_ms = fields["backend_queue_ms"]
            master_queue_ms = fields["master_queue_prediction_ms"]
            raw_queue_ms = fields["queue_raw_at_ready_ms"]
            agent_queue_ms = fields["agent_queue_at_ready_ms"]
            fields.update({
                "actual_queue_ms": actual_queue_ms,
                "master_queue_error_ms": actual_queue_ms - master_queue_ms,
                "master_queue_absolute_error_ms": abs(
                    actual_queue_ms - master_queue_ms),
                "raw_queue_error_ms": actual_queue_ms - raw_queue_ms,
                "raw_queue_absolute_error_ms": abs(
                    actual_queue_ms - raw_queue_ms),
                "agent_queue_at_ready_error_ms":
                    actual_queue_ms - agent_queue_ms,
                "agent_queue_at_ready_absolute_error_ms": abs(
                    actual_queue_ms - agent_queue_ms),
            })
        rows.append(fields)
    return rows


def parse_rtp(path):
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if RTP_MARKER not in line and COMPONENT_MARKER not in line:
            continue
        fields = parse_fields(line)
        batch_id = int(fields.get("batch_id", 0))
        if batch_id <= 0:
            continue
        record = records.setdefault(batch_id, {})
        if "handler_total_us" in fields:
            record["handler"] = fields
        if "create_to_start_us" in fields:
            record["execution"] = fields
        if COMPONENT_MARKER in line:
            record["components"] = fields
    return records


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def join(agent_records, rtp_records):
    rows = []
    for schedule_batch_id, agent in agent_records.items():
        rtp = rtp_records.get(schedule_batch_id)
        if not rtp or "handler" not in rtp or "execution" not in rtp or \
                "components" not in rtp:
            continue
        handler = rtp["handler"]
        execution = rtp["execution"]
        components = rtp["components"]
        actual_queue_us = max(
            0.0,
            agent["queue_end_canrun_time_us"] - agent["ready_time_us"],
        )
        create_to_start_us = execution["create_to_start_us"]
        scheduler_queue_us = execution["scheduler_queue_us"]
        enqueue_to_canrun_us = execution["enqueue_to_canrun_us"]
        actual_execution_ms = agent["actual_execution_ms"]
        master_execution_ms = agent["master_execution_ms"]
        agent_execution_ms = agent["agent_execution_ms"]
        agent_queue_at_ready_ms = agent.get("agent_queue_at_ready_ms")
        row = {
            "schedule_batch_id": schedule_batch_id,
            "host": agent["host"],
            "timestamp": agent["timestamp"],
            "local_batch_id": int(agent["local_batch_id"]),
            "request_count": int(agent["request_count"]),
            "master_queue_ms": agent["master_queue_ms"],
            "raw_queue_ms": agent["raw_queue_ms"],
            "preceding_batch_count": int(agent["preceding_batch_count"]),
            "queue_fit_step": int(agent["queue_fit_step"]),
            "actual_queue_ms": actual_queue_us / 1000.0,
            "reported_actual_queue_ms": agent["actual_queue_ms"],
            "actual_queue_timestamp_error_ms":
                (actual_queue_us / 1000.0) - agent["actual_queue_ms"],
            "request_count_difference":
                int(agent["request_count"]) - int(execution["requests"]),
            "master_queue_error_ms":
                (actual_queue_us / 1000.0) - agent["master_queue_ms"],
            "master_queue_absolute_error_ms": abs(
                (actual_queue_us / 1000.0) - agent["master_queue_ms"]),
            "raw_queue_error_ms":
                (actual_queue_us / 1000.0) - agent["raw_queue_ms"],
            "raw_queue_absolute_error_ms": abs(
                (actual_queue_us / 1000.0) - agent["raw_queue_ms"]),
            "ready_to_scheduler_enqueue_ms":
                (actual_queue_us - enqueue_to_canrun_us) / 1000.0,
            "create_to_start_ms": create_to_start_us / 1000.0,
            "scheduler_queue_ms": scheduler_queue_us / 1000.0,
            "enqueue_to_canrun_ms": enqueue_to_canrun_us / 1000.0,
            "after_canrun_to_timing_log_ms":
                (scheduler_queue_us - enqueue_to_canrun_us) / 1000.0,
            "enqueue_group_ms": handler["enqueue_group_us"] / 1000.0,
            "canrun_to_running_ms":
                execution["canrun_to_running_us"] / 1000.0,
            "loading_cache_ms": execution["loading_cache_us"] / 1000.0,
            "master_execution_ms": master_execution_ms,
            "agent_execution_ms": agent_execution_ms,
            "actual_execution_ms": actual_execution_ms,
            "master_execution_error_ms":
                actual_execution_ms - master_execution_ms,
            "agent_execution_error_ms":
                actual_execution_ms - agent_execution_ms,
            "master_execution_ape_pct":
                abs(actual_execution_ms - master_execution_ms) * 100.0 /
                actual_execution_ms,
            "agent_execution_ape_pct":
                abs(actual_execution_ms - agent_execution_ms) * 100.0 /
                actual_execution_ms,
        }
        if agent_queue_at_ready_ms is not None:
            row["agent_queue_at_ready_ms"] = agent_queue_at_ready_ms
            row["agent_queue_at_ready_error_ms"] = (
                (actual_queue_us / 1000.0) - agent_queue_at_ready_ms)
            row["agent_queue_at_ready_absolute_error_ms"] = abs(
                row["agent_queue_at_ready_error_ms"])
        elif "fitted_queue_ms" in agent:
            row["legacy_finish_recomputed_queue_ms"] = agent["fitted_queue_ms"]
        for key, value in components.items():
            if key != "batch_id":
                row[key] = int(value)
        rows.append(row)
    return rows


def summarize(rows):
    fields = [
        "actual_queue_ms",
        "actual_queue_timestamp_error_ms",
        "request_count_difference",
        "master_queue_error_ms",
        "master_queue_absolute_error_ms",
        "raw_queue_error_ms",
        "raw_queue_absolute_error_ms",
        "agent_queue_at_ready_error_ms",
        "agent_queue_at_ready_absolute_error_ms",
        "ready_to_scheduler_enqueue_ms",
        "create_to_start_ms",
        "scheduler_queue_ms",
        "enqueue_to_canrun_ms",
        "after_canrun_to_timing_log_ms",
        "enqueue_group_ms",
        "canrun_to_running_ms",
        "loading_cache_ms",
        "master_execution_error_ms",
        "agent_execution_error_ms",
        "master_execution_ape_pct",
        "agent_execution_ape_pct",
    ]
    return {
        field: distribution([row[field] for row in rows if field in row])
        for field in fields
    }


def training_semantics(rows):
    if not rows:
        return "unavailable"
    present_count = sum(
        row["queue_features_field_present"] for row in rows
    )
    valid_count = sum(row["queue_features_valid"] for row in rows)
    if present_count == 0:
        return "legacy_finish_time_recomputed_or_unavailable"
    if present_count == len(rows) and valid_count == len(rows):
        return "ready_time_frozen"
    return "mixed_or_missing"


def summarize_training(rows):
    valid_rows = [row for row in rows if row["queue_features_valid"]]
    fields = [
        "actual_queue_ms",
        "master_queue_error_ms",
        "master_queue_absolute_error_ms",
        "raw_queue_error_ms",
        "raw_queue_absolute_error_ms",
        "agent_queue_at_ready_error_ms",
        "agent_queue_at_ready_absolute_error_ms",
    ]
    return {
        field: distribution([row[field] for row in valid_rows])
        for field in fields
    }


def summarize_training_stages(rows):
    valid_rows = [row for row in rows if row["queue_features_valid"]]
    stages = {
        "cold_fit_step_lt_50": [
            row for row in valid_rows
            if row["queue_fit_step_at_ready"] < 50
        ],
        "transition_fit_step_50_99": [
            row for row in valid_rows
            if 50 <= row["queue_fit_step_at_ready"] < 100
        ],
        "steady_fit_step_ge_100": [
            row for row in valid_rows
            if row["queue_fit_step_at_ready"] >= 100
        ],
        "steady_fit_step_ge_200": [
            row for row in valid_rows
            if row["queue_fit_step_at_ready"] >= 200
        ],
    }
    return {
        name: {
            "sample_count": len(stage_rows),
            "summary_ms": summarize_training(stage_rows),
        }
        for name, stage_rows in stages.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Join NaviSched Agent and RTP per-batch Queue timing logs")
    parser.add_argument("--agent-log", required=True)
    parser.add_argument("--rtp-log", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    agent_records = parse_agent(args.agent_log)
    training_rows = parse_training_samples(args.agent_log)
    rtp_records = parse_rtp(args.rtp_log)
    rows = join(agent_records, rtp_records)
    filename_host = Path(args.agent_log).name.removesuffix(".agent.log").replace("_", ".")
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", filename_host):
        for row in rows:
            if not row["host"]:
                row["host"] = filename_host
        for row in training_rows:
            if not row["host"]:
                row["host"] = filename_host
    ready_training_rows = [
        row for row in training_rows if row["queue_features_valid"]
    ]
    result = {
        "agent_batch_count": len(agent_records),
        "rtp_batch_count": len(rtp_records),
        "rtp_component_count": sum(
            "components" in record for record in rtp_records.values()),
        "joined_batch_count": len(rows),
        "agent_training_sample_count": len(training_rows),
        "ready_prediction_training_sample_count": len(ready_training_rows),
        "training_queue_prediction_semantics": training_semantics(
            training_rows),
        "agent_queue_prediction_semantics": (
            "ready_time_frozen"
            if rows and all("agent_queue_at_ready_ms" in row for row in rows)
            else "legacy_finish_time_recomputed"
            if rows and all("legacy_finish_recomputed_queue_ms" in row for row in rows)
            else "mixed_or_unavailable"
        ),
        "summary_ms": summarize(rows),
        "training_summary_ms": summarize_training(training_rows),
        "training_stages": summarize_training_stages(training_rows),
        "training_samples": ready_training_rows,
        "batches": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
