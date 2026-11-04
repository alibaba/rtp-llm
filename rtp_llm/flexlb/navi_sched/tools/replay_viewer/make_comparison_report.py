#!/usr/bin/env python3

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def distribution(summary, name):
    value = summary["latency_distribution"][name]
    return {
        "average_ms": value["average_ms"],
        "p50_ms": value["p50_ms"],
        "p90_ms": value["p90_ms"],
        "p99_ms": value["p99_ms"],
        "max_ms": value["max_ms"],
    }


def traffic_segments(summary):
    biz_metrics = summary.get("biz_metrics", [])
    if summary.get("traffic_mode") == "mixed" and len(biz_metrics) == 1:
        values = biz_metrics[0].get("traffic_sources", [])
        return [
            {
                "name": value["name"],
                "request_count": value["request_count"],
                "prompt_token_distribution": value[
                    "prompt_token_distribution"
                ],
                "request_share": value.get("request_share"),
                "actual_cache": value.get("actual_cache"),
                "batch_size_distribution": value.get(
                    "batch_size_distribution"
                ),
                "queue_wait": distribution(
                    value, "queue_wait_ms"
                ),
                "execution_latency": distribution(
                    value, "execution_latency_ms"
                ),
                "end_to_end_latency": distribution(
                    value, "client_total_latency_ms"
                ),
            }
            for value in values
        ]
    return [
        {
            "name": value["biz_name"],
            "request_count": value["request_count"],
            "prompt_token_distribution": value[
                "prompt_token_distribution"
            ],
            "request_share": (
                value["request_count"] / summary["request_count"]
            ),
            "actual_cache": value.get("actual_cache"),
            "batch_size_distribution": value.get(
                "batch_size_distribution"
            ),
            "queue_wait": distribution(value, "queue_wait_ms"),
            "execution_latency": distribution(
                value, "execution_latency_ms"
            ),
            "end_to_end_latency": distribution(
                value, "client_total_latency_ms"
            ),
        }
        for value in biz_metrics
    ]


def make_run(arguments, group="Scheduler comparison"):
    label, scheduler, lambda_value, loop_count, path = arguments
    with Path(path).open(encoding="utf-8") as source:
        summary = json.load(source)
    agents = summary["execution_agents"]
    request_count = sum(agent["request_count"] for agent in agents)
    batch_count = sum(agent["batch_count"] for agent in agents)
    request_counts = [agent["request_count"] for agent in agents]
    utilizations = [
        (agent["logical_busy_ms"] / agent["logical_elapsed_ms"]
         if agent["logical_elapsed_ms"] else 0.0)
        for agent in agents
    ]
    loss = summary.get("cost_scheduler_loss")
    scheduler_metrics = summary.get("scheduler_metrics") or {}
    return {
        "group": group,
        "label": label,
        "scheduler": scheduler,
        "experiment_launch_config": summary.get("experiment_launch_config"),
        "scheduler_config": summary.get("scheduler_config"),
        "effective_scheduler_config": summary.get(
            "effective_scheduler_config"
        ),
        "schedule_wait_count": summary.get("schedule_wait_count"),
        "schedule_wait_time_ms": summary.get("schedule_wait_time_ms"),
        "agent_readiness": summary.get("agent_readiness"),
        "random_seed": summary.get("random_seed"),
        "lambda": None if lambda_value == "-" else float(lambda_value),
        "max_loop_count": (
            None if loop_count == "-" else int(loop_count)
        ),
        "request_count": summary["request_count"],
        "agent_count": len(agents),
        "wall_duration_ms": summary["wall_duration_ms"],
        "logical_drain_time_ms": summary["logical_drain_time_ms"],
        "completion_throughput_qps": (
            summary["request_count"] * 1000 /
            (summary["logical_replay_duration_ms"] +
             summary["logical_drain_time_ms"])
        ),
        "average_batch_size": request_count / batch_count,
        "average_utilization": statistics.mean(utilizations),
        "busy_agent_equivalents": sum(utilizations),
        "min_utilization": min(utilizations),
        "max_utilization": max(utilizations),
        "utilization_stddev": statistics.pstdev(utilizations),
        "cache_hit_rate": summary["cache_hit_rate"],
        "agent_request_min": min(request_counts),
        "agent_request_max": max(request_counts),
        "loss": loss.get("distribution") if loss else None,
        "schedule_count": scheduler_metrics.get("schedule_count"),
        "average_requests_per_schedule": scheduler_metrics.get(
            "average_requests_per_schedule"
        ),
        "schedule_latency_us": scheduler_metrics.get(
            "latency_us_distribution"
        ),
        "sampled_loop_count": scheduler_metrics.get(
            "sampled_loop_count_distribution"
        ),
        "sampled_stop_reasons": scheduler_metrics.get(
            "sampled_stop_reasons"
        ),
        "repeat_validation": summary.get("repeat_validation"),
        "decision_prediction_accuracy": summary.get(
            "decision_prediction_accuracy"
        ),
        "queue_prediction_accuracy": summary.get(
            "queue_prediction_accuracy"
        ),
        "traffic_mode": summary.get("traffic_mode", "single"),
        "traffic_segments": traffic_segments(summary),
        "prompt_token_distribution": summary.get(
            "prompt_token_distribution"
        ),
        "biz_metrics": summary.get("biz_metrics", []),
        "queue_wait": distribution(summary, "queue_wait_ms"),
        "execution_latency": distribution(
            summary, "execution_latency_ms"
        ),
        "end_to_end_latency": distribution(
            summary, "client_total_latency_ms"
        ),
        "other_latency_ms": (
            summary["latency_distribution"]["client_total_latency_ms"]
            ["average_ms"]
            - summary["latency_distribution"]["batch_wait_ms"]["average_ms"]
            - summary["latency_distribution"]["queue_wait_ms"]["average_ms"]
            - summary["latency_distribution"]["execution_latency_ms"]
            ["average_ms"]
        ),
        "source": str(Path(path).resolve()),
    }


def make_optimization_checkpoints(path):
    if not path:
        return None
    replay_path = Path(path)
    with replay_path.open(encoding="utf-8") as source:
        replay = json.load(source)
    runs = replay.get("scheduler_info", {}).get("optimization_runs", [])
    if not runs:
        return None
    run = runs[0]
    steps = run.get("steps", [])
    checkpoints = []
    for loop_count in (1, 2, 3, 5, 10, 20, 50, 100):
        if len(steps) < loop_count:
            continue
        phase_names = (
            "forward_us",
            "backward_us",
            "update_us",
            "projection_us",
            "bind_us",
        )
        phases = {
            name: sum(step.get(name, 0) for step in steps[:loop_count])
            for name in phase_names
        }
        projected_loss = (
            steps[loop_count].get("forward_value", 0.0)
            if loop_count < len(steps)
            else run.get("final_forward_value", 0.0)
        )
        checkpoints.append({
            "loop_count": loop_count,
            "projected_loss": projected_loss,
            "elapsed_us": sum(
                step.get("elapsed_us", 0) for step in steps[:loop_count]
            ),
            "phases_us": phases,
        })
    return {
        "request_count": len(replay.get("requests", [])),
        "agent_count": len(replay.get("nodes", [])),
        "source": str(replay_path.resolve()),
        "checkpoints": checkpoints,
    }


def load_optional_json(path):
    if not path:
        return None
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def run_display_identity(run):
    lambda_value = run.get("lambda")
    traffic_mode = run.get("traffic_mode")
    if traffic_mode == "single":
        traffic_mode = "independent"
    identity = (
        run.get("group"),
        run.get("scheduler"),
        None if lambda_value is None else round(float(lambda_value), 8),
        run.get("max_loop_count"),
        run.get("agent_count"),
        traffic_mode,
    )
    if run.get("group") == "Batch window sweep":
        identity += (
            run.get("schedule_wait_count"),
            run.get("schedule_wait_time_ms"),
        )
    return identity


def run_identity(run):
    return run_display_identity(run) + (run.get("agent_readiness"),)


def relative_change(current, previous):
    if previous == 0:
        return None
    return current / previous - 1.0


def make_regression(runs, previous_report):
    if not previous_report:
        return None
    previous_runs = {
        run_identity(run): run
        for run in previous_report.get("runs", [])
    }
    comparisons = []
    for run in runs:
        previous = previous_runs.get(run_identity(run))
        if not previous:
            continue
        e2e_change = relative_change(
            run["end_to_end_latency"]["average_ms"],
            previous["end_to_end_latency"]["average_ms"],
        )
        p99_change = relative_change(
            run["end_to_end_latency"]["p99_ms"],
            previous["end_to_end_latency"]["p99_ms"],
        )
        qps_change = relative_change(
            run["completion_throughput_qps"],
            previous["completion_throughput_qps"],
        )
        cache_hit_change = run["cache_hit_rate"] - previous["cache_hit_rate"]
        utilization_change = (
            run["average_utilization"] - previous["average_utilization"]
        )
        failures = []
        if e2e_change is not None and e2e_change > 0.10:
            failures.append("average E2E regressed by more than 10%")
        if p99_change is not None and p99_change > 0.15:
            failures.append("P99 E2E regressed by more than 15%")
        if qps_change is not None and qps_change < -0.10:
            failures.append("completion QPS regressed by more than 10%")
        if cache_hit_change < -0.03:
            failures.append("cache hit rate regressed by more than 3 pp")
        comparisons.append({
            "group": run["group"],
            "label": run["label"],
            "scheduler": run["scheduler"],
            "lambda": run["lambda"],
            "agent_count": run["agent_count"],
            "status": "failed" if failures else "passed",
            "failures": failures,
            "e2e_average_change": e2e_change,
            "e2e_p99_change": p99_change,
            "completion_qps_change": qps_change,
            "cache_hit_change": cache_hit_change,
            "utilization_change": utilization_change,
        })
    return {
        "baseline_generated_at": previous_report.get("generated_at"),
        "matched_run_count": len(comparisons),
        "failed_run_count": sum(
            comparison["status"] == "failed"
            for comparison in comparisons
        ),
        "comparisons": comparisons,
    }


def merge_runs(base_runs, new_runs):
    new_identities = {
        run_display_identity(run) for run in new_runs
    }
    return new_runs + [
        run for run in base_runs
        if run_display_identity(run) not in new_identities
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Scheduler comparison")
    parser.add_argument("--agent-count", type=int)
    parser.add_argument("--client-count", type=int, required=True)
    parser.add_argument("--time-compression", type=float, required=True)
    parser.add_argument("--optimization-replay", default="")
    parser.add_argument("--scalability", default="")
    parser.add_argument("--base-report", default="")
    parser.add_argument("--previous-report", default="")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        default=[],
        metavar=("LABEL", "SCHEDULER", "LAMBDA", "LOOPS", "SUMMARY"),
    )
    parser.add_argument(
        "--grouped-run",
        action="append",
        nargs=6,
        default=[],
        metavar=(
            "GROUP", "LABEL", "SCHEDULER", "LAMBDA", "LOOPS", "SUMMARY"
        ),
    )
    args = parser.parse_args()
    base_report = load_optional_json(args.base_report)
    previous_report = load_optional_json(args.previous_report)
    base_runs = list(base_report.get("runs", [])) if base_report else []
    new_runs = [make_run(run) for run in args.run]
    new_runs.extend(make_run(run[1:], run[0]) for run in args.grouped_run)
    runs = merge_runs(base_runs, new_runs)
    if not runs:
        parser.error("at least one --run or --grouped-run is required")
    agent_counts = sorted({run["agent_count"] for run in runs})
    report = {
        key: value
        for key, value in (base_report or {}).items()
        if key not in {
            "title",
            "generated_at",
            "experiment",
            "runs",
            "optimization",
            "scalability",
            "regression",
        }
    }
    report.update({
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "agent_count": (
                args.agent_count
                if args.agent_count is not None
                else (agent_counts[0] if len(agent_counts) == 1 else None)
            ),
            "agent_counts": agent_counts,
            "client_count": args.client_count,
            "time_compression": args.time_compression,
            "request_count": runs[0]["request_count"],
        },
        "runs": runs,
        "optimization": (
            make_optimization_checkpoints(args.optimization_replay)
            if args.optimization_replay
            else (base_report or {}).get("optimization")
        ),
        "scalability": (
            load_optional_json(args.scalability)
            if args.scalability
            else (base_report or {}).get("scalability")
        ),
        "regression": make_regression(new_runs, previous_report),
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    regression = report.get("regression") or {}
    if args.fail_on_regression and regression.get("failed_run_count", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
