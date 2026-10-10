"""Request-cohort projections published by the performance case."""

import math
from collections import defaultdict

from analysis.statistics import percentile_nr
from analysis.time_buckets import TimeBuckets
from monitoring.metric_store import series_row


def values(evidence, result):
    lo = evidence["window"]["start_epoch_ms"]
    duration = evidence["criteria"]["measure_s"]
    grid = TimeBuckets(lo / 1000, 1)
    output = {}

    for key in ("input_tps", "output_tps", "inflight"):
        output["request/" + key] = [[lo/1000 + w["t"], w[key]] for w in result["windows"]]
    arrivals = []
    records = evidence.get("flow", {}).get("records", [])
    terminals_by_id = {r.get("rid"): r for r in records}
    for issued in evidence.get("flow", {}).get("issued", []):
        r = dict(
            issued,
            **{
                k: v
                for k, v in terminals_by_id.get(issued.get("rid"), {}).items()
                if k not in issued
            },
        )
        if isinstance(r.get("send_start_epoch_ms"), (int, float)):
            arrivals.append(r)
    sent = grid.group(arrivals, timestamp_ms=lambda r: r["send_start_epoch_ms"])
    done = grid.group(
        [r for r in records if isinstance(r.get("send_start_epoch_ms"), (int, float))
         and isinstance(r.get("total_ms"), (int, float))],
        timestamp_ms=lambda r: r["send_start_epoch_ms"] + r["total_ms"],
    )

    def p99(values):
        return percentile_nr(
            [v for v in values if type(v) in (int, float) and math.isfinite(v)], 0.99)

    metrics = defaultdict(list)
    for i in range(math.ceil(duration)):
        arrivals, terminals = sent[i], done[i]
        ok = [r for r in arrivals if r.get("status") == "ok"]
        dt = min(1, duration - i)
        vals = {
            "sent_qps": len(arrivals) / dt,
            "completed_qps": len(terminals) / dt,
            "success_qps": sum(r.get("status") == "ok" for r in terminals) / dt,
            "error_qps": sum(r.get("status") != "ok" for r in terminals) / dt,
            "arrival_success_ratio": len(ok) / len(arrivals) if arrivals else None,
            "ttft_p99_ms": p99([r.get("ttft_ms") for r in ok]),
            "e2e_p99_ms": p99([r.get("total_ms") for r in ok]),
            "tpot_p99_ms": p99(
                [
                    (r["total_ms"] - r["ttft_ms"]) / (r["observed_output_tokens"] - 1)
                    for r in ok
                    if r.get("observed_output_tokens", 0) > 1
                    and isinstance(r.get("total_ms"), (int, float))
                    and isinstance(r.get("ttft_ms"), (int, float))
                ]
            ),
            "input_tokens_mean": (
                sum(r["input_len"] for r in arrivals) / len(arrivals)
                if arrivals
                else None
            ),
            "output_tokens_mean": (
                sum(r.get("observed_output_tokens", 0) for r in ok) / len(ok)
                if ok
                else None
            ),
        }
        for k, v in vals.items():
            metrics[k].append((i, v))
    for name, points in metrics.items():
        output["request/" + name] = [[grid.epoch_s(t), value] for t, value in points]
    return output


def produce(directory, evidence, result):
    from monitoring.metric_store import MetricStore, export_metrics, publish
    from monitoring.query_plan import load_plan
    export_metrics(directory, load_plan("master_performance.yaml"))
    store = MetricStore.read(directory)
    epoch = evidence["provenance"]["env_epoch"]
    for identity, points in values(evidence, result).items():
        publish(store, identity, store.document["definitions"][identity],
                [series_row(points, epoch=epoch, source="client", labels={})],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence.get("window"), algorithm="request_cohort_seconds"))

    for name, value in result["metrics"].items():
        key = name.split("/")[-1]
        identity = "performance_gate/" + key
        if name.startswith("mock/") and not key.endswith("_engine_count"):
            identity += "_scrape_engine_mean"
        publish(store, identity, store.document["definitions"][identity],
                [series_row([[evidence["window"]["end_epoch_ms"]/1000, value]],
                            epoch=epoch, source="performance_gate", labels={})],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence["window"], measurement_validity="INVALID" if result["verdict"] == "INVALID" else "VALID",
                    algorithm="absolute_performance_gate"))
    store.save(directory)
