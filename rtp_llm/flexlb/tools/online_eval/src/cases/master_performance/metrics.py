"""Request-cohort projections published by the performance case."""

import math
from collections import defaultdict


def values(evidence, result):
    lo = evidence.get("window", {}).get("start_epoch_ms", 0)
    duration = evidence.get("criteria", {}).get("measure_s", 1)
    output = {}

    for key in ("input_tps", "output_tps", "inflight"):
        output["request/" + key] = [[lo/1000 + w["t"], w[key]] for w in result["windows"]]
    sent, done = defaultdict(list), defaultdict(list)
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
            sent[math.floor((r["send_start_epoch_ms"] - lo) / 1000)].append(r)
    for r in records:
        if isinstance(r.get("send_start_epoch_ms"), (int, float)) and isinstance(
            r.get("total_ms"), (int, float)
        ):
            done[
                math.floor((r["send_start_epoch_ms"] + r["total_ms"] - lo) / 1000)
            ].append(r)

    def p99(values):
        values = sorted(
            v for v in values if isinstance(v, (int, float)) and math.isfinite(v)
        )
        return values[max(0, math.ceil(0.99 * len(values)) - 1)] if values else None

    metrics = defaultdict(list)
    for i in range(math.ceil(duration)):
        arrivals, terminals = sent[i], done[i]
        ok = [r for r in arrivals if r.get("status") == "ok"]
        dt = min(1, duration - i)
        vals = {
            "发送 QPS": len(arrivals) / dt,
            "完成 QPS": len(terminals) / dt,
            "成功 QPS": sum(r.get("status") == "ok" for r in terminals) / dt,
            "错误 QPS": sum(r.get("status") != "ok" for r in terminals) / dt,
            "到达 cohort 成功率": len(ok) / len(arrivals) if arrivals else None,
            "TTFT p99": p99([r.get("ttft_ms") for r in ok]),
            "E2E p99": p99([r.get("total_ms") for r in ok]),
            "TPOT p99": p99(
                [
                    (r["total_ms"] - r["ttft_ms"]) / (r["observed_output_tokens"] - 1)
                    for r in ok
                    if r.get("observed_output_tokens", 0) > 1
                    and isinstance(r.get("total_ms"), (int, float))
                    and isinstance(r.get("ttft_ms"), (int, float))
                ]
            ),
            "输入长度均值": (
                sum(r["input_len"] for r in arrivals) / len(arrivals)
                if arrivals
                else None
            ),
            "实际输出长度均值": (
                sum(r.get("observed_output_tokens", 0) for r in ok) / len(ok)
                if ok
                else None
            ),
        }
        for k, v in vals.items():
            metrics[k].append((i, v))
    request_ids = {
        "发送 QPS": "sent_qps", "完成 QPS": "completed_qps",
        "成功 QPS": "success_qps", "错误 QPS": "error_qps",
        "到达 cohort 成功率": "arrival_success_ratio",
        "TTFT p99": "ttft_p99_ms", "E2E p99": "e2e_p99_ms",
        "TPOT p99": "tpot_p99_ms", "输入长度均值": "input_tokens_mean",
        "实际输出长度均值": "output_tokens_mean",
    }
    for name, points in metrics.items():
        output["request/" + request_ids[name]] = [[lo/1000+t, value] for t, value in points]
    return output


def produce(directory, evidence, result):
    from monitoring.metric_store import MetricStore, export_metrics, publish
    from monitoring.query_plan import load_plan
    export_metrics(directory, load_plan("master_performance.yaml"))
    store = MetricStore.read(directory)
    for identity, points in values(evidence, result).items():
        publish(store, identity, store.document["definitions"][identity],
                [dict(epoch="1", source="client", labels={}, points=points)],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence.get("window"), algorithm="request_cohort_seconds"))

    for name, value in result["metrics"].items():
        key = name.split("/")[-1]
        identity = "performance_gate/" + key
        if name.startswith("mock/") and not key.endswith("_engine_count"):
            identity += "_scrape_engine_mean"
        publish(store, identity, store.document["definitions"][identity],
                [dict(epoch="1", source="performance_gate", labels={},
                      points=[[evidence["window"]["end_epoch_ms"]/1000, value]])],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence["window"], measurement_validity="INVALID" if result["verdict"] == "INVALID" else "VALID",
                    algorithm="absolute_performance_gate"))
    store.save(directory)
