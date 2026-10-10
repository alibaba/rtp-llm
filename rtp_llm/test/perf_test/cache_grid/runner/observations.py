"""Common, strict request-list observations for scalar and grouped measurements."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    current_metrics,
    run_prefill_rt,
)

CONTRACT = "batch_max_server_first_token_cost_time_ms"


def statistic(values, estimator):
    if estimator == "min":
        return min(values)
    if estimator == "trimmed":
        return median(sorted(values)[:-1]) if len(values) > 1 else values[0]
    return median(values)


def normalize_metric(item, source, index, estimator="median"):
    if item["status"] != "ok":
        raise ValueError("status")
    batch = item["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("invalid_batch_size")
    groups = item.get("request_groups")
    if groups is None:
        if batch != 1:
            raise ValueError("missing_request_distribution")
        groups = [
            dict(
                count=1,
                input_len=item["input_len"],
                cache_len=item.get("cache_len_requested"),
            )
        ]
    expected = []
    if not isinstance(groups, list) or any(not isinstance(g, dict) for g in groups):
        raise ValueError("invalid_request_groups")
    for group in groups:
        count, inp, cache = (group.get(k) for k in ("count", "input_len", "cache_len"))
        if (
            any(type(v) is not int for v in (count, inp, cache))
            or count <= 0
            or not 0 <= cache < inp
        ):
            raise ValueError("invalid_geometry")
        expected.extend([(inp, cache)] * count)
    if len(expected) != batch:
        raise ValueError("request_count_mismatch")
    runs = item["runs"]
    if (
        not runs
        or len(runs) != item["measure_runs"]
        or item["success_runs"] != len(runs)
    ):
        raise ValueError("incomplete_runs")
    rounds, distributions, walls = [], [], []
    for run in runs:
        requests = run.get("requests", [run])
        if len(requests) != batch or (
            "requests" in run and run.get("valid") is not True
        ):
            raise ValueError("invalid_round")
        distribution, times = [], []
        for request, (inp, cache) in zip(requests, expected):
            if request.get("success") is not True:
                raise ValueError("run_failed")
            if request.get("input_len") != inp or request.get("output_len") != 1:
                raise ValueError("request_shape_mismatch")
            reuse = request.get("reuse_len")
            if type(reuse) is not int or not 0 <= reuse < inp:
                raise ValueError("invalid_observed_geometry")
            if reuse != cache:
                raise ValueError("requested_reuse_mismatch")
            latency = run_prefill_rt(request)
            if latency is None:
                raise ValueError("invalid_latency")
            times.append(latency)
            distribution.append([inp, reuse, inp - reuse])
        distributions.append(distribution)
        rounds.append(max(times))
        walls.append(run.get("batch_wall_time_ms", run.get("client_wall_time_ms")))
    if any(d != distributions[0] for d in distributions):
        raise ValueError("observed_reuse_not_constant")
    distribution = distributions[0]
    # Request permutations are the same geometry for split/leakage purposes.
    geometry = json.dumps(sorted(distribution), separators=(",", ":"))
    return dict(
        source_run=str(source),
        case_id=item.get("case_id", index),
        metric_index=index,
        batch_size=batch,
        request_distribution=distribution,
        geometry_key=hashlib.sha256(geometry.encode()).hexdigest(),
        input_len=sum(r[0] for r in distribution),
        cache_len=sum(r[1] for r in distribution),
        compute_len=sum(r[2] for r in distribution),
        target_ms=statistic(rounds, estimator),
        formal_rounds_ms=rounds,
        wall_times_ms=walls,
        measurement_contract=CONTRACT,
    )


def collect(paths, *, batch_size=None, estimator="median", partial=False):
    rows, sources, excluded = [], [], []
    counts = Counter()
    seen_paths = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate result source: {path}")
        seen_paths.add(path)
        content = path.read_bytes()
        data = json.loads(content)
        if not data.get("complete") and not partial:
            raise ValueError(f"incomplete result (use --partial): {path}")
        metrics = current_metrics(data, str(path))
        sources.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(content).hexdigest(),
                complete=bool(data.get("complete")),
                rows=len(metrics),
            )
        )
        for index, item in enumerate(metrics):
            try:
                if batch_size is not None and item["batch_size"] != batch_size:
                    raise ValueError("batch_size")
                rows.append(normalize_metric(item, path, index, estimator))
            except (ValueError, TypeError, KeyError) as error:
                reason = str(error)
                counts[reason] += 1
                excluded.append(
                    dict(
                        source_run=str(path),
                        metric_index=index,
                        case_id=item.get("case_id", index),
                        reason=reason,
                    )
                )
    audit = dict(
        sources=sources,
        excluded=excluded,
        rejected_counts=dict(counts),
        valid_observation_count=len(rows),
        estimator=estimator,
        measurement_contract=CONTRACT,
        partial=any(not s["complete"] for s in sources),
        counts_by_batch=dict(Counter(str(r["batch_size"]) for r in rows)),
    )
    return rows, audit
