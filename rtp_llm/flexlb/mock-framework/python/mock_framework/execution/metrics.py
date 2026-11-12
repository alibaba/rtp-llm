"""Build metric queries and check sampling coverage for a Case."""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from ..config.validation import duration
from . import ExecutionError
from .step.step_registry import Targets


@dataclass(frozen=True)
class MetricArgs(Targets):
    range: Annotated[list[str], "range"] = field(default_factory=lambda: ["-60s", "0s"])
    mode: Literal["interval", "series"] = "interval"


@dataclass(frozen=True)
class LatencyArgs(MetricArgs):
    statistic: Literal["mean", "p95", "p99"] = "mean"


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ExecutionError("Query time requires timezone")
    return parsed.timestamp()


async def instant(ctx, expression, at):
    monitor = ctx.runtime.monitoring
    if monitor is None:
        raise ExecutionError("This Case needs a runner-managed Prometheus")
    try:
        result = await monitor.query(expression, at, ctx.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f"Prometheus query failed: {error}") from error
    ctx.records.emit(
        event="metric_query", query_time=ctx.query_time, at=at, promql=expression
    )
    return result


def query_window(ctx, args):
    end = epoch(ctx.query_time)
    start, stop = (duration(v, negative=True) for v in args.range)
    millis = (stop - start) * 1000
    if millis < 1 or millis != int(millis):
        raise ExecutionError("Metric window must contain integral milliseconds")
    return end + float(stop), int(millis)


def scope(ctx, args, role):
    available = {item["rpc"] for item in ctx.runtime.topology.active(role)}
    targets = sorted(available) if args.targets is None else args.targets
    if not targets or set(targets) - available:
        raise ExecutionError(f"No known {role} metric targets or unknown address")
    expression = "|".join(re.escape(address) for address in targets)
    return f"role={json.dumps(role)},instance=~{json.dumps(expression)}", targets


def history_range(ctx, name, labels, millis):
    if labels is not None:
        return f"{name}{{{labels}}}[{millis}ms]"
    # A derived worker gauge: sum priority at each scrape before averaging over time.
    step = max(1, round(ctx.runtime.monitoring.interval * 1000))
    return f"({name})[{millis}ms:{step}ms]"


def _check_history(name, targets, series):
    labels = [samples[0].labels for samples in series]
    if {label.get("instance") for label in labels} != set(targets) or any(
        sample.value < 2 for samples in series for sample in samples
    ):
        raise ExecutionError("Incomplete metric coverage for selected instances")
    if name.endswith("_bucket"):
        buckets = {label.get("le") for label in labels}
        pairs = {(label.get("instance"), label.get("le")) for label in labels}
        # Client instances export the same histogram. Each must cover every observed bound.
        if (
            None in buckets
            or "+Inf" not in buckets
            or len(buckets) < 2
            or len(labels) != len(targets) * len(buckets)
            or len(pairs) != len(labels)
        ):
            raise ExecutionError(
                "Incomplete histogram bucket coverage for selected instances"
            )
    elif len(labels) != len(targets):
        raise ExecutionError("Incomplete metric coverage for selected instances")


async def query_counter(ctx, args, role, numerator, denominator=None, *, kind=None):
    millis = counter_window_ms(ctx, args)
    labels, targets = scope(ctx, args, role)
    numerator_labels = labels + (f",kind={json.dumps(kind)}" if kind else "")
    histories = [(numerator, numerator_labels, targets)]

    def aggregate(name, selector):
        return f"sum(increase({name}{{{selector}}}[{millis}ms]))"

    expression = aggregate(numerator, numerator_labels)
    if denominator:
        histories.append((denominator, labels, targets))
        expression += " / " + aggregate(denominator, labels)
    else:
        expression += " / " + str(Decimal(millis) / 1000)
    return await evaluate(ctx, args, expression, histories)


def sample_period(ctx):
    # Like Grafana's minimum rate interval: enough scrapes for a stable counter delta.
    monitor = ctx.runtime.monitoring
    if monitor is None:
        raise ExecutionError("This Case needs a runner-managed Prometheus")
    return max(0.001, monitor.interval) * 4


def counter_window_ms(ctx, args):
    _, millis = query_window(ctx, args)
    if args.mode == "series":
        return max(1, round(sample_period(ctx) * 1000))
    return millis


async def series_query(ctx, expression, start, end, step):
    monitor = ctx.runtime.monitoring
    if monitor is None:
        raise ExecutionError("This Case needs a runner-managed Prometheus")
    try:
        values = await monitor.query_range(expression, start, end, step, ctx.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f"Prometheus range query failed: {error}") from error
    expected = int(round((end - start) / step)) + 1
    for samples in values:
        if len(samples) != expected or any(
            abs(sample.sample_time - (start + i * step)) > 0.001
            for i, sample in enumerate(samples)
        ):
            raise ExecutionError("Incomplete metric range coverage")
    ctx.records.emit(
        event="metric_query",
        query_time=ctx.query_time,
        start=start,
        end=end,
        step=step,
        promql=expression,
    )
    return values


async def evaluate(ctx, args, expression, histories):
    at, millis = query_window(ctx, args)
    start = at - millis / 1000
    if args.mode == "series":
        period = sample_period(ctx)
        step = period / 4
        count = int(millis / 1000 / step)
        if not 1 <= count <= 10000:
            raise ExecutionError("series requires 1–10000 sampling points")
        start = at - (count - 1) * step
    ctx.records.emit(
        event="metric_window", query_time=ctx.query_time, start=start, end=at
    )

    if args.mode == "interval":
        for name, labels, targets in histories:
            counts = await instant(
                ctx, f"count_over_time({history_range(ctx, name, labels, millis)})", at
            )
            _check_history(name, targets, [[sample] for sample in counts])
        values = await instant(ctx, expression, at)
        if len(values) != 1 or not math.isfinite(values[0].value):
            raise ExecutionError("Metric query must return one finite sample")
        return values[0].value
    period_millis = max(1, round(period * 1000))
    for name, labels, targets in histories:
        coverage = await series_query(
            ctx,
            f"count_over_time({history_range(ctx, name, labels, period_millis)})",
            start,
            at,
            step,
        )
        _check_history(name, targets, coverage)
    values = await series_query(ctx, expression, start, at, step)
    if len(values) != 1:
        raise ExecutionError("Metric query must return one complete series")
    points = values[0]
    ctx.records.emit(
        event="metric_series",
        samples=[{"time": p.sample_time, "value": p.value} for p in points],
    )
    return [p.value for p in points]


async def gauge(ctx, args, role, name):
    _, millis = query_window(ctx, args)
    labels, targets = scope(ctx, args, role)
    worker = f"sum by(instance)({name}{{{labels}}})"
    expression = (
        f"sum(avg_over_time({history_range(ctx, worker, None, millis)}))"
        if args.mode == "interval"
        else f"sum({worker})"
    )
    return await evaluate(ctx, args, expression, [(worker, None, targets)])


async def latency(ctx, args, name):
    millis = counter_window_ms(ctx, args)
    labels, targets = scope(ctx, args, "client")
    if args.statistic == "mean":
        histories = [(name + suffix, labels, targets) for suffix in ("_sum", "_count")]
        expression = f"sum(increase({name}_sum{{{labels}}}[{millis}ms])) / sum(increase({name}_count{{{labels}}}[{millis}ms]))"
    else:
        histories = [(name + "_bucket", labels, targets)]
        quantile = {"p95": 0.95, "p99": 0.99}[args.statistic]
        expression = f"histogram_quantile({quantile}, sum by(le)(increase({name}_bucket{{{labels}}}[{millis}ms])))"
    return await evaluate(ctx, args, expression, histories)
