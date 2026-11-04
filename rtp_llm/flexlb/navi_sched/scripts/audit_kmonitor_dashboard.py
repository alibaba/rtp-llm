#!/usr/bin/env python3
"""Collect NaviSched Grafana/OpenTSDB series with timestamps and original tags."""

import argparse
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


DEFAULT_DEPLOYMENTS = (
    "na130_navi_sched",
    "cn_MI308X_2TP_spot",
    "cn_RTX_PRO_5000_72GB_4TP_spot",
)


def dashboard_targets(dashboard_path):
    payload = json.loads(Path(dashboard_path).read_text(encoding="utf-8"))
    targets = []

    def visit(value):
        if isinstance(value, dict):
            for target in value.get("targets", []):
                if target.get("metric", "").startswith("navi_sched."):
                    targets.append(target)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload["dashboard"])
    return targets


def resolve_tags(tags, deployment):
    resolved = {}
    for key, value in tags.items():
        if value in ("$deployment", "[[deployment]]"):
            resolved[key] = deployment
        elif value in (
            "$agent_host",
            "[[agent_host]]",
            "$host",
            "$host_ip",
            "$__all",
            "All",
            "None",
        ):
            resolved[key] = "*"
        elif isinstance(value, str) and (
            value.startswith("$") or value.startswith("[[")
        ):
            # An unresolved dashboard variable must not silently constrain data.
            resolved[key] = "*"
        else:
            resolved[key] = value
    return resolved


def unique_targets(targets, deployment):
    result = []
    seen = set()
    for target in targets:
        tags = resolve_tags(target.get("tags", {}), deployment)
        key = (target["metric"], tuple(sorted(tags.items())))
        if key in seen:
            continue
        seen.add(key)
        result.append(target)
    return result


def query_payload(targets, deployment, start, end, granularity_seconds=0):
    queries = []
    seen = set()
    for target in unique_targets(targets, deployment):
        metric = target["metric"]
        tags = resolve_tags(target.get("tags", {}), deployment)
        key = (metric, tuple(sorted(tags.items())))
        if key in seen:
            continue
        seen.add(key)
        target_granularity = target.get("granularity", "60")
        downsample_seconds = (
            granularity_seconds
            if granularity_seconds > 0
            else target_granularity
            if str(target_granularity).isdigit()
            else 60
        )
        queries.append({
            "metric": metric,
            "aggregator": target.get("aggregator", "global_avg"),
            "downsample": "%ss-%s" % (
                downsample_seconds,
                target.get("downsampleAggregator", "avg"),
            ),
            "tags": tags,
            "tenant": target.get("tenant", "default"),
            "datasourceId": 5,
            "datasource": {"type": "opentsdbdruid", "uid": "yRF6SvhMz"},
            "intervalMs": 1000,
            "maxDataPoints": 900,
        })
    return {"start": start, "end": end, "limit": "avg:top:100", "queries": queries}


def request_json(endpoint, cookie_path, payload, attempts):
    jar = http.cookiejar.MozillaCookieJar(cookie_path)
    jar.load(ignore_discard=True, ignore_expires=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with opener.open(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("KMon query failed after %d attempts: %s" % (attempts, last_error))


def newest_point(series):
    dps = series.get("dps", {})
    if not dps:
        return None, None
    timestamp = max(int(point) for point in dps)
    return timestamp, dps[str(timestamp)] if str(timestamp) in dps else dps[timestamp]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", required=True)
    parser.add_argument(
        "--cookies",
        default=str(Path.home() / ".kmonitor-auth" / "cookies.txt"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("KMON_QUERY_ENDPOINT", ""),
        help="KMonitor query API URL; defaults to KMON_QUERY_ENDPOINT.",
    )
    parser.add_argument(
        "--deployment",
        action="append",
        dest="deployments",
        help="Deployment to audit; repeat for multiple deployments.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--granularity-seconds",
        type=int,
        default=0,
        help="Override every dashboard target's downsample interval.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=10,
        help="Number of unique dashboard queries per KMonitor request.",
    )
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or KMON_QUERY_ENDPOINT is required")
    if args.query_chunk_size <= 0:
        parser.error("--query-chunk-size must be positive")

    now = int(time.time())
    start = now - args.minutes * 60
    targets = dashboard_targets(args.dashboard)
    deployments = tuple(args.deployments or DEFAULT_DEPLOYMENTS)
    collected = []
    failures = []
    for deployment in deployments:
        deployment_targets = unique_targets(targets, deployment)
        for offset in range(0, len(deployment_targets), args.query_chunk_size):
            chunk = deployment_targets[offset : offset + args.query_chunk_size]
            payload = query_payload(
                chunk,
                deployment,
                start,
                now,
                args.granularity_seconds,
            )
            try:
                response = request_json(
                    args.endpoint, args.cookies, payload, args.attempts
                )
            except RuntimeError as error:
                failures.append({
                    "deployment": deployment,
                    "query_offset": offset,
                    "query_count": len(chunk),
                    "error": str(error),
                })
                continue
            for series in response:
                timestamp, value = newest_point(series)
                collected.append({
                    "deployment": deployment,
                    "metric": series.get("metric"),
                    "tags": series.get("tags", {}),
                    "newest_timestamp": timestamp,
                    "newest_value": value,
                    "point_count": len(series.get("dps", {})),
                    "dps": series.get("dps", {}),
                })

    grouped = defaultdict(int)
    for item in collected:
        grouped[(item["deployment"], item["metric"])] += 1
    output = {
        "collected_at": now,
        "window_start": start,
        "target_count": len(targets),
        "unique_metric_count": len({target["metric"] for target in targets}),
        "query_chunk_size": args.query_chunk_size,
        "series_count": len(collected),
        "series_per_deployment_metric": [
            {"deployment": deployment, "metric": metric, "series_count": count}
            for (deployment, metric), count in sorted(grouped.items())
        ],
        "failures": failures,
        "series": collected,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "series": len(collected),
        "failures": failures,
        "unique_metrics": output["unique_metric_count"],
    }, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
