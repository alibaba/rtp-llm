"""TSDB query operations; no exporter reads or process lifecycle."""

import json
import math
import time
import urllib.parse
import urllib.request

from monitoring.query_plan import definitions
from monitoring.contracts import MetricContractError


class PrometheusClient:
    """Query a bound TSDB using an explicitly shared target-kind inventory."""

    def __init__(self, url, *, query_plan, target_kinds, max_gap_s):
        self.url = url
        self.query_plan = query_plan
        self.target_kinds = target_kinds
        self.max_gap = max_gap_s

    def api(self, endpoint, timeout=5, *, deadline=None, **params):
        if self.url is None:
            raise RuntimeError("Prometheus client has no bound TSDB URL")
        url = self.url + "/api/v1/" + endpoint + "?" + urllib.parse.urlencode(params)
        if deadline is not None:
            timeout = min(timeout, deadline.remaining())
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
        if deadline is not None:
            deadline.check()
        if data.get("status") != "success" or data.get("warnings"):
            raise RuntimeError("Prometheus query failed or incomplete: " + str(data))
        return data["data"]

    def query(self, expression, when=None, timeout=5, *, deadline=None):
        return self.api(
            "query", timeout=timeout, query=expression, time=when or time.time(),
            **({"deadline": deadline} if deadline is not None else {})
        )["result"]

    def selector(self, name):
        if name not in self.target_kinds:
            raise ValueError("unknown monitored source " + name)
        return "{job=" + json.dumps(name) + "}"

    def instant(self, name, timeout=5):
        selector = self.selector(name)
        now = time.time()
        health = self.query("up" + selector, now, timeout)
        if len(health) != 1 or health[0]["value"][1] != "1":
            raise RuntimeError("Prometheus target unavailable: " + name)
        rows = self.query(f"{selector}[{round(self.max_gap * 1000)}ms]", now, timeout)
        if not any(r["metric"].get("__name__") == "up" for r in rows):
            raise RuntimeError("Prometheus target stale: " + name)
        latest_scrape = max(
            r["values"][-1][0] for r in rows if r["metric"].get("__name__") == "up"
        )
        return [
            dict(metric=r["metric"], value=r["values"][-1])
            for r in rows
            if r["values"][-1][0] == latest_scrape
            and not r["metric"]["__name__"].startswith(("up", "scrape_"))
        ]

    def metric_definition(self, metric_id):
        definition = definitions(self.query_plan).get(metric_id)
        if definition is None or "promql" not in definition:
            raise MetricContractError("undeclared Prometheus metric: " + metric_id)
        return definition

    def metric_rows(self, metric_id, *, source, start, end, deadline=None):
        """Read a declared raw metric with actual scrape times, never lookback filling."""
        definition = self.metric_definition(metric_id)
        if definition.get("mode") != "scrape":
            raise ValueError("raw gate input requires scrape mode: " + metric_id)
        if self.target_kind(source) != definition["source_kind"]:
            raise ValueError("metric source kind mismatch: " + metric_id)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("invalid metric query window")
        expression = definition["promql"].replace("${selector}", self.selector(source))
        rows, cursor = [], start
        while cursor < end:
            right = min(end, cursor + 60)
            rows.extend(self.query(expression + f"[{math.ceil((right - cursor) * 1000)}ms]", right,
                                   **({"deadline": deadline} if deadline is not None else {})))
            cursor = right
        for row in rows:
            if set(definition["labels"]) - set(row["metric"]):
                raise ValueError("metric lacks required labels: " + metric_id)
            row["metric_id"] = metric_id
        return rows

    def metric_snapshot(self, metric_ids, *, source, timeout=5):
        rows = self.instant(source, timeout)
        result = []
        for identity in metric_ids:
            definition = self.metric_definition(identity)
            if definition.get("mode") != "scrape" or self.target_kind(source) != definition["source_kind"]:
                raise ValueError("snapshot requires a raw metric with matching source kind")
            name = definition["promql"].split("${selector}")[0]
            for row in rows:
                if row["metric"].get("__name__") == name:
                    if set(definition["labels"]) - set(row["metric"]):
                        raise ValueError("metric lacks required labels: " + identity)
                    result.append(dict(row, metric_id=identity))
        return result

    def target_kind(self, source):
        if source not in self.target_kinds:
            raise ValueError("unknown monitored source: " + source)
        return self.target_kinds[source]
