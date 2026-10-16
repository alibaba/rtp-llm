"""TSDB query operations; no exporter reads or process lifecycle."""

import json
import math
import re
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
            "query", timeout=timeout, query=expression, time=time.time() if when is None else when,
            **({"deadline": deadline} if deadline is not None else {})
        )["result"]

    def selector(self, name, *, metric_names=None, labels=None):
        if name not in self.target_kinds:
            raise ValueError("unknown monitored source " + name)
        matchers = ["job=" + json.dumps(name)]
        if metric_names is not None:
            names = sorted(set(metric_names))
            if not names:
                raise ValueError("snapshot requires at least one metric")
            matchers.append('__name__=~' + json.dumps("|".join(re.escape(v) for v in names)))
        for key, value in sorted((labels or {}).items()):
            if key in {"job", "__name__"} or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key):
                raise ValueError("invalid snapshot label: " + key)
            if not isinstance(value, str):
                raise ValueError("snapshot label value must be a string")
            matchers.append(key + "=" + json.dumps(value))
        return "{" + ",".join(matchers) + "}"

    def instant(self, name, timeout=5, *, metric_names=None, labels=None):
        selector = self.selector(name)
        selected = self.selector(name, metric_names=metric_names, labels=labels)
        now = time.time()
        window = f"[{round(self.max_gap * 1000)}ms]"
        # Health and values must come from the same actual scrape. An instant
        # vector can look back across a gap and turn stale data into a sample.
        health = self.query("up" + selector + window, now, timeout)
        if len(health) != 1 or not health[0].get("values"):
            raise RuntimeError("Prometheus target stale: " + name)
        latest_scrape, up = health[0]["values"][-1]
        if up != "1":
            raise RuntimeError("Prometheus target unavailable: " + name)
        rows = self.query(selected + window, latest_scrape, timeout)
        return [
            dict(metric=r["metric"], value=r["values"][-1])
            for r in rows
            if r.get("values") and r["values"][-1][0] == latest_scrape
            and not r["metric"]["__name__"].startswith(("up", "scrape_"))
        ]

    def metric_definition(self, metric_id):
        definition = definitions(self.query_plan).get(metric_id)
        if definition is None or "promql" not in definition:
            raise MetricContractError("undeclared Prometheus metric: " + metric_id)
        return definition

    def metric_rows(self, metric_id, *, source, start, end, deadline=None):
        """Read a declared raw metric with actual scrape times, never lookback filling."""
        return [row for chunk in self.metric_chunks(metric_id, source=source, start=start,
                                                   end=end, deadline=deadline) for row in chunk]

    def metric_chunks(self, metric_id, *, source, start, end, deadline=None, labels=None):
        """Yield validated chunks so archives retain evidence if a later query fails."""
        definition = self.metric_definition(metric_id)
        if definition.get("mode") != "scrape":
            raise ValueError("raw gate input requires scrape mode: " + metric_id)
        if self.target_kind(source) != definition["source_kind"]:
            raise ValueError("metric source kind mismatch: " + metric_id)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("invalid metric query window")
        expression = definition["promql"].replace("${selector}", self.selector(source, labels=labels))
        cursor = start
        while cursor < end:
            if deadline is not None:
                deadline.check()
            right = min(end, cursor + 60)
            rows = self.query(expression + f"[{math.ceil((right - cursor) * 1000)}ms]", right,
                              **({"deadline": deadline} if deadline is not None else {}))
            for row in rows:
                if set(definition["labels"]) - set(row["metric"]):
                    raise ValueError("metric lacks required labels: " + metric_id)
                row["metric_id"] = metric_id
            yield rows
            cursor = right

    def metric_snapshot(self, metric_ids, *, source, timeout=5, labels=None):
        selected = {}
        for identity in metric_ids:
            definition = self.metric_definition(identity)
            if definition.get("mode") != "scrape" or self.target_kind(source) != definition["source_kind"]:
                raise ValueError("snapshot requires a raw metric with matching source kind")
            name = definition["promql"].split("${selector}")[0]
            selected.setdefault(name, []).append((identity, definition))
        rows = self.instant(source, timeout, metric_names=selected, labels=labels)
        result = []
        for row in rows:
            for identity, definition in selected.get(row["metric"].get("__name__"), ()):
                if set(definition["labels"]) - set(row["metric"]):
                    raise ValueError("metric lacks required labels: " + identity)
                result.append(dict(row, metric_id=identity))
        return result

    def target_kind(self, source):
        if source not in self.target_kinds:
            raise ValueError("unknown monitored source: " + source)
        return self.target_kinds[source]
