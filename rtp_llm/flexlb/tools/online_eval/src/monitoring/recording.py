"""Record a bounded observation window from a query client into raw evidence."""

from dataclasses import dataclass
from pathlib import Path
import time

from artifacts.json_io import write_json
from monitoring.query_plan import queries_for_targets, plan_hash


@dataclass(frozen=True)
class ArchiveWindow:
    directory: Path
    query_plan_name: str
    collection_plan: dict
    started: float
    interval: float
    targets: dict
    target_bounds: dict


def record_queries(window, client, *, deadline=None):
    end = time.time()

    queries, required_queries = queries_for_targets(
        client.query_plan, window.targets, client.selector, window.interval,
        client.target_kinds,
    )
    result = dict(
        missing_queries=[],
        query_plan=window.query_plan_name,
        query_plan_sha256=plan_hash(client.query_plan),
        backend="prometheus",
        metric_plan=client.query_plan,
        collection_plan=window.collection_plan,
        target_kinds={source: client.target_kind(source) for source in window.targets},
        start=window.started,
        end=end,
        step=window.interval,
        targets=window.targets,
        target_bounds=window.target_bounds,
        queries={},
        errors=[],
    )
    for key, expression in queries.items():
        rows = []
        # API query_range has a maximum point count; chunk long experiments.
        start, target_end = window.target_bounds.get(
            key.split("/", 1)[0], [window.started, end]
        )
        query_end = target_end or end
        cursor = start
        try:
            while cursor <= query_end:
                if deadline is not None:
                    deadline.check()
                right = min(query_end, cursor + window.interval * 1000)
                source, metric = key.split("/", 1)
                definition = client.query_plan["sources"][client.target_kind(source)].get(metric, {})
                if definition.get("mode") == "scrape":
                    if right <= cursor:
                        break
                    rows.extend(client.metric_rows(client.target_kind(source) + "/" + metric,
                                                 source=source, start=cursor, end=right, deadline=deadline))
                else:
                    data = client.api("query_range", query=expression, start=cursor,
                                    end=right, step=window.interval,
                                       **({"deadline": deadline} if deadline is not None else {}))
                    rows.extend(data["result"])
                cursor = right if definition.get("mode") == "scrape" and right < query_end else right + window.interval
            result["queries"][key] = dict(promql=expression, result=rows)
            if not rows:
                result["missing_queries"].append(key)
                if key in required_queries:
                    result["errors"].append(
                        dict(query=key, error="required monitor series absent")
                    )
        except Exception as exc:
            result["queries"][key] = dict(promql=expression, result=rows)
            result["errors"].append(dict(query=key, error=str(exc)))
    write_json(window.directory / "queries.json", result)
    return result
