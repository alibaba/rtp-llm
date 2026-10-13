"""Convert explicit Prometheus query archives without acquiring other sources."""

import json
import math
import time
from pathlib import Path
import copy

from monitoring.identity import SOURCE_KINDS
from monitoring.metric_store import MetricStore, MetricContractError, atomic_json
from monitoring.query_plan import definitions, plan_hash, queries_for_targets


def _finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def _archive_series(directory, anchor, *, archives=None):
    """Read only monitor query archives; return curves, provenance and health gaps."""
    series, sources, gaps, errors = {}, {}, {}, []
    if archives is None:
        paths = sorted(Path(directory).glob("telemetry/*/queries.json"))
        if (Path(directory) / "queries.json").is_file():
            paths.append(Path(directory) / "queries.json")
        archives = ((path, json.loads(path.read_text())) for path in paths)
    for path, data in archives:
        epoch = path.parent.name if path.parent != Path(directory) else "1"
        errors.extend(dict(source=epoch, **error) for error in data.get("errors", []))
        errors.extend(
            dict(source=epoch, query=query, error="monitor series absent",
                 severity="diagnostic")
            for query in data.get("missing_queries", [])
        )
        for query_id, query in data["queries"].items():
            source, metric = query_id.split("/", 1)
            for row in query["result"]:
                labels = dict(row["metric"])
                labels.pop("__name__", None)
                name = metric
                for label in ("job", "instance"):
                    labels.pop(label, None)
                key = f"{epoch}/{source}/{name}/" + json.dumps(labels, sort_keys=True)
                points = [[float(t) - anchor, _finite(v)] for t, v in row["values"]]
                series.setdefault(key, []).extend(points)
                sources[key] = dict(
                    path=str(path),
                    promql=query["promql"],
                    start=data["start"],
                    end=data["end"],
                    step=data["step"],
                    backend="prometheus",
                )
                if metric == "up":
                    gaps.setdefault(f"{epoch}/{source}/collection", []).extend(
                        t for t, value in points if value != 1
                    )
        for source in data["targets"]:
            query = data["queries"].get(source + "/up", {})
            points = sorted(
                (float(t), _finite(v))
                for row in query.get("result", [])
                for t, v in row["values"]
            )
            if not points:
                errors.append(
                    dict(source=f"{epoch}/{source}", error="no Prometheus up samples")
                )
            elif (
                points[0][0]
                - data.get("target_bounds", {}).get(source, [data["start"], None])[0]
                > 2 * data["step"]
                or (
                    data.get("target_bounds", {}).get(source, [None, None])[1]
                    or data["end"]
                )
                - points[-1][0]
                > 2 * data["step"]
            ):
                errors.append(
                    dict(
                        source=f"{epoch}/{source}",
                        error="incomplete monitoring coverage",
                    )
                )
    # A failed scrape is a gap in every associated curve; never bridge it.
    for key, points in series.items():
        source = "/".join(key.split("/")[:2])
        failed = set(gaps.get(source + "/collection", []))
        merged = {}
        for t, value in points:
            if not math.isfinite(t):
                raise ValueError("non-finite monitor timestamp")
            if t in merged and merged[t] != value:
                raise ValueError("conflicting monitor samples at one timestamp: " + key)
            merged[t] = value
        for t in failed:
            merged[t] = None
        series[key] = sorted([t, value] for t, value in merged.items())
    return series, sources, {k: v for k, v in gaps.items() if v}, errors


def _legacy_source_kind(name):
    """Read historical archives whose target kinds were not frozen explicitly."""
    return "mock" if name == "mock" else "client" if name.startswith("client-") else "master"


def export_metrics(directory, plan=None, *, archive_directory=None):
    """Materialize Prometheus query evidence into one run artifact, retaining producers.

    This is an explicit archive conversion, not an alternative acquisition source.
    No logs, debug snapshots or request journals can fill an absent query.
    """

    directory = Path(directory)
    selected_plan = plan
    archive_root = Path(archive_directory) if archive_directory is not None else directory
    paths = sorted(archive_root.glob("telemetry/*/queries.json"))
    if (archive_root / "queries.json").is_file():
        paths.append(archive_root / "queries.json")
    archives = [(path, json.loads(path.read_text())) for path in paths]
    series, sources, gaps, errors = _archive_series(archive_root, 0, archives=archives)
    plans, declared, observations = {}, {}, {}
    for path, data in archives:
        epoch = path.parent.name if path.parent != archive_root else "1"
        archive_plan = data.get("metric_plan")
        frozen = definitions(archive_plan or selected_plan) if archive_plan or selected_plan else {}
        if archive_plan:
            plans[epoch] = dict(name=data["query_plan"], sha256=plan_hash(archive_plan), definition=archive_plan)
        kinds = data.get("target_kinds", {})
        for query_id, query in data["queries"].items():
            source, metric = query_id.split("/", 1)
            kind = kinds[source] if source in kinds else _legacy_source_kind(source)
            if kind not in SOURCE_KINDS:
                raise MetricContractError("invalid archived source kind: " + str(kind))
            identity = kind + "/" + metric
            spec = frozen.get(identity, dict(source_type="prometheus", source_kind=kind,
                unit="unspecified", value_kind="gauge", labels=[], promql=query["promql"]))
            if identity in declared and declared[identity] != spec:
                raise MetricContractError("conflicting frozen metric definitions: " + identity)
            declared[identity] = spec
            result = query.get("result", [])
            if not result:
                observations.setdefault(identity, []).append(dict(metric_id=identity,
                    epoch=epoch, source=source, labels={}, points=[], status="ERROR" if any(
                        e.get("query") == query_id for e in data.get("errors", [])) else "ABSENT",
                    provenance=dict(source_type="prometheus", path=str(path), promql=query["promql"])))
            seen = set()
            for raw in result:
                labels = dict(raw["metric"])
                labels.pop("__name__", None)
                name = metric
                for label in ("job", "instance"):
                    labels.pop(label, None)
                key = f"{epoch}/{source}/{name}/" + json.dumps(labels, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                points = series.get(key, [])
                provenance = dict(sources.get(key, {}), source_type="prometheus",
                                  timestamp_kind="scrape" if spec.get("mode") == "scrape" else "evaluation",
                                  measurement=copy.deepcopy(spec.get("measurement", {})))
                observations.setdefault(identity, []).append(dict(metric_id=identity,
                    epoch=epoch, source=source, labels=labels, points=points, series_key=key,
                    status="ERROR" if any(e.get("query") == query_id for e in data.get("errors", []))
                           else "PRESENT" if any(v is not None for _, v in points) else "ABSENT",
                    provenance=provenance))
        for identity, spec in frozen.items():
            if identity in declared and declared[identity] != spec:
                raise MetricContractError("conflicting frozen metric definitions: " + identity)
            declared[identity] = spec
    if plan is not None:
        for identity, spec in definitions(plan).items():
            if identity in declared and declared[identity] != spec:
                raise MetricContractError("conflicting selected metric plan: " + identity)
            declared[identity] = spec
        plans["selected"] = dict(sha256=plan_hash(plan), definition=plan)
    destination = directory / "metrics.json"
    if destination.exists():
        previous = MetricStore.read(directory).document
        for identity, spec in previous["definitions"].items():
            if "producer" in spec or not archives:
                if identity in declared and declared[identity] != spec:
                    raise MetricContractError("conflicting producer definition: " + identity)
                declared[identity] = spec
                observations[identity] = previous["metrics"].get(identity, [])
    if destination.exists() and not archives:
        # A metrics-only frozen archive still carries collection validity.
        # Retaining its observations must retain gaps/errors and plan identity too.
        gaps = copy.deepcopy(previous.get("collection_gaps", {}))
        errors = copy.deepcopy(previous.get("errors", []))
        plans = dict(previous.get("plans", {}), **plans)
    metadata = previous.get("run", {}) if destination.exists() else {}
    for identity in declared:
        observations.setdefault(identity, [])
    document = dict(metrics_schema_version=1, run=metadata, plans=plans, definitions=declared,
                    metrics=observations, collection_gaps=gaps, errors=errors)
    store = MetricStore(document)
    atomic_json(destination, store.document)
    return store


def archive_session(session, *, materialize=True, deadline=None):
    end = time.time()

    queries, required_queries = queries_for_targets(
        session.query_plan, session.targets, session.selector, session.interval,
        session.target_kinds,
    )
    result = dict(
        missing_queries=[],
        query_plan=session.query_plan_name,
        query_plan_sha256=session.query_plan_sha256,
        backend="prometheus",
        metric_plan=session.query_plan,
        collection_plan=session.collection_plan,
        target_kinds={source: session.target_kind(source) for source in session.targets},
        start=session.started,
        end=end,
        step=session.interval,
        targets=session.targets,
        target_bounds=session.target_bounds,
        queries={},
        errors=[],
    )
    for key, expression in queries.items():
        rows = []
        # API query_range has a maximum point count; chunk long experiments.
        start, target_end = session.target_bounds.get(
            key.split("/", 1)[0], [session.started, end]
        )
        query_end = target_end or end
        cursor = start
        try:
            while cursor <= query_end:
                if deadline is not None:
                    deadline.check()
                right = min(query_end, cursor + session.interval * 1000)
                source, metric = key.split("/", 1)
                definition = session.query_plan["sources"][session.target_kind(source)].get(metric, {})
                if definition.get("mode") == "scrape":
                    if right <= cursor:
                        break
                    rows.extend(session.metric_rows(session.target_kind(source) + "/" + metric,
                                                 source=source, start=cursor, end=right, deadline=deadline))
                else:
                    data = session.api("query_range", query=expression, start=cursor,
                                    end=right, step=session.interval,
                                       **({"deadline": deadline} if deadline is not None else {}))
                    rows.extend(data["result"])
                cursor = right if definition.get("mode") == "scrape" and right < query_end else right + session.interval
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
    atomic_json(session.directory / "queries.json", result)
    root = session.directory.parent.parent if session.directory.parent.name == "telemetry" else session.directory
    if materialize:
        if deadline is not None:
            deadline.check()
        export_metrics(root)
        if deadline is not None:
            deadline.check()
    if result["errors"]:
        if deadline is not None:
            deadline.check()
        raise RuntimeError("incomplete Prometheus archive")
