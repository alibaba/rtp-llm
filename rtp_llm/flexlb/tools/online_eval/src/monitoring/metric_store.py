"""Frozen numeric observations shared by gates and report projections."""

import copy
import json
import math
from pathlib import Path

from schema_contract import matches_schema


class MetricContractError(ValueError):
    """An ID, identity or definition violates the declared metric contract."""


class MetricUnavailable(ValueError):
    """Declared data cannot establish a valid measurement."""


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
    temporary.replace(path)


class MetricStore:
    """Read a frozen artifact; never query a source or reinterpret current YAML."""

    def __init__(self, document):
        if not matches_schema(document, "metrics_schema_version", 1):
            raise MetricContractError("unsupported metric artifact schema")
        if (not isinstance(document.get("definitions"), dict)
                or not isinstance(document.get("metrics"), dict)
                or set(document["metrics"]) - set(document["definitions"])):
            raise MetricContractError("invalid metric artifact inventory")
        self.document = document

    @classmethod
    def read(cls, directory):
        return cls(json.loads((Path(directory) / "metrics.json").read_text()))

    def save(self, directory):
        atomic_json(Path(directory) / "metrics.json", self.document)

    def select(self, metric_id, *, labels=None, start=None, end=None,
               source=None, epoch=None, min_samples=1, max_gap_s=None):
        """Select without reducing labels, filling gaps, or changing timestamps."""
        if metric_id not in self.document["definitions"]:
            raise MetricContractError("undeclared metric id: " + metric_id)
        if self.document["definitions"][metric_id]["unit"] == "unspecified":
            raise MetricContractError("metric definition was not frozen: " + metric_id)
        if type(min_samples) is not int or min_samples < 1:
            raise MetricContractError("min_samples must be a positive integer")
        if (start is None) != (end is None) or (start is not None and (
                not math.isfinite(start) or not math.isfinite(end) or start >= end)):
            raise MetricContractError("selection requires an increasing finite window")
        if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s <= 0):
            raise MetricContractError("max_gap_s must be positive")
        selected = []
        for row in self.document["metrics"].get(metric_id, []):
            if source is not None and row["source"] != source:
                continue
            if epoch is not None and row["epoch"] != str(epoch):
                continue
            if any(row["labels"].get(k) != v for k, v in (labels or {}).items()):
                continue
            if row["provenance"].get("evidence", {}).get("measurement_validity") == "INVALID":
                raise MetricUnavailable(metric_id + ": measurement is invalid")
            if row["status"] == "ERROR":
                raise MetricUnavailable(metric_id + ": source query failed")
            points = [p for p in row["points"] if start is None or start <= p[0] <= end]
            if any(type(t) not in (int, float) or not math.isfinite(t)
                   or (v is not None and (type(v) not in (int, float) or not math.isfinite(v)))
                   for t, v in points):
                raise MetricUnavailable(metric_id + ": non-finite observation")
            valid = [p for p in points if p[1] is not None]
            if len(valid) < min_samples or len(valid) != len(points):
                raise MetricUnavailable(metric_id + ": missing/invalid samples")
            if max_gap_s is not None:
                if (any(b[0] - a[0] > max_gap_s for a, b in zip(valid, valid[1:]))
                        or (start is not None and (valid[0][0] - start > max_gap_s
                                                   or end - valid[-1][0] > max_gap_s))):
                    raise MetricUnavailable(metric_id + ": insufficient time coverage")
            selected.append(dict(row, points=points))
        if not selected:
            raise MetricUnavailable(metric_id + ": no matching series")
        return selected

    def reduce(self, metric_id, *, op, **selection):
        """Scalar reduction requires one selected series; fleet aggregation is explicit."""
        if op not in {"min", "max", "first", "last"}:
            raise MetricContractError("unsupported reduction: " + str(op))
        rows = self.select(metric_id, **selection)
        if len(rows) != 1:
            raise MetricContractError("scalar reduction requires exactly one series")
        values = [v for _, v in rows[0]["points"]]
        return {"min": min, "max": max, "first": lambda x: x[0],
                "last": lambda x: x[-1]}[op](values)

    def series(self, anchor):
        series, sources = {}, {}
        for observations in self.document["metrics"].values():
            for row in observations:
                key = row.get("series_key")
                if not row["points"]:
                    continue
                if key is None:
                    key = "/".join((str(row["epoch"]), row["source"], row["metric_id"],
                                    json.dumps(row["labels"], sort_keys=True)))
                if key in series:
                    raise MetricContractError("duplicate archived series key: " + key)
                series[key] = [[t - anchor, value] for t, value in row["points"]]
                sources[key] = dict(row["provenance"], metric_id=row["metric_id"],
                                    unit=self.document["definitions"][row["metric_id"]]["unit"])
        gaps = {k: [t - anchor for t in v] for k, v in self.document["collection_gaps"].items()}
        return series, sources, gaps, copy.deepcopy(self.document["errors"])


def export_metrics(directory, plan=None, *, archive_directory=None):
    """Materialize Prometheus query evidence into one run artifact, retaining producers.

    This is an explicit archive conversion, not an alternative acquisition source.
    No logs, debug snapshots or request journals can fill an absent query.
    """
    from monitoring.query_archive import series as archive_series
    from monitoring.query_plan import definitions, plan_hash

    directory = Path(directory)
    selected_plan = plan
    archive_root = Path(archive_directory) if archive_directory is not None else directory
    paths = sorted(archive_root.glob("telemetry/*/queries.json"))
    if (archive_root / "queries.json").is_file():
        paths.append(archive_root / "queries.json")
    archives = [(path, json.loads(path.read_text())) for path in paths]
    series, sources, gaps, errors = archive_series(archive_root, 0)
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
            kind = kinds.get(source, "mock" if source == "mock" else
                             "client" if source.startswith("client-") else "master")
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
        gaps = copy.deepcopy(previous["collection_gaps"])
        errors = copy.deepcopy(previous["errors"])
        plans = dict(previous.get("plans", {}), **plans)
    metadata = previous.get("run", {}) if destination.exists() else {}
    for identity in declared:
        observations.setdefault(identity, [])
    document = dict(metrics_schema_version=1, run=metadata, plans=plans, definitions=declared,
                    metrics=observations, collection_gaps=gaps, errors=errors)
    atomic_json(destination, document)
    return MetricStore(document)


def series_row(points, *, epoch, source, labels):
    """A numeric stream's identity; physical origin is in definition.source_type."""
    if type(epoch) is not int or epoch < 1:
        raise MetricContractError("produced series requires a positive environment epoch")
    if type(source) is not str or not source.strip() or not isinstance(labels, dict):
        raise MetricContractError("produced series requires source identity and labels")
    return dict(epoch=str(epoch), source=source, labels=dict(labels), points=points)


def publish(store, metric_id, definition, rows, *, producer, evidence):
    """A declared Python producer publishes numeric data with explicit provenance."""
    import hashlib
    from importlib.util import find_spec
    from monitoring.producers import PRODUCERS, output_contract

    declared = store.document["definitions"].get(metric_id)
    if declared is None or declared != definition or declared.get("producer") != producer or producer not in PRODUCERS:
        raise MetricContractError("undeclared producer or definition mismatch: " + metric_id)
    try:
        contract = output_contract(metric_id, definition)
        if any(contract.get(key) != definition.get(key) for key in ('measurement', 'collection')):
            raise MetricContractError('published measurement does not match its implementation: ' + metric_id)
    except ValueError as exc:
        raise MetricContractError(str(exc)) from exc
    calculation_module = ('analysis.request_metrics' if 'calculation' in definition
                          else definition['measurement']['method'].rsplit('.', 1)[0])
    calculation_sha256 = hashlib.sha256(Path(find_spec(calculation_module).origin).read_bytes()).hexdigest()
    module = PRODUCERS[producer][0]
    producer_sha256 = hashlib.sha256(Path(find_spec(module).origin).read_bytes()).hexdigest()
    result, identities = [], set()
    for row in rows:
        key = (str(row["epoch"]), row["source"], json.dumps(row["labels"], sort_keys=True))
        if key in identities:
            raise MetricContractError("duplicate produced series identity: " + metric_id)
        identities.add(key)
        if set(definition["labels"]) - set(row["labels"]):
            raise MetricContractError("produced metric lacks required labels: " + metric_id)
        points = row["points"]
        if any(type(t) not in (int, float) or not math.isfinite(t)
               or (v is not None and (type(v) not in (int, float) or not math.isfinite(v)))
               for t, v in points) or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise MetricContractError("invalid or unordered produced samples: " + metric_id)
        result.append(dict(metric_id=metric_id, epoch=str(row["epoch"]), source=row["source"],
            labels=row["labels"], points=points, status="PRESENT" if any(v is not None for _, v in points) else "ABSENT",
            provenance=dict(source_type=definition["source_type"], producer=producer,
                            producer_module=module, producer_sha256=producer_sha256,
                            calculation_module=calculation_module, calculation_sha256=calculation_sha256,
                            calculation=copy.deepcopy(definition.get("calculation")),
                            measurement=copy.deepcopy(definition["measurement"]), evidence=evidence)))
    store.document["metrics"][metric_id] = result
