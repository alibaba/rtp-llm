"""Frozen numeric observations shared by gates and report projections."""

from contextlib import contextmanager
from contextvars import ContextVar

import copy
import json
import math
from pathlib import Path

from schema_contract import matches_schema


_READ_SESSION = ContextVar('frozen_metric_reads', default=None)


@contextmanager
def frozen_metric_reads():
    """Reuse only unchanged artifacts within a report publication, never across runs."""
    token = _READ_SESSION.set({})
    try:
        yield
    finally:
        _READ_SESSION.reset(token)


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
        if (not isinstance(document.get("collection_gaps", {}), dict)
                or not isinstance(document.get("errors", []), list)):
            raise MetricContractError("invalid metric artifact diagnostics")
        for metric_id, observations in document["metrics"].items():
            if not isinstance(observations, list):
                raise MetricContractError("metric observations must be a list")
            for row in observations:
                if (not isinstance(row, dict)
                        or not isinstance(row.get("labels"), dict)
                        or not isinstance(row.get("provenance"), dict)
                        or not isinstance(row["provenance"].get("evidence", {}), dict)):
                    raise MetricContractError("invalid metric observation metadata")
                if (row.get("metric_id") != metric_id
                        or any(type(row.get(key)) is not str or not row[key].strip()
                               for key in ("source", "epoch", "status"))
                        or not isinstance(row.get("points"), list)):
                    raise MetricContractError("invalid metric observation identity or fields")
        self.document = document

    @classmethod
    def read(cls, directory):
        path = (Path(directory) / "metrics.json").resolve()
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cache = _READ_SESSION.get()
        if cache is not None and path in cache and cache[path][0] == signature:
            return cache[path][1]
        with path.open(encoding='utf-8') as stream:
            store = cls(json.load(stream))
        if cache is not None:
            cache[path] = (signature, store)
        return store

    def save(self, directory):
        atomic_json(Path(directory) / "metrics.json", self.document)

    def select(self, metric_id, *, labels=None, start=None, end=None,
               source=None, epoch=None, min_samples=1, max_gap_s=None):
        """Select without filling gaps; max_gap_s bounds coverage, not failed samples."""
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
            points = [p for p in _ordered_points(row) if start is None or start <= p[0] <= end]
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
                series[key] = [[t - anchor, value] for t, value in _ordered_points(row)]
                sources[key] = dict(row["provenance"], metric_id=row["metric_id"],
                                    unit=self.document["definitions"][row["metric_id"]]["unit"])
        gaps = {k: [t - anchor for t in v] for k, v in self.document.get("collection_gaps", {}).items()}
        return series, sources, gaps, copy.deepcopy(self.document.get("errors", []))


def _ordered_points(row):
    previous = None
    for point in row["points"]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise MetricUnavailable(row["metric_id"] + ": malformed observation")
        timestamp = point[0]
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise MetricUnavailable(row["metric_id"] + ": non-finite observation timestamp")
        if previous is not None and timestamp <= previous:
            raise MetricUnavailable(row["metric_id"] + ": non-increasing observation timestamps")
        previous = timestamp
        yield point


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
    from monitoring.producer_contract import output_contract
    from case_api.registry import registry

    declared = store.document["definitions"].get(metric_id)
    if declared is None or declared != definition or declared.get("producer") != producer or producer not in registry().producers:
        raise MetricContractError("undeclared producer or definition mismatch: " + metric_id)
    try:
        contract = output_contract(metric_id, definition)
    except MetricContractError:
        raise
    except ValueError as exc:
        raise MetricContractError(str(exc)) from exc
    if any(contract.get(key) != definition.get(key) for key in ('measurement', 'collection', 'implementation')):
        raise MetricContractError('published measurement does not match its implementation: ' + metric_id)
    calculation_module = contract['implementation']['module']
    def module_digest(name):
        try:
            spec = find_spec(name)
            if spec is None or spec.origin is None:
                raise MetricContractError("producer module has no source file: " + name)
            return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        except MetricContractError:
            raise
        except (ImportError, ValueError, OSError) as exc:
            raise MetricContractError("cannot read producer module: " + name) from exc

    calculation_sha256 = module_digest(calculation_module)
    module = registry().producers[producer].module
    producer_sha256 = module_digest(module)
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
