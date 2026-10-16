"""Bounded raw window capture; consumers read the same frozen MetricStore."""
from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path

from artifacts.json_io import write_json
from file_digest import sha256_file
from monitoring.contracts import MetricUnavailable
from monitoring.metric_store import MetricStore
from runtime.observation import capture_limits


@dataclass(frozen=True)
class FrozenMetricWindow:
    encoded: str

    def store(self):
        document = json.loads(self.encoded)
        if document["capture"]["status"] != "COMPLETE":
            raise MetricUnavailable("raw window capture is incomplete")
        return MetricStore(document)

    def field_rows(self):
        """Project declared fields without I/O, filling or cross-source joins."""
        store = self.store()
        return {field: [dict(metric_id=binding["metric"], metric=row["labels"], values=row["points"])
                for row in store.document["metrics"].get(binding["metric"], [])
                if all(row["labels"].get(k) == v for k, v in binding["labels"].items())]
                for field, binding in store.document["capture"]["fields"].items()}

    @classmethod
    def read(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("metric_window_schema_version") != 1 or set(manifest.get("files", {})) != {"metrics.json"}:
            raise ValueError("invalid raw window manifest")
        if sha256_file(directory / "metrics.json") != manifest["files"]["metrics.json"]:
            raise ValueError("raw window evidence digest mismatch")
        result = cls((directory / "metrics.json").read_text())
        result.store()
        return result


def capture_window(client, fields, *, source, windows, epoch, deadline, limits, directory=None):
    """Freeze declared (start, end] windows under one caller-owned deadline.

    max_samples counts distinct raw timestamps across the selected series;
    max_bytes bounds the retained encoded artifact, including identities.
    Missing fields/series are preserved for consumer-specific completeness
    checks. Query failures never fall back to live snapshots or another source.
    """
    deadline.check()
    limits = capture_limits(limits, "capture")
    if not fields or not windows or not isinstance(source, str) or not source or not str(epoch):
        raise ValueError("raw window requires fields, source, epoch and windows")
    fields = copy.deepcopy(fields)
    for field, binding in fields.items():
        if (not isinstance(field, str) or not field or set(binding) != {"metric", "labels"}
                or not isinstance(binding["metric"], str) or not isinstance(binding["labels"], dict)):
            raise ValueError("invalid raw field binding")
    windows = [list(w) for w in windows]
    if any(len(w) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in w)
           or w[0] >= w[1] for w in windows):
        raise ValueError("invalid raw measurement window")
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
    definitions = {b["metric"]: copy.deepcopy(client.metric_definition(b["metric"])) for b in fields.values()}
    document = dict(metrics_schema_version=1, definitions=definitions,
        metrics={identity: [] for identity in definitions}, errors=[], collection_gaps={},
        capture=dict(status="INCOMPLETE", fields=fields, source=source, epoch=str(epoch),
                     windows=windows, boundary="(start,end]", limits=limits))
    retained = len(json.dumps(document, separators=(",", ":"), allow_nan=False).encode())
    timestamps, series, seen = set(), {}, {}

    def reserve(size, timestamp=None):
        nonlocal retained
        if timestamp is not None and timestamp not in timestamps:
            if len(timestamps) >= limits["max_samples"]:
                raise ValueError("raw window sample budget exceeded")
            timestamps.add(timestamp)
        if retained + size > limits["max_bytes"]:
            raise ValueError("raw window byte budget exceeded")
        retained += size

    def freeze():
        encoded = json.dumps(document, separators=(",", ":"), allow_nan=False)
        if directory is not None:
            write_json(directory / "metrics.json", document)
            write_json(directory / "manifest.json", dict(metric_window_schema_version=1,
                files={"metrics.json": sha256_file(directory / "metrics.json")}))
        return FrozenMetricWindow(encoded)

    try:
        reserve(0)
        requests = {(b["metric"], json.dumps(b["labels"], sort_keys=True)) for b in fields.values()}
        for identity, selector in sorted(requests):
            labels = json.loads(selector)
            for start, end in windows:
                for chunk in client.metric_chunks(identity, source=source, start=start, end=end,
                                                   deadline=deadline, labels=labels):
                    deadline.check()
                    for raw in chunk:
                        raw_labels = dict(raw["metric"])
                        if any(raw_labels.get(k) != v for k, v in labels.items()):
                            raise ValueError("raw metric does not match selected labels")
                        if set(definitions[identity].get("labels", [])) - set(raw_labels):
                            raise ValueError("raw metric lacks identity labels")
                        key = (identity, json.dumps(raw_labels, sort_keys=True))
                        for timestamp, value in raw["values"]:
                            deadline.check()
                            value = float(value)
                            if not math.isfinite(timestamp) or not math.isfinite(value):
                                raise ValueError("non-finite raw measurement")
                            if not start < timestamp <= end:
                                continue
                            point_key = (*key, timestamp)
                            if point_key in seen:
                                if seen[point_key] != value:
                                    raise ValueError("conflicting duplicate raw measurement")
                                continue
                            if key not in series:
                                row = dict(metric_id=identity, source=source, epoch=str(epoch),
                                    labels=raw_labels, points=[], status="PRESENT",
                                    provenance=dict(source_type="prometheus", timestamp_kind="scrape",
                                                    measurement=definitions[identity].get("measurement", {})))
                                reserve(len(json.dumps(row, separators=(",", ":")).encode()) + 1)
                                series[key] = row
                                document["metrics"][identity].append(row)
                            point = [timestamp, value]
                            reserve(len(json.dumps(point, separators=(",", ":")).encode()) + 1, timestamp)
                            series[key]["points"].append(point)
                            seen[point_key] = value
        if not timestamps:
            raise MetricUnavailable("raw measurement samples are missing")
        for row in series.values():
            row["points"].sort()
        deadline.check()
        document["capture"]["status"] = "COMPLETE"
        return freeze()
    except BaseException as error:
        document["capture"]["status"] = "INCOMPLETE"
        document["errors"].append(dict(source=source, error=str(error)))
        for row in series.values():
            row["status"] = "ERROR"
        freeze()
        raise
