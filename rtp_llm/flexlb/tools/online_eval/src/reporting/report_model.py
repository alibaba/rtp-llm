"""Shared frozen experiment conditions and measurement identity, without point arrays."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import json
from typing import Mapping
from reporting.values import freeze, thaw


@dataclass(frozen=True)
class ExperimentConditions:
    configuration: object
    workload: object
    environment: object
    criteria: object
    statistic_sources: object

    def __post_init__(self):
        for name in ("configuration", "workload", "environment", "criteria", "statistic_sources"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @classmethod
    def read(cls, metadata):
        return cls(configuration=copy.deepcopy(metadata.get('configuration')),
                   workload=copy.deepcopy(metadata.get('workload')),
                   environment=copy.deepcopy(metadata.get('environment')),
                   criteria=copy.deepcopy(metadata.get('criteria')),
                   statistic_sources=copy.deepcopy(metadata.get('statistic_sources')))

    def to_dict(self):
        return thaw(dict(configuration=self.configuration, workload=self.workload,
                    environment=self.environment, criteria=self.criteria,
                    statistic_sources=self.statistic_sources))


@dataclass(frozen=True)
class MeasurementIdentity:
    metric_id: str | None
    labels: Mapping
    measurement: object
    calculation: object
    axis: str
    unit: str | None
    scale: float
    source_type: str | None

    def __post_init__(self):
        for name in ("labels", "measurement", "calculation"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @classmethod
    def read(cls, data, panel_unit=None):
        provenance = data.get('provenance') or {}
        return cls(data.get('metric_id'), data.get('labels', {}),
                   data.get('measurement', provenance.get('measurement')),
                   provenance.get('calculation'), data.get('axis', 'y'),
                   data.get('unit', panel_unit), data.get('scale', 1),
                   data.get('source_type', provenance.get('source_type')))

    def signature(self):
        if self.metric_id is None:
            return None
        return json.dumps(thaw(dict(metric_id=self.metric_id, labels=self.labels,
            measurement=self.measurement, calculation=self.calculation, axis=self.axis,
            unit=self.unit, scale=self.scale, source_type=self.source_type)),
            sort_keys=True, allow_nan=False)


def panel_document(*, identity, title, axes, series, caption, presets=None):
    """Build the same final panel contract for default and selected producers."""
    if not axes or any(curve.get('axis', 'y') not in axes for curve in series):
        raise ValueError('panel metric axis is not declared')
    result = dict(id=identity, title=title, timeX=True, axes=axes, series=series, caption=caption)
    if presets is not None:
        result['presets'] = presets
    return result

def compare_controls(left, right, *, required=(), allowed=()):
    """Compare all control leaves, except explicitly varied experiment subtrees.

    required/allowed paths use JSON Pointer syntax. Missing required evidence is
    unknown even when both sides omit it. Unknown added controls are compared.
    """
    missing = object()
    differences = []

    def at(value, pointer):
        for part in pointer.strip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict) or key not in value:
                return missing
            value = value[key]
        return value

    def walk(a, b, path):
        if any(path == p or path.startswith(p + "/") for p in allowed):
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(
                    a.get(key, missing),
                    b.get(key, missing),
                    path + "/" + key.replace("~", "~0").replace("/", "~1"),
                )
        elif a is missing or b is missing or a != b:
            differences.append(
                dict(
                    path=path,
                    status="MISSING" if a is missing or b is missing else "DIFFERENT",
                    baseline=None if a is missing else a,
                    candidate=None if b is missing else b,
                )
            )

    walk(left, right, "")
    absent = [
        p
        for p in required
        if any(at(v, p) is missing or at(v, p) is None for v in (left, right))
    ]
    return dict(
        status="UNKNOWN" if absent else "DIFFERENT" if differences else "ALIGNED",
        aligned=not absent and not differences,
        missing=absent,
        differences=differences,
        allowed=list(allowed),
    )
