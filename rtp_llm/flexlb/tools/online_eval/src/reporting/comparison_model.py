"""Typed frozen bundle inputs and stable measurement identities for comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Optional

from reporting.spec import validate


@dataclass(frozen=True)
class ComparisonControls:
    configuration: object
    workload: object
    environment: object
    criteria: object
    statistic_sources: object

    @classmethod
    def read(cls, metadata):
        return cls(*(copy.deepcopy(metadata.get(name)) for name in cls.__dataclass_fields__))

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ChartPoint:
    x: float | str
    y: float | None


@dataclass(frozen=True)
class ChartSeries:
    name: str
    metric_id: str | None
    labels: dict
    unit: str | None
    axis: str
    scale: float
    measurement: object
    calculation: object
    source_type: str | None
    points: tuple[ChartPoint, ...]

    @classmethod
    def read(cls, data, panel_unit):
        provenance = data.get('provenance') or {}
        return cls(data['name'], data.get('metric_id'), data.get('labels', {}),
            data.get('unit', panel_unit), data.get('axis', 'y'), data.get('scale', 1),
            data.get('measurement', provenance.get('measurement')), provenance.get('calculation'),
            data.get('source_type', provenance.get('source_type')),
            tuple(ChartPoint(point['x'], point['y']) for point in data['points']))

    def signature(self):
        if self.metric_id is None:
            return None
        identity = dict(metric_id=self.metric_id, labels=self.labels, measurement=self.measurement,
                        calculation=self.calculation, axis=self.axis, unit=self.unit,
                        scale=self.scale, source_type=self.source_type)
        return json.dumps(identity, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ChartPanel:
    identity: str
    title: str
    kind: str
    unit: str | None
    axes: dict
    time_x: bool
    series: tuple[ChartSeries, ...]
    document: dict = field(repr=False)

    @classmethod
    def read(cls, data):
        return cls(data['id'], data.get('title', data['id']), data.get('type', 'line'),
            data.get('unit'), data['axes'], data.get('timeX', False),
            tuple(ChartSeries.read(series, data.get('unit')) for series in data.get('series', [])),
            copy.deepcopy(data))

    def to_dict(self):
        return copy.deepcopy(self.document)

    def signature(self):
        identities = [series.signature() for series in self.series]
        if any(identity is None for identity in identities):
            return None
        return dict(type=self.kind, unit=self.unit, axes=self.axes, timeX=self.time_x,
                    series=sorted(identities))


@dataclass(frozen=True)
class FrozenReport:
    title: Optional[str]
    subtitle: Optional[str]
    time_origin_label: Optional[str]
    events: tuple[dict, ...]
    panels: tuple[ChartPanel, ...]
    sections: tuple[dict, ...]
    kpis: tuple[dict, ...]
    time_min: Optional[float]
    time_max: Optional[float]
    run_meta: dict
    controls: ComparisonControls

    @classmethod
    def read(cls, spec):
        validate(spec)
        metadata = copy.deepcopy(spec['run_meta'])
        if not isinstance(metadata.get('comparison_controls'), dict):
            raise ValueError('frozen report lacks comparison controls')
        axis = spec.get('timeAxis') or {}
        return cls(spec.get('title'), spec.get('subtitle'), spec.get('timeOriginLabel'),
                   tuple(copy.deepcopy(spec.get('events', []))), tuple(ChartPanel.read(panel) for panel in spec.get('panels', [])),
                   tuple(copy.deepcopy(spec.get('sections', []))), tuple(copy.deepcopy(spec.get('kpis', []))),
                   axis.get('min'), axis.get('max'), metadata,
                   ComparisonControls.read(metadata['comparison_controls']))


@dataclass(frozen=True)
class FrozenRun:
    directory: Path
    analysis: dict
    report: FrozenReport
    manifest_sha256: str


def measurement_signature(panel):
    return ChartPanel.read(panel).signature()
