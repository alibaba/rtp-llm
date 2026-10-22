"""Typed frozen bundle inputs and stable measurement identities for comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Optional

from reporting.spec import validate
from reporting.report_model import ExperimentConditions, MeasurementIdentity


@dataclass(frozen=True)
class ChartPanel:
    identity: str
    title: str
    kind: str
    unit: str | None
    axes: dict
    time_x: bool
    series: tuple[MeasurementIdentity, ...]
    document: dict = field(repr=False)

    @classmethod
    def read(cls, data):
        return cls(data['id'], data.get('title', data['id']), data.get('type', 'line'),
            data.get('unit'), data['axes'], data.get('timeX', False),
            tuple(MeasurementIdentity.read(series, data.get('unit')) for series in data.get('series', [])),
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
    controls: ExperimentConditions

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
                   ExperimentConditions.read(metadata['comparison_controls']))


def measurement_signature(panel):
    return ChartPanel.read(panel).signature()
