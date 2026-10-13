"""Shared frozen experiment conditions and measurement identity, without point arrays."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import json


@dataclass(frozen=True)
class ExperimentConditions:
    configuration: object
    workload: object
    environment: object
    criteria: object
    statistic_sources: object

    @classmethod
    def read(cls, metadata):
        return cls(configuration=copy.deepcopy(metadata.get('configuration')),
                   workload=copy.deepcopy(metadata.get('workload')),
                   environment=copy.deepcopy(metadata.get('environment')),
                   criteria=copy.deepcopy(metadata.get('criteria')),
                   statistic_sources=copy.deepcopy(metadata.get('statistic_sources')))

    def to_dict(self):
        return dict(configuration=self.configuration, workload=self.workload,
                    environment=self.environment, criteria=self.criteria,
                    statistic_sources=self.statistic_sources)


@dataclass(frozen=True)
class MeasurementIdentity:
    metric_id: str | None
    labels: dict
    measurement: object
    calculation: object
    axis: str
    unit: str | None
    scale: float
    source_type: str | None

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
        return json.dumps(dict(metric_id=self.metric_id, labels=self.labels,
            measurement=self.measurement, calculation=self.calculation, axis=self.axis,
            unit=self.unit, scale=self.scale, source_type=self.source_type),
            sort_keys=True, allow_nan=False)


def panel_document(*, identity, title, axes, series, caption, presets=None):
    """Build the same final panel contract for default and selected producers."""
    if not axes or any(curve.get('axis', 'y') not in axes for curve in series):
        raise ValueError('panel metric axis is not declared')
    result = dict(id=identity, title=title, timeX=True, axes=axes, series=series, caption=caption)
    if presets is not None:
        result['presets'] = presets
    return result
