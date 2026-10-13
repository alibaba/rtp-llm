"""Presentation value objects; mappings are decoded only at the view boundary."""
from __future__ import annotations

from dataclasses import dataclass, fields, field
from enum import Enum
import math
import copy
import re
import string


def mapping(value, required, allowed, message):
    if not isinstance(value, dict) or not required <= value.keys() or value.keys()-allowed:
        raise ValueError(message)
    return value


def text(value):
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class SectionPresentation:
    title: str
    opened: bool
    columns: tuple[str, ...] | None = None

    @classmethod
    def read(cls, value):
        mapping(value, {'title', 'opened'}, {'title', 'opened', 'columns'}, 'invalid section fields')
        if not text(value['title']) or type(value['opened']) is not bool:
            raise ValueError('invalid section fields')
        columns = value.get('columns')
        if 'columns' in value and (not isinstance(columns, list) or not columns or not all(map(text, columns))):
            raise ValueError('invalid section columns')
        return cls(value['title'], value['opened'], tuple(columns) if columns is not None else None)

    def to_dict(self):
        result = dict(title=self.title, opened=self.opened)
        if self.columns is not None:
            result['columns'] = list(self.columns)
        return result


@dataclass(frozen=True)
class SubtitleTemplate:
    template: str
    cache_behavior: bool = False

    def __post_init__(self):
        if not text(self.template):
            raise ValueError('subtitle is required')
        allowed = {'verdict', 'monitoring_status'}
        if self.cache_behavior:
            allowed.update({'expected_behavior', 'observed_behavior'})
        for _, field, spec, conversion in string.Formatter().parse(self.template):
            if field is not None and field not in allowed:
                raise ValueError('unknown subtitle placeholder')
            if field is not None and (spec or conversion):
                raise ValueError('subtitle placeholders do not accept format specs or conversions')

    def render(self, *, verdict, monitoring_status=None, expected_behavior=None, observed_behavior=None):
        return self.template.format(verdict=verdict, monitoring_status=monitoring_status,
                                    expected_behavior=expected_behavior, observed_behavior=observed_behavior)


@dataclass(frozen=True)
class CurvePresentation:
    name: str
    group: str
    axis: str
    source_unit: str | None = None
    unit: str | None = None
    color: str | None = None
    hidden: bool = False
    scale: float = 1
    metric_id: str | None = None
    labels: tuple[tuple[str, str], ...] | None = None

    @classmethod
    def read(cls, value):
        mapping(value, {'name', 'group', 'axis'}, {field.name for field in fields(cls)}, 'invalid metric presentation')
        if not all(text(value[key]) for key in ('name', 'group', 'axis')):
            raise ValueError('invalid metric labels')
        if any(type(value[key]) is not str for key in ('source_unit', 'unit', 'color') if key in value):
            raise ValueError('invalid metric unit or color')
        if 'metric_id' in value and type(value['metric_id']) is not str:
            raise ValueError('metric_id must be a string')
        if 'hidden' in value and type(value['hidden']) is not bool:
            raise ValueError('invalid metric visibility')
        scale = value.get('scale', 1)
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('invalid metric scale')
        if scale != 1 and not {'source_unit', 'unit'} <= value.keys():
            raise ValueError('scaled metric requires source_unit and unit')
        labels = value.get('labels')
        if 'labels' in value and (not isinstance(labels, dict) or any(
                type(key) is not str or type(val) is not str for key, val in labels.items())):
            raise ValueError('invalid metric labels binding')
        return cls(**{key: val for key, val in value.items() if key != 'labels'},
                   labels=tuple(labels.items()) if labels is not None else None)

    def to_dict(self):
        result = dict(name=self.name, group=self.group, axis=self.axis, hidden=self.hidden, scale=self.scale)
        for key, value in (('source_unit', self.source_unit), ('unit', self.unit), ('color', self.color),
                           ('metric_id', self.metric_id), ('labels', dict(self.labels) if self.labels is not None else None)):
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class AxisPresentation:
    title: str
    position: str
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def read(cls, value):
        mapping(value, {'title', 'position'}, {'title', 'position', 'min', 'max'}, 'invalid panel axes')
        if not text(value['title']) or value['position'] not in ('left', 'right'):
            raise ValueError('invalid panel axes')
        for key in ('min', 'max'):
            if key in value and (type(value[key]) not in (int, float) or not math.isfinite(value[key])):
                raise ValueError('invalid panel axis bounds')
        minimum, maximum = value.get('min'), value.get('max')
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError('invalid panel axis bounds: min must be less than max')
        return cls(value['title'], value['position'], minimum, maximum)

    def to_dict(self):
        value = dict(title=self.title, position=self.position)
        if self.minimum is not None:
            value['min'] = self.minimum
        if self.maximum is not None:
            value['max'] = self.maximum
        return value


class SelectionKind(str, Enum):
    IDS = 'ids'
    ROLES = 'roles'
    AGGREGATIONS = 'aggregations'
    VISIBLE = 'visible'
    NAMES = 'names'
    CONTAINS = 'contains'
    GROUPS = 'groups'


@dataclass(frozen=True)
class PresetSelection:
    kind: SelectionKind
    values: tuple[str, ...] = ()

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError('invalid panel presets')
        key, items = next(iter(value.items()))
        try:
            kind = SelectionKind(key)
        except (ValueError, TypeError):
            raise ValueError('invalid panel presets') from None
        if kind == SelectionKind.VISIBLE:
            if items is not True:
                raise ValueError('invalid panel presets')
            return cls(kind)
        if not isinstance(items, list) or not items or not all(map(text, items)):
            raise ValueError('invalid panel presets')
        return cls(kind, tuple(items))

    def to_dict(self):
        return {self.kind.value: True if self.kind == SelectionKind.VISIBLE else list(self.values)}

    def matches(self, *, name, group, hidden, curve_id=None, role=None, aggregation=None):
        if self.kind == SelectionKind.IDS:
            return curve_id in self.values
        if self.kind == SelectionKind.ROLES:
            return role in self.values
        if self.kind == SelectionKind.AGGREGATIONS:
            return aggregation in self.values
        if self.kind == SelectionKind.VISIBLE:
            return not hidden
        if self.kind == SelectionKind.NAMES:
            return name in self.values
        if self.kind == SelectionKind.CONTAINS:
            return any(fragment in name for fragment in self.values)
        return group in self.values


@dataclass(frozen=True)
class PanelPresentation:
    id: str
    title: str
    curve_ids: tuple[str, ...]
    axes: tuple[tuple[str, AxisPresentation], ...]
    presets: tuple[tuple[str, PresetSelection], ...]
    caption: str = ''
    empty_caption: str = '没有有效观测数据；缺失值不补零。'
    event_ids: tuple[str, ...] | None = None

    @classmethod
    def read(cls, value):
        mapping(value, {'id', 'title', 'curve_ids'}, {field.name for field in fields(cls)}, 'invalid panel presentation')
        if not all(text(value[key]) for key in ('id', 'title')):
            raise ValueError('invalid panel id or title')
        for key in ('caption', 'empty_caption'):
            if key in value and not text(value[key]):
                raise ValueError('invalid panel '+key)
        curves = value['curve_ids']
        if not isinstance(curves, list) or not curves or not all(map(text, curves)) or len(set(curves)) != len(curves):
            raise ValueError('invalid panel metric_ids: duplicate or empty selection')
        axes, presets = value.get('axes', {}), value.get('presets', {})
        if not isinstance(axes, dict) or not all(map(text, axes)):
            raise ValueError('invalid panel axes')
        if not isinstance(presets, dict) or not all(map(text, presets)):
            raise ValueError('invalid panel presets')
        events = value.get('event_ids')
        if events is not None and (not isinstance(events, list) or not all(map(text, events)) or len(set(events)) != len(events)):
            raise ValueError('invalid panel event_ids')
        return cls(value['id'], value['title'], tuple(curves),
            tuple((key, AxisPresentation.read(axis)) for key, axis in axes.items()),
            tuple((key, PresetSelection.read(selector)) for key, selector in presets.items()),
            value.get('caption', ''), value.get('empty_caption', cls.empty_caption), tuple(events) if events is not None else None)

    def to_dict(self):
        result = dict(id=self.id, title=self.title, curve_ids=list(self.curve_ids),
            axes={key: axis.to_dict() for key, axis in self.axes},
            empty_caption=self.empty_caption)
        if self.caption:
            result['caption'] = self.caption
        if self.presets:
            result['presets'] = {key: value.to_dict() for key, value in self.presets}
        if self.event_ids is not None:
            result['event_ids'] = list(self.event_ids)
        return result

    def validate_bindings(self, curves):
        axes = dict(self.axes)
        for identity in self.curve_ids:
            if identity not in curves:
                raise ValueError('invalid panel metric_ids')
            if curves[identity].axis not in axes:
                raise ValueError('panel metric axis is not declared')


@dataclass(frozen=True)
class ReportPresentation:
    subtitle: str
    identity: str | None = None
    producer: str | None = None

    def to_dict(self):
        result = dict(subtitle=self.subtitle)
        if self.identity is not None:
            result['id'] = self.identity
        if self.producer is not None:
            result['producer'] = self.producer
        return result


@dataclass(frozen=True)
class MetricSelection:
    query_plan: str
    diagnostic_only: tuple[str, ...]

    def to_dict(self):
        return dict(query_plan=self.query_plan, diagnostic_only=list(self.diagnostic_only))


@dataclass(frozen=True)
class ChartDefinition:
    curves: dict[str, CurvePresentation]
    panels: tuple[PanelPresentation, ...]
    events: dict
    event_ids: tuple[str, ...]
    detail_labels: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()
    default_visible: tuple[str, ...] = ()
    max_points_per_series: int = 256
    default_presets: dict = field(default_factory=dict)

    def to_dict(self, kind):
        result = dict(events=copy.deepcopy(self.events), event_ids=list(self.event_ids))
        if kind == 'default':
            result.update(group_by=['epoch', 'source', 'metric'], detail_labels=list(self.detail_labels),
                summaries=list(self.summaries), default_visible=list(self.default_visible),
                max_points_per_series=self.max_points_per_series,
                presets={key:list(values) for key,values in self.default_presets.items()})
        else:
            result.update(curves={key: value.to_dict() for key, value in self.curves.items()},
                          panels=[value.to_dict() for value in self.panels])
        return result


@dataclass(frozen=True)
class ViewDefinition:
    kind: str
    report: ReportPresentation
    metrics: MetricSelection | None
    charts: ChartDefinition
    sections: dict[str, SectionPresentation]
    subtitle: SubtitleTemplate

    @classmethod
    def read(cls, data):
        charts, report = data['charts'], data['report']
        return cls(kind=data['kind'],
            report=ReportPresentation(report['subtitle'], report.get('id'), report.get('producer')),
            metrics=MetricSelection(data['metrics']['query_plan'], tuple(data['metrics'].get('diagnostic_only', [])))
                    if 'metrics' in data else None,
            charts=ChartDefinition(
                curves={key: CurvePresentation.read(value) for key, value in charts.get('curves', {}).items()},
                panels=tuple(PanelPresentation.read(value) for value in charts.get('panels', [])),
                events=copy.deepcopy(charts.get('events', {})), event_ids=tuple(charts.get('event_ids', [])),
                detail_labels=tuple(charts.get('detail_labels', [])), summaries=tuple(charts.get('summaries', [])),
                default_visible=tuple(charts.get('default_visible', [])),
                max_points_per_series=charts.get('max_points_per_series', 256),
                default_presets={key:tuple(values) for key,values in charts.get('presets', {}).items()}),
            sections={key: _read_at(SectionPresentation, value, 'sections.'+key) for key, value in data.get('sections', {}).items()},
            subtitle=SubtitleTemplate(report['subtitle'], cache_behavior=report.get('producer') == 'cache-gate'))

    def to_dict(self):
        result = dict(report_view_schema_version=1, kind=self.kind, report=self.report.to_dict())
        if self.metrics is not None:
            result['metrics'] = self.metrics.to_dict()
        result['charts'] = self.charts.to_dict(self.kind)
        if self.kind == 'selected':
            result['sections'] = {key: value.to_dict() for key, value in self.sections.items()}
        return result


def _read_at(model, value, location):
    try:
        return model.read(value)
    except ValueError as exc:
        raise ValueError(location + ': ' + str(exc)) from exc
