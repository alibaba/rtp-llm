"""Presentation value objects; mappings are decoded only at the view boundary."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
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

    def render(self, *, verdict, monitoring_status, expected_behavior=None, observed_behavior=None):
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
            raise ValueError('curve must bind a declared metric id and labels')
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

    def matches(self, *, name, group, hidden):
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
    event_ids: tuple[str, ...] = ()

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
        events = value.get('event_ids', [])
        if not isinstance(events, list) or not all(map(text, events)) or len(set(events)) != len(events):
            raise ValueError('invalid panel event_ids')
        return cls(value['id'], value['title'], tuple(curves),
            tuple((key, AxisPresentation.read(axis)) for key, axis in axes.items()),
            tuple((key, PresetSelection.read(selector)) for key, selector in presets.items()),
            value.get('caption', ''), value.get('empty_caption', cls.empty_caption), tuple(events))

    def validate_bindings(self, curves):
        axes = dict(self.axes)
        for identity in self.curve_ids:
            if identity not in curves:
                raise ValueError('invalid panel metric_ids')
            if curves[identity].axis not in axes:
                raise ValueError('panel metric axis is not declared')
