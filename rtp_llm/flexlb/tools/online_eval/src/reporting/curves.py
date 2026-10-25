"""Materialize measured curves and project them into declared view panels."""

import copy
import math

from reporting.catalog import PALETTE
from reporting.report_model import panel_document


def has_data(curve):
    """Zero is a measurement; nulls and non-finite values are gaps."""
    return any(type(point['y']) in (int, float) and math.isfinite(point['y'])
               for point in curve['points'])


def materialize(curve_id, style, points, *, origin=0, labels=None,
                name=None, unit=None, hidden=None, provenance=None, description=None,
                color_index=0):
    if name is None:
        name = style.name if labels is None else style.name.format(**labels)
    group = style.group if labels is None else style.group.format(**labels)
    scale = style.scale
    result = dict(
        curve_id=curve_id, metric_id=style.metric_id, labels=copy.deepcopy(labels or {}), scale=scale,
        name=name, group=group, axis=style.axis,
        unit=style.unit if style.unit is not None else unit,
        color=style.color or PALETTE[color_index % len(PALETTE)],
        hidden=style.hidden if hidden is None else hidden,
        points=[dict(x=t-origin, y=value*scale if type(value) in (int, float)
                     and math.isfinite(value) else None) for t, value in points],
    )
    if provenance is not None:
        result['provenance'] = copy.deepcopy(provenance)
    if description is not None:
        result['description'] = description
    return result


def project_panels(curves, presentation):
    """Keep missing observations visible; configuration lines are added afterward."""
    panels = []
    for value in presentation.charts.panels:
        descriptor = value
        selected = [dict(curve) for curve_id in descriptor.curve_ids
                    for curve in curves if curve['curve_id'] == curve_id]
        populated = {curve['curve_id'] for curve in selected if has_data(curve)}
        missing = [presentation.charts.curves[curve_id].name
                   for curve_id in descriptor.curve_ids if curve_id not in populated]
        caption = (descriptor.caption if populated else
                   descriptor.empty_caption)
        if populated and missing:
            caption += ' 缺少有效曲线：' + '、'.join(missing) + '。'
        panel = panel_document(identity=descriptor.id, title=descriptor.title,
                     axes={key: axis.to_dict() for key, axis in descriptor.axes},
                     series=selected, caption=caption)
        apply_panel_presets(panel, descriptor)
        panels.append(panel)
    return panels


def apply_panel_presets(panel, descriptor):
    """Resolve against final series, including explicitly added configuration lines."""
    if descriptor.presets:
        panel['presets'] = resolve_presets(panel['series'], {name: (selector,) for name, selector in descriptor.presets})


def resolve_presets(series, selectors):
    """Both view producers select stable identities/roles through one resolver."""
    return {name: [curve['name'] for curve in series if any(selector.matches(
        name=curve['name'], group=curve['group'], hidden=curve['hidden'],
        curve_id=curve.get('curve_id'), role=curve.get('role'), aggregation=curve.get('aggregation')) for selector in selection)]
        for name, selection in selectors.items()}
