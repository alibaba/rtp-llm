"""Materialize measured curves and project them into declared view panels."""

import copy
import math

from reporting.catalog import PALETTE


def has_data(curve):
    """Zero is a measurement; nulls and non-finite values are gaps."""
    return any(type(point['y']) in (int, float) and math.isfinite(point['y'])
               for point in curve['points'])


def materialize(curve_id, style, points, *, origin=0, labels=None,
                name=None, unit=None, hidden=None, provenance=None, description=None,
                color_index=0):
    if name is None:
        name = style["name"] if labels is None else style["name"].format(**labels)
    group = style["group"] if labels is None else style["group"].format(**labels)
    scale = style.get('scale', 1)
    result = dict(
        curve_id=curve_id, metric_id=style['metric_id'],
        name=name, group=group, axis=style['axis'],
        unit=style.get('unit', unit),
        color=style.get('color') or PALETTE[color_index % len(PALETTE)],
        hidden=style.get('hidden', False) if hidden is None else hidden,
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
    for descriptor in presentation['charts']['panels']:
        selected = [dict(curve, hidden=False) for curve_id in descriptor['curve_ids']
                    for curve in curves if curve['curve_id'] == curve_id]
        populated = {curve['curve_id'] for curve in selected if has_data(curve)}
        missing = [presentation['charts']['curves'][curve_id]['name']
                   for curve_id in descriptor['curve_ids'] if curve_id not in populated]
        caption = (descriptor.get('caption', '') if populated else
                   descriptor.get('empty_caption', '没有有效观测数据；缺失值不补零。'))
        if populated and missing:
            caption += ' 缺少有效曲线：' + '、'.join(missing) + '。'
        panels.append(dict(id=descriptor['id'], title=descriptor['title'],
                           timeX=True, axes=copy.deepcopy(descriptor['axes']),
                           series=selected, caption=caption))
    return panels
