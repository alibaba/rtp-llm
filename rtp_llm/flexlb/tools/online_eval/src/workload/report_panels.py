"""Project archived workload series into bounded, auditable chart panels."""

import hashlib
import json
import math

from reporting.catalog import PERFORMANCE_COLORS, performance_axes, performance_metric_style
from reporting.view_config import template


def _sample(points, limit):
    """Keep extrema and one explicit gap per time bucket; never turn a gap into zero."""
    if len(points) <= limit:
        return [dict(x=x, y=y) for x, y in points]
    width = max(1, math.ceil(len(points) / max(1, (limit - 2) // 3)))
    selected = {0, len(points) - 1}
    for start in range(0, len(points), width):
        bucket = range(start, min(start + width, len(points)))
        valid = [i for i in bucket if points[i][1] is not None]
        if valid:
            selected.add(min(valid, key=lambda i: points[i][1]))
            selected.add(max(valid, key=lambda i: points[i][1]))
        missing = next((i for i in bucket if points[i][1] is None), None)
        if missing is not None:
            selected.add(missing)
    return [dict(x=points[i][0], y=points[i][1]) for i in sorted(selected)]


def _identity(key):
    parts = key.split("/", 3)
    if len(parts) != 4:
        raise ValueError("invalid archived series key: " + key)
    epoch, source, metric, raw_labels = parts
    labels = json.loads(raw_labels)
    if not isinstance(labels, dict):
        raise ValueError("archived series labels must be a mapping")
    return (epoch, source, metric), labels


def _summarize(members, method):
    values = [dict(points) for _, points, _ in members]
    times = sorted({t for series in values for t in series})
    result = []
    for t in times:
        sample = [series.get(t) for series in values]
        if any(value is None for value in sample):
            result.append((t, None))
        else:
            result.append((t, sum(sample) / len(sample) if method == "mean" else max(sample)))
    return result


def build_panels(series, sources, declaration=None):
    view = template((declaration or {}).get("template", "workload"))
    visible = (declaration or {}).get("default_visible", view["default_visible"])
    presets = {**view["presets"], **(declaration or {}).get("presets", {})}
    grouped = {}
    for key, points in series.items():
        base, labels = _identity(key)
        residual = {name: value for name, value in labels.items()
                    if name not in view["detail_labels"]}
        group = (*base, json.dumps(residual, sort_keys=True))
        grouped.setdefault(group, []).append((key, points, labels))
    panels = []
    for (epoch, source, metric, residual), members in sorted(grouped.items()):
        members.sort(key=lambda row: row[0])
        role = str(members[0][2].get("role", ""))
        title, category, axis, _ = performance_metric_style(source, metric, role)
        curves = []
        high_cardinality = len(members) > 1
        limit = view["max_points_per_series"]
        methods = view["summaries"] if high_cardinality else []
        for index, method in enumerate(methods):
            keys = [key for key, _, _ in members]
            raw = _summarize(members, method)
            curves.append(dict(
                name=("均值" if method == "mean" else "最大值") + f" · {len(members)} series",
                group="汇总", axis=axis, color=PERFORMANCE_COLORS[index],
                hidden=method not in visible, points=_sample(raw, limit),
                description="缺少任一原始 series 的采样时留空，不补零。",
                provenance=dict(kind="derived", source_series_keys=keys,
                                calculation=f"same-timestamp {method}; require all sources present",
                                projection=f"bucket min/max/gap; max_points={limit}"),
            ))
        for index, (key, points, labels) in enumerate(members):
            name = json.dumps(labels, sort_keys=True, ensure_ascii=False) if labels else key
            curves.append(dict(
                name=name, group=category, axis=axis,
                color=PERFORMANCE_COLORS[(index + len(methods)) % len(PERFORMANCE_COLORS)],
                hidden=high_cardinality,
                points=_sample(points, limit), description=key,
                provenance=dict(kind="archived", source_series_key=key,
                                source=sources.get(key),
                                projection=f"bucket min/max/gap; max_points={limit}"),
            ))
        choices = {}
        for name, kinds in presets.items():
            choices[name] = [
                curve["name"] for curve in curves
                if (curve["provenance"]["kind"] == "archived"
                    and ("detail" in kinds or not high_cardinality))
                or any(
                    curve["provenance"].get("calculation", "").split(";")[0].endswith(kind)
                    for kind in kinds if kind != "detail"
                )
            ]
        panels.append(dict(
            id="metric-" + hashlib.sha256("/".join((epoch, source, metric, residual)).encode()).hexdigest(),
            title=f"{epoch}/{source} · {title}" + (f" · {residual}" if residual != "{}" else ""),
            caption="同指标同图；明细可从指标选择器检索。仅 HTML 采样，analysis.json 保留原始全集。",
            overlay=True, timeX=True, axes={axis: performance_axes()[axis]},
            series=curves, presets=choices,
        ))
    return panels
