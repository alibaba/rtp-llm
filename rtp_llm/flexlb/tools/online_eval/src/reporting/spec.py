"""Validate the single producer contract without converting chart data."""

import math

from schema_contract import matches_schema, version_fields

REPORT_SPEC_SCHEMA_VERSION = 1


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(spec, *, versioned=False):
    if not isinstance(spec, dict):
        raise ValueError("report spec must be an object")
    if (versioned or version_fields(spec)) and not matches_schema(
            spec, "report_spec_schema_version", REPORT_SPEC_SCHEMA_VERSION):
        raise ValueError("unsupported report spec version")
    if spec.get("timeAxis") is not None:
        bounds = spec["timeAxis"]
        if (not isinstance(bounds, dict) or not _finite(bounds.get("min"))
                or not _finite(bounds.get("max")) or bounds["min"] >= bounds["max"]):
            raise ValueError("invalid report timeAxis")
    ids = set()
    for panel in spec.get("panels", []):
        identity = panel["id"]
        if identity in ids:
            raise ValueError("duplicate panel id: " + identity)
        ids.add(identity)
        kind = panel.get("type", "line")
        if kind not in {"line", "bar", "scatter"}:
            raise ValueError("unsupported panel type")
        if {"x", "xNums", "overlay", "representation"} & panel.keys():
            raise ValueError("panel " + identity + ": legacy chart fields are unsupported; emit series.points")
        if "timeX" in panel and type(panel["timeX"]) is not bool:
            raise ValueError("timeX must be boolean: " + identity)
        axes = panel.get("axes")
        if (not isinstance(axes, dict) or not axes
                or any(type(key) is not str or not key or key == "x"
                       or not isinstance(settings, dict) for key, settings in axes.items())):
            raise ValueError("panel lacks axes: " + identity)
        for series in panel.get("series", []):
            axis = series.get("axis", "y")
            if axis not in axes:
                raise ValueError(f"panel {identity} series {series.get('name', '')!r} uses undeclared axis {axis!r}")
            if "data" in series or not isinstance(series.get("points"), list):
                raise ValueError("panel " + identity + ": every series must emit points, never data")
            for field in ("points", "statistics_points"):
                if field not in series or series[field] is None:
                    continue
                for point in series[field]:
                    if (not isinstance(point, dict) or not {"x", "y"} <= point.keys()
                            or not (_finite(point["x"]) or not panel.get("timeX") and type(point["x"]) is str)
                            or point["y"] is not None and not _finite(point["y"])):
                        raise ValueError(f"panel {identity}: invalid {field} coordinate")
