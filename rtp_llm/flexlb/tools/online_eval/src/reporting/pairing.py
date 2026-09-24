"""Presentation-only alignment and pairing of already analyzed chart series."""

import copy
import math


def event_anchor(events, name):
    """Return a unique finite event time, or None when alignment is unavailable."""
    if name is None:
        return None
    times = [event.get("t") for event in events if event.get("name") == name]
    if len(times) != 1 or type(times[0]) not in (int, float) or not math.isfinite(times[0]):
        return None
    return times[0]


def shifted_panel(panel, anchor):
    """Copy a panel and express time coordinates relative to its anchor."""
    result = copy.deepcopy(panel)
    if anchor is None:
        return result
    if result.get("xNums"):
        result["xNums"] = [value - anchor for value in result["xNums"]]
        result["x"] = [str(value) for value in result["xNums"]]
    for series in result.get("series", []):
        if series.get("points") is not None:
            for point in series["points"]:
                point["x"] -= anchor
    return result


def align_series(points, anchor):
    """Express (time, value) samples against a declared stage or event time."""
    return [[time - anchor, value] for time, value in points]


def pair_samples(left, right):
    """Align sparse numeric series without inventing zero-valued samples."""
    left, right = dict(left), dict(right)
    axis = sorted(left.keys() | right.keys())
    return axis, [left.get(time) for time in axis], [right.get(time) for time in axis], [
        right[time] - left[time] if time in left and time in right
        and left[time] is not None and right[time] is not None else None
        for time in axis
    ]


def paired_overlay(panels, labels=("A", "B"), *, color_for=None, hidden_for=None):
    """Pair series and presets while retaining null gaps and source order."""
    if len(panels) != len(labels):
        raise ValueError("one label is required for each comparison panel")
    series, presets, colors = [], {}, {}
    for index, (label, panel) in enumerate(zip(labels, panels)):
        for source in panel.get("series", []):
            curve = copy.deepcopy(source)
            original_name = curve["name"]
            colors.setdefault(original_name, curve.get("color"))
            curve["name"] = f"{label} · {original_name}"
            curve["dash"] = [6, 4] if index == 0 else []
            if color_for is not None:
                curve["color"] = color_for(original_name, label, colors[original_name])
            else:
                curve["color"] = colors[original_name]
            if hidden_for is not None:
                curve["hidden"] = hidden_for(original_name)
            series.append(curve)
        for name, selection in panel.get("presets", {}).items():
            presets.setdefault(name, []).extend(f"{label} · {item}" for item in selection)
    return series, presets
