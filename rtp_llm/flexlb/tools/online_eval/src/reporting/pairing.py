"""Presentation-only event alignment of already analyzed chart series."""

import copy
import math


def event_anchor(events, name):
    """Return a unique finite event time, or None when alignment is unavailable."""
    if name is None:
        return None
    times = [event.get("t") for event in events if event.get("id") == name]
    if len(times) != 1 or type(times[0]) not in (int, float) or not math.isfinite(times[0]):
        return None
    return times[0]


def shifted_panel(panel, anchor):
    """Copy a panel and express time coordinates relative to its anchor."""
    result = copy.deepcopy(panel)
    if anchor is None or not panel.get("timeX"):
        return result
    for event in result.get("events", []):
        event["t"] -= anchor
    shifted = set()
    for series in result.get("series", []):
        for field in ("points", "statistics_points"):
            if series.get(field) is not None and id(series[field]) not in shifted:
                shifted.add(id(series[field]))
                for point in series[field]:
                    point["x"] -= anchor
    return result
