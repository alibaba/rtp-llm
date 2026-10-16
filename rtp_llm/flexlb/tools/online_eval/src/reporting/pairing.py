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


def merge_nearby_events(events, tolerance_s=1.0):
    """Display one marker for close equivalent events from distinct runs.

    Cluster width is bounded by tolerance, rather than chaining nearby pairs.
    Full original events remain in members; alignment always uses original input.
    A zero tolerance disables merging.
    """
    if type(tolerance_s) not in (int, float) or not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError('event merge tolerance must be finite and nonnegative')
    if tolerance_s == 0:
        return copy.deepcopy(events)
    groups, unchanged = {}, []
    for event in events:
        if not event.get('id') or not event.get('run'):
            unchanged.append(copy.deepcopy(event))
            continue
        key = (event['id'], event['event_name'])
        groups.setdefault(key, []).append(event)
    result = unchanged
    for (identity, name), candidates in groups.items():
        clusters = []
        for event in sorted(candidates, key=lambda e: e['t']):
            if (not clusters or event['t'] - clusters[-1][0]['t'] > tolerance_s
                    or any(e['run'] == event['run'] for e in clusters[-1])):
                clusters.append([])
            clusters[-1].append(event)
        for members in clusters:
            if len(members) == 1:
                result.append(copy.deepcopy(members[0]))
            else:
                result.append(dict(id=identity, name=name + ' ×' + str(len(members)),
                    t=sum(e['t'] for e in members) / len(members),
                    members=copy.deepcopy(members)))
    return sorted(result, key=lambda e: (e['t'], e.get('name', '')))
