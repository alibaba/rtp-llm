"""Lossless adapter from historical chart shapes to one panel contract."""

import copy


def normalize_spec(spec):
    """Keep chart data intact and expose a common time-point representation.

    Historical ``x``/``xNums`` + ``series.data`` and overlay ``series.points``
    remain readable. Time charts gain ``points`` without dropping their legacy
    fields; category bars and scatter panels keep their original coordinates.
    """
    result = copy.deepcopy(spec)
    for panel in result.get("panels", []):
        panel["representation"] = "multi" if panel.get("overlay") else "standard"
        if panel.get("type") == "scatter" or not panel.get("timeX"):
            continue
        times = panel.get("xNums") or []
        for series in panel.get("series", []):
            if series.get("points") is None and len(series.get("data", [])) == len(times):
                series["points"] = [dict(x=time, y=value)
                                    for time, value in zip(times, series["data"])]
    return result
