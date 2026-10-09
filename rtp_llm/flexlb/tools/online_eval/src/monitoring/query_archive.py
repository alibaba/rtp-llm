"""Convert explicit Prometheus query archives without acquiring other sources."""

import json
import math
from pathlib import Path


def _finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def series(directory, anchor):
    """Read only monitor query archives; return curves, provenance and health gaps."""
    series, sources, gaps, errors = {}, {}, {}, []
    paths = sorted(Path(directory).glob("telemetry/*/queries.json"))
    if (Path(directory) / "queries.json").is_file():
        paths.append(Path(directory) / "queries.json")
    for path in paths:
        data = json.loads(path.read_text())
        epoch = path.parent.name if path.parent != Path(directory) else "1"
        errors.extend(dict(source=epoch, **error) for error in data.get("errors", []))
        errors.extend(
            dict(source=epoch, query=query, error="monitor series absent",
                 severity="diagnostic")
            for query in data.get("missing_queries", [])
        )
        for query_id, query in data["queries"].items():
            source, metric = query_id.split("/", 1)
            for row in query["result"]:
                labels = dict(row["metric"])
                labels.pop("__name__", None)
                name = metric
                for label in ("job", "instance"):
                    labels.pop(label, None)
                key = f"{epoch}/{source}/{name}/" + json.dumps(labels, sort_keys=True)
                points = [[float(t) - anchor, _finite(v)] for t, v in row["values"]]
                series.setdefault(key, []).extend(points)
                sources[key] = dict(
                    path=str(path),
                    promql=query["promql"],
                    start=data["start"],
                    end=data["end"],
                    step=data["step"],
                    backend="prometheus",
                )
                if metric == "up":
                    gaps.setdefault(f"{epoch}/{source}/collection", []).extend(
                        t for t, value in points if value != 1
                    )
        for source in data["targets"]:
            query = data["queries"].get(source + "/up", {})
            points = sorted(
                (float(t), _finite(v))
                for row in query.get("result", [])
                for t, v in row["values"]
            )
            if not points:
                errors.append(
                    dict(source=f"{epoch}/{source}", error="no Prometheus up samples")
                )
            elif (
                points[0][0]
                - data.get("target_bounds", {}).get(source, [data["start"], None])[0]
                > 2 * data["step"]
                or (
                    data.get("target_bounds", {}).get(source, [None, None])[1]
                    or data["end"]
                )
                - points[-1][0]
                > 2 * data["step"]
            ):
                errors.append(
                    dict(
                        source=f"{epoch}/{source}",
                        error="incomplete monitoring coverage",
                    )
                )
    # A failed scrape is a gap in every associated curve; never bridge it.
    for key, points in series.items():
        source = "/".join(key.split("/")[:2])
        failed = set(gaps.get(source + "/collection", []))
        merged = {}
        for t, value in points:
            if not math.isfinite(t):
                raise ValueError("non-finite monitor timestamp")
            if t in merged and merged[t] != value:
                raise ValueError("conflicting monitor samples at one timestamp: " + key)
            merged[t] = value
        for t in failed:
            merged[t] = None
        series[key] = sorted([t, value] for t, value in merged.items())
    return series, sources, {k: v for k, v in gaps.items() if v}, errors
