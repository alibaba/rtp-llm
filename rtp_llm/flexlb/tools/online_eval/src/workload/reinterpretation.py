"""Read historical evidence without changing its archive; publish into a new directory."""

import hashlib
import json
import shutil
from pathlib import Path


def load_reinterpretation(parser, args, analyzer_path):
    if not args.reinterpret:
        parser.error("offline adjudication requires --reinterpret; comparisons use frozen bundles")
    source, destination = args.evidence.resolve(), args.output.resolve()
    if destination == source.parent or source.parent in destination.parents:
        parser.error("reinterpretation output must be outside the source archive")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error("reinterpretation requires an empty output directory; frozen reports cannot be overwritten")
    args.evidence, args.output = source, destination
    raw = source.read_bytes()
    evidence = json.loads(raw)
    evidence["reinterpretation"] = dict(
        source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(analyzer_path).read_bytes()).hexdigest(),
    )
    return evidence


def import_metrics(destination, source, plan_name):
    """Copy frozen metrics and convert raw archives, always writing at destination."""
    from monitoring.metric_store import MetricStore, export_metrics
    from monitoring.query_plan import load_plan

    destination, source = Path(destination), Path(source)
    destination.mkdir(parents=True, exist_ok=True)
    if (source / "metrics.json").is_file():
        MetricStore.read(source).save(destination)
    archives = list(source.glob("telemetry/*/queries.json"))
    if (source / "queries.json").is_file():
        archives.append(source / "queries.json")
    for path in archives:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return export_metrics(destination, load_plan(plan_name))
