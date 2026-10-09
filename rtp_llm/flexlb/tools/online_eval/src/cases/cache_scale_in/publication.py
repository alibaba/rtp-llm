"""Publish the frozen case verdict, metric projections and report."""

from pathlib import Path

from workload.gate_evidence import write_evidence


def publish_cache(directory, evidence, result, prepared=None):
    from cases.cache_scale_in.metrics import produce
    from cases.cache_scale_in.report import write_report

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_evidence(directory / "cache-gate-evidence.json", evidence)
    produce(directory, evidence, result)
    return write_report(directory, evidence, result, prepared=prepared)
