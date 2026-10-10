"""Publish the frozen case verdict, metric projections and report."""

from pathlib import Path

from workload.gate_evidence import write_evidence


def publish_performance(directory, evidence, result):
    from cases.master_performance.metrics import produce
    from cases.master_performance.report import write_report

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_evidence(directory / "performance-gate-evidence.json", evidence)
    produce(directory, evidence, result)
    return write_report(directory, evidence, result)
