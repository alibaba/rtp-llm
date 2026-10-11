"""One explicit frozen-gate capability, shared by live delivery and offline replay."""

from dataclasses import dataclass
from pathlib import Path
import re
from inspect import signature
from typing import Callable

from cases.extension_contracts import validate_call
from workload.gate_result import freeze_gate, load_gate


@dataclass(frozen=True)
class GateCommit:
    result: dict
    path: Path


@dataclass(frozen=True)
class GatePlugin:
    """Business functions compute/project/present; this capability owns publication.

    analyze(evidence) is pure. project_metrics(directory, evidence, result) publishes
    numeric projections only. write_report(directory, evidence, result, *, run=..., presentation=...)
    reads the frozen decision; neither delivery callback may re-adjudicate it.
    """

    name: str
    analyze: Callable[[dict], dict]
    project_metrics: Callable
    write_report: Callable

    def __post_init__(self):
        if type(self.name) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', self.name):
            raise ValueError('invalid gate name')
        for name, count in (('analyze', 1), ('project_metrics', 3), ('write_report', 3)):
            validate_call(getattr(self, name), count, 'gate.' + self.name + '.' + name)
        try:
            signature(self.write_report).bind(object(), object(), object(), run=None, presentation={})
        except TypeError as exc:
            raise ValueError('gate.' + self.name + '.write_report must accept run and presentation keywords') from exc

    def evaluate(self, evidence):
        """Explicit adjudication; delivery never calls this method implicitly."""
        from analysis.checks import CheckResult
        from analysis.gates import gate_checks
        result = self.analyze(evidence)
        verdict, _ = gate_checks([CheckResult(**row) for row in result['checks']], [])
        if verdict != result['verdict']:
            raise ValueError('gate verdict disagrees with typed checks')
        return result

    def commit(self, directory, evidence):
        """Live gate: evaluate once and commit before metric/report delivery."""
        result = self.evaluate(evidence)
        return GateCommit(result, freeze_gate(directory, self.name, evidence, result))

    def project(self, directory):
        """Project committed decisions; an aborted run may have no gate commit."""
        if not (Path(directory) / (self.name + '-gate-manifest.json')).is_file():
            return
        evidence, result = load_gate(directory, self.name)
        self.project_metrics(directory, evidence, result)

    def render(self, directory, run, presentation):
        """Selected-view adapter. Missing/corrupt committed evidence fails loudly."""
        evidence, result = load_gate(directory, self.name)
        return self.write_report(directory, evidence, result, run=run, presentation=presentation) / 'report.html'

    def publish(self, directory, evidence, result, *, report_options=None):
        """Publish an existing decision without running analyze, even in replay."""
        freeze_gate(directory, self.name, evidence, result)
        self.project(directory)
        frozen_evidence, frozen_result = load_gate(directory, self.name)
        options = dict(report_options or {})
        if 'run' in options:
            raise ValueError('offline publication does not accept a live run')
        return self.write_report(directory, frozen_evidence, frozen_result, **options)
