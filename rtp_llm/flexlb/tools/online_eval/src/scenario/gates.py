"""One explicit frozen-gate capability, shared by live delivery and offline replay."""

from dataclasses import dataclass
from pathlib import Path
import re
import json
from inspect import signature
from typing import Callable, Optional

from input_contract import validate_call
from artifacts.gate import freeze_gate, load_gate


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

    producer: Optional[str] = None
    describe: Optional[Callable] = None
    view: Optional[str] = None
    validate_view: Optional[Callable] = None
    subtitle_fields: frozenset[str] = frozenset()

    def __post_init__(self):
        if type(self.name) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', self.name):
            raise ValueError('invalid gate name')
        if (self.producer is None) != (self.describe is None):
            raise ValueError('gate producer requires both name and description')
        if self.producer is not None:
            if type(self.producer) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', self.producer):
                raise ValueError('invalid gate producer name')
            validate_call(self.describe, 3, 'gate.' + self.name + '.describe')
        if (self.view is None) != (self.validate_view is None):
            raise ValueError('gate view requires both name and validator')
        if self.view is not None:
            if type(self.view) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*\.yaml', self.view) or self.view == 'default.yaml':
                raise ValueError('invalid gate view name')
            validate_call(self.validate_view, 3, 'gate.' + self.name + '.validate_view')
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

    def render_with_deadline(self, directory, run, presentation, deadline):
        deadline.check()
        evidence, result = load_gate(directory, self.name)
        deadline.check()
        path = self.write_report(directory, evidence, result, run=run, presentation=presentation) / "report.html"
        deadline.check()
        return path

    def publish(self, directory, evidence, result, *, report_options=None):
        """Publish an existing decision without running analyze, even in replay."""
        freeze_gate(directory, self.name, evidence, result)
        self.project(directory)
        frozen_evidence, frozen_result = load_gate(directory, self.name)
        options = dict(report_options or {})
        if 'run' in options:
            raise ValueError('offline publication does not accept a live run')
        return self.write_report(directory, frozen_evidence, frozen_result, **options)


@dataclass(frozen=True)
class StagedGateProjection:
    action: str
    project: object

    def __post_init__(self):
        validate_call(self.project, 3, "staged_gate.project")

    def __call__(self, directory):
        from monitoring.metric_store import MetricStore
        directory = Path(directory)
        result = json.loads((directory / "result.json").read_text())
        store = MetricStore.read(directory)
        for stage in result["stages"]:
            if stage["action"] == self.action:
                for check in stage["checks"]:
                    if check["status"] in {"PASS", "FAIL", "WARNING"}:
                        self.project(store, stage, check)
        store.save(directory)
