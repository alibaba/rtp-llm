"""Normalize checked frozen bundles and experiment containers for comparison."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from file_io import read_json
from reporting.core import load_bundle
from reporting.experiment import ExperimentReader
from reporting.renderer import render


@dataclass(frozen=True)
class RunSnapshot:
    source: Path
    analysis: dict
    spec: dict
    digest: str
    input_kind: str
    view: str | None
    execution: dict
    controls: dict = field(init=False)
    outcome: dict = field(init=False)

    def __post_init__(self):
        controls = self.spec["run_meta"].get("comparison_controls")
        if not isinstance(controls, dict):
            raise ValueError("frozen report lacks comparison controls")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "outcome", original_result(self.analysis, self.execution))

    def original_html(self):
        if self.input_kind == "bundle":
            return (self.source / "report.html").read_text(encoding="utf-8")
        # Render the frozen layout only; no current YAML or domain analyzer.
        spec = dict(self.spec, sections=[s for s in self.spec.get("sections", []) if s["type"] != "links"])
        return render(spec)


def load_snapshot(path, *, view=None):
    path = Path(path).resolve()
    if path.is_dir():
        if view is not None:
            raise ValueError("view selection applies only to experiment ZIP inputs")
        bundle = load_bundle(path)
        if bundle.manifest.get("kind") != "run":
            raise ValueError("comparison input must be a run bundle: " + str(path))
        return RunSnapshot(
            path,
            bundle.analysis(),
            bundle.spec,
            bundle.digest,
            "bundle",
            None,
            read_json(path / "execution.json")
            if bundle.manifest["files"].get("execution.json", {}).get("path") == "execution.json"
            else {},
        )
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError("comparison input must be a run bundle directory or experiment ZIP: " + str(path))
    with ExperimentReader(path) as reader:
        if reader.kind != "experiment":
            raise ValueError("raw attachment is not an experiment comparison input")
        selected = view or reader.preferred_view
        if selected not in reader.views:
            raise ValueError("comparison requires a frozen view: " + selected)
        spec = reader.view_spec(selected)
        execution = reader.read_json("execution.json")
        analysis = reader.analysis() if "analysis.json" in reader.entries else {"execution": execution}
        from hashlib import sha256

        digest = sha256(reader.archive.read("manifest.json")).hexdigest()
        return RunSnapshot(path, analysis, spec, digest, "experiment", selected, execution)


def original_result(analysis: dict, execution: dict) -> dict:
    """Read the frozen gate and validity separately from execution status."""
    outcome = analysis.get("outcome") or {}
    executed = execution.get("outcome") or {}
    workload = analysis.get("workload") or {}
    return dict(
        verdict=outcome.get("gate") or executed.get("gate") or analysis.get("verdict"),
        validity=outcome.get("validity")
        or executed.get("validity")
        or analysis.get("runtime_validity")
        or workload.get("runtime_validity")
        or execution.get("workload", {}).get("runtime_validity"),
        status=execution.get("status") or analysis.get("status"),
    )
