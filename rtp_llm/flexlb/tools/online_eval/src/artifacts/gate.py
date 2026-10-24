"""Atomic, verified gate artifacts independent of metric and HTML delivery."""

import hashlib
import json
from pathlib import Path
from analysis.checks import CheckResult
from analysis.gates import gate_checks
from artifacts.errors import ArtifactNotProduced
from schema_contract import matches_schema
from artifacts.json_io import write_json


def freeze_gate(directory, name, evidence, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    verdict, _ = gate_checks([CheckResult(**row) for row in result["checks"]], [])
    if verdict != result["verdict"]:
        raise ValueError("gate verdict disagrees with typed checks")
    source = directory / f"{name}-gate-evidence.json"
    target = directory / f"{name}-gate-result.json"
    write_json(source, evidence)
    write_json(target, result)
    write_json(directory / f"{name}-gate-manifest.json", dict(
        gate_result_schema_version=1,
        evidence=dict(file=source.name, sha256=_sha(source)),
        result=dict(file=target.name, sha256=_sha(target)),
    ))
    return target


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate(directory, name):
    directory = Path(directory)
    committed = directory / f"{name}-gate-manifest.json"
    if not committed.is_file():
        raise ArtifactNotProduced("gate result was not committed: " + name)
    manifest = json.loads(committed.read_text(encoding='utf-8'))
    if not matches_schema(manifest, "gate_result_schema_version", 1):
        raise ValueError("unsupported frozen gate manifest")
    values = []
    for key, filename in (("evidence", f"{name}-gate-evidence.json"),
                          ("result", f"{name}-gate-result.json")):
        entry = manifest[key]
        if entry["file"] != filename:
            raise ValueError("unexpected gate artifact path")
        path = directory / filename
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise ValueError("gate artifact checksum mismatch: " + filename)
        values.append(json.loads(path.read_text(encoding='utf-8')))
    return tuple(values)


def gate_check(identity, result, path, expected):
    return CheckResult(identity, {"INVALID": "ERROR", "PASS": "PASS", "FAIL": "FAIL"}[result["verdict"]],
                       detail="; ".join(result.get("errors", [])), actual=result["verdict"], expected=expected,
                       evidence=dict(result=str(path), sha256=_sha(Path(path)), checks=result["checks"]))


def commit_gate(directory, name, evidence, analyze):
    result = analyze(evidence)
    return result, freeze_gate(directory, name, evidence, result)


def project_gate(directory, name, project):
    if (Path(directory) / (name + "-gate-manifest.json")).is_file():
        evidence, result = load_gate(directory, name)
        project(directory, evidence, result)


def render_gate(directory, run, presentation, deadline=None, *, name, write_report):
    if deadline is not None:
        deadline.check()
    evidence, result = load_gate(directory, name)
    if deadline is not None:
        deadline.check()
    path = write_report(directory, evidence, result, run=run, presentation=presentation) / "report.html"
    if deadline is not None:
        deadline.check()
    return path


def publish_gate(directory, name, evidence, result, project, write_report, *, report_options=None):
    freeze_gate(directory, name, evidence, result)
    project_gate(directory, name, project)
    evidence, result = load_gate(directory, name)
    options = dict(report_options or {})
    if 'run' in options:
        raise ValueError('offline publication does not accept a live run')
    return write_report(directory, evidence, result, **options)
