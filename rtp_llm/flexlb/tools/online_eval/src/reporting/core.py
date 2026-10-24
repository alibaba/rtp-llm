"""Checked report documents and recoverable publication of portable bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from file_digest import sha256_file
from file_io import read_json, write_text
from reporting.components import run_meta
from reporting.evidence_archive import project, restore, verify_references
from reporting.renderer import _render_validated
from reporting.report_model import ExperimentConditions
from reporting.spec import REPORT_SPEC_SCHEMA_VERSION, validate
from schema_contract import matches_schema


def _slug(identity):
    text = str(identity)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-") or "report"
    return safe if safe == text and len(safe) <= 96 else safe[:96] + "-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def bundle_path(root, kind, identity):
    if kind not in {"run", "comparison"}:
        raise ValueError("unsupported report kind: " + kind)
    return Path(root) / "reports" / kind / _slug(identity)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def write_bundle(root, kind, identity, analysis, spec, *, meta=None, producer=None, role=None, attachments=None):
    """Finish a staged bundle before replacement; restore the previous bundle on failure."""
    attachments = _attachments(attachments)
    spec = _prepare_spec(kind, identity, analysis, spec, meta)
    directory = bundle_path(root, kind, identity)
    directory.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".report-", dir=directory.parent))
    staged, previous = scratch / "bundle", scratch / "previous"
    try:
        _write_bundle(root, staged, kind, identity, analysis, spec, producer=producer, role=role, attachments=attachments)
        if directory.exists():
            directory.replace(previous)
        try:
            staged.replace(directory)
        except BaseException:
            if previous.exists():
                previous.replace(directory)
            raise
    finally:
        # Preserve the backup if both publication and rollback fail.
        if directory.exists() or not previous.exists():
            shutil.rmtree(scratch, ignore_errors=True)
    return directory


def _attachments(attachments):
    attachments = dict((attachments or {}).items())
    for name, content in attachments.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
            or name in {".", "..", "manifest.json", "analysis.json", "report-spec.json", "report.html"}
            or not isinstance(content, str)
        ):
            raise ValueError("invalid report attachment path or content")
    return attachments


def _prepare_spec(kind, identity, analysis, spec, meta):
    validate(spec)
    spec = dict(spec)
    spec["report_spec_schema_version"] = REPORT_SPEC_SCHEMA_VERSION
    spec["run_meta"] = copy.deepcopy(
        meta
        or spec.get("run_meta")
        or run_meta(dict(id=str(identity), kind=kind), clock=dict(axis=spec.get("timeAxis"), origin=spec.get("timeOriginLabel")))
    )
    controls = ExperimentConditions(
        spec["run_meta"].get("configuration"),
        spec["run_meta"].get("workload"),
        spec["run_meta"].get("environment"),
        analysis.get("criteria"),
        analysis.get("statistic_sources"),
    )
    spec["run_meta"]["comparison_controls"] = controls.to_dict()
    _validate_run_meta(spec["run_meta"])
    return spec


def _write_bundle(root, directory, kind, identity, analysis, spec, *, producer=None, role=None, attachments=None):
    """Publish one canonical report; manifest is written last as the commit record."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    archived_analysis = _analysis_nodes(analysis, lambda node: project(root, directory, node, files))
    result = dict(report_analysis_schema_version=1, kind=kind, producer=producer, run_meta=spec["run_meta"], result=archived_analysis)
    # Serialize and release each output before building the next large document.
    for name, produce in (
        ("analysis.json", lambda: _json(result)),
        ("report-spec.json", lambda: _json(spec)),
        ("report.html", lambda: _render_validated(spec)),
    ):
        content = produce()
        write_text(directory / name, content)
        files[name] = dict(path=name, sha256=hashlib.sha256(content.encode()).hexdigest())
        del content
    for name, content in attachments.items():
        if name in files:
            raise ValueError("invalid report attachment path")
        write_text(directory / name, content)
        files[name] = dict(path=name, sha256=hashlib.sha256(content.encode()).hexdigest())
    manifest = dict(
        report_manifest_schema_version=1,
        kind=kind,
        id=str(identity),
        producer=producer,
        identity=spec["run_meta"]["identity"],
        entrypoint="report.html",
        files=files,
    )
    if role is not None:
        manifest["role"] = role
    write_text(directory / "manifest.json", _json(manifest))


@dataclass(frozen=True)
class ReportBundle:
    directory: Path
    manifest: dict
    spec: dict
    result: dict
    digest: str

    def analysis(self) -> dict:
        return _analysis_nodes(self.result, lambda node: restore(self.directory, node))


def load_bundle(path: str | Path) -> ReportBundle:
    """Verify each member once and retain the decoded documents for downstream readers."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    encoded = (directory / "manifest.json").read_bytes()
    manifest = json.loads(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    if not matches_schema(manifest, "report_manifest_schema_version", 1):
        raise ValueError("unsupported report manifest version")
    files = manifest.get("files")
    required = {"analysis.json", "report-spec.json", "report.html"}
    if not isinstance(files, dict) or not required <= files.keys() or any(files[name].get("path") != name for name in required):
        raise ValueError("report manifest lacks required bundle files")
    documents = {}
    for name, entry in files.items():
        target = directory / entry["path"]
        if target.resolve().parent != directory.resolve():
            raise ValueError("report file escapes bundle")
        if name in {"analysis.json", "report-spec.json"}:
            encoded = target.read_bytes()
            actual = hashlib.sha256(encoded).hexdigest()
        else:
            actual = sha256_file(target)
        if actual != entry["sha256"]:
            raise ValueError("report checksum mismatch: " + entry["path"])
        if name in {"analysis.json", "report-spec.json"}:
            documents[name] = json.loads(encoded)
    spec, analysis = documents["report-spec.json"], documents["analysis.json"]
    validate(spec, versioned=True)
    _validate_run_meta(spec.get("run_meta"))
    if not matches_schema(analysis, "report_analysis_schema_version", 1):
        raise ValueError("unsupported report analysis version")
    _validate_run_meta(analysis.get("run_meta"))
    _analysis_nodes(analysis["result"], lambda node: verify_references(node, files) or node)
    return ReportBundle(directory, manifest, spec, analysis["result"], digest)


def read_bundle(path):
    return load_bundle(path).directory


def _validate_run_meta(meta):
    if not matches_schema(meta, "run_meta_schema_version", 1):
        raise ValueError("unsupported run metadata version")
    runs = meta.get("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("run metadata runs must be a mapping")
    for value in runs.values():
        _validate_run_meta(value)


def load_analysis(path):
    path = Path(path)
    if path.is_dir() or path.name == "analysis.json" and (path.parent / "manifest.json").exists():
        return load_bundle(path).analysis()
    value = read_json(path)
    if not matches_schema(value, "report_analysis_schema_version", 1):
        raise ValueError("unsupported report analysis version")
    _validate_run_meta(value.get("run_meta"))
    return _analysis_nodes(value["result"], lambda node: restore(path.parent, node))


def _analysis_nodes(analysis, operation):
    """The publication contract owns run envelopes; the codec handles one node."""
    result = operation(analysis)
    if isinstance(result.get("run"), dict):
        result["run"] = _analysis_nodes(result["run"], operation)
    if isinstance(result.get("runs"), dict):
        result["runs"] = {label: _analysis_nodes(run, operation) for label, run in result["runs"].items()}
    return result
