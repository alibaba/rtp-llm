import copy
import hashlib
import json
import re
from pathlib import Path

from reporting.spec import REPORT_SPEC_SCHEMA_VERSION, validate
from schema_contract import matches_schema


def details(title, value, *, opened=False):
    return dict(type="details", title=title, value=value, opened=opened)


def table(title, columns, rows, *, opened=True):
    return dict(type="table", title=title, columns=list(columns), rows=list(rows), opened=opened)


def links(title, items):
    return dict(type="links", title=title, items=list(items))


def run_meta(
    identity,
    *,
    implementation=None,
    workload=None,
    configuration=None,
    environment=None,
    clock=None,
    evidence=None,
    runs=None,
):
    """None means unknown, never proof of matching experiment conditions."""
    metadata = dict(
        run_meta_schema_version=1,
        identity=identity,
        implementation=implementation,
        workload=workload,
        configuration=configuration,
        environment=environment,
        clock=clock,
        evidence=evidence,
    )
    if runs is not None:
        metadata["runs"] = runs
    return metadata


def compare_controls(left, right, *, required=(), allowed=()):
    """Compare all control leaves, except explicitly varied experiment subtrees.

    required/allowed paths use JSON Pointer syntax. Missing required evidence is
    unknown even when both sides omit it. Unknown added controls are compared.
    """
    missing = object()
    differences = []

    def at(value, pointer):
        for part in pointer.strip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict) or key not in value:
                return missing
            value = value[key]
        return value

    def walk(a, b, path):
        if any(path == p or path.startswith(p + "/") for p in allowed):
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(
                    a.get(key, missing),
                    b.get(key, missing),
                    path + "/" + key.replace("~", "~0").replace("/", "~1"),
                )
        elif a is missing or b is missing or a != b:
            differences.append(
                dict(
                    path=path,
                    status="MISSING" if a is missing or b is missing else "DIFFERENT",
                    baseline=None if a is missing else a,
                    candidate=None if b is missing else b,
                )
            )

    walk(left, right, "")
    absent = [
        p
        for p in required
        if any(at(v, p) is missing or at(v, p) is None for v in (left, right))
    ]
    return dict(
        status="UNKNOWN" if absent else "DIFFERENT" if differences else "ALIGNED",
        aligned=not absent and not differences,
        missing=absent,
        differences=differences,
        allowed=list(allowed),
    )


def _slug(identity):
    text = str(identity)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-") or "report"
    return (
        safe
        if safe == text
        else safe[:96] + "-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    )


def bundle_path(root, kind, identity):
    if kind not in {"run", "comparison"}:
        raise ValueError("unsupported report kind: " + kind)
    return Path(root) / "reports" / kind / _slug(identity)


def render(spec):
    from reporting.renderer import render as render_html
    return render_html(spec)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _atomic(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_bundle(root, kind, identity, analysis, spec, *, meta=None, producer=None, role=None):
    """Publish one canonical report; manifest is written last as the commit record."""
    validate(spec)
    spec = copy.deepcopy(spec)
    spec["report_spec_schema_version"] = REPORT_SPEC_SCHEMA_VERSION
    legacy_meta = spec.get("meta") or {}
    spec["run_meta"] = meta or run_meta(
        dict(id=str(identity), kind=kind),
        implementation=legacy_meta.get("version"),
        workload=legacy_meta.get("dataset"),
        configuration=legacy_meta.get("params"),
        environment=legacy_meta.get("env"),
        clock=dict(axis=spec.get("timeAxis"), origin=spec.get("timeOriginLabel")),
        evidence=legacy_meta.get("sources"),
    )
    _validate_run_meta(spec["run_meta"])
    directory = bundle_path(root, kind, identity)
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(
        report_analysis_schema_version=1,
        kind=kind,
        producer=producer,
        run_meta=spec["run_meta"],
        result=analysis,
    )
    outputs = {
        "analysis.json": _json(result),
        "report-spec.json": _json(spec),
        "report.html": render(spec),
    }
    # Layout adapters emit links relative to this final bundle directory.
    for name, content in outputs.items():
        _atomic(directory / name, content)
    manifest = dict(
        report_manifest_schema_version=1,
        kind=kind,
        id=str(identity),
        producer=producer,
        identity=spec["run_meta"]["identity"],
        entrypoint="report.html",
        files={
            name: dict(path=name, sha256=hashlib.sha256(content.encode()).hexdigest())
            for name, content in outputs.items()
        },
    )
    if role is not None:
        manifest["role"] = role
    _atomic(directory / "manifest.json", _json(manifest))
    return directory


def discover_reports(root, *, role=None, kind="run"):
    """Find verified bundles by kind and optional producer-declared role."""
    if kind not in {"run", "comparison"}:
        raise ValueError("unsupported report kind: " + str(kind))
    reports = []
    for path in sorted((Path(root) / "reports" / kind).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if role is not None and manifest.get("role") != role:
            continue
        directory = read_bundle(path)
        entrypoint = manifest["entrypoint"]
        if entrypoint not in manifest["files"]:
            raise ValueError("report entrypoint is not a verified bundle file")
        reports.append(str((directory / entrypoint).resolve()))
    return reports


def read_bundle(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    manifest = json.loads((directory / "manifest.json").read_text())
    if not matches_schema(manifest, "report_manifest_schema_version", 1):
        raise ValueError("unsupported report manifest version")
    files = manifest.get("files")
    required = {"analysis.json", "report-spec.json", "report.html"}
    if (not isinstance(files, dict) or not required <= files.keys()
            or any(files[name].get("path") != name for name in required)):
        raise ValueError("report manifest lacks required bundle files")
    for entry in files.values():
        target = directory / entry["path"]
        if target.resolve().parent != directory.resolve():
            raise ValueError("report file escapes bundle")
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("report checksum mismatch: " + entry["path"])
    spec = json.loads((directory / "report-spec.json").read_text())
    validate(spec, versioned=True)
    _validate_run_meta(spec.get("run_meta"))
    analysis = json.loads((directory / "analysis.json").read_text())
    if not matches_schema(analysis, "report_analysis_schema_version", 1):
        raise ValueError("unsupported report analysis version")
    _validate_run_meta(analysis.get("run_meta"))
    return directory


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
    if path.is_dir():
        path = read_bundle(path) / "analysis.json"
    if path.name == "analysis.json" and (path.parent / "manifest.json").exists():
        read_bundle(path.parent)
    value = json.loads(path.read_text())
    if not matches_schema(value, "report_analysis_schema_version", 1):
        raise ValueError("unsupported report analysis version")
    _validate_run_meta(value.get("run_meta"))
    return value["result"]
