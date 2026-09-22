import copy
import hashlib
import json
import re
from pathlib import Path

SCHEMA_VERSION = 1


def details(title, value):
    return dict(type="details", title=title, value=value)


def table(title, columns, rows):
    return dict(type="table", title=title, columns=list(columns), rows=list(rows))


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
    evidence=None
):
    """None means unknown, never proof of matching experiment conditions."""
    return dict(
        schema_version=1,
        identity=identity,
        implementation=implementation,
        workload=workload,
        configuration=configuration,
        environment=environment,
        clock=clock,
        evidence=evidence,
    )


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
    if kind not in {"run", "comparison", "sweep"}:
        raise ValueError("unsupported report kind: " + kind)
    return Path(root) / "reports" / kind / _slug(identity)


def render(spec):
    if spec.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported report spec version")
    ids = set()
    for panel in spec.get("panels", []):
        if panel.get("type", "line") not in {"line", "bar", "scatter"}:
            raise ValueError("unsupported panel type")
        if panel["id"] in ids:
            raise ValueError("duplicate panel id: " + panel["id"])
        ids.add(panel["id"])
    from reporting.renderer import render as render_html

    return render_html(spec)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _atomic(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_bundle(root, kind, identity, analysis, spec, *, meta=None, producer=None):
    """Publish one canonical report; manifest is written last as the commit record."""
    directory = bundle_path(root, kind, identity)
    directory.mkdir(parents=True, exist_ok=True)
    spec = copy.deepcopy(spec)
    spec["schema_version"] = SCHEMA_VERSION
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
    result = dict(
        schema_version=SCHEMA_VERSION,
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
        schema_version=SCHEMA_VERSION,
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
    _atomic(directory / "manifest.json", _json(manifest))
    return directory


def read_bundle(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported report manifest version")
    for entry in manifest["files"].values():
        target = directory / entry["path"]
        if target.resolve().parent != directory.resolve():
            raise ValueError("report file escapes bundle")
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("report checksum mismatch: " + entry["path"])
    return directory


def load_analysis(path):
    path = Path(path)
    if path.is_dir():
        path = read_bundle(path) / "analysis.json"
    if path.name == "analysis.json" and (path.parent / "manifest.json").exists():
        read_bundle(path.parent)
    value = json.loads(path.read_text())
    if "run_meta" in value and "result" in value:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported report analysis version")
        return value["result"]
    return value  # Explicit read adapter for pre-bundle archives; never dual-write.
