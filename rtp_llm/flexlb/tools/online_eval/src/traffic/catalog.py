"""Versioned input catalog shared by Python launchers and the Java fixture test."""
import hashlib
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
CATALOG = DATA / "catalog.json"


def validate_model_name(entry):
    name = Path(entry["model"]).name
    prefix = f"{entry['codec']}_v{entry['codec_version']}_"
    if not (name.startswith(prefix) and
            re.fullmatch(r"[0-9a-f]{8,16}\.xz", name[len(prefix):])):
        raise ValueError(f"unconventional snapshot name: {name}")


def catalog():
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), dict):
        raise ValueError("invalid data/catalog.json")
    return document


def model_entry(name=None):
    document = catalog()
    name = name or document["default_trace"]
    if name not in document["models"]:
        raise ValueError(f"unknown data model {name!r}")
    entry = document["models"][name]
    validate_model_name(entry)
    paths = {}
    for key in ("model", "manifest", "java_fixture", "calibration_profile"):
        if key not in entry and key in ("java_fixture", "calibration_profile"):
            continue
        if key not in entry:
            raise ValueError(f"{name}: missing {key} declaration")
        path = (DATA / entry[key]).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file():
            raise ValueError(f"{name}: missing {key}: {path}")
        paths[key] = path
    return entry, paths


def verify_model(name):
    entry, paths = model_entry(name)
    raw = paths["model"].read_bytes()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if manifest.get("bytes") != len(raw) or manifest.get("sha256") != digest:
        raise ValueError(f"{name}: model and manifest bytes/SHA256 disagree")
    if manifest.get("manifest_schema_version") != 1 or manifest.get("codec") != {"name": entry["codec"], "version": entry["codec_version"]}:
        raise ValueError(f"{name}: manifest schema/codec identity disagrees")
    if manifest.get("capture_source") != entry.get("source"):
        raise ValueError(f"{name}: capture source disagrees with manifest")
    from traffic.prefix_lineage import decode
    metadata, events = decode(raw)
    for field in ("version", "block_size", "count", "provenance"):
        if manifest.get(field) != metadata.get(field):
            raise ValueError(f"{name}: manifest {field} disagrees with model contents")
    if manifest["count"] != len(events):
        raise ValueError(f"{name}: event count disagrees with manifest")
    if paths["manifest"].name != paths["model"].stem + ".manifest.json":
        raise ValueError(f"{name}: model and manifest must share a prefix")
    if not digest.startswith(paths["model"].stem.rsplit("_", 1)[-1]):
        raise ValueError(f"{name}: filename digest differs from model bytes")
    for key, suffix in (("java_fixture", ".templates.json"), ("calibration_profile", ".profile.json")):
        if key in paths and paths[key].name != paths["model"].stem + suffix:
            raise ValueError(f"{name}: {key} must share the model prefix")
    return entry, paths, manifest, digest


def verify_default_model():
    document = catalog()
    entry, paths, manifest, digest = verify_model(document["default_trace"])
    if "java_fixture" not in paths or "calibration_profile" not in paths:
        raise ValueError("default model lacks Java fixture or calibration profile")
    fixture = json.loads(paths["java_fixture"].read_text(encoding="utf-8"))
    profile = json.loads(paths["calibration_profile"].read_text(encoding="utf-8"))
    if fixture.get("source_sha256") != digest:
        raise ValueError("catalog Java fixture source SHA256 disagrees with model")
    calibration = profile.get("calibration") or {}
    if calibration.get("model_sha256") != digest or calibration.get("provenance") != manifest.get("provenance"):
        raise ValueError("catalog calibration provenance disagrees with manifest")
    if entry.get("codec") != "prefix_lineage" or entry.get("codec_version") != manifest.get("version"):
        raise ValueError("catalog codec identity disagrees with manifest")
    return entry, paths
