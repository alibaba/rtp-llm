"""Declared mock performance presets; no name-specific Python branches."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "config/performance_presets.json"


def _registry():
    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"invalid performance preset registry: {INDEX}")
    return entries


def preset_names():
    return tuple(_registry())


def resolve_performance_calibration(performance, source_path):
    """Materialize a named test calibration without altering capture records."""
    reference = performance.get("calibration")
    if reference is None:
        return performance
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("performance calibration must be a relative file path")
    target = (Path(source_path).resolve().parent / reference).resolve()
    calibration_root = (ROOT / "data/performance").resolve()
    if target.parent != calibration_root or not target.is_file():
        raise ValueError(f"missing registered mock calibration: {target}")
    from flexlb_profile_data import load_mock_calibration

    calibration = load_mock_calibration(target)
    decode = performance.setdefault("decode", {})
    if not isinstance(decode, dict):
        raise ValueError("performance.decode must be a mapping")
    for key, value in calibration["decode"].items():
        if key in decode and decode[key] != value:
            raise ValueError(f"performance.decode.{key} disagrees with {target}")
        decode[key] = value
    performance.pop("calibration")
    performance["calibration_id"] = calibration["id"]
    performance["calibration_model"] = calibration["model"]
    performance["calibration_hardware"] = calibration["hardware"]
    performance["calibration_status"] = calibration["status"]
    performance["calibration_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return performance


def load_performance_bundle(path):
    """Validate one Engine/Master record, then project its paired components."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("performance document must be an object")
    if "schema_version" in document:
        from flexlb_profile_data import load_mock_calibration

        load_mock_calibration(path)
        for key in ("schema_version", "id", "model", "hardware", "status", "source", "prefill_expression"):
            document.pop(key)
        document["calibration"] = Path(path).name
    capture = document.pop("capture", None)
    if capture is not None:
        if not isinstance(capture, dict) or capture.get("status") not in (
            "verified", "legacy_unverified", "orphan"
        ):
            raise ValueError("invalid capture provenance status")
        record = (json.dumps(document, indent=2) + "\n").encode()
        if hashlib.sha256(record).hexdigest() != capture.get("record_sha256"):
            raise ValueError("performance capture record checksum mismatch")
        if capture["status"] == "verified" and (
            not capture.get("identity", {}).get("deployment") or not capture.get("capture_window")
        ):
            raise ValueError("verified capture requires deployment and window")
    master = document.pop("master", {})
    if not isinstance(master, dict) or set(master) - {"config_overrides", "provenance"}:
        raise ValueError("invalid paired master settings")
    if master:
        from flexlb_cfg import ConfigOverride
        overrides = master.get("config_overrides")
        provenance = master.get("provenance")
        if not isinstance(overrides, dict) or not isinstance(provenance, dict):
            raise ValueError("paired master requires config_overrides and provenance")
        if provenance.get("status") not in {"verified", "legacy_unverified"} or not provenance.get("source"):
            raise ValueError("paired master requires provenance status and source")
        ConfigOverride(**overrides)
    resolve_performance_calibration(document, path)
    return document, capture, master


def load_performance_file(path):
    """Compatibility projection for consumers that only read Engine settings."""
    performance, capture, _ = load_performance_bundle(path)
    return performance, capture


def capture_defaults(name):
    """Observed topology and KV pool sizes, if the preset has a capture record."""
    entries = _registry()
    if name not in entries:
        raise ValueError(f"invalid perf_preset {name!r}")
    _, capture = load_performance_file(ROOT / entries[name]["performance"])
    if not capture or capture["status"] == "orphan":
        return {}
    fields = ("n_prefill", "n_decode", "prefill_kv_pool_blocks", "decode_kv_pool_blocks")
    if any(type(capture.get(key)) is not int or capture[key] <= 0 for key in fields):
        raise ValueError(f"perf_preset {name!r} has invalid capture topology or KV pool capacity")
    return {key: capture[key] for key in fields}


def load_preset(name):
    entries = _registry()
    if not isinstance(name, str) or name not in entries:
        raise ValueError(f"invalid perf_preset {name!r}; expected one of {tuple(entries)}")
    entry = entries[name]
    if not isinstance(entry, dict) or not isinstance(entry.get("performance"), str):
        raise ValueError(f"invalid perf_preset declaration {name!r}")
    path = (ROOT / entry["performance"]).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"perf_preset {name!r} has missing performance file: {path}")
    performance, capture, master = load_performance_bundle(path)
    if "calibration" in entry:
        performance["calibration"] = entry["calibration"]
        resolve_performance_calibration(performance, INDEX)
    runtime = {key: value for key, value in entry.items()
               if key not in {"performance", "calibration"}}
    if set(runtime) - {"mock_heap", "mock_extra_args"}:
        raise ValueError(f"perf_preset {name!r} has unknown runtime options: {set(runtime)}")
    if "mock_heap" in runtime and not isinstance(runtime["mock_heap"], str):
        raise ValueError(f"perf_preset {name!r} has invalid mock_heap")
    if "mock_extra_args" in runtime and (not isinstance(runtime["mock_extra_args"], list)
                                              or not all(isinstance(arg, str) for arg in runtime["mock_extra_args"])):
        raise ValueError(f"perf_preset {name!r} has invalid mock_extra_args")
    if capture and capture["status"] != "orphan" and "mock_extra_args" in capture:
        if "mock_extra_args" in runtime:
            raise ValueError(f"perf_preset {name!r} duplicates capture mock settings")
        runtime["mock_extra_args"] = capture["mock_extra_args"]
    if master:
        runtime["master_config_overrides"] = master["config_overrides"]
        runtime["master_provenance"] = master["provenance"]
    return performance, runtime
