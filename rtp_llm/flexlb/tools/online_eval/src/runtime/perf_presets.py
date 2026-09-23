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


def load_performance_file(path):
    """Read a capture record; return only mock-engine performance fields."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("performance document must be an object")
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
    return document, capture


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
    performance, capture = load_performance_file(path)
    runtime = {key: value for key, value in entry.items() if key != "performance"}
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
    return performance, runtime
