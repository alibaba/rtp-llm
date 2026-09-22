"""Declared mock performance presets; no name-specific Python branches."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "config/perf_presets/index.json"


def _registry():
    entries = json.loads(INDEX.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"invalid performance preset registry: {INDEX}")
    return entries


def preset_names():
    return tuple(_registry())


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
    performance = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(performance, dict):
        raise ValueError(f"perf_preset {name!r} performance must be an object")
    runtime = {key: value for key, value in entry.items() if key != "performance"}
    if set(runtime) - {"mock_heap", "mock_extra_args"}:
        raise ValueError(f"perf_preset {name!r} has unknown runtime options: {set(runtime)}")
    if "mock_heap" in runtime and not isinstance(runtime["mock_heap"], str):
        raise ValueError(f"perf_preset {name!r} has invalid mock_heap")
    if "mock_extra_args" in runtime and (not isinstance(runtime["mock_extra_args"], list)
                                              or not all(isinstance(arg, str) for arg in runtime["mock_extra_args"])):
        raise ValueError(f"perf_preset {name!r} has invalid mock_extra_args")
    return performance, runtime
