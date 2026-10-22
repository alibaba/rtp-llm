"""Discover pinned input files without importing their codecs."""
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TRACE = "glm-5.3_20260921_1400_15m"
DEFAULT_PROFILE = DEFAULT_TRACE


def trace_models():
    return {path.stem: path for path in sorted((DATA / "traffic_trace").glob("*.xz"))}


def model_path(name=DEFAULT_TRACE):
    try:
        return trace_models()[name]
    except KeyError:
        raise ValueError(f"unknown traffic model {name!r}") from None


def profile_path(name=DEFAULT_PROFILE):
    for path in (DATA / "synthetic_parameters").glob("*.profile.json"):
        if path.name == f"{name}.profile.json":
            return path
    raise ValueError(f"unknown synthetic parameter profile {name!r}")
