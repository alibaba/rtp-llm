"""Explicit generation and validation of versioned result golden files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.results import ResultSchemaError, parse_result_envelope
from rtp_llm.test.perf_test.storage import atomic_write_json


class GoldenMismatchError(ValueError):
    """Raised when a committed golden is invalid or differs from the result."""


def normalized_golden(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical schema representation used by golden operations."""
    return parse_result_envelope(result).to_dict()


def write_golden(result: Mapping[str, Any], output: Union[str, Path]) -> Path:
    """Explicitly replace a golden with a validated, deterministic snapshot."""
    path = Path(output)
    atomic_write_json(path, normalized_golden(result))
    return path


def validate_golden(result: Mapping[str, Any], golden: Union[str, Path]) -> None:
    """Validate a golden without modifying it."""
    path = Path(golden)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        normalized_stored = normalized_golden(stored)
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise GoldenMismatchError(f"invalid golden {path}: {error}") from error
    normalized_result = normalized_golden(result)
    if canonical_json(normalized_result) != canonical_json(normalized_stored):
        raise GoldenMismatchError(f"result does not match golden: {path}")


def load_result(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object for a golden command."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid result {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid result {source}: expected an object")
    return value
