"""Strict YAML primitives shared by environment and stage compilation."""

import math
import re
from scenario.loader import ScenarioError

ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def fail(path, message):
    raise ScenarioError(f"{path}: {message}")


def mapping(value, path, allowed, required=()):
    if not isinstance(value, dict):
        fail(path, "expected mapping")
    if any(not isinstance(key, str) for key in value):
        fail(path, "mapping keys must be strings")
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        fail(path, f"unknown fields {sorted(extra)}, missing fields {sorted(missing)}")
    return value


def identifier(value, path):
    if not isinstance(value, str) or not ID.fullmatch(value):
        fail(path, "expected lower-case identifier")
    return value


def number(value, path, minimum=0, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or value < minimum
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        fail(path, f"expected {'integer' if integer else 'number'} >= {minimum}")
    return value


def names(value, path, vocabulary):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        fail(path, "expected string list")
    if len(set(value)) != len(value) or set(value) - set(vocabulary):
        fail(path, f"duplicate or unknown values: {value}")
    return value
