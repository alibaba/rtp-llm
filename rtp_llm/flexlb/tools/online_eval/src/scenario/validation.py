"""Strict YAML primitives shared by environment and stage compilation."""

import math
import re
from document_io import DocumentError as ScenarioError
from input_contract import fail

ID = re.compile(r"[a-z][a-z0-9_]*\Z")



def mapping(value, path, allowed, required=()):
    from input_contract import FieldSetError, mapping_fields
    try:
        return mapping_fields(value, allowed, path, required=required)
    except FieldSetError as exc:
        raise ScenarioError(f"{path}: unknown fields {exc.extra}, missing fields {exc.missing}") from exc
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc


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
