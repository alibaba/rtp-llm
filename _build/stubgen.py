"""Validate generated stubs without changing their runtime extension API."""

import ast
import importlib
from pathlib import Path
import re


def normalize_keyword_field(source: str) -> tuple[str, bool]:
    """Represent QuantMethod's runtime ``None`` through its typed member map.

    pybind11-stubgen 3.0 accepts ``None.isidentifier()`` and emits an illegal
    declaration. Python cannot spell this attribute using dot syntax. Keep its
    original declaration as documentation and retain the existing typed
    ``__members__`` access; do not invent a differently named enum member.
    Every other declaration, default value and reference still has to parse.
    """
    lines = source.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.rstrip() == "class QuantMethod:"), None)
    changed = False
    if start is not None:
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].strip() and not lines[i][0].isspace()), len(lines))
        matches = [i for i in range(start + 1, end)
                   if re.fullmatch(r"    None: typing\.ClassVar\[QuantMethod\](?:[ \t]+#[^\r\n]*)?\r?\n?", lines[i])]
        if matches:
            if len(matches) != 1:
                raise ValueError("Ambiguous QuantMethod keyword declarations")
            if not any(re.match(r"    __members__: typing\.ClassVar\[dict\[str, QuantMethod\]\]", line)
                       for line in lines[start + 1:end]):
                raise ValueError("QuantMethod keyword requires the existing typed __members__ mapping")
            i = matches[0]
            lines[i] = ('    # Runtime keyword member: QuantMethod.__members__["None"].\n'
                        + "    # " + lines[i][4:])
            changed = True
    normalized = "".join(lines)
    ast.parse(normalized)
    return normalized, changed


def validate_generated_stub(path: str, module_name: str) -> dict:
    """Validate one complete real generated module, preserving runtime members."""
    target = Path(path)
    source = target.read_text(encoding="utf-8")
    normalized, changed = normalize_keyword_field(source)
    proof = {"module": module_name, "AST_parse": True, "keyword_field_documented": changed}
    if changed:
        # Import the genuine extension already loaded by the production preload.
        module = importlib.import_module(module_name)
        enum = module.QuantMethod
        member = getattr(enum, "None")
        mapped = enum.__members__["None"]
        if member != mapped or int(member) != 0 or int(mapped) != 0:
            raise ValueError("QuantMethod runtime keyword no longer matches its original binding")
        proof["runtime_keyword"] = {"name": "None", "value": int(member),
                                    "getattr_matches_members": True,
                                    "all_members": {name: int(value) for name, value in enum.__members__.items()}}
        target.write_text(normalized, encoding="utf-8")
    return proof
