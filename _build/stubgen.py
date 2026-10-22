"""Validate generated stubs without changing their runtime extension API."""

import ast
import importlib
import inspect
import re
from pathlib import Path


def normalize_keyword_field(source: str) -> tuple[str, bool]:
    """Represent QuantMethod's runtime ``None`` through its typed member map.

    pybind11-stubgen 3.0 accepts ``None.isidentifier()`` and emits an illegal
    declaration. Python cannot spell this attribute using dot syntax. Keep its
    original declaration as documentation and retain the existing typed
    ``__members__`` access; do not invent a differently named enum member.
    Every other declaration, default value and reference still has to parse.
    """
    lines = source.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "class QuantMethod:"),
        None,
    )
    changed = False
    if start is not None:
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i][0].isspace()
            ),
            len(lines),
        )
        matches = [
            i
            for i in range(start + 1, end)
            if re.fullmatch(
                r"    None: typing\.ClassVar\[QuantMethod\](?:[ \t]+#[^\r\n]*)?\r?\n?",
                lines[i],
            )
        ]
        if matches:
            if len(matches) != 1:
                raise ValueError("Ambiguous QuantMethod keyword declarations")
            if not any(
                re.match(
                    r"    __members__: typing\.ClassVar\[dict\[str, QuantMethod\]\]",
                    line,
                )
                for line in lines[start + 1 : end]
            ):
                raise ValueError(
                    "QuantMethod keyword requires the existing typed __members__ mapping"
                )
            i = matches[0]
            lines[i] = (
                '    # Runtime keyword member: QuantMethod.__members__["None"].\n'
                + "    # "
                + lines[i][4:]
            )
            changed = True
    normalized = "".join(lines)
    ast.parse(normalized)
    return normalized, changed


def validate_generated_stub(path: str, module_name: str) -> dict:
    """Validate one complete real generated module, preserving runtime members."""
    target = Path(path)
    source = target.read_text(encoding="utf-8")
    normalized, changed = normalize_keyword_field(source)
    proof = {
        "module": module_name,
        "AST_parse": True,
        "keyword_field_documented": changed,
    }
    if changed:
        # Import the genuine extension already loaded by the production preload.
        module = importlib.import_module(module_name)
        enum = module.QuantMethod
        member = getattr(enum, "None")
        mapped = enum.__members__["None"]
        if member != mapped or int(member) != 0 or int(mapped) != 0:
            raise ValueError(
                "QuantMethod runtime keyword no longer matches its original binding"
            )
        proof["runtime_keyword"] = {
            "name": "None",
            "value": int(member),
            "getattr_matches_members": True,
            "all_members": {
                name: int(value) for name, value in enum.__members__.items()
            },
        }
        target.write_text(normalized, encoding="utf-8")
    return proof


def validate_generated_module(output_dir: str, module_name: str) -> dict:
    """Validate stubgen's flat or package layout against the real module tree.

    Stubgen emits ``__init__.pyi`` for a package or a module containing native
    submodules, and a flat ``.pyi`` for a leaf module. Imported modules and
    aliases are not child modules. Keep the original single-file validation,
    including keyword/runtime checks, for every required and emitted stub.
    """
    output = Path(output_dir)
    required = {}

    def visit(module):
        name = module.__name__
        children = [
            value
            for attr, value in inspect.getmembers(module, inspect.ismodule)
            if value.__name__ == name + "." + attr
        ]
        base = output.joinpath(*name.split("."))
        flat, package = base.with_suffix(".pyi"), base / "__init__.pyi"
        if flat.exists() and package.exists():
            raise ValueError(f"Ambiguous generated stub layouts for {name}")
        target = package if children or hasattr(module, "__path__") else flat
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f"Required generated stub missing: {target}")
        required[target] = name
        for child in children:
            visit(child)

    visit(importlib.import_module(module_name))
    root = output.joinpath(*module_name.split("."))
    emitted = set(required)
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Generated stub tree contains a symlink: {path}")
            if path.is_file() and path.suffix == ".pyi":
                emitted.add(path)
    proofs = []
    for path in sorted(emitted):
        relative = path.relative_to(output)
        name = required.get(path)
        if name is None:
            parts = relative.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            name = ".".join(parts)
        proofs.append(
            {"path": str(relative), **validate_generated_stub(str(path), name)}
        )
    return {"module": module_name, "AST_parse": True, "files": proofs}
