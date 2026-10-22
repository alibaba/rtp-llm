"""Validate generated stubs without changing their runtime extension API."""

import ast
import importlib
import inspect
import io
import re
import tokenize
from pathlib import Path

_FAST_TOPK_MODULE = "librtp_compute_ops.rtp_llm_ops"
_FAST_TOPK_NAME = "fast_topk_transform_fused"
_FAST_TOPK_SIGNATURE = (
    "def fast_topk_transform_fused(score: torch.Tensor, lengths: torch.Tensor, "
    "dst_page_table: torch.Tensor, src_page_table: torch.Tensor | None = None, "
    "cu_seqlens_q: torch.Tensor, row_starts: torch.Tensor | None = None) -> None:"
)


def normalize_fast_topk_defaults(source: str, module_name: str) -> tuple[str, bool]:
    """Express the existing nontrailing pybind default using exact overloads.

    An explicitly supplied src_page_table keeps the original positional order.
    Omitting it requires cu_seqlens_q (and any row_starts) by keyword. Neither
    overload makes cu_seqlens_q optional or changes the extension's call API.
    Only the observed complete signature in its real module is rewritten.
    """
    if module_name != _FAST_TOPK_MODULE:
        return source, False
    lines = source.splitlines(keepends=True)
    definitions = [
        token.start[0] - 1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME
        and token.string == "def"
        and token.start[1] == 0
        and lines[token.start[0] - 1].rstrip("\r\n") == _FAST_TOPK_SIGNATURE
    ]
    if not definitions:
        return source, False
    if len(definitions) != 1:
        raise ValueError("Ambiguous fast_topk_transform_fused declarations")
    index = definitions[0]
    previous = next(
        (line.strip() for line in reversed(lines[:index]) if line.strip()), ""
    )
    if previous.startswith("@"):
        raise ValueError("Unexpected decorated fast_topk_transform_fused declaration")
    explicit = _FAST_TOPK_SIGNATURE.replace(
        "src_page_table: torch.Tensor | None = None,",
        "src_page_table: torch.Tensor | None,",
    )
    omitted = _FAST_TOPK_SIGNATURE.replace(
        "src_page_table: torch.Tensor | None = None, ", "*, "
    )
    lines[index] = (
        "import typing\n"
        "@typing.overload\n"
        f"{explicit}\n"
        "    ...\n"
        "@typing.overload\n"
        f"{omitted}\n"
    )
    # The original generated body/docstring stays on the second overload.
    return "".join(lines), True


def verify_fast_topk_docstring(doc: str) -> None:
    """Use the actual generator parser to verify the complete runtime signature."""
    from pybind11_stubgen import arg_parser, stub_parser_from_args
    from pybind11_stubgen.printer import Printer
    from pybind11_stubgen.structs import Identifier

    parser = stub_parser_from_args(arg_parser().parse_args([_FAST_TOPK_MODULE]))
    functions = parser.parse_function_docstring(
        Identifier(_FAST_TOPK_NAME), doc.splitlines()
    )
    if (
        len(functions) != 1
        or Printer(invalid_expr_as_ellipses=False).print_function(functions[0])[0]
        != _FAST_TOPK_SIGNATURE
    ):
        raise ValueError("fast_topk_transform_fused runtime signature changed")


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
    normalized, overloads = normalize_fast_topk_defaults(source, module_name)
    normalized, changed = normalize_keyword_field(normalized)
    proof = {
        "module": module_name,
        "AST_parse": True,
        "keyword_field_documented": changed,
    }
    if overloads:
        module = importlib.import_module(module_name)
        function = getattr(module, _FAST_TOPK_NAME)
        if not inspect.isbuiltin(function) or function.__module__ != module_name:
            raise ValueError(
                "fast_topk_transform_fused requires its real Native binding"
            )
        verify_fast_topk_docstring(function.__doc__ or "")
        proof["fast_topk_overloads"] = {
            "runtime_signature": function.__doc__.splitlines()[0],
            "count": 2,
            "cu_seqlens_q_required": True,
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
    if changed or overloads:
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
