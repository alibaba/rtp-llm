"""Generation-text contracts; these tests do not substitute a Native module."""

import ast
import importlib
import inspect
import itertools
import subprocess
import sys

import pytest

from _build.stubgen import (
    normalize_fast_topk_defaults,
    normalize_keyword_field,
    validate_generated_module,
    validate_generated_stub,
    verify_fast_topk_docstring,
)

STUB = """from __future__ import annotations
import typing
class QuantMethod:
    None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>
    Awq: typing.ClassVar[QuantMethod]
    __members__: typing.ClassVar[dict[str, QuantMethod]]
    def __int__(self) -> int: ...
class Other:
    def method(self, value: int = 3) -> None: ...
"""


def test_keyword_is_documented_without_alias_or_lost_typed_map():
    with pytest.raises(SyntaxError):
        ast.parse(STUB)
    normalized, changed = normalize_keyword_field(STUB)
    assert changed
    ast.parse(normalized)
    original = (
        "    None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>\n"
    )
    replacement = (
        '    # Runtime keyword member: QuantMethod.__members__["None"].\n'
        "    # None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>\n"
    )
    assert normalized.replace(replacement, original) == STUB
    assert "None_" not in normalized
    assert "    __members__: typing.ClassVar[dict[str, QuantMethod]]\n" in normalized


def test_valid_source_is_byte_identical():
    valid = "from __future__ import annotations\nclass Plain:\n    value: int\n"
    assert normalize_keyword_field(valid) == (valid, False)


def test_normalization_is_idempotent():
    normalized, _ = normalize_keyword_field(STUB)
    assert normalize_keyword_field(normalized) == (normalized, False)


@pytest.mark.parametrize(
    "invalid",
    [
        STUB + "def f(value=QuantMethod.None): ...\n",
        STUB + "value: QuantMethod.None\n",
        STUB + "class OtherEnum:\n    None: int\n",
        STUB.replace("Awq: typing.ClassVar[QuantMethod]", "Awq: invalid::CxxType"),
    ],
)
def test_other_keywords_references_and_defaults_still_fail(invalid):
    with pytest.raises(SyntaxError):
        normalize_keyword_field(invalid)


def test_keyword_requires_existing_typed_members_map():
    without_map = STUB.replace(
        "    __members__: typing.ClassVar[dict[str, QuantMethod]]\n", ""
    )
    with pytest.raises(ValueError, match="existing typed __members__"):
        normalize_keyword_field(without_map)


def test_duplicate_keyword_is_rejected():
    duplicate = STUB.replace(
        "    Awq:", "    None: typing.ClassVar[QuantMethod]\n    Awq:"
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        normalize_keyword_field(duplicate)


def test_different_keyword_annotation_is_not_silently_dropped():
    with pytest.raises(SyntaxError):
        normalize_keyword_field(
            STUB.replace("None: typing.ClassVar[QuantMethod]", "None: int")
        )


def test_name_in_docstring_is_unchanged():
    source = (
        'class Other:\n    """None: typing.ClassVar[QuantMethod]"""\n    field: int\n'
    )
    assert normalize_keyword_field(source) == (source, False)


@pytest.fixture
def generated_python_layout(tmp_path, monkeypatch):
    """Run the real tool on ordinary Python, never impersonate a Native .so."""
    created = []

    def generate(package):
        name = "stub_layout_example"
        source = tmp_path / "source"
        source.mkdir()
        if package:
            module = source / name
            module.mkdir()
            (module / "__init__.py").write_text("from . import child\n")
            child = module / "child"
            child.mkdir()
            (child / "__init__.py").write_text("from . import leaf\n")
            (child / "leaf.py").write_text("def value() -> int: return 3\n")
        else:
            (source / (name + ".py")).write_text("def value() -> int: return 3\n")
        output = tmp_path / "output"
        subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", name, "-o", str(output)],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        )
        monkeypatch.syspath_prepend(str(source))
        importlib.invalidate_caches()
        created.append(name)
        return output, name

    yield generate
    for name in created:
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(name + "."):
                del sys.modules[loaded]


@pytest.mark.parametrize("package", [False, True])
def test_real_stubgen_flat_and_nested_package_layouts(generated_python_layout, package):
    output, name = generated_python_layout(package)
    if package:
        # The original flat-only validator fails on this real tool output.
        assert not (output / (name + ".pyi")).exists()
        with pytest.raises(FileNotFoundError):
            validate_generated_stub(str(output / (name + ".pyi")), name)
    proof = validate_generated_module(str(output), name)
    paths = [entry["path"] for entry in proof["files"]]
    expected = (
        [f"{name}/__init__.pyi", f"{name}/child/__init__.pyi", f"{name}/child/leaf.pyi"]
        if package
        else [f"{name}.pyi"]
    )
    assert paths == expected
    assert all(entry["AST_parse"] for entry in proof["files"])


@pytest.mark.parametrize(
    "missing", ["__init__.pyi", "child/__init__.pyi", "child/leaf.pyi"]
)
def test_missing_required_package_stub_fails(generated_python_layout, missing):
    output, name = generated_python_layout(True)
    (output / name / missing).unlink()
    with pytest.raises(FileNotFoundError, match="Required generated stub missing"):
        validate_generated_module(str(output), name)


def test_invalid_child_stub_still_fails(generated_python_layout):
    output, name = generated_python_layout(True)
    (output / name / "child/leaf.pyi").write_text("field: invalid::CxxType\n")
    with pytest.raises(SyntaxError):
        validate_generated_module(str(output), name)


def test_unexpected_emitted_stub_is_also_ast_checked(generated_python_layout):
    output, name = generated_python_layout(True)
    (output / name / "extra.pyi").write_text("field: invalid::CxxType\n")
    with pytest.raises(SyntaxError):
        validate_generated_module(str(output), name)


def test_flat_and_package_ambiguity_fails(generated_python_layout):
    output, name = generated_python_layout(True)
    (output / (name + ".pyi")).write_text("value: int\n")
    with pytest.raises(ValueError, match="Ambiguous generated stub layouts"):
        validate_generated_module(str(output), name)


def test_missing_flat_stub_fails(generated_python_layout):
    output, name = generated_python_layout(False)
    (output / (name + ".pyi")).unlink()
    with pytest.raises(FileNotFoundError, match="Required generated stub missing"):
        validate_generated_module(str(output), name)


# Exact generated line retained in CI 82325810, not a substitute Native module.
FAST_TOPK_HEADER = (
    "def fast_topk_transform_fused(score: torch.Tensor, lengths: torch.Tensor, "
    "dst_page_table: torch.Tensor, src_page_table: torch.Tensor | None = None, "
    "cu_seqlens_q: torch.Tensor, row_starts: torch.Tensor | None = None) -> None:"
)
FAST_TOPK_STUB = (
    "from __future__ import annotations\nimport torch\n"
    + FAST_TOPK_HEADER
    + '\n    """Fast TopK Transform Fused kernel"""\n'
)
FAST_TOPK_MODULE = "librtp_compute_ops.rtp_llm_ops"


def test_observed_nontrailing_default_becomes_two_exact_overloads():
    with pytest.raises(SyntaxError, match="non-default argument follows default"):
        ast.parse(FAST_TOPK_STUB)
    normalized, changed = normalize_fast_topk_defaults(FAST_TOPK_STUB, FAST_TOPK_MODULE)
    assert changed
    functions = [
        node for node in ast.parse(normalized).body if isinstance(node, ast.FunctionDef)
    ]
    assert len(functions) == 2
    assert [ast.unparse(node.decorator_list[0]) for node in functions] == [
        "typing.overload",
        "typing.overload",
    ]
    assert [arg.arg for arg in functions[0].args.args] == [
        "score",
        "lengths",
        "dst_page_table",
        "src_page_table",
        "cu_seqlens_q",
        "row_starts",
    ]
    assert [arg.arg for arg in functions[1].args.args] == [
        "score",
        "lengths",
        "dst_page_table",
    ]
    assert [arg.arg for arg in functions[1].args.kwonlyargs] == [
        "cu_seqlens_q",
        "row_starts",
    ]
    assert len(functions[0].args.defaults) == 1
    assert functions[1].args.kw_defaults[0] is None
    assert ast.unparse(functions[1].args.kw_defaults[1]) == "None"
    assert all(ast.unparse(node.returns) == "None" for node in functions)
    assert normalized.count('"""Fast TopK Transform Fused kernel"""') == 1
    assert normalize_fast_topk_defaults(normalized, FAST_TOPK_MODULE) == (
        normalized,
        False,
    )


def test_overloads_preserve_every_original_argument_binding_shape():
    """Use standard Python Signature binding; no Native functions are imitated."""
    normalized, _ = normalize_fast_topk_defaults(FAST_TOPK_STUB, FAST_TOPK_MODULE)
    functions = [
        node for node in ast.parse(normalized).body if isinstance(node, ast.FunctionDef)
    ]
    signatures = []
    for node in functions:
        parameters = []
        first_default = len(node.args.args) - len(node.args.defaults)
        for index, arg in enumerate(node.args.args):
            default = None if index >= first_default else inspect.Parameter.empty
            parameters.append(
                inspect.Parameter(
                    arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default
                )
            )
        for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            default = inspect.Parameter.empty if default_node is None else None
            parameters.append(
                inspect.Parameter(
                    arg.arg, inspect.Parameter.KEYWORD_ONLY, default=default
                )
            )
        signatures.append(inspect.Signature(parameters))
    names = [
        "score",
        "lengths",
        "dst_page_table",
        "src_page_table",
        "cu_seqlens_q",
        "row_starts",
    ]
    required = {"score", "lengths", "dst_page_table", "cu_seqlens_q"}
    # Includes all-keyword, mixed, omitted/explicit-None src and row, duplicates,
    # missing required arguments, extra positionals, and unknown keywords.
    for positional_count, presence in itertools.product(
        range(8), itertools.product([False, True], repeat=7)
    ):
        args = tuple(None if i in (3, 5) else i for i in range(positional_count))
        kwargs = {
            name: None if name in ("src_page_table", "row_starts") else name
            for name, included in zip(names + ["unknown"], presence)
            if included
        }
        supplied = set(names[:positional_count]) | set(kwargs)
        expected = (
            positional_count <= 6
            and "unknown" not in kwargs
            and not (set(names[:positional_count]) & set(kwargs))
            and required <= supplied
        )
        matches = []
        for signature in signatures:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                continue
            bound.apply_defaults()
            values = dict(bound.arguments)
            values.setdefault("src_page_table", None)
            matches.append(values)
        assert bool(matches) == expected, (args, kwargs)
        if expected:
            original = dict(zip(names, args)) | kwargs
            original.setdefault("src_page_table", None)
            original.setdefault("row_starts", None)
            assert matches == [original], (args, kwargs)


@pytest.mark.parametrize(
    "optional", ["torch.Tensor | None", "typing.Optional[torch.Tensor]"]
)
def test_real_generator_parser_matches_complete_runtime_doc_signature(optional):
    doc = (
        FAST_TOPK_HEADER.removeprefix("def ")
        .removesuffix(":")
        .replace("torch.Tensor | None", optional)
    )
    verify_fast_topk_docstring(doc + "\n\nFast TopK Transform Fused kernel\n")


@pytest.mark.parametrize(
    "old,new",
    [
        ("cu_seqlens_q: torch.Tensor", "cu_seqlens_q: torch.Tensor = None"),
        (
            "src_page_table: torch.Tensor | None = None",
            "src_page_table: torch.Tensor | None",
        ),
        ("cu_seqlens_q: torch.Tensor", "cu_seqlens_q: int"),
        (" -> None:", " -> torch.Tensor:"),
        ("fast_topk_transform_fused", "other_function"),
    ],
)
def test_changed_runtime_doc_signature_is_rejected(old, new):
    doc = FAST_TOPK_HEADER.replace(old, new).removeprefix("def ").removesuffix(":")
    with pytest.raises(ValueError, match="runtime signature changed"):
        verify_fast_topk_docstring(doc)


def test_real_unrelated_builtin_doc_is_not_accepted():
    with pytest.raises(ValueError, match="runtime signature changed"):
        verify_fast_topk_docstring(len.__doc__)


@pytest.mark.parametrize(
    "module", ["other_module", "librtp_compute_ops", FAST_TOPK_MODULE + "_extra"]
)
def test_nontrailing_default_fix_is_exact_module_only(module):
    assert normalize_fast_topk_defaults(FAST_TOPK_STUB, module) == (
        FAST_TOPK_STUB,
        False,
    )
    with pytest.raises(SyntaxError):
        normalize_keyword_field(FAST_TOPK_STUB)


def test_unknown_invalid_signature_and_other_errors_still_fail():
    altered = FAST_TOPK_STUB.replace("lengths: torch.Tensor", "lengths: int")
    assert normalize_fast_topk_defaults(altered, FAST_TOPK_MODULE) == (altered, False)
    with pytest.raises(SyntaxError):
        normalize_keyword_field(altered)
    normalized, _ = normalize_fast_topk_defaults(
        FAST_TOPK_STUB + "bad: invalid::CxxType\n", FAST_TOPK_MODULE
    )
    with pytest.raises(SyntaxError):
        normalize_keyword_field(normalized)


def test_duplicate_or_decorated_observed_signature_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        normalize_fast_topk_defaults(
            FAST_TOPK_STUB + FAST_TOPK_HEADER + "\n    ...\n", FAST_TOPK_MODULE
        )
    decorated = FAST_TOPK_STUB.replace(
        FAST_TOPK_HEADER, "@typing.overload\n" + FAST_TOPK_HEADER
    )
    with pytest.raises(ValueError, match="Unexpected decorated"):
        normalize_fast_topk_defaults(decorated, FAST_TOPK_MODULE)


def test_signature_text_in_docstring_is_not_rewritten():
    source = '"""\n' + FAST_TOPK_HEADER + '\n"""\nvalue: int\n'
    assert normalize_fast_topk_defaults(source, FAST_TOPK_MODULE) == (source, False)
