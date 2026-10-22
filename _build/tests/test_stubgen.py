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


# Exact post header is in CI 82334777. Pre is derived from the actual binding
# and C++ declaration; the real generator parser checks both forms below.
MOE_HEADERS = {
    "moe_pre_reorder": (
        "def moe_pre_reorder(input: torch.Tensor, topk_ids: torch.Tensor, "
        "token_expert_indices: torch.Tensor, expert_map: torch.Tensor | None = None, "
        "n_expert: int, n_local_expert: int, topk: int, align_block_size: int | None = None, "
        "permuted_input: torch.Tensor, expert_first_token_offset: torch.Tensor, "
        "inv_permuted_idx: torch.Tensor, permuted_idx: torch.Tensor) -> None:"
    ),
    "moe_post_reorder": (
        "def moe_post_reorder(permuted_hidden_states: torch.Tensor, topk_weights: torch.Tensor, "
        "inv_permuted_idx: torch.Tensor, expert_first_token_offset: torch.Tensor | None = None, "
        "topk: int, hidden_states: torch.Tensor) -> None:"
    ),
}


def _moe_stub(name):
    return (
        "from __future__ import annotations\nimport torch\n"
        + MOE_HEADERS[name]
        + '\n    """Original MoE documentation."""\n'
    )


@pytest.mark.parametrize("name", list(MOE_HEADERS))
def test_moe_exact_overloads_preserve_required_arguments_and_documentation(name):
    from _build.stubgen import normalize_moe_defaults

    source = _moe_stub(name)
    with pytest.raises(SyntaxError, match="non-default argument follows default"):
        ast.parse(source)
    normalized, changed = normalize_moe_defaults(source, FAST_TOPK_MODULE)
    assert changed == [name]
    functions = [
        node for node in ast.parse(normalized).body if isinstance(node, ast.FunctionDef)
    ]
    assert len(functions) == (4 if name == "moe_pre_reorder" else 2)
    assert all(
        ast.unparse(node.decorator_list[0]) == "typing.overload" for node in functions
    )
    assert all(
        not node.args.defaults and all(value is None for value in node.args.kw_defaults)
        for node in functions
    )
    assert all(ast.unparse(node.returns) == "None" for node in functions)
    assert normalized.count('"""Original MoE documentation."""') == 1
    assert normalize_moe_defaults(normalized, FAST_TOPK_MODULE) == (normalized, [])


@pytest.mark.parametrize("name", list(MOE_HEADERS))
def test_moe_overloads_preserve_all_positional_keyword_call_shapes(name):
    """Standard Signature binding checks call shapes, not Native execution."""
    from _build.stubgen import normalize_moe_defaults

    normalized, _ = normalize_moe_defaults(_moe_stub(name), FAST_TOPK_MODULE)
    functions = [
        node for node in ast.parse(normalized).body if isinstance(node, ast.FunctionDef)
    ]
    signatures = [
        inspect.Signature(
            [
                *[
                    inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for arg in node.args.args
                ],
                *[
                    inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY)
                    for arg in node.args.kwonlyargs
                ],
            ]
        )
        for node in functions
    ]
    names = [arg.arg for arg in functions[0].args.args]
    optional = (
        {"expert_map", "align_block_size"}
        if name == "moe_pre_reorder"
        else {"expert_first_token_offset"}
    )
    required = set(names) - optional
    for positional_count, presence in itertools.product(
        range(len(names) + 2), itertools.product([False, True], repeat=len(names) + 1)
    ):
        args = tuple(
            None if i < len(names) and names[i] in optional else i
            for i in range(positional_count)
        )
        kwargs = {
            key: None if key in optional else key
            for key, included in zip(names + ["unknown"], presence)
            if included
        }
        expected = (
            positional_count <= len(names)
            and "unknown" not in kwargs
            and not (set(names[:positional_count]) & set(kwargs))
            and required <= (set(names[:positional_count]) | set(kwargs))
        )
        matches = []
        for signature in signatures:
            try:
                values = dict(signature.bind(*args, **kwargs).arguments)
            except TypeError:
                continue
            for key in optional:
                values.setdefault(key, None)
            matches.append(values)
        assert bool(matches) == expected, (name, args, kwargs)
        if expected:
            original = dict(zip(names, args)) | kwargs
            for key in optional:
                original.setdefault(key, None)
            assert matches == [original], (name, args, kwargs)


@pytest.mark.parametrize("name", list(MOE_HEADERS))
@pytest.mark.parametrize("optional_style", ["union", "typing"])
def test_moe_real_parser_accepts_exact_binding_doc(name, optional_style):
    from _build.stubgen import verify_moe_docstring

    doc = MOE_HEADERS[name].removeprefix("def ").removesuffix(":")
    if optional_style == "typing":
        doc = doc.replace(
            "torch.Tensor | None", "typing.Optional[torch.Tensor]"
        ).replace("int | None", "typing.Optional[int]")
    verify_moe_docstring(name, doc + "\n\nOriginal MoE documentation.\n")


@pytest.mark.parametrize("name", list(MOE_HEADERS))
@pytest.mark.parametrize(
    "change", ["required_default", "optional_required", "type", "return", "name"]
)
def test_moe_changed_runtime_signature_fails_closed(name, change):
    from _build.stubgen import verify_moe_docstring

    header = MOE_HEADERS[name]
    replacements = {
        "required_default": ("topk: int", "topk: int = None"),
        "optional_required": (" | None = None", " | None"),
        "type": ("topk: int", "topk: float"),
        "return": (" -> None:", " -> torch.Tensor:"),
        "name": (name, name + "_other"),
    }
    old, new = replacements[change]
    doc = header.replace(old, new).removeprefix("def ").removesuffix(":")
    with pytest.raises(ValueError, match="runtime signature changed"):
        verify_moe_docstring(name, doc)


@pytest.mark.parametrize("name", list(MOE_HEADERS))
def test_moe_duplicate_decorated_wrong_module_and_doc_literal_are_rejected(name):
    from _build.stubgen import normalize_moe_defaults

    source = _moe_stub(name)
    with pytest.raises(ValueError, match="Ambiguous"):
        normalize_moe_defaults(
            source + MOE_HEADERS[name] + "\n    ...\n", FAST_TOPK_MODULE
        )
    with pytest.raises(ValueError, match="Unexpected decorated"):
        normalize_moe_defaults(
            source.replace(MOE_HEADERS[name], "@typing.overload\n" + MOE_HEADERS[name]),
            FAST_TOPK_MODULE,
        )
    for module in ["librtp_compute_ops", FAST_TOPK_MODULE + "_extra", "other_module"]:
        assert normalize_moe_defaults(source, module) == (source, [])
    doc_literal = '"""\n' + MOE_HEADERS[name] + '\n"""\nvalue: int\n'
    assert normalize_moe_defaults(doc_literal, FAST_TOPK_MODULE) == (doc_literal, [])


@pytest.mark.parametrize("name", list(MOE_HEADERS))
def test_moe_unknown_invalid_signature_or_other_errors_still_fail(name, tmp_path):
    from _build.stubgen import normalize_moe_defaults

    source = _moe_stub(name).replace("topk: int", "topk: float")
    assert normalize_moe_defaults(source, FAST_TOPK_MODULE) == (source, [])
    path = tmp_path / "unknown.pyi"
    path.write_text(source)
    with pytest.raises(SyntaxError):
        validate_generated_stub(str(path), FAST_TOPK_MODULE)
    assert path.read_text() == source
    normalized, _ = normalize_moe_defaults(
        _moe_stub(name) + "bad: invalid::CxxType\n", FAST_TOPK_MODULE
    )
    with pytest.raises(SyntaxError):
        normalize_keyword_field(normalized)


def test_all_three_exact_nontrailing_signatures_compose_without_hiding_errors():
    from _build.stubgen import normalize_moe_defaults

    source = FAST_TOPK_STUB + "\n".join(
        _moe_stub(name).split("import torch\n", 1)[1] for name in MOE_HEADERS
    )
    normalized, fast = normalize_fast_topk_defaults(source, FAST_TOPK_MODULE)
    normalized, moe = normalize_moe_defaults(normalized, FAST_TOPK_MODULE)
    assert fast and moe == list(MOE_HEADERS)
    normalize_keyword_field(normalized)
    assert (
        len(
            [
                node
                for node in ast.parse(normalized).body
                if isinstance(node, ast.FunctionDef)
            ]
        )
        == 8
    )


def test_moe_runtime_gate_precedes_file_write_without_native_substitute(
    tmp_path, monkeypatch
):
    from _build.stubgen import normalize_moe_defaults

    # An ordinary existing module is not a Native binding; no fake success path.
    path = tmp_path / "ordinary.pyi"
    path.write_text(_moe_stub("moe_post_reorder"))
    original = path.read_bytes()
    assert normalize_moe_defaults(path.read_text(), "builtins")[1] == []
    with pytest.raises(SyntaxError):
        validate_generated_stub(str(path), "builtins")
    assert path.read_bytes() == original
    # Force only a failed import, without assuming the environment lacks Native.
    # This never supplies a substitute binding or a successful runtime proof.
    original_import = importlib.import_module

    def fail_native_import(name, *args, **kwargs):
        if name == FAST_TOPK_MODULE:
            raise ModuleNotFoundError("controlled Native import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fail_native_import)
    with pytest.raises(ModuleNotFoundError, match="controlled Native import failure"):
        validate_generated_stub(str(path), FAST_TOPK_MODULE)
    assert path.read_bytes() == original
