"""Generation-text contracts; these tests do not substitute a Native module."""

import ast
import importlib
import subprocess
import sys

import pytest

from _build.stubgen import (
    normalize_keyword_field,
    validate_generated_module,
    validate_generated_stub,
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
