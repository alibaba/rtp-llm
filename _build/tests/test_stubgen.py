"""Generation-text contracts; these tests do not substitute a Native module."""
import ast

import pytest

from _build.stubgen import normalize_keyword_field


STUB = '''from __future__ import annotations
import typing
class QuantMethod:
    None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>
    Awq: typing.ClassVar[QuantMethod]
    __members__: typing.ClassVar[dict[str, QuantMethod]]
    def __int__(self) -> int: ...
class Other:
    def method(self, value: int = 3) -> None: ...
'''


def test_keyword_is_documented_without_alias_or_lost_typed_map():
    with pytest.raises(SyntaxError):
        ast.parse(STUB)
    normalized, changed = normalize_keyword_field(STUB)
    assert changed
    ast.parse(normalized)
    original = '    None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>\n'
    replacement = ('    # Runtime keyword member: QuantMethod.__members__["None"].\n'
                   '    # None: typing.ClassVar[QuantMethod]  # value = <QuantMethod.None: 0>\n')
    assert normalized.replace(replacement, original) == STUB
    assert 'None_' not in normalized
    assert '    __members__: typing.ClassVar[dict[str, QuantMethod]]\n' in normalized


def test_valid_source_is_byte_identical():
    valid = 'from __future__ import annotations\nclass Plain:\n    value: int\n'
    assert normalize_keyword_field(valid) == (valid, False)


def test_normalization_is_idempotent():
    normalized, _ = normalize_keyword_field(STUB)
    assert normalize_keyword_field(normalized) == (normalized, False)


@pytest.mark.parametrize("invalid", [
    STUB + 'def f(value=QuantMethod.None): ...\n',
    STUB + 'value: QuantMethod.None\n',
    STUB + 'class OtherEnum:\n    None: int\n',
    STUB.replace('Awq: typing.ClassVar[QuantMethod]', 'Awq: invalid::CxxType'),
])
def test_other_keywords_references_and_defaults_still_fail(invalid):
    with pytest.raises(SyntaxError):
        normalize_keyword_field(invalid)


def test_keyword_requires_existing_typed_members_map():
    without_map = STUB.replace('    __members__: typing.ClassVar[dict[str, QuantMethod]]\n', '')
    with pytest.raises(ValueError, match='existing typed __members__'):
        normalize_keyword_field(without_map)


def test_duplicate_keyword_is_rejected():
    duplicate = STUB.replace('    Awq:', '    None: typing.ClassVar[QuantMethod]\n    Awq:')
    with pytest.raises(ValueError, match='Ambiguous'):
        normalize_keyword_field(duplicate)


def test_different_keyword_annotation_is_not_silently_dropped():
    with pytest.raises(SyntaxError):
        normalize_keyword_field(STUB.replace('None: typing.ClassVar[QuantMethod]', 'None: int'))


def test_name_in_docstring_is_unchanged():
    source = 'class Other:\n    """None: typing.ClassVar[QuantMethod]"""\n    field: int\n'
    assert normalize_keyword_field(source) == (source, False)
