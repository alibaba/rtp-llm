import json
from pathlib import Path
from unittest.mock import patch

import pytest

from artifacts.file_io import write_text
from artifacts.json_io import write_json


@pytest.mark.parametrize('writer,invalid', [(write_text, None), (write_json, {'value': float('nan')})])
def test_serialization_failure_preserves_previous_file_and_removes_temporary(tmp_path, writer, invalid):
    path = tmp_path / 'checkpoint'
    path.write_text('previous complete document')
    with pytest.raises((ValueError, TypeError)):
        writer(path, invalid)
    assert path.read_text() == 'previous complete document'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('writer,value', [(write_text, '正文\n'), (write_json, {'name': '正文'})])
def test_replace_failure_preserves_previous_file_and_removes_temporary(tmp_path, writer, value):
    path = tmp_path / 'checkpoint'
    path.write_text('previous complete document')
    with patch.object(Path, 'replace', side_effect=OSError('cannot publish')):
        with pytest.raises(OSError, match='cannot publish'):
            writer(path, value)
    assert path.read_text() == 'previous complete document'
    assert list(tmp_path.iterdir()) == [path]


def test_json_and_utf8_text_keep_exact_serialization_and_return_values(tmp_path):
    path = tmp_path / 'nested' / 'checkpoint'
    value = {'name': '正文', 'rows': [1, None]}
    for indent in (None, 2):
        assert write_json(str(path), value, indent=indent) == path
        expected = json.dumps(value, indent=indent, allow_nan=False,
                              separators=(',', ':') if indent is None else None)
        assert path.read_text() == expected + ('\n' if indent is not None else '')
    assert write_text(path, '正文\n') is None
    assert path.read_bytes() == '正文\n'.encode('utf-8')
    assert list(path.parent.iterdir()) == [path]
