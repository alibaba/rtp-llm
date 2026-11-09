import json
import subprocess
import sys

import pytest

from mock_framework.execution.test.helpers import FRAMEWORK


@pytest.mark.parametrize('rows', [
    [['tokens', [1], 0, 3, 50, []]],
    [['tokens', [1], 1, 3, 50, []], ['tokens', [2], 0, 3, 50, []], ['end', 2]],
    [['tokens', [1], 0, 3, 50, [[1, 0, 2]]], ['end', 1]],
    [['chain', 2, 0, 1], ['request', 2, 0, 3, 50, []], ['end', 1]],
])
def test_indexed_builder_rejects_invalid_data_before_publishing(tmp_path, rows):
    source = tmp_path / 'source.jsonl'
    target = tmp_path / 'traffic.mckidx'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    command = [sys.executable, str(FRAMEWORK / 'scripts/build-indexed-traffic.py'), str(source), str(target)]
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert not target.exists()
