import json
import struct
import subprocess
import sys

import pytest

from mock_framework.execution.test.helpers import FRAMEWORK
from mock_framework.execution.test.helpers.input import write_inputs


@pytest.mark.parametrize('rows', [
    [['tokens', [1], 0, {"max_output_tokens": 3}, 50, []]],
    [['tokens', [1], 1, {"max_output_tokens": 3}, 50, []], ['tokens', [2], 0, {"max_output_tokens": 3}, 50, []], ['end', 2]],
    [['tokens', [1], 0, {"max_output_tokens": 3}, 50, [[1, 0, 2]]], ['end', 1]],
    [['tokens', [1], 0, {"observed_output_tokens": 4, "max_output_tokens": 3}, 50, []], ['end', 1]],
    [['tokens', [1], 0, {"default_max_output_tokens": 3}, 50, []], ['end', 1]],
    [['chain', 2, 0, 1], ['request', 2, 0, {"max_output_tokens": 3}, 50, []], ['end', 1]],
])
def test_indexed_builder_rejects_invalid_data_before_publishing(tmp_path, rows):
    source = tmp_path / 'source.jsonl'
    target = tmp_path / 'traffic.mckidx'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    command = [sys.executable, str(FRAMEWORK / 'scripts/build-indexed-traffic.py'), str(source), str(target)]
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert not target.exists()


def test_output_cap_dictionary_and_observed_zero_are_independent(tmp_path):
    target = tmp_path / 'traffic.mckidx'
    write_inputs(target, [
        ([1], 0, {"observed_output_tokens": 37, "max_output_tokens": 8192}, 50, []),
        ([1], 1, {"max_output_tokens": 8192}, 50, []),
        ([1], 2, {"observed_output_tokens": 0}, 50, []),
        ([1], 3, {}, 50, []),
    ])
    data = target.read_bytes()
    magic, count, _, _, _, _, caps, payload, _, missing, _ = struct.unpack_from(">8s9q32s", data)
    assert (magic, count, caps, missing) == (b"MCKINDEX", 4, 1, 1)
    assert struct.unpack_from(">i", data, payload - 8)[0] == 8192
    outputs = []
    for i in range(count):
        offset, _ = struct.unpack_from(">qi", data, 112 + i * 16)
        _, _, observed, _, cap_id = struct.unpack_from(">qqiiH", data, payload + offset)
        outputs.append((observed, cap_id))
    assert outputs == [(37, 1), (-1, 1), (0, 0), (-1, 0)]
