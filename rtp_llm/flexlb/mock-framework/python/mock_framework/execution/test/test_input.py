import struct

import zstandard
from mock_framework.execution.test.helpers.input import write_inputs


def test_output_cap_dictionary_and_observed_zero_are_independent(tmp_path):
    target = tmp_path / "traffic.mckidx"
    write_inputs(
        target,
        [
            (1, 0, {"observed_output_tokens": 37, "max_output_tokens": 8192}, 50, []),
            (1, 1, {"max_output_tokens": 8192}, 50, []),
            (1, 2, {"observed_output_tokens": 0}, 50, []),
            (1, 3, {}, 50, []),
        ],
    )
    data = target.read_bytes()
    magic, count, _, _, _, _, caps, payload, _, missing, _ = struct.unpack_from(
        ">8s9q32s", data
    )
    assert (magic, count, caps, missing) == (b"MCKINDEX", 4, 1, 1)
    assert struct.unpack_from(">i", data, payload - 8)[0] == 8192
    offset, size, raw_size = struct.unpack_from(">qii", data, 112)
    page = zstandard.ZstdDecompressor().decompress(
        data[payload + offset : payload + offset + size]
    )
    assert len(page) == raw_size
    outputs = []
    for i in range(count):
        (offset,) = struct.unpack_from(">I", page, i * 4)
        _, _, observed, _, cap_id = struct.unpack_from(
            ">qqiiH", page, count * 4 + offset
        )
        outputs.append((observed, cap_id))
    assert outputs == [(37, 1), (-1, 1), (0, 0), (-1, 0)]
