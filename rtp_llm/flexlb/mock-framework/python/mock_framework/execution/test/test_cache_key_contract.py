"""Check Mock keys against the real text path and the DSV4.1 media span contract."""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mock_framework.execution.test.helpers import FRAMEWORK

ROOT = FRAMEWORK.parents[2]


@pytest.fixture(scope="module")
def native_hash(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-dsv-hash")
    source = directory / "hash.cc"
    source.write_text(
        """#include <cstdint>
#include <vector>
#include "rtp_llm/cpp/utils/HashUtil.h"
extern "C" int64_t hash_array(int64_t seed, const int32_t* values, int count) {
    return rtp_llm::hashInt64Array(seed, values, values + count);
}
"""
    )
    library = directory / "hash.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fwrapv",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    function = dll.hash_array
    function.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    function.restype = ctypes.c_int64

    def blocks(chunks, seed=0):
        result = []
        for chunk in chunks:
            values = (ctypes.c_int32 * len(chunk))(*chunk)
            seed = function(seed, values, len(chunk))
            result.append(seed)
        return result

    return blocks


def media_span_segments(tokens, spans, block_size):
    # DSV4.1 hashes the token block, then the overlapping media rows and their
    # positions. The main Frontend uses image digests instead of row hashes.
    segments = []
    for begin in range(0, len(tokens) - block_size + 1, block_size):
        identity = []
        for start, rows in spans:
            first = max(begin, start)
            end = min(begin + block_size, start + len(rows))
            if first < end:
                identity.extend(
                    [-42, first - begin, len(rows), first - start, end - first]
                )
                identity.extend(rows[first - start : end - start])
        segments.extend([tokens[begin : begin + block_size], identity])
    return segments


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 0x4453563431494631, 0x4453563431494231])
@pytest.mark.parametrize(
    "tokens,spans,block",
    [
        ([1, 2, 3, 4, 5], [], 2),
        ([-2147483648, -1, 0, 2147483647], [], 2),
        (
            [10, -200, -200, -200, -200, 20, 30],
            [(1, [-1, 42, -2147483648, 2147483647])],
            2,
        ),
        (
            [10, -200, -200, -200, -200, 20, 30],
            [(1, [-1, 43, -2147483648, 2147483647])],
            2,
        ),
        ([10, -200, -200, -200, -200, 20], [(1, [11, 12]), (3, [21, 22])], 2),
        ([1, 2, 3], [], 16),
        (list(range(1, 1101)), [], 128),
        ([10] + [-200] * 700 + list(range(100, 600)), [(1, list(range(700)))], 128),
    ],
)
def test_mock_signed_block_keys_match_text_and_media_contracts(
    native_hash, monkeypatch, tokens, spans, block, seed, stride
):
    monkeypatch.setitem(
        sys.modules,
        "rtp_llm.ops",
        SimpleNamespace(cpp_get_block_cache_keys=native_hash),
    )
    spec = importlib.util.spec_from_file_location(
        "real_dsv_cache_keys", ROOT / "rtp_llm/server/cache_key_routing.py"
    )
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    if spans:
        expected = native_hash(media_span_segments(tokens, spans, block), seed)[1::2]
    else:
        expected = frontend.get_block_cache_keys(tokens, block, cache_key_seed=seed)
    java = (
        str(Path(os.environ["JAVA_HOME"]) / "bin/java")
        if "JAVA_HOME" in os.environ
        else shutil.which("java")
    )
    classpath = os.pathsep.join(
        str(FRAMEWORK / entry)
        for entry in (
            "java/protocol/target/test-classes",
            "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar",
        )
    )
    completed = subprocess.run(
        [
            java,
            "-cp",
            classpath,
            "org.flexlb.mock.protocol.CacheKeyProbe",
            str(block),
            ",".join(map(str, tokens)),
            ";".join(
                f"{offset}:" + ",".join(map(str, hashes)) for offset, hashes in spans
            ),
            str(seed),
            str(stride),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = [int(value) for value in completed.stdout.strip().split(",") if value]
    assert actual == frontend.route_cache_keys_for_page_rr(expected, True, stride)
