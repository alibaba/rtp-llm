from mock_framework.execution.test.helpers import FRAMEWORK
"""Compare real DSV Frontend + C++ hash implementation with Mock Client/Engine keys."""
import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = FRAMEWORK.parents[2]


@pytest.fixture(scope='module')
def native_hash(tmp_path_factory):
    directory = tmp_path_factory.mktemp('real-dsv-hash')
    source = directory / 'hash.cc'
    source.write_text('''#include <cstdint>
#include <vector>
#include "rtp_llm/cpp/utils/HashUtil.h"
extern "C" int64_t hash_array(int64_t seed, const int32_t* values, int count) {
    return rtp_llm::hashInt64Array(seed, values, values + count);
}
''')
    library = directory / 'hash.so'
    subprocess.run(['c++', '-std=c++17', '-fwrapv', '-shared', '-fPIC', '-I', str(ROOT), str(source), '-o', str(library)], check=True)
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


@pytest.mark.parametrize('seed', [0, 0x4453563431494631, 0x4453563431494231])
@pytest.mark.parametrize('tokens,spans,block', [
    ([1, 2, 3, 4, 5], [], 2),
    ([-2147483648, -1, 0, 2147483647], [], 2),
    ([10, -200, -200, -200, -200, 20, 30], [(1, [-1, 42, -2147483648, 2147483647])], 2),
    ([10, -200, -200, -200, -200, 20, 30], [(1, [-1, 43, -2147483648, 2147483647])], 2),
    ([10, -200, -200, -200, -200, 20], [(1, [11, 12]), (3, [21, 22])], 2),
    ([1, 2, 3], [], 16),
])
def test_real_frontend_and_mock_have_identical_signed_block_keys(native_hash, monkeypatch, tokens, spans, block, seed):
    monkeypatch.setitem(sys.modules, 'rtp_llm.ops', SimpleNamespace(cpp_get_block_cache_keys=native_hash))
    spec = importlib.util.spec_from_file_location('real_dsv_cache_keys', ROOT / 'rtp_llm/server/cache_key_routing.py')
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    expected = frontend.get_block_cache_keys(tokens, block, multimodal_cache_spans=spans, cache_key_seed=seed)
    java = str(Path(os.environ['JAVA_HOME']) / 'bin/java') if 'JAVA_HOME' in os.environ else shutil.which('java')
    classpath = os.pathsep.join(str(FRAMEWORK / 'java/shared/cache/target' / entry) for entry in ('test-classes', 'classes'))
    completed = subprocess.run([java, '-cp', classpath, 'org.flexlb.mock.cache.CacheKeyProbe', str(block),
        ','.join(map(str, tokens)), ';'.join(f'{offset}:' + ','.join(map(str, hashes)) for offset, hashes in spans), str(seed)],
        capture_output=True, text=True, check=True)
    actual = [int(value) for value in completed.stdout.strip().split(',') if value]
    assert actual == expected
    assert frontend.route_cache_keys_for_page_rr(actual, True, 2) == expected[1::2]
