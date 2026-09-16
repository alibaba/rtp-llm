"""Keep mainline kernel benchmarks and their strict speed assertions runnable."""

from pathlib import Path
import subprocess
import sys

import pytest

_CASES = [
    pytest.param(
        "test_topk_v3",
        2,
        marks=pytest.mark.gpu(type="L20D_TEST", count=2),
        id="test_topk_v3_perf",
    ),
    pytest.param(
        "test_dsv4_persistent_topk",
        1,
        marks=pytest.mark.gpu(type="L20D_TEST", count=1),
        id="test_dsv4_persistent_topk_perf",
    ),
    pytest.param(
        "test_fused_decode_meta_comprehensive",
        1,
        marks=pytest.mark.gpu(type="L20D_TEST", count=1),
        id="test_fused_decode_meta_perf",
    ),
]


@pytest.mark.manual
@pytest.mark.perf
@pytest.mark.cuda
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("module,gpu_count", _CASES)
def test_perf_dsv4_kernel(module, gpu_count):
    from rtp_llm.test.cuda13_preflight import prepare_cuda13_runtime

    prepare_cuda13_runtime(gpu_count)
    module = "rtp_llm.models_py.modules.dsv4.fp8.test." + module
    result = subprocess.run([sys.executable, "-m", module], timeout=14300, check=False)
    assert (
        result.returncode == 0
    ), f"Kernel correctness or performance assertion failed: {module}"
