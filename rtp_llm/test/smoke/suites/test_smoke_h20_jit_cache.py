"""Native pytest adapter for the H20 remote-JIT lifecycle smoke."""

import unittest

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params


SMOKE_CASES = {
    "jit_cache_deepseek_v2_lite": {
        "task_info": "data/model/deepseek_v2/q_r_mla_pymodel.json",
        "smoke_args": "--warm_up 0 --hack_layer_num 2 --load_method scratch "
        "--test_block_num 100 --act_type BF16 --quantization FP8_PER_BLOCK "
        "--seq_size_per_block 64 --tp_size 2 --reuse_cache 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 3600,
    }
}

SUITE_NAME = "smoke_h20_jit_cache"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_jit_cache(test_name: str, test_config: dict, monkeypatch):
    from rtp_llm.utils.test import jit_cache_smoke_test

    monkeypatch.setenv("SMOKE_ARGS", test_config["smoke_args"])
    suite = unittest.TestSuite(
        [jit_cache_smoke_test.JitCacheSmokeTest("test_deepseek_v2_lite")]
    )
    result = unittest.TestResult()
    suite.run(result)
    if result.skipped:
        pytest.fail(f"JIT cache smoke did not execute: {result.skipped[0][1]}")
    assert not result.errors, result.errors
    assert not result.failures, result.failures
