"""Native pytest adapter for the ROCm remote-JIT lifecycle smoke."""

import unittest

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params


SMOKE_CASES = {
    "jit_cache_qwen3_rocm": {
        "task_info": "data/model/qwen3/q_r_new_model_py.json",
        "smoke_args": "--warm_up 0 --use_swizzleA 1 --use_asm_pa 1 "
        "--disable_flashinfer_native 1 --use_aiter_pa 1 --seq_size_per_block 16 "
        "--act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 3600,
    }
}

SUITE_NAME = "smoke_rocm_jit_cache"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_jit_cache(test_name: str, test_config: dict, monkeypatch):
    from rtp_llm.utils.test import jit_cache_smoke_test

    monkeypatch.setenv("SMOKE_ARGS", test_config["smoke_args"])
    suite = unittest.TestSuite(
        [jit_cache_smoke_test.JitCacheSmokeTest("test_qwen3_rocm")]
    )
    result = unittest.TestResult()
    suite.run(result)
    if result.skipped:
        pytest.fail(f"JIT cache smoke did not execute: {result.skipped[0][1]}")
    assert not result.errors, result.errors
    assert not result.failures, result.failures
