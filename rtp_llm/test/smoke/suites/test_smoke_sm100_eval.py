"""Pytest entry for smoke suite ``smoke_sm100_eval``."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "qwen3_moe_tau2_bench_sm100": {
        "task_info": "data/model/qwen3_moe/q_r_30b_tau2_bench_tp2_sm100.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --max_seq_len 16384 "
        "--reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 "
        "--seq_size_per_block 64 --concurrency_limit 64 "
        "--quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 "
        "--use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 "
        "--world_size 2",
        "gpu_type": "SM100_ARM",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "SM100_ARM", "eval"],
        "timeout": 6000,
    },
}

SUITE_NAME = "smoke_sm100_eval"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(6000)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_sm100_eval(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
