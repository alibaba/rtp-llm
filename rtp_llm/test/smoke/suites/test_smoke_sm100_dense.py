"""Pytest entry for smoke suite ``smoke_sm100_dense``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "dense_tp1_sm100": {
        "task_info": "data/model/qwen3/q_r_l20a_fp4_tp1_py.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 "
        "--reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 "
        "--concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
        "gpu_type": "SM100_ARM",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "SM100_ARM"],
        "timeout": 600,
    },
    "dense_tp2_sm100": {
        "task_info": "data/model/qwen3/q_r_l20a_fp4_tp2_py.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 "
        "--reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 "
        "--concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
        "gpu_type": "SM100_ARM",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "SM100_ARM"],
        "timeout": 600,
    },
    "fp8_attention_sm100": {
        "task_info": "data/model/qwen3/q_r_block_fp8_sm100.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 "
        "--reserver_runtime_mem_mb 178125 --warm_up 0",
        "gpu_type": "SM100_ARM",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "SM100_ARM"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_sm100_dense"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_sm100_dense(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
