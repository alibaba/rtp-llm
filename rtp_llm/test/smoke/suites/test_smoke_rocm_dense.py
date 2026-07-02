"""Pytest entry for smoke suite ``smoke_rocm_dense``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "rocm_dense_qwen3_8b_hipgraph_tp2": {
        "task_info": "data/model/qwen3/q_r_new_model_py.json",
        "smoke_args": "--use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 "
        "--warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 "
        "--act_type BF16 --test_block_num 1000 "
        "--reserver_runtime_mem_mb 70000 --enable_cuda_graph 1 "
        "--enable_cuda_graph_debug_mode 1 --decode_capture_config "
        "'1,2,3,4,5,6,7,8' --tp_size 2 --world_size 2",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
    "rocm_dense_qwen3_8b_ptpc": {
        "task_info": "data/model/qwen3/ptpc_q_r_8b.json",
        "smoke_args": "--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 "
        "--use_asm_pa 1 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 "
        "--seq_size_per_block 16 --act_type BF16 --test_block_num 1000 "
        "--reserver_runtime_mem_mb 70000",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_rocm_dense"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_dense(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
