"""Pytest entry for ROCm Qwen3.5 interleaved MRoPE graph coverage."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test


SMOKE_CASES = {
    "rocm_qwen35_bf16_mrope_cg": {
        "task_info": "data/model/qwen35/qwen35_bf16_rocm.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --seq_size_per_block 1024 "
        "--kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 "
        "--tp_size 1 --world_size 1 --use_asm_pa 0 --use_aiter_pa 1 "
        "--use_triton_pa 1 --reserver_runtime_mem_mb 40480 --enable_cuda_graph 1 "
        "--enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4' "
        "--reuse_cache 1",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
    "rocm_qwen35_bf16_mrope_cg_asm_pa": {
        "task_info": "data/model/qwen35/qwen35_bf16_rocm.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --seq_size_per_block 1024 "
        "--kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 "
        "--tp_size 1 --world_size 1 --use_asm_pa 1 --use_aiter_pa 1 "
        "--use_triton_pa 1 --reserver_runtime_mem_mb 40480 --enable_cuda_graph 1 "
        "--enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4' "
        "--reuse_cache 1",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_rocm_qwen35_mrope_cg"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_qwen35_mrope(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
