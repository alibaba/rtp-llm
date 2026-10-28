"""Pytest entry for smoke suite ``smoke_rocm_basic``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "rocm_basic_qwen25_generation_prefill_cuda_graph": {
        "task_info": "data/model/qwen25/q_r_generation_prefill_cuda_graph_mi308x.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --seq_size_per_block 16 --test_block_num 1000 "
        "--concurrency_limit 5 --max_context_batch_size 5 --reuse_cache 0 --use_aiter_pa 1 "
        "--use_asm_pa 1 --use_triton_pa 1 --disable_flash_infer 1 --enable_cuda_graph 1 "
        "--enable_cuda_graph_debug_mode 1 --decode_capture_config '1' "
        "--generation_prefill_cuda_graph_max_requests 5 "
        "--generation_prefill_capture_config '64,128,256,384,512,768,1024'",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
    "rocm_basic_cache_reuse": {
        "task_info": "data/model/qwen2/q_r_reuse.json",
        "smoke_args": "--reuse_cache 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa "
        "1 --act_type FP16",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
    "rocm_basic_batch_cache_reuse": {
        "task_info": "data/model/qwen3/q_r_308x_batch_cache.json",
        "smoke_args": "--reuse_cache 1 --enable_cuda_graph 1 --seq_size_per_block 16 "
        "--use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
    "rocm_basic_beam_search_tp2": {
        "task_info": "data/model/qwen25/bs_q_r_mi308x.json",
        "smoke_args": "--tp_size 2 --warm_up 0 --seq_size_per_block 16 "
        "--use_asm_pa 0 --use_aiter_pa 1 --disable_flash_infer 1 "
        "--act_type BF16",
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_rocm_basic"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_basic(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
