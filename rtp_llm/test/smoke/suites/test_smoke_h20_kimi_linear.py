"""Pytest entry for H20 Kimi Linear smoke coverage."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test


SMOKE_CASES = {
    "kimi_bf16_basic": {
        "task_info": "data/model/kimi_linear/q_r_bf16_tp2.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--ssm_state_dtype fp32 --reserver_runtime_mem_mb 8192",
        "envs": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "kimi_kernel_block": {
        "task_info": "data/model/kimi_linear/q_r_bf16_tp2_kernel_block_size_64.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--kernel_seq_size_per_block 64 --ssm_state_dtype fp32 "
        "--reserver_runtime_mem_mb 8192",
        "envs": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "kimi_cudagraph": {
        "task_info": "data/model/kimi_linear/q_r_cuda_graph.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --max_seq_len 128 "
        "--enable_cuda_graph 1 --warm_up 0 --concurrency_limit 8 "
        "--reserver_runtime_mem_mb 8192 --tp_size 2 --ssm_state_dtype fp32",
        "envs": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "kimi_long_reuse_memcache": {
        "task_info": "data/model/kimi_linear/q_r_bf16_tp2_long_input_reuse_cache.json",
        "smoke_args": "--tp_size 2 --act_type BF16 --max_seq_len 16384 "
        "--seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 "
        "--enable_memory_cache 1 --memory_cache_size_mb 2048 "
        "--write_cache_sync 1 --ssm_state_dtype fp32 "
        "--reserver_runtime_mem_mb 8192",
        "envs": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "kimi_tool_call": {
        "task_info": "data/model/kimi_linear/q_r_bf16_tp2_tool_call.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--ssm_state_dtype fp32 --reserver_runtime_mem_mb 8192",
        "envs": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "kimi_pd": {
        "task_info": "data/model/kimi_linear/q_r_bf16_tp2_pd_sep.json",
        "smoke_args": {
            "prefill": "--seq_size_per_block 2048 --act_type BF16 "
            "--role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 "
            "--tp_size 2 --ssm_state_dtype fp32 --reserver_runtime_mem_mb 8192",
            "decode": "--seq_size_per_block 2048 --act_type BF16 "
            "--role_type DECODE --cache_store_rdma_mode 0 --use_local 1 "
            "--tp_size 2 --ssm_state_dtype fp32 --reserver_runtime_mem_mb 8192",
        },
        "envs": {
            "prefill": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
            "decode": ["TRITON_AUTOTUNE_CACHE_MODE=cached"],
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_kimi_linear"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_kimi_linear(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
