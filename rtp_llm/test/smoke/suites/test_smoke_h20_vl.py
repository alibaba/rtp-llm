"""Pytest entry for H20 multimodal smoke coverage."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test


SMOKE_CASES = {
    "qwen3_vl": {
        "task_info": "data/model/qwen_vl/q_r_3.json",
        "smoke_args": {
            "llm": "--act_type BF16 --use_local 1 --tp_size 2 --reuse_cache 1",
            "vit": "--act_type BF16 --use_local 1 --use_local_preprocess 1",
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "qwen3_vl_cp2": {
        "task_info": "data/model/qwen_vl/q_r_3_cp2.json",
        "smoke_args": {
            "prefill": "--warm_up 0 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --use_local_preprocess 1 --role_type PREFILL "
            "--tp_size 2 --world_size 2 --dp_size 1 --reuse_cache 1 "
            "--enable_cuda_graph 0 --cp_rotate_method ALL_GATHER",
            "decode": "--warm_up 0 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --role_type DECODE --tp_size 2 --world_size 2 "
            "--dp_size 1 --reuse_cache 1 --enable_cuda_graph 0 "
            "--cp_rotate_method PREFILL_CP",
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "qwen3_vl_gpu_batch": {
        "task_info": "data/model/qwen_vl/q_r_3_gpu_batch.json",
        "smoke_args": {
            "llm": "--act_type BF16 --use_local 1 --tp_size 2 --reuse_cache 0",
            "vit": "--act_type BF16 --use_local 1 --use_local_preprocess 1 "
            "--gpu_batch_wait_ms 500 --gpu_max_batch_size 8 --mm_cache_item_num 0",
        },
        "concurrency_test": True,
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "qwen3_vl_moe": {
        "task_info": "data/model/qwen_vl/q_r_3_moe.json",
        "smoke_args": "--act_type BF16 --use_local 1 --enable_xqa off",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "qwen35_moe_vl_fp8": {
        "task_info": "data/model/qwen35/q_r_35b_moe_vl_fp8.json",
        "smoke_args": {
            "prefill": "--use_local 1 --role_type PREFILL --tp_size 2 "
            "--act_type BF16 --seq_size_per_block 2048 --max_seq_len 8192 "
            "--enable_cuda_graph 0 --warm_up 0 --concurrency_limit 8 "
            "--reserver_runtime_mem_mb 8192",
            "decode": "--use_local 1 --role_type DECODE --tp_size 2 "
            "--act_type BF16 --seq_size_per_block 2048 --max_seq_len 8192 "
            "--enable_cuda_graph 1 --warm_up 0 --concurrency_limit 8 "
            "--reserver_runtime_mem_mb 8192 --use_deepep_moe 1 "
            "--use_deepep_low_latency 1",
        },
        "envs": {
            "prefill": [
                "ACCL_LOW_LATENCY_OPTIMIZE=1",
                "DSV4_FP8_QUANT_KERNEL=legacy",
            ],
            "decode": [
                "ACCL_LOW_LATENCY_OPTIMIZE=1",
                "DSV4_FP8_QUANT_KERNEL=legacy",
            ],
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_vl"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_vl(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
