"""Pytest entry for smoke suite ``smoke_h20_mla``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "mla_kernel_block_size": {
        "task_info": "data/model/glm5/glm_5_fp8_q_r_h20.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 512 --act_type BF16 "
        "--enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1 "
        "--fp8_kv_cache 1 --kernel_seq_size_per_block 64",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_cp_pd": {
        "task_info": "data/model/glm5/glm_5_fp8_q_r_h20_cp.json",
        "smoke_args": {
            "prefill": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL "
            "--seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 "
            "--world_size 2 --warm_up 0 --use_deepep_moe 1 "
            "--use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER "
            "--use_all_gather=0",
            "decode": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE "
            "--seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 "
            "--warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 "
            "--cp_rotate_method PREFILL_CP --use_all_gather=0",
        },
        "envs": {"prefill": [], "decode": []},
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_pure_cp_pd": {
        "task_info": "data/model/glm5/glm_5_fp8_q_r_h20_cp.json",
        "smoke_args": {
            "prefill": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL "
            "--seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 "
            "--world_size 2 --warm_up 0 --use_deepep_moe 0 --use_all_gather 1 "
            "--moe_strategy fp8_per_block_pure_cp --cp_rotate_method ALL_GATHER",
            "decode": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 "
            "--use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE "
            "--seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 "
            "--warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 "
            "--cp_rotate_method PREFILL_CP --use_all_gather 0",
        },
        "envs": {"prefill": [], "decode": []},
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_glm4_moe_lite": {
        "task_info": "data/model/glm4_moe_lite/q_r_h20.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --prefill_chunk_size 64 --act_type BF16 "
        "--enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_mla"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_mla(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
