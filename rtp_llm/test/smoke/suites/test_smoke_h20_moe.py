"""Pytest entry for smoke suite ``smoke_h20_moe``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "moe_masked_fp8_tp2": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_masked_without_deepep_tp2.json",
        "smoke_args": "--moe_strategy fp8_per_block_no_dp_masked --quantization FP8_PER_BLOCK "
        "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 "
        "--reserver_runtime_mem_mb 16005 --seq_size_per_block 64 "
        "--concurrency_limit 64",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_w4a8_int4": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_w4a8_int4_ptpc.json",
        "smoke_args": "--quantization W4A8_INT4_PER_CHANNEL --warm_up 0 --act_type BF16 "
        "--reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_w4a8_int4_compressed": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_w4a8_int4_ptpc_compressed.json",
        "smoke_args": "--quantization W4A8_INT4_PER_CHANNEL_COMPRESSED --warm_up 0 "
        "--act_type BF16 --reserver_runtime_mem_mb 16005 "
        "--seq_size_per_block 64 --concurrency_limit 64",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_deepep_continuous_dp2": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 "
        "--use_deepep_moe 1 --use_deepep_low_latency 0 --dp_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_deepep_ll_tp2": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_tp2_ll.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 "
        "--use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 2 --world_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_deepep_normal_tp2": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_tp2.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 "
        "--use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size "
        "2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_headwise": {
        "task_info": "data/model/qwen3_moe/q_r_30b_py_headwise.json",
        "smoke_args": "",
        "envs": [
            "WARM_UP=0",
            "LOAD_PYTHON_MODEL=1",
            "ACT_TYPE=BF16",
            "TP_SIZE=1",
            "WORLD_SIZE=1",
            "RESERVER_RUNTIME_MEM_MB=8192",
            "MAX_SEQ_LEN=32768",
        ],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_fp8pt_batched_cudagraph": {
        "task_info": "data/model/qwen3_moe/q_r_fp8_per_tensor_batched_cuda_graph_30b.json",
        "smoke_args": "--decode_capture_config '1,2' --reserver_runtime_mem_mb 20000 "
        "--warm_up 0 --act_type BF16 --enable_cuda_graph 1  "
        "--use_deepep_moe 1 --use_deepep_low_latency 1 --dp_size 2 "
        "--ep_size 2",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_fp8pt_load_quant": {
        "task_info": "data/model/qwen3_moe/q_r_fp8_per_tensor_30b_load_quant.json",
        "smoke_args": "--warm_up 0 --act_type BF16 --quantization FP8_DYNAMIC_PER_TENSOR "
        "--use_deepep_moe 1 --use_deepep_low_latency 0 --force_cpu_load_weights "
        "1 --dp_size 2 --ep_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "moe_cp_pd": {
        "task_info": "data/model/qwen3_moe/q_r_30b_fp8_py_cp2.json",
        "smoke_args": {
            "prefill": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 "
            "--reserver_runtime_mem_mb 8192 --role_type PREFILL "
            "--seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 "
            "--world_size 2 --warm_up 0 --use_deepep_moe 1 "
            "--use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER",
            "decode": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 "
            "--reserver_runtime_mem_mb 8192 --role_type DECODE "
            "--seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 "
            "--warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 "
            "--cp_rotate_method PREFILL_CP",
        },
        "envs": {"prefill": [], "decode": ["ACCL_LOW_LATENCY_OPTIMIZE=1"]},
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_moe"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_moe(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
