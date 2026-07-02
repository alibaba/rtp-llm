"""Pytest entry for smoke suite ``smoke_h20_mla``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "mla_fp8_redundant_expert_tp2": {
        "task_info": "data/model/deepseek_v2/q_r_3090_mla_r24.json",
        "smoke_args": "--masked_max_token_num 0 --redundant_expert 24 --act_type BF16 "
        "--quantization FP8_PER_BLOCK --tp_size 2 --world_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_fp8_reuse_absorb_tp2": {
        "task_info": "data/model/deepseek_v2/q_r_3090_mla.json",
        "smoke_args": "--load_method scratch --reuse_cache 1 --seq_size_per_block 8 "
        "--act_type BF16 --quantization FP8_PER_BLOCK --absorb_opt_len 1 "
        "--tp_size 2 --world_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_noquant_dp2": {
        "task_info": "data/model/deepseek_v2/q_r_mla_pymodel.json",
        "smoke_args": "--seq_size_per_block 8 --act_type BF16 --tp_size 1 --dp_size 2 --world_size "
        "2 --reserver_runtime_mem_mb 16697",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_cudagraph_pad_reuse": {
        "task_info": "data/model/deepseek_v2/q_r_mla_pymodel.json",
        "smoke_args": "--warm_up 0 --test_block_num 1000 --tp_size 1 --world_size 1 "
        "--reuse_cache 1 --seq_size_per_block 64 --act_type BF16 "
        "--decode_capture_config '2' --enable_cuda_graph 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_cudagraph_fp8pt_deepep_dp2": {
        "task_info": "data/model/deepseek_v2/q_r_mla_cudagraph_per_tensor.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--enable_cuda_graph 1 --quantization FP8_DYNAMIC_PER_TENSOR "
        "--use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 "
        "--world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_cudagraph_fp8pb_deepep_dp2": {
        "task_info": "data/model/deepseek_v2/q_r_mla_pymodel_cudagraph.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--enable_cuda_graph 1 --quantization FP8_PER_BLOCK "
        "--use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 "
        "--world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
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
    "mla_cudagraph_deepep_tp2": {
        "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_cuda_graph.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--enable_cuda_graph 1 --reserver_runtime_mem_mb 20000 --tp_size 2 "
        "--world_size 2 --dp_size 1 --fp8_kv_cache 1 --use_deepep_moe 1 "
        "--use_deepep_low_latency 1",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_fp8_basic": {
        "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 0 "
        "--tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_not_fast_path_reuse": {
        "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_long.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--reserver_runtime_mem_mb 49343 --enable_cuda_graph 0 --reuse_cache "
        "1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 "
        "--fp8_kv_cache 1",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "mla_fast_path_reuse": {
        "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_66_seq.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph "
        "0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 "
        "--dp_size 1 --fp8_kv_cache 0",
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
    "mla_load_quant_tp2": {
        "task_info": "data/model/deepseek-r1-4layer/r1_fp8_q_r_h20.json",
        "smoke_args": "--cache_store_rdma_mode 0 --use_local 1 --seq_size_per_block 64 "
        "--decode_entrance 1 --act_type bf16 --quantization FP8_PER_BLOCK "
        "--tp_size 2 --reserver_runtime_mem_mb 5026",
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
