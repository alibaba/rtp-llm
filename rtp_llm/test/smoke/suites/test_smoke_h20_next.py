"""Pytest entry for smoke suite ``smoke_h20_next``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "next_mtp_basic": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 "
        "--reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp "
        "--gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path "
        "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_mtp_reuse": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_reuse_cache.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 "
        "--reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp "
        "--gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path "
        "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --reuse_cache 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_mtp_cudagraph_deepep": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_cudagraph.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 "
        "--sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type "
        "eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 "
        "--sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph 1 "
        "--decode_capture_config '1,2,3,4' --use_deepep_moe 1 "
        "--use_deepep_low_latency 1",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_mtp_pd_reuse": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_pd.json",
        "smoke_args": {
            "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 "
            "--act_type BF16 --role_type PREFILL --cache_store_rdma_mode "
            "0 --use_local 1 --tp_size 2 --max_seq_len 12800 "
            "--reserver_runtime_mem_mb 10000 --sp_model_type "
            "qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle "
            "--sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 "
            "--sp_act_type bf16 --reuse_cache 1",
            "decode": "--load_cache_timeout_ms 120000 --act_type BF16 "
            "--seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 "
            "--reserver_runtime_mem_mb 10000 --warm_up 0 --sp_model_type "
            "qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle "
            "--sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 "
            "--sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph "
            "1 --decode_capture_config '1,2,3,4' --use_deepep_moe 1 "
            "--use_deepep_low_latency 1 --role_type DECODE "
            "--cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1",
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_fp8_basic": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_kernel_block": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_kernel_block_size_128.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--kernel_seq_size_per_block 128",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_cudagraph_deepep": {
        "task_info": "data/model/qwen3_next/q_r_next_cuda_graph.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --max_seq_len 128 "
        "--use_deepep_moe 1 --use_deepep_low_latency 1 --enable_cuda_graph 1 "
        "--warm_up 0  --concurrency_limit 8 --reserver_runtime_mem_mb 8192 "
        "--tp_size 2",
        "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_long_reuse_memcache": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_cache.json",
        "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step "
        "2 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb "
        "1024 --write_cache_sync 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_long_reuse_remote": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_remote_cache.json",
        "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 "
        "--reuse_cache 1 --enable_remote_cache 1 --write_cache_sync 1 "
        "--reco_put_timeout_ms 17000 --reco_get_timeout_ms 17000 "
        "--reco_get_broadcast_timeout 20000 --reco_put_broadcast_timeout 20000",
        "envs": ["KVCM_LOG_LEVEL=DEBUG"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_bf16_basic": {
        "task_info": "data/model/qwen35/qwen35_bf16_tp2.json",
        "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_load_quant_tp2": {
        "task_info": "data/model/qwen35/qwen35_bf16_tp2_load_quant.json",
        "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --quantization "
        "fp8_per_block",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "next_pd": {
        "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_pd_sep.json",
        "smoke_args": {
            "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type "
            "BF16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 "
            "--tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32",
            "decode": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type "
            "BF16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 "
            "--tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32",
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_next"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_next(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
