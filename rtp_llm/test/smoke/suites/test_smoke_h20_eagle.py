"""Pytest entry for smoke suite ``smoke_h20_eagle``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "eagle_mtp_tp2": {
        "task_info": "data/model/qwen2_14b/q_r_mtp.json",
        "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle "
        "--gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp "
        "--sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 "
        "--reserver_runtime_mem_mb 21954 --tp_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "eagle_mtp_reuse": {
        "task_info": "data/model/qwen2_14b/q_r_mtp_reuse_cache.json",
        "smoke_args": "--reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 "
        "--write_cache_sync 1 --max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type "
        "eagle --gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp "
        "--sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 "
        "--reserver_runtime_mem_mb 21954 --tp_size 2",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "eagle_mtp_cudagraph": {
        "task_info": "data/model/qwen2_14b/q_r_mtp_cudagraph.json",
        "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE "
        "--redundant_expert 0 --act_type FP16 --concurrency_limit 64 "
        "--frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 24096 "
        "--seq_size_per_block 64 --enable_xqa 1 --sp_type eagle "
        "--gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path "
        "/mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 "
        "--decode_capture_config '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16'  "
        "--enable_cuda_graph 1 --load_method scratch --tp_size 1 --world_size 1 "
        "--dp_size 1",
        "envs": ["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "eagle_mtp_cudagraph_concurrent": {
        "task_info": "data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
        "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE "
        "--redundant_expert 0 --act_type FP16 --concurrency_limit 16 "
        "--frontend_server_count 1 --warm_up 0 "
        "--reserver_runtime_mem_mb 42000 --seq_size_per_block 64 "
        "--enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 "
        "--sp_model_type qwen_2-mtp --sp_checkpoint_path "
        "/mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 "
        "--decode_capture_config "
        "'1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16' "
        "--prefill_capture_config '80:1' --enable_cuda_graph 1 "
        "--tp_size 2",
        "envs": ["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
        "concurrency_test": True,
    },
    "eagle_mtp_no_cudagraph_concurrent": {
        "task_info": "data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
        "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode "
        "NONE --redundant_expert 0 --act_type FP16 "
        "--concurrency_limit 16 --frontend_server_count 1 --warm_up "
        "0 --reserver_runtime_mem_mb 42000 --seq_size_per_block 64 "
        "--enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 "
        "--sp_model_type qwen_2-mtp --sp_checkpoint_path "
        "/mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 "
        "--tp_size 2",
        "envs": ["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
        "concurrency_test": True,
    },
    "eagle_remote_cache_tp2": {
        "task_info": "data/model/qwen_sp/q_r_remote_cache_sp_tpsize2.json",
        "smoke_args": "--warm_up 0 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type "
        "qwen_2-mtp --tp_size 2 --sp_checkpoint_path "
        "/mnt/nas1/mtp_reg/qwen2_14b_draft/ --act_type FP16 --reuse_cache 1 "
        "--seq_size_per_block 8 --max_seq_len 16384 --ft_disable_custom_ar 1 "
        "--warm_up 0 --reserver_runtime_mem_mb 21954 --test_block_num 500 "
        "--enable_remote_cache true --enable_device_cache 0 "
        "--enable_memory_cache 0 --reco_put_timeout_ms 12000 "
        "--reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 "
        "--reco_put_broadcast_timeout 15000",
        "envs": ["KVCM_LOG_LEVEL=DEBUG"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
        "sleep_time_qr": 20,
    },
}

SUITE_NAME = "smoke_h20_eagle"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_eagle(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
