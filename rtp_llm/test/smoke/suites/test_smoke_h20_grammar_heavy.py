"""Pytest entry for the H20 Qwen3.5 grammar stress suite."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "qwen35_grammar_concurrent_no_mtp": {
        "task_info": "data/model/qwen35/q_r_mtp_grammar.json",
        "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 "
        "--max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 "
        "--think_mode 0 --load_method scratch --concurrency_limit 8",
        "envs": [
            "NCCL_DISABLE_ABORT=1",
            "NCCL_DEBUG=INFO",
            "LOG_LEVEL=INFO",
            "PYTHONUNBUFFERED=TRUE",
        ],
        "concurrency_test": True,
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "qwen35_grammar_pd_mtp_reasoning": {
        "task_info": "data/model/qwen35/q_r_mtp_grammar_reasoning.json",
        "smoke_args": {
            "prefill": "--act_type BF16 --warm_up 0 --seq_size_per_block 2048 "
            "--role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 "
            "--tp_size 1 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 "
            "--sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 "
            "--sp_type eagle --sp_checkpoint_path "
            "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 "
            "--think_mode 1 --load_method scratch",
            "decode": "--load_cache_timeout_ms 120000 --act_type BF16 "
            "--warm_up 0 --seq_size_per_block 2048 --role_type DECODE "
            "--cache_store_rdma_mode 0 --use_local 1 --tp_size 1 "
            "--max_seq_len 12800 --reserver_runtime_mem_mb 10000 "
            "--sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 "
            "--sp_type eagle --sp_checkpoint_path "
            "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 "
            "--think_mode 1 --load_method scratch",
        },
        "envs": {
            "prefill": ["PYTHONUNBUFFERED=TRUE"],
            "decode": ["PYTHONUNBUFFERED=TRUE"],
        },
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_grammar_heavy"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_grammar_heavy(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
