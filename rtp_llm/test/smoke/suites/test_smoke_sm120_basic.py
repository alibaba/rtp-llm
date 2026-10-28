"""Pytest entry for RTX 5000 Pro / SM120 smoke coverage."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test


SMOKE_CASES = {
    "generation_prefill_cuda_graph_sm120": {
        "task_info": "data/model/qwen25/q_r_generation_prefill_cuda_graph_sm120.json",
        "smoke_args": "--act_type BF16 --warm_up 1 --seq_size_per_block 64 "
        "--concurrency_limit 5 --max_context_batch_size 5 --enable_cuda_graph 1 "
        "--decode_capture_config '1' --generation_prefill_cuda_graph_max_requests 5 "
        "--generation_prefill_capture_config '64,256'",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "embedding_bert_sm120": {
        "task_info": "data/model/bert/q_r.json",
        "smoke_args": "--seq_size_per_block 16 --act_type FP16",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "softmax_probs_sm120": {
        "task_info": "data/model/qwen25/q_r_softmax_probs_sm120.json",
        "smoke_args": "--act_type FP16 --warm_up 0",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "fp16_sm120": {
        "task_info": "data/model/qwen25/q_r_s_fp16_sm120.json",
        "smoke_args": "--act_type FP16 --warm_up 0",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "bf16_sm120": {
        "task_info": "data/model/qwen25/q_r_s_bf16_sm120.json",
        "smoke_args": "--act_type BF16 --warm_up 0",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "bf16_cuda_graph_sm120": {
        "task_info": "data/model/qwen25/q_r_s_bf16_sm120.json",
        "smoke_args": "--act_type BF16 --warm_up 0 --seq_size_per_block 64 "
        "--enable_cuda_graph 1 --decode_capture_config '1,2'",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "random_seed_sm120": {
        "task_info": "data/model/qwen25/test_random_seed_sm120.json",
        "smoke_args": "--act_type FP16 --warm_up 0",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
    "logits_index_sm120": {
        "task_info": "data/model/qwen25/logits_index_q_r_sm120.json",
        "smoke_args": "--act_type FP16 --warm_up 0",
        "gpu_type": "RTX_5000_PRO",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_sm120_basic"
_test_params = build_smoke_params(
    pytest,
    {SUITE_NAME: SMOKE_CASES},
    composite_suites={"maga_model_smoke_light": [SUITE_NAME]},
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_sm120_basic(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
