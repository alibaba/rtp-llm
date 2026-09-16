"""Pytest entry for smoke suite ``smoke_h20_dense``.

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "dense_generation_prefill_cuda_graph": {
        "task_info": "data/model/qwen25/q_r_generation_prefill_cuda_graph.json",
        "smoke_args": "--act_type BF16 --warm_up 0 --seq_size_per_block 64 --test_block_num 1000 "
        "--concurrency_limit 5 --max_context_batch_size 5 --enable_cuda_graph 1 --decode_capture_config '1' "
        "--generation_prefill_cuda_graph_max_requests 5 --generation_prefill_capture_config '64,256'",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_generation_prefill_cuda_graph_fallback": {
        "task_info": "data/model/qwen25/q_r_generation_prefill_cuda_graph_fallback.json",
        "smoke_args": "--act_type BF16 --warm_up 0 --seq_size_per_block 64 --test_block_num 1000 "
        "--concurrency_limit 1 --max_context_batch_size 1 --reuse_cache 0 --enable_cuda_graph 1 "
        "--decode_capture_config '1' --generation_prefill_cuda_graph_max_requests 1 "
        "--generation_prefill_capture_config '64'",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_fp8kv_cudagraph": {
        "task_info": "data/model/qwen25/q_r_new_model_py_fp8_kv_cache_cudagraph.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num "
        "1000 --fp8_kv_cache 1 --enable_cuda_graph 1  --disable_flash_infer 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_fp8kv_flashinfer_prefill": {
        "task_info": "data/model/qwen25/q_r_new_model_py_fp8_kv_cache_flashinfer_prefill.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 0 "
        "--disable_flash_infer 0 --frontend_server_count 1",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_fp8_prequant_tp2": {
        "task_info": "data/model/qwen3/q_r_block_fp8.json",
        "smoke_args": "--disable_flash_infer 1 --act_type BF16 --reserver_runtime_mem_mb "
        "8192 --tp_size 2 --warm_up 0",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_fp8pb_dynamic": {
        "task_info": "data/model/qwen3/q_r_h20.json",
        "smoke_args": "--disable_flash_infer 1 --quantization FP8_PER_BLOCK --act_type BF16 "
        "--warm_up 0",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_fp8pt_dynamic": {
        "task_info": "data/model/qwen3/q_r_h20_per_tensor_w13.json",
        "smoke_args": "--disable_flash_infer 1 --quantization FP8_DYNAMIC_PER_TENSOR --act_type "
        "BF16",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_override_yarn": {
        "task_info": "data/model/qwen3/q_r_override_yarn.json",
        "smoke_args": '--reserver_runtime_mem_mb 20000 --json_model_override_args \'{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"beta_slow":1.0,"beta_fast":1.0,"mscale":1.0,"extrapolation_factor":1.0}}\' --seq_size_per_block 64 --act_type BF16 --warm_up 0',
        "envs": ["LOAD_PYTHON_MODEL=1"],
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_pdfusion_ratio_prompt_batch_alternation": {
        "task_info": "data/model/qwen25/q_r_pdfusion_ratio_prompt_batch.json",
        "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 "
        "--disable_flash_infer 1 --tp_size 1 --dp_size 2 --world_size 2 "
        "--pdfusion_scheduler_mode ratio --decode_prefill_ratio 3",
        "concurrency_test": True,
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
    "dense_prompt_scoring": {
        "task_info": "data/model/qwen25/q_r_prompt_scoring.json",
        "smoke_args": "--act_type BF16 --warm_up 0",
        "gpu_type": "H20",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "H20"],
        "timeout": 600,
    },
}

SUITE_NAME = "smoke_h20_dense"

_test_params = build_smoke_params(
    pytest,
    {SUITE_NAME: SMOKE_CASES},
    composite_suites={"maga_model_smoke_light": [SUITE_NAME]},
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_dense(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
