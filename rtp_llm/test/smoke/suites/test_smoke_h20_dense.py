"""Pytest entry for smoke suite ``smoke_h20_dense`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {   'dense_fp8kv_cudagraph': {   'task_info': 'data/model/qwen25/q_r_new_model_py_fp8_kv_cache.json',
                                 'smoke_args': '--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num '
                                               '1000 --fp8_kv_cache 1 --enable_cuda_graph 1  --disable_flash_infer 1',
                                 'gpu_type': 'H20',
                                 'platform': 'cuda',
                                 'markers': ['smoke', 'cuda', 'H20'],
                                 'timeout': 600},
    'dense_fp8_prequant_tp2': {   'task_info': 'data/model/qwen3/q_r_block_fp8.json',
                                  'smoke_args': '--disable_flash_infer 1 --act_type BF16 --reserver_runtime_mem_mb '
                                                '8192 --tp_size 2 --warm_up 0',
                                  'gpu_type': 'H20',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'H20'],
                                  'timeout': 600},
    'dense_fp8pb_dynamic': {   'task_info': 'data/model/qwen3/q_r_h20.json',
                               'smoke_args': '--disable_flash_infer 1 --quantization FP8_PER_BLOCK --act_type BF16 '
                                             '--warm_up 0',
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600},
    'dense_fp8pt_dynamic': {   'task_info': 'data/model/qwen3/q_r_h20_per_tensor_w13.json',
                               'smoke_args': '--disable_flash_infer 1 --quantization FP8_DYNAMIC_PER_TENSOR --act_type '
                                             'BF16',
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600},
    'dense_override_yarn': {   'task_info': 'data/model/qwen3/q_r_override_yarn.json',
                               'smoke_args': '--reserver_runtime_mem_mb 20000 --json_model_override_args \'{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"beta_slow":1.0,"beta_fast":1.0,"mscale":1.0,"extrapolation_factor":1.0}}\' --seq_size_per_block 64 --act_type BF16 --warm_up 0',
                               'envs': ['LOAD_PYTHON_MODEL=1'],
                               'gpu_type': 'H20',
                               'platform': 'cuda',
                               'markers': ['smoke', 'cuda', 'H20'],
                               'timeout': 600}}

SUITE_NAME = "smoke_h20_dense"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_dense(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
