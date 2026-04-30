"""Pytest entry for smoke suite ``smoke_rocm_basic`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {   'rocm_basic_cache_reuse': {   'task_info': 'data/model/qwen2/q_r_reuse.json',
                                  'smoke_args': '--reuse_cache 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa '
                                                '1 --act_type FP16',
                                  'gpu_type': 'MI308X-ROCM7',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                  'timeout': 600},
    'rocm_basic_batch_cache_reuse': {   'task_info': 'data/model/qwen3/q_r_308x_batch_cache.json',
                                        'smoke_args': '--reuse_cache 1 --enable_cuda_graph 1 --seq_size_per_block 16 '
                                                      '--use_aiter_pa 1 --use_asm_pa 1 --act_type FP16',
                                        'gpu_type': 'MI308X-ROCM7',
                                        'platform': 'cuda',
                                        'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                        'timeout': 600}}

SUITE_NAME = "smoke_rocm_basic"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_basic(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
