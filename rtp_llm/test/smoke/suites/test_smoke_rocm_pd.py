"""Pytest entry for smoke suite ``smoke_rocm_pd`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {   'rocm_pd_qwen3_8b': {   'task_info': 'data/model/qwen3/q_r_new_model_py.json',
                            'smoke_args': {   'prefill': '--test_block_num 10 --warm_up 0 --seq_size_per_block 16 '
                                                         '--act_type bf16 --use_swizzleA 1 --use_asm_pa 1 '
                                                         '--disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 '
                                                         '--role_type PREFILL --world_size 1',
                                              'decode': '--test_block_num 10 --warm_up 0 --seq_size_per_block 16 '
                                                        '--act_type bf16 --use_swizzleA 1 --use_asm_pa 1 '
                                                        '--disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 '
                                                        '--role_type DECODE --world_size 1'},
                            'gpu_type': 'MI308X-ROCM7',
                            'platform': 'cuda',
                            'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                            'timeout': 600}}

SUITE_NAME = "smoke_rocm_pd"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_pd(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
