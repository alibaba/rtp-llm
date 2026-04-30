"""Pytest entry for smoke suite ``smoke_rocm_moe`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {   'rocm_moe_qwen3_30b_basic': {   'task_info': 'data/model/qwen3_moe/q_r_30b_amd_py.json',
                                    'smoke_args': '--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type '
                                                  'BF16 --reserver_runtime_mem_mb 51200 --tp_size 1 --world_size 1 '
                                                  '--ep_size 1',
                                    'gpu_type': 'MI308X-ROCM7',
                                    'platform': 'cuda',
                                    'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                    'timeout': 600},
    'rocm_moe_qwen3_30b_tp2': {   'task_info': 'data/model/qwen3_moe/q_r_30b_amd_py_tp2.json',
                                  'smoke_args': '--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type '
                                                'BF16 --reserver_runtime_mem_mb 51200 --tp_size 2 --world_size 2 '
                                                '--ep_size 1',
                                  'gpu_type': 'MI308X-ROCM7',
                                  'platform': 'cuda',
                                  'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                  'timeout': 600}}

SUITE_NAME = "smoke_rocm_moe"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_moe(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
