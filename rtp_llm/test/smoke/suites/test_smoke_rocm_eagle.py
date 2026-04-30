"""Pytest entry for smoke suite ``smoke_rocm_eagle`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {   'rocm_eagle_qwen2_14b': {   'task_info': 'data/model/qwen2_14b/q_r_mtp_rocm.json',
                                'smoke_args': '--max_seq_len 16384 --tp_size 1 --use_asm_pa 1 --ft_disable_custom_ar 1 '
                                              '--sp_type eagle --gen_num_per_cycle 4 --warm_up 0 --act_type BF16 '
                                              '--sp_model_type qwen_2-mtp --sp_checkpoint_path '
                                              '/mnt/nas1/mtp_reg/qwen2_14b_draft/ --reserver_runtime_mem_mb 4002',
                                'gpu_type': 'MI308X-ROCM7',
                                'platform': 'cuda',
                                'markers': ['smoke', 'cuda', 'MI308X_ROCM7'],
                                'timeout': 600}}

SUITE_NAME = "smoke_rocm_eagle"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_eagle(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
