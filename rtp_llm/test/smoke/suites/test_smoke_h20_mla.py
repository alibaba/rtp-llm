"""Pytest entry for smoke suite ``smoke_h20_mla`` — one file per suite (PR12 / B0).

All runner / parametrize / env logic lives in rtp_llm.test.smoke_framework.
This file is intentionally tiny: data + parametrize + dispatch.
"""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test

from .smoke_h20_mla_cases import SMOKE_CASES

SUITE_NAME = "smoke_h20_mla"

_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_h20_mla(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
