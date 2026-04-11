"""Smoke tests dispatched to remote GPU workers.

This file does NOT import rtp_llm or any heavy dependencies.
It reads smoke_defs.py (pure Python/Starlark compatible) to generate test params,
then the remote-gpu plugin dispatches each test to a NativeLink worker.
"""
import os
import sys
from pathlib import Path

import pytest

# Import smoke_defs without importing rtp_llm
# smoke_defs.py is designed to be Python/Starlark dual-compatible (no heavy imports)
# Monorepo: internal_source is sibling of github-opensource — walk parents to find it.
_here = Path(__file__).resolve().parent
_smoke_dir = None
for base in _here.parents:
    cand = base / "internal_source" / "rtp_llm" / "test" / "smoke"
    if (cand / "smoke_defs.py").is_file():
        _smoke_dir = cand
        break
if _smoke_dir is None:
    raise RuntimeError(
        "Cannot find internal_source/rtp_llm/test/smoke/smoke_defs.py "
        "(expected RTP-LLM monorepo with github-opensource + internal_source)."
    )
sys.path.insert(0, str(_smoke_dir.parent))
from smoke.smoke_defs import build_smoke_params

_test_params = build_smoke_params(pytest)


@pytest.mark.manual
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke(test_name: str, test_config: dict):
    """Smoke test stub - actual execution happens on remote GPU worker.

    When --remote is used, the plugin intercepts this and runs it remotely.
    When run locally (no --remote), it runs the actual smoke test.
    """
    # If we get here, we're running locally (no --remote plugin)
    # Import the actual runner only at execution time
    from smoke.smoke_test import run_smoke_test
    run_smoke_test(test_name, test_config)
