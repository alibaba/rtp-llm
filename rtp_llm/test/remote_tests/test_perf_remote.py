"""Perf tests dispatched to remote GPU workers.

This file does NOT import rtp_llm or any heavy dependencies.
It reads perf_defs.py (pure Python, no heavy imports) to generate test params,
then the remote-gpu plugin dispatches each test to a NativeLink worker.
"""

import sys
from pathlib import Path

import pytest

# Import perf_defs without importing rtp_llm
# perf_defs.py is designed to be Python-only (no heavy imports)
# Monorepo: internal_source is sibling of github-opensource — walk parents to find it.
_here = Path(__file__).resolve().parent
_perf_dir = None
for base in _here.parents:
    cand = base / "internal_source" / "rtp_llm" / "test" / "perf_test"
    if (cand / "perf_defs.py").is_file():
        _perf_dir = cand
        break
if _perf_dir is None:
    raise RuntimeError(
        "Cannot find internal_source/rtp_llm/test/perf_test/perf_defs.py "
        "(expected RTP-LLM monorepo with github-opensource + internal_source)."
    )
sys.path.insert(0, str(_perf_dir.parent))
from perf_test.perf_defs import build_perf_params

_test_params = build_perf_params(pytest)


@pytest.mark.manual
@pytest.mark.perf
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_perf(test_name: str, test_config: dict):
    """Perf test stub - actual execution happens on remote GPU worker.

    When --remote is used, the plugin intercepts this and runs it remotely.
    When run locally (no --remote), it runs the actual perf test.
    """
    # If we get here, we're running locally (no --remote plugin)
    # Import the actual runner only at execution time
    from perf_test.perf_runner import run_perf_test

    run_perf_test(test_name, test_config)
