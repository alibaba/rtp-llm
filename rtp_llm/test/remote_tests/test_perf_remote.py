"""Perf tests dispatched to remote GPU workers.

This file does NOT import rtp_llm or any heavy dependencies at collection time.

`rtp_llm.test.perf_test` is a PEP 420 namespace package contributed to by both OSS
(framework: perf_runner.py, test_entry.py, batch_decode_test.py, ...) and
internal_source (data: perf_defs.PERF_TESTS, baselines/, test_data/).

In OSS-only checkouts the data module is missing — collection skips gracefully via
pytest.importorskip rather than RuntimeError-ing the whole session.
"""

import pytest

# Internal-only test data; OSS-only checkouts skip the whole module.
perf_defs = pytest.importorskip(
    "rtp_llm.test.perf_test.perf_defs",
    reason="perf test data lives in internal_source; not present in OSS-only checkout",
)
from rtp_llm.test.perf_test.perf_runner import build_perf_params  # noqa: E402

_test_params = build_perf_params(pytest, perf_defs.PERF_TESTS)


@pytest.mark.manual
@pytest.mark.perf
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_perf(test_name: str, test_config: dict):
    """Perf test stub - actual execution happens on remote GPU worker.

    When --remote is used, the plugin intercepts this and runs it remotely.
    When run locally (no --remote), it runs the actual perf test.
    """
    from rtp_llm.test.perf_test.perf_runner import run_perf_test

    run_perf_test(test_name, test_config)
