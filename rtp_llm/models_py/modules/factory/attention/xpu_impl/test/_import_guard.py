"""Shared skip-vs-fail policy for XPU unit tests with optional imports.

Some XPU unit tests import `rtp_llm.ops` / vllm_flash_attn helpers that only
build in an XPU toolchain. When that import fails:

  - On a real XPU test run, it is a genuine regression (broken build/pybind
    config), not an environment gap, so the test must FAIL.
  - Everywhere else (CPU dev boxes, non-XPU CI), the import genuinely may be
    unavailable, so the test is SKIPPED.

"Real XPU test run" is detected via either:
  - `TEST_USING_DEVICE=XPU`, the bazel test-env convention injected by
    `device_test_envs()` (see //bazel:device_defs.bzl) through
    `select({"@//:using_xpu": ...})` on the `py_test` targets in this
    directory's BUILD file.
  - `RTP_LLM_DEVICE_TYPE=xpu`, the runtime override some CI jobs / manual
    runs set directly.

This module is pure stdlib (no torch / rtp_llm import) on purpose: it must
keep working even when the package under test is completely unbuildable, and
it is unit-tested on its own in test_import_guard.py without depending on
any package-level preflight import.
"""

import os
from unittest import SkipTest


def is_xpu_test_env() -> bool:
    """True when this test run is targeting XPU."""
    return (
        os.environ.get("TEST_USING_DEVICE", "").strip().upper() == "XPU"
        or os.environ.get("RTP_LLM_DEVICE_TYPE", "").strip().lower() == "xpu"
    )


def skip_or_fail_on_missing_import(
    test_case, import_ok: bool, import_err: BaseException, what: str = "required import"
) -> None:
    """Fail on XPU, skip elsewhere, when a guarded import failed.

    Call from `setUp()` (or a test method) with the `_IMPORT_OK` /
    `_IMPORT_ERR` values captured by the module-level try/except around the
    optional import.
    """
    if import_ok:
        return
    if is_xpu_test_env():
        test_case.fail(f"{what} failed on XPU: {import_err!r}")
    raise SkipTest(f"{what} unavailable: {import_err!r}")
