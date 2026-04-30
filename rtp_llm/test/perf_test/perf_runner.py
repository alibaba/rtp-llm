"""Perf test framework runner — invoked by test_perf_remote.py.

This file is FRAMEWORK code (lives in OSS). Test data (PERF_TESTS dict, baselines/,
test_data/) lives in internal_source as a namespace-package contribution to
`rtp_llm.test.perf_test`. The dict is passed in via build_perf_params(...) so the
framework has no compile-time dependency on internal_source.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List


def build_perf_params(pytest_module, perf_tests: dict):
    """Build pytest.param list from a PERF_TESTS dict.

    Caller passes the dict (typically from rtp_llm.test.perf_test.perf_defs which
    lives in internal_source). Keeps framework decoupled from data location.
    """
    params = []
    for test_name, config in perf_tests.items():
        marks = []
        for marker_name in config.get("markers", []):
            marks.append(getattr(pytest_module.mark, marker_name))
        marks.append(pytest_module.mark.manual)

        gpu_type = config.get("gpu_type", "H20")
        gpu_count = config.get("gpu_count", 1)
        marks.append(pytest_module.mark.gpu(type=gpu_type, count=gpu_count))

        params.append(
            pytest_module.param(test_name, config, id=test_name, marks=marks)
        )
    return params


def _resolve_data_dir() -> Path:
    """Locate the perf test data directory.

    Data (baselines/, test_data/) lives next to perf_defs.py. perf_defs.py lives
    in internal_source as a namespace-package contribution to rtp_llm.test.perf_test.
    """
    from rtp_llm.test.perf_test import perf_defs

    return Path(perf_defs.__file__).resolve().parent


def run_perf_test(test_name: str, test_config: dict):
    """Run a single perf test case.

    1. Set environment variables from test_config (restored in finally to avoid
       cross-case leakage when multiple perf cases run in the same pytest session).
    2. Build sys.argv for batch_decode_test.main()
    3. Run engine + benchmark
    4. Validate results against baseline (raise on regression)
    """
    # Track keys we set so we can restore them in finally — same pattern smoke uses.
    _env_keys_set: list = []
    for k, v in test_config.get("envs", {}).items():
        _env_keys_set.append((k, os.environ.get(k)))
        os.environ[k] = v
    _env_keys_set.append(("PERF_TEST_NAME", os.environ.get("PERF_TEST_NAME")))
    os.environ["PERF_TEST_NAME"] = test_name

    data_dir = _resolve_data_dir()
    argv = _build_argv(test_name, test_config, data_dir)

    baseline_path = None
    baseline_rel = test_config.get("baseline", "")
    if baseline_rel:
        candidate = str(data_dir / baseline_rel)
        if os.path.exists(candidate):
            baseline_path = candidate
        else:
            logging.warning("Baseline file not found: %s", candidate)

    from rtp_llm.test.perf_test.batch_decode_test import main
    from rtp_llm.test.perf_test.test_entry import (
        _print_new_golden,
        _try_convert_model_path,
        upload_results_to_oss,
        validate_against_baseline,
        write_summary_to_odps,
        write_test_meta,
    )

    saved_argv = sys.argv
    try:
        sys.argv = _try_convert_model_path(argv)

        start_time = time.time()
        result_dir = main()
        duration = time.time() - start_time

        write_test_meta(result_dir)
        _print_new_golden(result_dir, baseline_path)

        if baseline_path:
            if not validate_against_baseline(result_dir, baseline_path):
                raise AssertionError(
                    f"PERF REGRESSION DETECTED for {test_name} — "
                    "see comparison table above for details"
                )
        else:
            oss_path = upload_results_to_oss(result_dir)
            write_summary_to_odps(result_dir, oss_path, duration)
    finally:
        sys.argv = saved_argv
        for key, old_val in reversed(_env_keys_set):
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def _build_argv(test_name: str, test_config: dict, data_dir: Path) -> List[str]:
    """Build sys.argv from test_config, matching the Bazel defs.bzl pattern.

    Substitutes `{perf_test_dir}` placeholders with the resolved data dir.
    """
    argv = ["perf_test"]

    argv.extend(["--model_type", test_config["model_type"]])
    argv.extend(["--checkpoint_path", test_config["checkpoint_path"]])
    tokenizer_path = test_config.get("tokenizer_path", test_config["checkpoint_path"])
    argv.extend(["--tokenizer_path", tokenizer_path])

    perf_args = list(test_config.get("perf_args", []))
    for i, arg in enumerate(perf_args):
        if "{perf_test_dir}" in arg:
            perf_args[i] = arg.replace("{perf_test_dir}", str(data_dir))
    argv.extend(perf_args)

    return argv
