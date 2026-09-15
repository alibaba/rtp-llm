"""Perf test framework runner — invoked by per-suite ``test_perf_*.py`` modules.

This file is FRAMEWORK code (lives in OSS). Test data (PERF_TESTS dicts, baselines/,
test_data/) lives in internal_source under ``rtp_llm/test/perf_test/suites/`` and
adjacent ``baselines``/``test_data`` directories. Each suite resolves its own
``data_dir`` from ``__file__`` and passes it into ``run_perf_test``, so the framework
stays decoupled from data location.
"""

import json
import math
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

        params.append(pytest_module.param(test_name, config, id=test_name, marks=marks))
    return params


def run_perf_test(test_name: str, test_config: dict, data_dir: Path):
    """Run a single perf test case.

    1. Set environment variables from test_config (restored in finally to avoid
       cross-case leakage when multiple perf cases run in the same pytest session).
    2. Build sys.argv for batch_decode_test.main()
    3. Run engine + benchmark
    4. Validate results against baseline (raise on regression)

    ``data_dir`` is resolved by the suite file (typically
    ``Path(__file__).resolve().parent.parent``) and points at the directory that
    contains ``baselines/`` and ``test_data/``. Passing it explicitly keeps the
    framework agnostic to where suite data is hosted (OSS vs internal).
    """
    # Track keys we set so we can restore them in finally — same pattern smoke uses.
    _env_keys_set: list = []
    for k, v in test_config.get("envs", {}).items():
        _env_keys_set.append((k, os.environ.get(k)))
        os.environ[k] = v
    _env_keys_set.append(("PERF_TEST_NAME", os.environ.get("PERF_TEST_NAME")))
    os.environ["PERF_TEST_NAME"] = test_name

    argv = _build_argv(test_name, test_config, data_dir)

    baseline_path = None
    baseline_rel = test_config.get("baseline", "")
    if baseline_rel:
        candidate = str(data_dir / baseline_rel)
        if not os.path.exists(candidate):
            raise AssertionError(
                f"Baseline file configured but not found for {test_name}: {candidate}"
            )
        if os.path.getsize(candidate) == 0:
            raise AssertionError(
                f"Baseline file configured but is empty for {test_name}: {candidate}"
            )
        baseline_path = candidate

    from rtp_llm.test.perf_test.batch_decode_test import main
    from rtp_llm.test.perf_test.perf_config import parse_args, prepare_config
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

        args, remaining = parse_args()
        grid_config = None
        if args.target_tpot <= 0 and not any(
            (args.dataset_name, args.dataset_path, args.dataset, args.test_json)
        ):
            grid_config = prepare_config(args, remaining)
            # Match main's explicit prefill batch-size override.
            from rtp_llm.test.perf_test.batch_decode_test import _explicit_batch_size_list

            if args.partial == 2:
                grid_config.batch_size_list = _explicit_batch_size_list(args) or [1]

        start_time = time.time()
        result_dir = main()
        duration = time.time() - start_time

        write_test_meta(result_dir)
        if grid_config is not None:
            _validate_grid_results(
                result_dir,
                grid_config.input_len_list,
                grid_config.batch_size_list,
                is_decode=args.partial == 1,
            )
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


def _validate_grid_results(result_dir, input_lengths, batch_sizes, *, is_decode):
    """Require real measurements even when a benchmark has no latency baseline."""
    result_path = Path(result_dir) / (
        "Decode_Result.json" if is_decode else "Prefill_Result.json"
    )
    try:
        with result_path.open() as result_file:
            result = json.load(result_file)
    except (OSError, ValueError) as exc:
        raise AssertionError(f"Missing or invalid perf result: {result_path}") from exc
    if not isinstance(result, dict) or result.get("mode") != "grid":
        raise AssertionError(f"Expected grid perf result: {result_path}")

    expected = {(seq, bs) for seq in input_lengths for bs in batch_sizes}
    metrics = result.get("metrics")
    if not expected or not isinstance(metrics, list) or not metrics:
        raise AssertionError(f"Empty perf measurement grid: {result_path}")

    actual = set()
    phase_metric = "avg_decode_time" if is_decode else "avg_prefill_time"
    for metric in metrics:
        if not isinstance(metric, dict):
            raise AssertionError(f"Invalid perf measurement: {metric!r}")
        point = (metric.get("input_len"), metric.get("batch_size"))
        if point not in expected or point in actual:
            raise AssertionError(f"Unexpected or duplicate perf point: {point}")
        actual.add(point)
        if metric.get("success_rate") != 1.0:
            raise AssertionError(f"Failed requests in perf point {point}: {metric}")
        value = metric.get(phase_metric)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise AssertionError(f"Invalid {phase_metric} at perf point {point}: {value}")
    if actual != expected:
        raise AssertionError(f"Missing perf measurement points: {sorted(expected - actual)}")


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
