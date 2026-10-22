"""Native pytest entry for the six existing CUDA 13 SM120 OSS smoke targets.

Arguments and task files mirror suites_sm120.bzl. Custom cases run the original
unittest classes, including their setup, assertions and cleanup.
"""

import hashlib
import importlib
import json
import os
import traceback
import unittest
from collections import Counter
from pathlib import Path

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params, get_gpu_count
from rtp_llm.test.smoke_framework.runner import run_smoke_test

SMOKE_CASES = {
    "qwen3_bf16_sm120": {
        "main": "sm120_qwen3_test.py",
        "smoke_args": "--role_type PDFUSION --act_type BF16 --warm_up 0 --tp_size 1 --world_size 1 --frontend_server_count 1 --max_seq_len 8192 --seq_size_per_block 64 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2' --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
    "generation_prefill_cuda_graph_sm120": {
        "task_info": "data/model/qwen25/q_r_generation_prefill_cuda_graph_sm120.json",
        "smoke_args": "--act_type BF16 --warm_up 1 --seq_size_per_block 64 --concurrency_limit 5 --max_context_batch_size 5 --enable_cuda_graph 1 --decode_capture_config '1' --generation_prefill_cuda_graph_max_requests 5 --generation_prefill_capture_config '64,256'",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
    "embedding_bert_sm120": {
        "task_info": "data/model/bert/q_r.json",
        "smoke_args": "--seq_size_per_block 16 --act_type FP16",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
    "dense_fp8pb_dynamic_sm120": {
        "task_info": "data/model/qwen3/q_r_1_7b_fp8pb_tp2_sm120.json",
        "envs": ["LOAD_PYTHON_MODEL=1"],
        "smoke_args": "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0 --tp_size 2 --world_size 2 --seq_size_per_block 2048 --max_seq_len 16384 --reserver_runtime_mem_mb 16005 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2' --reuse_cache 0",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
    "dense_fp8pb_reuse_cache_tp2_sm120": {
        "main": "sm120_reuse_cache_test.py",
        "smoke_args": "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0 --tp_size 2 --world_size 2 --seq_size_per_block 2048 --max_seq_len 16384 --reserver_runtime_mem_mb 16005 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2' --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
    "moe_fp8pb_tp2_sm120": {
        "main": "sm120_moe_test.py",
        "smoke_args": "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 2048 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
        "gpu_type": "RTX_5000_PRO_CU13",
        "platform": "cuda",
        "markers": ["smoke", "cuda", "RTX_5000_PRO_CU13"],
        "timeout": 7200,
    },
}

_CUSTOM_CLASSES = {
    "sm120_qwen3_test.py": "SM120Qwen3Test",
    "sm120_reuse_cache_test.py": "SM120ReuseCacheTest",
    "sm120_moe_test.py": "SM120MoeTest",
}

SUITE_NAME = "smoke_sm120_cuda13"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


class _OriginalTestResult(unittest.TestResult):
    """Keep real unittest identities and every subtest event in the outer report."""

    def __init__(self):
        super().__init__()
        self.methods = []
        self.events = []

    def startTest(self, test):
        super().startTest(test)
        self.methods.append(test.id())

    def _record(self, test, status, err=None, reason=None, subtest=None):
        owner = getattr(test, "test_case", test)
        child = (
            subtest if subtest is not None else (test if owner is not test else None)
        )
        self.events.append(
            {
                "method_id": owner.id(),
                "kind": "subtest" if child is not None else "method",
                "subtest_id": child.id() if child is not None else None,
                "status": status,
                "reason": reason,
                "traceback": "".join(traceback.format_exception(*err)) if err else None,
            }
        )

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed", err)

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", err)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason=reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", err)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        status = (
            "passed"
            if err is None
            else ("failed" if issubclass(err[0], test.failureException) else "error")
        )
        self._record(test, status, err, subtest=subtest)


def _run_original_class(test_class, target_name, output_dir, record_property):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
    expected = suite.countTestCases()
    result = _OriginalTestResult()
    interrupted = None
    try:
        assert expected > 0, "Original custom smoke class must not be empty"
        suite.run(result)
    except BaseException:
        interrupted = traceback.format_exc()
        raise
    finally:
        verdicts = {}
        for method in result.methods:
            statuses = {e["status"] for e in result.events if e["method_id"] == method}
            verdicts[method] = next(
                (
                    value
                    for value in (
                        "error",
                        "failed",
                        "unexpected_success",
                        "expected_failure",
                        "skipped",
                        "passed",
                    )
                    if value in statuses
                ),
                "incomplete",
            )
        report = {
            "outer_target": target_name,
            "original_class": test_class.__module__ + "." + test_class.__qualname__,
            "expected_methods": expected,
            "actual_methods_run": result.testsRun,
            "method_verdicts": verdicts,
            "method_counts": dict(Counter(verdicts.values())),
            "subtest_counts": dict(
                Counter(e["status"] for e in result.events if e["kind"] == "subtest")
            ),
            "events": result.events,
            "interrupted": interrupted,
        }
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        # The complete payload also travels in the normal JUnit user properties.
        # It remains recoverable when optional undeclared-output collection is off.
        record_property("original_unittest_report", payload)
        record_property(
            "original_unittest_report_sha256",
            hashlib.sha256(payload.encode()).hexdigest(),
        )
        record_property("original_unittest_methods_run", result.testsRun)
        directory = Path(output_dir) / "sm120_original_unittest"
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / (target_name + ".json")
        report_path.write_text(payload)
        record_property("original_unittest_report_path", str(report_path))
    return expected, result


@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_sm120_cuda13(
    test_name, test_config, monkeypatch, tmp_path, record_property
):
    main = test_config.get("main")
    if main is None:
        run_smoke_test(test_name, test_config)
        return

    # Match custom_smoke_test's declared environment; pytest owns restoration.
    count = str(get_gpu_count(test_config))
    monkeypatch.setenv("SMOKE_ARGS", test_config["smoke_args"])
    monkeypatch.setenv("GPU_COUNT", count)
    monkeypatch.setenv("WORLD_SIZE", count)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    if "TEST_TMPDIR" not in os.environ:
        monkeypatch.setenv("TEST_TMPDIR", str(tmp_path))
    if "TEST_UNDECLARED_OUTPUTS_DIR" not in os.environ:
        monkeypatch.setenv("TEST_UNDECLARED_OUTPUTS_DIR", str(tmp_path))
    module = importlib.import_module("rtp_llm.test.smoke." + main.removesuffix(".py"))
    expected, result = _run_original_class(
        getattr(module, _CUSTOM_CLASSES[main]),
        test_name,
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"],
        record_property,
    )
    assert result.testsRun == expected, result
    assert not result.skipped, result.skipped
    assert not result.expectedFailures, result.expectedFailures
    assert not result.unexpectedSuccesses, result.unexpectedSuccesses
    assert not result.errors, result.errors
    assert not result.failures, result.failures
