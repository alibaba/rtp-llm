import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import ResponseInfo
from rtp_llm.test.perf_test.perf_runner import _validate_grid_results, run_perf_test
from rtp_llm.test.perf_test.perf_utils import (
    filter_bs_by_kvcache,
    query_engine_status,
)


def _metric(seq=65536, bs=128, success_rate=1.0, latency=12.0):
    return {
        "input_len": seq,
        "batch_size": bs,
        "success_rate": success_rate,
        "avg_decode_time": latency,
        "avg_prefill_time": latency,
    }


def _write_result(tmp_path, metrics, is_decode=True):
    path = tmp_path / ("Decode_Result.json" if is_decode else "Prefill_Result.json")
    path.write_text(json.dumps({"mode": "grid", "metrics": metrics}))
    return path


@pytest.mark.parametrize("is_decode", [True, False])
def test_complete_grid_keeps_benchmark_without_latency_baseline(tmp_path, is_decode):
    metrics = [_metric(seq, bs, latency=1e6) for seq in (1024, 4096) for bs in (1, 8)]
    _write_result(tmp_path, metrics, is_decode)
    _validate_grid_results(tmp_path, [1024, 4096], [1, 8], is_decode=is_decode)


@pytest.mark.parametrize(
    "metrics, message",
    [
        ([], "Empty perf measurement"),
        ([_metric(bs=64)], "Missing perf measurement"),
        ([_metric(bs=64), _metric(bs=64)], "duplicate perf point"),
        ([_metric(bs=64), _metric(seq=131072)], "Unexpected"),
        ([_metric(bs=64), _metric(success_rate=0.046875)], "Failed requests"),
        ([_metric(bs=64), _metric(success_rate=0.0)], "Failed requests"),
        ([_metric(bs=64), _metric(latency=0)], "Invalid avg_decode_time"),
        ([_metric(bs=64), _metric(latency=float("nan"))], "Invalid avg_decode_time"),
        ([_metric(bs=64), _metric(latency=float("inf"))], "Invalid avg_decode_time"),
    ],
)
def test_invalid_or_missing_grid_points_cannot_pass(tmp_path, metrics, message):
    _write_result(tmp_path, metrics)
    with pytest.raises(AssertionError, match=message):
        _validate_grid_results(tmp_path, [65536], [64, 128], is_decode=True)


@pytest.mark.parametrize("content", [None, "{", "[]"])
def test_missing_or_malformed_result_cannot_pass(tmp_path, content):
    if content is not None:
        (tmp_path / "Decode_Result.json").write_text(content)
    with pytest.raises(AssertionError):
        _validate_grid_results(tmp_path, [65536], [128], is_decode=True)


def test_latency_trimming_preserves_failures_from_every_measurement():
    successful = ResponseInfo(
        {"aux_info": {"input_len": 65536, "output_len": 3,
                      "first_token_cost_time": 10.0, "cost_time": 30.0}}
    )
    failed = ResponseInfo({}, False)
    rounds = [[failed, failed]] + [[successful, successful] for _ in range(4)]
    with patch("rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"):
        runner = BatchPerfImpl(1234, 1, 2, "query", profile=False, warmup_runs=0,
                               profile_runs=0, measure_runs=5)
    with patch.object(runner, "_set_concurrency"), patch.object(
        runner, "_curl_server_responses", side_effect=rounds
    ):
        result = runner.run()
    assert result.avg_decode_time == 10.0
    assert result.total_requests == 10
    assert result.success_requests == 8
    assert result.fail_requests == 2


@pytest.mark.parametrize("is_decode", [True, False])
def test_benchmark_requests_the_full_per_dp_batch(is_decode):
    with patch("rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"):
        runner = BatchPerfImpl(1234, 8, 128, "query", is_decode=is_decode)
    response = Mock(status_code=200)
    response.json.return_value = {"status": "ok"}
    with patch(
        "rtp_llm.test.perf_test.batch_perf_impl.requests.post", return_value=response
    ) as post:
        runner._set_concurrency()
    post.assert_called_once_with(
        "http://127.0.0.1:1234/update_scheduler_info",
        json={
            "batch_size": 16,
            "mode": "decode" if is_decode else "prefill",
            "require_full_batch": True,
        },
        timeout=60,
    )


@pytest.mark.parametrize("payload", [{"error": "Failed on some addresses"}, {}])
def test_scheduler_update_http_200_error_cannot_start_benchmark(payload):
    with patch("rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"):
        runner = BatchPerfImpl(1234, 8, 128, "query")
    response = Mock(status_code=200, text=json.dumps(payload))
    response.json.return_value = payload
    with patch(
        "rtp_llm.test.perf_test.batch_perf_impl.requests.post", return_value=response
    ) as post, patch("rtp_llm.test.perf_test.batch_perf_impl.time.sleep"):
        with pytest.raises(Exception, match="failed to set concurrency after retries"):
            runner._set_concurrency()
    assert post.call_count == 20


def test_scheduler_update_retries_dp_error_until_explicit_success():
    with patch("rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"):
        runner = BatchPerfImpl(1234, 8, 128, "query")
    failed = Mock(status_code=200, text='{"error": "DP update failed"}')
    failed.json.return_value = {"error": "DP update failed"}
    success = Mock(status_code=200)
    success.json.return_value = {"status": "ok"}
    with patch(
        "rtp_llm.test.perf_test.batch_perf_impl.requests.post",
        side_effect=[failed, success],
    ) as post, patch("rtp_llm.test.perf_test.batch_perf_impl.time.sleep"):
        runner._set_concurrency()
    assert post.call_count == 2


def _query_cache_status(cache):
    cache_response = Mock()
    cache_response.json.return_value = cache
    worker_response = Mock()
    worker_response.json.return_value = {"frontend_concurrency_limit": 1024}
    with patch(
        "rtp_llm.test.perf_test.perf_utils.requests.get",
        side_effect=[cache_response, worker_response],
    ):
        return query_engine_status(1234)


@pytest.mark.parametrize("shape", ["direct", "wrapped", "empty_results"])
def test_grid_cache_status_preserves_native_token_units(shape):
    cache = {"total_kv_cache": "3854336", "block_size": "2048"}
    if shape == "wrapped":
        cache["results"] = [cache.copy()]
    elif shape == "empty_results":
        cache["results"] = []
    status = _query_cache_status(cache)
    assert status["max_kv_tokens"] == 3_854_336
    assert status["total_kv_cache_blocks"] == 1882


def test_grid_capacity_uses_the_smallest_dp_shard():
    status = _query_cache_status({
        "results": [
            {"total_kv_cache": "8192", "block_size": "2048"},
            {"total_kv_cache": "2048", "block_size": "1024"},
        ],
    })
    assert status["max_kv_tokens"] == 2048
    assert status["total_kv_cache_blocks"] == 2


@pytest.mark.parametrize("batches", [[16, 32], [64, 128]])
def test_capacity_filter_cannot_turn_an_incomplete_ppu_grid_green(tmp_path, batches):
    original_batches = batches.copy()
    status = _query_cache_status(
        {"total_kv_cache": "3854336", "block_size": "2048"}
    )
    runnable = filter_bs_by_kvcache(batches, 131072, status["max_kv_tokens"])
    _write_result(tmp_path, [_metric(seq=131072, bs=bs) for bs in runnable])
    with pytest.raises(AssertionError, match="Empty|Missing"):
        _validate_grid_results(tmp_path, [131072], batches, is_decode=True)
    assert batches == original_batches


@pytest.mark.parametrize("is_decode", [True, False])
@pytest.mark.parametrize("valid", [True, False])
def test_no_baseline_wrapper_validates_results_and_restores_state(
    tmp_path, monkeypatch, is_decode, valid
):
    report = _write_result(tmp_path, [_metric()] if valid else [], is_decode)
    args = SimpleNamespace(
        target_tpot=0, dataset_name="", dataset_path="", dataset="", test_json="",
        partial=1 if is_decode else 2,
    )
    config = SimpleNamespace(input_len_list=[65536], batch_size_list=[128])
    benchmark = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, "rtp_llm.test.perf_test.batch_decode_test",
                        SimpleNamespace(main=benchmark,
                                        _explicit_batch_size_list=lambda args: [128]))
    monkeypatch.setitem(sys.modules, "rtp_llm.test.perf_test.perf_config",
                        SimpleNamespace(parse_args=lambda: (args, []),
                                        prepare_config=lambda *args: config))
    entry = SimpleNamespace(
        _try_convert_model_path=lambda argv: argv, _print_new_golden=Mock(),
        upload_results_to_oss=Mock(return_value=""), validate_against_baseline=Mock(),
        write_summary_to_odps=Mock(), write_test_meta=Mock(),
    )
    monkeypatch.setitem(sys.modules, "rtp_llm.test.perf_test.test_entry", entry)
    monkeypatch.setenv("PERF_TEST_NAME", "previous-test")
    saved_argv = sys.argv
    test_config = {"model_type": "qwen35_moe", "checkpoint_path": "/model"}
    if valid:
        run_perf_test("ppu-perf", test_config, tmp_path)
        entry.upload_results_to_oss.assert_called_once_with(str(tmp_path))
    else:
        with pytest.raises(AssertionError, match="Empty perf measurement"):
            run_perf_test("ppu-perf", test_config, tmp_path)
        entry.upload_results_to_oss.assert_not_called()
        entry.write_summary_to_odps.assert_not_called()
    benchmark.assert_called_once()
    entry.write_test_meta.assert_called_once_with(str(tmp_path))
    entry.validate_against_baseline.assert_not_called()
    assert report.exists()
    assert sys.argv is saved_argv
    assert os.environ["PERF_TEST_NAME"] == "previous-test"
