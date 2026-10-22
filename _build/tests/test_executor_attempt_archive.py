"""CPU contracts for preserving remote executor attempt evidence.

These report-only contracts stay outside fixed GPU profile inventories.
The existing fake executors model controller responses, not Native execution.
"""

import json
from pathlib import Path

import pytest

from rtp_llm.test.remote_tests import endpoint_info
from rtp_llm.test.remote_tests import executor as executor_module
from rtp_llm.test.remote_tests import remote_execution_pb2
from rtp_llm.test.remote_tests.executor import ExecutionResult, FailoverRemoteExecutor
from rtp_llm.test.remote_tests.executor_failover_test import (
    _FakeCAS,
    _FakeExecutor,
    _reset_fake_executor,
)


class _StreamWritingExecutor(_FakeExecutor):
    def execute(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        for name in ("stdout", "stderr"):
            path = kwargs.get(f"stream_{name}_file")
            if path is not None:
                path.write_bytes((f"live {name} {len(self.calls)}").encode())
        if kwargs.get("on_stage") is not None:
            kwargs["on_stage"]("EXECUTING", f"operations/{len(self.calls)}")
        if isinstance(result, BaseException):
            raise result
        # The caller can tail these exact paths before execute returns.
        assert kwargs["stream_stdout_file"].read_bytes().startswith(b"live stdout")
        return result


def _stream_archive_executor(monkeypatch, tmp_path, results, **options):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = results
    executor = FailoverRemoteExecutor(
        "grpc://executor.service:50052",
        _FakeCAS(),
        max_failovers=1,
        executor_factory=_StreamWritingExecutor,
        **options,
    )
    paths = {
        "stream_stdout_file": tmp_path / "remote_stdout.log",
        "stream_stderr_file": tmp_path / "remote_stderr.log",
    }
    return executor, paths


def test_failover_preserves_each_attempt_raw_stream_and_result(monkeypatch, tmp_path):
    first = ExecutionResult(
        125,
        stdout_raw=b"first raw",
        stderr_raw=b"first err",
        stdout_digest=remote_execution_pb2.Digest(hash="a" * 64, size_bytes=9),
        stderr_digest=remote_execution_pb2.Digest(hash="b" * 64, size_bytes=9),
        operation_name="operations/1",
        last_stage="COMPLETED",
        operation_done=True,
        infra_category="infra_stall",
        response_status_code=0,
        cached_result=False,
    )
    final = ExecutionResult(
        9,
        stdout_raw=b"",
        stderr_raw=b"timeout",
        operation_name="operations/2",
        response_status_code=4,
    )
    executor, paths = _stream_archive_executor(monkeypatch, tmp_path, [first, final])
    stages = []
    result = executor.execute(
        command=["original"],
        env_vars={"SECRET": "never-copy"},
        no_cache=True,
        on_stage=lambda *args: stages.append(args),
        **paths,
    )
    assert result is final and result.failover_attempts == 1
    roots = list(tmp_path.glob("attempts-*"))
    assert len(roots) == 1
    assert stages == [("EXECUTING", "operations/1"), ("EXECUTING", "operations/2")]
    one, two = roots[0] / "attempt-0001", roots[0] / "attempt-0002"
    assert (one / "remote_stdout.log").read_bytes() == b"live stdout 1"
    assert (one / "remote_stderr.log").read_bytes() == b"live stderr 1"
    assert (one / "result_stdout.log").read_bytes() == b"first raw"
    assert (two / "result_stdout.log").read_bytes() == b""
    assert (two / "result_stderr.log").read_bytes() == b"timeout"
    receipt = json.loads((one / "result.json").read_text())
    assert receipt["attempt"] == 1
    assert receipt["executor_endpoint"] == "grpc://10.0.0.1:50052"
    assert receipt["result"]["operation_name"] == "operations/1"
    assert receipt["result"]["stdout_digest"] == {"hash": "a" * 64, "size_bytes": 9}
    assert receipt["result"]["exit_code"] == 125
    assert receipt["collection_errors"] == []
    assert "never-copy" not in (one / "result.json").read_text()
    assert paths["stream_stdout_file"].read_bytes() == b"live stdout 2"
    assert all(
        call["stream_stdout_file"] == paths["stream_stdout_file"]
        for call in _FakeExecutor.calls
    )
    assert all(call["no_cache"] is True for call in _FakeExecutor.calls)
    assert _FakeExecutor.calls[0]["env_vars"] == {"SECRET": "never-copy"}
    assert _FakeExecutor.calls[1]["env_vars"] == {
        "SECRET": "never-copy",
        "RTP_REMOTE_EXECUTOR_FAILOVER_ATTEMPT": "1",
    }


def test_failover_archive_is_unique_across_outer_execute_retries(monkeypatch, tmp_path):
    executor, paths = _stream_archive_executor(
        monkeypatch, tmp_path, [ExecutionResult(1), ExecutionResult(0)]
    )
    assert executor.execute(command=["original"], **paths).exit_code == 1
    assert executor.execute(command=["original"], **paths).exit_code == 0
    roots = list(tmp_path.glob("attempts-*"))
    assert len(roots) == 2
    assert sorted(
        (root / "attempt-0001/remote_stdout.log").read_bytes() for root in roots
    ) == [b"live stdout 1", b"live stdout 2"]
    assert sorted(
        json.loads((root / "attempt-0001/result.json").read_text())["result"][
            "exit_code"
        ]
        for root in roots
    ) == [0, 1]


def test_failover_archives_exception_and_observed_operation(monkeypatch, tmp_path):
    error = RuntimeError("original failure")
    executor, paths = _stream_archive_executor(monkeypatch, tmp_path, [error])
    with pytest.raises(RuntimeError) as caught:
        executor.execute(command=["original"], **paths)
    assert caught.value is error
    receipt = json.loads(
        next(tmp_path.glob("attempts-*/attempt-0001/result.json")).read_text()
    )
    assert receipt["exception_type"] == "RuntimeError"
    assert receipt["result"] is None
    assert receipt["observed_operation"] == "operations/1"
    assert receipt["observed_stage"] == "EXECUTING"
    assert (
        next(tmp_path.glob("attempts-*/attempt-0001/remote_stdout.log")).read_bytes()
        == b"live stdout 1"
    )
    assert len(_FakeExecutor.calls) == len(_FakeExecutor.closed) == 1


@pytest.mark.parametrize("exit_code", [0, 9, 125])
def test_failover_archive_creation_failure_preserves_exit(
    monkeypatch, tmp_path, caplog, exit_code
):
    final = ExecutionResult(exit_code)
    executor, paths = _stream_archive_executor(monkeypatch, tmp_path, [final])

    def fail(*args, **kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(executor_module.tempfile, "mkdtemp", fail)
    assert executor.execute(command=["original"], **paths) is final
    assert len(_FakeExecutor.calls) == 1
    assert "cannot create archive" in caplog.text


def test_failover_archive_write_failure_preserves_exception(
    monkeypatch, tmp_path, caplog
):
    error = RuntimeError("original execution failure")
    executor, paths = _stream_archive_executor(monkeypatch, tmp_path, [error])
    original = Path.write_text

    def fail_receipt(path, *args, **kwargs):
        if path.name == "result.json":
            raise OSError("report unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_receipt)
    with pytest.raises(RuntimeError) as caught:
        executor.execute(command=["original"], **paths)
    assert caught.value is error
    assert "archive failed" in caplog.text
    assert len(_FakeExecutor.calls) == 1


def test_failover_archive_missing_stream_records_error_and_preserves_raw(
    monkeypatch, tmp_path, caplog
):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info, "resolve_ipv4_addresses", lambda host, port: ["10.0.0.1"]
    )
    final = ExecutionResult(7, stdout_raw=b"actual response")
    _FakeExecutor.results = [final]
    executor = FailoverRemoteExecutor(
        "grpc://executor.service:50052", _FakeCAS(), executor_factory=_FakeExecutor
    )
    assert (
        executor.execute(
            command=["original"], stream_stdout_file=tmp_path / "missing.log"
        )
        is final
    )
    root = next(tmp_path.glob("attempts-*/attempt-0001"))
    receipt = json.loads((root / "result.json").read_text())
    assert receipt["collection_errors"] == [
        {"file": "remote_stdout.log", "error_type": "FileNotFoundError"}
    ]
    assert (root / "result_stdout.log").read_bytes() == b"actual response"
    assert "incomplete archive" in caplog.text


def test_failover_without_stream_paths_does_not_create_archive(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info, "resolve_ipv4_addresses", lambda host, port: ["10.0.0.1"]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no stream location requested")

    monkeypatch.setattr(executor_module.tempfile, "mkdtemp", forbidden)
    final = ExecutionResult(0)
    _FakeExecutor.results = [final]
    executor = FailoverRemoteExecutor(
        "grpc://executor.service:50052", _FakeCAS(), executor_factory=_FakeExecutor
    )
    assert executor.execute(command=["original"]) is final
