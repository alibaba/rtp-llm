"""Offline regressions for dispatch and artifact boundaries."""

import io
import json
import os
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from rtp_llm.test.remote_tests import plugin, remote_exec_rtp
from rtp_llm.test.remote_tests.cas_client import UploadBudget
from rtp_llm.test.remote_tests.executor import ExecutionResult
from rtp_llm.test.remote_tests.junit_merge import merge_reports
from rtp_llm.test.remote_tests.outcome_integrity_test import _controller, _result, _xml
from rtp_llm.test.remote_tests.output_collector import _safe_extract


def test_unknown_submission_cannot_be_retried_without_operation_name():
    assert not ExecutionResult(exit_code=-1).safe_to_retry
    assert not ExecutionResult(
        exit_code=-1, operation_name="operations/one"
    ).safe_to_retry
    assert ExecutionResult(exit_code=-1, submission_state="not_sent").safe_to_retry
    assert ExecutionResult(exit_code=-1, operation_done=True).safe_to_retry


def test_outer_retry_stops_on_lost_first_response(monkeypatch):
    controller = plugin.RemoteREAPIPlugin.__new__(plugin.RemoteREAPIPlugin)
    controller._ensure_remote_clients = lambda: None
    calls = []
    result = ExecutionResult(exit_code=-1)
    controller.executor = SimpleNamespace(
        execute=lambda **kw: calls.append(kw) or result
    )
    assert controller._execute_with_retry() is result
    assert len(calls) == 1


def test_collect_only_never_initializes_or_uploads():
    controller = plugin.RemoteREAPIPlugin.__new__(plugin.RemoteREAPIPlugin)
    controller.mode = plugin.RemoteDispatchMode.SESSION
    config = SimpleNamespace(option=SimpleNamespace(collectonly=True))
    controller.config = config
    session = SimpleNamespace(config=config)
    controller._ensure_remote_clients = lambda: pytest.fail(
        "must not initialize transport"
    )
    assert controller.pytest_collection(session) is None
    assert controller.pytest_collection_modifyitems([], config, session) is None
    assert controller.pytest_runtestloop(session) is None


@pytest.mark.parametrize("action_exit", [-1, 1, 124, 137])
def test_stdout_success_never_overrides_action_failure(tmp_path, action_exit):
    controller = _controller(tmp_path)
    result = _result(_xml("a.py::test_ok"))
    result.exit_code = action_exit
    assert controller._parse_remote_output(result, None) != 0


def test_all_cached_removes_only_owned_zero_execution_diagnostic(tmp_path):
    worker = tmp_path / "worker.xml"
    worker.write_text("<testsuite tests='0'/>")
    Path(str(worker) + ".collection.local.json").write_text(
        json.dumps({"selected": ["a.py::test_ok"], "cached": ["a.py::test_ok"]})
    )
    merged = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([worker], merged, required=[worker])
    controller = _controller(tmp_path)
    controller.config.getoption = lambda *a, **kw: False
    controller._session_cached_entries = {
        "a.py::test_ok": SimpleNamespace(
            test_nodeid="a.py::test_ok",
            junit_testcase_xml=(
                '<testcase name="test_ok"><properties><property name="nodeid" '
                'value="a.py::test_ok"/></properties></testcase>'
            ),
        )
    }
    result = _result(merged.read_text(), exit_code=5)
    assert controller._parse_remote_output(result, None) == 0
    # Missing reports are different from legitimate cache deselection.
    root = ET.fromstring(merged.read_text())
    root.find(".//error").text += "\nmissing.xml: required report was not supplied"
    assert (
        controller._parse_remote_output(
            _result(ET.tostring(root, encoding="unicode"), exit_code=5), None
        )
        != 0
    )
    controller.no_cache = True
    assert controller._parse_remote_output(result, None) != 0


def test_upload_budget_deadline_and_cancellation():
    expired = UploadBudget(time.monotonic() - 1, 30)
    with pytest.raises(TimeoutError):
        expired.rpc_timeout()
    budget = UploadBudget(time.monotonic() + 10, 30)
    assert 0 < budget.rpc_timeout() <= 10
    budget.cancelled.set()
    with pytest.raises(TimeoutError):
        budget.remaining()


def test_extract_rejects_sibling_links_and_special_members(tmp_path):
    dest = tmp_path / "output"
    dest.mkdir()
    sibling = tmp_path / "output-other"
    sibling.mkdir()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../output-other"
        archive.addfile(link)
        hardlink = tarfile.TarInfo("indirect-escape")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "escape"
        archive.addfile(hardlink)
        entry = tarfile.TarInfo("escape/leak")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
        entry = tarfile.TarInfo("good")
        entry.size = 2
        archive.addfile(entry, io.BytesIO(b"ok"))
        safe_link = tarfile.TarInfo("good-link")
        safe_link.type = tarfile.SYMTYPE
        safe_link.linkname = "good"
        archive.addfile(safe_link)
        device = tarfile.TarInfo("device")
        device.type = tarfile.CHRTYPE
        archive.addfile(device)
    payload.seek(0)
    with tarfile.open(fileobj=payload) as archive:
        _safe_extract(archive, dest)
    assert not (sibling / "leak").exists()
    assert not (dest / "device").exists()
    assert not (dest / "indirect-escape").is_symlink()
    assert (dest / "good-link").read_bytes() == b"ok"


def test_runtime_archive_tracks_deletion_and_equal_mtime_replacement(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(remote_exec_rtp, "_CORE_RUNTIME_LIBS", ["core.so"])
    libs = tmp_path / "rtp_llm/libs"
    libs.mkdir(parents=True)
    core = libs / "core.so"
    core.write_bytes(b"old")
    other = libs / "other.so"
    other.write_bytes(b"extra")
    archive = tmp_path / remote_exec_rtp._prepare_runtime_libs_archive(tmp_path)
    original = core.stat()
    core.write_bytes(b"new")
    os.utime(core, ns=(original.st_atime_ns, original.st_mtime_ns))
    other.unlink()
    remote_exec_rtp._prepare_runtime_libs_archive(tmp_path)
    with tarfile.open(archive) as stream:
        assert stream.extractfile("rtp_llm/libs/core.so").read() == b"new"
        assert "rtp_llm/libs/other.so" not in stream.getnames()


def test_partial_cache_cannot_hide_an_empty_required_entry(tmp_path):
    from rtp_llm.test.remote_tests.junit_merge import (
        remove_replayed_zero_execution_error,
    )

    paths = [tmp_path / "a.xml", tmp_path / "b.xml"]
    for path in paths:
        path.write_text("<testsuite/>")
    Path(str(paths[0]) + ".collection.local.json").write_text(
        json.dumps({"selected": ["a.py::test_ok"], "cached": ["a.py::test_ok"]})
    )
    Path(str(paths[1]) + ".collection.local.json").write_text(
        json.dumps({"selected": [], "cached": []})
    )
    report = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports(paths, report, required=paths)
    root = ET.fromstring(report.read_text())
    case = ET.fromstring(_xml("a.py::test_ok")).find(".//testcase")
    assert not remove_replayed_zero_execution_error(root, [case])
    assert root.find(".//error") is not None


def test_cache_worker_records_selection_before_its_own_deselection(pytester):
    pytester.makepyfile("def test_one(): pass\ndef test_two(): pass\n")
    cache = pytester.path / "cached.txt"
    nodeid = (
        "test_cache_worker_records_selection_before_its_own_deselection.py::test_one"
    )
    cache.write_text(nodeid + "\n")
    report = pytester.path / "report.xml"
    result = pytester.runpytest(
        "-p",
        "rtp_llm.test.remote_tests.cache_collection",
        "-k",
        "one",
        "--rtp-cache-deselect-file=" + str(cache),
        "--junitxml=" + str(report),
    )
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    evidence = json.loads(Path(str(report) + ".collection.local.json").read_text())
    assert evidence == {"selected": [nodeid], "cached": [nodeid]}


def test_upload_deadline_cancels_transport(monkeypatch, tmp_path):
    budgets = []
    closed = threading.Event()

    def upload(root, files, *, progress, budget):
        budgets.append(budget)
        budget.cancelled.wait(timeout=2)
        budget.remaining()

    cas = SimpleNamespace(upload_directory=upload, close=closed.set)
    monkeypatch.setattr(
        plugin,
        "select_remote_timeout_policy",
        lambda *a, **kw: SimpleNamespace(
            session_budget_seconds=0.02, rpc_timeout_seconds=30
        ),
    )
    config = SimpleNamespace(
        getoption=lambda name, **kw: name == "-q",
        get_terminal_writer=lambda: SimpleNamespace(),
    )
    with pytest.raises(TimeoutError):
        plugin._upload_directory_with_progress(cas, tmp_path, [], config)
    assert closed.is_set()
    assert budgets[0].cancelled.is_set()


def test_cas_rpc_uses_the_smaller_remaining_deadline():
    from rtp_llm.test.remote_tests import remote_execution_pb2 as pb
    from rtp_llm.test.remote_tests.cas_client import CASClient

    observed = []
    client = CASClient.__new__(CASClient)
    client.metadata = []
    client.instance_name = ""
    client.stub = SimpleNamespace(
        FindMissingBlobs=lambda request, **kw: observed.append(kw["timeout"])
        or pb.FindMissingBlobsResponse()
    )
    client._find_missing(
        [pb.Digest(hash="one", size_bytes=1)],
        budget=UploadBudget(time.monotonic() + 1, 30),
    )
    assert len(observed) == 1 and 0 < observed[0] <= 1


def test_failover_does_not_resubmit_when_first_response_is_lost(monkeypatch):
    from rtp_llm.test.remote_tests import executor as execution
    from rtp_llm.test.remote_tests.executor_failover_test import (
        _FakeCAS,
        _FakeExecutor,
        _reset_fake_executor,
    )

    _reset_fake_executor()
    from rtp_llm.test.remote_tests import endpoint_info

    monkeypatch.setattr(
        endpoint_info, "resolve_ipv4_addresses", lambda *a: ["10.0.0.1", "10.0.0.2"]
    )
    result = ExecutionResult(exit_code=-1, infra_category="executor_rpc")
    _FakeExecutor.results = [result, ExecutionResult(exit_code=0)]
    executor = execution.FailoverRemoteExecutor(
        "grpc://unit.invalid:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )
    assert executor.execute(command=["unused"]) is result
    assert len(_FakeExecutor.endpoints) == 1


def test_complete_mixed_cache_replays_empty_entry_without_hiding_fresh_result(tmp_path):
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text("<testsuite/>")
    b.write_text(_xml("b.py::test_ok"))
    for path, selected, cached in [
        (a, ["a.py::test_ok"], ["a.py::test_ok"]),
        (b, ["b.py::test_ok"], []),
    ]:
        Path(str(path) + ".collection.local.json").write_text(
            json.dumps(dict(selected=selected, cached=cached))
        )
    merged = tmp_path / "merged.xml"
    with pytest.raises(ValueError):
        merge_reports([a, b], merged, required=[a, b])
    controller = _controller(tmp_path)
    controller.config.getoption = lambda *a, **kw: False
    case = ET.fromstring(_xml("a.py::test_ok")).find(".//testcase")
    controller._session_cached_entries = {
        "a.py::test_ok": SimpleNamespace(
            test_nodeid="a.py::test_ok",
            junit_testcase_xml=ET.tostring(case, encoding="unicode"),
        )
    }
    assert (
        controller._parse_remote_output(_result(merged.read_text(), exit_code=5), None)
        == 0
    )


def test_execute_receipt_is_durable_before_first_rpc(tmp_path):
    from rtp_llm.test.remote_tests import remote_execution_pb2 as pb
    from rtp_llm.test.remote_tests.executor import RemoteExecutor
    from rtp_llm.test.remote_tests.executor_failover_test import _FakeCAS

    executor = RemoteExecutor.__new__(RemoteExecutor)
    executor.cas = _FakeCAS()
    executor.grpc_uri = "grpc://unit.invalid:50052"
    executor.reapi_targets_combined = "offline"
    executor.metadata = []
    executor.instance_name = ""
    receipts = []

    def execute(request, **kwargs):
        paths = list(tmp_path.glob("execute-attempt-*/submission.json"))
        assert len(paths) == 1
        receipt = json.loads(paths[0].read_text())
        assert receipt["submission_state"] == "unknown"
        assert receipt["operation_name"] is None
        assert receipt["action_digest"] == request.action_digest.hash
        receipts.extend(paths)
        return iter(())  # Lost first response, no operation name.

    executor.stub = SimpleNamespace(Execute=execute)
    result = executor.execute(
        ["unused"],
        pb.Digest(hash="input", size_bytes=1),
        stream_stdout_file=tmp_path / "stdout.log",
    )
    assert result.attempt_receipt == str(receipts[0])
    assert not result.safe_to_retry


def test_cancelled_cas_cannot_open_late_upload_channel(monkeypatch):
    from unittest.mock import Mock

    from rtp_llm.test.remote_tests import cas_client

    client = cas_client.CASClient.__new__(cas_client.CASClient)
    client._channel_lock = threading.Lock()
    client._closed = False
    client.channel = Mock()
    extra = Mock()
    client._extra_channels = [extra]
    client.close()
    monkeypatch.setattr(
        cas_client.grpc,
        "insecure_channel",
        lambda *a, **kw: pytest.fail("late channel"),
    )
    with pytest.raises(RuntimeError, match="closed"):
        client.new_bytestream_stub()
    client.channel.close.assert_called_once()
    extra.close.assert_called_once()
