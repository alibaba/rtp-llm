"""Test the real pytest/report boundary; these fixtures never execute Native ops."""

import hashlib
import importlib.metadata
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from rtp_llm.test.remote_tests import plugin as remote_plugin
from rtp_llm.test.remote_tests.executor import ExecutionResult
from rtp_llm.test.remote_tests.remote_exec_rtp import RemoteRuntimeConfig
from rtp_llm.test.remote_tests.remote_timeout_policy import select_remote_timeout_policy
from rtp_llm.test.smoke_framework.runs_plugin import (
    RepeatExecution,
    _REPEAT_EXECUTION,
    repeat_execution,
)


def _runs_plugin_args():
    """Use the installed entrypoint identity or the source-only module identity."""
    module = "rtp_llm.test.smoke_framework.runs_plugin"
    for entrypoint in importlib.metadata.entry_points(group="pytest11"):
        if entrypoint.name == "smoke-runs-per-test" and entrypoint.value == module:
            return ("-p", entrypoint.name)
    return ("-p", module)


def _controller(root):
    controller = remote_plugin.RemoteREAPIPlugin.__new__(remote_plugin.RemoteREAPIPlugin)
    controller.rootdir = root
    controller._session_cached_entries = {}
    controller._cached_items = {}
    controller._test_cache = None
    controller._cache_manifest = None
    controller._collect_outputs = False
    controller.ci_profile = None
    controller.config = SimpleNamespace(get_terminal_writer=lambda: None)
    controller.executor = SimpleNamespace(reapi_targets_combined="test transport")
    return controller


def _xml(nodeid, outcome="passed"):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    case = ET.SubElement(suite, "testcase", name="test_func", time="0.25")
    props = ET.SubElement(case, "properties")
    ET.SubElement(props, "property", name="nodeid", value=nodeid)
    if outcome in {"xfail", "xpass"}:
        ET.SubElement(props, "property", name="rtp_remote_wasxfail", value="expected reason")
    if outcome in {"skip", "xfail"}:
        ET.SubElement(case, "skipped", type="pytest.xfail" if outcome == "xfail" else "pytest.skip", message="reason")
    if outcome in {"failure", "error"}:
        ET.SubElement(case, outcome, message="real worker assertion")
    return ET.tostring(root, encoding="unicode")


def _result(xml, exit_code=0, response_status_code=0):
    stdout = f"EXIT_CODE={exit_code}\n<<<JUNIT_XML>>>{xml}<<<END_JUNIT_XML>>>"
    return ExecutionResult(exit_code=exit_code, response_status_code=response_status_code, stdout_raw=stdout.encode())


@pytest.mark.parametrize("outcome, expected", [
    ("passed", "passed"), ("skip", "skipped"), ("xfail", "skipped"),
    ("xpass", "passed"), ("failure", "failed"), ("error", "failed"),
])
def test_per_test_reports_preserve_worker_outcome(pytester, outcome, expected):
    item = pytester.getitem("def test_func(): pass")
    recorder = pytester.make_hook_recorder(item.config.pluginmanager)
    controller = _controller(pytester.path)
    nodeid = remote_plugin._per_test_nodeid(item, pytester.path)
    controller._report_per_test(item, _result(_xml(nodeid, outcome)))
    reports = recorder.getreports("pytest_runtest_logreport")
    call = next(report for report in reports if report.when == "call")
    assert call.outcome == expected
    assert call.duration == 0.25
    assert getattr(call, "wasxfail", None) == ("expected reason" if outcome in {"xfail", "xpass"} else None)
    assert dict(call.user_properties)["remote_cached"] == "false"


@pytest.mark.parametrize("bad", ["", "<broken", "<testsuite/>", "wrong_identity"])
def test_per_test_missing_or_wrong_worker_report_fails(pytester, bad):
    item = pytester.getitem("def test_func(): pass")
    recorder = pytester.make_hook_recorder(item.config.pluginmanager)
    controller = _controller(pytester.path)
    xml = _xml("other.py::test_func") if bad == "wrong_identity" else bad
    controller._report_per_test(item, _result(xml))
    call = next(report for report in recorder.getreports("pytest_runtest_logreport") if report.when == "call")
    assert call.failed


def test_strict_profile_rejects_remote_skip(pytester):
    item = pytester.getitem("def test_func(): pass")
    recorder = pytester.make_hook_recorder(item.config.pluginmanager)
    controller = _controller(pytester.path)
    controller.config._rtp_ci_forbid_skips = True
    controller._report_per_test(item, _result(_xml(remote_plugin._per_test_nodeid(item, pytester.path), "skip")))
    assert next(report for report in recorder.getreports("pytest_runtest_logreport") if report.when == "call").failed


@pytest.mark.parametrize("count, expected", [(1, 2), (2, 2), (3, 2)])
def test_session_cache_count_is_validated_after_no_tests_normalization(tmp_path, count, expected):
    controller = _controller(tmp_path)
    controller.ci_profile = "fixture_profile"
    controller.config._rtp_ci_minimum_count = 1
    controller.config._rtp_ci_expected_count = expected
    controller.config.getoption = lambda name, default=None: False
    controller._session_cached_entries = {
        str(i): SimpleNamespace(junit_testcase_xml=(
            f'<testcase name="test_{i}" time="0.1"><properties>'
            f'<property name="nodeid" value="pkg/test.py::test_{i}"/>'
            '</properties></testcase>'
        )) for i in range(count)
    }
    assert controller._parse_remote_output(_result('<testsuites><testsuite tests="0"/></testsuites>', exit_code=5), None) == (0 if count == expected else 4)


@pytest.mark.parametrize("failure", ["missing", "malformed", "failure", "error"])
def test_session_zero_exit_cannot_hide_missing_or_failed_report(tmp_path, failure):
    controller = _controller(tmp_path)
    xml = "" if failure == "missing" else "<broken" if failure == "malformed" else _xml("pkg/test.py::test_func", failure)
    assert controller._parse_remote_output(_result(xml), None) != 0


def test_session_infra_error_cannot_be_reset_by_cache(tmp_path):
    controller = _controller(tmp_path)
    controller._session_cached_entries = {"cached": SimpleNamespace(junit_testcase_xml='<testcase name="cached" time="0.1"/>')}
    assert controller._parse_remote_output(_result("", exit_code=5, response_status_code=14), None) != 0


def test_replicas_select_original_worker_test_and_have_distinct_actions(pytester):
    pytester.makepyfile("def test_func(): pass")
    _, recorder = pytester.inline_genitems(*_runs_plugin_args(), "--runs-per-test=3")
    items = recorder.getcalls("pytest_collection_finish")[0].session.items
    assert len(items) == 3
    controller = _controller(pytester.path)
    controller.config.option = SimpleNamespace(markexpr="")
    controller.timeout_policy = select_remote_timeout_policy("py_ut_sm8x", per_test=True)
    runtime = RemoteRuntimeConfig(ignore_args=[], env_vars={}, platform_properties={"gpu": "H20", "gpu_count": "1"}, remote_setup_prefix="")
    commands = []
    for item in items:
        item.add_marker(pytest.mark.gpu("H20", count=1))
        replica = repeat_execution(item)
        assert replica is not None
        assert replica.nodeid == items[0].nodeid
        worker_nodeid = remote_plugin._per_test_nodeid(item, pytester.path)
        assert worker_nodeid.endswith("::test_func")
        shell = controller._build_command(item, runtime)
        assert "RTP_REMOTE_TEST_REPLICA=" in shell[2]
        assert replica.action_identity in shell[2]
        assert "test_func[run" not in shell[2].split("python -m pytest", 1)[1].split("2>&1; ec=$?", 1)[0]
        commands.append(hashlib.sha256(json.dumps(shell).encode()).hexdigest())
    assert len(set(commands)) == 3
    assert [repeat_execution(item).index for item in items] == [1, 2, 3]


def test_repeated_execution_never_stores_a_success_cache_entry(pytester):
    item = pytester.getitem("def test_func(): pass")
    item.stash[_REPEAT_EXECUTION] = RepeatExecution(item.nodeid, 1, 2, "invocation")
    controller = _controller(pytester.path)
    controller._test_cache = object()
    controller._cache_manifest = {}
    controller._store_test_result_in_cache = lambda *args: pytest.fail("replicas must not populate result cache")
    controller._report_per_test(item, _result(_xml(remote_plugin._per_test_nodeid(item, pytester.path))))


def test_worker_junit_metadata_preserves_xfail_classification(pytester):
    item = pytester.getitem("def test_func(): pass")
    controller = _controller(pytester.path)
    controller.config.option = SimpleNamespace(markexpr="")
    controller.timeout_policy = select_remote_timeout_policy("py_ut_sm8x", per_test=True)
    item.add_marker(pytest.mark.gpu("H20", count=1))
    runtime = RemoteRuntimeConfig(ignore_args=[], env_vars={}, platform_properties={}, remote_setup_prefix="")
    shell = controller._build_command(item, runtime)[2]
    inner_script = shlex.split(shell)[-1]
    metadata_plugin = inner_script.split("<< '_PER_TEST_JUNIT_PY_'\n", 1)[1].split("\n_PER_TEST_JUNIT_PY_", 1)[0]
    pytester.makepyfile(rtp_remote_per_test_junit_plugin=metadata_plugin)
    tests = pytester.makepyfile(test_worker_outcomes='''
        import pytest
        def test_pass(): assert True
        def test_skip(): pytest.skip("fixture skip")
        @pytest.mark.xfail(reason="expected reason")
        def test_xfail(): assert False
        @pytest.mark.xfail(reason="unexpected reason")
        def test_xpass(): assert True
    ''')
    xml = pytester.path / "worker.xml"
    result = pytester.runpytest_subprocess("-p", "rtp_remote_per_test_junit_plugin", str(tests), f"--junitxml={xml}", "-q")
    result.assert_outcomes(passed=1, skipped=1, xfailed=1, xpassed=1)
    cases = {case.get("name"): case for case in ET.parse(xml).iter("testcase")}
    assert cases["test_xfail"].find("skipped").get("type") == "pytest.xfail"
    for name, reason in [("test_xfail", "expected reason"), ("test_xpass", "unexpected reason")]:
        properties = {prop.get("name"): prop.get("value") for prop in cases[name].findall("./properties/property")}
        assert properties["rtp_remote_wasxfail"] == reason
        assert properties["nodeid"].endswith("::" + name)


def test_repeated_local_runs_keep_fresh_fixture_lifecycles(pytester):
    output = pytester.path / "executed.txt"
    tests = pytester.makepyfile(test_lifecycle=f'''
        import pytest
        import uuid
        from pathlib import Path
        @pytest.fixture
        def state():
            value = []
            yield value
            assert len(value) == 1
        def test_fresh(state):
            assert state == []
            state.append(uuid.uuid4().hex)
            with Path({str(output)!r}).open("a") as stream:
                stream.write(state[0] + "\\n")
    ''')
    result = pytester.runpytest_subprocess(*_runs_plugin_args(), str(tests), "--runs-per-test=3", "-q")
    result.assert_outcomes(passed=3)
    executions = output.read_text().splitlines()
    assert len(executions) == len(set(executions)) == 3


def test_repeat_dispatch_bypasses_manifest_and_action_caches(pytester, monkeypatch):
    from rtp_llm.test.remote_tests import remote_execution_pb2
    pytester.makepyfile("def test_func(): pass")
    _, recorder = pytester.inline_genitems(*_runs_plugin_args(), "--runs-per-test=2")
    items = recorder.getcalls("pytest_collection_finish")[0].session.items
    for item in items:
        item.add_marker(pytest.mark.gpu("H20", count=1))
    controller = _controller(pytester.path)
    controller.config.option = SimpleNamespace(markexpr="")
    controller.timeout_policy = select_remote_timeout_policy("py_ut_sm8x", per_test=True)
    controller._input_root = remote_execution_pb2.Digest(hash="original-input", size_bytes=1)
    controller._test_cache_enabled = True
    controller._test_cache = SimpleNamespace(
        load_manifest=lambda digest: {"tests": {}},
        lookup=lambda *args: pytest.fail("replica must never read a result cache entry"),
    )
    controller._ensure_test_cache = lambda: None
    controller._ensure_uploaded = lambda items: None
    controller._apply_controller_timeout_marker = lambda item: None
    controller._futures = {}
    controller.concurrency = 2
    controller.no_cache = False
    controller.timeout = 60
    runtime = RemoteRuntimeConfig(ignore_args=[], env_vars={}, platform_properties={}, remote_setup_prefix="")
    monkeypatch.setattr(remote_plugin, "build_runtime_config", lambda *args, **kwargs: runtime)
    requests = []
    def capture_dispatch(**kwargs):
        requests.append(kwargs)
        return ExecutionResult(exit_code=0)
    controller._execute_queued_with_retry = capture_dispatch
    controller._per_test_collection_modifyitems(items)
    for future in controller._futures.values():
        future.result(timeout=5)
    controller._pool.shutdown()
    assert len(requests) == 2
    assert all(request["no_cache"] for request in requests)
    assert requests[0]["command"] != requests[1]["command"]
    assert not controller._cached_items


@pytest.mark.parametrize("xml", ["", "<broken", "<not-junit/>"])
def test_cached_cases_cannot_mask_missing_worker_report(tmp_path, xml):
    controller = _controller(tmp_path)
    controller._session_cached_entries = {"cached": SimpleNamespace(junit_testcase_xml='<testcase name="cached" time="0.1"/>')}
    assert controller._parse_remote_output(_result(xml, exit_code=5), None) != 0
