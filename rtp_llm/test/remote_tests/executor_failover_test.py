import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from google.protobuf.message import DecodeError

from rtp_llm.test import ci_profile_plugin
from rtp_llm.test.remote_tests import (
    cas_client,
    endpoint_info,
    executor as executor_module,
    output_collector,
    plugin as remote_plugin,
    remote_exec_rtp,
    remote_execution_pb2,
)
from rtp_llm.test.remote_tests.executor import (
    ExecutionResult,
    FailoverRemoteExecutor,
    RemoteExecutor,
)
from rtp_llm.test.remote_tests.remote_timeout_policy import select_remote_timeout_policy
from rtp_llm.test.utils import device_resource


class _FakeCAS:
    grpc_uri = "grpc://cas.service:50051"

    def upload_blob(self, data):
        return remote_execution_pb2.Digest(hash="abc", size_bytes=len(data))

    def download_blob(self, digest):
        return b""


class _CapturingCAS(_FakeCAS):
    def __init__(self):
        self.uploaded_blobs = []

    def upload_blob(self, data):
        self.uploaded_blobs.append(data)
        return remote_execution_pb2.Digest(
            hash=f"blob-{len(self.uploaded_blobs)}", size_bytes=len(data)
        )


class _FakeExecutor:
    results = []
    endpoints = []
    cancelled = []
    closed = []
    calls = []

    def __init__(self, endpoint, cas, metadata):
        self.grpc_uri = endpoint
        self.reapi_targets_combined = f"cas={cas.grpc_uri} | executor={endpoint}"
        self.endpoints.append(endpoint)

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)

    def cancel_operation(self, operation_name, timeout=5):
        self.cancelled.append(operation_name)
        return True

    def close(self):
        self.closed.append(self.grpc_uri)

    def download_output(self, digest):
        return ""


def _reset_fake_executor():
    _FakeExecutor.results = []
    _FakeExecutor.endpoints = []
    _FakeExecutor.cancelled = []
    _FakeExecutor.closed = []
    _FakeExecutor.calls = []


class _FakeCollectedItem:
    def __init__(self, path: Path, *, smoke: bool = False, perf: bool = False):
        self.fspath = str(path)
        self._smoke = smoke
        self._perf = perf

    def get_closest_marker(self, name):
        if name == "smoke" and self._smoke:
            return object()
        if name == "perf" and self._perf:
            return object()
        return None


class _FakeRemoteItem:
    name = "test_case"
    nodeid = "test_remote.py::test_case"

    def __init__(self, path: Path, *, gpu_type: str = "H20", gpu_count: int = 1):
        self.fspath = str(path)
        self._gpu_type = gpu_type
        self._gpu_count = gpu_count

    def get_closest_marker(self, name):
        if name == "gpu":
            return SimpleNamespace(
                kwargs={"type": self._gpu_type, "count": self._gpu_count}
            )
        return None


def test_safe_rel_maps_sibling_internal_source_to_worker_path(tmp_path):
    rootdir = tmp_path / "repo" / "github-opensource"
    item_path = (
        tmp_path
        / "repo"
        / "internal_source"
        / "rtp_llm"
        / "test"
        / "smoke"
        / "suites"
        / "test_smoke_h20_dense_internal.py"
    )
    item_path.parent.mkdir(parents=True)
    item_path.write_text("def test_placeholder(): pass\n")
    rootdir.mkdir(parents=True)

    rel = remote_exec_rtp._safe_rel_to_rootdir(item_path.resolve(), rootdir.resolve())

    assert (
        rel
        == "internal_source/rtp_llm/test/smoke/suites/test_smoke_h20_dense_internal.py"
    )
    assert ".." not in Path(rel).parts


def test_collect_remote_files_checks_smoke_lfs_pointers(tmp_path):
    repo = tmp_path / "repo"
    rootdir = repo / "github-opensource"
    internal = repo / "internal_source"
    suite = internal / "rtp_llm" / "test" / "smoke" / "suites" / "test_smoke.py"
    pointer = internal / "rtp_llm" / "test" / "smoke" / "data" / "model.bin"
    suite.parent.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    suite.write_text("def test_placeholder(): pass\n")
    pointer.write_bytes(b"oss-lfs v1\n")
    rootdir.mkdir()
    (rootdir / "internal_source").symlink_to("../internal_source")

    item = _FakeCollectedItem(suite.resolve(), smoke=True)

    try:
        remote_exec_rtp.collect_remote_files(rootdir.resolve(), [item])
    except RuntimeError as exc:
        assert "oss-lfs pointer" in str(exc)
        assert "internal_source/rtp_llm/test/smoke/data/model.bin" in str(exc)
    else:
        raise AssertionError("expected collect_remote_files to reject LFS pointers")


def test_collect_remote_files_includes_perf_data(tmp_path):
    repo = tmp_path / "repo"
    rootdir = repo / "github-opensource"
    internal = repo / "internal_source"
    suite = internal / "rtp_llm" / "test" / "perf_test" / "suites" / "test_perf.py"
    data = (
        internal
        / "rtp_llm"
        / "test"
        / "perf_test"
        / "test_data"
        / "qwen"
        / "distribution.csv"
    )
    baseline = (
        internal
        / "rtp_llm"
        / "test"
        / "perf_test"
        / "baselines"
        / "qwen_perf.json"
    )
    suite.parent.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    baseline.parent.mkdir(parents=True)
    suite.write_text("def test_placeholder(): pass\n")
    data.write_text("upper,count\n128,1\n")
    baseline.write_text("{}\n")
    rootdir.mkdir()
    (rootdir / "internal_source").symlink_to("../internal_source")

    files = remote_exec_rtp.collect_remote_files(
        rootdir.resolve(), [_FakeCollectedItem(suite.resolve(), perf=True)]
    )

    assert (
        "internal_source/rtp_llm/test/perf_test/test_data/qwen/distribution.csv"
        in files
    )
    assert "internal_source/rtp_llm/test/perf_test/baselines/qwen_perf.json" in files


class _FakeProfileConfig:
    def __init__(self, *, remote_session: bool):
        self.rootpath = Path(".")
        self.args = []
        self.option = SimpleNamespace(
            markexpr="",
            remote_gpu_type=None,
            timeout=None,
            tbstyle=None,
            verbose=0,
        )
        self._remote_session = remote_session

    def getoption(self, name, default=None):
        if name == "--rtp-ci-profile":
            return "py_ut_sm9x"
        if name == "--remote-session":
            return self._remote_session
        if name == "--timeout":
            return None
        return default


def test_reapi_smoke_uses_per_test_remote_profile_gpu_type_does_not_override_marker(
    monkeypatch,
):
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_pytest_ci_section",
        lambda root: {"default_pytest_cli": ""},
    )
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_profile",
        lambda root, name: {"markexpr": "H20", "gpu_type": "H20"},
    )

    per_test_config = _FakeProfileConfig(remote_session=False)
    ci_profile_plugin.pytest_configure(per_test_config)

    assert per_test_config.option.markexpr == "H20"
    assert per_test_config.option.remote_gpu_type is None

    session_config = _FakeProfileConfig(remote_session=True)
    ci_profile_plugin.pytest_configure(session_config)

    assert session_config.option.remote_gpu_type == "H20"


def test_remote_session_rejects_non_pyut_profile():
    class _PluginConfig(_FakeProfileConfig):
        def getoption(self, name, default=None):
            if name == "--rtp-ci-profile":
                return "smoke_h20_internal"
            if name == "--remote-session":
                return True
            if name == "--remote":
                return False
            return super().getoption(name, default)

    with pytest.raises(pytest.UsageError) as excinfo:
        remote_plugin.pytest_configure(_PluginConfig(remote_session=True))

    assert "--remote-session is only supported for py-ut profiles" in str(
        excinfo.value
    )


def test_per_test_command_exports_marker_gpu_count(tmp_path):
    rootdir = tmp_path / "repo"
    test_file = rootdir / "test_remote.py"
    rootdir.mkdir()
    test_file.write_text("def test_case(): pass\n")
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.rootdir = rootdir
    plugin.config = SimpleNamespace(option=SimpleNamespace(markexpr=""))
    plugin._collect_outputs = False
    plugin.timeout_policy = select_remote_timeout_policy(
        "smoke_h20_internal", per_test=True
    )

    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[],
        env_vars={},
        platform_properties={"gpu": "H20", "gpu_count": "1"},
        remote_setup_prefix="",
    )
    command = plugin._build_command(
        _FakeRemoteItem(test_file, gpu_type="H20", gpu_count=1), runtime
    )
    shell = command[2]

    assert "export GPU_COUNT=1;" in shell
    assert "export WORLD_SIZE=1;" in shell
    assert "export GPU_COUNT_PER_WORKER=1;" in shell
    assert shell.index("export GPU_COUNT=1;") < shell.index(
        "python rtp_llm/test/utils/device_resource.py"
    )


def test_session_command_locks_total_gpu_pool_and_slices_workers():
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.workers = 4
    plugin._collect_outputs = False
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr="H20", keyword=""),
        rootpath=Path("."),
    )
    plugin.timeout_policy = select_remote_timeout_policy("py_ut_sm8x", per_test=False)

    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[],
        env_vars={},
        platform_properties={"gpu": "H20", "gpu_count": "4"},
        remote_setup_prefix="",
    )
    command = plugin._build_session_command("", runtime, ci_profile=None)
    shell = command[2]

    assert (
        "export GPU_COUNT=4; unset WORLD_SIZE; export GPU_COUNT_PER_WORKER=1;"
        in shell
    )
    assert (
        "export GPU_COUNT=4; unset WORLD_SIZE; export GPU_COUNT_PER_WORKER=2;"
        in shell
    )
    assert (
        "export GPU_COUNT=4; unset WORLD_SIZE; export GPU_COUNT_PER_WORKER=4;"
        in shell
    )


def test_executor_pool_resolves_hostname_inside_remote_framework(monkeypatch):
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )

    pool = endpoint_info.ExecutorEndpointPool("grpc://scheduler.example.test:50052")

    assert pool.source_uri == "grpc://scheduler.example.test:50052"
    assert pool.current_endpoint() == "grpc://10.0.0.1:50052"
    assert pool.advance() == "grpc://10.0.0.2:50052"


def test_timeout_policy_maps_ci_profiles():
    ut = select_remote_timeout_policy("ut-sm9x", per_test=False)
    smoke = select_remote_timeout_policy("smoke-sm100-internal", per_test=False)
    perf = select_remote_timeout_policy("perf-test", per_test=False)
    per_test = select_remote_timeout_policy("ut-sm9x", per_test=True)
    per_test_smoke = select_remote_timeout_policy("smoke-ppu-internal", per_test=True)
    per_test_perf = select_remote_timeout_policy("perf-sm9x", per_test=True)

    assert (
        ut.session_budget_seconds,
        ut.action_timeout_seconds,
        ut.supervisor_timeout_seconds,
        ut.pytest_timeout_seconds,
    ) == (3000, 1500, 1440, 300)
    assert (
        smoke.session_budget_seconds,
        smoke.action_timeout_seconds,
        smoke.supervisor_timeout_seconds,
        smoke.pytest_timeout_seconds,
    ) == (3600, 3000, 2880, 600)
    assert (
        perf.session_budget_seconds,
        perf.action_timeout_seconds,
        perf.supervisor_timeout_seconds,
        perf.pytest_timeout_seconds,
    ) == (6600, 6300, 6120, 1800)
    assert (
        per_test.session_budget_seconds,
        per_test.action_timeout_seconds,
        per_test.supervisor_timeout_seconds,
        per_test.pytest_timeout_seconds,
        per_test.queued_timeout_seconds,
    ) == (300, 150, 130, 100, 60)
    assert (
        per_test_smoke.session_budget_seconds,
        per_test_smoke.action_timeout_seconds,
        per_test_smoke.supervisor_timeout_seconds,
        per_test_smoke.pytest_timeout_seconds,
        per_test_smoke.queued_timeout_seconds,
        per_test_smoke.heartbeat_stall_seconds,
    ) == (3300, 3000, 2880, 600, 180, 2400)
    assert (
        per_test_perf.session_budget_seconds,
        per_test_perf.action_timeout_seconds,
        per_test_perf.supervisor_timeout_seconds,
        per_test_perf.pytest_timeout_seconds,
        per_test_perf.min_retry_remaining_seconds,
    ) == (6600, 6300, 6120, 1800, 1200)


def _done_operation(exit_code=0):
    op = remote_execution_pb2.Operation(name="operations/done", done=True)
    resp = remote_execution_pb2.ExecuteResponse(
        result=remote_execution_pb2.ActionResult(exit_code=exit_code)
    )
    op.response.Pack(resp)
    return op


def test_execute_uses_action_timeout_and_rpc_deadline():
    cas = _CapturingCAS()
    executor = RemoteExecutor("grpc://scheduler.example.test:50052", cas)
    seen = {}

    class _Stub:
        def Execute(self, request, metadata, timeout):
            seen["rpc_timeout"] = timeout
            yield _done_operation()

    executor.stub = _Stub()

    result = executor.execute(
        command=["bash", "-c", "true"],
        input_root_digest=remote_execution_pb2.Digest(hash="root", size_bytes=1),
        timeout=7200,
        action_timeout_seconds=1500,
        rpc_timeout_seconds=1620,
        queued_timeout_seconds=300,
    )

    assert result.exit_code == 0
    assert seen["rpc_timeout"] == 1620
    action_timeouts = []
    for data in cas.uploaded_blobs:
        action = remote_execution_pb2.Action()
        try:
            action.ParseFromString(data)
        except DecodeError:
            continue
        if action.timeout.seconds:
            action_timeouts.append(action.timeout.seconds)
    assert action_timeouts == [1500]


def test_explicit_queued_watchdog_policy_is_independent_of_action_timeout():
    assert RemoteExecutor._queued_watchdog_seconds(1500, configured_seconds=60) == 60


def test_failover_budget_keeps_per_test_smoke_action_timeout():
    policy = select_remote_timeout_policy("smoke-ppu-internal", per_test=True)

    kwargs = FailoverRemoteExecutor._kwargs_for_budget(
        {
            "action_timeout_seconds": policy.action_timeout_seconds,
            "rpc_timeout_seconds": policy.rpc_timeout_seconds,
            "queued_timeout_seconds": policy.queued_timeout_seconds,
        },
        remaining=policy.session_budget_seconds,
    )

    assert kwargs["action_timeout_seconds"] == policy.action_timeout_seconds
    assert kwargs["queued_timeout_seconds"] == policy.queued_timeout_seconds


def test_device_resource_remote_child_inherits_output_without_pipe(monkeypatch):
    seen = {}

    class _Popen:
        pid = 12345
        returncode = 0

        def __init__(self, argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setenv("RTP_REMOTE_SESSION_ID", "session-output")
    monkeypatch.setattr(device_resource.subprocess, "Popen", _Popen)
    monkeypatch.setattr(device_resource, "_session_pids", lambda *args, **kwargs: [])

    assert device_resource._run_child(["python", "-c", "print('x')"]) == 0
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["env"]["RTP_DEVICE_RESOURCE_OWNER_PID"]
    assert "stdout" not in seen["kwargs"]
    assert "stderr" not in seen["kwargs"]


def test_executor_pool_samples_single_ip_service_discovery_answers(monkeypatch):
    answers = iter([["10.0.0.1"], ["10.0.0.2"]])

    monkeypatch.setattr(endpoint_info, "_FORCED_RESOLVE_SLEEP_SECONDS", 0)
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: next(answers, ["10.0.0.2"]),
    )

    pool = endpoint_info.ExecutorEndpointPool("grpc://scheduler.example.test:50052")

    assert pool.endpoints() == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert pool.advance() == "grpc://10.0.0.2:50052"


def test_executor_pool_falls_back_when_primary_hostname_unresolved(monkeypatch):
    def fake_resolve(host, port):
        if host == "scheduler.example.test":
            return []
        if host == "scheduler.daily":
            return ["10.0.0.9"]
        return []

    monkeypatch.setattr(endpoint_info, "resolve_ipv4_addresses", fake_resolve)

    pool = endpoint_info.ExecutorEndpointPool(
        "grpc://scheduler.example.test:50052",
        fallback_uri="grpc://scheduler.daily:50052",
    )

    assert pool.source_uri == "grpc://scheduler.example.test:50052"
    assert pool.active_source_uri == "grpc://scheduler.daily:50052"
    assert pool.current_endpoint() == "grpc://10.0.0.9:50052"


def test_executor_pool_appends_fallback_endpoints_for_failover(monkeypatch):
    def fake_resolve(host, port):
        if host == "scheduler.example.test":
            return ["10.0.0.1"]
        if host == "scheduler.daily":
            return ["10.0.0.9"]
        return []

    monkeypatch.setattr(endpoint_info, "resolve_ipv4_addresses", fake_resolve)

    pool = endpoint_info.ExecutorEndpointPool(
        "grpc://scheduler.example.test:50052",
        fallback_uri="grpc://scheduler.daily:50052",
    )

    assert pool.endpoints() == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.9:50052",
    ]
    assert pool.advance() == "grpc://10.0.0.9:50052"


def test_executor_pool_does_not_fallback_for_literal_ip(monkeypatch):
    calls = []

    def fake_resolve(host, port):
        calls.append((host, port))
        return ["10.0.0.9"]

    monkeypatch.setattr(endpoint_info, "resolve_ipv4_addresses", fake_resolve)

    pool = endpoint_info.ExecutorEndpointPool(
        "grpc://127.0.0.1:50052",
        fallback_uri="grpc://scheduler.daily:50052",
    )

    assert pool.current_endpoint() == "grpc://127.0.0.1:50052"
    assert pool.active_source_uri == "grpc://127.0.0.1:50052"
    assert calls == []


def test_default_reapi_endpoints_keep_hostnames(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp-llm": {
                    "remote": {
                        "executor-daily": "scheduler.example",
                        "cas-daily": "cas.example",
                        "executor-port": 50052,
                        "cas-port": 50051,
                    }
                }
            }
        },
    )

    executor_ep, cas_ep = remote_exec_rtp.resolve_default_reapi_endpoints(
        rootdir=Path("."),
        env="daily",
    )

    assert executor_ep == "grpc://scheduler.example:50052"
    assert cas_ep == "grpc://cas.example:50051"


def test_remote_setup_eviction_uses_venv_lock():
    command = remote_exec_rtp.build_remote_setup_command(Path("."))

    assert "evict_locked_venvs" in command
    assert 'flock -n "$lock" rm -rf "$d"' in command
    assert (
        "find /home/admin/venvs -maxdepth 1 -type d -name 'rtp-llm-*'   "
        "-mtime +7 -exec rm -rf" not in command
    )
    assert "evict_locked_venvs -mmin +360" in command
    assert "evict_locked_venvs -mmin +60" in command
    assert "restored rtp_llm/libs from runtime libs archive" in command


def test_remote_setup_exports_profile_env():
    command = remote_exec_rtp.build_remote_setup_command(
        Path("."), setup_env={"RTP_BAZEL_CONFIG": "--config=custom"}
    )

    assert "export RTP_BAZEL_CONFIG=--config=custom;" in command
    assert (
        "RTP_BAZEL_CONFIG=--config=custom /opt/conda310/bin/python "
        "internal_source/ci/prepare_venv.py"
    ) in command


def test_sm100_markexpr_prefers_arm_pool_alias():
    gpu_type = remote_exec_rtp.infer_gpu_type_from_markexpr(
        "manual and smoke and (SM100 or SM100_ARM)"
    )
    runtime = remote_exec_rtp.build_runtime_config(
        Path("."),
        remote_exec_rtp.GPURequest(gpu_type=gpu_type, gpu_count=4),
    )

    assert gpu_type == "SM100_ARM"
    assert runtime.platform_properties["gpu"] == "SM100_ARM"


def test_sm100_deprecated_marker_maps_to_arm_reapi_pool():
    runtime = remote_exec_rtp.build_runtime_config(
        Path("."),
        remote_exec_rtp.GPURequest(gpu_type="SM100", gpu_count=4),
    )

    assert runtime.platform_properties["gpu"] == "SM100_ARM"


def test_resolve_ci_profile_gpu_type(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp_llm": {
                    "pytest_ci": {
                        "profiles": {"smoke_sm100_internal": {"gpu_type": "SM100_ARM"}}
                    }
                }
            }
        },
    )

    assert (
        remote_exec_rtp.resolve_ci_profile_gpu_type(Path("."), "smoke_sm100_internal")
        == "SM100_ARM"
    )


def test_resolve_ci_profile_remote_env(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp_llm": {
                    "pytest_ci": {
                        "profiles": {
                            "custom_remote": {
                                "remote_env": {
                                    "RTP_BAZEL_CONFIG": "--config=custom"
                                }
                            }
                        }
                    }
                }
            }
        },
    )

    assert remote_exec_rtp.resolve_ci_profile_remote_env(
        Path("."), "custom_remote"
    ) == {"RTP_BAZEL_CONFIG": "--config=custom"}


def test_build_runtime_config_uses_gpu_type_remote_env(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp_llm": {
                    "pytest_ci": {
                        "profiles": {
                            "custom_remote": {
                                "gpu_type": "CUSTOM_GPU",
                                "remote_env": {
                                    "RTP_BAZEL_CONFIG": "--config=custom"
                                },
                            }
                        }
                    }
                }
            }
        },
    )

    runtime = remote_exec_rtp.build_runtime_config(
        Path("."), remote_exec_rtp.GPURequest(gpu_type="CUSTOM_GPU", gpu_count=4)
    )

    assert runtime.env_vars["RTP_BAZEL_CONFIG"] == "--config=custom"
    assert "export RTP_BAZEL_CONFIG=--config=custom;" in runtime.remote_setup_prefix


def test_collect_session_files_packs_runtime_lib_archive(tmp_path):
    (tmp_path / "rtp_llm" / "libs").mkdir(parents=True)
    (tmp_path / "rtp_llm" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "rtp_llm" / "sample.py").write_text("x = 1\n", encoding="utf-8")
    for name in (
        "libth_transformer_config.so",
        "libth_transformer.so",
        "librtp_compute_ops.so",
        "libdependency.so.1",
    ):
        p = tmp_path / "rtp_llm" / "libs" / name
        p.write_bytes(f"{name}\n".encode())
        p.chmod(0o755)

    files = remote_exec_rtp.collect_session_files(tmp_path)

    archive_rel = ".pytest_cache/remote_inputs/rtp_llm_libs.tar"
    assert archive_rel in files
    assert "rtp_llm/sample.py" in files
    assert "rtp_llm/libs/libth_transformer_config.so" not in files
    with tarfile.open(tmp_path / archive_rel, "r") as tar:
        names = set(tar.getnames())
    assert "rtp_llm/libs/libth_transformer_config.so" in names
    assert "rtp_llm/libs/libdependency.so.1" in names


def test_collect_session_files_requires_staged_runtime_libs(tmp_path):
    (tmp_path / "rtp_llm" / "libs").mkdir(parents=True)

    try:
        remote_exec_rtp.collect_session_files(tmp_path)
    except RuntimeError as exc:
        assert "Run `python setup.py build_ext --inplace` first" in str(exc)
    else:
        raise AssertionError("collect_session_files should require staged runtime libs")


def test_cas_find_missing_retries_transient_unavailable(monkeypatch):
    class _TransientRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

    class _Stub:
        calls = 0

        def FindMissingBlobs(self, request, metadata):
            self.calls += 1
            if self.calls == 1:
                raise _TransientRpcError()
            return remote_execution_pb2.FindMissingBlobsResponse(
                missing_blob_digests=[
                    remote_execution_pb2.Digest(hash="abc", size_bytes=1)
                ]
            )

    monkeypatch.setattr(cas_client.time, "sleep", lambda _: None)
    client = object.__new__(cas_client.CASClient)
    client.stub = _Stub()
    client.instance_name = ""
    client.metadata = []

    missing = client._find_missing(
        [remote_execution_pb2.Digest(hash="abc", size_bytes=1)]
    )

    assert missing == {"abc"}
    assert client.stub.calls == 2


def test_online_default_endpoint_config_adds_daily_executor_fallback(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp-llm": {
                    "remote": {
                        "executor-online": "scheduler.example.test",
                        "cas-online": "cas.example.test",
                        "executor-daily": "scheduler.daily",
                        "cas-daily": "cas.daily",
                        "executor-port": 50052,
                        "cas-port": 50051,
                    }
                }
            }
        },
    )

    endpoints = remote_exec_rtp.resolve_default_reapi_endpoint_config(
        rootdir=Path("."),
        env="online",
    )

    assert endpoints.executor == "grpc://scheduler.example.test:50052"
    assert endpoints.cas == "grpc://cas.example.test:50051"
    assert endpoints.fallback_executor == "grpc://scheduler.daily:50052"


def test_failover_retries_on_next_executor_ip(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [
        ExecutionResult(
            exit_code=-1,
            infra_category="executor_rpc",
            operation_name="operations/1",
            last_stage="QUEUED",
        ),
        ExecutionResult(exit_code=0),
    ]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "true"])

    assert result.exit_code == 0
    assert result.failover_attempts == 1
    assert _FakeExecutor.endpoints == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert _FakeExecutor.cancelled == ["operations/1"]


def test_failover_refuses_retry_when_global_budget_is_low(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [
        ExecutionResult(
            exit_code=-1,
            infra_category="watchdog_timeout",
            operation_name="operations/low-budget",
            last_stage="QUEUED",
        ),
        ExecutionResult(exit_code=0),
    ]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(
        command=["bash", "-c", "true"],
        global_deadline_epoch=executor_module.time.time() + 100,
        min_retry_remaining_seconds=360,
    )

    assert result.exit_code == -1
    assert result.failover_attempts == 0
    assert _FakeExecutor.endpoints == ["grpc://10.0.0.1:50052"]
    assert len(_FakeExecutor.calls) == 1


def test_failover_retries_worker_io_status_with_scheduler_exit_code(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [
        ExecutionResult(
            exit_code=-178,
            infra_category="executor_worker_io",
            operation_name="operations/2",
            last_stage="COMPLETED",
        ),
        ExecutionResult(exit_code=0),
    ]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "true"])

    assert result.exit_code == 0
    assert result.failover_attempts == 1
    assert _FakeExecutor.endpoints == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert _FakeExecutor.cancelled == ["operations/2"]
    assert _FakeExecutor.calls[0].get("env_vars") is None
    assert _FakeExecutor.calls[1]["env_vars"] == {
        "RTP_REMOTE_EXECUTOR_FAILOVER_ATTEMPT": "1"
    }


def test_classifies_nativelink_worker_io_as_infra():
    category = RemoteExecutor._classify_execute_response_infra(
        exit_code=-178,
        status_code=6,
        status_message=(
            "File exists (os error 17) : Could not create directory "
            "nativelink/work/72a5e817/work/.. : --- : Job cancelled because "
            "it attempted to execute too many times 4 > 3 times"
        ),
        stdout_raw=b"",
        stderr_raw=b"",
    )

    assert category == "executor_worker_io"


def test_classifies_remote_setup_network_failure_as_infra():
    category = RemoteExecutor._classify_execute_response_infra(
        exit_code=1,
        status_code=0,
        status_message="",
        stdout_raw=b">>>PHASE:pip_install_failed 123 rc=1",
        stderr_raw=(
            b"[prepare_venv] uv pip install --compile-bytecode -e .[dev]\n"
            b"error: Request failed after 3 retries\n"
            b"Caused by: Failed to fetch: https://example/simple/pytest-asyncio/\n"
            b"Caused by: operation timed out\n"
        ),
    )

    assert category == "worker_setup_network"


def test_classifies_remote_gpu_xid_as_infra():
    category = RemoteExecutor._classify_execute_response_infra(
        exit_code=1,
        status_code=0,
        status_message="",
        stdout_raw=(
            b">>>RTP_GPU_INFRA_FAILURE\n"
            b"NVRM: Xid (PCI:0019:01:00): 31, pid=123, name=python\n"
            b"nvAssertFailedNoLog: Assertion failed\n"
        ),
        stderr_raw=b"",
    )

    assert category == "worker_gpu_xid"


def test_output_collector_emits_gpu_infra_marker():
    postscript = output_collector.make_tar_postscript()

    assert ">>>RTP_GPU_INFRA_FAILURE" in postscript
    assert "NVRM: Xid" in postscript
    assert "gpu_state_*.log" in postscript


def test_failover_retries_worker_gpu_xid(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [
        ExecutionResult(
            exit_code=1,
            infra_category="worker_gpu_xid",
            operation_name="operations/gpu-xid",
            last_stage="COMPLETED",
        ),
        ExecutionResult(exit_code=0),
    ]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "true"])

    assert result.exit_code == 0
    assert result.failover_attempts == 1
    assert _FakeExecutor.endpoints == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert _FakeExecutor.cancelled == ["operations/gpu-xid"]


def _operation_with_stage(stage_name):
    op = remote_execution_pb2.Operation(name="operations/stage-regression")
    meta = remote_execution_pb2.ExecuteOperationMetadata(
        stage=getattr(remote_execution_pb2.ExecutionStage, stage_name)
    )
    op.metadata.Pack(meta)
    return op


def test_execute_classifies_executing_to_queued_regression_as_infra(monkeypatch):
    executor = RemoteExecutor("grpc://scheduler.example.test:50052", _FakeCAS())

    class _Stub:
        def Execute(self, request, metadata, timeout):
            yield _operation_with_stage("QUEUED")
            yield _operation_with_stage("EXECUTING")
            yield _operation_with_stage("QUEUED")

    cancelled = []
    executor.stub = _Stub()
    monkeypatch.setattr(
        executor,
        "cancel_operation",
        lambda operation_name: cancelled.append(operation_name) or True,
    )

    result = executor.execute(
        command=["bash", "-c", "true"],
        input_root_digest=remote_execution_pb2.Digest(hash="root", size_bytes=1),
        timeout=7200,
    )

    assert result.exit_code == -1
    assert result.infra_category == "executor_stage_regressed"
    assert result.last_stage == "QUEUED"
    assert cancelled == ["operations/stage-regression"]


def test_default_queued_watchdog_is_shorter_than_aone_timeout(monkeypatch):
    monkeypatch.delenv("RTP_REMOTE_QUEUED_TIMEOUT_SECONDS", raising=False)

    assert RemoteExecutor._queued_watchdog_seconds(7200) == 300


def test_execute_classifies_queued_watchdog_as_infra(monkeypatch):
    executor = RemoteExecutor("grpc://scheduler.example.test:50052", _FakeCAS())

    class _CancelledRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.CANCELLED

        def details(self):
            return "cancelled by queued watchdog"

    class _ImmediateShortTimer:
        intervals = []

        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            self.intervals.append(interval)

        def start(self):
            if self.interval <= 1:
                self.callback()

        def cancel(self):
            self.cancelled = True

    class _QueuedCall:
        cancelled = False

        def __iter__(self):
            yield _operation_with_stage("QUEUED")
            raise _CancelledRpcError()

        def cancel(self):
            self.cancelled = True
            return True

    class _Stub:
        def __init__(self):
            self.call = _QueuedCall()

        def Execute(self, request, metadata, timeout):
            return self.call

    cancelled = []
    stub = _Stub()
    executor.stub = stub
    monkeypatch.setenv("RTP_REMOTE_QUEUED_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(executor_module.threading, "Timer", _ImmediateShortTimer)
    monkeypatch.setattr(
        executor,
        "cancel_operation",
        lambda operation_name: cancelled.append(operation_name) or True,
    )

    result = executor.execute(
        command=["bash", "-c", "true"],
        input_root_digest=remote_execution_pb2.Digest(hash="root", size_bytes=1),
        timeout=7200,
    )

    assert result.exit_code == -1
    assert result.infra_category == "watchdog_timeout"
    assert result.last_stage == "QUEUED"
    assert result.operation_name == "operations/stage-regression"
    assert b"reason=queued" in result.stderr_raw
    assert cancelled == ["operations/stage-regression"]
    assert stub.call.cancelled is True
    assert 1 in _ImmediateShortTimer.intervals


def test_failover_does_not_retry_test_failures(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [ExecutionResult(exit_code=1)]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "false"])

    assert result.exit_code == 1
    assert _FakeExecutor.endpoints == ["grpc://10.0.0.1:50052"]
    assert _FakeExecutor.cancelled == []


def test_failover_executor_does_not_close_concurrent_actions(monkeypatch):
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1"],
    )
    started = []
    all_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    class _BlockingExecutor:
        closed = []

        def __init__(self, endpoint, cas, metadata):
            self.grpc_uri = endpoint
            self.reapi_targets_combined = f"cas={cas.grpc_uri} | executor={endpoint}"

        def execute(self, **kwargs):
            with lock:
                started.append(self.grpc_uri)
                if len(started) == 2:
                    all_started.set()
            assert release.wait(timeout=5)
            return ExecutionResult(exit_code=0)

        def close(self):
            self.closed.append(self.grpc_uri)

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.example.test:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_BlockingExecutor,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, command=["bash", "-c", "true"])
        second = pool.submit(executor.execute, command=["bash", "-c", "true"])
        assert all_started.wait(timeout=5)
        assert _BlockingExecutor.closed == []
        release.set()
        assert first.result(timeout=5).exit_code == 0
        assert second.result(timeout=5).exit_code == 0

    assert _BlockingExecutor.closed == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.1:50052",
    ]
