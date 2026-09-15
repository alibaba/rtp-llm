import json
import os
import shlex
import subprocess
import sys
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


def test_collect_remote_files_includes_smoke_jpeg(tmp_path):
    rootdir = tmp_path / "repo"
    suite = rootdir / "rtp_llm" / "test" / "smoke" / "suites" / "test_smoke.py"
    image = rootdir / "rtp_llm" / "test" / "smoke" / "data" / "model" / "1.jpeg"
    suite.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    suite.write_text("def test_placeholder(): pass\n")
    image.write_bytes(b"jpeg fixture")

    files = remote_exec_rtp.collect_remote_files(
        rootdir.resolve(), [_FakeCollectedItem(suite.resolve(), smoke=True)]
    )

    assert "rtp_llm/test/smoke/data/model/1.jpeg" in files


def test_collect_remote_files_includes_tipc_jit_sources(tmp_path):
    rootdir = tmp_path / "repo"
    suite = rootdir / "rtp_llm" / "test" / "smoke" / "suites" / "test_smoke.py"
    source = rootdir / "rtp_llm" / "model_loader" / "tipc" / "csrc" / "ipc.cc"
    header = rootdir / "rtp_llm" / "model_loader" / "tipc" / "csrc" / "ipc.h"
    suite.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    suite.write_text("def test_placeholder(): pass\n")
    source.write_text("// CUDA IPC implementation\n")
    header.write_text("// CUDA IPC declarations\n")

    files = remote_exec_rtp.collect_remote_files(
        rootdir.resolve(), [_FakeCollectedItem(suite.resolve(), smoke=True)]
    )

    assert "rtp_llm/model_loader/tipc/csrc/ipc.cc" in files
    assert "rtp_llm/model_loader/tipc/csrc/ipc.h" in files


def test_collect_repo_runtime_files_includes_source_contracts(tmp_path):
    source_contracts = {
        "3rdparty/cub_compat.h",
        "arch_config/arch_select.bzl",
        "rtp_llm/cpp/cache/test/BUILD",
        "rtp_llm/cpp/cache/test/KVCacheManagerCPSlotMapperTest.cc",
        "rtp_llm/cpp/cache/test/SharedBlockCacheTest.cc",
        "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc",
        "rtp_llm/cpp/disaggregate/cache_store/test/BUILD",
        "rtp_llm/cpp/model_rpc/test/BUILD",
        "rtp_llm/cpp/models/logits_processor/test/BUILD",
        "rtp_llm/cpp/models/logits_processor/test/SpecLogitsVerifyRunnerPerfTest.cc",
        "rtp_llm/cpp/models/test/BUILD",
        "rtp_llm/cpp/models/test/PyWrappedModelCacheStoreIntegrationTest.cc",
        "rtp_llm/cpp/normal_engine/speculative/test/BUILD",
        "rtp_llm/cpp/normal_engine/speculative/test/MtpBatchStreamProcessorTest.cc",
        "rtp_llm/cpp/pybind/BUILD",
        "rtp_llm/cpp/telemetry/TraceAttributes.h",
        "rtp_llm/cpp/utils/test/BUILD",
        "rtp_llm/dash_sc/BUILD",
        "rtp_llm/dash_sc/test/BUILD",
        "rtp_llm/models_py/bindings/cuda/ops/tests/BUILD",
        "rtp_llm/models_py/bindings/cuda/ops/tests/CudaSamplerTest.cc",
        "rtp_llm/models_py/modules/dsv4/decode/test/BUILD",
        "rtp_llm/utils/test/BUILD",
    }
    assert set(remote_exec_rtp._SOURCE_CONTRACT_FILES) == source_contracts

    for relative_path in source_contracts:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source contract\n", encoding="utf-8")

    files = remote_exec_rtp._collect_repo_runtime_files(
        tmp_path, include_libs=False
    )

    assert source_contracts.issubset(files)


def test_collect_repo_runtime_files_includes_nested_test_data(tmp_path):
    fixture = (
        tmp_path
        / "rtp_llm"
        / "dash_sc"
        / "test"
        / "data"
        / "mrcr_deepseek_v4_think_leak_cases.json"
    )
    fixture.parent.mkdir(parents=True)
    fixture.write_text("[]\n", encoding="utf-8")

    files = remote_exec_rtp._collect_repo_runtime_files(
        tmp_path, include_libs=False
    )

    assert str(fixture.relative_to(tmp_path)) in files


def test_collect_repo_runtime_files_includes_cutlass_groupgemm_configs(tmp_path):
    configs = [
        tmp_path
        / "rtp_llm"
        / "models_py"
        / "kernels"
        / "cuda"
        / "fp8_kernel"
        / "cutlass_groupgemm"
        / "H20_128_7168.json",
        tmp_path
        / "internal_source"
        / "rtp_llm"
        / "models_py"
        / "kernels"
        / "cuda"
        / "fp8_kernel"
        / "cutlass_groupgemm"
        / "L20X_128_7168.json",
    ]
    for config in configs:
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("{}\n", encoding="utf-8")

    files = remote_exec_rtp._collect_repo_runtime_files(
        tmp_path, include_libs=False
    )

    assert {str(config.relative_to(tmp_path)) for config in configs}.issubset(files)


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


def test_collect_remote_files_stages_ppu_runtime_libs(tmp_path, monkeypatch):
    rootdir = tmp_path / "repo"
    suite = rootdir / "test_ppu.py"
    sdk_cuda = tmp_path / "ppu_sdk" / "CUDA_SDK" / "lib64"
    sdk_cupti = tmp_path / "ppu_sdk" / "CUDA_SDK" / "extras" / "CUPTI" / "lib64"
    sdk_lib = tmp_path / "ppu_sdk" / "lib"
    sdk_cuda.mkdir(parents=True)
    sdk_cupti.mkdir(parents=True)
    sdk_lib.mkdir(parents=True)
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text("def test_placeholder(): pass\n")
    for name in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        (sdk_cuda / name).write_bytes(name.encode())
    (sdk_cupti / "libcupti.so.12").write_bytes(b"cupti")
    (sdk_lib / "libhgml.so").write_bytes(b"hgml")
    (sdk_lib / "libuki.so").write_bytes(b"uki")
    monkeypatch.setattr(
        remote_exec_rtp,
        "_ppu_runtime_search_dirs",
        lambda: [sdk_cuda, sdk_cupti, sdk_lib],
    )

    files = remote_exec_rtp.collect_remote_files(
        rootdir.resolve(), [_FakeRemoteItem(suite.resolve(), gpu_type="PPU-ZW810E")]
    )

    runtime_dir = ".pytest_cache/remote_inputs/ppu_runtime"
    expected = {
        f"{runtime_dir}/libcudart.so.12",
        f"{runtime_dir}/libcublas.so.12",
        f"{runtime_dir}/libcublasLt.so.12",
        f"{runtime_dir}/libcupti.so.12",
        f"{runtime_dir}/libhgml.so",
        f"{runtime_dir}/libuki.so",
    }
    assert expected.issubset(files)
    for rel in expected:
        assert (rootdir / rel).is_symlink()


def test_ppu_runtime_search_dirs_include_sdk_and_system_cupti(monkeypatch):
    monkeypatch.setenv("PPU_SDK", "/opt/ppu")
    monkeypatch.delenv("PPU_HOME", raising=False)
    monkeypatch.delenv("SDK_ROOT", raising=False)

    search_dirs = remote_exec_rtp._ppu_runtime_search_dirs()

    assert Path("/opt/ppu/CUDA_SDK/extras/CUPTI/lib64") in search_dirs
    assert Path("/opt/ppu/CUDA_SDK/targets/x86_64-linux/lib") in search_dirs
    assert Path("/usr/local/cuda/extras/CUPTI/lib64") in search_dirs


def test_collect_remote_files_ppu_requires_controller_runtime_libs(
    tmp_path, monkeypatch
):
    suite = tmp_path / "test_ppu.py"
    suite.write_text("def test_placeholder(): pass\n")
    monkeypatch.setattr(remote_exec_rtp, "_ppu_runtime_search_dirs", lambda: [])

    with pytest.raises(RuntimeError, match="CUDA runtime.*PPU device runtime"):
        remote_exec_rtp.collect_remote_files(
            tmp_path.resolve(),
            [_FakeRemoteItem(suite.resolve(), gpu_type="PPU-ZW810E")],
        )


def test_collect_remote_files_non_ppu_does_not_require_ppu_runtime_libs(
    tmp_path, monkeypatch
):
    suite = tmp_path / "test_cuda.py"
    suite.write_text("def test_placeholder(): pass\n")
    monkeypatch.setattr(remote_exec_rtp, "_ppu_runtime_search_dirs", lambda: [])

    files = remote_exec_rtp.collect_remote_files(
        tmp_path.resolve(), [_FakeRemoteItem(suite.resolve(), gpu_type="H20")]
    )

    assert str(suite.relative_to(tmp_path)) in files


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


def test_ci_profile_count_rejects_zero_without_exact_baseline():
    config = _FakeProfileConfig(remote_session=True)
    config._rtp_ci_minimum_count = 1
    config._rtp_ci_expected_count = None

    with pytest.raises(pytest.UsageError, match="must never pass as 0/0"):
        ci_profile_plugin.validate_ci_profile_count(
            config, 0, context="reported"
        )

    ci_profile_plugin.validate_ci_profile_count(config, 17, context="reported")


def test_remote_session_result_enforces_exact_count_and_skip_contract():
    config = _FakeProfileConfig(remote_session=True)
    config._rtp_ci_minimum_count = 1
    config._rtp_ci_expected_count = 21
    config._rtp_ci_forbid_skips = True

    assert (
        remote_plugin._validate_session_profile_result(
            config, "py_ut_ppu", tests=21, skipped=0
        )
        == ""
    )
    assert "expected 21" in remote_plugin._validate_session_profile_result(
        config, "py_ut_ppu", tests=20, skipped=0
    )
    assert "reported 1 skipped" in remote_plugin._validate_session_profile_result(
        config, "py_ut_ppu", tests=21, skipped=1
    )


def test_per_test_command_exports_marker_gpu_count(tmp_path, monkeypatch):
    monkeypatch.setenv("RTP_REMOTE_GPU_MEMORY_PREFLIGHT", "1")
    rootdir = tmp_path / "repo"
    test_file = rootdir / "test_remote.py"
    rootdir.mkdir()
    test_file.write_text("def test_case(): pass\n")
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.rootdir = rootdir
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr=""), _rtp_ci_forbid_skips=True
    )
    plugin._collect_outputs = False
    plugin.timeout_policy = select_remote_timeout_policy(
        "smoke_h20_internal", per_test=True
    )

    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[],
        env_vars={},
        platform_properties={"gpu": "H20", "gpu_count": "1"},
        remote_setup_prefix="echo REMOTE_SETUP; ",
    )
    command = plugin._build_command(
        _FakeRemoteItem(test_file, gpu_type="H20", gpu_count=1), runtime
    )
    shell = command[2]

    assert "export GPU_COUNT=1;" in shell
    assert "export WORLD_SIZE=1;" in shell
    assert "export GPU_COUNT_PER_WORKER=1;" in shell
    assert "export RTP_REMOTE_FORBID_SKIPS=1;" in shell
    assert "export RTP_REMOTE_HEARTBEAT_KEEPALIVE=1;" not in shell
    assert "test_remote.py::test_case" in shell
    assert "-k test_case" not in shell
    assert "--query-gpu=memory.used" in shell
    assert "need 1 (limit=1024 MiB)" in shell
    assert shell.index("--query-gpu=memory.used") < shell.index("echo REMOTE_SETUP")
    assert shell.index("export GPU_COUNT=1;") < shell.index(
        "python rtp_llm/test/utils/device_resource.py"
    )

    plugin.timeout_policy = select_remote_timeout_policy("perf_sm9x", per_test=True)
    perf_shell = plugin._build_command(
        _FakeRemoteItem(test_file, gpu_type="H20", gpu_count=1), runtime
    )[2]
    assert "export RTP_REMOTE_HEARTBEAT_KEEPALIVE=1;" in perf_shell


def test_session_command_locks_total_gpu_pool_and_slices_workers(monkeypatch):
    monkeypatch.setenv("RTP_REMOTE_GPU_MEMORY_PREFLIGHT", "1")
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
        remote_setup_prefix="echo REMOTE_SETUP; ",
    )
    command = plugin._build_session_command("", runtime, ci_profile=None)
    shell = command[2]

    assert "need 4 (limit=1024 MiB)" in shell
    assert shell.index("--query-gpu=memory.used") < shell.index("echo REMOTE_SETUP")
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


def test_session_command_forwards_profile_ignore_paths(monkeypatch):
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_pytest_ci_section",
        lambda root: {"default_pytest_cli": ""},
    )
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_profile",
        lambda root, name: {
            "markexpr": "not SM100_ARM",
            "ignore_paths": [
                "rtp_llm/test/dsv4",
                "rtp_llm/models_py/modules/dsv4",
            ],
        },
    )
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.workers = 1
    plugin._collect_outputs = False
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr="not SM100_ARM", keyword=""),
        rootpath=Path("."),
    )
    plugin.timeout_policy = select_remote_timeout_policy("py_ut_sm8x", per_test=False)
    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[],
        env_vars={},
        platform_properties={"gpu": "A10", "gpu_count": "1"},
        remote_setup_prefix="",
    )

    shell = plugin._build_session_command("", runtime, ci_profile="py_ut_sm8x")[2]

    assert "--ignore=rtp_llm/test/dsv4" in shell
    assert "--ignore=rtp_llm/models_py/modules/dsv4" in shell


def test_amd_session_forwards_routing_profile_and_all_gpu_tiers():
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.workers = 8
    plugin._collect_outputs = False
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr="MI308X", keyword=""),
        rootpath=Path(__file__).resolve().parents[3],
    )
    plugin.timeout_policy = select_remote_timeout_policy("py_ut_amd", per_test=False)
    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[], env_vars={},
        platform_properties={"gpu": "MI308X", "gpu_count": "8"},
        remote_setup_prefix="",
    )
    shell = plugin._build_session_command("", runtime, ci_profile="py_ut_amd")[2]
    assert "export RTP_PYTEST_CI_PROFILE=py_ut_amd" in shell
    for tier in (1, 2, 4, 8):
        assert f"export GPU_COUNT_PER_WORKER={tier};" in shell
    assert "test_rocm_beam_search_op.py" in shell
    assert "test_gdn_decode.py" in shell


def test_amd_session_rejects_missing_targets_despite_matching_total():
    config = _FakeProfileConfig(remote_session=True)
    config._rtp_ci_minimum_count = 1
    config._rtp_ci_expected_count = 286
    config._rtp_ci_forbid_skips = True
    error = remote_plugin._validate_session_profile_result(
        config, "py_ut_amd", tests=286, skipped=0,
        nodeids=[f"other.py::test_other[{i}]" for i in range(286)],
    )
    assert "AMD baseline coverage missing" in error
    assert "test_inline_fp8_quant" in error
    assert "RocmBeamSearchOpTest.simpleTest" in error


def test_session_command_runs_profile_isolated_paths_in_fresh_processes(monkeypatch):
    isolated_path = (
        "rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/"
        "test/test_allgather_cp_impl.py"
    )
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_pytest_ci_section",
        lambda root: {"default_pytest_cli": ""},
    )
    monkeypatch.setattr(
        ci_profile_plugin,
        "_get_profile",
        lambda root, name: {
            "markexpr": "H20 and not manual",
            "isolated_paths": [isolated_path],
        },
    )
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.workers = 4
    plugin._collect_outputs = False
    plugin.config = SimpleNamespace(
        option=SimpleNamespace(markexpr="H20 and not manual", keyword=""),
        rootpath=Path("."),
    )
    plugin.timeout_policy = select_remote_timeout_policy("py_ut_sm9x", per_test=False)
    runtime = remote_exec_rtp.RemoteRuntimeConfig(
        ignore_args=[],
        env_vars={},
        platform_properties={"gpu": "H20", "gpu_count": "4"},
        remote_setup_prefix="",
    )

    shell = plugin._build_session_command("", runtime, ci_profile="py_ut_sm9x")[2]
    isolated_start = shell.index(f"--- Isolated file: {isolated_path} ---")
    parallel_start = shell.index("--- Phase: 1-GPU tests")
    isolated_command = shell[isolated_start:parallel_start]
    parallel_command = shell[parallel_start:]

    assert "export GPU_COUNT=1" in isolated_command
    assert "-n 0" in isolated_command
    assert isolated_path in isolated_command
    assert "--junitxml=bazel-testlogs/pytest/test_isolated_0.xml" in isolated_command
    assert f"--ignore={isolated_path}" in parallel_command
    assert "--max-worker-restart=0" in parallel_command
    assert "test_isolated_*.xml" in shell
    cleanup = shell.index("rm -f bazel-testlogs/pytest/test.xml")
    assert cleanup < isolated_start

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
    per_test_eval = select_remote_timeout_policy(
        "smoke_sm100_eval_oss", per_test=True
    )
    per_test_perf = select_remote_timeout_policy("perf-sm9x", per_test=True)

    assert (
        ut.session_budget_seconds,
        ut.action_timeout_seconds,
        ut.supervisor_timeout_seconds,
        ut.pytest_timeout_seconds,
        ut.min_retry_remaining_seconds,
    ) == (3600, 3000, 2880, 300, 900)
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
        per_test_eval.profile_class,
        per_test_eval.session_budget_seconds,
        per_test_eval.action_timeout_seconds,
        per_test_eval.supervisor_timeout_seconds,
        per_test_eval.pytest_timeout_seconds,
        per_test_eval.heartbeat_stall_seconds,
    ) == ("per_test_eval", 7500, 7200, 7080, 6000, 6600)
    assert (
        per_test_perf.session_budget_seconds,
        per_test_perf.action_timeout_seconds,
        per_test_perf.supervisor_timeout_seconds,
        per_test_perf.pytest_timeout_seconds,
        per_test_perf.min_retry_remaining_seconds,
    ) == (6600, 6300, 6120, 1800, 1200)


def test_per_test_deadline_starts_after_local_queue_wait(monkeypatch):
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.timeout_policy = select_remote_timeout_policy(
        "smoke_sm100_oss", per_test=True
    )
    captured = {}
    now = [100.0]
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def _block_worker():
        blocker_started.set()
        assert release_blocker.wait(timeout=5)

    def _capture_execute(**kwargs):
        captured.update(kwargs)
        return ExecutionResult(exit_code=0)

    monkeypatch.setattr(remote_plugin.time, "time", lambda: now[0])
    plugin._execute_with_retry = _capture_execute

    with ThreadPoolExecutor(max_workers=1) as pool:
        blocker = pool.submit(_block_worker)
        assert blocker_started.wait(timeout=5)
        queued = pool.submit(plugin._execute_queued_with_retry, command=["true"])
        now[0] = 250.0
        release_blocker.set()
        blocker.result(timeout=5)
        assert queued.result(timeout=5).exit_code == 0

    assert captured["global_deadline_epoch"] == (
        250.0 + plugin.timeout_policy.session_budget_seconds
    )


def test_per_test_controller_wait_does_not_consume_worker_case_timeout():
    plugin = object.__new__(remote_plugin.RemoteREAPIPlugin)
    plugin.timeout_policy = select_remote_timeout_policy(
        "smoke_sm100_oss", per_test=True
    )
    source_timeout = pytest.mark.timeout(600).mark

    class FakeItem:
        own_markers = [source_timeout]

        def add_marker(self, marker, append=True):
            if append:
                self.own_markers.append(marker.mark)
            else:
                self.own_markers.insert(0, marker.mark)

    item = FakeItem()
    plugin._apply_controller_timeout_marker(item)

    assert item.own_markers[0].name == "timeout"
    assert item.own_markers[0].args == (3600,)
    assert item.own_markers[1].args == (600,)


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


def test_remote_setup_and_pytest_keep_heartbeat_alive_during_long_work(tmp_path):
    command = remote_exec_rtp.build_remote_setup_command(Path("."))

    assert "prepare_venv.py >logs/prepare_venv.out" in command
    assert "PV_PID=$!" in command
    assert "pip_install_active" in command
    assert 'wait "$PV_PID"; PV_RC=$?' in command
    assert 'OUT=$(cat logs/prepare_venv.out)' in command
    assert subprocess.run(
        ["bash", "-n"], input=command, text=True, capture_output=True, check=False
    ).returncode == 0

    helper_rel = "internal_source/ci/prepare_rocm_deps.sh"
    helper = tmp_path / helper_rel
    helper.parent.mkdir(parents=True)
    helper.write_text("return 37\n")
    assert helper_rel in remote_exec_rtp._collect_base_files(tmp_path)
    native_setup = command.split("mkdir -p logs; ", 1)[1].split(
        "if [ -f internal_source/ci/prepare_venv.py ]; then ", 1
    )[0]
    failed = subprocess.run(
        ["bash", "-c", native_setup + "echo installer_started"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 37
    assert "installer_started" not in failed.stdout
    helper.write_text("export CMAKE_PREFIX_PATH=/native/grpc\n")
    ready = subprocess.run(
        ["bash", "-c", native_setup + 'echo "$CMAKE_PREFIX_PATH"'],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert ready.stdout.strip() == "/native/grpc"

    heartbeat_plugin = remote_plugin._heartbeat_plugin_shell()
    assert (
        "threading.Thread(target=_heartbeat_loop, daemon=True).start()"
        in heartbeat_plugin
    )
    assert "_heartbeat_stop.wait(60)" in heartbeat_plugin
    assert "_touch('pytest_active')" in heartbeat_plugin
    assert "RTP_REMOTE_HEARTBEAT_KEEPALIVE" in heartbeat_plugin
    assert "_heartbeat_stop.set()" in heartbeat_plugin


def test_remote_setup_exports_profile_env():
    command = remote_exec_rtp.build_remote_setup_command(
        Path("."), setup_env={"RTP_BAZEL_CONFIG": "--config=custom"}
    )

    assert "export RTP_BAZEL_CONFIG=--config=custom;" in command
    assert (
        "RTP_BAZEL_CONFIG=--config=custom /opt/conda310/bin/python "
        "internal_source/ci/prepare_venv.py"
    ) in command


@pytest.mark.parametrize("profile_library_path", [None, "/profile/rocm/lib"])
def test_dependency_install_does_not_inherit_runtime_libraries(profile_library_path):
    setup_env = {
        "RTP_BAZEL_CONFIG": "--config=rocm",
        "RTP_LLM_REQUIRE_ACCELERATOR": "1",
    }
    if profile_library_path:
        setup_env["LD_LIBRARY_PATH"] = profile_library_path
    command = remote_exec_rtp.build_remote_setup_command(
        Path("."), setup_env=setup_env
    )
    invocation = command.split(
        "if [ -f internal_source/ci/prepare_venv.py ]; then ", 1
    )[1].split(">logs/prepare_venv.out", 1)[0]
    args = shlex.split(invocation)
    python_index = args.index("/opt/conda310/bin/python")
    # Run an environment probe in place of the installer, with the exact
    # generated env/assignment arguments used by the remote worker.
    args[python_index:] = [
        sys.executable,
        "-c",
        "import json, os; print(json.dumps({k: os.getenv(k) for k in "
        "['LD_LIBRARY_PATH', 'RTP_BAZEL_CONFIG', 'RTP_LLM_REQUIRE_ACCELERATOR']}))",
    ]
    runtime_env = dict(os.environ, LD_LIBRARY_PATH="/opt/conda310/lib:/opt/rocm/lib")
    result = subprocess.run(
        args, env=runtime_env, text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == {
        "LD_LIBRARY_PATH": None,
        "RTP_BAZEL_CONFIG": "--config=rocm",
        "RTP_LLM_REQUIRE_ACCELERATOR": "1",
    }
    assert runtime_env["LD_LIBRARY_PATH"] == "/opt/conda310/lib:/opt/rocm/lib"
    assert "export LD_LIBRARY_PATH=" in command
    if profile_library_path:
        assert f"export LD_LIBRARY_PATH={profile_library_path};" in command


def test_sm100_markexpr_prefers_explicit_arm_pool():
    gpu_type = remote_exec_rtp.infer_gpu_type_from_markexpr(
        "manual and smoke and (SM100 or SM100_ARM)"
    )
    runtime = remote_exec_rtp.build_runtime_config(
        Path("."),
        remote_exec_rtp.GPURequest(gpu_type=gpu_type, gpu_count=4),
    )

    assert gpu_type == "SM100_ARM"
    assert runtime.platform_properties["gpu"] == "SM100_ARM"


def test_sm100_marker_keeps_distinct_reapi_pool():
    runtime = remote_exec_rtp.build_runtime_config(
        Path("."),
        remote_exec_rtp.GPURequest(gpu_type="SM100", gpu_count=4),
    )

    assert runtime.platform_properties["gpu"] == "SM100"


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


def test_frontend_ci_profile_declares_a10_gpu_type():
    repo_root = Path(__file__).resolve().parents[3]

    assert (
        remote_exec_rtp.resolve_ci_profile_gpu_type(repo_root, "py_ut_frontend")
        == "A10"
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


def test_load_pyproject_merges_sibling_internal_overlay(tmp_path):
    repo_root = tmp_path / "github-opensource"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        """
[tool.rtp_llm.pytest_ci.profiles.smoke_sm100_oss]
gpu_type = "SM100_ARM"
""".strip(),
        encoding="utf-8",
    )
    internal_root = tmp_path / "internal_source"
    internal_root.mkdir()
    (internal_root / "pyproject_internal.toml").write_text(
        """
[tool.rtp-llm.pytest_ci.gpu_env.SM100_ARM]
FT_SERVER_TEST = "1"
PATH = "/usr/bin:/bin"
""".strip(),
        encoding="utf-8",
    )

    remote_exec_rtp._load_pyproject.cache_clear()
    try:
        config = remote_exec_rtp._load_pyproject(repo_root)
    finally:
        remote_exec_rtp._load_pyproject.cache_clear()

    pytest_ci = config["tool"]["rtp_llm"]["pytest_ci"]
    assert pytest_ci["profiles"]["smoke_sm100_oss"]["gpu_type"] == "SM100_ARM"
    assert pytest_ci["gpu_env"]["SM100_ARM"] == {
        "FT_SERVER_TEST": "1",
        "PATH": "/usr/bin:/bin",
    }


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
                                    "PROFILE_ONLY": "profile-value",
                                },
                            }
                        },
                        "gpu_env": {
                            "CUSTOM_GPU": {
                                "RTP_BAZEL_CONFIG": "--config=custom",
                                "SHARED_VALUE": "shared-value",
                            }
                        },
                    }
                }
            }
        },
    )

    runtime = remote_exec_rtp.build_runtime_config(
        Path("."), remote_exec_rtp.GPURequest(gpu_type="CUSTOM_GPU", gpu_count=4)
    )

    assert runtime.env_vars["RTP_BAZEL_CONFIG"] == "--config=custom"
    assert runtime.env_vars["PROFILE_ONLY"] == "profile-value"
    assert runtime.env_vars["SHARED_VALUE"] == "shared-value"
    assert "export RTP_BAZEL_CONFIG=--config=custom;" in runtime.remote_setup_prefix


def test_collect_session_files_packs_runtime_lib_archive(tmp_path):
    (tmp_path / "rtp_llm" / "libs").mkdir(parents=True)
    (tmp_path / "rtp_llm" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "rtp_llm" / "sample.py").write_text("x = 1\n", encoding="utf-8")
    solution_data = (
        tmp_path
        / "rtp_llm/models_py/modules/factory/linear/impl/rocm/data/solutions.json"
    )
    solution_data.parent.mkdir(parents=True)
    solution_data.write_text("{}\n", encoding="utf-8")
    stub = tmp_path / "rtp_llm/ops/librtp_compute_ops/__init__.pyi"
    stub.parent.mkdir(parents=True)
    stub.write_text("class CacheConfig: ...\n", encoding="utf-8")
    cuda_graph_source = tmp_path / "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc"
    cuda_graph_source.parent.mkdir(parents=True)
    cuda_graph_source.write_text("// source contract\n", encoding="utf-8")
    models_test_build = tmp_path / "rtp_llm/cpp/models/test/BUILD"
    models_test_build.parent.mkdir(parents=True)
    models_test_build.write_text("# model test targets\n", encoding="utf-8")
    pybind_build = tmp_path / "rtp_llm/cpp/pybind/BUILD"
    pybind_build.parent.mkdir(parents=True)
    pybind_build.write_text("# pybind targets\n", encoding="utf-8")
    cub_compat_header = tmp_path / "3rdparty/cub_compat.h"
    cub_compat_header.parent.mkdir(parents=True)
    cub_compat_header.write_text("// CUB compatibility contract\n", encoding="utf-8")
    for name in (
        "libth_transformer_config.so",
        "libth_grammar_tokenizer_info.so",
        "libth_transformer.so",
        "librtp_compute_ops.so",
        "libdependency.so.1",
    ):
        p = tmp_path / "rtp_llm" / "libs" / name
        p.write_bytes(f"{name}\n".encode())
        p.chmod(0o755)

    binary = tmp_path / "rtp_llm/libs/test/rocm_beam_search_op_test"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native test binary")
    binary.chmod(0o755)
    files = remote_exec_rtp.collect_session_files(tmp_path)

    archive_rel = ".pytest_cache/remote_inputs/rtp_llm_libs.tar"
    assert archive_rel in files
    assert "rtp_llm/sample.py" in files
    assert "rtp_llm/ops/librtp_compute_ops/__init__.pyi" in files
    assert "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc" in files
    assert "rtp_llm/cpp/models/test/BUILD" in files
    assert "rtp_llm/cpp/pybind/BUILD" in files
    assert "3rdparty/cub_compat.h" in files
    assert (
        "rtp_llm/models_py/modules/factory/linear/impl/rocm/data/solutions.json"
        in files
    )
    assert "rtp_llm/libs/libth_transformer_config.so" not in files
    with tarfile.open(tmp_path / archive_rel, "r") as tar:
        names = set(tar.getnames())
        assert tar.getmember("rtp_llm/libs/test/rocm_beam_search_op_test").mode & 0o111
    assert "rtp_llm/libs/libth_transformer_config.so" in names
    assert "rtp_llm/libs/libth_grammar_tokenizer_info.so" in names
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


def test_execute_can_watch_requeued_action_until_completion(monkeypatch):
    executor = RemoteExecutor("grpc://scheduler.example.test:50052", _FakeCAS())

    class _Stub:
        def Execute(self, request, metadata, timeout):
            yield _operation_with_stage("QUEUED")
            yield _operation_with_stage("EXECUTING")
            yield _operation_with_stage("QUEUED")
            yield _operation_with_stage("EXECUTING")
            yield _done_operation()

    cancelled = []
    executor.stub = _Stub()
    monkeypatch.setenv("RTP_REMOTE_ALLOW_EXECUTING_REQUEUE", "1")
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

    assert result.exit_code == 0
    assert result.last_stage == "COMPLETED"
    assert cancelled == []


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
