"""CPU lifecycle regressions. No inference service or remote execution is started."""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).absolute().parents[3]


def load_definitions(path, names, namespace):
    """Exercise complete infrastructure functions without importing Native ops."""
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def test_perf_environment_restored_when_preparation_fails(monkeypatch, tmp_path):
    from rtp_llm.test.perf_test.perf_runner import run_perf_test

    monkeypatch.setenv("REPAIR_ENV", "before")
    monkeypatch.setenv("PERF_TEST_NAME", "before-name")
    argv = sys.argv
    with pytest.raises(AssertionError, match="not found"):
        run_perf_test(
            "after",
            {"envs": {"REPAIR_ENV": "after"}, "argv": [], "baseline": "missing.json"},
            tmp_path,
        )
    assert os.environ["REPAIR_ENV"] == "before"
    assert os.environ["PERF_TEST_NAME"] == "before-name"
    assert sys.argv is argv


@pytest.mark.parametrize("stage", ["optional", "task", "constructor"])
def test_smoke_environment_restored_through_all_preparation(
    monkeypatch, tmp_path, stage
):
    import json

    from rtp_llm.test.smoke import common_def
    from rtp_llm.test.smoke_framework import runner

    monkeypatch.setenv("REPAIR_ENV", "before")
    monkeypatch.delenv("OPTIONAL_REPAIR_ENV", raising=False)
    old_gpu = os.environ.get("GPU_COUNT")

    def optional(config):
        os.environ["OPTIONAL_REPAIR_ENV"] = "changed"
        if stage == "optional":
            raise RuntimeError("preparation")

    monkeypatch.setattr(runner, "_configure_optional_internal_env", optional)
    monkeypatch.setattr(common_def, "REL_PATH", str(tmp_path))
    if stage == "constructor":
        (tmp_path / "task.json").write_text(
            json.dumps(dict(model_type="unit", model_path="unused", query_result=[]))
        )

    def constructor(**kw):
        raise RuntimeError("preparation")

    monkeypatch.setattr(runner, "get_runner_type", lambda *a: constructor)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.run_smoke_test(
            "test",
            {"task_info": "task.json", "gpu_type": "H20", "envs": ["REPAIR_ENV=after"]},
        )
    assert os.environ["REPAIR_ENV"] == "before"
    assert "OPTIONAL_REPAIR_ENV" not in os.environ
    assert os.environ.get("GPU_COUNT") == old_gpu


def test_unittest_repetition_resolves_original_method_and_fresh_instances(pytester):
    pytester.makepyfile(
        """
import unittest
seen = []
class Repeated(unittest.TestCase):
    def setUp(self):
        assert all(self is not previous for previous in seen)
        seen.append(self)
    def test_first(self):
        assert self._testMethodName == "test_first"
    def test_second(self):
        assert self._testMethodName == "test_second"
"""
    )
    result = pytester.runpytest(
        "-p", "rtp_llm.test.smoke_framework.runs_plugin", "--runs-per-test=3", "-v"
    )
    result.assert_outcomes(passed=6)
    assert "test_first[run02/3] PASSED" in result.stdout.str()
    assert "test_second[run03/3] PASSED" in result.stdout.str()


def gpu_hooks():
    return load_definitions(
        ROOT / "conftest.py",
        [
            "_xdist_gpu_slot",
            "pytest_configure_node",
            "_release_gpu_slot",
            "pytest_testnodedown",
        ],
        {"pytest": pytest},
    )


def test_controller_quarantines_slot_until_process_exit():
    hooks = gpu_hooks()
    config = SimpleNamespace()

    def node(name):
        gateway = Mock(id=name)
        gateway.spec.popen = True
        channel = gateway.remote_exec.return_value
        channel.receive.side_effect = lambda **kw: channel.send.call_args.args[0]
        return SimpleNamespace(
            config=config,
            gateway=gateway,
            workerinput={"workercount": 2},
            workeroutput={},
        )

    one, two, replacement = node("gw0"), node("gw1"), node("gw2")
    hooks["pytest_configure_node"](one)
    hooks["pytest_configure_node"](two)
    two.gateway._io.popen.wait.side_effect = subprocess.TimeoutExpired("worker", 10)
    hooks["pytest_testnodedown"](two, None)
    with pytest.raises(pytest.UsageError, match="No released"):
        hooks["pytest_configure_node"](replacement)
    two.gateway._io.popen.wait.side_effect = None
    hooks["pytest_testnodedown"](two, None)
    hooks["pytest_configure_node"](replacement)
    assert replacement.workerinput["rtp_gpu_slot"] == 1
    assert config._rtp_gpu_slot_allocations == {"gw0": 0, "gw2": 1}


def test_controller_bootstraps_actual_owned_local_gateway():
    import execnet

    hooks = gpu_hooks()
    group = execnet.Group()
    gateway = group.makegateway(f"popen//python={sys.executable}")
    config = SimpleNamespace()
    node = SimpleNamespace(
        config=config, gateway=gateway, workerinput={"workercount": 1}
    )
    try:
        hooks["pytest_configure_node"](node)
        channel = gateway.remote_exec(
            "import os; channel.send(os.environ['_RTP_XDIST_ASSIGNED_SLOT'])"
        )
        assert channel.receive(timeout=10) == "0"
        channel.waitclose(timeout=10)
        hooks["_release_gpu_slot"](node)
        assert gateway._io.popen.poll() is not None
        assert config._rtp_gpu_slot_allocations == {}
    finally:
        group.terminate(timeout=10)


def test_plain_package_import_does_not_load_native_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rtp_llm; assert not any(n in sys.modules for n in ('torch', 'triton', 'rtp_llm.ops', 'librtp_compute_ops'))",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("layout", ["native", "runfiles", "manifest"])
def test_extension_layouts(tmp_path, monkeypatch, layout):
    import rtp_llm
    from rtp_llm.test.utils.extension_loader import load_test_extension

    source = tmp_path / "source/rtp_llm"
    source.mkdir(parents=True)
    monkeypatch.setattr(rtp_llm, "__path__", [str(source)])
    native = "rtp_llm/libs/test/fixture.py"
    bazel = "rtp_llm/cpp/test/fixture.py"
    if layout == "native":
        path = source.parent / native
    elif layout == "runfiles":
        path = tmp_path / "runfiles/workspace" / bazel
        monkeypatch.setenv("RUNFILES_DIR", str(tmp_path / "runfiles"))
        monkeypatch.setenv("TEST_WORKSPACE", "workspace")
    else:
        path = tmp_path / "real/fixture.py"
        manifest = tmp_path / "MANIFEST"
        manifest.write_text(f"workspace/{bazel} {path}\n")
        monkeypatch.setenv("RUNFILES_MANIFEST_FILE", str(manifest))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("value = 17\n")
    assert load_test_extension("fixture", native, bazel).value == 17


def test_extension_existing_load_error_is_not_replaced_by_fallback(
    tmp_path, monkeypatch
):
    import rtp_llm
    from rtp_llm.test.utils.extension_loader import load_test_extension

    package = tmp_path / "rtp_llm"
    package.mkdir()
    monkeypatch.setattr(rtp_llm, "__path__", [str(package)])
    (package / "native.py").write_text('raise ImportError("dependency ABI failure")')
    (package / "fallback.py").write_text("value = 17")
    with pytest.raises(ImportError, match="dependency ABI failure"):
        load_test_extension("fixture", "rtp_llm/native.py", "rtp_llm/fallback.py")


@pytest.mark.parametrize("late", [False, True])
def test_partial_parallel_submission_reclaims_owned_and_late_services(late):
    import concurrent.futures
    import logging
    import threading
    from typing import Any, Dict, List, Tuple

    tree = ast.parse((ROOT / "rtp_llm/test/smoke/case_runner.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CaseRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "start_servers_parallel"
    )
    started = threading.Event()
    owned = Mock()
    observed = []

    class Pool:
        def __enter__(self):
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.count = 0
            return self

        def submit(self, fn, config):
            self.count += 1
            if self.count == 2:
                assert started.wait(timeout=2)
                raise RuntimeError("second submission failed")
            return self.pool.submit(fn, config)

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    def start_server(*args, server_manager_callback, cancel_event, **kw):
        observed.append(cancel_event)
        if not late:
            server_manager_callback(owned)
        started.set()
        assert cancel_event.wait(timeout=2)
        if late:
            server_manager_callback(owned)
        args[1].ret = False
        return owned

    futures = SimpleNamespace(
        ThreadPoolExecutor=lambda **kw: Pool(),
        as_completed=concurrent.futures.as_completed,
        CancelledError=concurrent.futures.CancelledError,
    )
    namespace = dict(
        threading=threading,
        logging=logging,
        concurrent=SimpleNamespace(futures=futures),
        TaskStates=lambda: SimpleNamespace(ret=True, err_msg=""),
        MagaServerManager=object,
        Any=Any,
        Dict=Dict,
        List=List,
        Tuple=Tuple,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "case_runner.py", "exec"),
        namespace,
    )
    configs = [
        dict(env_dict={}, task_info=None, port=1, role_name=name)
        for name in ("first", "second")
    ]
    with pytest.raises(RuntimeError, match="second submission"):
        namespace["start_servers_parallel"](
            SimpleNamespace(start_server=start_server), configs
        )
    assert observed[0].is_set()
    assert owned.stop_server.call_count >= 1


def test_nvidia_preload_honors_disabled_user_site_and_tries_environment_alternatives():
    import ctypes

    calls = []
    site = SimpleNamespace(
        ENABLE_USER_SITE=False,
        getsitepackages=lambda: ["/environment"],
        getusersitepackages=lambda: pytest.fail("disabled user-site queried"),
    )

    def load(path, **kw):
        calls.append(path)
        if path.endswith("first.so"):
            raise OSError("incompatible dependency")

    namespace = dict(
        site=site,
        os=os,
        ctypes=SimpleNamespace(CDLL=load, RTLD_GLOBAL=ctypes.RTLD_GLOBAL),
        glob=SimpleNamespace(
            glob=lambda pattern: ["/environment/first.so", "/environment/second.so"]
        ),
    )
    load_definitions(
        ROOT / "rtp_llm/ops/__init__.py", ["_preload_nvidia_deps"], namespace
    )
    namespace["_preload_nvidia_deps"]()
    assert calls == ["/environment/first.so", "/environment/second.so"] * 6


def rocm_layout_functions():
    from typing import Optional

    enum = SimpleNamespace(BASE=0, FP8=1)
    ns = dict(
        Optional=Optional,
        AttentionConfigs=object,
        FMHAConfig=object,
        KvCacheDataType=enum,
        ASM_DECODE_HEAD_SIZES={128},
    )
    load_definitions(
        ROOT / "rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py",
        [
            "_kv_vector_width",
            "_validate_v_geometry",
            "prefill_writes_vectorized_v",
            "implementation_v_layouts",
            "select_compatible_v_layout",
        ],
        ns,
    )
    return ns


@pytest.mark.parametrize("head", [128, 256])
@pytest.mark.parametrize("fp8", [False, True])
def test_rocm_layout_pairing_uses_final_candidates(head, fp8):
    ns = rocm_layout_functions()
    config = SimpleNamespace(
        size_per_head=head,
        kernel_tokens_per_block=32,
        dtype=SimpleNamespace(itemsize=2),
        kv_cache_dtype=int(fp8),
    )
    flags = SimpleNamespace(use_asm_pa=True)
    impl = lambda name: SimpleNamespace(NAME=name)
    choose = ns["select_compatible_v_layout"]
    if fp8:
        with pytest.raises(ValueError, match="layout mismatch"):
            choose(config, [impl("aiter_asm"), impl("aiter")], [impl("aiter")], flags)
    else:
        assert (
            choose(config, [impl("aiter_asm"), impl("aiter")], [impl("aiter")], flags)
            is False
        )
    assert choose(config, [impl("aiter_asm")], [impl("aiter_triton")], flags) is True
    with pytest.raises(ValueError, match="layout mismatch"):
        choose(config, [impl("aiter_asm")], [impl("aiter")], flags)


@pytest.mark.parametrize("page,vectorized", [(16, True), (8, False)])
def test_rocm_paged_writer_and_reader_use_negotiated_layout(page, vectorized):
    import __future__

    path = ROOT / "rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "AiterPrefillImplPaged"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    asm, nonasm, triton = Mock(), Mock(), Mock()
    ns = dict(
        AiterPrefillAttnOpPaged=Mock(),
        AiterPrefillAttnOpTriton=triton,
        FusedRopeKVCachePrefillOpAsm=asm,
        FusedRopeKVCachePrefillOpNonAsm=nonasm,
        common=SimpleNamespace(create_write_cache_store_impl=Mock()),
        _kv_vector_width=lambda config: 8,
    )
    exec(
        compile(
            ast.Module(body=[init], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    config = SimpleNamespace(
        need_rope_kv_cache=True,
        head_num=8,
        kv_head_num=1,
        size_per_head=128,
        kernel_tokens_per_block=page,
    )
    obj = SimpleNamespace(
        use_unified_attention=False,
        _select_backend=lambda _: "triton",
        _prepare_backend=lambda _: None,
    )
    ns["__init__"](
        obj,
        config,
        SimpleNamespace(is_cuda_graph=False),
        fmha_config=SimpleNamespace(use_asm_pa=False),
        cache_vectorized_v=vectorized,
    )
    assert obj.linear_v is (not vectorized)
    (asm if vectorized else nonasm).assert_called_once_with(config)
    (nonasm if vectorized else asm).assert_not_called()
    assert triton.call_args.kwargs["linear_v"] is (not vectorized)
    config.kernel_tokens_per_block = 16
    with pytest.raises(ValueError, match="requires vectorized"):
        ns["__init__"](
            obj, config, SimpleNamespace(is_cuda_graph=False), cache_vectorized_v=False
        )


def test_embedding_selects_vit_backend_for_each_request():
    import __future__

    import asyncio
    from unittest.mock import AsyncMock

    path = ROOT / "rtp_llm/embedding/embedding_endpoint.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EmbeddingEndpoint"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_embeddings_grpc"
    )
    stub = SimpleNamespace(
        embedding=AsyncMock(
            return_value=SimpleNamespace(output_is_tensor=False, output_map=[])
        )
    )
    ns = dict(
        pb2_grpc=SimpleNamespace(EmbeddingRpcServiceStub=lambda _: stub),
        pb2=SimpleNamespace(
            MMPreprocessConfigPB=SimpleNamespace,
            MultimodalInputPB=SimpleNamespace,
            EmbeddingInputPB=SimpleNamespace,
        ),
        RoleType=SimpleNamespace(VIT="vit"),
        trans_from_tensor=lambda x: x,
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    host = Mock()
    host.get_backend_role_addrs.side_effect = [
        [SimpleNamespace(ip="vit-one", grpc_port=1)],
        [SimpleNamespace(ip="vit-two", grpc_port=2)],
    ]
    endpoint = SimpleNamespace(
        _channel_pool=SimpleNamespace(get=AsyncMock()),
        address="embedding",
        host_service=host,
    )
    inputs = SimpleNamespace(
        multimodal_inputs=[Mock()],
        token_ids=Mock(),
        token_type_ids=Mock(),
        input_lengths=Mock(),
        input_length=1,
    )

    async def run():
        for _ in range(2):
            await ns["generate_embeddings_grpc"](endpoint, inputs, SimpleNamespace())

    asyncio.run(run())
    assert [c.args[0].vit_role_addr for c in stub.embedding.call_args_list] == [
        "vit-one:1",
        "vit-two:2",
    ]
    assert host.get_backend_role_addrs.call_count == 2
