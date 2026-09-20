import ast
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.version import Version
from setuptools import Distribution, find_namespace_packages

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Hosts that serve internal-only / non-publicly-reproducible build artifacts. Direct wheel pins
# from these must never reach a publicly published wheel's install metadata. Kept deliberately
# narrow (see the review decision): the OSS `rtp-opensource.*` bucket and `download.pytorch.org`
# are public and allowed.
INTERNAL_ONLY_HOST_MARKERS = ("sinian-metrics-platform",)

# Platform extras whose stack is merged into install_requires of a PUBLICLY published wheel.
# `rocm` is intentionally excluded: its wheel is not published to public indexes, so its
# internal `sinian-metrics-platform` pins never ship to external users. If ROCm wheels ever
# become publicly published, add "rocm" here and relocate those pins to the internal overlay.
PUBLIC_PLATFORM_EXTRAS = ("cuda12", "cuda12_arm", "cuda12_9", "cuda13", "cuda13_arm")


def _oss_optional_extras() -> dict:
    """Load [project.optional-dependencies] from the OSS extras file that ships in this repo."""
    extras_file = PROJECT_ROOT / "_build" / "oss_optional_extras.toml"
    with open(extras_file, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("optional-dependencies", {})


def _requirement_url(req: str) -> str:
    """Return the direct-reference URL of a `name @ url` requirement, else ''."""
    parts = req.split(" @ ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _is_local_path_reference(url: str) -> bool:
    """True if a direct reference points at a local path rather than a remote artifact."""
    if not url:
        return False
    if url.startswith("file:"):
        return True
    # Absolute or relative filesystem paths (never valid for a published wheel).
    return url.startswith(("/", "./", "../"))


def _load_platform_module():
    """Load _build/platform.py in isolation (stdlib-only, no side effects)."""
    spec = importlib.util.spec_from_file_location(
        "_rtp_build_platform_under_test", PROJECT_ROOT / "_build" / "platform.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "_rtp_llm_setup_under_test", PROJECT_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(os.environ, {"RTP_BAZEL_CONFIG": "--config=cuda12_9"}, clear=False):
        spec.loader.exec_module(module)
    return module


class BuildPackagingContractTest(TestCase):
    def test_native_test_command_preserves_multiple_cpp_targets(self):
        setup_module = _load_setup_module()
        command = setup_module.BazelTest(Distribution())
        command.test_target = (
            "//rtp_llm/cpp/utils/test:oom //rtp_llm/cpp/cuda_graph/tests:retry"
        )
        with patch.object(setup_module, "rewrite_torch_root"), patch.object(
            setup_module, "detect_build_config", return_value="cuda13"
        ), patch.object(
            setup_module, "_get_bazel_cmd_prefix", return_value=(["bazel"], [])
        ), patch.object(
            setup_module,
            "_run_bazel_with_retry",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            command.run()
        self.assertEqual(
            run.call_args.args[0][1:4],
            [
                "test",
                "//rtp_llm/cpp/utils/test:oom",
                "//rtp_llm/cpp/cuda_graph/tests:retry",
            ],
        )

    def test_arch_select_has_unique_top_level_functions(self):
        source = (PROJECT_ROOT / "arch_config" / "arch_select.bzl").read_text(
            encoding="utf-8"
        )
        functions = [
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        ]
        duplicates = sorted(
            name for name in set(functions) if functions.count(name) > 1
        )
        self.assertEqual(
            duplicates,
            [],
            "Starlark rejects duplicate top-level function definitions",
        )

    def test_rpc_writer_cancellation_declares_memory_exporter(self):
        build_text = (
            PROJECT_ROOT / "rtp_llm" / "cpp" / "model_rpc" / "test" / "BUILD"
        ).read_text(encoding="utf-8")
        start = build_text.index('name = "rpc_writer_cancellation_test"')
        target = build_text[start : build_text.index("\n)\n", start)]
        self.assertIn(
            "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
            target,
        )

    def test_python_native_rocm_extras_pin_validated_kernel_stack(self):
        expected_urls = {
            "aiter": "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/aiter/aiter-0.1.21.dev80%2Bg987203ba5.d20260825-cp310-cp310-linux_x86_64.whl",
            "triton": "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton-3.7.0%2Bamd.rocm7.2.0.gitd0d77a509-cp310-cp310-linux_x86_64.whl",
            "triton-kernels": "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/triton/triton_kernels-1.0.0%2Bamd.rocm7.2.0.gitd0d77a509-py3-none-any.whl",
        }
        requirements = {
            requirement.name: requirement
            for requirement in map(Requirement, _oss_optional_extras()["rocm"])
        }
        self.assertNotIn(
            "amdsmi",
            requirements,
            "Python-native metadata must not contain the invalid amd_smi.tar direct reference",
        )
        self.assertEqual(
            str(requirements["pybind11"].specifier),
            "==3.0.1",
            "AITER JIT modules must share the prebuilt tensor registry ABI",
        )
        self.assertEqual(
            str(requirements["flydsl"].specifier),
            "==0.3.1",
            "FLA kernels require an explicit FlyDSL dependency",
        )
        self.assertEqual(
            requirements["amd-mori"].url,
            "git+https://github.com/ROCm/mori.git@dafdcfcf1e27b0c981b90903ab198b90d29e6867",
            "MoriEP must build from the pinned v1.2.2 source on older workers",
        )
        for package, expected_url in expected_urls.items():
            self.assertEqual(
                requirements[package].url,
                expected_url,
                f"{package} must use the validated Python-native ROCm wheel",
            )

    def test_transformers5_uses_compatible_xgrammar_metadata(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        requirements = {
            requirement.name: requirement
            for requirement in map(
                Requirement,
                pyproject["tool"]["rtp-llm"]["base-dependencies"],
            )
        }
        transformers_version = next(
            Version(specifier.version)
            for specifier in requirements["transformers"].specifier
            if specifier.operator == "=="
        )
        xgrammar_version = next(
            Version(specifier.version)
            for specifier in requirements["xgrammar"].specifier
            if specifier.operator == "=="
        )

        if transformers_version.major >= 5:
            self.assertGreaterEqual(
                xgrammar_version,
                Version("0.2.6rc1"),
                "xgrammar 0.2.5 metadata requires transformers<5",
            )
        self.assertEqual(
            str(requirements["xgrammar"].marker),
            'platform_machine != "aarch64"',
            "ARM must preserve Bazel's optional-xgrammar dependency selection",
        )

    def test_xgrammar_platform_extras_use_compatible_tvm_ffi(self):
        xgrammar_minimum_tvm_ffi = Version("0.1.10")
        for extra_name in ("cuda12", "cuda12_9"):
            extra_requirements = {
                requirement.name: requirement
                for requirement in map(Requirement, _oss_optional_extras()[extra_name])
            }
            tvm_ffi_version = next(
                Version(specifier.version)
                for specifier in extra_requirements["apache-tvm-ffi"].specifier
                if specifier.operator == "=="
            )
            self.assertGreaterEqual(
                tvm_ffi_version,
                xgrammar_minimum_tvm_ffi,
                f"{extra_name} must satisfy xgrammar's apache-tvm-ffi requirement",
            )

    def test_cub_compat_does_not_require_cuda_header_on_rocm(self):
        compat_header = (PROJECT_ROOT / "3rdparty/cub_compat.h").read_text()

        self.assertNotIn("#include <cub/version.cuh>", compat_header)
        self.assertIn("defined(CUB_VERSION)", compat_header)

    def test_dash_sc_protos_are_part_of_python_build_outputs(self):
        setup_module = _load_setup_module()

        expected = {
            "rtp_llm/dash_sc/proto/model_config_pb2.py",
            "rtp_llm/dash_sc/proto/model_config_pb2_grpc.py",
            "rtp_llm/dash_sc/proto/predict_v2_pb2.py",
            "rtp_llm/dash_sc/proto/predict_v2_pb2_grpc.py",
            "rtp_llm/dash_sc/proto/__init__.py",
        }
        self.assertEqual(set(setup_module.DASH_SC_PROTO_OUTPUTS), expected)
        self.assertTrue(expected.issubset(set(setup_module.PROTO_OUTPUTS)))

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            proto_dir = project_root / "rtp_llm/dash_sc/proto"
            generator = proto_dir / "create_grpc_proto.py"
            with patch.object(setup_module.subprocess, "run") as run:
                setup_module._generate_dash_sc_proto_files(project_root)
            run.assert_called_once_with(
                [setup_module.sys.executable, str(generator), str(proto_dir)],
                check=True,
            )

    def test_core_build_stages_grammar_tokenizer_binding(self):
        setup_module = _load_setup_module()

        self.assertIn(
            (
                "core",
                "//:th_grammar_tokenizer_info",
                ("libth_grammar_tokenizer_info.so",),
            ),
            setup_module._CORE_BAZEL_STAGED_OUTPUTS,
        )

    def test_core_build_stages_native_repetition_tracker(self):
        setup_module = _load_setup_module()

        self.assertIn(
            (
                "core",
                "//rtp_llm/cpp/repetition:online_repetition_tracker",
                ("libonline_repetition_tracker.so",),
            ),
            setup_module._CORE_BAZEL_STAGED_OUTPUTS,
        )

    def test_stubgen_preloads_native_modules_in_runtime_order(self):
        setup_module = _load_setup_module()

        self.assertEqual(
            setup_module._pybind_stubgen_preload("libth_transformer"),
            "rtp_llm.ops.ensure_engine_ops_loaded(); ",
        )
        self.assertEqual(
            setup_module._pybind_stubgen_preload("librtp_compute_ops"),
            "rtp_llm.ops.ensure_compute_ops_loaded(); ",
        )
        self.assertEqual(
            setup_module._pybind_stubgen_preload("libth_transformer_config"),
            "",
        )

    def _bazel_cmd_prefix(self, setup_module, scope=None):
        env = {"XDG_CACHE_HOME": "/tmp/rtp-llm-test-cache"}
        if scope is not None:
            env["RTP_BAZEL_CACHE_SCOPE"] = scope
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                setup_module,
                "parse_bazel_config",
                return_value=["--config=cuda12_9"],
            ),
            patch.object(setup_module, "_get_remote_bazel_args", return_value=[]),
            patch.object(setup_module, "_get_local_jobs_args", return_value=[]),
        ):
            if scope is None:
                os.environ.pop("RTP_BAZEL_CACHE_SCOPE", None)
            return setup_module._get_bazel_cmd_prefix("cuda12_9")

    def test_bazel_cache_scope_defaults_to_platform_cache(self):
        setup_module = _load_setup_module()

        cmd, build_args = self._bazel_cmd_prefix(setup_module)

        self.assertEqual(
            cmd,
            [
                "bazelisk",
                "--output_user_root=/tmp/rtp-llm-test-cache/bazel_cuda12_9_cache",
            ],
        )
        self.assertEqual(build_args, ["--config=cuda12_9"])

    def test_bazel_cache_scope_isolates_cpp_ut_cache(self):
        setup_module = _load_setup_module()

        cmd, build_args = self._bazel_cmd_prefix(setup_module, scope="cpp_ut")

        self.assertEqual(
            cmd,
            [
                "bazelisk",
                "--output_user_root=/tmp/rtp-llm-test-cache/bazel_cuda12_9_cpp_ut_cache",
            ],
        )
        self.assertEqual(build_args, ["--config=cuda12_9"])

    def test_bazel_cache_scope_rejects_unsafe_path_content(self):
        setup_module = _load_setup_module()

        with self.assertRaisesRegex(ValueError, "RTP_BAZEL_CACHE_SCOPE"):
            self._bazel_cmd_prefix(setup_module, scope="../../cpp-ut")

    def test_remote_bazel_tests_allow_for_gpu_lock_queueing(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=True):
            args = setup_module._with_default_remote_test_timeout(["--config=cuda12_9"])

        self.assertEqual(args, ["--config=cuda12_9", "--test_timeout=900"])

    def test_explicit_remote_test_timeout_is_preserved(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=True):
            args = setup_module._with_default_remote_test_timeout(
                ["--config=cuda12_9", "--test_timeout=1200"]
            )

        self.assertEqual(args, ["--config=cuda12_9", "--test_timeout=1200"])

    def test_local_bazel_tests_keep_native_timeout(self):
        setup_module = _load_setup_module()

        with patch.object(setup_module, "is_remote_enabled", return_value=False):
            args = setup_module._with_default_remote_test_timeout(["--config=cuda12_9"])

        self.assertEqual(args, ["--config=cuda12_9"])

    def test_bazel_no_test_targets_is_failure(self):
        setup_module = _load_setup_module()

        with self.assertRaisesRegex(SystemExit, "4"):
            setup_module._require_successful_bazel_test(4)

    def test_bazel_test_success_is_accepted(self):
        setup_module = _load_setup_module()

        self.assertIsNone(setup_module._require_successful_bazel_test(0))

    def test_build_cleanup_removes_only_stale_test_artifacts(self):
        setup_module = _load_setup_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "project"
            output_root = tmp_path / "bazel_cuda12_9_cache"
            testlogs = (
                output_root
                / "install-hash"
                / "execroot"
                / "rtp_llm"
                / "bazel-out"
                / "k8-opt"
                / "testlogs"
            )
            testlogs.mkdir(parents=True)
            (testlogs / "old-test.xml").write_text("old", encoding="utf-8")
            keep = output_root / "install-hash" / "action-cache"
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text("keep", encoding="utf-8")

            project_root.mkdir()
            (project_root / "bazel-testlogs").symlink_to(testlogs)
            remote_logs = project_root / ".pytest_cache" / "remote_stream_logs"
            remote_logs.mkdir(parents=True)
            (remote_logs / "old.log").write_text("old", encoding="utf-8")

            setup_module._clean_stale_test_artifacts(
                project_root,
                ["bazelisk", f"--output_user_root={output_root}"],
            )

            self.assertFalse(testlogs.exists())
            self.assertFalse((project_root / "bazel-testlogs").exists())
            self.assertFalse(remote_logs.exists())
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep")

    def test_cuda129_stages_cuda_graph_pytest_binding_outside_wheel_glob(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs(
            "cuda12_9", ["--config=cuda12_9"]
        )
        runner = [
            entry
            for entry in staged
            if entry[1] == "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"
        ]

        self.assertEqual(
            runner,
            [
                (
                    "test",
                    "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner",
                    (
                        (
                            "libtest_cuda_graph_runner.so",
                            "test/libtest_cuda_graph_runner.so",
                        ),
                    ),
                )
            ],
        )

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        excluded = pyproject["tool"]["setuptools"]["exclude-package-data"]["rtp_llm"]
        self.assertIn("libs/test/*", excluded)

    def test_cuda129_stages_pywrapped_model_pytest_binding(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs(
            "cuda12_9", ["--config=cuda12_9"]
        )
        self.assertIn(
            (
                "test",
                "//rtp_llm/cpp/models/test:th_pywrapped_model_cache_store_integration_test",
                (
                    (
                        "libth_pywrapped_model_cache_store_integration_test.so",
                        "test/libth_pywrapped_model_cache_store_integration_test.so",
                    ),
                ),
            ),
            staged,
        )

    def test_pywrapped_model_integration_test_is_native_pytest_only(self):
        build_file = PROJECT_ROOT / "rtp_llm/cpp/models/test/BUILD"
        self.assertTrue(
            build_file.exists(), "source-only Bazel BUILD file is not staged"
        )
        build_text = build_file.read_text(encoding="utf-8")
        target_block = re.search(
            r'py_test\(\s*name = "pywrapped_model_cache_store_integration_test",'
            r".*?\n\)",
            build_text,
            re.S,
        )
        self.assertIsNotNone(target_block)
        self.assertIn('tags = ["manual"]', target_block.group(0))

        test_file = (
            PROJECT_ROOT
            / "rtp_llm/cpp/models/test/pywrapped_model_cache_store_integration_test.py"
        )
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        test_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "PyWrappedModelCacheStoreIntegrationTest"
        )
        decorators = {ast.unparse(node) for node in test_class.decorator_list}
        self.assertIn(
            "pytest.mark.gpu(type='MI308X' if torch.version.hip else 'H20')",
            decorators,
        )
        test_text = test_file.read_text(encoding="utf-8")
        self.assertIn("subprocess.run(", test_text)
        self.assertIn('"--native-scenario"', test_text)
        self.assertIn("_run_native_scenario_isolated", test_text)
        self.assertNotIn("ensure_compute_ops_loaded()", test_text)

        binding_file = (
            PROJECT_ROOT
            / "rtp_llm/cpp/models/test/PyWrappedModelCacheStoreIntegrationTest.cc"
        )
        binding_text = binding_file.read_text(encoding="utf-8")
        self.assertIn("registerPyOpDefs(m)", binding_text)
        self.assertNotIn('py::module_::import("librtp_compute_ops")', binding_text)
        self.assertIn(
            "std::make_shared<StoreContext>(shared_from_this())", binding_text
        )
        self.assertIn(
            "store_context->store(request_block_buffers, timeout_ms)", binding_text
        )

        build_text = build_file.read_text(encoding="utf-8")
        self.assertIn(
            '"//rtp_llm/models_py/bindings/core:exec_ops_test_lib"', build_text
        )

    def test_config_pickle_test_is_h20_pytest_only(self):
        build_file = PROJECT_ROOT / "rtp_llm/cpp/pybind/BUILD"
        self.assertTrue(
            build_file.exists(), "source-only Bazel BUILD file is not staged"
        )
        build_text = build_file.read_text(encoding="utf-8")
        target_block = re.search(
            r'py_test\(\s*name = "config_pickle_test",.*?\n\)',
            build_text,
            re.S,
        )
        self.assertIsNotNone(target_block)
        self.assertIn('tags = ["manual"]', target_block.group(0))

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        self.assertIn(
            "rtp_llm/cpp/pybind",
            pyproject["tool"]["pytest"]["ini_options"]["testpaths"],
        )

        test_file = PROJECT_ROOT / "rtp_llm/cpp/pybind/config_pickle_test.py"
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        test_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GrammarConfigPickleTest"
        )
        decorators = {ast.unparse(node) for node in test_class.decorator_list}
        self.assertIn("pytest.mark.H20", decorators)

    def test_cuda_graph_runner_defers_native_import_until_execution(self):
        module_path = (
            PROJECT_ROOT / "rtp_llm/cpp/cuda_graph/tests/cuda_graph_test_runner.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_rtp_cuda_graph_test_runner_contract", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        with patch.dict("sys.modules", {"libtest_cuda_graph_runner": None}):
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(
                ImportError, "libtest_cuda_graph_runner.so not found"
            ):
                module.CudaGraphRunner()

    def test_cuda_graph_unittest_fixtures_do_not_initialize_during_collection(self):
        for filename, class_name in (
            ("cuda_graph_decode_padding.py", "TestCudaGraphDecodePadding"),
            ("cuda_graph_prefill.py", "TestCudaGraphPrefill"),
        ):
            path = PROJECT_ROOT / "rtp_llm/cpp/cuda_graph/tests" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
            test_class = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            methods = {
                node.name
                for node in test_class.body
                if isinstance(node, ast.FunctionDef)
            }
            self.assertIn("setUp", methods)
            self.assertNotIn("__init__", methods)

    def test_rocm_stages_generation_graph_and_beam_search_bindings(self):
        setup_module = _load_setup_module()

        staged = setup_module._selected_bazel_staged_outputs("rocm", ["--config=rocm"])

        self.assertIn(
            "//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner",
            [entry[1] for entry in staged],
        )
        self.assertIn(
            (
                "test",
                "//rtp_llm/models_py/bindings/rocm/ops/tests:beam_search_op_test_bin",
                (("beam_search_op_test_bin", "test/rocm_beam_search_op_test"),),
            ),
            staged,
        )

    def test_rdma_exporter_and_generated_aiter_source_are_staged(self):
        setup_module = _load_setup_module()
        entries = setup_module._selected_bazel_staged_outputs("rocm", ["--config=rocm"])
        targets = {
            "//:mm_rdma_exporter",
            "//rtp_llm/models_py/triton_kernels:aiter_gdr_decode_padding_source",
        }
        selected = [entry for entry in entries if entry[1] in targets]
        self.assertEqual({entry[1] for entry in selected}, targets)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bazel-bin").mkdir()
            (root / "bazel-bin/libmm_rdma_exporter.so").write_bytes(b"exporter")
            source = (
                root
                / "bazel-bin/rtp_llm/models_py/triton_kernels/fla/_aiter_gdr_decode_padding.py"
            )
            source.parent.mkdir(parents=True)
            source.write_text("PADDING_BACKEND = 'patched'\n")
            setup_module.stage_bazel_outputs(root, selected)
            self.assertEqual(
                (root / "rtp_llm/libs/libmm_rdma_exporter.so").read_bytes(), b"exporter"
            )
            generated = (
                root
                / "rtp_llm/models_py/triton_kernels/fla/_aiter_gdr_decode_padding.py"
            )
            self.assertEqual(generated.read_text(), source.read_text())

    def test_cuda13_stages_all_flashinfer_shared_dependencies(self):
        setup_module = _load_setup_module()
        build_tree = ast.parse(
            (PROJECT_ROOT / "3rdparty/flashinfer/flashinfer_cu13.BUILD").read_text()
        )
        names = {
            node.value.args[0].value
            for node in build_tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "sub_lib"
        }
        self.assertEqual(len(names), 11)
        for config in ("cuda13", "cuda13_arm"):
            entries = setup_module._selected_bazel_staged_outputs(
                config, [f"--config={config}"]
            )
            selected = [entry for entry in entries if "@flashinfer_cpp_cu13//:" in entry[1]]
            self.assertEqual({entry[1].split(":")[-1] for entry in selected}, names)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                outputs = root / "bazel-bin/external/flashinfer_cpp_cu13"
                outputs.mkdir(parents=True)
                for name in names:
                    (outputs / f"lib{name}.so").write_bytes(name.encode())
                setup_module.stage_bazel_outputs(root, selected)
                for name in names:
                    self.assertEqual(
                        (root / "rtp_llm/libs" / f"lib{name}.so").read_bytes(),
                        name.encode(),
                    )
                (outputs / "libflashinfer_single_decode.so").unlink()
                with self.assertRaisesRegex(RuntimeError, "libflashinfer_single_decode.so"):
                    setup_module.stage_bazel_outputs(root, selected)
        for config in ("cuda12_9", "rocm", "cuda13_ppu"):
            entries = setup_module._selected_bazel_staged_outputs(
                config, [f"--config={config}"]
            )
            self.assertFalse(any("@flashinfer_cpp_cu13//:" in entry[1] for entry in entries))

    def test_mainline_python_and_sm120_methods_have_native_routes(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as stream:
            config = tomllib.load(stream)
        profiles = config["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        restored = {
            'rtp_llm/model_loader/test/test_stacked_moe_weight.py',
            'rtp_llm/model_loader/test/test_pure_tp_preshard.py',
            'rtp_llm/model_loader/test/test_compressed_w4a8_int4_per_channel.py',
            'rtp_llm/model_loader/test/test_per_channel_fp8_helpers.py',
            'rtp_llm/model_loader/test/test_compressed_w8a8_int8_per_channel.py',
            'rtp_llm/model_loader/test/test_swizzle_load_config.py',
            'rtp_llm/model_loader/test/test_model_weight_info.py',
            'rtp_llm/models/qwen3_next/test/qwen3_next_mtp_test.py',
            'rtp_llm/models/test/dsv4_kv_cache_test.py',
            'rtp_llm/models/test/deepseek_v4_dspark_commit_only_test.py',
            'rtp_llm/models/test/moe_config_propagation_test.py',
            'rtp_llm/tools/convert/weights_convert_test.py',
            'rtp_llm/cpp/models/context_parallel/test/context_parallel_py_wrapper_test.py',
            'rtp_llm/cpp/models/eplb/test/eplb_py_wrapper_test.py',
        }
        for name in ("py_ut_sm8x", "py_ut_oss_sm8x"):
            profile = profiles[name]
            self.assertTrue(restored <= set(profile["isolated_paths"]))
            self.assertFalse(restored & set(profile.get("ignore_paths", [])))
            for path in restored:
                self.assertTrue((PROJECT_ROOT / path).is_file())
                self.assertTrue(any(path == root or path.startswith(root.rstrip("/") + "/")
                                    for root in profile["paths"]))
                self.assertFalse(any(path.startswith(arg.split("=", 1)[1])
                                     for arg in config["tool"]["pytest"]["ini_options"]["addopts"]
                                     if arg.startswith("--ignore=")))
        from rtp_llm.test.remote_tests.plugin import RemoteREAPIPlugin
        from rtp_llm.test.remote_tests.remote_exec_rtp import RemoteRuntimeConfig
        from rtp_llm.test.remote_tests.remote_timeout_policy import select_remote_timeout_policy

        cpu_path = "rtp_llm/model_loader/test/test_pure_tp_preshard.py"
        for name in ("py_ut_sm8x", "py_ut_oss_sm8x"):
            profile = profiles[name]
            self.assertEqual(profile["isolated_env"], {cpu_path: {"CUDA_VISIBLE_DEVICES": ""}})
            plugin = object.__new__(RemoteREAPIPlugin)
            plugin.workers = 4
            plugin._collect_outputs = False
            plugin.config = SimpleNamespace(
                option=SimpleNamespace(markexpr=profile["markexpr"], keyword=""),
                rootpath=PROJECT_ROOT,
            )
            plugin.timeout_policy = select_remote_timeout_policy(name, per_test=False)
            runtime = RemoteRuntimeConfig(ignore_args=[], env_vars={},
                                          platform_properties={"gpu": "A10", "gpu_count": "4"},
                                          remote_setup_prefix="")
            shell = plugin._build_session_command("", runtime, ci_profile=name)[2]
            cpu_start = shell.index(f"--- Isolated file: {cpu_path} ---")
            next_start = shell.index("--- Isolated file:", cpu_start + 20)
            self.assertIn("device_resource.py env CUDA_VISIBLE_DEVICES= python -m pytest",
                          shell[cpu_start:next_start])
            self.assertEqual(shell.count("env CUDA_VISIBLE_DEVICES="), 1)
        loader_source = (PROJECT_ROOT / "rtp_llm/model_loader/test/test_compressed_w8a8_int8_per_channel.py").read_text()
        self.assertIn('pytest.mark.gpu(type="A10")', loader_source)
        self.assertNotIn('pytest.mark.gpu(type="H20")', loader_source)
        sm120 = profiles["py_ut_sm120"]
        self.assertEqual(sm120["gpu_type"], "RTX_5000_PRO")
        self.assertEqual(len(sm120["paths"]), 7)
        self.assertTrue(sm120["isolate_all_paths"])
        self.assertTrue(sm120["require_isolated_tests"])
        for path in sm120["paths"]:
            self.assertTrue((PROJECT_ROOT / path.split("::")[0]).is_file())
        trt = [path for path in sm120["paths"] if "trtllm_fmha_v2" in path]
        self.assertEqual(len(trt), 2)
        self.assertTrue(all(path.endswith("BF16") for path in trt))

    def test_cuda_wrapper_bindings_are_required_staged_test_inputs(self):
        setup_module = _load_setup_module()
        expected = {
            f"//rtp_llm/cpp/models/{name}/test:th_{name}_py_wrapper_test":
            f"libth_{name}_py_wrapper_test.so"
            for name in ("context_parallel", "eplb")
        }
        expected["//rtp_llm/cpp/cache/test:cache_config_creator_py_test"] = "libcache_config_creator_py_test.so"
        for name in ("context_parallel", "eplb"):
            source = (PROJECT_ROOT / f"rtp_llm/cpp/models/{name}/test/{name}_py_wrapper_test.py").read_text()
            self.assertNotIn("TEST_SRCDIR", source)
        entries = setup_module._selected_bazel_staged_outputs("cuda12_9", ["--config=cuda12_9"])
        selected = [entry for entry in entries if entry[1] in expected]
        self.assertEqual({entry[1] for entry in selected}, set(expected))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for target, library in expected.items():
                directory = root / "bazel-bin" / target[2:].split(":")[0]
                directory.mkdir(parents=True)
                (directory / library).write_bytes(library.encode())
            setup_module.stage_bazel_outputs(root, selected)
            for library in expected.values():
                self.assertEqual((root / "rtp_llm/libs/test" / library).read_bytes(), library.encode())
            target, library = next(iter(expected.items()))
            (root / "bazel-bin" / target[2:].split(":")[0] / library).unlink()
            with self.assertRaisesRegex(RuntimeError, library):
                setup_module.stage_bazel_outputs(root, selected)
        for config in ("rocm", "cuda12_9_arm", "cuda13", "cuda13_arm", "cuda13_ppu"):
            entries = setup_module._selected_bazel_staged_outputs(config, [f"--config={config}"])
            self.assertFalse(any(entry[1] in expected for entry in entries))

    def test_dynamic_version_uses_release_version(self):
        setup_module = _load_setup_module()
        release_text = (PROJECT_ROOT / "rtp_llm" / "release_version.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r'^RELEASE_VERSION\s*=\s*["\']([^"\']+)["\']', release_text, re.M
        )
        assert match is not None
        expected = match.group(1)

        self.assertEqual(setup_module.get_release_version(), expected)
        self.assertEqual(setup_module.get_version_with_platform(), f"{expected}+cu129")

    def test_public_platform_extras_have_no_internal_only_wheel_pins(self):
        """Publicly published wheels must not carry internal-only wheel sources in their metadata.

        setup.get_all_dependencies() merges the auto-detected platform's extras into
        install_requires, so any internal-only direct wheel pin in a public platform stack would
        leak the internal source into the published wheel's install metadata. Assert none of the
        public platform extras reference an internal-only host.
        """
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if any(marker in url for marker in INTERNAL_ONLY_HOST_MARKERS):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders,
            [],
            "Public platform extras must not pin internal-only wheels; move these to the "
            f"internal overlay (internal_source/pyproject_internal.toml):\n{offenders}",
        )

    def test_public_platform_extras_have_no_local_path_dependencies(self):
        """Public platform extras must not reference local filesystem paths (non-reproducible)."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                if _is_local_path_reference(_requirement_url(req)):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders,
            [],
            f"Public platform extras must not use local paths:\n{offenders}",
        )

    def test_public_platform_extras_use_https_for_direct_wheels(self):
        """Direct wheel pins in public platform extras must be fetched over HTTPS, not plaintext."""
        extras = _oss_optional_extras()
        offenders = []
        for extra in PUBLIC_PLATFORM_EXTRAS:
            for req in extras.get(extra, []):
                url = _requirement_url(req)
                if url.startswith("http://"):
                    offenders.append(f"{extra}: {req}")
        self.assertEqual(
            offenders,
            [],
            f"Public platform extras must use https:// wheel URLs:\n{offenders}",
        )

    def test_cuda129_rtp_kernel_pin_contains_sm120_cubins(self):
        """Keep the SM120-capable rtp-kernel build across packaging migrations."""
        public_deps = _oss_optional_extras()["cuda12_9"]
        public_pins = [req for req in public_deps if req.startswith("rtp_kernel @ ")]
        self.assertEqual(len(public_pins), 1)
        self.assertIn("/rtp_kernel_260612/", public_pins[0])

        internal_overlay = PROJECT_ROOT / "internal_source" / "pyproject_internal.toml"
        if internal_overlay.exists():
            with open(internal_overlay, "rb") as f:
                internal_config = tomllib.load(f)
            if "project" not in internal_config:
                # OSS CI retains only dispatch configuration after stripping
                # the internal package and its dependency overlay.
                self.assertEqual(set(internal_config), {"tool"})
                self.assertEqual(set(internal_config["tool"]), {"rtp-llm"})
                self.assertEqual(set(internal_config["tool"]["rtp-llm"]), {"remote"})
                return
            internal_extras = internal_config["project"]["optional-dependencies"]
            internal_pins = [
                req
                for req in internal_extras["cuda12_9"]
                if req.startswith("rtp_kernel @ ")
            ]
            self.assertEqual(len(internal_pins), 1)
            self.assertIn("/rtp_kernel_260612/", internal_pins[0])

    def test_pytest_testpaths_all_exist(self):
        """Every configured pytest testpath must exist.

        A stale/typo'd testpath (e.g. rtp_llm/models/multimodal/test vs the real
        rtp_llm/multimodal/test) is silently dropped by pytest, so those tests never run in CI.
        Assert each path resolves so such drift fails loudly at contract-test time instead.
        """
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        testpaths = pyproject["tool"]["pytest"]["ini_options"]["testpaths"]
        missing = [p for p in testpaths if not (PROJECT_ROOT / p).exists()]
        self.assertEqual(
            missing,
            [],
            f"pyproject testpaths point at non-existent directories: {missing}",
        )

    def test_py_ut_amd_profile_collects_rocm_sources(self):
        """ROCm collection must not import unrelated CUDA-only modules."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        profile = profiles["py_ut_amd"]
        expected_paths = [
            "rtp_llm/models_py/triton_kernels/fla/test/test_aiter_flydsl_gdn_decode.py",
            "rtp_llm/cpp/cuda_graph/tests/cuda_graph_tagged_cache_test.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_aiter_flydsl_gdn_prefill.py",
            "rtp_llm/cpp/cuda_graph/tests/cuda_graph_copy_kernel_test.py",
            "rtp_llm/cpp/models/test/pywrapped_model_cache_store_integration_test.py",
            "rtp_llm/model_loader/test/test_inline_fp8_quant.py",
            "rtp_llm/models_py/modules/factory/fused_moe/defs/test/fused_moe_allreduce_contract_test.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_gdn_decode.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_flydsl_chunk_gdn_shape_gate.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_l2norm.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_chunk_prefill.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_gdn_block_prefill.py",
            "rtp_llm/models_py/triton_kernels/moe/test/test_remap_local_ids.py",
            "rtp_llm/models_py/model_desc/test/generic_moe_allreduce_test.py",
            "rtp_llm/models_py/bindings/rocm/ops/tests/test_rocm_beam_search_op.py",
            "rtp_llm/models_py/modules/base/rocm/test/",
            "rtp_llm/models_py/modules/factory/attention/rocm_impl/test/",
            "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/",
            "rtp_llm/models_py/modules/factory/linear/impl/rocm/test/",
            "rtp_llm/models_py/model_desc/test/qwen3_next_qkvz_ba_fusion_test.py",
            "rtp_llm/models_py/model_desc/test/qwen_gdn_graph_replay_test.py",
            "rtp_llm/models_py/distributed/test/moriep_test.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_flydsl_chunk_gdn_cache_store.py",
            "rtp_llm/utils/test/ckpt_database_test.py",
            "rtp_llm/utils/test/jit_cache_smoke_test.py",
        ]
        self.assertEqual(profile["paths"], expected_paths)
        self.assertTrue(all((PROJECT_ROOT / path).exists() for path in expected_paths))
        self.assertIn("MI308X", profile["markexpr"])
        self.assertIn("not MI308X", profiles["py_ut_sm8x"]["markexpr"])

    def test_broad_py_ut_profiles_reject_empty_runs(self):
        """Main-tracking UT profiles need a stable non-zero execution contract."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        for name in (
            "py_ut_sm8x",
            "py_ut_oss_sm8x",
            "py_ut_sm9x",
            "py_ut_l20",
            "py_ut_oss_l20",
            "py_ut_sm100",
            "py_ut_sm100_arm",
            "py_ut_amd",
            "py_ut_frontend",
        ):
            self.assertEqual(
                profiles[name].get("minimum_count"),
                1,
                f"{name} must fail instead of reporting a successful 0/0 run",
            )

        self.assertTrue(profiles["py_ut_oss_sm8x"].get("forbid_skips"))
        self.assertIn("not open_skip", profiles["py_ut_oss_sm8x"]["markexpr"])
        self.assertIn("not frontend", profiles["py_ut_oss_sm8x"]["markexpr"])
        self.assertEqual(
            profiles["py_ut_sm100"].get("paths"),
            [
                "rtp_llm/models_py/kernels/cuda/test/pack_ue8m0_kernel_test.py",
                "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/fp8_deepgemm_linear_sm100_test.py",
            ],
        )
        self.assertIn("not SM100_ARM", profiles["py_ut_sm100"]["markexpr"])

        for name, expected_count in {
            "py_ut_sm8x": 2968,
            "py_ut_oss_sm8x": 2968,
            "py_ut_sm9x": 526,
            "py_ut_sm100": 20,
            "py_ut_sm120": 118,
            "py_ut_l20": 86,
            "py_ut_oss_l20": 84,
            "py_ut_sm100_arm": 104,
            "py_ut_amd": 405,
            "py_ut_frontend": 71,
        }.items():
            self.assertEqual(profiles[name].get("expected_count"), expected_count)
            self.assertTrue(
                profiles[name].get("forbid_skips"),
                f"{name} must reject partial runs with skipped tests",
            )
        l20 = profiles["py_ut_l20"]
        self.assertEqual(l20["gpu_type"], "L20")
        self.assertTrue(l20["isolate_all_paths"])
        self.assertTrue(l20["require_isolated_tests"])
        self.assertEqual(l20["paths"], [
            "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_py_flashinfer_mha_decode.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_flashinfer_prefill/test_py_flashinfer_paged_mha_prefill.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_flashinfer_prefill/test_py_flashinfer_ragged_mha_prefill.py",
            "rtp_llm/models_py/triton_kernels/common/test/silu_mul_masked_test.py",
            "rtp_llm/models_py/kernels/cuda/test/per_token_group_quant_8bit_test.py",
        ])
        oss_l20 = profiles["py_ut_oss_l20"]
        self.assertEqual(oss_l20["paths"], l20["paths"][:-1])
        self.assertEqual(oss_l20["gpu_type"], "L20")
        self.assertTrue(oss_l20["isolate_all_paths"])
        self.assertTrue(oss_l20["require_isolated_tests"])
        internal_root = PROJECT_ROOT.parent
        internal_overlay = internal_root / "internal_source" / "pyproject_internal.toml"
        if (internal_root / ".git").exists() and internal_overlay.exists():
            with open(internal_overlay, "rb") as f:
                internal_config = tomllib.load(f)
            internal_profiles = internal_config["tool"]["rtp_llm"]["pytest_ci"][
                "profiles"
            ]
            self.assertEqual(internal_profiles["py_ut_gb200"].get("minimum_count"), 1)
            self.assertEqual(
                internal_profiles["py_ut_gb200"].get("expected_count"), 104
            )
            self.assertTrue(internal_profiles["py_ut_gb200"].get("forbid_skips"))
            self.assertEqual(
                internal_profiles["py_ut_gb200"].get("ignore_paths"),
                profiles["py_ut_sm100_arm"]["ignore_paths"],
            )
            self.assertEqual(internal_profiles["py_ut_ppu"].get("minimum_count"), 1)
            self.assertEqual(internal_profiles["py_ut_ppu"].get("expected_count"), 23)

            sm100_arm_env = internal_config["tool"]["rtp_llm"]["pytest_ci"]["gpu_env"][
                "SM100_ARM"
            ]
            self.assertEqual(
                sm100_arm_env["PATH"],
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            )
            self.assertEqual(
                sm100_arm_env["LD_LIBRARY_PATH"],
                "/opt/rocm/lib:/opt/conda310/lib/:/usr/local/nvidia/lib64:"
                "/usr/lib64:/usr/local/cuda/lib64:/opt/amdgpu/lib64:"
                "/usr/local/cuda/extras/CUPTI/lib64",
            )
            for cuda13_only_env in (
                "LD_PRELOAD",
                "DG_JIT_CPP_STANDARD",
                "TRITON_PTXAS_PATH",
            ):
                self.assertNotIn(cuda13_only_env, sm100_arm_env)

            arm_requirements = {
                requirement.name: requirement
                for requirement in map(
                    Requirement,
                    internal_config["project"]["optional-dependencies"]["cuda12_arm"],
                )
            }
            expected_arm_wheels = {
                "deep_gemm": "deep_gemm-2.1.1+ee8d9c7.cu129",
                "rtp_kernel": "rtp_kernel-0.1.0+47e3c4a4.cu129",
                "flash_mla": "flash_mla-1.0.0+fcaae18.cu129",
            }
            for package, wheel_build in expected_arm_wheels.items():
                self.assertIn(
                    wheel_build,
                    arm_requirements[package].url,
                    f"{package} must match the golden-validated GB200 runtime stack",
                )

            self.assertNotIn(
                "xgrammar",
                arm_requirements,
                "ARM must not override the base platform marker for xgrammar",
            )
            expected_arm_versions = {
                "apache-tvm-ffi": "==0.1.8.post2",
                "cuda-pathfinder": "==1.3.2",
                "cuda-tile": "==1.4.0",
                "nvidia-cudnn-frontend": "==1.16.0",
                "nvidia-ml-py": "==13.580.82",
            }
            for package, version in expected_arm_versions.items():
                self.assertEqual(
                    str(arm_requirements[package].specifier),
                    version,
                    f"{package} must match the deterministic GB200 dependency lock",
                )

    def test_rocm_unit_cases_are_routed_by_mi308x_marker(self):
        """ROCm-only cases must be deselected before running on CUDA workers."""

        def is_mi308x_marker(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "gpu"
                and any(
                    keyword.arg == "type"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "MI308X"
                    for keyword in node.keywords
                )
            )

        module_marked = [
            "rtp_llm/models_py/modules/factory/attention/rocm_impl/test/test_aiter_prefill_op.py",
            "rtp_llm/models_py/modules/factory/attention/rocm_impl/test/test_fused_qkv_transpose_v3.py",
            "rtp_llm/models_py/modules/factory/attention/rocm_impl/test/test_aiter_decode_triton_noasm.py",
            "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/test_generic_moe_allreduce.py",
            "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/torch_moe_ref_test.py",
            "rtp_llm/models_py/distributed/test/moriep_test.py",
            "rtp_llm/models_py/triton_kernels/fla/test/test_flydsl_chunk_gdn_cache_store.py",
        ]
        for relative_path in module_marked:
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text())
            pytestmark = next(
                (
                    node.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "pytestmark"
                        for target in node.targets
                    )
                ),
                None,
            )
            self.assertIsNotNone(pytestmark, relative_path)
            self.assertTrue(
                any(is_mi308x_marker(node) for node in ast.walk(pytestmark)),
                relative_path,
            )

        method_marked = {
            "rtp_llm/models_py/model_desc/test/qwen3_next_qkvz_ba_fusion_test.py": {
                "test_in_proj_ba_no_swizzle_when_unaligned",
                "test_in_proj_ba_keeps_swizzle_when_aligned",
            },
            "rtp_llm/utils/test/ckpt_database_test.py": {
                "test_recycling_enabled_on_real_rocm_build",
            },
        }
        for relative_path, method_names in method_marked.items():
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text())
            methods = {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in method_names
            }
            self.assertEqual(methods.keys(), method_names, relative_path)
            for method_name, method in methods.items():
                self.assertTrue(
                    any(is_mi308x_marker(node) for node in method.decorator_list),
                    f"{relative_path}:{method_name}",
                )

    def test_removed_legacy_trt_ops_are_not_collected(self):
        legacy_tests = (
            "test_trt_nonpadded.py",
            "test_trt_padded.py",
            "test_trt_paged_nonpadded.py",
        )
        test_dir = (
            PROJECT_ROOT
            / "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/trt_tests"
        )
        self.assertFalse(any((test_dir / name).exists() for name in legacy_tests))

        current_tests = (
            PROJECT_ROOT
            / "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_trtllm_fmha_v2_prefill.py",
            PROJECT_ROOT
            / "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_trtllm_fmha_v2_paged_prefill.py",
        )
        for test_file in current_tests:
            text = test_file.read_text(encoding="utf-8")
            self.assertIn("TRTLLMFMHAv2", text)
            self.assertNotIn("from rtp_llm.ops.compute_ops import TRTAttnOp", text)

    def test_sm9x_unit_cases_are_executed_or_explicitly_routed(self):
        def parse(relative_path):
            return ast.parse((PROJECT_ROOT / relative_path).read_text())

        def decorators(node):
            return {
                child.attr
                for decorator in node.decorator_list
                for child in ast.walk(decorator)
                if isinstance(child, ast.Attribute)
            }

        def is_gpu_marker(node, gpu_type, count=None):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "gpu"
            ):
                return False
            kwargs = {
                keyword.arg: keyword.value.value
                for keyword in node.keywords
                if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
            }
            return kwargs.get("type") == gpu_type and (
                count is None or kwargs.get("count") == count
            )

        indexer_tree = parse("rtp_llm/models_py/modules/hybrid/test/indexer_test.py")
        cuda_version_assignments = [
            node
            for node in indexer_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CUDA_VERSION_OK"
                for target in node.targets
            )
        ]
        self.assertEqual(len(cuda_version_assignments), 1)
        self.assertIsInstance(cuda_version_assignments[0].value, ast.Call)
        top_level_indexer_imports = [
            node
            for node in indexer_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and "models_py.modules.hybrid.indexer" in ast.unparse(node)
        ]
        self.assertEqual(top_level_indexer_imports, [])
        lazy_loader = next(
            node
            for node in indexer_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_load_indexer_test_symbols"
        )
        lazy_loader_text = ast.unparse(lazy_loader)
        self.assertIn("models_py.modules.hybrid.indexer", lazy_loader_text)
        self.assertIn("models_py.modules.hybrid.test.indexer_ref", lazy_loader_text)

        indexer_ref_tree = parse("rtp_llm/models_py/modules/hybrid/test/indexer_ref.py")
        tilelang_imports = [
            node
            for node in indexer_ref_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and "tilelang" in ast.unparse(node)
        ]
        self.assertEqual(tilelang_imports, [])

        fp8_tree = parse(
            "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/fp8_linear_test.py"
        )
        mask_class = next(
            node
            for node in fp8_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_H20WithoutUE8M0Tests"
        )
        masked_methods = {
            target.id
            for node in mask_class.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            masked_methods,
            {
                "test_fp8_input_with_cached_scales",
                "test_fp8_input_without_cached_scales",
                "test_fp8_input_cache_miss_m_exceeds_max_len",
                "test_global_scale_cache_sharing",
                "test_fp8_input_reproducibility",
            },
        )
        module_assignments = {
            target.id
            for node in fp8_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("CudaFp8DeepGEMMLinearTest", module_assignments)

        pack_tree = parse(
            "rtp_llm/models_py/kernels/cuda/test/pack_ue8m0_kernel_test.py"
        )
        pack_module_markers = [
            marker
            for node in pack_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            and isinstance(node.value, (ast.List, ast.Tuple))
            for marker in node.value.elts
        ]
        self.assertEqual(len(pack_module_markers), 1)
        self.assertTrue(is_gpu_marker(pack_module_markers[0], "H20"))
        self.assertFalse(is_gpu_marker(pack_module_markers[0], "SM100_ARM"))
        deep_gemm_class = next(
            node
            for node in pack_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TestDeepGemmIntegration"
        )
        self.assertTrue(
            any(
                is_gpu_marker(marker, "SM100")
                for marker in deep_gemm_class.decorator_list
            )
        )
        self.assertFalse(
            any(
                is_gpu_marker(marker, "SM100_ARM")
                for marker in deep_gemm_class.decorator_list
            )
        )
        self.assertEqual(
            sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                for node in deep_gemm_class.body
            ),
            2,
        )

        sm100_linear_tree = parse(
            "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/"
            "fp8_deepgemm_linear_sm100_test.py"
        )
        sm100_linear_markers = [
            marker
            for node in sm100_linear_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            and isinstance(node.value, (ast.List, ast.Tuple))
            for marker in node.value.elts
        ]
        self.assertEqual(len(sm100_linear_markers), 1)
        self.assertTrue(is_gpu_marker(sm100_linear_markers[0], "SM100"))
        self.assertEqual(
            sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                for node in ast.walk(sm100_linear_tree)
            ),
            1,
        )

        cp_tree = parse(
            "rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/flashmla_sparse_cp_op_test.py"
        )
        cp_test = next(
            node
            for node in ast.walk(cp_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_cp_tp2_matches_non_cp"
        )
        self.assertTrue(
            any(is_gpu_marker(node, "H20", count=2) for node in cp_test.decorator_list)
        )
        cp_worker_text = ast.unparse(
            next(
                node
                for node in cp_tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_tp2_worker"
            )
        )
        self.assertIn(
            "rtp_llm.models_py.distributed.symm_mem.init_symm_mem_communicator",
            cp_worker_text,
        )
        self.assertNotIn("collective_torch.init_symm_mem_communicator", cp_worker_text)

        strategy_tree = parse(
            "rtp_llm/models_py/modules/factory/fused_moe/tests/test_cuda_strategies.py"
        )
        w4a8_class = next(
            node
            for node in strategy_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "TestCudaW4a8Int4PerChannelNoDPStrategy"
        )
        self.assertNotIn("skip", decorators(w4a8_class))

        xqa_tree = parse("rtp_llm/models_py/kernels/cuda/test/test_xqa_batch_decode.py")
        xqa_test = next(
            node
            for node in ast.walk(xqa_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_xqa_decode_comprehensive"
        )
        self.assertNotIn("skip", decorators(xqa_test))
        test_cases = next(
            node.value
            for node in xqa_test.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "test_cases"
                for target in node.targets
            )
        )
        self.assertEqual(len(test_cases.elts), 4)

        fused_quant_tree = parse(
            "rtp_llm/models_py/kernels/cuda/test/"
            "fused_silu_mul_token_quant_batched_test.py"
        )
        fused_quant_text = ast.unparse(fused_quant_tree)
        self.assertNotIn("TODO: fix q_out mismatch", fused_quant_text)
        self.assertIn("torch.nn.functional.silu(gates)", fused_quant_text)

    def test_sm9x_profile_and_plot_cases_are_manual_perf_tests(self):
        expected_methods = {
            "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/fp8_linear_test.py": {
                "test_profile_cuda_fp8_deepgemm_linear"
            },
            "rtp_llm/models_py/triton_kernels/common/test/silu_mul_masked_test.py": {
                "test_silu_mul_masked_fp8_beam_search_skew_performance",
                "test_profile_fp8_silu_mul_masked",
                "test_profile_bf16_silu_mul_masked",
                "test_plot_silu_mul_masked_fp8_latency_vs_num_local_experts",
                "test_plot_silu_mul_masked_fp8_latency_vs_expected_m",
                "test_plot_silu_mul_masked_fp8_latency_vs_moe_intermediate_size",
                "test_plot_silu_mul_masked_bf16_latency_vs_num_local_experts",
                "test_plot_silu_mul_masked_bf16_latency_vs_expected_m",
                "test_plot_silu_mul_masked_bf16_latency_vs_moe_intermediate_size",
            },
        }
        for relative_path, method_names in expected_methods.items():
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text())
            methods = {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in method_names
            }
            self.assertEqual(methods.keys(), method_names)
            for method_name, method in methods.items():
                decorator_names = {
                    child.attr
                    for decorator in method.decorator_list
                    for child in ast.walk(decorator)
                    if isinstance(child, ast.Attribute)
                }
                self.assertTrue(
                    {"manual", "perf"}.issubset(decorator_names),
                    f"{relative_path}:{method_name}",
                )
                self.assertNotIn("skip", decorator_names)

    def test_cpp_disabled_inventory_matches_upstream_main(self):
        """C++ remains GTest-owned, including its default-disabled inventory."""
        expected = {
            "DISABLED_MallocAutoInjectReducesBlockCount",
            "DISABLED_MallocWithoutCPAllocatesFullBlocks",
            "DISABLED_AllocatorMapperControlsMalloc",
            "DISABLED_InsertAutoInjectsMapper",
            "DISABLED_FlatFallbackLargeLru",
            "DISABLED_PrefixTreeLongSessionChains",
            "DISABLED_DeepSeekFlashDecodeB35P3",
            "DISABLED_benchmarkScoreTokenIdsTorchCopyVsMemcpy",
            "DISABLED_benchmarkLatestFlashinferSamplingVsCurrentRtp",
            "DISABLED_compareLatestFlashinferSamplingAccuracyVsCurrentRtp",
        }
        actual = set()
        pattern = re.compile(r"TEST(?:_F|_P)?\([^,]+,\s*(DISABLED_[A-Za-z0-9_]+)")
        for source_path in (PROJECT_ROOT / "rtp_llm").rglob("*"):
            if source_path.suffix in {".cc", ".cpp", ".cu", ".h", ".hpp"}:
                actual.update(pattern.findall(source_path.read_text(errors="ignore")))
        self.assertEqual(actual, expected)

        for build_path in (
            "rtp_llm/cpp/cache/test/BUILD",
            "rtp_llm/cpp/normal_engine/speculative/test/BUILD",
            "rtp_llm/models_py/bindings/cuda/ops/tests/BUILD",
        ):
            self.assertNotIn(
                "RTP_LLM_MANUAL_BENCHMARKS",
                (PROJECT_ROOT / build_path).read_text(),
            )

    def test_cpp_device_pin_targets_request_the_resources_they_assert(self):
        def target_block(relative_path, target_name):
            build_text = (PROJECT_ROOT / relative_path).read_text()
            marker_pos = build_text.index(f'name = "{target_name}"')
            block_start = build_text.rfind("cc_test(", 0, marker_pos)
            block_end = build_text.index("\n)", marker_pos) + 2
            return build_text[block_start:block_end]

        device_pin = target_block("rtp_llm/cpp/utils/test/BUILD", "device_pin_test")
        self.assertIn('"GPU_COUNT": "1"', device_pin)
        self.assertIn('"gpu_count": "1"', device_pin)
        self.assertIn('"gpu": "A10"', device_pin)

        cache_store = target_block(
            "rtp_llm/cpp/disaggregate/cache_store/test/BUILD",
            "cache_store_gtest",
        )
        self.assertIn('"GPU_COUNT": "2"', cache_store)
        self.assertIn('"gpu_count": "2"', cache_store)
        self.assertIn('"gpu": "A10"', cache_store)

        block_pool = target_block(
            "rtp_llm/cpp/cache/test/BUILD", "block_pool_device_malloc_test"
        )
        self.assertIn('"GPU_COUNT": "2"', block_pool)
        self.assertIn('"gpu_count": "2"', block_pool)

    def test_non_sm100_py_ut_profiles_ignore_dsv4(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        dsv4_paths = ["rtp_llm/test/dsv4", "rtp_llm/models_py/modules/dsv4"]
        self.assertEqual(profiles["py_ut_sm8x"]["ignore_paths"], dsv4_paths)
        self.assertEqual(profiles["py_ut_sm9x"]["ignore_paths"], dsv4_paths)
        self.assertEqual(profiles["py_ut_sm100_arm"]["ignore_paths"], [
            "rtp_llm/cpp/models/context_parallel/test/context_parallel_py_wrapper_test.py",
            "rtp_llm/cpp/models/eplb/test/eplb_py_wrapper_test.py",
        ])

    def test_dsv4_decode_bazel_targets_are_routed_to_sm100_pytest(self):
        from rtp_llm.test.ci_profile_plugin import _get_profile

        # Check the effective profile, including exclusions added at runtime.
        # Checking only the TOML entry misses broad CUDA 13 directory ignores.
        profile = _get_profile(PROJECT_ROOT, "py_ut_sm100_arm")
        ignored = [PROJECT_ROOT / path for path in profile.get("ignore_paths", [])]
        expected_cases = {
            "decode_attn_metadata_test.py": 14,
            "decode_metadata_in_place_test.py": 4,
            "decode_fmha_impl_test.py": 5,
            "kv_write_decode_test.py": 11,
            "fp8_kv_quant_decode_test.py": 7,
            "fp8_sparse_attn_decode_test.py": 3,
            "indexer_decode_test.py": 4,
        }
        test_dir = PROJECT_ROOT / "rtp_llm/models_py/modules/dsv4/decode/test"
        build_source = (test_dir / "BUILD").read_text()
        build_targets = re.findall(r"py_test\(\n(.*?)\n\)", build_source, re.DOTALL)

        for filename, expected_count in expected_cases.items():
            test_path = test_dir / filename
            self.assertFalse(
                any(test_path == path or path in test_path.parents for path in ignored),
                f"Original SM100_ARM test excluded from its effective profile: {filename}",
            )
            tree = ast.parse((test_dir / filename).read_text())
            gpu_markers = [
                marker
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
                for marker in node.value.elts
                if isinstance(marker, ast.Call)
                and isinstance(marker.func, ast.Attribute)
                and marker.func.attr == "gpu"
            ]
            self.assertEqual(len(gpu_markers), 1, filename)
            marker_kwargs = {
                keyword.arg: keyword.value.value
                for keyword in gpu_markers[0].keywords
                if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
            }
            self.assertEqual(marker_kwargs.get("type"), "SM100_ARM", filename)

            test_count = sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
                for node in ast.walk(tree)
            )
            self.assertEqual(test_count, expected_count, filename)
            target = next(
                (block for block in build_targets if f'srcs = ["{filename}"]' in block),
                None,
            )
            self.assertIsNotNone(target, filename)
            self.assertIn('"SM100_ARM"', target, filename)
            self.assertIn('"gpu": "SM100_ARM"', target, filename)

            unconditional_skips = [
                decorator
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "skip"
            ]
            self.assertEqual(unconditional_skips, [], filename)

        self.assertEqual(sum(expected_cases.values()), 48)

    def test_sm9x_profile_isolates_legacy_native_test_targets(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profile = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]["py_ut_sm9x"]
        isolated_paths = profile["isolated_paths"]
        expected = {
            "rtp_llm/models_py/bindings/cuda/test/concat_and_cache_mla/test_dpsk_bf16.py",
            "rtp_llm/models_py/bindings/cuda/test/concat_and_cache_mla/test_dpsk32_fp8.py",
            "rtp_llm/models_py/bindings/cuda/test/concat_and_cache_mla/test_model1_fp8.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test/test_allgather_cp_impl.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test/test_allgather_overlap_impl.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test/test_alltoall_cp_impl.py",
            "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/test_flashinfer_prefill/test_py_flashinfer_hybrid_mha_prefill.py",
            "rtp_llm/models_py/modules/hybrid/test/indexer_test.py",
        }

        self.assertEqual(set(isolated_paths), expected)
        self.assertTrue(all((PROJECT_ROOT / path).is_file() for path in isolated_paths))

    def test_legacy_flashinfer_hybrid_route_stays_multi_arch_and_isolated(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        hybrid_path = (
            "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/"
            "test_flashinfer_prefill/test_py_flashinfer_hybrid_mha_prefill.py"
        )
        hybrid_source = (PROJECT_ROOT / hybrid_path).read_text()

        self.assertIn("pytest.mark.multi_arch_cuda", hybrid_source)
        self.assertIn(hybrid_path, profiles["py_ut_sm8x"]["isolated_paths"])
        self.assertIn(hybrid_path, profiles["py_ut_sm9x"]["isolated_paths"])
        self.assertIn("multi_arch_cuda", profiles["py_ut_sm9x"]["markexpr"])

    def test_sparse_mla_cp_keeps_legacy_manual_contract(self):
        test_path = (
            PROJECT_ROOT
            / "rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test/"
            "flashmla_sparse_cp_op_test.py"
        )
        source = ast.unparse(ast.parse(test_path.read_text()))

        self.assertIn("pytest.mark.gpu(type='H20')", source)
        self.assertIn("pytest.mark.manual", source)

    def test_smoke_profiles_enforce_native_collection_parity(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profiles = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
        expected_counts = {
            "smoke_h20_light_oss": 16,
            "smoke_h20_full_oss": 58,
            "smoke_sm8x_light_oss": 9,
            "smoke_sm8x_full_oss": 9,
            "smoke_sm100_oss": 12,
            "smoke_sm100_eval_oss": 1,
            "smoke_sm120_oss": 8,
            "smoke_rocm_oss": 26,
            "smoke_amd_internal": 3,
            "smoke_rocm_qwen35_mtp_manual": 1,
            "smoke_remote_cache_oss": 11,
        }
        for name, expected_count in expected_counts.items():
            self.assertEqual(profiles[name]["expected_count"], expected_count)
            self.assertTrue(profiles[name]["forbid_skips"])

        self.assertEqual(
            profiles["smoke_sm120_oss"]["paths"],
            ["rtp_llm/test/smoke/suites/"],
        )
        self.assertEqual(profiles["smoke_sm120_internal"]["expected_count"], 1)

        manifest_source = (
            PROJECT_ROOT / "rtp_llm/test/smoke_framework/manifest.py"
        ).read_text()
        self.assertIn(
            'pytest_module.mark.timeout(int(config.get("timeout", 600)))',
            " ".join(manifest_source.split()),
        )
        for suite_name in ("test_smoke_sm100_dense.py", "test_smoke_sm100_moe.py"):
            suite_source = (
                PROJECT_ROOT / "rtp_llm/test/smoke/suites" / suite_name
            ).read_text()
            self.assertNotIn("@pytest.mark.timeout(7200)", suite_source)

    def test_standalone_perf_pipeline_uses_native_pytest(self):
        """Internal perf CI must not regress to Bazel-owned Python tests."""
        perf_yaml = PROJECT_ROOT.parent / ".aoneci" / "perf.yaml"
        if not perf_yaml.exists():
            return

        source = perf_yaml.read_text()
        self.assertNotIn("BAZEL_TEST_TARGET", source)
        self.assertNotIn("basic_test.sh test", source)
        self.assertNotIn("perf-a100:", source)
        self.assertNotIn("perf-amd:", source)
        self.assertIn('PYTEST_PROFILE: "perf_sm9x"', source)
        self.assertIn('PYTEST_PROFILE: "perf_ppu_internal"', source)
        self.assertIn("--remote-no-test-cache", source)
        self.assertIn("--remote-concurrency=1", source)

        main_source = (PROJECT_ROOT.parent / ".aoneci" / "main.yaml").read_text()
        self.assertIn('PYTEST_PROFILE: "perf_ppu_internal"', main_source)
        self.assertIn("--remote-concurrency=1", main_source)

    def test_non_model_smokes_use_python_native_entrypoints(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        scripts = pyproject["project"]["scripts"]
        self.assertEqual(
            scripts["rtp-llm-json-format-smoke"],
            "rtp_llm.dash_sc.json_format_e2e_smoke:main",
        )
        self.assertEqual(
            scripts["rtp-llm-grammar-validation-smoke"],
            "rtp_llm.dash_sc.grammar_validation_smoke:main",
        )

        dash_build = (PROJECT_ROOT / "rtp_llm/dash_sc/BUILD").read_text()
        dash_test_build = (PROJECT_ROOT / "rtp_llm/dash_sc/test/BUILD").read_text()
        utils_test_build = (PROJECT_ROOT / "rtp_llm/utils/test/BUILD").read_text()
        self.assertNotIn('name = "json_format_e2e_smoke"', dash_build)
        self.assertNotIn('name = "grammar_validation_smoke"', dash_build)
        self.assertNotIn('name = "mrcr_smoke_test"', dash_test_build)
        self.assertNotIn("custom_smoke_test(", utils_test_build)
        self.assertNotIn(
            "bazel",
            (PROJECT_ROOT / "rtp_llm/dash_sc/grammar_validation_smoke.py")
            .read_text()
            .lower(),
        )

        frontend_paths = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"][
            "py_ut_frontend"
        ]["paths"]
        self.assertIn(
            "rtp_llm/dash_sc/test/inference/mrcr_smoke_test.py", frontend_paths
        )

    def test_smoke_runner_defers_runtime_only_imports_during_collection(self):
        runner_path = PROJECT_ROOT / "rtp_llm/test/smoke_framework/runner.py"
        tree = ast.parse(runner_path.read_text(), filename=str(runner_path))
        runtime_only_modules = {
            "rtp_llm.test.smoke.case_runner",
            "rtp_llm.test.smoke.multi_inst_case_runner",
            "rtp_llm.test.smoke.task_info",
            "rtp_llm.test.smoke.utils",
            "rtp_llm.test.smoke.rel_path_config",
        }
        eager_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(runtime_only_modules & eager_imports)

    def test_smoke_runner_preserves_legacy_declared_env_contract(self):
        runner_source = (
            PROJECT_ROOT / "rtp_llm/test/smoke_framework/runner.py"
        ).read_text()

        self.assertIn('append(f"WORLD_SIZE={ws}")', runner_source)
        self.assertNotIn("ENABLE_STABLE_SCATTER_ADD", runner_source)
        self.assertNotIn("DETERMINISTIC_GEMM", runner_source)
        self.assertNotIn("USE_GATHER_BATCH_SCHEDULER", runner_source)

        legacy_entry_source = (PROJECT_ROOT / "rtp_llm/test/smoke/entry.py").read_text()
        self.assertNotIn("USE_GATHER_BATCH_SCHEDULER", legacy_entry_source)

    def test_h20_full_smoke_profile_preserves_remote_jit_cache_contract(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profile = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"][
            "smoke_h20_full_oss"
        ]
        self.assertEqual(
            profile["remote_env"],
            {"REMOTE_JIT_DIR": "/tmp/rtp-llm/.remote_jit_cache"},
        )

    def test_rocm_smoke_profile_preserves_remote_jit_cache_contract(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        profile = pyproject["tool"]["rtp_llm"]["pytest_ci"]["profiles"][
            "smoke_rocm_oss"
        ]
        self.assertEqual(
            profile["remote_env"],
            {"REMOTE_JIT_DIR": "/tmp/rtp-llm/.remote_jit_cache"},
        )

    def test_jit_cache_smoke_adapters_do_not_reexport_testcase(self):
        """Imported TestCase classes are collected again from every adapter module."""
        suite_dir = PROJECT_ROOT / "rtp_llm" / "test" / "smoke" / "suites"
        expected_cases = {
            "test_smoke_h20_jit_cache.py": (
                "jit_cache_deepseek_v2_lite",
                "test_deepseek_v2_lite",
            ),
            "test_smoke_rocm_jit_cache.py": (
                "jit_cache_qwen3_rocm",
                "test_qwen3_rocm",
            ),
        }
        for name, (case_name, unittest_name) in expected_cases.items():
            path = suite_dir / name
            source = path.read_text()
            tree = ast.parse(source)
            direct_imports = [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "rtp_llm.utils.test.jit_cache_smoke_test"
                for alias in node.names
                if alias.name == "JitCacheSmokeTest"
            ]
            self.assertEqual(
                direct_imports,
                [],
                f"{name} exposes JitCacheSmokeTest to pytest collection",
            )
            smoke_cases = next(
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "SMOKE_CASES"
                    for target in node.targets
                )
            )
            self.assertEqual(set(smoke_cases), {case_name})
            self.assertIn(f'JitCacheSmokeTest("{unittest_name}")', source)
            self.assertNotIn("pytest.skip(", source)
            self.assertIn("pytest.fail(", source)

    def test_deepgemm_optional_symbol_does_not_block_available_symbols(self):
        wrapper_path = (
            PROJECT_ROOT
            / "rtp_llm"
            / "models_py"
            / "kernels"
            / "cuda"
            / "deepgemm_wrapper.py"
        )
        tree = ast.parse(wrapper_path.read_text())
        init_func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_lazy_init_deep_gemm"
        )
        namespace = {
            "List": list,
            "has_deep_gemm": lambda: True,
            "resolve_symbol": lambda module, new, old: getattr(
                module, new, getattr(module, old, None)
            ),
            "_deep_gemm_impl_new_map": {
                "available": "available_impl",
                "optional": "optional_impl",
            },
            "_deep_gemm_impl_old_map": {
                "available": "available_impl",
                "optional": "optional_impl",
            },
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[init_func], type_ignores=[])
                ),
                str(wrapper_path),
                "exec",
            ),
            namespace,
        )

        available_impl = object()
        fake_deep_gemm = SimpleNamespace(available_impl=available_impl)
        compiler_env = {
            "CPATH": "/existing/cpath",
            "CPLUS_INCLUDE_PATH": "/existing/cplus",
            "NVCC_PREPEND_FLAGS": "--existing-flag",
        }
        with (
            patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}),
            patch.dict(os.environ, compiler_env, clear=False),
        ):
            namespace["_lazy_init_deep_gemm"](["available", "optional"])
            self.assertEqual(
                {key: os.environ[key] for key in compiler_env}, compiler_env
            )

        self.assertIs(namespace["_available_impl"], available_impl)
        self.assertIsNone(namespace.get("_optional_impl"))

    def test_model_rpc_test_does_not_leak_ops_mocks(self):
        test_path = (
            PROJECT_ROOT
            / "rtp_llm"
            / "cpp"
            / "model_rpc"
            / "test"
            / "model_rpc_client_test.py"
        )
        tree = ast.parse(test_path.read_text())

        leaked_modules = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                value = target.value
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "sys"
                    and value.attr == "modules"
                ):
                    leaked_modules.append(ast.unparse(target))

        self.assertEqual(leaked_modules, [])

    def test_rocm_wheel_version_matches_dependency_abi(self):
        """The rocm wheel version suffix must track the ROCm ABI the rocm extras are built for.

        get_version_with_platform() stamps every OSS ROCm wheel with this suffix, so a stale value
        (e.g. rocm62 while the deps/toolchain moved to ROCm 7.2) makes cache/publish/rollback pick
        the wrong binary stack. Derive the ABI from the suffix and assert the rocm extras' wheels
        actually reference it — and that no wheel references a different ROCm ABI.
        """
        platform_module = _load_platform_module()
        suffix = platform_module.PLATFORM_CONFIG_VERSIONS.get("rocm", "")
        m = re.fullmatch(r"rocm(\d)(\d+)", suffix)
        self.assertIsNotNone(m, f"unexpected rocm version suffix {suffix!r}")
        expected_abi = f"{m.group(1)}.{m.group(2)}"  # rocm72 -> "7.2"

        rocm_reqs = _oss_optional_extras().get("rocm", [])
        rocm_abis = set(re.findall(r"rocm(\d+\.\d+)", " ".join(rocm_reqs)))
        self.assertIn(
            expected_abi,
            rocm_abis,
            f"rocm suffix {suffix!r} (ABI {expected_abi}) not found in rocm extras "
            f"wheel URLs (found ABIs: {sorted(rocm_abis)})",
        )
        stale = rocm_abis - {expected_abi}
        self.assertEqual(
            stale,
            set(),
            f"rocm extras reference ROCm ABIs {sorted(stale)} != suffix {expected_abi}",
        )

    def test_pytest_entry_points_are_packaged_with_tests(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        packages = set(
            find_namespace_packages(
                where=str(PROJECT_ROOT), include=["rtp_llm", "rtp_llm.*"]
            )
        )

        self.assertIn("rtp_llm.test.remote_tests", packages)
        self.assertIn("rtp_llm.test.smoke_framework", packages)
        find_cfg = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertNotIn("exclude", find_cfg)

        entry_points = pyproject["project"]["entry-points"]["pytest11"]
        for target in entry_points.values():
            module_name = target.split(":", 1)[0]
            module_path = PROJECT_ROOT / (module_name.replace(".", "/") + ".py")
            self.assertTrue(module_path.exists(), module_name)

        package_data = pyproject["tool"]["setuptools"]["package-data"]["rtp_llm"]
        self.assertIn("test/**/*.proto", package_data)
        self.assertIn("test/**/*.json", package_data)
        self.assertIn("dash_sc/test/**/*.json", package_data)
        self.assertIn("models_py/**/data/*.json", package_data)
        self.assertIn(
            "models_py/kernels/cuda/fp8_kernel/cutlass_groupgemm/*.json",
            package_data,
        )
