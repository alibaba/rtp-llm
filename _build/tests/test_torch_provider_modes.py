"""Source/repository and CI controller contracts; no Torch or Bazel execution."""

import ast
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from rtp_llm.test.test_torch_repository_discovery import (
    _Context,
    _implementation,
    _RepositoryFailure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_ROOT = PROJECT_ROOT / "internal_source"
TARGETS = ("torch_api", "torch", "torch_libs")
EXTERNAL_ARCHIVES = {
    "@rtp_llm//:using_ppu": "@torch_2.11_py312_ppu",
    "@rtp_llm//:using_rocm": "@torch_rocm",
    "@rtp_llm//:using_arm": "@torch_2.3_py310_cpu_aarch64",
    "@rtp_llm//:using_cuda13_arm": "@torch_2.11_py310_cuda-aarch64",
    "@rtp_llm//:cuda_pre_12_9": "@torch_2.6_py310_cuda",
    "@rtp_llm//:using_cuda13_x86": "@torch_2.11_py310_cuda",
    "@rtp_llm//:using_cuda12_9_x86": "@torch_2.8_py310_cuda",
    "//conditions:default": "@torch_2.1_py310_cpu",
}
INTERNAL_ARCHIVES = {
    **EXTERNAL_ARCHIVES,
    "@rtp_llm//:using_cuda12_arm": "@torch_2.9_py310_cuda-aarch64",
}


def _function(path, name):
    return next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _record_aliases(path):
    """Execute only the plain macro to record its declarations, not providers."""
    aliases = []
    scope = {
        "native": SimpleNamespace(alias=lambda **kwargs: aliases.append(kwargs)),
        "select": lambda choices: choices,
    }
    function = _function(path, "torch_archive_aliases")
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope
    )
    scope[function.name]()
    return aliases


def _archives(path):
    return {
        ast.literal_eval(
            next(item.value for item in call.keywords if item.arg == "name")
        ): call
        for call in ast.walk(ast.parse(path.read_text()))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "http_archive"
    }


class TorchProviderModeTest(TestCase):
    def test_hermetic_never_discovers_host_torch_or_uses_native_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            context = _Context(
                directory,
                {
                    "RTP_TORCH_PROVIDER": "hermetic",
                    "TORCH_ROOT": "/nonexistent/native/torch",
                    "PYTHON_BIN_PATH": "/nonexistent/native/python",
                },
            )
            _implementation()(context)
        self.assertEqual(context.commands, [])
        self.assertEqual(context.links, [])
        self.assertEqual(
            context.files,
            [
                (
                    "BUILD",
                    'load("@arch_config//:arch_select.bzl", "torch_archive_aliases")\ntorch_archive_aliases()\n',
                )
            ],
        )

    def test_unknown_provider_fails_before_discovery_or_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = _Context(directory, {"RTP_TORCH_PROVIDER": "typo"})
            with self.assertRaisesRegex(_RepositoryFailure, "native or hermetic"):
                _implementation()(context)
        self.assertEqual(context.commands, [])
        self.assertEqual(context.files, [])
        self.assertEqual(context.links, [])

    def test_explicit_native_retains_installed_torch_override(self):
        with tempfile.TemporaryDirectory() as directory:
            torch = Path(directory) / "torch"
            torch.mkdir()
            context = _Context(
                directory,
                {"RTP_TORCH_PROVIDER": "native", "TORCH_ROOT": str(torch)},
            )
            _implementation()(context)
        self.assertEqual(context.commands, [])
        self.assertEqual(context.links, [(str(torch), "torch")])
        self.assertEqual(
            context.files[0][1], (PROJECT_ROOT / "BUILD.pytorch.native").read_text()
        )

    def test_public_aliases_have_exact_main_archives_and_real_target_names(self):
        aliases = _record_aliases(PROJECT_ROOT / "arch_config/arch_select.bzl")
        self.assertEqual([alias["name"] for alias in aliases], list(TARGETS))
        for alias in aliases:
            with self.subTest(target=alias["name"]):
                self.assertEqual(
                    alias["actual"],
                    {
                        condition: repo + "//:" + alias["name"]
                        for condition, repo in EXTERNAL_ARCHIVES.items()
                    },
                )
                self.assertEqual(alias["visibility"], ["//visibility:public"])
                self.assertEqual(
                    alias["target_compatible_with"],
                    {
                        "@rtp_llm//:using_cuda13_arm": [],
                        "@rtp_llm//:using_cuda12_arm": ["@platforms//:incompatible"],
                        "//conditions:default": [],
                    },
                )
                self.assertTrue(
                    all(
                        not target.startswith("@torch//")
                        for target in alias["actual"].values()
                    )
                )

    def test_paired_aliases_preserve_ppu_and_cuda12_arm_archives(self):
        aliases = _record_aliases(INTERNAL_ROOT / "bazel/arch_select.bzl")
        self.assertEqual([alias["name"] for alias in aliases], list(TARGETS))
        for alias in aliases:
            with self.subTest(target=alias["name"]):
                self.assertEqual(
                    alias["actual"],
                    {
                        condition: repo + "//:" + alias["name"]
                        for condition, repo in INTERNAL_ARCHIVES.items()
                    },
                )
                self.assertNotIn("target_compatible_with", alias)
                self.assertEqual(alias["visibility"], ["//visibility:public"])
                self.assertTrue(
                    all(
                        not target.startswith("@torch//")
                        for target in alias["actual"].values()
                    )
                )

    def test_every_alias_archive_is_pinned_and_uses_actual_three_target_template(self):
        for root, expected in (
            (PROJECT_ROOT, EXTERNAL_ARCHIVES),
            (INTERNAL_ROOT, INTERNAL_ARCHIVES),
        ):
            archives = _archives(root / "deps/http.bzl")
            for repo in set(expected.values()):
                with self.subTest(owner=str(root), repository=repo):
                    call = archives[repo[1:]]
                    args = {keyword.arg: keyword.value for keyword in call.keywords}
                    self.assertRegex(
                        ast.literal_eval(args["sha256"]), r"^[0-9a-f]{64}$"
                    )
                    url_globals = {}
                    host_file = root / "deps/oss_host.bzl"
                    if host_file.exists():
                        for node in ast.parse(host_file.read_text()).body:
                            if isinstance(node, ast.Assign) and isinstance(
                                node.value, ast.Constant
                            ):
                                for target in node.targets:
                                    if isinstance(target, ast.Name):
                                        url_globals[target.id] = ast.literal_eval(
                                            node.value
                                        )
                    urls = eval(
                        compile(ast.Expression(args["urls"]), "archive_urls", "eval"),
                        {"__builtins__": {}},
                        url_globals,
                    )
                    self.assertTrue(urls)
                    self.assertTrue(
                        all(url.startswith(("http://", "https://")) for url in urls)
                    )
                    label = ast.literal_eval(args["build_file"].args[0])
                    template = PROJECT_ROOT / label.removeprefix("@rtp_llm//").replace(
                        ":", "/"
                    ).lstrip("/")
                    names = {
                        ast.literal_eval(
                            next(
                                item.value
                                for item in call.keywords
                                if item.arg == "name"
                            )
                        )
                        for call in ast.walk(ast.parse(template.read_text()))
                        if isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "cc_library"
                    }
                    self.assertTrue(set(TARGETS) <= names)

    def test_native_and_archive_templates_keep_distinct_actual_cuda13_sources(self):
        native = (PROJECT_ROOT / "BUILD.pytorch.native").read_text()
        archive = (PROJECT_ROOT / "BUILD.pytorch").read_text()
        self.assertIn('"nvidia/nvshmem/lib/libnvshmem_host.so.3"', native)
        self.assertIn(
            '"@pip_gpu_cuda13_torch_nvidia_nvshmem_cu13//:site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3"',
            archive,
        )
        self.assertNotIn("@pip_gpu_cuda13_torch_", native)
        self.assertNotIn('"nvidia/nvshmem/lib/libnvshmem_host.so.3"', archive)
        listing = (PROJECT_ROOT / "_build/sdist_sources.txt").read_text()
        self.assertIn("include WORKSPACE BUILD BUILD.* *.bzl", listing)

    def test_all_three_dependency_entrypoints_keep_shared_alias_identity(self):
        for path in (
            PROJECT_ROOT / "arch_config/arch_select.bzl",
            INTERNAL_ROOT / "bazel/arch_select.bzl",
            PROJECT_ROOT / "bazel/defs.bzl",
        ):
            with self.subTest(owner=str(path)):
                function = _function(path, "torch_deps")
                returned = next(
                    node.value
                    for node in ast.walk(function)
                    if isinstance(node, ast.Return)
                )
                self.assertEqual(
                    ast.literal_eval(returned),
                    ["@torch//:" + target for target in TARGETS],
                )

    def test_workspace_keeps_native_template_and_real_dsv4_install_owner(self):
        tree = ast.parse((PROJECT_ROOT / "WORKSPACE").read_text())
        calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        torch = next(
            node.value
            for node in calls
            if isinstance(node.value.func, ast.Name)
            and node.value.func.id == "torch_local_repository"
        )
        self.assertEqual(
            ast.literal_eval(
                next(item.value for item in torch.keywords if item.arg == "build_file")
            ),
            "//:BUILD.pytorch.native",
        )
        dsv4 = next(
            node.value
            for node in calls
            if isinstance(node.value.func, ast.Name)
            and node.value.func.id == "pip_parse"
        )
        self.assertEqual(
            {keyword.arg: ast.literal_eval(keyword.value) for keyword in dsv4.keywords},
            {
                "name": "pip_dsv4_test",
                "requirements_lock": "//rtp_llm/test/pytest:requirements_lock.txt",
                "python_interpreter": "/opt/conda310/bin/python3",
                "extra_pip_args": [
                    "--extra-index-url=https://mirrors.aliyun.com/pypi/simple/"
                ],
                "timeout": 3600,
            },
        )
        self.assertTrue(
            (PROJECT_ROOT / "rtp_llm/test/pytest/requirements_lock.txt").is_file()
        )
        names = [
            node.value.func.id
            for node in calls
            if isinstance(node.value.func, ast.Name)
        ]
        self.assertEqual(names.count("pip_dsv4_test_install_deps"), 1)
        self.assertLess(names.index("pip_deps"), names.index("pip_parse"))
        self.assertLess(
            names.index("pip_parse"), names.index("pip_dsv4_test_install_deps")
        )
        self.assertIn(
            'load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")',
            (PROJECT_ROOT / "rtp_llm/models_py/modules/dsv4/test/defs.bzl").read_text(),
        )

    def test_provider_selection_is_repository_only_and_has_no_forced_default(self):
        arguments = [
            shlex.split(line, comments=True)
            for line in (PROJECT_ROOT / ".bazelrc").read_text().splitlines()
        ]
        self.assertIn(["build", "--repo_env=RTP_TORCH_PROVIDER"], arguments)
        self.assertFalse(
            any(
                "RTP_TORCH_PROVIDER=" in argument
                for command in arguments
                for argument in command
            )
        )
        self.assertNotIn(
            "RTP_TORCH_PROVIDER", (INTERNAL_ROOT / "ci/run_native_ci.sh").read_text()
        )

    def test_basic_main_child_selects_hermetic_for_sync_and_build_without_env_leak(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "github-opensource").mkdir()
            (root / "internal_source/ci").mkdir(parents=True)
            (root / "internal_source/ci/before_script.sh").write_text(":\n")
            for helper in ("collect_bazel_build_artifacts.py", "collect_vipsrv_dns.py"):
                (root / "internal_source/ci" / helper).write_bytes(
                    (INTERNAL_ROOT / "ci" / helper).read_bytes()
                )
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "bazelisk").write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                "with open(os.environ['COMMAND_LOG'],'a') as f: f.write(json.dumps({'argv':sys.argv[1:],'mode':os.environ.get('RTP_TORCH_PROVIDER')})+'\\n')\n"
                "from pathlib import Path\n"
                "for arg in sys.argv:\n"
                "    if arg.startswith('--build_event_json_file='):\n"
                "        Path(arg.split('=',1)[1]).write_text(json.dumps({'started': {'uuid': 'fixture'}})+'\\n'+json.dumps({'finished': {'exitCode': {'code': 0}, 'overallSuccess': True}})+'\\n')\n"
            )
            (binaries / "dig").write_text("#!/bin/sh\necho 127.0.0.1\n")
            for path in binaries.iterdir():
                path.chmod(0o755)
            log = root / "commands.jsonl"
            inherited_mode = os.environ.get("RTP_TORCH_PROVIDER")
            env = dict(
                os.environ,
                PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                BUILDS_DIR=str(root),
                AONE_CI_ENV=str(root / "ci.env"),
                AONE_CI_OUTPUTS=str(root),
                COMMAND_LOG=str(log),
                BAZEL_BUILD_TARGET="//rtp_llm:rtp_llm_lib",
                BAZEL_BUILD_TEST_TARGET="none",
                RTP_TORCH_PROVIDER="native",
                DRY_RUN="false",
            )
            result = subprocess.run(
                [
                    "bash",
                    str(INTERNAL_ROOT / "ci/basic_test.sh"),
                    "build",
                    "--config=cuda13",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(
                [call["argv"][:2] for call in calls],
                [["--batch", "sync"], ["--batch", "build"]],
            )
            self.assertTrue(all(call["mode"] == "hermetic" for call in calls))
            self.assertIn("//rtp_llm:rtp_llm_lib", calls[-1]["argv"])
            self.assertEqual(os.environ.get("RTP_TORCH_PROVIDER"), inherited_mode)
            self.assertEqual(env["RTP_TORCH_PROVIDER"], "native")
