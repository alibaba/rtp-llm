"""Checkout-only CPU checks; actual Bazel resolution must be verified separately.

This AST/mock helper deliberately has no Bazel target and is not a hermetic
runfiles test. Run it from the reviewed public checkout; it installs no packages.
"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

if os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR"):
    raise RuntimeError("This is a checkout-only helper, not a Bazel runfiles test")
ROOT = Path(__file__).absolute().parent.parent


def load_bzl(relative, **overrides):
    source = (ROOT / relative).read_text()
    scope = {
        "load": lambda *args, **kwargs: None,
        "select": lambda values: values,
        "json": SimpleNamespace(encode=json.dumps),
        "native": SimpleNamespace(py_library=lambda **kwargs: None),
    }
    for node in ast.parse(source).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "load":
                repo = call.args[0].value.split("//")[0]
                for alias in call.keywords:
                    scope[alias.arg] = (
                        lambda name, repo=repo: repo + ":" + name.replace("_", "-")
                    )
    scope.update(overrides)
    exec(compile(source, str(ROOT / relative), "exec"), scope)
    return scope


class DependencySelectionTest(unittest.TestCase):
    def test_repository_flag_tracks_boolean_environment_and_rejects_invalid(self):
        def fail(message):
            raise ValueError(message)

        scope = load_bzl(
            "deps/sm90_dependency_config/repositories.bzl",
            repository_rule=lambda **kwargs: kwargs,
            fail=fail,
        )
        self.assertEqual(
            scope["sm90_dependency_config"]["environ"], ["RTP_LLM_SM90_DEPS"]
        )
        for value, expected in [
            (None, False),
            ("", False),
            ("0", False),
            ("false", False),
            ("FALSE", False),
            ("1", True),
            ("true", True),
            ("TRUE", True),
        ]:
            with self.subTest(value=value):
                files = {}
                environ = {} if value is None else {"RTP_LLM_SM90_DEPS": value}
                context = SimpleNamespace(
                    os=SimpleNamespace(environ=environ), file=files.__setitem__
                )
                scope["_sm90_dependency_config_impl"](context)
                self.assertEqual(files["config.bzl"], "SM90_DEPS = %s\n" % expected)
        context = SimpleNamespace(
            os=SimpleNamespace(environ={"RTP_LLM_SM90_DEPS": "yes"}),
            file=lambda *args: None,
        )
        with self.assertRaisesRegex(ValueError, "must be"):
            scope["_sm90_dependency_config_impl"](context)

    def test_both_pip_repositories_have_distinct_fixed_locks(self):
        calls = []
        scope = load_bzl(
            "deps/pip.bzl",
            pip_parse=lambda **kwargs: calls.append(kwargs),
            package_annotation=lambda **kwargs: kwargs,
        )
        scope["pip_deps"]()
        repositories = {call["name"]: call for call in calls}
        self.assertEqual(
            repositories["pip_gpu_cuda13_torch"]["requirements_lock"],
            "@rtp_deps//:requirements_lock_torch_gpu_cuda13.txt",
        )
        self.assertEqual(
            repositories["pip_gpu_cuda13_sm90_torch"]["requirements_lock"],
            "@rtp_deps//:requirements_lock_torch_gpu_cuda13_sm90.txt",
        )
        self.assertEqual(
            repositories["pip_cuda13_arm_torch"]["requirements_lock"],
            "@rtp_deps//:requirements_lock_cuda13_arm.txt",
        )

    def test_x86_repository_and_metadata_select_together_while_arm_stays_upstream(self):
        metadata = []
        for enabled in (False, True):
            scope = load_bzl("arch_config/arch_select.bzl", SM90_DEPS=enabled)
            self.assertEqual(
                scope["requirement_gpu_cuda13"]("deep-gemm"),
                ("@pip_gpu_cuda13_sm90_torch" if enabled else "@pip_gpu_cuda13_torch")
                + ":deep-gemm",
            )
            selected = scope["whl_deps"]()
            x86 = selected["@rtp_llm//:using_cuda13_x86"]
            self.assertIn(
                "nvidia-cutlass-dsl==" + ("4.8.1" if enabled else "4.7.1"), x86
            )
            self.assertEqual(any("bbbb3c3" in value for value in x86), enabled)
            self.assertEqual(any("gc_moe@" in value for value in x86), enabled)
            metadata.append(selected)
        self.assertEqual(
            metadata[0]["@rtp_llm//:using_cuda13_arm"],
            metadata[1]["@rtp_llm//:using_cuda13_arm"],
        )
        default = (ROOT / "deps/requirements_lock_torch_gpu_cuda13.txt").read_text()
        sm90 = (ROOT / "deps/requirements_lock_torch_gpu_cuda13_sm90.txt").read_text()
        self.assertIn("cuda-python==13.0.3", default)
        self.assertIn("nvidia-cutlass-dsl==4.7.1", default)
        self.assertNotIn("bbbb3c3", default)
        self.assertIn("cuda-python==13.2.0", sm90)
        self.assertIn("nvidia-cutlass-dsl==4.8.1", sm90)
        self.assertIn("bbbb3c3", sm90)

    def test_default_build_does_not_resolve_gc_moe_or_sm90_sidecar(self):
        source = (ROOT / "rtp_llm/models_py/BUILD").read_text()
        for enabled in (False, True):
            packages, rules = [], {}

            def sm90_requirement(name):
                packages.append(name)
                return "@pip_gpu_cuda13_sm90_torch:" + name

            scope = dict(
                SM90_DEPS=enabled,
                load=lambda *a, **k: None,
                exports_files=lambda *a: None,
                requirement=lambda *a: None,
                sm90_requirement=sm90_requirement,
                glob=lambda *a, **k: [],
                select=lambda d: d.get(
                    "@//:using_cuda13_x86", d.get("//conditions:default")
                ),
            )
            for name in ("py_library", "filegroup", "py_test", "py_binary"):
                scope[name] = lambda **kwargs: rules.update({kwargs["name"]: kwargs})
            exec(compile(source, "models_py/BUILD", "exec"), scope)
            self.assertEqual(packages, ["gc-moe"] if enabled else [])
            deps = rules["modules"]["deps"]
            self.assertEqual("@flash_mla_sm90_sidecar//:pkg" in deps, enabled)

    def test_native_nvidia_libraries_follow_the_python_repository(self):
        source = (ROOT / "BUILD.pytorch").read_text()
        for enabled in (False, True):
            rules = {}
            scope = dict(
                SM90_DEPS=enabled,
                load=lambda *a, **k: None,
                glob=lambda *a, **k: [],
                select=lambda d: d.get(
                    "@//:using_cuda13_x86", d.get("//conditions:default", [])
                ),
                config_setting=lambda **k: None,
                filegroup=lambda **k: None,
                cc_library=lambda **k: rules.update({k["name"]: k}),
            )
            exec(compile(source, "BUILD.pytorch", "exec"), scope)
            wheel_libraries = [
                label for label in rules["torch"]["srcs"] if label.startswith("@pip_")
            ]
            self.assertTrue(wheel_libraries)
            prefix = (
                "@pip_gpu_cuda13_sm90_torch_" if enabled else "@pip_gpu_cuda13_torch_"
            )
            self.assertTrue(all(label.startswith(prefix) for label in wheel_libraries))

    def test_default_wheel_excludes_sidecar_and_preserves_path_policy(self):
        source = ast.parse((ROOT / "rtp_llm/BUILD").read_text())
        assignment = next(
            node
            for node in source.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "sm90_backend_wheel_deps"
                for t in node.targets
            )
        )
        wheel = next(
            node.value
            for node in source.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "py_wheel"
            and any(
                k.arg == "name"
                and isinstance(k.value, ast.Constant)
                and k.value.value == "rtp_llm_whl"
                for k in node.value.keywords
            )
        )
        strip = next(k.value for k in wheel.keywords if k.arg == "strip_path_prefixes")
        for enabled in (False, True):
            scope = dict(SM90_DEPS=enabled, select=lambda d: d["@//:using_cuda13_x86"])
            exec(
                compile(
                    ast.Module(body=[assignment], type_ignores=[]), "wheel", "exec"
                ),
                scope,
            )
            self.assertEqual(
                scope["sm90_backend_wheel_deps"],
                ["@flash_mla_sm90_sidecar//:wheel_package"] if enabled else [],
            )
            self.assertEqual(
                eval(compile(ast.Expression(strip), "wheel", "eval"), scope),
                ["site-packages/"] if enabled else [],
            )


if __name__ == "__main__":
    unittest.main()
