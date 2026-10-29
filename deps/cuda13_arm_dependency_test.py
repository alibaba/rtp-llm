"""CPU-only contracts for CUDA 13 ARM dependency selection and wheel packaging.

These checks execute the Python-compatible selector macros with stub Bazel
APIs. They do not replace Bazel analysis or an aarch64 build/import smoke test.
Run directly from either the public checkout or the containing internal tree.
"""

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ARM = "@rtp_llm//:using_cuda13_arm"
X86 = "@rtp_llm//:using_cuda13_x86"


def load_macros(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    # load() binds Bazel symbols; provide test doubles instead.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "load"
        )
    ]
    calls = []
    namespace = {
        "native": SimpleNamespace(
            py_library=lambda **kwargs: calls.append(kwargs),
            alias=lambda **kwargs: calls.append(kwargs),
        ),
        "select": lambda branches: branches,
        "fail": lambda message: (_ for _ in ()).throw(ValueError(message)),
        "attr": SimpleNamespace(label_list=lambda **kwargs: None),
        "repository_rule": lambda **kwargs: lambda **args: calls.append(args),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("requirement_"):
            prefix = node.id
            namespace[prefix] = lambda name, prefix=prefix: prefix + ":" + name
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, calls


def normalized(text):
    return unquote(text).replace(" ", "").lower().replace("_", "-")


class Cuda13ArmDependencyTest(unittest.TestCase):
    def selectors(self):
        yield ROOT / "arch_config/arch_select.bzl"
        internal = ROOT.parent / "internal_source/bazel/arch_select.bzl"
        if internal.exists():
            yield internal

    def test_python_requirements_select_arm_lock_and_shared_layout(self):
        for path in self.selectors():
            macros, calls = load_macros(path)
            for name in (
                "torch",
                "av",
                "xgrammar",
                "flash-attn-4",
                "pynvvideocodec",
                "flashinfer-python",
                "flashinfer-jit-cache-sm100a",
            ):
                calls.clear()
                macros["requirement"]([name])
                deps = calls[-1]["deps"][ARM]
                self.assertIn("requirement_cuda13_arm:" + name, deps, (path, name))
                self.assertFalse(any("requirement_gpu_cuda13:" in x for x in deps))
                if name in ("flash-attn-4", "flashinfer-python"):
                    self.assertIn("@cuda13_arm_wheel_layout//:layout", deps)
                    self.assertIn(
                        "@cuda13_wheel_layout//:layout", calls[-1]["deps"][X86]
                    )
            for name in ("flash_attn", "flash-attn-3"):
                calls.clear()
                macros["requirement"]([name])
                self.assertEqual(calls[-1]["deps"][ARM], [])

    def test_cpp_and_cpu_decoder_dependencies(self):
        macros, calls = load_macros(ROOT / "arch_config/arch_select.bzl")
        self.assertTrue(
            all(
                "torch_2.11_py310_cuda-aarch64" in item
                for item in macros["torch_deps"]()[ARM]
            )
        )
        macros["flashinfer_deps"]()
        self.assertEqual(calls[-1]["actual"][ARM], "@flashinfer_cpp_cu13//:flashinfer")
        self.assertEqual(macros["platform_deps"]()[ARM], ["av==16.1.0"])
        self.assertIn(
            'name = "torch_2.11_py310_cuda-aarch64"',
            (ROOT / "deps/http.bzl").read_text(),
        )

    def test_layout_wheels_match_selected_architecture(self):
        macros, calls = load_macros(
            ROOT / "3rdparty/python_wheel_layout/repositories.bzl"
        )
        for arch, prefix, name in (
            ("x86_64", "@pip_gpu_cuda13_torch_", "cuda13_wheel_layout"),
            ("aarch64", "@pip_cuda13_arm_torch_", "cuda13_arm_wheel_layout"),
        ):
            macros["cuda13_wheel_layout"](architecture=arch)
            self.assertEqual(calls[-1]["name"], name)
            wheels = calls[-1]["wheels"]
            self.assertEqual(len(wheels), 9)
            for label in wheels:
                self.assertTrue(label.startswith(prefix), label)
                self.assertTrue(
                    label.endswith(arch + ".whl") or label.endswith("py3-none-any.whl"),
                    label,
                )
            self.assertTrue(any("sm100a" in x for x in wheels))
            self.assertTrue(any("sm103a" in x for x in wheels))
        with self.assertRaises(ValueError):
            macros["cuda13_wheel_layout"](architecture="unsupported")
        self.assertIn(
            'cuda13_wheel_layout(architecture = "aarch64")',
            (ROOT / "WORKSPACE").read_text(),
        )

    def test_release_pins_match_build_requirements_and_lock(self):
        for path in self.selectors():
            macros, _ = load_macros(path)
            deps = path.parents[1] / "deps"
            req = normalized((deps / "requirements_cuda13_arm.txt").read_text())
            lock = normalized((deps / "requirements_lock_cuda13_arm.txt").read_text())
            pins = macros["whl_deps"]()[ARM]
            for pin in pins:
                value = normalized(pin)
                # Source lists may use a bare wheel URL instead of name@URL.
                source = value.split("@", 1)[-1]
                self.assertIn(source, req, (path, pin))
                self.assertIn(source, lock, (path, pin))
                self.assertNotIn("x86-64.whl", value)
            for package in (
                "deep-gemm",
                "rtp-kernel",
                "flash-attn-4",
                "pynvvideocodec",
            ):
                self.assertTrue(
                    any(normalized(pin).startswith(package + "@") for pin in pins)
                )
            nv = next(pin for pin in pins if pin.startswith("pynvvideocodec@"))
            block = re.search(
                r"pynvvideocodec @ .*\n\s+--hash=sha256:([0-9a-f]{64})",
                (deps / "requirements_lock_cuda13_arm.txt").read_text(),
            )
            self.assertIsNotNone(block)
            self.assertEqual(
                block.group(1),
                "6c653a236bb61e98fd787601bfa8095150888a8e6cabf604306fa8ebbeb696dc",
            )
            self.assertIn("cp310-cp310-manylinux_2_28_aarch64.whl", nv)

    def test_arm_config_is_not_overridden_by_duplicate_block(self):
        rc = (ROOT / ".bazelrc").read_text()
        self.assertEqual(rc.count("build:cuda13_base --config=cuda12\n"), 1)
        self.assertEqual(rc.count("build:cuda13_arm --config=cuda13_base\n"), 1)
        for scope in ("action_env", "host_action_env"):
            values = re.findall(
                r"^build:cuda13_arm --"
                + scope
                + r' TF_CUDA_COMPUTE_CAPABILITIES="([^"]+)"$',
                rc,
                re.M,
            )
            self.assertEqual(values, ["10.0,10.3"])

    def test_internal_image_uses_cuda13_override_on_both_architectures(self):
        package = ROOT.parent / "internal_source/package"
        if not package.exists():
            self.skipTest("internal image packaging is not in the public checkout")
        docker = (package / "maga.Dockerfile").read_text()
        override = (package / "cuda13_uv_overrides.txt").read_text()
        self.assertIn("nvidia-cutlass-dsl[cu13]==4.7.1", override)
        block = next(
            block
            for block in docker.split("RUN ")
            if "cat /tmp/cuda13_uv_overrides.txt" in block
        )
        self.assertIn('[ "${EXPECTED_CUDA_MAJOR:-}" = "13" ]; then', block)
        self.assertNotIn("uname -m", block)
        self.assertIn("--overrides /tmp/xgrammar_uv_overrides.txt", block)


if __name__ == "__main__":
    unittest.main()
