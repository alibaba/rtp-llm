import os
import shlex
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import _build.platform as build_platform


def _declared_bazel_build_options():
    """Read this product rc's build stanzas; no imports or Bazel execution."""
    rc = Path(__file__).resolve().parents[2] / ".bazelrc"
    configs = {}
    for line in rc.read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0].startswith("build:"):
            configs.setdefault(tokens[0].split(":", 1)[1], []).extend(tokens[1:])
    return configs


def _expand_declared_config(configs, name, stack=()):
    """Expand only local --config references for the declared platform contract."""
    if name in stack:
        raise AssertionError("Cyclic local build config: " + name)
    result = []
    for token in configs[name]:
        if token.startswith("--config="):
            result.extend(
                _expand_declared_config(
                    configs, token.split("=", 1)[1], stack + (name,)
                )
            )
        else:
            result.append(token)
    return result


class BuildPlatformDetectionTest(TestCase):
    def tearDown(self):
        build_platform._cached_build_config = None

    def test_cuda13_detection_keeps_architecture_and_wheel_abi(self):
        for machine, config in (("x86_64", "cuda13"), ("aarch64", "cuda13_arm")):
            with self.subTest(machine=machine), patch.dict(os.environ, {}, clear=True):
                build_platform._cached_build_config = None
                with (
                    patch.object(
                        build_platform, "_detect_overlay_build_config", return_value=""
                    ),
                    patch.object(build_platform, "_detect_cuda", return_value=True),
                    patch.object(
                        build_platform,
                        "_get_cuda_version_from_json",
                        return_value="13.2.0",
                    ),
                    patch.object(
                        build_platform.platform, "machine", return_value=machine
                    ),
                ):
                    self.assertEqual(
                        build_platform.detect_build_config(verbose=False), config
                    )
                    self.assertEqual(build_platform.get_pip_extras(), config)
                    self.assertEqual(
                        build_platform.get_platform_config_versions()[config], "cu130"
                    )

    def test_cuda_version_comparison_uses_numeric_minor(self):
        self.assertEqual(
            build_platform._get_cuda_config_from_version("12.10.0"), "cuda12_9"
        )
        self.assertEqual(
            build_platform._get_cuda_config_from_version("12.9.20250531"),
            "cuda12_9",
        )
        self.assertEqual(
            build_platform._get_cuda_config_from_version("12.6.0"), "cuda12_6"
        )
        self.assertEqual(
            build_platform._get_cuda_config_from_version("not-a-version"),
            "cuda12_6",
        )

    def test_explicit_cpu_arm_configs_fail_fast_as_deprecated(self):
        for config in ("cpu", "arm"):
            build_platform._cached_build_config = None
            with self.subTest(config=config):
                with patch.dict(
                    os.environ, {"RTP_BAZEL_CONFIG": f"--config={config}"}, clear=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "deprecated"):
                        build_platform.detect_build_config(verbose=False)

    def test_deps_only_without_accelerator_reports_deprecated_cpu_arm_path(self):
        with patch.dict(os.environ, {"RTP_SKIP_BAZEL_BUILD": "1"}, clear=True):
            with patch.object(
                build_platform, "_detect_overlay_build_config", return_value=""
            ):
                with patch.object(build_platform, "_detect_cuda", return_value=False):
                    with patch.object(
                        build_platform, "_detect_rocm", return_value=False
                    ):
                        with self.assertRaisesRegex(RuntimeError, "CPU/ARM"):
                            build_platform.detect_build_config(verbose=False)

    def test_cuda129_detection_keeps_architecture_and_wheel_abi(self):
        for machine, config, extra in (
            ("x86_64", "cuda12_9", "cuda12_9"),
            ("aarch64", "cuda12_9_arm", "cuda12_arm"),
        ):
            with self.subTest(machine=machine), patch.dict(os.environ, {}, clear=True):
                build_platform._cached_build_config = None
                with (
                    patch.object(
                        build_platform, "_detect_overlay_build_config", return_value=""
                    ),
                    patch.object(build_platform, "_detect_cuda", return_value=True),
                    patch.object(
                        build_platform,
                        "_get_cuda_version_from_json",
                        return_value="12.9.20250531",
                    ),
                    patch.object(
                        build_platform.platform, "machine", return_value=machine
                    ),
                ):
                    self.assertEqual(
                        build_platform.detect_build_config(verbose=False), config
                    )
                    self.assertEqual(build_platform.get_pip_extras(), extra)
                    self.assertEqual(
                        build_platform.get_platform_config_versions()[config], "cu129"
                    )

    def test_builtin_packaging_configs_have_real_bazel_declarations(self):
        configs = _declared_bazel_build_options()
        for config in build_platform.PLATFORM_CONFIG_VERSIONS:
            with self.subTest(config=config):
                self.assertIn(config, configs)
                self.assertTrue(_expand_declared_config(configs, config))

    def test_cuda129_arm_bazel_config_preserves_sdk_and_native_selection(self):
        configs = _declared_bazel_build_options()
        options = _expand_declared_config(configs, "cuda12_9_arm")
        self.assertIn("--config=cuda12_9_base", configs["cuda12_9_arm"])
        self.assertNotIn("--config=cuda12_9", configs["cuda12_9_arm"])
        self.assertNotIn("--config=cuda13_base", configs["cuda12_9_arm"])
        self.assertNotIn("--config=sm10x", configs["cuda12_9_arm"])
        self.assertIn("--cpu=aarch64", options)
        self.assertIn("--host_cpu=aarch64", options)
        definitions = dict(
            token.removeprefix("--define=").split("=", 1)
            for token in options
            if token.startswith("--define=")
        )
        self.assertEqual(definitions["using_cuda12_arm"], "true")
        self.assertEqual(definitions["using_cuda12_x86"], "false")
        self.assertEqual(definitions["using_cuda12_9_x86"], "false")
        self.assertEqual(definitions["using_cuda13_x86"], "false")
        self.assertNotEqual(definitions.get("using_cuda13_arm"), "true")
        self.assertEqual(definitions["use_remote_kv_cache"], "false")
        self.assertIn("--repo_env=TF_CUDA_VERSION=12.9", options)
        self.assertIn("--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0", options)
        for flag in ("--action_env", "--host_action_env"):
            values = dict(
                options[i + 1].split("=", 1)
                for i, token in enumerate(options[:-1])
                if token == flag
            )
            self.assertEqual(values["TF_CUDA_VERSION"], "12.9")
            self.assertEqual(values["TF_CUDA_COMPUTE_CAPABILITIES"], "10.0")
        self.assertIn("--copt=-DUSING_CUDA12_9=1", options)
        self.assertIn("--copt=-D_GLIBCXX_USE_CXX11_ABI=1", options)
        self.assertNotIn("--copt=-DUSING_CUDA13=1", options)
