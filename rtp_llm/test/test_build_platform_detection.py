import os
from unittest import TestCase
from unittest.mock import patch

import _build.platform as build_platform


class BuildPlatformDetectionTest(TestCase):
    def tearDown(self):
        build_platform._cached_build_config = None

    def test_cuda13_detection_keeps_architecture_and_wheel_abi(self):
        for machine, config in (("x86_64", "cuda13"), ("aarch64", "cuda13_arm")):
            with self.subTest(machine=machine), patch.dict(os.environ, {}, clear=True):
                build_platform._cached_build_config = None
                with patch.object(
                    build_platform, "_detect_overlay_build_config", return_value=""
                ), patch.object(
                    build_platform, "_detect_cuda", return_value=True
                ), patch.object(
                    build_platform, "_get_cuda_version_from_json", return_value="13.2.0"
                ), patch.object(
                    build_platform.platform, "machine", return_value=machine
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
