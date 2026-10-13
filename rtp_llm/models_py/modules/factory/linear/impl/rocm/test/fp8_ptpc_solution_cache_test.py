import json
import tempfile
import unittest
from pathlib import Path
from unittest import SkipTest, mock

try:
    import aiter  # noqa: F401

    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


class RocmFp8PTPCSolutionCacheTest(unittest.TestCase):
    """Solution-cache metadata and fallback tests that do not require a GPU."""

    def setUp(self):
        if not AITER_AVAILABLE:
            raise SkipTest("aiter required for FP8 PTPC solution-cache tests")
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )

        RocmFp8PTPCLinearWithSwizzle._load_solution_cache.cache_clear()
        self.addCleanup(RocmFp8PTPCLinearWithSwizzle._load_solution_cache.cache_clear)

    @staticmethod
    def _payload(
        validated_versions=None,
        arch_prefix="gfx942",
        hip_prefix="7.2",
    ):
        if validated_versions is None:
            validated_versions = ["0.1.21.dev80+g987203ba5"]
        return {
            "format_version": 2,
            "target": {
                "arch_prefix": arch_prefix,
                "torch_hip_prefix": hip_prefix,
                "validated_aiter_versions": validated_versions,
            },
            "solutions": [
                {
                    "m": 512,
                    "k": 768,
                    "n": 2304,
                    "epilogue": "bias",
                    "solution_index": 271680,
                }
            ],
        }

    def _load_payload(self, linear_cls, payload, runtime):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "solutions.json"
            cache_path.write_text(json.dumps(payload))
            with (
                mock.patch.object(
                    linear_cls, "_solution_cache_path", return_value=cache_path
                ),
                mock.patch.object(linear_cls, "_runtime_target", return_value=runtime),
            ):
                linear_cls._load_solution_cache.cache_clear()
                return linear_cls._load_solution_cache()

    def test_runtime_target_device_probe_assertion_disables_cache(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )

        payload = self._payload()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "solutions.json"
            cache_path.write_text(json.dumps(payload))
            with (
                mock.patch.object(
                    RocmFp8PTPCLinearWithSwizzle,
                    "_solution_cache_path",
                    return_value=cache_path,
                ),
                mock.patch(
                    "torch.cuda.current_device", side_effect=AssertionError("bad")
                ),
                mock.patch(
                    "importlib.metadata.version",
                    return_value="0.1.21.dev80+g987203ba5.d20260825",
                ),
            ):
                runtime = RocmFp8PTPCLinearWithSwizzle._runtime_target()
                self.assertEqual(runtime.arch, "")
                self.assertEqual(
                    RocmFp8PTPCLinearWithSwizzle._load_solution_cache(), {}
                )

    def test_runtime_target_missing_aiter_metadata_disables_cache(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        payload = self._payload(arch_prefix="", hip_prefix="")
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "solutions.json"
            cache_path.write_text(json.dumps(payload))
            with (
                mock.patch.object(
                    linear_cls, "_solution_cache_path", return_value=cache_path
                ),
                mock.patch.object(
                    fp8_ptpc_linear.importlib_metadata,
                    "version",
                    side_effect=fp8_ptpc_linear.importlib_metadata.PackageNotFoundError,
                ),
            ):
                runtime = linear_cls._runtime_target()
                self.assertEqual(runtime.aiter_version, "")
                self.assertEqual(linear_cls._load_solution_cache(), {})

    def test_stale_aiter_cache_warns_and_falls_back(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        payload = self._payload(validated_versions=["0.1.21.dev80+g987203ba5"])
        runtime = fp8_ptpc_linear._RuntimeTarget(
            "gfx942:sramecc+:xnack-", "7.2.0", "0.1.22.dev1+gabcdef01"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "solutions.json"
            cache_path.write_text(json.dumps(payload))
            with (
                mock.patch.object(
                    linear_cls, "_solution_cache_path", return_value=cache_path
                ),
                mock.patch.object(linear_cls, "_runtime_target", return_value=runtime),
                self.assertLogs(fp8_ptpc_linear.logger.name, level="WARNING") as logs,
            ):
                self.assertEqual(linear_cls._load_solution_cache(), {})

        rendered_logs = "\n".join(logs.output)
        self.assertIn("0.1.21.dev80+g987203ba5", rendered_logs)
        self.assertIn("0.1.22.dev1+gabcdef01", rendered_logs)
        self.assertIn("default heuristics", rendered_logs)

    def test_matching_target_loads_complete_cache(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        payload = self._payload()
        runtime = fp8_ptpc_linear._RuntimeTarget(
            "gfx942:sramecc+:xnack-",
            "7.2.0",
            "0.1.21.dev80+g987203ba5.d20260825",
        )
        self.assertEqual(
            self._load_payload(linear_cls, payload, runtime),
            linear_cls._parse_solution_cache(payload),
        )

    def test_platform_mismatch_logs_info_and_falls_back(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        runtime = fp8_ptpc_linear._RuntimeTarget(
            "gfx950", "7.2.0", "0.1.21.dev80+g987203ba5.d20260825"
        )
        with self.assertLogs(fp8_ptpc_linear.logger.name, level="INFO") as logs:
            self.assertEqual(
                self._load_payload(linear_cls, self._payload(), runtime), {}
            )
        self.assertIn("does not target this platform", "\n".join(logs.output))

    def test_invalid_version_list_is_rejected(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        runtime = fp8_ptpc_linear._RuntimeTarget("gfx942", "7.2.0", "anything")
        for invalid_versions in ("0.1.21.dev80+g987203ba5", [""], [None]):
            with self.subTest(validated_versions=invalid_versions):
                with self.assertLogs(
                    fp8_ptpc_linear.logger.name, level="WARNING"
                ) as logs:
                    self.assertEqual(
                        self._load_payload(
                            linear_cls,
                            self._payload(validated_versions=invalid_versions),
                            runtime,
                        ),
                        {},
                    )
                self.assertIn("invalid target metadata", "\n".join(logs.output))

    def test_unvalidated_cache_is_explicitly_disabled(self):
        from rtp_llm.models_py.modules.factory.linear.impl.rocm import fp8_ptpc_linear

        linear_cls = fp8_ptpc_linear.RocmFp8PTPCLinearWithSwizzle
        payload = self._payload(validated_versions=[])
        runtime = fp8_ptpc_linear._RuntimeTarget("gfx942", "7.2.0", "anything")
        with self.assertLogs(fp8_ptpc_linear.logger.name, level="WARNING") as logs:
            self.assertEqual(self._load_payload(linear_cls, payload, runtime), {})
        self.assertIn("intentionally disabled", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
