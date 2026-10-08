"""CPU regression against the measured reference, including launcher/worker parity."""

import contextlib
import hashlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import qwen35_e1pd4_perf_smoke as preset
import qwen35_epd_fusion_4gpu_smoke as smoke

REFERENCE = json.loads(
    Path(__file__).with_name("qwen35_e1pd4_perf_reference.json").read_text()
)


def parse(*options):
    with mock.patch.object(smoke, "run_launcher", side_effect=lambda a: a):
        return preset.main(["--output=/tmp/e1pd4-preset-test", *options])


class PerformancePresetTest(unittest.TestCase):
    def test_reference_launcher_options(self):
        # Read the old command as independent evidence, not newly generated defaults.
        command = REFERENCE["command"][2:]
        with mock.patch.object(smoke, "run_launcher", side_effect=lambda a: a):
            old = smoke.main([x for x in command if x != "--execute"])
        new = parse()
        for key in preset.DEFAULTS:
            self.assertEqual(getattr(new, key), getattr(old, key), key)
        self.assertEqual(new.gpus, old.gpus)
        self.assertEqual(new.encoder_gpus, old.encoder_gpus)
        old_env = [x for x in old.bazel_option if x.startswith("--test_env=")]
        self.assertEqual(len(old_env), 6)
        # Portable package/cache paths may change; all other historical envs match.
        for value in old_env:
            if not any(k in value for k in ("PATH=", "CACHE_DIR=")):
                self.assertIn(value, new.bazel_option)
        self.assertIn("--jobs=32", new.bazel_option)

    def test_reference_services_match_all_env_and_args(self):
        a = parse("--trace-run-id=" + preset.REFERENCE_RUN)
        env, args = smoke.server_config(
            a.gpus,
            a.profile,
            a.moe_strategy,
            a.decode_graph,
            a.fp8_kv_cache,
            a.native_fp8_attn,
            a.seq_size_per_block,
        )
        env, args = smoke.throughput_config(
            env,
            args,
            a.scheduler_policy,
            a.decode_prefill_ratio,
            a.schedule_trace,
            a.trace_run_id,
            a.coord_mode,
        )
        env, args = smoke.capacity_config(
            env,
            args,
            a.rank_concurrency,
            a.kv_cache_mb,
            a.graph_batches,
            a.decode_graph,
            a.runtime_reserve_mb,
        )
        configs = smoke.dp2_service_configs(
            a.gpus,
            a.encoder_gpus,
            env,
            args,
            dict(encoder_0=13328, encoder_1=14000, fusion=12184),
            9,
            a.encoder_profile,
            a.encoder_rdma_pool_bytes,
        )
        for actual, expected in zip(configs, REFERENCE["configs"]):
            actual["env"].pop("CUDA_COREDUMP_FILE", None)
            self.assertEqual(actual, expected)
        self.assertEqual(len(configs), 2)

    def test_bazel_round_trip_keeps_preset_and_resource_paths(self):
        a = parse(
            "--dg-jit-cache=/tmp/jit",
            "--pycparser-path=/tmp/pycparser",
            "--gpus=0,1,2,3",
            "--encoder-gpus=7",
        )
        command = smoke.bazel_command(a)
        self.assertIn(preset.TARGET, command)
        self.assertFalse(any("DEEP_GEMM_DIAGNOSTIC_PATH" in x for x in command))
        self.assertFalse(any("--deep-gemm-path" in x for x in command))
        for flag in (
            "--test_env=GPU_COUNT=5",
            "--test_env=WORLD_SIZE=5",
            "--test_env=CUDA_VISIBLE_DEVICES=7,0,1,2,3",
            "--run_under=//rtp_llm/test/utils:gpu_lock",
            "--test_timeout=21600",
        ):
            self.assertIn(flag, command)
        options = [
            x[len("--test_arg=") :] for x in command if x.startswith("--test_arg=")
        ]
        with mock.patch.object(preset, "validate_runtime"), mock.patch.object(
            smoke, "run_worker", side_effect=lambda a: a
        ):
            worker = preset.main(["--worker", *options])
        for field in (
            *preset.DEFAULTS,
            "trace_run_id",
            "dg_jit_cache",
            "pycparser_path",
            "gpus",
            "encoder_gpus",
        ):
            self.assertEqual(getattr(worker, field), getattr(a, field), field)

    def test_reject_performance_drift_and_extra_bazel_options(self):
        for option in (
            "--steady-concurrency=256",
            "--coord-mode=off",
            "--encoder-profile=baseline",
            "--encoder-gpus=0,1",
            "--bazel-option=--test_env=DG_MEGA_MOE_FP8_GRID_SYNC_DIAG=1",
            "--fixed-output-tokens=512",
            "--deep-gemm-path=/tmp/legacy-package",
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    parse(option)

    def test_unique_trace_and_dry_run_does_not_access_dependencies(self):
        self.assertNotEqual(parse().trace_run_id, parse().trace_run_id)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new-run"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), mock.patch.object(
                preset,
                "validate_runtime",
                side_effect=AssertionError("dry run accessed runtime"),
            ):
                self.assertEqual(preset.main([f"--output={out}"]), 0)
            self.assertFalse(out.exists())
            self.assertIn(preset.TARGET, shlex.split(stdout.getvalue()))

    def test_launcher_does_not_require_local_deep_gemm(self):
        with tempfile.TemporaryDirectory() as directory:
            a = parse(f"--dg-jit-cache={directory}", f"--pycparser-path={directory}")
            with mock.patch.object(
                preset.importlib.util,
                "find_spec",
                side_effect=AssertionError("launcher accessed worker package"),
            ):
                preset.validate_runtime(a)

    def test_missing_runtime_assets_fail_before_launcher(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            smoke, "run_launcher"
        ) as launcher:
            for missing in ("dg-jit-cache", "pycparser-path"):
                options = {
                    "dg-jit-cache": directory,
                    "pycparser-path": directory,
                }
                options[missing] = directory + "/missing"
                with self.subTest(missing=missing), self.assertRaises(ValueError):
                    preset.main(
                        ["--execute", "--output=/tmp/unused"]
                        + [f"--{key}={value}" for key, value in options.items()]
                    )
            launcher.assert_not_called()

    def test_missing_worker_package(self):
        with tempfile.TemporaryDirectory() as directory:
            a = parse(f"--dg-jit-cache={directory}", f"--pycparser-path={directory}")
            a.worker = True
            for spec in (None, SimpleNamespace(origin=None)):
                with mock.patch.object(
                    preset.importlib.util, "find_spec", return_value=spec
                ), self.assertRaisesRegex(ValueError, "missing from worker runtime"):
                    preset.validate_runtime(a)

    def test_published_wheel_identity_for_both_architectures(self):
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = root / "deep_gemm"
                package.mkdir()
                (package / "__init__.py").write_text("")
                header = package / "fixed.cuh"
                header.write_bytes(b"epoch fix")
                binary = package / f"_C.cpython-310-{arch}-linux-gnu.so"
                binary.write_bytes(b"published binary")
                a = parse(f"--dg-jit-cache={root}", f"--pycparser-path={root}")
                a.worker = True
                spec = SimpleNamespace(origin=str(package / "__init__.py"))
                hashes = {"fixed.cuh": hashlib.sha256(header.read_bytes()).hexdigest()}
                binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
                with mock.patch.dict(
                    preset.DEEP_GEMM_SHA256, hashes, clear=True
                ), mock.patch.dict(
                    preset.DEEP_GEMM_BINARY_SHA256, {arch: binary_hash}, clear=True
                ), mock.patch.object(
                    preset.platform, "machine", return_value=arch
                ), mock.patch.object(
                    preset.importlib.util, "find_spec", return_value=spec
                ), mock.patch.dict(
                    os.environ
                ):
                    # No dist-info or local override is required after gpu_lock staging.
                    preset.validate_runtime(a)
                    self.assertEqual(os.environ["MM_RDMA_READ_TIMEOUT_MS"], "30000")
                    self.assertEqual(
                        a.preset_metadata["deep_gemm_package"], str(package.resolve())
                    )
                    for path in (header, binary):
                        original = path.read_bytes()
                        path.write_bytes(b"old or modified package")
                        with self.assertRaisesRegex(ValueError, "differs from"):
                            preset.validate_runtime(a)
                        path.unlink()
                        with self.assertRaisesRegex(ValueError, "differs from"):
                            preset.validate_runtime(a)
                        path.write_bytes(original)
                    spec.origin = str(root / "shadowed/deep_gemm/__init__.py")
                    with self.assertRaisesRegex(ValueError, "differs from"):
                        preset.validate_runtime(a)


if __name__ == "__main__":
    unittest.main()
