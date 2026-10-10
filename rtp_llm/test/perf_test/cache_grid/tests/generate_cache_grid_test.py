import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint as profile_fingerprint,
)
from rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid import (
    DEFAULT_SEED,
    build_grid,
    comma_separated_ints,
    generate_cache_lengths,
    generate_fixed_cache_lengths,
    generate_input_lengths,
)


class GenerateCacheGridTest(unittest.TestCase):
    def test_input_points_are_aligned_and_keep_strict_1m_boundary(self):
        values = generate_input_lengths(
            256, 1048575, 128, 32, "stratified", DEFAULT_SEED
        )
        self.assertIn(1048575, values)
        self.assertTrue(all(x % 128 == 0 for x in values if x != 1048575))

    def test_same_seed_is_reproducible(self):
        first = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        second = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 65408)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(x % 128 == 0 for x in first))

    def test_different_seed_preserves_boundaries(self):
        first = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        second = generate_cache_lengths(65536, 128, 16, 7, 104723)
        self.assertNotEqual(first, second)
        self.assertEqual((first[0], first[-1]), (second[0], second[-1]))

    def test_cache_alignment_governs_cache_dimension_only(self):
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=65536,
            alignment=128,
            cache_alignment=512,
            input_points=8,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=1,
            max_cases=1000,
            allow_large_grid=False,
            measure_runs=3,
        )
        plan = build_grid(args)
        for case in plan["cases"]:
            self.assertEqual(case["cache_len"] % 512, 0)
        self.assertEqual(plan["generator"]["cache_alignment"], 512)
        # The input dimension keeps its own alignment.
        inputs = {case["input_len"] for case in plan["cases"]}
        self.assertTrue(inputs)
        self.assertTrue(all(x % 128 == 0 for x in inputs))

    def test_cache_alignment_zero_is_rejected(self):
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=65536,
            alignment=128,
            cache_alignment=0,
            input_points=8,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=1,
            max_cases=1000,
            allow_large_grid=False,
            measure_runs=3,
        )
        with self.assertRaisesRegex(ValueError, "cache-alignment"):
            build_grid(args)

    def test_case_limit_rejects_oversized_plan(self):
        args = argparse.Namespace(
            min_input_len=256,
            max_input_len=8192,
            alignment=128,
            cache_alignment=128,
            input_points=32,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=16,
            cache_ratio_points=7,
            batch_size=1,
            max_cases=5,
            allow_large_grid=False,
            measure_runs=3,
        )
        with self.assertRaisesRegex(ValueError, "exceeding --max-cases"):
            build_grid(args)


class GenerateFixedCacheSweepTest(unittest.TestCase):
    def _args(self, **overrides):
        values = dict(
            grid_mode="fixed-cache-sweep",
            max_input_len=32768,
            cache_alignment=4096,
            fixed_cache_len=[4096, 8192],
            random_cache_count=None,
            min_cache_len=0,
            max_cache_len=None,
            compute_step=4096,
            min_compute_len=None,
            batch_size=1,
            measure_runs=3,
            seed=DEFAULT_SEED,
            max_cases=1000,
            allow_large_grid=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_explicit_cache_lengths_generate_nonduplicate_compute_sweeps(self):
        plan = build_grid(self._args())
        geometries = [(case["input_len"], case["cache_len"]) for case in plan["cases"]]
        self.assertEqual(len(geometries), len(set(geometries)))
        self.assertEqual(plan["summary"]["cache_count"], 2)
        first_slice = [case for case in plan["cases"] if case["cache_len"] == 4096]
        self.assertEqual(
            [case["input_len"] - case["cache_len"] for case in first_slice],
            list(range(4096, 32768 - 4096 + 1, 4096)),
        )
        self.assertTrue(all(case["input_len"] <= 32768 for case in plan["cases"]))

    def test_random_cache_lengths_are_unique_aligned_and_reproducible(self):
        args = self._args(
            fixed_cache_len=None,
            random_cache_count=4,
            min_cache_len=4096,
            max_cache_len=20480,
        )
        first = build_grid(args)
        second = build_grid(args)
        caches = first["generator"]["cache_sampling"]["values"]
        self.assertEqual(first["cases"], second["cases"])
        self.assertEqual(len(caches), len(set(caches)))
        self.assertTrue(all(value % 4096 == 0 for value in caches))

    def test_compute_steps_run_coarse_to_fine_without_duplicates(self):
        plan = build_grid(
            self._args(
                fixed_cache_len=[4096],
                compute_step=[16384, 8192, 4096],
            )
        )
        cases = plan["cases"]
        self.assertEqual(
            [case["input_len"] - case["cache_len"] for case in cases],
            [4096, 20480, 12288, 28672, 8192, 16384, 24576],
        )
        self.assertEqual(
            [case["refinement_level"] for case in cases],
            [0, 0, 1, 1, 2, 2, 2],
        )
        self.assertEqual(len(cases), len({case["input_len"] for case in cases}))
        self.assertEqual(
            plan["generator"]["compute_sampling"]["steps"],
            [16384, 8192, 4096],
        )

    def test_compute_steps_must_be_unique_and_coarse_to_fine(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_grid(self._args(compute_step=[8192, 8192]))
        with self.assertRaisesRegex(ValueError, "coarse-to-fine"):
            build_grid(self._args(compute_step=[4096, 8192]))

    def test_manual_and_random_cache_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_grid(self._args(random_cache_count=2))

    def test_duplicate_explicit_cache_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_grid(self._args(fixed_cache_len=[4096, 4096]))

    def test_random_count_cannot_exceed_aligned_candidates(self):
        with self.assertRaisesRegex(ValueError, "only 2 aligned values"):
            generate_fixed_cache_lengths(
                explicit=[],
                random_count=3,
                minimum=0,
                maximum=4096,
                alignment=4096,
                seed=DEFAULT_SEED,
            )

    def test_fixed_cache_cli_value_is_comma_separated(self):
        self.assertEqual(comma_separated_ints("4096, 8192,16384"), [4096, 8192, 16384])
        with self.assertRaises(argparse.ArgumentTypeError):
            comma_separated_ints("4096,,8192")


class GenerateCacheGridProfileTest(unittest.TestCase):
    COMMON_ARGS = [
        "--min-input-len",
        "2048",
        "--max-input-len",
        "65536",
        "--alignment",
        "128",
        "--input-points",
        "8",
        "--cache-points-per-input",
        "6",
        "--cache-ratio-points",
        "2",
    ]

    def _run_grid(self, extra_args, tmp):
        output = Path(tmp) / "grid.json"
        cmd = (
            [
                sys.executable,
                "-m",
                "rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid",
                "--output",
                str(output),
            ]
            + self.COMMON_ARGS
            + extra_args
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_without_profile_no_profile_keys_in_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run_grid(["--cache-alignment", "128"], tmp)
        self.assertNotIn("profile", payload)
        self.assertNotIn("profile_sha256", payload)

    def test_with_profile_embeds_profile_and_sha256(self):
        profile = {
            "schema_version": 1,
            "label": "test",
            "cache_grid": {"cache_alignment": 512},
        }
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            payload = self._run_grid(["--profile", str(profile_path)], tmp)
        self.assertIn("profile", payload)
        self.assertIn("profile_sha256", payload)
        self.assertEqual(payload["profile"], profile)
        self.assertEqual(payload["profile_sha256"], profile_fingerprint(profile))
        for case in payload["cases"]:
            self.assertEqual(case["cache_len"] % 512, 0)

    def test_cli_cache_alignment_wins_over_profile(self):
        profile = {
            "schema_version": 1,
            "cache_grid": {"cache_alignment": 512},
        }
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            payload = self._run_grid(
                ["--profile", str(profile_path), "--cache-alignment", "256"], tmp
            )
        for case in payload["cases"]:
            self.assertEqual(case["cache_len"] % 256, 0)
        self.assertEqual(payload["generator"]["cache_alignment"], 256)

    def test_explicit_alignment_matches_direct_build(self):
        """The CLI and direct API use the same explicit alignment."""
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run_grid(["--cache-alignment", "128"], tmp)
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=65536,
            alignment=128,
            cache_alignment=128,
            input_points=8,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=1,
            max_cases=20000,
            allow_large_grid=False,
            measure_runs=3,
        )
        expected = build_grid(args)
        self.assertEqual(payload["cases"], expected["cases"])
        self.assertEqual(payload["generator"], expected["generator"])
        self.assertEqual(payload["summary"], expected["summary"])

    def test_fixed_cache_sweep_cli_accepts_comma_separated_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run_grid(
                [
                    "--grid-mode",
                    "fixed-cache-sweep",
                    "--fixed-cache-len",
                    "4096,8192",
                    "--compute-step",
                    "16384,4096",
                    "--cache-alignment",
                    "4096",
                ],
                tmp,
            )
        self.assertEqual(payload["generator"]["cache_sampling"]["values"], [4096, 8192])
        self.assertEqual(
            payload["generator"]["compute_sampling"]["steps"], [16384, 4096]
        )


if __name__ == "__main__":
    unittest.main()
