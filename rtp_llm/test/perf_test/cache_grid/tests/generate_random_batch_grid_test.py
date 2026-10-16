import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid import (
    build_plans,
    generate_grid,
    main,
    parse_args,
    round_up,
)


class RandomBatchGridTest(unittest.TestCase):
    def setUp(self):
        stdout = patch("sys.stdout", new_callable=io.StringIO)
        stdout.start()
        self.addCleanup(stdout.stop)

    def args(self, *extra):
        return parse_args(["--output", "/tmp/random_batch_test.json", *extra])

    def test_reproducible_constrained_batches(self):
        args = self.args("--num-cases", "300", "--kv-budget-tokens", "400000")
        grid = generate_grid(args)
        self.assertEqual(grid, generate_grid(args))
        for case in grid["cases"]:
            groups = case["request_groups"]
            self.assertEqual(sum(g["count"] for g in groups), case["batch_size"])
            self.assertLessEqual(
                sum(g["input_len"] for g in groups), args.max_batch_tokens
            )
            peak = 0
            for g in groups:
                self.assertLessEqual(g["input_len"], 262144)
                self.assertEqual(g["input_len"] % 256, 0)
                self.assertEqual(g["cache_len"] % 4096, 0)
                self.assertLess(g["cache_len"], g["input_len"])
                if g["cache_len"]:
                    self.assertGreaterEqual(g["input_len"] - g["cache_len"], 4096)
                peak += round_up(g["input_len"], 4096) + (4096 if g["cache_len"] else 0)
            self.assertEqual(peak, case["estimated_peak_kv_tokens"])
            self.assertLessEqual(peak, 400000)

    def test_small_unaligned_minimum_with_tight_kv_budget(self):
        grid = generate_grid(
            self.args(
                "--num-cases",
                "1",
                "--batch-sizes",
                "2",
                "--min-input-tokens",
                "257",
                "--max-input-tokens",
                "512",
                "--kv-budget-tokens",
                "8192",
            )
        )
        self.assertEqual(grid["cases"][0]["estimated_peak_kv_tokens"], 8192)

    def test_impossible_and_oversized_inputs_rejected(self):
        for extra in [
            ("--max-input-tokens", "0"),
            ("--max-batch-tokens", "100"),
            ("--kv-budget-tokens", "100"),
            ("--commit-tail-tokens", "1"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                generate_grid(self.args(*extra))

    def test_exhausted_shape_space_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct batches"):
            generate_grid(
                self.args(
                    "--num-cases",
                    "2",
                    "--batch-sizes",
                    "1",
                    "--min-input-tokens",
                    "256",
                    "--max-input-tokens",
                    "256",
                )
            )

    def test_directory_plans_do_not_divide_length_by_batch(self):
        args = parse_args(
            [
                "--output-dir",
                "/tmp/batch_plans",
                "--num-cases",
                "5",
                "--batch-sizes",
                "1",
                "8",
                "31",
            ]
        )
        plans = build_plans(args)
        self.assertEqual(plans, build_plans(args))
        for _, grid in plans:
            batches = {case["batch_size"] for case in grid["cases"]}
            self.assertEqual(len(batches), 1)
            batch = batches.pop()
            limit = grid["summary"]["max_request_tokens"]
            self.assertLessEqual(limit, 262144)
            self.assertEqual(limit, 262144)
            self.assertTrue(
                all(c["input_tokens_sum"] <= 1048576 for c in grid["cases"])
            )

    def test_batch_limit_override(self):
        args = parse_args(
            [
                "--output-dir",
                "/tmp/batch_plans",
                "--num-cases",
                "5",
                "--batch-sizes",
                "31",
                "--batch-input-limit",
                "31:65536",
            ]
        )
        self.assertEqual(
            build_plans(args)[0][1]["summary"]["max_request_tokens"], 65536
        )
        args.batch_input_limit = ["31:262145"]
        with self.assertRaises(ValueError):
            build_plans(args)

    def test_custom_commit_tail_alignment(self):
        grid = generate_grid(self.args("--commit-tail-tokens", "8192"))
        for case in grid["cases"]:
            for group in case["request_groups"]:
                self.assertEqual(group["cache_len"] % 8192, 0)

    def test_long_requests_are_sampled_in_both_output_modes(self):
        options = [
            "--batch-sizes",
            "8",
            "--num-cases",
            "100",
            "--min-input-tokens",
            "2048",
            "--max-input-tokens",
            "49152",
            "--max-batch-tokens",
            "65536",
            "--cache-alignment",
            "1024",
            "--commit-tail-tokens",
            "1024",
        ]
        for workspace in ([], ["--workspace-tokens", "65536"]):
            for output in ("--output", "--output-dir"):
                args = parse_args([output, "/tmp/not-written", *options, *workspace])
                grid = build_plans(args)[0][1]
                lengths = [
                    g["input_len"] for c in grid["cases"] for g in c["request_groups"]
                ]
                self.assertGreater(max(lengths), 32768)
                self.assertTrue(
                    all(2048 <= n <= 49152 and n % 256 == 0 for n in lengths)
                )
                for c in grid["cases"]:
                    self.assertLessEqual(c["input_tokens_sum"], 65536)
                    if workspace:
                        self.assertLessEqual(
                            sum(
                                round_up(g["input_len"] + 1, 1024)
                                for g in c["request_groups"]
                            ),
                            65536,
                        )

    def test_fixed_workspace_rejects_unusable_probe_or_minimum(self):
        for extra in (
            ("--workspace-tokens", "0"),
            ("--workspace-tokens", "8192"),
            ("--workspace-tokens", "16384", "--batch-sizes", "4"),
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                generate_grid(self.args(*extra))

    def test_fixed_workspace_and_kv_budget_are_independent(self):
        args = self.args("--workspace-tokens", "131072", "--kv-budget-tokens", "100000")
        for c in generate_grid(args)["cases"]:
            self.assertLessEqual(c["estimated_peak_kv_tokens"], 100000)
            self.assertLessEqual(
                sum(round_up(g["input_len"] + 1, 4096) for g in c["request_groups"]),
                131072,
            )

    def test_generic_grid_accepts_other_geometry_and_larger_budget(self):
        grid = generate_grid(
            self.args(
                "--num-cases",
                "10",
                "--batch-sizes",
                "2",
                "--min-input-tokens",
                "524288",
                "--max-input-tokens",
                "1048576",
                "--max-batch-tokens",
                "4194304",
                "--cache-alignment",
                "64",
                "--input-alignment",
                "64",
                "--commit-tail-tokens",
                "64",
            )
        )
        self.assertIsNone(grid["generator"]["workspace_tokens"])
        for case in grid["cases"]:
            for group in case["request_groups"]:
                self.assertGreater(group["input_len"], 262144)
                self.assertEqual(group["cache_len"] % 64, 0)

    def test_directory_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "grids"
            self.assertIsNone(
                main(
                    [
                        "--output-dir",
                        str(output),
                        "--num-cases",
                        "3",
                        "--batch-sizes",
                        "2",
                        "16",
                    ]
                )
            )
            files = sorted(output.glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertEqual(
                [
                    {
                        case["batch_size"]
                        for case in json.loads(path.read_text())["cases"]
                    }
                    for path in files
                ],
                [{2}, {16}],
            )


if __name__ == "__main__":
    unittest.main()
