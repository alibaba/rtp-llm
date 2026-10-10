import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import _configure_cache_batch_limits
from rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid import (
    generate_grid,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    grid_token_budget,
    grid_workspace_tokens,
    validate_token_budget,
    workspace_capacity,
)
from rtp_llm.test.perf_test.dataset import extract_arg


class WorkspaceBudgetTest(unittest.TestCase):
    def case(self, batch=4, length=8192):
        return dict(case_id=1, batch_size=batch, input_len=length, cache_len=4096)

    def validate(self, cases, **overrides):
        options = dict(workspace_tokens=65536, commit_tail=4096, block=4096)
        options.update(overrides)
        return validate_token_budget(cases, **options)

    def test_long_request_with_short_peers_uses_packed_sum(self):
        case = dict(
            batch_size=4,
            request_groups=[
                dict(count=1, input_len=40000, cache_len=0),
                dict(count=3, input_len=4096, cache_len=0),
            ],
        )
        stats = self.validate([case])
        self.assertEqual(stats[1]["workspace_tokens"], 65536)
        self.assertGreater(4 * 40000, 65536)

    def test_per_request_padding_and_output_reserve(self):
        with self.assertRaisesRegex(ValueError, "padded token sum"):
            self.validate([self.case(length=16384)])
        self.assertEqual(self.validate([self.case()])[1]["workspace_tokens"], 49152)

    def test_seed_counts_and_probe(self):
        case = self.case()
        self.assertEqual(self.validate([case])[-1]["requests"], 4)
        case["prefix_policy"] = "shared_by_group"
        self.assertEqual(self.validate([case])[-1]["requests"], 1)
        with self.assertRaisesRegex(ValueError, "seed tail"):
            self.validate([self.case(length=4097)])
        with self.assertRaisesRegex(ValueError, "probe"):
            self.validate([], workspace_tokens=8192)
        with self.assertRaisesRegex(ValueError, "probe"):
            self.validate([], commit_tail=65536)

    def test_generated_grid_respects_explicit_workspace(self):
        args = parse_args(
            [
                "--output",
                "/tmp/not-written.json",
                "--workspace-tokens",
                "131072",
                "--cache-alignment",
                "1024",
                "--commit-tail-tokens",
                "1024",
                "--batch-sizes",
                "2",
                "4",
                "8",
                "--num-cases",
                "200",
            ]
        )
        grid = generate_grid(args)
        self.assertEqual(grid, generate_grid(args))
        self.assertEqual(grid_workspace_tokens(grid), 131072)
        stats = self.validate(
            grid["cases"], workspace_tokens=131072, block=1024, commit_tail=1024
        )
        self.assertTrue(all(s["workspace_tokens"] <= 131072 for s in stats))
        self.assertTrue(
            any(
                max(g["input_len"] for g in c["request_groups"])
                > 131072 // c["batch_size"]
                for c in grid["cases"]
            )
        )

    def test_backend_accepts_tp1_tp4_tp8_and_keeps_fixed_capacity(self):
        for tp in (1, 4, 8):
            for method in ("DISABLED", "ALL_GATHER"):
                with self.subTest(tp=tp, method=method):
                    args = argparse.Namespace(
                        cache_workspace_tokens=65536,
                        dp_size=1,
                        concurrency_limit=1,
                        max_seq_len=8192,
                        cache_shared_seed=False,
                        cache_commit_tail_tokens=1024,
                        decode_test_length=1,
                    )
                    remaining = [
                        f"--tp_size={tp}",
                        f"--cp_rotate_method={method}",
                        "--max_context_batch_size=32",
                    ]
                    _configure_cache_batch_limits(
                        args, remaining, [self.case()], cache_alignment=1024
                    )
                    self.assertEqual(args.max_seq_len, 65536)
                    self.assertEqual(
                        extract_arg(remaining, "max_context_batch_size"), "32"
                    )
                    self.assertEqual(
                        extract_arg(remaining, "max_batch_tokens_size"), "65536"
                    )
                    self.assertEqual(args.concurrency_limit, 1)
                    with self.assertRaisesRegex(ValueError, "padded token sum"):
                        _configure_cache_batch_limits(
                            args,
                            remaining,
                            [self.case(length=16384)],
                            cache_alignment=1024,
                        )

    def test_metadata_requires_explicit_positive_budget(self):
        self.assertIsNone(grid_workspace_tokens({}))
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            grid_workspace_tokens({"generator": {"workspace_policy": "unknown"}})
        for value in (0, -1, True, 1.5, "65536"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                grid_workspace_tokens({"generator": {"workspace_tokens": value}})
        self.assertEqual(
            grid_workspace_tokens({"generator": {"workspace_tokens": 2097152}}), 2097152
        )
        self.assertEqual(
            grid_token_budget({"generator": {"workspace_tokens": 65536}}), 65536
        )
        self.assertEqual(workspace_capacity(66000, 1024), 65536)

    def test_incompatible_cp_padding_is_rejected(self):
        args = argparse.Namespace(
            cache_workspace_tokens=65536,
            dp_size=1,
            concurrency_limit=1,
            max_seq_len=8192,
            cache_shared_seed=False,
            cache_commit_tail_tokens=1024,
            decode_test_length=1,
        )
        with self.assertRaisesRegex(ValueError, "CP execution alignment"):
            _configure_cache_batch_limits(
                args,
                ["--tp_size=3", "--cp_rotate_method=ALL_GATHER"],
                [self.case()],
                cache_alignment=1024,
            )


if __name__ == "__main__":
    unittest.main()
