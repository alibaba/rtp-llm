"""Offline failure gates for the complete K3 PD smoke."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3.text_smoke import Runner, SmokeFailure, parse_args


def runner():
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:1",
        decode_health_url="http://127.0.0.1:2/health",
        decode_role_addrs=[],
        namespace="offline-orthogonal",
        block_size=4096,
        reuse_unit_tokens=32768,
        chunk_tokens=65536,
        max_tokens=32,
        mtp_chunk_max_tokens=32,
        suite="main-text-64k-capped",
    )
    return Runner(args)


class OrthogonalSmokeOfflineTest(unittest.TestCase):
    def test_full_model_cache_pressure_is_bounded(self):
        smoke = runner()
        smoke._orthogonal_device_blocks = lambda: 69

        def fit_prompt(head, tail, target, *, enable_thinking=None):
            self.assertIs(enable_thinking, False)
            return head + tail, [1] * target

        smoke.fit_prompt = fit_prompt
        stages = []
        smoke._required_stage = lambda name, cases: stages.append((name, cases))
        smoke._orthogonal_bounded_cache_pressure()
        self.assertEqual(len(stages), 24)
        self.assertTrue(all(cases[0].preparation_only for _, cases in stages))

    def test_checkpoint_requires_full_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"num_hidden_layers":93}')
            engine = root / "engine.log"
            engine.touch()
            proto = (
                root / "runfiles/rtp_llm/rtp_llm/cpp/model_rpc/proto/"
                "model_rpc_service_pb2.py"
            )
            proto.parent.mkdir(parents=True)
            proto.touch()
            argv = [
                "text_smoke.py",
                "--base-url",
                "http://127.0.0.1:1",
                "--decode-health-url",
                "http://127.0.0.1:2/health",
                "--decode-role-addr",
                "127.0.0.1:2:3",
                "--output",
                str(root / "result.json"),
                "--namespace",
                "test",
                "--block-size",
                "4096",
                "--long-prefix-checkpoint",
                str(checkpoint),
                "--prefill-engine-log",
                str(engine),
                "--prefill-rpc-runfiles",
                str(root / "runfiles"),
                "--prefill-grpc-port",
                "4",
            ]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            self.assertEqual(args.suite, "main-text-64k-capped")
            self.assertEqual(args.run_kind, "full-93")
            self.assertEqual(args.decode_dp_size, 1)
            self.assertTrue(args.require_mtp)
            self.assertEqual(
                args.orthogonal_phases, ("cache", "cancel", "page", "chunk", "decode")
            )
            for option in (
                ["--orthogonal-phases", "cache"],
                ["--suite", "main-text-64k-capped"],
                ["--require-mtp"],
                ["--decode-dp-size", "1"],
            ):
                with self.subTest(option=option), patch.object(
                    sys, "argv", argv + option
                ):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parse_args()
            with patch.object(
                sys, "argv", argv + ["--decode-role-addr", "127.0.0.1:5:6"]
            ):
                args = parse_args()
            self.assertEqual(args.decode_dp_size, 2)
            (checkpoint / "config.json").write_text('{"num_hidden_layers":4}')
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(
                io.StringIO()
            ) as error:
                with self.assertRaises(SystemExit) as rejected:
                    parse_args()
            self.assertEqual(rejected.exception.code, 2)
            self.assertIn("requires a complete 93-layer checkpoint", error.getvalue())

    def test_cache_seed_reaches_kda_checkpoint_before_reuse_probe(self):
        smoke = runner()
        smoke.args.reuse_unit_tokens = 4096
        targets = []

        def fit_prompt(prefix, suffix, target, *, enable_thinking=None):
            self.assertIs(enable_thinking, False)
            targets.append(target)
            return prefix + suffix, [0] * target

        smoke.fit_prompt = fit_prompt
        smoke._required_stage = lambda *args, **kwargs: (_ for _ in ()).throw(
            SmokeFailure("stop after seed")
        )
        with self.assertRaisesRegex(SmokeFailure, "stop after seed"):
            smoke._orthogonal_cache_seed()
        self.assertEqual(targets, [32769])

    def test_cache_seed_stays_within_64k_q(self):
        smoke = runner()
        targets = []

        def fit_prompt(head, tail, target, *, enable_thinking=None):
            self.assertIs(enable_thinking, False)
            targets.append(target)
            return head + tail, [0] * target

        smoke.fit_prompt = fit_prompt
        smoke._required_stage = lambda *args, **kwargs: (_ for _ in ()).throw(
            SmokeFailure("stop after seed")
        )
        with self.assertRaisesRegex(SmokeFailure, "stop after seed"):
            smoke._orthogonal_cache_seed()
        self.assertEqual(targets, [32769])

    def test_cache_probe_rejects_missing_host_hit_without_extra_pressure(self):
        smoke = runner()

        def fit_prompt(head, tail, target, *, enable_thinking=None):
            self.assertIs(enable_thinking, False)
            return head + tail, [0] * target

        smoke.fit_prompt = fit_prompt
        names = []

        def stage(name, cases, concurrent=False):
            names.append(name)
            if name.startswith("tier-device-probe-"):
                self.assertEqual(cases[0].max_tokens, 64)
                smoke.records.append(
                    {"prefill_cache_tiers": {"device": 32768, "memory": 0}}
                )
            if name == "tier-memory-probe":
                self.assertEqual(cases[0].max_tokens, 64)
                smoke.records.append(
                    {"prefill_cache_tiers": {"device": 32768, "memory": 0}}
                )

        smoke._required_stage = stage
        smoke._orthogonal_cache_seed()
        with self.assertRaisesRegex(SmokeFailure, "did not demote"):
            smoke._orthogonal_cache()
        self.assertEqual(names[:2], ["tier-memory-seed-0", "tier-device-probe-0"])
        self.assertFalse(any("evict" in name for name in names))
        self.assertEqual(names[-1], "tier-memory-probe")

    def test_long_history_seed_never_exceeds_uncached_64k_budget(self):
        smoke = runner()

        class Tokens:
            def __init__(self, count):
                self.count = count

            def __len__(self):
                return self.count

        def tokenize(prompt, *, enable_thinking=None):
            self.assertIs(enable_thinking, False)
            return Tokens(27 + prompt.count(" x") + prompt.count(" z"))

        smoke.tokenize = tokenize
        calls = []

        def stage(name, cases, concurrent=False):
            case = cases[0]
            calls.append(case)
            self.assertTrue(case.preparation_only or case.thinking_disabled)
            length = len(smoke.tokenize(case.prompt, enable_thinking=False))
            reuse = 0 if len(calls) == 1 else ((length - 32740) // 32768) * 32768
            if name == "orthogonal_kv_final":
                reuse = 589_824
            smoke.records.append({"input_len": length, "effective_reuse_len": reuse})
            smoke.stages.append({"name": name})

        smoke._required_stage = stage
        smoke._orthogonal_chunk_kv()
        seeds = [case for case in calls if case.preparation_only]
        self.assertLessEqual(len(seeds), 20)
        self.assertTrue(all(case.max_tokens == 4 for case in seeds))
        self.assertTrue(all(case.allow_long_history for case in calls))
        self.assertTrue(
            all(
                row["input_len"] - row["effective_reuse_len"] <= 65536
                for row in smoke.records[1:]
            )
        )
        self.assertEqual(calls[-1].name, "orthogonal_kv_final")

    def test_decode_global_64_routes_32_per_dp_owner(self):
        smoke = runner()
        smoke.decode_role_addrs = [{"role": "DECODE"}, {"role": "DECODE"}]
        groups = []
        smoke._required_stage = lambda name, cases, **kwargs: groups.append(
            (name, cases, kwargs)
        )
        smoke._orthogonal_decode_batches()
        self.assertEqual([len(cases) for _, cases, _ in groups], [1, 4, 8, 63, 64, 64])
        self.assertEqual(
            [
                sum(case.decode_owner_rank == rank for case in groups[4][1])
                for rank in (0, 1)
            ],
            [32, 32],
        )
        self.assertIsNone(groups[4][2].get("admission_wave_size"))
        self.assertEqual(groups[4][2].get("admission_gap_s", 0), 0)
        self.assertEqual(
            [cases[0].max_tokens for _, cases, _ in groups],
            [256, 256, 256, 2560, 1280, 2560],
        )
        self.assertEqual(
            [
                sum(case.decode_owner_rank == rank for case in groups[3][1])
                for rank in (0, 1)
            ],
            [32, 31],
        )
        self.assertTrue(all(group[2]["grouped_pd"] for group in groups[1:]))
        self.assertEqual(groups[4][0], "orthogonal_decode_04_batch_64")
        self.assertTrue(groups[4][2]["grouped_pd"])
        self.assertEqual(groups[3][1][0].max_tokens, 2560)
        self.assertEqual(groups[4][1][0].max_tokens, 1280)
        for index, case in enumerate(groups[4][1]):
            original = smoke.record_case(case.name, index % 2, words=56)
            self.assertEqual(case.prompt, original.prompt)
            self.assertEqual(case.expected_json, original.expected_json)
            self.assertEqual(len(case.expected_json["value"].split()), 56)
        name, cases, options = groups[5]
        self.assertEqual(name, "orthogonal_decode_graph_long_output_batch_64")
        self.assertEqual(len(cases), 64)
        self.assertEqual(
            [
                sum(case.decode_owner_rank == owner for case in cases)
                for owner in (0, 1)
            ],
            [32, 32],
        )
        self.assertTrue(options["concurrent"] and options["grouped_pd"])
        for case in cases:
            self.assertTrue(case.name.startswith(name + "_"))
            self.assertEqual(len(case.expected_json["value"].split()), 112)
            self.assertEqual(case.max_tokens, 2560)
            self.assertTrue(case.require_mtp_draft and case.thinking_disabled)
        self.assertFalse(groups[0][2]["grouped_pd"])
        self.assertTrue(all(group[2]["grouped_pd"] for group in groups[1:]))
        for index, local_batch in ((1, 2), (2, 4)):
            self.assertEqual(
                [
                    sum(case.decode_owner_rank == owner for case in groups[index][1])
                    for owner in (0, 1)
                ],
                [local_batch, local_batch],
            )

    def test_cached_64k_crosses_decode_dp_owner_without_extra_requests(self):
        smoke = runner()
        smoke.args.reuse_unit_tokens = 4096
        smoke.decode_role_addrs = [{"role": "DECODE"}, {"role": "DECODE"}]
        smoke.fit_prompt = lambda head, tail, target: (head + tail, [1] * target)
        stages = []
        smoke.run_stage = lambda name, cases, concurrent=False: stages.append(
            (name, cases)
        )
        smoke._required_stage = smoke.run_stage
        smoke.run_single_prefill_64k()
        self.assertEqual(sum(len(cases) for _, cases in stages), 6)
        name, cases = stages[1]
        self.assertEqual(name, "decode_dp_cross_owner_cached_64k")
        self.assertEqual(cases[0].decode_owner_rank, 1)
        self.assertEqual(cases[0].expected_reuse_len, 61440)
        self.assertTrue(cases[0].require_mtp)


if __name__ == "__main__":
    unittest.main()
