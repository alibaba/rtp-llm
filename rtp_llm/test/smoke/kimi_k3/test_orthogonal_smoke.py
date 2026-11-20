"""Offline failure gates for the complete K3 PD smoke."""

import contextlib
import io
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
                ["--batch-size", "17"],
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

    def test_prefill_admission_rejects_more_than_16_requests(self):
        smoke = runner()
        with self.assertRaisesRegex(SmokeFailure, "must not exceed 16"):
            smoke.request_cases([None] * 17, concurrent=True)
        with self.assertRaisesRegex(SmokeFailure, "must not exceed 16"):
            smoke.request_cases_group([None] * 17)


if __name__ == "__main__":
    unittest.main()
