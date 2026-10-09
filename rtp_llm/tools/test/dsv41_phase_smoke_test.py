"""Reject incomplete native smokes and distinguish forced decode from EOS."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from rtp_llm.tools.dsv41_phase_smoke import TOPOLOGY, verify_run, verify_sources


def fixture(count=9, candidate=None):
    prompts = [[3, 4], [5, 6]]
    reference = {
        "world_size": 8,
        "outputs": [
            {
                "prompt_ids": prompts[0],
                "generated_ids": [11, 12, 1, 14, 15, 16, 17, 18, 19],
            },
            {
                "prompt_ids": prompts[1],
                "generated_ids": [21, 22, 23, 24, 25, 26, 27, 28, 29],
            },
        ],
    }
    ids = candidate or [row["generated_ids"] for row in reference["outputs"]]
    manifest = {
        "topology": TOPOLOGY.copy(),
        "new_tokens": count,
        "native_role": "PREFILL" if count == 1 else "PDFUSION",
        "profiles": ["baseline"],
        "fake_balance_expert": False,
        "eos_disabled": True,
        "eos_token_id": 1,
        "prompt_token_ids": prompts,
    }
    records = [
        f"rank={rank} phase={pattern} precision=baseline "
        f"output_tokens={ids[pattern % 2][:count]} hidden_sha256={['a' * 64] * count}"
        for pattern in range(4)
        for rank in (range(8) if pattern in (1, 2) else [0 if pattern == 0 else 7])
    ]
    tail = (
        f"checkpoint-to-native-engine full_nas=True new_tokens={count} "
        "context=256 precision=['baseline'] generation liveness PASS\n"
        "Ran 1 test in 123.4s\n\nOK\n"
        "exitcode: 0\n"
        "[GPU_LOCK_NO_KILL] leased GPUs verified idle; no signals\n"
        "release done\n"
    )
    return (
        f"full-engine admission bytes=123 prompts={prompts}\n"
        + "\n".join(records)
        + "\n"
        + tail,
        reference,
        manifest,
    )


class NativePhaseSmokeTest(unittest.TestCase):
    def test_exact_prefill_and_real_eight_step_decode(self):
        for phase, count in (("prefill", 1), ("decode", 9)):
            result = verify_run(phase, *fixture(count))
            self.assertTrue(result["all_requested_ids_exact"])
            self.assertEqual(result["record_count"], 18)
            self.assertEqual(result["real_decode_steps_per_request"], count - 1)
            self.assertFalse(result["http_qualified"])
            self.assertFalse(result["cep4pp2_qualified"])
            self.assertFalse(result["hidden_logit_accuracy_qualified"])

    def test_missing_eos_is_a_reference_failure(self):
        log, reference, manifest = fixture()
        candidate = [row["generated_ids"].copy() for row in reference["outputs"]]
        candidate[0][2] = 99
        result = verify_run("decode", *fixture(candidate=candidate))
        self.assertFalse(result["all_requested_ids_exact"])
        self.assertEqual(result["comparisons"][0]["first_mismatch_step"], 3)
        self.assertFalse(result["comparisons"][0]["through_eos_exact"])

    def test_difference_after_eos_is_reported_and_fails_forced_decode(self):
        _, reference, _ = fixture()
        candidate = [row["generated_ids"].copy() for row in reference["outputs"]]
        candidate[0][6] = 99
        result = verify_run("decode", *fixture(candidate=candidate))
        self.assertFalse(result["all_requested_ids_exact"])
        row = result["comparisons"][0]
        self.assertTrue(row["through_eos_exact"])
        self.assertEqual(row["first_mismatch_step"], 7)
        self.assertEqual(row["forced_steps_after_reference_eos"], 6)

    def test_missing_and_duplicate_rank_records_rejected(self):
        log, reference, manifest = fixture()
        admission, first, rest = log.split("\n", 2)
        for altered in (
            admission + "\n" + rest,
            admission + "\n" + first + "\n" + first + "\n" + rest,
        ):
            with self.assertRaisesRegex(ValueError, "18 rank/phase|Duplicate"):
                verify_run("decode", altered, reference, manifest)

    def test_hidden_hash_disagreement_rejected(self):
        log, reference, manifest = fixture()
        with self.assertRaisesRegex(ValueError, "Repeated native"):
            verify_run(
                "decode", log.replace("a" * 64, "b" * 64, 1), reference, manifest
            )

    def test_truncated_and_skipped_runs_rejected(self):
        log, reference, manifest = fixture()
        for altered in (
            log.replace("OK", "FAILED"),
            log.replace("OK", "OK (skipped=1)"),
            log[: log.index("checkpoint-to-native")],
            log.replace("release done", ""),
            log.replace("exitcode: 0", "exitcode: 1"),
        ):
            with self.assertRaisesRegex(ValueError, "teardown"):
                verify_run("decode", altered, reference, manifest)

    def test_synthetic_decode_and_diagnostic_profiles_rejected(self):
        log, reference, manifest = fixture()
        for field, value in (
            ("native_role", "DECODE"),
            ("profiles", ["tp8_projection"]),
            ("fake_balance_expert", True),
            ("topology", {**TOPOLOGY, "pp_size": 2}),
        ):
            changed = copy.deepcopy(manifest)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "unpatched DP8EP8"):
                verify_run("decode", log, reference, changed)
        with self.assertRaisesRegex(ValueError, "Diagnostic precision"):
            verify_run(
                "decode",
                log.replace("precision=baseline", "precision=mhc_fp32", 1),
                reference,
                manifest,
            )

    def test_wrong_or_short_reference_rejected(self):
        log, reference, manifest = fixture()
        for altered in (
            {**reference, "world_size": 4},
            {**reference, "outputs": reference["outputs"][:1]},
            {**reference, "outputs": reference["outputs"] * 2},
            {
                **reference,
                "outputs": [
                    {**row, "generated_ids": row["generated_ids"][:3]}
                    for row in reference["outputs"]
                ],
            },
        ):
            with self.assertRaises(ValueError):
                verify_run("decode", log, altered, manifest)
        with self.assertRaisesRegex(ValueError, "admission prompt IDs"):
            verify_run(
                "decode",
                log.replace("prompts=[[3, 4], [5, 6]]", "prompts=[[8, 9], [5, 6]]"),
                reference,
                manifest,
            )

    def test_frozen_source_report_and_asset_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, checkpoint = root / "repo", root / "checkpoint"
            sources = [
                "rtp_llm/models_py/modules/dsv4/fp8/test/test_v41_checkpoint_engine_sm120.py",
                "rtp_llm/models_py/model_desc/deepseek_v41_model.py",
                "rtp_llm/cpp/normal_engine/test/V41NativeEngineTest.cc",
            ]
            official = (
                "model.py",
                "kernel.py",
                "engram.py",
                "convert.py",
                "config.json",
            )
            assets = (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
            paths = [
                *[repo / name for name in sources],
                *[checkpoint / "inference" / name for name in official],
                *[checkpoint / name for name in assets],
                root / "reference.json",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen")
            digest = hashlib.sha256(b"frozen").hexdigest()
            manifest = {
                "checkpoint": str(checkpoint),
                "source_sha256": dict.fromkeys(sources, digest),
                "checkpoint_asset_sha256": dict.fromkeys(assets, digest),
                "reference_report_sha256": digest,
            }
            reference = {"official_source_sha256": dict.fromkeys(official, digest)}
            verify_sources(repo, checkpoint, paths[-1], reference, manifest)
            for path in paths:
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    verify_sources(repo, checkpoint, paths[-1], reference, manifest)
                path.write_text("frozen")


if __name__ == "__main__":
    unittest.main()
