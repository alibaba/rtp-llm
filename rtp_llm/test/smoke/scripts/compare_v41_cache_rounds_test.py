import copy
import unittest

from rtp_llm.test.smoke.scripts.compare_v41_cache_rounds import (
    DIGEST,
    PHASES,
    compare,
    hash_ids,
    validate,
)


def sample(loader):
    tag = ("a" if loader == "oldfst" else "b") * 32
    d = dict(
        schema=1,
        loader=loader,
        reader="fastsafetensors",
        run_id=tag,
        vocab_size=2000,
        eos_id=1,
        input_ids_sha256=DIGEST,
        input_len=783,
        phases={},
    )
    for phase in PHASES:
        out = [1606, 3, 1]
        row = dict(
            run_id=tag,
            phase=phase,
            offset=1,
            history_sha256_le_i32="c" * 64,
            committed_sha256_le_i32=hash_ids([1606]),
            draft=[3, 4, 5],
            target=[3, 1, 7, 8],
            first_verify_logits={
                "draft_id": 3,
                "sampled_target_id": 3,
                "draft_id_target_logit": 7.5,
                "sampled_id_target_logit": 7.5,
            },
            accept_len=2,
            accepted_draft_ids=[3],
            result=[3, 1],
        )
        d["phases"][phase] = dict(
            first_token=1606,
            output_ids=out,
            expected_ids=out[:],
            matches_golden=True,
            draft_rounds=1,
            pd_sep=True,
            prefill_reuse_len=512 if phase == "hit" else 0,
            rounds=[row],
        )
    return d


class CacheRoundsTest(unittest.TestCase):
    def test_only_compare_causal_target_rows(self):
        a, b = sample("oldfst"), sample("newfst")
        row = b["phases"]["hit"]["rounds"][0]
        row["draft"][1] = 99
        row["target"][2] = 42
        r = compare(a, b)
        self.assertEqual(set(r["phases"]["hit"]["first_diff"]), {"draft"})
        row["target"][1] = 73
        self.assertIn("target", compare(a, b)["phases"]["hit"]["first_diff"])

    def test_fail_closed_on_partial_or_misaligned_evidence(self):
        a = sample("oldfst")
        for key, bad in [
            ("offset", 3),
            ("committed_sha256_le_i32", "d" * 64),
            ("run_id", "e" * 32),
            ("result", [3, 99]),
            ("accept_len", 5),
        ]:
            b = copy.deepcopy(a)
            b["phases"]["hit"]["rounds"][0][key] = bad
            with self.assertRaises(ValueError):
                validate(b)
        b = sample("newfst")
        b["phases"]["hit"]["rounds"][0]["history_sha256_le_i32"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "mismatched native contexts"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
