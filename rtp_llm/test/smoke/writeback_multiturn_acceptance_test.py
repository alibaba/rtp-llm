import tempfile
import unittest
from pathlib import Path

from writeback_multiturn_acceptance import (
    MultiturnAcceptance,
    build_turn,
    common_prefix_length,
    complete_prefix,
    expected_reuse,
    validate_cold_control,
    validate_metrics,
)


class CharacterTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class MultiturnAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.aux = {
            "reuse_len": 1024,
            "actual_reuse_len": 1536,
            "writeback_first_use_len": 512,
            "prefill_total_reuse_len": 1536,
            "prefill_local_reuse_len": 1536,
            "decode_total_reuse_len": 1792,
        }

    def test_complete_prefix_excludes_last_sample(self):
        self.assertEqual(complete_prefix(480, 32, 512), 0)
        self.assertEqual(complete_prefix(480, 33, 512), 512)
        self.assertEqual(complete_prefix(992, 33, 512), 1024)

    def test_bounded_replay_can_trim_an_available_writeback_unit(self):
        self.assertEqual(expected_reuse(10252, 10240, 512, 128), 9728)
        self.assertEqual(expected_reuse(10368, 10240, 512, 128), 10240)
        self.assertEqual(expected_reuse(992, 512, 512, 128), 512)

    def test_first_use_is_separate_from_ordinary_reuse(self):
        validate_metrics(self.aux, 1536, 512)

    def test_prefill_reuse_cannot_be_replaced_by_decode_reuse(self):
        self.aux["prefill_total_reuse_len"] = 0
        with self.assertRaisesRegex(AssertionError, "prefill_total_reuse_len"):
            validate_metrics(self.aux, 1536, 512)

    def test_actual_reuse_must_include_first_use(self):
        self.aux["actual_reuse_len"] = 1024
        with self.assertRaisesRegex(AssertionError, "does not equal"):
            validate_metrics(self.aux, 1536, 512)

    def test_first_use_must_match_previous_publication(self):
        with self.assertRaisesRegex(AssertionError, "writeback_first_use_len"):
            validate_metrics(self.aux, 1536, 0)

    def test_missing_metrics_are_not_defaulted_to_zero(self):
        del self.aux["actual_reuse_len"]
        with self.assertRaisesRegex(AssertionError, "Missing integer metric"):
            validate_metrics(self.aux, 1536)

    def test_common_prefix_tracks_tokenization_drift(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(common_prefix_length([1, 2], [1, 2, 3]), 2)
        self.assertEqual(common_prefix_length([], [1]), 0)

    def test_aligned_turn_crosses_exactly_one_new_cp_boundary(self):
        tokenizer = CharacterTokenizer()
        history = []
        for turn in range(1, 21):
            prompt, input_ids, count = build_turn(
                tokenizer, history, "aligned", turn, 1024, True
            )
            self.assertEqual(input_ids, tokenizer.encode(prompt))
            self.assertEqual(input_ids[:len(history)], history)
            self.assertLessEqual(count, 128)
            self.assertEqual(complete_prefix(len(input_ids), count, 1024), turn * 1024)
            history = input_ids + tokenizer.encode("x" * count)

    def test_natural_followup_does_not_force_boundary_padding(self):
        tokenizer = CharacterTokenizer()
        _, history, _ = build_turn(tokenizer, [], "natural", 1, 1024, False)
        prompt, input_ids, count = build_turn(
            tokenizer, history, "natural", 2, 1024, False
        )
        self.assertEqual(input_ids[:len(history)], history)
        self.assertEqual(count, 128)
        self.assertNotIn("repeated words", prompt[len(history):])

    def test_single_token_request_cannot_verify_pd_path(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = MultiturnAcceptance(CharacterTokenizer(), 1, 2, directory, 512)
            with self.assertRaisesRegex(AssertionError, "at least two"):
                runner.generate(Path(directory), "probe", "a", [97], 1, True)

    def test_cold_control_allows_fresh_pd_transfer_but_not_decode_cache(self):
        cold = {key: 0 for key in self.aux}
        cold.update(
            decode_total_reuse_len=2048,
            decode_remote_reuse_len=2048,
            decode_local_reuse_len=0,
            decode_memory_reuse_len=0,
            decode_disk_reuse_len=0,
        )
        validate_cold_control(cold)
        cold["decode_local_reuse_len"] = 512
        with self.assertRaisesRegex(AssertionError, "decode_local_reuse_len"):
            validate_cold_control(cold)


if __name__ == "__main__":
    unittest.main()
