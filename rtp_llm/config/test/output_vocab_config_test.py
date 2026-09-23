import json
import os
import tempfile
import unittest

from rtp_llm.config.output_vocab_config import (
    OUTPUT_TOKENS_FILENAME,
    load_output_vocab_config,
    load_output_vocab_ids,
    parse_output_tokens,
)


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


class OutputVocabConfigTest(unittest.TestCase):
    def test_flat_text_tokens_use_exact_vocab_and_add_eos(self):
        self.assertEqual(
            parse_output_tokens(
                ["a", "b", "a"],
                model_vocab_size=10,
                tokenizer_vocab={"a": 7, "b": 2},
                extra_token_ids=(0,),
            ),
            [0, 2, 7],
        )

    def test_grouped_text_tokens_are_one_static_union(self):
        self.assertEqual(
            parse_output_tokens(
                [["C0", "C1"], ["C2", "C1"]],
                model_vocab_size=10,
                tokenizer_vocab={"C0": 6, "C1": 4, "C2": 8},
                extra_token_ids=(0,),
            ),
            [0, 4, 6, 8],
        )

    def test_flat_and_grouped_canonical_ids_are_supported(self):
        cases = [
            ([7, 2, 7], [0, 2, 7]),
            ([[7, 2], [7]], [0, 2, 7]),
            ([0, 2, 0], [0, 2]),
        ]
        for raw_tokens, expected in cases:
            with self.subTest(raw_tokens=raw_tokens):
                self.assertEqual(
                    parse_output_tokens(
                        raw_tokens,
                        model_vocab_size=10,
                        extra_token_ids=(0,),
                    ),
                    expected,
                )

    def test_rejects_mixed_or_malformed_source_shapes(self):
        invalid_cases = [
            ([], "non-empty JSON array"),
            ({"steps": [["a"]]}, "non-empty JSON array"),
            (["a", ["b"]], "cannot mix"),
            ([["a"], []], "must not be empty"),
            ([[["a"]]], "only one organizational group level"),
            (["a", 2], "only token strings or only canonical integer"),
            ([True, 2], "only token strings or only canonical integer"),
        ]
        for raw_tokens, message in invalid_cases:
            with self.subTest(raw_tokens=raw_tokens):
                with self.assertRaisesRegex(ValueError, message):
                    parse_output_tokens(
                        raw_tokens,
                        model_vocab_size=10,
                        tokenizer_vocab={"a": 1, "b": 2},
                    )

    def test_rejects_unknown_text_and_unavailable_exact_vocab(self):
        with self.assertRaisesRegex(ValueError, "exact token-to-ID mapping"):
            parse_output_tokens(["a"], model_vocab_size=10)
        with self.assertRaisesRegex(ValueError, "absent from the exact tokenizer"):
            parse_output_tokens(
                ["missing"], model_vocab_size=10, tokenizer_vocab={"a": 1}
            )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            parse_output_tokens(["a"], model_vocab_size=10, tokenizer_vocab={"a": True})

    def test_rejects_invalid_ids_and_non_pruning_sets(self):
        invalid_cases = [
            ([-1, 2], "outside"),
            ([0, 10], "outside"),
            (list(range(10)), "proper subset"),
        ]
        for raw_tokens, message in invalid_cases:
            with self.subTest(raw_tokens=raw_tokens):
                with self.assertRaisesRegex(ValueError, message):
                    parse_output_tokens(raw_tokens, model_vocab_size=10)

    def test_validates_input_embedding_coverage(self):
        self.assertEqual(
            parse_output_tokens([0, 7], model_vocab_size=10, input_vocab_size=0),
            [0, 7],
        )
        with self.assertRaisesRegex(ValueError, "input embedding size 7"):
            parse_output_tokens([0, 7], model_vocab_size=10, input_vocab_size=7)

    def test_loads_algorithm_style_grouped_text_file(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([["C0", "C1"], ["C2", "C3"]], writer)

            self.assertEqual(
                load_output_vocab_ids(
                    checkpoint_path,
                    model_vocab_size=10,
                    tokenizer=FakeTokenizer({"C0": 6, "C1": 4, "C2": 8, "C3": 2}),
                    extra_token_ids=(0,),
                ),
                [0, 2, 4, 6, 8],
            )

            self.assertEqual(
                load_output_vocab_config(
                    checkpoint_path,
                    model_vocab_size=10,
                    tokenizer=FakeTokenizer({"C0": 6, "C1": 4, "C2": 8, "C3": 2}),
                    extra_token_ids=(0,),
                ),
                ([0, 2, 4, 6, 8], [[2, 3], [1, 4]]),
            )

    def test_nested_groups_are_sorted_deduplicated_and_may_overlap(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([[7, 2, 7], [5, 2]], writer)

            self.assertEqual(
                load_output_vocab_config(
                    checkpoint_path,
                    model_vocab_size=10,
                    extra_token_ids=(0,),
                ),
                ([0, 2, 5, 7], [[1, 3], [1, 2]]),
            )

    def test_flat_manifest_has_no_level_groups(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([7, 2, 7], writer)

            self.assertEqual(
                load_output_vocab_config(
                    checkpoint_path,
                    model_vocab_size=10,
                    extra_token_ids=(0,),
                ),
                ([0, 2, 7], []),
            )

    def test_nested_group_rejects_eos_or_other_extra_token(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([[1, 0], [2]], writer)

            with self.assertRaisesRegex(ValueError, "EOS/extra token IDs"):
                load_output_vocab_config(
                    checkpoint_path,
                    model_vocab_size=10,
                    extra_token_ids=(0,),
                )

    def test_single_group_is_static_union_and_may_contain_eos(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([[1, 0, 2]], writer)

            self.assertEqual(
                load_output_vocab_config(
                    checkpoint_path,
                    model_vocab_size=10,
                    extra_token_ids=(0,),
                ),
                ([0, 1, 2], []),
            )

    def test_id_file_does_not_require_tokenizer_vocab(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump([[7, 2], [7]], writer)

            self.assertEqual(
                load_output_vocab_ids(
                    checkpoint_path,
                    model_vocab_size=10,
                    extra_token_ids=(0,),
                ),
                [0, 2, 7],
            )

    def test_missing_manifest_disables_pruning_and_logs_path(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with self.assertLogs(level="INFO") as logs:
                self.assertEqual(
                    load_output_vocab_config(
                        checkpoint_path, model_vocab_size=10, extra_token_ids=(0,)
                    ),
                    ([], []),
                )
            self.assertIn(
                os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME), logs.output[0]
            )
            self.assertIn("skipping output vocabulary pruning", logs.output[0])
            self.assertEqual(
                load_output_vocab_ids(checkpoint_path, model_vocab_size=10), []
            )

    def test_reports_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
            with open(config_path, "w", encoding="utf-8") as writer:
                writer.write("not-json")
            with self.assertRaisesRegex(ValueError, OUTPUT_TOKENS_FILENAME):
                load_output_vocab_ids(checkpoint_path, model_vocab_size=10)

    def test_unreadable_manifest_is_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            os.mkdir(os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME))
            with self.assertRaisesRegex(ValueError, OUTPUT_TOKENS_FILENAME):
                load_output_vocab_config(checkpoint_path, model_vocab_size=10)


if __name__ == "__main__":
    unittest.main()
