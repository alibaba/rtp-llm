"""Real Qwen2 BPE boundary regressions; runnable without inference bindings/GPU."""

import unittest
from pathlib import Path
from unittest.mock import Mock

from tokenizers import Tokenizer

from rtp_llm.dash_sc.think import (
    MAX_THINK_ANCHOR_DECODE_TOKENS,
    MAX_THINK_ANCHOR_SCAN_TOKENS,
    build_empty_think_phase2_input_ids,
    match_think_anchor,
)


class ThinkAnchorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = (
            Path(__file__).resolve().parents[2]
            / "test/tokenizer_test/testdata/qwen2_tokenizer/tokenizer.json"
        )
        cls.tokenizer = Tokenizer.from_file(str(fixture))
        cls.bos = cls.encode("<think>\n")
        cls.bare = cls.encode("<think>")
        cls.empty = cls.encode("<think>\n\n</think>\n\n")
        cls.empty_suffix = cls.encode("\n</think>\n\n")

    @classmethod
    def encode(cls, text):
        return cls.tokenizer.encode(text, add_special_tokens=False).ids

    def match(self, ids, decode=None):
        return match_think_anchor(
            ids,
            bos_tokens=self.bos,
            bare_bos_tokens=self.bare,
            anchor="<think>\n",
            decode=decode or self.tokenizer.decode,
            max_token_id=self.tokenizer.get_vocab_size() - 1,
        )

    def test_complete_partial_and_closed_tags(self):
        for prefix in ("", "abc ", "中文🙂 ", "prompt\n"):
            for tail, expected in (
                ("<think>", True),
                ("<think>\n", True),
                ("<think>\n\n", True),
                ("", False),
                ("<th", False),
                ("<think> ", False),
                ("<think>\n\n</think>\n\n", False),
                ("plain\n", False),
            ):
                with self.subTest(prefix=prefix, tail=tail):
                    ids = self.encode(prefix + tail)
                    anchored, matched = self.match(ids)
                    self.assertEqual(anchored, expected)
                    if matched:
                        self.assertEqual(ids[-len(matched) :], matched)
                        self.assertEqual(
                            self.tokenizer.decode(matched).rstrip("\n"), "<think>"
                        )

    def test_exact_raw_and_bare_anchors_do_not_decode(self):
        decoder = Mock(side_effect=AssertionError("unexpected decode"))
        self.assertNotEqual(self.bos[-1], self.bare[-1])
        for suffix in (self.bos, self.bare):
            self.assertEqual(self.match([42] + suffix, decoder), (True, suffix))

    def test_long_newline_suffix_is_retained_for_phase2_removal(self):
        newline = self.encode("\n")
        for count in (65, 128, 4096):
            ids = self.bare + newline * count
            self.assertEqual(self.match(ids), (True, ids))
            self.assertEqual(
                self.match(self.encode("<th") + newline * count), (False, [])
            )

    def test_excessive_newline_tail_has_bounded_decode_work(self):
        decoder = Mock(wraps=self.tokenizer.decode)
        ids = self.bare + self.encode("\n") * 100_000
        with self.assertRaisesRegex(ValueError, "8192 trailing input tokens"):
            self.match(ids, decoder)
        decoded_tokens = sum(len(call.args[0]) for call in decoder.call_args_list)
        self.assertEqual(decoded_tokens, MAX_THINK_ANCHOR_SCAN_TOKENS)

    def test_complete_anchor_at_scan_limit_is_recognized(self):
        newline = self.encode("\n")
        self.assertEqual(len(newline), 1)
        ids = self.bare + newline * (MAX_THINK_ANCHOR_SCAN_TOKENS - len(self.bare))
        self.assertEqual(self.match(ids), (True, ids))
        with self.assertRaisesRegex(ValueError, "8192 trailing input tokens"):
            self.match(ids + newline)

    def test_invisible_tokens_cannot_amplify_decode_work(self):
        # SentencePiece decoders may map valid special tokens to empty text.
        # Exercise both a short nonempty tail and a sparsely encoded full tag.
        pieces = {0: "", **dict(enumerate("<think>", start=1))}
        for ids in (
            [0] * MAX_THINK_ANCHOR_SCAN_TOKENS + [1],
            [token for char in range(1, 8) for token in [char] + [0] * 63],
            [token for char in range(1, 8) for token in [char] + [0] * 1023],
        ):
            with self.subTest(size=len(ids)):
                decoder = Mock(
                    side_effect=lambda ids, **kwargs: "".join(pieces[i] for i in ids)
                )
                with self.assertRaisesRegex(ValueError, "decoded input tokens"):
                    self.match(ids, decoder)
                self.assertLessEqual(
                    sum(len(call.args[0]) for call in decoder.call_args_list),
                    MAX_THINK_ANCHOR_DECODE_TOKENS,
                )

    def test_context_merged_anchor_does_not_scan_preceding_prompt(self):
        decoder = Mock(wraps=self.tokenizer.decode)
        ids = self.encode("ordinary prompt " * 100 + "abc <think>\n\n")
        self.assertEqual(self.match(ids, decoder), (True, []))
        self.assertLessEqual(decoder.call_count, 8)

    def test_empty_phase2_closes_each_anchor_exactly_once(self):
        for text in (
            "abc <think>\n\n",
            "prompt\n<think>",
            "prompt\n<think>\n\n",
            "plain",
        ):
            with self.subTest(text=text):
                ids = self.encode(text)
                anchored, matched = self.match(ids)
                result = build_empty_think_phase2_input_ids(
                    ids,
                    matched,
                    self.empty,
                    prompt_has_think_anchor=anchored,
                    empty_think_suffix_tokens=self.empty_suffix,
                )
                decoded = self.tokenizer.decode(result)
                self.assertEqual(decoded.count("<think>"), 1)
                self.assertEqual(decoded.count("</think>"), 1)
                self.assertTrue(decoded.endswith("</think>\n\n"))
                if anchored and not matched:
                    self.assertEqual(result[: len(ids)], ids)
                    self.assertTrue(decoded.startswith(text))

    def test_custom_decoder_without_keyword_arguments(self):
        def decode(ids):
            return self.tokenizer.decode(ids, skip_special_tokens=False)

        self.assertEqual(self.match(self.encode("abc <think>\n\n"), decode), (True, []))

    def test_custom_decoder_without_introspectable_signature(self):
        tokenizer = self.tokenizer

        class Decoder:
            @property
            def __signature__(self):
                raise ValueError("no signature for extension callable")

            def __call__(self, ids):
                return tokenizer.decode(ids, skip_special_tokens=False)

        self.assertEqual(
            self.match(self.encode("abc <think>\n\n"), Decoder()), (True, [])
        )

    def test_signature_error_does_not_replace_decoder_error(self):
        for signature_error in (ValueError, TypeError):
            with self.subTest(signature_error=signature_error):

                class Decoder:
                    @property
                    def __signature__(self):
                        raise signature_error("no signature")

                    def __call__(self, ids, **kwargs):
                        raise TypeError("original decoder error")

                with self.assertRaisesRegex(TypeError, "original decoder error"):
                    self.match(self.encode("plain"), Decoder())

    def test_decoder_internal_type_error_is_not_silently_retried(self):
        def decode(ids, skip_special_tokens=False):
            raise TypeError("decoder implementation error")

        with self.assertRaisesRegex(TypeError, "decoder implementation error"):
            self.match(self.encode("plain"), decode)

    def test_invalid_tail_token_ids_are_rejected(self):
        for ids in ([-1], [self.tokenizer.get_vocab_size()], [2**31 - 1]):
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(ValueError, "vocabulary"),
            ):
                self.match(ids)


if __name__ == "__main__":
    unittest.main()
