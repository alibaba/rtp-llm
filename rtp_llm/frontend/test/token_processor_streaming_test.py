from types import SimpleNamespace
from unittest import TestCase, main

import numpy as np

from rtp_llm.frontend.token_processor import TokenProcessor, TokenProcessorPerStream


class CharacterTokenizer:
    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)

    def batch_decode(self, sequences):
        return [self.decode(tokens) for tokens in sequences]


class TokenProcessorStreamingTest(TestCase):
    def processor(self, *, beams=False):
        return TokenProcessorPerStream(
            beams,
            2,
            TokenProcessor(CharacterTokenizer(), SimpleNamespace(eos_token_id=0)),
        )

    def batch(
        self, processor, sequences, *, incremental=False, finished=False, stops=()
    ):
        return processor.batch_decode_tokens(
            [np.asarray(tokens, dtype=np.int32) for tokens in sequences],
            [finished] * len(sequences),
            False,
            [],
            list(stops),
            return_incremental=incremental,
        )

    def test_two_rounds_native_1d_cumulative_text(self):
        processor = self.processor()
        self.assertEqual(self.batch(processor, [[65], [88, 89]]), ([1, 2], ["A", "XY"]))
        self.assertEqual(
            self.batch(processor, [[66, 67], [90]], finished=True),
            ([3, 3], ["ABC", "XYZ"]),
        )

    def test_two_rounds_incremental_text_and_empty_delta(self):
        processor = self.processor()
        self.assertEqual(
            self.batch(processor, [[65], [88]], incremental=True),
            ([1, 1], ["A", "X"]),
        )
        self.assertEqual(
            self.batch(processor, [[], [89, 90]], incremental=True),
            ([1, 3], ["", "YZ"]),
        )
        self.assertEqual(
            self.batch(processor, [[66], []], incremental=True, finished=True),
            ([2, 3], ["B", ""]),
        )

    def test_stop_ids_applied_to_accumulated_tokens(self):
        processor = self.processor()
        self.assertEqual(
            self.batch(processor, [[65, 66], [88]], stops=[[66, 67]]),
            ([2, 1], ["A", "X"]),
        )
        self.assertEqual(
            self.batch(processor, [[67], [89]], stops=[[66, 67]], finished=True),
            ([3, 2], ["A", "XY"]),
        )

    def test_single_stream_accepts_native_1d_and_singleton_2d(self):
        processor = self.processor()
        args = dict(
            finished=False,
            print_stop_words=False,
            stop_word_str_list=[],
            stop_word_ids=[],
        )
        self.assertEqual(processor.decode_tokens(0, np.array([65]), **args), (1, "A"))
        self.assertEqual(
            processor.decode_tokens(0, np.array([[66]]), **args), (2, "AB")
        )

    def test_singleton_2d_batch_compatibility_and_eos(self):
        processor = self.processor()
        self.assertEqual(
            self.batch(processor, [[[65, 0]], [[88]]]), ([1, 1], ["A", "X"])
        )
        self.assertEqual(
            self.batch(processor, [[[66]], [[89, 0]]]), ([2, 2], ["AB", "XY"])
        )

    def test_beam_outputs_replace_history(self):
        processor = self.processor(beams=True)
        self.assertEqual(self.batch(processor, [[65], [88]]), ([1, 1], ["A", "X"]))
        self.assertEqual(
            self.batch(processor, [[66, 67], [89]], finished=True),
            ([2, 1], ["BC", "Y"]),
        )


if __name__ == "__main__":
    main()
