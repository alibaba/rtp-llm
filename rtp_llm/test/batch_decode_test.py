from types import SimpleNamespace
from unittest import TestCase, main

import numpy as np
import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.token_processor import TokenProcessor, TokenProcessorPerStream
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs
from rtp_llm.utils.word_util import get_stop_word_slices


class Tokenizer:
    def __init__(self):
        self.batch_calls = 0

    def decode(self, tokens, **kwargs):
        return "".join(f"C{token}" for token in tokens)

    def batch_decode(self, tokens, **kwargs):
        self.batch_calls += 1
        return [self.decode(row, **kwargs) for row in tokens]


class BatchDecodeTest(TestCase):
    def test_variable_beam_batches_and_ragged_sid_lengths(self):
        pipeline = object.__new__(Pipeline)
        pipeline.tokenizer = Tokenizer()
        pipeline._special_tokens = SimpleNamespace(eos_token_id=0)
        config = GenerateConfig(variable_num_beams=[2, 7, 1, 3], out_prefix="sid:")
        states, buffers, tokens = [], [], []
        for width in (2, 7, 1, 3):
            rows = [[11, 12 + i, 0] if i % 2 else [11, 0] for i in range(width)]
            outputs = GenerateOutputs(
                generate_outputs=[
                    GenerateOutput(
                        output_ids=torch.tensor([row], dtype=torch.int32), finished=True
                    )
                    for row in rows
                ]
            )
            texts, lengths, states, buffers, tokens = pipeline.decode_tokens(
                config, outputs, [], [], [], [], states, buffers, tokens
            )
            self.assertEqual(
                texts,
                [
                    "sid:" + pipeline.tokenizer.decode([x for x in row if x != 0])
                    for row in rows
                ],
            )
            self.assertEqual(lengths, [sum(x != 0 for x in row) for row in rows])
            self.assertEqual(len(states), width)
        self.assertEqual(pipeline.tokenizer.batch_calls, 4)

    def test_stop_words_ignore_eos_and_custom_tokenizer(self):
        pipeline = object.__new__(Pipeline)
        pipeline.tokenizer = Tokenizer()
        pipeline._special_tokens = SimpleNamespace(eos_token_id=0)
        outputs = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([[11, 12, 0]], dtype=torch.int32),
                    finished=True,
                )
            ]
        )
        config = GenerateConfig(num_beams=2, ignore_eos=True)
        texts, lengths, *_ = pipeline.decode_tokens(
            config, outputs, [], [], [], [], [], [], []
        )
        self.assertEqual(texts, ["C11C12C0"])
        self.assertEqual(lengths, [3])
        config.ignore_eos = False
        texts, lengths, *_ = pipeline.decode_tokens(
            config, outputs, ["C12"], get_stop_word_slices(["C12"]), [], [], [], [], []
        )
        self.assertEqual(texts, ["C11"])
        self.assertEqual(lengths, [2])

        class CustomTokenizer(BaseTokenizer):
            def decode(self, tokens, **kwargs):
                return "custom:" + str(tokens)

        wrapped = object.__new__(CustomTokenizer)
        wrapped.tokenizer = Tokenizer()
        self.assertEqual(wrapped.batch_decode([[1], [2]]), ["custom:[1]", "custom:[2]"])
        self.assertEqual(wrapped.tokenizer.batch_calls, 0)

    def test_cpp_adapter_batch_keeps_non_beam_incremental_behavior(self):
        tokenizer = Tokenizer()
        processor = TokenProcessor(tokenizer, SimpleNamespace(eos_token_id=0))
        stream = TokenProcessorPerStream(False, 1, processor)
        self.assertEqual(
            stream.decode_tokens_batch([np.array([11])], [False], False, [], [], True),
            ([1], ["C11"]),
        )
        self.assertEqual(
            stream.decode_tokens_batch(
                [np.array([12, 0])], [True], False, [], [], True
            ),
            ([2], ["C12"]),
        )
        beams = TokenProcessorPerStream(True, 2, processor)
        for width in (2, 7, 1, 3):
            lengths, texts = beams.decode_tokens_batch(
                [np.array([11, 12 + i, 0]) for i in range(width)],
                [True] * width,
                False,
                [],
                [],
            )
            self.assertEqual(lengths, [2] * width)
            self.assertEqual(texts, [f"C11C{12 + i}" for i in range(width)])


if __name__ == "__main__":
    main()
