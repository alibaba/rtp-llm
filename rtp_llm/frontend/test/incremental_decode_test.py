import logging
import logging.config
import os
import random
from types import SimpleNamespace
from unittest import TestCase, main

import numpy as np

from rtp_llm.frontend.token_processor import TokenProcessor, TokenProcessorPerStream
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import LlamaTokenizer, QWenTokenizer

os.environ["FT_SERVER_TEST"] = "1"


class IncrementalDecodeTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizers = self._get_tokenizer_list()
        self.inputs = [
            "你好,你的名字是什么",
            "hello, what's your name?",
            "sxsadasfdjsadfas asdas djbnasdb asj asiokdnaskd asnkdnaskd naskdnas knask",
        ]

    def _get_tokenizer_list(self):
        ret = []
        tokenizer_pairs = [
            (
                QWenTokenizer,
                "rtp_llm/test/model_test/fake_test/testdata/qwen_7b/tokenizer/",
            ),
            (
                LlamaTokenizer,
                "rtp_llm/test/model_test/fake_test/testdata/llama/fake/hf_source/",
            ),
        ]
        for cls, path in tokenizer_pairs:
            ret.append(cls(path))
        return ret

    def _run_incremental_decode(self, tokenizer, all_input_ids, skip_special_tokens):
        text = ""
        state = DecodingState()
        for i in range(0, len(all_input_ids), 1):
            out = IncrementDecodingUtils.detokenize_incrementally(
                tokenizer, all_input_ids[: i + 1], state, skip_special_tokens
            )
            text += out
        return text

    def _run_incremental_decode_random(
        self, tokenizer, all_input_ids, skip_special_tokens
    ):
        text = ""
        state = DecodingState()
        index = 0
        while index < len(all_input_ids):
            out = IncrementDecodingUtils.detokenize_incrementally(
                tokenizer, all_input_ids[:index], state, skip_special_tokens
            )
            text += out
            index += random.randint(1, 2)
        out = IncrementDecodingUtils.detokenize_incrementally(
            tokenizer, all_input_ids[: len(all_input_ids)], state, skip_special_tokens
        )
        text += out
        return text

    def test_simple(self):
        for input in self.inputs:
            for tokenizer in self._get_tokenizer_list():
                logging.info("Test Tokenizer: " + str(tokenizer.__class__))
                tokens = tokenizer.encode(input)

                base_output = tokenizer.decode(tokens)
                cmp_output = self._run_incremental_decode(tokenizer, tokens, False)
                self.assertEqual(base_output, cmp_output)

    def test_random_step(self):
        for input in self.inputs:
            for tokenizer in self._get_tokenizer_list():
                logging.info("Test Tokenizer: " + str(tokenizer.__class__))
                tokens = tokenizer.encode(input)

                base_output = tokenizer.decode(tokens)
                cmp_output = self._run_incremental_decode_random(
                    tokenizer, tokens, False
                )
                self.assertEqual(base_output, cmp_output)


class TokenProcessorPerStreamTest(TestCase):
    """The C++ HTTP adapter passes one-dimensional int32 arrays per sequence."""

    def make_processor(self, beams=False, size=1):
        tokenizer = SimpleNamespace(decode=lambda ids: "".join(chr(i) for i in ids))
        return TokenProcessorPerStream(
            beams, size, TokenProcessor(tokenizer, SimpleNamespace(eos_token_id=0))
        )

    def decode(self, processor, tokens, i=0, finished=False, incremental=False):
        return processor.decode_tokens(
            i, np.asarray(tokens, dtype=np.int32), finished, False, [], [], incremental
        )

    def test_first_single_beam_chunk(self):
        processor = self.make_processor()
        self.assertEqual(self.decode(processor, [65, 66]), (2, "AB"))
        self.assertEqual(processor.ouput_tokens_list[0].shape, (2,))
        self.assertEqual(processor.ouput_tokens_list[0].dtype, np.int32)

    def test_single_beam_accumulates_chunks_and_removes_eos(self):
        processor = self.make_processor()
        self.assertEqual(self.decode(processor, [65]), (1, "A"))
        self.assertEqual(self.decode(processor, [66, 67]), (3, "ABC"))
        self.assertEqual(self.decode(processor, [0], finished=True), (3, "ABC"))

    def test_single_beam_incremental_text(self):
        processor = self.make_processor()
        self.assertEqual(self.decode(processor, [65], incremental=True), (1, "A"))
        self.assertEqual(self.decode(processor, [66], incremental=True), (2, "B"))

    def test_empty_chunk_preserves_history(self):
        processor = self.make_processor()
        self.assertEqual(self.decode(processor, []), (0, ""))
        self.decode(processor, [65])
        self.assertEqual(self.decode(processor, []), (1, "A"))

    def test_sequences_have_independent_histories(self):
        processor = self.make_processor(size=2)
        self.decode(processor, [65], i=0)
        self.decode(processor, [66], i=1)
        self.assertEqual(self.decode(processor, [67], i=0), (2, "AC"))
        self.assertEqual(self.decode(processor, [68], i=1), (2, "BD"))

    def test_beam_search_replaces_history_instead_of_appending(self):
        processor = self.make_processor(beams=True)
        self.assertEqual(self.decode(processor, [65]), (1, "A"))
        self.assertEqual(self.decode(processor, [66, 67]), (2, "BC"))
        self.assertEqual(self.decode(processor, [66, 67, 0], finished=True), (2, "BC"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()
