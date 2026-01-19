import logging
import logging.config
import os
import random
from unittest import TestCase, main

from rtp_llm.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.tokenizer_factory.tokenizers import LlamaTokenizer, QWenTokenizer


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()
