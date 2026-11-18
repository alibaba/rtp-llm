import logging
import logging.config
import os
from unittest import TestCase, main

import pytest

pytestmark = [pytest.mark.gpu(type="A10")]

import numpy as np

from rtp_llm.frontend.token_processor import TokenProcessor, TokenProcessorPerStream
from rtp_llm.frontend.tokenizer_factory.tokenizers import LlamaTokenizer, QWenTokenizer

os.environ["FT_SERVER_TEST"] = "1"


class MockSpecialTokens:
    def __init__(self, eos_token_id=2):
        self.eos_token_id = eos_token_id


class AsciiTokenizer:
    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class TokenProcessorTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizers = self._get_tokenizer_list()
        self.inputs = [
            ["你好,你的名字是什么", "Hello, what's your name?"],
            ["Testing batch decode", "Another test string"],
            [
                "sxsadasfdjsadfas asdas djbnasdb asj asiokdnaskd asnkdnaskd naskdnas knask",
                "Another long string for testing batch processing",
            ],
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

    def test_batch_decode_tokens_simple(self):
        """Test basic functionality of batch_decode_tokens"""
        for tokenizer in self.tokenizers:
            with self.subTest(tokenizer=type(tokenizer).__name__):
                # Create TokenProcessor and TokenProcessorPerStream
                special_tokens = MockSpecialTokens()
                token_processor = TokenProcessor(tokenizer, special_tokens)

                # Test with has_num_beams=True (simpler case)
                stream_processor = TokenProcessorPerStream(
                    has_num_beams=True, size=2, token_processor=token_processor
                )

                # Encode test inputs
                batch_texts = ["Hello world", "Test string"]
                batch_token_ids = []
                for text in batch_texts:
                    encoded = tokenizer.encode(text)
                    # Convert to numpy array with correct shape (1, len)
                    batch_token_ids.append(np.array([encoded], dtype=np.int32))

                # Call batch_decode_tokens
                batch_finished = [True, True]
                print_stop_words = False
                stop_word_str_list = []
                stop_word_ids = []

                output_lens, final_texts = stream_processor.batch_decode_tokens(
                    batch_token_ids=batch_token_ids,
                    batch_finished=batch_finished,
                    print_stop_words=print_stop_words,
                    stop_word_str_list=stop_word_str_list,
                    stop_word_ids=stop_word_ids,
                    return_incremental=False,
                )

                self.assertEqual(len(output_lens), 2)
                self.assertEqual(len(final_texts), 2)
                for i, text in enumerate(batch_texts):
                    expected_tokens = tokenizer.encode(text)
                    self.assertEqual(output_lens[i], len(expected_tokens))
                    self.assertEqual(final_texts[i], tokenizer.decode(expected_tokens))

    def test_batch_decode_tokens_incremental(self):
        """Batched decoding must match independent streams over multiple chunks."""
        for tokenizer in self.tokenizers:
            for incremental in (True, False):
                with self.subTest(
                    tokenizer=type(tokenizer).__name__, incremental=incremental
                ):
                    processor = TokenProcessor(tokenizer, MockSpecialTokens())
                    batched = TokenProcessorPerStream(False, 2, processor)
                    single = TokenProcessorPerStream(False, 2, processor)
                    token_ids = [
                        tokenizer.encode(text)
                        for text in ("Hello world", "Test string")
                    ]
                    for ids in token_ids:
                        self.assertGreaterEqual(len(ids), 2)
                    cuts = [len(ids) // 2 for ids in token_ids]
                    outputs = []
                    for step in (0, 1):
                        chunks = [
                            np.array(
                                [ids[:cut] if step == 0 else ids[cut:]], dtype=np.int32
                            )
                            for ids, cut in zip(token_ids, cuts)
                        ]
                        finished = step == 1
                        lengths, texts = batched.batch_decode_tokens(
                            chunks, [finished, finished], False, [], [], incremental
                        )
                        expected = [
                            single.decode_tokens(
                                i, chunk, finished, False, [], [], incremental
                            )
                            for i, chunk in enumerate(chunks)
                        ]
                        self.assertEqual(lengths, [item[0] for item in expected])
                        self.assertEqual(texts, [item[1] for item in expected])
                        outputs.append(texts)

                    actual = (
                        [first + last for first, last in zip(*outputs)]
                        if incremental
                        else outputs[-1]
                    )
                    self.assertEqual(
                        actual, [tokenizer.decode(ids) for ids in token_ids]
                    )

    def test_batch_incremental_stop_words_and_padding(self):
        processor = TokenProcessor(AsciiTokenizer(), MockSpecialTokens(eos_token_id=0))
        batched = TokenProcessorPerStream(False, 2, processor)
        single = TokenProcessorPerStream(False, 2, processor)
        steps = [
            ([[ord(c) for c in "aST"] + [0], [ord(c) for c in "xy"] + [0]], False),
            ([[ord(c) for c in "OPb"] + [0], [ord(c) for c in "z!"] + [0]], True),
        ]
        actual = []
        for token_rows, finished in steps:
            chunks = [np.array([row], dtype=np.int32) for row in token_rows]
            lengths, texts = batched.batch_decode_tokens(
                chunks, [finished, finished], False, ["STOP"], [[ord("!")]], True
            )
            expected = [
                single.decode_tokens(
                    i, chunk, finished, False, ["STOP"], [[ord("!")]], True
                )
                for i, chunk in enumerate(chunks)
            ]
            self.assertEqual(lengths, [item[0] for item in expected])
            self.assertEqual(texts, [item[1] for item in expected])
            actual.append(texts)
        self.assertEqual(actual, [["a", "xy"], ["", "z"]])

    def test_batch_decode_tokens_with_stop_words(self):
        """Test batch_decode_tokens with stop words"""
        for tokenizer in self.tokenizers:
            with self.subTest(tokenizer=type(tokenizer).__name__):
                # Create TokenProcessor and TokenProcessorPerStream
                special_tokens = MockSpecialTokens()
                token_processor = TokenProcessor(tokenizer, special_tokens)

                # Test with has_num_beams=True
                stream_processor = TokenProcessorPerStream(
                    has_num_beams=True, size=1, token_processor=token_processor
                )

                # Encode test input
                batch_texts = ["Hello world. Stop here."]
                batch_token_ids = []
                for text in batch_texts:
                    encoded = tokenizer.encode(text)
                    batch_token_ids.append(np.array([encoded], dtype=np.int32))

                # Call batch_decode_tokens with stop words
                batch_finished = [True]
                print_stop_words = False
                stop_word_str_list = ["world"]
                stop_word_ids = []

                output_lens, final_texts = stream_processor.batch_decode_tokens(
                    batch_token_ids=batch_token_ids,
                    batch_finished=batch_finished,
                    print_stop_words=print_stop_words,
                    stop_word_str_list=stop_word_str_list,
                    stop_word_ids=stop_word_ids,
                    return_incremental=False,
                )
                self.assertEqual(len(output_lens), 1)
                self.assertEqual(len(final_texts), 1)
                self.assertEqual(
                    final_texts[0], tokenizer.decode(encoded).split("world", 1)[0]
                )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()
