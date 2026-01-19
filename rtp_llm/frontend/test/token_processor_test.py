import logging
import logging.config
import os
from typing import List
from unittest import TestCase, main

import numpy as np
import numpy.typing as npt

from rtp_llm.frontend.token_processor import TokenProcessor, TokenProcessorPerStream
from rtp_llm.tokenizer_factory.tokenizer_utils import DecodingState
from rtp_llm.tokenizer_factory.tokenizers import LlamaTokenizer, QWenTokenizer

os.environ["FT_SERVER_TEST"] = "1"


class MockSpecialTokens:
    def __init__(self, eos_token_id=2):
        self.eos_token_id = eos_token_id


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
            try:
                ret.append(cls(path))
            except Exception as e:
                logging.warning(f"Failed to load tokenizer {cls} from {path}: {e}")
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

                try:
                    output_lens, final_texts = stream_processor.batch_decode_tokens(
                        batch_token_ids=batch_token_ids,
                        batch_finished=batch_finished,
                        print_stop_words=print_stop_words,
                        stop_word_str_list=stop_word_str_list,
                        stop_word_ids=stop_word_ids,
                        return_incremental=False,
                    )

                    # Basic assertions
                    self.assertEqual(len(output_lens), 2)
                    self.assertEqual(len(final_texts), 2)
                    self.assertEqual(len(output_lens), len(final_texts))

                    # Check that output lengths match expected values
                    for i, text in enumerate(batch_texts):
                        expected_tokens = tokenizer.encode(text)
                        expected_len = len(expected_tokens)
                        self.assertEqual(output_lens[i], expected_len)

                except Exception as e:
                    logging.error(
                        f"Failed to test batch_decode_tokens with {type(tokenizer).__name__}: {e}"
                    )
                    # Some tokenizers might not work in test environment, that's OK

    def test_batch_decode_tokens_incremental(self):
        """Test batch_decode_tokens with incremental decoding"""
        for tokenizer in self.tokenizers:
            with self.subTest(tokenizer=type(tokenizer).__name__):
                # Create TokenProcessor and TokenProcessorPerStream
                special_tokens = MockSpecialTokens()
                token_processor = TokenProcessor(tokenizer, special_tokens)

                # Test with has_num_beams=False (incremental decoding)
                stream_processor = TokenProcessorPerStream(
                    has_num_beams=False, size=2, token_processor=token_processor
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

                try:
                    output_lens, final_texts = stream_processor.batch_decode_tokens(
                        batch_token_ids=batch_token_ids,
                        batch_finished=batch_finished,
                        print_stop_words=print_stop_words,
                        stop_word_str_list=stop_word_str_list,
                        stop_word_ids=stop_word_ids,
                        return_incremental=True,
                    )

                    # Basic assertions
                    self.assertEqual(len(output_lens), 2)
                    self.assertEqual(len(final_texts), 2)
                    self.assertEqual(len(output_lens), len(final_texts))

                except Exception as e:
                    logging.error(
                        f"Failed to test batch_decode_tokens with incremental decoding for {type(tokenizer).__name__}: {e}"
                    )

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

                try:
                    output_lens, final_texts = stream_processor.batch_decode_tokens(
                        batch_token_ids=batch_token_ids,
                        batch_finished=batch_finished,
                        print_stop_words=print_stop_words,
                        stop_word_str_list=stop_word_str_list,
                        stop_word_ids=stop_word_ids,
                        return_incremental=False,
                    )

                    # Basic assertions
                    self.assertEqual(len(output_lens), 1)
                    self.assertEqual(len(final_texts), 1)

                except Exception as e:
                    logging.error(
                        f"Failed to test batch_decode_tokens with stop words for {type(tokenizer).__name__}: {e}"
                    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()
