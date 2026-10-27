import threading
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from rtp_llm.config.exceptions import EmbeddingInputException, ExceptionType
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.metrics import GaugeMetrics
from rtp_llm.models.downstream_modules.reranker.api_datatype import (
    VoyageRerankerRequest,
)
from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
    Qwen3RerankerHandler,
    Qwen3RerankerRenderer,
    _longest_first_lengths,
)

TOKEN_FALSE_ID = 3
TOKEN_TRUE_ID = 5


class FakeTokenizer:
    """Maps every character to its code point, enough to check token layout."""

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        tokens = [ord(c) for c in text]
        if return_tensors == "np":
            return np.array([tokens], dtype=np.int64)
        return tokens

    def __call__(self, texts, max_length, **kwargs):
        return {"input_ids": [np.array(self.encode(t)[:max_length]) for t in texts]}


def _make_raw_renderer(max_seq_len: int) -> Qwen3RerankerRenderer:
    config = ModelConfig()
    config.max_seq_len = max_seq_len
    config.qwen3_reranker_input_format = "raw"
    renderer = Qwen3RerankerRenderer.__new__(Qwen3RerankerRenderer)
    renderer.config_ = config
    renderer.tokenizer_ = FakeTokenizer()
    renderer.lock = threading.Lock()
    return renderer


def _make_template_renderer(max_seq_len: int) -> Qwen3RerankerRenderer:
    renderer = _make_raw_renderer(max_seq_len)
    renderer.config_.qwen3_reranker_input_format = "template"
    renderer.prefix_tokens_ = np.array([1], dtype=np.int64)
    renderer.suffix_tokens_ = np.array([2], dtype=np.int64)
    return renderer


def _make_handler(fixed_head: bool, lm_head: torch.Tensor) -> Qwen3RerankerHandler:
    config = ModelConfig()
    config.data_type = "fp32"
    config.qwen3_reranker_fixed_head = fixed_head
    handler = Qwen3RerankerHandler(config, TOKEN_FALSE_ID, TOKEN_TRUE_ID)
    handler.device = torch.device("cpu")
    handler.init({handler.lm_head_weight_name: lm_head})
    return handler


class Qwen3RerankerConfigDefaultTest(TestCase):
    def test_model_config_defaults(self):
        config = ModelConfig()
        self.assertTrue(config.qwen3_reranker_fixed_head)
        self.assertEqual(config.qwen3_reranker_input_format, "template")


class Qwen3RerankerRawInputTest(TestCase):
    def test_raw_input_concatenates_query_and_document_tokens(self):
        renderer = _make_raw_renderer(max_seq_len=1024)
        request = VoyageRerankerRequest(query="qq", documents=["abc", "d"])
        inputs = renderer.create_input(request)
        self.assertEqual(inputs.input_lengths.tolist(), [5, 3])
        self.assertEqual(
            inputs.token_ids.tolist(),
            [ord(c) for c in "qqabc"] + [ord(c) for c in "qqd"],
        )
        self.assertEqual(inputs.token_ids.dtype, torch.int32)
        self.assertEqual(inputs.token_type_ids.tolist(), [0] * 8)

    def _assert_input_error(self, request, exception_type):
        renderer = _make_raw_renderer(max_seq_len=8)
        # The dedicated subclass makes the frontend return HTTP 400.
        with self.assertRaises(EmbeddingInputException) as error:
            renderer.create_input(request)
        self.assertEqual(error.exception.exception_type, exception_type)

    def test_raw_input_rejects_empty_documents(self):
        self._assert_input_error(
            VoyageRerankerRequest(query="q", documents=[]),
            ExceptionType.EMPTY_PROMPT_ERROR,
        )

    def test_raw_input_rejects_zero_token_pair_at_any_position(self):
        for documents in (["", "a", "b"], ["a", "", "b"], ["a", "b", ""]):
            with self.subTest(documents=documents):
                self._assert_input_error(
                    VoyageRerankerRequest(query="", documents=documents),
                    ExceptionType.EMPTY_PROMPT_ERROR,
                )

    def test_raw_input_keeps_empty_document_with_non_empty_query(self):
        renderer = _make_raw_renderer(max_seq_len=8)
        request = VoyageRerankerRequest(query="q", documents=["", "a"])
        inputs = renderer.create_input(request)
        self.assertEqual(inputs.input_lengths.tolist(), [1, 2])

    def test_raw_input_rejects_long_pair_without_truncation(self):
        self._assert_input_error(
            VoyageRerankerRequest(
                query="qq", documents=["ab", "abcdefg"], truncation=False
            ),
            ExceptionType.LONG_PROMPT_ERROR,
        )

    def test_raw_input_truncates_document_first_when_longer(self):
        renderer = _make_raw_renderer(max_seq_len=4)
        request = VoyageRerankerRequest(query="qq", documents=["abcdef"])
        inputs = renderer.create_input(request)
        self.assertEqual(inputs.input_lengths.tolist(), [4])
        self.assertEqual(inputs.token_ids.tolist(), [ord(c) for c in "qqab"])

    def test_raw_input_long_query_keeps_document_tokens(self):
        renderer = _make_raw_renderer(max_seq_len=4)
        request = VoyageRerankerRequest(query="qqqqqqqqqq", documents=["abc", "d"])
        inputs = renderer.create_input(request)
        self.assertEqual(inputs.input_lengths.tolist(), [4, 4])
        self.assertEqual(
            inputs.token_ids.tolist(),
            [ord(c) for c in "qqab"] + [ord(c) for c in "qqqd"],
        )

    def test_longest_first_lengths_match_fast_tokenizer(self):
        # Expected values from tokenizers' Rust "longest_first" pair truncation.
        cases = {
            (2, 3, 8): (2, 3),
            (2, 6, 4): (2, 2),
            (10, 3, 4): (2, 2),
            (10, 1, 4): (3, 1),
            (5, 5, 7): (3, 4),
            (6, 5, 7): (4, 3),
            (2, 4, 3): (1, 2),
            (10, 9, 5): (3, 2),
        }
        for (query_len, doc_len, max_len), expected in cases.items():
            with self.subTest(query_len=query_len, doc_len=doc_len, max_len=max_len):
                self.assertEqual(
                    _longest_first_lengths(query_len, doc_len, max_len), expected
                )


class Qwen3RerankerMetricTest(TestCase):
    def test_both_formats_report_pre_pipeline_rt(self):
        for renderer in (_make_raw_renderer(1024), _make_template_renderer(1024)):
            with self.subTest(format=renderer.config_.qwen3_reranker_input_format):
                with patch(
                    "rtp_llm.models.downstream_modules.reranker.qwen3_reranker.kmonitor"
                ) as kmonitor:
                    renderer.create_input(
                        VoyageRerankerRequest(query="q", documents=["a", "b"])
                    )
                reported = [call.args[0] for call in kmonitor.report.call_args_list]
                self.assertEqual(
                    reported,
                    [
                        GaugeMetrics.PRE_PIPELINE_RT_METRIC,
                        GaugeMetrics.INPUT_TOKEN_SIZE_METRIC,
                    ],
                )


class Qwen3RerankerHandlerTest(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab_size = 8
        self.hidden_size = 16
        self.lm_head = torch.randn(self.vocab_size, self.hidden_size)
        self.hidden_states = torch.randn(5, self.hidden_size)
        self.input_lengths = torch.tensor([2, 3], dtype=torch.int32)
        last_token = self.hidden_states[[1, 4]]
        logits = last_token @ self.lm_head.t()
        yes_no = torch.stack(
            [logits[:, TOKEN_FALSE_ID], logits[:, TOKEN_TRUE_ID]], dim=1
        )
        self.expected = torch.log_softmax(yes_no, dim=1)[:, 1].exp()

    def _forward(self, handler: Qwen3RerankerHandler) -> torch.Tensor:
        return handler.forward(
            torch.zeros(5, dtype=torch.int32), self.hidden_states, self.input_lengths
        )

    def test_fixed_head_matches_yes_no_softmax_without_bias(self):
        handler = _make_handler(True, self.lm_head)
        self.assertEqual(tuple(handler.yes_no_weight.shape), (2, self.hidden_size))
        self.assertEqual(handler.yes_no_weight.dtype, torch.float32)
        self.assertIsNone(handler.linear)
        torch.testing.assert_close(self._forward(handler), self.expected)

    def test_fixed_head_is_deterministic_across_instances(self):
        first = self._forward(_make_handler(True, self.lm_head))
        torch.manual_seed(1234)
        second = self._forward(_make_handler(True, self.lm_head))
        self.assertTrue(torch.equal(first, second))

    def test_legacy_head_still_available(self):
        handler = _make_handler(False, self.lm_head)
        self.assertIsNone(handler.yes_no_weight)
        self.assertIsNotNone(handler.linear)
        self.assertEqual(self._forward(handler).shape, self.expected.shape)


if __name__ == "__main__":
    main()
