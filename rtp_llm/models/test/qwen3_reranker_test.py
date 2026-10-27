from unittest import TestCase, main

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.downstream_modules.reranker.qwen3_reranker import (
    Qwen3RerankerHandler,
)

TOKEN_FALSE_ID = 3
TOKEN_TRUE_ID = 5


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
