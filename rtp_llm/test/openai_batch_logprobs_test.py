"""Regression coverage for OpenAI batch logprobs and K3 reasoning boundaries."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ReturnAllProbsMode
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_OPEN,
    THINK_TO_RESPONSE,
)
from rtp_llm.openai.api_datatype import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatMessage,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _Tokenizer:
    """Small vocabulary with native-style, indivisible XTML control tokens."""

    eos_token_id = 0
    chat_template = "{{ messages[-1]['content'] }}"

    def __init__(self):
        self.vocab = [chr(index) for index in range(128)] + [
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "<|end_of_msg|>",
            "think",
            "response",
            "message",
        ]
        self.vocab[0] = "<eos>"
        self.ids = {text: index for index, text in enumerate(self.vocab)}
        self.pieces = sorted(self.ids, key=len, reverse=True)

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)

    def tokenize(self, text):
        return [self.vocab[token] for token in self.encode(text)]

    def encode(self, text, **_kwargs):
        result = []
        while text:
            piece = next(piece for piece in self.pieces if text.startswith(piece))
            result.append(self.ids[piece])
            text = text[len(piece) :]
        return result

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.vocab[int(token)] for token in token_ids)

    def apply_chat_template(self, _messages, *, thinking, **_kwargs):
        return self.encode("prompt" + (THINK_OPEN if thinking else RESPONSE_OPEN))


class BatchLogprobsReproTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()
        self.backend = SimpleNamespace(batch_enqueue=AsyncMock(), enqueue=AsyncMock())

    def _endpoint(self, model_type="test"):
        return OpenaiEndpoint(
            model_config=SimpleNamespace(
                model_type=model_type,
                model_name=model_type,
                max_seq_len=256,
                template_type=None,
                ckpt_path="",
                render_config=RenderConfig(),
                generate_env_config=GenerateEnvConfig(),
                special_tokens=SimpleNamespace(
                    eos_token_id=0, stop_words_id_list=[], stop_words_str_list=[]
                ),
            ),
            misc_config=PyMiscellaneousConfig(),
            vit_config=VitConfig(),
            tokenizer=self.tokenizer,
            backend_rpc_server_visitor=self.backend,
        )

    def _output(self, text, *, output_len=None, finished=True, **probabilities):
        ids = self.tokenizer.encode(text)
        return GenerateOutputs(
            [
                GenerateOutput(
                    output_ids=torch.tensor([ids], dtype=torch.int32),
                    aux_info=AuxInfo(input_len=4, output_len=output_len or len(ids)),
                    finished=finished,
                    **probabilities,
                )
            ]
        )

    @staticmethod
    def _request(**options):
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test")], **options
        )

    async def test_batch_compact_logprobs_preserve_distinct_rows_and_order(self):
        first, second = self.tokenizer.encode("AB")
        self.backend.batch_enqueue.return_value = [
            self._output("X"),
            self._output(
                "AB",
                token_logprobs=torch.tensor([-0.25, -0.75]),
                top_logprob_token_ids=torch.tensor([[first, second], [second, first]]),
                top_logprobs=torch.tensor([[-0.25, -1.5], [-0.75, -1.0]]),
                logprobs_offset=0,
                logprobs_count=2,
            ),
        ]
        plain, scored = await self._endpoint().batch_chat_completion(
            100,
            BatchChatCompletionRequest(
                requests=[self._request(), self._request(logprobs=True, top_logprobs=2)]
            ),
        )

        self.assertEqual(plain.choices[0].message.content, "X")
        self.assertIsNone(plain.choices[0].logprobs)
        choice = scored.choices[0]
        self.assertEqual(choice.message.content, "AB")
        self.assertEqual(
            [(row.token, row.logprob) for row in choice.logprobs.content],
            [("A", -0.25), ("B", -0.75)],
        )
        self.assertEqual(
            [
                [(item.token, item.logprob) for item in row.top_logprobs]
                for row in choice.logprobs.content
            ],
            [[("A", -0.25), ("B", -1.5)], [("B", -0.75), ("A", -1.0)]],
        )
        self.backend.batch_enqueue.assert_awaited_once()
        inputs = self.backend.batch_enqueue.await_args.args[0]
        self.assertEqual([item.request_id for item in inputs], [100, 101])
        self.assertFalse(inputs[0].generate_config.return_logprobs)
        self.assertTrue(inputs[1].generate_config.return_logprobs)
        self.assertEqual(inputs[1].generate_config.top_logprobs, 2)
        self.backend.enqueue.assert_not_awaited()

    async def test_batch_legacy_logprobs_preserve_incremental_rows(self):
        async def outputs():
            for i, (token, probability) in enumerate((("A", 0.75), ("B", 0.625)), 1):
                row = torch.zeros((1, len(self.tokenizer)))
                row[0, self.tokenizer.ids[token]] = probability
                row[0, self.tokenizer.ids["X"]] = 1 - probability
                yield self._output(token, output_len=i, finished=i == 2, all_probs=row)

        legacy = {"logprobs": True, "logprobs_mode": "default", "top_logprobs": 1}
        self.backend.enqueue.side_effect = [
            outputs(),
            FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "speculative decoding does not support return_all_probs",
            ),
        ]
        self.backend.batch_enqueue.return_value = [self._output("X")]
        response, plain, rejected = await self._endpoint().batch_chat_completion(
            100,
            BatchChatCompletionRequest(
                requests=[
                    self._request(**legacy),
                    self._request(),
                    self._request(**legacy),
                ]
            ),
        )

        choice = response.choices[0]
        self.assertEqual(choice.message.content, "AB")
        self.assertEqual([row.token for row in choice.logprobs.content], ["A", "B"])
        for row, expected in zip(choice.logprobs.content, (0.75, 0.625), strict=True):
            self.assertAlmostEqual(row.logprob, math.log(expected))
        self.assertEqual(plain.choices[0].message.content, "X")
        self.assertEqual(
            rejected["error"]["error_code"], ExceptionType.UNSUPPORTED_OPERATION
        )
        self.backend.batch_enqueue.assert_awaited_once()
        self.assertEqual(
            self.backend.batch_enqueue.await_args.args[0][0].request_id, 101
        )
        legacy_inputs = [call.args[0] for call in self.backend.enqueue.await_args_list]
        self.assertEqual([item.request_id for item in legacy_inputs], [100, 102])
        config = legacy_inputs[0].generate_config
        self.assertEqual(config.return_all_probs, ReturnAllProbsMode.DEFAULT)
        self.assertTrue(config.is_streaming)

    async def test_k3_batch_compact_logprobs_start_after_reasoning(self):
        prefix = "r" + THINK_TO_RESPONSE
        suffix = "AB" + RESPONSE_CLOSE + "<|close|>message<|sep|>"
        offset, count = len(self.tokenizer.encode(prefix)), len(
            self.tokenizer.encode(suffix)
        )
        self.backend.batch_enqueue.return_value = [
            self._output(
                prefix + suffix,
                token_logprobs=torch.tensor([-0.25, -0.75] + [-1.0] * (count - 2)),
                top_logprob_token_ids=torch.empty((count, 0), dtype=torch.int32),
                top_logprobs=torch.empty((count, 0)),
                logprobs_offset=offset,
                logprobs_count=count,
            )
        ]
        (response,) = await self._endpoint("kimi_k3").batch_chat_completion(
            100,
            BatchChatCompletionRequest(
                requests=[
                    self._request(enable_thinking=True, logprobs=True, top_logprobs=0)
                ]
            ),
        )

        choice = response.choices[0]
        self.assertEqual(choice.message.reasoning_content, "r")
        self.assertEqual(choice.message.content, "AB")
        self.assertEqual(
            [(row.token, row.logprob) for row in choice.logprobs.content],
            [("A", -0.25), ("B", -0.75)],
        )
        config = self.backend.batch_enqueue.await_args.args[0][0].generate_config
        self.assertTrue(config.return_logprobs)
        self.assertFalse(config.is_streaming)


if __name__ == "__main__":
    unittest.main()
