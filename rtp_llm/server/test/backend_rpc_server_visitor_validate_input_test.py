import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig, RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import GenerateInput


def _make_input(
    prompt_length: int, max_new_tokens: int, **generate_config_kwargs
) -> GenerateInput:
    return GenerateInput(
        request_id=1,
        token_ids=torch.arange(prompt_length, dtype=torch.int32),
        mm_inputs=[],
        generate_config=GenerateConfig(
            max_new_tokens=max_new_tokens, **generate_config_kwargs
        ),
    )


class _RecordingModelRpcClient:
    def __init__(self):
        self.inputs = []
        self.batches = []

    async def enqueue(self, input):
        self.inputs.append(input)
        if False:
            yield None

    async def batch_enqueue(self, inputs):
        self.batches.append(inputs)
        return []


class BackendRPCServerVisitorValidateInputTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_rpc_client = _RecordingModelRpcClient()
        pd_sep_config = SimpleNamespace(
            max_rpc_timeout_ms=0,
            decode_entrance=False,
            role_type=RoleType.FRONTEND,
            to_string=lambda: "test",
        )
        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.ModelRpcClient",
            return_value=self.model_rpc_client,
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.HostService"
        ) as host_service_type, patch(
            "rtp_llm.server.backend_rpc_server_visitor.MasterClient"
        ):
            host_service_type.return_value.service_available = False
            self.visitor = BackendRPCServerVisitor(
                max_seq_len=32,
                seq_size_per_block=1,
                pd_sep_config=pd_sep_config,
                addresses=[],
            )

    async def test_prefill_only_accepts_nonempty_prompt_without_generation_capacity(
        self,
    ):
        for prompt_length in (1, self.visitor.max_seq_len):
            with self.subTest(prompt_length=prompt_length):
                input = _make_input(prompt_length, max_new_tokens=0)
                stream = await self.visitor.enqueue(input)
                self.assertEqual([output async for output in stream], [])
                self.assertIs(self.model_rpc_client.inputs[-1], input)

    async def test_prefill_only_rejects_empty_or_overlong_prompt(self):
        for prompt_length in (0, self.visitor.max_seq_len + 1):
            with self.subTest(prompt_length=prompt_length):
                with self.assertRaises(FtRuntimeException) as ctx:
                    await self.visitor.enqueue(
                        _make_input(prompt_length, max_new_tokens=0)
                    )
                self.assertEqual(
                    ctx.exception.exception_type, ExceptionType.LONG_PROMPT_ERROR
                )
        self.assertEqual(self.model_rpc_client.inputs, [])

    async def test_enqueue_normalizes_prompt_scoring_before_rpc_forwarding(self):
        input = _make_input(1, max_new_tokens=1, return_prompt_logits=True)
        input.generate_config.max_new_tokens = 0
        input.generate_config.is_streaming = True
        input.generate_config.reuse_cache = True
        input.generate_config.can_use_pd_separation = True

        stream = await self.visitor.enqueue(input)
        self.assertEqual([output async for output in stream], [])

        forwarded_config = self.model_rpc_client.inputs[-1].generate_config
        self.assertEqual(forwarded_config.max_new_tokens, 1)
        self.assertTrue(forwarded_config.return_prompt_logits)
        self.assertFalse(forwarded_config.is_prefill_only())
        self.assertFalse(forwarded_config.is_streaming)
        self.assertFalse(forwarded_config.reuse_cache)
        self.assertFalse(forwarded_config.can_use_pd_separation)

    async def test_enqueue_rejects_invalid_prefill_only_generate_config(self):
        input = _make_input(1, max_new_tokens=1, return_logits=True)
        input.generate_config.max_new_tokens = 0

        with self.assertRaisesRegex(FtRuntimeException, "return_logits") as ctx:
            await self.visitor.enqueue(input)

        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertEqual(self.model_rpc_client.inputs, [])

    async def test_batch_enqueue_accepts_valid_prefill_only_generate_config(self):
        inputs = [
            _make_input(1, max_new_tokens=0),
            _make_input(self.visitor.max_seq_len, max_new_tokens=0),
        ]

        outputs = await self.visitor.batch_enqueue(inputs)

        self.assertEqual(outputs, [])
        self.assertEqual(self.model_rpc_client.batches, [inputs])

    async def test_batch_enqueue_rejects_invalid_prefill_only_generate_config(self):
        input = _make_input(1, max_new_tokens=1, return_logits=True)
        input.generate_config.max_new_tokens = 0

        with self.assertRaisesRegex(FtRuntimeException, "return_logits") as ctx:
            await self.visitor.batch_enqueue([input])

        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertEqual(self.model_rpc_client.batches, [])

    def test_negative_max_new_tokens_is_bad_request_at_config_construction(self):
        with self.assertRaises(FtRuntimeException) as ctx:
            GenerateConfig(max_new_tokens=-1)

        self.assertEqual(
            ctx.exception.exception_type.category, ExceptionCategory.BAD_REQUEST
        )

    async def test_positive_generation_still_requires_remaining_capacity(self):
        input = _make_input(self.visitor.max_seq_len - 1, max_new_tokens=1)
        stream = await self.visitor.enqueue(input)
        self.assertEqual([output async for output in stream], [])
        self.assertIs(self.model_rpc_client.inputs[-1], input)

        with self.assertRaises(FtRuntimeException) as ctx:
            await self.visitor.enqueue(
                _make_input(self.visitor.max_seq_len, max_new_tokens=1)
            )
        self.assertEqual(ctx.exception.exception_type, ExceptionType.LONG_PROMPT_ERROR)


if __name__ == "__main__":
    unittest.main()
