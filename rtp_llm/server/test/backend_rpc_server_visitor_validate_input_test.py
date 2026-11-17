import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from pydantic import ValidationError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.host_service import HostServiceArgs
from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType


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
        self.visitor = self._make_visitor()

    def _make_visitor(self, vit_separation=None, vit_domain=""):
        server_config = ServerConfig()
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
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.HostServiceArgs.create_from_env",
            return_value=HostServiceArgs(
                prefill_domain="prefill", decode_domain="decode", vit_domain=vit_domain
            ),
        ):
            host_service_type.return_value.service_available = False
            return BackendRPCServerVisitor(
                max_seq_len=32,
                seq_size_per_block=1,
                pd_sep_config=pd_sep_config,
                addresses=[],
                server_config=server_config,
                vit_separation=vit_separation,
            )

    def _multimodal_input(self, mm_type=MMUrlType.IMAGE, **kwargs):
        input = _make_input(1, **kwargs)
        input.mm_inputs = [
            MultimodalInput(
                "https://example.invalid/media",
                mm_type,
                torch.empty(0),
                MMPreprocessConfig(),
            )
        ]
        return input

    async def test_remote_vit_without_domain_rejects_before_routing_or_rpc(self):
        visitor = self._make_visitor(vit_separation=VitSeparation.VIT_SEPARATION_REMOTE)
        visitor.host_service.service_available = True
        visitor.route_ips = AsyncMock()
        for mm_type in MMUrlType:
            for is_streaming in (False, True):
                for max_new_tokens in (0, 1):
                    with self.subTest(
                        mm_type=mm_type,
                        is_streaming=is_streaming,
                        max_new_tokens=max_new_tokens,
                    ):
                        input = self._multimodal_input(
                            mm_type,
                            max_new_tokens=max_new_tokens,
                            is_streaming=is_streaming,
                        )
                        with self.assertRaisesRegex(
                            FtRuntimeException, "VIT domain.*VIT_SEPARATION=2"
                        ) as ctx:
                            await visitor.enqueue(input)
                        self.assertEqual(
                            ctx.exception.exception_type,
                            ExceptionType.MM_NOT_SUPPORTED_ERROR,
                        )
        visitor.route_ips.assert_not_awaited()
        self.assertEqual(self.model_rpc_client.inputs, [])

    async def test_remote_vit_without_domain_rejects_entire_batch_before_routing(self):
        visitor = self._make_visitor(vit_separation=VitSeparation.VIT_SEPARATION_REMOTE)
        visitor.host_service.service_available = True
        visitor.route_ips = AsyncMock()
        inputs = [
            _make_input(1, max_new_tokens=1),
            self._multimodal_input(max_new_tokens=1),
        ]
        with self.assertRaises(FtRuntimeException) as ctx:
            await visitor.batch_enqueue(inputs)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.MM_NOT_SUPPORTED_ERROR
        )
        visitor.route_ips.assert_not_awaited()
        self.assertEqual(self.model_rpc_client.batches, [])

    async def test_remote_vit_without_domain_forwards_text_single_and_batch(self):
        visitor = self._make_visitor(vit_separation=VitSeparation.VIT_SEPARATION_REMOTE)
        input = _make_input(1, max_new_tokens=1)
        stream = await visitor.enqueue(input)
        self.assertEqual([output async for output in stream], [])
        self.assertEqual(self.model_rpc_client.inputs, [input])
        self.assertEqual(await visitor.batch_enqueue([input]), [])
        self.assertEqual(self.model_rpc_client.batches, [[input]])

    async def test_other_vit_configurations_preserve_multimodal_requests(self):
        for separation, domain in (
            (None, ""),
            (VitSeparation.VIT_SEPARATION_LOCAL, ""),
            (VitSeparation.VIT_SEPARATION_LOCAL, "vit"),
            (VitSeparation.VIT_SEPARATION_ROLE, ""),
            (VitSeparation.VIT_SEPARATION_REMOTE, "vit"),
        ):
            with self.subTest(separation=separation, domain=domain):
                visitor = self._make_visitor(
                    vit_separation=separation, vit_domain=domain
                )
                input = self._multimodal_input(max_new_tokens=1)
                stream = await visitor.enqueue(input)
                self.assertEqual([output async for output in stream], [])
                self.assertIs(self.model_rpc_client.inputs[-1], input)
                self.assertEqual(await visitor.batch_enqueue([input]), [])
                self.assertEqual(self.model_rpc_client.batches[-1], [input])

    def test_remote_vit_readiness_requires_vit_only_with_configured_domain(self):
        roles = [
            RoleAddr(role=role, ip="127.0.0.1", http_port=8000, grpc_port=8001)
            for role in (RoleType.DECODE, RoleType.PREFILL)
        ]
        for domain in ("", "vit"):
            with self.subTest(domain=domain):
                visitor = self._make_visitor(
                    vit_separation=VitSeparation.VIT_SEPARATION_REMOTE,
                    vit_domain=domain,
                )
                visitor.host_service.get_backend_role_addrs.return_value = roles
                self.assertEqual(
                    RoleType.VIT in visitor.backend_role_list, bool(domain)
                )
                self.assertEqual(visitor.is_backend_service_ready(), not domain)
                visitor.host_service.get_backend_role_addrs.return_value = roles[:1]
                self.assertFalse(visitor.is_backend_service_ready())
                visitor.host_service.get_backend_role_addrs.return_value = roles + [
                    RoleAddr(
                        role=RoleType.VIT,
                        ip="127.0.0.1",
                        http_port=8000,
                        grpc_port=8001,
                    )
                ]
                self.assertTrue(visitor.is_backend_service_ready())

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
        with self.assertRaises(ValidationError) as ctx:
            GenerateConfig(max_new_tokens=-1)

        errors = ctx.exception.errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["loc"], ("max_new_tokens",))
        self.assertIn(
            "max_new_tokens must be greater than or equal to 0, got -1",
            errors[0]["msg"],
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
