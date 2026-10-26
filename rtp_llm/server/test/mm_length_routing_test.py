import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient, trans_input
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalLengthsPB
from rtp_llm.ops import (
    MMPreprocessConfig,
    MultimodalInput,
    PDSepConfig,
    RoleType,
    VitSeparation,
)
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.mm_cache_routing import multimodal_length_tokens
from rtp_llm.utils.base_model_datatypes import GenerateInput, MultimodalTokenExpansion


class MMLatePrefillActivationTest(unittest.TestCase):
    def create_visitor(self, separation, flag="1", multimodal=True, decode=False):
        pd_config = PDSepConfig()
        pd_config.role_type = RoleType.PDFUSION
        pd_config.decode_entrance = decode
        module = "rtp_llm.server.backend_rpc_server_visitor"
        with (
            patch.dict(os.environ, {"RTP_LLM_LOCAL_MM_LATE_PREFILL": flag}),
            patch(f"{module}.ModelRpcClient"),
            patch(f"{module}.HostService"),
            patch(f"{module}.MasterClient"),
        ):
            return BackendRPCServerVisitor(
                max_seq_len=100,
                seq_size_per_block=16,
                pd_sep_config=pd_config,
                addresses=["127.0.0.1:8001", "127.0.0.1:8002"],
                vit_separation=separation,
                mm_model_config=SimpleNamespace() if multimodal else None,
            )

    def test_real_local_binding_activates_late_prefill(self):
        visitor = self.create_visitor(VitSeparation.VIT_SEPARATION_LOCAL)
        self.assertTrue(visitor.local_mm_late_prefill)

    def test_disabled_flag_keeps_eager_path(self):
        visitor = self.create_visitor(VitSeparation.VIT_SEPARATION_LOCAL, flag="0")
        self.assertFalse(visitor.local_mm_late_prefill)

    def test_other_topologies_and_text_models_do_not_activate(self):
        for separation in (
            None,
            VitSeparation.VIT_SEPARATION_ROLE,
            VitSeparation.VIT_SEPARATION_REMOTE,
        ):
            with self.subTest(separation=separation):
                self.assertFalse(self.create_visitor(separation).local_mm_late_prefill)
        self.assertFalse(
            self.create_visitor(
                VitSeparation.VIT_SEPARATION_LOCAL, multimodal=False
            ).local_mm_late_prefill
        )
        self.assertFalse(
            self.create_visitor(
                VitSeparation.VIT_SEPARATION_LOCAL, decode=True
            ).local_mm_late_prefill
        )


class MMLocalDPAffinityTest(unittest.IsolatedAsyncioTestCase):
    async def test_length_probe_and_prefill_share_one_dp_owner(self):
        client = ModelRpcClient(["127.0.0.1:8001", "127.0.0.1:8002"], {})
        self.addAsyncCleanup(client.close)
        request = GenerateInput(
            3,
            torch.tensor([1, 90, 5, 91, 2]),
            [MultimodalInput("video.mp4", 2, torch.empty(0), MMPreprocessConfig())],
            GenerateConfig(),
        )
        config = SimpleNamespace(mm_sep_tokens=[[90, 91]], include_sep_tokens=False)
        probe = AsyncMock(return_value=MultimodalLengthsPB(lengths=[7]))
        with patch.object(
            client._channel_pool, "get", new=AsyncMock()
        ) as channel, patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.RpcServiceStub",
            return_value=SimpleNamespace(GetMultimodalLengths=probe),
        ):
            await client.prepare_local_multimodal(request, config, 100)
        channel.assert_awaited_once_with("127.0.0.1:8002")
        self.assertEqual(request.generate_config.role_addrs[0].grpc_port, 8002)
        self.assertEqual(request.generate_config.role_addrs[0].role, RoleType.PDFUSION)
        self.assertTrue(request.mm_token_expansion.length_only)
        self.assertEqual(request.mm_token_expansion.spans, [(2, 7)])
        self.assertFalse(probe.call_args.args[0].HasField("multimodal_token_layout"))


class MMLengthRoutingTest(unittest.TestCase):
    def test_lengths_preserve_text_and_mark_wire_as_pending(self):
        tokens, spans = multimodal_length_tokens(
            [1, 90, 5, 91, 2, 90, 6, 91, 3], [[90, 91]], False, [2, 3], 100
        )
        self.assertEqual(list(tokens), [1, 90, 0, 0, 91, 2, 90, 0, 0, 0, 91, 3])
        self.assertEqual(spans, [(2, 2), (7, 3)])
        request = GenerateInput(1, torch.tensor([1]), [], GenerateConfig())
        request.mm_token_expansion = MultimodalTokenExpansion(
            tokens, spans, length_only=True
        )
        wire = trans_input(request)
        self.assertTrue(wire.multimodal_token_layout.length_only)
        self.assertEqual(list(wire.token_ids), list(tokens))

    def test_bad_lengths_and_oversize_are_rejected_before_allocation(self):
        for lengths in ([], [0], [-1], [True], [1 << 40], [3, 4], [99]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                multimodal_length_tokens(
                    [1, 90, 5, 91, 2], [[90, 91]], False, lengths, 100
                )

    def test_included_markers_are_counted_once(self):
        tokens, spans = multimodal_length_tokens(
            [1, 90, 5, 91, 2], [[90, 91]], True, [7], 100
        )
        self.assertEqual(list(tokens), [1] + [0] * 7 + [2])
        self.assertEqual(spans, [(1, 7)])


if __name__ == "__main__":
    unittest.main()
