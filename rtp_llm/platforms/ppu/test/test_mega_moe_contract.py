"""Graph lifetime and ownership contracts for the optional EP8 MegaMoE path."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.ppu_mega_moe import (
    PpuMegaMoEFP4Strategy,
    PpuMegaMoEResources,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import PpuMoeConfig


class MegaMoEContractTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PpuMoeConfig(
            layer_id=0,
            dim=4096,
            moe_inter_dim=2048,
            n_routed_experts=256,
            n_activated_experts=6,
            swiglu_limit=10.0,
            ep_size=8,
            ep_rank=0,
            n_local_experts=32,
            local_expert_start=0,
            local_expert_end=32,
            max_tokens_per_rank=128,
        )

    def test_only_supported_geometry_is_selected(self):
        self.assertTrue(PpuMegaMoEFP4Strategy.can_handle(self.cfg))
        for change in (
            {"tp_size": 2},
            {"ep_size": 4},
            {"dim": 7168},
            {"moe_inter_dim": 4096},
            {"n_activated_experts": 8},
        ):
            self.assertFalse(
                PpuMegaMoEFP4Strategy.can_handle(replace(self.cfg, **change))
            )
        with self.assertRaisesRegex(ValueError, "BF16 routed output"):
            PpuMegaMoEFP4Strategy(
                self.cfg, resources=PpuMegaMoEResources(), output_dtype=torch.float32
            )

    def test_backend_choice_is_a_model_snapshot(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        a = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_MOE_BACKEND"] = "deepep"
        b = PpuDecodeProvider(options)
        self.assertEqual(a._moe_backend, "megamoe")
        self.assertEqual(b._moe_backend, "deepep")
        self.assertEqual(a._mega_module_name, "deep_gemm_mega")
        c = PpuDecodeProvider(
            {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        )
        self.assertIsNot(a._mega_resources, c._mega_resources)
        with self.assertRaisesRegex(ValueError, "must be deep_gemm_mega"):
            PpuDecodeProvider(
                {
                    **DECODE_EXECUTION_OPTIONS,
                    "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe",
                    "DSV4_PPU_MEGAMOE_PACKAGE": "deep_gemm",
                }
            )
        with self.assertRaisesRegex(ValueError, "must be deepep or megamoe"):
            PpuDecodeProvider({**options, "DSV4_PPU_DECODE_MOE_BACKEND": "typo"})

    def test_dispatch_preserves_bf16_and_separate_reduction(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = object()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        x = torch.zeros((8, 4096), dtype=torch.bfloat16)
        indices = torch.zeros((8, 8), dtype=torch.int64)
        weights = torch.ones((8, 8))
        active_mask = torch.ones(8, dtype=torch.bool)
        strategy.forward(x, weights, indices, active_token_mask=active_mask)
        strategy._mega.mega_moe_pre_dispatch.assert_called_once()
        self.assertIs(
            strategy._mega.mega_moe_pre_dispatch.call_args.kwargs["active_token_mask"],
            active_mask,
        )
        selected = strategy._mega.fp4_fp4_mega_moe
        selected.assert_called_once()
        self.assertEqual(selected.call_args.kwargs, {
            "activation_clamp": 10.0, "preserve_bf16": True, "fuse_reduce": False,
        })

    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.group", new=SimpleNamespace(WORLD=object()))
    def test_eager_allocation_is_reused_in_graphs_and_across_layers(
        self, size, rank, capture
    ):
        resources = PpuMegaMoEResources()
        wrapper = SimpleNamespace(ll_num_max_token_per_rank=256)
        module = SimpleNamespace(
            get_symm_buffer_for_mega_moe=Mock(return_value=object())
        )
        first = resources.bind(self.cfg, wrapper, module)
        capture.return_value = True
        self.assertIs(
            first, resources.bind(replace(self.cfg, layer_id=42), wrapper, module)
        )
        module.get_symm_buffer_for_mega_moe.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "different workspace layouts"):
            resources.bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=512), module
            )

    @patch("torch.cuda.is_current_stream_capturing", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.group", new=SimpleNamespace(WORLD=object()))
    def test_first_allocation_inside_capture_is_rejected(self, size, rank, capture):
        module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock())
        with self.assertRaisesRegex(RuntimeError, "eager warmup"):
            PpuMegaMoEResources().bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=256), module
            )
        module.get_symm_buffer_for_mega_moe.assert_not_called()

    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.group", new=SimpleNamespace(WORLD=object()))
    def test_wrong_communicator_is_rejected_before_allocation(self, size, rank):
        with self.assertRaisesRegex(RuntimeError, "EP8 WORLD"):
            PpuMegaMoEResources().bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=256), Mock()
            )


if __name__ == "__main__":
    unittest.main()
