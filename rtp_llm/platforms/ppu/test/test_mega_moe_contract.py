"""Graph lifetime and ownership contracts for the optional EP8 MegaMoE path."""

import unittest
import os
import sys
import tempfile
from pathlib import Path
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
    def test_transport_library_comes_from_isolated_wheel(self):
        from rtp_llm.platforms.ppu.models.dsv4 import ppu_mega_sparse as sparse

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'deep_gemm_mega'
            (package / 'lib').mkdir(parents=True)
            library = package / 'lib/libsignal_gin.so'
            library.touch()
            with patch.object(sparse, 'find_spec', return_value=SimpleNamespace(origin=str(package / '__init__.py'))), \
                 patch.object(sparse.C, 'CDLL') as load:
                self.assertIs(sparse._transport_library(library.name), load.return_value)
                load.assert_called_once_with(str(library))
                with self.assertRaisesRegex(RuntimeError, 'rebuild'):
                    sparse._transport_library('libsparse_gin.so')

    def test_transport_library_requires_installed_package(self):
        from rtp_llm.platforms.ppu.models.dsv4 import ppu_mega_sparse as sparse

        for package in (None, SimpleNamespace(origin=None)):
            with patch.object(sparse, 'find_spec', return_value=package):
                with self.assertRaisesRegex(RuntimeError, 'requires'):
                    sparse._transport_library('libsignal_gin.so')

    def test_signal_transport_is_opt_in_for_four_nodes(self):
        from rtp_llm.platforms.ppu.models.dsv4 import ppu_mega_sparse as sparse

        for nodes, enabled in ((1, '1'), (2, '1'), (4, '0'), (4, '1')):
            base = SimpleNamespace(nodes=nodes, node=0, dim=4096, experts=256 // nodes,
                                   cross_group=object())
            with patch.dict(os.environ, {'DSV4_PPU_MEGAMOE_SIGNAL_ONLY': enabled}), \
                 patch.object(sparse, 'PpuMegaCommunicator'), \
                 patch.object(sparse, 'Exchange') as original, \
                 patch.object(sparse, 'SignalExchange') as signal, \
                 patch.object(torch, 'empty'), patch.object(torch, 'empty_like'):
                sparse.SparseHierarchy(base, 12)
                expected, unused = (signal, original) if nodes == 4 and enabled == '1' else (original, signal)
                self.assertEqual(expected.call_count, 2)
                unused.assert_not_called()

    @patch('torch.cuda.is_current_stream_capturing', return_value=False)
    @patch('torch.distributed.get_rank', return_value=0)
    @patch('torch.distributed.get_world_size', return_value=8)
    @patch('torch.distributed.group', new=SimpleNamespace(WORLD=object()))
    def test_signal_flag_keeps_ep8_resources_local(self, size, rank, capture):
        hierarchy = Mock(side_effect=AssertionError('EP8 must not create a hierarchy'))
        module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock(return_value=object()))
        with patch.dict(os.environ, {'DSV4_PPU_MEGAMOE_SIGNAL_ONLY': '1'}), \
             patch.dict(sys.modules, {
                 'rtp_llm.platforms.ppu.models.dsv4.ppu_mega_hierarchy':
                     SimpleNamespace(PpuMegaMoEHierarchy=hierarchy)}):
            PpuMegaMoEResources().bind(self.cfg, SimpleNamespace(ll_num_max_token_per_rank=128), module)
        hierarchy.assert_not_called()
        module.get_symm_buffer_for_mega_moe.assert_called_once()

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
        for size in (8, 16, 32):
            for tp in (1,):
                self.assertTrue(PpuMegaMoEFP4Strategy.can_handle(replace(
                    self.cfg, ep_size=size, tp_size=tp, n_local_experts=256 // size,
                    local_expert_end=256 // size,
                )))
        for change in (
            {"tp_size": 2},
            {"tp_size": 8},
            {"ep_size": 16, "tp_size": 16},
            {"ep_size": 32, "tp_size": 32},
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

    def test_large_ep_capacity_uses_speculative_admission_bound(self):
        cases = ((16, 1, 12, "collective", 16), (32, 1, 12, "collective", 16),
                 (16, 1, 24, "collective", 32), (32, 1, 48, "collective", 48),
                 (32, 1, 12, "sparse", 12), (32, 1, 1, "sparse", 1))
        cases = [(size, tp, logical, transport, '0', capacity)
                 for size, tp, logical, transport, capacity in cases] + [
                     (16, 1, 12, 'collective', '1', 12),
                     (16, 1, 15, 'collective', '1', 15),
                     (32, 1, 12, 'collective', '1', 16),
                     (32, 1, 12, 'sparse', '1', 12)]
        for size, tp, logical, transport, exact, capacity in cases:
            local_group = object()
            hierarchy = Mock(return_value=SimpleNamespace(
                local_group=local_group, nodes=size // 8, experts=256 // (size // 8)
            ))
            fake = SimpleNamespace(PpuMegaMoEHierarchy=hierarchy)
            with patch.dict(os.environ, {"DSV4_PPU_MEGAMOE_TRANSPORT": transport,
                                        "DSV4_PPU_EP16_EXACT_CAPACITY": exact}), patch.dict(sys.modules, {
                "rtp_llm.platforms.ppu.models.dsv4.ppu_mega_hierarchy": fake
            }), patch("torch.distributed.group", SimpleNamespace(WORLD=object())), \
                patch("torch.distributed.get_world_size", return_value=size), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.cuda.is_current_stream_capturing", return_value=False):
                cfg = replace(self.cfg, ep_size=size, tp_size=tp, n_local_experts=256 // size,
                              local_expert_end=256 // size)
                wrapper = SimpleNamespace(ll_num_max_token_per_rank=64,
                                          config=SimpleNamespace(ll_num_max_token=logical))
                module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock())
                PpuMegaMoEResources().bind(cfg, wrapper, module)
                hierarchy.assert_called_once_with(cfg, capacity)
                module.get_symm_buffer_for_mega_moe.assert_called_once_with(
                    local_group, 256 // (size // 8), capacity * (size // 8), 8, 4096, 2048
                )

    def test_ep8_does_not_import_cross_host_transport(self):
        self._assert_ep8_transport_isolation("direct")

    def test_ep8_does_not_import_sparse_transport(self):
        self._assert_ep8_transport_isolation("sparse")

    def _assert_ep8_transport_isolation(self, transport):
        group = object()
        module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock())
        with patch.dict(os.environ, {"DSV4_PPU_MEGAMOE_TRANSPORT": transport,
                                    "DSV4_PPU_EP16_EXACT_CAPACITY": "1"}), \
            patch.dict(sys.modules, {
                "rtp_llm.platforms.ppu.models.dsv4.ppu_mega_hierarchy": None,
                "rtp_llm.platforms.ppu.models.dsv4.ppu_mega_collective": None,
                "rtp_llm.platforms.ppu.models.dsv4.ppu_mega_sparse": None,
            }), patch("torch.distributed.group", SimpleNamespace(WORLD=group)), \
            patch("torch.distributed.get_world_size", return_value=8), \
            patch("torch.distributed.get_rank", return_value=0), \
            patch("torch.cuda.is_current_stream_capturing", return_value=False):
            resources = PpuMegaMoEResources()
            resources.bind(self.cfg, SimpleNamespace(ll_num_max_token_per_rank=256), module)
            self.assertIsNone(resources.hierarchy)
            module.get_symm_buffer_for_mega_moe.assert_called_once_with(
                group, 256, 256, 8, 4096, 2048
            )

    def test_large_ep_tile_hint_excludes_transport_padding(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = replace(self.cfg, ep_size=32, n_local_experts=8,
                               local_expert_end=8)
        strategy._sym = object()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        x = torch.zeros((6, 4096), dtype=torch.bfloat16)
        padded = torch.zeros((64, 4096), dtype=torch.bfloat16)
        ids = torch.zeros((6, 8), dtype=torch.int64)
        weights = torch.ones((6, 8))
        hierarchy = Mock()
        hierarchy.dispatch.return_value = (padded, torch.ones((64, 8)),
                                            torch.zeros((64, 8), dtype=torch.int64))
        strategy._resources = SimpleNamespace(hierarchy=hierarchy)
        strategy.forward(x, weights, ids)
        self.assertEqual(strategy._mega.fp4_fp4_mega_moe.call_args.kwargs["expected_m"], 5)
        self.assertEqual(hierarchy.combine.call_args.args[1], 6)

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
        with self.assertRaisesRegex(RuntimeError, "EP WORLD"):
            PpuMegaMoEResources().bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=256), Mock()
            )


if __name__ == "__main__":
    unittest.main()
