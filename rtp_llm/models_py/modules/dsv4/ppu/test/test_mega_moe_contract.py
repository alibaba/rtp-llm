"""Graph lifetime and ownership contracts for the optional EP8 MegaMoE path."""

import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from rtp_llm.models_py.modules.dsv4.ppu.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_moe import (
    PpuMegaMoEFP4Strategy,
    PpuMegaMoEFP8Strategy,
    PpuMegaMoEW4A8Strategy,
    PpuMegaMoEResources,
    promote_mxfp4_to_fp8,
)
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_moe_config import PpuMoeConfig
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import (
    PpuSharedExpertExecutor,
)
from rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider import PpuDecodeProvider


class MegaMoEContractTest(unittest.TestCase):
    def test_tp_exact_combines_fp32_branches_after_collective(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_tp_moe import PpuTPMoE

        model = PpuTPMoE.__new__(PpuTPMoE)
        torch.nn.Module.__init__(model)
        model._fp32_reduce = True
        model._shared_phases = False
        model.route_scale = 2.5
        model.gate = Mock(return_value=(torch.ones(1, 6), torch.zeros(1, 6)))
        routed = torch.tensor([[1.003, -2.001]], dtype=torch.float32)
        shared = torch.tensor([[0.002, 0.003]], dtype=torch.float32)
        model.shared_experts = Mock(return_value=shared)
        model._strategy = Mock(return_value=routed)
        model._reduce = Mock(side_effect=lambda branches: branches * 4)
        result = model._forward_local_chunk(torch.ones(1, 2), None)
        self.assertEqual(model._reduce.call_args.args[0].dtype, torch.float32)
        self.assertEqual(tuple(model._reduce.call_args.args[0].shape), (2, 1, 2))
        self.assertEqual(model._strategy.call_args.kwargs["route_scale"], 2.5)
        self.assertTrue(torch.equal(result, (routed * 4 + shared * 4).bfloat16()))

    @staticmethod
    def _exact_module():
        capability = {
            "output_dtype": "float32",
            "preweighted": True,
            "route_weight_stage": "before_l2_quant",
            "g1_rounding": "bf16",
            "activation_quantization": "block128_ue8m0_eps1e-4",
            "input_activation_scale": "block128_ue8m0_fp32_storage",
            "g2_activation_scale": "block128_ue8m0_fp32_storage",
        }
        return SimpleNamespace(
            MEGA_MOE_CAPABILITIES={"fp8_fp8_mega_moe_fp32": capability},
            get_symm_buffer_for_mega_moe=Mock(),
            transform_weights_for_fp8_mega_moe=Mock(),
            mega_moe_pre_dispatch_ue8m0=Mock(),
            fp8_fp8_mega_moe_fp32=Mock(),
        )

    @staticmethod
    def _w4a8_module():
        capability = {
            "output_dtype": "float32",
            "preweighted": True,
            "weight_dtype": "mxfp4_k32_ue8m0",
            "route_weight_stage": "before_l2_quant",
            "g1_rounding": "bf16",
            "activation_quantization": "block128_ue8m0_eps1e-4",
            "input_activation_scale": "block128_ue8m0_fp32_storage",
            "g2_activation_scale": "block128_ue8m0_fp32_storage",
        }
        return SimpleNamespace(
            MEGA_MOE_CAPABILITIES={"fp8_fp4_mega_moe_fp32": capability},
            get_symm_buffer_for_mega_moe=Mock(),
            transform_weights_for_w4a8_mega_moe=Mock(),
            mega_moe_pre_dispatch_ue8m0=Mock(),
            fp8_fp4_mega_moe_fp32=Mock(),
        )

    def test_w4a8_requires_the_full_capability_contract(self):
        module = self._w4a8_module()
        self.assertTrue(PpuMegaMoEW4A8Strategy.package_has_exact_capability(module))
        module.MEGA_MOE_CAPABILITIES["fp8_fp4_mega_moe_fp32"]["weight_dtype"] = "fp8"
        self.assertFalse(PpuMegaMoEW4A8Strategy.package_has_exact_capability(module))
        module = self._w4a8_module()
        module.fp8_fp4_mega_moe_fp32 = None
        self.assertFalse(PpuMegaMoEW4A8Strategy.package_has_exact_capability(module))
        self.assertFalse(
            PpuMegaMoEW4A8Strategy.package_has_exact_capability(self._exact_module())
        )

    def test_w4a8_forward_pads_routes_and_uses_exact_kernel(self):
        strategy = object.__new__(PpuMegaMoEW4A8Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._max_tokens = 96
        strategy._sym = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        strategy._resources = Mock()
        for name in ("_w13", "_s13", "_inv13", "_w2", "_s2", "_inv2"):
            setattr(strategy, name, object())
        strategy._mega = self._w4a8_module()
        padded_ids = torch.zeros((2, 8), dtype=torch.int64)
        padded_weights = torch.ones((2, 8), dtype=torch.float32)
        with patch(
            "rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding.pad_inactive_routes",
            return_value=(padded_ids, padded_weights),
        ) as padding:
            result = strategy(
                torch.zeros((2, 4096), dtype=torch.bfloat16),
                torch.ones((2, 6), dtype=torch.float32),
                torch.zeros((2, 6), dtype=torch.int32),
            )
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(padding.call_args.args[0].dtype, torch.int64)
        self.assertEqual(padding.call_args.args[2], 8)
        published = strategy._mega.mega_moe_pre_dispatch_ue8m0.call_args.args[1]
        self.assertEqual(published.shape, (2, 8))
        self.assertEqual(published.dtype, torch.int64)
        call = strategy._mega.fp8_fp4_mega_moe_fp32.call_args
        self.assertEqual(len(call.args[1]), 3)
        self.assertEqual(len(call.args[2]), 3)
        self.assertFalse(call.kwargs["fast_math"])

    def test_w4a8_rejects_inputs_above_the_token_cap(self):
        strategy = object.__new__(PpuMegaMoEW4A8Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._max_tokens = 96
        strategy._sym = object()
        strategy._resources = Mock()
        strategy._mega = self._w4a8_module()
        with self.assertRaisesRegex(ValueError, "W4A8_MAX_TOKENS"):
            strategy(
                torch.zeros((97, 4096), dtype=torch.bfloat16),
                torch.ones((97, 8), dtype=torch.float32),
                torch.zeros((97, 8), dtype=torch.int64),
            )

    def test_exact_routea_requires_the_full_capability_contract(self):
        module = self._exact_module()
        self.assertTrue(PpuMegaMoEFP8Strategy.package_has_exact_capability(module))
        module.MEGA_MOE_CAPABILITIES["fp8_fp8_mega_moe_fp32"][
            "route_weight_stage"
        ] = "combine"
        self.assertFalse(PpuMegaMoEFP8Strategy.package_has_exact_capability(module))

    def test_mxfp4_promotion_is_lossless_and_block128(self):
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is None:
            self.skipTest("torch lacks float8_e8m0fnu")
        packed = torch.full((1, 128, 64), 0x22, dtype=torch.uint8)
        raw_scale = torch.ones((1, 128, 4), dtype=e8m0)
        weight, scale = promote_mxfp4_to_fp8(packed, raw_scale)
        self.assertEqual(weight.shape, (1, 128, 128))
        self.assertEqual(scale.shape, (1, 1, 1))
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(
            torch.equal(weight.float() * scale.item(), torch.ones(1, 128, 128))
        )

    def test_exact_routea_accepts_int32_and_int64_and_emits_fp32(self):
        strategy = object.__new__(PpuMegaMoEFP8Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        strategy._resources = Mock()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = self._exact_module()
        x = torch.zeros((2, 4096), dtype=torch.bfloat16)
        weights = torch.ones((2, 8), dtype=torch.float32)
        for dtype in (torch.int32, torch.int64):
            strategy._mega.mega_moe_pre_dispatch_ue8m0.reset_mock()
            strategy._mega.fp8_fp8_mega_moe_fp32.reset_mock()
            result = strategy(x, weights, torch.zeros((2, 8), dtype=dtype))
            self.assertEqual(result.dtype, torch.float32)
            published = strategy._mega.mega_moe_pre_dispatch_ue8m0.call_args.args[1]
            self.assertEqual(published.dtype, dtype)
            call = strategy._mega.fp8_fp8_mega_moe_fp32.call_args
            self.assertTrue(call.kwargs["fast_math"])
            self.assertNotIn("quantization", call.kwargs)

    def test_exact_routea_pads_six_int32_routes(self):
        strategy = object.__new__(PpuMegaMoEFP8Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        strategy._resources = Mock()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = self._exact_module()
        padded_ids = torch.zeros((2, 8), dtype=torch.int64)
        padded_weights = torch.ones((2, 8), dtype=torch.float32)
        with patch(
            "rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding.pad_inactive_routes",
            return_value=(padded_ids, padded_weights),
        ) as padding:
            strategy(
                torch.zeros((2, 4096), dtype=torch.bfloat16),
                torch.ones((2, 6), dtype=torch.float32),
                torch.zeros((2, 6), dtype=torch.int32),
            )
        self.assertEqual(padding.call_args.args[0].dtype, torch.int64)
        published = strategy._mega.mega_moe_pre_dispatch_ue8m0.call_args.args[1]
        self.assertEqual(published.shape, (2, 8))
        self.assertEqual(published.dtype, torch.int64)

    def test_provider_selects_routea_only_for_supported_config(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        provider = PpuDecodeProvider(options)
        topologies = ((8, 1, 4096), (16, 1, 4096), (8, 2, 4096), (8, 1, 8192))
        for available in (False, True):
            for ep, tp, dim in topologies:
                cfg = replace(
                    self.cfg,
                    ep_size=ep,
                    tp_size=tp,
                    dim=dim,
                    n_local_experts=256 // ep,
                    local_expert_end=256 // ep,
                )
                with (
                    self.subTest(available=available, ep=ep, tp=tp, dim=dim),
                    patch.object(
                        PpuMegaMoEW4A8Strategy,
                        "package_is_available",
                        return_value=False,
                    ),
                    patch.object(
                        PpuMegaMoEFP8Strategy,
                        "package_is_available",
                        return_value=available,
                    ),
                    patch.object(
                        PpuMegaMoEFP8Strategy, "__init__", return_value=None
                    ) as exact,
                    patch.object(
                        PpuMegaMoEFP4Strategy, "__init__", return_value=None
                    ) as fallback,
                ):
                    strategy = provider._build_moe_strategy(cfg)
                if available and (ep, tp, dim) == (8, 1, 4096):
                    self.assertIsInstance(strategy, PpuMegaMoEFP8Strategy)
                    self.assertEqual(
                        set(exact.call_args.kwargs), {"resources", "module_name"}
                    )
                    self.assertEqual(
                        exact.call_args.kwargs["module_name"], "deep_gemm_route_a"
                    )
                    fallback.assert_not_called()
                else:
                    self.assertIsInstance(strategy, PpuMegaMoEFP4Strategy)
                    self.assertEqual(
                        fallback.call_args.kwargs["module_name"], "deep_gemm_mega"
                    )
                    self.assertEqual(
                        fallback.call_args.kwargs["expected_m_policy"], provider._moe_hint
                    )
                    exact.assert_not_called()

    def test_provider_prefers_w4a8_then_routea_then_legacy(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        provider = PpuDecodeProvider(options)
        with (
            patch.object(
                PpuMegaMoEW4A8Strategy, "package_is_available", return_value=True
            ),
            patch.object(PpuMegaMoEW4A8Strategy, "__init__", return_value=None) as w4a8,
            patch.object(PpuMegaMoEFP8Strategy, "__init__", return_value=None) as routea,
            patch.object(PpuMegaMoEFP4Strategy, "__init__", return_value=None) as legacy,
        ):
            strategy = provider._build_moe_strategy(self.cfg)
        self.assertIsInstance(strategy, PpuMegaMoEW4A8Strategy)
        self.assertEqual(w4a8.call_args.kwargs["module_name"], "deep_gemm_route_a")
        self.assertEqual(w4a8.call_args.kwargs["max_tokens"], 96)
        routea.assert_not_called()
        legacy.assert_not_called()
        with (
            patch.object(
                PpuMegaMoEW4A8Strategy, "package_is_available", return_value=False
            ),
            patch.object(
                PpuMegaMoEFP8Strategy, "package_is_available", return_value=True
            ),
            patch.object(PpuMegaMoEW4A8Strategy, "__init__", return_value=None) as w4a8,
            patch.object(PpuMegaMoEFP8Strategy, "__init__", return_value=None) as routea,
            patch.object(PpuMegaMoEFP4Strategy, "__init__", return_value=None) as legacy,
        ):
            strategy = provider._build_moe_strategy(self.cfg)
        self.assertIsInstance(strategy, PpuMegaMoEFP8Strategy)
        self.assertEqual(routea.call_args.kwargs["module_name"], "deep_gemm_route_a")
        w4a8.assert_not_called()
        legacy.assert_not_called()
        with (
            patch.object(
                PpuMegaMoEW4A8Strategy, "package_is_available", return_value=False
            ),
            patch.object(
                PpuMegaMoEFP8Strategy, "package_is_available", return_value=False
            ),
            patch.object(PpuMegaMoEW4A8Strategy, "__init__", return_value=None) as w4a8,
            patch.object(PpuMegaMoEFP8Strategy, "__init__", return_value=None) as routea,
            patch.object(PpuMegaMoEFP4Strategy, "__init__", return_value=None) as legacy,
        ):
            strategy = provider._build_moe_strategy(self.cfg)
        self.assertIsInstance(strategy, PpuMegaMoEFP4Strategy)
        self.assertEqual(legacy.call_args.kwargs["module_name"], "deep_gemm_mega")
        w4a8.assert_not_called()
        routea.assert_not_called()

    def test_provider_exact_backend_switch_can_force_routea(self):
        options = {
            **DECODE_EXECUTION_OPTIONS,
            "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe",
            "DSV4_PPU_MEGAMOE_EXACT_BACKEND": "routea",
        }
        provider = PpuDecodeProvider(options)
        with (
            patch.object(
                PpuMegaMoEW4A8Strategy, "package_is_available", return_value=True
            ),
            patch.object(
                PpuMegaMoEFP8Strategy, "package_is_available", return_value=True
            ),
            patch.object(PpuMegaMoEW4A8Strategy, "__init__", return_value=None) as w4a8,
            patch.object(PpuMegaMoEFP8Strategy, "__init__", return_value=None) as routea,
        ):
            strategy = provider._build_moe_strategy(self.cfg)
        self.assertIsInstance(strategy, PpuMegaMoEFP8Strategy)
        routea.assert_called_once()
        w4a8.assert_not_called()

    def test_provider_validates_exact_backend_and_w4a8_cap(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        self.assertEqual(PpuDecodeProvider(options)._exact_backend, "w4a8")
        self.assertEqual(
            PpuDecodeProvider(
                {**options, "DSV4_PPU_MEGAMOE_W4A8_MAX_TOKENS": "48"}
            )._w4a8_max_tokens,
            48,
        )
        with self.assertRaisesRegex(ValueError, "w4a8 or routea"):
            PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_EXACT_BACKEND": "fp4"})
        for value in ("0", "-1", "abc"):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                PpuDecodeProvider(
                    {**options, "DSV4_PPU_MEGAMOE_W4A8_MAX_TOKENS": value}
                )

    def test_provider_defers_selection_until_ep_config_is_available(self):
        provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        with patch("rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_ep_moe.PpuEPMoE") as ctor:
            provider.build_moe(None, is_decode_role=True)
        self.assertEqual(
            ctor.call_args.kwargs["strategy_type"], provider._build_moe_strategy
        )

    def test_transport_library_comes_from_isolated_wheel(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe import ppu_mega_sparse as sparse

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "deep_gemm_mega"
            (package / "lib").mkdir(parents=True)
            library = package / "lib/libsignal_gin.so"
            library.touch()
            with patch.object(
                sparse,
                "find_spec",
                return_value=SimpleNamespace(origin=str(package / "__init__.py")),
            ), patch.object(sparse.C, "CDLL") as load:
                self.assertIs(
                    sparse._transport_library(library.name), load.return_value
                )
                load.assert_called_once_with(str(library))
                with self.assertRaisesRegex(RuntimeError, "rebuild"):
                    sparse._transport_library("libsparse_gin.so")

    def test_transport_library_requires_installed_package(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe import ppu_mega_sparse as sparse

        for package in (None, SimpleNamespace(origin=None)):
            with patch.object(sparse, "find_spec", return_value=package):
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    sparse._transport_library("libsignal_gin.so")

    def test_signal_transport_is_opt_in_for_four_nodes(self):
        from rtp_llm.models_py.modules.dsv4.ppu.moe import ppu_mega_sparse as sparse

        for nodes, enabled in ((1, "1"), (2, "1"), (4, "0"), (4, "1")):
            base = SimpleNamespace(
                nodes=nodes,
                node=0,
                dim=4096,
                experts=256 // nodes,
                cross_group=object(),
            )
            with patch.dict(
                os.environ, {"DSV4_PPU_MEGAMOE_SIGNAL_ONLY": enabled}
            ), patch.object(sparse, "PpuMegaCommunicator"), patch.object(
                sparse, "Exchange"
            ) as original, patch.object(
                sparse, "SignalExchange"
            ) as signal, patch.object(
                torch, "empty"
            ), patch.object(
                torch, "empty_like"
            ):
                sparse.SparseHierarchy(base, 12)
                expected, unused = (
                    (signal, original)
                    if nodes == 4 and enabled == "1"
                    else (original, signal)
                )
                self.assertEqual(expected.call_count, 2)
                unused.assert_not_called()

    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.group", new=SimpleNamespace(WORLD=object()))
    def test_signal_flag_keeps_ep8_resources_local(self, size, rank, capture):
        hierarchy = Mock(side_effect=AssertionError("EP8 must not create a hierarchy"))
        module = SimpleNamespace(
            get_symm_buffer_for_mega_moe=Mock(return_value=object())
        )
        with patch.dict(os.environ, {"DSV4_PPU_MEGAMOE_SIGNAL_ONLY": "1"}), patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_hierarchy": SimpleNamespace(
                    PpuMegaMoEHierarchy=hierarchy
                )
            },
        ):
            PpuMegaMoEResources().bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=128), module
            )
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
                self.assertTrue(
                    PpuMegaMoEFP4Strategy.can_handle(
                        replace(
                            self.cfg,
                            ep_size=size,
                            tp_size=tp,
                            n_local_experts=256 // size,
                            local_expert_end=256 // size,
                        )
                    )
                )
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
        self.assertEqual(a._mega_fp8_module_name, "deep_gemm_route_a")
        c = PpuDecodeProvider(
            {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        )
        self.assertIsNot(a._mega_resources, c._mega_resources)
        with self.assertRaisesRegex(ValueError, "deep_gemm_mega"):
            PpuDecodeProvider(
                {
                    **DECODE_EXECUTION_OPTIONS,
                    "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe",
                    "DSV4_PPU_MEGAMOE_PACKAGE": "deep_gemm_route_a",
                }
            )
        with self.assertRaisesRegex(ValueError, "must be deepep or megamoe"):
            PpuDecodeProvider({**options, "DSV4_PPU_DECODE_MOE_BACKEND": "typo"})

    def test_megamoe_mode_is_snapshotted_and_validated(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        plain = PpuDecodeProvider(options)
        streamed = PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_MODE": "streamed"})
        staged = PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_MODE": "staged"})
        self.assertFalse(plain._mega_streamed)
        self.assertTrue(streamed._mega_streamed)
        self.assertTrue(staged._shared_staged)
        self.assertFalse(staged._mega_streamed)
        with self.assertRaisesRegex(ValueError, "plain, streamed or staged"):
            PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_MODE": "split"})
        with self.assertRaisesRegex(ValueError, "require the MegaMoE backend"):
            PpuDecodeProvider(
                {
                    **options,
                    "DSV4_PPU_DECODE_MOE_BACKEND": "deepep",
                    "DSV4_PPU_MEGAMOE_MODE": "streamed",
                }
            )
        with self.assertRaisesRegex(ValueError, "replaced by DSV4_PPU_MEGAMOE_MODE"):
            PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_STREAMED": "1"})

    def test_staged_shared_options_are_snapshotted_and_validated(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        default = PpuDecodeProvider(options)
        staged = PpuDecodeProvider({**options, "DSV4_PPU_MEGAMOE_MODE": "staged"})
        self.assertFalse(default.build_shared_expert_executor().staged)
        self.assertTrue(staged.build_shared_expert_executor().staged)
        with self.assertRaisesRegex(ValueError, "MegaMoE backend"):
            PpuDecodeProvider(
                {
                    **options,
                    "DSV4_PPU_DECODE_MOE_BACKEND": "deepep",
                    "DSV4_PPU_MEGAMOE_MODE": "staged",
                }
            )
        with self.assertRaisesRegex(ValueError, "early overlap"):
            PpuSharedExpertExecutor(staged=True)

    def test_streamed_dispatch_and_callback_rejection(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = object()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        x = torch.zeros((8, 4096), dtype=torch.bfloat16)
        indices = torch.zeros((8, 8), dtype=torch.int64)
        weights = torch.ones((8, 8))
        for streamed in (False, True):
            strategy._streamed = streamed
            strategy._mega.reset_mock()
            strategy.forward(x, weights, indices)
            selected = (
                strategy._mega.fp4_fp4_mega_moe_streamed
                if streamed
                else strategy._mega.fp4_fp4_mega_moe
            )
            selected.assert_called_once()
            self.assertTrue(selected.call_args.kwargs["preserve_bf16"])
            self.assertEqual("fuse_reduce" in selected.call_args.kwargs, not streamed)
        strategy._mega.reset_mock()
        strategy._streamed = True
        with self.assertRaisesRegex(ValueError, "after_gemm1"):
            strategy.forward(x, weights, indices, after_gemm1=lambda: None)
        strategy._mega.mega_moe_pre_dispatch.assert_not_called()

    def test_streamed_tile_rejection_falls_back_to_plain_kernel(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = object()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        strategy._mega.fp4_fp4_mega_moe_streamed.side_effect = RuntimeError(
            "streamed MegaMoE requires Flash EP8 BM64/BN128 and G2 BM32"
        )
        strategy._streamed = True
        x = torch.zeros((8, 4096), dtype=torch.bfloat16)
        indices = torch.zeros((8, 8), dtype=torch.int64)
        weights = torch.ones((8, 8))
        strategy.forward(x, weights, indices)
        self.assertFalse(strategy._streamed)
        plain = strategy._mega.fp4_fp4_mega_moe
        plain.assert_called_once()
        self.assertFalse(plain.call_args.kwargs["fuse_reduce"])
        self.assertTrue(plain.call_args.kwargs["preserve_bf16"])
        strategy.forward(x, weights, indices)
        self.assertEqual(strategy._mega.fp4_fp4_mega_moe_streamed.call_count, 1)
        self.assertEqual(plain.call_count, 2)

    def test_streamed_unrelated_runtime_error_still_raises(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = object()
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        strategy._mega.fp4_fp4_mega_moe_streamed.side_effect = RuntimeError("boom")
        strategy._streamed = True
        with self.assertRaisesRegex(RuntimeError, "boom"):
            strategy.forward(
                torch.zeros((8, 4096), dtype=torch.bfloat16),
                torch.ones((8, 8)),
                torch.zeros((8, 8), dtype=torch.int64),
            )
        self.assertTrue(strategy._streamed)
        strategy._mega.fp4_fp4_mega_moe.assert_not_called()

    def test_missing_native_phase_api_rejected_before_forward(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy._mega = SimpleNamespace(fp4_fp4_mega_moe=lambda *args: None)
        with self.assertRaisesRegex(RuntimeError, "after_gemm1"):
            strategy.require_staged_shared()
        strategy._mega = SimpleNamespace(
            fp4_fp4_mega_moe=lambda *args, after_gemm1=None: None
        )
        strategy.require_staged_shared()

    def test_dispatch_preserves_bf16_and_separate_reduction(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = self.cfg
        strategy._sym = object()
        strategy._streamed = False
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
        self.assertEqual(
            selected.call_args.kwargs,
            {
                "activation_clamp": 10.0,
                "preserve_bf16": True,
                "fuse_reduce": False,
            },
        )

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
        cases = (
            (16, 1, 12, "collective", 16),
            (32, 1, 12, "collective", 16),
            (16, 1, 24, "collective", 32),
            (32, 1, 48, "collective", 48),
            (32, 1, 12, "sparse", 12),
            (32, 1, 1, "sparse", 1),
        )
        cases = [
            (size, tp, logical, transport, "0", capacity)
            for size, tp, logical, transport, capacity in cases
        ] + [
            (16, 1, 12, "collective", "1", 12),
            (16, 1, 15, "collective", "1", 15),
            (32, 1, 12, "collective", "1", 16),
            (32, 1, 12, "sparse", "1", 12),
        ]
        for size, tp, logical, transport, exact, capacity in cases:
            local_group = object()
            hierarchy = Mock(
                return_value=SimpleNamespace(
                    local_group=local_group, nodes=size // 8, experts=256 // (size // 8)
                )
            )
            fake = SimpleNamespace(PpuMegaMoEHierarchy=hierarchy)
            with patch.dict(
                os.environ,
                {
                    "DSV4_PPU_MEGAMOE_TRANSPORT": transport,
                    "DSV4_PPU_EP16_EXACT_CAPACITY": exact,
                },
            ), patch.dict(
                sys.modules,
                {"rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_hierarchy": fake},
            ), patch(
                "torch.distributed.group", SimpleNamespace(WORLD=object())
            ), patch(
                "torch.distributed.get_world_size", return_value=size
            ), patch(
                "torch.distributed.get_rank", return_value=0
            ), patch(
                "torch.cuda.is_current_stream_capturing", return_value=False
            ):
                cfg = replace(
                    self.cfg,
                    ep_size=size,
                    tp_size=tp,
                    n_local_experts=256 // size,
                    local_expert_end=256 // size,
                )
                wrapper = SimpleNamespace(
                    ll_num_max_token_per_rank=64,
                    config=SimpleNamespace(ll_num_max_token=logical),
                )
                module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock())
                PpuMegaMoEResources().bind(cfg, wrapper, module)
                hierarchy.assert_called_once_with(cfg, capacity)
                module.get_symm_buffer_for_mega_moe.assert_called_once_with(
                    local_group,
                    256 // (size // 8),
                    capacity * (size // 8),
                    8,
                    4096,
                    2048,
                )

    def test_ep8_does_not_import_cross_host_transport(self):
        self._assert_ep8_transport_isolation("direct")

    def test_ep8_does_not_import_sparse_transport(self):
        self._assert_ep8_transport_isolation("sparse")

    def _assert_ep8_transport_isolation(self, transport):
        group = object()
        module = SimpleNamespace(get_symm_buffer_for_mega_moe=Mock())
        with patch.dict(
            os.environ,
            {
                "DSV4_PPU_MEGAMOE_TRANSPORT": transport,
                "DSV4_PPU_EP16_EXACT_CAPACITY": "1",
            },
        ), patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_hierarchy": None,
                "rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_collective": None,
                "rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_sparse": None,
            },
        ), patch(
            "torch.distributed.group", SimpleNamespace(WORLD=group)
        ), patch(
            "torch.distributed.get_world_size", return_value=8
        ), patch(
            "torch.distributed.get_rank", return_value=0
        ), patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        ):
            resources = PpuMegaMoEResources()
            resources.bind(
                self.cfg, SimpleNamespace(ll_num_max_token_per_rank=256), module
            )
            self.assertIsNone(resources.hierarchy)
            module.get_symm_buffer_for_mega_moe.assert_called_once_with(
                group, 256, 256, 8, 4096, 2048
            )

    def test_large_ep_tile_hint_excludes_transport_padding(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = replace(
            self.cfg, ep_size=32, n_local_experts=8, local_expert_end=8
        )
        strategy._sym = object()
        strategy._streamed = False
        strategy._w13 = strategy._s13 = strategy._w2 = strategy._s2 = object()
        strategy._mega = Mock()
        x = torch.zeros((6, 4096), dtype=torch.bfloat16)
        padded = torch.zeros((64, 4096), dtype=torch.bfloat16)
        ids = torch.zeros((6, 8), dtype=torch.int64)
        weights = torch.ones((6, 8))
        hierarchy = Mock()
        hierarchy.dispatch.return_value = (
            padded,
            torch.ones((64, 8)),
            torch.zeros((64, 8), dtype=torch.int64),
        )
        strategy._resources = SimpleNamespace(hierarchy=hierarchy)
        strategy.forward(x, weights, ids)
        self.assertEqual(
            strategy._mega.fp4_fp4_mega_moe.call_args.kwargs["expected_m"], 5
        )
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
