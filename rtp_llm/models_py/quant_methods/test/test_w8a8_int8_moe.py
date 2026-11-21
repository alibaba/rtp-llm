"""Real checkpoint/placement tests, not PPU execution or multicard tests."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.w8a8_int8 import W8A8_METHOD
from rtp_llm.models_py.quant_methods.w8a8_int8_moe import _require_grouped_symbols
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _MoeModel(RtpModule):
    supports_expert_parallel = True

    def __init__(self, config, load_config):
        super().__init__()
        tp, ep = load_config.tp_size, load_config.ep_size
        parallel = types.SimpleNamespace(
            tp_size=tp,
            tp_rank=load_config.tp_rank,
            ep_size=ep,
            ep_rank=load_config.ep_rank,
            dp_size=1,
            dp_rank=0,
            world_size=max(tp, ep),
            local_rank=load_config.ep_rank,
            local_world_size=max(tp, ep),
            num_nodes=1,
            get_attn_tp_size=lambda: tp,
            get_attn_tp_rank=lambda: load_config.tp_rank,
        )
        moe_config = types.SimpleNamespace(
            use_deepep_low_latency=False,
            use_all_gather=ep == 1,
            use_deepep_moe=ep > 1,
            use_mori_ep=False,
            moe_strategy="auto",
            ll_num_max_token=1,
            masked_max_token_num=1,
        )
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=1,
            hidden_size=config.hidden_size,
            moe_intermediate_size=getattr(config, "intermediate_size", 12),
            tp_size=tp,
            tp_rank=load_config.tp_rank,
            ep_size=ep,
            ep_rank=load_config.ep_rank,
            params_dtype=torch.bfloat16,
            model_config=config,
            parallelism_config=parallel,
            moe_config=moe_config,
            quant_config=load_config.quant_config,
            layer_idx=0,
            prefix="experts",
        )


class W8A8Int8MoETest(unittest.TestCase):
    def test_real_consumer_classes_import_without_cuda_fp8_native_symbol(self):
        # Import the actual PPU closure, not stand-in executors. This checks
        # Python dependency isolation only, never PPU kernel/DeepEP execution.
        from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.strategy.w8a8_int8 import (
            PpuW8A8Int8DpNormalDeepGemmStrategy,
            PpuW8A8Int8EpLowLatencyDeepGemmStrategy,
        )

        for strategy in (
            PpuW8A8Int8DpNormalDeepGemmStrategy,
            PpuW8A8Int8EpLowLatencyDeepGemmStrategy,
        ):
            attributes = strategy().get_attributes()
            self.assertTrue(callable(attributes.router_class))
            self.assertTrue(callable(attributes.executor_class))
            self.assertIn("ppu_impl", attributes.router_class.__module__)
            self.assertIn("ppu_impl", attributes.executor_class.__module__)

    def checkpoint(self):
        values = {}
        for expert in range(4):
            for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                rows, columns = (8, 12) if projection == "down_proj" else (12, 8)
                prefix = f"experts.{expert}.{projection}."
                values[prefix + "weight"] = (
                    (
                        (torch.arange(rows * columns) + expert * 19 + index * 7) % 255
                        - 127
                    )
                    .to(torch.int8)
                    .reshape(rows, columns)
                )
                values[prefix + "weight_scale"] = (
                    torch.arange(rows).float() + 1 + expert + index
                ).reshape(rows, 1) / 31
        return values

    def load(
        self, values, tp=1, tp_rank=0, ep=1, ep_rank=0, redundant=0, method="scratch"
    ):
        source = CompressedW8A8Int8PerChannelQuantConfig()
        config = types.SimpleNamespace(
            model_type="ppu_w8a8_moe_fixture",
            expert_num=4,
            moe_k=1,
            moe_style=2,
            hidden_size=8,
            activation_type="SiGLU",
            data_type="bf16",
            quant_config=source,
            num_layers=1,
            phy2log_path="",
            attn_config=types.SimpleNamespace(head_num=2),
            moe_topk_group=1,
            eplb_config=types.SimpleNamespace(
                redundant_expert=redundant,
                phy_exp_num=lambda logical: logical + redundant,
                enable_eplb=lambda: False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(values, os.path.join(directory, "model.safetensors"))
            with mock.patch.dict(MODEL_REGISTRY, {config.model_type: _MoeModel}):
                loader = NewModelLoader(
                    config,
                    NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        load_method="fastsafetensors" if method == "fst" else method,
                        force_cpu_load_weights=method == "fst",
                        tp_size=tp,
                        tp_rank=tp_rank,
                        ep_size=ep,
                        ep_rank=ep_rank,
                        quant_config=QuantizationConfig(
                            W8A8_METHOD, source_config=source
                        ),
                    ),
                    model_path=directory,
                )
                model = loader.load()
                self.assertEqual(
                    loader._selected_load_method.value,
                    "fastsafetensors" if method == "fst" else "scratch",
                )
                return model.experts

    def test_native_fst_cpu_checkpoint_tp_ep_redundant_matches_scratch(self):
        for rank in range(2):
            for redundant in (0, 2):
                with self.subTest(rank=rank, redundant=redundant):
                    values = self.checkpoint()
                    actual = self.load(values, 2, rank, 2, rank, redundant, "fst")
                    expected = self.load(values, 2, rank, 2, rank, redundant)
                    for name in ("w13", "w2", "w13_scale", "w2_scale"):
                        torch.testing.assert_close(
                            getattr(actual, name),
                            getattr(expected, name),
                            rtol=0,
                            atol=0,
                        )

    def test_stacked_experts_native_fst_and_scratch_match_separate_keys(self):
        separate = self.checkpoint()
        stacked = {}
        for field in ("weight", "weight_scale"):
            stacked[f"experts.gate_up_proj.{field}"] = torch.stack(
                [
                    torch.cat(
                        [
                            separate[f"experts.{expert}.{p}_proj.{field}"]
                            for p in ("gate", "up")
                        ]
                    )
                    for expert in range(4)
                ]
            )
            stacked[f"experts.down_proj.{field}"] = torch.stack(
                [separate[f"experts.{expert}.down_proj.{field}"] for expert in range(4)]
            )
        for method in ("scratch", "fst"):
            for ep, redundant in ((1, 0), (2, 0), (2, 2)):
                for rank in range(2):
                    with self.subTest(
                        method=method, ep=ep, redundant=redundant, rank=rank
                    ):
                        ep_rank = rank if ep == 2 else 0
                        actual = self.load(
                            stacked, 2, rank, ep, ep_rank, redundant, method
                        )
                        expected = self.load(separate, 2, rank, ep, ep_rank, redundant)
                        for name in ("w13", "w2", "w13_scale", "w2_scale"):
                            torch.testing.assert_close(
                                getattr(actual, name),
                                getattr(expected, name),
                                rtol=0,
                                atol=0,
                            )
        invalid = dict(stacked)
        invalid["experts.gate_up_proj.weight"] = stacked[
            "experts.gate_up_proj.weight"
        ].float()
        for method in ("scratch", "fst"):
            with self.assertRaises(TypeError):
                self.load(invalid, method=method)

    def test_full_loader_real_checkpoint_tp_ep_redundant(self):
        checkpoint = self.checkpoint()
        cases = [(tp, rank, 1, 0, 0) for tp in (1, 2, 4) for rank in range(tp)]
        cases += [
            (2, rank, 2, rank, redundant) for redundant in (0, 2) for rank in range(2)
        ]
        for tp, rank, ep, ep_rank, redundant in cases:
            with self.subTest(tp=tp, rank=rank, ep=ep, redundant=redundant):
                layer = self.load(checkpoint, tp, rank, ep, ep_rank, redundant)
                for local, logical in enumerate(
                    layer._expert_placement.local_physical_to_logical
                ):
                    start = layer.moe_expert_tp_rank * layer.moe_inter_tp
                    rows = slice(start, start + layer.moe_inter_tp)
                    expected_weight, expected_scale = [], []
                    for name in ("up_proj", "gate_proj"):
                        expected_weight.append(
                            checkpoint[f"experts.{logical}.{name}.weight"][rows]
                        )
                        expected_scale.append(
                            checkpoint[f"experts.{logical}.{name}.weight_scale"][rows]
                        )
                    torch.testing.assert_close(
                        layer.w13[local], torch.cat(expected_weight), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        layer.w13_scale[local],
                        torch.cat(expected_scale),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        layer.w2[local],
                        checkpoint[f"experts.{logical}.down_proj.weight"][:, rows],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        layer.w2_scale[local],
                        checkpoint[f"experts.{logical}.down_proj.weight_scale"],
                        rtol=0,
                        atol=0,
                    )
                self.assertIsNone(layer.fused_moe)
                self.assertFalse(layer._post_load_done)

    def test_missing_scales_and_bad_integer_dtype_rejected(self):
        values = self.checkpoint()
        del values["experts.0.up_proj.weight_scale"]
        with self.assertRaisesRegex(RuntimeError, "auxiliary tensors"):
            self.load(values)
        values = self.checkpoint()
        values["experts.0.up_proj.weight"] = values["experts.0.up_proj.weight"].to(
            torch.uint8
        )
        with self.assertRaises(TypeError):
            self.load(values)
        values = self.checkpoint()
        values["experts.0.up_proj.weight_scale"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.load(values)

    def test_grouped_preflight_checks_callable_symbols_and_deepep_apis(self):
        masked = "m_grouped_gemm_int8_int8_bf16_nt_masked"
        contiguous = "m_grouped_gemm_int8_int8_bf16_nt_contiguous"
        for value in (None, object()):
            native = types.SimpleNamespace(**{masked: value})
            with mock.patch(
                "rtp_llm.models_py.quant_methods.w8a8_int8_moe.importlib.import_module",
                return_value=native,
            ):
                with self.assertRaisesRegex(RuntimeError, "grouped INT8"):
                    _require_grouped_symbols(True)
        native = types.SimpleNamespace(
            **{masked: lambda: None, contiguous: lambda: None}
        )
        with mock.patch(
            "rtp_llm.models_py.quant_methods.w8a8_int8_moe.importlib.import_module",
            side_effect=[native, types.SimpleNamespace(Buffer=type("Buffer", (), {}))],
        ):
            with self.assertRaisesRegex(RuntimeError, "Buffer APIs"):
                _require_grouped_symbols(False)

    def test_preferred_noncallable_alias_is_not_hidden_by_callable_fallback(self):
        from rtp_llm.models_py.quant_methods.w8a8_int8 import resolved_backend_symbol

        native = types.SimpleNamespace(first=object(), second=lambda: None)
        self.assertFalse(callable(resolved_backend_symbol(native, ("first", "second"))))
        for attributes in (
            {
                "m_grouped_gemm_int8_int8_bf16_nt_masked": object(),
                "m_grouped_int8_gemm_nt_masked": lambda: None,
            },
            {
                "m_grouped_gemm_int8_int8_bf16_nt_masked": lambda: None,
                "m_grouped_gemm_int8_int8_bf16_nt_contiguous": lambda: None,
                "m_grouped_gemm_int8_int8_bf16_nt_nopad": object(),
            },
        ):
            with mock.patch(
                "rtp_llm.models_py.quant_methods.w8a8_int8_moe.importlib.import_module",
                return_value=types.SimpleNamespace(**attributes),
            ):
                with self.assertRaisesRegex(RuntimeError, "grouped INT8"):
                    _require_grouped_symbols(False)


if __name__ == "__main__":
    unittest.main()
