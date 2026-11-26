"""NVFP4 dispatch contracts and opt-in SM120 GPU regression tests."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.config.qwen4_exp_config import build_qwen4_quant_config
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor import (
    Nvfp4HybridExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
    CudaNvfp4CutlassStrategy,
    CudaNvfp4HybridStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)

MODULE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor"


def config_for(
    role="DECODE",
    ep=2,
    hidden=2560,
    intermediate=640,
    experts=512,
    top_k=10,
    model_recipe=QWEN4_FP8_NVFP4_RECIPE,
    **tuning,
):
    quant = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=False, model_recipe=model_recipe, **tuning
    )
    pc = SimpleNamespace(
        role_type=role,
        tp_size=ep,
        ep_size=ep,
        dp_size=1,
        pp_size=1,
        get_ffn_tp_size=lambda: 1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            quant_config=quant,
            data_type=torch.bfloat16,
            eplb_config=SimpleNamespace(enable_eplb=lambda: False),
        ),
        parallelism_config=pc,
        ep_size=ep,
        tp_size=ep,
        moe_quant_method="modelopt_fp4",
        moe_strategy="nvfp4_cutlass",
        has_redundant_experts=False,
        moe_config=SimpleNamespace(
            fp4_moe_op="cutlass",
            moe_strategy="nvfp4_cutlass",
            use_all_gather=True,
            use_deepep_moe=False,
        ),
        hidden_size=hidden,
        moe_inter_dim=intermediate,
        n_local_experts=experts // ep,
        expert_num=experts,
        moe_k=top_k,
        moe_w1_layout="up_gate",
    )


class Nvfp4HybridExecutorTest(unittest.TestCase):
    def test_registry_selects_by_stage_and_conditions_without_model_name(self):
        registry = StrategyRegistry()
        baseline, hybrid = CudaNvfp4CutlassStrategy(), CudaNvfp4HybridStrategy()
        registry.register(baseline)
        registry.register(hybrid)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch(f"{MODULE}.uses_locked_wheel", return_value=True),
        ):
            for enabled, role, expected in [
                ("1", "DECODE", hybrid),
                ("1", "PREFILL", baseline),
                ("1", "PDFUSION", baseline),
                ("0", "DECODE", baseline),
            ]:
                with (
                    self.subTest(enabled=enabled, role=role),
                    patch.dict("os.environ", {"RTP_NVFP4_HYBRID_MOE": enabled}),
                ):
                    self.assertIs(
                        registry.get_strategy(
                            config_for(role=role, decode_b12x_max_rows=2)
                        ),
                        expected,
                    )
            with patch.dict("os.environ", {"RTP_NVFP4_HYBRID_MOE": "1"}):
                for ep, k, n, e, top_k in [
                    (1, 2048, 1024, 64, 4),
                    (4, 4096, 1536, 128, 8),
                ]:
                    self.assertIs(
                        registry.get_strategy(
                            config_for(
                                ep=ep, hidden=k, intermediate=n, experts=e, top_k=top_k
                            )
                        ),
                        baseline,
                    )
                self.assertIs(
                    registry.get_strategy(
                        config_for(intermediate=192, decode_b12x_max_rows=2)
                    ),
                    baseline,
                )
                config = config_for(decode_b12x_max_rows=2)
                self.assertIs(
                    registry.get_strategy(config_for(ep=8, decode_b12x_max_rows=2)),
                    baseline,
                )
                config.moe_w1_layout = "gate_up"
                self.assertIs(registry.get_strategy(config), baseline)
                with patch("torch.cuda.get_device_capability", return_value=(12, 1)):
                    self.assertIs(
                        registry.get_strategy(config_for(decode_b12x_max_rows=2)),
                        baseline,
                    )
            with patch.dict(os.environ):
                os.environ.pop("RTP_NVFP4_HYBRID_MOE", None)
                self.assertIs(
                    registry.get_strategy(config_for(model_recipe=None)), baseline
                )
                self.assertIs(
                    registry.get_strategy(
                        config_for(model_recipe=None, decode_b12x_max_rows=2)
                    ),
                    baseline,
                )
                config = config_for(model_recipe=QWEN4_FP8_NVFP4_RECIPE)
                self.assertIs(registry.get_strategy(config), hybrid)
                with patch(f"{MODULE}.uses_locked_wheel", return_value=False):
                    self.assertIs(registry.get_strategy(config), baseline)

    def test_locked_wheel_admits_measured_decode_geometry(self):
        registry = StrategyRegistry()
        baseline, hybrid = CudaNvfp4CutlassStrategy(), CudaNvfp4HybridStrategy()
        registry.register(baseline)
        registry.register(hybrid)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch(f"{MODULE}.uses_locked_wheel", return_value=True),
            patch.dict(os.environ, {"RTP_NVFP4_HYBRID_MOE": "1"}),
        ):
            self.assertIs(registry.get_strategy(config_for()), hybrid)
            for config in (
                config_for(role="PREFILL"),
                config_for(intermediate=768),
                config_for(top_k=6),
                config_for(ep=4),
            ):
                self.assertIs(registry.get_strategy(config), baseline)

    def test_recipe_defaults_and_opt_out(self):
        for requested_rows in (0, 2):
            requested = json.dumps(
                dict(
                    method="modelopt_fp4",
                    bits=4,
                    group_size=16,
                    is_quanted=False,
                    model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                    decode_b12x_max_rows=requested_rows,
                )
            )
            config = build_qwen4_quant_config(requested, source_fp8=True)
            self.assertEqual(config.decode_b12x_max_rows, requested_rows)
        self.assertEqual(config_for().model_config.quant_config.decode_b12x_max_rows, 2)
        for bad in (-1, True, 17):
            with self.assertRaisesRegex(ValueError, "decode_b12x_max_rows"):
                config_for(decode_b12x_max_rows=bad)
        for timeout in (-1, True, 1.5, 2**31):
            with self.assertRaisesRegex(ValueError, "decode_lsa_timeout_ms"):
                config_for(decode_lsa_timeout_ms=timeout)


@unittest.skipUnless(
    os.environ.get("RTP_TEST_B12X_NVFP4_GPU") == "1",
    "requires SM120 and the pinned FlashInfer b12x runtime",
)
class B12xNvfp4GpuTest(unittest.TestCase):
    @staticmethod
    def swizzled_scales(experts, rows, cols, offset, device):
        # Logical block16 scales vary across experts, rows and K tiles.
        logical = (
            torch.arange(experts * rows * (cols // 16), device=device)
            .reshape(experts, rows, cols // 16)
            .add(offset)
            .remainder(29)
            .add(4)
            .div(128)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        packed = (
            logical.reshape(experts, rows // 128, 4, 32, cols // 64, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .view(experts, rows, cols // 16)
            .view(torch.int32)
        )
        return logical, packed

    @staticmethod
    def packed_weights(experts, hidden, intermediate, device, *, nonuniform=False):
        packed = 0x2222222222222222
        w1 = torch.full(
            (experts, 2 * intermediate, hidden // 16),
            packed,
            dtype=torch.int64,
            device=device,
        )
        w2 = torch.full(
            (experts, hidden, intermediate // 16),
            packed,
            dtype=torch.int64,
            device=device,
        )
        scale1 = torch.ones(
            (experts, 2 * intermediate, hidden // 16), device=device
        ).to(torch.float8_e4m3fn)
        scale2 = torch.ones((experts, hidden, intermediate // 16), device=device).to(
            torch.float8_e4m3fn
        )
        alpha = torch.ones(experts, device=device)
        inverse_act_scale = torch.ones((), device=device)
        if nonuniform:

            def fill_codes(weight, offset):
                expert = torch.arange(weight.shape[0], device=device)[:, None, None]
                row = torch.arange(weight.shape[1], device=device)[None, :, None]
                column = torch.arange(weight.shape[2], device=device)[None, None, :]
                weight.zero_()
                for lane in range(16):
                    codes = (
                        expert + 2 * row + column + row // intermediate + lane + offset
                    ) % 5 + 1
                    weight.bitwise_or_(codes << (4 * lane))

            fill_codes(w1, 0)
            fill_codes(w2, 2)
            alpha = torch.linspace(0.75, 1.25, experts, device=device)
            inverse_act_scale = torch.full((), 1.25, device=device)
        return (
            w1,
            w2,
            [
                inverse_act_scale,
                scale1.view(torch.int32),
                alpha,
                inverse_act_scale * 0.75 if nonuniform else inverse_act_scale,
                scale2.view(torch.int32),
                alpha.flip(0) if nonuniform else alpha,
            ],
        )

    def test_production_shape_matches_cutlass(self):
        from pathlib import Path

        from flashinfer import ActivationType
        from flashinfer.fused_moe import cutlass_fused_moe
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_decode_069 import (
            LockedDecodeMoE,
            uses_locked_wheel,
        )
        from rtp_llm.models_py.triton_kernels.moe.qwen4_topk import _moe_topk

        self.assertTrue(uses_locked_wheel())
        device = torch.device("cuda", torch.cuda.current_device())
        experts, hidden, intermediate, top_k = 256, 2560, 640, 10
        torch.manual_seed(647)
        capture_root = os.getenv("RTP_TEST_B12X_CAPTURE_DIR")
        fixture = None
        if capture_root:
            fixture = torch.load(
                Path(capture_root) / "I8192-B16-R0-L0-S0.pt",
                map_location=device,
                weights_only=True,
            )
        model = object()
        for ep_rank in (0, 1):
            if fixture is not None:
                pack = torch.load(
                    Path(capture_root) / f"weights-R{ep_rank}-L0.pt",
                    map_location=device,
                    weights_only=True,
                )
                w1, w2, scales = (
                    pack["fc1_expert_weights"],
                    pack["fc2_expert_weights"],
                    pack["quant_scales"],
                )
            else:
                w1, w2, scales = self.packed_weights(
                    experts, hidden, intermediate, device
                )
                w1.random_()
                w2.random_()
                for index, rows, cols in (
                    (1, 2 * intermediate, hidden),
                    (4, hidden, intermediate),
                ):
                    _, scales[index] = self.swizzled_scales(
                        experts, rows, cols, index + ep_rank, device
                    )
            self.assertEqual(w1.shape, (experts, 2 * intermediate, hidden // 16))
            self.assertEqual(w2.shape, (experts, hidden, intermediate // 16))
            adapter = LockedDecodeMoE(w1, w2, scales, ep_rank, model, top_k, True)
            self.assertTrue(adapter.micro)
            self.assertTrue(adapter.profile_table_matches)
            for rows in (2, 4, 8, 16):
                with self.subTest(ep_rank=ep_rank, rows=rows):
                    if fixture is not None:
                        x = fixture["input"][:rows].contiguous()
                        ids = (
                            fixture["token_selected_experts"][:rows]
                            .to(torch.int32)
                            .contiguous()
                        )
                        weights = (
                            fixture["token_final_scales"][:rows].float().contiguous()
                        )
                    else:
                        x = (
                            torch.randn(
                                rows, hidden, device=device, dtype=torch.bfloat16
                            )
                            * 0.125
                        )
                        logits = torch.randn(rows, 2 * experts, device=device)
                        ids = torch.empty(
                            (rows, top_k), dtype=torch.int32, device=device
                        )
                        weights = torch.empty(
                            (rows, top_k), dtype=torch.float32, device=device
                        )
                        _moe_topk[(rows,)](
                            logits,
                            ids,
                            weights,
                            STRIDE=2 * experts,
                            num_warps=1,
                            enable_fp_fusion=False,
                        )
                    self.assertEqual(x.shape, (rows, hidden))
                    self.assertEqual(ids.shape, (rows, top_k))
                    actual = adapter.execute(x, ids, weights)

                    expected = cutlass_fused_moe(
                        input=x,
                        input_sf=None,
                        token_selected_experts=ids,
                        token_final_scales=weights,
                        fc1_expert_weights=w1,
                        fc2_expert_weights=w2,
                        output_dtype=torch.bfloat16,
                        quant_scales=scales,
                        tp_size=1,
                        tp_rank=0,
                        ep_size=2,
                        ep_rank=ep_rank,
                        activation_type=ActivationType.Swiglu,
                        enable_pdl=False,
                    )
                    if isinstance(expected, list):
                        expected = expected[0]
                    local = (ids // experts == ep_rank).any(dim=1)
                    self.assertGreater(expected.float().norm().item(), 0)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertEqual(torch.count_nonzero(actual[~local]).item(), 0)
                    delta = actual.float() - expected.float()
                    relative_l2 = (delta.norm() / expected.float().norm()).item()
                    self.assertLessEqual(relative_l2, 0.05)
                    remote_ids = (
                        torch.arange(top_k, device=device, dtype=ids.dtype)
                        + (1 - ep_rank) * experts
                    )
                    remote_only = adapter.execute(
                        x, remote_ids.expand_as(ids).contiguous(), weights
                    )
                    self.assertEqual(torch.count_nonzero(remote_only).item(), 0)
                    print(
                        f"B12X_PRODUCTION_CASE ep_rank={ep_rank} rows={rows} "
                        f"capture={fixture is not None} rel_l2={relative_l2:.8f} "
                        f"max_abs={delta.abs().max().item():.8f}",
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
