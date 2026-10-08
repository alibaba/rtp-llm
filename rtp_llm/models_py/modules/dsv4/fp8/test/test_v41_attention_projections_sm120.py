"""Construct real attention projections with V4/V4.1 scale geometry on SM120."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4 import utils
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    _v4_fp8_linear,
    _v4_fp8_linear_from_dict,
)
from rtp_llm.utils.model_weight import W


def make_attention(weights, config, *, tp_size=1, tp_rank=0, layer_id=21):
    return AttentionFP8(
        layer_id=layer_id,
        cache_layer_id=0,
        dim=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        q_lora_rank=config["q_lora_rank"],
        head_dim=config["head_dim"],
        rope_head_dim=config["qk_rope_head_dim"],
        o_lora_rank=config["o_lora_rank"],
        o_groups=config["o_groups"],
        window_size=128,
        # The upstream V4.1 owner disables the V4 compressor during super init.
        compress_ratio=0,
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=65536,
        max_batch_size=2,
        max_seq_len=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        norm_eps=config.get("rms_norm_eps", 1e-6),
        layer_weights=weights,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


class AttentionProjectionsSM120Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4110)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def synthetic(self, block):
        config = dict(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=256,
            qk_rope_head_dim=64,
            q_lora_rank=256,
            o_lora_rank=256,
            o_groups=4,
        )
        weights = {}
        for name, n, k in (
            ("wq_a", 256, 512),
            ("wq_b", 2048, 256),
            ("wkv", 256, 512),
            ("wo_a", 1024, 512),
            ("wo_b", 512, 1024),
        ):
            weights[getattr(W, f"v4_attn_{name}_w")] = torch.randn(
                n, k, device="cuda"
            ).to(torch.float8_e4m3fn)
            weights[getattr(W, f"v4_attn_{name}_s")] = torch.randint(
                120, 125, (n // block, k // block), device="cuda", dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
        weights[W.v4_attn_q_norm] = torch.ones(256, device="cuda", dtype=torch.bfloat16)
        weights[W.v4_attn_kv_norm] = torch.ones(
            256, device="cuda", dtype=torch.bfloat16
        )
        weights[W.v4_attn_sink] = torch.arange(8, device="cuda", dtype=torch.float32)
        return weights, config

    def assert_bytes(self, actual, expected):
        torch.testing.assert_close(
            actual.view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def check_constructor(self, weights, config, block, tp_size, tp_rank):
        attn = make_attention(weights, config, tp_size=tp_size, tp_rank=tp_rank)
        self.assertEqual((attn.layer_id, attn.cache_layer_id), (21, 0))
        local_heads = config["num_attention_heads"] // tp_size
        local_groups = config["o_groups"] // tp_size
        query_rows = local_heads * config["head_dim"]
        output_rows = local_groups * config["o_lora_rank"]
        query_slice = slice(tp_rank * query_rows, (tp_rank + 1) * query_rows)
        output_slice = slice(tp_rank * output_rows, (tp_rank + 1) * output_rows)
        for name, rows, cols in (
            ("wq_a", slice(None), slice(None)),
            ("wq_b", query_slice, slice(None)),
            ("wkv", slice(None), slice(None)),
            ("wo_b", slice(None), output_slice),
        ):
            op = getattr(attn, name)
            expected_weight = weights[getattr(W, f"v4_attn_{name}_w")][rows, cols]
            sr = (
                slice(rows.start // block, rows.stop // block)
                if rows.start is not None
                else rows
            )
            sc = (
                slice(cols.start // block, cols.stop // block)
                if cols.start is not None
                else cols
            )
            scales = weights[getattr(W, f"v4_attn_{name}_s")][sr, sc].contiguous()
            self.assert_bytes(op.weight, expected_weight)
            if block == 32:
                self.assertIsInstance(op, utils.V41MXFP8Linear)
                self.assert_bytes(op._sm120_linear.scales, scales)
                x = torch.randn(7, op.K, device="cuda", dtype=torch.bfloat16)
                quant, packed = op._quantize_input(x)
                exponent = torch.stack(
                    [(packed >> (8 * byte)) & 255 for byte in range(4)], dim=-1
                ).reshape(7, -1)[:, : op.K // 32]
                a = quant.float() * torch.exp2(
                    exponent.float() - 127
                ).repeat_interleave(32, 1)
                b = expected_weight.float() * scales.float().repeat_interleave(
                    32, 0
                ).repeat_interleave(32, 1)
                expected = (a @ b.T).bfloat16()
                actual = op(x)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
            else:
                self.assertNotIsInstance(op, utils.V41MXFP8Linear)
                reference = utils._v4_fp8_linear(expected_weight.contiguous(), scales)
                x = torch.randn(
                    7, expected_weight.shape[1], device="cuda", dtype=torch.bfloat16
                )
                torch.testing.assert_close(op(x), reference(x), rtol=0, atol=0)
        self.assert_bytes(attn.wo_a_w, weights[W.v4_attn_wo_a_w][output_slice])
        expected_scale = weights[W.v4_attn_wo_a_s][
            output_slice.start // block : output_slice.stop // block
        ]
        self.assert_bytes(attn.wo_a_s, expected_scale)
        self.assert_bytes(attn._wo_a_stk_w.flatten(0, 1), attn.wo_a_w)
        packed = attn._wo_a_stk_s
        unpacked = (
            torch.stack([(packed >> (8 * byte)) & 255 for byte in range(4)], dim=-1)
            .flatten(-2)
            .to(torch.uint8)
        )
        expected = (
            expected_scale.view(torch.uint8)
            .reshape(local_groups, config["o_lora_rank"] // block, -1)
            .repeat_interleave(block, 1)
        )
        torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            attn.attn_sink,
            weights[W.v4_attn_sink][
                tp_rank * local_heads : (tp_rank + 1) * local_heads
            ],
        )

    @torch.no_grad()
    def test_group32_factories_preserve_checkpoint_scales(self):
        weights, _ = self.synthetic(32)
        weight, scale = weights[W.v4_attn_wq_a_w], weights[W.v4_attn_wq_a_s]
        for op in (
            _v4_fp8_linear(weight, scale),
            _v4_fp8_linear_from_dict(weights, W.v4_attn_wq_a_w, W.v4_attn_wq_a_s),
        ):
            self.assertIsInstance(op, utils.V41MXFP8Linear)
            self.assertIs(op.weight, weight)
            self.assertFalse(hasattr(op, "quantize_input"))
        self.assertIs(weights[W.v4_attn_wq_a_s], scale)

    @torch.no_grad()
    def test_group32_native_constructor_and_tp_slices(self):
        weights, config = self.synthetic(32)
        for tp_size in (1, 2, 4):
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    self.check_constructor(weights, config, 32, tp_size, rank)

    @torch.no_grad()
    def test_v4_block128_native_constructor_regression(self):
        weights, config = self.synthetic(128)
        for tp_size in (1, 2, 4):
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    self.check_constructor(weights, config, 128, tp_size, rank)

    @torch.no_grad()
    def test_nas_attention_projections(self):
        from safetensors import safe_open

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        config = json.loads((root / "config.json").read_text())["text_config"]
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in (0, 21):
            weights = {}
            suffixes = {
                W.v4_attn_q_norm: "q_norm.weight",
                W.v4_attn_kv_norm: "kv_norm.weight",
                W.v4_attn_sink: "attn_sink",
            }
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                suffixes[getattr(W, f"v4_attn_{name}_w")] = name + ".weight"
                suffixes[getattr(W, f"v4_attn_{name}_s")] = name + ".scale"
            for key, suffix in suffixes.items():
                name = f"layers.{layer}.attn.{suffix}"
                with safe_open(
                    root / index[name], framework="pt", device="cpu"
                ) as file:
                    weights[key] = file.get_tensor(name).cuda()
            for key in (W.v4_attn_q_norm, W.v4_attn_kv_norm):
                weights[key] = weights[key].bfloat16()
            for tp_size, rank in ((1, 0), (2, 1), (8, 7)):
                with self.subTest(layer=layer, tp_size=tp_size, rank=rank):
                    self.check_constructor(weights, config, 32, tp_size, rank)


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            "Attention projection qualification requires real SM120 hardware"
        )
    return tests


if __name__ == "__main__":
    unittest.main()
