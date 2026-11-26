import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import (
    Qwen3NextGdnParameters,
)
from rtp_llm.models_py.new_models.qwen3_next.test.gated_groupwise_test import (
    _encode,
    _fixture,
    _load_config,
)
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.language import Qwen3NextForCausalLM
from rtp_llm.ops import HybridAttentionType
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import get_all_weights


def owner(method, bits, rank=0, ignored=(), enabled=True):
    cfg = NS(
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        linear_conv_kernel_dim=4,
    )
    quant = QuantizationConfig(
        method, NS(bits=bits, group_size=32), ignored_layers=ignored
    )
    return Qwen3NextGdnParameters(
        cfg,
        128,
        1e-6,
        4,
        rank,
        torch.bfloat16,
        quant,
        "layers.0.linear_attn",
        grouped_control_source=enabled,
    )


def encoded(method, bits, order):
    rng = torch.Generator().manual_seed(771)
    q = torch.randint(1 << bits, (128, 32), generator=rng)
    z = torch.randint(1 << bits, (4, 32), generator=rng)
    z[0, 0] = (1 << bits) - 1  # GPTQ zero offset may be 16 or 256; never wrap.
    scale = torch.randint(1, 20, (4, 32), generator=rng).float() / 8
    raw = {}
    for component, cols in order.items():
        key = "in_proj_" + component + "."
        raw.update(
            {
                key + n: t
                for n, t in _encode(
                    method, bits, [x[:, cols] for x in (q, z, scale)]
                ).items()
            }
        )
        if method == "gptq":
            raw[key + "g_idx"] = torch.arange(128, dtype=torch.int32) // 32
            raw[key + "bias"] = torch.zeros(len(cols))
    groups = torch.arange(128) // 32
    decoded = (
        (q - z[groups] - int(method == "gptq")).float() * scale[groups]
    ).bfloat16()
    return raw, decoded


class ControlTest(unittest.TestCase):
    def test_production_qwen35_owner_routes_checkpoint_keys(self):
        cfg, _, _ = _fixture("gptq", 4)
        cfg.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR
        ]
        linear = cfg.linear_attention_config
        linear.linear_num_key_heads, linear.linear_num_value_heads = 8, 16
        linear.linear_value_head_dim = linear.linear_key_head_dim = 16
        quant = QuantizationConfig("gptq", cfg.quant_config)
        opts = replace(
            _load_config(tp_size=2, tp_rank=1, quant_config=quant),
            compute_dtype=torch.bfloat16,
        )
        model = Qwen3NextForCausalLM(cfg, opts)
        tensors, logical = encoded(
            "gptq", 4, {"b": list(range(16)), "a": list(range(16, 32))}
        )
        with tempfile.TemporaryDirectory() as folder:
            file = str(Path(folder) / "model.safetensors")
            name = "model.language_model.layers.0.linear_attn."
            save_file({name + k: v for k, v in tensors.items()}, file)
            loader = NewModelLoader(cfg, opts, model_path=folder)
            resolve = loader._checkpoint_read_spec_resolver(
                model, model.checkpoint_weight_name_filter(), None
            )
            model.load_weights(get_all_weights([file], read_spec_resolver=resolve))
            torch.testing.assert_close(
                model.layers[0].linear_attn.in_proj_ba_w,
                logical[:, list(range(8, 16)) + list(range(24, 32))],
                rtol=0,
                atol=0,
            )
        cfg.model_type = "qwen3_next"
        other = Qwen3NextForCausalLM(cfg, opts)
        self.assertIsNone(other.layers[0].linear_attn._grouped_controls)

    def test_load_actual_file_controls_and_gemm(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                for fused in (False, True):
                    order = (
                        {
                            "ba": [
                                c
                                for i in range(8)
                                for c in (2 * i, 2 * i + 1, 16 + 2 * i, 17 + 2 * i)
                            ]
                        }
                        if fused
                        else {"b": list(range(16)), "a": list(range(16, 32))}
                    )
                    source, full = encoded(method, bits, order)
                    with (
                        self.subTest(method=method, bits=bits, fused=fused),
                        tempfile.TemporaryDirectory() as folder,
                    ):
                        path = str(Path(folder) / "weights.safetensors")
                        save_file(source, path)
                        for rank in range(4):
                            leaf = owner(method, bits, rank)
                            for key, tensor in get_all_weights([path]):
                                leaf.load_weights({key: tensor})
                            cols = list(range(rank * 4, rank * 4 + 4))
                            expected = full[:, cols + [x + 16 for x in cols]]
                            torch.testing.assert_close(
                                leaf.in_proj_ba_w, expected, atol=0, rtol=0
                            )
                            x = torch.eye(128, dtype=torch.bfloat16)[:16]
                            torch.testing.assert_close(
                                x @ leaf.in_proj_ba_w, x @ expected, atol=0, rtol=0
                            )

    def test_mixed_exclusion_validation_and_scope(self):
        raw, full = encoded("gptq", 4, {"a": list(range(16, 32)), "b": list(range(16))})
        leaf = owner("gptq", 4, ignored=["layers.0.linear_attn.in_proj_a"])
        leaf.load_weights({"in_proj_a.weight": full[:, 16:].T})
        for key, tensor in raw.items():
            if key.startswith("in_proj_b."):
                leaf.load_weights({key: tensor})
        torch.testing.assert_close(
            leaf.in_proj_ba_w, torch.cat((full[:, :4], full[:, 16:20]), 1)
        )
        for field, source, error in (
            ("qweight", raw["in_proj_a.qweight"].float(), "int32"),
            ("g_idx", torch.ones(128, dtype=torch.int32), "activation-ordered"),
            ("bias", torch.ones(16), "zero bias"),
            ("scales", torch.full((4, 16), float("nan")), "scales"),
        ):
            with self.assertRaisesRegex(ValueError, error):
                owner("gptq", 4).load_weights({"in_proj_a." + field: source})
        with self.assertRaisesRegex(ValueError, "excluded"):
            owner("gptq", 4, ignored=["layers.0.linear_attn.in_proj_a"]).load_weights(
                {"in_proj_a.scales": raw["in_proj_a.scales"]}
            )
        with self.assertRaisesRegex(ValueError, "Mixed"):
            leaf.load_weights({"in_proj_a.qweight": raw["in_proj_a.qweight"]})
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            owner("gptq", 4, enabled=False).load_weights(
                {"in_proj_a.scales": raw["in_proj_a.scales"]}
            )


if __name__ == "__main__":
    unittest.main()
