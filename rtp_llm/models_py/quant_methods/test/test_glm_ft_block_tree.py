"""Whole GLM parameter tree, real legacy-layout files, CPU owner staging only."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
)
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.glm4_moe.parameters import install_glm4_parameters
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.test.test_block_fp8_ft import write_ft


def tree(root, channel=False):
    source = (
        Fp8PerChannelCompressedQuantConfig(is_quanted=True)
        if channel
        else Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=True)
    )
    config = SimpleNamespace(
        ckpt_path=str(root),
        num_layers=1,
        moe_layer_index=[0],
        moe_inter_size=128,
        qk_norm=True,
        tie_word_embeddings=False,
        normalize_lm_head_weight=False,
        logit_scale=1.0,
        scoring_func=1,
        routed_scaling_factor=1.0,
        moe_n_group=1,
        moe_topk_group=1,
        has_moe_norm=True,
        activation_type="SiGLU",
        quant_config=source,
        expert_num=2,
        hidden_size=128,
        moe_k=1,
    )
    cfg = dict(
        model_config=config,
        num_layers=1,
        vocab_size=8,
        lm_head_tp_size=1,
        lm_head_tp_rank=0,
        hidden_size=128,
        num_heads=1,
        num_kv_heads=1,
        head_dim=128,
        attn_tp_size=1,
        attn_tp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        ep_size=1,
        ep_rank=0,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=128,
        params_dtype=torch.bfloat16,
        lm_head_params_dtype=torch.bfloat16,
        rms_norm_eps=1e-6,
        moe_config=None,
        quant_config=QuantizationConfig(
            "fp8_per_channel" if channel else "fp8_block", source
        ),
        parallelism_config=SimpleNamespace(
            dp_size=1, dp_rank=0, world_size=1, local_world_size=1
        ),
    )
    model = RtpModule()
    install_glm4_parameters(model, cfg, layer_ids=range(1), partitioned=False)
    return model


def inputs(channel=False):
    g = torch.Generator().manual_seed(132)
    hf = {}
    ft = {}
    h = 128
    fp = torch.float8_e4m3fn

    def val(shape):
        return torch.randn(shape, generator=g).bfloat16()

    def simple(src, target, shape, transpose=False, dtype=None):
        value = val(shape)
        if dtype:
            value = value.to(dtype)
        hf[target] = value
        ft[src] = value.T.contiguous() if transpose else value

    simple("global.embedding", "embed_tokens.weight", (8, h))
    simple("global.lm_head", "lm_head.weight", (8, h))
    simple("global.final_layernorm.gamma", "norm.weight", (h,))
    src = "layers.0."
    old = "self_attention_weights."
    for oldkey, hfkey in (
        ("pre_layernorm_weights.gamma", "input_layernorm.weight"),
        ("post_layernorm_weights.gamma", "post_attention_layernorm.weight"),
        (old + "q_layernorm.gamma", "self_attn.q_norm.weight"),
        (old + "k_layernorm.gamma", "self_attn.k_norm.weight"),
    ):
        simple(src + oldkey, src + hfkey, (h,))
    simple(
        src + "partial_moe_weights.gate.kernel", src + "mlp.gate.weight", (2, h), True
    )
    simple(
        src + "partial_moe_weights.e_score_correction_bias",
        src + "mlp.gate.e_score_correction_bias",
        (2,),
        dtype=torch.float32,
    )

    def dense(stem, names):
        ws = []
        ss = []
        for name in names:
            w = val((h, h)).to(fp)
            s = torch.rand((1, 1), generator=g) + 0.01
            hf[src + name + ".weight"] = w
            hf[src + name + ".weight_scale_inv"] = s
            ws.append(w.view(torch.uint8))
            ss.append(s)
        ft[src + stem + ".kernel"] = torch.cat(ws).view(fp).reshape(h, -1)
        ft[src + stem + ".weight_only_quant_scale"] = torch.cat(ss).reshape(1, -1)

    dense(old + "query_weight", ["self_attn." + p + "_proj" for p in "qkv"])
    dense(old + "attention_output_weight", ["self_attn.o_proj"])
    biases = [val((h,)) for _ in range(3)]
    for p, b in zip("qkv", biases):
        hf[src + f"self_attn.{p}_proj.bias"] = b
    ft[src + old + "query_weight.bias"] = torch.cat(biases)[None, :]
    for proj, stem in (
        ("gate", "intermediate_weight"),
        ("up", "intermediate_weight3"),
        ("down", "intermediate_weight2"),
    ):
        dense("ffn_weights." + stem, ["mlp.shared_experts." + proj + "_proj"])
    for stem, projs in (
        ("intermediate_weight", ("up", "gate")),
        ("intermediate_weight2", ("down",)),
    ):
        ws = []
        ss = []
        for e in range(2):
            wp = []
            sp = []
            for proj in projs:
                key = src + f"mlp.experts.{e}.{proj}_proj."
                w = val((h, h)).to(fp)
                s = torch.rand((1, 1), generator=g) + 0.01
                hf[key + "weight"] = w
                hf[key + "weight_scale_inv"] = s
                wp.append(w.view(torch.uint8))
                sp.append(s)
            ws.append(torch.cat(wp))
            ss.append(torch.cat(sp))
        ft[src + "partial_moe_weights." + stem + ".kernel"] = torch.stack(ws).view(fp)
        ft[src + "partial_moe_weights." + stem + ".weight_only_quant_scale"] = (
            torch.stack(ss)
        )
    return hf, ft


def channel_inputs():
    """The per-channel producer retains expert axes and flat-reshapes dense scales."""
    hf, ft = inputs()
    for k in list(hf):
        if k.endswith(".weight_scale_inv"):
            s = hf.pop(k)
            hf[k.removesuffix("_inv")] = s.reshape(()) * (
                1 + torch.arange(128).reshape(128, 1) / 256
            )
    for k in list(ft):
        if not k.endswith(".weight_only_quant_scale"):
            continue
        v = ft[k]
        if "partial_moe_weights" in k:
            ft[k] = (
                v.unsqueeze(-2) * (1 + torch.arange(128).reshape(128, 1) / 256)
            ).flatten(1, 2)
        else:
            ft[k] = (v.reshape(-1, 1) * (1 + torch.arange(128) / 256)).reshape(1, -1)
    # Old FfnWeight merges the shared gate/up producers before quantization.
    p = "layers.0.ffn_weights.intermediate_weight"
    w = [
        ft.pop(p + t + ".kernel").reshape(128, 128).view(torch.uint8) for t in ("", "3")
    ]
    s = [ft.pop(p + t + ".weight_only_quant_scale") for t in ("", "3")]
    ft[p + "13.kernel"] = torch.cat(w).view(torch.float8_e4m3fn).reshape(128, 256)
    ft[p + "13.weight_only_quant_scale"] = torch.cat(s, dim=1)
    return hf, ft


class GlmFTBlockTest(unittest.TestCase):
    def test_whole_tree(self):
        self._whole_tree(False)

    def test_channel_tree(self):
        self._whole_tree(True)

    def _whole_tree(self, channel):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="glm4_moe",
                        num_hidden_layers=1,
                        first_k_dense_replace=0,
                        n_shared_experts=1,
                        use_qk_norm=True,
                    )
                )
            )
            hf, ft = channel_inputs() if channel else inputs()
            write_ft(root, ft)
            base = tree(root, channel)
            self.assertTrue(base._layout.correction_bias)
            for name, tensor in hf.items():
                base.load_weights({name: tensor})
            NewModelLoader._validate_loaded_weights(base)
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    m = tree(root, channel)
                    cp = FTCheckpoint.open(root, CheckpointTopology())
                    cp.read_backend = method
                    load_gqa_ffn_ft_checkpoint(m, cp)
                    NewModelLoader._validate_loaded_weights(m)
                    a, b = base.state_dict(), m.state_dict()
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        if channel and key.endswith(
                            ("experts.w13_scale", "experts.w2_scale")
                        ):
                            # HF only assembles these derived runtime views during
                            # device postprocessing; compare their actual channel
                            # sources instead of an unused HF zero buffer.
                            if key.endswith("w13_scale"):
                                expected = torch.cat(
                                    (
                                        base.layers[0].mlp.experts._up_ch_scales,
                                        base.layers[0].mlp.experts._gate_ch_scales,
                                    ),
                                    dim=1,
                                )
                            else:
                                expected = base.layers[0].mlp.experts._down_ch_scales
                            torch.testing.assert_close(expected, b[key], atol=0, rtol=0)
                            continue
                        torch.testing.assert_close(
                            a[key].float(), b[key].float(), atol=0, rtol=0, msg=key
                        )


if __name__ == "__main__":
    unittest.main()
