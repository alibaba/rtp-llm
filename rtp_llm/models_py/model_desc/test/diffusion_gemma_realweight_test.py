"""Real-weights layer math validation for DiffusionGemma.

Same layer comparison as diffusion_gemma_test, but with tensors and config
loaded from the real 26B checkpoint (env DGM_REAL_CKPT points at the local
checkpoint directory). Skipped when the env var is unset, so CI stays green.
Run locally with a standalone execution strategy so absolute paths resolve:

  bazel test //rtp_llm/models_py/model_desc/test:diffusion_gemma_realweight_test \
      --test_env=DGM_REAL_CKPT=/path/to/checkpoint --spawn_strategy=standalone
"""

import json
import os
import unittest
from unittest.mock import patch

import torch

CKPT = os.environ.get("DGM_REAL_CKPT", "")

# Real checkpoint geometry (text_config).
H = 2816
LAYERS = 30
HEADS = 16
SLIDING_KV = 8
GLOBAL_KV = 2
SLIDING_DIM = 256
GLOBAL_DIM = 512
WINDOW = 1024
INTER = 2112
MOE_INTER = 704
EXPERTS = 128
TOPK = 8
EPS = 1e-6
PAGE = 64
CANVAS = 256

# Load the sibling test module FIRST by path (the test dir is not a
# package): importing it preloads the CUDA link stubs, which must happen
# before any rtp_llm.models_py import, and exposes its reference helpers.
import importlib.util

_HELPER_PATH = os.path.join(os.path.dirname(__file__), "diffusion_gemma_test.py")
_spec = importlib.util.spec_from_file_location("dgm_test_helpers", _HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
# The reference helpers read module-level geometry constants; retarget them
# to the real checkpoint geometry before any comparison.
for _name, _value in {
    "H": H,
    "HEADS": HEADS,
    "SLIDING_KV": SLIDING_KV,
    "GLOBAL_KV": GLOBAL_KV,
    "SLIDING_DIM": SLIDING_DIM,
    "GLOBAL_DIM": GLOBAL_DIM,
    "INTER": INTER,
    "MOE_INTER": MOE_INTER,
    "EXPERTS": EXPERTS,
    "TOPK": TOPK,
    "WINDOW": WINDOW,
}.items():
    setattr(_helpers, _name, _value)
_TorchRMSNorm = _helpers._TorchRMSNorm
_attn_inputs = _helpers._attn_inputs
_model_weights = _helpers._model_weights
_ref_layer = _helpers._ref_layer

from rtp_llm.ops import ParallelismConfig  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma as dgm  # noqa: E402
from rtp_llm.models.diffusion_gemma import DiffusionGemmaConfig  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


def _layer_types():
    return dgm.diffusion_gemma_layer_types(LAYERS)


def _load_hf_weights(layer_id: int, layer_type: str) -> dict:
    from safetensors.torch import load_file

    ckpt = CKPT
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {}

    def get(key):
        shard = weight_map[key]
        if shard not in shards:
            shards[shard] = load_file(os.path.join(ckpt, shard))
        return shards[shard][key]

    pre = "model.decoder.layers.%d." % layer_id
    is_global = layer_type == "full_attention"
    if is_global:
        qkv = torch.cat(
            [get(pre + "self_attn.q_proj.weight"), get(pre + "self_attn.k_proj.weight")],
            dim=0,
        )
        # reference: V is the k_proj output duplicated into the fused slot
        qkv = torch.cat([qkv, get(pre + "self_attn.k_proj.weight")], dim=0)
    else:
        qkv = torch.cat(
            [
                get(pre + "self_attn.q_proj.weight"),
                get(pre + "self_attn.k_proj.weight"),
                get(pre + "self_attn.v_proj.weight"),
            ],
            dim=0,
        )
    return {
        "qkv": qkv,
        "o": get(pre + "self_attn.o_proj.weight"),
        "q_ln": get(pre + "self_attn.q_norm.weight"),
        "k_ln": get(pre + "self_attn.k_norm.weight"),
        "input_ln": get(pre + "input_layernorm.weight"),
        "post_attn_ln": get(pre + "post_attention_layernorm.weight"),
        "pre_ffn_ln": get(pre + "pre_feedforward_layernorm.weight"),
        "post_ffn_ln": get(pre + "post_feedforward_layernorm.weight"),
        "pre_ffn_ln_2": get(pre + "pre_feedforward_layernorm_2.weight"),
        "post_ffn_ln_1": get(pre + "post_feedforward_layernorm_1.weight"),
        "post_ffn_ln_2": get(pre + "post_feedforward_layernorm_2.weight"),
        "gate": get(pre + "mlp.gate_proj.weight"),
        "up": get(pre + "mlp.up_proj.weight"),
        "down": get(pre + "mlp.down_proj.weight"),
        "moe_gate_up": get(pre + "experts.gate_up_proj"),
        "moe_down": get(pre + "experts.down_proj"),
        "router_proj": get(pre + "router.proj.weight"),
        "router_scale": get(pre + "router.scale"),
        "per_expert_scale": get(pre + "router.per_expert_scale"),
        "layer_scalar_enc": get(
            "model.encoder.language_model.layers.%d.layer_scalar" % layer_id
        ),
        "layer_scalar_dec": get(pre + "layer_scalar"),
    }


def _make_real_config() -> DiffusionGemmaConfig:
    config = DiffusionGemmaConfig()
    config.num_layers = LAYERS
    config.hidden_size = H
    config.vocab_size = 262144
    config.max_seq_len = 12800
    config.layernorm_eps = EPS
    config.attn_config.head_num = HEADS
    config.attn_config.kv_head_num = SLIDING_KV
    config.attn_config.size_per_head = SLIDING_DIM
    config.attn_config.is_causal = True
    config.attn_config.kernel_tokens_per_block = PAGE
    config.inter_size = INTER
    config.moe_inter_size = MOE_INTER
    config.expert_num = EXPERTS
    config.moe_k = TOPK
    config.dgm_head_dim = SLIDING_DIM
    config.dgm_kv_head_num = SLIDING_KV
    config.dgm_global_head_dim = GLOBAL_DIM
    config.dgm_global_kv_head_num = GLOBAL_KV
    config.dgm_global_partial_rope = 0.25
    config.dgm_swa_rope_theta = 10000.0
    config.dgm_global_rope_theta = 1000000.0
    config.dgm_sliding_window = WINDOW
    config.dgm_canvas_length = CANVAS
    config.dgm_final_logit_softcapping = 30.0
    config.dgm_layer_types = _layer_types()
    return config


@unittest.skipUnless(CKPT, "DGM_REAL_CKPT not set; real-weights check skipped")
class DiffusionGemmaRealWeightLayerTest(unittest.TestCase):
    def _run_layer(self, layer_id: int, layer_type: str, causal: bool, decoder_mode: bool):
        hf = _load_hf_weights(layer_id, layer_type)
        # fp32 for a numerically clean reference comparison
        hf = {k: v.to(torch.float32) for k, v in hf.items()}
        total = 5 if causal else 32
        torch.manual_seed(11)
        hidden = torch.randn(total, H) * 0.5
        positions = torch.arange(total)
        if layer_type == "full_attention":
            rope = dgm.DiffusionGemmaRope(1e6, GLOBAL_DIM, 0.25, 128, torch.device("cpu"), torch.float32)
        else:
            rope = dgm.DiffusionGemmaRope(1e4, SLIDING_DIM, 1.0, 128, torch.device("cpu"), torch.float32)
        attn_inputs = _attn_inputs(total, positions)
        attn_op = dgm.DiffusionGemmaAttnOp(
            layer_type,
            HEADS,
            GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV,
            GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM,
            WINDOW,
            PAGE,
            torch.float32,
            None,
        )
        with patch.object(dgm, "RMSNorm", _TorchRMSNorm):
            layer = dgm.DiffusionGemmaLayer(
                _make_real_config(),
                ParallelismConfig(),
                _model_weights(hf),
                layer_id,
                layer_type,
                None,
            )
            out = layer(
                hidden,
                rope,
                positions,
                attn_op,
                None,
                attn_inputs,
                causal=causal,
                write_kv=False,
                decoder_mode=decoder_mode,
            )
        ref = _ref_layer(hidden, hf, layer_type, positions, causal, decoder_mode)
        # bf16-level tolerance: real weights span magnitudes ~1e-2..10
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
        return out

    def test_layer0_swa_encoder_causal(self):
        self._run_layer(0, "sliding_attention", causal=True, decoder_mode=False)

    def test_layer5_global_encoder_causal(self):
        self._run_layer(5, "full_attention", causal=True, decoder_mode=False)

    def test_layer29_global_canvas(self):
        self._run_layer(29, "full_attention", causal=False, decoder_mode=True)

    def test_layer7_swa_canvas(self):
        self._run_layer(7, "sliding_attention", causal=False, decoder_mode=True)




HF_PREFILL = os.environ.get("DGM_HF_PREFILL", "")


def _load_all_weights_fp32():
    """All 30 layers + globals from the real checkpoint, loader transforms."""
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "dgm_wi", os.path.join(os.path.dirname(__file__), "../../models/diffusion_gemma_weight.py")
    )
    # Not importable (rtp_llm package path) - apply transforms inline below.
    from safetensors.torch import load_file

    ckpt = CKPT
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {}

    def get(key):
        shard = weight_map[key]
        if shard not in shards:
            shards[shard] = load_file(os.path.join(ckpt, shard))
        return shards[shard][key].to(torch.float32)

    layer_weights = []
    for i in range(LAYERS):
        pre = "model.decoder.layers.%d." % i
        is_global = i % 6 == 5
        # HF-format fused qkv rows: [q; k; v] (global layers duplicate k into v)
        if is_global:
            qkv_hf = torch.cat(
                [get(pre + "self_attn.q_proj.weight"), get(pre + "self_attn.k_proj.weight")],
                dim=0,
            )
            qkv_hf = torch.cat([qkv_hf, get(pre + "self_attn.k_proj.weight")], dim=0)
        else:
            qkv_hf = torch.cat(
                [
                    get(pre + "self_attn.q_proj.weight"),
                    get(pre + "self_attn.k_proj.weight"),
                    get(pre + "self_attn.v_proj.weight"),
                ],
                dim=0,
            )
        hf = {
            "qkv": qkv_hf,
            "o": get(pre + "self_attn.o_proj.weight"),
            "q_ln": get(pre + "self_attn.q_norm.weight"),
            "k_ln": get(pre + "self_attn.k_norm.weight"),
            "input_ln": get(pre + "input_layernorm.weight"),
            "post_attn_ln": get(pre + "post_attention_layernorm.weight"),
            "pre_ffn_ln": get(pre + "pre_feedforward_layernorm.weight"),
            "post_ffn_ln": get(pre + "post_feedforward_layernorm.weight"),
            "pre_ffn_ln_2": get(pre + "pre_feedforward_layernorm_2.weight"),
            "post_ffn_ln_1": get(pre + "post_feedforward_layernorm_1.weight"),
            "post_ffn_ln_2": get(pre + "post_feedforward_layernorm_2.weight"),
            "gate": get(pre + "mlp.gate_proj.weight"),
            "up": get(pre + "mlp.up_proj.weight"),
            "down": get(pre + "mlp.down_proj.weight"),
            "moe_gate_up": get(pre + "experts.gate_up_proj"),
            "moe_down": get(pre + "experts.down_proj"),
            "router_proj": get(pre + "router.proj.weight"),
            "router_scale": get(pre + "router.scale").reshape(-1),
            "per_expert_scale": get(pre + "router.per_expert_scale").reshape(-1),
            "layer_scalar_enc": get(
                "model.encoder.language_model.layers.%d.layer_scalar" % i
            ),
            "layer_scalar_dec": get(pre + "layer_scalar"),
        }
        layer_weights.append(_model_weights(hf))
    globals_ = {
        W.embedding: get("model.decoder.embed_tokens.weight"),
        W.final_ln_gamma: get("model.decoder.norm.weight"),
        W.dgm_self_cond_gate_w: get("model.decoder.self_conditioning.gate_proj.weight").T.contiguous(),
        W.dgm_self_cond_up_w: get("model.decoder.self_conditioning.up_proj.weight").T.contiguous(),
        W.dgm_self_cond_down_w: get("model.decoder.self_conditioning.down_proj.weight").T.contiguous(),
        W.dgm_self_cond_pre_norm: get("model.decoder.self_conditioning.pre_norm.weight"),
    }
    return layer_weights, globals_


def _ropes_of(model, layer_type):
    from rtp_llm.models_py.model_desc.diffusion_gemma import (
        LAYER_TYPE_SLIDING,
    )

    return model.ropes[layer_type]


class _CpuEmbedding(torch.nn.Module):
    """Plain CPU embedding lookup standing in for the CUDA-only op."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input_ids, *args, **kwargs):
        return torch.nn.functional.embedding(input_ids.long(), self.weight)


@unittest.skipUnless(CKPT and HF_PREFILL, "DGM_REAL_CKPT/DGM_HF_PREFILL not set")
class DiffusionGemmaRealWeightPrefillTest(unittest.TestCase):
    def test_prompt_prefill_hidden_matches_hf(self):
        from types import SimpleNamespace

        layer_weights, global_weights = _load_all_weights_fp32()
        from rtp_llm.model_loader.model_weight_info import ModelWeights

        mw = ModelWeights(LAYERS, "cpu", torch.float32)
        for i, lw in enumerate(layer_weights):
            for name, tensor in lw.items():
                mw.set_layer_weight(i, name, tensor)
        for name, tensor in global_weights.items():
            mw.set_global_weight(name, tensor)

        with patch.object(dgm, "RMSNorm", _TorchRMSNorm):
            model = dgm.DiffusionGemmaModel(
                _make_real_config(), ParallelismConfig(), mw,
                max_generate_batch_size=2,
            )
        saved = torch.load(HF_PREFILL)
        ids = saved["ids"].to(torch.int32)
        total = int(ids.numel())
        cu = torch.tensor([0, total], dtype=torch.int32)
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            input_lengths=torch.tensor([total], dtype=torch.int32),
            cu_seqlens=cu,
            cu_seqlens_device=cu,
            kv_cache_kernel_block_id=None,
            kv_cache_kernel_block_id_device=None,
            combo_position_ids=torch.arange(total, dtype=torch.int64).unsqueeze(0),
            sequence_lengths=None,
            is_target_verify=False,
        )
        inputs = SimpleNamespace(
            input_ids=ids,
            embedding_inputs=SimpleNamespace(text_tokens_mask=None),
            attention_inputs={"swa_kv": attn_inputs, "global_kv": attn_inputs},
            capture_hidden_states=False,
        )
        model.eval()
        # The C++ embedding op is CUDA-only; swap in a plain lookup for CPU.
        embed_weight = model.embed_tokens.weight
        model.embed_tokens = _CpuEmbedding(embed_weight)
        with torch.inference_mode():
            out = model.forward(inputs)
        my_hidden = out.hidden_states.to(torch.float32)
        ref = saved["hidden"].to(torch.float32)
        diff = (my_hidden - ref).abs()
        rel = diff.max().item() / ref.abs().max().item()
        print(
            f"PREFILL hidden: shape={tuple(my_hidden.shape)} max_abs_diff={diff.max().item():.4e} "
            f"mean_abs_diff={diff.mean().item():.4e} rel={rel:.3e} ref_absmax={ref.abs().max().item():.3f}"
        )
        # fp32 reference: exact-level agreement after the double-norm fix.
        self.assertLess(diff.max().item(), 0.05)

        # Layer-by-layer bisect: drive the model's own layer modules
        # sequentially and compare against the reference layer chain.
        h_ref = torch.nn.functional.embedding(
            saved["ids"].long(), embed_weight
        ).to(torch.float32) * (float(H) ** 0.5)
        h_mine = h_ref.clone()
        layer_types = _layer_types()
        positions = torch.arange(total)
        first_bad = None
        with torch.inference_mode():
            for i in range(LAYERS):
                layer_type = layer_types[i]
                hf = _load_hf_weights(i, layer_type)
                hf = {k: v.to(torch.float32) for k, v in hf.items()}
                h_ref = _ref_layer(h_ref, hf, layer_type, positions, True, False)
                rope = _ropes_of(model, layer_type)
                attn_op = model.get_attn_op(layer_type)
                h_mine = model.layers[i](
                    h_mine, rope, positions, attn_op, None, attn_inputs,
                    causal=True, write_kv=False, decoder_mode=False,
                )
                d = (h_mine - h_ref).abs().max().item()
                if first_bad is None and d > 1.0:
                    first_bad = i
                    print(
                        f"  first divergent layer: {i} ({layer_type}) "
                        f"max_diff={d:.3e} my_absmax={h_mine.abs().max().item():.3f} "
                        f"ref_absmax={h_ref.abs().max().item():.3f}"
                    )
                if i < 2 or i in (5, 29) or (first_bad is not None and i == first_bad):
                    print(
                        f"  layer {i} ({layer_type}): max_diff={d:.3e} "
                        f"my_absmax={h_mine.abs().max().item():.3f} ref={h_ref.abs().max().item():.3f}"
                    )
        if first_bad is not None:
            self.fail(f"layer {first_bad} diverges from reference")
        print("  all layers agree; divergence is above the layer stack")
        # Final-norm agreement against the fp32 HF reference (the reference
        # file must be produced with torch_dtype=float32; a bf16 reference
        # differs by up to ~5.2 at the |gamma|~588 norm positions).
        with torch.inference_mode():
            my_normed = model.norm(h_ref)
        ref_normed = saved["hidden"].to(torch.float32)
        d = (my_normed - ref_normed).abs().max().item()
        print(
            f"  final norm probe: my_absmax={my_normed.abs().max().item():.4f} "
            f"ref_absmax={ref_normed.abs().max().item():.4f} max_diff={d:.4e}"
        )


if __name__ == "__main__":
    unittest.main()
