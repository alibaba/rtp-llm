"""MOSS-VL desc unit test: cross-attention layer + interleaved mrope vs reference.

Small fp32 configuration; the reference layer (MossVLCrossAttentionDecoderLayer
from the checkpoint's modeling_moss_vl.py) and our layer run the same synthetic
weights through prefill (new vision states) and decode (cached vision KV).
Weight tensors are placed into the RTP W-name contract exactly as the loader
produces them (merge_qkv_hf / transpose transforms).
"""

import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from typing import Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W, merge_qkv_hf, transpose

CKPT = os.environ.get(
    "MOSS_VL_CKPT_PATH", "/mnt/hf3fs/3fs/liukan.lk/models/MOSS-VL-Instruct-0708"
)

HIDDEN = 64
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 32
INTER = 128
EPS = 1e-6
VOCAB = 96
MROPE_SECTION = (6, 5, 5)
CROSS_LAYERS = [1, 3]
NUM_LAYERS = 4


def _stub_callable():
    class _Shim:
        def __call__(self, *args, **kwargs):
            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return self

        def __getattr__(self, name):
            return _stub_callable()

    return _Shim()


def _install_transformers_shims():
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        generic.OutputRecorder = _stub_callable()
    # transformers 5.x removed the 'default' rope init key that the reference
    # config uses (rope_type 'default'); alias it to a plain inv_freq init
    # (identical math to the 4.x default: no scaling, base from config).
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _plain_default_rope_init(config, device):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS.setdefault("default", _plain_default_rope_init)
    for module_name, names in (
        ("transformers.integrations", ["use_kernel_forward_from_hub"]),
        ("transformers.modeling_layers", ["GradientCheckpointingLayer"]),
        ("transformers.processing_utils", ["Unpack"]),
        ("transformers.utils", ["TransformersKwargs", "auto_docstring"]),
        ("transformers.utils.deprecation", ["deprecate_kwarg"]),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names:
            if not hasattr(module, name):
                if name == "GradientCheckpointingLayer":
                    module.GradientCheckpointingLayer = type(
                        "GradientCheckpointingLayer", (torch.nn.Module,), {}
                    )
                else:
                    setattr(module, name, _stub_callable())


def _load_reference():
    _install_transformers_shims()
    cfg_spec = importlib.util.spec_from_file_location(
        "moss_vl_desc_ref_configuration", os.path.join(CKPT, "configuration_moss_vl.py")
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    from pathlib import Path

    source = Path(CKPT, "modeling_moss_vl.py").read_text().replace(
        "from .configuration_moss_vl import",
        "from moss_vl_desc_ref_configuration import",
    )
    spec = importlib.util.spec_from_file_location(
        "moss_vl_desc_ref_modeling", os.path.join(CKPT, "modeling_moss_vl.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moss_vl_desc_ref_modeling"] = mod
    sys.modules["moss_vl_desc_ref_configuration"] = cfg_mod
    exec(
        compile(source, os.path.join(CKPT, "modeling_moss_vl.py"), "exec"), mod.__dict__
    )
    return cfg_mod, mod


def _reference_text_config(cfg_mod):
    config = cfg_mod.MossVLTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        rope_theta=10000.0,
        max_position_embeddings=512,
        cross_attention_layers=CROSS_LAYERS,
        attention_bias=False,
    )
    config.mrope_section = list(MROPE_SECTION)
    config.rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": list(MROPE_SECTION),
        "rope_type": "default",
    }
    config._attn_implementation = "eager"
    return config


def _our_config() -> ModelConfig:
    config = ModelConfig()
    config.mm_related_params.config["cross_attention_layers"] = list(CROSS_LAYERS)
    config.hidden_size = HIDDEN
    config.num_layers = NUM_LAYERS
    config.inter_size = INTER
    config.vocab_size = VOCAB
    config.layernorm_eps = EPS
    config.attn_config.head_num = HEADS
    config.attn_config.kv_head_num = KV_HEADS
    config.attn_config.size_per_head = HEAD_DIM
    config.attn_config.rope_config.dim = HEAD_DIM
    config.attn_config.rope_config.base = 10000
    config.attn_config.rope_config.style = 7
    config.attn_config.rope_config.index_factor = 3
    config.attn_config.rope_config.mrope_dim1, config.attn_config.rope_config.mrope_dim2, config.attn_config.rope_config.mrope_dim3 = MROPE_SECTION
    config.attn_config.rope_config.mrope_interleaved = True
    config.activation_type = "SiGLU"
    return config


def _weights_from_reference(ref_layer) -> Dict[str, torch.Tensor]:
    """Map reference cross-layer state_dict to the RTP W-name contract."""
    state = ref_layer.state_dict()
    prefix = "cross_attn."
    weights: Dict[str, torch.Tensor] = {
        W.pre_ln_gamma: state["input_layernorm.weight"].clone(),
        W.post_ln_gamma: state["post_attention_layernorm.weight"].clone(),
        W.q_ln_gamma: state[prefix + "q_norm.weight"].clone(),
        W.k_ln_gamma: state[prefix + "k_norm.weight"].clone(),
        W.cross_attn_attn_gate: state["cross_attn_attn_gate"].clone(),
        W.cross_attn_mlp_gate: state["cross_attn_mlp_gate"].clone(),
        W.attn_qkv_w: merge_qkv_hf(
            [
                state[prefix + "q_proj.weight"].clone(),
                state[prefix + "k_proj.weight"].clone(),
                state[prefix + "v_proj.weight"].clone(),
            ]
        ),
        W.attn_o_w: transpose([state[prefix + "o_proj.weight"].clone()]),
        W.ffn_w1: transpose([state["mlp.gate_proj.weight"].clone()]),
        W.ffn_w3: transpose([state["mlp.up_proj.weight"].clone()]),
        W.ffn_w2: transpose([state["mlp.down_proj.weight"].clone()]),
    }
    return weights


class MossVLDescCrossLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(CKPT):
            raise unittest.SkipTest(f"checkpoint not available: {CKPT}")
        from rtp_llm.models_py.model_desc.mossvl import (
            MossVLCrossAttentionDecoderLayer,
            MossVLCrossKVStore,
            MossVLInterleavedMrope,
        )

        cls.cfg_mod, cls.ref_mod = _load_reference()
        cls.ref_config = _reference_text_config(cls.cfg_mod)
        torch.manual_seed(11)
        # bf16 on both sides: this is the production dtype and the production
        # RMSNorm dispatch (flashinfer) requires a half dtype.
        cls.dtype = torch.bfloat16
        cls.ref_layer = cls.ref_mod.MossVLCrossAttentionDecoderLayer(
            cls.ref_config, layer_idx=CROSS_LAYERS[0]
        ).to("cuda", cls.dtype)
        cls.ours_layer = MossVLCrossAttentionDecoderLayer(
            _our_config(), ParallelismConfig(), CROSS_LAYERS[0],
            _weights_from_reference(cls.ref_layer), None, None,
        ).to("cuda", cls.dtype)
        cls.store = MossVLCrossKVStore(NUM_LAYERS)
        cls.rope = MossVLInterleavedMrope(HEAD_DIM, 10000.0, MROPE_SECTION).to("cuda")
        cls.ref_rotary = cls.ref_mod.MossVLTextRotaryEmbedding(config=cls.ref_config).to("cuda")

    def _positions(self, seq_len, markers, eh, ew):
        """Text 3D positions with Qwen2-VL-style marker shifts (reference math)."""
        pos = [[i, i, i] for i in range(seq_len)]
        shift = max(eh, ew) + 1
        total_shift = 0
        for marker in markers:
            for p in range(marker + 1, seq_len):
                for k in range(3):
                    pos[p][k] += shift - 1
            # marker itself sits at t_val + max_hw
            t_val = pos[marker][0]
            pos[marker] = [t_val + max(eh, ew)] * 3
            total_shift += shift
        return torch.tensor(pos, dtype=torch.long, device="cuda")

    def test_mrope_cos_sin_matches_reference(self):
        seq = 12
        positions = self._positions(seq, markers=[4], eh=2, ew=3)
        ours_cos, ours_sin = self.rope(positions)
        ref_positions = positions.t().reshape(3, 1, seq)  # reference (3, B, L)
        # fp32 zeros: the reference casts cos/sin to the probe dtype
        ref_cos, ref_sin = self.ref_rotary(
            torch.zeros(1, seq, HIDDEN, device="cuda"), ref_positions
        )
        ref_cos = ref_cos.reshape(seq, HEAD_DIM)
        ref_sin = ref_sin.reshape(seq, HEAD_DIM)
        self.assertEqual(ours_cos.shape, ref_cos.shape)
        self.assertLess((ours_cos - ref_cos).abs().max().item(), 1e-5)
        self.assertLess((ours_sin - ref_sin).abs().max().item(), 1e-5)

    def test_cross_layer_prefill_and_decode_match_reference(self):
        seq = 12
        eh, ew = 2, 3
        rows = eh * ew + 1  # tokens + separator
        markers = [4]
        torch.manual_seed(21)
        hidden = torch.randn(seq, HIDDEN, device="cuda").to(self.dtype)
        features = torch.randn(rows, HIDDEN, device="cuda").to(self.dtype)
        positions = self._positions(seq, markers, eh, ew)
        # vision positions: tokens (t, t+y, t+x), separator at marker position
        t_val = int(positions[markers[0], 0]) - max(eh, ew)
        vision_positions = []
        for y in range(eh):
            for x in range(ew):
                vision_positions.append([t_val, t_val + y, t_val + x])
        vision_positions.append([int(positions[markers[0], 0])] * 3)
        vision_positions = torch.tensor(vision_positions, dtype=torch.long, device="cuda")

        ours_cos, ours_sin = self.rope(positions)
        k_cos, k_sin = self.rope(vision_positions)
        mask = torch.zeros(seq, rows, device="cuda")
        mask[: markers[0]] = float("-inf")

        with torch.no_grad():
            ours_out = self.ours_layer(
                hidden, features, ours_cos, ours_sin, k_cos, k_sin, self.store, mask
            )
            # decode step: one new token, no new vision states
            dec_hidden = torch.randn(1, HIDDEN, device="cuda").to(self.dtype)
            dec_positions = torch.tensor(
                [[seq, seq, seq]], dtype=torch.long, device="cuda"
            )
            dec_cos, dec_sin = self.rope(dec_positions)
            ours_dec = self.ours_layer(
                dec_hidden, None, dec_cos, dec_sin, None, None, self.store, None
            )

        # reference side
        with torch.no_grad():
            ref_cache = self.ref_mod.DynamicCache()
            q_pos = positions.t().reshape(3, 1, seq)
            v_pos = vision_positions.t().reshape(3, 1, rows)
            ref_pe = self.ref_rotary(torch.zeros(1, seq, HIDDEN, device="cuda", dtype=self.dtype), q_pos)
            ref_vpe = self.ref_rotary(torch.zeros(1, rows, HIDDEN, device="cuda", dtype=self.dtype), v_pos)
            ref_mask = mask.reshape(1, 1, seq, rows).clone()
            ref_mask[ref_mask == float("-inf")] = torch.finfo(torch.float32).min
            ref_out = self.ref_layer(
                hidden.unsqueeze(0),
                position_embeddings=ref_pe,
                cross_attention_states=features.unsqueeze(0),
                cross_attention_mask=ref_mask,
                past_key_values=ref_cache,
                use_cache=True,
                cache_position=torch.arange(seq, device="cuda"),
                vision_cache_position=torch.arange(rows, device="cuda"),
                vision_position_embeddings=ref_vpe,
            )
            dec_q_pos = dec_positions.t().reshape(3, 1, 1)
            ref_dec_pe = self.ref_rotary(
                torch.zeros(1, 1, HIDDEN, device="cuda", dtype=self.dtype), dec_q_pos
            )
            ref_dec = self.ref_layer(
                dec_hidden.unsqueeze(0),
                position_embeddings=ref_dec_pe,
                cross_attention_states=None,
                past_key_values=ref_cache,
                use_cache=True,
                cache_position=torch.tensor([seq], device="cuda"),
                # continuation positions: must not start at 0, otherwise the
                # reference treats the cross KV cache as absent
                vision_cache_position=torch.arange(rows, 2 * rows, device="cuda"),
            )

        prefill_diff = (ours_out.float() - ref_out[0].float()).abs().max().item()
        decode_diff = (ours_dec.float() - ref_dec[0].float()).abs().max().item()
        print(f"[desc] cross-layer prefill max diff={prefill_diff:.3e}")
        print(f"[desc] cross-layer decode  max diff={decode_diff:.3e}")
        # single layer, bf16 production dtype
        self.assertLess(prefill_diff, 5e-2, f"prefill diff {prefill_diff}")
        self.assertLess(decode_diff, 5e-2, f"decode diff {decode_diff}")

        # row-visibility variant: reference full_text_row_masked_out_mask vs
        # our row_visible (contributions zeroed, residual survives)
        from rtp_llm.models_py.model_desc.mossvl import build_moss_vl_cross_inputs

        cross = build_moss_vl_cross_inputs(
            [features], [markers[0]], [(1, eh, ew)], positions,
            self.rope, hidden.device, hidden.dtype,
        )
        self.store.reset()
        with torch.no_grad():
            ours_masked = self.ours_layer(
                hidden, features, ours_cos, ours_sin, k_cos, k_sin, self.store,
                cross.attn_mask, cross.row_visible,
            )
            ftm = torch.ones(1, 1, seq, 1, device="cuda", dtype=self.dtype)
            ftm[..., : markers[0], :] = 0.0
            ref_masked = self.ref_layer(
                hidden.unsqueeze(0),
                position_embeddings=ref_pe,
                cross_attention_states=features.unsqueeze(0),
                cross_attention_mask=ref_mask,
                full_text_row_masked_out_mask=ftm,
                past_key_values=self.ref_mod.DynamicCache(),
                use_cache=False,
                cache_position=torch.arange(seq, device="cuda"),
                vision_cache_position=torch.arange(rows, device="cuda"),
                vision_position_embeddings=ref_vpe,
            )
        masked_diff = (ours_masked.float() - ref_masked[0].float()).abs().max().item()
        print(f"[desc] cross-layer row-visibility max diff={masked_diff:.3e}")
        self.assertLess(masked_diff, 5e-2, f"row-visibility diff {masked_diff}")


if __name__ == "__main__":
    unittest.main()
