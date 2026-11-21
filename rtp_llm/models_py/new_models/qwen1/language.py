"""Qwen-7B-Chat's HF fused-MHA/gated-MLP checkpoint layout."""

import re

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen1CheckpointMapper(WeightsMapper):
    _layer = re.compile(r"^transformer\.h\.(\d+)\.(.+)$")
    _layers = {
        "ln_1.weight": "input_layernorm.weight",
        "ln_2.weight": "post_attention_layernorm.weight",
        # HF stores [Q; K; V] along the output dimension. The QKV leaf takes
        # its shard of each projection, not one slice of their concatenation.
        "attn.c_attn.weight": "self_attn.qkv_proj.weight",
        "attn.c_attn.bias": "self_attn.qkv_proj.bias",
        "attn.c_proj.weight": "self_attn.o_proj.weight",
        # Qwen1 computes silu(w2(x)) * w1(x).
        "mlp.w2.weight": "mlp.gate_proj.weight",
        "mlp.w1.weight": "mlp.up_proj.weight",
        "mlp.c_proj.weight": "mlp.down_proj.weight",
    }

    def __init__(self):
        super().__init__(
            exact_mapping={
                "transformer.wte.weight": "embed_tokens.weight",
                "transformer.ln_f.weight": "norm.weight",
            }
        )

    def map_name(self, name):
        match = self._layer.fullmatch(name)
        if match and match[2] in self._layers:
            return f"layers.{match[1]}.{self._layers[match[2]]}"
        return super().map_name(name)


class Qwen1ForCausalLM(Qwen2ForCausalLM):
    """Reuse the biased QKV/RMSNorm/SwiGLU runtime, not Qwen2's HF layout.

    QWen_7B resolves rotary/logn settings, MHA heads and the per-branch MLP
    width from config.json. Preserve that config through prefill and decode.
    """

    WEIGHTS_MAPPER = Qwen1CheckpointMapper()
    ppu_dense_streamed_scratch = True
