"""Qwen-7B-Chat's HF fused-MHA/gated-MLP checkpoint layout."""

import re

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen1CheckpointMapper(WeightsMapper):
    _layer = re.compile(r"^transformer\.h\.(\d+)\.(.+)$")
    _layers = {
        "ln_1": "input_layernorm",
        "ln_2": "post_attention_layernorm",
        # HF stores [Q; K; V] along the output dimension. The QKV leaf takes
        # its shard of each projection, not one slice of their concatenation.
        "attn.c_attn": "self_attn.qkv_proj",
        "attn.c_proj": "self_attn.o_proj",
        # Qwen1 computes silu(w2(x)) * w1(x).
        "mlp.w2": "mlp.gate_proj",
        "mlp.w1": "mlp.up_proj",
        "mlp.c_proj": "mlp.down_proj",
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
        if match:
            owner, _, field = match[2].rpartition(".")
            if owner in self._layers:
                # Preserve the actual checkpoint field. The projection's
                # QuantMethod validates and shards qweight/qzeros/scales etc.
                return f"layers.{match[1]}.{self._layers[owner]}.{field}"
        return super().map_name(name)


class Qwen1ForCausalLM(Qwen2ForCausalLM):
    """Reuse the biased QKV/RMSNorm/SwiGLU runtime, not Qwen2's HF layout.

    QWen_7B resolves rotary/logn settings, MHA heads and the per-branch MLP
    width from config.json. Preserve that config through prefill and decode.
    """

    WEIGHTS_MAPPER = Qwen1CheckpointMapper()
    ppu_dense_streamed_scratch = True

    @staticmethod
    def online_checkpoint_name(name):
        # Preserve original transformer.h.* checkpoint dispatch. Public RL
        # clients may instead send canonical model/decoder-layer spellings.
        if name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
            return name.removeprefix("model.")
        return name
