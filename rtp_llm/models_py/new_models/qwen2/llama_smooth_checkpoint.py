"""The llama2.pt dialect written by utils/smooth_quant_convert/llama."""

import re

from rtp_llm.models_py.new_models.qwen2.trt_checkpoint import Qwen2CheckpointMapper


class LlamaSmoothCheckpointMapper(Qwen2CheckpointMapper):
    _layer = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    _projection = {
        "attention.query_key_value": "self_attn.qkv_proj",
        "attention.dense": "self_attn.o_proj",
        "mlp.fc": "mlp.gate_proj",
        "mlp.gate": "mlp.up_proj",
        "mlp.proj": "mlp.down_proj",
    }

    def _describe(self, name):
        m = self._layer.fullmatch(name)
        if m is not None:
            for source, target in self._projection.items():
                if m[2].startswith(source + "."):
                    return f"layers.{m[1]}.{target}", m[2][len(source) + 1 :]
        return None, None

    def unused_calibration(self, name):
        # The converter emits alternate static calibrations alongside the
        # per-output weight scale. The dynamic-per-token executor uses neither.
        _, field = self._describe(name)
        return field in ("scale_x_orig_quant", "scale_w_quant_orig")

    def map_name(self, name):
        owner, field = self._describe(name)
        aliases = {
            "weight.int8.col": "qweight",
            "scale_w_quant_orig.col": "scales",
            "smoother": "smoother",
        }
        if field in aliases:
            return owner + "." + aliases[field]
        m = self._layer.fullmatch(name)
        if m and m[2] == "post_layernorm.weight":
            return f"layers.{m[1]}.post_attention_layernorm.weight"
        return super().map_name(name)

    def apply(self, weights):
        def converted():
            for name, value in weights:
                if self.unused_calibration(name):
                    continue
                owner, field = self._describe(name)
                if owner and owner.endswith(".qkv_proj"):
                    # MHA writes [H, 3, head_width] and [3, head_width].
                    # GQA already writes the flat concatenated Q/K/V widths.
                    if (
                        field == "weight.int8.col"
                        and value.ndim == 3
                        and value.shape[1] == 3
                    ):
                        value = value.flatten(1)
                    elif (
                        field == "scale_w_quant_orig.col"
                        and value.ndim == 2
                        and value.shape[0] == 3
                    ):
                        value = value.flatten()
                yield name, value

        return super().apply(converted())
