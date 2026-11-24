"""Import JinaBertWeightInfo exports, including its [up, gate] GLU."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.new_models.bert.ft_checkpoint import EncoderBindings, matrix


def load_jina_ft_checkpoint(model, checkpoint):
    b = EncoderBindings(checkpoint)
    b.full_table("global.embedding", "embeddings.word_embeddings.weight")
    b.full_table(
        "global.token_type_embedding.weight", "embeddings.token_type_embeddings.weight"
    )
    b.norm("global.pre_decoder_layernorm", "embeddings.LayerNorm", bias_suffix="bias")

    for index, layer in enumerate(model.encoder.layer):
        src, dst = f"layers.{index}.", f"encoder.layer.{index}."
        b.separate_qkv(
            src + "self_attention_weights.query_weight",
            [dst + "attention.self." + n for n in ("query", "key", "value")],
        )
        for qk in ("q", "k"):

            def local_norm(values, layer=layer):
                v = values[0]
                if v.shape != (layer.full_width,):
                    raise ValueError("Expected replicated Jina FT Q/K normalization")
                return v.chunk(layer.tp_size)[layer.tp_rank]

            for old, new in (("gamma", "weight"), ("beta", "bias")):
                b.field(
                    src + f"self_attention_weights.{qk}_layernorm.{old}",
                    dst + f"attention.self.layer_norm_{qk}.{new}",
                    local_norm,
                )
        b.linear(
            src + "self_attention_weights.attention_output_weight",
            dst + "attention.output.dense",
        )
        b.norm(src + "post_layernorm_weights", dst + "attention.output.LayerNorm")
        b.norm(src + "post_layernorm_weights_2", dst + "layer_norm_1")
        b.norm(src + "post_ffn_layernorm_weights", dst + "layer_norm_2")
        # Unlike the decoder MLP, Jina's original first half is up (w3),
        # followed by gate (w1). The exporter wrote them independently.
        b.append(
            RuntimeTensorBinding(
                (
                    src + "ffn_weights.intermediate_weight3.kernel",
                    src + "ffn_weights.intermediate_weight.kernel",
                ),
                dst + "mlp.up_gated_layer.weight",
                lambda values: torch.cat([matrix((v,)) for v in values]),
            )
        )
        b.linear(src + "ffn_weights.intermediate_weight2", dst + "mlp.down_layer")
    b.custom_parameters(
        model,
        (
            (slot, role, dtype or model.embeddings.LayerNorm.weight.dtype)
            for slot, role, dtype in model._custom.values()
        ),
    )

    def validate_slopes(value):
        expected = torch.tensor(model.encoder.layer[0].slopes, dtype=torch.float32)
        if value.dtype != expected.dtype or not torch.equal(value, expected):
            raise ValueError("Jina FT ALiBi slopes disagree with the configured heads")

    checkpoint.load_parameters(
        model, tuple(b), {"global.linear_bias_slopes": validate_slopes}
    )
