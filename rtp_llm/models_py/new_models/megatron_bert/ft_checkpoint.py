"""Megatron-BERT FT tensors use pre-attention and pre-FFN LayerNorm."""

from torch import nn

from rtp_llm.models_py.new_models.bert.ft_checkpoint import EncoderBindings


def load_megatron_ft_checkpoint(model, checkpoint):
    if (
        not {"global.final_layernorm.gamma", "global.final_layernorm.beta"}
        <= checkpoint.entries.keys()
    ):
        raise ValueError(
            "Older Megatron-BERT FT exports omitted encoder.ln (the learned final norm); "
            "its values cannot be recovered. Re-export from the original HF checkpoint."
        )
    b = EncoderBindings(checkpoint)
    b.full_table("global.embedding", "embeddings.word_embeddings.weight")
    b.full_table(
        "global.position_encoding.weight",
        "embeddings.position_embeddings.weight",
        max_rows=model.embeddings.position_embeddings.num_embeddings,
    )
    token_types = "global.token_type_embedding.weight"
    if (
        token_types in checkpoint.entries
        and model.embeddings.token_type_embeddings is None
    ):
        rows, width = checkpoint.read_rank_slice(
            token_types, checkpoint.topology.tp_rank
        )
        base = model.embeddings.word_embeddings.weight
        if rows <= 0 or width * checkpoint.topology.tp_size != base.shape[1]:
            raise ValueError("Invalid Megatron-BERT FT token type table")
        model.embeddings.token_type_embeddings = nn.Embedding(
            rows, base.shape[1], dtype=base.dtype
        )
        model.embeddings.token_type_embeddings.requires_grad_(False)
    if model.embeddings.token_type_embeddings is not None:
        b.full_table(token_types, "embeddings.token_type_embeddings.weight")
    b.norm("global.final_layernorm", "final_norm")
    for index, layer in enumerate(model.layers):
        src = dst = f"layers.{index}."
        b.norm(src + "pre_layernorm_weights", dst + "attn_norm")
        b.norm(src + "post_layernorm_weights", dst + "ffn_norm")
        b.separate_qkv(
            src + "self_attention_weights.query_weight",
            [dst + n for n in ("query", "key", "value")],
        )
        b.linear(src + "self_attention_weights.attention_output_weight", dst + "out")
        b.linear(src + "ffn_weights.intermediate_weight3", dst + "intermediate")
        b.linear(src + "ffn_weights.intermediate_weight2", dst + "output")
    b.custom_parameters(
        model,
        (
            (slot, role, model._custom_dtype[slot])
            for slot, role in model._custom.items()
        ),
    )
    checkpoint.load_parameters(model, tuple(b))
