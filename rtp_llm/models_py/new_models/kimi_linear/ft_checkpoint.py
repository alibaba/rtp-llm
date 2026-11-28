"""Kimi Linear's KDA and MLA layers share a checkpoint but not a layout."""

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.new_models.hybrid_ft import convolution, identity, matrix
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _attention(checkpoint, layer, src, dst, swizzled):
    if not layer.is_linear:
        from rtp_llm.models_py.new_models.deepseek_v3.ft_checkpoint import (
            mla_ft_bindings,
        )

        return mla_ft_bindings(checkpoint, layer, src, dst, swizzled)
    kda = layer.self_attn
    if (kda.o_proj.tp_size, kda.o_proj.tp_rank) != (
        checkpoint.topology.tp_size,
        checkpoint.topology.tp_rank,
    ):
        raise ValueError("KDA FT projection topology does not match the producer")
    bindings = []
    src, dst = src + "linear_attn.", dst + "self_attn."
    for old, target in (
        ("in_proj_qkv", "qkv_proj"),
        ("b_proj", "b_proj"),
        ("f_a_proj", "f_a_proj"),
        ("f_b_proj", "f_b_proj"),
        ("g_a_proj", "g_a_proj"),
        ("g_b_proj", "g_b_proj"),
        ("out_proj", "o_proj"),
    ):
        module = getattr(kda, target)
        if not isinstance(module.quant_method, UnquantizedLinearMethod):
            raise ValueError(f"KDA FT has no declared quantized layout for {target}")
        bindings.append(
            RuntimeTensorBinding(
                (src + old + ".weight",),
                dst + target + ".weight",
                lambda v, decode=swizzled and old == "out_proj": matrix(v, decode),
            )
        )
    for old, target, fn in (
        ("conv1d.weight", "conv1d.weight", convolution),
        ("dt_bias_kda", "dt_bias", identity),
        ("A_log", "A_log", identity),
        ("norm.weight", "o_norm.weight", identity),
    ):
        bindings.append(RuntimeTensorBinding((src + old,), dst + target, fn))
    return bindings, {}


def load_kimi_linear_ft(model, checkpoint):
    def ffn(layer):
        if layer.block_sparse_moe is not None:
            return layer.block_sparse_moe, "block_sparse_moe", "shared_experts"
        return layer.mlp, "mlp", "shared_experts"

    load_gqa_ffn_ft_checkpoint(
        model, checkpoint, attention_bindings=_attention, layer_ffn=ffn
    )
