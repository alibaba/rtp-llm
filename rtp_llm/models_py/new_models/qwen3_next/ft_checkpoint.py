"""Read Qwen3-Next/3.5's already-reordered rank-local GDN, gates and MTP."""

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding, ft_layer_bindings
from rtp_llm.models_py.new_models.hybrid_ft import (
    convolution,
    identity,
    linear_bindings,
    matrix,
)
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint
from rtp_llm.utils.swizzle_utils import should_swizzle_linear_attn_ba


def _gdn(checkpoint, layer, src, dst, swizzled):
    gdn = getattr(layer, "linear_attn", None)
    if gdn is None:
        return None  # Ordinary GQA, handled by the shared converter.
    src, dst = src + "linear_attn.", dst + "linear_attn."
    partition = (gdn.attn_tp_size, gdn.attn_tp_rank)
    bindings = []
    for projection in ("in_proj_qkvz", "out_proj"):
        bindings.extend(
            linear_bindings(
                checkpoint,
                getattr(gdn, projection),
                src + projection + ".weight",
                dst + projection,
                src + projection + ".scale",
                swizzled,
                partition,
            )
        )

    def ba(values):
        raw = values[0]
        if swizzled and should_swizzle_linear_attn_ba(raw):
            return matrix(values, True).T
        return raw

    for old, new, fn in (
        ("in_proj_ba.weight", "in_proj_ba_w", ba),
        ("conv1d.weight", "conv1d_w", convolution),
        ("dt_bias", "dt_bias", identity),
        ("A_log", "a_log", identity),
        ("norm.weight", "norm_w", identity),
    ):
        bindings.append(RuntimeTensorBinding((src + old,), dst + new, fn))
    return bindings, {}


def load_qwen_next_ft(model, checkpoint):
    # Shared adapter also validates conflicting/invalid flag values before copy.
    swizzled = any(
        k.lower().replace("_", "") == "useswizzlea" and str(v).lower() in ("true", "1")
        for k, v in checkpoint.env_params.items()
    )
    extra = []
    for idx, key, layer in ft_layer_bindings(model):
        src, dst = f"layers.{idx}.", f"layers.{key}."
        if hasattr(layer, "self_attn"):
            gate = layer.self_attn.gate
            extra.extend(
                linear_bindings(
                    checkpoint,
                    gate,
                    src + "self_attention_weights.gate.weight",
                    dst + "self_attn.gate",
                    src + "self_attention_weights.gate.scale",
                    swizzled,
                    (gate.tp_size, gate.tp_rank),
                )
            )
        if getattr(layer.mlp, "shared_expert_gate", None) is not None:
            extra.append(
                RuntimeTensorBinding(
                    (src + "ffn_weights.shared_expert_gate.kernel",),
                    dst + "mlp.shared_expert_gate.weight",
                    matrix,
                )
            )
    if model._expects_mtp:
        for old, target, fn in (
            (
                "multi_tokens_predict_enorm.weight",
                "pre_fc_norm_embedding.weight",
                identity,
            ),
            (
                "multi_tokens_predict_hnorm.weight",
                "pre_fc_norm_hidden.weight",
                identity,
            ),
            (
                "multi_tokens_predict_eh_proj.weight",
                "fc.weight",
                lambda values: matrix(values, swizzled),
            ),
        ):
            extra.append(RuntimeTensorBinding(("global." + old,), target, fn))
    load_gqa_ffn_ft_checkpoint(
        model,
        checkpoint,
        extra_bindings=extra,
        attention_bindings=_gdn,
        layer_ffn=lambda layer: (layer.mlp, "mlp", "shared_expert"),
    )
