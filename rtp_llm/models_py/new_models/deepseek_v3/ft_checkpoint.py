"""Rank-local MLA FT import. The producer exports fused A, B and absorb tensors."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint
from rtp_llm.models_py.quant_methods.block_fp8_ft import (
    block_ft_bindings,
    block_ft_scale,
    block_ft_source_is_rocm,
    block_ft_weight,
)
from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight
from rtp_llm.models_py.quant_methods.fp8 import Fp8BlockLinearMethod
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.utils.model_weight import W


def _identity(values):
    return values[0]


def _dense(values, swizzled=False):
    if values[0].ndim != 2:
        raise ValueError("Expected a rank-local FT matrix")
    value = values[0].T
    return decode_dense_weight(value) if swizzled else value


def _projections(checkpoint, source, owners):
    """Map one rank's contiguous serialized matrix to one or more row owners."""
    shape = (
        sum(layer.weight.shape[0] for _, layer in owners),
        owners[0][1].weight.shape[1],
    )
    block = all(
        isinstance(layer.quant_method, Fp8BlockLinearMethod) for _, layer in owners
    )
    if not block and not all(
        isinstance(layer.quant_method, UnquantizedLinearMethod) for _, layer in owners
    ):
        raise ValueError(f"No MLA FT decoder for quantization of {source}")

    def matrix(values):
        if block:
            return block_ft_weight(values[0], shape, packed)
        result = _dense(values)
        if result.shape != shape:
            raise ValueError(
                f"Invalid fused FT matrix {source}: {result.shape} != {shape}"
            )
        return result

    if block:
        scale_source = source.removesuffix(".kernel") + ".weight_only_quant_scale"
        rank = checkpoint.topology.tp_rank
        sample = (slice(0, 1), slice(0, 1))
        dtype = checkpoint.read_rank_slice(source, rank, sample).dtype
        packed = (
            checkpoint.read_rank_slice(scale_source, rank, sample).dtype == torch.int32
        )
        block_ft_source_is_rocm(
            checkpoint.index_metadata.get("ft_export_backend"), dtype, packed
        )
    bindings = []
    start = 0
    for target, layer in owners:
        rows, cols = layer.weight.shape
        if cols != shape[1]:
            raise ValueError(f"Inconsistent input widths for {source}")
        bindings.append(
            RuntimeTensorBinding(
                (source,),
                target + ".weight",
                lambda values, start=start, rows=rows: matrix(values).narrow(
                    0, start, rows
                ),
            )
        )
        if block:
            if start % 128:
                raise ValueError(f"Unaligned block-FP8 plane boundary in {source}")

            def scales(values, start=start, rows=rows):
                return block_ft_scale(values[0], shape, source_dtype=dtype).narrow(
                    0, start // 128, (rows + 127) // 128
                )

            def install(values, layer=layer, scales=scales):
                layer.weight_scale_inv.copy_(scales(values))
                layer._packed_block_fp8_ft = {"weight_scale_inv"} if packed else set()

            bindings.append(
                RuntimeTensorBinding(
                    (scale_source,),
                    target + ".weight_scale_inv",
                    scales,
                    install=install,
                )
            )
        start += rows
    return bindings


def mla_ft_bindings(checkpoint, layer, src, dst, swizzled):
    att = layer.self_attn
    target = dst + "self_attn."
    result = []

    def direct(name, path):
        result.append(RuntimeTensorBinding((src + name,), target + path, _identity))

    def linear(name, *paths):
        result.extend(
            _projections(
                checkpoint,
                src + name,
                [(target + path, att.get_submodule(path)) for path in paths],
            )
        )

    linear(
        W.mla_fusedqkrope_w if att.q_lora_rank else W.mla_fusedqkrope_no_lora_w,
        "q_a_proj",
        "kv_a_proj_with_mqa",
    )
    direct(W.mla_kv_a_ln_gamma, "kv_a_layernorm.weight")
    if att.q_lora_rank:
        linear(W.mla_q_b_w, "q_b_proj")
        direct(W.mla_q_a_ln_gamma, "q_a_layernorm.weight")
    linear(W.mla_kv_b_w, "kv_b_proj")
    if isinstance(att.o_proj.quant_method, Fp8BlockLinearMethod):
        result.extend(
            block_ft_bindings(
                att.o_proj,
                src + W.attn_o_w,
                src + W.attn_o_s,
                target + "o_proj",
                checkpoint,
                swizzled=swizzled,
            )
        )
    elif isinstance(att.o_proj.quant_method, UnquantizedLinearMethod):
        result.append(
            RuntimeTensorBinding(
                (src + W.attn_o_w,),
                target + "o_proj.weight",
                lambda values: _dense(values, swizzled),
            )
        )
    else:
        raise ValueError("Unsupported MLA output projection FT quantization")

    if getattr(att, "indexer", None) is not None:
        linear(W.mla_indexer_qb_w, "indexer.wq_b")
        linear(W.mla_indexer_k_w, "indexer.wk")
        linear(W.mla_indexer_weights_proj_w, "indexer.weights_proj")
        direct(W.mla_indexer_k_norm_w, "indexer.k_norm.weight")
        direct(W.mla_indexer_k_norm_b, "indexer.k_norm.bias")

    # For quantized files these are independent pre-quantization results:
    # reproducing them from the exported prefill KV-B would change decode.
    for name, attr, shape in (
        (W.mla_kc, "_kc_w", (att.num_heads, att.nope_head_dim, att.kv_lora_rank)),
        (W.mla_vc, "_vc_w", (att.num_heads, att.kv_lora_rank, att.v_head_dim)),
    ):
        att.register_buffer(
            attr, att.kv_a_layernorm.weight.new_empty(shape), persistent=True
        )
        direct(name, attr)
    att._ft_absorb_weights = True
    return result, {}


def load_mla_ft_checkpoint(model, checkpoint):
    tag = model.config.model_type
    if tag not in {
        "deepseek2",
        "deepseek3",
        "deepseek_v31",
        "deepseek_v32",
        "glm_5",
        "kimi_k2",
        "kimi_k25",
        "glm4_moe_lite",
        "deepseek-v3-mtp",
        "glm_5_mtp",
    }:
        raise ValueError(f"No MLA FT model contract for {tag!r}")
    options = {}
    if tag in {"deepseek-v3-mtp", "glm_5_mtp"}:
        prefix = "global." if tag == "glm_5_mtp" else "layers.0."
        swizzled = any(
            k.lower().replace("_", "") == "useswizzlea"
            and str(v).lower() in ("1", "true")
            for k, v in checkpoint.env_params.items()
        )
        options["final_norm_keys"] = (
            prefix + W.multi_tokens_predict_final_ln_gamma,
            prefix + W.multi_tokens_predict_final_ln_beta,
        )
        options["extra_bindings"] = [
            RuntimeTensorBinding((prefix + name,), "mtp_block." + dest, convert)
            for name, dest, convert in (
                (W.multi_tokens_predict_enorm, "e_norm.weight", _identity),
                (W.multi_tokens_predict_hnorm, "h_norm.weight", _identity),
                (
                    W.multi_tokens_predict_eh_proj,
                    "fc.weight",
                    lambda values: _dense(values, swizzled),
                ),
            )
        ]
    return load_gqa_ffn_ft_checkpoint(
        model, checkpoint, attention_bindings=mla_ft_bindings, **options
    )
