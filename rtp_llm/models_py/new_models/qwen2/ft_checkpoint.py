"""Qwen2/Llama's boundary conversion for legacy dense FT runtime snapshots.

Historical names exist only at this import boundary. No runtime weight-name
registry or OldLoader/device conversion dependency is introduced here.
"""

import torch

from rtp_llm.models_py.ft_checkpoint import (
    RuntimeTensorBinding,
    ft_layer_bindings,
    ft_lm_head_source,
    validate_format_parameter_ownership,
)
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _identity(values):
    return values[0]


def _transpose(values):
    if values[0].ndim != 2:
        raise ValueError("Dense FT kernel must be a two-dimensional matrix")
    return values[0].T


def _bias(values):
    value = values[0]
    if value.ndim != 1 and not (value.ndim == 2 and value.shape[0] == 1):
        raise ValueError("Dense FT QKV bias must have shape [N] or [1, N]")
    return value.reshape(-1)


def _validate_cache_reciprocal(value):
    # Non-MLA, non-per-tensor-FP8 exporters synthesize exactly this FP32 [1]
    # unit for FP8 KV cache. It is not a learned activation-quantization scale.
    if value.shape != (1,) or value.dtype != torch.float32 or value.item() != 1.0:
        raise ValueError(
            "Invalid FT FP8 KV-cache dummy reciprocal: expected FP32 [1.0]"
        )


def load_dense_ft_checkpoint(model, checkpoint):
    if type(model).__name__ not in (
        "Qwen1ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2EmbeddingBackbone",
        "LlamaForCausalLM",
    ):
        raise ValueError("No dense FT adapter for this model")
    return load_gqa_ffn_ft_checkpoint(model, checkpoint)


def load_gqa_ffn_ft_checkpoint(
    model,
    checkpoint,
    *,
    extra_bindings=(),
    attention_bindings=None,
    final_norm_keys=("global.final_layernorm.gamma", "global.final_layernorm.beta"),
    layer_ffn=None,
    source_validators=None,
):
    from rtp_llm.models_py.quant_methods.fp8 import (
        Fp8BlockLinearMethod,
        Fp8LinearMethod,
        Fp8PerChannelLinearMethod,
    )
    from rtp_llm.models_py.quant_methods.per_tensor_int8 import (
        PerTensorInt8LinearMethod,
    )
    from rtp_llm.models_py.quant_methods.per_tensor_int8_ft import static_int8_bindings

    fp8_methods = (Fp8LinearMethod, Fp8PerChannelLinearMethod, Fp8BlockLinearMethod)

    swizzle_flags = set()
    for key, value in checkpoint.env_params.items():
        normalized = key.lower().replace("_", "")
        if normalized == "useswizzlea":
            if str(value).lower() not in ("0", "1", "false", "true", ""):
                raise ValueError("FT use_swizzleA metadata must be a boolean")
            swizzle_flags.add(str(value).lower() in ("1", "true"))
        if normalized == "armgemmusekai" and str(value).lower() not in (
            "0",
            "false",
            "none",
            "",
        ):
            raise ValueError("Historical ARM Kai FT layout has no current-main decoder")
    if len(swizzle_flags) > 1:
        raise ValueError("Conflicting FT use_swizzleA metadata")
    swizzled = True in swizzle_flags

    def kernel(values):
        value = _transpose(values)
        if swizzled:
            from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight

            value = decode_dense_weight(value)
        return value

    layers = ft_layer_bindings(model)
    owned_attention = {}
    if layer_ffn is None:
        layer_ffn = lambda layer: (layer.mlp, "mlp", "shared_experts")
    for layer_id, key, layer in layers:
        owned_attention[key] = (
            attention_bindings(
                checkpoint, layer, f"layers.{layer_id}.", f"layers.{key}.", swizzled
            )
            if attention_bindings is not None
            else None
        )
        mlp, _, shared_name = layer_ffn(layer)
        ffn = getattr(mlp, shared_name, None) if hasattr(mlp, "experts") else mlp
        for linear in (
            *(
                (layer.self_attn.qkv_proj, layer.self_attn.o_proj)
                if owned_attention[key] is None
                else ()
            ),
            *((ffn.gate_up_proj, ffn.down_proj) if ffn is not None else ()),
        ):
            if not isinstance(
                linear.quant_method,
                (UnquantizedLinearMethod, PerTensorInt8LinearMethod, *fp8_methods),
            ):
                raise ValueError(
                    "Dense FT has no declared adapter for this quantized runtime layout"
                )
    if getattr(model.lm_head, "output_vocab_ids", None):
        raise ValueError(
            "Legacy FT checkpoints cannot be combined with vocabulary pruning"
        )

    tp = checkpoint.topology.tp_size
    rank = checkpoint.topology.tp_rank

    def fp8_bindings(
        linear, source, destination, projection, layer_prefix, ffn_width=None
    ):
        from rtp_llm.models_py.quant_methods.fp8_ft import (
            PerChannelFP8FTProjection,
            PerTensorFP8FTProjection,
        )

        if isinstance(linear.quant_method, Fp8BlockLinearMethod):
            from rtp_llm.models_py.quant_methods.block_fp8_ft import block_ft_bindings

            keys = (source,) if isinstance(source, str) else source
            scales = tuple(
                k.removesuffix(".kernel") + ".weight_only_quant_scale" for k in keys
            )
            return block_ft_bindings(
                linear,
                keys,
                scales,
                destination,
                checkpoint,
                swizzled=swizzled,
                projection=projection,
                ffn_width=ffn_width,
            )
        scale = source.removesuffix(".kernel") + ".weight_only_quant_scale"
        if isinstance(linear.quant_method, Fp8PerChannelLinearMethod):
            state = PerChannelFP8FTProjection(
                linear, checkpoint, source, scale, projection, swizzled=swizzled
            )
            return [
                RuntimeTensorBinding(
                    (source, scale),
                    destination + ".weight",
                    _identity,
                    lambda: (state.weight(),),
                ),
                RuntimeTensorBinding(
                    (source, scale),
                    destination + ".weight_scale",
                    _identity,
                    lambda: (state.scale(),),
                ),
            ]
        state = PerTensorFP8FTProjection(
            linear, checkpoint, source, scale, projection, swizzled=swizzled
        )
        # These spellings are only legacy serializer boundary names. The quant
        # module owns all tensor layout, fusion and activation-scale semantics.
        stems = {
            "qkv": "pre_layernorm_weights",
            "out": "self_attention_weights.attention_output_weight",
            "gate_up": "post_layernorm_weights",
            "down": "ffn_weights.intermediate_weight2",
        }
        stem = layer_prefix + stems[projection]
        direct, reciprocal = stem + ".static_quant", stem + ".static_quant_reciprocal"
        dynamic = linear.quant_method.quant_config.activation_dynamic
        if dynamic and (
            direct in checkpoint.entries or reciprocal in checkpoint.entries
        ):
            raise ValueError(
                "FP8 FT static source disagrees with dynamic activation configuration"
            )
        if not dynamic and not {direct, reciprocal}.issubset(checkpoint.entries):
            raise ValueError("FP8 FT static activation metadata is incomplete")
        actual, companion = (
            (direct, reciprocal) if projection == "out" else (reciprocal, direct)
        )
        auxiliary = () if dynamic else (actual, companion)
        return [
            RuntimeTensorBinding(
                (source, scale),
                destination + ".weight",
                _identity,
                lambda: (state.weight(),),
            ),
            RuntimeTensorBinding(
                (scale,),
                destination + ".weight_scale",
                _identity,
                lambda: (state.scale(),),
            ),
            RuntimeTensorBinding(
                (scale, *auxiliary),
                destination + ".input_scale",
                _identity,
                lambda: (
                    state.input_scale(
                        None if dynamic else actual, companion, projection != "down"
                    ),
                ),
            ),
        ]

    embedding = model.embed_tokens
    head = model.lm_head
    preserved = validate_format_parameter_ownership(model, ft=True)
    head_source = ft_lm_head_source(checkpoint)
    if head is None and "global.lm_head" in checkpoint.entries:
        # A generation FT archive is also a valid source for a non-generation
        # deployment of this backbone. Only exclude the unused *projection*;
        # the last stage still needs its final norm, and undeclared tensors
        # must continue to fail the strict FT coverage check.
        from rtp_llm.ops import TaskType

        cfg = model.config
        task = (
            cfg.get("task_type")
            if isinstance(cfg, dict)
            else getattr(cfg, "task_type", None)
        )
        if task is not None and task != TaskType.LANGUAGE_MODEL:
            del checkpoint.entries["global.lm_head"]
    if any(
        (leaf.tp_size, leaf.tp_rank) != (tp, rank)
        for leaf in (embedding, head)
        if leaf is not None
    ):
        raise ValueError("FT import requires matching embedding/head TP topology")

    def embedding_rows(_):
        # OldLoader shards embedding on hidden, NewLoader on vocabulary. Read
        # only this rank's destination vocabulary rows from each peer shard.
        rows, hidden = embedding.vocab_size, embedding.embedding_dim
        chunks = []
        for peer in range(tp):
            if checkpoint.read_rank_slice("global.embedding", peer) != (
                rows,
                hidden // tp,
            ):
                raise ValueError("FT embedding hidden-partition layout mismatch")
            chunks.append(
                checkpoint.read_rank_slice(
                    "global.embedding",
                    peer,
                    (
                        slice(embedding.vocab_start_idx, embedding.vocab_end_idx),
                        slice(None),
                    ),
                )
            )
        return embedding.prepare_local_checkpoint_weight(torch.cat(chunks, dim=1))

    def head_rows(_):
        # sp_0_pad8 partitions padded vocabulary, which changes rank boundaries.
        # A tied source may have more rows than the configured output vocabulary.
        # Read the source's actual shard stride, not the destination's padding.
        first_shape = checkpoint.read_rank_slice("global.lm_head", 0)
        if len(first_shape) != 2 or first_shape[1] != head.hidden_size:
            raise ValueError("FT lm_head padded-vocabulary layout mismatch")
        old_rows, hidden = first_shape
        if old_rows * tp < head.checkpoint_vocab_size:
            raise ValueError(
                "FT lm_head does not cover the configured output vocabulary"
            )
        begin = min(rank * head.vocab_size_per_partition, head.checkpoint_vocab_size)
        end = begin + len(head.local_checkpoint_rows)
        chunks = []
        for peer in range(tp):
            if checkpoint.read_rank_slice("global.lm_head", peer) != (old_rows, hidden):
                raise ValueError("FT lm_head padded-vocabulary layout mismatch")
            left, right = max(begin, peer * old_rows), min(end, (peer + 1) * old_rows)
            if left < right:
                chunks.append(
                    checkpoint.read_rank_slice(
                        "global.lm_head",
                        peer,
                        (
                            slice(left - peer * old_rows, right - peer * old_rows),
                            slice(None),
                        ),
                    )
                )
        selected = (
            torch.cat(chunks, dim=0)
            if chunks
            else head.weight.detach().new_empty((0, hidden))
        )
        return head.prepare_local_checkpoint_weight(selected)

    def ffn_local(names, projection, local_width, hidden):
        # Main pads intermediate_size to TP*64 when swizzleA is enabled.
        # Consequently the original FT rank boundaries differ from the model's
        # unpadded local parameter boundaries, just as with padded vocabulary.
        width = ((local_width + 63) // 64) * 64 if swizzled else local_width
        global_width = local_width * tp
        begin, end = rank * local_width, (rank + 1) * local_width
        parts = [[], []] if projection == "gate_up" else [[]]
        for peer in range(tp):
            left, right = max(begin, peer * width), min(end, (peer + 1) * width)
            padding = max(0, global_width - peer * width)
            needs_padding_check = width != local_width and padding < width
            expected = (
                (width, hidden)
                if projection == "down"
                else (hidden, width * (2 if len(names) == 1 else 1))
            )
            for name in names:
                if checkpoint.read_rank_slice(name, peer) != expected:
                    raise ValueError("FT FFN padded-partition layout mismatch")
            if left >= right and not needs_padding_check:
                continue
            raw = tuple(
                checkpoint.read_rank_slice(name, peer, (slice(None), slice(None)))
                for name in names
            )
            if projection == "down":
                canonical = (kernel(raw),)
                axis = 1
            elif len(names) == 1:
                canonical = kernel(raw).split(width, dim=0)
                axis = 0
            else:
                # ROCm's rewrite list includes separate w3, not w1.
                canonical = (_transpose(raw[:1]), kernel(raw[1:]))
                axis = 0
            for output, value in zip(parts, canonical):
                if (
                    needs_padding_check
                    and torch.count_nonzero(
                        value.narrow(axis, padding, width - padding)
                    ).item()
                ):
                    raise ValueError("FT FFN padding contains nonzero values")
                if left < right:
                    output.append(value.narrow(axis, left - peer * width, right - left))
        merged = [
            torch.cat(values, dim=1 if projection == "down" else 0) for values in parts
        ]
        return (merged[0] if projection == "down" else torch.cat(merged, dim=0),)

    bindings = list(extra_bindings)
    if embedding is not None and "embed_tokens.weight" not in preserved:
        bindings.append(
            RuntimeTensorBinding(
                ("global.embedding",), "embed_tokens.weight", embedding_rows
            )
        )
    if head is not None and "lm_head.weight" not in preserved:
        bindings.append(
            RuntimeTensorBinding(("global.lm_head",), "lm_head.weight", head_rows)
        )
    if model.norm is not None:
        bindings.append(
            RuntimeTensorBinding((final_norm_keys[0],), "norm.weight", _identity)
        )
    validators = dict(source_validators or {})
    if final_norm_keys[1] in checkpoint.entries:

        def zero_final_beta(value):
            if (
                value.shape != model.norm.weight.shape
                or value.dtype != model.norm.weight.dtype
                or torch.count_nonzero(value).item()
            ):
                raise ValueError("Dense RMSNorm FT final beta must be exactly zero")

        validators[final_norm_keys[1]] = zero_final_beta
    for layer_id, key, layer in layers:
        src = f"layers.{layer_id}."
        dst = f"layers.{key}."
        cache_unit = (
            src
            + "self_attention_weights.attention_output_weight.static_quant_reciprocal"
        )
        if (
            owned_attention[key] is None
            and cache_unit in checkpoint.entries
            and not isinstance(
                layer.self_attn.o_proj.quant_method,
                (Fp8LinearMethod, PerTensorInt8LinearMethod),
            )
        ):
            validators[cache_unit] = _validate_cache_reciprocal
        fields = [
            ("pre_layernorm_weights.gamma", "input_layernorm.weight", _identity),
            (
                "post_layernorm_weights.gamma",
                "post_attention_layernorm.weight",
                _identity,
            ),
        ]
        if owned_attention[key] is not None:
            attn_fields, attn_validators = owned_attention[key]
            bindings.extend(attn_fields)
            validators.update(attn_validators)
        else:
            fields.extend(
                (
                    (
                        "self_attention_weights.query_weight.kernel",
                        "self_attn.qkv_proj.weight",
                        kernel,
                    ),
                    (
                        "self_attention_weights.attention_output_weight.kernel",
                        "self_attn.o_proj.weight",
                        kernel,
                    ),
                )
            )
        for old, target, convert in fields:
            if target in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
                linear = (
                    layer.self_attn.qkv_proj
                    if "qkv_proj" in target
                    else layer.self_attn.o_proj
                )
                if isinstance(linear.quant_method, PerTensorInt8LinearMethod):
                    bindings.extend(
                        static_int8_bindings(
                            linear,
                            src + old,
                            dst + target.removesuffix(".weight"),
                            "qkv" if "qkv_proj" in target else "out",
                            src,
                            swizzled,
                        )
                    )
                    continue
                if isinstance(linear.quant_method, fp8_methods):
                    bindings.extend(
                        fp8_bindings(
                            linear,
                            src + old,
                            dst + target.removesuffix(".weight"),
                            "qkv" if "qkv_proj" in target else "out",
                            src,
                        )
                    )
                    continue
            bindings.append(RuntimeTensorBinding((src + old,), dst + target, convert))
        if owned_attention[key] is None and layer.self_attn.qkv_proj.bias is not None:
            bindings.append(
                RuntimeTensorBinding(
                    (src + "self_attention_weights.query_weight.bias",),
                    dst + "self_attn.qkv_proj.bias",
                    lambda values, shape=layer.self_attn.qkv_proj.bias.shape: _bias(
                        values
                    ).reshape(shape),
                )
            )
        for qk in ("q", "k") if owned_attention[key] is None else ():
            if getattr(layer.self_attn, qk + "_norm", None) is not None:
                bindings.append(
                    RuntimeTensorBinding(
                        (src + "self_attention_weights." + qk + "_layernorm.gamma",),
                        dst + "self_attn." + qk + "_norm.weight",
                        _identity,
                    )
                )
        mlp, mlp_path, shared_name = layer_ffn(layer)
        ffn, ffn_target = mlp, mlp_path + "."
        if hasattr(mlp, "experts"):
            from rtp_llm.models_py.quant_methods.block_fp8_ft import (
                block_moe_ft_bindings,
            )
            from rtp_llm.models_py.quant_methods.channel_tensor_moe_ft import (
                nonblock_moe_ft_bindings,
            )

            expert_binding = (
                block_moe_ft_bindings
                if mlp.experts._quant_family == "fp8_per_block"
                else nonblock_moe_ft_bindings
            )
            bindings.extend(
                expert_binding(
                    mlp.experts, src, dst + mlp_path + ".experts", checkpoint, layer_id
                )
            )
            bindings.append(
                RuntimeTensorBinding(
                    (src + "partial_moe_weights.gate.kernel",),
                    dst + mlp_path + ".gate.weight",
                    kernel,
                )
            )
            if getattr(mlp.gate, "e_score_correction_bias", None) is not None:
                bindings.append(
                    RuntimeTensorBinding(
                        (src + "partial_moe_weights.e_score_correction_bias",),
                        dst + mlp_path + ".gate.e_score_correction_bias",
                        _identity,
                    )
                )
            ffn, ffn_target = (
                getattr(mlp, shared_name, None),
                mlp_path + "." + shared_name + ".",
            )
        if ffn is None:
            continue
        down_names = (src + "ffn_weights.intermediate_weight2.kernel",)
        gate_names = (
            (src + "ffn_weights.intermediate_weight13.kernel",)
            if src + "ffn_weights.intermediate_weight13.kernel" in checkpoint.entries
            else (
                src + "ffn_weights.intermediate_weight.kernel",
                src + "ffn_weights.intermediate_weight3.kernel",
            )
        )
        local_width = ffn.down_proj.weight.shape[1]
        hidden = ffn.down_proj.weight.shape[0]
        ffn_width = None
        if isinstance(ffn.down_proj.quant_method, Fp8BlockLinearMethod):
            from rtp_llm.models_py.quant_methods.block_fp8_ft import block_ft_ffn_width

            ffn_width = block_ft_ffn_width(
                checkpoint, down_names[0], local_width, hidden
            )
            if ffn_width != local_width and not isinstance(
                ffn.gate_up_proj.quant_method, Fp8BlockLinearMethod
            ):
                raise ValueError(
                    "Padded FT FFN requires matching gate/up and down layouts"
                )
        for names, projection, target in (
            (down_names, "down", ffn_target + "down_proj.weight"),
            (gate_names, "gate_up", ffn_target + "gate_up_proj.weight"),
        ):
            linear = ffn.down_proj if projection == "down" else ffn.gate_up_proj
            if isinstance(linear.quant_method, PerTensorInt8LinearMethod):
                if len(names) != 1:
                    raise ValueError("Static INT8 FT requires a fused gate/up kernel")
                bindings.extend(
                    static_int8_bindings(
                        linear,
                        names[0],
                        dst + target.removesuffix(".weight"),
                        projection,
                        src,
                        swizzled,
                    )
                )
                continue
            if isinstance(linear.quant_method, fp8_methods):
                if len(names) != 1 and not isinstance(
                    linear.quant_method, Fp8BlockLinearMethod
                ):
                    raise ValueError(
                        "Per-tensor FP8 FT requires the producer's fused gate/up tensor"
                    )
                bindings.extend(
                    fp8_bindings(
                        linear,
                        names[0] if len(names) == 1 else names,
                        dst + target.removesuffix(".weight"),
                        projection,
                        src,
                        ffn_width=ffn_width,
                    )
                )
                continue
            bindings.append(
                RuntimeTensorBinding(
                    names,
                    dst + target,
                    _identity,
                    lambda names=names, projection=projection, width=local_width, hidden=hidden: (
                        ffn_local(names, projection, width, hidden)
                    ),
                )
            )
    handler = getattr(model, "handler_heads", None)
    if handler is not None:
        bindings.extend(handler.ft_bindings(checkpoint))
    checkpoint.load_parameters(model, tuple(bindings), validators)
    tied_config = getattr(model, "tie_word_embeddings", False)
    # Historic FT files always emit two tensors and have no origin marker.
    # Keep serving their exact bytes, but do not guess future update ownership.
    model._ft_unknown_tied_vocab = tied_config and head_source is None and (
        head is not None or getattr(model, "_pipeline_partitioned", False)
    )
    if head is not None and "lm_head.weight" not in preserved:
        mirrored = tied_config and head_source == "embedding"
        head._explicit_checkpoint_loaded = not mirrored
        if mirrored:
            head._tied_checkpoint_loaded = True
            config = model.config
            rows = config.get("input_vocab_size") if isinstance(config, dict) else getattr(config, "input_vocab_size", None)
            head._tied_checkpoint_source_rows = rows or head.checkpoint_vocab_size
