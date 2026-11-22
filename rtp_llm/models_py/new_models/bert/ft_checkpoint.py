"""Boundary conversions for the BERT family's rank-local FT exports."""

import torch
from torch import nn

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding


def identity(values):
    return values[0]


def matrix(values):
    value = values[0]
    if value.ndim != 2:
        raise ValueError("Expected a two-dimensional encoder FT matrix")
    return value.T


def vector(values):
    value = values[0]
    if value.ndim == 2 and value.shape[0] == 1:
        return value[0]
    if value.ndim != 1:
        raise ValueError("Expected an encoder FT vector or [1, width] QKV bias")
    return value


class EncoderBindings(list):
    """Short-lived, model-local bindings; never used by the runtime forward."""

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        for key, value in checkpoint.env_params.items():
            if key.lower().replace("_", "") in ("useswizzlea", "armgemmusekai"):
                if str(value).lower() not in ("0", "false", "", "none"):
                    raise ValueError(
                        f"Unsupported packed encoder FT layout: {key}={value}"
                    )

    def field(self, source, target, convert=identity):
        self.append(RuntimeTensorBinding((source,), target, convert))

    def norm(self, source, target, *, bias_suffix="beta"):
        self.field(source + ".gamma", target + ".weight")
        self.field(source + "." + bias_suffix, target + ".bias")

    def linear(self, source, target):
        self.field(source + ".kernel", target + ".weight", matrix)
        self.field(source + ".bias", target + ".bias", vector)

    def separate_qkv(self, source, targets):
        for suffix, field, convert in (
            (".kernel", ".weight", matrix),
            (".bias", ".bias", vector),
        ):
            for i, target in enumerate(targets):

                def part(values, convert=convert, i=i):
                    tensor = convert(values)
                    if tensor.shape[0] % 3:
                        raise ValueError(
                            "Encoder FT QKV must contain three equal partitions"
                        )
                    return tensor.chunk(3, dim=0)[i]

                self.field(source + suffix, target + field, part)

    def full_table(self, source, target, *, max_rows=None):
        # FT partitions tables on hidden width. Jina and Megatron use replicated
        # tables; Megatron's active position table may be a prefix of the source.
        c = self.checkpoint

        def read():
            shape = c.read_rank_slice(source, c.topology.tp_rank)
            if len(shape) != 2 or (max_rows is not None and shape[0] < max_rows):
                raise ValueError(
                    f"Invalid encoder FT embedding shape {shape} for {source}"
                )
            if any(
                c.read_rank_slice(source, r) != shape for r in range(c.topology.tp_size)
            ):
                raise ValueError(
                    f"Inconsistent encoder FT embedding partitions for {source}"
                )
            parts = [
                c.read_rank_slice(source, r, (slice(max_rows), slice(None)))
                for r in range(c.topology.tp_size)
            ]
            return (torch.cat(parts, dim=1),)

        self.append(RuntimeTensorBinding((source,), target, identity, read))

    def custom_parameters(self, model, slots):
        # Jina/Megatron allocate arbitrary handler heads from checkpoint shape.
        for slot, role, dtype in slots:
            source = "global." + role
            value = self.checkpoint.read(source)
            if not value.is_floating_point() or value.dtype != dtype:
                raise ValueError(f"Unexpected encoder FT handler dtype for {source}")
            model.custom_params[slot] = nn.Parameter(
                torch.empty_like(value), requires_grad=False
            )
            self.field(source, "custom_params." + slot)


def load_bert_ft_checkpoint(model, checkpoint):
    from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod

    from .static_ft import bind_static_projection
    from .static_int8 import BertStaticInt8Linear

    bindings = EncoderBindings(checkpoint)
    static = isinstance(
        next(iter(model.encoder.layer.values())).qkv.quant_method, BertStaticInt8Linear
    )
    if static and not (model.first and model.last):
        raise ValueError(
            "Static INT8 BERT FT needs the preceding layer's activation scale; PP is not supported"
        )
    if model.first:
        for source, target in (
            ("embedding", "word_embeddings"),
            ("position_encoding.weight", "position_embeddings"),
            ("token_type_embedding.weight", "token_type_embeddings"),
        ):
            bindings.field("global." + source, "embeddings." + target + ".weight")
        if static:

            def norm_weight(values):
                value = values[0]
                if value.ndim != 2 or value.shape[1] != 1:
                    raise ValueError("Expected a static INT8 BERT FT norm column")
                return value[:, 0]

            bindings.field(
                "global.pre_decoder_layernorm.gamma",
                "embeddings.LayerNorm.weight",
                norm_weight,
            )
            bindings.field(
                "global.pre_decoder_layernorm.bias", "embeddings.LayerNorm.bias"
            )
        else:
            bindings.norm(
                "global.pre_decoder_layernorm", "embeddings.LayerNorm", bias_suffix="bias"
            )
    else:
        # select_stage handles embedding/position; the old writer replicated
        # these additional globals, but only stage zero has BERT embeddings.
        for name in (
            "token_type_embedding.weight",
            "pre_decoder_layernorm.gamma",
            "pre_decoder_layernorm.bias",
        ):
            checkpoint.entries.pop("global." + name, None)
    if not model.last:
        for role in model._ft_handler_roles:
            checkpoint.entries.pop("global." + role, None)
    for index, layer in model.encoder.layer.items():
        src, dst = f"layers.{index}.", f"encoder.layer.{index}."
        if not static and not isinstance(
            layer.qkv.quant_method, UnquantizedLinearMethod
        ):
            raise ValueError("Unsupported BERT FT projection quantization")
        for name, target in (
            ("self_attention_weights.query_weight", "qkv"),
            ("self_attention_weights.attention_output_weight", "attn_output"),
            ("ffn_weights.intermediate_weight3", "intermediate"),
            ("ffn_weights.intermediate_weight2", "output"),
        ):
            if static:
                activation = {
                    "qkv": (
                        "global.pre_decoder_layernorm"
                        if int(index) == 0
                        else f"layers.{int(index) - 1}.post_ffn_layernorm_weights"
                    ),
                    "attn_output": src + name,
                    "intermediate": src + "post_layernorm_weights",
                    "output": src + name,
                }[target]
                bind_static_projection(
                    bindings,
                    getattr(layer, target),
                    src + name,
                    dst + target,
                    activation,
                )
            else:
                bindings.linear(src + name, dst + target)
        bindings.norm(src + "post_layernorm_weights", dst + "attn_norm")
        bindings.norm(src + "post_ffn_layernorm_weights", dst + "output_norm")
    bindings.extend(model.handler_heads.ft_bindings(checkpoint))
    checkpoint.load_parameters(model, tuple(bindings))
