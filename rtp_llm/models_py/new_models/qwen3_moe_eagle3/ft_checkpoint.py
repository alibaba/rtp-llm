"""Rank-local FT records for the single-block full-vocabulary EAGLE3 draft."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding


def _record_prefix(body):
    # FFN and attention TP need not select the same subgroup. Two shards can
    # have identical shapes but belong to different ranks.
    attn = body.midlayer.self_attn.qkv_proj
    ffn = body.midlayer.mlp.gate_up_proj
    return f"eagle3.v1.a{attn.tp_size}r{attn.tp_rank}.f{ffn.tp_size}r{ffn.tp_rank}."


def load_eagle3_ft(model, checkpoint):
    """Restore only body parameters; never import into target-owned vocabulary."""
    prefix = "layers.0." + _record_prefix(model.body)
    if not any(k.startswith("layers.0.eagle3.v1.") for k in checkpoint.entries):
        return _load_descriptor_ft(model, checkpoint)
    if prefix + "norm.weight" not in checkpoint.entries:
        raise ValueError(
            "EAGLE3 FT archive lacks this rank's full-vocabulary draft body/final norm; "
            "re-export the original EAGLE3 checkpoint for this TP configuration"
        )
    bindings = tuple(
        RuntimeTensorBinding((prefix + name,), "body." + name, lambda values: values[0])
        for name, _ in model.body.named_parameters()
    )
    checkpoint.load_parameters(model, bindings)


def _load_descriptor_ft(model, checkpoint):
    """Older serializer spellings: accept only a self-contained compatible block.

    The stock Eagle3 descriptor exported the input and hidden norms but not
    the *learned output norm*. No input or target norm is equivalent to it.
    A complete locally produced archive need not have our versioned prefix.
    """
    layer = "layers.0."
    if "global.final_layernorm.gamma" not in checkpoint.entries:
        raise ValueError("EAGLE3 FT missing learned global.final_layernorm.gamma; "
                         "re-export the original EAGLE3 checkpoint")
    if any(layer + "self_attention_weights." + s + "_layernorm.gamma" in checkpoint.entries
           for s in ("q", "k", "qk")):
        raise ValueError("This EAGLE3 FT contains Q/K normalization, unlike the full-vocabulary draft")
    for key, value in checkpoint.env_params.items():
        if key.lower().replace("_", "") in ("useswizzlea", "armgemmusekai") and str(value).lower() not in (
            "", "0", "false", "none",
        ):
            raise ValueError("EAGLE3 descriptor FT requires canonical floating kernels; re-export the original EAGLE3")

    identity = lambda values: values[0]
    def matrix(values):
        if values[0].ndim != 2:
            raise ValueError("EAGLE3 FT kernel must be a matrix")
        return values[0].T

    # The source descriptor may store the same input norm under two names.
    # If both are present, neither can silently shadow the other.
    def input_norm(values):
        first = values[0]
        if any(t.dtype != first.dtype or not torch.equal(first, t) for t in values[1:]):
            raise ValueError("Conflicting EAGLE3 FT input normalization weights")
        return first

    bindings = [
        RuntimeTensorBinding(("global.final_layernorm.gamma",), "body.norm.weight", identity),
        RuntimeTensorBinding((layer + "eagle3_fc.weight",), "body.fc.weight", matrix),
    ]
    mid = "body.midlayer."
    norm_keys = tuple(layer + n for n in ("eagle3_input.gamma", "pre_layernorm_weights.gamma")
                      if layer + n in checkpoint.entries)
    bindings.append(RuntimeTensorBinding(norm_keys, mid + "input_layernorm.weight", input_norm))
    for source, target, op in (
        ("eagle3_fc.gamma", "hidden_norm.weight", identity),
        ("post_layernorm_weights.gamma", "post_attention_layernorm.weight", identity),
        ("self_attention_weights.query_weight.kernel", "self_attn.qkv_proj.weight", matrix),
        ("self_attention_weights.attention_output_weight.kernel", "self_attn.o_proj.weight", matrix),
        ("ffn_weights.intermediate_weight2.kernel", "mlp.down_proj.weight", matrix),
    ):
        bindings.append(RuntimeTensorBinding((layer + source,), mid + target, op))
    ffn = layer + "ffn_weights.intermediate_weight"
    if ffn + "13.kernel" in checkpoint.entries:
        bindings.append(RuntimeTensorBinding((ffn + "13.kernel",), mid + "mlp.gate_up_proj.weight", matrix))
    else:
        def merge(values):
            if values[0].shape != values[1].shape or values[0].dtype != values[1].dtype:
                raise ValueError("EAGLE3 FT gate and up projection layouts disagree")
            return torch.cat([matrix((v,)) for v in values], dim=0)
        bindings.append(RuntimeTensorBinding((ffn + ".kernel", ffn + "3.kernel"),
                                             mid + "mlp.gate_up_proj.weight", merge))
    validators = {}
    if "global.final_layernorm.beta" in checkpoint.entries:
        def zero_beta(t):
            norm = model.body.norm.weight
            if t.shape != norm.shape or t.dtype != norm.dtype or torch.count_nonzero(t).item():
                raise ValueError("EAGLE3 RMSNorm may only discard a correctly shaped zero beta")
        validators["global.final_layernorm.beta"] = zero_beta
    # FTCheckpoint checks the entire source/target set, dtype and *local*
    # shape before writing, including [2H, Q + K + V] and [3H, H] sources.
    checkpoint.load_parameters(model, tuple(bindings), validators)


class Eagle3FTExporter:
    """Run the same checkpoint-to-parameter transforms as the serving draft.

    The target is deliberately not constructed. Both vocabulary parameters are
    loans and are never serialized. Export is a CPU operation on one local body.
    """

    def __init__(self, config, parallel, database):
        if config.num_layers != 1 or getattr(config.quant_config, "quant_type", "none") != "none":
            raise ValueError("EAGLE3 FT export requires the one-block unquantized draft")
        if config.lora_infos or config.ptuning_path:
            raise ValueError("EAGLE3 FT export requires an original draft without LoRA or p-tuning")
        self.config = config
        self.parallel = parallel
        self.database = database

    @torch.no_grad()
    def dump_weight_as_ft_style(self, device, output_dir):
        from rtp_llm.models_py.model_loader import NewModelLoader
        from rtp_llm.models_py.new_models.qwen3_moe_eagle3.service import build_eagle3_body
        from rtp_llm.tools.convert.ft_export_writer import write_ft_shards

        pc = self.parallel
        dtype = self.config.compute_dtype
        with torch.device("cpu"):
            body = build_eagle3_body(
                self.config, dtype, pc.get_attn_tp_size(), pc.get_attn_tp_rank(),
                pc.get_ffn_tp_size(), pc.get_ffn_tp_rank(),
            )

        def source():
            # Published draft weights omit vocabulary. Ignore any copied
            # target matrices, exactly as serving does for an active loan.
            for name in self.database.get_pretrain_tensor_names():
                if name in ("model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight"):
                    continue
                tensors = self.database.load_tensor(name, data_type=None)
                if len(tensors) != 1:
                    raise ValueError(f"EAGLE3 FT requires exactly one checkpoint tensor for {name}")
                yield name, tensors[0]

        body.load_weights(source())
        NewModelLoader._validate_loaded_weights(body)
        prefix = _record_prefix(body)
        write_ft_shards(
            ((0, prefix + name, value.detach()) for name, value in body.named_parameters()),
            output_dir, tp_rank=pc.tp_rank, dp_rank=pc.dp_rank, ep_rank=pc.ep_rank,
        )
