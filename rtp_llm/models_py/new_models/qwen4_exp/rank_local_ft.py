"""Import the CUDA BF16 Qwen4 target archive at the loader boundary.

An FT shard already has local Q/K/V, expert placement and GDN row order.
Never send it through the HF readers or apply those transformations again.
"""

import json

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4CheckpointPlan
from rtp_llm.models_py.quant_methods.channel_tensor_moe_ft import (
    nonblock_moe_ft_bindings,
)


class RankPlan:
    positive = Qwen4CheckpointPlan.positive
    prefix = ""
    source_kind = "bf16"
    source_fp8 = dense_fp8 = vocab_fp8 = False
    dense_storage = vocab_storage = "source_bf16"

    def lowrank(self, relative):
        return self.lowranks[relative]


def rank_local_plan(config, load, ft):
    plan = RankPlan()
    plan.root = ft.root
    plan.model_config = config
    plan.source_config = json.loads((ft.root / "config.json").read_text())
    plan.text = plan.source_config.get("text_config", plan.source_config)
    if plan.source_config.get("quantization_config") or ft.index_metadata.get(
        "ft_export_backend"
    ) not in (None, "cuda"):
        raise ValueError("Qwen4 FT import requires a CUDA BF16 runtime snapshot")
    topology = ft.topology
    for size, rank in (
        (load.attn_tp_size, load.attn_tp_rank),
        (load.ffn_tp_size, load.ffn_tp_rank),
        (load.lm_head_tp_size, load.lm_head_tp_rank),
    ):
        if (size, rank) != (topology.tp_size, topology.tp_rank):
            raise ValueError(
                "Qwen4 FT projection topology differs from the exported TP group"
            )
    plan.ple_prefetch_mb = (
        config.get("qwen4_ple_prefetch_mb", 256)
        if isinstance(config, dict)
        else getattr(config, "qwen4_ple_prefetch_mb", 256)
    )
    plan.num_layers = plan.positive("num_hidden_layers")
    plan.hidden_size = plan.positive("hidden_size")
    plan.hc_mult = plan.positive("hc_count")
    plan.vocab_size = plan.positive("vocab_size")
    plan.layer_types, plan.shapes, plan.lowranks = [], {}, {}
    plan.ple_layers = tuple(n - 1 for n in plan.text.get("ple_layer_ids", ()))
    if len(set(plan.ple_layers)) != len(plan.ple_layers) or any(
        not 0 <= n < plan.num_layers for n in plan.ple_layers
    ):
        raise ValueError("Invalid Qwen4 FT PLE layer ids")

    def hc(source, dest):
        shape = ft._shape(*ft.entries[source + "_mix_down"])
        if (
            len(shape) != 2
            or shape[1] != plan.hidden_size * plan.hc_mult
            or not shape[0]
        ):
            raise ValueError(f"Invalid Qwen4 FT HC projection: {source}")
        plan.lowranks[dest] = shape[0]

    hc("global.qwen4.hc.mixer", "hyper_connection_mixer")
    for i in range(plan.num_layers):
        p = f"layers.{i}."
        for unit, dest in (
            ("attn", "attn_hyper_connection"),
            ("mlp", "mlp_hyper_connection"),
        ):
            hc(p + "qwen4.hc." + unit, p + dest)
        gdn = p + "linear_attn.in_proj_qkvz.weight" in ft.entries
        full = p + "self_attention_weights.query_weight.kernel" in ft.entries
        if full == gdn:
            raise ValueError(f"Qwen4 FT layer {i} must have one attention owner")
        plan.layer_types.append("linear_attention" if gdn else "full_attention")
        down = p + "ffn_weights.intermediate_weight2.kernel"
        if down in ft.entries:
            width, hidden = ft._shape(*ft.entries[down])
            if hidden != plan.hidden_size or width <= 0:
                raise ValueError(f"Malformed Qwen4 shared expert {down}")
            plan.shapes[p + "mlp.shared_expert.gate_proj.weight"] = (
                width * load.ffn_tp_size,
                hidden,
            )
        if p + "ffn_weights.shared_expert_gate.kernel" in ft.entries:
            plan.shapes[p + "mlp.shared_expert_gate.weight"] = (1, plan.hidden_size)
    if plan.text.get("layer_types", plan.layer_types) != plan.layer_types:
        raise ValueError(
            "Qwen4 FT attention types do not match the model configuration"
        )
    return plan


def load_rank_local(model, ft):
    bindings = []

    def identity(xs):
        return xs[0]

    def transpose(xs):
        return xs[0].t()

    def add(src, dst, conv=identity):
        bindings.append(RuntimeTensorBinding((src,), dst, conv))

    def hc(src, dest, inject):
        for suffix, leaf in (
            ("norm", "hc_norm"),
            ("mix_down", "input_mix_weight_down"),
            ("mix_up", "input_mix_weight_up"),
            ("inject", "block_inject_weight"),
        ):
            if suffix != "inject" or inject:
                add(src + "_" + suffix, dest + "." + leaf + ".weight")

    add("global.embedding", "embed_tokens.weight")
    add("global.lm_head", "lm_head.weight")
    hc("global.qwen4.hc.mixer", "hyper_connection_mixer", False)
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}."
        for unit, dest in (
            ("attn", "attn_hyper_connection"),
            ("mlp", "mlp_hyper_connection"),
        ):
            hc(p + "qwen4.hc." + unit, p + dest, True)
        fields = (
            (
                (
                    "linear_attn.in_proj_qkvz.weight",
                    "linear_attn.in_proj_qkvz.weight",
                    transpose,
                ),
                ("linear_attn.in_proj_ba.weight", "linear_attn.in_proj_ba_w", identity),
                (
                    "linear_attn.conv1d.weight",
                    "linear_attn.conv1d_w",
                    lambda xs: xs[0].squeeze(1) if xs[0].ndim == 3 else xs[0],
                ),
                ("linear_attn.dt_bias", "linear_attn.dt_bias", identity),
                ("linear_attn.A_log", "linear_attn.a_log", identity),
                ("linear_attn.norm.weight", "linear_attn.norm_w", identity),
                (
                    "linear_attn.out_proj.weight",
                    "linear_attn.out_proj.weight",
                    transpose,
                ),
            )
            if layer.layer_type == "linear_attention"
            else (
                (
                    "self_attention_weights.query_weight.kernel",
                    "self_attn.qkv_proj.weight",
                    transpose,
                ),
                (
                    "self_attention_weights.attention_output_weight.kernel",
                    "self_attn.o_proj.weight",
                    transpose,
                ),
                (
                    "self_attention_weights.gate.weight",
                    "self_attn.gate.weight",
                    transpose,
                ),
                (
                    "self_attention_weights.q_layernorm.gamma",
                    "self_attn.q_norm.weight",
                    identity,
                ),
                (
                    "self_attention_weights.k_layernorm.gamma",
                    "self_attn.k_norm.weight",
                    identity,
                ),
            )
        )
        for src, dst, conv in fields:
            add(p + src, p + dst, conv)
        if getattr(getattr(layer, "self_attn", None), "qsa_indexer", None) is not None:
            for src, dest in (
                ("qk_proj.weight", "index_qk_proj"),
                ("q_ln.gamma", "q_layernorm"),
                ("k_ln.gamma", "k_layernorm"),
            ):
                add(
                    p + "qwen4.indexer." + src,
                    p + "self_attn.qsa_indexer." + dest + ".weight",
                )
        add(p + "partial_moe_weights.gate.kernel", p + "mlp.gate.weight", transpose)
        bindings.extend(
            nonblock_moe_ft_bindings(layer.mlp.experts, p, p + "mlp.experts", ft, i)
        )
        if layer.mlp.shared_expert is not None:
            target = p + "mlp.shared_expert."
            add(
                p + "ffn_weights.intermediate_weight2.kernel",
                target + "down_proj.weight",
                transpose,
            )
            merged = p + "ffn_weights.intermediate_weight13.kernel"
            for j, (src, dest) in enumerate(
                (
                    ("intermediate_weight", "gate_proj"),
                    ("intermediate_weight3", "up_proj"),
                )
            ):
                if merged in ft.entries:
                    add(
                        merged,
                        target + dest + ".weight",
                        lambda xs, j=j: xs[0].t().chunk(2, dim=0)[j],
                    )
                else:
                    add(
                        p + "ffn_weights." + src + ".kernel",
                        target + dest + ".weight",
                        transpose,
                    )
        if layer.mlp.shared_expert_gate is not None:
            add(
                p + "ffn_weights.shared_expert_gate.kernel",
                p + "mlp.shared_expert_gate.weight",
                transpose,
            )
        if layer.ple is not None:
            for src, dest in (
                ("conv.weight", "conv1d.weight"),
                ("key_proj.weight", "key_proj.weight"),
                ("value_proj.weight", "value_proj.weight"),
                ("norm_conv.gamma", "norm_conv.weight"),
                ("norm_key.gamma", "norm_key.weight"),
                ("norm_query.gamma", "norm_query.weight"),
                ("multipliers", "ple_embedding.layer_multipliers"),
                ("ngram_offsets", "ple_embedding.ngram_heads_offsets"),
                ("ngram_vocab_sizes", "ple_embedding.ngram_heads_vocab_sizes"),
            ):
                add(p + "qwen4.ple." + src, p + "ple." + dest)
            table = layer.ple.ple_embedding.ngram_embedding
            for index in table.shard_indices:
                key = p + f"qwen4.ple.ngram_shards.{index}"
                shape = ft._shape(*ft.entries[key])
                table._validate_shape(shape)
                name = f"shard_{index}"
                param = getattr(table, name)
                # Allocate by header on the host: never send a PLE table to CUDA.
                param.data = torch.empty(shape, device="cpu", dtype=param.dtype)
                bindings.append(
                    RuntimeTensorBinding(
                        (key,),
                        p + "ple.ple_embedding.ngram_embedding." + name,
                        identity,
                        install=lambda xs, table=table, name=name: table.load_weights(
                            {name + ".weight": xs[0]}
                        ),
                    )
                )
    ft.load_parameters(model, tuple(bindings))
