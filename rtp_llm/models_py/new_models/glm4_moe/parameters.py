"""Install the actual GLM parameter tree on one main-runtime pipeline stage."""

import math

from torch import nn

from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.norm import RMSResNorm

from .checkpoint_layout import Glm4CheckpointLayout
from .decoder import Glm4MoeDecoderLayer


def install_glm4_parameters(owner, cfg, *, layer_ids, partitioned):
    state = cfg.get("resident_construction")
    if state is None:
        layout = Glm4CheckpointLayout.read(cfg["model_config"])
    else:
        fields = state["glm_layout"]
        layout = Glm4CheckpointLayout(**{**fields, "routed_layers": frozenset(fields["routed_layers"])})
    install_gqa_moe_parameters(
        owner,
        cfg,
        layer_ids=layer_ids,
        partitioned=partitioned,
        layout=layout,
        qkv_bias=True,
    )


def install_gqa_moe_parameters(owner, cfg, *, layer_ids, partitioned, layout, qkv_bias):
    model_config = cfg["model_config"]
    owner._layout = layout
    owner._owned_layer_ids = layer_ids
    owner._pipeline_partitioned = partitioned
    first = 0 in owner._owned_layer_ids
    last = cfg["num_layers"] - 1 in owner._owned_layer_ids
    owner.tie_word_embeddings = model_config.tie_word_embeddings
    owner.normalize_lm_head_weight = model_config.normalize_lm_head_weight
    owner.logit_scale = float(model_config.logit_scale)
    if not math.isfinite(owner.logit_scale):
        raise ValueError("logit_scale must be finite")
    owner._lm_head_postprocessed = False
    owner.embed_tokens = (
        VocabParallelEmbedding(
            cfg["vocab_size"],
            cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
            allow_vocab_padding=True,
        )
        if first
        else None
    )
    owned = {
        str(i): Glm4MoeDecoderLayer(
            cfg,
            i,
            dense_width=owner._layout.dense_width,
            shared_width=owner._layout.shared_width,
            moe_layers=owner._layout.routed_layers,
            use_qk_norm=owner._layout.use_qk_norm,
            correction_bias=owner._layout.correction_bias,
            qkv_bias=qkv_bias,
        )
        for i in owner._owned_layer_ids
    }
    owner.layers = (
        nn.ModuleDict(owned)
        if owner._pipeline_partitioned
        else nn.ModuleList(owned.values())
    )
    owner.norm = (
        RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        if last
        else None
    )
    # Non-generation tasks consume normalized hidden states. Any raw output
    # projection requested by a classifier/reranker belongs to its handler,
    # not to the TP-sharded, optionally normalized/scaled generation head.
    needs_lm_head = last
    if needs_lm_head and hasattr(model_config, "task_type"):
        from rtp_llm.ops import TaskType

        needs_lm_head = model_config.task_type == TaskType.LANGUAGE_MODEL
    owner.lm_head = (
        ParallelLMHead(
            cfg["vocab_size"],
            cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
            **output_vocab_head_kwargs(model_config),
        )
        if needs_lm_head
        else None
    )
    if owner.tie_word_embeddings and first and owner.lm_head is not None:
        owner.embed_tokens.tie_output_head(owner.lm_head)
