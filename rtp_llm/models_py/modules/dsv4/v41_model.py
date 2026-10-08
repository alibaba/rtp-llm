"""Loader-to-stage assembly behind the V4.1 native serving gate.

Construction performs no checkpoint I/O, cache allocation or communication.
The worker owns the supplied Engram lifetime and must admit every EP operator
collectively before execution. Native request/cache admission remains external.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.v41_attention_projection import (
    build_v41_projections,
)
from rtp_llm.models_py.modules.dsv4.v41_block import build_v41_block, start_v41_residual
from rtp_llm.models_py.modules.dsv4.v41_collective import V41CollectiveMoE, V41EPPlan
from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout
from rtp_llm.models_py.modules.dsv4.v41_engram_stage import V41EngramStage
from rtp_llm.models_py.modules.dsv4.v41_feed_forward import build_v41_feed_forward
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE, _assert_values
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import _require_sm120, _scale_bytes
from rtp_llm.models_py.modules.dsv4.v41_projection import _tensor
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts, _fp4_bytes
from rtp_llm.models_py.modules.dsv4.v41_stage import V41Stage
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class V41ModelStage:
    """Stage-owned numerical model; LM head and Engram teardown belong to the worker.

    Use ``start`` on embedding stages, then ``stage.run`` or an admitted CP
    forward. Stage one resumes through the validated PP adapter. This is not a
    GptModelBase registration and cannot infer native scheduler metadata.
    """

    stage: V41Stage
    embedding: torch.Tensor | None

    def start(self, input_ids, valid_rows):
        if self.embedding is None:
            raise ValueError("V4.1 stage one must resume the validated PP frame")
        if (
            input_ids.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.device != self.stage.device
            or not input_ids.is_contiguous()
            or input_ids.numel() > self.stage.max_rows
        ):
            raise ValueError("V4.1 embedding requires bounded device token IDs")
        _tensor(
            valid_rows, input_ids.shape, torch.bool, self.stage.device, "token mask"
        )
        safe = torch.where(valid_rows, input_ids, 0)
        _assert_values(
            ((safe >= 0) & (safe < self.embedding.shape[0])).all(),
            "V4.1 live token ID is outside the embedding vocabulary",
        )
        return start_v41_residual(F.embedding(safe, self.embedding), valid_rows)


def _validate_config(config, max_positions):
    if not isinstance(config, V41Config):
        raise ValueError("V4.1 assembly requires validated checkpoint metadata")
    config.validate()
    text = config.text
    fixed = {
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "sliding_window": 128,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-20,
        "n_shared_experts": 1,
        "scoring_func": "sqrtsoftplus",
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "engram_layer_ids": [1, 14],
    }
    for key, value in fixed.items():
        if text.get(key) != value:
            raise ValueError(f"V4.1 assembly requires checkpoint {key}={value!r}")
    if text["compress_ratios"][:40] != [0] * 2 + [2] * 18 + [1] * 20:
        raise ValueError("V4.1 assembly requires the checkpoint compression schedule")
    if text["q_lora_rank"] % 32 or text["o_lora_rank"] % 32:
        raise ValueError("V4.1 projection ranks require group-32 alignment")
    if type(max_positions) is not int or not 1 <= max_positions <= min(
        32768, text["max_position_embeddings"]
    ):
        raise ValueError("V4.1 assembly requires a bounded 32K context")
    scaling = text.get("rope_scaling")
    if not isinstance(scaling, Mapping) or scaling.get("rope_type") != "yarn":
        raise ValueError("V4.1 assembly requires explicit checkpoint YaRN metadata")
    for value in (
        text.get("rope_theta"),
        text.get("compress_rope_theta"),
        scaling.get("factor"),
        scaling.get("beta_fast"),
        scaling.get("beta_slow"),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("V4.1 RoPE parameters must be finite and positive")
    original = scaling.get("original_max_position_embeddings")
    if (
        type(original) is not int
        or original <= 0
        or min(text["rope_theta"], text["compress_rope_theta"]) <= 1
        or scaling["factor"] < 1
        or scaling["beta_fast"] < scaling["beta_slow"]
    ):
        raise ValueError("V4.1 YaRN range/base/factor is invalid")


def _layer_specs(text, source, local_experts):
    """Exact TP1 loader geometry, checked for every layer before any conversion."""
    dim, inter = text["hidden_size"], text["moe_intermediate_size"]
    qrank, orank = text["q_lora_rank"], text["o_lora_rank"]
    specs = {}

    def add(name, shape, dtype):
        specs[getattr(W, name)] = (shape, dtype)

    for name, n, k in (
        ("attn_wq_a", qrank, dim),
        ("attn_wq_b", 64 * 512, qrank),
        ("attn_wkv", 512, dim),
        ("attn_wo_a", 8 * orank, 4096),
        ("attn_wo_b", dim, 8 * orank),
        ("shared_w13", 2 * inter, dim),
        ("shared_w2", dim, inter),
    ):
        add("v4_" + name + "_w", (n, k), torch.float8_e4m3fn)
        add("v4_" + name + "_s", (n // 32, k // 32), "scale")
    for name, shape in (
        ("attn_q_norm", (qrank,)),
        ("attn_kv_norm", (512,)),
        ("attn_norm", (dim,)),
        ("ffn_norm", (dim,)),
        ("router_w", (text["n_routed_experts"], dim)),
    ):
        add("v4_" + name, shape, torch.bfloat16)
    add("v4_attn_sink", (64,), torch.float32)
    add("v4_router_bias", (text["n_routed_experts"],), torch.float32)
    for kind in ("attn", "ffn"):
        for name, shape in (("fn", (24, 4 * dim)), ("base", (24,)), ("scale", (3,))):
            add("v4_hc_" + kind + "_" + name, shape, torch.float32)
    for name, n, k in (("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)):
        add("v4_routed_" + name + "_w", (local_experts, n, k // 2), "fp4")
        add("v4_routed_" + name + "_s", (local_experts, n, k // 32), "scale")
    if source.owns_kv:
        add("v4_compressor_wkv", (512, dim), torch.bfloat16)
        add("v4_compressor_norm", (512,), torch.bfloat16)
        add("v41_indexer_wk", (128, 512), torch.bfloat16)
        add("v41_indexer_k_norm", (128,), torch.bfloat16)
        if source.compress_ratio == 2:
            add("v4_compressor_wgate", (512, dim), torch.bfloat16)
    if source.owns_index:
        add("v4_indexer_wq_b_w", (4096, qrank), torch.float8_e4m3fn)
        add("v4_indexer_wq_b_s", (128, qrank // 32), "scale")
        add("v4_indexer_weights_proj_w", (32, dim), torch.bfloat16)
    return specs


def _validate_layer(weights, specs, device, layer):
    if not isinstance(weights, Mapping) or set(weights) != set(specs):
        raise ValueError(
            f"V4.1 L{layer} loader keys disagree with stage/source ownership"
        )
    for name, (shape, dtype) in specs.items():
        value = weights[name]
        if not isinstance(value, torch.Tensor) or value.requires_grad:
            raise ValueError(f"V4.1 L{layer} {name} must be an inference tensor")
        if dtype == "scale":
            value, dtype = _scale_bytes(value), torch.uint8
        elif dtype == "fp4":
            value, dtype = _fp4_bytes(value), torch.uint8
        elif dtype == torch.float8_e4m3fn:
            if value.dtype != dtype:
                raise ValueError(f"V4.1 L{layer} {name} requires E4M3 weights")
            value, dtype = value.view(torch.uint8), torch.uint8
            if device.type == "cpu" and bool(((value & 127) == 127).any()):
                raise ValueError(f"V4.1 L{layer} {name} must be finite")
        _tensor(value, shape, dtype, device, f"L{layer} {name}")


def _local_experts(weights, text, expert_ids, total_rows, backend):
    tensors = {}
    for name in ("w", "s"):
        view = _fp4_bytes if name == "w" else _scale_bytes
        # The loader owns separate gate/up stacks. Merge bytes gate-first once.
        tensors["w13_" + name] = torch.cat(
            [
                view(weights[getattr(W, "v4_routed_" + part + "_" + name)])
                for part in ("w1", "w3")
            ],
            dim=1,
        )
        tensors["w2_" + name] = weights[getattr(W, "v4_routed_w2_" + name)]
    return V41LocalMoE(
        V41RoutedExperts(
            tensors,
            swiglu_limit=text["swiglu_limit"],
            backend="eager" if backend == "eager" else "deepgemm",
        ),
        expert_start_id=expert_ids.start,
        max_tokens=total_rows,
        masked_token_limit=total_rows,
        max_topk=text["num_experts_per_tok"],
    )


def build_v41_model_stage(
    config,
    model_weights,
    *,
    role,
    pp_rank,
    ep_rank,
    world_ranks,
    generation,
    process_group,
    engram,
    rows_per_rank,
    max_positions=32768,
    backend="flashinfer",
):
    """Bind ModelWeights without consuming or modifying any loader dictionaries.

    All local layers are validated before allocation/conversion. Loader references
    can be released only after complete worker startup, including peer admission.
    Converted grouped-output, merged FP4 and prepared-scale allocations coexist
    with loader tensors during startup; the worker must budget that peak.
    ``eager`` constructs CPU diagnostic components, never a collective fallback.
    """
    _validate_config(config, max_positions)
    text = config.text
    if backend not in ("flashinfer", "eager"):
        raise ValueError("Unknown V4.1 assembly backend")
    if (
        role not in ("prefill", "decode")
        or type(pp_rank) is not int
        or pp_rank not in ((0, 1) if role == "prefill" else (0,))
        or type(rows_per_rank) is not int
        or not 1 <= rows_per_rank <= (32 if role == "prefill" else 128)
    ):
        raise ValueError("V4.1 assembly requires bounded CEP4PP2 or DP8EP8 ownership")
    begin, end = (
        ((0, 21) if pp_rank == 0 else (21, 40)) if role == "prefill" else (0, 40)
    )
    expert_ids = config.local_expert_ids(4 if role == "prefill" else 8, ep_rank)
    dispatch = V41DispatchLayout(
        text["hidden_size"], text["num_experts_per_tok"], text["n_routed_experts"]
    )
    plans = {
        layer: V41EPPlan(
            dispatch,
            role,
            pp_rank,
            layer,
            world_ranks,
            generation,
            rows_per_rank,
            swiglu_limit=text["swiglu_limit"],
            intermediate_dim=text["moe_intermediate_size"],
        )
        for layer in range(begin, end)
    }
    if process_group is None:
        raise ValueError("V4.1 assembly requires an explicit EP process group")
    layers = model_weights.weights
    if (
        not isinstance(layers, (list, tuple))
        or len(layers) != 40
        or any(layers[layer] for layer in range(40) if not begin <= layer < end)
    ):
        raise ValueError(
            "V4.1 loader requires 40 global layer slots with only stage-local weights"
        )
    globals_ = model_weights.global_weights
    if not isinstance(globals_, Mapping):
        raise ValueError("V4.1 loader requires global weights")
    anchor = W.embedding if begin == 0 else W.final_ln_gamma
    if not isinstance(globals_.get(anchor), torch.Tensor):
        raise ValueError("V4.1 loader is missing its stage-owned embedding/final norm")
    device, dim = globals_[anchor].device, text["hidden_size"]
    if device.type != ("cpu" if backend == "eager" else "cuda"):
        raise ValueError("V4.1 eager assembly requires CPU; FlashInfer requires CUDA")
    if backend == "flashinfer":
        _require_sm120(device)
        if dim % 128 or text["moe_intermediate_size"] % 128:
            raise ValueError("V4.1 SM120 expert dimensions require 128 alignment")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 model assembly must precede graph capture")
    embedding = globals_.get(W.embedding)
    final_norm = globals_.get(W.final_ln_gamma)
    if begin == 0:
        _tensor(
            embedding, (text["vocab_size"], dim), torch.bfloat16, device, "embedding"
        )
    elif embedding is not None:
        raise ValueError("V4.1 stage one cannot own the embedding")
    if end == 40:
        _tensor(final_norm, (dim,), torch.bfloat16, device, "final norm")
    elif final_norm is not None or W.lm_head in globals_:
        raise ValueError("V4.1 stage zero cannot own final readout/LM head")
    if any(
        name in globals_
        for name in (W.v4_hc_head_fn, W.v4_hc_head_base, W.v4_hc_head_scale)
    ):
        raise ValueError("V4.1 cannot load the V4 learned HC head")
    owned = tuple(layer for layer in text["engram_layer_ids"] if begin <= layer < end)
    if (
        not isinstance(engram, V41EngramStage)
        or engram.layer_ids != owned
        or engram.device != device
        or engram.backend != backend
    ):
        raise ValueError("V4.1 assembly requires the stage-owned Engram device/backend")
    sources = config.layer_sources()
    for layer in range(begin, end):
        _validate_layer(
            layers[layer],
            _layer_specs(text, sources[layer], len(expert_ids)),
            device,
            layer,
        )

    scaling = text["rope_scaling"]
    frequencies = {
        compressed: precompute_freqs_cis(
            64,
            max_positions,
            scaling["original_max_position_embeddings"] if compressed else 0,
            text["compress_rope_theta"] if compressed else text["rope_theta"],
            scaling["factor"],
            scaling["beta_fast"],
            scaling["beta_slow"],
            device=device,
        )
        for compressed in {
            bool(sources[layer].compress_ratio) for layer in range(begin, end)
        }
    }
    blocks = []
    for layer in range(begin, end):
        weights, source = layers[layer], sources[layer]
        attention, kv = build_v41_projections(
            source,
            weights,
            frequencies[bool(source.compress_ratio)],
            backend=backend,
            max_rows=rows_per_rank,
        )
        local = _local_experts(
            weights, text, expert_ids, len(world_ranks) * rows_per_rank, backend
        )
        routed = V41CollectiveMoE(local, plans[layer], process_group)
        feed_forward = build_v41_feed_forward(
            layer,
            weights,
            routed,
            max_rows=rows_per_rank,
            topk=text["num_experts_per_tok"],
            gate_temp=text.get("gate_temp", 1.0),
            route_scale=text["routed_scaling_factor"],
            norm_topk_prob=text["norm_topk_prob"],
            swiglu_limit=text["swiglu_limit"],
            backend=backend,
        )
        blocks.append(build_v41_block(weights, attention, kv, feed_forward))
    return V41ModelStage(
        V41Stage(
            blocks,
            config,
            role=role,
            pp_rank=pp_rank,
            engram=engram,
            final_norm=final_norm,
            max_positions=max_positions,
        ),
        embedding,
    )
