"""PPU DSpark checkpoint loading; the target/draft pools and EP groups stay separate.

Calculation source: SGLang (Apache-2.0), b313eda591b53f263c811ac001239855506f64ba,
deepseek_v4_dspark.py: DSparkV4StageMixin, _dspark_stage_config, commit_hidden_states.
Only computation/weight names are adapted; no SGLang runtime is imported.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    _build_freqs_cis,
    _CheckpointSource,
    _load_dense,
    _load_vector,
    _require,
    load_ppu_weights,
)


def draft_config(config):
    """Use real draft geometry rather than the target's 384-expert layers."""
    result = dict(config)
    layers = int(config["num_nextn_predict_layers"])
    if layers != 3:
        raise ValueError("V4.1 DSpark requires exactly three real draft layers")
    result.update(
        num_hidden_layers=layers,
        num_layers=layers,
        compress_ratios=[0] * layers,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[],
        engram_num_embeddings=[],
        n_routed_experts=int(config["dspark_n_routed_experts"]),
        num_experts_per_tok=int(config["dspark_num_experts_per_tok"]),
        original_seq_len=0,
        ppu_model_identity="dspark-draft",
    )
    return result


@dataclass
class DraftWeights:
    config: dict
    bundle: object
    main_proj: object
    main_norm: torch.Tensor
    confidence_head: object = None


def load_draft_weights(
    config,
    checkpoint,
    *,
    ep_rank=0,
    ep_size=1,
    device=None,
    commit_only=False,
    confidence_enabled=False,
):
    config = draft_config(config)
    geometry = V41Geometry.from_config(config)
    source = _CheckpointSource(checkpoint)
    device = torch.device(device or "cpu")
    main_proj = _load_dense(source, "mtp.0.main_proj")
    main_proj.weight = main_proj.weight.to(device)
    main_norm = _require(
        _load_vector(source, "mtp.0.main_norm.weight"), "mtp.0.main_norm.weight"
    ).to(device)
    if tuple(main_proj.weight.shape) != (
        geometry.hidden_size,
        len(config["dspark_target_layer_ids"]) * geometry.hidden_size,
    ):
        raise ValueError("draft main projection disagrees with target capture width")
    if commit_only:
        # P never evaluates proposal/FFN; do not load an unused second MoE.
        layers = {}
        for layer in range(3):
            wkv = _load_dense(source, f"mtp.{layer}.attn.wkv")
            wkv.weight = wkv.weight.to(device)
            layers[layer] = SimpleNamespace(
                wkv=wkv,
                kv_norm=_require(
                    _load_vector(source, f"mtp.{layer}.attn.kv_norm.weight"),
                    f"mtp.{layer}.attn.kv_norm.weight",
                ).to(device),
            )
        freqs = _build_freqs_cis(geometry, geometry.rope_theta, device)
        bundle = SimpleNamespace(
            geometry=geometry,
            layers=layers,
            embedding=torch.empty(
                (0, geometry.hidden_size), dtype=torch.bfloat16, device=device
            ),
            freqs_for=lambda _: freqs,
        )
    else:
        bundle = load_ppu_weights(
            config,
            checkpoint,
            ep_rank=ep_rank,
            ep_size=ep_size,
            device=device,
            layer_prefix="mtp",
            final_norm_name="mtp.2.norm.weight",
            load_lm_head=False,
        )
    confidence = None
    if confidence_enabled and not commit_only:
        confidence = _require(
            source.get("mtp.2.confidence_head.proj.weight"),
            "mtp.2.confidence_head.proj.weight",
        ).to(device=device, dtype=torch.float32)
        if tuple(confidence.shape) != (
            1,
            geometry.hidden_size + int(config["dspark_markov_rank"]),
        ):
            raise ValueError("draft confidence head must be FP32 [1,H+markov_rank]")
    return DraftWeights(config, bundle, main_proj, main_norm, confidence)
