"""
DeepSeek V3.2 (4-layer) for new-loader.

Top-level model: DeepSeekV32ForCausalLM
  - load_weights() applies WEIGHTS_MAPPER (prefix_mapping={"model.": ""})
    then delegates to RtpModule's streaming dispatch via super().
  - __init__ builds all submodules with HF-compatible names.
  - process_weights_after_loading() fuses QKV projections and KV cache
    projections (kc/vc from kv_b).
  - RoPE cos/sin cache is computed on-the-fly from config.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from .model import DeepSeekV32DecoderLayer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  RoPE helpers (mirrors DeepSeekV2._create_rope_w)
# ------------------------------------------------------------------ #


def _build_rope_cache(
    config_json: dict,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cos/sin cache from config.json + max_seq_len.

    Returns [cos|sin] concatenated, shape [max_seq_len, rope_head_dim], float32.
    """
    from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
        DeepseekV3RotaryEmbedding,
        DeepseekV3YarnRotaryEmbedding,
    )

    rope_scaling = config_json.get("rope_scaling")
    rope_parameters = config_json.get("rope_parameters", {})
    rope_theta = rope_parameters.get(
        "rope_theta", config_json.get("rope_theta", 10000.0)
    )
    rope_head_dim = config_json["qk_rope_head_dim"]

    has_yarn = rope_scaling is not None

    if not has_yarn:
        rotary_emb = DeepseekV3RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            device=device,
        )
    else:
        # device kw is required: DeepSeekV32 uses yarn scaling, and IndexerOp
        # stores cos_sin_cache as a plain Python attr (not a buffer), so any
        # later .to(cuda) does NOT propagate into the indexer's copy. Build
        # on cuda from the start so the indexer captures a cuda reference.
        rotary_emb = DeepseekV3YarnRotaryEmbedding(
            rope_head_dim,
            max_seq_len,
            rope_theta,
            device=device,
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=float(rope_scaling.get("beta_fast", 32)),
            beta_slow=float(rope_scaling.get("beta_slow", 1)),
            mscale=rope_scaling["mscale"],
            mscale_all_dim=rope_scaling["mscale_all_dim"],
        )

    half_rope_dim = rope_head_dim // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    return torch.cat([cos_cache, sin_cache], dim=-1).contiguous().to(torch.float32)


def _read_config_json(ckpt_path: str) -> Dict[str, Any]:
    """Read config.json from ckpt path, return empty dict if not found."""
    if not ckpt_path:
        return {}
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.loads(f.read())


# ------------------------------------------------------------------ #
#  Config extraction (mirrors DeepSeekV2._from_hf)
# ------------------------------------------------------------------ #


def _extract_config_values(
    model_config: Any, load_config: Any, config_json: dict = None
) -> Dict[str, Any]:
    """Read config from either ModelConfig (C++ pybind) or HF dict.

    config_json is the raw config.json dict; used to resolve fields that
    the old-loader's _create_config overwrites on model_config (e.g.
    inter_size -> n_shared_experts * moe_intermediate_size).
    """

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    hidden_size = _get(model_config, "hidden_size", 7168)
    num_layers = _get(
        model_config, "num_layers", _get(model_config, "num_hidden_layers", 4)
    )
    vocab_size = _get(model_config, "vocab_size", 102400)
    max_seq_len = _get(model_config, "max_seq_len", 8192)

    # Attention config
    attn_config = _get(model_config, "attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 128)
        q_lora_rank = _get(attn_config, "q_lora_rank", 1536)
        kv_lora_rank = _get(attn_config, "kv_lora_rank", 512)
        nope_head_dim = _get(attn_config, "nope_head_dim", 128)
        rope_head_dim = _get(attn_config, "rope_head_dim", 64)
        v_head_dim = _get(attn_config, "v_head_dim", 128)
        is_sparse = _get(attn_config, "is_sparse", False)
        indexer_head_dim = _get(attn_config, "indexer_head_dim", 128)
        indexer_head_num = _get(attn_config, "indexer_head_num", 64)
        indexer_topk = _get(attn_config, "indexer_topk", 2048)
    else:
        num_heads = _get(model_config, "num_attention_heads", 128)
        q_lora_rank = _get(model_config, "q_lora_rank", 1536)
        kv_lora_rank = _get(model_config, "kv_lora_rank", 512)
        nope_head_dim = _get(model_config, "qk_nope_head_dim", 128)
        rope_head_dim = _get(model_config, "qk_rope_head_dim", 64)
        v_head_dim = _get(model_config, "v_head_dim", 128)
        is_sparse = _get(model_config, "is_sparse", False)
        indexer_head_dim = _get(model_config, "index_head_dim", 128)
        indexer_head_num = _get(model_config, "index_n_heads", 64)
        indexer_topk = _get(model_config, "index_topk", 2048)

    rms_norm_eps = _get(
        model_config, "rms_norm_eps", _get(model_config, "layernorm_eps", 1e-6)
    )

    # MoE config
    num_experts = _get(
        model_config, "expert_num", _get(model_config, "n_routed_experts", 256)
    )
    top_k = _get(model_config, "moe_k", _get(model_config, "num_experts_per_tok", 8))
    # ModelConfig.moe_inter_size defaults to 0 and the legacy DeepSeek
    # loader never overwrites it — reading model_config first would yield
    # 0, breaking the expert buffer shapes. Prefer config.json's
    # moe_intermediate_size, fall through to the default only if absent.
    moe_intermediate_size = None
    if config_json:
        moe_intermediate_size = config_json.get("moe_intermediate_size")
    if not moe_intermediate_size:
        mc_moe = _get(model_config, "moe_inter_size", 0) or _get(
            model_config, "moe_intermediate_size", 0
        )
        moe_intermediate_size = mc_moe or 2048
    n_shared_experts = _get(model_config, "n_shared_experts", 1)
    if config_json:
        n_shared_experts = config_json.get("n_shared_experts", n_shared_experts)
    shared_expert_intermediate_size = _get(
        model_config,
        "shared_expert_intermediate_size",
        None,
    )
    if shared_expert_intermediate_size is None and config_json:
        shared_expert_intermediate_size = config_json.get(
            "shared_expert_intermediate_size",
            n_shared_experts * moe_intermediate_size,
        )
    if shared_expert_intermediate_size is None:
        shared_expert_intermediate_size = n_shared_experts * moe_intermediate_size

    # NOTE: model_config.inter_size is overridden by the old-loader to
    # n_shared_experts * moe_intermediate_size.  Read the real dense FFN
    # width from config_json (HF ckpt) when available.
    dense_intermediate_size = None
    if config_json:
        dense_intermediate_size = config_json.get("intermediate_size")
    if dense_intermediate_size is None:
        dense_intermediate_size = _get(
            model_config,
            "intermediate_size",
            _get(model_config, "inter_size", 18432),
        )

    # first_k_dense_replace / moe_layer_freq are not propagated onto
    # ModelConfig by the legacy loader either — read from config.json,
    # falling back only as a last resort. Wrong values here mismap dense
    # vs MoE layers and routes dense ckpt tensors into MoEBlock.
    first_k_dense_replace = None
    moe_layer_freq = None
    if config_json:
        first_k_dense_replace = config_json.get("first_k_dense_replace")
        moe_layer_freq = config_json.get("moe_layer_freq")
    if first_k_dense_replace is None:
        first_k_dense_replace = _get(model_config, "first_k_dense_replace", 1)
    if moe_layer_freq is None:
        moe_layer_freq = _get(model_config, "moe_layer_freq", 1)
    moe_layer_index = [
        i
        for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]

    scoring_func = _get(model_config, "scoring_func", 1)  # 0=softmax, 1=sigmoid
    routed_scaling_factor = _get(model_config, "routed_scaling_factor", 1.0)
    n_group = _get(model_config, "moe_n_group", _get(model_config, "n_group", None))
    topk_group = _get(
        model_config, "moe_topk_group", _get(model_config, "topk_group", None)
    )
    has_moe_norm = _get(
        model_config, "has_moe_norm", _get(model_config, "norm_topk_prob", None)
    )
    if config_json:
        if n_group is None:
            n_group = config_json.get("n_group", 1)
        if topk_group is None:
            topk_group = config_json.get("topk_group", 1)
        if has_moe_norm is None:
            has_moe_norm = config_json.get("norm_topk_prob", False)
    n_group = n_group if n_group is not None else 1
    topk_group = topk_group if topk_group is not None else 1
    has_moe_norm = has_moe_norm if has_moe_norm is not None else False
    if config_json:
        routed_scaling_factor = config_json.get(
            "routed_scaling_factor", routed_scaling_factor
        )
    # has_e_score_correction is not a ModelConfig field — the legacy loader
    # detects it from ckpt key presence on the weight class side. Derive it
    # here from config.json's topk_method ("noaux_tc" => correction bias).
    has_e_score_correction = _get(model_config, "has_e_score_correction", False)
    if not has_e_score_correction and config_json:
        has_e_score_correction = config_json.get("topk_method") == "noaux_tc"

    # Rope interleave style.
    # The old loader sets these on model_config.attn_config.rope_config
    # (is_neox_style / indexer_is_neox_style), NOT as direct attributes on
    # model_config.  So _get(model_config, "rope_interleave", ...) returns the
    # default.  Read from config_json to get the raw value — this matters for
    # GLM-5 which may set rope_interleave=False / indexer_rope_interleave=True.
    rope_interleave = _get(model_config, "rope_interleave", None)
    if rope_interleave is None and config_json:
        rope_interleave = config_json.get("rope_interleave", True)
    if rope_interleave is None:
        rope_interleave = True
    is_neox_style = not rope_interleave

    indexer_rope_interleave = _get(model_config, "indexer_rope_interleave", None)
    if indexer_rope_interleave is None and config_json:
        indexer_rope_interleave = config_json.get("indexer_rope_interleave", False)
    if indexer_rope_interleave is None:
        indexer_rope_interleave = False
    indexer_is_neox_style = not indexer_rope_interleave

    # Parallelism
    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    ep_size = getattr(load_config, "ep_size", 1)
    ep_rank = getattr(load_config, "ep_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    # DeepSeek-V3.2 runs its NON-expert linears (MLA q/kv/o projections, dense
    # FFN, shared expert, indexer) in bf16: the MLA absorb path derives kc/vc
    # via torch.bmm (no fp8 kernel), and fp8 GEMM diverges from the validated
    # bf16 reference output. So for an already-quantized fp8-per-block ckpt,
    # route those linears to the dequant-to-bf16 method. The routed experts
    # keep fp8 — DeepSeekV32Experts._EXTRA_QUANT_MAP maps "fp8_block_dequant"
    # back to "fp8_per_block".
    if quant_config is not None and getattr(quant_config, "quant_type", "") == (
        "fp8_block"
    ):
        quant_config = QuantizationConfig(quant_type="fp8_block_dequant")
    params_dtype = getattr(load_config, "compute_dtype", torch.bfloat16)
    parallelism_config = getattr(load_config, "parallelism_config", None)
    moe_config = getattr(load_config, "moe_config", None)

    # Kernel tokens per block
    blocksize = _get(model_config, "kernel_tokens_per_block", 64)
    if attn_config is not None:
        blocksize = _get(attn_config, "kernel_tokens_per_block", blocksize)

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        v_head_dim=v_head_dim,
        rms_norm_eps=rms_norm_eps,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        dense_intermediate_size=dense_intermediate_size,
        moe_layer_index=moe_layer_index,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        n_group=n_group,
        topk_group=topk_group,
        has_moe_norm=has_moe_norm,
        has_e_score_correction=has_e_score_correction,
        is_sparse=is_sparse,
        indexer_head_dim=indexer_head_dim,
        indexer_head_num=indexer_head_num,
        indexer_topk=indexer_topk,
        is_neox_style=is_neox_style,
        indexer_is_neox_style=indexer_is_neox_style,
        blocksize=blocksize,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class DeepSeekV32ForCausalLM(GptModelBase):
    """DeepSeek V3.2 (4-layer) for new-loader.

    WEIGHTS_MAPPER only strips "model." prefix.  All submodule names match
    HF ckpt keys directly, so RtpModule.load_weights can dispatch weights
    without any fusion-time mapping.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    @staticmethod
    def _read_config_json(ckpt_path: str) -> Dict[str, Any]:
        """Read config.json from ckpt path.

        Overridable by subclasses (e.g. DeepSeek VL V2 merges the nested
        ``language_config`` section into the top-level dict so that
        _extract_config_values can find MLA / MoE fields).
        """
        return _read_config_json(ckpt_path)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if (
                    name == "lm_head.weight"
                    or name.startswith("lm_head.")
                    or name == "model.lm_head.weight"
                    or name.startswith("model.lm_head.")
                ):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head:
            logger.info(
                "[DeepSeekV32] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
    ):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )

        # Resolve ckpt_path from model_config.ckpt_path (C++ pybind attribute).
        ckpt_path = ""
        if hasattr(model_config, "ckpt_path") and model_config.ckpt_path:
            ckpt_path = model_config.ckpt_path

        # Read config.json early — _extract_config_values needs it to
        # resolve fields that old-loader's _create_config overwrites on
        # model_config (e.g. inter_size).
        # _read_config_json is overridable so multimodal variants (e.g.
        # DeepSeek VL V2) can merge nested sub-configs into the top level.
        config_json = self._read_config_json(ckpt_path)

        cfg = _extract_config_values(model_config, load_config, config_json)

        # --- RoPE cache: read config.json directly for full rope fields ---
        device = torch.device("cuda")
        cos_sin_cache = _build_rope_cache(
            config_json if config_json else cfg,
            cfg["max_seq_len"],
            device,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        # --- Embedding ---
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        # --- Decoder layers ---
        moe_layer_set = set(cfg["moe_layer_index"])
        self.layers = nn.ModuleList()

        for i in range(cfg["num_layers"]):
            is_moe = i in moe_layer_set
            layer = DeepSeekV32DecoderLayer(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                q_lora_rank=cfg["q_lora_rank"],
                kv_lora_rank=cfg["kv_lora_rank"],
                nope_head_dim=cfg["nope_head_dim"],
                rope_head_dim=cfg["rope_head_dim"],
                v_head_dim=cfg["v_head_dim"],
                layer_idx=i,
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                params_dtype=cfg["params_dtype"],
                layernorm_eps=cfg["rms_norm_eps"],
                quant_config=cfg["quant_config"],
                model_config=cfg["model_config"],
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                is_moe_layer=is_moe,
                dense_intermediate_size=cfg["dense_intermediate_size"],
                moe_intermediate_size=cfg["moe_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
                has_shared_expert=True,
                scoring_func=cfg["scoring_func"],
                routed_scaling_factor=cfg["routed_scaling_factor"],
                n_group=cfg["n_group"],
                topk_group=cfg["topk_group"],
                has_moe_norm=cfg["has_moe_norm"],
                correction_bias=cfg["has_e_score_correction"],
                is_sparse=cfg["is_sparse"],
                index_n_heads=cfg["indexer_head_num"],
                index_head_dim=cfg["indexer_head_dim"],
                index_topk=cfg["indexer_topk"],
                indexer_is_neox_style=cfg["indexer_is_neox_style"],
                cos_sin_cache=cos_sin_cache,
                blocksize=cfg["blocksize"],
            )
            self.layers.append(layer)

        # --- Final norm ---
        self.norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

        # --- LM head ---
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

    def initialize(self, init_resource):
        """Build ModelWeights view after all post-load hooks have run.

        Called by C++ PyWrappedModel after weight loading +
        process_weights_after_loading completes, before any prepare_fmha_impl /
        forward.  By this point every self_attn module has _fused_qkv_a_w /
        _kc_w / _vc_w populated, so _build_weights_dict() is safe to call.
        """
        ok = super().initialize(init_resource)
        self._ensure_weight_assembled()
        return ok

    def _ensure_weight_assembled(self):
        """Build the ModelWeights view that prepare_fmha_impl / MlaImpl expects.

        Cannot run inside __init__ (params not loaded yet) or inside
        process_weights_after_loading (parent hook fires before children's,
        so layer.self_attn._fused_qkv_a_w etc. would still be None).
        Called from initialize() once all child post-load hooks have run, and
        also kept as a lazy fallback in forward() for direct-execution paths.
        """
        if self.weight is not None:
            return
        num_layers = len(self.layers)
        device = next(self.parameters()).device
        weights = ModelWeights(
            num_layers=num_layers,
            device=str(device),
            dtype=self.cos_sin_cache.dtype,
        )
        weights.set_global_weight(W.rope_cos_sin_cache, self.cos_sin_cache)
        for i, layer in enumerate(self.layers):
            for key, tensor in layer.self_attn._build_weights_dict().items():
                weights.set_layer_weight(i, key, tensor)
        self.weight = weights

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            self._ensure_weight_assembled()
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
