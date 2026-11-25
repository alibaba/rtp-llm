"""Qwen4.5-Flash DFlash2 (DFlashQwenDraftModel / SP_TYPE_DSPARK) model definition."""

from dataclasses import dataclass
import functools
import inspect
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.qwen4_5_config import (
    MXFP4_GROUP_SIZE,
    parse_qwen4_5_quant_config,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, CustomAtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import FULL_ATTN_KV_TAG
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    KVCacheConfig,
    KVCacheSpecDesc,
    KVCacheSpecType,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
    TpCacheTransferMode,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    choose_available,
    identity,
    merge_qkv_hf,
    sp_0,
    sp_id,
    tolerate_failed,
    transpose,
)


class DFLASH2_W:
    """Additional weight keys for Qwen4.5 DFlash2 modules."""

    hc_mixer_norm_w = "dflash2.hc_mixer.hc_norm.weight"
    hc_mixer_norm_b = "dflash2.hc_mixer.hc_norm.bias"
    hc_mixer_down = "dflash2.hc_mixer.input_mix_weight_down.weight"
    hc_mixer_up = "dflash2.hc_mixer.input_mix_weight_up.weight"
    selector_hidden_proj = "dflash2.candidate_selector.hidden_projection.weight"

    attn_output_gate = "dflash2.self_attn.output_gate_proj.weight"
    attn_conv_base = "dflash2.attention_conv.base_kernel"
    attn_conv_proj = "dflash2.attention_conv.kernel_projection.weight"
    mlp_conv_base = "dflash2.mlp_conv.base_kernel"
    mlp_conv_proj = "dflash2.mlp_conv.kernel_projection.weight"

    attn_hc_norm_w = "dflash2.attn_hc.hc_norm.weight"
    attn_hc_norm_b = "dflash2.attn_hc.hc_norm.bias"
    attn_hc_inject = "dflash2.attn_hc.block_inject_weight.weight"
    attn_hc_down = "dflash2.attn_hc.input_mix_weight_down.weight"
    attn_hc_up = "dflash2.attn_hc.input_mix_weight_up.weight"

    mlp_hc_norm_w = "dflash2.mlp_hc.hc_norm.weight"
    mlp_hc_norm_b = "dflash2.mlp_hc.hc_norm.bias"
    mlp_hc_inject = "dflash2.mlp_hc.block_inject_weight.weight"
    mlp_hc_down = "dflash2.mlp_hc.input_mix_weight_down.weight"
    mlp_hc_up = "dflash2.mlp_hc.input_mix_weight_up.weight"

    moe_latent_down = "dflash2.mlp.latent_down_proj"
    moe_latent_gate = "dflash2.mlp.latent_gate_proj"
    moe_latent_up = "dflash2.mlp.latent_up_proj"
    moe_latent_scale_gate = "dflash2.mlp.latent_scale_gate"
    moe_pre_latent_ln_w = "dflash2.mlp.pre_moe_latent_layernorm.weight"
    moe_gate_w = "dflash2.mlp.gate.weight"
    moe_expert_bias = "dflash2.mlp.expert_bias"
    moe_shared_gate_proj = "dflash2.mlp.shared_expert.gate_proj.weight"
    moe_shared_up_proj = "dflash2.mlp.shared_expert.up_proj.weight"
    moe_shared_down_proj = "dflash2.mlp.shared_expert.down_proj.weight"
    moe_shared_expert_gate = "dflash2.mlp.shared_expert_gate.weight"
    moe_expert_gate_w = "dflash2.mlp.experts.{e}.gate_proj.weight"
    moe_expert_gate_s = "dflash2.mlp.experts.{e}.gate_proj.weight_scale"
    moe_expert_up_w = "dflash2.mlp.experts.{e}.up_proj.weight"
    moe_expert_up_s = "dflash2.mlp.experts.{e}.up_proj.weight_scale"
    moe_expert_down_w = "dflash2.mlp.experts.{e}.down_proj.weight"
    moe_expert_down_s = "dflash2.mlp.experts.{e}.down_proj.weight_scale"


@dataclass
class DFlash2ArchSpec:
    """Architecture specification matching real DFlashQwenDraftModel (erke_config.json)."""

    num_hidden_layers: int = 5
    hidden_size: int = 5120
    vocab_size: int = 248320
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000.0
    rope_is_neox_style: bool = True
    layer_types: Tuple[str, ...] = (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )
    swa_window_size: int = 2048
    full_attention_layer_ids: Tuple[int, ...] = (4,)
    is_causal: bool = False
    use_attn_output_gate: bool = True
    qk_layernorm: bool = True
    zero_centered_block_norms: bool = True
    zero_centered_hidden_norm: bool = True
    zero_centered_hc_norm: bool = True

    num_target_layers: int = 40
    target_layer_ids: Tuple[int, ...] = (0, 5, 10, 15, 20)
    target_hidden_source: str = "post_next_hc_mix"
    target_hidden_size: int = 5120
    fc_in_features: int = 25600
    kv_inject_method: str = "fc"
    use_pre_fc_norm: bool = False

    block_size: int = 8
    gamma: int = 8
    sample_from_anchor: bool = True
    mask_token_id: int = 248280
    selector_rank: int = 256
    selector_top_k: int = 16
    input_embedding_scale: float = 1.0

    conv_kernel_size: int = 2
    conv_group_size: int = 16
    conv_base_kernel_identity_folded: bool = True

    use_hc: bool = True
    hc_count: int = 4
    hc_lowrank: int = 128
    hc_per_branch_norm: bool = True
    hc_rmsnorm_bias: bool = True
    hc_prev_mix: bool = True
    use_final_hc_mixer: bool = True
    has_final_layernorm: bool = False

    first_k_dense_replace: int = 0
    num_experts: int = 640
    num_experts_per_tok: int = 10
    routed_expert_hidden_size: int = 2560
    moe_intermediate_size: int = 2560
    shared_expert_intermediate_size: int = 1280
    hidden_act_gate_beta: float = 4.0
    hidden_act_up_beta: float = 25.0
    clip_mlp_output: float = 1000.0
    router_precision: str = "fp32_logits"
    router_normalize_topk_scores: bool = True
    moe_router_enable_expert_bias: bool = True
    shared_expert_gate: bool = True
    quant_method: str = "mxfp4"
    tie_word_embeddings: bool = False

    @property
    def query_width(self) -> int:
        return int(self.block_size) + int(not self.sample_from_anchor)

    @property
    def rotary_dim(self) -> int:
        return max(2, int(round(self.head_dim * self.partial_rotary_factor)))

    @classmethod
    def from_config_dict(cls, config_json: Dict[str, Any]) -> "DFlash2ArchSpec":
        draft = config_json.get("dflash_config") or {}
        qcfg = config_json.get("quantization_config") or {}
        zc = draft.get("zero_centered_gamma") or {}
        rope_params = config_json.get("rope_parameters") or {}

        num_layers = int(config_json.get("num_hidden_layers", 5))
        hidden_size = int(config_json.get("hidden_size", 5120))
        vocab_size = int(
            config_json.get("draft_vocab_size") or config_json.get("vocab_size", 248320)
        )
        num_heads = int(config_json.get("num_attention_heads", 32))
        num_kv_heads = int(config_json.get("num_key_value_heads", 1))
        head_dim = int(config_json.get("head_dim", hidden_size // num_heads))

        raw_layer_types = config_json.get("layer_types")
        if raw_layer_types is None:
            raw_layer_types = ["sliding_attention"] * max(0, num_layers - 1) + [
                "full_attention"
            ]
        layer_types = tuple(str(x) for x in raw_layer_types)

        full_ids_raw = draft.get("full_attention_layer_ids")
        if full_ids_raw is None:
            full_ids = tuple(
                i for i, kind in enumerate(layer_types) if kind == "full_attention"
            )
        else:
            full_ids = tuple(int(i) for i in full_ids_raw)

        target_ids = dspark_parse_target_layer_ids(config_json) or [0]
        target_hidden = int(draft.get("target_hidden_size", hidden_size))
        fc_in = int(draft.get("fc_in_features", len(target_ids) * target_hidden))

        partial_rotary = float(
            draft.get(
                "partial_rotary_factor",
                config_json.get(
                    "partial_rotary_factor",
                    rope_params.get("partial_rotary_factor", 0.25),
                ),
            )
        )
        rope_theta = float(
            config_json.get("rope_theta", rope_params.get("rope_theta", 10000.0))
        )

        return cls(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(config_json.get("rms_norm_eps", 1e-6)),
            partial_rotary_factor=partial_rotary,
            rope_theta=rope_theta,
            rope_is_neox_style=bool(draft.get("rope_is_neox_style", True)),
            layer_types=layer_types,
            swa_window_size=int(
                draft.get(
                    "swa_window_size",
                    max(1, int(config_json.get("sliding_window", 2049)) - 1),
                )
            ),
            full_attention_layer_ids=full_ids,
            is_causal=bool(draft.get("is_causal", config_json.get("is_causal", False))),
            use_attn_output_gate=bool(draft.get("use_attn_output_gate", True)),
            qk_layernorm=bool(draft.get("qk_layernorm", True)),
            zero_centered_block_norms=bool(zc.get("block_norms", True)),
            zero_centered_hidden_norm=bool(zc.get("hidden_norm", True)),
            zero_centered_hc_norm=bool(zc.get("hc_norm", True)),
            num_target_layers=int(
                draft.get("num_target_layers", config_json.get("num_target_layers", 40))
            ),
            target_layer_ids=tuple(target_ids),
            target_hidden_source=str(
                draft.get("target_hidden_source", "post_next_hc_mix")
            ),
            target_hidden_size=target_hidden,
            fc_in_features=fc_in,
            kv_inject_method=str(draft.get("kv_inject_method", "fc")),
            use_pre_fc_norm=bool(draft.get("use_pre_fc_norm", False)),
            block_size=int(
                draft.get(
                    "block_size",
                    config_json.get("block_size", config_json.get("gen_num_per_cycle", 8)),
                )
            ),
            gamma=int(
                config_json.get(
                    "gamma",
                    draft.get("block_size", config_json.get("gen_num_per_cycle", 8)),
                )
            ),
            sample_from_anchor=bool(
                draft.get(
                    "sample_from_anchor", config_json.get("sample_from_anchor", True)
                )
            ),
            mask_token_id=int(
                draft.get(
                    "mask_token_id",
                    config_json.get(
                        "dspark_noise_token_id", config_json.get("eos_token_id", 0)
                    ),
                )
            ),
            selector_rank=int(
                draft.get("selector_rank", config_json.get("markov_rank", 256))
            ),
            selector_top_k=int(draft.get("selector_top_k", 16)),
            input_embedding_scale=float(draft.get("input_embedding_scale", 1.0)),
            conv_kernel_size=int(draft.get("conv_kernel_size", 2)),
            conv_group_size=int(draft.get("conv_group_size", 16)),
            conv_base_kernel_identity_folded=bool(
                draft.get("conv_base_kernel_identity_folded", True)
            ),
            use_hc=bool(draft.get("use_hc", True)),
            hc_count=int(draft.get("hc_count", config_json.get("hc_count", 4))),
            hc_lowrank=int(draft.get("hc_lowrank", config_json.get("hc_lowrank", 128))),
            hc_per_branch_norm=bool(
                draft.get(
                    "hc_per_branch_norm", config_json.get("hc_per_branch_norm", True)
                )
            ),
            hc_rmsnorm_bias=bool(config_json.get("hc_rmsnorm_bias", True)),
            hc_prev_mix=bool(config_json.get("hc_prev_mix", True)),
            use_final_hc_mixer=bool(draft.get("use_final_hc_mixer", True)),
            has_final_layernorm=bool(draft.get("has_final_layernorm", False)),
            first_k_dense_replace=int(config_json.get("first_k_dense_replace", 0)),
            num_experts=int(config_json.get("num_experts", 640)),
            num_experts_per_tok=int(config_json.get("num_experts_per_tok", 10)),
            routed_expert_hidden_size=int(
                config_json.get("routed_expert_hidden_size", 2560)
            ),
            moe_intermediate_size=int(config_json.get("moe_intermediate_size", 2560)),
            shared_expert_intermediate_size=int(
                config_json.get("shared_expert_intermediate_size", 1280)
            ),
            hidden_act_gate_beta=float(config_json.get("hidden_act_gate_beta", 4.0)),
            hidden_act_up_beta=float(config_json.get("hidden_act_up_beta", 25.0)),
            clip_mlp_output=float(config_json.get("clip_mlp_output", 1000.0)),
            router_precision=str(config_json.get("router_precision", "fp32_logits")),
            router_normalize_topk_scores=bool(
                draft.get(
                    "moe_norm_topk_scores",
                    config_json.get("router_normalize_topk_scores", True),
                )
            ),
            moe_router_enable_expert_bias=bool(
                config_json.get("moe_router_enable_expert_bias", True)
            ),
            shared_expert_gate=bool(
                config_json.get(
                    "shared_expert_gate", config_json.get("use_shared_expert_gate", True)
                )
            ),
            quant_method=str(qcfg.get("quant_method", "mxfp4")),
            tie_word_embeddings=bool(config_json.get("tie_word_embeddings", False)),
        )


def validate_dflash2_config_dict(config_json: Dict[str, Any]) -> DFlash2ArchSpec:
    """Validate DFlash2 config invariants matching erke_config.json."""
    spec = DFlash2ArchSpec.from_config_dict(config_json)
    if spec.is_causal:
        raise ValueError("DFlash2 draft attention must be non-causal (is_causal=False)")
    if len(spec.layer_types) != spec.num_hidden_layers:
        raise ValueError(
            f"layer_types length {len(spec.layer_types)} != num_hidden_layers {spec.num_hidden_layers}"
        )
    for lt in spec.layer_types:
        if lt not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported DFlash2 layer type: {lt!r}")
    expected_full = tuple(
        i for i, lt in enumerate(spec.layer_types) if lt == "full_attention"
    )
    if expected_full != spec.full_attention_layer_ids:
        raise ValueError(
            f"full_attention_layer_ids {spec.full_attention_layer_ids} != {expected_full}"
        )
    if len(spec.target_layer_ids) * spec.target_hidden_size != spec.fc_in_features:
        raise ValueError(
            f"fc_in_features {spec.fc_in_features} != len({spec.target_layer_ids}) * {spec.target_hidden_size}"
        )
    if spec.block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {spec.block_size}")
    if spec.selector_rank <= 0 or spec.selector_top_k <= 0:
        raise ValueError("selector_rank and selector_top_k must be positive")
    if spec.hidden_size % spec.conv_group_size != 0:
        raise ValueError(
            f"hidden_size={spec.hidden_size} must be divisible by conv_group_size={spec.conv_group_size}"
        )
    if spec.conv_kernel_size <= 0 or spec.conv_kernel_size > spec.query_width:
        raise ValueError(
            f"conv_kernel_size={spec.conv_kernel_size} must be in [1, query_width={spec.query_width}]"
        )
    return spec


def dspark_offset_d2t_to_absolute(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Convert draft-to-target relative token offsets to absolute int32 target IDs."""
    d2t = tensors[0]
    if d2t.dim() != 1:
        raise ValueError(f"Expected 1D d2t offset tensor, got shape {tuple(d2t.shape)}")
    base = torch.arange(d2t.numel(), dtype=torch.int32, device=d2t.device)
    return (base + d2t.to(torch.int32)).contiguous()


def dspark_parse_target_layer_ids(config_json: Dict[str, Any]) -> Optional[List[int]]:
    """Parse DFlash2 / DSpARK target hidden layer IDs into sorted 0-based indices."""
    dflash_cfg = config_json.get("dflash_config") or {}
    explicit = (
        dflash_cfg.get("target_layer_ids")
        if dflash_cfg.get("target_layer_ids") is not None
        else config_json.get("dspark_target_layer_ids")
    )
    if explicit is None:
        explicit = config_json.get("target_layer_ids")
    if explicit is not None:
        ids = sorted({int(x) for x in explicit})
        if any(x < 0 for x in ids):
            raise ValueError(f"Negative 0-based target layer index in {ids}")
        return ids

    aux_ids = config_json.get("aux_hidden_state_layer_ids")
    if aux_ids is None:
        return None
    raw = [int(x) for x in aux_ids]
    if not raw:
        return []
    is_zero_based = bool(
        config_json.get("aux_hidden_state_layer_ids_zero_based", False)
    ) or (0 in raw)
    converted = raw if is_zero_based else [x - 1 for x in raw]
    ids = sorted(set(converted))
    if any(x < 0 for x in ids):
        raise ValueError(f"Invalid target layer ids {raw} -> {ids}")
    return ids


def _parse_dtype(dtype_name: Any) -> Tuple[str, DataType, torch.dtype, str]:
    text = str(dtype_name or "bfloat16").lower().strip()
    if text in ("bf16", "bfloat16", "torch.bfloat16"):
        return "bf16", DataType.TYPE_BF16, torch.bfloat16, "bfloat16"
    if text in ("fp16", "float16", "half", "torch.float16"):
        return "fp16", DataType.TYPE_FP16, torch.float16, "float16"
    if text in ("fp32", "float32", "float", "torch.float32"):
        return "fp32", DataType.TYPE_FP32, torch.float32, "float32"
    raise ValueError(f"Unsupported torch_dtype for Qwen4_5DSpark: {dtype_name!r}")


def _zero_tensor(
    _ts: List[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype).contiguous()


class Qwen4_5DSparkWeight(ModelDeployWeightInfo):
    """Weight info mapping DFlashQwenDraftModel HF tensors onto SP_TYPE_DSPARK."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: Optional[ParallelismConfig] = None,
        hw_kernel_config: Optional[HWKernelConfig] = None,
        kv_cache_config: Optional[KVCacheConfig] = None,
        merge_lora: bool = False,
        moe_strategy: str = "auto",
        **kwargs: Any,
    ):
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
        if hw_kernel_config is None:
            hw_kernel_config = HWKernelConfig()
        if kv_cache_config is None:
            kv_cache_config = KVCacheConfig()
        super().__init__(
            model_config,
            parallelism_config,
            hw_kernel_config,
            kv_cache_config,
            merge_lora=merge_lora,
            moe_strategy=moe_strategy,
            **kwargs,
        )
        self._ckpt_keys: Optional[set[str]] = None

    def _process_meta(self, meta_dict: Any, weight_keys: Iterable[str]) -> None:
        del meta_dict
        self._ckpt_keys = set(weight_keys) if weight_keys is not None else None

    def _has_key(self, key: str) -> Optional[bool]:
        if self._ckpt_keys is None:
            return None
        return key in self._ckpt_keys

    def _pick_key(self, *candidates: str) -> str:
        if self._ckpt_keys is not None:
            for candidate in candidates:
                if candidate in self._ckpt_keys:
                    return candidate
        return candidates[0]

    def _layer_prefix(self, layer_id: int) -> str:
        if self._ckpt_keys is not None and any(
            k.startswith(f"model.layers.{layer_id}.") for k in self._ckpt_keys
        ):
            return f"model.layers.{layer_id}."
        if self._ckpt_keys is not None and any(
            k.startswith(f"layers.{layer_id}.") for k in self._ckpt_keys
        ):
            return f"layers.{layer_id}."
        return f"layers.{layer_id}."

    def _optional_zero(
        self,
        name: str,
        candidates: Sequence[str],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> AtomicWeight:
        chosen = self._pick_key(*candidates)
        if self._has_key(chosen) is False:
            return AtomicWeight(
                name, [], functools.partial(_zero_tensor, shape=shape, dtype=dtype)
            )
        if self._has_key(chosen) is True:
            return AtomicWeight(name, [CkptWeightInfo(chosen, identity)], identity)
        return AtomicWeight(
            name,
            [CkptWeightInfo(chosen, functools.partial(tolerate_failed, origin_func=identity))],
            functools.partial(
                choose_available,
                origin_func_list=[
                    identity,
                    functools.partial(_zero_tensor, shape=shape, dtype=dtype),
                ],
            ),
        )

    def _spec(self) -> DFlash2ArchSpec:
        spec = getattr(self.model_config, "_dflash2_spec", None)
        if isinstance(spec, DFlash2ArchSpec):
            return spec
        if self.model_config.ckpt_path:
            cfg_path = os.path.join(self.model_config.ckpt_path, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return DFlash2ArchSpec.from_config_dict(json.load(f))
        return DFlash2ArchSpec(
            num_hidden_layers=int(self.model_config.num_layers),
            hidden_size=int(self.model_config.hidden_size),
            vocab_size=int(self.model_config.vocab_size),
            num_attention_heads=int(self.model_config.attn_config.head_num),
            num_key_value_heads=int(self.model_config.attn_config.kv_head_num),
            head_dim=int(self.model_config.attn_config.size_per_head),
            selector_rank=int(
                getattr(self.model_config, "dspark_markov_rank", 256) or 256
            ),
        )

    def _get_hf_layer_weight_info(
        self,
        layer_id: int,
        attn_config: AttnConfig,
        spec: DFlash2ArchSpec,
    ) -> List[WeightModule]:
        p = self._layer_prefix(layer_id)
        sa = f"{p}self_attn."
        layer_weights: List[WeightModule] = [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(f"{sa}q_proj.weight", identity),
                    CkptWeightInfo(f"{sa}k_proj.weight", identity),
                    CkptWeightInfo(f"{sa}v_proj.weight", identity),
                ],
                merge_qkv_hf,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(f"{sa}o_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [CkptWeightInfo(f"{sa}q_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [CkptWeightInfo(f"{sa}k_norm.weight", identity)],
                identity,
            ),
        ]

        if spec.use_attn_output_gate:
            layer_weights.append(
                CustomAtomicWeight(
                    DFLASH2_W.attn_output_gate,
                    [CkptWeightInfo(f"{sa}output_gate_proj.weight", identity)],
                    identity,
                    split_func=sp_0,
                )
            )

        for conv_prefix, base_name, proj_name in (
            ("attention_conv", DFLASH2_W.attn_conv_base, DFLASH2_W.attn_conv_proj),
            ("mlp_conv", DFLASH2_W.mlp_conv_base, DFLASH2_W.mlp_conv_proj),
        ):
            layer_weights.extend(
                [
                    CustomAtomicWeight(
                        base_name,
                        [CkptWeightInfo(f"{p}{conv_prefix}.base_kernel", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        proj_name,
                        [
                            CkptWeightInfo(
                                f"{p}{conv_prefix}.kernel_projection.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                ]
            )

        if spec.use_hc:
            hc_p_attn = (
                f"{p}attn_hyper_connection."
                if self._has_key(f"{p}attn_hyper_connection.hc_norm.weight") is True
                else f"{p}attn_hc."
            )
            hc_p_mlp = (
                f"{p}mlp_hyper_connection."
                if self._has_key(f"{p}mlp_hyper_connection.hc_norm.weight") is True
                else f"{p}mlp_hc."
            )
            layer_weights.extend(
                [
                    CustomAtomicWeight(
                        DFLASH2_W.attn_hc_norm_w,
                        [CkptWeightInfo(f"{hc_p_attn}hc_norm.weight", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.attn_hc_norm_b,
                        [CkptWeightInfo(f"{hc_p_attn}hc_norm.bias", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.attn_hc_inject,
                        [
                            CkptWeightInfo(
                                f"{hc_p_attn}block_inject_weight.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.attn_hc_down,
                        [
                            CkptWeightInfo(
                                f"{hc_p_attn}input_mix_weight_down.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.attn_hc_up,
                        [
                            CkptWeightInfo(
                                f"{hc_p_attn}input_mix_weight_up.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.mlp_hc_norm_w,
                        [CkptWeightInfo(f"{hc_p_mlp}hc_norm.weight", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.mlp_hc_norm_b,
                        [CkptWeightInfo(f"{hc_p_mlp}hc_norm.bias", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.mlp_hc_inject,
                        [
                            CkptWeightInfo(
                                f"{hc_p_mlp}block_inject_weight.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.mlp_hc_up,
                        [
                            CkptWeightInfo(
                                f"{hc_p_mlp}input_mix_weight_up.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                ]
            )
            if not spec.hc_prev_mix or layer_id + 1 < self._num_layers:
                layer_weights.append(
                    CustomAtomicWeight(
                        DFLASH2_W.mlp_hc_down,
                        [
                            CkptWeightInfo(
                                f"{hc_p_mlp}input_mix_weight_down.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    )
                )

        mlp = f"{p}mlp."
        for wname, raw_name in (
            (DFLASH2_W.moe_latent_down, f"{mlp}latent_down_proj"),
            (DFLASH2_W.moe_latent_gate, f"{mlp}latent_gate_proj"),
            (DFLASH2_W.moe_latent_up, f"{mlp}latent_up_proj"),
            (DFLASH2_W.moe_latent_scale_gate, f"{mlp}latent_scale_gate"),
        ):
            chosen = self._pick_key(f"{raw_name}.weight", raw_name)
            layer_weights.append(
                CustomAtomicWeight(wname, [CkptWeightInfo(chosen, identity)], identity, split_func=sp_id)
            )

        layer_weights.extend(
            [
                CustomAtomicWeight(
                    DFLASH2_W.moe_pre_latent_ln_w,
                    [CkptWeightInfo(f"{mlp}pre_moe_latent_layernorm.weight", identity)],
                    identity,
                split_func=sp_id,
                ),
                CustomAtomicWeight(
                    DFLASH2_W.moe_gate_w,
                    [CkptWeightInfo(f"{mlp}gate.weight", identity)],
                    identity,
                split_func=sp_id,
                ),
                CustomAtomicWeight(
                    DFLASH2_W.moe_shared_gate_proj,
                    [CkptWeightInfo(f"{mlp}shared_expert.gate_proj.weight", identity)],
                    identity,
                split_func=sp_id,
                ),
                CustomAtomicWeight(
                    DFLASH2_W.moe_shared_up_proj,
                    [CkptWeightInfo(f"{mlp}shared_expert.up_proj.weight", identity)],
                    identity,
                split_func=sp_id,
                ),
                CustomAtomicWeight(
                    DFLASH2_W.moe_shared_down_proj,
                    [CkptWeightInfo(f"{mlp}shared_expert.down_proj.weight", identity)],
                    identity,
                split_func=sp_id,
                ),
            ]
        )
        if spec.moe_router_enable_expert_bias:
            layer_weights.append(
                CustomAtomicWeight(
                    DFLASH2_W.moe_expert_bias,
                    [CkptWeightInfo(f"{mlp}expert_bias", identity)],
                    identity,
                    data_type=torch.float32,
                split_func=sp_id,
                )
            )
        if spec.shared_expert_gate:
            layer_weights.append(
                CustomAtomicWeight(
                    DFLASH2_W.moe_shared_expert_gate,
                    [CkptWeightInfo(f"{mlp}shared_expert_gate.weight", identity)],
                    identity,
                split_func=sp_id,
                )
            )
        ep_size, ep_rank = self._effective_ep()
        if spec.num_experts % ep_size != 0:
            raise ValueError(
                f"DFlash2 num_experts={spec.num_experts} must be divisible by ep_size={ep_size}"
            )
        local_experts = spec.num_experts // ep_size
        ep_start = ep_rank * local_experts
        ep_end = ep_start + local_experts
        for e in range(ep_start, ep_end):
            ep = f"{mlp}experts.{e}."
            for wfmt, ckpt_suffix in (
                (DFLASH2_W.moe_expert_gate_w, "gate_proj.weight"),
                (DFLASH2_W.moe_expert_gate_s, "gate_proj.weight_scale"),
                (DFLASH2_W.moe_expert_up_w, "up_proj.weight"),
                (DFLASH2_W.moe_expert_up_s, "up_proj.weight_scale"),
                (DFLASH2_W.moe_expert_down_w, "down_proj.weight"),
                (DFLASH2_W.moe_expert_down_s, "down_proj.weight_scale"),
            ):
                layer_weights.append(
                    CustomAtomicWeight(
                        wfmt.format(e=e),
                        [CkptWeightInfo(f"{ep}{ckpt_suffix}", identity)],
                        identity,
                        data_type=torch.uint8,
                    split_func=sp_id,
                    )
                )

        return layer_weights

    def _get_weight_info(self) -> ModelWeightInfo:
        spec = self._spec()
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

        fc_key = self._pick_key("fc.weight", "model.fc.weight")
        hidden_norm_key = self._pick_key("hidden_norm.weight", "model.hidden_norm.weight")
        embed_key = self._pick_key("model.embed_tokens.weight", "embed_tokens.weight")
        lm_head_key = self._pick_key("lm_head.weight", "model.lm_head.weight")
        if self._has_key(lm_head_key) is False and (
            getattr(self.model_config, "tie_word_embeddings", False)
            or self._has_key(embed_key) is True
        ):
            lm_head_key = embed_key

        target_vocab = int(
            self.model_config.input_vocab_size or self.model_config.vocab_size
        )
        draft_vocab = int(self.model_config.vocab_size)
        markov_rank = int(
            getattr(self.model_config, "dspark_markov_rank", spec.selector_rank)
            or spec.selector_rank
        )
        weight_dtype = self.model_config.attn_config.dtype
        if not isinstance(weight_dtype, torch.dtype):
            weight_dtype = torch.bfloat16

        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(embed_key, identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(lm_head_key, identity)],
                identity,
                # Keep BF16 checkpoint dtype unless enable_fp32_lm_head is explicitly requested,
                # so PyWrappedModel::computeLogits dispatches BF16 TensorCore GEMM instead of FP32 sgemm.
                data_type=torch.float32 if getattr(self, "enable_fp32_lm_head", False) else None,
            ),
            AtomicWeight(
                W.dspark_fc_w,
                [CkptWeightInfo(fc_key, identity)],
                transpose,
            ),
            AtomicWeight(
                W.dspark_hidden_norm_gamma,
                [CkptWeightInfo(hidden_norm_key, identity)],
                identity,
            ),
            self._optional_zero(
                W.dspark_markov_w1,
                (
                    "candidate_selector.predecessor_codebook",
                    "model.candidate_selector.predecessor_codebook",
                    "markov_head.markov_w1.weight",
                ),
                shape=(target_vocab, markov_rank),
                dtype=weight_dtype,
            ),
            self._optional_zero(
                W.dspark_markov_w2,
                (
                    "candidate_selector.successor_codebook",
                    "model.candidate_selector.successor_codebook",
                    "markov_head.markov_w2.weight",
                ),
                shape=(draft_vocab, markov_rank),
                dtype=weight_dtype,
            ),
            CustomAtomicWeight(
                DFLASH2_W.selector_hidden_proj,
                [
                    CkptWeightInfo(
                        self._pick_key(
                            "candidate_selector.hidden_projection.weight",
                            "model.candidate_selector.hidden_projection.weight",
                        ),
                        identity,
                    )
                ],
                identity,
            split_func=sp_id,
            ),
        ]

        if spec.has_final_layernorm:
            norm_key = self._pick_key("model.norm.weight", "norm.weight")
            weights.append(
                AtomicWeight(
                    W.final_ln_gamma,
                    [CkptWeightInfo(norm_key, identity)],
                    identity,
                )
            )
        else:
            weights.append(
                AtomicWeight(
                    W.final_ln_gamma,
                    [],
                    functools.partial(
                        _zero_tensor,
                        shape=(int(self._hidden_size),),
                        dtype=weight_dtype,
                    ),
                )
            )

        if spec.use_final_hc_mixer:
            mixer_p = (
                "hyper_connection_mixer."
                if self._has_key("hyper_connection_mixer.hc_norm.weight") is True
                else "hc_mixer."
            )
            weights.extend(
                [
                    CustomAtomicWeight(
                        DFLASH2_W.hc_mixer_norm_w,
                        [CkptWeightInfo(f"{mixer_p}hc_norm.weight", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.hc_mixer_norm_b,
                        [CkptWeightInfo(f"{mixer_p}hc_norm.bias", identity)],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.hc_mixer_down,
                        [
                            CkptWeightInfo(
                                f"{mixer_p}input_mix_weight_down.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                    CustomAtomicWeight(
                        DFLASH2_W.hc_mixer_up,
                        [
                            CkptWeightInfo(
                                f"{mixer_p}input_mix_weight_up.weight", identity
                            )
                        ],
                        identity,
                    split_func=sp_id,
                    ),
                ]
            )

        d2t_key = self._pick_key("d2t", "model.d2t")
        if self._has_key(d2t_key) is True:
            weights.append(
                AtomicWeight(
                    W.multi_tokens_predict_d2t_map,
                    [CkptWeightInfo(d2t_key, identity)],
                    dspark_offset_d2t_to_absolute,
                    data_type=torch.int32,
                )
            )

        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id, attn_config, spec)
            for layer_id in range(self._num_layers)
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _effective_ep(self) -> Tuple[int, int]:
        ep_size = int(getattr(self, "ep_size", 1) or 1)
        ep_rank = int(getattr(self, "ep_rank", 0) or 0)
        tp_size = int(getattr(self, "tp_size", 1) or 1)
        tp_rank = int(getattr(self, "tp_rank", 0) or 0)
        dp_size = int(getattr(self, "dp_size", 1) or 1)
        dp_rank = int(getattr(self, "dp_rank", 0) or 0)
        if ep_size <= 1 and dp_size > 1:
            eff_ep = dp_size * tp_size
            return eff_ep, dp_rank * tp_size + tp_rank
        if ep_size <= 1 and tp_size > 1 and dp_size == 1:
            return tp_size, tp_rank
        return max(1, ep_size), max(0, ep_rank)

    def get_weight_info(
        self, database: Optional[Any] = None
    ) -> ModelWeightInfo:
        # Qwen4_5DSparkWeight explicitly declares all MXFP4 expert weight/scale
        # descriptors and keeps attention weights (including separate
        # output_gate_proj) unquantized. Bypass generic to_quant_weight_info(),
        # which wraps W.attn_qkv_w in MixedFp4Weight (split_q_gate) for Qwen3.
        saved_quant_algo = self._quant_algo
        self._quant_algo = None
        try:
            parent_fn = super().get_weight_info
            if len(inspect.signature(parent_fn).parameters) == 0:
                return parent_fn()
            return parent_fn(database)
        finally:
            self._quant_algo = saved_quant_algo


def _parse_dflash2_mxfp4_group_size(config_json: Dict[str, Any]) -> int:
    raw_qcfg = config_json.get("quantization_config")
    if isinstance(raw_qcfg, dict):
        group_0 = (raw_qcfg.get("config_groups") or {}).get("group_0") or {}
        weights_cfg = group_0.get("weights") if isinstance(group_0, dict) else None
        if isinstance(weights_cfg, dict) and weights_cfg.get("group_size"):
            return int(weights_cfg["group_size"])
        if raw_qcfg.get("group_size"):
            return int(raw_qcfg["group_size"])
    return int(MXFP4_GROUP_SIZE)


class _Qwen4_5DSparkModelConfig(ModelConfig):
    """ModelConfig subclass handling MXFP4 quantized DFlash2 draft checkpoints."""

    _dflash2_spec: Optional[Any] = None
    _dflash2_quant_method: Optional[str] = None
    _dflash2_mxfp4_group_size: int = MXFP4_GROUP_SIZE

    def init_precision_config(
        self, kv_cache_config: Optional[Any], act_type: Optional[str]
    ) -> None:
        if self._dflash2_quant_method == "mxfp4":
            from rtp_llm.config.model_config import apply_quant_scope_to_pricing
            from rtp_llm.config.quant_config import MXFp4QuarkQuantConfig
            from rtp_llm.utils.weight_type import WEIGHT_TYPE

            gsize = int(self._dflash2_mxfp4_group_size or MXFP4_GROUP_SIZE)
            quant_config = MXFp4QuarkQuantConfig(
                bits=4,
                group_size=gsize,
                is_quanted=True,
            )
            self.quant_algo.setQuantAlgo("mxfp4-quark", 4, gsize)
            apply_quant_scope_to_pricing(self, quant_config)
            self.expert_weight_dtype = "fp4"
            self.dense_weight_dtype = "bf16"

            if kv_cache_config is not None and getattr(
                kv_cache_config, "fp8_kv_cache", False
            ):
                self.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            else:
                self.attn_config.kv_cache_dtype = KvCacheDataType.BASE

            dt = (
                WEIGHT_TYPE.from_str(act_type)
                if act_type
                else (
                    WEIGHT_TYPE.from_str(self.config_dtype)
                    if self.config_dtype
                    else WEIGHT_TYPE.BF16
                )
            )
            if dt == WEIGHT_TYPE.FP32:
                dt = WEIGHT_TYPE.BF16
            self.data_type = dt.to_str()
            self.quant_config = quant_config
            return

        super().init_precision_config(kv_cache_config, act_type)


class Qwen4_5DSpark(BaseModel):
    """5-layer hybrid non-causal DFlashQwenDraftModel for Qwen4.5-Flash."""

    @staticmethod
    def get_weight_cls():
        return Qwen4_5DSparkWeight

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)
        return cls._create_config_from_json(ckpt_path, config_json)

    @classmethod
    def _create_config_from_json(
        cls, ckpt_path: str, config_json: Dict[str, Any]
    ) -> ModelConfig:
        spec = validate_dflash2_config_dict(config_json)
        data_type_str, _, torch_dtype, dtype_str = _parse_dtype(
            config_json.get("torch_dtype", config_json.get("dtype", "bfloat16"))
        )
        target_vocab_size = int(
            config_json.get("target_vocab_size")
            or config_json.get("input_vocab_size")
            or config_json.get("vocab_size", spec.vocab_size)
        )
        max_seq_len = int(config_json.get("max_position_embeddings", 8192))
        tokens_per_block = int(config_json.get("tokens_per_block", 16) or 16)
        kernel_tokens_per_block = int(
            config_json.get("kernel_tokens_per_block", tokens_per_block)
            or tokens_per_block
        )

        config = _Qwen4_5DSparkModelConfig()
        config.ckpt_path = ckpt_path
        config.model_type = "qwen4_5_dspark"
        config.num_layers = spec.num_hidden_layers
        config.hidden_size = spec.hidden_size
        config.inter_size = int(
            config_json.get("intermediate_size", spec.shared_expert_intermediate_size)
        )
        config.dense_inter_size = config.inter_size
        config.moe_inter_size = spec.moe_intermediate_size
        config.vocab_size = spec.vocab_size
        config.input_vocab_size = target_vocab_size
        config.max_seq_len = max_seq_len
        config.layernorm_eps = spec.rms_norm_eps
        config.partial_rotary_factor = spec.partial_rotary_factor
        config.qk_norm = spec.qk_layernorm
        config.norm_type = "rmsnorm"
        config.layernorm_type = "pre_layernorm"
        config.activation_type = "SiGLU"
        config.has_positional_encoding = False
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = spec.has_final_layernorm
        config.has_lm_head = True
        config.tie_word_embeddings = spec.tie_word_embeddings
        # Keep LM head in checkpoint BF16 dtype instead of framework default FP32 upcast.
        config.enable_fp32_lm_head = False
        config.data_type = data_type_str
        config.config_dtype = dtype_str
        config.gen_num_per_cycle = spec.block_size
        config.expert_num = spec.num_experts
        config.moe_k = spec.num_experts_per_tok
        first_k_dense = int(config_json.get("first_k_dense_replace", 0) or 0)
        config.moe_layer_index = list(range(first_k_dense, spec.num_hidden_layers))
        config.hc_mult = spec.hc_count
        config.hc_eps = spec.rms_norm_eps
        raw_qcfg = config_json.get("quantization_config")
        if raw_qcfg is None or (
            isinstance(raw_qcfg, dict)
            and raw_qcfg.get("quant_method") in ("mxfp4", None)
        ):
            qcfg = parse_qwen4_5_quant_config(config_json)
            config._dflash2_quant_method = qcfg.quant_method
        else:
            config._dflash2_quant_method = (
                raw_qcfg.get("quant_method") if isinstance(raw_qcfg, dict) else None
            )
        config._dflash2_mxfp4_group_size = _parse_dflash2_mxfp4_group_size(
            config_json
        )
        if hasattr(config, "hybrid_attention_config"):
            config.hybrid_attention_config.enable_hybrid_attention = False
            config.hybrid_attention_config.hybrid_attention_types = []
        if hasattr(config, "linear_attention_config"):
            config.linear_attention_config.linear_num_value_heads = 0

        config.attn_config.head_num = spec.num_attention_heads
        config.attn_config.kv_head_num = spec.num_key_value_heads
        config.attn_config.size_per_head = spec.head_dim
        config.attn_config.rope_head_dim = spec.rotary_dim
        config.attn_config.is_causal = False
        config.attn_config.need_rope_kv_cache = True
        config.attn_config.use_mla = False
        config.attn_config.sliding_window = spec.swa_window_size
        config.attn_config.tokens_per_block = tokens_per_block
        config.attn_config.kernel_tokens_per_block = kernel_tokens_per_block
        config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        config.attn_config.dtype = torch_dtype
        config.attn_config.max_seq_len = max_seq_len
        config.attn_config.gen_num_per_cycle = spec.block_size
        config.attn_config.rope_config.style = RopeStyle.Base
        config.attn_config.rope_config.dim = spec.rotary_dim
        config.attn_config.rope_config.base = int(round(spec.rope_theta))

        config.dspark_noise_token_id = spec.mask_token_id
        config.dspark_target_layer_ids = list(spec.target_layer_ids)
        config.dspark_markov_rank = spec.selector_rank
        config.dspark_sample_from_anchor = spec.sample_from_anchor
        config._dflash2_spec = spec
        cls._post_build_model_config(config)
        return config

    @classmethod
    def _apply_kv_cache_config(
        cls, model_config: ModelConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        if kv_cache_config is not None:
            seq_size = int(getattr(kv_cache_config, "seq_size_per_block", 0) or 0)
            if seq_size > 0:
                model_config.attn_config.tokens_per_block = seq_size
                model_config.attn_config.kernel_tokens_per_block = seq_size
            kernel_seq_size = int(
                getattr(kv_cache_config, "kernel_seq_size_per_block", 0) or 0
            )
            if kernel_seq_size > 0:
                model_config.attn_config.kernel_tokens_per_block = kernel_seq_size
        cls._post_build_model_config(model_config)

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        model_config.attn_config.is_causal = False
        desc = KVCacheSpecDesc()
        desc.tag = FULL_ATTN_KV_TAG
        desc.cache_type = KVCacheSpecType.MHA
        desc.tp_transfer_mode = TpCacheTransferMode.MHA_HEADS
        num_layers = max(1, int(getattr(model_config, "num_layers", 1) or 1))
        model_config.kv_cache_spec_descs = [[desc] for _ in range(num_layers)]

    @classmethod
    def _checkpoint_weight_keys(cls, ckpt_path: str) -> Optional[set[str]]:
        if not ckpt_path:
            return None
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                wmap = json.load(f).get("weight_map") or {}
            return set(wmap.keys())
        single_path = os.path.join(ckpt_path, "model.safetensors")
        if os.path.isfile(single_path):
            from safetensors import safe_open

            with safe_open(single_path, framework="pt", device="cpu") as handle:
                return set(handle.keys())
        return None

    @classmethod
    def speculative_weight_alias_names(
        cls, target_model: "BaseModel", draft_model_config: ModelConfig
    ) -> Tuple[str, ...]:
        """Allow borrowing target embed_tokens/lm_head when omitted on disk."""
        target_cfg = target_model.model_config
        ckpt_path = draft_model_config.ckpt_path or ""
        keys = cls._checkpoint_weight_keys(ckpt_path)
        if (
            target_cfg.vocab_size == draft_model_config.vocab_size
            and target_cfg.hidden_size == draft_model_config.hidden_size
            and keys is not None
        ):
            has_embed = (
                "model.embed_tokens.weight" in keys or "embed_tokens.weight" in keys
            )
            has_lm = "lm_head.weight" in keys or "model.lm_head.weight" in keys
            aliases: List[str] = []
            if not has_embed:
                aliases.append(W.embedding)
            if not has_lm and (
                not has_embed
                or not getattr(draft_model_config, "tie_word_embeddings", False)
            ):
                aliases.append(W.lm_head)
            return tuple(aliases)
        return ()

    def support_cuda_graph(self) -> bool:
        return True

    def support_pp(self) -> bool:
        return False

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import Qwen4_5DSparkModel

        kv_dtype_str = (
            "fp8_e4m3"
            if getattr(self.model_config.attn_config, "kv_cache_dtype", None) == KvCacheDataType.FP8
            else "bf16"
        )
        self.py_model = Qwen4_5DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=getattr(self, "hw_kernel_config", None),
            dflash_kv_dtype=kv_dtype_str,
            dense_linear_mode=getattr(self.model_config, "dense_linear_mode", None),
        )


register_model(
    "qwen4_5_dspark",
    Qwen4_5DSpark,
    ["DFlashQwenDraftModel", "Qwen4_5DSparkForCausalLM"],
)
