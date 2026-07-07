"""Qwen3-MoE for the new loader.

Supports BF16 / FP8 / FP4 / W4A8 ckpts with TP and EP parallelism. Reuses
dense Qwen3's attention block and the loader-agnostic FusedMoeFactory.
Per-expert ckpt streaming is handled inside Qwen3Experts.load_weights, which
stacks the streamed `experts.{i}.{gate|up|down}_proj.weight` tensors into
the [E, ...] W.moe_w1 / W.moe_w2 layout that the existing FusedMoe
executors expect.

EP support: when ep_size > 1, only local experts are allocated and loaded.
The NewModelLoader._apply_ep_filter pre-filters the weight stream so only
weights for experts in [start_expert, end_expert) arrive. BaseMoEExperts
remaps global expert IDs to local IDs internally.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import FusedMoeFactory, SelectTopk
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.new_models.qwen3.language import Qwen3Attention
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3Experts(BaseMoEExperts):
    """Qwen3-MoE expert module with FP4/W4A8 quantization support.

    Inherits EP/TP/buffer/id-remapping and FP8 quantization (per-tensor,
    per-channel, per-block) from BaseMoEExperts.
    Adds FP4 (modelopt) and W4A8 quantization support.
    """

    _EXTRA_QUANT_MAP = {
        "modelopt_fp4": "fp4",
        "FP4": "fp4",
        "W4A8_INT4_PER_CHANNEL": "w4a8",
        "W4A8_INT4_PER_CHANNEL_COMPRESSED": "w4a8",
    }

    # ------------------------------------------------------------------ #
    #  Buffer allocation — extend base for FP4/W4A8
    # ------------------------------------------------------------------ #

    def _init_buffers(self, params_dtype: torch.dtype):
        if self.quant_method is not None:
            super()._init_buffers(params_dtype)
            return
        E = self.num_local_experts
        M_tp = self.moe_inter_tp
        H = self.hidden_size
        qf = self._quant_family

        if qf == "fp4":
            gs = getattr(self._model_config, "group_size", 16)
            self._group_size = gs
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            self.register_buffer(
                "w13_scale",
                torch.zeros(E, 2 * M_tp, H // gs, dtype=torch.float8_e4m3fn),
            )
            self.register_buffer(
                "w2_scale",
                torch.zeros(E, H, M_tp // gs, dtype=torch.float8_e4m3fn),
            )
            self.register_buffer("w13_scale_2", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("w2_scale_2", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("w13_input_scale", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("w2_input_scale", torch.zeros(E, dtype=torch.float32))
            self._fp4_gate_s2: list = [0.0] * E
            self._fp4_up_s2: list = [0.0] * E
            self._fp4_gate_i_s: list = [0.0] * E
            self._fp4_up_i_s: list = [0.0] * E

        elif qf == "w4a8":
            self._group_size = getattr(self._model_config, "group_size", 128)
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=params_dtype),
                requires_grad=False,
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=params_dtype),
                requires_grad=False,
            )

        else:
            # FP8 per-tensor/per-channel/per-block and unquantized handled by base.
            super()._init_buffers(params_dtype)

    # ------------------------------------------------------------------ #
    #  Scale dispatch — extend base for FP4
    # ------------------------------------------------------------------ #

    def _dispatch_scale(
        self, local_id: int, proj: str, param_name: str, tensor: torch.Tensor
    ):
        qf = self._quant_family

        if qf in ("fp8_per_tensor", "fp8_per_channel", "fp8_per_block"):
            super()._dispatch_scale(local_id, proj, param_name, tensor)
            return

        if param_name == "weight_scale" and qf == "fp4":
            self._copy_fp4_weight_scale(local_id, proj, tensor)

        elif param_name == "weight_scale_2" and qf == "fp4":
            val = tensor.float().squeeze().item()
            if proj == "gate_proj":
                self._fp4_gate_s2[local_id] = val
            elif proj == "up_proj":
                self._fp4_up_s2[local_id] = val
            elif proj == "down_proj":
                self.w2_scale_2[local_id] = val

        elif param_name == "input_scale" and qf == "fp4":
            val = tensor.float().squeeze().item()
            if proj == "gate_proj":
                self._fp4_gate_i_s[local_id] = val
            elif proj == "up_proj":
                self._fp4_up_i_s[local_id] = val
            elif proj == "down_proj":
                self.w2_input_scale[local_id] = val

    # ------------------------------------------------------------------ #
    #  FP4 scale copy helper
    # ------------------------------------------------------------------ #

    def _copy_fp4_weight_scale(self, expert_id: int, proj: str, tensor: torch.Tensor):
        if proj in ("gate_proj", "up_proj"):
            start = self.tp_rank * self.moe_inter_tp
            sliced = tensor.narrow(0, start, self.moe_inter_tp).contiguous()
            row_start = self.moe_inter_tp if proj == "gate_proj" else 0
            self.w13_scale.data[
                expert_id, row_start : row_start + self.moe_inter_tp
            ].copy_(sliced)
        elif proj == "down_proj":
            tp_cols = self.w2_scale.shape[2]
            start = self.tp_rank * tp_cols
            sliced = tensor.narrow(1, start, tp_cols).contiguous()
            self.w2_scale.data[expert_id].copy_(sliced)

    # ------------------------------------------------------------------ #
    #  Post-load processing — extend base for FP4/W4A8
    # ------------------------------------------------------------------ #

    def process_weights_after_loading(self):
        if self.quant_method is not None:
            super().process_weights_after_loading()
            return
        qf = self._quant_family
        if qf == "fp4":
            self._fp4_merge_scales()
            self._fp4_postprocess_weights()
        elif qf == "w4a8":
            self._quantize_to_int4()
        # FP8 per-tensor/per-channel/per-block handled by base.
        super().process_weights_after_loading()

    def _fp4_merge_scales(self):
        for e in range(self.num_local_experts):
            self.w13_scale_2[e] = max(self._fp4_gate_s2[e], self._fp4_up_s2[e])
            self.w13_input_scale[e] = max(self._fp4_gate_i_s[e], self._fp4_up_i_s[e])
        del self._fp4_gate_s2
        del self._fp4_up_s2
        del self._fp4_gate_i_s
        del self._fp4_up_i_s

    def _fp4_postprocess_weights(self):
        moe_config = self._moe_config
        fp4_moe_op = (
            getattr(moe_config, "fp4_moe_op", "trtllm")
            if moe_config is not None
            else "trtllm"
        )

        if fp4_moe_op == "cutedsl":
            from rtp_llm.device.device_impl import CudaImpl

            self.register_buffer(
                "w13_scale", CudaImpl.swizzle_blockscale(self.w13_scale)
            )
            self.register_buffer("w2_scale", CudaImpl.swizzle_blockscale(self.w2_scale))
        elif fp4_moe_op == "trtllm":
            M_tp = self.moe_inter_tp

            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                get_w2_permute_indices_with_cache,
            )

            from rtp_llm.device.device_impl import CudaImpl

            cache: dict = {}
            w13_shape = [
                self.w13.shape[0],
                self.w13.shape[1],
                self.w13.shape[2] * 2,
            ]
            new_w13, new_s13 = CudaImpl.prepare_static_weights_for_trtllm_fp4_moe(
                self.w13.data,
                self.w13_scale,
                w13_shape,
                _maybe_get_cached_w3_w1_permute_indices,
                cache,
            )
            self.w13 = nn.Parameter(new_w13, requires_grad=False)
            self.register_buffer("w13_scale", new_s13)

            w2_shape = [
                self.w2.shape[0],
                self.w2.shape[1],
                self.w2.shape[2] * 2,
            ]
            new_w2, new_s2 = CudaImpl.prepare_static_weights_for_trtllm_fp4_moe(
                self.w2.data,
                self.w2_scale,
                w2_shape,
                get_w2_permute_indices_with_cache,
                cache,
            )
            self.w2 = nn.Parameter(new_w2, requires_grad=False)
            self.register_buffer("w2_scale", new_s2)

    def _quantize_to_int4(self):
        from rtp_llm.model_loader.w4a8_int4_per_channel_quant_weight import (
            quantize_weight_to_int4b,
        )

        E = self.num_experts
        gs = self._group_size
        device = self.w13.data.device

        w13_data = self.w13.data
        w2_data = self.w2.data
        N1, K1 = w13_data.shape[1], w13_data.shape[2]
        N2, K2 = w2_data.shape[1], w2_data.shape[2]

        w13_q = torch.empty(E, N1, K1 // 2, device=device, dtype=torch.int8)
        w13_s = torch.empty(
            E, K1 // gs, N1, 8, device=device, dtype=torch.float8_e4m3fn
        )
        w2_q = torch.empty(E, N2, K2 // 2, device=device, dtype=torch.int8)
        w2_s = torch.empty(E, K2 // gs, N2, 8, device=device, dtype=torch.float8_e4m3fn)

        for e in range(E):
            w13_q[e], w13_s[e] = quantize_weight_to_int4b(w13_data[e], gs)
            w2_q[e], w2_s[e] = quantize_weight_to_int4b(w2_data[e], gs)

        self.w13 = nn.Parameter(w13_q, requires_grad=False)
        self.w2 = nn.Parameter(w2_q, requires_grad=False)
        self.register_buffer("w13_scale", w13_s)
        self.register_buffer("w2_scale", w2_s)

    # ------------------------------------------------------------------ #
    #  Override: weights dict for FP4/W4A8
    # ------------------------------------------------------------------ #

    def _build_weights_dict(self) -> Dict[str, torch.Tensor]:
        if self.quant_method is not None:
            return super()._build_weights_dict()
        weights_dict = super()._build_weights_dict()
        qf = self._quant_family
        if qf == "w4a8":
            weights_dict[W.moe_s1] = self.w13_scale
            weights_dict[W.moe_s2] = self.w2_scale
        elif qf == "fp4":
            weights_dict[W.moe_s1] = self.w13_scale
            weights_dict[W.moe_s2] = self.w2_scale
            weights_dict[W.moe_w1_s2] = self.w13_scale_2
            weights_dict[W.moe_w2_s2] = self.w2_scale_2
            weights_dict[W.moe_w1_i_s] = self.w13_input_scale
            weights_dict[W.moe_w2_i_s] = self.w2_input_scale
        return weights_dict


class Qwen3MoeBlock(RtpModule):
    """Routed-expert MoE block (Qwen3-MoE has no shared expert)."""

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        dp_size = int(getattr(parallelism_config, "dp_size", 1))
        moe_pure_tp_mode = tp_size > 1 and dp_size == 1 and ep_size == 1
        experts_tp_size = tp_size if moe_pure_tp_mode else 1
        experts_tp_rank = tp_rank if moe_pure_tp_mode else 0

        # Router: tiny BF16 linear from hidden -> num_experts. NOT TP-sharded
        # (output dim is num_experts, small). compressed-tensors `ignore` list
        # keeps gate at BF16, so quant_config is None here regardless of phase.
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_experts,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)
        self.experts = Qwen3Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            tp_size=experts_tp_size,
            tp_rank=experts_tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=params_dtype,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=(
                self.experts.fused_moe.topk_ids_dtype
                if self.experts.fused_moe is not None
                else torch.int32
            ),
            device=hidden_states.device,
        )
        self.select_topk(router_logits_fp32, topk_ids, topk_weights)
        return self.experts(hidden_states, topk_weights, topk_ids)


class Qwen3MoeDecoderLayer(RtpModule):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        attn_tp_size: int,
        attn_tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.input_layernorm = RMSResNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        # Reuse dense Qwen3 attention: same qk_norm + bias=False contract.
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=attn_tp_size,
            tp_rank=attn_tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSResNorm(
            hidden_size, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.mlp = Qwen3MoeBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # FusedMoe PureTP routers reduce TP-sharded expert outputs in finalize().
        # Do not all_reduce here, otherwise TP>1 applies the MoE contribution twice.
        return hidden_states, residual


def _extract_moe_config_values(model_config: Any, load_config: Any) -> Dict[str, Any]:
    """Pull all fields needed to build Qwen3MoE layers.

    Mirrors _extract_config_values from new_models/qwen3/language.py but adds
    MoE-specific fields. Tolerates either ModelConfig (C++ pybind) or HF dict.
    """

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    hidden_size = _get(model_config, "hidden_size", 2048)
    num_layers = _get(
        model_config, "num_layers", _get(model_config, "num_hidden_layers", 48)
    )
    vocab_size = _get(model_config, "vocab_size", 151936)

    attn_config = _get(model_config, "attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 32)
        num_kv_heads = _get(attn_config, "kv_head_num", num_heads)
        head_dim = _get(attn_config, "size_per_head", hidden_size // num_heads)
    else:
        num_heads = _get(model_config, "num_attention_heads", 32)
        num_kv_heads = _get(model_config, "num_key_value_heads", num_heads)
        head_dim = _get(model_config, "head_dim", hidden_size // num_heads)

    rms_norm_eps = _get(
        model_config,
        "layernorm_eps",
        _get(model_config, "rms_norm_eps", 1e-6),
    )

    # MoE-specific fields. ModelConfig has expert_num/moe_k/moe_inter_size set
    # by the old-loader-side _create_config (Qwen2Moe._create_config); HF dict
    # uses the original names.
    num_experts = _get(model_config, "expert_num", _get(model_config, "num_experts", 0))
    top_k = _get(model_config, "moe_k", _get(model_config, "num_experts_per_tok", 0))
    moe_intermediate_size = _get(
        model_config,
        "moe_inter_size",
        _get(model_config, "moe_intermediate_size", 0),
    )
    if num_experts <= 0 or top_k <= 0 or moe_intermediate_size <= 0:
        raise ValueError(
            f"Qwen3-MoE config missing fields: expert_num={num_experts}, "
            f"moe_k={top_k}, moe_inter_size={moe_intermediate_size}"
        )

    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    ep_size = getattr(load_config, "ep_size", 1)
    ep_rank = getattr(load_config, "ep_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.bfloat16)
    parallelism_config = getattr(load_config, "parallelism_config", None)
    moe_config = getattr(load_config, "moe_config", None)
    enable_fp32_lm_head = getattr(model_config, "enable_fp32_lm_head", True)
    if parallelism_config is not None and hasattr(
        parallelism_config, "get_attn_tp_size"
    ):
        attn_tp_size = int(parallelism_config.get_attn_tp_size())
        attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
    else:
        attn_tp_size = tp_size
        attn_tp_rank = tp_rank

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=num_layers,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


class Qwen3MoeForCausalLM(GptModelBase):

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        import logging

        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head:
            logging.info(
                "[Qwen3MoeForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
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

        cfg = _extract_moe_config_values(model_config, load_config)

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    moe_intermediate_size=cfg["moe_intermediate_size"],
                    num_experts=cfg["num_experts"],
                    top_k=cfg["top_k"],
                    head_dim=cfg["head_dim"],
                    layer_idx=i,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    attn_tp_size=cfg["attn_tp_size"],
                    attn_tp_rank=cfg["attn_tp_rank"],
                    ep_size=cfg["ep_size"],
                    ep_rank=cfg["ep_rank"],
                    model_config=cfg["model_config"],
                    parallelism_config=cfg["parallelism_config"],
                    moe_config=cfg["moe_config"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                )
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
