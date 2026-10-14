"""gpt-oss (20b / 120b) python execution model.

Structure follows GenericMoeModel/GenericMoeDecoderLayer: pre-norm RMS
residual blocks of GQA MHA attention (with per-head sinks and alternating
sliding/full windows) + all-routed MXFP4 MoE (Fp8Fp4MoeLayer on the
DeepGEMM path).

gpt-oss specifics handled here:
  - Sinks: per-layer ``W.attn_sink`` [head_num] tensors registered into the
    trtllm-gen attention ops (softmax denominator-only column). Backends
    without native sink support are a step-4 concern.
  - Window: even layers are sliding_attention (window_left = sliding_window
    - 1); odd layers full. Both registered via the same per-layer extras.
  - MoE: router ``topk_softmax`` (raw biased logits -> top-4 -> softmax over
    the selected), routed_scaling_factor 1.0, no shared experts, experts
    carry biases (consumed by the grouped-fp4 executor), activation
    ``gate * sigmoid(1.702 * gate) * (up + 1)`` with clamp limit 7.
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.gpt_oss.gpt_oss_dequant import (
    dequant_mxfp4_stacked,
    use_native_mxfp4,
)
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import CausalAttention, Embedding, RMSResNorm
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.ops import HWKernelConfig, HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

# gpt-oss experts: gate * sigmoid(alpha * gate) * (up + 1) (quick-gelu style)
GPT_OSS_SWIGLU_ALPHA = 1.702
GPT_OSS_SWIGLU_UP_SHIFT = 1.0


def _dequant_expert_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Replace packed MXFP4 expert tensors with BF16 dequantized ones.

    Consumes ``W.moe_w1``/``W.moe_s1`` (``[E, 2I, K/2]`` nibbles +
    ``[E, 2I, K/32]`` scales, gate rows first) and ``W.moe_w2``/``W.moe_s2``,
    mutating the dict in place so the packed source tensors are freed. The
    dequantized ``W.moe_w1`` is re-ordered to ``[up; gate]`` — the halves
    convention of the generic Triton fused-MoE executor's silu_and_mul
    kernel (Qwen's stack_moe_w1 order). Router, biases, and attention
    weights pass through untouched.
    """
    w13_packed = weights.pop(W.moe_w1)
    s13 = weights.pop(W.moe_s1)
    w2_packed = weights.pop(W.moe_w2)
    s2 = weights.pop(W.moe_s2)
    w13 = dequant_mxfp4_stacked(w13_packed, s13)  # [E, 2I, K] [gate; up]
    inter = w13.shape[1] // 2
    weights[W.moe_w1] = torch.cat(
        [w13[:, inter:], w13[:, :inter]], dim=1
    )  # [up; gate]
    # gate_up bias follows its weight rows: swap its halves too, so the
    # triton executor's [up; gate] GEMM1 output gets the matching bias halves.
    b13 = weights.get(W.moe_b1)
    if b13 is not None:
        weights[W.moe_b1] = torch.cat(
            [b13[:, inter:], b13[:, :inter]], dim=1
        )
    weights[W.moe_w2] = dequant_mxfp4_stacked(w2_packed, s2)  # [E, K, I]
    return weights


class GptOssDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: Optional[MoeConfig],
        max_generate_batch_size: int = 0,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            config.quant_config,
            hw_kernel_config,
            layer_idx,
        )

        moe_strategy = "auto"
        if moe_config is not None:
            moe_strategy = getattr(moe_config, "moe_strategy", None) or "auto"
        # sm103 (B300) runs the packed MXFP4 experts natively; sm90/sm12x
        # dequantize to BF16 once here and run the generic Triton lane.
        # The BF16 lane pins the pure-TP Triton strategy explicitly (an
        # "auto" quant=None request could otherwise match a DeepEP lane).
        if use_native_mxfp4():
            moe_weights = weights
            expert_quant = "FP8_FP4"
            w1_layout = "gate_up"
            # The gpt-oss expert contract (per-expert biases + gated
            # activation alpha/shift) is only implemented by grouped_fp4;
            # an "auto" request must not silently land on another
            # FP8_FP4 executor that drops the contract.
            if moe_strategy == "auto":
                moe_strategy = "grouped_fp4"
        else:
            moe_weights = _dequant_expert_weights(weights)
            expert_quant = None
            w1_layout = "up_gate"
            if moe_strategy == "auto":
                moe_strategy = "no_quant_cpp"
        self.mlp = Fp8Fp4MoeLayer(
            layer_id=layer_idx,
            dim=config.hidden_size,
            moe_inter_dim=config.moe_inter_size,
            n_routed_experts=config.expert_num,
            n_activated_experts=config.moe_k,
            n_shared_experts=0,
            score_func="topk_softmax",
            route_scale=float(config.routed_scaling_factor),
            swiglu_limit=float(config.swiglu_limit),
            n_hash_layers=0,
            vocab_size=int(config.vocab_size),
            layer_weights=moe_weights,
            ep_size=int(parallelism_config.ep_size or 1),
            ep_rank=int(parallelism_config.ep_rank or 0),
            max_tokens_per_rank=max(
                int(config.max_seq_len or 4096), int(max_generate_batch_size or 0)
            ),
            strategy=moe_strategy,
            model_type="gpt_oss",
            moe_w1_layout=w1_layout,
            swiglu_alpha=1.702,
            swiglu_up_shift=1.0,
            moe_quant_method=expert_quant,
        )

        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states, None)
        return hidden_states, residual


class GptOssModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: Optional[MoeConfig],
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma),
            eps=model_config.layernorm_eps,
        )
        self._init_capture_context(
            self._capture_canonical_layer, self._capture_canonical_final
        )
        # per-layer attention sinks / windows, registered into the selected
        # fmha impl on each prepare_fmha_impl (idempotent; pointer-stable)
        self._layer_sinks: Dict[int, Optional[torch.Tensor]] = {}
        self._layer_window_left: Dict[int, int] = {}
        sliding_window = int(model_config.attn_config.sliding_window or 0)
        hybrid_types = model_config.hybrid_attention_config.hybrid_attention_types
        for layer_idx, layer_weights in enumerate(weights.weights):
            sink = layer_weights.get(W.attn_sink)
            if sink is not None:
                self._layer_sinks[layer_idx] = sink.to(torch.float32).contiguous()
            layer_type = (
                hybrid_types[layer_idx]
                if layer_idx < len(hybrid_types)
                else HybridAttentionType.NONE
            )
            if layer_type == HybridAttentionType.SLIDING_WINDOW and sliding_window > 0:
                # flashinfer window_left is inclusive: attend [p-W+1, p]
                self._layer_window_left[layer_idx] = sliding_window - 1
        # layer weights beyond layer_num (MTP etc.) are not consumed here

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual capture requires a residual tensor")
        return hidden_states + residual

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual finalization requires a residual tensor")
        return self.norm(hidden_states, residual)[0]

    def _register_attn_extras(self, fmha_impl: Any) -> None:
        impls = (
            fmha_impl.values() if isinstance(fmha_impl, dict) else [fmha_impl]
        )
        import os as _os
        for impl in impls:
            if _os.environ.get("GPT_OSS_SINK_DEBUG"):
                print(
                    f"[sinkdbg] REG impl={type(impl).__name__} has_extras="
                    f"{hasattr(impl, 'register_layer_attn_extras')} "
                    f"sinks={len(self._layer_sinks)}",
                    flush=True,
                )
            register = getattr(impl, "register_layer_attn_extras", None)
            if register is not None:
                register(self._layer_sinks, self._layer_window_left)

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Any:
        fmha_impl = super().prepare_fmha_impl(
            inputs, is_cuda_graph, cuda_graph_selection_mode
        )
        self._register_attn_extras(fmha_impl)
        return fmha_impl

    def embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        return self.embed_tokens(inputs.input_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.embedding(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
            self._register_attn_extras(fmha_impl)
        capture = self.capture_context(inputs.capture_hidden_states)
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            layer_kv_cache = (
                self.kv_cache.get_layer_cache(i) if self.kv_cache else None
            )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=layer_kv_cache,
            )
            capture.capture_layer(i, hidden_states, residual)

        return capture.finalize(hidden_states, residual)


__all__ = ["GptOssModel", "GptOssDecoderLayer"]
