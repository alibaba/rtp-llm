"""Opt-in GLM-5 decode bridge to the external ``cuda_dag_runtime`` package."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.utils.model_weight import W

_ENABLE_ENV = "RTP_LLM_GLM52_CUDA_DAG"
_SUPPORTED_MODEL_TYPES = frozenset(("glm_5", "glm_5_mtp"))


def _load_runtime() -> Any:
    return importlib.import_module("cuda_dag_runtime")


def resolve_glm52_cuda_dag_enabled() -> bool:
    value = os.environ.get(_ENABLE_ENV, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"invalid {_ENABLE_ENV}={value!r}")


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _linear_weight(linear: Any) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scales", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("CUDA-DAG requires FP8 weight and weight_scales")
    return weight, scale


def _page_block_table(attn_inputs: Any) -> torch.Tensor | None:
    """Return RTP's request-local table at CUDA-DAG's 64-token granularity."""
    table = getattr(attn_inputs, "kv_cache_block_id_device", None)
    return table if isinstance(table, torch.Tensor) and table.numel() > 0 else None


def _tokens_per_request(
    rows: int,
    context_lens: Any,
    block_table: Any,
) -> int | None:
    if (
        rows <= 0
        or not isinstance(context_lens, torch.Tensor)
        or context_lens.dtype != torch.int32
        or context_lens.numel() != rows
        or not isinstance(block_table, torch.Tensor)
        or block_table.dim() != 2
        or block_table.size(0) <= 0
    ):
        return None
    request_bs = int(block_table.size(0))
    return rows // request_bs if rows % request_bs == 0 else None


def _unsupported_parallelism_reason(parallelism: Any) -> str | None:
    raw_tp = int(getattr(parallelism, "tp_size", 1) or 1)
    attn_tp = int(parallelism.get_attn_tp_size())
    cp = getattr(parallelism, "prefill_cp_config", None)
    if cp is not None:
        is_enabled = getattr(cp, "is_enabled", None)
        is_prefill_enabled = getattr(cp, "is_prefill_enabled", None)
        if (
            (callable(is_enabled) and is_enabled())
            or (callable(is_prefill_enabled) and is_prefill_enabled())
            or bool(getattr(cp, "kv_cache_sharded", False))
            or int(getattr(cp, "prefill_cp_size", 0) or 0) > 1
        ):
            return "CUDA-DAG requires CP=1 and an unsharded KV cache"
    if raw_tp != 1 or attn_tp != 1:
        return "CUDA-DAG requires TP=1"
    return None


@dataclass(frozen=True)
class _CudaDagOutput:
    residual: torch.Tensor
    topk_indices: torch.Tensor
    moe_hidden_states: torch.Tensor
    routed_indices: torch.Tensor
    routed_weights: torch.Tensor


@dataclass(frozen=True)
class _Dispatch:
    implementation: Any
    params: Any
    rows: int
    has_indexer: bool
    reuses_indexer: bool
    block_table: torch.Tensor | None
    context_lens: Any
    reason: str | None


@dataclass(frozen=True)
class _MoeBindings:
    norm_weight: torch.Tensor
    router_weight: torch.Tensor
    correction_bias: torch.Tensor
    activation: torch.Tensor
    scale: torch.Tensor
    routed_indices: torch.Tensor
    routed_weights: torch.Tensor

    @property
    def pointer_key(self) -> tuple[int, ...]:
        return tuple(
            int(t.data_ptr())
            for t in (
                self.activation,
                self.scale,
                self.routed_indices,
                self.routed_weights,
            )
        )


class Glm52CudaDagAdapter:
    """Bind one GLM layer to a cached CUDA-DAG plan."""

    def __init__(
        self,
        *,
        layer_idx: int,
        config: Any,
        parallelism_config: Any,
        self_attn: Any,
        input_layernorm: Any,
        mlp: Any,
        post_attention_layernorm: Any,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.config = config
        self.parallelism_config = parallelism_config
        self.self_attn = self_attn
        self.input_layernorm = input_layernorm
        self.mlp = mlp
        self.post_attention_layernorm = post_attention_layernorm
        self._router_weight = self._prepare_router_weight(mlp)
        self._plans: dict[tuple[Any, ...], Any] = {}
        self._draft_prefill_clone = False
        self._disabled_reason = self._static_disabled_reason()

    def _static_disabled_reason(self) -> str | None:
        reason = _unsupported_parallelism_reason(self.parallelism_config)
        if reason is not None:
            return reason
        if str(getattr(self.config, "model_type", "")) not in _SUPPORTED_MODEL_TYPES:
            return "unsupported model type"
        attn = self.config.attn_config
        if not bool(getattr(attn, "use_mla", False)):
            return "attention is not MLA"
        if int(getattr(attn, "kernel_tokens_per_block", 0)) != 64:
            return "CUDA-DAG requires 64-token KV pages"

        # CUDA-DAG writes router outputs directly into MegaMoE's stable input
        # buffers. Keep this first integration limited to the deployed ABI.
        config = self.config
        if (
            self._router_weight is None
            or getattr(self.mlp, "correction_bias", None) is None
            or not callable(
                getattr(
                    getattr(self.mlp, "fused_moe", None),
                    "prepacked_input_views",
                    None,
                )
            )
            or bool(getattr(self.mlp, "_use_mega_moe_fused_shared", False))
            or getattr(self.mlp, "shared_expert", None) is None
            or getattr(self.mlp, "shared_expert_gate", None) is not None
            or int(getattr(self.mlp, "ffn_tp_size", 1)) != 1
            or (
                int(getattr(config, "expert_num", 0)),
                int(getattr(config, "moe_k", 0)),
                int(getattr(config, "moe_n_group", 0)),
                int(getattr(config, "moe_topk_group", 0)),
                bool(getattr(config, "has_moe_norm", False)),
                float(getattr(config, "routed_scaling_factor", 0.0)),
            )
            != (256, 8, 1, 1, True, 2.5)
        ):
            return "unsupported MegaMoE contract"
        return None

    @staticmethod
    def _prepare_router_weight(mlp: Any) -> torch.Tensor | None:
        weight = getattr(getattr(mlp, "gate", None), "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != (256, 6144)
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return None
        return weight

    def clone_for_cuda_graph(
        self,
        *,
        mlp: Any,
        draft_prefill: bool = False,
    ) -> Glm52CudaDagAdapter:
        clone = object.__new__(type(self))
        clone.layer_idx = self.layer_idx
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.self_attn = self.self_attn
        clone.input_layernorm = self.input_layernorm
        clone.mlp = mlp
        clone.post_attention_layernorm = self.post_attention_layernorm
        clone._router_weight = self._router_weight
        clone._plans = self._plans
        clone._draft_prefill_clone = bool(draft_prefill)
        clone._disabled_reason = self._disabled_reason
        return clone

    def _moe_bindings(self, rows: int) -> _MoeBindings | None:
        # The ABI owns shapes and dtypes. Only M is dynamic here; an oversized
        # row count is rejected by prepacked_input_views().
        try:
            activation, scale, routed_indices, routed_weights = (
                self.mlp.fused_moe.prepacked_input_views(rows)
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        return _MoeBindings(
            self.post_attention_layernorm.weight.data,
            self._router_weight,
            self.mlp.correction_bias,
            activation,
            scale,
            routed_indices,
            routed_weights,
        )

    @staticmethod
    def _attention_impl(fmha_impl: Any) -> Any:
        if hasattr(fmha_impl, "weights") and hasattr(fmha_impl, "fmha_params"):
            return fmha_impl
        nested = getattr(fmha_impl, "fmha_impl", None)
        if nested is not None and hasattr(nested, "weights"):
            return nested
        raise TypeError("cannot locate SparseMlaImpl")

    def _dispatch(
        self, hidden_states: torch.Tensor, fmha_impl: Any, kv_cache: Any
    ) -> _Dispatch:
        implementation = self._attention_impl(fmha_impl)
        attn_inputs = implementation.attn_inputs
        params = implementation.fmha_params
        rows = int(hidden_states.size(0)) if hidden_states.dim() == 2 else 0
        has_indexer = bool(getattr(self.self_attn, "has_indexer", False))
        reuses_indexer = bool(getattr(self.self_attn, "reuse_topk_indices", False))
        block_table = _page_block_table(attn_inputs)
        context_lens = getattr(params, "expanded_seq_lens", None)
        tokens_per_request = _tokens_per_request(rows, context_lens, block_table)
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_draft_prefill = (
            str(getattr(self.config, "model_type", "")) == "glm_5_mtp"
            and self._draft_prefill_clone
            and is_prefill
            and not is_target_verify
        )

        reason = None
        if kv_cache is None:
            reason = "KV cache is unavailable"
        elif hidden_states.dim() != 2 or rows <= 0 or hidden_states.size(1) != 6144:
            reason = "unsupported hidden-state shape"
        elif is_prefill and not is_target_verify and not is_draft_prefill:
            reason = "ordinary prefill is unsupported"
        elif has_indexer and tokens_per_request is None:
            reason = "invalid Indexer request metadata"
        elif is_draft_prefill:
            expected = int(getattr(self.config, "gen_num_per_cycle", 0) or 0) + 1
            if expected <= 1 or tokens_per_request != expected:
                reason = "draft-prefill width does not match MTP"

        return _Dispatch(
            implementation,
            params,
            rows,
            has_indexer,
            reuses_indexer,
            block_table,
            context_lens,
            reason,
        )

    @staticmethod
    def _metadata(
        runtime: Any,
        implementation: Any,
        context_lens: torch.Tensor,
        request_ids: torch.Tensor,
        request_bs: int,
    ) -> Any:
        cache = implementation._glm_cuda_dag_metadata
        key = (int(context_lens.numel()), request_bs)
        metadata = cache.get(key)
        if metadata is None:
            metadata = runtime.prepare_glm52_runtime_attention_metadata(
                context_lens=context_lens,
                request_ids=request_ids,
                request_bs=request_bs,
            )
            cache[key] = metadata
        else:
            metadata.validate_source(context_lens, request_ids, request_bs=request_bs)
        return metadata

    @staticmethod
    def _moe_kwargs(moe: _MoeBindings, epsilon: float) -> dict[str, Any]:
        return {
            "moe_norm_weight": moe.norm_weight,
            "moe_norm_epsilon": epsilon,
            "router_weight": moe.router_weight,
            "correction_bias": moe.correction_bias,
            "moe_activation": moe.activation,
            "moe_scale": moe.scale,
            "routed_indices": moe.routed_indices,
            "routed_weights": moe.routed_weights,
        }

    def _prepare_main_plan(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        implementation: Any,
        kv_cache: Any,
        indices: torch.Tensor,
        moe: _MoeBindings,
    ) -> Any:
        runtime = _load_runtime()
        attention = self.self_attn
        params = implementation.fmha_params
        weights = implementation.weights[self.layer_idx]
        b0_weight, b0_scale = _linear_weight(attention.fused_qkv_a_proj)
        d0_weight, d0_scale = _linear_weight(attention.q_b_proj)
        f3_weight, f3_scale = _linear_weight(attention.o_proj)
        return runtime.prepare_glm52_runtime_main_only_attention_moe_prep_plan(
            hidden=hidden_states,
            residual=residual,
            norm_weight=self.input_layernorm.weight.data,
            b0_weight=b0_weight,
            b0_weight_scale=b0_scale,
            q_norm_weight=attention.q_a_layernorm.weight.data,
            kv_norm_weight=attention.kv_a_layernorm.weight.data,
            cos_sin=implementation._cos_sin_cache,
            positions=params.positions_d,
            slot_mapping=params.slot_mapping,
            mla_cache=kv_cache.kv_cache_base,
            d0_weight=d0_weight,
            d0_weight_scale=d0_scale,
            d2_weight=weights[W.mla_kc],
            reused_indices=indices,
            f1_weight=weights[W.mla_vc],
            f3_weight=f3_weight,
            f3_weight_scale=f3_scale,
            softmax_scale=float(implementation.fmha_impl.scale),
            input_norm_epsilon=float(self.input_layernorm.variance_epsilon),
            q_norm_epsilon=float(attention.q_a_layernorm.variance_epsilon),
            kv_norm_epsilon=float(attention.kv_a_layernorm.variance_epsilon),
            storage_plan=implementation._glm_cuda_dag_storage_plans.get(
                ("main", int(hidden_states.size(0)))
            ),
            mode="programmatic",
            **self._moe_kwargs(
                moe, float(self.post_attention_layernorm.variance_epsilon)
            ),
        )

    def _prepare_full_plan(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        dispatch: _Dispatch,
        kv_cache: Any,
        moe: _MoeBindings,
    ) -> Any:
        if dispatch.block_table is None:
            raise RuntimeError("CUDA-DAG requires a 64-token page table")
        runtime = _load_runtime()
        attention = self.self_attn
        indexer = attention.indexer
        implementation = dispatch.implementation
        params = dispatch.params
        weights = implementation.weights[self.layer_idx]
        metadata = self._metadata(
            runtime,
            implementation,
            dispatch.context_lens,
            params.batch_indice_d,
            int(dispatch.block_table.size(0)),
        )
        b0_weight, b0_scale = _linear_weight(attention.fused_qkv_a_proj)
        b1_weight, b1_scale = _linear_weight(indexer.wk)
        d0_weight, d0_scale = _linear_weight(attention.q_b_proj)
        d1_weight, d1_scale = _linear_weight(indexer.wq_b)
        f3_weight, f3_scale = _linear_weight(attention.o_proj)
        return runtime.prepare_glm52_runtime_attention_moe_prep_plan(
            hidden=hidden_states,
            residual=residual,
            norm_weight=self.input_layernorm.weight.data,
            c0_weight=indexer.weights_proj.weight,
            b0_weight=b0_weight,
            b0_weight_scale=b0_scale,
            q_norm_weight=attention.q_a_layernorm.weight.data,
            kv_norm_weight=attention.kv_a_layernorm.weight.data,
            cos_sin=implementation._cos_sin_cache,
            positions=params.positions_d,
            slot_mapping=params.slot_mapping,
            mla_cache=kv_cache.kv_cache_base,
            b1_weight=b1_weight,
            b1_weight_scale=b1_scale,
            b1_norm_weight=indexer.k_norm.weight.data,
            b1_norm_bias=indexer.k_norm.beta.data,
            indexer_cache=kv_cache.kv_scale_base,
            d0_weight=d0_weight,
            d0_weight_scale=d0_scale,
            d1_weight=d1_weight,
            d1_weight_scale=d1_scale,
            d2_weight=weights[W.mla_kc],
            context_lens=dispatch.context_lens,
            block_table=dispatch.block_table,
            request_ids=params.batch_indice_d,
            f1_weight=weights[W.mla_vc],
            f3_weight=f3_weight,
            f3_weight_scale=f3_scale,
            softmax_scale=float(implementation.fmha_impl.scale),
            input_norm_epsilon=float(self.input_layernorm.variance_epsilon),
            q_norm_epsilon=float(attention.q_a_layernorm.variance_epsilon),
            kv_norm_epsilon=float(attention.kv_a_layernorm.variance_epsilon),
            indexer_k_norm_epsilon=float(indexer.k_norm.variance_epsilon),
            runtime_metadata=metadata,
            storage_plan=implementation._glm_cuda_dag_storage_plans.get(
                ("full", int(hidden_states.size(0)))
            ),
            programmatic=True,
            **self._moe_kwargs(
                moe, float(self.post_attention_layernorm.variance_epsilon)
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
        prev_topk_indices: torch.Tensor | None,
    ) -> _CudaDagOutput:
        dispatch = self._dispatch(hidden_states, fmha_impl, kv_cache)
        if dispatch.reason is not None:
            raise RuntimeError(
                f"CUDA-DAG model contract failed at layer {self.layer_idx}: "
                f"{dispatch.reason}"
            )
        moe = self._moe_bindings(dispatch.rows)
        if moe is None:
            raise RuntimeError("MegaMoE backend does not support CUDA-DAG prepacking")

        path = "full" if dispatch.has_indexer else "main"
        if path == "main" and (
            not dispatch.reuses_indexer or prev_topk_indices is None
        ):
            raise RuntimeError("main-only CUDA-DAG layer requires prior TopK indices")
        request_bs = (
            int(dispatch.block_table.size(0))
            if dispatch.block_table is not None and dispatch.has_indexer
            else 0
        )
        key = (
            path,
            dispatch.rows,
            request_bs,
            id(dispatch.implementation),
            moe.pointer_key,
        )
        plan = self._plans.get(key)
        if plan is None:
            if _is_capturing():
                raise RuntimeError(
                    "CUDA-DAG plan cache miss during capture; warm up this shape first"
                )
            if dispatch.has_indexer:
                plan = self._prepare_full_plan(
                    hidden_states, residual, dispatch, kv_cache, moe
                )
            else:
                plan = self._prepare_main_plan(
                    hidden_states,
                    residual,
                    dispatch.implementation,
                    kv_cache,
                    prev_topk_indices,
                    moe,
                )
            self._plans[key] = plan
            dispatch.implementation._glm_cuda_dag_storage_plans.setdefault(
                (path, dispatch.rows), plan
            )

        outputs = plan(hidden_states, residual)
        topk_indices = (
            outputs.sparse_indices if dispatch.has_indexer else outputs.reused_indices
        )
        return _CudaDagOutput(
            outputs.residual,
            topk_indices,
            outputs.moe_hidden_states,
            outputs.routed_indices,
            outputs.routed_weights,
        )


def should_enable_glm52_cudadag(
    layers: Any,
    layer_num: int,
    hidden_states: torch.Tensor,
    fmha_impl: Any,
    kv_cache: Any,
) -> bool:
    """Return whether the complete model forward should use CUDA-DAG."""
    if layer_num <= 0:
        return False

    # GLM layers share one model/parallelism/MegaMoE contract. Checking layer 0
    # avoids repeating identical validation across every layer.
    adapter = layers[0]._glm52_cuda_dag_adapter
    if adapter is None or adapter._disabled_reason is not None:
        return False

    first_cache = kv_cache.get_layer_cache(0) if kv_cache is not None else None
    dispatch = adapter._dispatch(hidden_states, fmha_impl, first_cache)

    # Layer 0 must seed the global TopK indices reused by main-only layers.
    return (
        dispatch.reason is None
        and dispatch.has_indexer
        and adapter._moe_bindings(dispatch.rows) is not None
    )
