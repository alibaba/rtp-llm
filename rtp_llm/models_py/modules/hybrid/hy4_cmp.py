"""Independent HY4 CMP attention path.

The caller owns main Query/cache preparation and output Gate. Target Indexer K
and Q projections use separate side streams and join at the existing fused Q/K
epilogue. Score/TopK stays on the K stream; long KV waits for the complete main
Query. MTP and TopK-reuse layers create no side-stream work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

_ENABLE_ENV = "RTP_LLM_HY4_CMP"
_INDEXER_FRONTEND_ENV = "RTP_LLM_HY4_CMP_INDEXER_FRONTEND"
_SUPPORTED_MODEL_TYPES = frozenset(("hy_v4", "hy_v4_mtp"))
_MAX_ROWS = 256


def _resolve_bool_env(name: str, *, default_on: bool = True) -> bool:
    value = os.environ.get(name, "1" if default_on else "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"invalid {name}={value!r}")


def resolve_hy4_cmp_enabled() -> bool:
    return _resolve_bool_env(_ENABLE_ENV)


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _mxfp8_quantize_hidden(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the canonical MXFP8 quantizer without importing CUDA code on CPU."""
    from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

    return mxfp8_quant_act_packed(hidden_states)


@dataclass
class _Events:
    caller_ready: torch.cuda.Event
    q_inputs_ready: torch.cuda.Event
    indexer_q_ready: torch.cuda.Event
    q_path_complete: torch.cuda.Event
    indexer_complete: torch.cuda.Event


def _record_stream(value: Any, stream: Any) -> None:
    """Track cross-stream consumers independently of execution dependencies."""
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for tensor in value:
            _record_stream(tensor, stream)


class Hy4Cmp:
    """Overlap independent HY4 attention branches without changing formulas."""

    _streams_by_device: dict[int, tuple[Any, Any]] = {}

    def __init__(
        self,
        *,
        config: Any,
        parallelism_config: Any,
        self_attn: Any,
    ) -> None:
        self.config = config
        self.parallelism_config = parallelism_config
        self.self_attn = self_attn
        self._indexer_frontend_parallel = _resolve_bool_env(_INDEXER_FRONTEND_ENV)
        self._events: Optional[_Events] = None
        self._disabled_reason = self._static_disabled_reason()

    def clone_for_cuda_graph(self, *, self_attn: Any) -> "Hy4Cmp":
        clone = object.__new__(type(self))
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.self_attn = self_attn
        clone._indexer_frontend_parallel = self._indexer_frontend_parallel
        # Each captured graph owns its event nodes.  Streams are device-global
        # and must be created before capture, so sharing them is intentional.
        clone._events = None
        clone._disabled_reason = self._disabled_reason
        return clone

    def _static_disabled_reason(self) -> Optional[str]:
        if str(getattr(self.config, "model_type", "")) not in _SUPPORTED_MODEL_TYPES:
            return "unsupported model type"
        raw_tp = int(getattr(self.parallelism_config, "tp_size", 1) or 1)
        attn_tp = int(self.parallelism_config.get_attn_tp_size())
        if raw_tp != 1 or attn_tp != 1:
            return "HY4 CMP requires TP=1"
        if getattr(self.self_attn, "gating_type", None) != "elementwise":
            return "HY4 CMP requires elementwise gated MLA"
        if getattr(self.self_attn, "gate_proj", None) is None:
            return "HY4 CMP requires gate_proj"
        if int(getattr(self.self_attn, "q_lora_rank", 0)) <= 0:
            return "HY4 CMP requires a low-rank Query projection"
        indexer = getattr(self.self_attn, "indexer", None)
        if indexer is not None and bool(getattr(indexer, "use_hadamard", True)):
            return "HY4 CMP requires the HY4 Indexer layout"
        return None

    @staticmethod
    def _dynamic_disabled_reason(
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
    ) -> Optional[str]:
        if not hidden_states.is_cuda or hidden_states.dim() != 2:
            return "HY4 CMP requires a 2D CUDA activation"
        rows = int(hidden_states.size(0))
        if rows <= 0 or rows > _MAX_ROWS:
            return "HY4 CMP supports 1..256 rows"
        if kv_cache is None:
            return "KV cache is unavailable"
        if not bool(fmha_impl.is_sparse()):
            return "HY4 CMP requires sparse MLA"
        if not (
            bool(getattr(fmha_impl, "supports_topk_late_binding", False))
            and callable(getattr(fmha_impl, "prepare_topk_independent_forward", None))
            and callable(getattr(fmha_impl, "finish_topk_dependent_forward", None))
        ):
            return "HY4 CMP requires separate MLA preparation and execution"
        if getattr(fmha_impl, "cp_params", None) is not None:
            return "HY4 CMP does not support context-parallel preparation"
        attn_inputs = fmha_impl.attn_inputs
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        is_decode_extension = bool(
            getattr(attn_inputs, "is_target_verify", False)
            or getattr(attn_inputs, "is_draft_extend", False)
        )
        if is_prefill and not is_decode_extension:
            return "ordinary prefill is unsupported"
        return None

    @classmethod
    def _side_streams(cls, device: torch.device) -> tuple[Any, Any]:
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        streams = cls._streams_by_device.get(device_index)
        if streams is None:
            if _is_capturing():
                raise RuntimeError("HY4 CMP streams must be created before capture")
            with torch.cuda.device(device):
                streams = (torch.cuda.Stream(), torch.cuda.Stream())
            cls._streams_by_device[device_index] = streams
        return streams

    @staticmethod
    def _new_events(device: torch.device) -> _Events:
        if _is_capturing():
            raise RuntimeError("HY4 CMP events must be created before capture")
        events = _Events(*(torch.cuda.Event() for _ in range(5)))
        with torch.cuda.device(device):
            for event in vars(events).values():
                event.record()
        return events

    def _serialize_score_after_q_path(self, fmha_impl: Any) -> bool:
        """Use the scorer's page table to apply the 512K KV threshold.

        The Indexer scorer prefers ``kv_cache_kernel_block_id_device``, whose
        width is expressed in ``kernel_tokens_per_block`` pages.  Multiplying
        the narrower physical-block table by that kernel page size can
        undercount capacity when the two granularities differ (for example,
        256-token physical blocks split into 64-token kernel pages).
        """
        attn_inputs = getattr(fmha_impl, "attn_inputs", None)
        if attn_inputs is None:
            nested = getattr(fmha_impl, "fmha_impl", None)
            attn_inputs = getattr(nested, "attn_inputs", None)
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        tokens_per_block = int(getattr(self.self_attn, "token_per_block", 64) or 64)
        if (
            not isinstance(block_table, torch.Tensor)
            or block_table.dim() != 2
            or block_table.numel() == 0
        ):
            block_table = getattr(attn_inputs, "kv_cache_block_id_device", None)
            attn_config = getattr(self.self_attn, "attn_config", None)
            tokens_per_block = int(
                getattr(attn_config, "tokens_per_block", tokens_per_block)
                or tokens_per_block
            )
        if (
            not isinstance(block_table, torch.Tensor)
            or block_table.dim() != 2
            or block_table.numel() == 0
        ):
            return False
        return int(block_table.size(1)) * tokens_per_block >= (1 << 19)

    def can_run(
        self, hidden_states: torch.Tensor, fmha_impl: Any, kv_cache: Any
    ) -> bool:
        return (
            self._disabled_reason is None
            and self._dynamic_disabled_reason(hidden_states, fmha_impl, kv_cache)
            is None
        )

    def _qkv_a(self, hidden: torch.Tensor, x_fp8: Any, x_scale: Any) -> tuple:
        """Keep the existing HY4 projection and Q-A normalization numerics."""
        from rtp_llm.models_py.modules.hybrid import mla_attention as kernels

        attn = self.self_attn
        projected = (
            attn.fused_qkv_a_proj(x_fp8, input_scales=x_scale)
            if x_fp8 is not None and x_scale is not None
            else attn.fused_qkv_a_proj(hidden)
        )
        q, kv = projected.split(
            [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1
        )
        q_c = q_fp8 = q_scale = None
        norm = attn.q_a_layernorm
        if attn._fuse_q_a_norm_mode == "mxfp8":
            q_fp8, q_scale = kernels.fused_strided_rmsnorm_per_token_fp8_quant(
                q,
                norm.weight.data,
                norm.variance_epsilon,
                group_size=32,
                scale_ue8m0=True,
                mxfp8_semantics=True,
            )
        elif attn._fuse_q_a_norm_mode == "fp8_dual":
            q_c, q_fp8, q_scale = (
                kernels.fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                    q,
                    norm.weight.data,
                    norm.variance_epsilon,
                    group_size=128,
                    scale_ue8m0=attn.q_b_proj.scale_ue8m0,
                )
            )
        elif attn._fuse_q_a_norm_mode == "bf16":
            q_c = kernels.fused_strided_rmsnorm(
                q, norm.weight.data, norm.variance_epsilon
            )
        else:
            q_c = norm(q.contiguous())
        return q_c, q_fp8, q_scale, kv

    def _main_query(
        self,
        hidden: torch.Tensor,
        q_inputs: tuple,
        kv: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
    ) -> tuple:
        """Main Q-B, caller-stream Gate, and complete Query/cache preparation."""
        from rtp_llm.models_py.modules.hybrid import mla_attention as kernels

        attn = self.self_attn
        q_c, q_fp8, q_scale = q_inputs
        q = (
            attn.q_b_proj(q_fp8, input_scales=q_scale)
            if q_fp8 is not None
            else attn.q_b_proj(q_c)
        )
        q = q.reshape(-1, attn.num_heads, attn.q_head_dim)
        # Keep the measured policy: no output Gate stream or overlap with MLA.
        gate = attn.gate_proj(hidden)
        kv, k_pe = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        norm = attn.kv_a_layernorm
        norm_kwargs = {}
        can_fuse = getattr(fmha_impl, "can_fuse_kv_norm_cache", None)
        if (
            attn._fuse_kv_a_norm
            and callable(can_fuse)
            and can_fuse(kv, norm.weight.data)
        ):
            norm_kwargs = dict(
                kv_norm_weight=norm.weight.data,
                kv_norm_eps=float(norm.variance_epsilon),
            )
        elif attn._fuse_kv_a_norm:
            kv = kernels.fused_strided_rmsnorm(
                kv, norm.weight.data, norm.variance_epsilon
            )
        else:
            kv = norm(kv.contiguous())
        prepared = fmha_impl.prepare_topk_independent_forward(
            q,
            kv,
            k_pe,
            kv_cache,
            attn.layer_idx,
            attn.attn_sink,
            **norm_kwargs,
        )
        return prepared, gate

    def _indexer_k(
        self, hidden: torch.Tensor, x_fp8: Any, x_scale: Any
    ) -> torch.Tensor:
        indexer = self.self_attn.indexer
        k = (
            indexer.wk(x_fp8, input_scales=x_scale)
            if x_fp8 is not None and x_scale is not None
            else indexer.wk(hidden)
        )
        return indexer.k_norm(k)

    def _indexer_q(self, q_inputs: tuple) -> torch.Tensor:
        indexer = self.self_attn.indexer
        q_c, q_fp8, q_scale = q_inputs
        q = (
            indexer.wq_b(q_fp8, input_scales=q_scale)
            if q_fp8 is not None
            else indexer.wq_b(q_c)
        )
        return q.view(-1, indexer.index_n_heads, indexer.index_head_dim)

    def _indexer_post(
        self,
        hidden: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
        x_fp32: Any,
    ) -> tuple:
        """Join Q/K at the existing fused epilogue, without repeating projections."""
        from rtp_llm.models_py.modules.hybrid import indexer as kernels

        indexer = self.self_attn.indexer
        op = indexer.indexer_op
        params = fmha_impl.fmha_params
        fused = None
        if kernels.fused_hy4_indexer_rope_quant_cache is not None and (
            not fmha_impl.attn_inputs.is_prefill
            or indexer._is_multi_token_decode(fmha_impl.attn_inputs)
        ):
            fused = kernels.fused_hy4_indexer_rope_quant_cache(
                q,
                k,
                params.positions_d,
                op.cos_sin_cache,
                params.slot_mapping,
                op._kv_cache_blocks(kv_cache),
                is_neox_style=op.is_neox_style,
            )
        if fused is None:
            query, key = op.apply_rope_and_rotate_q_k(q, k, params.positions_d)
            fused = op.quant_q_k(query, key, kv_cache, params.slot_mapping)
        q_fp8, q_scale = fused
        weights = indexer._get_logits_head_gate(hidden, q_scale, x_fp32=x_fp32)
        return q_fp8, weights

    def _score_topk(
        self, prepared: tuple, fmha_impl: Any, kv_cache: Any
    ) -> torch.Tensor:
        return self.self_attn.indexer._compute_topk(
            *prepared,
            kv_cache,
            fmha_impl.fmha_params,
            fmha_impl.attn_inputs,
            fmha_impl.cp_params,
        )

    def mla_prologue(
        self,
        hidden: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any,
        prev_topk_indices: Any,
        reuse_topk: bool,
        x_fp8: Any,
        x_scale: Any,
        x_fp32: Any,
    ) -> tuple:
        """Submit the entire DAG here; no callbacks into the ordinary forward."""
        from contextlib import nullcontext

        attn = self.self_attn
        # Publish one shared quantization to both projection branches. Reuse
        # layers still need this representation for the main QKV-A projection.
        if (
            x_fp8 is None
            and x_scale is None
            and attn._reuse_mxfp8_hidden_quant
            and hidden.is_contiguous()
        ):
            x_fp8, x_scale = _mxfp8_quantize_hidden(hidden)
        index_fp8 = x_fp8 if attn._reuse_mxfp8_hidden_quant else None
        index_scale = x_scale if attn._reuse_mxfp8_hidden_quant else None
        indexed = attn.indexer is not None and not reuse_topk
        parallel_tail = indexed and str(self.config.model_type) == "hy_v4"
        parallel_frontend = parallel_tail and self._indexer_frontend_parallel
        defer_score = indexed and self._serialize_score_after_q_path(fmha_impl)
        index_stream = indexer_q_stream = events = None
        if parallel_tail:
            index_stream, indexer_q_stream = self._side_streams(hidden.device)
            if self._events is None:
                self._events = self._new_events(hidden.device)
            events = self._events

        # K only needs normalized hidden; Q will wait for the Q-A result.
        k = None
        if indexed:
            if parallel_frontend:
                events.caller_ready.record()
                index_stream.wait_event(events.caller_ready)
                _record_stream((hidden, index_fp8, index_scale, x_fp32), index_stream)
            with (
                torch.cuda.stream(index_stream) if parallel_frontend else nullcontext()
            ):
                k = self._indexer_k(hidden, index_fp8, index_scale)

        q_c, q_fp8, q_scale, kv = self._qkv_a(hidden, x_fp8, x_scale)
        q_inputs = (q_c, q_fp8, q_scale)
        indexer_prepared = topk = None
        if indexed:
            if parallel_frontend:
                events.q_inputs_ready.record()
                indexer_q_stream.wait_event(events.q_inputs_ready)
                _record_stream(q_inputs, indexer_q_stream)
            with (
                torch.cuda.stream(indexer_q_stream)
                if parallel_frontend
                else nullcontext()
            ):
                indexer_q = self._indexer_q(q_inputs)
                if parallel_frontend:
                    events.indexer_q_ready.record()
            if parallel_frontend:
                # Q projection does NOT wait for K. Only the fused Q/K
                # consumer joins the branches; K is already on this stream.
                index_stream.wait_event(events.indexer_q_ready)
                _record_stream(indexer_q, index_stream)
            with (
                torch.cuda.stream(index_stream) if parallel_frontend else nullcontext()
            ):
                indexer_prepared = self._indexer_post(
                    hidden,
                    indexer_q,
                    k,
                    fmha_impl,
                    kv_cache,
                    x_fp32,
                )
            if parallel_tail and not parallel_frontend:
                events.caller_ready.record()
                index_stream.wait_event(events.caller_ready)
                _record_stream(indexer_prepared, index_stream)
            if not defer_score:
                with (
                    torch.cuda.stream(index_stream) if parallel_tail else nullcontext()
                ):
                    topk = self._score_topk(indexer_prepared, fmha_impl, kv_cache)
                    if parallel_tail:
                        events.indexer_complete.record()

        # Caller stream stays responsible for the main path. At long KV the
        # score tail cannot start until this includes absorbed-Q and Gate.
        mla_inputs, gate = self._main_query(hidden, q_inputs, kv, fmha_impl, kv_cache)
        if defer_score:
            if parallel_tail:
                events.q_path_complete.record()
                index_stream.wait_event(events.q_path_complete)
            with torch.cuda.stream(index_stream) if parallel_tail else nullcontext():
                topk = self._score_topk(indexer_prepared, fmha_impl, kv_cache)
                if parallel_tail:
                    events.indexer_complete.record()
        if parallel_tail:
            caller = torch.cuda.current_stream(hidden.device)
            caller.wait_event(events.indexer_complete)
            _record_stream(topk, caller)
        if reuse_topk:
            topk = prev_topk_indices
        return mla_inputs, gate, topk

    def mla_epilogue(self, output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.modules.hybrid import mla_attention as kernels

        attn = self.self_attn
        output = output.reshape(gate.shape).contiguous()
        if attn._fuse_gated_mla_quant:
            fp8, scale = kernels.sigmoid_mul_fp8_quant_fwd(
                output,
                gate,
                quant_group_size=attn._gated_mla_quant_group_size,
                scale_ue8m0=attn._gated_mla_scale_ue8m0,
                round_scale_to_pow2=attn._gated_mla_round_scale_to_pow2,
                column_major_scales=True,
            )
            return attn.o_proj(fp8, input_scales=scale)
        return attn.o_proj(output * torch.sigmoid(gate))

    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Any = None,
        x_fp8: Any = None,
        x_scale: Any = None,
        prev_topk_indices: Any = None,
        force_reuse_topk_indices: bool = False,
        return_topk: bool = False,
        x_fp32: Any = None,
    ) -> Any:
        if not self.can_run(hidden_states, fmha_impl, kv_cache):
            raise RuntimeError(
                "Unsupported HY4 CMP call must select the ordinary path before execution"
            )
        reuse_topk = self.self_attn.reuse_topk_indices or force_reuse_topk_indices
        if reuse_topk and prev_topk_indices is None:
            raise RuntimeError("HY4 CMP reuse layer requires previous TopK indices")
        if (x_fp8 is None) != (x_scale is None):
            raise ValueError("HY4 CMP FP8 input and scales must be paired")
        mla_inputs, gate, topk = self.mla_prologue(
            hidden_states,
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            reuse_topk,
            x_fp8,
            x_scale,
            x_fp32,
        )
        output = fmha_impl.finish_topk_dependent_forward(mla_inputs, topk)
        output = self.mla_epilogue(output, gate)
        return (output, topk) if return_topk else output


def should_enable_hy4_cmp(
    layers: Any,
    layer_num: int,
    hidden_states: torch.Tensor,
    fmha_impl: Any,
    kv_cache: Any,
) -> bool:
    """Select one HY4 CMP execution mode for the complete model call.

    Per-layer capability decisions make a captured graph depend on whichever
    layer happens to reject the fast path first.  Match GLM5 CMP's contract:
    validate every active layer up front, then either schedule every layer with
    CMP or keep the complete forward on the baseline path.
    """
    if layer_num <= 0:
        return False

    cmps = []
    for layer in layers[:layer_num]:
        layer_cmp = getattr(layer, "hy4_cmp", None)
        if layer_cmp is None or layer_cmp._disabled_reason is not None:
            return False
        cmps.append(layer_cmp)
    if len(cmps) != layer_num:
        return False

    first_cache = kv_cache.get_layer_cache(0) if kv_cache is not None else None
    if (
        cmps[0]._dynamic_disabled_reason(hidden_states, fmha_impl, first_cache)
        is not None
    ):
        return False
    # The first DSA layer seeds request-local TopK.  Later layers may reuse it,
    # but a model call without any initial Indexer is not a complete CMP DAG.
    return getattr(cmps[0].self_attn, "indexer", None) is not None


__all__ = [
    "Hy4Cmp",
    "resolve_hy4_cmp_enabled",
    "should_enable_hy4_cmp",
]
