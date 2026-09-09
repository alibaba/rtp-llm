"""Independent HY4 CMP attention and MoE preparation path.

Target Query/cache preparation and output Gate use a dedicated high-priority
main stream. Indexer K and Q finish their own RoPE/quantization and join only
before score. When the HY4 group-32 native operator is available, QKV-A and
the raw FP32 head gate share one main-stream kernel; otherwise the head gate
keeps the existing Indexer-Q overlap. All branches join back to the caller
before sparse MLA. Score/TopK stays on the K stream; long KV waits for complete
main Query preparation. Indexed target and MTP layers use the same three-stream
schedule; TopK-reuse layers stay on the caller because they have no Indexer
work to overlap.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

_ENABLE_ENV = "RTP_LLM_HY4_CMP"
_INDEXER_FRONTEND_ENV = "RTP_LLM_HY4_CMP_INDEXER_FRONTEND"
_SUPPORTED_MODEL_TYPES = frozenset(("hy_v4", "hy_v4_mtp"))
_MAX_ROWS = 256
logger = logging.getLogger(__name__)


def _load_hy4_ops() -> Any:
    return importlib.import_module("rtp_kernel.hy4")


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
    caller_to_main: torch.cuda.Event
    side_streams_complete: torch.cuda.Event
    frontend_ready: torch.cuda.Event
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
    """Own HY4 stream scheduling, output buffers and prepacked MoE preparation."""

    _streams_by_device: dict[int, tuple[Any, Any, Any]] = {}

    def __init__(
        self,
        *,
        config: Any,
        parallelism_config: Any,
        self_attn: Any,
        mlp: Any = None,
    ) -> None:
        self.config = config
        self.parallelism_config = parallelism_config
        self.self_attn = self_attn
        self.mlp = mlp
        self._indexer_frontend_parallel = _resolve_bool_env(_INDEXER_FRONTEND_ENV)
        self._events: Optional[_Events] = None
        self._qkv_head_gate_initialized = False
        self._qkv_head_gate_ops = None
        self._qkv_head_gate_weight = None
        self._qkv_weight = None
        self._qkv_weight_scale = None
        self._qkv_head_gate_disabled_reason = None
        self._disabled_reason = self._static_disabled_reason()

    def clone_for_cuda_graph(self, *, self_attn: Any, mlp: Any = None) -> "Hy4Cmp":
        clone = object.__new__(type(self))
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.self_attn = self_attn
        clone.mlp = mlp
        clone._indexer_frontend_parallel = self._indexer_frontend_parallel
        clone._qkv_head_gate_initialized = self._qkv_head_gate_initialized
        clone._qkv_head_gate_ops = self._qkv_head_gate_ops
        clone._qkv_head_gate_weight = self._qkv_head_gate_weight
        clone._qkv_weight = self._qkv_weight
        clone._qkv_weight_scale = self._qkv_weight_scale
        clone._qkv_head_gate_disabled_reason = self._qkv_head_gate_disabled_reason
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

    def _initialize_optional_qkv_head_gate(self, ops: Any = None) -> None:
        """Prepare immutable native inputs after the complete call selects CMP."""
        if self._qkv_head_gate_initialized:
            return
        self._qkv_head_gate_initialized = True
        if getattr(self.self_attn, "indexer", None) is None:
            self._qkv_head_gate_disabled_reason = "not an indexed HY4 layer"
            return
        try:
            if ops is None:
                ops = _load_hy4_ops()
            if not callable(getattr(ops, "qkv_a_head_gate", None)) or not callable(
                getattr(ops, "pack_head_gate_weight", None)
            ):
                raise AttributeError("rtp_kernel.hy4 lacks the fused operator")

            qkv = self.self_attn.fused_qkv_a_proj
            weight = getattr(qkv, "weight", None)
            pack_weight_scale = getattr(qkv, "_packed_weight_scale", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.dtype != torch.float8_e4m3fn
                or tuple(weight.shape) != (2624, 6144)
                or not weight.is_contiguous()
                or getattr(qkv, "bias", None) is not None
                or int(getattr(qkv, "input_quant_group_size", 0)) != 32
                or not bool(getattr(qkv, "input_quant_scale_ue8m0", False))
                or not callable(pack_weight_scale)
            ):
                raise ValueError("QKV-A is not the HY4 group-32 MXFP8 layout")
            capability = (
                torch.cuda.get_device_capability(weight.device)
                if weight.is_cuda
                else (0, 0)
            )
            if capability[0] != 10:
                raise ValueError("HY4 fused QKV-A requires SM100")

            weight_scale = pack_weight_scale()
            if (
                not isinstance(weight_scale, torch.Tensor)
                or weight_scale.dtype != torch.int32
                or tuple(weight_scale.shape) != (2624, 48)
                or weight_scale.stride(0) != 1
                or weight_scale.stride(1) != 2624
                or weight_scale.device != weight.device
            ):
                raise ValueError("QKV-A weight scale is not packed group-32 UE8M0")

            head_linear = self.self_attn.indexer.weights_proj
            head_weight = getattr(head_linear, "weight", None)
            if (
                not isinstance(head_weight, torch.Tensor)
                or head_weight.dtype != torch.float32
                or tuple(head_weight.shape) != (32, 6144)
                or head_weight.device != weight.device
                or getattr(head_linear, "bias", None) is not None
            ):
                raise ValueError("raw head gate is not FP32 [32,6144]")
            packed_head_weight = ops.pack_head_gate_weight(head_weight)
            if (
                packed_head_weight.dtype != torch.float32
                or tuple(packed_head_weight.shape) != (96, 64, 32)
                or not packed_head_weight.is_contiguous()
                or packed_head_weight.device != weight.device
            ):
                raise ValueError("raw head-gate weight pack returned an invalid layout")

            self._qkv_head_gate_ops = ops
            self._qkv_head_gate_weight = packed_head_weight
            self._qkv_weight = weight
            self._qkv_weight_scale = weight_scale
        except (
            ImportError,
            OSError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            self._qkv_head_gate_disabled_reason = str(error)
            logger.info("HY4 fused QKV-A/head-gate unavailable: %s", error)

    def _can_fuse_qkv_head_gate(
        self,
        hidden: torch.Tensor,
        x_fp8: Any,
        x_scale: Any,
        buffers: dict[str, Any],
        precleared_raw_gate: Optional[torch.Tensor],
    ) -> bool:
        if self._qkv_head_gate_ops is None:
            return False
        rows = int(hidden.shape[0])
        aligned_rows = (rows + 3) // 4 * 4
        projected = buffers.get("qkv")
        raw_gate = buffers.get("raw_gate")
        return (
            hidden.dtype == torch.bfloat16
            and tuple(hidden.shape) == (rows, 6144)
            and hidden.is_contiguous()
            and isinstance(x_fp8, torch.Tensor)
            and x_fp8.dtype == torch.float8_e4m3fn
            and tuple(x_fp8.shape) == (rows, 6144)
            and x_fp8.is_contiguous()
            and x_fp8.device == hidden.device
            and isinstance(x_scale, torch.Tensor)
            and x_scale.dtype == torch.int32
            and tuple(x_scale.shape) == (rows, 48)
            and x_scale.stride(0) == 1
            and x_scale.stride(1) == aligned_rows
            and x_scale.device == hidden.device
            and isinstance(projected, torch.Tensor)
            and projected.dtype == torch.bfloat16
            and tuple(projected.shape) == (rows, 2624)
            and projected.is_contiguous()
            and projected.device == hidden.device
            and isinstance(raw_gate, torch.Tensor)
            and raw_gate.dtype == torch.float32
            and tuple(raw_gate.shape) == (rows, 32)
            and raw_gate.is_contiguous()
            and raw_gate.device == hidden.device
            and raw_gate is precleared_raw_gate
        )

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
    def _side_streams(cls, device: torch.device) -> tuple[Any, Any, Any]:
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        streams = cls._streams_by_device.get(device_index)
        if streams is None:
            if _is_capturing():
                raise RuntimeError("HY4 CMP streams must be created before capture")
            with torch.cuda.device(device):
                streams = (
                    torch.cuda.Stream(priority=-1),
                    torch.cuda.Stream(),
                    torch.cuda.Stream(),
                )
            cls._streams_by_device[device_index] = streams
        return streams

    @staticmethod
    def _new_events(device: torch.device) -> _Events:
        if _is_capturing():
            raise RuntimeError("HY4 CMP events must be created before capture")
        events = _Events(*(torch.cuda.Event() for _ in range(7)))
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

    def allocate_raw_head_gate_output(
        self,
        source: torch.Tensor,
        *,
        fmha_impl: Any,
        kv_cache: Any,
        force_reuse_topk_indices: bool = False,
    ) -> Optional[torch.Tensor]:
        """Reserve producer-cleared storage only for a complete fused path.

        This preflight deliberately happens before the input producer.  If the
        Indexer split cannot use its fused Q/K frontend, returning ``None``
        keeps the existing producer-generated FP32 hidden for the fallback
        head-gate projection.
        """
        indexer = getattr(self.self_attn, "indexer", None)
        rows = int(source.shape[0]) if source.dim() >= 1 else 0
        if (
            self._qkv_head_gate_ops is None
            or not self._indexer_frontend_parallel
            or indexer is None
            or bool(getattr(self.self_attn, "reuse_topk_indices", False))
            or force_reuse_topk_indices
            or not source.is_cuda
            or source.dim() < 1
            or int(source.shape[-1]) != 6144
            or rows <= 0
            or rows > _MAX_ROWS
            or int(getattr(indexer, "index_n_heads", 0)) != 32
            or (
                rows <= 32
                and getattr(indexer, "_hy4_small_t_head_gate_weight", None)
                is not None
            )
            or not self._can_preallocate_fused_raw_gate(
                rows, source.device, fmha_impl, kv_cache
            )
        ):
            return None
        return torch.empty(
            (rows, 32), device=source.device, dtype=torch.float32
        )

    def _can_preallocate_fused_raw_gate(
        self,
        rows: int,
        device: torch.device,
        fmha_impl: Any,
        kv_cache: Any,
    ) -> bool:
        """Mirror split-Indexer metadata checks without launching or allocating."""
        indexer = self.self_attn.indexer
        q_linear = getattr(indexer, "wq_b", None)
        k_linear = getattr(indexer, "wk", None)
        q_weight = getattr(q_linear, "weight", None)
        k_weight = getattr(k_linear, "weight", None)
        native_weights = (
            self._qkv_weight,
            self._qkv_weight_scale,
            self._qkv_head_gate_weight,
        )

        def output_dtype(weight: Any) -> Optional[torch.dtype]:
            if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                return None
            if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                return weight.dtype
            return torch.bfloat16

        if (
            not bool(getattr(self.self_attn.fused_qkv_a_proj, "supports_out", False))
            or not bool(getattr(q_linear, "supports_out", False))
            or not bool(getattr(k_linear, "supports_out", False))
            or output_dtype(q_weight) != torch.bfloat16
            or output_dtype(k_weight) != torch.bfloat16
            or q_weight.device != device
            or k_weight.device != device
            or any(
                not isinstance(weight, torch.Tensor) or weight.device != device
                for weight in native_weights
            )
            or int(q_weight.shape[0])
            != int(indexer.index_n_heads) * int(indexer.index_head_dim)
            or int(k_weight.shape[0]) != int(indexer.index_head_dim)
            or int(k_weight.shape[1]) != 6144
        ):
            return False

        try:
            params = fmha_impl.fmha_params
            positions = params.positions_d
            slot_mapping = params.slot_mapping
            cos_sin_cache = indexer.indexer_op.cos_sin_cache
            cache = indexer.indexer_op._kv_cache_blocks(kv_cache)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return bool(
            isinstance(positions, torch.Tensor)
            and positions.dim() == 1
            and positions.numel() == rows
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
            and positions.device == device
            and isinstance(slot_mapping, torch.Tensor)
            and slot_mapping.dim() == 1
            and slot_mapping.numel() == rows
            and slot_mapping.dtype == torch.int64
            and slot_mapping.is_contiguous()
            and slot_mapping.device == device
            and isinstance(cos_sin_cache, torch.Tensor)
            and cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.dim() == 2
            and cos_sin_cache.shape[1] >= 64
            and cos_sin_cache.stride(1) == 1
            and cos_sin_cache.device == device
            and isinstance(cache, torch.Tensor)
            and cache.dtype == torch.uint8
            and cache.dim() == 3
            and cache.shape[2] == 132
            and cache.is_contiguous()
            and cache.device == device
        )

    @staticmethod
    def _project(
        linear: Any,
        hidden: torch.Tensor,
        scale: Any = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {} if out is None else {"out": out}
        if scale is not None:
            kwargs["input_scales"] = scale
        return linear(hidden, **kwargs)

    def _allocate_buffers(
        self,
        hidden: torch.Tensor,
        indexed: bool,
        raw_head_gate_output: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Allocate explicit outputs on caller; the final event joins every use."""
        attn = self.self_attn
        rows = hidden.shape[0]
        projections = {
            "qkv": attn.fused_qkv_a_proj,
            "main_q": attn.q_b_proj,
            "gate": attn.gate_proj,
        }
        if indexed:
            projections.update(
                index_q=attn.indexer.wq_b,
                index_k=attn.indexer.wk,
            )
        buffers = {"raw_gate": raw_head_gate_output}
        for name, linear in projections.items():
            if getattr(linear, "supports_out", False):
                dtype = linear.weight.dtype
                if dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    dtype = torch.bfloat16
                buffers[name] = torch.empty(
                    (rows, linear.weight.shape[0]), device=hidden.device, dtype=dtype
                )
            else:
                buffers[name] = None
        if (
            indexed
            and buffers["index_k"] is not None
            and getattr(attn.indexer.k_norm, "supports_out", False)
        ):
            buffers["index_k_norm"] = torch.empty_like(buffers["index_k"])
        buffers["absorbed_q"] = torch.empty(
            (rows, attn.num_heads, attn.kv_lora_rank + attn.qk_rope_head_dim),
            device=hidden.device,
            dtype=(
                buffers["main_q"].dtype
                if buffers["main_q"] is not None
                else hidden.dtype
            ),
        )
        if (
            attn._fuse_q_a_norm_mode == "mxfp8"
            and attn.q_lora_rank <= 8192
            and attn.q_lora_rank % 128 == 0
        ):
            from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
                _allocate_mxfp8_scale,
            )

            buffers["q_a"] = (
                torch.empty(
                    (rows, attn.q_lora_rank),
                    device=hidden.device,
                    dtype=torch.float8_e4m3fn,
                ),
                _allocate_mxfp8_scale(rows, attn.q_lora_rank, hidden.device),
            )
        return buffers

    def _prepare_split_indexer(
        self, buffers: dict[str, Any], fmha_impl: Any, kv_cache: Any
    ) -> bool:
        """Select the independent path before launching either Q or K."""
        indexer = self.self_attn.indexer
        q, k = buffers.get("index_q"), buffers.get("index_k")
        if q is None or k is None:
            return False
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_hy4_indexer_rope_quant import (
            can_fuse_hy4_indexer_rope_quant_cache,
        )

        q = q.view(-1, indexer.index_n_heads, indexer.index_head_dim)
        op = indexer.indexer_op
        params = fmha_impl.fmha_params
        cache = op._kv_cache_blocks(kv_cache)
        if not can_fuse_hy4_indexer_rope_quant_cache(
            q,
            k,
            params.positions_d,
            op.cos_sin_cache,
            params.slot_mapping,
            cache,
            is_neox_style=op.is_neox_style,
        ):
            return False
        # Only the existing FP32 head-gate path may be moved early.
        weight = getattr(indexer.weights_proj, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.float32
            or weight.shape != (indexer.index_n_heads, indexer.wk.weight.shape[1])
        ):
            return False
        if buffers.get("raw_gate") is None:
            buffers["raw_gate"] = (
                torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
                if getattr(indexer.weights_proj, "supports_out", False)
                else None
            )
        buffers["index_q"] = q
        buffers["index_cache"] = cache
        buffers["index_fp8"] = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        buffers["index_scale"] = torch.empty(
            q.shape[:2], device=q.device, dtype=torch.float32
        )
        buffers["head_weights"] = torch.empty_like(buffers["index_scale"])
        return True

    def _indexer_rope_quant(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        fmha_impl: Any,
        buffers: dict[str, Any],
        branch: str,
        raw_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_hy4_indexer_rope_quant import (
            fused_hy4_indexer_rope_quant_cache,
        )

        indexer = self.self_attn.indexer
        op = indexer.indexer_op
        params = fmha_impl.fmha_params
        result = fused_hy4_indexer_rope_quant_cache(
            q,
            k,
            params.positions_d,
            op.cos_sin_cache,
            params.slot_mapping,
            buffers["index_cache"],
            is_neox_style=op.is_neox_style,
            branch=branch,
            out=(buffers["index_fp8"], buffers["index_scale"]),
            raw_head_gate=raw_gate,
            head_weights=buffers["head_weights"] if raw_gate is not None else None,
            head_scale=indexer.softmax_scale * indexer.weights_scale,
        )
        if result is None:
            raise RuntimeError("HY4 Indexer inputs changed after CMP preflight")
        return buffers["index_fp8"], buffers["head_weights"].unsqueeze(-1)

    def _qkv_a(
        self,
        hidden: torch.Tensor,
        x_fp8: Any,
        x_scale: Any,
        buffers: dict,
        fused_raw_gate: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Keep the existing HY4 projection and Q-A normalization numerics."""
        from rtp_llm.models_py.modules.hybrid import mla_attention as kernels

        attn = self.self_attn
        if fused_raw_gate is not None:
            projected, produced_gate = self._qkv_head_gate_ops.qkv_a_head_gate(
                x_fp8,
                x_scale,
                self._qkv_weight,
                self._qkv_weight_scale,
                hidden,
                self._qkv_head_gate_weight,
                out=buffers["qkv"],
                gate_output=fused_raw_gate,
            )
            if produced_gate is not fused_raw_gate:
                raise RuntimeError("HY4 fused QKV-A replaced the raw-gate buffer")
        else:
            projected = self._project(
                attn.fused_qkv_a_proj,
                x_fp8 if x_fp8 is not None else hidden,
                x_scale,
                buffers.get("qkv"),
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
                out=buffers.get("q_a"),
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
        buffers: dict,
    ) -> tuple:
        """Main Q-B, output Gate, and complete Query/cache preparation."""
        from rtp_llm.models_py.modules.hybrid import mla_attention as kernels

        attn = self.self_attn
        q_c, q_fp8, q_scale = q_inputs
        q = self._project(
            attn.q_b_proj,
            q_fp8 if q_fp8 is not None else q_c,
            q_scale if q_fp8 is not None else None,
            buffers.get("main_q"),
        )
        q = q.reshape(-1, attn.num_heads, attn.q_head_dim)
        # Keep the measured policy: no output Gate stream or overlap with MLA.
        gate = self._project(attn.gate_proj, hidden, out=buffers.get("gate"))
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
            q_transformed=buffers["absorbed_q"],
            **norm_kwargs,
        )
        return prepared, gate

    def _indexer_k(
        self,
        hidden: torch.Tensor,
        x_fp8: Any,
        x_scale: Any,
        out: Optional[torch.Tensor] = None,
        norm_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        indexer = self.self_attn.indexer
        k = self._project(
            indexer.wk, x_fp8 if x_fp8 is not None else hidden, x_scale, out
        )
        return (
            indexer.k_norm(k) if norm_out is None else indexer.k_norm(k, out=norm_out)
        )

    def _indexer_q(
        self, q_inputs: tuple, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indexer = self.self_attn.indexer
        q_c, q_fp8, q_scale = q_inputs
        q = self._project(
            indexer.wq_b,
            q_fp8 if q_fp8 is not None else q_c,
            q_scale if q_fp8 is not None else None,
            out.view(out.shape[0], -1) if out is not None else None,
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
        raw_head_gate_output: Optional[torch.Tensor],
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
        parallel_tail = indexed
        parallel_frontend = parallel_tail and self._indexer_frontend_parallel
        defer_score = indexed and self._serialize_score_after_q_path(fmha_impl)
        if raw_head_gate_output is not None and (
            raw_head_gate_output.dtype != torch.float32
            or tuple(raw_head_gate_output.shape) != (hidden.shape[0], 32)
            or not raw_head_gate_output.is_contiguous()
            or raw_head_gate_output.device != hidden.device
        ):
            raise ValueError("invalid precleared HY4 raw head-gate output ABI")
        buffers = self._allocate_buffers(hidden, indexed, raw_head_gate_output)
        # Frontend-off retains the joint launch; splitting it would add a
        # launch without opening an overlap window.
        split_indexer = parallel_frontend and self._prepare_split_indexer(
            buffers, fmha_impl, kv_cache
        )
        fuse_qkv_head_gate = split_indexer and self._can_fuse_qkv_head_gate(
            hidden, x_fp8, x_scale, buffers, raw_head_gate_output
        )
        main_stream = index_stream = indexer_q_stream = events = None
        caller = None
        if parallel_tail:
            main_stream, index_stream, indexer_q_stream = self._side_streams(
                hidden.device
            )
            if self._events is None:
                self._events = self._new_events(hidden.device)
            events = self._events
            caller = torch.cuda.current_stream(hidden.device)
            events.caller_to_main.record()
            main_stream.wait_event(events.caller_to_main)
            _record_stream(
                (hidden, x_fp8, x_scale, x_fp32, raw_head_gate_output), main_stream
            )

        with torch.cuda.stream(main_stream) if parallel_tail else nullcontext():
            # K only needs normalized hidden; Q will wait for the Q-A result.
            k = None
            raw_gate = buffers["raw_gate"] if fuse_qkv_head_gate else None
            if indexed:
                if parallel_frontend:
                    events.frontend_ready.record()
                    index_stream.wait_event(events.frontend_ready)
                    _record_stream(
                        (hidden, index_fp8, index_scale, x_fp32), index_stream
                    )
                with (
                    torch.cuda.stream(index_stream)
                    if parallel_frontend
                    else nullcontext()
                ):
                    k = self._indexer_k(
                        hidden,
                        index_fp8,
                        index_scale,
                        buffers.get("index_k"),
                        buffers.get("index_k_norm"),
                    )
                    if split_indexer:
                        self._indexer_rope_quant(
                            buffers["index_q"], k, fmha_impl, buffers, "k"
                        )
                if split_indexer and not fuse_qkv_head_gate:
                    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
                        project_fp32_logits_head_gate,
                    )

                    if parallel_frontend:
                        indexer_q_stream.wait_event(events.frontend_ready)
                        _record_stream((hidden, x_fp32), indexer_q_stream)
                    with (
                        torch.cuda.stream(indexer_q_stream)
                        if parallel_frontend
                        else nullcontext()
                    ):
                        # Independent of Q-A: overlap FP32 raw head gates with
                        # main-stream QKV-A. Only the final scale fold waits Q.
                        raw_gate = project_fp32_logits_head_gate(
                            hidden,
                            attn.indexer.weights_proj,
                            x_fp32=x_fp32,
                            out=buffers.get("raw_gate"),
                            small_t_weight=getattr(
                                attn.indexer, "_hy4_small_t_head_gate_weight", None
                            ),
                        )

            q_c, q_fp8, q_scale, kv = self._qkv_a(
                hidden,
                x_fp8,
                x_scale,
                buffers,
                raw_gate if fuse_qkv_head_gate else None,
            )
            q_inputs = (q_c, q_fp8, q_scale)
            indexer_prepared = topk = None
            if indexed:
                if parallel_frontend:
                    events.q_inputs_ready.record()
                    indexer_q_stream.wait_event(events.q_inputs_ready)
                    _record_stream((q_inputs, raw_gate), indexer_q_stream)
                with (
                    torch.cuda.stream(indexer_q_stream)
                    if parallel_frontend
                    else nullcontext()
                ):
                    indexer_q = self._indexer_q(q_inputs, buffers.get("index_q"))
                    if split_indexer:
                        indexer_prepared = self._indexer_rope_quant(
                            indexer_q,
                            buffers["index_k"],
                            fmha_impl,
                            buffers,
                            "q",
                            raw_gate,
                        )
                    if parallel_frontend:
                        events.indexer_q_ready.record()
                if parallel_frontend:
                    # Q-ready includes RoPE, quantization and head weights.
                    # K cache is already complete on index; join before score.
                    index_stream.wait_event(events.indexer_q_ready)
                    _record_stream(
                        indexer_prepared if split_indexer else indexer_q, index_stream
                    )
                with (
                    torch.cuda.stream(index_stream)
                    if parallel_frontend
                    else nullcontext()
                ):
                    if not split_indexer:
                        indexer_prepared = self._indexer_post(
                            hidden,
                            indexer_q,
                            k,
                            fmha_impl,
                            kv_cache,
                            x_fp32,
                        )
                if parallel_tail and not parallel_frontend:
                    events.frontend_ready.record()
                    index_stream.wait_event(events.frontend_ready)
                    _record_stream(indexer_prepared, index_stream)
                if not defer_score:
                    with (
                        torch.cuda.stream(index_stream)
                        if parallel_tail
                        else nullcontext()
                    ):
                        topk = self._score_topk(indexer_prepared, fmha_impl, kv_cache)
                        if parallel_tail:
                            events.indexer_complete.record()

            # At long KV the score tail waits for the complete main path. The
            # dependency includes absorbed-Q, cache preparation, and Gate.
            mla_inputs, gate = self._main_query(
                hidden, q_inputs, kv, fmha_impl, kv_cache, buffers
            )
            if defer_score:
                if parallel_tail:
                    events.q_path_complete.record()
                    index_stream.wait_event(events.q_path_complete)
                with (
                    torch.cuda.stream(index_stream) if parallel_tail else nullcontext()
                ):
                    topk = self._score_topk(indexer_prepared, fmha_impl, kv_cache)
                    if parallel_tail:
                        events.indexer_complete.record()
            if parallel_tail:
                main_stream.wait_event(events.indexer_complete)
                events.side_streams_complete.record()
        if parallel_tail:
            caller.wait_event(events.side_streams_complete)
            # TopK and unsupported Linear fallbacks may still allocate on a
            # side stream. Explicit caller-owned buffers were joined above.
            _record_stream(
                (getattr(mla_inputs, "q_transformed", mla_inputs), gate, topk), caller
            )
        if reuse_topk:
            topk = prev_topk_indices
        return mla_inputs, gate, topk

    def moe_prepacked_input_views(self, rows: int) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Return validated plain-MegaMoE views for the HY4 iHC producer."""
        if not getattr(self.mlp, "_hy4_mega_moe_prepack", False):
            return None, None, None, None
        try:
            views = self.mlp.fused_moe.prepacked_input_views(int(rows))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None, None, None, None
        if not isinstance(views, (tuple, list)) or len(views) != 4:
            return None, None, None, None
        activation, scale, topk_ids, topk_weights = views
        if not all(isinstance(value, torch.Tensor) for value in views):
            return None, None, None, None
        expected = (
            (activation, torch.float8_e4m3fn, (rows, self.mlp.hidden_dim)),
            (scale, torch.int32, (rows, self.mlp.hidden_dim // 128)),
            (topk_ids, self.mlp.fused_moe.topk_ids_dtype, (rows, self.mlp.top_k)),
            (topk_weights, torch.float32, (rows, self.mlp.top_k)),
        )
        if any(
            value.dtype != dtype
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or value.device != activation.device
            for value, dtype, shape in expected
        ):
            return None, None, None, None
        return activation, scale, topk_ids, topk_weights

    def _prepare_router(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Write HY4 FP32 routing results directly into MegaMoE buffers."""
        if not getattr(self.mlp, "_hy4_mega_moe_prepack", False):
            raise RuntimeError("HY4 MegaMoE prepack is unsupported")
        from rtp_llm.models_py.modules import GroupTopK

        router_logits = torch.matmul(hidden_states.float(), self.mlp.gate_weight)
        group_topk = GroupTopK()
        group_topk(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            scores=router_logits,
            correction_bias=self.mlp.correction_bias,
            n_group=self.mlp.config.moe_n_group,
            topk_group=self.mlp.config.moe_topk_group,
            topk=self.mlp.top_k,
            renormalize=self.mlp.config.has_moe_norm,
            routed_scaling_factor=self.mlp.config.routed_scaling_factor,
        )

    def forward_target_moe(
        self, channels: torch.Tensor, ihc: Any, norm: Any, *, fuse_mxfp8: bool
    ) -> torch.Tensor:
        """Prepare iHC, shared-expert quantization and routed MoE inputs together."""
        mlp_input_fp8 = None
        mlp_input_scale = None
        mega_activation = None
        mega_scale = None
        routed_indices = None
        routed_weights = None
        prepacked_ihc = None
        if fuse_mxfp8:
            (
                mega_activation,
                mega_scale,
                routed_indices,
                routed_weights,
            ) = self.moe_prepacked_input_views(int(channels.size(0)))
            if mega_activation is not None and mega_scale is not None:
                prepacked_ihc = ihc.pre_normed_mxfp8_to_mega_moe(
                    channels,
                    norm,
                    mega_activation,
                    mega_scale,
                )

        if prepacked_ihc is not None:
            (
                mlp_input,
                mlp_post_gate,
                mlp_input_fp8,
                mlp_input_scale,
            ) = prepacked_ihc
        elif fuse_mxfp8:
            (
                mlp_input,
                mlp_post_gate,
                mlp_input_fp8,
                mlp_input_scale,
            ) = ihc.pre_normed_mxfp8(channels, norm)
        else:
            mlp_input, mlp_post_gate = ihc.pre_normed(channels, norm)
        if prepacked_ihc is not None:
            assert routed_indices is not None and routed_weights is not None
            self._prepare_router(mlp_input, routed_indices, routed_weights)
            mlp_output = self.mlp.forward_prepacked(
                mlp_input,
                routed_indices,
                routed_weights,
                x_fp8=mlp_input_fp8,
                x_scale=mlp_input_scale,
            )
        elif mlp_input_fp8 is not None and mlp_input_scale is not None:
            mlp_output = self.mlp(
                mlp_input, x_fp8=mlp_input_fp8, x_scale=mlp_input_scale
            )
        else:
            mlp_output = self.mlp(mlp_input)
        channels = ihc.post(mlp_output, channels, mlp_post_gate)
        return channels

    def forward_mtp_moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: Any,
        *,
        fuse_mxfp8: bool,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Prepare MTP residual/norm and MoE buffers, or select normal fallback."""
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )

        if fuse_mxfp8 and hidden_states.dim() == 2:
            mega_fp8, mega_scale, topk_ids, topk_weights = (
                self.moe_prepacked_input_views(int(hidden_states.size(0)))
            )
            if mega_fp8 is not None and mega_scale is not None:
                assert topk_ids is not None and topk_weights is not None
                bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                    hidden_states,
                    residual,
                    norm.weight.data,
                    norm.variance_epsilon,
                    group_size=32,
                    scale_ue8m0=True,
                    mxfp8_semantics=True,
                    mega_mxfp8_out=mega_fp8,
                    mega_mxfp8_scale_out=mega_scale,
                )
                self._prepare_router(bf16_hs, topk_ids, topk_weights)
                hidden_states = self.mlp.forward_prepacked(
                    bf16_hs,
                    topk_ids,
                    topk_weights,
                    x_fp8=fp8_hs,
                    x_scale=scale,
                )
                return hidden_states, residual
        return None

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
        raw_head_gate_output: Optional[torch.Tensor] = None,
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
            raw_head_gate_output,
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
    if getattr(cmps[0].self_attn, "indexer", None) is None:
        return False

    # Optional native fusion must be initialized before any CUDA Graph capture.
    # A missing or old rtp-kernel wheel only disables this producer fusion; the
    # existing HY4 CMP schedule remains the complete fallback.
    uninitialized = [
        cmp for cmp in cmps if not cmp._qkv_head_gate_initialized
    ]
    if uninitialized and not _is_capturing():
        try:
            ops = _load_hy4_ops()
        except (ImportError, OSError, AttributeError, RuntimeError) as error:
            for cmp in uninitialized:
                cmp._qkv_head_gate_initialized = True
                cmp._qkv_head_gate_disabled_reason = str(error)
            logger.info("HY4 fused QKV-A/head-gate provider unavailable: %s", error)
        else:
            for cmp in uninitialized:
                cmp._initialize_optional_qkv_head_gate(ops)
    return True


__all__ = [
    "Hy4Cmp",
    "resolve_hy4_cmp_enabled",
    "should_enable_hy4_cmp",
]
