"""HY4 fixed-ABI CMP: one model preflight, three plans, no in-DAG fallback.

The fixed Indexer K consumes BF16 hidden states, preserving the validated HY4
reference semantics. Native group-32 QKV/raw-gate is mandatory above M=32.
Native producers publish the three PDL boundaries; each native consumer waits
before its first dependent read. Ordinary events retain caller and lifetime joins.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import torch

_ENABLE_ENV = "RTP_LLM_HY4_CMP"
_SUPPORTED_MODEL_TYPES = frozenset(("hy_v4", "hy_v4_mtp"))
_MAX_ROWS = 256


def _load_hy4_ops():
    return importlib.import_module("rtp_kernel.hy4")


def resolve_hy4_cmp_enabled():
    value = os.environ.get(_ENABLE_ENV, "1").strip().lower()
    if value not in ("1", "true", "yes", "on", "0", "false", "no", "off", ""):
        raise ValueError(f"invalid {_ENABLE_ENV}={value!r}")
    return value in ("1", "true", "yes", "on")


def _is_capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class Hy4CmpPlan(IntEnum):
    SMALL_T = 0
    NATIVE_G32 = 1
    REUSE_TOPK = 2


def select_fixed_plan(rows: int, reuse_topk: bool) -> Hy4CmpPlan:
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError("HY4 CMP requires 1..256 rows")
    if reuse_topk:
        return Hy4CmpPlan.REUSE_TOPK
    return Hy4CmpPlan.SMALL_T if rows <= 32 else Hy4CmpPlan.NATIVE_G32


@dataclass
class _Events:
    caller_to_main: Any
    side_streams_complete: Any
    frontend_ready: Any
    frontend_complete: Any
    q_inputs_ready: Any
    indexer_q_ready: Any
    q_path_complete: Any
    indexer_complete: Any


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for tensor in value:
            _record_stream(tensor, stream)


def _page_table(inputs):
    table = inputs.kv_cache_kernel_block_id_device
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        table = inputs.kv_cache_block_id_device
    return table


def _tensor(value, shape, dtype, device, *, contiguous=True):
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(shape)
        and value.dtype == dtype
        and value.device == device
        and (not contiguous or value.is_contiguous())
    )


class Hy4Cmp:
    _streams_by_device = {}

    def __init__(self, *, config, parallelism_config, self_attn, mlp=None):
        self.config = config
        self.parallelism_config = parallelism_config
        self.self_attn = self_attn
        self.mlp = mlp
        self._draft_prefill_clone = False
        self._events = None
        self._initialized = False
        self._plan = None
        self._positions = None
        self._ops = None
        self._packed_head_weight = None
        self._small_head_weight = None
        self._index_k_weight = None
        self._output_bmm_weight = None
        self._router_weight = None
        self._topk_row_starts = None
        self._weight_scales = {}
        self._disabled_reason = self._static_disabled_reason()

    def _static_disabled_reason(self):
        a, p = self.self_attn, self.parallelism_config
        if self.config.model_type not in _SUPPORTED_MODEL_TYPES:
            return "unsupported model"
        if p.tp_size != 1 or p.get_attn_tp_size() != 1:
            return "HY4 CMP requires TP=1"
        shape = (
            a.num_heads,
            a.q_lora_rank,
            a.kv_lora_rank,
            a.qk_nope_head_dim,
            a.qk_rope_head_dim,
            a.v_head_dim,
            a.token_per_block,
        )
        if shape != (64, 2048, 512, 192, 64, 256, 64):
            return "unsupported attention shape"
        if a.gating_type != "elementwise" or a._fuse_q_a_norm_mode != "mxfp8":
            return "unsupported gate or Q-A norm"
        if not (
            a._fuse_kv_a_norm
            and a._reuse_mxfp8_hidden_quant
            and a._fuse_gated_mla_quant
            and a._gated_mla_quant_group_size == 32
        ):
            return "required HY4 fusions are disabled"
        i = a.indexer
        if i is not None and (
            i.use_hadamard
            or i.index_n_heads != 32
            or i.index_head_dim != 128
            or i.index_topk != 2048
            or not i.wk_bf16_input
            or not i.bf16_compute
        ):
            return "unsupported Indexer ABI"
        return None

    def _validate_weight_abi(self):
        a = self.self_attn
        device = a.fused_qkv_a_proj.weight.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 10:
            raise ValueError("HY4 native group-32 requires an SM100 device")
        projections = dict(
            qkv=(a.fused_qkv_a_proj, 2624, 6144),
            q_b=(a.q_b_proj, 16384, 2048),
            o=(a.o_proj, 6144, 16384),
        )
        if a.indexer is not None:
            projections["index_q"] = (a.indexer.wq_b, 4096, 2048)
        scales = {}
        for name, (linear, n, k) in projections.items():
            if not (
                _tensor(linear.weight, (n, k), torch.float8_e4m3fn, device)
                and linear.bias is None
                and linear.input_quant_group_size == 32
                and linear.input_quant_scale_ue8m0
                and linear.supports_out
            ):
                raise ValueError(f"HY4 {name} projection violates group-32 ABI")
            scale = linear._packed_weight_scale()
            if not (
                _tensor(scale, (n, k // 128), torch.int32, device, contiguous=False)
                and scale.stride() == (1, (n + 3) // 4 * 4)
            ):
                raise ValueError(f"HY4 {name} scale violates column-major ABI")
            scales[name] = scale
        for norm, width in [(a.q_a_layernorm, 2048), (a.kv_a_layernorm, 512)]:
            if not _tensor(norm.weight, (width,), torch.bfloat16, device):
                raise ValueError("HY4 norm weight ABI mismatch")
        if not _tensor(a.attn_sink, (64,), torch.float32, device):
            raise ValueError("HY4 learnable sink ABI mismatch")
        if not (
            _tensor(
                a.gate_proj.weight,
                (16384, 6144),
                torch.bfloat16,
                device,
                contiguous=False,
            )
            and a.gate_proj.supports_out
            and a.gate_proj.bias is None
        ):
            raise ValueError("HY4 MLA output gate ABI mismatch")
        if a.indexer is not None:
            i = a.indexer
            if not (
                _tensor(
                    i.wk.weight, (128, 6144), torch.bfloat16, device, contiguous=False
                )
                and i.wk.supports_out
                and i.k_norm.supports_out
                and _tensor(i.k_norm.weight, (128,), torch.bfloat16, device)
                and _tensor(i.k_norm.beta, (128,), torch.bfloat16, device)
                and _tensor(
                    i.weights_proj.weight,
                    (32, 6144),
                    torch.float32,
                    device,
                    contiguous=False,
                )
            ):
                raise ValueError("HY4 Indexer BF16-K/FP32-head ABI mismatch")
        return scales

    def initialize_for_cmp(self, ops):
        """Bind the mandatory native provider before stream capture."""
        if self._initialized:
            return
        if _is_capturing():
            raise RuntimeError("HY4 CMP provider must be initialized before capture")
        if not self._weight_scales:
            self._weight_scales = self._validate_weight_abi()
        a = self.self_attn
        device = a.fused_qkv_a_proj.weight.device
        if a.indexer is not None:
            i = a.indexer
            if not (
                callable(ops.qkv_a_head_gate_partials)
                and callable(ops.pack_head_gate_weight)
            ):
                raise RuntimeError("rtp_kernel.hy4 fixed ABI is incomplete")
            self._index_k_weight = i.wk.weight.detach().contiguous()
            self._topk_row_starts = torch.zeros(256, device=device, dtype=torch.int32)
            self._small_head_weight = i.weights_proj.weight.detach().contiguous()
            packed = ops.pack_head_gate_weight(i.weights_proj.weight)
            if not _tensor(packed, (96, 64, 32), torch.float32, device):
                raise ValueError("HY4 native packed head weight ABI mismatch")
            self._packed_head_weight = packed
        for function in (
            ops.output_bmm_gate_quant,
            ops.router_proj,
            ops.router_topk,
            ops.q_b_proj,
            ops.qkv_post,
            ops.mtp_norm_quant,
            ops.target_norm_quant,
            ops.indexer_k_cache,
            ops.indexer_q_proj,
            ops.indexer_q_post,
            ops.indexer_q_post_partials,
            ops.canonicalize_topk,
            ops.get_pdl,
        ):
            if not callable(function):
                raise RuntimeError("rtp_kernel.hy4 native PDL ABI is incomplete")
        if self.mlp is not None and (
            self.config.model_type == "hy_v4_mtp" or a.layer_idx != 0
        ):
            self._router_weight = self.mlp.gate_weight.detach().t().contiguous()
        self._ops = ops
        self._initialized = True

    def clone_for_cuda_graph(self, *, self_attn, mlp=None, draft_prefill=False):
        clone = object.__new__(type(self))
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone._initialized = self._initialized
        clone._plan = None
        clone._positions = None
        clone._disabled_reason = self._disabled_reason
        clone._ops = self._ops
        clone._packed_head_weight = self._packed_head_weight
        clone._small_head_weight = self._small_head_weight
        clone._index_k_weight = self._index_k_weight
        clone._output_bmm_weight = self._output_bmm_weight
        clone._router_weight = self._router_weight
        clone._topk_row_starts = self._topk_row_starts
        clone._weight_scales = self._weight_scales
        clone.self_attn = self_attn
        clone.mlp = mlp
        clone._draft_prefill_clone = bool(draft_prefill)
        clone._events = None
        return clone

    def _dynamic_disabled_reason(
        self, hidden, fmha, cache, *, require_indexer=True, page_table=None
    ):
        rows = hidden.shape[0] if hidden.ndim else 0
        device = hidden.device
        if not (
            hidden.is_cuda
            and hidden.ndim == 2
            and hidden.shape[1] == 6144
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
            and 1 <= rows <= 256
        ):
            return "unsupported activation ABI"
        if cache is None or not fmha.is_sparse() or fmha.cp_params is not None:
            return "unsupported sparse cache or CP preparation"
        if not (
            fmha.supports_topk_late_binding and callable(fmha.can_fuse_kv_norm_cache)
        ):
            return "late-bound TopK or fused KV norm/cache is unavailable"
        if not fmha.can_fuse_kv_norm_cache(
            hidden[:, :512], self.self_attn.kv_a_layernorm.weight
        ):
            return "backend cannot fuse KV norm/cache"
        inputs, params = fmha.attn_inputs, fmha.fmha_params
        table = _page_table(inputs) if page_table is None else page_table
        if not (
            isinstance(table, torch.Tensor)
            and table.ndim == 2
            and table.shape[0] > 0
            and table.shape[1] > 0
            and rows % table.shape[0] == 0
            and table.device == device
            and table.dtype == torch.int32
            and table.is_contiguous()
        ):
            return "invalid request page table"
        width = rows // table.shape[0]
        if inputs.is_prefill and not inputs.is_target_verify:
            if not (
                self.config.model_type == "hy_v4_mtp"
                and self._draft_prefill_clone
                and width == self.config.gen_num_per_cycle + 1
                and width > 1
            ):
                return "ordinary prefill or invalid MTP draft width"
        multi_token = (
            inputs.is_target_verify
            or inputs.is_draft_extend
            or (self._draft_prefill_clone and inputs.is_prefill)
        )
        if not multi_token and (
            width != 1
            or not _tensor(params.kvlen_d, (rows,), torch.int32, device)
            or not _tensor(inputs.decode_cu_seqlens_d, (rows + 1,), torch.int32, device)
        ):
            return "invalid single-token decode metadata"
        if not _tensor(params.slot_mapping, (rows,), torch.int64, device):
            return "invalid slot_mapping"
        if not _tensor(params.expanded_seq_lens, (rows,), torch.int32, device):
            return "invalid expanded_seq_lens"
        positions = params.positions_d
        if not (
            isinstance(positions, torch.Tensor)
            and positions.dtype in (torch.int32, torch.int64)
            and _tensor(positions, (rows,), positions.dtype, device)
        ):
            return "invalid positions"
        working_entry = fmha.pinned_mla_groups.get(self.self_attn.layer_idx)
        mla_cache = cache.kv_cache_base
        if working_entry is not None:
            working, group_layer = working_entry
            if (
                cache.kv_cache_base.data_ptr()
                != working.backing[group_layer].data_ptr()
                or working.capacity < rows * 2048
            ):
                return "invalid tiered MLA backing or resident capacity"
            mla_cache = working.resident[group_layer]
        if not (
            isinstance(mla_cache, torch.Tensor)
            and mla_cache.ndim == 3
            and mla_cache.shape[1] == 64
            and mla_cache.device == device
            and mla_cache.dtype == torch.uint8
            and mla_cache.is_contiguous()
            and mla_cache.shape[-1] == 656
        ):
            return "invalid packed sparse MLA cache"
        cosine = fmha._cos_sin_cache
        if not (
            isinstance(cosine, torch.Tensor)
            and cosine.device == device
            and cosine.ndim == 2
            and cosine.shape[1] == 64
            and cosine.is_contiguous()
            and cosine.dtype == torch.float32
            and not fmha._is_neox_style
        ):
            return "invalid MLA RoPE cache"
        if require_indexer and self.self_attn.indexer is not None:
            op = self.self_attn.indexer.indexer_op
            if not isinstance(cache.kv_scale_base, torch.Tensor):
                return "missing Indexer cache"
            index_cache = op._kv_cache_blocks(cache)
            cosine = op.cos_sin_cache
            if not (
                index_cache.device == device
                and index_cache.dtype == torch.uint8
                and index_cache.ndim == 3
                and index_cache.shape[1:] == (64, 132)
                and index_cache.is_contiguous()
                and not op.is_neox_style
                and cosine.device == device
                and cosine.dtype == torch.float32
                and cosine.ndim == 2
                and cosine.shape[1] >= 64
                and cosine.stride(1) == 1
            ):
                return "invalid Indexer cache ABI"
        return None

    def select_plan(self, rows, force_reuse_topk_indices=False):
        return select_fixed_plan(
            rows, self.self_attn.reuse_topk_indices or force_reuse_topk_indices
        )

    def allocate_raw_head_gate_output(self, source, *, force_reuse_topk_indices=False):
        if (
            self.select_plan(source.shape[0], force_reuse_topk_indices)
            != Hy4CmpPlan.NATIVE_G32
        ):
            return None
        return torch.empty(
            (source.shape[0], 32), dtype=torch.float32, device=source.device
        )

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
        events = _Events(*(torch.cuda.Event() for _ in range(8)))
        with torch.cuda.device(device):
            for event in (
                events.caller_to_main,
                events.frontend_ready,
                events.frontend_complete,
                events.q_inputs_ready,
                events.indexer_q_ready,
                events.q_path_complete,
                events.indexer_complete,
                events.side_streams_complete,
            ):
                event.record()
        return events

    def _serialize_score_after_q_path(self, fmha_impl):
        inputs = fmha_impl.attn_inputs
        table = inputs.kv_cache_kernel_block_id_device
        page_size = 64
        if table is None or table.numel() == 0:
            table = inputs.kv_cache_block_id_device
            page_size = self.self_attn.attn_config.tokens_per_block
        return table.shape[1] * page_size >= (1 << 19)

    def _allocate_buffers(self, hidden, indexed, raw_head_gate_output=None):
        from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
            _allocate_mxfp8_scale,
        )

        rows, device = hidden.shape[0], hidden.device

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)

        buffers = dict(
            qkv=empty((rows, 2624)),
            main_q=empty((rows, 64, 192)),
            gate=empty((rows, 16384)),
            absorbed_q=empty((rows, 64, 576)),
            q_a=(
                empty((rows, 2048), torch.float8_e4m3fn),
                _allocate_mxfp8_scale(rows, 2048, device),
            ),
        )
        if indexed:
            buffers.update(
                raw_gate=(
                    raw_head_gate_output
                    if raw_head_gate_output is not None
                    else empty((rows, 32), torch.float32)
                ),
                gate_partials=(
                    empty((96, rows, 32), torch.float32) if rows > 32 else None
                ),
                index_q=empty((rows, 32, 128)),
                index_fp8=empty((rows, 32, 128), torch.float8_e4m3fn),
                index_scale=empty((rows, 32), torch.float32),
                head_weights=empty((rows, 32), torch.float32),
            )
        return buffers

    def _qkv_a(
        self,
        hidden,
        x_fp8,
        x_scale,
        buffers,
        fmha,
        cache,
        native_gate=False,
        notify_event=0,
    ):
        a = self.self_attn
        if native_gate:
            self._ops.qkv_a_head_gate_partials(
                x_fp8,
                x_scale,
                a.fused_qkv_a_proj.weight,
                self._weight_scales["qkv"],
                hidden,
                self._packed_head_weight,
                out=buffers["qkv"],
                gate_output=buffers["gate_partials"],
            )
        else:
            a.fused_qkv_a_proj(x_fp8, input_scales=x_scale, out=buffers["qkv"])
        self._ops.qkv_post(
            buffers["qkv"],
            a.q_a_layernorm.weight,
            a.kv_a_layernorm.weight,
            fmha._cos_sin_cache,
            buffers["positions"],
            buffers["mla_slots"],
            *buffers["q_a"],
            buffers["mla_cache"],
            a.q_a_layernorm.variance_epsilon,
            a.kv_a_layernorm.variance_epsilon,
            self._ops.get_pdl(),
            notify_event,
        )
        return buffers["q_a"]

    def _main_query(self, hidden, q_inputs, fmha, cache, buffers):
        a = self.self_attn
        self._ops.q_b_proj(
            *q_inputs,
            a.q_b_proj.weight,
            self._weight_scales["q_b"],
            fmha._cos_sin_cache,
            buffers["positions"],
            buffers["main_q"],
            buffers["absorbed_q"],
        )
        a.gate_proj(hidden, out=buffers["gate"])
        prepared = fmha.prepare_hy4_native_query(
            buffers["main_q"],
            buffers["absorbed_q"],
            cache,
            a.layer_idx,
            a.attn_sink,
        )
        return prepared, buffers["gate"]

    def _small_raw_gate(self, hidden, out):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
            _fused_logits_head_gate_small_t_kernel,
        )

        _fused_logits_head_gate_small_t_kernel[(hidden.shape[0], 32)](
            hidden,
            self._small_head_weight,
            out,
            out,
            1.0,
            6144,
            hidden.stride(0),
            6144,
            1,
            32,
            32,
            BLOCK_K=8192,
            APPLY_SCALE=False,
            num_warps=4,
        )

    def _prepare_score(self, rows, fmha, cache):
        import deep_gemm

        # Metadata producers run on caller before the CMP DAG. After the Q
        # programmatic event, the first submitted consumer is the scoring GEMM.
        table = _page_table(fmha.attn_inputs)
        width = rows // table.shape[0]
        if width > 1:
            table = table.repeat_interleave(width, dim=0, output_size=rows)
        lengths = fmha.fmha_params.expanded_seq_lens
        kvlen = lengths.view(rows, 1)
        metadata = deep_gemm.get_paged_mqa_logits_metadata(
            kvlen, 64, deep_gemm.get_num_sms()
        )
        kv = self.self_attn.indexer.indexer_op._kv_cache_blocks(cache)
        return kv.view(-1, 64, 1, 132), table, kvlen, metadata, lengths

    def _score_topk(self, prepared, score_plan):
        import deep_gemm

        q, weights = prepared
        kv, table, kvlen, metadata, lengths = score_plan
        max_length = table.shape[1] * 64
        logits = deep_gemm.fp8_paged_mqa_logits(
            q.unsqueeze(1),
            kv,
            weights.view(-1, 32),
            kvlen,
            table,
            metadata,
            max_length,
            clean_logits=False,
        )
        topk = torch.empty((q.shape[0], 2048), device=q.device, dtype=torch.int32)
        self.self_attn.indexer.indexer_op.hy4_topk(
            logits, self._topk_row_starts[: q.shape[0]], lengths, topk, max_length
        )
        self._ops.canonicalize_topk(topk)
        return topk

    def mla_prologue(
        self, source, fmha, cache, prev_topk, plan, producer, producer_inputs, raw_gate
    ):
        indexed = plan != Hy4CmpPlan.REUSE_TOPK
        buffers = self._allocate_buffers(source, indexed, raw_gate)
        # Canonicalize metadata once on caller, before any side-stream launch.
        buffers["positions"] = self._positions
        working_entry = fmha.pinned_mla_groups.get(self.self_attn.layer_idx)
        buffers["mla_cache"] = cache.kv_cache_base
        buffers["mla_slots"] = fmha.fmha_params.slot_mapping
        if working_entry is not None:
            rows = source.shape[0]
            buffers["mla_cache"] = torch.empty(
                ((rows + 63) // 64, 64, 656), device=source.device, dtype=torch.uint8
            )
            buffers["mla_slots"] = torch.arange(
                rows, device=source.device, dtype=torch.int64
            )
        if not indexed:
            if working_entry is not None:
                fmha.prefetch_kv(self.self_attn.layer_idx, prev_topk)
            hidden, x_fp8, x_scale = producer(0)
            q_inputs = self._qkv_a(hidden, x_fp8, x_scale, buffers, fmha, cache)
            prepared, gate = self._main_query(hidden, q_inputs, fmha, cache, buffers)
            if working_entry is not None:
                fmha.finish_hy4_cache_write(prepared, cache, buffers["mla_cache"])
            return prepared, gate, prev_topk

        i = self.self_attn.indexer
        score_plan = self._prepare_score(source.shape[0], fmha, cache)
        buffers["index_cache"] = i.indexer_op._kv_cache_blocks(cache)
        main, index, index_q = self._side_streams(source.device)
        events = self._events
        if events is None:
            raise RuntimeError("HY4 events must be initialized by model preflight")
        caller = torch.cuda.current_stream(source.device)
        events.caller_to_main.record(caller)
        main.wait_event(events.caller_to_main)
        _record_stream(producer_inputs, main)
        defer_score = self._serialize_score_after_q_path(fmha)
        with torch.cuda.stream(main):
            hidden, x_fp8, x_scale = producer(events.frontend_ready.cuda_event)
            index.wait_event(events.frontend_ready)
            with torch.cuda.stream(index):
                _record_stream(hidden, index)
                self._ops.indexer_k_cache(
                    hidden,
                    self._index_k_weight,
                    i.k_norm.weight,
                    i.k_norm.beta,
                    i.indexer_op.cos_sin_cache,
                    buffers["positions"],
                    fmha.fmha_params.slot_mapping,
                    buffers["index_cache"],
                    i.k_norm.variance_epsilon,
                )
            if plan == Hy4CmpPlan.SMALL_T:
                # The retained small-T Triton gate has no GDC wait.
                events.frontend_complete.record()
                index_q.wait_event(events.frontend_complete)
                with torch.cuda.stream(index_q):
                    _record_stream(hidden, index_q)
                    self._small_raw_gate(hidden, buffers["raw_gate"])
            q_inputs = self._qkv_a(
                hidden,
                x_fp8,
                x_scale,
                buffers,
                fmha,
                cache,
                notify_event=events.q_inputs_ready.cuda_event,
                native_gate=plan == Hy4CmpPlan.NATIVE_G32,
            )
            index_q.wait_event(events.q_inputs_ready)
            with torch.cuda.stream(index_q):
                self._ops.indexer_q_proj(
                    *q_inputs,
                    i.wq_b.weight,
                    self._weight_scales["index_q"],
                    buffers["index_q"].view(-1, 4096),
                )
                post_inputs = (
                    buffers["index_q"],
                    i.indexer_op.cos_sin_cache,
                    buffers["positions"],
                    buffers["raw_gate"],
                )
                post_outputs = (
                    buffers["index_fp8"],
                    buffers["index_scale"],
                    buffers["head_weights"],
                    events.indexer_q_ready.cuda_event,
                )
                if plan == Hy4CmpPlan.SMALL_T:
                    self._ops.indexer_q_post(*post_inputs, *post_outputs)
                else:
                    self._ops.indexer_q_post_partials(
                        *post_inputs, buffers["gate_partials"], *post_outputs
                    )
                score_inputs = (
                    buffers["index_fp8"],
                    buffers["head_weights"].unsqueeze(-1),
                )
            index.wait_event(events.indexer_q_ready)
            if not defer_score:
                with torch.cuda.stream(index):
                    topk = self._score_topk(score_inputs, score_plan)
                    if working_entry is not None:
                        fmha.prefetch_kv(self.self_attn.layer_idx, topk)
                    events.indexer_complete.record()
            prepared, gate = self._main_query(hidden, q_inputs, fmha, cache, buffers)
            if defer_score:
                events.q_path_complete.record()
                index.wait_event(events.q_path_complete)
                with torch.cuda.stream(index):
                    topk = self._score_topk(score_inputs, score_plan)
                    if working_entry is not None:
                        fmha.prefetch_kv(self.self_attn.layer_idx, topk)
                    events.indexer_complete.record()
            main.wait_event(events.indexer_complete)
            events.side_streams_complete.record()
        caller.wait_event(events.side_streams_complete)
        # Unlike fixed buffers, TopK is allocated on index. Event ordering
        # alone does not protect its storage from caching-allocator reuse.
        topk.record_stream(caller)
        if working_entry is not None:
            fmha.finish_hy4_cache_write(prepared, cache, buffers["mla_cache"])
        return prepared, gate, topk

    def moe_prepacked_input_views(self, rows):
        views = self.mlp.fused_moe.prepacked_input_views(rows)
        device = self.mlp.gate_weight.device
        expected = [
            ((rows, 6144), torch.float8_e4m3fn),
            ((rows, 48), torch.int32),
            ((rows, 8), torch.int64),
            ((rows, 8), torch.float32),
        ]
        if len(views) != 4 or any(
            not _tensor(v, shape, dtype, device)
            for v, (shape, dtype) in zip(views, expected)
        ):
            raise RuntimeError("HY4 MegaMoE prepacked ABI mismatch")
        return views

    def _prepare_router(self, hidden, ids, weights):
        logits = torch.empty(
            (hidden.shape[0], 256), device=hidden.device, dtype=torch.float32
        )
        self._ops.router_proj(hidden, self._router_weight, logits)
        self._ops.router_topk(logits, self.mlp.correction_bias, ids, weights)

    def _target_producer(self, channels, ihc, norm, raw_gate=None, mega=None, out=None):
        from rtp_llm.models_py.modules.hy_v4.ihc import (
            maybe_fused_ihc_pre_normed_grouped,
        )

        extra = (
            {}
            if mega is None
            else dict(mega_mxfp8_out=mega[0], mega_mxfp8_scale_out=mega[1])
        )
        result = maybe_fused_ihc_pre_normed_grouped(
            channels,
            ihc.fn_weight,
            ihc.scale,
            ihc.base,
            norm.weight,
            magnitude=ihc.magnitude,
            hc_eps=ihc.hc_eps,
            ihc_norm_eps=ihc.norm_eps,
            read_norm_eps=norm.variance_epsilon,
            chunk_size=ihc.chunk_size,
            emit_mxfp8=True,
            raw_gate_clear_out=raw_gate,
            out=out,
            **extra,
        )
        if result is None:
            raise RuntimeError("HY4 iHC producer invariant changed after preflight")
        return result

    def target_input(self, channels, ihc, norm):
        raw_gate = self.allocate_raw_head_gate_output(channels)
        return (*self._target_producer(channels, ihc, norm, raw_gate), raw_gate)

    def mtp_input(self, hidden, residual, norm, force_reuse_topk_indices=False):
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )

        raw_gate = self.allocate_raw_head_gate_output(
            hidden, force_reuse_topk_indices=force_reuse_topk_indices
        )
        values = fused_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden,
            residual,
            norm.weight,
            norm.variance_epsilon,
            group_size=32,
            scale_ue8m0=True,
            mxfp8_semantics=True,
            raw_gate_clear_out=raw_gate,
        )
        return (*values, raw_gate)

    def _allocate_frontend(self, source, *, target):
        from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
            _allocate_mxfp8_scale,
        )

        if self._plan is None:
            raise RuntimeError("HY4 CMP plan must be selected at model entry")
        rows, device = source.shape[0], source.device
        hidden = torch.empty((rows, 6144), device=device, dtype=torch.bfloat16)
        fp8 = torch.empty((rows, 6144), device=device, dtype=torch.float8_e4m3fn)
        scale = _allocate_mxfp8_scale(rows, 6144, device)
        raw_gate = torch.empty((rows, 32), device=device, dtype=torch.float32)
        post_gate = (
            torch.empty((rows, 4), device=device, dtype=torch.float32)
            if target
            else None
        )
        return hidden, fp8, scale, raw_gate, post_gate

    def _finish_attention(self, prepared, gate, topk, fmha):
        latent = fmha.finish_hy4_native_attention(prepared, topk)
        rows = latent.shape[0]
        fp8 = torch.empty(
            (rows, 16384), device=latent.device, dtype=torch.float8_e4m3fn
        )
        scale = torch.empty(
            (128, (rows + 3) // 4 * 4), device=latent.device, dtype=torch.int32
        ).t()[:rows]
        self._ops.output_bmm_gate_quant(
            latent, self._output_bmm_weight, gate, fp8, scale
        )
        return self.self_attn.o_proj(fp8, input_scales=scale), topk

    def forward_target_attention(
        self, channels, ihc, norm, fmha, cache, prev_topk=None
    ):
        hidden, fp8, scale, raw_gate, post_gate = self._allocate_frontend(
            channels, target=True
        )

        import deep_gemm
        from rtp_llm.models_py.modules.hy_v4.ihc_triton import _deepgemm_num_splits

        rows = channels.shape[0]
        splits = _deepgemm_num_splits(rows, 24576, channels.device.index)
        partials = torch.empty(
            (splits, rows, 8), device=channels.device, dtype=torch.float32
        )
        squares = torch.empty(
            (splits, rows), device=channels.device, dtype=torch.float32
        )

        def produce(notify_event):
            deep_gemm.tf32_hc_prenorm_gemm(
                channels.view(rows, 24576), ihc.fn_weight, partials, squares, splits
            )
            self._ops.target_norm_quant(
                channels,
                partials,
                squares,
                ihc.scale,
                ihc.base,
                norm.weight,
                hidden,
                fp8,
                scale,
                raw_gate,
                post_gate,
                ihc.norm_eps,
                norm.variance_epsilon,
                ihc.hc_eps,
                ihc.magnitude,
                notify_event,
            )
            return hidden, fp8, scale

        prepared, gate, topk = self.mla_prologue(
            channels,
            fmha,
            cache,
            prev_topk,
            self._plan,
            produce,
            (channels, partials, squares),
            raw_gate,
        )
        output, topk = self._finish_attention(prepared, gate, topk, fmha)
        return output, topk, post_gate

    def forward_mtp_attention(
        self, source, residual, norm, fmha, cache, prev_topk=None
    ):
        hidden, fp8, scale, raw_gate, _ = self._allocate_frontend(source, target=False)

        def produce(notify_event):
            self._ops.mtp_norm_quant(
                source,
                residual,
                norm.weight,
                hidden,
                fp8,
                scale,
                raw_gate,
                norm.variance_epsilon,
                notify_event,
            )
            return hidden, fp8, scale

        prepared, gate, topk = self.mla_prologue(
            source,
            fmha,
            cache,
            prev_topk,
            self._plan,
            produce,
            (source, residual),
            raw_gate,
        )
        return self._finish_attention(prepared, gate, topk, fmha)

    def forward_target_moe(self, channels, ihc, norm):
        # Layer zero is dense; routed layers have a preflight-validated MegaMoE.
        if self.self_attn.layer_idx == 0:
            hidden, post_gate, fp8, scale = self._target_producer(channels, ihc, norm)
            output = self.mlp(hidden, x_fp8=fp8, x_scale=scale)
        else:
            mega = self.moe_prepacked_input_views(channels.shape[0])
            hidden, post_gate, fp8, scale = self._target_producer(
                channels, ihc, norm, mega=mega
            )
            self._prepare_router(hidden, mega[2], mega[3])
            output = self.mlp.forward_prepacked(
                hidden, mega[2], mega[3], x_fp8=fp8, x_scale=scale
            )
        return ihc.post(output, channels, post_gate)

    def forward_mtp_moe(self, hidden, residual, norm):
        from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
            fused_add_rmsnorm_fp8_quant_with_bf16_output,
        )

        mega = self.moe_prepacked_input_views(hidden.shape[0])
        bf16, fp8, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
            hidden,
            residual,
            norm.weight,
            norm.variance_epsilon,
            group_size=32,
            scale_ue8m0=True,
            mxfp8_semantics=True,
            mega_mxfp8_out=mega[0],
            mega_mxfp8_scale_out=mega[1],
            round_residual_bf16=True,
        )
        self._prepare_router(bf16, mega[2], mega[3])
        output = self.mlp.forward_prepacked(
            bf16, mega[2], mega[3], x_fp8=fp8, x_scale=scale
        )
        return output, residual

    def forward_attention(
        self,
        hidden_states,
        fmha_impl,
        kv_cache=None,
        x_fp8=None,
        x_scale=None,
        prev_topk_indices=None,
        force_reuse_topk_indices=False,
        return_topk=False,
        raw_head_gate_output=None,
    ):
        if not self._initialized:
            raise RuntimeError(
                "HY4 CMP must be selected and initialized at model entry"
            )
        rows, device = hidden_states.shape[0], hidden_states.device
        self._positions = fmha_impl.fmha_params.positions_d.to(torch.int32)
        plan = self.select_plan(rows, force_reuse_topk_indices)
        if plan == Hy4CmpPlan.REUSE_TOPK and prev_topk_indices is None:
            raise RuntimeError("HY4 CMP reuse requires previous TopK indices")
        if not (
            _tensor(x_fp8, (rows, 6144), torch.float8_e4m3fn, device)
            and _tensor(x_scale, (rows, 48), torch.int32, device, contiguous=False)
            and x_scale.stride() == (1, (rows + 3) // 4 * 4)
        ):
            raise RuntimeError("HY4 producer violated fixed activation/scale ABI")
        if plan == Hy4CmpPlan.NATIVE_G32 and not _tensor(
            raw_head_gate_output, (rows, 32), torch.float32, device
        ):
            raise RuntimeError("HY4 native plan requires producer-cleared raw gate")

        def precomputed_input(notify_event):
            if notify_event:
                self._events.frontend_ready.record()
            return hidden_states, x_fp8, x_scale

        prepared, gate, topk = self.mla_prologue(
            hidden_states,
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            plan,
            precomputed_input,
            (hidden_states, x_fp8, x_scale, raw_head_gate_output),
            raw_head_gate_output,
        )
        output, topk = self._finish_attention(prepared, gate, topk, fmha_impl)
        return (output, topk) if return_topk else output


def _producer_supported(layer, cmp, hidden, channels):
    device = hidden.device
    for norm in (layer.input_layernorm, layer.post_attention_layernorm):
        if not _tensor(norm.weight, (6144,), torch.bfloat16, device):
            return False
    if cmp.config.model_type == "hy_v4_mtp":
        return (
            layer._fuse_hy4_cmp_input_norm_quant
            and layer._fuse_hy4_cmp_post_norm_quant_moe
        )
    if not (
        layer._fuse_attn_ihc_mxfp8
        and layer._fuse_mlp_ihc_mxfp8
        and channels is not None
    ):
        return False
    from rtp_llm.models_py.modules.hy_v4.ihc_triton import (
        ihc_pre_is_supported,
        _use_deepgemm_prenorm,
    )

    return _use_deepgemm_prenorm(channels) and all(
        ihc.chunk_size >= hidden.shape[0]
        and ihc_pre_is_supported(channels, ihc.fn_weight, ihc.scale, ihc.base)
        for ihc in (layer.attn_ihc, layer.mlp_ihc)
    )


def _moe_supported(cmp, hidden):
    mlp, c, device = cmp.mlp, cmp.config, hidden.device
    if c.model_type == "hy_v4" and cmp.self_attn.layer_idx == 0:
        return mlp.accepts_mxfp8_input and _dense_weights_supported(mlp, device, 18432)
    if not mlp._hy4_mega_moe_prepack:
        return False
    if not (
        mlp.num_experts == 256
        and mlp.top_k == 8
        and c.moe_n_group == c.moe_topk_group == 1
        and c.has_moe_norm
        and abs(c.routed_scaling_factor - 2.827) < 1e-7
        and _tensor(mlp.gate_weight, (6144, 256), torch.float32, device)
        and _tensor(mlp.correction_bias, (256,), torch.float32, device)
        and mlp.shared_expert is not None
        and mlp.shared_expert.accepts_mxfp8_input
        and _dense_weights_supported(mlp.shared_expert, device, 2048)
    ):
        return False
    mega = mlp.fused_moe.mega_moe
    if any(
        not isinstance(weight, torch.Tensor) or weight.device != device
        for weight in (
            mega._mega_l1_w,
            mega._mega_l1_sf,
            mega._mega_l2_w,
            mega._mega_l2_sf,
        )
    ):
        return False
    cmp.moe_prepacked_input_views(hidden.shape[0])
    return True


def _dense_weights_supported(mlp, device, intermediate):
    return all(
        _tensor(linear.weight, shape, torch.float8_e4m3fn, device)
        and linear.weight_scale.device == device
        and linear.input_quant_group_size == 32
        and linear.input_quant_scale_ue8m0
        for linear, shape in (
            (mlp.up_proj, (2 * intermediate, 6144)),
            (mlp.down_proj, (6144, intermediate)),
        )
    )


def should_enable_hy4_cmp(
    layers,
    layer_num,
    hidden_states,
    fmha_impl,
    kv_cache,
    *,
    channels=None,
    residual=None,
    force_reuse_topk_indices=False,
    prev_topk_indices=None,
):
    """Preflight every active layer before any CMP kernel is submitted."""
    if layer_num <= 0 or kv_cache is None:
        return False
    active = list(layers[:layer_num])
    if len(active) != layer_num:
        return False
    cmps = [layer.hy4_cmp for layer in active]
    if any(cmp is None or cmp._disabled_reason is not None for cmp in cmps):
        return False
    if force_reuse_topk_indices and prev_topk_indices is None:
        raise RuntimeError("HY4 forced TopK reuse requires seed indices")
    if force_reuse_topk_indices and not _tensor(
        prev_topk_indices,
        (hidden_states.shape[0], 2048),
        torch.int32,
        hidden_states.device,
    ):
        raise ValueError("HY4 seed TopK violates the fixed index ABI")
    if cmps[0].config.model_type == "hy_v4_mtp" and not _tensor(
        residual, hidden_states.shape, torch.bfloat16, hidden_states.device
    ):
        return False
    has_topk = force_reuse_topk_indices
    for idx, (layer, cmp) in enumerate(zip(active, cmps)):
        cache = kv_cache.get_layer_cache(idx)
        inputs = fmha_impl.attn_inputs
        table = _page_table(inputs)
        tables = inputs.kv_cache_kernel_block_id_device_by_group
        if tables is not None and len(tables):
            groups = inputs.kv_cache_layer_to_group
            group = 0 if groups is None else int(groups[idx].item())
            if not 0 <= group < len(tables):
                return False
            table = tables[group]
        reuse = cmp.self_attn.reuse_topk_indices or force_reuse_topk_indices
        if (
            cmp._dynamic_disabled_reason(
                hidden_states,
                fmha_impl,
                cache,
                require_indexer=not reuse,
                page_table=table,
            )
            is not None
        ):
            return False
        if cmp.self_attn.fused_qkv_a_proj.weight.device != hidden_states.device:
            return False
        output_weight = fmha_impl.hy4_output_weight(cmp.self_attn.layer_idx)
        if not (
            isinstance(output_weight, torch.Tensor)
            and output_weight.shape == (64, 512, 256)
            and output_weight.dtype == torch.bfloat16
            and output_weight.device == hidden_states.device
        ):
            return False
        if reuse and not has_topk:
            return False
        if not reuse:
            if cmp.self_attn.indexer is None:
                return False
            has_topk = True
        if not (
            _producer_supported(layer, cmp, hidden_states, channels)
            and _moe_supported(cmp, hidden_states)
        ):
            return False
    uninitialized = [cmp for cmp in cmps if not cmp._initialized]
    if uninitialized and _is_capturing():
        raise RuntimeError("HY4 provider must be warmed before CUDA Graph capture")
    for cmp in cmps:
        if not cmp._weight_scales:
            try:
                cmp._weight_scales = cmp._validate_weight_abi()
            except ValueError:
                return False
    # Native symbol/packing failures are deployment errors, never silent fallback.
    if uninitialized:
        ops = _load_hy4_ops()
        for cmp in uninitialized:
            cmp.initialize_for_cmp(ops)
    for cmp in cmps:
        if cmp._output_bmm_weight is None:
            if _is_capturing():
                raise RuntimeError("HY4 output weight must be packed before capture")
            cmp._output_bmm_weight = (
                fmha_impl.hy4_output_weight(cmp.self_attn.layer_idx)
                .detach()
                .transpose(1, 2)
                .contiguous()
            )
    positions = fmha_impl.fmha_params.positions_d.to(torch.int32)
    for cmp in cmps:
        cmp._positions = positions
        cmp._plan = cmp.select_plan(hidden_states.shape[0], force_reuse_topk_indices)
        if cmp._plan != Hy4CmpPlan.REUSE_TOPK:
            cmp._side_streams(hidden_states.device)
            if cmp._events is None:
                cmp._events = cmp._new_events(hidden_states.device)
    return True


__all__ = [
    "Hy4Cmp",
    "Hy4CmpPlan",
    "select_fixed_plan",
    "resolve_hy4_cmp_enabled",
    "should_enable_hy4_cmp",
]
