"""High-bandwidth NVLink5 AllToAll (dispatch + combine) for Qwen 4.5-Flash DP+EP MoE.

Unlocks ``DP=4 Attention + EP=4 MoE`` (and ``dp2_tp2_ep4`` / ragged DP prefill)
over 2560-dim latent MoE tokens with two complementary, accuracy-exact paths:

  1. **Equal-token dense dispatch / combine** (CUDA-Graph-safe, zero host sync):
     Packs ``hidden`` (BF16 or pre-quantized FP8+UE8M0 bytes, cutting NVLink
     bytes by ~1.94x) together with ``topk_ids`` and ``topk_weights`` into one
     16B-aligned buffer and performs a single ``all_gather_into_tensor`` (or
     ``torch.distributed._symmetric_memory`` P2P transfer) on dispatch and
     ``reduce_scatter_tensor`` / P2P sum on combine.
  2. **Unequal / sparse token-deduplicated AllToAll** (ragged DP prefill & sparse
     routing): sends each token at most once per destination EP rank with
     non-local ``topk_ids`` masked to ``-1``, runs ``all_to_all_single``, and
     accumulates returned partial ``[L]`` outputs in FP32 on combine.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rtp_llm.models_py.modules.qwen4_5.activation import (
    _QUANT_EPS,
    _compute_ue8m0_fp8_block32,
)
from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    _agree_bool,
    _detect_single_node,
    _group_identity,
)

logger = logging.getLogger(__name__)

_DEFAULT_HIDDEN_DIM = 2560
_DEFAULT_TOP_K = 10
_DEFAULT_MAX_TOKENS_PER_RANK = 2048
_SUPPORTED_A2A_BACKENDS = ("auto", "symm_mem", "nccl")

_A2A_CACHE_LOCK = threading.Lock()
_A2A_CACHE: dict[Tuple[Any, ...], "Qwen4_5AllToAllComm"] = {}


@dataclass
class AllToAllDispatchState:
    """Metadata carried from :meth:`Qwen4_5AllToAllComm.dispatch` to :meth:`combine`."""

    mode: str  # "noop" | "dense_gather" | "routed_a2a"
    local_num_tokens: int
    hidden_dim: int
    dtype: torch.dtype
    hidden: Optional[torch.Tensor]  # [M_ep, L] BF16 (or None when fp8-only until needed)
    hidden_fp8: Optional[Tuple[torch.Tensor, torch.Tensor]]  # (x_q [M_ep, L], x_sc [M_ep, L//32])
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    # routed_a2a combine fields
    send_src_tok: Optional[torch.Tensor] = None
    send_splits: Optional[List[int]] = None
    recv_splits: Optional[List[int]] = None
    # Packed dense_gather payload (all_buf, L, sc_off, meta_off, row_bytes, top_k)
    packed_fp8: Optional[Tuple[torch.Tensor, int, int, int, int, int]] = None
    # Single-barrier causal tracking (paired GPU dispatch before combine)
    symm_ready_gen: int = -1
    captured_in_graph: bool = False


@dataclass
class AllToAllAsyncHandle:
    """Async stream/event handle returned by ``dispatch_async`` / ``combine_async``."""

    kind: str  # "dispatch" | "combine"
    comm_stream: Optional[torch.cuda.Stream]
    done_event: Optional[torch.cuda.Event]
    main_stream: Optional[torch.cuda.Stream]
    capturing: bool
    result: Any
    keepalive: Tuple[Any, ...] = ()


# ---------------------------------------------------------------------------
# Triton pack / unpack / dedup / combine kernels
# ---------------------------------------------------------------------------


@triton.jit
def _pack_dense_dp_bf16_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    out_ptr,
    M,
    L,
    top_k,
    row_bytes,
    x_bytes,
    stride_x: tl.int64,
    stride_ids: tl.int64,
    stride_w: tl.int64,
    BLOCK_B: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    # Copy x as uint16 elements (L elements -> 2*L bytes)
    out_u16 = (out_ptr + m * row_bytes).to(tl.pointer_type(tl.uint16))
    x_u16 = (x_ptr + m * stride_x).to(tl.pointer_type(tl.uint16))
    for l0 in range(0, L, BLOCK_B):
        offs = l0 + tl.arange(0, BLOCK_B)
        ml = offs < L
        tl.store(out_u16 + offs, tl.load(x_u16 + offs, mask=ml, other=0), mask=ml)

    # Pack topk_ids (int32) and topk_weights (float32 as uint32 bits) right after x_bytes
    offs_k = tl.arange(0, 32)
    mk = offs_k < top_k
    ids_val = tl.load(ids_ptr + m * stride_ids + offs_k, mask=mk, other=-1).to(tl.int32)
    w_val = tl.load(w_ptr + m * stride_w + offs_k, mask=mk, other=0.0).to(tl.float32)
    meta_i32 = (out_ptr + m * row_bytes + x_bytes).to(tl.pointer_type(tl.int32))
    tl.store(meta_i32 + offs_k, ids_val, mask=mk)
    meta_f32 = (out_ptr + m * row_bytes + x_bytes + top_k * 4).to(tl.pointer_type(tl.float32))
    tl.store(meta_f32 + offs_k, w_val, mask=mk)


@triton.jit
def _unpack_dense_dp_bf16_kernel(
    in_ptr,
    x_ptr,
    ids_ptr,
    w_ptr,
    M,
    L,
    top_k,
    row_bytes,
    x_bytes,
    stride_x: tl.int64,
    stride_ids: tl.int64,
    stride_w: tl.int64,
    BLOCK_B: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    in_u16 = (in_ptr + m * row_bytes).to(tl.pointer_type(tl.uint16))
    x_u16 = (x_ptr + m * stride_x).to(tl.pointer_type(tl.uint16))
    for l0 in range(0, L, BLOCK_B):
        offs = l0 + tl.arange(0, BLOCK_B)
        ml = offs < L
        tl.store(x_u16 + offs, tl.load(in_u16 + offs, mask=ml, other=0), mask=ml)

    offs_k = tl.arange(0, 32)
    mk = offs_k < top_k
    meta_i32 = (in_ptr + m * row_bytes + x_bytes).to(tl.pointer_type(tl.int32))
    tl.store(ids_ptr + m * stride_ids + offs_k, tl.load(meta_i32 + offs_k, mask=mk, other=-1), mask=mk)
    meta_f32 = (in_ptr + m * row_bytes + x_bytes + top_k * 4).to(tl.pointer_type(tl.float32))
    tl.store(w_ptr + m * stride_w + offs_k, tl.load(meta_f32 + offs_k, mask=mk, other=0.0), mask=mk)


@triton.jit
def _pack_dense_dp_bf16_to_fp8_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    out_ptr,
    M,
    L,
    num_groups,
    top_k,
    row_bytes,
    sc_off,
    meta_off,
    stride_x: tl.int64,
    stride_ids: tl.int64,
    stride_w: tl.int64,
    eps: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Quantize BF16 -> FP8+UE8M0 in-register and pack into contiguous NVLink buffer."""
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    GROUPS: tl.constexpr = BLOCK_B // 32
    out_row = out_ptr + m * row_bytes
    x_row = x_ptr + m * stride_x
    for l0 in range(0, L, BLOCK_B):
        offs = l0 + tl.arange(0, BLOCK_B)
        ml = offs < L
        x_blk = tl.load(x_row + offs, mask=ml, other=0.0).to(tl.float32)
        q, biased = _compute_ue8m0_fp8_block32(x_blk.reshape((GROUPS, 32)), eps=eps)
        tl.store(out_row + offs, q.reshape((BLOCK_B,)).to(tl.uint8, bitcast=True), mask=ml)
        offs_g = (l0 // 32) + tl.arange(0, GROUPS)
        tl.store(out_row + sc_off + offs_g, biased.to(tl.uint8), mask=offs_g < num_groups)

    offs_k = tl.arange(0, 32)
    mk = offs_k < top_k
    ids_val = tl.load(ids_ptr + m * stride_ids + offs_k, mask=mk, other=-1).to(tl.int32)
    w_val = tl.load(w_ptr + m * stride_w + offs_k, mask=mk, other=0.0).to(tl.float32)
    tl.store((out_row + meta_off).to(tl.pointer_type(tl.int32)) + offs_k, ids_val, mask=mk)
    tl.store((out_row + meta_off + top_k * 4).to(tl.pointer_type(tl.float32)) + offs_k, w_val, mask=mk)


@triton.jit
def _unpack_dense_dp_fp8_kernel(
    in_ptr,
    xq_ptr,
    xsc_ptr,
    ids_ptr,
    w_ptr,
    M,
    L,
    num_groups,
    top_k,
    row_bytes,
    sc_off,
    meta_off,
    stride_xq: tl.int64,
    stride_xsc: tl.int64,
    stride_ids: tl.int64,
    stride_w: tl.int64,
    BLOCK_B: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    in_row = in_ptr + m * row_bytes
    for l0 in range(0, L, BLOCK_B):
        offs = l0 + tl.arange(0, BLOCK_B)
        ml = offs < L
        tl.store(xq_ptr + m * stride_xq + offs, tl.load(in_row + offs, mask=ml, other=0), mask=ml)
    for g0 in range(0, num_groups, 256):
        offs_g = g0 + tl.arange(0, 256)
        mg = offs_g < num_groups
        tl.store(xsc_ptr + m * stride_xsc + offs_g, tl.load(in_row + sc_off + offs_g, mask=mg, other=0), mask=mg)

    offs_k = tl.arange(0, 32)
    mk = offs_k < top_k
    tl.store(
        ids_ptr + m * stride_ids + offs_k,
        tl.load((in_row + meta_off).to(tl.pointer_type(tl.int32)) + offs_k, mask=mk, other=-1),
        mask=mk,
    )
    tl.store(
        w_ptr + m * stride_w + offs_k,
        tl.load((in_row + meta_off + top_k * 4).to(tl.pointer_type(tl.float32)) + offs_k, mask=mk, other=0.0),
        mask=mk,
    )


@triton.jit
def _count_dedup_dest_ranks_kernel(
    ids_ptr,
    send_counts_ptr,
    tok_rank_mask_ptr,
    M,
    local_num_experts,
    world_size,
    stride_ids: tl.int64,
    TOP_K: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    seen_mask = 0
    for k_idx in tl.static_range(TOP_K):
        eid = tl.load(ids_ptr + m * stride_ids + k_idx).to(tl.int32)
        if eid >= 0:
            dest = eid // local_num_experts
            if dest >= 0 and dest < world_size:
                bit = 1 << dest
                if (seen_mask & bit) == 0:
                    seen_mask = seen_mask | bit
                    tl.atomic_add(send_counts_ptr + dest, 1)
    tl.store(tok_rank_mask_ptr + m, seen_mask)


@triton.jit
def _scatter_dedup_tokens_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    tok_rank_mask_ptr,
    offsets_ptr,
    write_pos_ptr,
    send_buf_ptr,
    send_src_tok_ptr,
    M,
    L,
    top_k,
    local_num_experts,
    world_size,
    row_bytes,
    x_bytes,
    stride_x: tl.int64,
    stride_ids: tl.int64,
    stride_w: tl.int64,
    BLOCK_B: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    mask_bits = tl.load(tok_rank_mask_ptr + m).to(tl.int32)
    if mask_bits == 0:
        return

    offs_k = tl.arange(0, 32)
    mk = offs_k < top_k
    ids_val = tl.load(ids_ptr + m * stride_ids + offs_k, mask=mk, other=-1).to(tl.int32)
    w_val = tl.load(w_ptr + m * stride_w + offs_k, mask=mk, other=0.0).to(tl.float32)
    x_u16 = (x_ptr + m * stride_x).to(tl.pointer_type(tl.uint16))

    for dest in range(0, world_size):
        if (mask_bits & (1 << dest)) != 0:
            local_slot = tl.atomic_add(write_pos_ptr + dest, 1).to(tl.int64)
            base_off = tl.load(offsets_ptr + dest).to(tl.int64)
            dst_idx = base_off + local_slot
            tl.store(send_src_tok_ptr + dst_idx, m.to(tl.int32))

            dst_u16 = (send_buf_ptr + dst_idx * row_bytes).to(tl.pointer_type(tl.uint16))
            for l0 in range(0, L, BLOCK_B):
                offs = l0 + tl.arange(0, BLOCK_B)
                ml = offs < L
                tl.store(dst_u16 + offs, tl.load(x_u16 + offs, mask=ml, other=0), mask=ml)

            lo_e = dest * local_num_experts
            hi_e = lo_e + local_num_experts
            masked_ids = tl.where((ids_val >= lo_e) & (ids_val < hi_e), ids_val, -1)
            meta_i32 = (send_buf_ptr + dst_idx * row_bytes + x_bytes).to(tl.pointer_type(tl.int32))
            tl.store(meta_i32 + offs_k, masked_ids, mask=mk)
            meta_f32 = (send_buf_ptr + dst_idx * row_bytes + x_bytes + top_k * 4).to(tl.pointer_type(tl.float32))
            tl.store(meta_f32 + offs_k, w_val, mask=mk)


@triton.jit
def _combine_scatter_add_kernel(
    returned_ptr,
    send_src_tok_ptr,
    out_fp32_ptr,
    num_send,
    L,
    stride_ret: tl.int64,
    stride_out: tl.int64,
    BLOCK_L: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    if idx >= num_send:
        return
    src_tok = tl.load(send_src_tok_ptr + idx).to(tl.int64)
    offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    ml = offs < L
    v = tl.load(returned_ptr + idx * stride_ret + offs, mask=ml, other=0.0).to(tl.float32)
    tl.atomic_add(out_fp32_ptr + src_tok * stride_out + offs, v, mask=ml)


@triton.jit
def _symm_p2p_sum_rows_kernel(
    ptrs_dev_ptr,
    out_ptr,
    scale_gate_ptr,
    start_row: tl.int64,
    M,
    L,
    stride_peer: tl.int64,
    stride_out: tl.int64,
    stride_sg: tl.int64,
    scale_gate_col: tl.constexpr,
    HAS_SCALE_GATE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    if m >= M:
        return
    offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    ml = offs < L
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    ptrs_base = tl.cast(ptrs_dev_ptr, tl.pointer_type(tl.int64))
    row = start_row + m
    for r in tl.static_range(WORLD_SIZE):
        base_addr = tl.load(ptrs_base + r).to(tl.int64)
        peer_ptr = tl.cast(base_addr, tl.pointer_type(tl.bfloat16))
        v = tl.load(peer_ptr + row * stride_peer + offs, mask=ml, other=0.0).to(tl.float32)
        acc += v
    out_val = acc.to(out_ptr.dtype.element_ty)
    if HAS_SCALE_GATE:
        sg_raw = tl.load(scale_gate_ptr + m * stride_sg + scale_gate_col).to(tl.float32)
        sig_sg = tl.div_rn(1.0, 1.0 + libdevice.exp(-sg_raw)).to(out_ptr.dtype.element_ty)
        out_val = (out_val.to(tl.float32) * sig_sg.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + m * stride_out + offs, out_val, mask=ml)


class Qwen4_5AllToAllComm:
    """High-bandwidth NVLink5 AllToAll (dispatch + combine) for Qwen 4.5-Flash EP MoE."""

    def __init__(
        self,
        group: Any,
        device: Optional[torch.device] = None,
        *,
        backend: Optional[str] = None,
        max_tokens_per_rank: Optional[int] = None,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        top_k: int = _DEFAULT_TOP_K,
        dtype: torch.dtype = torch.bfloat16,
        eager_init: bool = True,
        single_node: Optional[bool] = None,
    ) -> None:
        self.group = group
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.dtype = dtype

        env_mt = os.environ.get("QWEN4_5_A2A_MAX_TOKENS")
        self.max_tokens_per_rank = int(
            max_tokens_per_rank
            if max_tokens_per_rank is not None
            else (env_mt or _DEFAULT_MAX_TOKENS_PER_RANK)
        )
        req_backend = (
            backend
            if backend is not None
            else os.environ.get("QWEN4_5_A2A_BACKEND", "auto")
        ).lower()
        if req_backend not in _SUPPORTED_A2A_BACKENDS:
            raise ValueError(
                f"Unsupported alltoall backend={req_backend!r}; expected {_SUPPORTED_A2A_BACKENDS}"
            )
        self.requested_backend = req_backend

        self._symm_hdl: Any = None
        self._symm_buf: Optional[torch.Tensor] = None
        self._symm_max_rows: int = 0
        self._tbo_stream: Optional[torch.cuda.Stream] = None
        self._tbo_events: Optional[Tuple[Tuple[torch.cuda.Event, torch.cuda.Event], ...]] = None
        self._is_gloo: bool = False
        self._gloo_tag_seq: int = 0
        self._disp_gen: int = 0
        self._last_comb_gen: int = -1
        self._last_comb_gen_by_slot: list[int] = [-1, -1]
        self._unclosed_slot_gen: list[int] = [-1, -1]
        self._m_local_dev: Optional[torch.Tensor] = None
        self._m_vec_dev: Optional[torch.Tensor] = None
        self._pack_ws: Optional[torch.Tensor] = None
        self._destroyed: bool = False

        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            self.world_size = 1
            self.rank = 0
            self.device = device or (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.single_node = True
            self.active_backend = "noop"
            return

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = device or (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.single_node = (
            _detect_single_node(group) if single_node is None else bool(single_node)
        )
        try:
            self._is_gloo = dist.get_backend(group) == "gloo"
        except Exception:
            self._is_gloo = False
        self.active_backend = "gloo" if self._is_gloo else "nccl"
        if self.device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
            self._ensure_tbo_resources()
        if eager_init and self.device.type == "cuda" and not self._is_gloo:
            self.init_workspace()

    def _ensure_tbo_resources(self) -> None:
        """Pre-allocate TBO comm stream and dual-slot events before CUDA Graph capture."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if self._tbo_stream is None or self._tbo_stream.device != self.device:
            self._tbo_stream = torch.cuda.Stream(device=self.device)
        if self._tbo_events is None:
            self._tbo_events = (
                (torch.cuda.Event(), torch.cuda.Event()),
                (torch.cuda.Event(), torch.cuda.Event()),
            )

    def init_workspace(self) -> str:
        """Eagerly initialize NVLink5 symmetric-memory workspace before CG capture."""
        if self.world_size <= 1 or self.group is None:
            self.active_backend = "noop"
            return self.active_backend
        if self.device.type != "cuda":
            self.active_backend = "gloo" if self._is_gloo else "nccl"
            return self.active_backend
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Qwen4_5AllToAllComm must be initialized eagerly before CUDA Graph capture."
            )
        self._ensure_tbo_resources()
        if self._is_gloo:
            self.active_backend = "gloo"
            return self.active_backend
        if self.requested_backend == "nccl":
            self.active_backend = "nccl"
            return self.active_backend
        if self._try_init_symm_mem():
            self.active_backend = "symm_mem"
            return self.active_backend
        self.active_backend = "nccl"
        return self.active_backend

    def _try_init_symm_mem(self) -> bool:
        if not self.single_node or self.world_size not in (2, 4, 8):
            return False
        group_name = getattr(self.group, "group_name", None)
        local_ok = False
        try:
            import torch.distributed._symmetric_memory as symm_mem

            cap_major, _ = torch.cuda.get_device_capability(self.device)
            if cap_major in (9, 10) and group_name is not None:
                local_ok = True
        except Exception as exc:
            logger.info("AllToAll symmetric_memory unavailable (%s)", exc)

        if not _agree_bool(local_ok, self.group, self.device):
            return False

        import torch.distributed._symmetric_memory as symm_mem

        hdl = None
        buf = None
        local_buf_ok = False
        max_rows = self.world_size * self.max_tokens_per_rank
        try:
            torch.cuda.set_device(self.device)
            buf = symm_mem.empty(
                (max_rows, self.hidden_dim), dtype=self.dtype, device=self.device
            )
            hdl = symm_mem.rendezvous(buf, str(group_name))
            local_buf_ok = hdl is not None
        except Exception as exc:
            logger.info("AllToAll symm_mem rendezvous failed (%s)", exc)

        if not _agree_bool(local_buf_ok, self.group, self.device):
            return False

        self._symm_buf = buf
        self._symm_hdl = hdl
        self._symm_max_rows = max_rows
        return True

    def can_symm_p2p_combine(
        self,
        *,
        use_symm_p2p: Optional[bool] = None,
        routed_dtype: Optional[torch.dtype] = None,
    ) -> bool:
        if self.world_size <= 1 or self.group is None:
            return False
        env_symm = os.environ.get("QWEN4_5_A2A_SYMM_P2P_COMBINE")
        want_symm = (
            bool(use_symm_p2p)
            if use_symm_p2p is not None
            else (True if env_symm is None else env_symm.strip().lower() not in ("0", "false", "no"))
        )
        return (
            want_symm
            and self.active_backend == "symm_mem"
            and self._symm_hdl is not None
            and self._symm_buf is not None
            and self._symm_max_rows >= self.world_size
            and (routed_dtype is None or routed_dtype == self.dtype)
        )

    def _get_pack_buf(self, nbytes: int, device: torch.device) -> torch.Tensor:
        capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if self._pack_ws is not None and self._pack_ws.device == device and int(self._pack_ws.numel()) >= nbytes:
            return self._pack_ws[:nbytes]
        if capturing:
            return torch.empty((nbytes,), dtype=torch.uint8, device=device)
        alloc = max(nbytes, 0 if self._pack_ws is None else int(self._pack_ws.numel()))
        buf = torch.empty((alloc,), dtype=torch.uint8, device=device)
        self._pack_ws = buf
        return buf[:nbytes]

    def dispatch(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: Optional[int] = None,
        equal_tokens: Optional[bool] = None,
        quantize_fp8: bool = False,
        return_packed: bool = False,
    ) -> AllToAllDispatchState:
        """Dispatch local DP latent tokens ``[M_local, L]`` across ``world_size`` EP ranks.

        Args:
            hidden: Local routed latent activations ``[M_local, L]`` (BF16).
            topk_ids: Global expert IDs ``[M_local, top_k]`` (int32/int64).
            topk_weights: Routing weights ``[M_local, top_k]``.
            num_experts: Total global experts (required when ``equal_tokens=False``).
            equal_tokens: Set ``True`` for equal-length DP CUDA Graph buckets (zero
                host sync), ``False`` for sparse token-dedup AllToAll on unequal/ragged
                DP batches, or ``None`` to auto-detect safely.
            quantize_fp8: If True on ``equal_tokens`` dispatch, quantizes local
                ``[M_local, L]`` to FP8+UE8M0 before NVLink transfer (halving payload).
        """
        M_local, L = hidden.shape
        top_k = int(topk_ids.shape[1])
        if self.world_size <= 1 or self.group is None:
            return AllToAllDispatchState(
                mode="noop",
                local_num_tokens=M_local,
                hidden_dim=L,
                dtype=hidden.dtype,
                hidden=hidden,
                hidden_fp8=None,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

        x_c = hidden if hidden.is_contiguous() else hidden.contiguous()
        ids_i32 = (
            topk_ids
            if (topk_ids.dtype == torch.int32 and topk_ids.is_contiguous())
            else topk_ids.to(torch.int32).contiguous()
        )
        w_f32 = (
            topk_weights
            if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
            else topk_weights.to(torch.float32).contiguous()
        )

        if equal_tokens is None:
            capturing = (
                torch.cuda.is_current_stream_capturing()
                or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
            )
            if capturing:
                equal_tokens = True
            else:
                m_dev = torch.device("cpu") if self._is_gloo else x_c.device
                if self._m_local_dev is None or self._m_local_dev.device != m_dev:
                    self._m_local_dev = torch.empty((1,), dtype=torch.int64, device=m_dev)
                    self._m_vec_dev = torch.empty((self.world_size,), dtype=torch.int64, device=m_dev)
                assert self._m_vec_dev is not None
                self._m_local_dev.fill_(M_local)
                dist.all_gather_into_tensor(self._m_vec_dev, self._m_local_dev, group=self.group)
                counts_cpu = self._m_vec_dev.tolist()
                equal_tokens = all(c == counts_cpu[0] for c in counts_cpu) and counts_cpu[0] > 0

        if equal_tokens:
            return self._dispatch_dense_equal(
                x_c,
                ids_i32,
                w_f32,
                M_local,
                L,
                top_k,
                quantize_fp8=quantize_fp8,
                return_packed=return_packed,
            )
        if num_experts is None or num_experts % self.world_size != 0:
            raise ValueError(
                f"Unequal/ragged routed AllToAll requires num_experts divisible by world_size={self.world_size}, got {num_experts}"
            )
        return self._dispatch_routed_a2a(
            x_c, ids_i32, w_f32, M_local, L, top_k, local_num_experts=num_experts // self.world_size
        )

    def _dispatch_dense_equal(
        self,
        x_c: torch.Tensor,
        ids_i32: torch.Tensor,
        w_f32: torch.Tensor,
        M_local: int,
        L: int,
        top_k: int,
        *,
        quantize_fp8: bool,
        return_packed: bool = False,
    ) -> AllToAllDispatchState:
        assert top_k <= 32, f"Expected top_k <= 32, got {top_k}"
        if M_local == 0:
            dev = x_c.device
            if quantize_fp8 and L % 32 == 0:
                return AllToAllDispatchState(
                    mode="dense_gather",
                    local_num_tokens=0,
                    hidden_dim=L,
                    dtype=x_c.dtype,
                    hidden=None,
                    hidden_fp8=(
                        torch.empty((0, L), dtype=torch.float8_e4m3fn, device=dev),
                        torch.empty((0, L // 32), dtype=torch.uint8, device=dev),
                    ),
                    topk_ids=torch.empty((0, top_k), dtype=torch.int32, device=dev),
                    topk_weights=torch.empty((0, top_k), dtype=torch.float32, device=dev),
                )
            return AllToAllDispatchState(
                mode="dense_gather",
                local_num_tokens=0,
                hidden_dim=L,
                dtype=x_c.dtype,
                hidden=torch.empty((0, L), dtype=x_c.dtype, device=dev),
                hidden_fp8=None,
                topk_ids=torch.empty((0, top_k), dtype=torch.int32, device=dev),
                topk_weights=torch.empty((0, top_k), dtype=torch.float32, device=dev),
            )
        M_global = self.world_size * M_local
        meta_bytes = ((2 * top_k * 4 + 15) // 16) * 16

        if quantize_fp8 and L % 32 == 0:
            num_groups = L // 32
            sc_off = ((L + 15) // 16) * 16
            meta_off = sc_off + ((num_groups + 15) // 16) * 16
            row_bytes = meta_off + meta_bytes

            pack_buf = self._get_pack_buf(M_local * row_bytes, x_c.device).view(M_local, row_bytes)
            block_b = 512 if L >= 512 else max(32, triton.next_power_of_2(L))
            _pack_dense_dp_bf16_to_fp8_kernel[(M_local,)](
                x_c,
                ids_i32,
                w_f32,
                pack_buf,
                M_local,
                L,
                num_groups,
                top_k,
                row_bytes,
                sc_off,
                meta_off,
                x_c.stride(0),
                ids_i32.stride(0),
                w_f32.stride(0),
                eps=float(_QUANT_EPS),
                BLOCK_B=block_b,
                num_warps=4,
            )
            all_buf = torch.empty((M_global, row_bytes), dtype=torch.uint8, device=x_c.device)
            self._collective_all_gather(all_buf, pack_buf)
            capturing = x_c.is_cuda and torch.cuda.is_current_stream_capturing()
            disp_gen = -1
            if not self._is_gloo and x_c.is_cuda:
                self._disp_gen += 1
                disp_gen = self._disp_gen
            if return_packed and x_c.is_cuda:
                return AllToAllDispatchState(
                    mode="dense_gather",
                    local_num_tokens=M_local,
                    hidden_dim=L,
                    dtype=x_c.dtype,
                    hidden=None,
                    hidden_fp8=None,
                    topk_ids=ids_i32[:0],
                    topk_weights=w_f32[:0],
                    packed_fp8=(all_buf, L, sc_off, meta_off, row_bytes, top_k),
                    symm_ready_gen=disp_gen,
                    captured_in_graph=capturing,
                )

            out_xq = torch.empty((M_global, L), dtype=torch.float8_e4m3fn, device=x_c.device)
            out_xsc = torch.empty((M_global, num_groups), dtype=torch.uint8, device=x_c.device)
            out_ids = torch.empty((M_global, top_k), dtype=torch.int32, device=x_c.device)
            out_w = torch.empty((M_global, top_k), dtype=torch.float32, device=x_c.device)
            _unpack_dense_dp_fp8_kernel[(M_global,)](
                all_buf,
                out_xq.view(torch.uint8),
                out_xsc,
                out_ids,
                out_w,
                M_global,
                L,
                num_groups,
                top_k,
                row_bytes,
                sc_off,
                meta_off,
                out_xq.stride(0),
                out_xsc.stride(0),
                out_ids.stride(0),
                out_w.stride(0),
                BLOCK_B=min(1024, triton.next_power_of_2(L)),
                num_warps=4,
            )
            return AllToAllDispatchState(
                mode="dense_gather",
                local_num_tokens=M_local,
                hidden_dim=L,
                dtype=x_c.dtype,
                hidden=None,
                hidden_fp8=(out_xq, out_xsc),
                topk_ids=out_ids,
                topk_weights=out_w,
                symm_ready_gen=disp_gen,
                captured_in_graph=capturing,
            )

        x_bytes = ((2 * L + 15) // 16) * 16
        row_bytes = x_bytes + meta_bytes
        pack_buf = self._get_pack_buf(M_local * row_bytes, x_c.device).view(M_local, row_bytes)
        _pack_dense_dp_bf16_kernel[(M_local,)](
            x_c,
            ids_i32,
            w_f32,
            pack_buf,
            M_local,
            L,
            top_k,
            row_bytes,
            x_bytes,
            x_c.stride(0),
            ids_i32.stride(0),
            w_f32.stride(0),
            BLOCK_B=min(1024, triton.next_power_of_2(L)),
            num_warps=4,
        )
        all_buf = torch.empty((M_global, row_bytes), dtype=torch.uint8, device=x_c.device)
        self._collective_all_gather(all_buf, pack_buf)

        out_x = torch.empty((M_global, L), dtype=x_c.dtype, device=x_c.device)
        out_ids = torch.empty((M_global, top_k), dtype=torch.int32, device=x_c.device)
        out_w = torch.empty((M_global, top_k), dtype=torch.float32, device=x_c.device)
        _unpack_dense_dp_bf16_kernel[(M_global,)](
            all_buf,
            out_x,
            out_ids,
            out_w,
            M_global,
            L,
            top_k,
            row_bytes,
            x_bytes,
            out_x.stride(0),
            out_ids.stride(0),
            out_w.stride(0),
            BLOCK_B=min(1024, triton.next_power_of_2(L)),
            num_warps=4,
        )
        capturing = x_c.is_cuda and torch.cuda.is_current_stream_capturing()
        disp_gen = -1
        if not self._is_gloo and x_c.is_cuda:
            self._disp_gen += 1
            disp_gen = self._disp_gen
        return AllToAllDispatchState(
            mode="dense_gather",
            local_num_tokens=M_local,
            hidden_dim=L,
            dtype=x_c.dtype,
            hidden=out_x,
            hidden_fp8=None,
            topk_ids=out_ids,
            topk_weights=out_w,
            symm_ready_gen=disp_gen,
            captured_in_graph=capturing,
        )

    def _dispatch_routed_a2a(
        self,
        x_c: torch.Tensor,
        ids_i32: torch.Tensor,
        w_f32: torch.Tensor,
        M_local: int,
        L: int,
        top_k: int,
        *,
        local_num_experts: int,
    ) -> AllToAllDispatchState:
        assert self.world_size <= 32, f"Expected world_size <= 32, got {self.world_size}"
        assert top_k <= 32, f"Expected top_k <= 32, got {top_k}"
        dev = x_c.device
        send_counts = torch.zeros((self.world_size,), dtype=torch.int32, device=dev)
        tok_rank_mask = torch.empty((M_local,), dtype=torch.int32, device=dev)
        if M_local > 0:
            _count_dedup_dest_ranks_kernel[(M_local,)](
                ids_i32,
                send_counts,
                tok_rank_mask,
                M_local,
                local_num_experts,
                self.world_size,
                ids_i32.stride(0),
                TOP_K=top_k,
                num_warps=4,
            )

        recv_counts = torch.empty((self.world_size,), dtype=torch.int32, device=dev)
        self._collective_all_to_all_single(recv_counts, send_counts)

        offsets = torch.cumsum(send_counts, dim=0) - send_counts
        send_splits = [int(v) for v in send_counts.tolist()]
        recv_splits = [int(v) for v in recv_counts.tolist()]
        total_send = sum(send_splits)
        total_recv = sum(recv_splits)

        x_bytes = ((2 * L + 15) // 16) * 16
        meta_bytes = ((2 * top_k * 4 + 15) // 16) * 16
        row_bytes = x_bytes + meta_bytes

        send_buf = torch.empty((total_send, row_bytes), dtype=torch.uint8, device=dev)
        send_src_tok = torch.empty((total_send,), dtype=torch.int32, device=dev)
        if M_local > 0 and total_send > 0:
            write_pos = torch.zeros((self.world_size,), dtype=torch.int32, device=dev)
            _scatter_dedup_tokens_kernel[(M_local,)](
                x_c,
                ids_i32,
                w_f32,
                tok_rank_mask,
                offsets,
                write_pos,
                send_buf,
                send_src_tok,
                M_local,
                L,
                top_k,
                local_num_experts,
                self.world_size,
                row_bytes,
                x_bytes,
                x_c.stride(0),
                ids_i32.stride(0),
                w_f32.stride(0),
                BLOCK_B=min(1024, triton.next_power_of_2(L)),
                num_warps=4,
            )

        recv_buf = torch.empty((total_recv, row_bytes), dtype=torch.uint8, device=dev)
        self._collective_all_to_all_single(
            recv_buf.view(-1),
            send_buf.view(-1),
            output_split_sizes=[c * row_bytes for c in recv_splits],
            input_split_sizes=[c * row_bytes for c in send_splits],
        )

        out_x = torch.empty((total_recv, L), dtype=x_c.dtype, device=dev)
        out_ids = torch.empty((total_recv, top_k), dtype=torch.int32, device=dev)
        out_w = torch.empty((total_recv, top_k), dtype=torch.float32, device=dev)
        if total_recv > 0:
            _unpack_dense_dp_bf16_kernel[(total_recv,)](
                recv_buf,
                out_x,
                out_ids,
                out_w,
                total_recv,
                L,
                top_k,
                row_bytes,
                x_bytes,
                out_x.stride(0),
                out_ids.stride(0),
                out_w.stride(0),
                BLOCK_B=min(1024, triton.next_power_of_2(L)),
                num_warps=4,
            )

        return AllToAllDispatchState(
            mode="routed_a2a",
            local_num_tokens=M_local,
            hidden_dim=L,
            dtype=x_c.dtype,
            hidden=out_x,
            hidden_fp8=None,
            topk_ids=out_ids,
            topk_weights=out_w,
            send_src_tok=send_src_tok,
            send_splits=send_splits,
            recv_splits=recv_splits,
        )

    @staticmethod
    def _apply_scale_gate_fallback(
        out: torch.Tensor,
        scale_gate: Optional[torch.Tensor],
        scale_gate_col: int,
    ) -> torch.Tensor:
        if scale_gate is None or out.shape[0] == 0:
            return out
        M_out = int(out.shape[0])
        sg_2d = scale_gate.reshape(-1, 1) if scale_gate.ndim == 1 else scale_gate.reshape(scale_gate.shape[0], -1)
        sg_col = sg_2d[:M_out, scale_gate_col : scale_gate_col + 1]
        sig = torch.sigmoid(sg_col.float()).to(out.dtype)
        return (out.float() * sig.float()).to(out.dtype)

    def combine(
        self,
        ep_output: torch.Tensor,
        state: AllToAllDispatchState,
        *,
        use_symm_p2p: Optional[bool] = None,
        scale_gate: Optional[torch.Tensor] = None,
        scale_gate_col: int = 0,
        slot: int = 0,
    ) -> torch.Tensor:
        """Combine EP partial expert outputs back into DP rank-local ``[M_local, L]``."""
        if state.mode == "noop" or self.world_size <= 1 or self.group is None:
            return self._apply_scale_gate_fallback(ep_output, scale_gate, scale_gate_col)

        L = state.hidden_dim
        M_local = state.local_num_tokens
        ep_c = ep_output if ep_output.is_contiguous() else ep_output.contiguous()
        has_sg = scale_gate is not None
        if has_sg:
            assert scale_gate is not None
            sg_2d = scale_gate.reshape(-1, 1) if scale_gate.ndim == 1 else scale_gate.reshape(scale_gate.shape[0], -1)
            if sg_2d.stride(-1) != 1:
                sg_2d = sg_2d.contiguous()
            sg_ptr: torch.Tensor = sg_2d
            sg_stride = int(sg_2d.stride(0))
            sg_col = int(scale_gate_col)
        else:
            sg_ptr = ep_c
            sg_stride = 0
            sg_col = 0

        if state.mode == "dense_gather":
            out = torch.empty((M_local, L), dtype=state.dtype, device=ep_c.device)
            if M_local == 0:
                return out
            M_global = self.world_size * M_local
            can_symm = (
                self.active_backend == "symm_mem"
                and self._symm_hdl is not None
                and self._symm_buf is not None
                and L == self.hidden_dim
                and ep_c.dtype == self.dtype
                and self._symm_max_rows >= self.world_size
            )
            env_symm = os.environ.get("QWEN4_5_A2A_SYMM_P2P_COMBINE")
            want_symm = (
                bool(use_symm_p2p)
                if use_symm_p2p is not None
                else (True if env_symm is None else env_symm.strip().lower() not in ("0", "false", "no"))
            )
            if want_symm and can_symm:
                assert self._symm_buf is not None and self._symm_hdl is not None
                BK = min(512, triton.next_power_of_2(L))
                env_sb = os.environ.get("QWEN4_5_A2A_SYMM_SINGLE_BARRIER")
                want_sb = (
                    True
                    if env_sb is None
                    else env_sb.strip().lower() not in ("0", "false", "no")
                )
                capturing = torch.cuda.is_current_stream_capturing()
                if M_global <= self._symm_max_rows:
                    half_rows = self._symm_max_rows // 2
                    dual_disjoint = 2 * M_global <= self._symm_max_rows
                    use_slot = int(slot) & 1 if dual_disjoint else 0
                    buf_row_off = half_rows if (use_slot == 1) else 0
                    ch = use_slot
                    if self._disp_gen > 0 and self._unclosed_slot_gen[use_slot] == self._disp_gen:
                        # Fence before overwriting the same symmetric buffer slice if a prior
                        # combine on this slot omitted its tail barrier without an intervening dispatch.
                        self._symm_hdl.barrier(channel=1 - ch)
                        self._unclosed_slot_gen[use_slot] = -1
                    has_paired_disp = (
                        want_sb
                        and state.symm_ready_gen >= 0
                        and state.symm_ready_gen != self._last_comb_gen
                        and state.symm_ready_gen != self._last_comb_gen_by_slot[use_slot]
                        and (not capturing or state.captured_in_graph)
                        and (
                            state.symm_ready_gen == self._disp_gen
                            or (dual_disjoint and (self._disp_gen - state.symm_ready_gen) <= 1)
                        )
                    )
                    self._symm_buf[buf_row_off : buf_row_off + M_global].copy_(ep_c)
                    self._symm_hdl.barrier(channel=ch)
                    _symm_p2p_sum_rows_kernel[(M_local, triton.cdiv(L, BK))](
                        self._symm_hdl.buffer_ptrs_dev,
                        out,
                        sg_ptr,
                        buf_row_off + self.rank * M_local,
                        M_local,
                        L,
                        self.hidden_dim,
                        out.stride(0),
                        sg_stride,
                        scale_gate_col=sg_col,
                        HAS_SCALE_GATE=has_sg,
                        WORLD_SIZE=self.world_size,
                        BLOCK_L=BK,
                        num_warps=4,
                    )
                    if state.symm_ready_gen >= 0:
                        self._last_comb_gen = state.symm_ready_gen
                        self._last_comb_gen_by_slot[use_slot] = state.symm_ready_gen
                    if has_paired_disp:
                        self._unclosed_slot_gen[use_slot] = self._disp_gen
                    else:
                        self._symm_hdl.barrier(channel=1 - ch)
                        self._unclosed_slot_gen[use_slot] = -1
                    return out
                if self._disp_gen > 0 and self._unclosed_slot_gen[0] == self._disp_gen:
                    self._symm_hdl.barrier(channel=1)
                    self._unclosed_slot_gen[0] = -1
                has_paired_disp = (
                    want_sb
                    and state.symm_ready_gen >= 0
                    and state.symm_ready_gen == self._disp_gen
                    and state.symm_ready_gen != self._last_comb_gen
                    and state.symm_ready_gen != self._last_comb_gen_by_slot[0]
                    and (not capturing or state.captured_in_graph)
                )
                chunk_local = max(1, self._symm_max_rows // self.world_size)
                ep_3d = ep_c.view(self.world_size, M_local, L)
                for m0 in range(0, M_local, chunk_local):
                    m1 = min(M_local, m0 + chunk_local)
                    cm = m1 - m0
                    total_c = self.world_size * cm
                    self._symm_buf[:total_c].copy_(ep_3d[:, m0:m1, :].reshape(total_c, L))
                    self._symm_hdl.barrier(channel=0)
                    sg_chunk_ptr = sg_ptr[m0:m1] if has_sg else ep_c
                    _symm_p2p_sum_rows_kernel[(cm, triton.cdiv(L, BK))](
                        self._symm_hdl.buffer_ptrs_dev,
                        out[m0:m1],
                        sg_chunk_ptr,
                        self.rank * cm,
                        cm,
                        L,
                        self.hidden_dim,
                        out.stride(0),
                        sg_stride,
                        scale_gate_col=sg_col,
                        HAS_SCALE_GATE=has_sg,
                        WORLD_SIZE=self.world_size,
                        BLOCK_L=BK,
                        num_warps=4,
                    )
                    if (m1 < M_local) or not has_paired_disp:
                        self._symm_hdl.barrier(channel=1)
                        if m1 >= M_local:
                            if state.symm_ready_gen >= 0:
                                self._last_comb_gen = state.symm_ready_gen
                                self._last_comb_gen_by_slot[0] = state.symm_ready_gen
                            self._unclosed_slot_gen[0] = -1
                    else:
                        self._last_comb_gen = state.symm_ready_gen
                        self._last_comb_gen_by_slot[0] = state.symm_ready_gen
                        self._unclosed_slot_gen[0] = self._disp_gen
                return out
            self._collective_reduce_scatter(out, ep_c)
            return self._apply_scale_gate_fallback(out, scale_gate, scale_gate_col)

        assert state.mode == "routed_a2a"
        assert state.send_splits is not None and state.recv_splits is not None
        assert state.send_src_tok is not None
        total_send = sum(state.send_splits)
        returned = torch.empty((total_send, L), dtype=ep_c.dtype, device=ep_c.device)
        self._collective_all_to_all_single(
            returned,
            ep_c,
            output_split_sizes=state.send_splits,
            input_split_sizes=state.recv_splits,
        )
        if M_local == 0:
            return torch.empty((0, L), dtype=state.dtype, device=ep_c.device)
        out_fp32 = torch.zeros((M_local, L), dtype=torch.float32, device=ep_c.device)
        if total_send > 0:
            BK = min(512, triton.next_power_of_2(L))
            _combine_scatter_add_kernel[(total_send, triton.cdiv(L, BK))](
                returned,
                state.send_src_tok,
                out_fp32,
                total_send,
                L,
                returned.stride(0),
                out_fp32.stride(0),
                BLOCK_L=BK,
                num_warps=4,
            )
        return self._apply_scale_gate_fallback(out_fp32.to(state.dtype), scale_gate, scale_gate_col)

    def _collective_all_gather(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self._is_gloo:
            src_cpu = src.detach().cpu().contiguous() if src.is_cuda else src.contiguous()
            chunks = [torch.empty_like(src_cpu) for _ in range(self.world_size)]
            dist.all_gather(chunks, src_cpu, group=self.group)
            if dst.numel() > 0:
                dst_cpu = torch.cat(chunks, dim=0)
                dst.copy_(dst_cpu.to(device=dst.device) if dst.is_cuda else dst_cpu)
            return
        dist.all_gather_into_tensor(dst, src, group=self.group)

    def _collective_reduce_scatter(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self._is_gloo:
            src_cpu = src.detach().cpu().contiguous() if src.is_cuda else src.contiguous()
            ws = self.world_size
            chunks = [c.contiguous() for c in src_cpu.chunk(ws, dim=0)]
            out_cpu = torch.empty(dst.shape, dtype=dst.dtype, device="cpu")
            dist.reduce_scatter(out_cpu, chunks, op=dist.ReduceOp.SUM, group=self.group)
            if dst.numel() > 0:
                dst.copy_(out_cpu.to(device=dst.device) if dst.is_cuda else out_cpu)
            return
        dist.reduce_scatter_tensor(dst, src, op=dist.ReduceOp.SUM, group=self.group)

    def _collective_all_to_all_single(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        *,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
    ) -> None:
        if self._is_gloo:
            ws = self.world_size
            me = self.rank
            src_cpu = src.detach().cpu().contiguous() if src.is_cuda else src.contiguous()
            if input_split_sizes is None:
                in_splits = [src_cpu.shape[0] // ws] * ws
            else:
                in_splits = [int(s) for s in input_split_sizes]
            if output_split_sizes is None:
                out_splits = [dst.shape[0] // ws] * ws
            else:
                out_splits = [int(s) for s in output_split_sizes]
            in_chunks = [t.contiguous() for t in torch.split(src_cpu, in_splits, dim=0)]
            out_chunks = [
                torch.empty((s, *dst.shape[1:]), dtype=dst.dtype, device="cpu")
                for s in out_splits
            ]
            if out_splits[me] > 0:
                out_chunks[me].copy_(in_chunks[me])
            # Gloo lacks alltoall/alltoall_base; emulate with tagged P2P CPU bounce.
            tag_base = (self._gloo_tag_seq & 0xFFFF) * (ws * ws)
            self._gloo_tag_seq += 1
            reqs = []
            for peer in range(ws):
                if peer == me:
                    continue
                g_peer = dist.get_global_rank(self.group, peer)
                if in_chunks[peer].numel() > 0:
                    reqs.append(
                        dist.isend(
                            in_chunks[peer],
                            dst=g_peer,
                            group=self.group,
                            tag=tag_base + me * ws + peer,
                        )
                    )
                if out_chunks[peer].numel() > 0:
                    reqs.append(
                        dist.irecv(
                            out_chunks[peer],
                            src=g_peer,
                            group=self.group,
                            tag=tag_base + peer * ws + me,
                        )
                    )
            for req in reqs:
                req.wait()
            if dst.numel() > 0:
                dst_cpu = torch.cat(out_chunks, dim=0)
                dst.copy_(dst_cpu.to(device=dst.device) if dst.is_cuda else dst_cpu)
            return
        if output_split_sizes is None:
            dist.all_to_all_single(dst, src, group=self.group)
        else:
            dist.all_to_all_single(
                dst,
                src,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=self.group,
            )

    def dispatch_async(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: Optional[int] = None,
        equal_tokens: Optional[bool] = None,
        quantize_fp8: bool = False,
        return_packed: bool = False,
        comm_stream: Optional[torch.cuda.Stream] = None,
        slot: int = 0,
    ) -> AllToAllAsyncHandle:
        """Launch AllToAll dispatch asynchronously on a secondary stream for TBO overlap.

        CG-safe when pre-initialized prior to capture and called with ``equal_tokens=True``;
        in eager unequal/sparse DP batches (``equal_tokens=False``), 0-token DP ranks still
        participate in all collectives on ``comm_stream``.
        """
        if self.world_size <= 1 or self.group is None or not hidden.is_cuda:
            st = self.dispatch(
                hidden,
                topk_ids,
                topk_weights,
                num_experts=num_experts,
                equal_tokens=equal_tokens,
                quantize_fp8=quantize_fp8,
                return_packed=return_packed,
            )
            return AllToAllAsyncHandle(
                kind="dispatch",
                comm_stream=None,
                done_event=None,
                main_stream=None,
                capturing=False,
                result=st,
            )

        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            self._ensure_tbo_resources()
        stream = comm_stream if comm_stream is not None else self._tbo_stream
        events = self._tbo_events[int(slot) & 1] if self._tbo_events is not None else None
        if stream is None or events is None or stream.device != hidden.device:
            st = self.dispatch(
                hidden,
                topk_ids,
                topk_weights,
                num_experts=num_experts,
                equal_tokens=equal_tokens,
                quantize_fp8=quantize_fp8,
                return_packed=return_packed,
            )
            return AllToAllAsyncHandle(
                kind="dispatch",
                comm_stream=None,
                done_event=None,
                main_stream=None,
                capturing=capturing,
                result=st,
            )

        ready_ev, done_ev = events
        main_stream = torch.cuda.current_stream(hidden.device)
        ready_ev.record(main_stream)
        exc_hit = False
        try:
            with torch.cuda.stream(stream):
                ready_ev.wait(stream)
                if not capturing:
                    hidden.record_stream(stream)
                    topk_ids.record_stream(stream)
                    topk_weights.record_stream(stream)
                st = self.dispatch(
                    hidden,
                    topk_ids,
                    topk_weights,
                    num_experts=num_experts,
                    equal_tokens=equal_tokens,
                    quantize_fp8=quantize_fp8,
                    return_packed=return_packed,
                )
        except BaseException:
            exc_hit = True
            raise
        finally:
            if not (capturing and exc_hit):
                done_ev.record(stream)
        return AllToAllAsyncHandle(
            kind="dispatch",
            comm_stream=stream,
            done_event=done_ev,
            main_stream=main_stream,
            capturing=capturing,
            result=st,
            keepalive=(hidden, topk_ids, topk_weights),
        )

    def wait_dispatch(self, handle: AllToAllAsyncHandle) -> AllToAllDispatchState:
        """Synchronize caller stream with an in-flight :meth:`dispatch_async`."""
        st: AllToAllDispatchState = handle.result
        if handle.done_event is not None and handle.comm_stream is not None:
            cur = torch.cuda.current_stream(handle.comm_stream.device)
            handle.done_event.wait(cur)
            if not handle.capturing:
                for t in (st.hidden, st.topk_ids, st.topk_weights, st.send_src_tok):
                    if t is not None and t.is_cuda:
                        t.record_stream(cur)
                if st.hidden_fp8 is not None:
                    for t in st.hidden_fp8:
                        if t.is_cuda:
                            t.record_stream(cur)
                if st.packed_fp8 is not None and st.packed_fp8[0].is_cuda:
                    st.packed_fp8[0].record_stream(cur)
        return st

    def combine_async(
        self,
        ep_output: torch.Tensor,
        state: AllToAllDispatchState,
        *,
        use_symm_p2p: Optional[bool] = None,
        scale_gate: Optional[torch.Tensor] = None,
        scale_gate_col: int = 0,
        comm_stream: Optional[torch.cuda.Stream] = None,
        slot: int = 0,
    ) -> AllToAllAsyncHandle:
        """Launch AllToAll combine asynchronously on a secondary stream for TBO overlap."""
        if state.mode == "noop" or self.world_size <= 1 or self.group is None or not ep_output.is_cuda:
            out = self.combine(
                ep_output,
                state,
                use_symm_p2p=use_symm_p2p,
                scale_gate=scale_gate,
                scale_gate_col=scale_gate_col,
            )
            return AllToAllAsyncHandle(
                kind="combine",
                comm_stream=None,
                done_event=None,
                main_stream=None,
                capturing=False,
                result=out,
            )

        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            self._ensure_tbo_resources()
        stream = comm_stream if comm_stream is not None else self._tbo_stream
        events = self._tbo_events[int(slot) & 1] if self._tbo_events is not None else None
        if stream is None or events is None or stream.device != ep_output.device:
            out = self.combine(
                ep_output,
                state,
                use_symm_p2p=use_symm_p2p,
                scale_gate=scale_gate,
                scale_gate_col=scale_gate_col,
                slot=slot,
            )
            return AllToAllAsyncHandle(
                kind="combine",
                comm_stream=None,
                done_event=None,
                main_stream=None,
                capturing=capturing,
                result=out,
            )

        ready_ev, done_ev = events
        main_stream = torch.cuda.current_stream(ep_output.device)
        ready_ev.record(main_stream)
        exc_hit = False
        try:
            with torch.cuda.stream(stream):
                ready_ev.wait(stream)
                if not capturing:
                    ep_output.record_stream(stream)
                    if state.send_src_tok is not None and state.send_src_tok.is_cuda:
                        state.send_src_tok.record_stream(stream)
                    if scale_gate is not None and scale_gate.is_cuda:
                        scale_gate.record_stream(stream)
                out = self.combine(
                    ep_output,
                    state,
                    use_symm_p2p=use_symm_p2p,
                    scale_gate=scale_gate,
                    scale_gate_col=scale_gate_col,
                    slot=slot,
                )
        except BaseException:
            exc_hit = True
            raise
        finally:
            if not (capturing and exc_hit):
                done_ev.record(stream)
        return AllToAllAsyncHandle(
            kind="combine",
            comm_stream=stream,
            done_event=done_ev,
            main_stream=main_stream,
            capturing=capturing,
            result=out,
            keepalive=(ep_output, scale_gate),
        )

    def wait_combine(self, handle: AllToAllAsyncHandle) -> torch.Tensor:
        """Synchronize caller stream with an in-flight :meth:`combine_async`."""
        out: torch.Tensor = handle.result
        if handle.done_event is not None and handle.comm_stream is not None:
            cur = torch.cuda.current_stream(handle.comm_stream.device)
            handle.done_event.wait(cur)
            if not handle.capturing and out.is_cuda:
                out.record_stream(cur)
        return out

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._symm_hdl = None
        self._symm_buf = None
        self._tbo_stream = None
        self._tbo_events = None
        self._m_local_dev = None
        self._m_vec_dev = None
        self._pack_ws = None
        self.active_backend = "nccl" if self.world_size > 1 else "noop"


def get_qwen4_5_all_to_all(
    group: Any,
    device: Optional[torch.device] = None,
    *,
    backend: Optional[str] = None,
    max_tokens_per_rank: Optional[int] = None,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    top_k: int = _DEFAULT_TOP_K,
    dtype: torch.dtype = torch.bfloat16,
    single_node: Optional[bool] = None,
) -> Qwen4_5AllToAllComm:
    """Return a cached, workspace-shared :class:`Qwen4_5AllToAllComm` for ``group``."""
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return Qwen4_5AllToAllComm(
            group,
            device=device,
            backend=backend,
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_dim=hidden_dim,
            top_k=top_k,
            dtype=dtype,
            single_node=single_node,
        )

    dev_obj = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else (device or torch.device("cuda", torch.cuda.current_device()))
    )
    key = (
        _group_identity(group),
        dev_obj.type,
        int(dev_obj.index or 0),
        backend or os.environ.get("QWEN4_5_A2A_BACKEND", "auto"),
        int(max_tokens_per_rank or os.environ.get("QWEN4_5_A2A_MAX_TOKENS") or _DEFAULT_MAX_TOKENS_PER_RANK),
        int(hidden_dim),
        int(top_k),
        str(dtype),
        single_node,
    )
    with _A2A_CACHE_LOCK:
        cached = _A2A_CACHE.get(key)
        if cached is not None and not cached._destroyed:
            return cached
        inst = Qwen4_5AllToAllComm(
            group,
            device=dev_obj,
            backend=backend,
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_dim=hidden_dim,
            top_k=top_k,
            dtype=dtype,
            eager_init=True,
            single_node=single_node,
        )
        _A2A_CACHE[key] = inst
        return inst


def destroy_qwen4_5_all_to_all() -> None:
    """Destroy all cached :class:`Qwen4_5AllToAllComm` workspaces."""
    with _A2A_CACHE_LOCK:
        for inst in _A2A_CACHE.values():
            inst.destroy()
        _A2A_CACHE.clear()


__all__ = [
    "AllToAllAsyncHandle",
    "AllToAllDispatchState",
    "Qwen4_5AllToAllComm",
    "destroy_qwen4_5_all_to_all",
    "get_qwen4_5_all_to_all",
]
