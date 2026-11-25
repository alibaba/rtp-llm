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
    start_row: tl.int64,
    M,
    L,
    stride_peer: tl.int64,
    stride_out: tl.int64,
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
    tl.store(out_ptr + m * stride_out + offs, acc.to(out_ptr.dtype.element_ty), mask=ml)


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
        self.active_backend = "nccl"
        if eager_init and self.device.type == "cuda":
            self.init_workspace()

    def init_workspace(self) -> str:
        """Eagerly initialize NVLink5 symmetric-memory workspace before CG capture."""
        if self.world_size <= 1 or self.group is None:
            self.active_backend = "noop"
            return self.active_backend
        if self.device.type != "cuda":
            self.active_backend = "nccl"
            return self.active_backend
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Qwen4_5AllToAllComm must be initialized eagerly before CUDA Graph capture."
            )
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

    def dispatch(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: Optional[int] = None,
        equal_tokens: Optional[bool] = None,
        quantize_fp8: bool = False,
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
        ids_i32 = topk_ids.to(torch.int32).contiguous()
        w_f32 = topk_weights.to(torch.float32).contiguous()

        if equal_tokens is None:
            capturing = (
                torch.cuda.is_current_stream_capturing()
                or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
            )
            if capturing:
                equal_tokens = True
            else:
                m_vec = torch.empty((self.world_size,), dtype=torch.int64, device=x_c.device)
                m_local_t = torch.tensor([M_local], dtype=torch.int64, device=x_c.device)
                dist.all_gather_into_tensor(m_vec, m_local_t, group=self.group)
                counts_cpu = m_vec.tolist()
                equal_tokens = all(c == counts_cpu[0] for c in counts_cpu) and counts_cpu[0] > 0

        if equal_tokens:
            return self._dispatch_dense_equal(
                x_c, ids_i32, w_f32, M_local, L, top_k, quantize_fp8=quantize_fp8
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

            pack_buf = torch.empty((M_local, row_bytes), dtype=torch.uint8, device=x_c.device)
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
            dist.all_gather_into_tensor(all_buf, pack_buf, group=self.group)

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
            )

        x_bytes = ((2 * L + 15) // 16) * 16
        row_bytes = x_bytes + meta_bytes
        pack_buf = torch.empty((M_local, row_bytes), dtype=torch.uint8, device=x_c.device)
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
        dist.all_gather_into_tensor(all_buf, pack_buf, group=self.group)

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
        return AllToAllDispatchState(
            mode="dense_gather",
            local_num_tokens=M_local,
            hidden_dim=L,
            dtype=x_c.dtype,
            hidden=out_x,
            hidden_fp8=None,
            topk_ids=out_ids,
            topk_weights=out_w,
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
        tok_rank_mask = torch.zeros((M_local,), dtype=torch.int32, device=dev)
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
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)

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
        dist.all_to_all_single(
            recv_buf.view(-1),
            send_buf.view(-1),
            output_split_sizes=[c * row_bytes for c in recv_splits],
            input_split_sizes=[c * row_bytes for c in send_splits],
            group=self.group,
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

    def combine(
        self,
        ep_output: torch.Tensor,
        state: AllToAllDispatchState,
        *,
        use_symm_p2p: Optional[bool] = None,
    ) -> torch.Tensor:
        """Combine EP partial expert outputs back into DP rank-local ``[M_local, L]``."""
        if state.mode == "noop" or self.world_size <= 1 or self.group is None:
            return ep_output

        L = state.hidden_dim
        M_local = state.local_num_tokens
        ep_c = ep_output if ep_output.is_contiguous() else ep_output.contiguous()

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
                and M_global <= self._symm_max_rows
            )
            want_symm = bool(use_symm_p2p) if use_symm_p2p is not None else False
            if want_symm and can_symm:
                assert self._symm_buf is not None and self._symm_hdl is not None
                self._symm_buf[:M_global].copy_(ep_c)
                self._symm_hdl.barrier(channel=0)
                BK = min(512, triton.next_power_of_2(L))
                _symm_p2p_sum_rows_kernel[(M_local, triton.cdiv(L, BK))](
                    self._symm_hdl.buffer_ptrs_dev,
                    out,
                    self.rank * M_local,
                    M_local,
                    L,
                    self.hidden_dim,
                    out.stride(0),
                    WORLD_SIZE=self.world_size,
                    BLOCK_L=BK,
                    num_warps=4,
                )
                self._symm_hdl.barrier(channel=1)
                return out
            dist.reduce_scatter_tensor(out, ep_c, op=dist.ReduceOp.SUM, group=self.group)
            return out

        assert state.mode == "routed_a2a"
        assert state.send_splits is not None and state.recv_splits is not None
        assert state.send_src_tok is not None
        total_send = sum(state.send_splits)
        returned = torch.empty((total_send, L), dtype=ep_c.dtype, device=ep_c.device)
        dist.all_to_all_single(
            returned,
            ep_c,
            output_split_sizes=state.send_splits,
            input_split_sizes=state.recv_splits,
            group=self.group,
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
        return out_fp32.to(state.dtype)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._symm_hdl = None
        self._symm_buf = None
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
    "AllToAllDispatchState",
    "Qwen4_5AllToAllComm",
    "destroy_qwen4_5_all_to_all",
    "get_qwen4_5_all_to_all",
]
