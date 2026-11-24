"""Opt-in TP2 P2P AllReduce + Hyper-Connection Residual Injection & Norm.

Stages TP-partial sublayer outputs (single-branch Attention/GDN `[rows, 2560]`
or packed dual-branch MoE `[2 * rows, 2560]`) into persistent 2-slot ping-pong
CUDA IPC buffers and fuses the cross-GPU BF16 reduction directly into the
subsequent Hyper-Connection residual writeback and grouped RMSNorm.
"""

from contextvars import ContextVar
import ctypes
import logging
import math
import os
from pathlib import Path
import socket
from typing import Optional

import torch
import torch.distributed as dist
import triton

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
    _norm_fp8_scale,
)
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)

_WORKSPACES = {}


class _DeferSlot:
    __slots__ = ("workspace", "consumed", "allow_single")

    def __init__(
        self, workspace: "Qwen4Tp2ArHcWorkspace", *, allow_single: bool = True
    ) -> None:
        self.workspace = workspace
        self.consumed = False
        self.allow_single = bool(allow_single)


class DeferredTp2Sublayer:
    __slots__ = ("local_tensor", "packed_dual", "workspace")

    def __init__(
        self,
        local_tensor: torch.Tensor,
        packed_dual: bool,
        workspace: "Qwen4Tp2ArHcWorkspace",
    ) -> None:
        self.local_tensor = local_tensor
        self.packed_dual = bool(packed_dual)
        self.workspace = workspace

    def materialize(self) -> torch.Tensor:
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        reduced = all_reduce(self.local_tensor.clone(), Group.TP)
        if self.packed_dual:
            rows = reduced.shape[0] // 2
            return reduced[:rows] + reduced[rows:]
        return reduced


_attn_tp2_defer_slot: ContextVar[Optional[_DeferSlot]] = ContextVar(
    "_attn_tp2_defer_slot", default=None
)
_moe_tp2_defer_slot: ContextVar[Optional[_DeferSlot]] = ContextVar(
    "_moe_tp2_defer_slot", default=None
)


def try_defer_attn_tp2(tensor: torch.Tensor, group) -> Optional[DeferredTp2Sublayer]:
    slot = _attn_tp2_defer_slot.get()
    if (
        slot is not None
        and not slot.consumed
        and group == collective_torch.Group.TP
        and slot.workspace.can_stage_single(tensor)
    ):
        slot.consumed = True
        return DeferredTp2Sublayer(tensor, False, slot.workspace)
    return None


def try_defer_moe_packed_tp2(packed: torch.Tensor) -> Optional[DeferredTp2Sublayer]:
    slot = _moe_tp2_defer_slot.get()
    if (
        slot is not None
        and not slot.consumed
        and slot.workspace.can_stage_packed(packed)
    ):
        slot.consumed = True
        return DeferredTp2Sublayer(packed, True, slot.workspace)
    return None


def try_defer_sublayer_tp2(tensor: torch.Tensor, group) -> Optional[DeferredTp2Sublayer]:
    deferred = try_defer_attn_tp2(tensor, group)
    if deferred is not None:
        return deferred
    slot = _moe_tp2_defer_slot.get()
    if (
        slot is not None
        and slot.allow_single
        and not slot.consumed
        and group == collective_torch.Group.TP
    ):
        if slot.workspace.can_stage_single(tensor):
            slot.consumed = True
            return DeferredTp2Sublayer(tensor, False, slot.workspace)
    return None


class Qwen4Tp2ArHcWorkspace:
    max_rows = 12288
    max_packed_rows = 6144

    def __init__(self, group, device: torch.device) -> None:
        from flashinfer.comm import cuda_ipc

        from rtp_llm.models_py.distributed.tp2_p2p_all_gather import (
            acquire_tp2_shared_ipc_buffers,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.ops = rtp_llm_ops
        self.closed = False
        assert dist.get_world_size(group) == 2
        with torch.cuda.device(self.device):
            self.signals = cuda_ipc.create_shared_buffer(
                self.ops.tp2_ar_hc_meta_size(), group
            )
            self.buffers, self.buffer_bytes, max_single_rows = (
                acquire_tp2_shared_ipc_buffers(group, self.device)
            )
            self.max_rows = int(max_single_rows)
            self.max_packed_rows = self.max_rows // 2
            cuda_ipc.cudart.cudaMemset(
                ctypes.c_void_p(self.signals[self.rank]),
                0,
                self.ops.tp2_ar_hc_meta_size(),
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)
            probe_x = torch.zeros((1, 10240), device=self.device, dtype=torch.bfloat16)
            probe_s = torch.zeros((1, 2560), device=self.device, dtype=torch.bfloat16)
            probe_g = torch.zeros((1, 4), device=self.device, dtype=torch.bfloat16)
            self.ops.tp2_ar_hc_inject_only(
                probe_x,
                probe_s,
                probe_g,
                self.buffers,
                self.signals,
                self.rank,
                False,
                False,
                self.buffer_bytes,
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        from flashinfer.comm import cuda_ipc

        from rtp_llm.models_py.distributed.tp2_p2p_all_gather import (
            release_tp2_shared_ipc_buffers,
        )

        self.closed = True
        torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier(group=self.group)
        cuda_ipc.free_shared_buffer(self.signals, self.group)
        release_tp2_shared_ipc_buffers(self.group, self.device)

    def supports_hyper(self, hyper_x: torch.Tensor) -> bool:
        return (
            isinstance(hyper_x, torch.Tensor)
            and hyper_x.is_cuda
            and hyper_x.device == self.device
            and hyper_x.dtype == torch.bfloat16
            and hyper_x.ndim >= 2
            and hyper_x.shape[-1] == 10240
            and 1 <= hyper_x.numel() // 10240 <= self.max_rows
            and hyper_x.is_contiguous()
            and not hyper_x.requires_grad
        )

    def can_stage_single(self, tensor: torch.Tensor) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= self.max_rows
            and tensor.shape[1] == 2560
            and tensor.is_contiguous()
            and tensor.data_ptr() % 16 == 0
            and not tensor.requires_grad
        )

    def can_stage_packed(self, packed: torch.Tensor) -> bool:
        return (
            isinstance(packed, torch.Tensor)
            and packed.is_cuda
            and packed.device == self.device
            and packed.dtype == torch.bfloat16
            and packed.ndim == 2
            and packed.shape[0] % 2 == 0
            and 1 <= packed.shape[0] // 2 <= self.max_packed_rows
            and packed.shape[1] == 2560
            and packed.is_contiguous()
            and packed.data_ptr() % 16 == 0
            and not packed.requires_grad
        )

    def try_inject_grouped_norm(
        self,
        hyper_x: torch.Tensor,
        deferred: DeferredTp2Sublayer,
        gates: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *,
        collect_scale: bool = True,
    ):
        rows = hyper_x.numel() // 10240 if hyper_x.ndim >= 2 and hyper_x.shape[-1] == 10240 else 0
        expected_sub_rows = 2 * rows if deferred.packed_dual else rows
        if not (
            cached_device_capability(self.device) == (9, 0)
            and self.supports_hyper(hyper_x)
            and 1 <= expected_sub_rows <= self.max_rows
            and deferred.local_tensor.shape == (expected_sub_rows, 2560)
            and gates.shape == (*hyper_x.shape[:-1], 4)
            and gamma.shape == (10240,)
            and all(
                t.dtype == torch.bfloat16
                and t.device == self.device
                and t.is_contiguous()
                and t.data_ptr() % 8 == 0
                and not t.requires_grad
                for t in (gates, gamma)
            )
            and hyper_x.data_ptr() % 16 == 0
            and gamma.data_ptr() % 16 == 0
            and isinstance(eps, (float, int))
            and math.isfinite(eps)
            and eps >= 0
        ):
            return None
        pdl = sm90_pdl_enabled(self.device)
        residual, normed, amax = self.ops.tp2_ar_hc_inject_norm(
            hyper_x,
            deferred.local_tensor,
            gates,
            gamma,
            self.buffers,
            self.signals,
            self.rank,
            float(eps),
            deferred.packed_dual,
            bool(collect_scale),
            bool(pdl),
            getattr(self, "buffer_bytes", 0),
        )
        scale = None
        if collect_scale and amax is not None:
            scale = torch.empty(1, device=self.device, dtype=torch.float32)
            block = min(65536, triton.next_power_of_2(rows * 4))
            _norm_fp8_scale[(1,)](
                amax,
                scale,
                N=rows * 4,
                BLOCK=block,
                PDL=pdl,
                num_warps=4,
                launch_pdl=pdl,
            )
        return residual, normed, scale

    def try_inject_only(
        self,
        hyper_x: torch.Tensor,
        deferred: DeferredTp2Sublayer,
        gates: torch.Tensor,
    ):
        rows = hyper_x.numel() // 10240 if hyper_x.ndim >= 2 and hyper_x.shape[-1] == 10240 else 0
        expected_sub_rows = 2 * rows if deferred.packed_dual else rows
        if not (
            cached_device_capability(self.device) == (9, 0)
            and self.supports_hyper(hyper_x)
            and 1 <= expected_sub_rows <= self.max_rows
            and deferred.local_tensor.shape == (expected_sub_rows, 2560)
            and gates.shape == (*hyper_x.shape[:-1], 4)
            and gates.dtype == torch.bfloat16
            and gates.device == self.device
            and gates.is_contiguous()
            and gates.data_ptr() % 8 == 0
            and hyper_x.data_ptr() % 16 == 0
            and not gates.requires_grad
        ):
            return None
        pdl = sm90_pdl_enabled(self.device)
        return self.ops.tp2_ar_hc_inject_only(
            hyper_x,
            deferred.local_tensor,
            gates,
            self.buffers,
            self.signals,
            self.rank,
            deferred.packed_dual,
            bool(pdl),
            getattr(self, "buffer_bytes", 0),
        )


def _tp2_peers_can_access(group, device: torch.device) -> bool:
    if dist.get_world_size(group) != 2:
        return False
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_path.read_text().strip() if boot_path.is_file() else ""
    info = (socket.gethostname(), boot_id, os.getuid(), int(device.index))
    peers = [None, None]
    dist.all_gather_object(peers, info, group=group)
    (h0, b0, u0, d0), (h1, b1, u1, d1) = peers
    return (
        (h0, b0, u0) == (h1, b1, u1)
        and d0 != d1
        and 0 <= d0 < torch.cuda.device_count()
        and 0 <= d1 < torch.cuda.device_count()
        and torch.cuda.can_device_access_peer(d0, d1)
        and torch.cuda.can_device_access_peer(d1, d0)
    )


def prepare_tp2_ar_hc_workspace(config, parallelism, device: Optional[torch.device] = None):
    # Prefer the verified standalone FlashInfer path for decode and MTP.
    # Setting this switch to 0 explicitly selects the fused HC implementation.
    if os.getenv("RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED", "1") == "1":
        return None
    if os.getenv("FT_DISABLE_CUSTOM_AR", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return None
    if not (
        getattr(config, "hc_mult", 0) == 4
        and getattr(config, "hidden_size", 0) == 2560
        and getattr(parallelism, "tp_size", 1) == 2
        and getattr(parallelism, "local_world_size", 2) >= 2
        and torch.version.hip is None
        and torch.cuda.is_available()
        and dist.is_initialized()
        and collective_torch._initialized
    ):
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not (
        device.type == "cuda"
        and cached_device_capability(device) == (9, 0)
        and torch.cuda.device_count() >= 2
    ):
        return None
    try:
        group = collective_torch._get_group(collective_torch.Group.TP)
    except (RuntimeError, ValueError):
        return None
    if not _tp2_peers_can_access(group, device):
        return None
    key = (id(group), device.index)
    if key in _WORKSPACES:
        return _WORKSPACES[key]
    ws = Qwen4Tp2ArHcWorkspace(group, device)
    _WORKSPACES[key] = ws
    logging.getLogger(__name__).info(
        "Qwen4 fused TP2 AllReduce + HC ready: device=%s max_rows=%s",
        device,
        ws.max_rows,
    )
    return ws
