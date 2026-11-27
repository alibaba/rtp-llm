"""TP2 P2P AllReduce + Hyper-Connection Residual Injection & Norm.

Stages TP-partial sublayer outputs (single-branch Attention/GDN `[rows, 2560]`
or packed dual-branch MoE `[2 * rows, 2560]`) and fuses cross-GPU BF16 reduction
directly into Hyper-Connection residual writeback, grouped RMSNorm, and per-tensor
FP8 scale generation. Decode and medium batches use a 3-slot Lamport one-way
NVLink push; large prefills use a 2-slot one-way push with release/acquire flags.
Enabled by default on SM90 TP2 when custom AllReduce is enabled; falls back to
standalone FlashInfer/NCCL AllReduce when prerequisites are not met.
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

from rtp_llm.models_py.distributed import collective_torch
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
    __slots__ = ("local_tensor", "packed_dual", "workspace", "ep_gather")

    def __init__(
        self,
        local_tensor: torch.Tensor,
        packed_dual: bool,
        workspace: "Qwen4Tp2ArHcWorkspace",
        *,
        ep_gather=None,
    ) -> None:
        self.local_tensor = local_tensor
        self.packed_dual = bool(packed_dual)
        self.workspace = workspace
        self.ep_gather = ep_gather

    def materialize(self) -> torch.Tensor:
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        if self.ep_gather is not None:
            from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
                ep_gather_global_routes,
            )

            eg = self.ep_gather
            routed = torch.empty_like(self.local_tensor)
            ep_gather_global_routes(
                eg.down_out,
                eg.topk_ids,
                eg.topk_weights,
                routed,
                eg.global_start,
                eg.num_local,
            )
            return all_reduce(routed + self.local_tensor, Group.TP, inplace=None)

        tensor = self.local_tensor
        if self.packed_dual:
            # Pre-sum local routed + shared branches before TP reduction so
            # fallback halves the token count and stays within FlashInfer's
            # single-shot workspace without extra clones or copies.
            rows = tensor.shape[0] // 2
            tensor = tensor[:rows] + tensor[rows:]
            return all_reduce(tensor, Group.TP, inplace=None)
        return all_reduce(tensor, Group.TP, inplace=False)


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


def can_defer_moe_ep_gather_tp2(rows: int, device: torch.device) -> bool:
    slot = _moe_tp2_defer_slot.get()
    return (
        slot is not None
        and not slot.consumed
        and slot.workspace.can_stage_ep_gather_rows(rows, device)
    )


def try_defer_moe_ep_gather_tp2(
    ep_gather, shared: torch.Tensor
) -> Optional[DeferredTp2Sublayer]:
    slot = _moe_tp2_defer_slot.get()
    if (
        slot is not None
        and not slot.consumed
        and slot.workspace.can_stage_ep_gather(ep_gather, shared)
    ):
        slot.consumed = True
        return DeferredTp2Sublayer(shared, False, slot.workspace, ep_gather=ep_gather)
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
    max_packed_rows = 12288

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
        self.signals = None
        self.buffers = None
        self.buffer_bytes = 0
        self.lamport_buffers = []
        self.lamport_buffer_bytes = 0
        self.max_lamport_rows = 0
        assert dist.get_world_size(group) == 2
        try:
            with torch.cuda.device(self.device):
                self.signals = cuda_ipc.create_shared_buffer(
                    self.ops.tp2_ar_hc_meta_size(), group
                )
                self.buffers, self.buffer_bytes, max_single_rows = (
                    acquire_tp2_shared_ipc_buffers(group, self.device)
                )
                self.max_rows = int(max_single_rows)
                self.max_packed_rows = self.max_rows
                cuda_ipc.cudart.cudaMemset(
                    ctypes.c_void_p(self.signals[self.rank]),
                    0,
                    self.ops.tp2_ar_hc_meta_size(),
                )
                raw_lamp_rows = os.getenv("RTP_LLM_TP2_LAMPORT_MAX_ROWS", "1024").strip()
                try:
                    parsed_lamp_rows = int(raw_lamp_rows) if raw_lamp_rows else 1024
                except ValueError:
                    parsed_lamp_rows = 1024
                req_lamp_rows = max(0, min(self.max_rows, parsed_lamp_rows))
                if req_lamp_rows > 0 and hasattr(self.ops, "tp2_ar_hc_lamport_buffer_size"):
                    self.max_lamport_rows = req_lamp_rows
                    self.lamport_buffer_bytes = int(
                        self.ops.tp2_ar_hc_lamport_buffer_size(req_lamp_rows)
                    )
                    self.lamport_buffers = cuda_ipc.create_shared_buffer(
                        self.lamport_buffer_bytes, group
                    )
                    self.ops.tp2_ar_hc_init_lamport_buffer(
                        self.lamport_buffers[self.rank], self.lamport_buffer_bytes
                    )
                torch.cuda.synchronize(self.device)
                dist.barrier(group=group)
                probe_x = torch.zeros((1, 10240), device=self.device, dtype=torch.bfloat16)
                probe_s = torch.zeros((1, 2560), device=self.device, dtype=torch.bfloat16)
                probe_g = torch.zeros((1, 4), device=self.device, dtype=torch.bfloat16)
                # Warm up both the Lamport path (if enabled) and the 2-slot push fallback.
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
                    self.lamport_buffers,
                    self.lamport_buffer_bytes,
                )
                if self.lamport_buffers:
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
                        [],
                        0,
                    )
                torch.cuda.synchronize(self.device)
                dist.barrier(group=group)
        except Exception:
            self._cleanup_partial_init()
            raise

    def _cleanup_partial_init(self) -> None:
        from flashinfer.comm import cuda_ipc

        from rtp_llm.models_py.distributed.tp2_p2p_all_gather import (
            release_tp2_shared_ipc_buffers,
        )

        self.closed = True
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        # Avoid collective barriers in free_shared_buffer during failure teardown;
        # release only this rank's local IPC allocations directly.
        if getattr(self, "lamport_buffers", None):
            try:
                if len(self.lamport_buffers) > self.rank and self.lamport_buffers[self.rank]:
                    cuda_ipc.cudart.cudaFree(ctypes.c_void_p(self.lamport_buffers[self.rank]))
            except Exception:
                pass
            self.lamport_buffers = []
        if getattr(self, "signals", None) is not None:
            try:
                if len(self.signals) > self.rank and self.signals[self.rank]:
                    cuda_ipc.cudart.cudaFree(ctypes.c_void_p(self.signals[self.rank]))
            except Exception:
                pass
            self.signals = None
        if getattr(self, "buffers", None) is not None:
            try:
                release_tp2_shared_ipc_buffers(self.group, self.device)
            except Exception:
                pass
            self.buffers = None

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
        if getattr(self, "lamport_buffers", None):
            cuda_ipc.free_shared_buffer(self.lamport_buffers, self.group)
            self.lamport_buffers = []
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

    def can_stage_ep_gather_rows(self, rows: int, device: torch.device) -> bool:
        return (
            device == self.device
            and 1 <= rows <= getattr(self, "max_lamport_rows", 0)
            and bool(getattr(self, "lamport_buffers", None))
            and getattr(self, "lamport_buffer_bytes", 0) > 0
            and hasattr(getattr(self, "ops", None), "tp2_ar_hc_inject_norm_ep_gather")
            and hasattr(getattr(self, "ops", None), "tp2_ar_hc_inject_only_ep_gather")
        )

    def can_stage_ep_gather(self, ep_gather, shared: torch.Tensor) -> bool:
        if not (
            isinstance(shared, torch.Tensor)
            and shared.is_cuda
            and shared.device == self.device
            and shared.dtype == torch.bfloat16
            and shared.ndim == 2
            and shared.shape[1] == 2560
            and shared.is_contiguous()
            and shared.data_ptr() % 16 == 0
            and not shared.requires_grad
        ):
            return False
        rows = int(shared.shape[0])
        if not self.can_stage_ep_gather_rows(rows, shared.device):
            return False
        down = ep_gather.down_out
        ids = ep_gather.topk_ids
        wts = ep_gather.topk_weights
        if not (isinstance(ids, torch.Tensor) and ids.ndim == 2):
            return False
        top_k = int(ids.shape[1])
        return (
            1 <= top_k <= 16
            and isinstance(down, torch.Tensor)
            and down.is_cuda
            and down.device == self.device
            and down.dtype == torch.bfloat16
            and down.shape == (rows * top_k, 2560)
            and down.is_contiguous()
            and down.data_ptr() % 16 == 0
            and not down.requires_grad
            and ids.is_cuda
            and ids.device == self.device
            and ids.dtype in (torch.int32, torch.int64)
            and ids.shape == (rows, top_k)
            and ids.is_contiguous()
            and isinstance(wts, torch.Tensor)
            and wts.is_cuda
            and wts.device == self.device
            and wts.dtype == torch.float32
            and wts.shape == (rows, top_k)
            and wts.is_contiguous()
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
        if deferred.ep_gather is not None:
            if not (
                cached_device_capability(self.device) == (9, 0)
                and self.supports_hyper(hyper_x)
                and self.can_stage_ep_gather(deferred.ep_gather, deferred.local_tensor)
                and deferred.local_tensor.shape[0] == rows
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
            eg = deferred.ep_gather
            residual, normed, scale = self.ops.tp2_ar_hc_inject_norm_ep_gather(
                hyper_x,
                eg.down_out,
                deferred.local_tensor,
                eg.topk_ids,
                eg.topk_weights,
                gates,
                gamma,
                self.lamport_buffers,
                self.signals,
                self.rank,
                int(eg.global_start),
                int(eg.num_local),
                float(eps),
                bool(collect_scale),
                bool(pdl),
                getattr(self, "lamport_buffer_bytes", 0),
            )
            return residual, normed, scale
        expected_sub_rows = 2 * rows if deferred.packed_dual else rows
        if not (
            cached_device_capability(self.device) == (9, 0)
            and self.supports_hyper(hyper_x)
            and 1 <= rows <= self.max_rows
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
        residual, normed, scale = self.ops.tp2_ar_hc_inject_norm(
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
            getattr(self, "lamport_buffers", []),
            getattr(self, "lamport_buffer_bytes", 0),
        )
        return residual, normed, scale

    def try_inject_only(
        self,
        hyper_x: torch.Tensor,
        deferred: DeferredTp2Sublayer,
        gates: torch.Tensor,
    ):
        rows = hyper_x.numel() // 10240 if hyper_x.ndim >= 2 and hyper_x.shape[-1] == 10240 else 0
        if deferred.ep_gather is not None:
            if not (
                cached_device_capability(self.device) == (9, 0)
                and self.supports_hyper(hyper_x)
                and self.can_stage_ep_gather(deferred.ep_gather, deferred.local_tensor)
                and deferred.local_tensor.shape[0] == rows
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
            eg = deferred.ep_gather
            return self.ops.tp2_ar_hc_inject_only_ep_gather(
                hyper_x,
                eg.down_out,
                deferred.local_tensor,
                eg.topk_ids,
                eg.topk_weights,
                gates,
                self.lamport_buffers,
                self.signals,
                self.rank,
                int(eg.global_start),
                int(eg.num_local),
                bool(pdl),
                getattr(self, "lamport_buffer_bytes", 0),
            )
        expected_sub_rows = 2 * rows if deferred.packed_dual else rows
        if not (
            cached_device_capability(self.device) == (9, 0)
            and self.supports_hyper(hyper_x)
            and 1 <= rows <= self.max_rows
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
            getattr(self, "lamport_buffers", []),
            getattr(self, "lamport_buffer_bytes", 0),
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
    logger = logging.getLogger(__name__)
    if not collective_torch.custom_all_reduce_enabled():
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC disabled by custom AR switch "
            "(FT_DISABLE_CUSTOM_AR=%r, cli_override=%s)",
            os.getenv("FT_DISABLE_CUSTOM_AR"),
            getattr(collective_torch, "_disable_custom_all_reduce_override", None),
        )
        return None
    cuda_available = False
    prereq_ok = (
        getattr(config, "hc_mult", 0) == 4
        and getattr(config, "hidden_size", 0) == 2560
        and getattr(parallelism, "tp_size", 1) == 2
        and getattr(parallelism, "local_world_size", 2) >= 2
        and torch.version.hip is None
        and (cuda_available := torch.cuda.is_available())
        and dist.is_initialized()
        and collective_torch._initialized
    )
    if not prereq_ok:
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC skipped: config/dist mismatch "
            "(hc_mult=%s hidden=%s tp=%s local_world=%s cuda=%s dist=%s coll=%s)",
            getattr(config, "hc_mult", 0),
            getattr(config, "hidden_size", 0),
            getattr(parallelism, "tp_size", 1),
            getattr(parallelism, "local_world_size", 2),
            cuda_available,
            dist.is_initialized(),
            collective_torch._initialized,
        )
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not (
        device.type == "cuda"
        and cached_device_capability(device) == (9, 0)
        and torch.cuda.device_count() >= 2
    ):
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC skipped: unsupported device=%s cc=%s dev_count=%s",
            device,
            cached_device_capability(device) if device.type == "cuda" else None,
            torch.cuda.device_count(),
        )
        return None
    try:
        group = collective_torch._get_group(collective_torch.Group.TP)
    except (RuntimeError, ValueError) as exc:
        logger.info("Qwen4 fused TP2 AllReduce + HC skipped: TP group unavailable (%s)", exc)
        return None

    from rtp_llm.models_py.distributed.symm_mem import _agree_across_group

    local_ops_ok = False
    try:
        from flashinfer.comm import cuda_ipc  # noqa: F401

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        local_ops_ok = hasattr(rtp_llm_ops, "tp2_ar_hc_inject_norm") and hasattr(
            rtp_llm_ops, "tp2_ar_hc_inject_only"
        )
    except Exception as exc:
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC prerequisite check failed (%s); "
            "falling back to FlashInfer AllReduce",
            exc,
        )
    if not _agree_across_group(group, local_ops_ok, "tp2_ar_hc_prereq"):
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC skipped: prerequisite check declined; "
            "falling back to FlashInfer AllReduce"
        )
        return None
    if not _tp2_peers_can_access(group, device):
        logger.info(
            "Qwen4 fused TP2 AllReduce + HC skipped: TP2 peer access unavailable on device=%s; "
            "falling back to FlashInfer AllReduce",
            device,
        )
        return None
    key = (id(group), device.index)
    if key in _WORKSPACES:
        return _WORKSPACES[key]
    ws = None
    local_ok = False
    local_err: Optional[Exception] = None
    try:
        ws = Qwen4Tp2ArHcWorkspace(group, device)
        local_ok = True
    except Exception as exc:
        local_err = exc
    if not _agree_across_group(group, local_ok, "tp2_ar_hc_init"):
        if ws is not None:
            try:
                ws._cleanup_partial_init()
            except Exception:
                pass
        logger.warning(
            "Qwen4 fused TP2 AllReduce + HC workspace init failed; "
            "falling back to FlashInfer AllReduce%s",
            f" ({local_err})" if local_err is not None else "",
        )
        return None
    _WORKSPACES[key] = ws
    logger.info(
        "Qwen4 fused TP2 AllReduce + HC ready: device=%s max_rows=%s lamport_rows=%s",
        device,
        ws.max_rows,
        ws.max_lamport_rows,
    )
    return ws
