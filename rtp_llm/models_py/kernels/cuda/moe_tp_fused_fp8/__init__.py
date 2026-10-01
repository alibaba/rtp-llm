"""Opt-in correctness baseline for pure-TP2 FC2 and FP8 communication fusion.

This module is not registered in the serving model path. The scalar FC2
baseline fixes a numerical contract before a tensor-core implementation is
introduced. It must not be used as a performance replacement for grouped GEMM.
"""

from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path
from typing import Optional

_MODULE = None
_SWITCH = "RTP_LLM_MOE_TP_FUSED_FP8_AR"


def read_enabled() -> bool:
    value = os.environ.get(_SWITCH, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{_SWITCH} must be 0 or 1")
    return value == "1"


def is_supported(device=None) -> bool:
    if not read_enabled():
        return False
    import torch

    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) == (12, 0)
    )


def load_native():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    import torch
    from torch.utils.cpp_extension import load

    if not is_supported():
        raise RuntimeError(
            "MoE TP fused FP8 baseline requires explicit enable and SM120"
        )
    root = Path(__file__).parent
    sources = sorted((*root.glob("*.cu"), *root.glob("*.cuh")))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    # Disallow implicit contraction in the block-scale/route arithmetic. The
    # FC2 reference intentionally uses explicit fmaf for each FP8 dot product.
    digest.update(b"scalar-v1-fmad-false")
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if previous is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    try:
        _MODULE = load(
            name=f"rtp_moe_tp_fused_fp8_{digest.hexdigest()[:12]}",
            sources=[str(root / "moe_tp_fused_fp8.cu")],
            extra_cuda_cflags=["-O3", "--fmad=false", "--expt-relaxed-constexpr"],
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    return _MODULE


def _gather(value, group):
    import torch.distributed as dist

    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _agree(error, signature, group, phase):
    gathered = _gather((error, signature), group)
    if any(item[0] is not None for item in gathered):
        raise ValueError(f"MoE TP fused FP8 {phase}: {gathered}")
    if any(item[1] != gathered[0][1] for item in gathered):
        raise ValueError(f"MoE TP fused FP8 {phase} differs across ranks: {gathered}")


class MoeTpFusedFp8Context:
    """Explicit, collective benchmark API with an independently owned workspace.

    All ranks must call create, operations, and close in the same order. Calls
    validate metadata collectively before launching. The native API is exposed
    only for already-validated fixed-input kernel benchmarks. No serving code
    imports or selects this module automatically. Operations enqueue work;
    call synchronize() on both ranks before accepting any result, to check the
    sticky device error status. Inputs and computed outputs must remain finite.
    """

    @classmethod
    def create(cls, group, device, max_numel: int, blocks: Optional[int] = None):
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "create the baseline context collectively outside graph capture"
            )
        error = None
        try:
            enabled = read_enabled()
        except Exception as exc:
            enabled, error = None, str(exc)
        _agree(error, enabled, group, "configuration")
        if not enabled:
            return None
        return cls(group, device, max_numel, blocks)

    def __init__(self, group, device, max_numel, blocks=None):
        import torch
        import torch.distributed as dist

        self.group = group
        self.device = torch.device(device)
        self.max_numel = max_numel
        self.native = None
        self.closed = False
        self.calls = 0
        error = None
        identity = None
        try:
            if not read_enabled():
                raise ValueError("construct through create; feature is disabled")
            if dist.get_world_size(group) != 2 or dist.get_backend(group) != "nccl":
                raise ValueError("requires a two-rank NCCL group for bootstrap")
            if dist.get_world_size() != 2:
                raise ValueError("standalone baseline requires a two-rank world")
            if self.device.type != "cuda":
                raise ValueError("requires a CUDA device")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if not is_supported(self.device):
                raise ValueError("requires NVIDIA SM120")
            if not isinstance(max_numel, int) or max_numel <= 0:
                raise ValueError("max_numel must be positive")
            properties = torch.cuda.get_device_properties(self.device)
            identity = (socket.gethostname(), str(properties.uuid))
            # Zero is an internal native sentinel: resolve from actual compiled
            # occupancy and workspace capacity, rather than limiting large TP
            # workloads to a fixed 32-CTA grid. Explicit benchmark grids remain.
            self.blocks = 0 if blocks is None else blocks
            if not isinstance(self.blocks, int) or (
                self.blocks < 1 and blocks is not None
            ):
                raise ValueError("blocks must be a positive integer")
            self.rank = dist.get_rank(group)
        except Exception as exc:
            error = repr(exc)
        _agree(error, (max_numel, blocks), group, "initialization")
        identities = _gather(identity, group)
        if (
            len({item[0] for item in identities}) != 1
            or len({item[1] for item in identities}) != 2
        ):
            raise ValueError("requires two distinct GPUs on the same host")
        _agree(None, self.blocks, group, "resident grid")
        # CPU-only validation keeps metadata collectives out of GPU timing and
        # can report asymmetric capture/shape errors without capturing NCCL.
        self.group = dist.new_group(
            ranks=dist.get_process_group_ranks(group),
            backend="gloo",
            # All world ranks participate. Use monotonic group names: hashed
            # rank-list names may reuse stale store keys after group teardown.
            use_local_synchronization=False,
        )
        group = self.group
        error, handle = None, None
        try:
            with torch.cuda.device(self.device):
                self.stream = torch.cuda.Stream(device=self.device)
                self.native = load_native().MoeTpFusedFp8(
                    max_numel, self.device.index, self.rank, self.blocks
                )
                self.blocks = self.native.blocks()
                torch.cuda.current_stream(self.device).synchronize()
                handle = self.native.get_ipc_handle()
        except Exception as exc:
            error = repr(exc)
        try:
            _agree(error, None, group, "JIT/allocation")
            _agree(None, self.blocks, group, "resolved resident grid")
            handles = _gather(handle, group)
            error = None
            try:
                self.native.open_peer(handles[1 - self.rank])
            except Exception as exc:
                error = repr(exc)
            _agree(error, None, group, "IPC mapping")
        except Exception:
            if self.native is not None:
                self.native.close_peer()
            dist.barrier(group=group)
            if self.native is not None:
                self.native.close()
            self.closed = True
            dist.destroy_process_group(self.group)
            raise

    def _validate_tensors(self, tensors, specs, phase):
        import torch

        error = None
        signature = []
        try:
            if self.closed:
                raise ValueError("context is closed")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("baseline does not support CUDA graph capture")
            for tensor, (shape, dtype) in zip(tensors, specs):
                if tensor.device != self.device or tensor.dtype != dtype:
                    raise ValueError("tensor device/dtype does not match contract")
                if tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous():
                    raise ValueError("tensor shape/layout does not match contract")
                signature.append((tuple(tensor.shape), str(tensor.dtype)))
            if tensors[-1].numel() <= 0 or tensors[-1].numel() > self.max_numel:
                raise ValueError("output does not fit the preallocated workspace")
        except Exception as exc:
            error = repr(exc)
        _agree(error, signature, self.group, phase)

    def _launch(self, method, tensors, out):
        import torch

        with torch.cuda.device(self.device):
            producer = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(producer)
            with torch.cuda.stream(self.stream):
                getattr(self.native, method)(*tensors, out)
                for tensor in (*tensors, out):
                    if tensor is not None:
                        tensor.record_stream(self.stream)
                done = torch.cuda.Event()
                done.record(self.stream)
            producer.wait_event(done)
        self.calls += 1
        return out

    def all_reduce(self, partial, out=None):
        import torch

        if out is None:
            out = torch.empty_like(partial)
        self._validate_tensors(
            (partial, out),
            ((partial.shape, torch.bfloat16), (partial.shape, torch.bfloat16)),
            "one-shot inputs",
        )
        error = None
        try:
            if not bool(torch.isfinite(partial).all().item()):
                raise ValueError("one-shot input must be finite")
            if max(partial.data_ptr(), out.data_ptr()) < min(
                partial.data_ptr() + partial.nbytes, out.data_ptr() + out.nbytes
            ):
                raise ValueError("one-shot output overlaps input allocation")
        except Exception as exc:
            error = repr(exc)
        _agree(error, None, self.group, "one-shot values/aliasing")
        return self._launch("all_reduce", (partial,), out)

    def fc2(
        self,
        a,
        a_scale,
        w,
        w_scale,
        ids,
        route_weights,
        gated_shared,
        *,
        fused=True,
        out=None,
    ):
        import torch

        error = None
        shape = None
        try:
            tokens, experts = a.shape[0], w.shape[0]
            if tokens <= 0 or not 0 < experts <= 256:
                raise ValueError("requires positive tokens and 1..256 experts")
            shape = (tokens, experts)
        except Exception as exc:
            error = repr(exc)
        _agree(error, shape, self.group, "FC2 dimensions")
        if out is None:
            out = torch.empty((tokens, 2048), dtype=torch.bfloat16, device=self.device)
        tensors = (a, a_scale, w, w_scale, ids, route_weights, gated_shared, out)
        specs = (
            ((tokens, 8, 256), torch.float8_e4m3fn),
            ((tokens, 8, 2), torch.float32),
            ((experts, 2048, 256), torch.float8_e4m3fn),
            ((experts, 16, 2), torch.float32),
            ((tokens, 8), torch.int32),
            ((tokens, 8), torch.float32),
            ((tokens, 2048), torch.bfloat16),
            ((tokens, 2048), torch.bfloat16),
        )
        self._validate_tensors(tensors, specs, "FC2 inputs")
        error = None
        try:
            if bool((ids >= experts).any().item()):
                raise ValueError("route ID exceeds the expert allocation")
            for tensor in (a, a_scale, w, w_scale, route_weights, gated_shared):
                if not bool(torch.isfinite(tensor.float()).all().item()):
                    raise ValueError("FC2 inputs must be finite")
            out_begin, out_end = out.data_ptr(), out.data_ptr() + out.nbytes
            for tensor in tensors[:-1]:
                begin, end = tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes
                if max(begin, out_begin) < min(end, out_end):
                    raise ValueError("FC2 output overlaps an input allocation")
        except Exception as exc:
            error = repr(exc)
        route_digest = None
        if error is None:
            digest = hashlib.sha256()
            for tensor in (ids, route_weights):
                digest.update(tensor.cpu().view(torch.uint8).numpy().tobytes())
            route_digest = digest.hexdigest()
        _agree(error, (bool(fused), route_digest), self.group, "FC2 route/mode")
        return self._launch("fused_fc2" if fused else "local_fc2", tensors[:-1], out)

    def synchronize(self):
        self.stream.synchronize()
        status = int(self.native.error_status().item())
        _agree(
            None if status == 0 else f"sticky native error status={status:#x}",
            None,
            self.group,
            "completion",
        )

    def close(self):
        import torch.distributed as dist

        if self.closed:
            return
        # A reported protocol error is sticky, but after both kernels finish
        # it must not prevent paired mapping/allocation teardown.
        self.stream.synchronize()
        errors = _gather(int(self.native.error_status().item()), self.group)
        dist.barrier(group=self.group)
        self.native.close_peer()
        dist.barrier(group=self.group)
        self.native.close()
        dist.destroy_process_group(self.group)
        self.closed = True
        if any(errors):
            raise RuntimeError(f"closed failed MoE TP fused FP8 context: {errors}")


__all__ = ["MoeTpFusedFp8Context", "is_supported", "load_native", "read_enabled"]
