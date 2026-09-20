"""Projection-only KTP planning and tensor layout for Kimi K3 Decode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_gather_into,
    all_to_all_single,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)

logger = logging.getLogger(__name__)
_LOGGED_PROJECTION_LAYOUTS: set[tuple[int, int, int, int]] = set()


def resolve_projection_local_heads(
    *, total_heads: int, attention_tp_size: int, ktp_size: int
) -> int:
    """Resolve the checkpoint-local projection heads for this topology.

    Projection-KTP replaces attention-TP sharding only when KTP is enabled.
    Prefill and legacy KTP1 Decode retain their attention-TP-local weights.
    """

    if attention_tp_size <= 0 or ktp_size <= 0:
        raise ValueError("attention TP and KTP sizes must be positive")
    projection_parallel_size = ktp_size if ktp_size > 1 else attention_tp_size
    if total_heads % projection_parallel_size:
        raise ValueError(
            f"KDA heads {total_heads} must be divisible by projection parallel "
            f"size {projection_parallel_size}"
        )
    return total_heads // projection_parallel_size


def validate_projection_ktp_sp_type(sp_type: str) -> str:
    """Validate the speculative session type for the score-model KTP path.

    ``PyModelInitResources.is_speculative`` describes the whole serving
    session, so it is true for both target and draft models. Target versus
    draft is instead encoded by the model's parallelism: the target model keeps
    KTP while Eagle3 and K3 MTP proposal construction force the draft model to
    KTP1.
    """

    normalized = str(sp_type).strip().lower()
    if normalized not in ("", "eagle3", "mtp"):
        raise RuntimeError(
            "Projection KTP supports ordinary Decode, Eagle3 target "
            "verification, or K3 MTP target verification; the speculative "
            "model itself must use KTP1, "
            f"got SP_TYPE={normalized!r}"
        )
    return normalized


@dataclass(frozen=True)
class KtpProjectionResult:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    output_gate: torch.Tensor


class KtpProjectionWorkspace:
    """Model-owned KDA communication buffers for the finite Graph buckets.

    The serial target runner shares these across KDA layers. Reassembly copies
    every head section before the next layer reuses the receive buffer. Keep
    the same addresses in eager warmup, capture, and replay: transient A2A
    allocations can hang KTP/MegaMoE replay with the supported NCCL build.
    This workspace must not be shared by concurrently executing models.
    """

    def __init__(self, physical_batches, *, ktp_size, total_heads, head_dim, device):
        if ktp_size <= 1 or total_heads % ktp_size or head_dim <= 0:
            raise ValueError("invalid KTP communication workspace geometry")
        local_heads = total_heads // ktp_size
        self.payload_width = local_heads * (5 * head_dim + 1)
        self.ktp_size = ktp_size
        self.device = torch.device(device)
        self.physical_batches = frozenset(
            batch for batch in physical_batches if batch > 0
        )
        self.buffers = {}

    def _is_capturing(self):
        return self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def get(self, physical_batch):
        buffers = self.buffers.get(physical_batch)
        if buffers is not None:
            return buffers
        if self._is_capturing():
            raise RuntimeError(
                "KTP communication bucket was not prepared before capture"
            )
        if physical_batch not in self.physical_batches:
            # Ordinary eager request sizes do not grow the Graph resource pool.
            return None
        shape = (self.ktp_size * physical_batch, self.payload_width)
        buffers = (
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
        )
        self.buffers[physical_batch] = buffers
        logger.info(
            "[K3_KTP_WORKSPACE] warmup_physical_batch=%d bytes=%d shared_across_kda_layers=True",
            physical_batch,
            sum(t.numel() * t.element_size() for t in buffers),
        )
        return buffers


def _apply_projection(
    inputs: torch.Tensor | QuantizedActivation, projection: Any
) -> torch.Tensor:
    """Apply either a dense weight tensor or a quantized Linear module."""

    if isinstance(projection, torch.Tensor):
        return torch.matmul(inputs, projection)
    return projection(inputs.contiguous())


def _all_gather_projection_input(
    local_input: torch.Tensor | QuantizedActivation,
    *,
    ktp_size: int,
    optimize: bool = False,
) -> torch.Tensor | QuantizedActivation:
    """Gather either BF16 rows or the complete group128 FP8 wire format.

    ``QuantizedActivation`` is deliberately not a Tensor subclass.  Its FP8
    values and UE8M0 scales therefore have to be communicated independently,
    then repacked into the row-major global KTP order expected by the FP8
    projection.  Per-rank scale padding is removed before the global wire is
    assembled; otherwise graph buckets 1 and 2 would attach the wrong scale to
    rows from ranks after rank 0.
    """

    if (
        not isinstance(local_input, QuantizedActivation)
        and optimize
        and local_input.is_cuda
        and local_input.dtype == torch.bfloat16
    ):
        local_input = local_input.contiguous()
        output = torch.empty(
            (ktp_size * local_input.shape[0], *local_input.shape[1:]),
            dtype=local_input.dtype,
            device=local_input.device,
        )
        return all_gather_into(local_input, output, group=Group.KTP)

    if not isinstance(local_input, QuantizedActivation):
        return all_gather(local_input.contiguous(), group=Group.KTP)

    local_rows, hidden_size = local_input.shape
    local_scale_columns = int(local_input.scale_wire.shape[1])
    scale_groups = int(local_input.scale_wire.shape[0])

    gathered_values_wire = all_gather(
        local_input.values.view(torch.uint8).contiguous(), group=Group.KTP
    )
    gathered_values = gathered_values_wire.view(torch.float8_e4m3fn)
    expected_values_shape = (ktp_size * local_rows, hidden_size)
    if tuple(gathered_values.shape) != expected_values_shape:
        raise RuntimeError(
            "KTP FP8 value AllGather returned shape "
            f"{tuple(gathered_values.shape)}, expected {expected_values_shape}"
        )

    gathered_scale_wire = all_gather(
        local_input.scale_wire.contiguous(), group=Group.KTP
    )
    expected_scale_shape = (ktp_size * scale_groups, local_scale_columns)
    if tuple(gathered_scale_wire.shape) != expected_scale_shape:
        raise RuntimeError(
            "KTP FP8 scale AllGather returned shape "
            f"{tuple(gathered_scale_wire.shape)}, expected {expected_scale_shape}"
        )

    # all_gather concatenates dim 0, yielding [rank, K-group, local-M-pad].
    # Drop each rank's local alignment columns before concatenating M globally.
    scale_by_rank = gathered_scale_wire.reshape(
        ktp_size, scale_groups, local_scale_columns
    )
    global_scale_wire = (
        scale_by_rank[:, :, :local_rows]
        .permute(1, 0, 2)
        .contiguous()
        .reshape(scale_groups, ktp_size * local_rows)
    )
    return QuantizedActivation(gathered_values, global_scale_wire)


def pack_ktp_projection_payload(
    gathered_hidden: torch.Tensor,
    fused_projection: Any,
    forget_up_projection: Any,
    *,
    total_heads: int,
    head_dim: int,
    forget_latent_size: int,
    ktp_size: int,
    ktp_rank: int,
    output: torch.Tensor | None = None,
    optimize: bool = False,
) -> torch.Tensor:
    """Project all owner rows with one rank's head shard and pack A2A input."""

    local_heads = total_heads // ktp_size
    local_projection_size = local_heads * head_dim
    projected = _apply_projection(gathered_hidden, fused_projection)
    sizes = [local_projection_size] * 4 + [forget_latent_size, total_heads]
    projected_width = sum(sizes)
    if projected.ndim != 2 or projected.shape[1] != projected_width:
        raise RuntimeError(
            "KTP fused projection must be 2D with width "
            f"{projected_width}, got shape={tuple(projected.shape)}"
        )
    # Keep the original row stride so the optimized FP8 producer can consume
    # this view without materializing the other fused-projection sections.
    forget_latent = projected.narrow(1, 4 * local_projection_size, forget_latent_size)
    use_strided_fp8_forget = (
        optimize
        and forget_latent_size == 128
        and forget_latent.is_cuda
        and forget_latent.dtype == torch.bfloat16
        and forget_latent.ndim == 2
        and forget_latent.stride(1) == 1
        and forget_latent.stride(0) >= 128
        and getattr(forget_up_projection, "K", None) == 128
        and getattr(forget_up_projection, "scale_ue8m0", False)
        and callable(getattr(forget_up_projection, "forward_quantized", None))
    )
    if use_strided_fp8_forget:
        from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_quant import (
            quantize_forget_latent_fp8,
        )

        raw_gate = forget_up_projection.forward_quantized(
            *quantize_forget_latent_fp8(forget_latent)
        )
    else:
        raw_gate = _apply_projection(forget_latent, forget_up_projection)
    output_shape = (projected.shape[0], 5 * local_projection_size + local_heads)
    if output is not None and (
        tuple(output.shape) != output_shape
        or output.device != projected.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "KTP projection pack output must be contiguous and match "
            f"shape={output_shape}, device={projected.device}"
        )
    use_optimized_pack = (
        optimize
        and projected.is_cuda
        and projected.dtype == torch.bfloat16
        and raw_gate.is_cuda
        and raw_gate.dtype == torch.bfloat16
        and projected.stride(1) == 1
        and raw_gate.stride(1) == 1
    )
    if use_optimized_pack:
        if output is not None and output.dtype != projected.dtype:
            raise ValueError(
                "optimized KTP projection pack output dtype must match "
                f"projection dtype={projected.dtype}, got {output.dtype}"
            )
        if output is None:
            output = torch.empty(
                output_shape, dtype=projected.dtype, device=projected.device
            )
        from rtp_llm.models_py.triton_kernels.kimi_kda.projection_ktp import (
            pack_ktp_projection_payload_cuda,
        )

        return pack_ktp_projection_payload_cuda(
            projected,
            raw_gate,
            output,
            local_projection_size=local_projection_size,
            forget_latent_size=forget_latent_size,
            local_heads=local_heads,
            ktp_rank=ktp_rank,
        )

    q, k, v, output_gate, _, full_raw_beta = torch.split(projected, sizes, dim=1)
    raw_beta = full_raw_beta.narrow(1, ktp_rank * local_heads, local_heads)
    packed = torch.cat((q, k, v, output_gate, raw_gate, raw_beta), dim=-1)
    if output is None:
        return packed
    output.copy_(packed)
    return output


def reassemble_ktp_projection_payload(
    received: torch.Tensor,
    *,
    ktp_size: int,
    physical_batch: int,
    local_projection_size: int,
    local_heads: int,
    optimize: bool = False,
) -> KtpProjectionResult:
    """Convert source-major A2A payload into owner-major full-head tensors."""

    payload_width = 5 * local_projection_size + local_heads
    expected = (ktp_size * physical_batch, payload_width)
    if tuple(received.shape) != expected:
        raise ValueError(
            f"KTP A2A payload shape {tuple(received.shape)} != expected {expected}"
        )
    if (
        optimize
        and received.is_cuda
        and received.dtype == torch.bfloat16
        and received.is_contiguous()
    ):
        from rtp_llm.models_py.triton_kernels.kimi_kda.projection_ktp import (
            reassemble_ktp_projection_payload_cuda,
        )

        q, k, v, output_gate, raw_gate, raw_beta = (
            reassemble_ktp_projection_payload_cuda(
                received,
                ktp_size=ktp_size,
                physical_batch=physical_batch,
                local_projection_size=local_projection_size,
                local_heads=local_heads,
            )
        )
        return KtpProjectionResult(q, k, v, raw_gate, raw_beta, output_gate)

    source_major = received.reshape(ktp_size, physical_batch, payload_width)
    sections = torch.split(
        source_major,
        [
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_heads,
        ],
        dim=-1,
    )

    def _heads_full(section: torch.Tensor) -> torch.Tensor:
        return section.permute(1, 0, 2).contiguous().reshape(physical_batch, -1)

    q, k, v, output_gate, raw_gate, raw_beta = (
        _heads_full(section) for section in sections
    )
    return KtpProjectionResult(q, k, v, raw_gate, raw_beta, output_gate)


def project_kda_inputs_ktp(
    hidden_states: torch.Tensor,
    fused_projection: Any,
    forget_up_projection: Any,
    *,
    total_heads: int,
    head_dim: int,
    forget_latent_size: int,
    ktp_size: int,
    ktp_rank: int,
    workspace: KtpProjectionWorkspace | None = None,
    optimize: bool = False,
) -> KtpProjectionResult:
    """Run KDA's projection-only KTP AllGather/GEMM/AllToAll pipeline."""

    physical_batch = int(hidden_states.shape[0])
    local_heads = total_heads // ktp_size
    local_projection_size = local_heads * head_dim
    layout_key = (ktp_rank, ktp_size, physical_batch, local_heads)
    if layout_key not in _LOGGED_PROJECTION_LAYOUTS:
        logger.info(
            "[K3_PROJECTION_KTP_LAYOUT] rank=%d size=%d physical_batch=%d "
            "heads=%d local_heads=%d collectives=AllGather,AllToAll",
            ktp_rank,
            ktp_size,
            physical_batch,
            total_heads,
            local_heads,
        )
        _LOGGED_PROJECTION_LAYOUTS.add(layout_key)
    gathered_hidden = _all_gather_projection_input(
        hidden_states,
        ktp_size=ktp_size,
        optimize=optimize,
    )
    buffers = workspace.get(physical_batch) if workspace is not None else None
    send = pack_ktp_projection_payload(
        gathered_hidden,
        fused_projection,
        forget_up_projection,
        total_heads=total_heads,
        head_dim=head_dim,
        forget_latent_size=forget_latent_size,
        ktp_size=ktp_size,
        ktp_rank=ktp_rank,
        output=buffers[0] if buffers is not None else None,
        optimize=optimize,
    )
    received = (
        all_to_all_single(send, group=Group.KTP, output=buffers[1])
        if buffers is not None
        else all_to_all_single(send, group=Group.KTP)
    )
    return reassemble_ktp_projection_payload(
        received,
        ktp_size=ktp_size,
        physical_batch=physical_batch,
        local_projection_size=local_projection_size,
        local_heads=local_heads,
        optimize=optimize,
    )


__all__ = [
    "KtpProjectionResult",
    "KtpProjectionWorkspace",
    "pack_ktp_projection_payload",
    "project_kda_inputs_ktp",
    "reassemble_ktp_projection_payload",
    "resolve_projection_local_heads",
    "validate_projection_ktp_sp_type",
]
