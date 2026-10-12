"""Select separate released W1/W3 GEMMs on explicit SM120 expert providers."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, _scale_bytes
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


def _shared_projection(owner, values, scales, kernel, *, out=None):
    if values.dtype != torch.float8_e4m3fn:
        raise ValueError("Shared projection requires native E4M3 inputs")
    output = owner._output(values, out, (scales,))
    rows, half = values.numel() // owner.K, owner.N // 2
    raw = _scale_bytes(scales)
    if (
        raw.shape != (rows, owner.K // 32)
        or raw.device != values.device
        or not raw.is_contiguous()
    ):
        raise ValueError("Shared projection requires linear group-32 activation scales")
    if not rows:
        return output
    gemm = kernel.fp8_gemm_kernel(
        half, owner.K, block_size=32, scale_dtype=kernel.FE8M0
    )
    parts = []
    for offset in (0, half):
        part = torch.empty((rows, half), device=values.device, dtype=torch.bfloat16)
        gemm(
            values.reshape(rows, owner.K),
            owner.weight[offset : offset + half],
            part,
            raw.view(torch.float8_e8m0fnu),
            owner.scales[offset // 32 : (offset + half) // 32].view(
                torch.float8_e8m0fnu
            ),
        )
        parts.append(part)
    output.view(rows, owner.N).copy_(torch.cat(parts, dim=-1))
    return output


def _routed_projection(owner, values, scales, key, kernel):
    if values.dtype != torch.float8_e4m3fn or values.device != owner.device:
        raise ValueError("Routed projection requires native E4M3 inputs")
    if (
        scales.shape != (*values.shape[:-1], owner.dim // 32)
        or scales.dtype != torch.uint8
        or scales.device != values.device
    ):
        raise ValueError("Routed projection requires linear group-32 activation scales")
    if (
        values.shape[-1] != owner.dim
        or not values.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("Routed projection requires contiguous admitted inputs")
    result = torch.empty(
        (*values.shape[:-1], 2 * owner.inter_dim),
        device=values.device,
        dtype=torch.bfloat16,
    )
    if key[0] == "masked":
        if values.ndim != 3 or values.shape[0] != owner.experts:
            raise ValueError("Routed masked projection disagrees with expert count")
        segments = [(values[e], scales[e], result[e]) for e in range(owner.experts)]
    elif key[0] == "contiguous":
        counts = key[1]
        if (
            values.ndim != 2
            or len(counts) != owner.experts
            or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != len(values)
        ):
            raise ValueError("Routed contiguous projection disagrees with row counts")
        segments, cursor = [], 0
        for count in counts:
            segments.append(
                (
                    values[cursor : cursor + count],
                    scales[cursor : cursor + count],
                    result[cursor : cursor + count],
                )
            )
            cursor += count
    else:
        raise ValueError("Unknown routed projection layout")
    gemm = kernel.fp4_gemm_kernel(
        owner.inter_dim, owner.dim, act_block_size=32, scale_dtype=kernel.FE8M0
    )
    for e, (a, a_scale, output) in enumerate(segments):
        if not len(a):
            continue
        parts = []
        for offset in (0, owner.inter_dim):
            part = torch.empty(
                (len(a), owner.inter_dim), device=values.device, dtype=torch.bfloat16
            )
            gemm(
                a,
                owner.w13[e, offset : offset + owner.inter_dim].view(
                    torch.float4_e2m1fn_x2
                ),
                part,
                a_scale.view(torch.float8_e8m0fnu),
                owner.s13[e, offset : offset + owner.inter_dim].view(
                    torch.float8_e8m0fnu
                ),
            )
            parts.append(part)
        output.copy_(torch.cat(parts, dim=-1))
    return result


def _admit_shared(owner):
    if type(owner) is not V41MXFP8Linear or owner.backend != "flashinfer":
        raise ValueError("Select native SM120 shared W13 providers")
    if (
        owner.N < 256
        or owner.N % 256
        or owner.K < 128
        or owner.K % 128
        or owner.weight.shape != (owner.N, owner.K)
        or owner.scales.shape != (owner.N // 32, owner.K // 32)
    ):
        raise ValueError(
            "Shared W13 provider metadata disagrees with separate projections"
        )
    if (
        owner.weight.dtype != torch.float8_e4m3fn
        or owner.scales.dtype != torch.uint8
        or not owner.weight.is_cuda
        or owner.scales.device != owner.weight.device
        or not owner.weight.is_contiguous()
        or not owner.scales.is_contiguous()
        or torch.cuda.get_device_capability(owner.weight.device) != (12, 0)
    ):
        raise ValueError(
            "Shared W13 requires immutable contiguous SM120 group-32 operands"
        )


def _admit_routed(owner):
    if type(owner) is not V41RoutedExperts or owner.backend != "deepgemm":
        raise ValueError("Select native SM120 routed W13 owners")
    if (
        owner.experts <= 0
        or min(owner.dim, owner.inter_dim) < 128
        or owner.dim % 128
        or owner.inter_dim % 128
        or owner.w13.shape != (owner.experts, 2 * owner.inter_dim, owner.dim // 2)
        or owner.s13.shape != (owner.experts, 2 * owner.inter_dim, owner.dim // 32)
    ):
        raise ValueError("Routed W13 metadata disagrees with separate projections")
    if (
        owner.w13.dtype != torch.uint8
        or owner.s13.dtype != torch.uint8
        or not owner.w13.is_cuda
        or owner.device != owner.w13.device
        or owner.s13.device != owner.device
        or not owner.w13.is_contiguous()
        or not owner.s13.is_contiguous()
        or torch.cuda.get_device_capability(owner.device) != (12, 0)
    ):
        raise ValueError(
            "Routed W13 requires immutable contiguous SM120 group-32 operands"
        )


@contextmanager
def expert_projection_reference(enabled, *, shared=(), routed=(), kernel=None):
    if type(enabled) is not bool:
        raise ValueError("Expert projection selection requires a bool")
    if not enabled:
        yield
        return
    shared, routed = tuple(shared), tuple(routed)
    if not shared and not routed:
        raise ValueError("Select explicit shared or routed projection owners")
    if len({id(p) for p in shared}) != len(shared) or len(
        {id(p) for p in routed}
    ) != len(routed):
        raise ValueError("Select distinct projection owners")
    for owner in shared:
        _admit_shared(owner)
    for owner in routed:
        _admit_routed(owner)
    kernel = load_reference_kernel() if kernel is None else kernel
    if (
        not hasattr(kernel, "FE8M0")
        or (shared and not callable(getattr(kernel, "fp8_gemm_kernel", None)))
        or (routed and not callable(getattr(kernel, "fp4_gemm_kernel", None)))
    ):
        raise ValueError("Expert projection requires the released GEMM provider")
    selected_shared, selected_routed = {id(p) for p in shared}, {id(p) for p in routed}
    original_dense, original_run = (
        V41MXFP8Linear.forward_quantized,
        V41RoutedExperts._run,
    )

    def dense_forward(owner, values, scales, *, out=None):
        if id(owner) in selected_shared:
            return _shared_projection(owner, values, scales, kernel, out=out)
        return original_dense(owner, values, scales, out=out)

    def routed_run(owner, x, valid, route_weights, key, gemm, *, out, input_scales):
        if id(owner) not in selected_routed:
            return original_run(
                owner,
                x,
                valid,
                route_weights,
                key,
                gemm,
                out=out,
                input_scales=input_scales,
            )

        def projection(values, scales, weight, raw_scales, packed_scales):
            if weight is owner.w13:
                return _routed_projection(owner, values, scales, key, kernel)
            return gemm(values, scales, weight, raw_scales, packed_scales)

        return original_run(
            owner,
            x,
            valid,
            route_weights,
            key,
            projection,
            out=out,
            input_scales=input_scales,
        )

    with ExitStack() as stack:
        if shared:
            stack.enter_context(
                patch.object(V41MXFP8Linear, "forward_quantized", dense_forward)
            )
        if routed:
            stack.enter_context(patch.object(V41RoutedExperts, "_run", routed_run))
        yield
