"""Test-only released output GEMMs through maintained SM120 expert owners."""

import hashlib
import importlib.util
import os
import sys
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, _scale_bytes
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts

_REFERENCE_MODULES = {}


def load_reference_kernel():
    path = Path(os.environ["DSV41_TEST_REFERENCE_KERNEL"]).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = (str(path), digest)
    if key not in _REFERENCE_MODULES:
        name = f"_dsv41_expert_reference_{digest}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _REFERENCE_MODULES[key] = module
    return _REFERENCE_MODULES[key]


def _dense_down(owner, values, scales, kernel, *, out=None):
    if not isinstance(owner, V41MXFP8Linear) or owner.backend != "flashinfer":
        raise ValueError("Shared output diagnostic requires the native group-32 owner")
    if values.dtype != torch.float8_e4m3fn:
        raise ValueError("Released output GEMM requires native E4M3 inputs")
    output = owner._output(values, out, (scales,))
    rows = values.numel() // owner.K
    raw = _scale_bytes(scales)
    if (
        raw.shape != (rows, owner.K // 32)
        or raw.device != values.device
        or not raw.is_contiguous()
    ):
        raise ValueError("Released output GEMM requires linear group-32 scales")
    if rows:
        gemm = kernel.fp8_gemm_kernel(
            owner.N, owner.K, block_size=32, scale_dtype=kernel.FE8M0
        )
        gemm(
            values.reshape(rows, owner.K),
            owner.weight,
            output.reshape(rows, owner.N),
            raw.view(torch.float8_e8m0fnu),
            owner.scales.view(torch.float8_e8m0fnu),
        )
    return output


def _routed_down(owner, values, scales, key, kernel):
    if owner.backend != "deepgemm":
        raise ValueError("Routed output diagnostic requires the native grouped owner")
    result = torch.empty(
        (*values.shape[:-1], owner.dim), device=values.device, dtype=torch.bfloat16
    )
    gemm = kernel.fp4_gemm_kernel(
        owner.dim, owner.inter_dim, act_block_size=32, scale_dtype=kernel.FE8M0
    )
    if key[0] == "masked":
        segments = [(values[e], scales[e], result[e]) for e in range(owner.experts)]
    elif key[0] == "contiguous":
        segments, cursor = [], 0
        for count in key[1]:
            segments.append(
                (
                    values[cursor : cursor + count],
                    scales[cursor : cursor + count],
                    result[cursor : cursor + count],
                )
            )
            cursor += count
    else:
        raise ValueError("Unknown native routed output layout")
    for e, (a, a_scale, out) in enumerate(segments):
        if a.shape[0]:
            gemm(
                a.contiguous(),
                owner.w2[e].view(torch.float4_e2m1fn_x2),
                out,
                a_scale.contiguous().view(torch.float8_e8m0fnu),
                owner.s2[e].view(torch.float8_e8m0fnu),
            )
    return result


@contextmanager
def expert_down_reference(enabled, *, shared=True, routed=False, kernel=None):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Released output-GEMM diagnostic is qualified only on SM120")
    if not (shared or routed):
        raise ValueError("Output-GEMM diagnostic requires a selected expert owner")
    kernel = load_reference_kernel() if kernel is None else kernel
    original_shared, original_run = W13SharedExpert.forward, V41RoutedExperts._run
    original_dense = V41MXFP8Linear.forward_quantized
    selected = ContextVar("v41_diagnostic_shared_down", default=None)

    def shared_forward(owner, *args, **kwargs):
        # The maintained upstream adapter owns packed-int32 quantization. Its
        # nested SM120 provider receives the canonical linear scale bytes.
        token = selected.set(getattr(owner.w2, "_sm120_linear", None))
        try:
            return original_shared(owner, *args, **kwargs)
        finally:
            selected.reset(token)

    def dense_forward(owner, values, scales, *, out=None):
        if owner is selected.get():
            return _dense_down(owner, values, scales, kernel, out=out)
        return original_dense(owner, values, scales, out=out)

    def routed_run(owner, x, valid, route_weights, key, gemm, *, out, input_scales):
        if owner.backend != "deepgemm":
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

        def output_gemm(values, scales, weight, raw_scales, packed_scales):
            if weight is owner.w2:
                return _routed_down(owner, values, scales, key, kernel)
            return gemm(values, scales, weight, raw_scales, packed_scales)

        return original_run(
            owner,
            x,
            valid,
            route_weights,
            key,
            output_gemm,
            out=out,
            input_scales=input_scales,
        )

    with ExitStack() as stack:
        if shared:
            stack.enter_context(
                patch.object(W13SharedExpert, "forward", shared_forward)
            )
            stack.enter_context(
                patch.object(V41MXFP8Linear, "forward_quantized", dense_forward)
            )
        if routed:
            stack.enter_context(patch.object(V41RoutedExperts, "_run", routed_run))
        yield
