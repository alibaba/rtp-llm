"""Fuse a V4.1 attention post-mix and the following FFN pre-mix/RMSNorm.

This seam is internal to one block, so Engram injection, auxiliary hidden
capture and the final delayed readout keep their existing boundaries. Both
flat prefill and batched decode layouts retain BF16 outputs. Cross-layer
decode can additionally request native FP8 output for the next projection.
The intra-layer decode seam can instead write routed/shared FP8 scales
directly into the following MegaMoE's buffers, retaining BF16 for its gate.

DeepGEMM initializes persistent split barriers per CUDA stream. Warm up on
the capture stream before recording a graph; an unwarmed capture stream
retains the unfused path instead of initializing barriers during capture.

SM90 low-token decode uses the vLLM two-kernel boundary: post mapping and the
next projection are fused first, followed by coefficient reduction, Sinkhorn,
delayed readout and RMSNorm in one epilogue kernel.  SM90 mid/large-batch
decode (``64 < M <= 512``) instead uses a three-kernel boundary: post mapping
(optionally consuming the deferred routed/shared pair), a split-K TF32
projection, and one epilogue that also applies the downstream layer RMSNorm.
"""

from __future__ import annotations

import importlib
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch

from rtp_llm.models_py.modules.dsv41._prefill_stream import default_eager_prefill
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv41.moe.mega_se_input_packer import MegaMoEInputs

QuantizedInput = tuple[torch.Tensor, torch.Tensor]
MHCOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
MHCQuantizedOutput = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, QuantizedInput
]

_WARMED_STREAMS: set[tuple[int, int]] = set()


@lru_cache(maxsize=1)
def _get_mega_mhc() -> Callable | None:
    try:
        module = importlib.import_module("deep_gemm")
    except ModuleNotFoundError as error:
        if error.name == "deep_gemm":
            return None
        raise
    operation = getattr(module, "mega_mhc", None)
    return operation if callable(operation) else None


@lru_cache(maxsize=1)
def _get_sm90_mega_mhc():
    try:
        return importlib.import_module(
            "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_two_kernel"
        )
    except (ImportError, OSError):
        # Optional TileLang dependencies are not present in every build.
        return None


@lru_cache(maxsize=1)
def _get_sm90_largebs_mega_mhc():
    try:
        return importlib.import_module(
            "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_largebs_three_kernel"
        )
    except (ImportError, OSError):
        # Optional TileLang dependencies are not present in every build.
        return None


def _is_dense_tensor(value, shape, dtype, device) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device == device
        and value.is_contiguous()
    )


def _stream_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    return stream.device.index, stream.cuda_stream


class _PrefillMHCStream:
    def __init__(self, device: torch.device):
        self.device = device
        self.key = _stream_key(device)
        self.warmed = self.key in _WARMED_STREAMS


_PREFILL_STREAM: ContextVar[_PrefillMHCStream | None] = ContextVar(
    "dsv41_mhc_prefill_stream", default=None
)


@contextmanager
def prefill_mhc_stream(device: torch.device, *, reuse_capture: bool = False):
    """The prefill layer loop submits mHC on its caller's stream.

    Attention and CP side streams restore that stream before returning. Keep
    standalone calls and externally captured forwards on the checked path.
    """
    with torch.cuda.device(device):
        stream = (
            None
            if torch.cuda.is_current_stream_capturing()
            else _PrefillMHCStream(device)
        )
        token = _PREFILL_STREAM.set(stream)
        try:
            with default_eager_prefill(
                reuse_capture and stream is not None and stream.key[1] == 0
            ):
                yield stream is not None
        finally:
            _PREFILL_STREAM.reset(token)


def _prefill_stream(device, use_prepared):
    stream = _PREFILL_STREAM.get() if use_prepared else None
    return stream if stream is not None and stream.device == device else None


def _stream_supports_mhc(device, use_prepared):
    if _prefill_stream(device, use_prepared) is not None:
        return True
    with torch.cuda.device(device):
        return (
            not torch.cuda.is_current_stream_capturing()
            or _stream_key(device) in _WARMED_STREAMS
        )


def _can_try_sm90_mega_mhc(attn_out: torch.Tensor) -> bool:
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.ndim != 3
        or attn_out.shape[-1] != 5120
    ):
        return False
    if tuple(torch.cuda.get_device_capability(attn_out.device)) != (9, 0):
        return False
    backend = _get_sm90_mega_mhc()
    return (
        backend is not None
        and 0 < attn_out.numel() // attn_out.shape[-1] <= backend.MAX_TOKENS
    )


def _can_try_sm90_largebs_mega_mhc(attn_out: torch.Tensor) -> bool:
    """Phase-2 three-kernel window: ``64 < M <= 512`` on SM90."""
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.ndim != 3
        or attn_out.shape[-1] != 5120
    ):
        return False
    if tuple(torch.cuda.get_device_capability(attn_out.device)) != (9, 0):
        return False
    backend = _get_sm90_largebs_mega_mhc()
    if backend is None:
        return False
    tokens = attn_out.numel() // attn_out.shape[-1]
    return backend.MIN_TOKENS < tokens <= backend.MAX_TOKENS


def _check_units(previous, next_hc, norm, device) -> bool:
    if device.type != "cuda":
        return False
    # Subclasses may change either mix equations or norm rounding, so only
    # bypass the implementations whose contracts this fusion reproduces.
    if type(previous) is not DelayedHCUnit or type(next_hc) is not DelayedHCUnit:
        return False
    if (
        previous.dim != 5120
        or next_hc.dim != 5120
        or previous.hc_mult != 4
        or next_hc.hc_mult != 4
        or next_hc._previous_ref is None
        or next_hc._previous_ref() is not previous
        or next_hc.hc_sinkhorn_iters < 1
        or type(norm).__module__ != "rtp_llm.models_py.modules.base.cuda.norm"
        or type(norm).__name__ != "RMSNorm"
    ):
        return False
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

    if type(norm) is not RMSNorm:
        return False
    contracts = (
        (next_hc.fn, (24, 20480), torch.float32),
        (next_hc.scale, (3,), torch.float32),
        (next_hc.base, (24,), torch.float32),
        (norm.weight, (5120,), torch.bfloat16),
    )
    if any(
        not _is_dense_tensor(tensor, shape, dtype, device)
        for tensor, shape, dtype in contracts
    ):
        return False
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor, _, _ in contracts
    ):
        return False
    if any(
        not isinstance(eps, (int, float)) or eps < 0
        for eps in (next_hc.norm_eps, next_hc.hc_eps, norm.variance_epsilon)
    ):
        return False
    return (
        torch.cuda.get_device_capability(device)[0] == 10
        and _get_mega_mhc() is not None
    )


@torch.no_grad()
def prepare_mega_mhc(previous, next_hc, norm) -> None:
    """Bind eligibility to a serving layer after loading its weights and norm.

    Serving updates copy values into the existing tensors. After changing fixed
    weight layouts or layer configuration, prepare the pair again before serving.
    """
    if type(next_hc) is not DelayedHCUnit:
        return
    next_hc._mega_mhc_supported = isinstance(next_hc.fn, torch.Tensor) and _check_units(
        previous, next_hc, norm, next_hc.fn.device
    )
    if next_hc._mega_mhc_supported:
        next_hc._mega_mhc_norm = weakref.ref(norm)


def _units_supported(previous, next_hc, norm, device, use_prepared) -> bool:
    if (
        use_prepared
        and torch.is_inference_mode_enabled()
        and not torch.is_grad_enabled()
    ):
        supported = getattr(next_hc, "_mega_mhc_supported", None)
        if supported is not None:
            return (
                supported
                and next_hc._previous_ref is not None
                and next_hc._previous_ref() is previous
                and next_hc._mega_mhc_norm() is norm
                and next_hc.fn.device == device
            )
    return _check_units(previous, next_hc, norm, device)


def _compact_execution_tokens(original_tokens, tokens, next_hc, norm):
    if (
        type(original_tokens) is not int
        or type(tokens) is not int
        or not 0 < tokens <= original_tokens <= 1 << 20
    ):
        return None
    get_num_sms = getattr(importlib.import_module("deep_gemm"), "get_num_sms", None)
    if not callable(get_num_sms):
        return None
    num_sms = get_num_sms()
    if type(num_sms) is not int or num_sms <= 0:
        return None

    def splits(rows):
        # Pinned DeepGEMM mega_mhc.hpp: H5120 has 80 K64 blocks.
        # Deterministic mode already forces 16 for both calls; never change it.
        maximum = min(64, max(16, num_sms // ((rows + 63) // 64)))
        blocks_per_split = (80 + maximum - 1) // maximum
        return (80 + blocks_per_split - 1) // blocks_per_split

    target = splits(original_tokens)
    if splits(tokens) == target:
        return tokens
    # Bound extra storage independently of original prefill length. At 152 SMs
    # the split27/20/16 representatives are 256/384/512; split40 needs no pad.
    if not all(
        eps > 0 for eps in (next_hc.norm_eps, next_hc.hc_eps, norm.variance_epsilon)
    ):
        return None
    for rows in range(((tokens + 63) // 64) * 64, 513, 64):
        if splits(rows) == target:
            return rows
    return None


def can_preserve_compact_mhc(
    previous: DelayedHCUnit,
    next_hc: DelayedHCUnit,
    norm: torch.nn.Module,
    *,
    original_tokens: int,
    compact_tokens: int,
    use_prepared: bool = False,
) -> bool:
    """Pre-compaction metadata gate, before the previous pre-mix is produced.

    This requires the same delayed-HC backend as the original row count; an
    unfused small-M fallback is not an arithmetic-preserving substitute.
    Runtime activations are still checked by ``is_supported`` after compaction.
    Revalidate the mirrored host split heuristic when upgrading DeepGEMM.
    """
    if type(next_hc) is not DelayedHCUnit or not isinstance(next_hc.fn, torch.Tensor):
        return False
    device = next_hc.fn.device
    if not _units_supported(previous, next_hc, norm, device, use_prepared):
        return False
    if (
        _compact_execution_tokens(original_tokens, compact_tokens, next_hc, norm)
        is None
    ):
        return False
    return _stream_supports_mhc(device, use_prepared)


def is_supported(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous: DelayedHCUnit,
    next_hc: DelayedHCUnit,
    norm: torch.nn.Module,
    *,
    use_prepared: bool = False,
) -> bool:
    """Check the exact delayed-HC/RMSNorm contract without reading GPU data."""
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.dtype != torch.bfloat16
        or attn_out.ndim not in (2, 3)
        or attn_out.shape[-1] != 5120
        or not attn_out.is_contiguous()
        or not 0 < attn_out.numel() // 5120 <= 1 << 20
    ):
        return False
    device = attn_out.device
    if not _units_supported(previous, next_hc, norm, device, use_prepared):
        return False
    leading = tuple(attn_out.shape[:-1])
    contracts = (
        (residual, leading + (4, 5120), torch.bfloat16),
        (post, leading + (4, 1), torch.float32),
        (comb, leading + (4, 4), torch.float32),
        (previous.pre_mix_out, leading + (4,), torch.float32),
    )
    if any(
        not _is_dense_tensor(tensor, shape, dtype, device)
        for tensor, shape, dtype in contracts
    ):
        return False
    if torch.is_grad_enabled() and (
        attn_out.requires_grad
        or any(tensor.requires_grad for tensor, _, _ in contracts)
    ):
        return False
    return _stream_supports_mhc(device, use_prepared)


def try_fused_post_pre(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous: DelayedHCUnit,
    next_hc: DelayedHCUnit,
    norm: torch.nn.Module,
    *,
    original_tokens: int | None = None,
    reuse_residual: bool = False,
    output_fp8: bool = False,
    moe_inputs: MegaMoEInputs | None = None,
    use_prepared: bool = False,
    combine: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> MHCOutput | MHCQuantizedOutput | None:
    """Return ``(residual, normalized_input, post, comb)`` or use the old path.

    Fixed-layout serving layers may reuse their static check with ``use_prepared``.

    A successful call also publishes the FFN pre-mix for the next sublayer's
    delayed readout. Unsupported inputs return ``None``; backend execution
    failures propagate to the caller. ``reuse_residual`` overwrites the input
    residual after its corresponding tiles have been loaded, or the padded
    copy when padding is needed.
    SM90 low-token decode uses an out-of-place residual because the projection
    CTAs read the old residual while one projection tile writes the new one.
    ``original_tokens`` preserves the installed backend's pre-compaction
    split heuristic using at most 512 execution rows when padding is needed.
    Returned leading-row views retain that bounded padded storage.
    ``output_fp8`` appends an (FP8, packed GEMM scales) pair using native
    DeepGEMM quantization and TMA-aligned GEMM scales. Its minimum scale
    is 2**-22 (legacy/v3: 2**-33), so tiny inputs may underflow to zero
    and are not bitwise equivalent to the legacy quantization path.
    ``moe_inputs`` instead writes caller-owned MegaMoE activation/scales,
    retaining the four-element return. It excludes QKV scales and compact
    execution padding; the caller may consume it only after a non-None result.
    ``combine`` carries the deferred ``(routed, shared)`` MoE pair; ``attn_out``
    is then only its geometry proxy. The SM90 two-kernel backend rebuilds
    ``bf16(routed + shared)`` inline -- the exact value the standalone add+cast
    kernel had materialized -- and any other backend returns ``None`` so the
    caller materializes the pair first.
    """
    if moe_inputs is not None and output_fp8:
        raise ValueError("Choose GEMM scales or MegaMoE scales, not both")
    if moe_inputs is not None and original_tokens is not None:
        return None
    if combine is not None:
        if (
            original_tokens is not None
            or reuse_residual
            or output_fp8
            or moe_inputs is not None
        ):
            return None
        if _can_try_sm90_mega_mhc(attn_out):
            sm90_backend = _get_sm90_mega_mhc()
            if sm90_backend is not None:
                result = sm90_backend.try_vllm_two_kernel_post_pre(
                    attn_out,
                    residual,
                    post,
                    comb,
                    previous,
                    next_hc,
                    norm,
                    combine=combine,
                )
                if result is not None:
                    return result
        if _can_try_sm90_largebs_mega_mhc(attn_out):
            largebs_backend = _get_sm90_largebs_mega_mhc()
            if largebs_backend is not None:
                return largebs_backend.try_sm90_largebs_post_pre(
                    attn_out,
                    residual,
                    post,
                    comb,
                    previous,
                    next_hc,
                    norm,
                    combine=combine,
                )
        return None
    if (
        original_tokens is None
        and not reuse_residual
        and not output_fp8
        and moe_inputs is None
    ):
        if _can_try_sm90_mega_mhc(attn_out):
            sm90_backend = _get_sm90_mega_mhc()
            if sm90_backend is not None:
                result = sm90_backend.try_vllm_two_kernel_post_pre(
                    attn_out, residual, post, comb, previous, next_hc, norm
                )
                if result is not None:
                    return result
        if _can_try_sm90_largebs_mega_mhc(attn_out):
            largebs_backend = _get_sm90_largebs_mega_mhc()
            if largebs_backend is not None:
                result = largebs_backend.try_sm90_largebs_post_pre(
                    attn_out, residual, post, comb, previous, next_hc, norm
                )
                if result is not None:
                    return result
    if not is_supported(
        attn_out,
        residual,
        post,
        comb,
        previous,
        next_hc,
        norm,
        use_prepared=use_prepared,
    ):
        return None
    operation = _get_mega_mhc()
    leading = tuple(attn_out.shape[:-1])
    tokens = attn_out.numel() // 5120
    if moe_inputs is not None and (
        moe_inputs.x.shape != (tokens, 5120) or moe_inputs.x.device != attn_out.device
    ):
        raise ValueError("MegaMoE output buffers must match the MegaMHC token rows")
    execution_tokens = tokens
    if original_tokens is not None:
        execution_tokens = _compact_execution_tokens(
            original_tokens, tokens, next_hc, norm
        )
        if execution_tokens is None:
            return None
    # Packed prefill already has the kernel's row layout. Singleton dimensions
    # can have arbitrary strides even on contiguous tensors; preserve the
    # original view normalization when those strides can differ.
    flat_rows = len(leading) == 1 and tokens > 1 and post.stride(-1) == 1
    if not flat_rows:
        attn_out = attn_out.view(tokens, 5120)
        residual = residual.view(tokens, 4, 5120)
        post = post.view(tokens, 4, 1)
        comb = comb.view(tokens, 4, 4)
    previous_pre = previous.pre_mix_out.view(tokens, 4, 1)
    if execution_tokens != tokens:

        def pad(value):
            padded = value.new_zeros((execution_tokens, *value.shape[1:]))
            padded[:tokens].copy_(value)
            return padded

        attn_out, residual, post, comb, previous_pre = (
            pad(value) for value in (attn_out, residual, post, comb, previous_pre)
        )
    new_residual = residual if reuse_residual else torch.empty_like(residual)
    new_pre = torch.empty_like(previous_pre)
    new_post = torch.empty_like(post)
    new_comb = torch.empty_like(comb)
    normalized = torch.empty_like(attn_out)
    quantized = None
    if output_fp8:
        q = torch.empty_like(normalized, dtype=torch.float8_e4m3fn)
        # DeepGEMM requires column-major packed scales with 16-byte alignment.
        sf = torch.empty(
            (40, (execution_tokens + 3) // 4 * 4),
            device=attn_out.device,
            dtype=torch.int32,
        ).T[:execution_tokens]
        quantized = (q, sf)
    arguments = dict(
        x=attn_out,
        residual=residual,
        shifted_prev_mix=previous_pre,
        post_mix=post,
        comb_res_mix=comb,
        # Let the native TF32 tensor map convert the checkpoint FP32
        # weights, including its round-to-nearest-even halfway rule.
        fn=next_hc.fn,
        mix_scales=next_hc.scale,
        mix_bases=next_hc.base,
        hc_mult=4,
        hc_norm_eps=next_hc.norm_eps,
        hc_pre_eps=next_hc.hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=next_hc.hc_eps,
        num_sinkhorn_iters=next_hc.hc_sinkhorn_iters,
        rmsnorm_weight=norm.weight,
        rmsnorm_eps=norm.variance_epsilon,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=new_pre,
        new_post_mix=new_post,
        new_comb_res_mix=new_comb,
        y_bf16=normalized,
    )
    if quantized is not None:
        arguments.update(y_fp8=q, y_gemm_sf=sf)
    elif moe_inputs is not None:
        arguments.update(
            y_fp8=moe_inputs.x,
            y_routed_sf=moe_inputs.x_sf,
            y_shared_sf=moe_inputs.shared_l1_acts_sf,
            shared_sf_block_m=moe_inputs.block_m,
        )
    prefill_stream = _prefill_stream(attn_out.device, use_prepared)
    if prefill_stream is not None:
        operation(**arguments)
        if not prefill_stream.warmed:
            _WARMED_STREAMS.add(prefill_stream.key)
            prefill_stream.warmed = True
    else:
        with torch.cuda.device(attn_out.device):
            capturing = torch.cuda.is_current_stream_capturing()
            operation(**arguments)
            if not capturing:
                _WARMED_STREAMS.add(_stream_key(attn_out.device))
    if quantized is not None:
        quantized = (q[:tokens].view(*leading, 5120), sf[:tokens])
    if flat_rows and execution_tokens == tokens:
        next_hc.pre_mix_out = new_pre.view(tokens, 4)
        # Keep the caller's residual metadata independent when sharing storage.
        if reuse_residual:
            new_residual = new_residual.view_as(residual)
        result = (new_residual, normalized, new_post, new_comb)
    else:
        next_hc.pre_mix_out = new_pre[:tokens].view(*leading, 4)
        result = (
            new_residual[:tokens].view(*leading, 4, 5120),
            normalized[:tokens].view(*leading, 5120),
            new_post[:tokens].view(*leading, 4, 1),
            new_comb[:tokens].view(*leading, 4, 4),
        )

    return (*result, quantized) if output_fp8 else result


__all__ = [
    "can_preserve_compact_mhc",
    "is_supported",
    "prefill_mhc_stream",
    "prepare_mega_mhc",
    "try_fused_post_pre",
]
