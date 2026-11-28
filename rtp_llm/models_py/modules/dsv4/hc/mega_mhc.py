"""DeepGEMM fused post+pre+norm (``deep_gemm.mega_mhc``) adapter for DSV4 mHC."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit

_HC_MULT = 4
_HC_MIX_COUNT = (2 + _HC_MULT) * _HC_MULT  # 24
_HC_POST_SCALE = 2.0
_RMSNORM_SCALE = 1.0
_BLOCK_M = 64
_BLOCK_K = 64
_MIN_SPLITS = 16
_MAX_SPLITS = 64
_HIDDEN_ALIGNMENT = 1024  # kDefaultNumSplits (16) * BLOCK_K (64)
_MAX_MEGA_MHC_TOKENS = 1 << 20


class _MegaMhcSpec(NamedTuple):
    dim: int
    hc_mult: int
    hc_sinkhorn_iters: int
    norm_eps: float
    hc_eps: float
    rmsnorm_eps: float


# Tracks CUDA streams (device_index, raw_cuda_stream_ptr) that have executed an
# eager mega_mhc launch, initializing DeepGEMM's per-stream split barriers
# before any CUDA graph capture runs on that stream.
_WARMED_STREAMS: set[Tuple[int, int]] = set()
_LAST_SPEC_BY_DEVICE: dict[int, _MegaMhcSpec] = {}


def _get_mega_mhc() -> Optional[Any]:
    try:
        import deep_gemm
    except ImportError:
        return None
    return getattr(deep_gemm, "mega_mhc", None)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _mega_num_splits(tokens: int, num_sms: int, hidden: int) -> int:
    """Mirror DeepGEMM ``get_num_splits(num_m_blocks, hidden, num_sms)``."""
    if tokens <= 0 or num_sms <= 0 or hidden <= 0:
        return _MIN_SPLITS
    num_m_blocks = _ceil_div(int(tokens), _BLOCK_M)
    num_k_blocks = _ceil_div(int(hidden), _BLOCK_K)
    max_num_splits = min(
        _MAX_SPLITS,
        max(_MIN_SPLITS, int(num_sms) // num_m_blocks),
    )
    return _ceil_div(num_k_blocks, _ceil_div(num_k_blocks, max_num_splits))


def representative_tokens(
    max_m: int,
    num_sms: int,
    *,
    hidden: int = 4096,
) -> Tuple[int, ...]:
    """Return the minimal token counts covering every ``num_splits`` bucket."""
    max_m = int(max_m)
    num_sms = int(num_sms)
    hidden = int(hidden)
    if max_m <= 0 or num_sms <= 0 or hidden <= 0:
        return ()

    max_m_blocks = min(_ceil_div(max_m, _BLOCK_M), (num_sms // _MIN_SPLITS) + 1)
    seen_splits: set[int] = set()
    reps: list[int] = []
    for m_block in range(1, max_m_blocks + 1):
        tokens = (m_block - 1) * _BLOCK_M + 1
        if tokens > max_m:
            break
        splits = _mega_num_splits(tokens, num_sms, hidden)
        if splits not in seen_splits:
            seen_splits.add(splits)
            reps.append(tokens)
    return tuple(reps)


def _normalize_cuda_device(device: Any) -> Optional[torch.device]:
    if device is None:
        return None
    try:
        dev = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if dev.type != "cuda":
        return None
    return dev


def _is_sm10x_device(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return int(major) == 10


def _same_cuda_device(tensor_device: torch.device, target: torch.device) -> bool:
    if tensor_device.type != "cuda" or target.type != "cuda":
        return False
    target_idx = (
        int(target.index)
        if target.index is not None
        else int(torch.cuda.current_device())
    )
    return int(tensor_device.index) == target_idx


def _stream_key(device: torch.device) -> Tuple[int, int]:
    index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    stream_ptr = int(torch.cuda.current_stream(index).cuda_stream)
    return (index, stream_ptr)


def can_use_on_current_stream(device: Any) -> bool:
    """Return True if ``deep_gemm.mega_mhc`` can launch safely on current stream."""
    dev = _normalize_cuda_device(device)
    if dev is None or _get_mega_mhc() is None or not _is_sm10x_device(dev):
        return False
    if torch.cuda.is_current_stream_capturing():
        return _stream_key(dev) in _WARMED_STREAMS
    return True


def _is_unpatched_previous_unit(previous: Any) -> bool:
    if type(previous) is not TileLangHCUnit:
        return False
    if getattr(previous, "_post_impl_prefill_fast", None) is not None:
        return False
    inst_dict = previous.__dict__
    if (
        "_post_impl" in inst_dict
        or "post" in inst_dict
        or "_post_impl_prefill_fast" in inst_dict
    ):
        return False
    cls = type(previous)
    return cls._post_impl is TileLangHCUnit._post_impl and cls.post is HCUnitBase.post


def _is_unpatched_next_unit(next_hc: Any) -> bool:
    if type(next_hc) is not TileLangHCUnit:
        return False
    if getattr(next_hc, "_pre_impl_prefill_fast", None) is not None:
        return False
    inst_dict = next_hc.__dict__
    if (
        "_pre_impl" in inst_dict
        or "pre" in inst_dict
        or "_pre_impl_prefill_fast" in inst_dict
    ):
        return False
    cls = type(next_hc)
    return cls._pre_impl is TileLangHCUnit._pre_impl and cls.pre is HCUnitBase.pre


def _get_rmsnorm_cls() -> Optional[type]:
    try:
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
    except ImportError:
        return None
    return RMSNorm


def _is_unpatched_rmsnorm(norm: Any) -> bool:
    rmsnorm_cls = _get_rmsnorm_cls()
    return (
        rmsnorm_cls is not None
        and type(norm) is rmsnorm_cls
        and "forward" not in norm.__dict__
        and type(norm).forward is rmsnorm_cls.forward
    )


def _check_static_tensor(
    tensor: Any,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and not tensor.requires_grad
        and _same_cuda_device(tensor.device, device)
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tensor.is_contiguous()
    )


def is_unit_pair_supported(
    previous: Any,
    next_hc: Any,
    norm: Any,
    device: Any,
) -> bool:
    """Return True if the ``(previous.post, next_hc.pre, norm)`` pair matches ``mega_mhc``."""
    dev = _normalize_cuda_device(device)
    if dev is None or not _is_sm10x_device(dev):
        return False
    if not (
        _is_unpatched_previous_unit(previous)
        and _is_unpatched_next_unit(next_hc)
        and _is_unpatched_rmsnorm(norm)
    ):
        return False
    if (
        int(previous.hc_mult) != _HC_MULT
        or int(next_hc.hc_mult) != _HC_MULT
        or int(previous.dim) != int(next_hc.dim)
    ):
        return False
    dim = int(next_hc.dim)
    if dim <= 0 or dim % _HIDDEN_ALIGNMENT != 0 or int(next_hc.hc_sinkhorn_iters) < 1:
        return False
    return (
        _check_static_tensor(
            next_hc.fn, (_HC_MIX_COUNT, _HC_MULT * dim), torch.float32, dev
        )
        and _check_static_tensor(next_hc.scale, (3,), torch.float32, dev)
        and _check_static_tensor(next_hc.base, (_HC_MIX_COUNT,), torch.float32, dev)
        and _check_static_tensor(norm.weight, (dim,), torch.bfloat16, dev)
    )


def is_supported(
    previous: Any,
    next_hc: Any,
    norm: Any,
    device: Any,
) -> bool:
    """Return True if ``mega_mhc`` is supported for the pair and current CUDA stream."""
    return is_unit_pair_supported(
        previous, next_hc, norm, device
    ) and can_use_on_current_stream(device)


def _check_runtime_tensor(
    tensor: Any,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and not tensor.requires_grad
        and _same_cuda_device(tensor.device, device)
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tensor.is_contiguous()
    )


def _validate_runtime_inputs(
    x_sub: Any,
    residual: Any,
    post: Any,
    comb: Any,
    dim: int,
    device: torch.device,
) -> Optional[Tuple[Tuple[int, ...], int]]:
    if not isinstance(residual, torch.Tensor) or residual.dim() not in (3, 4):
        return None
    leading = tuple(int(v) for v in residual.shape[:-2])
    if not _check_runtime_tensor(
        residual, leading + (_HC_MULT, dim), torch.bfloat16, device
    ):
        return None
    if not (
        _check_runtime_tensor(x_sub, leading + (dim,), torch.bfloat16, device)
        and _check_runtime_tensor(post, leading + (_HC_MULT, 1), torch.float32, device)
        and _check_runtime_tensor(
            comb, leading + (_HC_MULT, _HC_MULT), torch.float32, device
        )
    ):
        return None
    tokens = 1
    for extent in leading:
        tokens *= extent
    if tokens <= 0 or tokens > _MAX_MEGA_MHC_TOKENS:
        return None
    return leading, tokens


def try_fused_post_pre(
    x_sub: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    previous: Any,
    next_hc: Any,
    norm: Any,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fuse ``previous.post(x_sub, residual, post, comb)`` -> ``next_hc.pre`` -> ``norm``.

    Returns ``(new_residual, x_pre_normed, new_post, new_comb)`` on success, or
    ``None`` when any eligibility or stream gate is not satisfied.
    """
    mega_mhc = _get_mega_mhc()
    if mega_mhc is None:
        return None
    if not isinstance(residual, torch.Tensor):
        return None
    dev = _normalize_cuda_device(residual.device)
    if dev is None:
        return None
    if not is_supported(previous, next_hc, norm, dev):
        return None

    dim = int(next_hc.dim)
    validated = _validate_runtime_inputs(x_sub, residual, post, comb, dim, dev)
    if validated is None:
        return None
    leading, tokens = validated

    x_2d = x_sub.reshape(tokens, dim)
    res_3d = residual.reshape(tokens, _HC_MULT, dim)
    post_3d = post.reshape(tokens, _HC_MULT, 1)
    comb_3d = comb.reshape(tokens, _HC_MULT, _HC_MULT)

    # deep_gemm.mega_mhc forbids new_residual aliasing residual.
    new_residual_3d = torch.empty_like(res_3d)
    new_post_3d = torch.empty(
        (tokens, _HC_MULT, 1), dtype=torch.float32, device=res_3d.device
    )
    new_comb_3d = torch.empty(
        (tokens, _HC_MULT, _HC_MULT), dtype=torch.float32, device=res_3d.device
    )
    normalized_2d = torch.empty(
        (tokens, dim), dtype=torch.bfloat16, device=res_3d.device
    )

    layer = f"L{next_hc.layer_id:02d}" if int(next_hc.layer_id) >= 0 else "Lxx"
    name = next_hc.name or "unit"
    with (
        _profiler.record_function_range(f"dsv4.hc.{layer}.{name}.mega_mhc"),
        torch.inference_mode(),
    ):
        mega_mhc(
            x_2d,
            res_3d,
            None,
            post_3d,
            comb_3d,
            next_hc.fn,
            next_hc.scale,
            next_hc.base,
            _HC_MULT,
            float(next_hc.norm_eps),
            float(next_hc.hc_eps),
            _HC_POST_SCALE,
            float(next_hc.hc_eps),
            int(next_hc.hc_sinkhorn_iters),
            norm.weight,
            float(norm.variance_epsilon),
            _RMSNORM_SCALE,
            new_residual_3d,
            None,
            new_post_3d,
            new_comb_3d,
            y_bf16=normalized_2d,
        )

    if not torch.cuda.is_current_stream_capturing():
        _WARMED_STREAMS.add(_stream_key(dev))
    dev_idx = (
        int(dev.index) if dev.index is not None else int(torch.cuda.current_device())
    )
    _LAST_SPEC_BY_DEVICE[dev_idx] = _MegaMhcSpec(
        dim=dim,
        hc_mult=int(next_hc.hc_mult),
        hc_sinkhorn_iters=int(next_hc.hc_sinkhorn_iters),
        norm_eps=float(next_hc.norm_eps),
        hc_eps=float(next_hc.hc_eps),
        rmsnorm_eps=float(norm.variance_epsilon),
    )

    return (
        new_residual_3d.reshape(*leading, _HC_MULT, dim),
        normalized_2d.reshape(*leading, dim),
        new_post_3d.reshape(*leading, _HC_MULT, 1),
        new_comb_3d.reshape(*leading, _HC_MULT, _HC_MULT),
    )


def _launch_capture_warmup(
    mega_mhc_fn: Any,
    spec: _MegaMhcSpec,
    device: torch.device,
) -> None:
    """Launch a 1-token eager ``deep_gemm.mega_mhc`` call to init stream barriers."""
    dim = spec.dim
    hc_mult = spec.hc_mult
    x = torch.zeros((1, dim), dtype=torch.bfloat16, device=device)
    residual = torch.zeros((1, hc_mult, dim), dtype=torch.bfloat16, device=device)
    post_mix = torch.zeros((1, hc_mult, 1), dtype=torch.float32, device=device)
    comb_mix = torch.zeros((1, hc_mult, hc_mult), dtype=torch.float32, device=device)
    fn = torch.zeros((_HC_MIX_COUNT, hc_mult * dim), dtype=torch.float32, device=device)
    mix_scales = torch.ones((3,), dtype=torch.float32, device=device)
    mix_bases = torch.zeros((_HC_MIX_COUNT,), dtype=torch.float32, device=device)
    weight = torch.ones((dim,), dtype=torch.bfloat16, device=device)
    new_residual = torch.empty_like(residual)
    new_post = torch.empty_like(post_mix)
    new_comb = torch.empty_like(comb_mix)
    y_bf16 = torch.empty_like(x)
    mega_mhc_fn(
        x,
        residual,
        None,
        post_mix,
        comb_mix,
        fn,
        mix_scales,
        mix_bases,
        hc_mult,
        spec.norm_eps,
        spec.hc_eps,
        _HC_POST_SCALE,
        spec.hc_eps,
        spec.hc_sinkhorn_iters,
        weight,
        spec.rmsnorm_eps,
        _RMSNORM_SCALE,
        new_residual,
        None,
        new_post,
        new_comb,
        y_bf16=y_bf16,
    )


@torch.inference_mode()
def prepare_capture_stream(
    device: Optional[Any] = None,
    *,
    dim: Optional[int] = None,
    hc_mult: int = _HC_MULT,
    hc_sinkhorn_iters: int = 20,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    rmsnorm_eps: float = 1.0e-6,
) -> bool:
    """Initialize DeepGEMM's per-stream ``split_barriers`` on current capture stream.

    Called by C++ ``CudaGraphRunner`` immediately after switching to
    ``capture_stream_`` and before ``graphCaptureBegin``. Returns ``False`` when
    ``deep_gemm.mega_mhc`` is unavailable, the device is non-SM10x, the current
    stream is already capturing or already warmed, or no ``mega_mhc`` spec has
    executed on this device yet (avoiding side effects on unrelated models).
    """
    mega_mhc_fn = _get_mega_mhc()
    if mega_mhc_fn is None:
        return False
    if device is None:
        if not torch.cuda.is_available():
            return False
        dev: Optional[torch.device] = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = _normalize_cuda_device(device)
    if dev is None or not _is_sm10x_device(dev):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False

    dev_idx = (
        int(dev.index) if dev.index is not None else int(torch.cuda.current_device())
    )
    key = _stream_key(dev)
    if key in _WARMED_STREAMS:
        return False

    if dim is not None:
        resolved_dim = int(dim)
        if (
            resolved_dim <= 0
            or resolved_dim % _HIDDEN_ALIGNMENT != 0
            or int(hc_mult) != _HC_MULT
            or int(hc_sinkhorn_iters) < 1
        ):
            return False
        spec: Optional[_MegaMhcSpec] = _MegaMhcSpec(
            dim=resolved_dim,
            hc_mult=int(hc_mult),
            hc_sinkhorn_iters=int(hc_sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rmsnorm_eps=float(rmsnorm_eps),
        )
    else:
        spec = _LAST_SPEC_BY_DEVICE.get(dev_idx)
        if spec is None:
            return False

    with torch.cuda.device(dev_idx):
        _launch_capture_warmup(mega_mhc_fn, spec, torch.device("cuda", dev_idx))
        torch.cuda.current_stream(dev_idx).synchronize()
    _WARMED_STREAMS.add(key)
    _LAST_SPEC_BY_DEVICE.setdefault(dev_idx, spec)
    return True


__all__ = [
    "_LAST_SPEC_BY_DEVICE",
    "_WARMED_STREAMS",
    "_get_mega_mhc",
    "_mega_num_splits",
    "can_use_on_current_stream",
    "is_supported",
    "is_unit_pair_supported",
    "prepare_capture_stream",
    "representative_tokens",
    "try_fused_post_pre",
]
