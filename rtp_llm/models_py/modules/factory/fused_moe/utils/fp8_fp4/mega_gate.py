"""DeepGEMM fused BF16 MoE gate (``bf16_mega_gate``) wrapper and graph helpers.

Replaces the unfused BF16 GEMM + sigmoid/sqrtsoftplus + bias + top-K + weight
normalization chain (and the MegaMoE Triton gate+pack epilogue when fused gate
projection is supported) with DeepGEMM's SM100 persistent/warp-specialized
``deep_gemm.bf16_mega_gate`` kernel.

This module intentionally maintains strict separation from ``Gate``:

* Unsupported hardware (non-SM100, CPU, ROCm, PPU), unsupported parameters
  (``softmax``, ``hidden % 256 != 0``, ``experts > 512``, ``experts % 4 != 0``,
  ``topk > 32``, ``MOE_GATE_FP32=1``, ``MOE_GATE_FUSED=0``, CED row projection,
  or unwarmed CUDA-graph capture streams) cleanly return ``False`` / ``None`` so
  callers fall back to Triton/eager routing.
* ``get_score_barriers`` in DeepGEMM lazily allocates a per-stream byte tensor
  the first time ``bf16_mega_gate`` executes with ``T > 0`` on a given stream.
  Calling inside CUDA-graph capture before that stream has been warmed triggers a
  C++ assertion; ``_WARMED_STREAMS`` and ``prepare_capture_stream_barrier``
  guarantee stream initialization before ``graphCaptureBegin``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

_SUPPORTED_SCORE_FUNCS = ("sigmoid", "sqrtsoftplus", "identity")
_MAX_NUM_TOKENS = 1 << 20
_MAX_EXPERTS = 512
_MAX_TOPK = 32
_HIDDEN_MULTIPLE = 256
_EXPERT_MULTIPLE = 4
_ALL_TRUE_MASK_MIN_CAP = 2048

# (device_index, cuda_stream_ptr) pairs that have already executed T > 0 on
# bf16_mega_gate outside graph capture, initializing DeepGEMM's per-stream
# score_barriers buffer.
_WARMED_STREAMS: set[tuple[int, int]] = set()
_LAST_SPEC_BY_DEVICE: dict[int, "MegaGateSpec"] = {}
_WARMED_JIT_KEYS: set[tuple[Any, ...]] = set()
_ALL_TRUE_MASK_CACHE: dict[int, torch.Tensor] = {}
_CUBLAS_INITIALIZED_DEVICES: set[int] = set()


@dataclass(frozen=True)
class MegaGateSpec:
    hidden: int
    num_experts: int
    topk: int
    scoring_func: str
    has_bias: bool
    is_hash: bool
    has_bias_vl: bool


def _get_bf16_mega_gate():
    try:
        import deep_gemm
    except Exception:
        return None
    fn = getattr(deep_gemm, "bf16_mega_gate", None)
    return fn if callable(fn) else None


def _get_bf16_mega_gate_config():
    try:
        import deep_gemm
    except Exception:
        return None
    fn = getattr(deep_gemm, "get_bf16_mega_gate_config", None)
    return fn if callable(fn) else None


def is_bf16_mega_gate_available() -> bool:
    return _get_bf16_mega_gate() is not None


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _is_on_cuda(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda)


def _is_sm100_device(device: torch.device) -> bool:
    if not isinstance(device, torch.device) or device.type != "cuda":
        return False
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(_device_index(device))
        return int(major) == 10
    except Exception:
        return False


def _stream_key(device: torch.device) -> tuple[int, int]:
    dev_idx = _device_index(device)
    stream_ptr = int(torch.cuda.current_stream(dev_idx).cuda_stream)
    return (dev_idx, stream_ptr)


def _ensure_cublas_initialized(device: torch.device) -> None:
    """Warm PyTorch's per-device cuBLAS handle once outside CUDA graph capture.

    When ``bf16_mega_gate`` replaces ``F.linear`` during eager warmup, ATen's
    thread-local cuBLAS handle may never have been created; a subsequent
    fallback to ``F.linear`` inside an unwarmed ``torch.cuda.graph`` capture
    would otherwise fail with ``CUBLAS_STATUS_NOT_INITIALIZED``.
    """
    if (
        not isinstance(device, torch.device)
        or device.type != "cuda"
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    dev_idx = _device_index(device)
    if dev_idx in _CUBLAS_INITIALIZED_DEVICES:
        return
    if not torch.cuda.is_available():
        return
    a = torch.zeros((1, 16), dtype=torch.bfloat16, device=device)
    b = torch.zeros((16, 16), dtype=torch.bfloat16, device=device)
    torch.nn.functional.linear(a, b)
    _CUBLAS_INITIALIZED_DEVICES.add(dev_idx)


def _as_fp32_contiguous(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


def _get_all_true_mask(device: torch.device, tokens: int) -> torch.Tensor:
    dev_idx = _device_index(device)
    cached = _ALL_TRUE_MASK_CACHE.get(dev_idx)
    if cached is not None and cached.device == device and cached.numel() >= tokens:
        return cached[:tokens]
    if torch.cuda.is_current_stream_capturing():
        return torch.ones((tokens,), dtype=torch.bool, device=device)
    alloc_size = max(tokens, _ALL_TRUE_MASK_MIN_CAP)
    cached = torch.ones((alloc_size,), dtype=torch.bool, device=device)
    _ALL_TRUE_MASK_CACHE[dev_idx] = cached
    return cached[:tokens]


def is_visual_routing_active(gate: Any, input_ids: Optional[torch.Tensor]) -> bool:
    """Return whether VL per-image bias routing applies to this gate call."""
    if getattr(gate, "bias_vl", None) is None:
        return False
    if not bool(getattr(gate, "_has_visual_tokens", False)):
        return False
    if input_ids is None:
        return False
    return bool(getattr(gate, "hash", False) or getattr(gate, "bias", None) is not None)


def _record_spec_for_gate(
    gate: Any,
    x: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    *,
    has_bias: bool,
    is_hash: bool,
    has_bias_vl: bool,
) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    dev_idx = _device_index(x.device)
    _LAST_SPEC_BY_DEVICE[dev_idx] = MegaGateSpec(
        hidden=int(x.shape[1]),
        num_experts=int(gate.weight.shape[0]),
        topk=int(gate.topk),
        scoring_func=str(gate.score_func),
        has_bias=has_bias,
        is_hash=is_hash,
        has_bias_vl=has_bias_vl,
    )


def spec_from_gate(gate: Any) -> Optional[MegaGateSpec]:
    """Extract a canonical ``MegaGateSpec`` from a Gate module if eligible."""
    weight = getattr(gate, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None
    if weight.dtype != torch.bfloat16:
        return None
    score_func = getattr(gate, "score_func", None)
    if score_func not in _SUPPORTED_SCORE_FUNCS:
        return None
    if getattr(gate, "_ced_row_projection", None) is not None:
        return None
    num_experts, hidden = int(weight.shape[0]), int(weight.shape[1])
    topk = int(getattr(gate, "topk", 0) or 0)
    if (
        hidden <= 0
        or hidden % _HIDDEN_MULTIPLE != 0
        or num_experts <= 0
        or num_experts > _MAX_EXPERTS
        or num_experts % _EXPERT_MULTIPLE != 0
        or topk <= 0
        or topk > num_experts
        or topk > _MAX_TOPK
    ):
        return None
    is_hash = bool(getattr(gate, "hash", False))
    bias = getattr(gate, "bias", None)
    bias_vl = getattr(gate, "bias_vl", None)
    has_bias = bias is not None
    # Note: hash layers may carry a checkpoint-loaded tid2eid while visual routing
    # can also activate bias_vl dynamically when image tokens appear.
    has_bias_vl = bias_vl is not None and (is_hash or has_bias)
    return MegaGateSpec(
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
        scoring_func=str(score_func),
        has_bias=has_bias,
        is_hash=is_hash,
        has_bias_vl=has_bias_vl,
    )


def can_use_bf16_mega_gate(
    gate: Any,
    x: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
) -> bool:
    """Return whether ``gate(x, input_ids)`` can execute via ``bf16_mega_gate``."""
    if os.environ.get("MOE_GATE_FP32", "0") == "1":
        return False
    if os.environ.get("MOE_GATE_FUSED", "1") == "0":
        return False

    weight = getattr(gate, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if getattr(gate, "_ced_row_projection", None) is not None:
        return False
    score_func = getattr(gate, "score_func", None)
    if score_func not in _SUPPORTED_SCORE_FUNCS:
        return False

    if (
        not isinstance(x, torch.Tensor)
        or not _is_on_cuda(x)
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
    ):
        return False
    if not _is_sm100_device(x.device):
        return False
    if not is_bf16_mega_gate_available():
        return False

    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if tokens <= 0 or tokens > _MAX_NUM_TOKENS:
        return False
    if hidden <= 0 or hidden % _HIDDEN_MULTIPLE != 0:
        return False

    if (
        not _is_on_cuda(weight)
        or weight.device != x.device
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
    ):
        return False
    num_experts, weight_hidden = int(weight.shape[0]), int(weight.shape[1])
    topk = int(getattr(gate, "topk", 0) or 0)
    if (
        weight_hidden != hidden
        or num_experts <= 0
        or num_experts > _MAX_EXPERTS
        or num_experts % _EXPERT_MULTIPLE != 0
        or topk <= 0
        or topk > num_experts
        or topk > _MAX_TOPK
    ):
        return False

    if torch.cuda.is_current_stream_capturing():
        if _stream_key(x.device) not in _WARMED_STREAMS:
            return False

    bias = getattr(gate, "bias", None)
    if bias is not None:
        if (
            not isinstance(bias, torch.Tensor)
            or not _is_on_cuda(bias)
            or bias.device != x.device
            or bias.dim() != 1
            or int(bias.shape[0]) != num_experts
            or bias.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            return False

    is_hash = bool(getattr(gate, "hash", False))
    if is_hash:
        tid2eid = getattr(gate, "tid2eid", None)
        if (
            input_ids is None
            or not isinstance(input_ids, torch.Tensor)
            or not _is_on_cuda(input_ids)
            or input_ids.device != x.device
            or input_ids.numel() != tokens
        ):
            return False
        if (
            not isinstance(tid2eid, torch.Tensor)
            or not _is_on_cuda(tid2eid)
            or tid2eid.device != x.device
            or tid2eid.dim() != 2
            or int(tid2eid.shape[1]) != topk
        ):
            return False

    if is_visual_routing_active(gate, input_ids):
        bias_vl = getattr(gate, "bias_vl", None)
        vocab_size = int(getattr(gate, "vocab_size", 0) or 0)
        if (
            not isinstance(bias_vl, torch.Tensor)
            or not _is_on_cuda(bias_vl)
            or bias_vl.device != x.device
            or bias_vl.dim() != 1
            or int(bias_vl.shape[0]) != num_experts
            or bias_vl.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or vocab_size <= 0
            or input_ids is None
            or not _is_on_cuda(input_ids)
            or input_ids.device != x.device
            or input_ids.numel() != tokens
        ):
            return False

    return True


def _validate_out_pair(
    out: tuple[torch.Tensor, torch.Tensor],
    *,
    tokens: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError("out must be a (topk_idx, topk_weights) tuple")
    out_idx, out_weights = out
    for name, tensor, expected_dtype in (
        ("out_idx", out_idx, torch.int64),
        ("out_weights", out_weights, torch.float32),
    ):
        if not _is_on_cuda(tensor) or tensor.device != device:
            raise ValueError(f"{name} must be on CUDA device {device}")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must be {expected_dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != (tokens, topk):
            raise ValueError(
                f"{name} shape mismatch: expected {(tokens, topk)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return out_idx, out_weights


def run_bf16_mega_gate(
    gate: Any,
    x: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    *,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    chunk_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute ``deep_gemm.bf16_mega_gate`` and return ``(topk_idx, topk_weights)``.

    Note: ``deep_gemm.bf16_mega_gate`` returns ``(topk_idx, topk_weights)``, whereas
    ``Gate.forward()`` returns ``(topk_weights, topk_idx)``.  ``run_bf16_mega_gate``
    matches DeepGEMM's ``(topk_idx, topk_weights)`` return and ``out`` convention;
    use :func:`try_bf16_mega_gate` when ``Gate.forward()`` order is expected.
    """
    bf16_mega_gate = _get_bf16_mega_gate()
    if bf16_mega_gate is None:
        raise RuntimeError("deep_gemm.bf16_mega_gate is unavailable")

    tokens = int(x.shape[0])
    topk = int(gate.topk)
    weight = gate.weight
    scoring_func = str(gate.score_func)
    routed_scaling_factor = float(getattr(gate, "route_scale", 1.0))

    bias_tensor = _as_fp32_contiguous(getattr(gate, "bias", None))
    is_hash = bool(getattr(gate, "hash", False))
    flat_ids = input_ids.reshape(-1) if input_ids is not None else None
    visual_active = is_visual_routing_active(gate, flat_ids)

    if visual_active:
        assert flat_ids is not None
        image_bias_tensor = _as_fp32_contiguous(gate.bias_vl)
        image_token_mask = (flat_ids >= int(gate.vocab_size)).contiguous()
    else:
        image_bias_tensor = None
        image_token_mask = None

    if is_hash:
        assert flat_ids is not None
        if image_token_mask is not None:
            fix_routing_mask = (~image_token_mask).contiguous()
            safe_ids = torch.where(image_token_mask, 0, flat_ids)
        else:
            fix_routing_mask = _get_all_true_mask(x.device, tokens)
            safe_ids = flat_ids
        unmapped_topk_idx = gate.tid2eid[safe_ids].to(dtype=torch.int64).contiguous()
    else:
        fix_routing_mask = None
        unmapped_topk_idx = None

    effective_chunk = int(chunk_rows) if chunk_rows > 0 else 0
    out_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    if out is not None:
        out_pair = _validate_out_pair(out, tokens=tokens, topk=topk, device=x.device)
    elif effective_chunk > 0 and tokens > effective_chunk:
        out_pair = (
            torch.empty((tokens, topk), dtype=torch.int64, device=x.device),
            torch.empty((tokens, topk), dtype=torch.float32, device=x.device),
        )

    def _call_range(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_out = (
            None
            if out_pair is None
            else (out_pair[0][start:end], out_pair[1][start:end])
        )
        return bf16_mega_gate(
            x[start:end],
            weight,
            topk,
            False,  # use_shared_as_routed
            0,  # num_shared_experts
            routed_scaling_factor,
            0,  # ep_rank (unused when to_physical_map is None)
            scoring_func=scoring_func,
            mask=None,
            bias=bias_tensor,
            image_bias=image_bias_tensor,
            image_token_mask=None if image_token_mask is None else image_token_mask[start:end],
            fix_routing_mask=None if fix_routing_mask is None else fix_routing_mask[start:end],
            to_physical_map=None,
            logical_count=None,
            unmapped_topk_idx=None if unmapped_topk_idx is None else unmapped_topk_idx[start:end],
            force_random=None,
            out=chunk_out,
        )

    if effective_chunk > 0 and tokens > effective_chunk:
        for start in range(0, tokens, effective_chunk):
            _call_range(start, min(start + effective_chunk, tokens))
        assert out_pair is not None
        topk_idx, topk_weights = out_pair
    else:
        topk_idx, topk_weights = _call_range(0, tokens)

    if tokens > 0:
        _WARMED_STREAMS.add(_stream_key(x.device))
        _ensure_cublas_initialized(x.device)
        _record_spec_for_gate(
            gate,
            x,
            flat_ids,
            has_bias=bias_tensor is not None,
            is_hash=is_hash,
            has_bias_vl=image_bias_tensor is not None,
        )

    return topk_idx, topk_weights


def try_bf16_mega_gate(
    gate: Any,
    x: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Run ``bf16_mega_gate`` if supported and return ``(topk_weights, topk_idx)``."""
    if not can_use_bf16_mega_gate(gate, x, input_ids):
        return None
    chunk_rows = int(getattr(gate, "_prefill_gate_chunk_rows", 0) or 0)
    topk_idx, topk_weights = run_bf16_mega_gate(
        gate, x, input_ids, chunk_rows=chunk_rows
    )
    return topk_weights, topk_idx


def _probe_token_candidates(max_tokens: int) -> list[int]:
    """Build candidate token counts preserving small dense steps + large steps."""
    cap = max(int(max_tokens), 0)
    if cap <= 0:
        return []
    dense_limit = min(cap, 2048)
    candidates = list(range(1, dense_limit + 1))
    if cap > dense_limit:
        val = 2048
        while val < cap:
            step = max(128, val // 8)
            val = min(val + step, cap)
            candidates.append(val)
    return candidates


def deduplicate_mega_gate_token_counts(
    *,
    hidden: int,
    num_experts: int,
    topk: int,
    max_tokens: int,
    config_fn=None,
) -> list[int]:
    """Return one token count per distinct DeepGEMM ``bf16_mega_gate`` heuristic bucket."""
    cap = max(int(max_tokens), 0)
    if cap <= 0:
        return []
    if config_fn is None:
        config_fn = _get_bf16_mega_gate_config()
    if config_fn is None:
        return [1] if cap == 1 else [1, cap]

    seen_configs: set[tuple[tuple[str, Any], ...]] = set()
    reps: list[int] = []
    for num_tokens in _probe_token_candidates(cap):
        cfg = config_fn(num_tokens, hidden, num_experts, topk)
        signature = tuple(sorted(cfg.items()))
        if signature not in seen_configs:
            seen_configs.add(signature)
            reps.append(num_tokens)
    if reps and reps[0] != 1:
        reps.insert(0, 1)
    return reps


def _spec_from_item(item: Any) -> Optional[MegaGateSpec]:
    if isinstance(item, MegaGateSpec):
        return item
    return spec_from_gate(item)


def collect_mega_gate_specs_from_model(model: Any) -> list[MegaGateSpec]:
    """Collect unique ``MegaGateSpec`` entries across all MoE Gate submodules."""
    if model is None:
        return []
    modules_iter = (
        model.modules()
        if hasattr(model, "modules") and callable(model.modules)
        else ()
    )
    seen: set[MegaGateSpec] = set()
    specs: list[MegaGateSpec] = []
    for module in modules_iter:
        if (
            hasattr(module, "weight")
            and hasattr(module, "score_func")
            and hasattr(module, "topk")
            and isinstance(getattr(module, "weight", None), torch.Tensor)
        ):
            spec = spec_from_gate(module)
            if spec is not None and spec not in seen:
                seen.add(spec)
                specs.append(spec)
    return specs


def _launch_spec(
    bf16_mega_gate,
    spec: MegaGateSpec,
    tokens: int,
    device: torch.device,
) -> None:
    x = torch.zeros((tokens, spec.hidden), dtype=torch.bfloat16, device=device)
    weight = torch.zeros(
        (spec.num_experts, spec.hidden), dtype=torch.bfloat16, device=device
    )
    bias = (
        torch.zeros((spec.num_experts,), dtype=torch.float32, device=device)
        if spec.has_bias
        else None
    )
    image_bias = (
        torch.zeros((spec.num_experts,), dtype=torch.float32, device=device)
        if spec.has_bias_vl
        else None
    )
    image_token_mask = (
        torch.zeros((tokens,), dtype=torch.bool, device=device)
        if spec.has_bias_vl
        else None
    )
    fix_routing_mask = (
        _get_all_true_mask(device, tokens) if spec.is_hash else None
    )
    unmapped_topk_idx = (
        torch.zeros((tokens, spec.topk), dtype=torch.int64, device=device)
        if spec.is_hash
        else None
    )
    bf16_mega_gate(
        x,
        weight,
        spec.topk,
        False,
        0,
        1.0,
        0,
        scoring_func=spec.scoring_func,
        mask=None,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        to_physical_map=None,
        logical_count=None,
        unmapped_topk_idx=unmapped_topk_idx,
        force_random=None,
        out=None,
    )


@torch.inference_mode()
def warmup_bf16_mega_gate_specs(
    specs: Iterable[Any],
    *,
    max_tokens: int,
    device: torch.device,
) -> int:
    """JIT-compile unique ``bf16_mega_gate`` layer specs across heuristic buckets."""
    if os.environ.get("MOE_GATE_FP32", "0") == "1":
        return 0
    if os.environ.get("MOE_GATE_FUSED", "1") == "0":
        return 0
    device = torch.device(device)
    if not _is_sm100_device(device):
        return 0
    bf16_mega_gate = _get_bf16_mega_gate()
    if bf16_mega_gate is None:
        return 0
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "bf16_mega_gate JIT warmup must not run inside CUDA graph capture"
        )

    cap = max(int(max_tokens), 0)
    if cap <= 0:
        return 0

    unique_specs: list[MegaGateSpec] = []
    seen_specs: set[MegaGateSpec] = set()
    for item in specs:
        spec = _spec_from_item(item)
        if spec is not None and spec not in seen_specs:
            seen_specs.add(spec)
            unique_specs.append(spec)
    if not unique_specs:
        return 0

    config_fn = _get_bf16_mega_gate_config()
    dev_idx = _device_index(device)
    _ensure_cublas_initialized(device)
    compiled = 0
    for spec in unique_specs:
        token_reps = deduplicate_mega_gate_token_counts(
            hidden=spec.hidden,
            num_experts=spec.num_experts,
            topk=spec.topk,
            max_tokens=cap,
            config_fn=config_fn,
        )
        for tokens in token_reps:
            jit_key = (dev_idx, spec, tokens)
            if jit_key in _WARMED_JIT_KEYS:
                continue
            _launch_spec(bf16_mega_gate, spec, tokens, device)
            _WARMED_JIT_KEYS.add(jit_key)
            compiled += 1
        _WARMED_STREAMS.add(_stream_key(device))
        _LAST_SPEC_BY_DEVICE[dev_idx] = spec
    if compiled > 0:
        logging.info(
            "[bf16_mega_gate] JIT warmup compiled %d variant(s) across %d unique spec(s), max_tokens=%d",
            compiled,
            len(unique_specs),
            cap,
        )
    return compiled


@torch.inference_mode()
def prepare_capture_stream_barrier(device: Optional[torch.device] = None) -> bool:
    """Initialize DeepGEMM's per-stream ``score_barriers`` on the current capture stream.

    Called by C++ ``CudaGraphRunner`` immediately after switching to
    ``capture_stream_`` and before ``graphCaptureBegin``.  Returns ``False`` when
    no DSV4 ``bf16_mega_gate`` spec has ever executed on this device or when the
    current stream is already warmed, avoiding any side effects on unrelated models.
    """
    if not is_bf16_mega_gate_available():
        return False
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if not _is_sm100_device(dev):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    dev_idx = _device_index(dev)
    key = _stream_key(dev)
    if key in _WARMED_STREAMS:
        return False
    spec = _LAST_SPEC_BY_DEVICE.get(dev_idx)
    if spec is None:
        return False
    bf16_mega_gate = _get_bf16_mega_gate()
    if bf16_mega_gate is None:
        return False
    with torch.cuda.device(dev_idx):
        _ensure_cublas_initialized(dev)
        _launch_spec(bf16_mega_gate, spec, 1, dev)
        torch.cuda.current_stream(dev_idx).synchronize()
    _WARMED_STREAMS.add(key)
    return True


__all__ = [
    "MegaGateSpec",
    "can_use_bf16_mega_gate",
    "collect_mega_gate_specs_from_model",
    "deduplicate_mega_gate_token_counts",
    "is_bf16_mega_gate_available",
    "is_visual_routing_active",
    "prepare_capture_stream_barrier",
    "run_bf16_mega_gate",
    "spec_from_gate",
    "try_bf16_mega_gate",
    "warmup_bf16_mega_gate_specs",
]
