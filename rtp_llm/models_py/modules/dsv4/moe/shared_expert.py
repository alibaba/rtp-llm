"""Shared expert execution policies for DSV4 MoE.

Open-source MoE stacks such as vLLM and SGLang commonly overlap shared experts
with routed MoE work on an auxiliary CUDA stream.  They do not rely on BF16
direct accumulation by default.  RTP keeps the existing FP32 accumulate contract
and only fuses the final add+cast when possible.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

from .warmup_sync import cuda_graph_warmup_forward_enabled


_SHARED_EXPERT_WORKSPACE_CACHE: dict[tuple, dict[str, torch.Tensor | int | torch.device]] = {}
_SHARED_EXPERT_STREAM_CACHE: dict[int, torch.cuda.Stream] = {}


def _mode() -> str:
    return os.environ.get("DSV4_SHARED_EXPERT_MODE", "sequential").strip().lower()


def strict_fused_moe_enabled() -> bool:
    return os.environ.get("DSV4_MOE_STRICT_FUSED", "1") != "0"


def _normalize_cuda_device(device: torch.device) -> torch.device | None:
    if not torch.cuda.is_available() or device.type != "cuda":
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.device("cuda", device_index)


def _ensure_shared_expert_stream(device: torch.device) -> torch.cuda.Stream | None:
    device = _normalize_cuda_device(device)
    if device is None:
        return None
    device_index = device.index
    assert device_index is not None
    stream = _SHARED_EXPERT_STREAM_CACHE.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SHARED_EXPERT_STREAM_CACHE[device_index] = stream
    return stream


def _get_shared_expert_stream(
    device: torch.device,
    *,
    allow_create: bool,
) -> torch.cuda.Stream:
    device = _normalize_cuda_device(device)
    if device is None:
        raise RuntimeError(f"shared expert overlap requires CUDA device, got {device}")
    device_index = device.index
    assert device_index is not None
    stream = _SHARED_EXPERT_STREAM_CACHE.get(device_index)
    if stream is not None:
        return stream
    if not allow_create:
        raise RuntimeError(
            "shared expert overlap stream was not created before CUDA graph "
            f"capture for device cuda:{device_index}"
        )
    stream = torch.cuda.Stream(device=device)
    _SHARED_EXPERT_STREAM_CACHE[device_index] = stream
    return stream


def _find_module_cuda_device(module: nn.Module) -> torch.device | None:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.is_cuda:
            return tensor.device

    for submodule in module.modules():
        for attr in ("weight", "weight_scales", "bias"):
            tensor = getattr(submodule, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                return tensor.device
    return None


class W13SharedExpert(nn.Module):
    """DSV4 shared expert with loader-merged gate/up projection.

    The checkpoint stores shared w1 and w3 separately, but the loader merges
    them into ``w13`` so inference never keeps duplicate split linears.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        expert_weights: dict[str, torch.Tensor],
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear

        w13_w = expert_weights["w13_w"]
        w13_s = expert_weights["w13_s"]
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_w.shape[0] != 2 * inter_dim or w13_w.shape[1] != dim:
            raise RuntimeError(
                "shared w13 weight shape mismatch: "
                f"got {tuple(w13_w.shape)}, expected {(2 * inter_dim, dim)}"
            )
        self.w13 = _v4_fp8_linear(w13_w, w13_s)
        self.w2 = _v4_fp8_linear(expert_weights["w2_w"], expert_weights["w2_s"])
        self.swiglu_limit = swiglu_limit

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        with record_function_range("dsv4.shared_expert.w13"):
            gate_up = self._apply_layer(self.w13, x).float()
            gate, up = gate_up.chunk(2, dim=-1)
        with record_function_range("dsv4.shared_expert.silu_mul"):
            from .expert import require_silu_mul_split

            hidden = require_silu_mul_split()(
                gate.contiguous(),
                up.contiguous(),
                clamp_limit=self.swiglu_limit,
            )
        if weights is not None:
            hidden = weights * hidden
        with record_function_range("dsv4.shared_expert.w2"):
            return self._apply_layer(self.w2, hidden.to(dtype))


class FusedSharedExpertFastPath:
    """Workspace-backed DSV4 shared expert path.

    It quantizes the BF16 input once, runs one merged w13 FP8 GEMM into a
    reusable BF16 gate_up buffer, fuses SwiGLU+FP8 quantization, then runs w2.
    The merged w13 weight is prepared outside the forward hot path.
    """

    _W13_WEIGHT_NAME = "_dsv4_shared_w13_weight"
    _W13_SCALE_NAME = "_dsv4_shared_w13_scale"

    def __init__(
        self,
        max_tokens_per_rank: int | None = None,
        dim: int | None = None,
        inter_dim: int | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dim = dim
        self.inter_dim = inter_dim
        self.swiglu_limit = swiglu_limit
        self._device: torch.device | None = None
        self._capacity = 0
        self._x_fp8: torch.Tensor | None = None
        self._x_scale_storage: torch.Tensor | None = None
        self._gate_up_bf16: torch.Tensor | None = None
        self._hidden_fp8: torch.Tensor | None = None
        self._hidden_scale_storage: torch.Tensor | None = None
        self._out_bf16: torch.Tensor | None = None

    @staticmethod
    def _linear_parts(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(linear, "weight", None)
        scale = getattr(linear, "weight_scales", None)
        if weight is None or scale is None:
            raise RuntimeError("shared expert FP8 linear does not expose weight/scale")
        return weight, scale

    @staticmethod
    def can_run(shared_experts: nn.Module, x: torch.Tensor) -> bool:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2):
            return False
        return all(hasattr(shared_experts, name) for name in ("w13", "w2"))

    @classmethod
    def has_merged_w13(cls, shared_experts: nn.Module) -> bool:
        return hasattr(shared_experts, "w13") or (
            hasattr(shared_experts, cls._W13_WEIGHT_NAME)
            and hasattr(shared_experts, cls._W13_SCALE_NAME)
        )

    @classmethod
    def _set_shared_buffer(
        cls,
        shared_experts: nn.Module,
        name: str,
        value: torch.Tensor,
    ) -> None:
        if name in shared_experts._buffers:
            shared_experts._buffers[name] = value
        else:
            shared_experts.register_buffer(name, value, persistent=False)

    @staticmethod
    def _merge_weight_scales(w1_s: torch.Tensor, w3_s: torch.Tensor) -> torch.Tensor:
        if w1_s.dtype != torch.int32:
            return torch.cat((w1_s, w3_s), dim=0).contiguous()

        rows = w1_s.size(0) + w3_s.size(0)
        cols = w1_s.size(1)
        aligned_rows = FusedSharedExpertFastPath._tma_aligned_rows(
            rows,
            w1_s.element_size(),
        )
        storage = torch.empty(
            (cols, aligned_rows),
            dtype=torch.int32,
            device=w1_s.device,
        )
        merged = storage.as_strided((rows, cols), (1, aligned_rows))
        merged[: w1_s.size(0)].copy_(w1_s)
        merged[w1_s.size(0) :].copy_(w3_s)
        return merged

    def prepare(self, shared_experts: nn.Module) -> None:
        """Validate the loader-prepared merged w13; no runtime concatenation."""
        if not hasattr(shared_experts, "w13"):
            raise RuntimeError("DSV4 shared expert requires loader-prepared w13")
        w13_w, w13_s = self._linear_parts(shared_experts.w13)
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_s.dim() != 2:
            raise RuntimeError(f"shared w13 scale must be 2D, got {w13_s.dim()}D")
        if w13_w.shape[0] % 2 != 0:
            raise RuntimeError(f"shared w13 rows must be even, got {w13_w.shape[0]}")

    @staticmethod
    def _tma_aligned_rows(rows: int, element_size: int) -> int:
        import deep_gemm

        return deep_gemm.get_tma_aligned_size(rows, element_size)

    @staticmethod
    def _scale_storage(
        num_packed_groups: int,
        capacity: int,
        device: torch.device,
    ) -> torch.Tensor:
        aligned_capacity = FusedSharedExpertFastPath._tma_aligned_rows(
            max(capacity, 1),
            torch.empty((), dtype=torch.int32).element_size(),
        )
        return torch.empty(
            (num_packed_groups, aligned_capacity),
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _scale_view(storage: torch.Tensor, tokens: int) -> torch.Tensor:
        aligned_tokens = FusedSharedExpertFastPath._tma_aligned_rows(
            max(tokens, 1),
            storage.element_size(),
        )
        return storage.as_strided(
            (tokens, storage.size(0)),
            (1, aligned_tokens),
        )

    def _ensure_workspace(self, x: torch.Tensor) -> None:
        T, D = x.shape
        if self.dim is None:
            self.dim = D
        if D != self.dim:
            raise RuntimeError(f"shared expert dim mismatch: got {D}, expected {self.dim}")
        if self.inter_dim is None:
            w13, _ = self._linear_parts(self._shared.w13)  # type: ignore[attr-defined]
            self.inter_dim = w13.shape[0] // 2
        inter = self.inter_dim
        assert inter is not None
        capacity = max(T, self.max_tokens_per_rank or 0, 1)
        if (
            self._device == x.device
            and self._capacity >= capacity
            and self._x_fp8 is not None
        ):
            return
        if D % 128 != 0 or inter % 128 != 0:
            raise RuntimeError(
                f"shared expert fused path requires D/inter divisible by 128, got {D}/{inter}"
            )
        key = (x.device, D, inter)
        cached = _SHARED_EXPERT_WORKSPACE_CACHE.get(key)
        if cached is not None and int(cached["capacity"]) >= capacity:
            self._device = x.device
            self._capacity = int(cached["capacity"])
            self._x_fp8 = cached["x_fp8"]  # type: ignore[assignment]
            self._x_scale_storage = cached["x_scale_storage"]  # type: ignore[assignment]
            self._gate_up_bf16 = cached["gate_up_bf16"]  # type: ignore[assignment]
            self._hidden_fp8 = cached["hidden_fp8"]  # type: ignore[assignment]
            self._hidden_scale_storage = cached["hidden_scale_storage"]  # type: ignore[assignment]
            self._out_bf16 = cached["out_bf16"]  # type: ignore[assignment]
            return
        self._device = x.device
        self._capacity = capacity
        self._x_fp8 = torch.empty(
            (capacity, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        self._x_scale_storage = self._scale_storage((D // 128 + 3) // 4, capacity, x.device)
        self._gate_up_bf16 = torch.empty(
            (capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=x.device,
        )
        self._hidden_fp8 = torch.empty(
            (capacity, inter),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        self._hidden_scale_storage = self._scale_storage(
            (inter // 128 + 3) // 4,
            capacity,
            x.device,
        )
        self._out_bf16 = torch.empty(
            (capacity, D),
            dtype=torch.bfloat16,
            device=x.device,
        )
        _SHARED_EXPERT_WORKSPACE_CACHE[key] = {
            "capacity": capacity,
            "device": x.device,
            "x_fp8": self._x_fp8,
            "x_scale_storage": self._x_scale_storage,
            "gate_up_bf16": self._gate_up_bf16,
            "hidden_fp8": self._hidden_fp8,
            "hidden_scale_storage": self._hidden_scale_storage,
            "out_bf16": self._out_bf16,
        }

    def run(self, shared_experts: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.can_run(shared_experts, x):
            raise RuntimeError(
                "DSV4 fused shared expert requires CUDA bf16 2D input and FP8 "
                "loader-merged shared w13/w2 weights"
            )
        self._shared = shared_experts
        self._ensure_workspace(x)
        T = x.size(0)
        assert self._x_fp8 is not None
        assert self._x_scale_storage is not None
        assert self._gate_up_bf16 is not None
        assert self._hidden_fp8 is not None
        assert self._hidden_scale_storage is not None
        assert self._out_bf16 is not None
        if not self.has_merged_w13(shared_experts):
            raise RuntimeError("DSV4 fused shared expert requires loader-prepared w13")

        x_fp8 = self._x_fp8[:T]
        x_scale = self._scale_view(self._x_scale_storage, T)
        gate_up = self._gate_up_bf16[:T]
        hidden_fp8 = self._hidden_fp8[:T]
        hidden_scale = self._scale_view(self._hidden_scale_storage, T)
        out = self._out_bf16[:T]
        if T == 0:
            return out

        from ._shared_expert_triton import quant_bf16_fp8_packed_ue8m0
        from ._silu_mul_fp8_quant_triton import silu_mul_fp8_quant_packed
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        quant_bf16_fp8_packed_ue8m0(x, x_fp8, x_scale, group_size=128, eps=1.0e-4)
        w13 = self._linear_parts(shared_experts.w13)
        w2 = self._linear_parts(shared_experts.w2)
        fp8_gemm_nt((x_fp8, x_scale), w13, gate_up, disable_ue8m0_cast=False)
        silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=self.swiglu_limit,
            group_size=128,
            output_q=hidden_fp8,
            output_scale=hidden_scale,
        )
        fp8_gemm_nt((hidden_fp8, hidden_scale), w2, out, disable_ue8m0_cast=False)
        return out


class FusedSharedExpertExecutor(FusedSharedExpertFastPath):
    """Backward-facing name for the fused shared expert workspace runner."""


class SharedExpertExecutor(ABC):
    name: str

    def prepare(self, shared_experts: nn.Module) -> None:
        return None

    @abstractmethod
    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> torch.Tensor:
        raise NotImplementedError


class SequentialSharedExpertExecutor(SharedExpertExecutor):
    name = "sequential"

    def __init__(
        self,
        fast_path: FusedSharedExpertFastPath | None = None,
    ) -> None:
        self._out: torch.Tensor | None = None
        self._fast_path = fast_path

    def prepare(self, shared_experts: nn.Module) -> None:
        if self._fast_path is not None:
            self._fast_path.prepare(shared_experts)

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        with record_function_range("dsv4.moe.shared_expert"):
            self._out = _run_shared_expert(shared_experts, x, self._fast_path)

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        out = self._out
        self._out = None
        return out


class OverlapSharedExpertExecutor(SharedExpertExecutor):
    """Run shared expert on an aux stream while routed MoE runs on current stream."""

    name = "overlap"

    def __init__(
        self,
        fast_path: FusedSharedExpertFastPath | None = None,
    ) -> None:
        self._active_stream: torch.cuda.Stream | None = None
        self._out: torch.Tensor | None = None
        self._fast_path = fast_path

    def prepare(self, shared_experts: nn.Module) -> None:
        if self._fast_path is not None:
            self._fast_path.prepare(shared_experts)
        device = _find_module_cuda_device(shared_experts)
        if device is not None:
            _ensure_shared_expert_stream(device)

    def _can_overlap(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and torch.cuda.is_available()):
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if cuda_graph_warmup_forward_enabled():
            return False
        if os.environ.get("MOEDBG", "0") != "0":
            return False
        threshold = int(
            os.environ.get("DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096")
        )
        return x.shape[0] <= threshold

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        if not self._can_overlap(x):
            self._active_stream = None
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)
            return
        capturing = torch.cuda.is_current_stream_capturing()
        stream = _get_shared_expert_stream(x.device, allow_create=not capturing)
        if not capturing:
            x.record_stream(stream)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)
        self._active_stream = stream

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        if self._active_stream is not None:
            torch.cuda.current_stream(self._out.device).wait_stream(self._active_stream)
        out = self._out
        self._out = None
        self._active_stream = None
        return out


def _run_shared_expert(
    shared_experts: nn.Module,
    x: torch.Tensor,
    fast_path: FusedSharedExpertFastPath | None,
) -> torch.Tensor:
    if fast_path is not None and fast_path.can_run(shared_experts, x):
        try:
            return fast_path.run(shared_experts, x)
        except Exception:
            if strict_fused_moe_enabled():
                raise
    if strict_fused_moe_enabled():
        raise RuntimeError(
            "DSV4_MOE_STRICT_FUSED=1 forbids generic Expert.forward shared path"
        )
    return shared_experts(x).float()


def get_shared_expert_executor(
    max_tokens_per_rank: int | None = None,
    dim: int | None = None,
    inter_dim: int | None = None,
    swiglu_limit: float = 0.0,
) -> SharedExpertExecutor:
    mode = _mode()
    fast_path = FusedSharedExpertExecutor(
        max_tokens_per_rank=max_tokens_per_rank,
        dim=dim,
        inter_dim=inter_dim,
        swiglu_limit=swiglu_limit,
    )
    if mode == "sequential":
        return SequentialSharedExpertExecutor(fast_path)
    if mode in ("auto", "overlap"):
        return OverlapSharedExpertExecutor(fast_path)
    raise ValueError(
        f"invalid DSV4_SHARED_EXPERT_MODE={mode!r}; expected auto|sequential|overlap"
    )


def combine_routed_and_shared(
    routed: torch.Tensor,
    shared: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if os.environ.get("DSV4_SHARED_EXPERT_BF16_ADD", "0") == "1":
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids DSV4_SHARED_EXPERT_BF16_ADD=1"
            )
        return (routed.to(out_dtype) + shared.to(out_dtype)).to(out_dtype)

    try:
        from ._shared_expert_triton import fused_moe_epilogue

        return fused_moe_epilogue(routed, shared, out_dtype, out=out)
    except Exception:
        if strict_fused_moe_enabled():
            raise
        return (routed.float() + shared.float()).to(out_dtype)
