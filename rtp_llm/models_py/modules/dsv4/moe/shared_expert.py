"""Shared expert execution policies for DSV4 MoE.

Open-source MoE stacks such as vLLM and SGLang commonly overlap shared experts
with routed MoE work on an auxiliary CUDA stream.  They do not rely on BF16
direct accumulation by default.  RTP keeps the existing FP32 accumulate contract
and only fuses the final add+cast when possible.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from rtp_llm.model_loader.weight_memory_saver import pausable_empty
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

from .warmup_sync import cuda_graph_warmup_forward_enabled

@dataclass(frozen=True)
class _SharedExpertWorkspaceViews:
    x_fp8: torch.Tensor
    x_scale: torch.Tensor
    gate_up_bf16: torch.Tensor
    hidden_fp8: torch.Tensor
    hidden_scale: torch.Tensor
    out_bf16: torch.Tensor


@dataclass
class _SharedExpertWorkspace:
    capacity: int
    device: torch.device
    x_fp8: torch.Tensor
    x_scale_storage: torch.Tensor
    gate_up_bf16: torch.Tensor
    hidden_fp8: torch.Tensor
    hidden_scale_storage: torch.Tensor
    out_bf16: torch.Tensor
    views: dict[int, _SharedExpertWorkspaceViews] = field(default_factory=dict)


_SHARED_EXPERT_WORKSPACE_CACHE: dict[tuple, _SharedExpertWorkspace] = {}
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
    device_index: int = device.index  # type: ignore[assignment]
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
    device_index: int = device.index  # type: ignore[assignment]
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
    for tensor in list(module.parameters(recurse=True)) + list(
        module.buffers(recurse=True)
    ):
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
        self._workspace: _SharedExpertWorkspace | None = None
        self._prepared_shared_experts: nn.Module | None = None
        self._w13_parts: tuple[torch.Tensor, torch.Tensor] | None = None
        self._w2_parts: tuple[torch.Tensor, torch.Tensor] | None = None

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
        w2_w, w2_s = self._linear_parts(shared_experts.w2)
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_s.dim() != 2:
            raise RuntimeError(f"shared w13 scale must be 2D, got {w13_s.dim()}D")
        if w13_w.shape[0] % 2 != 0:
            raise RuntimeError(f"shared w13 rows must be even, got {w13_w.shape[0]}")
        inferred_inter_dim = int(w13_w.shape[0]) // 2
        self.inter_dim = inferred_inter_dim
        self._prepared_shared_experts = shared_experts
        self._w13_parts = (w13_w, w13_s)
        self._w2_parts = (w2_w, w2_s)

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
        return pausable_empty(
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

    def _ensure_workspace(self, x: torch.Tensor) -> _SharedExpertWorkspace:
        T, D = x.shape
        if self.dim is None:
            self.dim = D
        if D != self.dim:
            raise RuntimeError(f"shared expert dim mismatch: got {D}, expected {self.dim}")
        inter: int = self.inter_dim  # type: ignore[assignment]
        capacity = max(T, self.max_tokens_per_rank or 0, 1)
        workspace = self._workspace
        if (
            workspace is not None
            and workspace.device == x.device
            and workspace.capacity >= capacity
        ):
            return workspace
        if D % 128 != 0 or inter % 128 != 0:
            raise RuntimeError(
                f"shared expert fused path requires D/inter divisible by 128, got {D}/{inter}"
            )
        key = (x.device, D, inter)
        workspace = _SHARED_EXPERT_WORKSPACE_CACHE.get(key)
        if workspace is not None and workspace.capacity >= capacity:
            self._workspace = workspace
            return workspace

        x_fp8 = pausable_empty(
            (capacity, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        x_scale_storage = self._scale_storage(
            (D // 128 + 3) // 4, capacity, x.device
        )
        gate_up_bf16 = pausable_empty(
            (capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=x.device,
        )
        hidden_fp8 = pausable_empty(
            (capacity, inter),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        hidden_scale_storage = self._scale_storage(
            (inter // 128 + 3) // 4,
            capacity,
            x.device,
        )
        out_bf16 = pausable_empty(
            (capacity, D),
            dtype=torch.bfloat16,
            device=x.device,
        )
        # Replace the cache entry as one versioned unit. Executors or captured
        # graphs that still reference the prior workspace keep its buffers
        # alive; each executor adopts this version when it next needs capacity.
        workspace = _SharedExpertWorkspace(
            capacity=capacity,
            device=x.device,
            x_fp8=x_fp8,
            x_scale_storage=x_scale_storage,
            gate_up_bf16=gate_up_bf16,
            hidden_fp8=hidden_fp8,
            hidden_scale_storage=hidden_scale_storage,
            out_bf16=out_bf16,
        )
        _SHARED_EXPERT_WORKSPACE_CACHE[key] = workspace
        self._workspace = workspace
        return workspace

    def _workspace_views(
        self,
        workspace: _SharedExpertWorkspace,
        tokens: int,
    ) -> _SharedExpertWorkspaceViews:
        cached = workspace.views.get(tokens)
        if cached is not None:
            return cached
        if tokens < 0 or tokens > workspace.capacity:
            raise RuntimeError(
                f"shared expert workspace tokens={tokens} exceed capacity={workspace.capacity}"
            )
        # All MoE layers in one forward use the same token count. Keep only
        # that shape so variable-length traffic cannot accumulate view objects
        # for every historical T over the process lifetime.
        workspace.views.clear()
        cached = _SharedExpertWorkspaceViews(
            x_fp8=workspace.x_fp8[:tokens],
            x_scale=self._scale_view(workspace.x_scale_storage, tokens),
            gate_up_bf16=workspace.gate_up_bf16[:tokens],
            hidden_fp8=workspace.hidden_fp8[:tokens],
            hidden_scale=self._scale_view(workspace.hidden_scale_storage, tokens),
            out_bf16=workspace.out_bf16[:tokens],
        )
        workspace.views[tokens] = cached
        return cached

    def run(self, shared_experts: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.can_run(shared_experts, x):
            raise RuntimeError(
                "DSV4 fused shared expert requires CUDA bf16 2D input and FP8 "
                "loader-merged shared w13/w2 weights"
            )
        return self._run_prepared(shared_experts, x)

    def _run_prepared(
        self, shared_experts: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        if self._prepared_shared_experts is not shared_experts:
            self.prepare(shared_experts)
        w13_parts: tuple[torch.Tensor, torch.Tensor] = (
            self._w13_parts  # type: ignore[assignment]
        )
        w2_parts: tuple[torch.Tensor, torch.Tensor] = (
            self._w2_parts  # type: ignore[assignment]
        )
        workspace = self._ensure_workspace(x)
        T = x.size(0)

        views = self._workspace_views(workspace, T)
        x_fp8 = views.x_fp8
        x_scale = views.x_scale
        gate_up = views.gate_up_bf16
        hidden_fp8 = views.hidden_fp8
        hidden_scale = views.hidden_scale
        out = views.out_bf16
        if T == 0:
            return out

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        from ._shared_expert_triton import quant_bf16_fp8_packed_ue8m0
        from ._silu_mul_fp8_quant_triton import silu_mul_fp8_quant_packed

        quant_bf16_fp8_packed_ue8m0(x, x_fp8, x_scale, group_size=128, eps=1.0e-4)
        fp8_gemm_nt(
            (x_fp8, x_scale),
            w13_parts,
            gate_up,
            disable_ue8m0_cast=False,
        )
        silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=self.swiglu_limit,
            group_size=128,
            output_q=hidden_fp8,
            output_scale=hidden_scale,
        )
        fp8_gemm_nt(
            (hidden_fp8, hidden_scale),
            w2_parts,
            out,
            disable_ue8m0_cast=False,
        )
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
        out: torch.Tensor = self._out  # type: ignore[assignment]
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
        self._input: torch.Tensor | None = None
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
        threshold = int(
            os.environ.get("DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096")
        )
        if x.shape[0] > threshold:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if cuda_graph_warmup_forward_enabled():
            return False
        if os.environ.get("MOEDBG", "0") != "0":
            return False
        return True

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        if not self._can_overlap(x):
            self._active_stream = None
            self._input = None
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)
            return
        stream = _get_shared_expert_stream(x.device, allow_create=True)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        self._active_stream = stream
        self._input = x
        with torch.cuda.stream(stream):
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = _run_shared_expert(shared_experts, x, self._fast_path)

    def finish(self) -> torch.Tensor:
        out: torch.Tensor = self._out  # type: ignore[assignment]
        if self._active_stream is not None:
            torch.cuda.current_stream(out.device).wait_stream(self._active_stream)
        self._input = None
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
            return fast_path._run_prepared(shared_experts, x)
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
