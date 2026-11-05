"""Shared-expert execution policies for fused MoE.

Open-source MoE stacks such as vLLM and SGLang commonly overlap shared experts
with routed MoE work on an auxiliary CUDA stream.  They do not rely on BF16
direct accumulation by default.  RTP keeps the existing FP32 accumulate contract
and only fuses the final add+cast when possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    strict_fused_moe_enabled,
)
from rtp_llm.models_py.utils.profiler import record_function_range

from .expert import require_silu_mul_split
from .quantized_linear import Mxfp8Linear, create_fp8_linear


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


class W13SharedExpert(nn.Module):
    """Merged shared experts with a combined gate/up projection.

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
        w13_w = expert_weights["w13_w"]
        w13_s = expert_weights["w13_s"]
        if w13_w.dim() != 2:
            raise RuntimeError(f"shared w13 weight must be 2D, got {w13_w.dim()}D")
        if w13_w.shape[0] != 2 * inter_dim or w13_w.shape[1] != dim:
            raise RuntimeError(
                "shared w13 weight shape mismatch: "
                f"got {tuple(w13_w.shape)}, expected {(2 * inter_dim, dim)}"
            )
        self.w13 = create_fp8_linear(w13_w, w13_s)
        self.w2 = create_fp8_linear(expert_weights["w2_w"], expert_weights["w2_s"])
        self.swiglu_limit = swiglu_limit

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if weights is None and self._can_run_mxfp8_fused_silu(x):
            return self._forward_mxfp8_fused_silu(x)
        dtype = x.dtype
        with record_function_range("moe.shared_expert.w13"):
            gate_up = self._apply_layer(self.w13, x).float()
            gate, up = gate_up.chunk(2, dim=-1)
        with record_function_range("moe.shared_expert.silu_mul"):
            hidden = require_silu_mul_split()(
                gate.contiguous(),
                up.contiguous(),
                clamp_limit=self.swiglu_limit,
            )
        if weights is not None:
            hidden = weights * hidden
        with record_function_range("moe.shared_expert.w2"):
            return self._apply_layer(self.w2, hidden.to(dtype))

    def _can_run_mxfp8_fused_silu(self, x: torch.Tensor) -> bool:
        """Fuse SwiGLU with the group-32 quantizer for MXFP8 linears.

        The fused kernel reads the merged BF16 gate_up halves in place, so the
        ``.float()`` cast, the two ``.contiguous()`` copies of the chunk halves
        and the ``hidden.to(dtype)`` cast disappear while keeping the FP32
        accumulate contract and the single FP32->BF16 rounding step.
        """
        return (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and isinstance(self.w13, Mxfp8Linear)
            and isinstance(self.w2, Mxfp8Linear)
        )

    def _forward_mxfp8_fused_silu(self, x: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_packed_g32,
        )

        with record_function_range("moe.shared_expert.w13"):
            gate_up = self.w13(x)
        with record_function_range("moe.shared_expert.silu_mul"):
            hidden_q, hidden_s = silu_mul_fp8_quant_packed_g32(
                gate_up, clamp_limit=self.swiglu_limit
            )
        with record_function_range("moe.shared_expert.w2"):
            return self.w2.forward_quantized(hidden_q, hidden_s)


class FusedSharedExpertFastPath:
    """Workspace-backed shared-expert path.

    It quantizes the BF16 input once, runs one merged w13 FP8 GEMM into a
    reusable BF16 gate_up buffer, fuses SwiGLU+FP8 quantization, then runs w2.
    The merged w13 weight is prepared outside the forward hot path.
    """

    _W13_WEIGHT_NAME = "_fused_moe_shared_w13_weight"
    _W13_SCALE_NAME = "_fused_moe_shared_w13_scale"

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
        if not all(hasattr(shared_experts, name) for name in ("w13", "w2")):
            return False
        # The workspace quantizes with block-128 scales; MXFP8 checkpoints need
        # group-32 scales and run W13SharedExpert.forward's fused path instead.
        return not any(
            isinstance(layer, Mxfp8Linear)
            for layer in (shared_experts.w13, shared_experts.w2)
        )

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

    def prepare(self, shared_experts: nn.Module) -> None:
        """Validate the loader-prepared merged w13; no runtime concatenation."""
        if not hasattr(shared_experts, "w13"):
            raise RuntimeError("shared expert requires loader-prepared w13")
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

    def _ensure_workspace(self, x: torch.Tensor) -> _SharedExpertWorkspace:
        T, D = x.shape
        if self.dim is None:
            self.dim = D
        if D != self.dim:
            raise RuntimeError(
                f"shared expert dim mismatch: got {D}, expected {self.dim}"
            )
        inter = self.inter_dim
        assert inter is not None
        capacity = max(T, self.max_tokens_per_rank or 0, 1)
        workspace = self._workspace
        if (
            workspace is not None
            and workspace.device == x.device
            and workspace.capacity >= capacity
            and workspace.hidden_fp8.size(1) == inter
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

        x_fp8 = torch.empty(
            (capacity, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        x_scale_storage = self._scale_storage((D // 128 + 3) // 4, capacity, x.device)
        gate_up_bf16 = torch.empty(
            (capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=x.device,
        )
        hidden_fp8 = torch.empty(
            (capacity, inter),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        hidden_scale_storage = self._scale_storage(
            (inter // 128 + 3) // 4,
            capacity,
            x.device,
        )
        out_bf16 = torch.empty(
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
                "fused shared expert requires CUDA bf16 2D input and FP8 "
                "loader-merged shared w13/w2 weights"
            )
        return self._run_prepared(shared_experts, x)

    def _run_prepared(self, shared_experts: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._prepared_shared_experts is not shared_experts:
            self.prepare(shared_experts)
        w13_parts = self._w13_parts
        w2_parts = self._w2_parts
        assert w13_parts is not None and w2_parts is not None
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
        from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
            quant_bf16_fp8_packed_ue8m0,
        )
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
            silu_mul_fp8_quant_packed,
        )

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
        with record_function_range("moe.shared_expert"):
            self._out = _run_shared_expert(shared_experts, x, self._fast_path)

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        out = self._out
        self._out = None
        return out


class MXFP8SharedExpertExecutor(SequentialSharedExpertExecutor):
    """Native group-32 GEMMs through MXFP8 shared-expert linears.

    This is a distinct supported quantization recipe: the shared expert owns
    block-32 weights, so the block-128 workspace path does not apply. The BF16
    return skips the standalone FP32 cast because the consumer
    (``fused_moe_epilogue``) loads and adds in FP32, matching the old
    ``.float()`` value bit-for-bit.
    """

    name = "mxfp8"

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        if x.shape[0] == 0:
            self._out = torch.empty_like(x, dtype=torch.float32)
            return
        with record_function_range("moe.shared_expert"):
            self._out = shared_experts(x)


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
            "MOE_STRICT_FUSED=1 forbids the generic Expert.forward "
            "shared-expert fallback"
        )
    return shared_experts(x).float()


def get_shared_expert_executor(
    max_tokens_per_rank: int | None = None,
    dim: int | None = None,
    inter_dim: int | None = None,
    swiglu_limit: float = 0.0,
    native_mxfp8: bool = False,
) -> SharedExpertExecutor:
    if native_mxfp8:
        return MXFP8SharedExpertExecutor()
    fast_path = FusedSharedExpertExecutor(
        max_tokens_per_rank=max_tokens_per_rank,
        dim=dim,
        inter_dim=inter_dim,
        swiglu_limit=swiglu_limit,
    )
    return SequentialSharedExpertExecutor(fast_path)


def combine_routed_and_shared(
    routed: torch.Tensor,
    shared: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        if out.shape != routed.shape:
            raise ValueError(
                f"combined MoE output shape {tuple(out.shape)} does not match "
                f"routed output shape {tuple(routed.shape)}"
            )
        if out.dtype != out_dtype or out.device != routed.device:
            raise ValueError(
                "combined MoE output must use the requested dtype and routed device"
            )

    try:
        from rtp_llm.models_py.triton_kernels.moe.shared_expert import (
            fused_moe_epilogue,
        )

        return fused_moe_epilogue(routed, shared, out_dtype, out=out)
    except Exception:
        if strict_fused_moe_enabled():
            raise
        combined = (routed.float() + shared.float()).to(out_dtype)
        if out is None:
            return combined
        out.copy_(combined)
        return out
