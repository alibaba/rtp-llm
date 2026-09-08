"""PPU shared expert; the loader's merged FP8 weights remain unchanged."""

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    SharedExpertExecutor,
    W13SharedExpert,
)


class PpuSharedExpert(W13SharedExpert):
    def __init__(self, *args, sglang_moe=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preserve_output_dtype = bool(sglang_moe)

    def forward(self, x, weights=None):
        if not self.preserve_output_dtype or weights is not None:
            return super().forward(x, weights)
        from rtp_llm.platforms.ppu.kernels.ppu_shared_swiglu import silu_mul_merged_bf16

        with record_function_range("dsv4.shared_expert.w13"):
            gate_up = self._apply_layer(self.w13, x)
        with record_function_range("dsv4.shared_expert.silu_mul"):
            hidden = silu_mul_merged_bf16(gate_up, self.swiglu_limit)
        with record_function_range("dsv4.shared_expert.w2"):
            return self._apply_layer(self.w2, hidden)


class PpuSharedExpertExecutor(SharedExpertExecutor):
    """Execute the selected PPU FP8 path without CUDA packed-scale dispatch.

    PPU linears own row-major FP32 scales. The CUDA fused executor accepts a
    different scale ABI, so this executor validates its module at preparation
    and preserves the SGLang BF16 shared result for the FP32 MoE epilogue.
    Errors propagate directly; there is no generic expert fallback.
    """

    name = "ppu_sequential"

    def __init__(self):
        self._shared = None
        self._out = None

    def prepare(self, shared_experts):
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear

        if not isinstance(shared_experts, PpuSharedExpert) or not all(
            isinstance(getattr(shared_experts, name, None), PpuFp8Linear)
            for name in ("w13", "w2")
        ):
            raise TypeError("PPU shared executor requires PPU FP8 w13/w2 linears")
        if not shared_experts.preserve_output_dtype:
            raise ValueError("PPU shared executor requires the SGLang BF16 contract")
        self._shared = shared_experts

    def start(self, shared_experts, x):
        if self._shared is None or shared_experts is not self._shared:
            raise RuntimeError("PPU shared executor must use its prepared module")
        with record_function_range("dsv4.moe.shared_expert"):
            self._out = shared_experts(x)

    def finish(self):
        if self._out is None:
            raise RuntimeError("PPU shared executor has no pending output")
        out, self._out = self._out, None
        return out
