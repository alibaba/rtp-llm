"""PPU shared expert; the loader's merged FP8 weights remain unchanged."""

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    SharedExpertExecutor,
    W13SharedExpert,
)


class PpuSharedExpert(W13SharedExpert):
    def __init__(self, *args, sglang_moe=False, platform_provider, **kwargs):
        from functools import partial
        from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear

        super().__init__(
            *args,
            linear_factory=partial(_v4_fp8_linear, platform_provider=platform_provider),
            **kwargs,
        )
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

    def __init__(self, stream_pool=None, *, start_before_routing=False):
        if start_before_routing and stream_pool is None:
            raise ValueError("Early shared execution requires a PPU overlap stream")
        self.start_before_routing = bool(start_before_routing)
        self._shared = None
        self._out = None
        self._stream_pool = stream_pool
        self._stream = None
        self.name = "ppu_overlap" if stream_pool is not None else "ppu_sequential"

    def prepare(self, shared_experts):
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear

        if self._out is not None:
            raise RuntimeError("PPU shared executor still has a pending output")
        if not isinstance(shared_experts, PpuSharedExpert) or not all(
            isinstance(getattr(shared_experts, name, None), PpuFp8Linear)
            for name in ("w13", "w2")
        ):
            raise TypeError("PPU shared executor requires PPU FP8 w13/w2 linears")
        if not shared_experts.preserve_output_dtype:
            raise ValueError("PPU shared executor requires the SGLang BF16 contract")
        if self._stream_pool is not None:
            self._stream = self._stream_pool.get(
                "shared_expert", shared_experts.w13.weight.device
            )
        self._shared = shared_experts

    def start(self, shared_experts, x):
        if self._shared is None or shared_experts is not self._shared:
            raise RuntimeError("PPU shared executor must use its prepared module")
        if self._out is not None:
            raise RuntimeError("PPU shared executor still has a pending output")
        if self._stream is None:
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = shared_experts(x)
            return
        if x.device != self._stream.device:
            raise ValueError("PPU shared input must use the prepared stream device")
        current = torch.cuda.current_stream(x.device)
        if not torch.cuda.is_current_stream_capturing():
            x.record_stream(self._stream)
        self._stream.wait_stream(current)
        try:
            with torch.cuda.stream(self._stream):
                with record_function_range("dsv4.moe.shared_expert"):
                    self._out = shared_experts(x)
        except Exception:
            current.wait_stream(self._stream)
            raise

    def finish(self):
        if self._out is None:
            raise RuntimeError("PPU shared executor has no pending output")
        if self._stream is not None:
            current = torch.cuda.current_stream(self._out.device)
            current.wait_stream(self._stream)
            if not torch.cuda.is_current_stream_capturing():
                self._out.record_stream(current)
        out, self._out = self._out, None
        return out
