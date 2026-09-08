"""PPU shared expert; the loader's merged FP8 weights remain unchanged."""

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert


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
