"""TP Prefill schedule that keeps shared/routed GEMMs apart."""

import torch

from rtp_llm.platforms.ppu.kernels.ppu_shared_swiglu import silu_mul_merged_bf16


class SharedPhases:
    def __init__(self, shared, x, stream):
        assert not getattr(shared, "fuse_activation_quant", False)
        self.shared = shared
        self.main = torch.cuda.current_stream(x.device)
        self.stream = stream
        self.output = torch.empty_like(x)
        self.output.record_stream(stream)
        x.record_stream(stream)
        stream.wait_stream(self.main)
        with torch.cuda.stream(stream):
            self.gate_up = shared.w13(x)
            self.up_done = stream.record_event()

    def before_up(self):
        self.main.wait_event(self.up_done)
        with torch.cuda.stream(self.stream):
            hidden = silu_mul_merged_bf16(self.gate_up, self.shared.swiglu_limit)
            self.quantized = self.shared.w2.quantize_input(hidden)

    def after_up(self):
        self.stream.wait_stream(self.main)
        with torch.cuda.stream(self.stream):
            self.shared.w2.forward_quantized(*self.quantized, out=self.output)
            self.done = self.stream.record_event()

    def before_down(self):
        self.main.wait_event(self.done)
