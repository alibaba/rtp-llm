"""FP8 attention with a PPU inverse-RoPE leaf before the grouped projection."""

from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

from ...kernels.ppu_inverse_rope import inverse_rope_inplace


class PpuRopeAttention(AttentionFP8):
    def _wo_a_from_bf16(self, o, freqs_cis, B, S):
        if self.wo_a is None:
            raise RuntimeError("PPU grouped output projection was not bound")
        output = o.view(B, S, self.n_heads, self.head_dim)
        inverse_rope_inplace(output, freqs_cis, self.rope_head_dim)
        return self.wo_a(output.reshape(B, S, self.n_groups, -1))
