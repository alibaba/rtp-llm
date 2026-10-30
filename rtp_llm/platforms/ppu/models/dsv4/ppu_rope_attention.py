"""FP8 attention with a PPU inverse-RoPE leaf before the grouped projection."""

from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.attention import AttentionFP8

from rtp_llm.platforms.ppu.kernels.ppu_inverse_rope import inverse_rope_inplace


class PpuRopeAttention(AttentionFP8):
    def _wo_a_from_bf16(self, o, freqs_cis, B, S):
        if self.wo_a is None:
            raise RuntimeError("PPU grouped output projection was not bound")
        output = o.view(B, S, self.n_heads, self.head_dim)
        if getattr(self.wo_a, "fuse_inverse_rope_quant", False):
            from ...kernels.ppu_decode_fused_quant import (
                inverse_rope_fp8_permute,
                supports_inverse_rope_fp8_permute,
            )
            from .ppu_wo_a import PpuWoAFp8Linear

            if (
                type(self.wo_a) is PpuWoAFp8Linear
                and self.wo_a.sglang_layout
                and self.wo_a.quantization == "v2_row"
                and supports_inverse_rope_fp8_permute(
                    output, freqs_cis, self.rope_head_dim, self.n_groups
                )
            ):
                payload, scale = inverse_rope_fp8_permute(
                    output,
                    freqs_cis,
                    self.rope_head_dim,
                    self.n_groups,
                    column_major_scales=True,
                )
                projected = self.wo_a.forward_permuted(payload, scale)
                return projected.view(B, S, self.n_groups, self.wo_a.rank)
        inverse_rope_inplace(output, freqs_cis, self.rope_head_dim)
        return self.wo_a(output.reshape(B, S, self.n_groups, -1))
