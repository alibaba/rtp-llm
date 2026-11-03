"""Decode-only MTP entrance; keep both GEMMs and their BF16 add unchanged."""

from ...kernels.ppu_mtp_input import mtp_input_quant, supports_mtp_input
from ...modules.linear.fp8_linear import PpuFp8Linear


class PpuMtpInputFusion:
    def __init__(self, enorm, hnorm, e_proj, h_proj):
        self.enorm, self.hnorm = enorm, hnorm
        self.e_proj, self.h_proj = e_proj, h_proj

    def __call__(self, embedding, hidden, positions):
        if not all(
            type(layer) is PpuFp8Linear
            and layer.quantization == "v2_column"
            and layer.k == layer.n == 4096
            and layer.weight.device == hidden.device
            for layer in (self.e_proj, self.h_proj)
        ):
            return None
        args = (
            embedding,
            hidden,
            positions,
            self.enorm.weight.data,
            self.hnorm.weight.data,
            self.enorm.variance_epsilon,
            self.hnorm.variance_epsilon,
        )
        if not supports_mtp_input(*args):
            return None
        e_pair, h_pair = mtp_input_quant(*args)
        return self.h_proj.forward_quantized(*h_pair).view_as(
            hidden
        ) + self.e_proj.forward_quantized(*e_pair).unsqueeze(1)
