"""Qwen4 TP8 FP8 projections without SM120's misaligned Split-K scales.

The pinned DeepGEMM normal GEMM can misalign packed scales after limiting its
Split-K factor. A single contiguous group uses the same FP8 ABI without Split-K.
"""

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_gemm_nt_contiguous,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)


class Qwen4Sm120Fp8Linear(CudaFp8DeepGEMMLinear):
    def __init__(self, original, indices):
        super().__init__(original.weight, original.weight_scales, bias=original.bias)
        self.cached_scales = original.cached_scales
        self.cached_scales_max_len = original.cached_scales_max_len
        self._indices = indices
        self._grouped_weight = self.weight.unsqueeze(0)
        self._grouped_scale = self.weight_scales.as_strided(
            (1, *self.weight_scales.shape),
            (
                self.weight_scales.stride(1) * self.weight_scales.shape[1],
                *self.weight_scales.stride(),
            ),
        )

    def forward_quantized(self, input_fp8, input_scales, out=None):
        m, _ = self._validate_input(input_fp8)
        output = self._prepare_output(input_fp8, m, out)
        indices = self._indices[:m]
        if m > self._indices.numel():
            # Oversized eager calls remain correct; admitted graph shapes use
            # the scheduler-sized allocation made before cache budgeting.
            indices = torch.zeros(m, dtype=torch.int32, device=input_fp8.device)
        m_grouped_fp8_gemm_nt_contiguous(
            (input_fp8, input_scales),
            (self._grouped_weight, self._grouped_scale),
            output,
            indices,
            disable_ue8m0_cast=False,
        )
        if self.bias is not None:
            output.add_(self.bias.to(output.dtype))
        return output


def install_tp8_fp8_linears(layer, config, pc, max_generate_batch_size):
    if not (
        getattr(config, "_qwen4_full_fp8_nvfp4", False)
        and pc.tp_size == pc.get_attn_tp_size() == pc.get_ffn_tp_size() == 8
        and pc.world_size == 8
        and pc.ep_size == pc.dp_size == pc.pp_size == 1
    ):
        return
    linears = [
        module
        for module in layer.modules()
        if type(module) is CudaFp8GEMMLinear and module._deepgemm_linear.scale_ue8m0
    ]
    if not linears or torch.cuda.get_device_capability(linears[0].weight.device) != (
        12,
        0,
    ):
        return
    max_tokens = max(
        config.moe_prefill_max_tokens_per_rank or config.max_seq_len,
        max_generate_batch_size
        * max(int(getattr(config, "gen_num_per_cycle", 0)) + 1, 1),
    )
    indices = torch.zeros(
        max_tokens, dtype=torch.int32, device=linears[0].weight.device
    )
    for module in linears:
        module._deepgemm_linear = Qwen4Sm120Fp8Linear(module._deepgemm_linear, indices)
