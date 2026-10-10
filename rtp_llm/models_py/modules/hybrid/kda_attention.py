"""Reusable packed, full-rank-gated KDA attention composition.

Models bind three projections, the output norm, and the recurrent implementations.
The packed input is [Q,K,V,output_gate,forget_latent,beta]. Equal Q/K/V head
geometry is required by this packed layout; other gate layouts can compose the
shared KdaPrefill/KdaDecode primitives directly.

Projection and communication behavior is preserved from the current model
implementation; no new projection backends or tuning options are added.
"""

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import (
    Fp8Activation,
    Group,
    _get_group,
    all_gather,
    reduce_scatter_bf16 as reduce_scatter,
)


class KdaAttention(nn.Module):
    def __init__(
        self,
        *,
        input_projection,
        forget_projection,
        output_projection,
        fa_width,
        norm,
        prefill,
        decode,
        fp8_output_norm=False,
        strided_fp8_forget=False,
        ag_local_fp8_overlap=False,
        ag_local_fp8_overlap_min_rows=4096,
    ):
        super().__init__()
        if (
            prefill.local_num_k_heads != prefill.local_num_v_heads
            or prefill.head_k_dim != prefill.head_v_dim
        ):
            raise ValueError("packed KDA requires equal Q/K/V head geometry")
        self.tp_size = prefill.parallelism_config.tp_size
        self.heads = prefill.local_num_v_heads
        self.dim = prefill.head_v_dim
        self.width = self.heads * self.dim
        self.fa_width = fa_width
        self.input, self.f_b, self.output = (
            input_projection,
            forget_projection,
            output_projection,
        )
        self.norm, self.prefill, self.decode = norm, prefill, decode
        self._fp8_output_norm = fp8_output_norm
        self._ag_local_fp8_overlap = ag_local_fp8_overlap
        self._ag_local_fp8_overlap_min_rows = ag_local_fp8_overlap_min_rows
        self._project_forget = (
            self._project_forget_strided_fp8
            if strided_fp8_forget
            else self._project_forget_contiguous
        )
        self._fp8_collective = None

    def _project_forget_contiguous(self, latent):
        return self.f_b(latent.contiguous())

    def _project_forget_strided_fp8(self, latent):
        from rtp_llm.models_py.triton_kernels.common.activation import (
            quantize_strided_group128_fp8,
        )

        values, scales = quantize_strided_group128_fp8(latent)
        return self.f_b.forward_quantized(values, scales)

    def forward(self, hidden, fmha, cache, attention_inputs, metadata):
        use_fused_ag = (
            self._fp8_collective is not None
            and self._fp8_collective.eligible_ag(hidden, attention_inputs, metadata)
        )
        use_overlap = (
            self._ag_local_fp8_overlap
            and self.tp_size > 1
            and hidden.shape[0] >= self._ag_local_fp8_overlap_min_rows
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and attention_inputs.is_prefill
            and not metadata.is_target_verify
            and not torch.cuda.is_current_stream_capturing()
        )
        if isinstance(hidden, Fp8Activation) and not use_fused_ag:
            raise RuntimeError("prequantized FP8 input requires the FP8 AG/GEMM path")
        if use_fused_ag:
            fused = self._fp8_collective.all_gather_gemm(hidden, self.input)
        elif use_overlap:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion import (
                all_gather_project_local_overlap,
            )

            fused = all_gather_project_local_overlap(
                hidden, self.input, _get_group(Group.TP)
            )
        else:
            full_hidden = (
                all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
            )
            fused = self.input(full_hidden)
        logical_width = 4 * self.width + self.fa_width + self.heads
        if fused.shape[-1] < logical_width:
            raise ValueError(
                f"KDA input projection returned {fused.shape[-1]} columns, "
                f"expected at least {logical_width}"
            )
        fused = fused[..., :logical_width]
        qkv, gate, fa, beta = fused.split(
            [3 * self.width, self.width, self.fa_width, self.heads], dim=-1
        )
        forget = self._project_forget(fa)
        kernel = (
            self.prefill
            if attention_inputs.is_prefill and not metadata.is_target_verify
            else self.decode
        )
        output = kernel(
            qkv if kernel is self.prefill else qkv.contiguous(),
            forget,
            beta,
            attention_inputs,
            cache,
            metadata,
        )
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None:
            # Paged KDA skips null-block rows, leaving their output unspecified.
            output = torch.where(valid_mask[:, None], output.reshape(-1, self.width), 0)
        if self._fp8_output_norm:
            from rtp_llm.models_py.triton_kernels.common.activation import (
                rmsnorm_sigmoid_gate_per_token_group_quant_fp8,
            )

            values, scales = rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                output.reshape(-1, self.heads, self.dim),
                gate.reshape(-1, self.heads, self.dim),
                self.norm.weight,
                self.norm.eps,
            )
        else:
            output = self.norm(
                output.reshape(-1, self.dim), gate.reshape(-1, self.dim)
            )
        use_fused_rs = (
            self._fp8_collective is not None
            and self._fp8_collective.eligible_rs(
                values.shape[0] if self._fp8_output_norm else 0,
                attention_inputs,
                metadata,
                self._fp8_output_norm,
            )
        )
        if use_fused_rs:
            return self._fp8_collective.gemm_reduce_scatter(
                values, scales, self.output
            )
        output = (
            self.output.forward_quantized(values, scales)
            if self._fp8_output_norm
            else self.output(output.reshape(-1, self.width))
        )
        return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
