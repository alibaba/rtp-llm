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
from rtp_llm.models_py.utils.module_profiler import profile_scope

_AG_LOCAL_FP8_OVERLAP_MIN_ROWS = 4096


class KdaAttention(nn.Module):
    supports_fp8_activation = True

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
        strided_fp8_prefill_forget=False,
        ag_local_fp8_overlap=False,
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
        self._project_forget = (
            self._project_forget_strided_fp8
            if strided_fp8_forget
            else self._project_forget_contiguous
        )
        self._project_prefill_forget = (
            self._project_forget_strided_fp8
            if strided_fp8_prefill_forget
            else self._project_forget
        )
        self._fp8_collective = None
        self._bf16_collective = None

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
            and hidden.shape[0] >= _AG_LOCAL_FP8_OVERLAP_MIN_ROWS
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and attention_inputs.is_prefill
            and not metadata.is_target_verify
            and not torch.cuda.is_current_stream_capturing()
        )
        if isinstance(hidden, Fp8Activation) and not use_fused_ag:
            raise RuntimeError("prequantized FP8 input requires the FP8 AG/GEMM path")
        if use_fused_ag:
            with profile_scope("RTP::attention.fp8_ag_gemm"):
                fused = self._fp8_collective.all_gather_gemm(hidden, self.input)
        elif self._bf16_collective is not None and self._bf16_collective.eligible(
            hidden.shape[0], attention_inputs, metadata,
        ):
            with profile_scope("RTP::attention.bf16_ag_gemm"):
                fused = self._bf16_collective.all_gather_gemm(hidden, self.input)
        elif use_overlap:
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion import (
                all_gather_project_local_overlap,
            )

            with profile_scope("RTP::attention.kda.ag_local_fp8_overlap"):
                fused = all_gather_project_local_overlap(
                    hidden, self.input, _get_group(Group.TP)
                )
        else:
            with profile_scope("RTP::attention.input_all_gather"):
                full_hidden = (
                    all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
                )
            with profile_scope("RTP::attention.kda.input_proj"):
                fused = self.input(full_hidden)
        logical_width = 4 * self.width + self.fa_width + self.heads
        if fused.shape[-1] < logical_width:
            raise ValueError(
                f"KDA input projection returned {fused.shape[-1]} columns, "
                f"expected at least {logical_width}"
            )
        fused = fused[..., :logical_width]
        outer_layout = getattr(metadata, "outer_layout", None)
        layout = outer_layout if outer_layout is not None else getattr(metadata, "query_layout", None)
        if layout is not None:
            if outer_layout is not None and outer_layout.all_valid:
                attention_inputs = layout.natural_attention(attention_inputs)
            else:
                with profile_scope("RTP::attention.kda.zigzag_to_natural"):
                    fused = layout.gather_natural(fused)
                    attention_inputs = layout.natural_attention(attention_inputs)
        qkv, gate, fa, beta = fused.split(
            [3 * self.width, self.width, self.fa_width, self.heads], dim=-1
        )
        kernel = (
            self.prefill
            if attention_inputs.is_prefill and not metadata.is_target_verify
            else self.decode
        )
        with profile_scope("RTP::attention.kda.forget_proj"):
            project_forget = (
                self._project_prefill_forget
                if kernel is self.prefill and not fa.is_contiguous()
                else self._project_forget
            )
            forget = project_forget(fa)
        with profile_scope("RTP::attention.kda.core"):
            output = kernel(
                qkv,
                forget,
                beta,
                attention_inputs,
                cache,
                metadata,
            )
        valid_mask = attention_inputs.valid_token_mask
        if valid_mask is not None and not (
            layout is not None and layout.natural_all_valid
        ):
            # Paged KDA skips null-block rows, leaving their output unspecified.
            output = torch.where(valid_mask[:, None], output.reshape(-1, self.width), 0)
            gate = torch.where(valid_mask[:, None], gate, 0)
        with profile_scope("RTP::attention.kda.output_norm_quant"):
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
                scatter_layout = None if outer_layout is not None and outer_layout.all_valid else layout
                output = self.norm(
                    output.reshape(-1, self.heads, self.dim),
                    gate.reshape(-1, self.heads, self.dim),
                    row_indices=(
                        scatter_layout.rankmajor_to_natural if scatter_layout is not None else None
                    ),
                    out=(
                        scatter_layout.reserve(
                            "output", scatter_layout.physical_rows, self.width, output.dtype
                        ).view(-1, self.heads, self.dim)
                        if scatter_layout is not None
                        else None
                    ),
                )
        if (layout is not None and self._fp8_output_norm
                and not (outer_layout is not None and outer_layout.all_valid)):
            with profile_scope("RTP::attention.kda.natural_to_zigzag"):
                values, scales = layout.scatter_rankmajor(values, scales)
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
            with profile_scope("RTP::attention.fp8_gemm_rs"):
                return self._fp8_collective.gemm_reduce_scatter(
                    values, scales, self.output
                )
        if self._bf16_collective is not None and self._bf16_collective.eligible(
            output.numel() // self.width, attention_inputs, metadata, gathered=True,
        ):
            with profile_scope("RTP::attention.bf16_gemm_rs"):
                return self._bf16_collective.gemm_reduce_scatter(
                    output.reshape(-1, self.width), self.output,
                )
        with profile_scope("RTP::attention.kda.output_proj"):
            output = (
                self.output.forward_quantized(values, scales)
                if self._fp8_output_norm
                else self.output(output.reshape(-1, self.width))
            )
        with profile_scope("RTP::attention.output_reduce_scatter"):
            return reduce_scatter(output, Group.TP) if self.tp_size > 1 else output
