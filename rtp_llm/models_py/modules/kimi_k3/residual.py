"""Kimi K3 attention-residual selection."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class KimiK3AttentionResidual(nn.Module):
    """Select over K3 block anchors and the running prefix residual."""

    def __init__(
        self,
        norm_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm_weight = norm_weight.reshape(-1).contiguous()
        # RTP's linear loader returns [hidden, 1] for this scalar projection.
        # Flatten at initialization: singleton strides can be non-unit even
        # when PyTorch considers that matrix contiguous. The native kernel
        # consumes a dense vector, and graph replay must not repack weights.
        self.projection_weight = projection_weight.reshape(-1).contiguous()
        self.eps = float(eps)
        self._verify_fp8 = False

    def configure_verify_fp8(self, enabled: bool) -> None:
        self._verify_fp8 = bool(enabled)

    def forward_fp8(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        output_norm_weight: torch.Tensor,
        output_norm_eps: float,
        num_blocks: int,
        block_write_idx: int = -1,
        delta: Optional[torch.Tensor] = None,
        shared_delta: Optional[torch.Tensor] = None,
        metadata=None,
    ):
        from rtp_llm.models_py.triton_kernels.common.activation import attn_res_fp8

        return attn_res_fp8(
            prefix_sum,
            block_residual,
            self.norm_weight,
            self.projection_weight,
            self.eps,
            output_norm_weight=output_norm_weight,
            output_norm_eps=output_norm_eps,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            delta=delta,
            shared_delta=shared_delta,
        )

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        output_norm_weight: Optional[torch.Tensor] = None,
        output_norm_eps: Optional[float] = None,
        delta: Optional[torch.Tensor] = None,
        shared_delta: Optional[torch.Tensor] = None,
        num_blocks: Optional[int] = None,
        block_write_idx: int = -1,
        metadata=None,
        also_quantize: bool = False,
        quantize_nvfp4: bool = False,
        split_nvfp4_quant: bool = False,
        rowwise_nvfp4: bool = False,
        decode_head_gate_columns: int = 0,
    ):
        if quantize_nvfp4 and output_norm_weight is None:
            raise ValueError("AttnRes NVFP4 requires output RMSNorm")
        if rowwise_nvfp4 and (not quantize_nvfp4 or also_quantize):
            raise ValueError("Rowwise NVFP4 AttnRes requires a packed output")
        if self._verify_fp8 and getattr(metadata, "is_target_verify", False):
            return self.forward_fp8(
                prefix_sum,
                block_residual,
                output_norm_weight=output_norm_weight,
                output_norm_eps=output_norm_eps,
                delta=delta,
                shared_delta=shared_delta,
                num_blocks=(
                    block_residual.shape[1] if num_blocks is None else num_blocks
                ),
                block_write_idx=block_write_idx,
            )
        if prefix_sum.ndim != 2:
            raise ValueError("AttnRes prefix_sum must have shape [tokens, hidden]")
        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != prefix_sum.shape[0]
            or block_residual.shape[2] != prefix_sum.shape[1]
        ):
            raise ValueError(
                "AttnRes block_residual must have shape [tokens, blocks, hidden]: "
                f"prefix_sum={tuple(prefix_sum.shape)}, "
                f"block_residual={tuple(block_residual.shape)}"
            )
        active_blocks = (
            block_residual.shape[1] if num_blocks is None else int(num_blocks)
        )
        if active_blocks < 0 or active_blocks > block_residual.shape[1]:
            raise ValueError("AttnRes valid block count is outside the residual bank")
        if block_write_idx < -1 or block_write_idx >= block_residual.shape[1]:
            raise ValueError("AttnRes block write index is outside the residual bank")
        if output_norm_weight is not None and output_norm_eps is None:
            raise ValueError(
                "output_norm_eps is required when output RMSNorm is requested"
            )
        if shared_delta is not None and (
            delta is None
            or shared_delta.shape != prefix_sum.shape
            or shared_delta.dtype != prefix_sum.dtype
            or shared_delta.device != prefix_sum.device
            or not shared_delta.is_contiguous()
        ):
            raise ValueError("AttnRes shared delta requires matching dense MoE deltas")
        if (
            prefix_sum.is_cuda
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.shape[1] == 7168
            and torch.cuda.get_device_capability(prefix_sum.device)[0] == 10
        ):
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            if not hasattr(rtp_llm_ops, "attn_res"):
                raise RuntimeError("Native AttnRes requires the CUDA 13 RTP build")
            # DSA's local Indexer head-gate GEMM also consumes the normalized BF16.
            packet_only = rowwise_nvfp4 and not decode_head_gate_columns
            output = prefix_sum if packet_only else torch.empty_like(prefix_sum)
            fp8 = None
            kwargs = {}
            decode_packet = None
            if decode_head_gate_columns:
                from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
                    DsaDecodeActivation,
                )

                decode_packet = DsaDecodeActivation.empty(
                    output,
                    decode_head_gate_columns,
                    rowwise=rowwise_nvfp4,
                )
                packet, fp8, _ = decode_packet.views()
            elif also_quantize:
                from rtp_llm.models_py.distributed.collective_torch import Fp8Activation

                rows, width = output.shape
                fp8 = Fp8Activation(
                    torch.empty_like(output, dtype=torch.float8_e4m3fn),
                    torch.empty(
                        (width // 512, (rows + 3) // 4 * 4),
                        dtype=torch.int32,
                        device=output.device,
                    ),
                )
            if fp8 is not None:
                kwargs.update(fp8_output=fp8.values, fp8_scales=fp8.scale_wire)
            if quantize_nvfp4:
                from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
                    PackedNvfp4Activation,
                )

                if decode_packet is None:
                    packet = PackedNvfp4Activation.empty(
                        *output.shape,
                        output.device,
                        rowwise=rowwise_nvfp4,
                    )
                if rowwise_nvfp4:
                    kwargs.update(nvfp4_packet=packet.wire, nvfp4_rowwise=True)
                else:
                    if not split_nvfp4_quant:
                        kwargs["nvfp4_packet"] = packet.wire
                    num_sms = torch.cuda.get_device_properties(
                        prefix_sum.device
                    ).multi_processor_count
                    amax_partials = output.new_empty(num_sms, dtype=torch.float32)
                    kwargs["amax_partials"] = amax_partials
            rtp_llm_ops.attn_res(
                prefix_sum,
                delta,
                block_residual,
                self.norm_weight,
                self.projection_weight,
                output_norm_weight,
                output,
                active_blocks,
                block_write_idx,
                self.eps,
                0.0 if output_norm_eps is None else float(output_norm_eps),
                shared_delta=shared_delta,
                **kwargs,
            )
            if quantize_nvfp4 and split_nvfp4_quant and not rowwise_nvfp4:
                # Give large prefill packets a packing grid independent of AttnRes.
                from rtp_llm.models_py.triton_kernels.common.nvfp4 import (
                    quantize_nvfp4_from_partials,
                )
                from rtp_llm.models_py.utils.module_profiler import profile_scope

                with profile_scope("RTP::attention.attn_res_nvfp4_packet_quant"):
                    quantize_nvfp4_from_partials(output, amax_partials, packet=packet)
            if decode_packet is not None:
                return decode_packet
            if quantize_nvfp4:
                return (output, fp8, packet) if fp8 is not None else packet
            return (output, fp8) if fp8 is not None else output
        if quantize_nvfp4:
            raise ValueError(
                "Packed AttnRes NVFP4 requires native SM10x BF16 [M,7168] input"
            )
        if shared_delta is not None:
            if delta is None:
                raise ValueError("AttnRes shared delta requires routed output")
            delta = delta + shared_delta
        if delta is not None:
            prefix_sum.add_(delta)
        if block_write_idx >= 0:
            block_residual[:, block_write_idx].copy_(prefix_sum)
        if active_blocks == 0:
            output = prefix_sum
        else:
            candidates = torch.cat(
                (block_residual[:, :active_blocks], prefix_sum.unsqueeze(1)), dim=1
            )
            candidates_float = candidates.float()
            normalized = candidates_float * torch.rsqrt(
                candidates_float.square().mean(dim=-1, keepdim=True) + self.eps
            )
            score_weight = (
                self.norm_weight.float() * self.projection_weight.reshape(-1).float()
            )
            probabilities = torch.softmax(
                (normalized * score_weight).sum(dim=-1), dim=-1
            )
            # Keep the mixture in FP32 through a fused output RMSNorm. Rounding
            # it to BF16 first changes the K3 decoder's normalization input.
            output = torch.einsum("tb,tbd->td", probabilities, candidates_float)
        if output_norm_weight is None:
            return output.to(dtype=prefix_sum.dtype)
        output_float = output.float()
        normalized = output_float * torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + output_norm_eps
        )
        return (output_norm_weight.float() * normalized).to(dtype=prefix_sum.dtype)


__all__ = ["KimiK3AttentionResidual"]
