"""Adapter wrapping FlashInfer's chunk_gated_delta_rule_sm100 to match
the existing RTP-LLM calling convention."""

import math
from typing import Optional, Tuple, Union

import torch

from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """RTP-LLM compatible wrapper around FlashInfer's Blackwell GDN prefill.

    Input convention (RTP-LLM):
        q, k: [B, T, H_qk, D]
        v:    [B, T, H_v, D]
        g:    [1, total_tokens, H_v] float32, log-space decay
        beta: [1, total_tokens, H_v] float32
        initial_state: [N, H_v, D, D] float32 or None
        cu_seqlens: [N+1] int32

    FlashInfer SM100 convention:
        q, k: [total_tokens, H_qk, D]
        v:    [total_tokens, H_v, D]
        gate: [total_tokens, H_v] float32, decay factor = exp(g)
        beta: [total_tokens, H_v] float32
        initial_state: [N, H_v, D, D] float32 or None
        cu_seqlens: [N+1] int32
    """
    B, T, H_qk, D = q.shape
    H_v = v.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if use_qk_l2norm_in_kernel:
        from rtp_llm.models_py.triton_kernels.fla.l2norm import (
            l2norm_fwd,
            l2norm_fwd_qk,
        )

        if (
            q.stride() == k.stride()
            and q.shape == k.shape
            and not q.is_contiguous()
            and q.stride(-1) == 1
            and q.ndim >= 3
            and k.data_ptr()
            == q.data_ptr() + q.shape[-2] * q.shape[-1] * q.element_size()
        ):
            q, k = l2norm_fwd_qk(q, k)
        else:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

    # Reshape [B, T, H, D] -> [total_tokens, H, D]
    total_tokens = B * T
    q_3d = q.reshape(total_tokens, H_qk, D).contiguous()
    k_3d = k.reshape(total_tokens, H_qk, D).contiguous()
    v_3d = v.reshape(total_tokens, H_v, D).contiguous()

    # Gate: log-space -> decay factor
    # g: [1, total_tokens, H_v] -> [total_tokens, H_v]
    gate = g.view(total_tokens, H_v).exp()

    # Beta: [1, total_tokens, H_v] -> [total_tokens, H_v]
    beta_2d = beta.view(total_tokens, H_v).float()

    # Output
    if output is None:
        output = torch.empty(total_tokens, H_v, D, dtype=v.dtype, device=v.device)
    else:
        output = output.reshape(total_tokens, H_v, D)

    # State
    fi_output_state = output_state if output_final_state else None

    chunk_gated_delta_rule_sm100(
        q=q_3d,
        k=k_3d,
        v=v_3d,
        gate=gate,
        beta=beta_2d,
        output=output,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_state=fi_output_state,
        scale=scale,
    )

    # Reshape output back to [B, T, H_v, D]
    output_4d = output.view(B, T, H_v, D)

    if output_final_state:
        return output_4d, fi_output_state
    return output_4d
