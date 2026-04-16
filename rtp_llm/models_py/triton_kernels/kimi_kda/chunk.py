# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Related files are modified and supported by the Moonshot AI Team
# Adapted for rtp-llm: forward-only, no backward, no CP.

import torch

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_fwd import chunk_kda_fwd


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        return_intermediate_states: bool = False,
    ):
        chunk_size = 64

        # Apply l2norm (ensure contiguous for view compatibility)
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q.contiguous())
            k = l2norm_fwd(k.contiguous())

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None
            else None
        )

        g_input = g

        (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = (
            chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=g_input,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                safe_gate=False,
                lower_bound=None,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=False,
                return_intermediate_states=return_intermediate_states,
            )
        )

        if return_intermediate_states:
            return o.type_as(q), final_state, h

        return o.type_as(q), final_state


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    return_intermediate_states: bool = False,
    **kwargs,
):
    r"""
    Standalone KDA (Kimi Delta Attention) chunk operator for inference.

    Args:
        q (torch.Tensor): queries of shape `[B, T, H, K]`.
        k (torch.Tensor): keys of shape `[B, T, H, K]`.
        v (torch.Tensor): values of shape `[B, T, H, V]`.
        g (torch.Tensor): gating tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor): betas of shape `[B, T, H]`.
        scale (Optional[float]): Scale factor. Default: `1 / sqrt(K)`.
        initial_state (Optional[torch.Tensor]): Initial state `[N, H, K, V]`.
        output_final_state (bool): Whether to output final state.
        use_qk_l2norm_in_kernel (bool): Apply L2norm to q,k internally.
        use_gate_in_kernel (bool): Compute KDA decay internally.
            If True, kwargs must contain `A_log` and optionally `dt_bias`.
        cu_seqlens (torch.LongTensor): Cumulative sequence lengths `[N+1]`.
        return_intermediate_states (bool): Return intermediate states for inference.

    Returns:
        (o, final_state) or (o, final_state, h) if return_intermediate_states=True.
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert (
        beta.shape == q.shape[:3]
    ), "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (
        *q.shape[:3],
        v.shape[-1],
    ), "v must be of shape (batch size, seq len, num of head, head dim)."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    return ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        return_intermediate_states,
    )
