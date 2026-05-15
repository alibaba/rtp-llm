# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
import os
from typing import Optional

import torch
from einops import rearrange

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
    chunk_gated_delta_rule_fwd_intra_a_only,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import fused_l2norm_qk, l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
    is_amd,
)
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

RCP_LN2 = 1.0 / 0.6931471805599453
_TRUE_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}

# All Qwen3.5/Qwen3.6 runtime (Hg, H, K, V) shapes that the FlyDSL megakernel
# targets. ENABLED_SHAPES is the subset with validated correctness AND acceptable
# performance. Shapes in TARGET but not ENABLED (e.g. (8,8,128,128)) have passed
# correctness but show FlyDSL perf regression vs Triton, so they stay on Triton.
# Future shapes should be added to TARGET first, then promoted to ENABLED after
# both correctness and performance validation.
FLYDSL_CHUNK_GDN_TARGET_SHAPES = frozenset(
    {
        (16, 16, 128, 128),
        (8, 8, 128, 128),
        (16, 32, 128, 128),
        (8, 16, 128, 128),
        (16, 48, 128, 128),
        (8, 24, 128, 128),
        (16, 64, 128, 128),
        (8, 32, 128, 128),
        (4, 16, 128, 128),
        (2, 8, 128, 128),
    }
)
FLYDSL_CHUNK_GDN_ENABLED_SHAPES = frozenset(
    {
        (16, 16, 128, 128),
        # (8, 8, 128, 128) — target shape but excluded: FlyDSL perf regression
        (16, 32, 128, 128),
        (8, 16, 128, 128),
        (16, 48, 128, 128),
        (16, 64, 128, 128),
        (8, 24, 128, 128),
        (8, 32, 128, 128),
        (4, 16, 128, 128),
        (2, 8, 128, 128),
    }
)

FLYDSL_CHUNK_GDN_MIN_SEQ_LEN = 64


@functools.lru_cache(maxsize=None)
def _use_flydsl_chunk_gdn() -> bool:
    """Cached read of USE_FLYDSL env var (evaluated once per process)."""
    return os.getenv("USE_FLYDSL", "0").strip().lower() in _TRUE_ENV_VALUES


def is_flydsl_chunk_gdn_enabled() -> bool:
    return _use_flydsl_chunk_gdn()


def _flydsl_chunk_gdn_shape(
    q: torch.Tensor, v: torch.Tensor
) -> tuple[int, int, int, int]:
    _, _, Hg, K = q.shape
    _, _, H, V = v.shape
    return (Hg, H, K, V)


def is_flydsl_chunk_gdn_shape_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> bool:
    if not is_amd:
        return False
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        return False
    if beta.dtype != torch.bfloat16:
        return False
    return _flydsl_chunk_gdn_shape(q, v) in FLYDSL_CHUNK_GDN_ENABLED_SHAPES


def is_flydsl_chunk_gdn_length_supported(q: torch.Tensor) -> bool:
    return q.shape[1] >= FLYDSL_CHUNK_GDN_MIN_SEQ_LEN


def _validate_flydsl_chunk_gdn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    B, T, Hg, K = q.shape
    _, _, H, V = v.shape
    errors = []
    if not is_amd:
        errors.append("AMD/ROCm backend is required")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        errors.append(f"q/k/v must be bf16, got {q.dtype}/{k.dtype}/{v.dtype}")
    if beta.dtype != torch.bfloat16:
        errors.append(f"beta must be bf16, got {beta.dtype}")
    shape = (Hg, H, K, V)
    if shape not in FLYDSL_CHUNK_GDN_ENABLED_SHAPES:
        target_note = (
            "target shape pending correctness"
            if shape in FLYDSL_CHUNK_GDN_TARGET_SHAPES
            else "not in Qwen3.5/Qwen3.6 target set"
        )
        errors.append(f"unsupported Hg/H/K/V={shape} ({target_note})")
    if B < 1 or T < 1:
        errors.append(f"expected non-empty input, got B={B}, T={T}")
    if errors:
        raise ValueError(
            "USE_FLYDSL=1 Chunk-GDN path is unsupported: " + "; ".join(errors)
        )


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # AMD: scale g to log2 domain (multiply cumsum by 1/ln2) so AMD-side
    # downstream kernels can use the single-instruction exp2.
    # NVIDIA: keep the original natural-log domain. The RCP_LN2 (~1.4427) scale
    # is not bit-exact in bf16/fp16 and accumulates non-trivial drift over long
    # contexts, which has been observed to change generation outputs on CUDA.
    if is_amd:
        g = chunk_local_cumsum(
            g,
            chunk_size=64,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
        )
        # AMD-optimized: fused kkt + solve_tril + recompute_w_u
        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
        )
        # The public API must return per-chunk h/v_new. The production FlyDSL
        # path is Qwen3Next's direct-store helper, which skips materialized h.
    else:
        g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
        # Original pipeline: separate kkt -> solve_tril -> recompute_w_u
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return g, o, A, final_state, w, h, v_new


@torch.compiler.disable
def chunk_gated_delta_rule_flydsl_with_cache_store(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    prefix_lengths: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    ssm_states: Optional[torch.Tensor] = None,
    seq_size_per_block: Optional[int] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Run the FlyDSL Chunk-GDN megakernel.

    When ssm_states is provided, the kernel also writes RTP SSM block cache
    state directly. Without ssm_states it runs the same fused no-store path.
    """
    if not _use_flydsl_chunk_gdn():
        raise RuntimeError(
            "chunk_gated_delta_rule_flydsl_with_cache_store requires USE_FLYDSL=1"
        )
    if head_first:
        raise ValueError(
            "head_first is deprecated and is not supported by the FlyDSL Chunk-GDN path."
        )
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
        )
    if cu_seqlens is not None and initial_state is not None:
        expected_states = len(cu_seqlens) - 1
        if initial_state.shape[0] != expected_states:
            raise ValueError(
                f"The number of initial states is expected to be {expected_states} "
                f"rather than {initial_state.shape[0]}."
            )
    _validate_flydsl_chunk_gdn_inputs(q=q, k=k, v=v, beta=beta)
    _, _, H, V = v.shape
    K = k.shape[-1]
    if ssm_states is not None:
        if prefix_lengths is None or block_map is None or seq_size_per_block is None:
            raise ValueError(
                "prefix_lengths, block_map and seq_size_per_block are required "
                "when FlyDSL writes ssm_states directly"
            )
        if ssm_states.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"unsupported ssm_states dtype for FlyDSL direct store: {ssm_states.dtype}"
            )
        if (
            ssm_states.stride(1) != K * V
            or ssm_states.stride(2) != K
            or ssm_states.stride(3) != 1
        ):
            raise ValueError(
                "FlyDSL direct store expects ssm_states layout [block, head, V, K] "
                "with contiguous per-head state"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if prefix_lengths is not None and prefix_lengths.dtype != torch.int32:
        prefix_lengths = prefix_lengths.to(torch.int32)
    if block_map is not None and block_map.dtype != torch.int32:
        block_map = block_map.to(torch.int32)
    if prefix_lengths is not None and not prefix_lengths.is_contiguous():
        prefix_lengths = prefix_lengths.contiguous()
    if block_map is not None and not block_map.is_contiguous():
        block_map = block_map.contiguous()

    if use_qk_l2norm_in_kernel:
        if is_amd:
            q, k = fused_l2norm_qk(q, k)
        else:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

    g = chunk_local_cumsum(
        g,
        chunk_size=64,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
    )
    A = chunk_gated_delta_rule_fwd_intra_a_only(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )

    # Lazy import: flydsl depends on ROCm-only packages (flydsl.compiler, rocdl)
    # that are unavailable on non-AMD environments.
    from rtp_llm.models_py.triton_kernels.fla.flydsl_chunk_gdn_mi308x import (
        megakernel_fwd,
    )

    flydsl_initial_state = (
        initial_state.float()
        if initial_state is not None and initial_state.dtype != torch.float32
        else initial_state
    )
    flydsl_cu_seqlens = (
        cu_seqlens.to(torch.long)
        if cu_seqlens is not None and cu_seqlens.dtype != torch.long
        else cu_seqlens
    )
    o, final_state = megakernel_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=flydsl_initial_state,
        output_final_state=output_final_state,
        cu_seqlens=flydsl_cu_seqlens,
        prefix_lengths=prefix_lengths,
        block_map=block_map,
        ssm_states=ssm_states,
        seq_size_per_block=seq_size_per_block,
    )
    return o.to(q.dtype), final_state


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            if is_amd:
                q, k = fused_l2norm_qk(q, k)
            else:
                # NOTE: fused_l2norm_qk is only validated on AMD/ROCm backend.
                # On CUDA, fall back to l2norm_fwd until the fused kernel is verified.
                q = l2norm_fwd(q)
                k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        return o.to(q.dtype), h, final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, h, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, h, final_state
