"""Triton: merge partial attention (v, logsumexp) over disjoint KV shards (cascade / CP).

Same math as ``flashinfer.cascade.merge_states`` / ``torch.logsumexp`` weighting, but
supports large ``head_dim`` (e.g. MLA ``kv_lora_rank`` 512) where FlashInfer CUDA
kernels may reject the launch configuration.

v: [seq_len, num_states, num_heads, head_dim]
s: [seq_len, num_states, num_heads] float32

out: [seq_len, num_heads, head_dim]

``num_states`` must be <= ``MAX_STATES`` (default 32), sufficient for typical CP width.
"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

# Upper bound on context-parallel width; loop is fully unrolled up to this constant.
MAX_STATES = 32


def merge_states_kv_cascade_torch_reference(
    v: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for cascade merge over disjoint KV shards.

    Same semantics as ``flashinfer.cascade.merge_states`` / Triton kernel:
    ``v``: ``[seq_len, num_states, num_heads, head_dim]``;
    ``s``: ``[seq_len, num_states, num_heads]`` (float32, log-sum-exp per shard).

    Returns:
        ``[seq_len, num_heads, head_dim]``, same dtype as ``v``.
    """
    lse_merged = torch.logsumexp(s, dim=1)
    w = torch.exp(s - lse_merged.unsqueeze(1))
    return (w.unsqueeze(-1) * v).sum(dim=1)


def _tl_dtype_from_torch(dt: torch.dtype):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton not available")
    if dt == torch.float16:
        return tl.float16
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float32:
        return tl.float32
    raise TypeError(f"Unsupported v dtype for triton merge: {dt}")


if _TRITON_AVAILABLE:

    @triton.jit
    def _merge_states_kv_cascade_kernel(
        v_ptr,
        s_ptr,
        out_ptr,
        seq_len,
        num_heads,
        head_dim,
        num_states,
        v_s0,
        v_s1,
        v_s2,
        v_s3,
        s_s0,
        s_s1,
        s_s2,
        o_s0,
        o_s1,
        o_s2,
        OUT_DTYPE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_STATES_CONST: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        seq_idx = pid_h // num_heads
        head_idx = pid_h % num_heads
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        s_base = s_ptr + seq_idx * s_s0 + head_idx * s_s2

        s_max = -1e30
        for k in range(MAX_STATES_CONST):
            mask_k = k < num_states
            sk = tl.load(s_base + k * s_s1, mask=mask_k, other=-1e30)
            s_max = tl.maximum(s_max, sk)

        exp_sum = 0.0
        for k in range(MAX_STATES_CONST):
            mask_k = k < num_states
            sk = tl.load(s_base + k * s_s1, mask=mask_k, other=-1e30)
            exp_sum += tl.where(mask_k, tl.exp(sk - s_max), 0.0)
        lse = tl.log(exp_sum) + s_max

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        v_base = v_ptr + seq_idx * v_s0 + head_idx * v_s2
        for k in range(MAX_STATES_CONST):
            mask_k = k < num_states
            sk = tl.load(s_base + k * s_s1, mask=mask_k, other=-1e30)
            wk = tl.where(mask_k, tl.exp(sk - lse), 0.0)
            vk = tl.load(
                v_base + k * v_s1 + offs_d * v_s3,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            acc += wk * vk

        out_base = out_ptr + seq_idx * o_s0 + head_idx * o_s1 + offs_d * o_s2
        tl.store(out_base, acc.to(OUT_DTYPE), mask=mask_d)

else:
    _merge_states_kv_cascade_kernel = None  # type: ignore[misc, assignment]


def triton_merge_states_kv_cascade(
    v: torch.Tensor,
    s: torch.Tensor,
    *,
    block_d: int = 128,
    max_states: int = MAX_STATES,
) -> torch.Tensor:
    """Merge cascade attention states; see module docstring.

    Args:
        v: Partial attention outputs, contiguous, shape
           ``[seq_len, num_states, num_heads, head_dim]``.
        s: Log-sum-exp per shard, contiguous float32, shape
           ``[seq_len, num_states, num_heads]``.
        block_d: Triton tile size along ``head_dim``.
        max_states: Must match the kernel compile-time ``MAX_STATES_CONST`` (default 32).

    Returns:
        Merged ``v``, shape ``[seq_len, num_heads, head_dim]``, same dtype as ``v``.
    """
    if not _TRITON_AVAILABLE or _merge_states_kv_cascade_kernel is None:
        raise RuntimeError("triton is not installed")
    if not v.is_cuda or not s.is_cuda:
        raise ValueError("triton merge expects CUDA tensors")
    if s.dtype != torch.float32:
        raise ValueError(f"s must be float32, got {s.dtype}")
    if not v.is_contiguous() or not s.is_contiguous():
        raise ValueError("v and s must be contiguous")

    seq_len, num_states, num_heads, head_dim = v.shape
    if s.shape != (seq_len, num_states, num_heads):
        raise ValueError(
            f"s shape {s.shape} does not match v {(seq_len, num_states, num_heads)}"
        )
    if num_states > max_states:
        raise ValueError(
            f"num_states={num_states} exceeds max_states={max_states}; increase MAX_STATES "
            "in merge_states_kv_cascade.py and recompile."
        )

    out = torch.empty(seq_len, num_heads, head_dim, device=v.device, dtype=v.dtype)
    out_dtype = _tl_dtype_from_torch(v.dtype)

    grid = (seq_len * num_heads, triton.cdiv(head_dim, block_d))

    v_s0, v_s1, v_s2, v_s3 = v.stride()
    s_s0, s_s1, s_s2 = s.stride()
    o_s0, o_s1, o_s2 = out.stride()

    _merge_states_kv_cascade_kernel[grid](
        v,
        s,
        out,
        seq_len,
        num_heads,
        head_dim,
        num_states,
        v_s0,
        v_s1,
        v_s2,
        v_s3,
        s_s0,
        s_s1,
        s_s2,
        o_s0,
        o_s1,
        o_s2,
        OUT_DTYPE=out_dtype,
        BLOCK_D=block_d,
        MAX_STATES_CONST=max_states,
    )
    return out
