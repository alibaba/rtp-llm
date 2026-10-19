"""Fused categorical-entropy Triton kernel (one launch, no intermediates).

Replaces ``torch.distributions.Categorical(logits=...).entropy()`` on hot
full-vocab paths: the torch path materializes the logsumexp-normalized
logits, the softmax probabilities, the clamped product and the final sum —
8+ full-width fp32 passes over an [N, V] logits block (the DiffusionGemma
proposer runs it twice per denoising step). This kernel launches once,
reads each row twice (row max, then the fused reductions; the second read
is L2-resident for the 1 MB rows of the proposer path) and writes one fp32
scalar per row:

    m = max(l)                       # loads clamped to fp32 finfo.min,
    s = sum(exp(l - m))              #   exactly like torch's Categorical
    u = sum((l - m) * exp(l - m))    #   .entropy clamp (-inf lanes -> 0)
    entropy = log(s) - u / s

which is the centered form of ``-sum(p * log p)``: identical algebra to
torch's ``lse = m + log(s); normed = l - lse; p = exp(normed);
entropy = -sum(normed * p)`` while avoiding the large-magnitude
cancellation of the naive ``lse - sum(p * l)`` form. Divergence from torch:
a row of all-(-inf) logits returns ``log(V)`` where torch yields NaN (the
proposer path is tanh-bounded, so this is unreachable there).

Agreement with the torch reference is fp32 reduction-order noise: on H20 the
fused-vs-reference gap measured <= 5.7e-6 over flat / sharp / softcap-bounded
[1024, 262144] blocks, and a float64 Categorical referee shows the kernel is
consistently closer to the true entropy than the fp32 torch chain itself
(kernel-vs-truth <= 2.1e-6, torch-vs-truth up to 5.5e-6). diffusion_gemma_
proposer_test asserts kernel-vs-truth < 1e-5 (contract bound) and sanity-
bounds the fused-vs-reference gap (GPU A/B, contract #5-iv). The kernel
compiles through the normal Triton JIT (TRITON_CACHE_DIR, managed by
jit_cache_manager as the "triton" component) — no local warmup logic is
needed beyond the first call.
"""

import torch
import triton
import triton.language as tl

# torch.finfo(torch.float32).min, inlined in the kernel body below: torch's
# Categorical clamps logits to this value before p_log_p so -inf lanes
# contribute exactly 0 instead of NaN.

_BLOCK_V = 4096
_NUM_WARPS = 8


@triton.jit
def _categorical_entropy_kernel(
    logits_ptr,
    entropy_ptr,
    vocab_size,
    stride_row,
    BLOCK_V: tl.constexpr,
):
    # int64 row offset: the API is shape-generic (rows * V can exceed 2^31
    # for >8GB fp32 blocks even though the proposer path is 1 GB per call).
    row = tl.program_id(axis=0).to(tl.int64)
    base = logits_ptr + row * stride_row
    # pass 1: row max of the clamped logits.
    m = float("-inf")
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        l = tl.load(base + idx, mask=mask, other=float("-inf"))
        l = tl.maximum(l, -3.4028234663852886e38)
        m = tl.maximum(m, tl.max(l, axis=0))
    # pass 2: s = sum(exp(l - m)); u = sum((l - m) * exp(l - m)).
    s = 0.0
    u = 0.0
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        l = tl.load(base + idx, mask=mask, other=float("-inf"))
        l = tl.maximum(l, -3.4028234663852886e38)
        centered = l - m
        e = tl.exp(centered)
        s += tl.sum(e, axis=0)
        u += tl.sum(centered * e, axis=0)
    entropy = tl.log(s) - u / s
    tl.store(entropy_ptr + row, entropy)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of ``Categorical(logits)`` over the last dim, fp32 in/out.

    Requires CUDA fp32 logits (the proposer's hot path); the caller owns
    the gate and the torch.distributions fallback for CPU / other dtypes
    (see diffusion_gemma_proposer._categorical_entropy). Returns fp32 with
    shape ``logits.shape[:-1]``.
    """
    if logits.dim() < 1:
        raise ValueError("categorical_entropy needs at least one dimension")
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise ValueError("categorical_entropy requires CUDA fp32 logits")
    vocab_size = int(logits.shape[-1])
    rows = logits.numel() // vocab_size if vocab_size else 0
    out_shape = logits.shape[:-1]
    if rows == 0:
        return torch.empty(out_shape, dtype=torch.float32, device=logits.device)
    flat = logits.reshape(rows, vocab_size)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = torch.empty(rows, dtype=torch.float32, device=flat.device)
    _categorical_entropy_kernel[(rows,)](
        flat,
        out,
        vocab_size,
        flat.stride(0),
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return out.view(out_shape)
