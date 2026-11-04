"""Fused categorical-sampling Triton kernel (one launch, no intermediates).

Replaces the torch EB-sampler chain ``softmax(logits) -> multinomial(probs, 1)``
on hot full-vocab paths: the torch path materializes a full [N, V] fp32
probability tensor (one read + one write of the logits block), then the
multinomial kernel walks it again (plus its own noise/cdf work) — three-plus
full-width passes and an [N, V] fp32 allocation per denoising step of the
DiffusionGemma proposer. This kernel launches once, reads each logits row
exactly once and writes one token id per (row, sample):

    token = argmax_v ( logit_v + g_v ),   g_v = -log(-log(u_v)),  u_v ~ U(0,1)

which is the Gumbel-max identity: for ``p = softmax(logits)`` the argmax of
``log(p) + g`` is an exact draw from ``Categorical(p)``, and the shared
``-logsumexp`` offset of ``log(p)`` cancels inside the argmax, so the softmax
never has to be formed. The temperature is applied by the caller before the
kernel (the proposer's ``processed`` logits are temperature-scaled once and
are reused by the entropy/argmax/self-conditioning consumers).

The per-lane uniforms come from a counter-based hash (the splitmix64
finalizer) over ``(seed, row, sample, vocab index)``:

    z = splitmix64_finalize(seed + pid * V + idx)
    u = (z & 0xFFFFFF) * 2^-24   # 24 uniform bits in [0, 1)

(the masked LOW 24 bits, not a right shift: Triton's ``>>`` on signed int64
is an arithmetic shift, so the high-bit extraction would sign-extend; the
finalizer's closing ``z ^ (z >> 31)`` avalanche leaves the low bits fully
mixed, so they are as uniform as the high ones)

so a fixed ``seed`` reproduces the exact same tokens (same-seed determinism,
the criterion the proposer relies on for golden reproducibility), and any
seed change reshuffles the stream. In infinite precision Gumbel-max is an
exact categorical draw; the finite-precision deviation is the 24-bit uniform
granularity and fp32 ``log`` rounding, which is statistically invisible at
the distribution level: the acceptance evidence in diffusion_gemma_
proposer_test compares the fused sampler against the torch softmax+
multinomial reference with >= 1e3-sample empirical distributions
(total-variation < 1%, chi-square vs the analytic softmax) rather than
token-for-token equality, because sampling is stochastic and the two paths
by design do not share an RNG stream.

The kernel compiles through the normal Triton JIT (TRITON_CACHE_DIR, managed
by jit_cache_manager as the "triton" component) — no local warmup logic is
needed beyond the first call. Divergence from torch: rows of all-(-inf)
logits still return a well-defined (noise-argmax) token where torch's
softmax+multinomial yields NaN/undefined; the proposer path is
tanh-bounded, so this is unreachable there.

Opt r5 softcap-inline mode: with ``softcap``/``out_scale`` the caller passes
the RAW pre-softcap logits (bf16 or fp32) and the kernel applies
``tanh(x/s) * out_scale`` in-register (contract #5-iii merged algebra,
libdevice ``__nv_tanhf`` — the entry point torch's CUDA tanh lowers to), so
the temperature/softcap elementwise materialization never runs for this
consumer.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

_BLOCK_V = 4096
_NUM_WARPS = 8


@triton.jit
def _splitmix64_mix(z):
    """splitmix64 finalizer: hashes one int64 counter into a well-mixed
    int64 (sequential counters give decorrelated outputs, which is exactly
    how the splitmix64 generator produces its stream). The two multiplier
    constants are inlined — Triton @jit bodies may only read globals that
    are triton.language.constexpr instances — as signed int64 decimals of
    0xBF58476D1CE4E5B9 and 0x94D049BB133111EB (the hex forms exceed
    INT64_MAX and Triton integer literals are signed)."""
    z = (z ^ (z >> 30)) * -4658895280553007687
    z = (z ^ (z >> 27)) * -7723592293110705685
    return z ^ (z >> 31)


@triton.jit
def _categorical_sample_kernel(
    logits_ptr,
    tokens_ptr,
    vocab_size,
    stride_row,
    seed,
    samples_per_row,
    softcap,
    out_scale,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # int64 pid: rows * samples * V can exceed 2^31 (the proposer's hot block
    # is [2048, 262144] fp32 = 2 GB; the A/B statistics path runs
    # 131072 draws over the same vocab).
    pid = tl.program_id(axis=0).to(tl.int64)
    row = pid // samples_per_row
    base = logits_ptr + row * stride_row
    # running (max, argmax) over the noisy logits; idx is int32 (V < 2^31).
    m = float("-inf")
    am = -1
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        l = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            # opt r5 softcap-inline: transform the RAW logits in-register
            # (tanh(x/s) * out_scale, contract #5-iii merged algebra via
            # libdevice __nv_tanhf) so the draw is over softmax of the
            # softcapped/temperature-scaled distribution without the
            # materialized [N, V] passes. Padded lanes keep -inf and lose
            # the argmax as in the plain path.
            l = tl.where(mask, libdevice.tanh(l / softcap) * out_scale, l)
        # per-lane counter stream: seed + pid * V + idx — unique per
        # (seed, row, sample, vocab lane), sequential across lanes.
        counter = seed + pid * vocab_size + idx
        z = _splitmix64_mix(counter)
        # 24 uniform bits from the masked low half, scaled by 2^-24 (the
        # constant is inlined: Triton @jit bodies may only read constexpr
        # globals; see module docstring for why the low bits, not a shift).
        u = (z & 0xFFFFFF).to(tl.float32) * 5.9604644775390625e-08
        # Gumbel(0,1) noise; u == 0 (probability 2^-24) gives g = -inf,
        # i.e. that lane simply loses the argmax.
        g = -tl.log(-tl.log(u))
        cand = l + g
        cmax, cidx = tl.max(cand, axis=0, return_indices=True)
        upd = cmax > m
        m = tl.where(upd, cmax, m)
        am = tl.where(upd, off + cidx, am)
    tl.store(tokens_ptr + pid, am.to(tl.int64))


def categorical_sample(
    logits: torch.Tensor,
    seed: int,
    samples_per_row: int = 1,
    softcap: Optional[float] = None,
    out_scale: Optional[float] = None,
) -> torch.Tensor:
    """Draw ``samples_per_row`` tokens per row from ``softmax(logits)``.

    Gumbel-max fused sampling over the last dim: fp32 CUDA logits in, int64
    token ids out (torch.multinomial-compatible dtype), one kernel launch,
    one logits read, no [N, V] probability materialization. ``seed`` is any
    Python integer (folded to signed 64-bit) controlling the deterministic
    counter-based RNG stream: same (seed, logits) -> same tokens.

    ``softcap``/``out_scale`` (opt r5 softcap-inline): when ``softcap`` is
    given, ``logits`` are the RAW pre-softcap rows (bf16 or fp32) and the
    draw runs over ``softmax(tanh(x/s) * out_scale)`` with the transform
    applied in-register — bf16 inputs upcast exactly like the
    ``.to(torch.float32)`` cast they replace.

    Returns shape ``logits.shape[:-1] + (samples_per_row,)``; with
    ``samples_per_row == 1`` (the proposer's call shape) the trailing 1 is
    squeezed, i.e. ``logits.shape[:-1]`` — matching
    ``torch.multinomial(probs, 1).squeeze(-1)``.
    """
    if logits.dim() < 1:
        raise ValueError("categorical_sample needs at least one dimension")
    if not logits.is_cuda:
        raise ValueError("categorical_sample requires CUDA logits")
    if softcap is None:
        if logits.dtype != torch.float32:
            raise ValueError("categorical_sample requires CUDA fp32 logits")
    elif logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(
            "categorical_sample softcap mode requires fp32/bf16/fp16 logits"
        )
    if samples_per_row < 1:
        raise ValueError("samples_per_row must be >= 1")
    vocab_size = int(logits.shape[-1])
    if vocab_size < 1:
        raise ValueError("categorical_sample needs a non-empty last dim")
    rows = logits.numel() // vocab_size
    out_shape = logits.shape[:-1] + (samples_per_row,)
    if samples_per_row == 1:
        out_shape = logits.shape[:-1]
    if rows == 0:
        return torch.empty(out_shape, dtype=torch.int64, device=logits.device)
    flat = logits.reshape(rows, vocab_size)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    if seed >= 1 << 63:
        seed -= 1 << 64
    out = torch.empty(
        rows * samples_per_row, dtype=torch.int64, device=flat.device
    )
    _categorical_sample_kernel[(rows * samples_per_row,)](
        flat,
        out,
        vocab_size,
        flat.stride(0),
        seed,
        samples_per_row,
        float(softcap) if softcap is not None else 1.0,
        float(out_scale) if out_scale is not None else 1.0,
        HAS_SOFTCAP=softcap is not None,
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return out.view(out_shape)
