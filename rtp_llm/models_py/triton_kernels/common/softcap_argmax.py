"""Fused softcap-transform + argmax Triton kernel (one launch, one read).

Opt r5 softcap-inline (⑥-clearing lane, contract #5-iii): the DiffusionGemma
proposer's per-denoising-step epilogue used to run three full-vocab torch
passes over the raw lm_head logits before any consumer could touch them —

    logits = linear(hidden, embed_weight).to(torch.float32)   # cast pass
    logits = torch.tanh(logits / softcap)                     # tanh pass
    processed = logits * (softcap / temperature)              # scale pass
    new_argmax = torch.argmax(processed, dim=-1)              # argmax pass

(profiled as the 52.6 ms tanh/mul pair plus 15.8 ms ArgMax over one decode
window). The transform is elementwise and the argmax only needs the
transformed values in registers, so this kernel reads each raw logits row
exactly once, computes ``c = tanh(x/s) * out_scale`` (the #5-iii merged
algebra — two roundings either way, values equal to the sequential chain to
the last fp32 ulp), writes ``c`` to the processed buffer (the one consumer
that still needs a materialized tensor: the next step's self-conditioning
softmax+matmul) and keeps a running per-row (max, argmax) over ``c``.

Argmax semantics: lower index wins on exact ties (``c == cmax`` picks
``tl.min`` over the matching offsets; cross-block updates use strict ``>``),
matching torch.argmax's first-maximal-index contract. The transform is
strictly monotonic in ``x`` for fixed positive ``s``/``out_scale``, so the
in-register argmax equals the argmax of the materialized chain up to fp32
tanh rounding — only saturation-collapsed top values (|x| > ~8.9*s, where
fp32 tanh rounds to +-1.0) can merge into ties, and ties resolve to the
first index on both paths. bf16 raw inputs upcast exactly like the
``.to(torch.float32)`` cast they replace.

The kernel compiles through the normal Triton JIT (TRITON_CACHE_DIR,
managed by jit_cache_manager as the "triton" component).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

_BLOCK_V = 4096
_NUM_WARPS = 8


@triton.jit
def _softcap_argmax_kernel(
    logits_ptr,
    processed_ptr,
    argmax_ptr,
    vocab_size,
    stride_row,
    softcap,
    out_scale,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # int64 row offset: rows * V can exceed 2^31 (the proposer's raw block is
    # [B*canvas, 262144]).
    row = tl.program_id(axis=0).to(tl.int64)
    base = logits_ptr + row * stride_row
    out_base = processed_ptr + row * vocab_size
    # running (max, argmax) over the transformed logits; am starts at 0 so an
    # all-(-inf)/all-tie row reports index 0 like torch.argmax.
    m = float("-inf")
    am = 0
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            c = tl.where(
                mask, libdevice.tanh(x / softcap) * out_scale, x
            )
        else:
            c = x
        # materialize the transformed row for the consumers that need the
        # tensor (self-conditioning softmax+matmul; reference fallbacks).
        tl.store(out_base + idx, c, mask=mask)
        # within-block (max, first-index argmax): deterministic lower-index
        # tie-break, matching torch.argmax's first-maximal-index contract.
        cmax = tl.max(c, axis=0)
        is_max = c == cmax
        first = tl.min(tl.where(is_max, idx, vocab_size), axis=0)
        # cross-block: strict > keeps the earliest block on ties.
        upd = cmax > m
        m = tl.where(upd, cmax, m)
        am = tl.where(upd, first, am)
    tl.store(argmax_ptr + row, am.to(tl.int64))


@triton.jit
def _softcap_softmax_argmax_kernel(
    logits_ptr,
    probs_ptr,
    argmax_ptr,
    vocab_size,
    stride_row,
    softcap,
    out_scale,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Softcap-transform + row softmax + row argmax, one kernel.

    Opt r9 ⑥ tail-clear lane, item 2 (softmax fold): the proposer's only
    materialized-``processed`` consumer is the next step's
    self-conditioning ``softmax(processed).to(embed_dtype)`` matmul — the
    ATen cunn_SoftMaxForward (42.2 ms / 122x per decode window) plus the
    full-vocab fp32->bf16 cast pass. This kernel reads the RAW logits
    instead, applies the same ``c = tanh(x/s) * out_scale`` transform
    in-register, and emits the softmax probabilities directly in the
    consumer's dtype — the fp32 ``processed`` materialization, the ATen
    softmax and the cast pass all drop.

    Pass 1 = the plain softcap_argmax (max, first-index argmax) loop.
    Pass 2 accumulates sum(exp(c - m)) in fp32. Pass 3 re-reads the row and
    stores exp(c - m) / sum as ``probs`` (bf16/fp16/fp32 — the division is
    fp32 either way, so vs the torch ``softmax(fp32).to(dtype)`` reference
    the difference is accumulation order plus exp/tanh rounding: bf16-ulp
    class, A/B-gated). Masked tail lanes carry c = -inf and contribute 0.
    """
    # int64 row offset: rows * V can exceed 2^31 (the proposer's raw block
    # is [B*canvas, 262144]).
    row = tl.program_id(axis=0).to(tl.int64)
    base = logits_ptr + row * stride_row
    probs_base = probs_ptr + row * vocab_size
    # ---- pass 1: (max, first-index argmax) of the transformed row.
    m = float("-inf")
    am = 0
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            c = tl.where(mask, libdevice.tanh(x / softcap) * out_scale, x)
        else:
            c = x
        cmax = tl.max(c, axis=0)
        is_max = c == cmax
        first = tl.min(tl.where(is_max, idx, vocab_size), axis=0)
        upd = cmax > m
        m = tl.where(upd, cmax, m)
        am = tl.where(upd, first, am)
    # ---- pass 2: sum of exp(c - m); masked lanes are exp(-inf - m) = 0.
    s = 0.0
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            c = tl.where(mask, libdevice.tanh(x / softcap) * out_scale, x)
        else:
            c = x
        s += tl.sum(tl.exp(c - m), axis=0)
    # ---- pass 3: write probs = exp(c - m) / s in the consumer dtype.
    inv = 1.0 / s
    for off in range(0, vocab_size, BLOCK_V):
        idx = off + tl.arange(0, BLOCK_V)
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_SOFTCAP:
            c = tl.where(mask, libdevice.tanh(x / softcap) * out_scale, x)
        else:
            c = x
        p = tl.exp(c - m) * inv
        tl.store(probs_base + idx, p.to(probs_ptr.dtype.element_ty), mask=mask)
    tl.store(argmax_ptr + row, am.to(tl.int64))


def softcap_argmax(
    logits: torch.Tensor,
    softcap: Optional[float],
    out_scale: Optional[float] = None,
    emit_softmax: bool = False,
    probs_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softcap-transform a raw logits block and take its row argmax.

    Computes ``processed = tanh(x / softcap) * out_scale`` and
    ``argmax = argmax(processed, dim=-1)`` in one launch: fp32/bf16 CUDA
    raw logits in, fp32 ``processed`` (``logits.shape``) and int64
    ``argmax`` (``logits.shape[:-1]``) out, one logits read, one processed
    write. ``softcap=None`` degenerates to a pure cast+argmax (identity
    transform, still one pass).

    With ``emit_softmax=True`` (opt r9 item 2) the first return value is
    instead ``softmax(processed)`` materialized directly in ``probs_dtype``
    (bf16 by default — the self-conditioning consumer's embed dtype): the
    raw logits are read three times, the row max / exp-sum / normalized
    probabilities are computed in-register, and the fp32 ``processed``
    tensor, the ATen softmax and the dtype cast never materialize. The
    argmax contract (first-maximal index, saturation ties included) is
    unchanged.

    Returns ``(processed_or_probs, argmax)``.
    """
    if logits.dim() < 1:
        raise ValueError("softcap_argmax needs at least one dimension")
    if not logits.is_cuda:
        raise ValueError("softcap_argmax requires CUDA logits")
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("softcap_argmax requires fp32/bf16/fp16 logits")
    if softcap is not None and not float(softcap) > 0.0:
        raise ValueError("softcap must be positive when given")
    if emit_softmax and probs_dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("emit_softmax needs a fp32/bf16/fp16 probs_dtype")
    vocab_size = int(logits.shape[-1])
    if vocab_size < 1:
        raise ValueError("softcap_argmax needs a non-empty last dim")
    rows = logits.numel() // vocab_size
    out_shape = logits.shape
    argmax_shape = logits.shape[:-1]
    if rows == 0:
        return (
            torch.empty(out_shape, dtype=probs_dtype, device=logits.device),
            torch.empty(argmax_shape, dtype=torch.int64, device=logits.device),
        )
    flat = logits.reshape(rows, vocab_size)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    argmax = torch.empty(rows, dtype=torch.int64, device=flat.device)
    if emit_softmax:
        probs = torch.empty(rows, vocab_size, dtype=probs_dtype, device=flat.device)
        _softcap_softmax_argmax_kernel[(rows,)](
            flat,
            probs,
            argmax,
            vocab_size,
            flat.stride(0),
            float(softcap) if softcap is not None else 1.0,
            float(out_scale) if out_scale is not None else 1.0,
            HAS_SOFTCAP=softcap is not None,
            BLOCK_V=_BLOCK_V,
            num_warps=_NUM_WARPS,
        )
        return probs.view(out_shape), argmax.view(argmax_shape)
    processed = torch.empty(rows, vocab_size, dtype=torch.float32, device=flat.device)
    _softcap_argmax_kernel[(rows,)](
        flat,
        processed,
        argmax,
        vocab_size,
        flat.stride(0),
        float(softcap) if softcap is not None else 1.0,
        float(out_scale) if out_scale is not None else 1.0,
        HAS_SOFTCAP=softcap is not None,
        BLOCK_V=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return processed.view(out_shape), argmax.view(argmax_shape)
