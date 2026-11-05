"""D1: local-row V4.1 CP router without the per-layer all-gather.

By default (``DSV41_CP_LOCAL_ROUTER``, default ON since R8; ``=0`` is the
escape hatch) this replaces :func:`canonical_v41_cp_routes`
(all-gather the full ``[4*chunk, 5120]`` BF16 batch every MoE layer, run the
FP32 router on ALL rows, then select this rank's rows) with a router on the
rank-local rows only, V4F-style.

Why padded: cuBLAS FP32 GEMM is NOT M-invariant on SM120. Captured from a
production 32K CP4-packed prefill (40 layers x 2 chunks x 8 ranks): ``F.linear`` at M=chunk(1024) vs M=full(4096) gives bitwise-equal
logits on only ~17% of elements (ULP diffs up to 1.2e-5), producing
non-bitwise top-k weights everywhere and 2 order-only top-k rows in 344k
(0 set flips; exact-tie rows with margin 0 exist in the same population —
same hazard class as the router-TF32 tie flips). Zero-padding the local rows
to the canonical full M before the GEMM restores bitwise identity: logits,
top-k ids AND weights matched canonical on every captured layer/chunk/rank
(variants D/E, r7_router_bitwise_v2.json). Per-row determinism of the fused
sqrtsoftplus epilogue across batch sizes was verified in the same analysis,
so only the rank's own rows are routed after the padded GEMM.

Savings vs canonical: the NCCL all-gather itself (~70 ms/chunk stage 0,
~63 ms/chunk stage 1; ~65% of all AllGather time — MIGRATION-AUDIT D1) plus
3/4 of the epilogue work (routes T rows instead of 4T). The GEMM keeps
M=4T by construction — that is the price of bitwise parity.

Default ON since R8 (user decision 1): the A/B pool gave median
−161.4 ms (3/3 sign) with every arm byte-identical to the parity build);
AllGather time −79..−84% per stage. ``DSV41_CP_LOCAL_ROUTER=0`` restores
the canonical gather path.

Prefill-only by construction: ``_v41_cp_ctx`` is set exclusively by the CP4
prefill propagation (transformer._propagate_cp_ctx); decode never binds it.
"""

import os

import torch


def cp_local_router_enabled() -> bool:
    """DSV41_CP_LOCAL_ROUTER: default ON (R8 decision 1, byte-exact via the
    padded path — parity sha == base). ``=0`` is the escape hatch
    back to the canonical all-gather; any other non-"1" value also falls
    back to canonical (conservative)."""
    return os.environ.get("DSV41_CP_LOCAL_ROUTER", "1") == "1"


def cp_local_padded_routes(gate, x, cp):
    """Drop-in replacement for ``canonical_v41_cp_routes`` (see module doc).

    Returns ``(weights, indices)`` for this rank's local rows, in local row
    order — identical layout to the canonical path's post-``index_select``
    outputs, so moe_layer chunk slicing and the EP strategy lockstep
    invariant are untouched for both flag values.
    """
    _require_supported(gate, x, cp)
    return padded_local_routes(gate, x, cp.padded_seq_len)


def padded_local_routes(gate, x, m_full):
    """Route ``x``'s own rows with the gate's local math, GEMM padded to
    ``m_full`` rows so cuBLAS picks the same kernel/split as the canonical
    full-batch GEMM (bitwise-identical logits per captured verification)."""
    t = x.shape[0]
    m_full = int(m_full)
    if m_full == t:
        return gate._forward_local(x, None)
    if m_full < t:
        raise ValueError(
            f"DSV41_CP_LOCAL_ROUTER padded M {m_full} < local rows {t}"
        )
    pad = torch.zeros((m_full, x.shape[1]), dtype=x.dtype, device=x.device)
    pad[:t] = x
    weights, indices = gate._forward_local(pad, None)
    return weights[:t].contiguous(), indices[:t].contiguous()


def _require_supported(gate, x, cp):
    """Mirror the canonical path's domain so flag=1 fails loudly on any
    configuration the verified bitwise evidence does not cover."""
    if getattr(gate, "hash", False) or not getattr(gate, "fp32_logits", False):
        raise NotImplementedError(
            "DSV41_CP_LOCAL_ROUTER requires the V4.1 FP32 non-hash gate"
        )
    if getattr(cp, "kv_cache_sharded", False) or int(cp.cp_size) != 4:
        raise NotImplementedError(
            "DSV41_CP_LOCAL_ROUTER requires eager unsharded CP4"
        )
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise NotImplementedError(
            "DSV41_CP_LOCAL_ROUTER requires contiguous CUDA BF16 x"
        )
    if x.shape != (cp.chunk_length, 5120) or cp.padded_seq_len != 4 * cp.chunk_length:
        raise ValueError("DSV41_CP_LOCAL_ROUTER geometry disagrees with its CP context")
    if torch.cuda.get_device_capability(x.device) != (12, 0):
        raise NotImplementedError("DSV41_CP_LOCAL_ROUTER is SM120-verified only")
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "DSV41_CP_LOCAL_ROUTER does not support graph capture"
        )
