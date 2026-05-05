"""PP4 wireup UT for Compressor → fused {pool+RMSNorm+RoPE+FP8 quant+scatter}.

Builds a real ``Compressor`` instance with random weights, attaches a
uint8 / 132B-per-slot FP8 indexer pool view via ``set_pool_context``, then
exercises both:

  * ``_forward_prefill_body`` (overlap=True, ratio=4, seqlen=64, NB=16),
  * ``forward_decode_vectorized`` at a boundary step (sp=3, ratio=4),

with ``DSV4_COMPRESSOR_FUSED=1`` and verifies the FP8-packed pool slots
round-trip (via ``dequantize_indexer_k``) to within rel_max < 0.15 of the
bf16 reference path (the same Compressor with the env OFF, writing to a
plain bf16 cache).  rel_max < 0.15 brackets the FP8 e4m3 quant noise
floor (matches ``test_v4_compressor_fused.test_round_trip_via_dequant``).

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_fused_wireup.py
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    dequantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4.compressor import Compressor

DEV = "cuda"
HEAD_DIM = INDEXER_HEAD_DIM  # 128
ROPE_HEAD_DIM = 64
RATIO = 4  # CSA / overlap
EPS = 1e-6


@contextmanager
def _env(name: str, value: str):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _build_compressor(dim: int = 256) -> Compressor:
    g = torch.Generator(device=DEV).manual_seed(7)
    coff = 2  # overlap
    weights = {
        "C.ape": torch.randn(
            RATIO, coff * HEAD_DIM, dtype=torch.float32, device=DEV, generator=g
        ),
        "C.wkv.weight": torch.randn(
            coff * HEAD_DIM, dim, dtype=torch.float32, device=DEV, generator=g
        )
        * 0.1,
        "C.wgate.weight": torch.randn(
            coff * HEAD_DIM, dim, dtype=torch.float32, device=DEV, generator=g
        )
        * 0.1,
        "C.norm.weight": (
            torch.randn(HEAD_DIM, dtype=torch.float32, device=DEV, generator=g) * 0.3
            + 1.0
        ).to(torch.bfloat16),
    }
    c = Compressor(
        dim=dim,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        compress_ratio=RATIO,
        max_batch_size=1,
        norm_eps=EPS,
        weights=weights,
        prefix="C",
    ).to(DEV)
    # freqs_cis: complex64 for prefill RoPE.
    seqlen = 4096
    angles = (
        torch.linspace(
            0, 2 * math.pi, ROPE_HEAD_DIM // 2, device=DEV, dtype=torch.float32
        )
        .unsqueeze(0)
        .expand(seqlen, -1)
    )
    angles = angles * (
        1.0 + 0.001 * torch.arange(seqlen, device=DEV, dtype=torch.float32).unsqueeze(1)
    )
    c.freqs_cis = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
    c.configure_kv_cache_shape(seqlen // RATIO)
    return c


def _attach_fp8_pool(
    c: Compressor, max_blocks: int = 32, eb: int = INDEXER_ENTRY_BYTES
):
    """Attach a fresh uint8/132B FP8 pool view + block_table."""
    pool = torch.zeros(
        max_blocks * eb, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=DEV
    )
    # Block table for B=1 — block_id 0 is reserved as 'null' by
    # _compute_pool_slots (treated as invalid). Use 1..max_blocks.
    bt = torch.arange(1, max_blocks + 1, device=DEV, dtype=torch.int32).unsqueeze(0)
    # State pool: not exercised in these UTs (we use is_fresh_prefill=True for
    # prefill to bypass; decode runs a single boundary step starting from
    # zeroed state). So pass a tiny dummy state pool.
    state_pool = torch.zeros(
        eb * max_blocks, 2 * 2 * HEAD_DIM, dtype=torch.float32, device=DEV
    )
    c.set_pool_context(
        kv_pool_view=pool,
        kv_block_table=bt,
        kv_eb=eb,
        state_pool_view=state_pool,
        state_block_table=bt,
        state_eb=eb,
    )
    return pool, bt, eb


def _detach_pool(c: Compressor):
    c.clear_pool_context()


def _gather_pool_slots(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Dequantize a list of pool-flat slots to fp32 [N, 128]."""
    pool_blocks = pool.view(-1, 1, INDEXER_ENTRY_BYTES)
    return dequantize_indexer_k(pool_blocks, slots, out_dtype=torch.float32)


def test_prefill_fused_matches_bf16():
    print("== prefill: fused vs bf16 ref ==")
    c = _build_compressor()
    seqlen = 64
    NB = seqlen // RATIO  # 16
    g = torch.Generator(device=DEV).manual_seed(101)
    x = torch.randn(1, seqlen, c.dim, dtype=torch.bfloat16, device=DEV, generator=g)

    # ── Reference: bf16 path (env OFF, no FP8 pool) ──
    pool_ref, bt_ref, eb = _attach_fp8_pool(c)
    # Force bf16 path: detach FP8 pool, run with kv_cache materialised by
    # base class (we'll capture self.kv_cache before scatter via a hook).
    # Simpler: just call with env OFF — the base path writes to
    # self.kv_cache (bf16) before _scatter_kv_cache_to_pool packs back into
    # the uint8 pool. The pool is then FP8-quantised by the framework
    # (quantize_indexer_k via write_kv_to_pool). To get a *bf16* reference
    # without FP8 quant in the way, swap the pool to a bf16-compatible
    # surrogate (kv_eb=0 → base class allocates a fresh bf16 cache and the
    # scatter is a no-op).
    c.set_pool_context(
        kv_pool_view=None,
        kv_block_table=None,
        kv_eb=0,
        state_pool_view=None,
        state_block_table=None,
        state_eb=0,
    )
    with _env("DSV4_COMPRESSOR_FUSED", "0"):
        c._dbg_prefix = None
        # Capture the bf16 kv before scatter by patching the helper.
        captured = {}
        orig_scatter = c._scatter_kv_cache_to_pool

        def _spy(bsz, block_mask=None):
            if c.kv_cache is not None:
                captured["kv"] = c.kv_cache[:bsz].detach().clone()
            orig_scatter(bsz, block_mask)

        c._scatter_kv_cache_to_pool = _spy  # type: ignore[assignment]
        c.forward(x, 0)
        c._scatter_kv_cache_to_pool = orig_scatter  # type: ignore[assignment]
    ref_kv_bf16 = captured["kv"]  # [1, T, 128]
    write_start = 0
    ref_compressed = ref_kv_bf16[0, write_start : write_start + NB].float()
    print(
        f"  ref shape: {tuple(ref_compressed.shape)}, "
        f"abs_max={ref_compressed.abs().max().item():.4f}"
    )

    # ── Fused path: env ON, FP8 pool attached, fresh state ──
    c2 = _build_compressor()
    # Mirror weights from c so we compare apples-to-apples.
    with torch.no_grad():
        c2.ape.copy_(c.ape)
        c2.wkv.weight.copy_(c.wkv.weight)
        c2.wgate.weight.copy_(c.wgate.weight)
        c2.norm.weight.copy_(c.norm.weight)
    c2.freqs_cis = c.freqs_cis
    pool_fused, bt_fused, eb = _attach_fp8_pool(c2)
    with _env("DSV4_COMPRESSOR_FUSED", "1"):
        c2.forward(x, 0)
    # Read back: per-position p in [0, NB), logical slot p, batch 0 → bt[0, p//eb]*eb + p%eb.
    bt_long = bt_fused.to(torch.long)
    pos = torch.arange(NB, device=DEV, dtype=torch.long)
    slots = bt_long[0, pos // eb] * eb + (pos % eb)
    deq = _gather_pool_slots(pool_fused, slots)
    print(
        f"  fused deq shape: {tuple(deq.shape)}, "
        f"abs_max={deq.abs().max().item():.4f}"
    )

    diff = (deq - ref_compressed).abs()
    ref_max = ref_compressed.abs().max().item()
    rel_max = diff.max().item() / (ref_max + 1e-6)
    rel_mean = diff.mean().item() / (ref_max + 1e-6)
    print(f"  rel_max={rel_max:.4f}  rel_mean={rel_mean:.4f}  ref_max={ref_max:.4f}")
    assert rel_max < 0.15, (
        f"prefill fused vs bf16: rel_max {rel_max:.4f} exceeds FP8 noise "
        f"floor (0.15)"
    )
    print("  OK")


def test_decode_boundary_fused_matches_bf16():
    print("== decode boundary: fused vs bf16 ref ==")
    # CSA decode at sp=3 (ratio=4) is the last token of the first window —
    # boundary, writes pool slot 0 (cache_logical = sp//ratio = 0).
    c = _build_compressor()
    g = torch.Generator(device=DEV).manual_seed(202)
    x = torch.randn(1, 1, c.dim, dtype=torch.bfloat16, device=DEV, generator=g)
    sp = torch.tensor([3], dtype=torch.int32, device=DEV)

    # Bf16 reference: NO pool attached; persistent state across calls via
    # monkey-patch.
    c.set_pool_context(
        kv_pool_view=None,
        kv_block_table=None,
        kv_eb=0,
        state_pool_view=None,
        state_block_table=None,
        state_eb=0,
    )
    # Pre-fill state: simulate prior 3 tokens via _bind_state_from_pool with
    # is_fresh_prefill=True (zero kv_state, -inf score_state). For the decode
    # to cover all G slots meaningfully we need score_state to NOT be all
    # -inf. Pre-populate by running 3 prior decodes with bf16 path; then
    # snapshot+replay the state into the fused-path Compressor.

    # State persists in self.kv_state / score_state ONLY across calls when
    # bound from pool. With null pool the base class re-zeroes every call.
    # Workaround: monkey-patch _bind_state_from_pool to keep the prior buf.
    persistent = {"kv": None, "sc": None}

    orig_bind_state = c._bind_state_from_pool
    orig_scatter_state = c._scatter_state_to_pool
    orig_bind_kv = c._bind_kv_cache_from_pool
    orig_scatter_kv = c._scatter_kv_cache_to_pool

    def _bs(bsz, is_fresh_prefill, device):
        if persistent["kv"] is None:
            orig_bind_state(bsz, True, device)
            persistent["kv"] = c.kv_state
            persistent["sc"] = c.score_state
        else:
            c.kv_state = persistent["kv"]
            c.score_state = persistent["sc"]

    def _ss(bsz):
        persistent["kv"] = c.kv_state
        persistent["sc"] = c.score_state

    persistent_kv = {"buf": None}

    def _bk(bsz, is_fresh_prefill, device, dtype):
        if persistent_kv["buf"] is None:
            persistent_kv["buf"] = torch.zeros(
                bsz, c._kv_cache_t, c._kv_cache_d, dtype=dtype, device=device
            )
        c.kv_cache = persistent_kv["buf"]

    def _sk(bsz, block_mask=None):
        persistent_kv["buf"] = c.kv_cache

    c._bind_state_from_pool = _bs  # type: ignore[assignment]
    c._scatter_state_to_pool = _ss  # type: ignore[assignment]
    c._bind_kv_cache_from_pool = _bk  # type: ignore[assignment]
    c._scatter_kv_cache_to_pool = _sk  # type: ignore[assignment]

    with _env("DSV4_COMPRESSOR_FUSED", "0"):
        for sp_prev in range(3):
            x_prev = torch.randn(
                1, 1, c.dim, dtype=torch.bfloat16, device=DEV, generator=g
            )
            c.forward_decode_vectorized(
                x_prev, torch.tensor([sp_prev], dtype=torch.int32, device=DEV)
            )
        c.forward_decode_vectorized(x, sp)
    ref_compressed = persistent_kv["buf"][0, 0].float().clone()  # bf16 cache slot 0
    print(f"  ref abs_max={ref_compressed.abs().max().item():.4f}")

    # Snapshot kv_state / score_state at the moment AFTER the 3 prior
    # decodes (i.e., before the boundary call). We need to re-derive: the
    # state at sp=3 right before the call has 3 tokens written into slots
    # ratio + sp_mod for sp_mod in 0..2. Replay the same priors against
    # c2 to seed state, then run the boundary step under fused.
    c2 = _build_compressor()
    with torch.no_grad():
        c2.ape.copy_(c.ape)
        c2.wkv.weight.copy_(c.wkv.weight)
        c2.wgate.weight.copy_(c.wgate.weight)
        c2.norm.weight.copy_(c.norm.weight)
    c2.freqs_cis = c.freqs_cis

    # Re-seed RNG so the same x_prev sequence is generated for c2's priors.
    g2 = torch.Generator(device=DEV).manual_seed(202)
    _ = torch.randn(1, 1, c2.dim, dtype=torch.bfloat16, device=DEV, generator=g2)  # x

    # No pool during priors — bf16 path with monkey-patched persistence.
    c2.set_pool_context(
        kv_pool_view=None,
        kv_block_table=None,
        kv_eb=0,
        state_pool_view=None,
        state_block_table=None,
        state_eb=0,
    )
    persistent2 = {"kv": None, "sc": None}
    orig_bind2 = c2._bind_state_from_pool

    def _bs2(bsz, is_fresh_prefill, device):
        if persistent2["kv"] is None:
            orig_bind2(bsz, True, device)
            persistent2["kv"] = c2.kv_state
            persistent2["sc"] = c2.score_state
        else:
            c2.kv_state = persistent2["kv"]
            c2.score_state = persistent2["sc"]

    def _ss2(bsz):
        persistent2["kv"] = c2.kv_state
        persistent2["sc"] = c2.score_state

    persistent_kv2 = {"buf": None}

    def _bk2(bsz, is_fresh_prefill, device, dtype):
        if persistent_kv2["buf"] is None:
            persistent_kv2["buf"] = torch.zeros(
                bsz, c2._kv_cache_t, c2._kv_cache_d, dtype=dtype, device=device
            )
        c2.kv_cache = persistent_kv2["buf"]

    def _sk2(bsz, block_mask=None):
        persistent_kv2["buf"] = c2.kv_cache

    c2._bind_state_from_pool = _bs2  # type: ignore[assignment]
    c2._scatter_state_to_pool = _ss2  # type: ignore[assignment]
    c2._bind_kv_cache_from_pool = _bk2  # type: ignore[assignment]
    c2._scatter_kv_cache_to_pool = _sk2  # type: ignore[assignment]

    with _env("DSV4_COMPRESSOR_FUSED", "0"):
        for sp_prev in range(3):
            x_prev = torch.randn(
                1, 1, c2.dim, dtype=torch.bfloat16, device=DEV, generator=g2
            )
            c2.forward_decode_vectorized(
                x_prev, torch.tensor([sp_prev], dtype=torch.int32, device=DEV)
            )
    # Restore real bind helpers, attach FP8 pool, run boundary under fused.
    # State buffer needs to be passed through the FP8 path's bind: re-route
    # _bind_state_from_pool to seed from persistent2.
    c2._bind_kv_cache_from_pool = type(c2)._bind_kv_cache_from_pool.__get__(c2)
    c2._scatter_kv_cache_to_pool = type(c2)._scatter_kv_cache_to_pool.__get__(c2)
    pool_fused, bt_fused, eb = _attach_fp8_pool(c2)
    with _env("DSV4_COMPRESSOR_FUSED", "1"):
        c2.forward_decode_vectorized(x, sp)

    # Pool slot for cache_logical=0, batch=0, eb=eb → bt[0,0]*eb + 0
    slot = (bt_fused.to(torch.long)[0, 0] * eb + 0).view(1)
    deq = _gather_pool_slots(pool_fused, slot)[0].float()
    print(f"  fused deq abs_max={deq.abs().max().item():.4f}")

    diff = (deq - ref_compressed).abs()
    ref_max = ref_compressed.abs().max().item()
    rel_max = diff.max().item() / (ref_max + 1e-6)
    rel_mean = diff.mean().item() / (ref_max + 1e-6)
    print(f"  rel_max={rel_max:.4f}  rel_mean={rel_mean:.4f}  ref_max={ref_max:.4f}")
    assert (
        rel_max < 0.15
    ), f"decode fused vs bf16: rel_max {rel_max:.4f} exceeds FP8 noise floor"
    print("  OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_prefill_fused_matches_bf16()
    test_decode_boundary_fused_matches_bf16()
    print("\nALL OK")
