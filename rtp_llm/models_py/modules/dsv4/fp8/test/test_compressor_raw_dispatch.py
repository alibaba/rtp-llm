"""UT for ``run_fused_compress_kv_write`` under the new dispatch.

After the rewrite, in-launch positions (``flat_idx >= 0``) always read
from ``kv_raw``; only prefix-cache hits (``flat_idx < 0``) read from
``state_cache`` via ``block_table``. This UT exercises that with
framework-faithful ``block_id`` / ``slot_mapping`` layouts:

  * **long prefill** (N > ``2*page_size``): the request only owns 2 state
    pool blocks, so the earliest logical block is unmapped in
    ``block_table``. Every in-launch position routes through raw — proves
    the kernel does not need state_cache to retain earlier positions.

  * **prefix-cache reuse**: launch 1 prefills positions ``[0, PAGE)``
    into phys block 1; launch 2 starts at ``sp=PAGE`` and reuses phys 1
    for its logical block 0 (the prefix-cache hit). The boundary near
    ``sp`` reads its prefix overlap from launch 1's state_cache and the
    rest from launch 2's raw kv.

  * **decode**: a single boundary token, ``disable_raw_path=True``;
    the entire overlap window comes from state_cache populated by a
    prior prefill launch.

Reference for all three: a monolithic "all-raw" launch covering every
position with an oversized state pool — captures the canonical FP8
bytes per boundary. Each scenario assembles its launches so the boundary
under test sees identical kv/score/ape data, and we assert byte-for-byte
equality of both the FP8 segment and the UE8M0/FP32 scale segment.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_raw_dispatch.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)

# --------------------------------------------------------------------------- #
# Layout constants                                                            #
# --------------------------------------------------------------------------- #
PAGE = 256  # state pool entries_per_block
COMPRESS_RATIO = 4
OVERLAP = True
COFF = 1 + int(OVERLAP)  # = 2 for ratio=4 layers

_TOKEN_STRIDE = {128: 128, 512: 576}
_SCALE_DIM = {128: 4, 512: 8}
_ROPE_DIM = {128: 128, 512: 64}


# --------------------------------------------------------------------------- #
# Tensor helpers                                                              #
# --------------------------------------------------------------------------- #
def _alloc_state_cache(head_dim: int, num_phys_blocks: int) -> torch.Tensor:
    """Phys block 0 is reserved as the kernel's ``unallocated`` sentinel
    (``block_table > 0`` is the valid check), so callers should request
    ``num_phys_blocks`` *usable* blocks; we allocate one extra at index 0."""
    width = COFF * head_dim
    return torch.zeros(
        (num_phys_blocks + 1, PAGE, 2 * width),
        dtype=torch.float32,
        device="cuda",
    )


def _alloc_kv_cache(head_dim: int, num_slots: int) -> torch.Tensor:
    """Single KV-pool block sized for ``num_slots`` boundaries.

    Layout: ``[num_slots * TOKEN_STRIDE bytes][num_slots * SCALE_DIM bytes]``,
    matching what the kernel indexes via ``KV_BLOCK_STRIDE``.
    """
    ts, sd = _TOKEN_STRIDE[head_dim], _SCALE_DIM[head_dim]
    block_bytes = num_slots * (ts + sd)
    flat = torch.zeros((1, block_bytes), dtype=torch.uint8, device="cuda")
    cache = flat.as_strided(
        size=(1, num_slots, ts),
        stride=(block_bytes, ts, 1),
    )
    cache._flat_backing = flat  # type: ignore[attr-defined]
    return cache


def _read_kv_slot(kv_cache: torch.Tensor, slot: int, head_dim: int, num_slots: int):
    """Return ``(kv_bytes, scale_bytes)`` for boundary ``slot`` in block 0."""
    ts, sd = _TOKEN_STRIDE[head_dim], _SCALE_DIM[head_dim]
    flat = kv_cache._flat_backing.view(-1)  # type: ignore[attr-defined]
    kv = flat[slot * ts : (slot + 1) * ts].clone()
    scale_base = num_slots * ts
    scale = flat[scale_base + slot * sd : scale_base + (slot + 1) * sd].clone()
    return kv, scale


# --------------------------------------------------------------------------- #
# slot_mapping / block_table builders (mirror fp8/compressor.py)              #
# --------------------------------------------------------------------------- #
def _build_state_slots(
    positions: torch.Tensor, block_table: torch.Tensor
) -> torch.Tensor:
    """``state_slots[t] = block_table[0, pos // PAGE] * PAGE + (pos % PAGE)``
    when the logical block is allocated (``block_table > 0``), else ``-1``.

    Matches ``_kv_attention_slot_mapping_impl`` in ``fp8/compressor.py`` modulo
    the boundary mask (state pool stores per-token, not per-boundary).
    """
    log = (positions // PAGE).to(torch.int64)
    in_table = log < block_table.shape[1]
    log_safe = log.clamp(max=block_table.shape[1] - 1)
    phys = block_table[0, log_safe].to(torch.int64)
    valid = in_table & (phys > 0)
    slots = phys * PAGE + (positions % PAGE)
    return torch.where(valid, slots, torch.full_like(slots, -1))


def _build_kv_slots(
    positions: torch.Tensor, kv_block_id: int, kv_block_size: int
) -> torch.Tensor:
    """Pack boundaries into a single KV-pool block. Boundary k in this
    launch → slot = ``kv_block_id * kv_block_size + k``; non-boundary → -1."""
    is_b = ((positions + 1) % COMPRESS_RATIO) == 0
    boundary_idx = torch.cumsum(is_b.to(torch.int64), dim=0) - 1
    slot = kv_block_id * kv_block_size + boundary_idx
    return torch.where(is_b, slot, torch.full_like(slot, -1))


def _global_boundary_idx(pos: int) -> int:
    return (pos + 1) // COMPRESS_RATIO - 1


# --------------------------------------------------------------------------- #
# Random data + launch                                                        #
# --------------------------------------------------------------------------- #
def _make_random(N: int, head_dim: int, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    g_ape = torch.Generator(device="cuda").manual_seed(seed + 9973)
    width = COFF * head_dim
    kv = torch.randn(N, width, dtype=torch.float32, device="cuda", generator=g) * 0.1
    score = torch.randn(N, width, dtype=torch.float32, device="cuda", generator=g) * 0.1
    ape = (
        torch.randn(
            COMPRESS_RATIO, width, dtype=torch.float32, device="cuda", generator=g_ape
        )
        * 0.1
    )
    return kv, score, ape


def _identity_cos_sin(head_dim: int, max_pos: int) -> torch.Tensor:
    rope_dim = _ROPE_DIM[head_dim]
    cs = torch.zeros(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    cs[:, : rope_dim // 2] = 1.0  # cos = 1, sin = 0 → identity rotation
    return cs


def _launch(
    *,
    kv_flat: torch.Tensor,
    score_flat: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slots: torch.Tensor,
    sp: int,
    head_dim: int,
    disable_raw: bool,
    skip_save: bool = False,
):
    rope_dim = _ROPE_DIM[head_dim]
    rms_w = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
    cos_sin = _identity_cos_sin(head_dim, int(positions.max().item()) + 16)
    token_to_req = torch.zeros(positions.shape[0], dtype=torch.int32, device="cuda")

    if not skip_save:
        run_save_partial_states(
            kv_flat,
            score_flat,
            ape,
            positions,
            state_cache,
            state_slots,
            compress_ratio=COMPRESS_RATIO,
        )
    run_fused_compress_kv_write(
        state_cache,
        token_to_req,
        positions,
        state_slots,
        state_block_table.to(torch.int32),
        rms_w,
        1e-6,
        cos_sin,
        kv_cache,
        kv_slots,
        kv_flat,
        score_flat,
        ape,
        sp,
        disable_raw_path=disable_raw,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=COMPRESS_RATIO,
        overlap=OVERLAP,
    )


def _run_reference(N_total: int, head_dim: int, *, seed: int):
    """Monolithic all-raw launch covering positions ``[0, N_total)``.
    State pool is sized to hold every position (all logical blocks mapped),
    so save_partial_states never returns -1 — but the fused writer reads
    purely from raw anyway (``flat_idx >= 0`` for all positions)."""
    n_logical = (N_total + PAGE - 1) // PAGE
    state_cache = _alloc_state_cache(head_dim, n_logical)
    bt = torch.arange(1, n_logical + 1, dtype=torch.int64, device="cuda").view(1, -1)

    positions = torch.arange(N_total, dtype=torch.int64, device="cuda")
    state_slots = _build_state_slots(positions, bt)
    n_boundaries = N_total // COMPRESS_RATIO
    kv_cache = _alloc_kv_cache(head_dim, n_boundaries)
    kv_slots = _build_kv_slots(positions, 0, n_boundaries)

    kv_flat, score_flat, ape = _make_random(N_total, head_dim, seed=seed)
    _launch(
        kv_flat=kv_flat,
        score_flat=score_flat,
        ape=ape,
        positions=positions,
        state_cache=state_cache,
        state_slots=state_slots,
        state_block_table=bt,
        kv_cache=kv_cache,
        kv_slots=kv_slots,
        sp=0,
        head_dim=head_dim,
        disable_raw=False,
    )
    return dict(
        kv_flat=kv_flat,
        score_flat=score_flat,
        ape=ape,
        kv_cache=kv_cache,
        n_boundaries=n_boundaries,
    )


def _assert_eq(
    tag: str, scenario: str, target_pos: int, ref_kv, ref_sc, got_kv, got_sc
) -> None:
    if torch.equal(ref_kv, got_kv) and torch.equal(ref_sc, got_sc):
        print(f"[{tag}] {scenario:<14s} pos={target_pos:<4d} OK")
        return
    diff_kv = (ref_kv.to(torch.int16) - got_kv.to(torch.int16)).abs().sum().item()
    diff_sc = (ref_sc.to(torch.int16) - got_sc.to(torch.int16)).abs().sum().item()
    raise AssertionError(
        f"[{tag}] {scenario}: boundary at pos={target_pos} mismatch "
        f"(diff_kv={diff_kv}, diff_scale={diff_sc})"
    )


# ============================================================================ #
# Scenario 1: long prefill — N > 2 * PAGE, only 2 state blocks per request
# ============================================================================ #
def _test_long_prefill(head_dim: int, *, tag: str) -> None:
    N = 600  # 3 logical blocks; only 2 mapped
    target_pos = 519  # boundary in logical block 2
    ref = _run_reference(N, head_dim, seed=42)

    state_cache = _alloc_state_cache(head_dim, 2)
    n_logical = (N + PAGE - 1) // PAGE  # 3
    bt = torch.zeros((1, n_logical), dtype=torch.int64, device="cuda")
    bt[0, n_logical - 2] = 1  # most-recent 2 logical blocks
    bt[0, n_logical - 1] = 2  # → phys 1, 2; logical 0 unmapped (= 0)

    positions = torch.arange(N, dtype=torch.int64, device="cuda")
    state_slots = _build_state_slots(positions, bt)
    n_boundaries = N // COMPRESS_RATIO
    kv_cache = _alloc_kv_cache(head_dim, n_boundaries)
    kv_slots = _build_kv_slots(positions, 0, n_boundaries)

    _launch(
        kv_flat=ref["kv_flat"],
        score_flat=ref["score_flat"],
        ape=ref["ape"],
        positions=positions,
        state_cache=state_cache,
        state_slots=state_slots,
        state_block_table=bt,
        kv_cache=kv_cache,
        kv_slots=kv_slots,
        sp=0,
        head_dim=head_dim,
        disable_raw=False,
    )

    bidx = _global_boundary_idx(target_pos)
    ref_kv, ref_sc = _read_kv_slot(ref["kv_cache"], bidx, head_dim, ref["n_boundaries"])
    got_kv, got_sc = _read_kv_slot(kv_cache, bidx, head_dim, n_boundaries)
    _assert_eq(tag, "long_prefill", target_pos, ref_kv, ref_sc, got_kv, got_sc)


# ============================================================================ #
# Scenario 2: prefix-cache reuse — launch 2 inherits launch 1's state pool
# ============================================================================ #
def _test_prefix_reuse(head_dim: int, *, tag: str) -> None:
    N1 = PAGE  # launch 1 fills logical block 0
    N2 = 8  # launch 2 prefills 8 new tokens
    target_pos = PAGE + COMPRESS_RATIO - 1  # 259, boundary in launch 2
    N_total = N1 + N2

    ref = _run_reference(N_total, head_dim, seed=7)

    # Shared state pool: phys 0 sentinel, phys 1 (logical 0), phys 2 (logical 1).
    state_cache = _alloc_state_cache(head_dim, 2)

    # ── Launch 1 ──
    bt1 = torch.tensor([[1]], dtype=torch.int64, device="cuda")
    pos1 = torch.arange(0, N1, dtype=torch.int64, device="cuda")
    slots1 = _build_state_slots(pos1, bt1)
    n_b1 = N1 // COMPRESS_RATIO
    kv_cache_1 = _alloc_kv_cache(head_dim, n_b1)
    kv_slots_1 = _build_kv_slots(pos1, 0, n_b1)
    _launch(
        kv_flat=ref["kv_flat"][:N1],
        score_flat=ref["score_flat"][:N1],
        ape=ref["ape"],
        positions=pos1,
        state_cache=state_cache,
        state_slots=slots1,
        state_block_table=bt1,
        kv_cache=kv_cache_1,
        kv_slots=kv_slots_1,
        sp=0,
        head_dim=head_dim,
        disable_raw=False,
    )

    # ── Launch 2: logical 0 reuses phys 1 (prefix hit), logical 1 → phys 2 ──
    bt2 = torch.tensor([[1, 2]], dtype=torch.int64, device="cuda")
    pos2 = torch.arange(N1, N1 + N2, dtype=torch.int64, device="cuda")
    slots2 = _build_state_slots(pos2, bt2)
    n_b2 = N2 // COMPRESS_RATIO
    kv_cache_2 = _alloc_kv_cache(head_dim, n_b2)
    kv_slots_2 = _build_kv_slots(pos2, 0, n_b2)
    _launch(
        kv_flat=ref["kv_flat"][N1:],
        score_flat=ref["score_flat"][N1:],
        ape=ref["ape"],
        positions=pos2,
        state_cache=state_cache,
        state_slots=slots2,
        state_block_table=bt2,
        kv_cache=kv_cache_2,
        kv_slots=kv_slots_2,
        sp=N1,
        head_dim=head_dim,
        disable_raw=False,
    )

    bidx_global = _global_boundary_idx(target_pos)
    bidx_local = bidx_global - n_b1
    ref_kv, ref_sc = _read_kv_slot(
        ref["kv_cache"], bidx_global, head_dim, ref["n_boundaries"]
    )
    got_kv, got_sc = _read_kv_slot(kv_cache_2, bidx_local, head_dim, n_b2)
    _assert_eq(tag, "prefix_reuse", target_pos, ref_kv, ref_sc, got_kv, got_sc)


# ============================================================================ #
# Scenario 3: decode — disable_raw_path; entire overlap window from cache
# ============================================================================ #
def _test_decode(head_dim: int, *, tag: str) -> None:
    """Pre-fill state_cache via a prefill launch covering ``[0, target_pos+1)``
    so all overlap positions live in the pool, then "decode" the boundary
    token at ``target_pos`` with N=1 and ``disable_raw_path=True`` — every
    overlap-window read must come from the cache."""
    target_pos = COMPRESS_RATIO * 70 - 1  # 279, in logical block 1
    N_pre = target_pos + 1  # 280
    ref = _run_reference(N_pre, head_dim, seed=11)

    state_cache = _alloc_state_cache(head_dim, 2)
    n_logical = (N_pre + PAGE - 1) // PAGE  # 2
    bt = torch.tensor([[1, 2]], dtype=torch.int64, device="cuda")

    # ── Prefill: write every position [0, target_pos+1) into state_cache. ──
    pos_pre = torch.arange(0, N_pre, dtype=torch.int64, device="cuda")
    slots_pre = _build_state_slots(pos_pre, bt)
    n_b_pre = N_pre // COMPRESS_RATIO
    kv_cache_pre = _alloc_kv_cache(head_dim, n_b_pre)
    kv_slots_pre = _build_kv_slots(pos_pre, 0, n_b_pre)
    _launch(
        kv_flat=ref["kv_flat"][:N_pre],
        score_flat=ref["score_flat"][:N_pre],
        ape=ref["ape"],
        positions=pos_pre,
        state_cache=state_cache,
        state_slots=slots_pre,
        state_block_table=bt,
        kv_cache=kv_cache_pre,
        kv_slots=kv_slots_pre,
        sp=0,
        head_dim=head_dim,
        disable_raw=False,
    )

    # ── Decode: N=1, disable_raw=True. kv_flat is a dummy (won't be read). ──
    pos_dec = torch.tensor([target_pos], dtype=torch.int64, device="cuda")
    slots_dec = _build_state_slots(pos_dec, bt)
    kv_cache_dec = _alloc_kv_cache(head_dim, 1)
    kv_slots_dec = torch.tensor([0], dtype=torch.int64, device="cuda")
    width = COFF * head_dim
    dummy = torch.zeros(1, width, dtype=torch.float32, device="cuda")
    _launch(
        kv_flat=dummy,
        score_flat=dummy,
        ape=ref["ape"],
        positions=pos_dec,
        state_cache=state_cache,
        state_slots=slots_dec,
        state_block_table=bt,
        kv_cache=kv_cache_dec,
        kv_slots=kv_slots_dec,
        sp=N_pre,
        head_dim=head_dim,
        disable_raw=True,
        skip_save=True,
    )

    bidx = _global_boundary_idx(target_pos)
    ref_kv, ref_sc = _read_kv_slot(ref["kv_cache"], bidx, head_dim, ref["n_boundaries"])
    got_kv, got_sc = _read_kv_slot(kv_cache_dec, 0, head_dim, 1)
    _assert_eq(tag, "decode", target_pos, ref_kv, ref_sc, got_kv, got_sc)


# ============================================================================ #
# Scenario 4: missing state blocks — out-of-table cache reads are skipped
# ============================================================================ #
def _test_missing_state_blocks_skip_write(head_dim: int, *, tag: str) -> None:
    """Boundary writer must not read past block_table when the logical
    state block is unavailable. With no valid raw/cache source, it should
    leave the destination KV slot untouched."""
    target_pos = 515  # boundary; overlap window straddles logical blocks 1/2
    state_cache = _alloc_state_cache(head_dim, 1)
    bt = torch.tensor([[0, 0]], dtype=torch.int64, device="cuda")
    pos_dec = torch.tensor([target_pos], dtype=torch.int64, device="cuda")
    state_slots = _build_state_slots(pos_dec, bt)
    kv_cache = _alloc_kv_cache(head_dim, 1)
    kv_slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    width = COFF * head_dim
    dummy = torch.zeros(1, width, dtype=torch.float32, device="cuda")
    ape = torch.zeros(COMPRESS_RATIO, width, dtype=torch.float32, device="cuda")

    _launch(
        kv_flat=dummy,
        score_flat=dummy,
        ape=ape,
        positions=pos_dec,
        state_cache=state_cache,
        state_slots=state_slots,
        state_block_table=bt,
        kv_cache=kv_cache,
        kv_slots=kv_slots,
        sp=target_pos + 1,
        head_dim=head_dim,
        disable_raw=True,
        skip_save=True,
    )

    got_kv, got_sc = _read_kv_slot(kv_cache, 0, head_dim, 1)
    if int(got_kv.sum().item()) != 0 or int(got_sc.sum().item()) != 0:
        raise AssertionError(f"[{tag}] missing_state_blocks should not write slot 0")
    print(f"[{tag}] missing_state_blocks pos={target_pos:<4d} OK")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    for hd in (128, 512):
        tag = "indexer" if hd == 128 else "sparse "
        _test_long_prefill(hd, tag=tag)
        _test_prefix_reuse(hd, tag=tag)
        _test_decode(hd, tag=tag)
        _test_missing_state_blocks_skip_write(hd, tag=tag)
    print("OK")


if __name__ == "__main__":
    main()
