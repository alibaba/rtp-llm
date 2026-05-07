"""UT for ``run_fused_compress_kv_write`` raw-vs-cache dispatch.

The state pool is a 2-block * 256-entry = 512-slot **cyclic** ring per request.
For a prefill launch with ``N > 512``, ``_save_partial_states`` overwrites the
earliest ``N - 512`` slots within the same launch, so the fused kernel cannot
read those overlap-window positions from the state cache. The dispatch logic
inside the kernel is::

    flat_idx  = pos - seq_start
    use_raw   = (0 <= flat_idx < n_batch) & (flat_idx < n_batch - CACHE_WINDOW)
    use_cache = mask_pos & ~use_raw

with ``CACHE_WINDOW == 512``.

This UT verifies the dispatch via a **differential-against-ground-truth** check:

  * Ground truth: run the kernel on a tiny launch (N == 8 << CACHE_WINDOW) so
    every overlap-window read is unambiguously cache-correct. Capture the FP8
    bytes produced for the boundary at ``pos == COMPRESS_RATIO - 1``.

  * Full launch (N == 600 > CACHE_WINDOW): the same boundary's overlap window
    [0, 7] now sits inside the overwritten prefix, so reading from the cache
    would yield wrong (cyclically-overwritten) data. The kernel must take the
    raw path and produce the **same** FP8 bytes as the ground truth.

  * Negative control: re-run the full launch with ``disable_raw_path=True`` and
    confirm the boundary's bytes **diverge** from the ground truth (the cache
    read indeed sources stale data). This proves the raw path is what saves us.

Two scenarios are exercised:

  * ``head_dim=128`` indexer kernel (overlap=True, ratio=4)
  * ``head_dim=512`` sparse-attn kernel (overlap=True, ratio=4) — this is the
    CSA layer's compressor; HCA (overlap=False, ratio=128) shares the same
    sparse-attn kernel + dispatch path so it's covered by the same logic.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_raw_dispatch.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)

# --------------------------------------------------------------------------- #
# Constants matching DSV4ConfigCreator + indexer/CSA layout.                  #
# --------------------------------------------------------------------------- #
STATE_BLOCK_SIZE = 256  # entries_per_block for state pools
FIXED_BLOCKS_PER_REQ = 2  # 2 * 256 = 512-slot cyclic ring
CACHE_WINDOW = STATE_BLOCK_SIZE * FIXED_BLOCKS_PER_REQ  # 512

COMPRESS_RATIO = 4
OVERLAP = True
COFF = 1 + int(OVERLAP)  # = 2 for ratio=4 layers


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _alloc_state_cache(head_dim: int, num_blocks: int) -> torch.Tensor:
    """Per-token (kv | score+ape) fp32 state cache. The kernel treats
    ``block_table[req, i] == 0`` as ``unallocated``, so we allocate one
    extra physical block at index 0 (unused) and have the block_table
    point to physical blocks 1..num_blocks."""
    width = COFF * head_dim
    return torch.zeros(
        (num_blocks + 1, STATE_BLOCK_SIZE, 2 * width),
        dtype=torch.float32,
        device="cuda",
    )


def _alloc_kv_cache(head_dim: int, num_slots: int) -> torch.Tensor:
    """Single contiguous KV pool block sized to fit ``num_slots`` boundaries.

    Layout per block: ``[num_slots, TOKEN_STRIDE]`` packed FP8/RoPE bytes,
    then ``[num_slots, SCALE_DIM]`` UE8M0 scales — matches what the kernel
    indexes (``cache_block_ptr + slot * TOKEN_STRIDE`` for KV and
    ``cache_block_ptr + num_slots * TOKEN_STRIDE + slot * SCALE_DIM`` for
    scales).
    """
    if head_dim == 128:
        token_stride, scale_dim = 128, 4
    elif head_dim == 512:
        token_stride, scale_dim = 576, 8
    else:
        raise ValueError(head_dim)
    block_bytes = num_slots * token_stride + num_slots * scale_dim
    # shape[1] must equal num_slots; stride(0) = block_bytes is what the
    # kernel reads as ``KV_BLOCK_STRIDE``.
    cache = torch.zeros(
        (1, num_slots, block_bytes // num_slots), dtype=torch.uint8, device="cuda"
    )
    # The above may not reproduce exact bytes-per-block if scale_dim doesn't
    # divide evenly. Build it explicitly via as_strided.
    flat = torch.zeros((1, block_bytes), dtype=torch.uint8, device="cuda")
    cache = flat.as_strided(
        size=(1, num_slots, token_stride), stride=(block_bytes, token_stride, 1)
    )
    # Keep the underlying flat alive by stashing it on the view.
    cache._flat_backing = flat  # type: ignore[attr-defined]
    return cache


def _build_block_table(num_blocks: int) -> torch.Tensor:
    """Single-request block table mapping logical block i → physical
    block (i + 1). Physical block 0 is reserved as the kernel's
    ``unallocated`` sentinel (block_table value > 0 is the valid check)."""
    return (torch.arange(num_blocks, dtype=torch.int32, device="cuda") + 1).view(
        1, num_blocks
    )


def _build_kv_block_table(num_blocks: int) -> torch.Tensor:
    return torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        1, num_blocks
    )


def _build_meta(N: int, sp: int, num_state_blocks: int):
    """Construct positions / token_to_req / state_slots / kv_slots for a
    single-request launch starting at absolute position ``sp``."""
    positions = torch.arange(sp, sp + N, dtype=torch.int64, device="cuda")
    token_to_req = torch.zeros(N, dtype=torch.int32, device="cuda")

    # State slot — cyclic over (num_state_blocks * STATE_BLOCK_SIZE) ring.
    # Physical block 0 in state_cache is the unallocated-sentinel reservoir
    # (block_table starts at physical id 1), so slot offsets are biased by
    # one block worth of entries.
    ring_size = num_state_blocks * STATE_BLOCK_SIZE
    state_slots = ((positions % ring_size) + STATE_BLOCK_SIZE).to(torch.int64)

    # KV slot: one per boundary (where (pos+1) % ratio == 0); -1 otherwise.
    is_boundary = ((positions + 1) % COMPRESS_RATIO) == 0
    boundary_idx = torch.cumsum(is_boundary.to(torch.int64), dim=0) - 1
    kv_slots = torch.where(is_boundary, boundary_idx, torch.full_like(boundary_idx, -1))
    return positions, token_to_req, state_slots, kv_slots


def _make_inputs(N: int, sp: int, head_dim: int, *, seed: int):
    """Build a self-contained launch context."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    # Raw kv/score: [N, COFF*head_dim] fp32. Use small magnitudes so
    # softmax+RMSNorm produce numerically stable results.
    width = COFF * head_dim
    kv_flat = (
        torch.randn(N, width, dtype=torch.float32, device="cuda", generator=g) * 0.1
    )
    score_flat = (
        torch.randn(N, width, dtype=torch.float32, device="cuda", generator=g) * 0.1
    )
    ape = (
        torch.randn(
            COMPRESS_RATIO, width, dtype=torch.float32, device="cuda", generator=g
        )
        * 0.1
    )

    rms_w = torch.ones(
        head_dim, dtype=torch.bfloat16, device="cuda"
    )  # identity-ish norm
    rope_head_dim = head_dim if head_dim == 128 else 64
    max_pos = sp + N + 16
    cos_sin = torch.zeros(max_pos, rope_head_dim, dtype=torch.float32, device="cuda")
    cos_sin[:, : rope_head_dim // 2] = 1.0  # cos = 1 (identity rotation)
    # sin = 0 (default)

    # State pool sized for 2 blocks per request.
    state_cache = _alloc_state_cache(head_dim, FIXED_BLOCKS_PER_REQ)
    state_block_table = _build_block_table(FIXED_BLOCKS_PER_REQ)

    # KV pool sized for all boundaries in this launch (single block).
    n_boundaries = ((sp + N) // COMPRESS_RATIO) - (sp // COMPRESS_RATIO)
    n_boundaries = max(n_boundaries, 1)
    kv_cache = _alloc_kv_cache(head_dim, n_boundaries)

    positions, token_to_req, state_slots, kv_slots = _build_meta(
        N, sp, FIXED_BLOCKS_PER_REQ
    )

    return dict(
        kv_flat=kv_flat,
        score_flat=score_flat,
        ape=ape,
        rms_w=rms_w,
        cos_sin=cos_sin,
        state_cache=state_cache,
        state_block_table=state_block_table,
        kv_cache=kv_cache,
        positions=positions,
        token_to_req=token_to_req,
        state_slots=state_slots,
        kv_slots=kv_slots,
        rope_head_dim=rope_head_dim,
    )


def _launch(ctx: dict, head_dim: int, *, disable_raw_path: bool):
    run_save_partial_states(
        ctx["kv_flat"],
        ctx["score_flat"],
        ctx["ape"],
        ctx["positions"],
        ctx["state_cache"],
        ctx["state_slots"],
        compress_ratio=COMPRESS_RATIO,
    )
    run_fused_compress_kv_write(
        ctx["state_cache"],
        ctx["token_to_req"],
        ctx["positions"],
        ctx["state_slots"],
        ctx["state_block_table"],
        ctx["rms_w"],
        1e-6,
        ctx["cos_sin"],
        ctx["kv_cache"],
        ctx["kv_slots"],
        ctx["kv_flat"],
        ctx["score_flat"],
        ctx["ape"],
        0,  # seq_start (will be overridden below for the full launch)
        disable_raw_path=disable_raw_path,
        head_dim=head_dim,
        rope_head_dim=ctx["rope_head_dim"],
        compress_ratio=COMPRESS_RATIO,
        overlap=OVERLAP,
    )


def _kv_slot_bytes(kv_cache: torch.Tensor, slot: int) -> torch.Tensor:
    """Return the FP8/RoPE bytes for a single boundary slot in block 0."""
    # kv_cache view is (1, num_slots, token_stride) but slot bytes also
    # depend on the trailing scale_dim region; we compare both KV bytes and
    # the scale region by reading the underlying flat backing.
    return kv_cache._flat_backing.clone()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Differential test                                                            #
# --------------------------------------------------------------------------- #
def _run_one(head_dim: int, *, tag: str):
    sp = 0
    target_pos = COMPRESS_RATIO - 1  # earliest boundary; overlap = [0..7]
    big_N = CACHE_WINDOW + 100  # 612, > 512 so target_pos is overwritten

    # ---------- Ground truth (small launch, no overwrite) ----------
    gt = _make_inputs(N=COMPRESS_RATIO * 2, sp=sp, head_dim=head_dim, seed=0)
    _launch(gt, head_dim, disable_raw_path=True)  # raw path off, pure cache
    gt_bytes = _kv_slot_bytes(gt["kv_cache"], slot=0).clone()

    # ---------- Full launch with raw enabled ----------
    full = _make_inputs(N=big_N, sp=sp, head_dim=head_dim, seed=0)
    # Mirror gt's first ``COMPRESS_RATIO * 2`` raw rows so the boundary at
    # target_pos has the same input data.
    full["kv_flat"][: COMPRESS_RATIO * 2].copy_(gt["kv_flat"])
    full["score_flat"][: COMPRESS_RATIO * 2].copy_(gt["score_flat"])
    # ape/rms_w/cos_sin already match (deterministic seed + same shapes).
    _launch(full, head_dim, disable_raw_path=False)
    raw_on_bytes = _kv_slot_bytes(full["kv_cache"], slot=0).clone()

    # ---------- Full launch with raw disabled (negative control) ----------
    full_neg = _make_inputs(N=big_N, sp=sp, head_dim=head_dim, seed=0)
    full_neg["kv_flat"][: COMPRESS_RATIO * 2].copy_(gt["kv_flat"])
    full_neg["score_flat"][: COMPRESS_RATIO * 2].copy_(gt["score_flat"])
    _launch(full_neg, head_dim, disable_raw_path=True)
    raw_off_bytes = _kv_slot_bytes(full_neg["kv_cache"], slot=0).clone()

    # Slot 0 in the KV pool corresponds to boundary at target_pos.
    # KV bytes for slot 0 sit at byte offset 0, length token_stride. Scale
    # bytes sit at ``num_slots * token_stride`` (different in gt vs full
    # because the launches have different boundary counts), so compare each
    # per-slot region using its own num_slots.
    token_stride = 128 if head_dim == 128 else 576
    scale_dim = 4 if head_dim == 128 else 8

    def _kv_and_scale(buf, num_slots):
        # buf is the (1, block_bytes) flat backing.
        flat = buf.view(-1)
        kv = flat[:token_stride]
        sstart = num_slots * token_stride
        scale = flat[sstart : sstart + scale_dim]
        return kv, scale

    gt_kv, gt_scale = _kv_and_scale(gt_bytes, gt["kv_cache"].shape[1])
    raw_on_kv, raw_on_scale = _kv_and_scale(raw_on_bytes, full["kv_cache"].shape[1])
    raw_off_kv, raw_off_scale = _kv_and_scale(
        raw_off_bytes, full_neg["kv_cache"].shape[1]
    )

    eq_on = torch.equal(gt_kv, raw_on_kv) and torch.equal(gt_scale, raw_on_scale)
    eq_off = torch.equal(gt_kv, raw_off_kv) and torch.equal(gt_scale, raw_off_scale)
    diff_on = (gt_kv.to(torch.int16) - raw_on_kv.to(torch.int16)).abs().sum().item()
    diff_off = (gt_kv.to(torch.int16) - raw_off_kv.to(torch.int16)).abs().sum().item()
    print(
        f"[{tag}] head_dim={head_dim} N={big_N} target_pos={target_pos} "
        f"gt vs raw_on bytes_diff={diff_on}  vs raw_off bytes_diff={diff_off}"
    )
    assert eq_on, (
        f"{tag}: with raw path enabled, boundary at pos={target_pos} (in the "
        f"overwritten prefix) must reproduce the ground-truth FP8 bytes; got "
        f"diff={diff_on}."
    )
    assert not eq_off, (
        f"{tag}: with raw path disabled, the same boundary must read stale "
        f"cache and diverge from ground truth — but bytes matched, suggesting "
        f"the cyclic overwrite isn't actually happening (or CACHE_WINDOW is "
        f"miscalibrated)."
    )


def main():
    assert torch.cuda.is_available(), "CUDA required"
    _run_one(head_dim=128, tag="indexer")
    _run_one(head_dim=512, tag="sparse")
    print("OK")


if __name__ == "__main__":
    main()
