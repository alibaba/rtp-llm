"""R48 — decoder SWA bounded replay / YOCO prefill truncation (scaffold).

Flag ``DSV41_DECODER_TRIM_KEEP`` (int; absent/0 = OFF = today's exact path;
140 = the R48 arm; negatives/non-ints rejected loudly). Default OFF.

Semantics (DeepSeek's decoder SWA bounded replay / YOCO prefill truncation):
the decoder layers (layer_id >= 21, VERIFY: the code's decoder boundary) run
only on the last KEEP tokens of the request. The decoder SWA window is 128
rows; KEEP=140 > 128 means the decoder attention over the last 140 tokens is
exactly what the rolling-128 SWA would compute (no halo needed, unlike R19's
exact-CED 3072+halo). Non-exact by design (precision-class).

This module is the SCAFFOLD: the flag + the pure CP4 zigzag tail geometry
(reused from R19's _v41_ced.py), CPU-unit-tested. The actual truncation of
the decoder layers lands in the implementation stage (R48b) on top of this.

Byte-exactness: flag OFF => zero behavior change (the module is inert).
"""

from __future__ import annotations

import os
import sys

DECODER_TRIM_KEEP_FLAG = "DSV41_DECODER_TRIM_KEEP"
# R58: mask-mode reference arm (DSV41_DECODER_TRIM_MASK, default 0). Full-length
# compute with the keep-tail trim semantics applied by MASKING the decoder SWA
# visibility to the tail (no compaction/padding/CP redistribution). This is the
# "no additional error" reference for the compact-mode equivalence gate.
DECODER_TRIM_MASK_FLAG = "DSV41_DECODER_TRIM_MASK"
# Decoder layers start at layer 21 (attention_v41.py:733 uses layer_id >= 21
# as the decoder boundary; the 40-layer model's decoder half is L21-39).
DECODER_FIRST_LAYER = 21
# The decoder SWA window is 128 rows; KEEP must exceed it (140 keeps the 12
# rows of gate/logit context beyond the rolling window).
DEFAULT_KEEP = 140
_SWA_WINDOW = 128

_ANNOUNCED = [False]


def decoder_trim_keep() -> int:
    """DSV41_DECODER_TRIM_KEEP: absent/0 = OFF (today's exact path). R58c proved
    the trim adds no additional error (routing-replay gate: bitwise vs the mask
    reference) but the net TTFT win is currently ~0 (the correctness-required
    graph-replay suspension over the compacted region eats the small-M GEMM
    savings), so it stays opt-in: a positive int = keep that many tail tokens
    in the decoder layers (140 = the validated DeepInfer rectangle). Reject
    negatives and non-integers loudly."""
    raw = os.environ.get(DECODER_TRIM_KEEP_FLAG, "0")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{DECODER_TRIM_KEEP_FLAG} must be an integer, got {raw!r}")
    if value < 0:
        raise ValueError(f"{DECODER_TRIM_KEEP_FLAG} must be >= 0, got {value}")
    if value and value <= _SWA_WINDOW:
        raise ValueError(
            f"{DECODER_TRIM_KEEP_FLAG}={value} <= the decoder SWA window "
            f"({_SWA_WINDOW}); the keep rectangle must exceed the window"
        )
    if value and not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        print(
            f"[DECODERTRIM] engaged ({DECODER_TRIM_KEEP_FLAG}={value}): decoder "
            f"layers L{DECODER_FIRST_LAYER}+ keep the last {value} request tokens",
            file=sys.stderr,
            flush=True,
        )
    return value


def decoder_trim_enabled() -> bool:
    return decoder_trim_keep() > 0


def decoder_trim_mask_keep() -> int:
    """DSV41_DECODER_TRIM_MASK (int, default 0): the R58 mask-mode reference arm.
    Full-length compute with the keep-tail trim semantics applied by MASKING the
    decoder SWA visibility to the tail (no compaction/padding/CP redistribution).
    The value is the DeepInfer KEEP target (140 -> CP-legal tail 144, like the
    compact arm). When > 0, mask mode engages and the compact arm is gated off
    (they are mutually exclusive reference vs optimized implementations of the
    same trim semantics)."""
    raw = os.environ.get(DECODER_TRIM_MASK_FLAG, "0")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{DECODER_TRIM_MASK_FLAG} must be an integer, got {raw!r}")
    if value < 0:
        raise ValueError(f"{DECODER_TRIM_MASK_FLAG} must be >= 0, got {value}")
    if value and value <= _SWA_WINDOW:
        raise ValueError(
            f"{DECODER_TRIM_MASK_FLAG}={value} <= the decoder SWA window ({_SWA_WINDOW})"
        )
    return value


def mask_tail() -> int:
    """The CP-legal compact tail for the mask-mode KEEP target (mirror of
    decoder_trim_tail but for the mask arm). 0 when mask mode is off."""
    keep = decoder_trim_mask_keep()
    if keep == 0:
        return 0
    length = keep
    while length % (2 * _CP) != 0 or (length // _CP) % 2 != 0:
        length += 1
    return length


DECODER_TRIM_MASK_EXACT_FLAG = "DSV41_DECODER_TRIM_MASK_EXACT"


def decoder_trim_mask_exact() -> bool:
    """DSV41_DECODER_TRIM_MASK_EXACT (default 0): with the mask arm on, put the
    visibility floor at prefix + n - KEEP (the exact keep-140 rectangle —
    DeepInfer's yoco_swa_visibility rect_lo = data_len - keep_tokens) instead
    of the default mask-matched floor prefix + n - tail (tail = the CP-legal
    compact length, 144 for KEEP=140 — the compact arm's exact semantics)."""
    return os.environ.get(DECODER_TRIM_MASK_EXACT_FLAG, "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def mask_start_global(prefix_length: int, seq_len_full: int, tail: int) -> int:
    """The trim-tail start (global position) = the first row the decoder layers
    keep. On the final chunk this is prefix + seq_len_full - tail. Decoder SWA
    visibility is masked to positions >= this."""
    return prefix_length + seq_len_full - tail


def mask_floor_global(
    prefix_length: int, seq_len_full: int, tail: int, keep: int, exact: bool
) -> int:
    """The SWA visibility floor the mask arm publishes for the final chunk.
    exact=False -> tail start (prefix + n - tail): the compact arm's exact
    semantics (the reference for the equivalence gate). exact=True ->
    prefix + n - keep: DeepInfer's exact keep-140 rectangle."""
    start = mask_start_global(prefix_length, seq_len_full, tail)
    return start + (tail - keep) if exact else start


def is_decoder_layer(layer_id: int) -> bool:
    """The decoder boundary (L21+) — the layers that truncate under R48."""
    return layer_id >= DECODER_FIRST_LAYER


# ---------------------------------------------------------------------------
# Pure CP4 zigzag tail geometry (self-contained copies of R19 _v41_ced's
# functions — kept dependency-free so the CPU test needs no torch import of
# the prefill package; verified equivalent to _v41_ced's in the test).
# ---------------------------------------------------------------------------


def _zigzag_owner_local(p: int, half: int, cp: int = 4) -> tuple:
    """Canonical position -> (owner rank, rank-local row) for a zigzag split
    into 2*cp halves of ``half`` rows (front half pair p, back half mirrored).
    Byte-identical mapping to _v41_ced.zigzag_owner_local."""
    pair = p // half
    owner = pair if pair < cp else 2 * cp - 1 - pair
    local = p % half + (0 if pair < cp else half)
    return owner, local


def tail_owner_rows(n_tokens: int, keep: int, cp: int = 4) -> dict:
    """Which CP rank owns which global rows of the KEEP-token tail.

    Returns {rank: [global rows]} for the last ``keep`` tokens of an
    ``n_tokens``-token request under the CP zigzag (each rank holds
    n_tokens/cp rows: front half + mirrored back half).
    """
    start = n_tokens - keep
    out = {r: [] for r in range(cp)}
    half = n_tokens // (2 * cp)  # rows per half per rank
    for p in range(start, n_tokens):
        owner, _local = _zigzag_owner_local(p, half, cp)
        out[owner].append(p)
    return out


# ---------------------------------------------------------------------------
# R48d — compact consumer-view geometry (the decoder layers run on a standalone
# compact tail sequence, CP4-re-zigzagged; mirrors _v41_ced's proven helpers).
#
# Design choice (documented in PROGRESS.md): the compact tail length L is the
# CP-legal ceiling of the DeepInfer KEEP target. KEEP=140 -> L=144 (the smallest
# length >= KEEP with L % (2*cp) == 0 AND even rows/rank). We keep L REAL tail
# rows (no synthetic pads): the mHC delayed-residual chain is row-local
# (collapse_delayed sums over the hc/streams axis, not the token axis), so the
# extra (L - KEEP) real context rows are SWA-evicted and do not change the final
# token's output, and there is no pad-K/V contamination vector to mask.
# ---------------------------------------------------------------------------

_CP = 4


def decoder_trim_tail() -> int:
    """The CP4-legal compact tail length for the configured KEEP target.

    L = smallest length >= KEEP with L % (2*cp) == 0 and (L//cp) even (so the
    compact zigzag splits evenly into front/back halves per rank). KEEP=140 ->
    L=144 (half=18, 36 rows/rank). Returns 0 when the trim is off.
    """
    keep = decoder_trim_keep()
    if keep == 0:
        return 0
    length = keep
    while length % (2 * _CP) != 0 or (length // _CP) % 2 != 0:
        length += 1
    return length


def compact_local_rows(tail: int, cp: int = _CP) -> int:
    """Rows per rank in the compact view."""
    return tail // cp


def zigzag_relative(rank: int, local_rows: int, cp: int = _CP) -> list:
    """Rank-local row idx -> canonical compact position (front + mirrored back).
    Byte-identical mapping to _v41_ced.zigzag_relative."""
    half = local_rows // 2
    total = local_rows * cp
    return [rank * half + i for i in range(half)] + [
        total - (rank + 1) * half + i for i in range(half)
    ]


def zigzag_unpad_restore(local_rows: int, total: int, cp: int = _CP) -> list:
    """Canonical compact position -> index into the rank-concatenated gather
    flat. Byte-identical mapping to _v41_ced.zigzag_unpad_restore."""
    half = local_rows // 2
    restore = [0] * total
    for p in range(total):
        owner, local = _zigzag_owner_local(p, half, cp)
        restore[p] = owner * local_rows + local
    return restore


def trim_tail_source_map(
    rank: int, rows: int, n: int, tail: int, cp: int = _CP
) -> tuple:
    """Restore path: this rank's ORIGINAL local rows that fall inside the chunk
    tail -> (dst local idx, src canonical tail idx). Mirrors _v41_ced.tail_source_map;
    handles arbitrary tails (the whole 144-tail sits on rank 0's back half in the
    original per-chunk layout, so only rank 0 has non-empty dst/src)."""
    half = rows // 2
    start = n - tail
    dst, src = [], []
    for i in range(rows):
        pair = rank if i < half else 2 * cp - 1 - rank
        p = pair * half + i % half
        if p >= start:
            dst.append(i)
            src.append(p - start)
    return dst, src


def is_final_chunk(prefix_length: int, seq_len_full: int, seq_len_total: int) -> bool:
    """True iff this chunk is the request's final chunk (rank-invariant under CP:
    prefix_length/seq_len_full/seq_len_total are all global scalars). The decoder
    trim compacts only the final chunk; earlier chunks skip the decoder layers."""
    return prefix_length + seq_len_full == seq_len_total
