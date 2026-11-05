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
# Decoder layers start at layer 21 (attention_v41.py:733 uses layer_id >= 21
# as the decoder boundary; the 40-layer model's decoder half is L21-39).
DECODER_FIRST_LAYER = 21
# The decoder SWA window is 128 rows; KEEP must exceed it (140 keeps the 12
# rows of gate/logit context beyond the rolling window).
DEFAULT_KEEP = 140
_SWA_WINDOW = 128

_ANNOUNCED = [False]


def decoder_trim_keep() -> int:
    """DSV41_DECODER_TRIM_KEEP: absent/0 = OFF (today's exact path); a
    positive int = keep that many tail tokens in the decoder layers. Reject
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
