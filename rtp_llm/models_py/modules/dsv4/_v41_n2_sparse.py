"""N2 (Batch 5, audit #8): sparse EP dispatch helpers.

Dense dispatch sends every row to every EP rank even when the rank owns
none of the row's experts (~18% wasted wire under uniform routing). Under
``DSV41_N2_SPARSE_DISPATCH`` each row goes only to the ranks owning at
least one of its active experts. The helpers here are pure tensor/index
math (CPU-testable); the exchange itself lives in
``V41NcclEpStrategy._sparse_dispatch_exchange``.

Bitwise contract: per-row expert partials are row-independent (each row's
routed sum is computed in the stable per-expert order regardless of the
received batch), and skipped (row, rank) pairs reconstruct to zero wire
bytes whose dequantized contribution is +0.0 — exactly what the dense
path's zero partial quantizes to. The parity sha gate is the backstop.
"""

import os
import sys

import torch

_N2_FLAG = "DSV41_N2_SPARSE_DISPATCH"
_N2_ANNOUNCED = False


def n2_sparse_dispatch_enabled() -> bool:
    """N2: sparse EP dispatch/return (default 0)."""
    return os.environ.get(_N2_FLAG, "0") == "1"


def _announce_n2(sent: int, dense: int) -> None:
    global _N2_ANNOUNCED
    if not _N2_ANNOUNCED:
        _N2_ANNOUNCED = True
        pct = 100.0 * sent / dense if dense else 0.0
        print(
            f"[N2SPARSE] sparse EP dispatch engaged (wire rows {sent}/{dense} = {pct:.1f}% of dense)",
            file=sys.stderr,
            flush=True,
        )


def sparse_destination_mask(
    indices: torch.Tensor, world: int, per: int
) -> torch.Tensor:
    """bool [rows, world]: row i goes to rank d iff one of its expert ids
    lies in ``[d*per, (d+1)*per)``. Invalid ids (-1) floor-divide to -1
    and match no rank; all-invalid rows go nowhere (their dense partials
    are zeros on every rank)."""
    owner = torch.div(indices.long(), per, rounding_mode="floor")
    ranks = torch.arange(world, device=indices.device)
    return (owner[:, :, None] == ranks[None, None, :]).any(dim=1)


def sparse_compaction_order(contrib: torch.Tensor) -> torch.Tensor:
    """Stable d-major compaction of bool [rows, world].

    Returns a fixed-shape int64 [rows*world] tensor whose first
    ``contrib.sum()`` entries are the flat d-major indices
    (``d*rows + row``) of the selected entries in ascending order
    (destinations ascending, source rows ascending within each
    destination). Fixed-shape sort keeps this CUDA-graph/sync free — the
    single host sync for split sizes happens in the caller.
    """
    rows, world = contrib.shape
    flat = contrib.t().reshape(-1)
    positions = torch.arange(rows * world, device=contrib.device)
    key = torch.where(flat, positions, positions.new_full((), rows * world))
    sorted_keys, _ = torch.sort(key, stable=True)
    return sorted_keys
