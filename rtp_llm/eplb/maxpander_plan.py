"""Deterministic static circulant placement for MaxPander (at most two replicas)."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np

DEFAULT_SEED = 0


def draft_redundant_experts(
    target_experts: int, target_redundant: int, draft_experts: int
) -> int:
    """Preserve the target redundancy ratio, not its absolute count, in DSpark."""
    if (
        target_experts <= 0
        or draft_experts <= 0
        or not 0 <= target_redundant <= target_experts
    ):
        raise ValueError(
            "MaxPander requires positive expert counts and redundancy in [0,1]"
        )
    redundant, remainder = divmod(target_redundant * draft_experts, target_experts)
    if remainder:
        raise ValueError(
            "target MaxPander ratio does not give an integral draft replica count"
        )
    return redundant


# --------------------------------------------------------------------------
# circulant placement (prediction-free, spectral-gap-optimal offsets)
# --------------------------------------------------------------------------


def circulant_fits(num_logical: int, num_physical: int, num_gpus: int) -> bool:
    if num_gpus <= 0 or num_logical <= 0 or num_physical < num_logical:
        return False
    if num_gpus == 1 and num_physical != num_logical:
        return False
    if num_logical % num_gpus != 0:
        return False
    extra = num_physical - num_logical
    if extra % num_gpus != 0:
        return False
    return (extra // num_gpus) <= (num_logical // num_gpus)


def _circulant_lambda2(offsets: tuple, num_gpus: int) -> float:
    G = num_gpus
    A = np.zeros((G, G), dtype=np.float64)
    for o in offsets:
        for g in range(G):
            A[g, (g + o) % G] = 1.0
            A[g, (g - o) % G] = 1.0
    deg = A.sum(axis=1)
    if np.any(deg <= 0):
        return 1.0
    dinv = 1.0 / np.sqrt(deg)
    a_norm = A * dinv[:, None] * dinv[None, :]
    eig = np.abs(np.linalg.eigvalsh(a_norm))
    eig.sort()
    return float(eig[-2]) if eig.size > 1 else 0.0


def best_offsets(num_gpus: int, rep_per: int, max_exhaustive: int = 20000) -> list:
    G = num_gpus
    if G <= 1 or rep_per <= 0:
        return []
    cand = list(range(1, G))
    m = min(rep_per, G - 1)
    if m >= G - 1:
        chosen = cand
    elif comb(G - 1, m) <= max_exhaustive:
        best, best_l2 = None, None
        for subset in itertools.combinations(cand, m):
            l2 = _circulant_lambda2(subset, G)
            if best_l2 is None or l2 < best_l2 - 1e-12:
                best, best_l2 = subset, l2
        chosen = list(best)
    else:
        chosen = []
        remaining = set(cand)
        while len(chosen) < m:
            pick, pick_l2 = None, None
            for o in sorted(remaining):
                l2 = _circulant_lambda2(tuple(chosen + [o]), G)
                if pick_l2 is None or l2 < pick_l2 - 1e-12:
                    pick, pick_l2 = o, l2
            chosen.append(pick)
            remaining.discard(pick)
    return [chosen[r % len(chosen)] for r in range(rep_per)]


def build_circulant_plan(
    num_logical: int, num_physical: int, num_gpus: int, seed: int = DEFAULT_SEED
):
    """Static circulant expander placement.

    Returns (phy2log [P] int64, log2phy [E, max_cnt] int64, logcnt [E] int64).
    Slot ids are contiguous per GPU: gpu = phys // (P // G).
    """
    if not circulant_fits(num_logical, num_physical, num_gpus):
        raise ValueError(
            f"circulant placement does not fit E={num_logical} "
            f"P={num_physical} G={num_gpus}"
        )
    E, G = num_logical, num_gpus
    cap = num_physical // G
    home_per = E // G
    rep_per = (num_physical - E) // G
    rng = np.random.default_rng(seed)

    place = np.zeros((E, G), dtype=np.bool_)
    perm = rng.permutation(E)
    home = [perm[g * home_per : (g + 1) * home_per] for g in range(G)]
    for g in range(G):
        place[home[g], g] = True

    if G == 1:
        offsets = [0] * rep_per
    else:
        offsets = best_offsets(G, rep_per)
    for g in range(G):
        for r in range(rep_per):
            expert = int(home[g][r])
            dst = (g + offsets[r]) % G
            place[expert, dst] = True

    assert place.sum() == num_physical
    assert np.all(place.sum(0) == cap)

    counts = place.sum(1).astype(np.int64)
    max_replicas = int(counts.max())
    lp = np.full((E, max_replicas), -1, dtype=np.int64)
    phy2log = np.full((num_physical,), -1, dtype=np.int64)
    next_slot = np.arange(G, dtype=np.int64) * cap
    for e in range(E):
        for col, g in enumerate(np.flatnonzero(place[e])):
            slot = int(next_slot[g])
            lp[e, col] = slot
            phy2log[slot] = e
            next_slot[g] += 1
    assert not np.any(phy2log < 0)
    return phy2log, lp, counts
