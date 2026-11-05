"""R58c routing-replay experiment (test 3 of the decoder-trim error analysis).

Records the MASK arm's discrete routing decisions (MoE expert top-k
indices+weights and sparse/indexer top-k selections) for the final-chunk tail
rows, and replays them into the COMPACT arm. Both arms then make IDENTICAL
discrete choices, so the residual compact-vs-mask final-logit difference is
pure continuous noise (GEMM M-tiling, attention accumulation order) — the
pass bar for "no additional error" is rel_l2 < 1e-2.

Record side (mask arm): nothing special — the MOEDBG recorder already captures
``L{layer:02d}_moe_topk_{indices,weights}`` and ``L{layer:02d}_sparse_topk``,
and on the mask arm the whole 144-row tail sits IN ORDER on rank 0's back half
(rows 880..1023 of the 1024-row chunk share), so the rank-0 dump file's last
144 rows ARE the canonical tail.

Replay side (compact arm): the trim plan publishes this rank's compact-row ->
canonical-tail index map when it engages; the consumers (MoE gate, sparse
selection) substitute the recorded rows whenever the compact geometry is
active. Decode steps and non-final chunks never publish the map, so they are
never replayed.

Env:
    DSV41_R58C_REPLAY_DIR   — directory containing the mask arm's per-chunk
                              dumps (``rank*_*_step{S}.pt``) (empty = off).
    DSV41_R58C_REPLAY_STEPS — comma list of the recorded forward steps forming
                              one request's chunk sequence (e.g. "0,1,2,3,4,5,6,7"
                              for the 8 x 4096-token chunks of warmup request 0).
                              The replay cycles through them per engaged
                              (compacted) chunk; determinism makes request 0's
                              routing valid for every identical request.
"""

import glob
import os
from typing import Dict, Optional, Tuple

import torch

_REPLAY_DIR = os.environ.get("DSV41_R58C_REPLAY_DIR", "")
_REPLAY_STEPS = [
    int(s)
    for s in os.environ.get("DSV41_R58C_REPLAY_STEPS", "").split(",")
    if s.strip()
]
# Backward compat: a single DSV41_R58C_REPLAY_STEP replays that step's routing
# into every engaged chunk (valid because a chunk's decoder-layer K/V is read
# only by that same chunk — the tail window (128) never crosses the 4096 chunk
# boundary and the sparse globals are per-chunk).
if not _REPLAY_STEPS and os.environ.get("DSV41_R58C_REPLAY_STEP", ""):
    _REPLAY_STEPS = [int(os.environ["DSV41_R58C_REPLAY_STEP"])]

# This rank's compact rows -> canonical tail indices (set by the trim plan
# while the decoder layers run compacted; None otherwise).
_ROW_INDEX: Optional[torch.Tensor] = None
# Per-forward-chunk counter: each engaged (compacted) chunk advances it; the
# current chunk's recorded step is _REPLAY_STEPS[counter % len(_REPLAY_STEPS)].
_CHUNK_COUNTER = 0
# Lazily loaded {step: blob} and {(step, layer_id, kind): canonical cpu tensor}.
_FILES: Dict[int, object] = {}
_CACHE: Dict[Tuple[int, int, str], Optional[torch.Tensor]] = {}


def replay_enabled() -> bool:
    return bool(_REPLAY_DIR) and bool(_REPLAY_STEPS)


def publish_row_index(row_index: Optional[torch.Tensor]) -> None:
    """DecoderTrimPlan.compact/restore_rows set/clear this rank's compact-row
    -> canonical-tail index map (CPU int64 [local_rows]). Each SET advances the
    engaged-chunk counter (chunk i of the stream replays step STEPS[i % n])."""
    global _ROW_INDEX, _CHUNK_COUNTER
    _ROW_INDEX = row_index
    if row_index is not None:
        _CHUNK_COUNTER += 1


def _current_step() -> int:
    if not _REPLAY_STEPS:
        return -1
    # The counter was advanced at publish time; chunk i uses STEPS[(i-1) % n].
    return _REPLAY_STEPS[(_CHUNK_COUNTER - 1) % len(_REPLAY_STEPS)]


def _load_file(step: int):
    if step not in _FILES:
        hits = sorted(
            glob.glob(os.path.join(_REPLAY_DIR, f"rank*_*_step{step:03d}.pt"))
        ) or sorted(glob.glob(os.path.join(_REPLAY_DIR, f"rank*_*_step{step}.pt")))
        if not hits:
            raise RuntimeError(f"R58c replay: no step-{step} dumps in {_REPLAY_DIR}")
        # The canonical tail (ascending positions) sits on the LAST 144 rows of
        # the stage-1 rank whose cp_rank == 0 (zigzag back half). Global rank
        # varies with the PP layout, so select by the dump's extra metadata.
        chosen = None
        for path in hits:
            try:
                blob = torch.load(path, map_location="cpu")
            except Exception:
                continue
            extra = blob.get("extra", {})
            if (
                int(extra.get("cp_rank", -1)) == 0
                and int(extra.get("cp_size", -1)) == 4
            ):
                chosen = blob
                break
        if chosen is None:
            raise RuntimeError(
                f"R58c replay: no cp_rank==0 dump among {len(hits)} step-{step} files"
            )
        _FILES[step] = chosen
    return _FILES[step]


def _canonical(layer_id: int, kind: str) -> Optional[torch.Tensor]:
    if not replay_enabled() or _CHUNK_COUNTER == 0:
        return None
    step = _current_step()
    key = (step, layer_id, kind)
    if key not in _CACHE:
        tensors = _load_file(step).get("tensors", {})
        _CACHE[key] = tensors.get(f"L{layer_id:02d}_{kind}")
    return _CACHE[key]


def _rows_for(x: torch.Tensor) -> Optional[torch.Tensor]:
    """This rank's canonical-tail row indices iff x is on the compact decoder
    geometry (the replay gate), else None."""
    if _ROW_INDEX is None or not replay_enabled():
        return None
    if x.shape[0] != _ROW_INDEX.numel():
        return None
    return _ROW_INDEX


def override_moe_routing(
    layer_id: int, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Replace the router's (weights, indices) with the mask arm's recorded
    tail-row routing when the compact geometry is active."""
    rows = _rows_for(x)
    if rows is None:
        return None
    rec_w = _canonical(layer_id, "moe_topk_weights")
    rec_i = _canonical(layer_id, "moe_topk_indices")
    if rec_w is None or rec_i is None:
        return None
    sel_w = rec_w.index_select(0, rows).to(device=x.device, dtype=weights.dtype)
    sel_i = rec_i.index_select(0, rows).to(device=x.device, dtype=indices.dtype)
    if sel_w.shape != weights.shape or sel_i.shape != indices.shape:
        return None
    return sel_w, sel_i


def override_sparse_topk(
    layer_id: int, x: torch.Tensor, selected: torch.Tensor
) -> Optional[torch.Tensor]:
    """Replace the indexer selection with the mask arm's recorded tail-row
    selection when the compact geometry is active."""
    rows = _rows_for(x)
    if rows is None:
        return None
    rec = _canonical(layer_id, "sparse_topk")
    if rec is None:
        return None
    sel = rec.index_select(0, rows).to(device=x.device, dtype=selected.dtype)
    if sel.shape != selected.shape:
        return None
    return sel
