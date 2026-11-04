"""D-side fixture loading and decode continuation (L3, plan section 7.5).

The P -> D state hand-off contract is L2's cache fixture format
(``PPUKVCache.export_fixture`` / ``load_fixture``: ``manifest.json`` +
``pools.pt`` with format version, producer snapshot, source CP, per-request
lengths, valid regions and a payload checksum). What the fixture does NOT
carry is the request-side decode state — the token history needed for the
Engram n-gram windows and the request ownership map. This module glues the
two halves together:

* :func:`load_fixture_state` — load + verify a fixture and rebuild a
  :class:`DecodeStepOrchestrator` whose requests are registered against the
  LOADED cache (no re-allocation, no re-prefill);
* :func:`continue_decode` — run N fixed continuation steps from the loaded
  state, sampling argmax tokens and appending them to each request's
  history (the n-gram window therefore spans the step boundary exactly like
  an uninterrupted run);
* :func:`compare_against_uninterrupted` — the offline contract check: the
  continuation from a loaded fixture must match a decode run that never went
  through the export/load round-trip (token-for-token, hidden allclose).

Everything here runs on the decode rank's own machine: no P service, no
network — the file IS the contract (plan 7.5: "文件 fixture 的交换是离线契约
验证，不需要提前启动完整 PD 服务").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator

__all__ = [
    "load_fixture_state",
    "continue_decode",
    "compare_against_uninterrupted",
]


def load_fixture_state(
    fixture_path,
    *,
    ops: Any,
    bundle,
    dp_rank: int = 0,
    dp_size: int = 1,
    request_histories: Dict[int, Sequence[int]],
    owner_map: Optional[Dict[int, int]] = None,
    device=None,
) -> DecodeStepOrchestrator:
    """Restore a decode orchestrator from a cache fixture directory.

    ``request_histories`` maps request id -> its full token history (prompt +
    generated so far); each must match the fixture's recorded length exactly
    (the manifest carries ``lengths`` — a mismatch means the fixture and the
    history describe different states and is refused loudly). ``owner_map``
    optionally pins each request's DP owner (plan 3.2.6); requests owned by
    another rank are simply not registered here.
    """
    cache, manifest = PPUKVCache.load_fixture(fixture_path)
    if owner_map is not None:
        # Filter BEFORE copying to the device. Registering only local requests
        # while retaining every owner's pools defeats Attention DP memory
        # ownership and lets unrelated requests occupy the local block pool.
        local_ids = [
            rid
            for rid in cache._request_tokens
            if int(owner_map.get(rid, dp_rank)) == dp_rank
        ]
        cache = cache.select_requests(local_ids)
    if device is not None:
        # Fixture payloads load on CPU. Move the pools to the decode device
        # AND record the device on the cache itself: the per-call block-table
        # tensors are rebuilt from ``cache.device`` (a CPU table under cuda
        # pools is a device-mismatch crash in the attention reader), and new
        # allocations must land on the same device as the restored pools.
        cache.device = torch.device(device)
        for key in list(cache._pools.keys()):
            cache._pools[key] = cache._pools[key].to(device)
    orchestrator = DecodeStepOrchestrator(
        ops=ops,
        bundle=bundle,
        cache=cache,
        device=device,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    lengths = {int(k): int(v) for k, v in manifest.get("lengths", {}).items()}
    for request_id, history in request_histories.items():
        request_id = int(request_id)
        owner = None if owner_map is None else int(owner_map.get(request_id, dp_rank))
        if owner is not None and owner != dp_rank:
            continue  # owned by another DP rank; its owner restores it there
        if request_id not in lengths:
            raise ValueError(
                f"request {request_id} has a history but the fixture does not "
                f"track it (fixture requests: {sorted(lengths)})"
            )
        if lengths[request_id] != len(history):
            raise ValueError(
                f"request {request_id}: history has {len(history)} tokens but "
                f"the fixture records {lengths[request_id]} (state mismatch; "
                "refusing to continue decode)"
            )
        orchestrator.add_request(request_id, history, owner_rank=owner)
    # The model must describe the same layer TOPOLOGY the fixture was
    # produced with. (Physical block sizes may differ: the fixture carries
    # its own pools; PPUKVCache.load_fixture already verified the fixture's
    # internal geometry signature.)
    manifest_geometry = manifest.get("geometry", {})
    expected = {
        "num_layers": bundle.geometry.num_layers,
        "compress_ratios": list(bundle.geometry.compress_ratios),
        "kv_source_layer_ids": list(bundle.geometry.kv_source_layer_ids),
        "index_source_layer_ids": list(bundle.geometry.index_source_layer_ids),
        "window": bundle.geometry.window,
    }
    for key, value in expected.items():
        if key in manifest_geometry and manifest_geometry[key] != value:
            raise ValueError(
                f"fixture geometry {key}={manifest_geometry[key]} does not match "
                f"the loaded model ({value}); refusing to continue decode from "
                "an incompatible fixture"
            )
    return orchestrator


def continue_decode(
    orchestrator: DecodeStepOrchestrator,
    steps: int,
    seed_tokens: Dict[int, int],
    *,
    placeholders: int = 0,
) -> List[DecodeStepResult]:
    """Run ``steps`` fixed continuation steps, feeding each step's argmax.

    ``seed_tokens`` maps each request to its FIRST continuation token (the
    one sampled from the prefill / fixture-transfer last-token logits). Each
    later step's input is the argmax of the previous step's logits, appended
    to the request history inside the orchestrator, so the Engram n-gram
    window spans the step boundary exactly like an uninterrupted run.
    """
    results: List[DecodeStepResult] = []
    pending: Dict[int, int] = {int(k): int(v) for k, v in seed_tokens.items()}
    for _ in range(int(steps)):
        active = [
            rid for rid, state in orchestrator.requests.items() if not state.finished
        ]
        if not active and not placeholders:
            break
        missing = [rid for rid in active if rid not in pending]
        if missing:
            raise RuntimeError(
                f"no input token for requests {missing}: seed_tokens must "
                "cover every active request (argmax of the previous step)"
            )
        result = orchestrator.step(
            {rid: pending[rid] for rid in active}, placeholders=placeholders
        )
        results.append(result)
        pending = {}
        if result.logits is None:
            break
        for row, request_id in enumerate(result.row_request_ids):
            pending[request_id] = int(result.logits[row].argmax())
    return results


def compare_against_uninterrupted(
    reference: DecodeStepOrchestrator,
    restored: DecodeStepOrchestrator,
    steps: int,
    seed_tokens: Dict[int, int],
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Dict[str, object]:
    """Offline contract check (plan 7.5): loaded fixture == uninterrupted run.

    ``reference`` continues from the in-memory state (no export/load);
    ``restored`` continues from the exported+reloaded fixture. Both consume
    the same seed tokens and N steps; the contract holds when every step's
    argmax token stream matches exactly and the hidden states stay allclose
    (bf16 codec round-trips can shift the last bits).
    """
    reference_steps = continue_decode(reference, steps, dict(seed_tokens))
    restored_steps = continue_decode(restored, steps, dict(seed_tokens))
    if len(reference_steps) != len(restored_steps):
        return {
            "match": False,
            "reason": f"step count {len(restored_steps)} != {len(reference_steps)}",
        }
    max_diff = 0.0
    for index, (left, right) in enumerate(zip(reference_steps, restored_steps)):
        if left.row_request_ids != right.row_request_ids:
            return {
                "match": False,
                "reason": f"step {index}: request order diverged",
            }
        if left.logits is None or right.logits is None:
            continue
        left_tokens = [int(row.argmax()) for row in left.logits]
        right_tokens = [int(row.argmax()) for row in right.logits]
        if left_tokens != right_tokens:
            return {
                "match": False,
                "reason": f"step {index}: token stream diverged "
                f"({left_tokens} != {right_tokens})",
            }
        if left.hidden.shape != right.hidden.shape:
            return {
                "match": False,
                "reason": f"step {index}: hidden shape diverged",
            }
        if not torch.isfinite(right.hidden).all():
            return {"match": False, "reason": f"step {index}: non-finite hidden"}
        diff = (left.hidden.float() - right.hidden.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(left.hidden, right.hidden, atol=atol, rtol=rtol):
            return {
                "match": False,
                "reason": f"step {index}: hidden diff {diff:.3e} exceeds tolerance",
                "max_abs_diff": max_diff,
            }
    return {
        "match": True,
        "steps": len(reference_steps),
        "max_abs_diff": max_diff,
    }
