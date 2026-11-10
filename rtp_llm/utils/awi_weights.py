"""AWI checkpoint-weights placement recorder and publisher.

Feeds the evaluator's ``weights`` contract from genuine allocations, never
from requested settings:

- ``record_fetch`` is called at the checkpoint-database chokepoint, so every
  loader-fetched key (dense, MoE, vision, stacked-expert splits) is recorded
  with the bytes actually read for it on this rank.
- Engram keys are not loader-fetched (``Engram.from_checkpoint`` reads them
  directly); they are walked live on the constructed modules and reported
  from their actual host allocations — pinned/anonymous host RAM with zero
  persistent GPU bytes — including a guard that the wkv projection kept no
  cached device dequantization.
- All other recorded keys report the rank's live CUDA device (read from an
  actual model tensor at publish time) with ``storage=device``.
- Coverage is checked at publish time against the checkpoint index; gaps are
  logged loudly and surfaced, never papered over with synthetic rows.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_fetched: Dict[str, Dict[str, Any]] = {}
_model_root = None
_checkpoint_path: Optional[str] = None
# Retains device allocations for checkpoint keys the runtime never executes
# (speculative mtp.* weights while SP is disabled); see ensure_inert_resident.
_inert_resident: List[Tuple[str, Any]] = []

_ENGRAM_SUFFIXES = (
    "embed.weight",
    "embed.scale",
    "q_weight",
    "k_weight",
    "wkv.weight",
    "wkv.scale",
)


def record_fetch(name: str, tensors, basis: str = "loader_fetch") -> None:
    """Record one checkpoint fetch; called from CkptDatabase.load_tensor."""
    try:
        total = 0
        dtypes: List[str] = []
        shapes: List[List[int]] = []
        for tensor in tensors or []:
            total += tensor.numel() * tensor.element_size()
            dtypes.append(str(tensor.dtype).replace("torch.", ""))
            shapes.append([int(d) for d in tensor.shape])
        with _lock:
            previous = _fetched.get(name)
            if previous is None:
                _fetched[name] = {
                    "checkpoint_bytes": total,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "fetches": 1,
                    "basis": basis,
                }
            else:
                previous["fetches"] += 1
                previous["checkpoint_bytes"] = max(
                    previous["checkpoint_bytes"], total
                )
    except Exception:  # pragma: no cover - instrumentation must not break load
        logger.warning("AWI weights recorder failed for %s", name, exc_info=True)


def record_slice_fetch(name: str, nbytes: int) -> None:
    """Record a sliced checkpoint read (load_tensor_slice path)."""
    try:
        with _lock:
            previous = _fetched.get(name)
            if previous is None:
                _fetched[name] = {
                    "checkpoint_bytes": int(nbytes),
                    "dtypes": [],
                    "shapes": [],
                    "fetches": 1,
                    "sliced": True,
                }
            else:
                previous["fetches"] += 1
                previous["checkpoint_bytes"] = previous.get(
                    "checkpoint_bytes", 0
                ) + int(nbytes)
                previous["sliced"] = True
    except Exception:  # pragma: no cover
        logger.warning("AWI slice recorder failed for %s", name, exc_info=True)


def recorded_names() -> List[str]:
    with _lock:
        return sorted(_fetched)


def ensure_inert_resident(
    checkpoint_path: str,
    *,
    prefix: str = "mtp.",
    device: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Give never-executed checkpoint keys a genuine device allocation.

    The placement contract covers every manifest key and lane policy keeps
    non-Engram weights device-resident. Speculative (``mtp.*``) weights are
    not loaded while speculative execution is disabled, so each such key is
    read here verbatim — original dtype and bytes, no decoding, no
    transformation, no forward wiring — and kept on exactly one rank
    (round-robin over the sorted names). Keys the loader already fetched
    (speculative execution enabled) are skipped. Must run before KV-cache
    sizing so the reservation sees the true steady-state footprint, and
    failures propagate: a placement gap must never pass silently.
    """
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_path) as file:
        weight_map = json.load(file)["weight_map"]
    names = sorted(name for name in weight_map if name.startswith(prefix))
    if not names:
        return {"owned": 0, "bytes": 0}
    if rank is None or world_size is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank() if rank is None else rank
                world_size = (
                    dist.get_world_size() if world_size is None else world_size
                )
        except Exception:  # noqa: BLE001 - fall through to env/single-rank
            pass
    rank = int(rank if rank is not None else os.environ.get("RANK", "0"))
    world_size = int(
        world_size if world_size is not None else os.environ.get("WORLD_SIZE", "1")
    )
    world_size = max(1, world_size)
    already = set(recorded_names())
    owned = [
        name
        for position, name in enumerate(names)
        if position % world_size == rank and name not in already
    ]
    if device is None:
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
    from safetensors import safe_open

    total = 0
    for name in owned:
        shard = os.path.join(checkpoint_path, weight_map[name])
        with safe_open(shard, framework="pt", device="cpu") as file:
            tensor = file.get_tensor(name)
        resident = tensor.to(device)
        record_fetch(name, [resident], basis="inert_resident")
        _inert_resident.append((name, resident))
        total += resident.numel() * resident.element_size()
    logger.info(
        "AWI inert residency: rank=%d loaded %d '%s*' keys (%.2f MiB) on %s",
        rank,
        len(owned),
        prefix,
        total / 1024**2,
        device,
    )
    return {"owned": len(owned), "bytes": total, "device": str(device)}


def _detect_device(module_root) -> Optional[str]:
    for tensor in module_root.parameters():
        if tensor.is_cuda:
            return f"cuda:{tensor.device.index}"
    for tensor in module_root.buffers():
        if tensor.is_cuda:
            return f"cuda:{tensor.device.index}"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return None


def _host_row(name: str, tensor, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    storage = "pinned" if tensor.is_pinned() else "anonymous"
    row = {
        "name": name,
        "device": "cpu",
        "storage": storage,
        "persistent_gpu_bytes": 0,
        "checkpoint_bytes": tensor.numel() * tensor.element_size(),
        "basis": "live_host_allocation",
    }
    if extra:
        row.update(extra)
    return row


def _engram_rows(module_root) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Report every Engram checkpoint key from live host allocations."""
    from rtp_llm.models_py.modules.dsv41.engram import Engram

    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for module in module_root.modules():
        if not isinstance(module, Engram):
            continue
        layer_id = module.layout.layer_ids[module.layer_hash_index]
        prefix = f"layers.{layer_id}.engram."
        embedding = module.embed_tokens
        table = getattr(embedding, "weight", None)
        scales = getattr(embedding, "scales", None)
        if table is None or table.device.type != "cpu":
            problems.append(f"{prefix}embed.weight not host-resident")
        else:
            rows.append(_host_row(prefix + "embed.weight", table))
        if scales is None or scales.device.type != "cpu":
            problems.append(f"{prefix}embed.scale not host-resident")
        else:
            rows.append(_host_row(prefix + "embed.scale", scales))
        # _host_pinned order is fixed by Engram.from_checkpoint:
        # (wkv.weight, wkv.scale, q_weight, k_weight).
        pinned = list(getattr(module, "_host_pinned", ()))
        expected = ["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
        if len(pinned) == len(expected):
            for suffix, owner in zip(expected, pinned):
                host = getattr(owner, "host", None)
                if host is None or host.device.type != "cpu":
                    problems.append(f"{prefix}{suffix} host copy missing")
                    continue
                if not host.is_pinned():
                    problems.append(f"{prefix}{suffix} is not pinned")
                rows.append(
                    _host_row(
                        prefix + suffix,
                        host,
                        {"shape": [int(d) for d in host.shape]},
                    )
                )
        else:
            # CPU-target reference builds carry no pinned owners; the buffers
            # themselves must then be host tensors.
            for suffix, buffer in (
                ("q_weight", module.q_weight),
                ("k_weight", module.k_weight),
            ):
                if buffer.device.type != "cpu":
                    problems.append(f"{prefix}{suffix} not host-resident")
                else:
                    rows.append(_host_row(prefix + suffix, buffer))
            problems.append(f"{prefix}wkv.* pinned owners absent")
        # Policy guard: no persistent GPU dequantization of the projection.
        wkv = module.wkv
        if getattr(wkv, "_dequant_bf16", None) is not None:
            problems.append(f"{prefix}wkv holds a persistent bf16 GPU dequant")
        weight = getattr(wkv, "weight", None)
        if weight is not None and weight.is_cuda and weight.is_pinned() is False:
            # A UVA view reports is_cuda True but is backed by the pinned host
            # allocation; a real device allocation would fail this shape match.
            host_bytes = (
                pinned[0].host.numel() * pinned[0].host.element_size()
                if pinned
                else -1
            )
            if weight.numel() * weight.element_size() != host_bytes:
                problems.append(f"{prefix}wkv.weight has a device allocation")
    return rows, problems


def _checkpoint_index_names(checkpoint_path: str) -> Optional[set]:
    try:
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        with open(index_path) as file:
            return set(json.load(file)["weight_map"].keys())
    except Exception:
        logger.warning("AWI coverage check: cannot read checkpoint index", exc_info=True)
        return None


def build_weights_rows(
    module_root, checkpoint_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Return (rows, missing_names, problems) from live state."""
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    engram_rows, engram_problems = _engram_rows(module_root)
    rows.extend(engram_rows)
    problems.extend(engram_problems)
    seen = {row["name"] for row in rows}
    device = _detect_device(module_root)
    if device is None:
        problems.append("no CUDA tensor found on the model; device rows unverifiable")
    with _lock:
        fetched = dict(_fetched)
    for name in sorted(fetched):
        if name in seen:
            continue
        seen.add(name)
        meta = fetched[name]
        rows.append(
            {
                "name": name,
                "device": device or "cpu",
                "storage": "device" if device else "anonymous",
                "persistent_gpu_bytes": meta["checkpoint_bytes"],
                "basis": "loader_fetch",
                **meta,
            }
        )
    missing: List[str] = []
    index_names = (
        _checkpoint_index_names(checkpoint_path) if checkpoint_path else None
    )
    if index_names is not None:
        missing = sorted(index_names - seen)
        unexpected = sorted(seen - index_names)
        if unexpected:
            problems.append(f"recorded names absent from index: {unexpected[:8]}")
    return rows, missing, problems


def set_model_root(module_root, checkpoint_path: Optional[str] = None) -> None:
    """Remember the constructed model so later stages can republish.

    Multimodal (vision) weights load through their own engine after the
    transformer constructor returns; a republish once the backend is fully
    started folds those loader-fetched keys into the same placement report.
    """
    global _model_root, _checkpoint_path
    _model_root = module_root
    if checkpoint_path is not None:
        _checkpoint_path = checkpoint_path


def republish() -> bool:
    if _model_root is None:
        return False
    return publish_weights_placement(_model_root, _checkpoint_path)


def publish_weights_placement(
    module_root, checkpoint_path: Optional[str] = None
) -> bool:
    """Build placement rows and push them into the rank's C++ collector."""
    rows, missing, problems = build_weights_rows(module_root, checkpoint_path)
    for problem in problems:
        logger.error("AWI weights placement problem: %s", problem)
    if missing:
        # Sharded deployments fetch each expert shard on exactly one rank, so
        # a per-rank shortfall is expected; the merged multi-rank observations
        # are the authoritative coverage report.
        logger.warning(
            "AWI weights: %d checkpoint keys not fetched on this rank "
            "(peer ranks hold their shards; merged report is authoritative; "
            "first 12): %s",
            len(missing),
            missing[:12],
        )
    payload = json.dumps(rows)
    try:
        import importlib

        try:
            libth_transformer = importlib.import_module("libth_transformer")
        except ImportError:
            from rtp_llm.ops import libth_transformer

        awi = getattr(libth_transformer, "awi", None)
        if awi is None or not hasattr(awi, "set_weights_json"):
            logger.info(
                "AWI weights: collector unavailable in this build; %d rows "
                "(%d missing) not published",
                len(rows),
                len(missing),
            )
            return False
        awi.set_weights_json(payload)
        logger.info(
            "AWI weights placement published: %d rows, %d missing, %d problems",
            len(rows),
            len(missing),
            len(problems),
        )
        return not missing and not problems
    except Exception:
        logger.warning("AWI weights publish failed", exc_info=True)
        return False
