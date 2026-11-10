"""Per-rank checkpoint weight placement census (AWI evaluator contract).

The evaluator requires /awi/observations to report EVERY checkpoint key with
its actual placement: Engram keys host-RAM resident (device=cpu,
persistent_gpu_bytes=0), every other key device-resident, transformed tensors
included with checkpoint provenance. Nothing here is derived from requested
settings; every row comes from a tensor that is actually allocated when the
census is written.

Provenance chain (one per backend rank):

1. ``snapshot_loader`` runs at the end of ``ModelLoader.load_weights`` while
   the checkpoint database is still open. It records, for every checkpoint
   key the loader consumes, the descriptor output tensor it materialized
   (data_ptr, device, dtype, bytes) and the merge group it belongs to.
   Keys the deployment does not consume (e.g. disabled MTP draft keys) are
   PARKED: loaded to the rank's device and retained, so they remain
   device-resident per the offload policy instead of silently missing.
2. Bind-time transform sites (exact-dequant linears, expert dequant stacks,
   host Engram tables) call ``register_transform`` / ``register_host_tensor``
   with the real resident tensors that REPLACE the loader allocations.
3. ``finalize_and_write`` runs after the python model is fully constructed.
   It merges the three sources per checkpoint key and publishes one JSON
   document per rank under ``AWI_REPORT_DIR`` (see utils.awi_telemetry); the
   frontend aggregates the rank documents for /awi/observations.

Rows of a merged descriptor tensor (many checkpoint keys -> one resident
stack, e.g. QKV merges or the 384-expert stacks) split its bytes equally
among the group members; the row records the full group as provenance.
"""

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set

import torch

from rtp_llm.utils.awi_telemetry import write_rank_document

_ENGRAM_KEY = re.compile(r"(^|\.)engram\.")

_lock = threading.Lock()
# input data_ptr -> transform record
_transforms_by_ptr: Dict[int, Dict[str, Any]] = {}
# checkpoint key -> host-resident tensor record (Engram module)
_host_tensors: Dict[str, Dict[str, Any]] = {}
# parked unconsumed keys: retained references keep them device-resident
_parked: List[torch.Tensor] = []
_parked_rows: Dict[str, Dict[str, Any]] = {}
# snapshot state from the most recent load_weights on this rank
_snapshot: Optional[Dict[str, Any]] = None
_census_written = False


def _device_str(tensor: torch.Tensor) -> str:
    device = str(tensor.device)
    if torch.version.hip is not None and device.startswith("cuda:"):
        device = "hip:" + device.split(":", 1)[1]
    return device


def _storage_str(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu":
        return "device"
    try:
        return "pinned" if tensor.is_pinned() else "anonymous"
    except Exception:  # noqa: BLE001
        return "anonymous"


def _tensor_record(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "device": _device_str(tensor),
        "storage": _storage_str(tensor),
        "bytes": tensor.numel() * tensor.element_size(),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "ptr": tensor.data_ptr(),
    }


def register_transform(
    input_tensors: List[torch.Tensor],
    output_tensors: List[torch.Tensor],
    transform: str,
) -> None:
    """Record a bind-time replacement of loader allocations.

    Called by exact-dequant linears / expert strategies with the source
    tensors they consumed (whose data_ptrs match the loader snapshot) and
    the resident tensors that replace them.
    """
    record = {
        "transform": transform,
        "outputs": [_tensor_record(t) for t in output_tensors],
        "out_bytes": sum(
            t.numel() * t.element_size() for t in output_tensors
        ),
        "device": _device_str(output_tensors[0]) if output_tensors else "cpu",
        "storage": _storage_str(output_tensors[0]) if output_tensors else "anonymous",
    }
    with _lock:
        for tensor in input_tensors:
            _transforms_by_ptr[tensor.data_ptr()] = record


def register_host_tensor(checkpoint_key: str, tensor: torch.Tensor) -> None:
    """Record a host-RAM resident checkpoint tensor (Engram tables)."""
    record = _tensor_record(tensor)
    record["name"] = checkpoint_key
    with _lock:
        _host_tensors[checkpoint_key] = record


def _park_unconsumed(
    universe: List[str],
    consumed: Set[str],
    device: str,
    database: Any,
) -> Dict[str, Any]:
    stats = {"parked": 0, "park_failed": [], "engram_pattern": 0}
    for name in universe:
        if name in consumed or _ENGRAM_KEY.search(name):
            if _ENGRAM_KEY.search(name):
                stats["engram_pattern"] += 1
            continue
        try:
            tensor = database.load_tensor(name)
            try:
                resident = tensor.to(device)
            except Exception:  # noqa: BLE001
                # Exotic dtypes (fp8/e8m0) still park byte-identically as
                # their raw uint8 view; the census reports the true dtype.
                resident = tensor.view(torch.uint8).to(device)
            _parked.append(resident)
            record = _tensor_record(resident)
            record["dtype"] = str(tensor.dtype).replace("torch.", "")
            record["shape"] = list(tensor.shape)
            record["parked"] = True
            _parked_rows[name] = record
            stats["parked"] += 1
        except Exception as error:  # noqa: BLE001
            stats["park_failed"].append({"name": name, "error": str(error)[:200]})
    return stats


def snapshot_loader(loader: Any, weights: Any, device: str) -> None:
    """Capture the loader-side census at the end of load_weights.

    Accumulates across multiple loaders (e.g. a separate vision or draft
    loader) so the finalized census covers the union of every checkpoint
    key each loader's database enumerates. Parking is per-database, so a
    key consumed by a later loader is never in an earlier loader's universe.
    """
    global _snapshot
    try:
        database = loader._load_config.database
        universe = list(database.get_pretrain_tensor_names())
        tensor_to_weight_map, _ = loader._generate_weight_info()
        consumed: Dict[str, Any] = {}
        for name, info in tensor_to_weight_map.items():
            component = info.weight
            layer_id = info.layer_id
            tensor = None
            try:
                if layer_id is not None:
                    tensor = weights.weights[layer_id].get(component.name)
                else:
                    tensor = weights.global_weights.get(component.name)
            except Exception:  # noqa: BLE001
                tensor = None
            # Capture the record only; holding tensor references here would
            # pin the pre-transform allocations (fp8/fp4 sources) through
            # bind-time dequant and OOM the rank. Untransformed descriptor
            # tensors stay resident because the constructed modules own
            # them; transformed ones are covered by register_transform.
            consumed[name] = {
                "descriptor": getattr(component, "name", "?"),
                "layer_id": layer_id,
                "record": _tensor_record(tensor) if tensor is not None else None,
            }
        with _lock:
            already = set(consumed.keys()) | set(_parked_rows.keys())
            parked_stats = _park_unconsumed(universe, already, device, database)
            if _snapshot is None:
                _snapshot = {
                    "device": device,
                    "universe": [],
                    "consumed": {},
                    "parked_stats": {"parked": 0, "park_failed": [], "engram_pattern": 0},
                    "snapshot_ns": time.time_ns(),
                }
            _snapshot["universe"].extend(universe)
            _snapshot["consumed"].update(consumed)
            _snapshot["parked_stats"]["parked"] += parked_stats["parked"]
            _snapshot["parked_stats"]["park_failed"].extend(parked_stats["park_failed"])
            _snapshot["parked_stats"]["engram_pattern"] += parked_stats["engram_pattern"]
        logging.info(
            "[awi] census snapshot: universe=%d consumed=%d parked=%d park_failed=%d",
            len(universe),
            len(consumed),
            parked_stats["parked"],
            len(parked_stats["park_failed"]),
        )
    except Exception:  # noqa: BLE001
        logging.exception("[awi] census snapshot failed")
        if _snapshot is None:
            _snapshot = None


def _merged_group_bytes(entries: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
    """Group consumed keys by resident tensor ptr (merge groups)."""
    groups: Dict[int, List[str]] = {}
    for name, entry in entries.items():
        record = entry.get("record")
        if record is None:
            continue
        groups.setdefault(record["ptr"], []).append(name)
    return groups


def finalize_and_write() -> Optional[str]:
    """Merge snapshot + transforms + host registrations; publish the rank doc."""
    global _census_written
    if _snapshot is None:
        logging.warning("[awi] census finalize without snapshot; skipping")
        return None
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    consumed = _snapshot["consumed"]
    groups = _merged_group_bytes(consumed)
    transformed = 0
    with _lock:
        host_snapshot = dict(_host_tensors)
        transforms = dict(_transforms_by_ptr)
        parked_rows = dict(_parked_rows)

    for name, entry in consumed.items():
        record = entry.get("record")
        base = {
            "name": name,
            "provenance": {
                "descriptor": entry["descriptor"],
                "layer_id": entry["layer_id"],
            },
        }
        if record is None:
            missing.append(name)
            continue
        ptr = record["ptr"]
        group = groups.get(ptr, [name])
        transform = transforms.get(ptr)
        if transform is not None:
            transformed += 1
            share = transform["out_bytes"] // max(1, len(group))
            rows.append(
                {
                    **base,
                    "device": transform["device"],
                    "storage": transform["storage"],
                    "persistent_gpu_bytes": (
                        0 if transform["storage"] != "device" else share
                    ),
                    "resident_bytes": share,
                    "provenance": {
                        **base["provenance"],
                        "transform": transform["transform"],
                        "merged_with": sorted(group),
                        "outputs": transform["outputs"],
                    },
                }
            )
            continue
        share = record["bytes"] // max(1, len(group))
        rows.append(
            {
                **base,
                "device": record["device"],
                "storage": record["storage"],
                "persistent_gpu_bytes": 0 if record["storage"] != "device" else share,
                "resident_bytes": share,
                "dtype": record["dtype"],
                "shape": record["shape"],
                "provenance": {
                    **base["provenance"],
                    "merged_with": sorted(group) if len(group) > 1 else None,
                },
            }
        )

    for name, record in host_snapshot.items():
        rows.append(
            {
                "name": name,
                "device": record["device"],
                "storage": record["storage"],
                "persistent_gpu_bytes": (
                    0 if record["storage"] != "device" else record["bytes"]
                ),
                "resident_bytes": record["bytes"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "provenance": {"host_module": "engram", "parked": False},
            }
        )

    for name, record in parked_rows.items():
        rows.append(
            {
                "name": name,
                "device": record["device"],
                "storage": record["storage"],
                "persistent_gpu_bytes": (
                    0 if record["storage"] != "device" else record["bytes"]
                ),
                "resident_bytes": record["bytes"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "provenance": {
                    "parked": True,
                    "note": "unconsumed by this deployment; kept device-resident per offload policy",
                },
            }
        )

    universe = set(_snapshot["universe"])
    covered = {row["name"] for row in rows}
    missing.extend(sorted(universe - covered))
    # Engram-pattern keys must have a real host registration; never infer.
    for name in sorted(universe):
        if _ENGRAM_KEY.search(name) and name not in host_snapshot:
            missing.append(f"{name} (engram key without host registration)")

    document = {
        "world_rank": int(os.environ.get("WORLD_RANK", -1)),
        "pid": os.getpid(),
        "generated_ns": time.time_ns(),
        "snapshot_ns": _snapshot["snapshot_ns"],
        "device": _snapshot["device"],
        "coverage": {
            "universe": len(universe),
            "rows": len(rows),
            "transformed": transformed,
            "host_registered": len(host_snapshot),
            "parked": _snapshot["parked_stats"]["parked"],
            "park_failed": _snapshot["parked_stats"]["park_failed"],
            "missing": missing,
        },
        "rows": rows,
    }
    _census_written = True
    path = write_rank_document("weights", document)
    logging.info(
        "[awi] census written: rows=%d missing=%d parked=%d host=%d -> %s",
        len(rows),
        len(missing),
        document["coverage"]["parked"],
        len(host_snapshot),
        path,
    )
    return path


def census_written() -> bool:
    return _census_written
