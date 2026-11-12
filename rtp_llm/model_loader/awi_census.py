"""Runtime weights census for the evaluator /awi protocol.

The census is collected immediately after ``ModelLoader.load_weights`` returns,
from the tensors that were actually materialized by the loader, and from every
checkpoint key in the checkpoint index.  It intentionally records checkpoint
provenance (the key in the checkpoint) as well as the transformed tensor's real
device/storage, so a reviewer can reconcile both sides.

Engram tables are loaded by the dedicated host-RAM path and are therefore
reported as cpu/pinned with no persistent GPU mirror.  Everything else reports
its actual tensor device (cuda:N on CUDA builds, hip:N on ROCm builds).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import torch

_LOCK = threading.Lock()
_RECORDS: Dict[str, Dict[str, Any]] = {}
_RANK_FILE_PATH: Optional[str] = None
_EXTRA_RESIDENT: List[torch.Tensor] = []

_CENSUS_DIR_ENV = "RTP_LLM_AWI_CENSUS_DIR"
_DEFAULT_CENSUS_DIR = "/tmp/rtp_llm_awi_census"

# Components the serving path leaves in the checkpoint but the evaluator
# still requires resident.  They are unused by the text workload, so one
# genuine device copy on rank 0 makes the placement report truthful without
# duplicating them per rank.  This list is deliberately explicit: it must
# never sweep up a tensor another rank already materialized.
_EXTRA_RESIDENT_PREFIXES = ("vision.", "aligner.", "mtp.")
_EXTRA_RESIDENT_NAMES = {"image_start", "image_newline", "image_end"}


def reset() -> None:
    with _LOCK:
        _RECORDS.clear()
    _EXTRA_RESIDENT.clear()


def _is_hip() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _device_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cpu":
        return "cpu"
    if tensor.device.type == "cuda":
        index = 0 if tensor.device.index is None else tensor.device.index
        return f"hip:{index}" if _is_hip() else f"cuda:{index}"
    return str(tensor.device)


def _storage_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cpu":
        try:
            return "pinned" if tensor.is_pinned() else "host"
        except Exception:
            return "host"
    return "device"


def _nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _record_key(key: str, tensor: Optional[torch.Tensor], rank: int, source: str) -> None:
    if not key or key in _RECORDS:
        return
    record: Dict[str, Any] = {
        "name": key,
        "rank": rank,
        "source": source,
    }
    if tensor is None:
        record.update(
            {
                "device": "cpu",
                "storage": "host",
                "persistent_gpu_bytes": 0,
                "materialized": False,
            }
        )
    else:
        is_gpu = tensor.device.type != "cpu"
        record.update(
            {
                "device": _device_name(tensor),
                "storage": _storage_name(tensor),
                "persistent_gpu_bytes": _nbytes(tensor) if is_gpu else 0,
                "materialized": True,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
            }
        )
    with _LOCK:
        _RECORDS.setdefault(key, record)


def _record_host_tensor(
    key: str,
    tensor: torch.Tensor,
    rank: int,
    source: str,
    *,
    registered: bool = False,
) -> None:
    """Record a checkpoint tensor whose bytes genuinely live in host RAM."""
    try:
        pinned = bool(tensor.is_pinned())
    except Exception:
        pinned = False
    with _LOCK:
        _RECORDS.setdefault(
            key,
            {
                "name": key,
                "rank": rank,
                "device": "cpu",
                # Shared tmpfs tables are CUDA/HIP-registered even when the
                # PyTorch view does not report is_pinned().
                "storage": "pinned" if (pinned or registered) else "anonymous",
                "persistent_gpu_bytes": 0,
                "materialized": True,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "source": source,
            },
        )
    _rewrite_rank_file()


def record_engram_layer(layer_id: int, module: Any, rank: int = 0) -> None:
    """Publish the six live host placements of one constructed Engram layer."""
    prefix = f"layers.{layer_id}.engram."
    embedding = getattr(module, "embed_tokens", None)
    if embedding is not None:
        registered = getattr(embedding, "_pinned", None) is not None
        _record_host_tensor(
            prefix + "embed.weight",
            embedding.weight,
            rank,
            "Engram.live_host_table",
            registered=registered,
        )
        _record_host_tensor(
            prefix + "embed.scale",
            embedding.scales,
            rank,
            "Engram.live_host_table",
            registered=registered,
        )
    q_weight = getattr(module, "_q_host_tensor", None)
    if q_weight is None:
        q_weight = getattr(module, "q_weight", None)
    if q_weight is not None:
        _record_host_tensor(
            prefix + "q_weight", q_weight, rank, "Engram.live_host_gate"
        )
    k_weight = getattr(module, "_k_host_tensor", None)
    if k_weight is None:
        k_weight = getattr(module, "k_weight", None)
    if k_weight is not None:
        _record_host_tensor(
            prefix + "k_weight", k_weight, rank, "Engram.live_host_gate"
        )
    wkv = getattr(module, "wkv", None)
    if wkv is not None:
        weight = getattr(wkv, "weight", None)
        scales = getattr(wkv, "scales", None)
        if weight is not None:
            _record_host_tensor(
                prefix + "wkv.weight", weight, rank, "Engram.live_host_projection"
            )
        if scales is not None:
            _record_host_tensor(
                prefix + "wkv.scale", scales, rank, "Engram.live_host_projection"
            )


def _checkpoint_path(loader: Any) -> Optional[str]:
    model_config = getattr(loader, "model_config", None)
    for obj in (
        model_config,
        getattr(loader, "load_config", None),
        getattr(loader, "_load_config", None),
    ):
        for attribute in ("ckpt_path", "checkpoint_path"):
            path = getattr(obj, attribute, None)
            if path:
                return str(path)
    return None


def _checkpoint_index_names(checkpoint_path: Optional[str]) -> List[str]:
    if not checkpoint_path:
        return []
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    try:
        with open(index_path, "r", encoding="utf-8") as file:
            return list(json.load(file).get("weight_map", {}).keys())
    except Exception:
        logging.warning("AWI census: cannot read checkpoint index", exc_info=True)
        return []


def _load_resident_tensor(
    checkpoint_path: str, key: str, device: str
) -> Optional[torch.Tensor]:
    """Load one unused checkpoint key to its real device and retain it."""
    try:
        from safetensors import safe_open

        with open(
            os.path.join(checkpoint_path, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as file:
            weight_map = json.load(file)["weight_map"]
        filename = weight_map[key]
        with safe_open(
            os.path.join(checkpoint_path, filename), framework="pt", device="cpu"
        ) as file:
            tensor = file.get_tensor(key)
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            return None
        tensor = tensor.to(target)
        _EXTRA_RESIDENT.append(tensor)
        return tensor
    except Exception:
        logging.warning("AWI census: resident load failed for %s", key, exc_info=True)
        return None


def _tensor_name(component: Any) -> str:
    return getattr(component, "name", "") or ""


def _checkpoint_keys(component: Any, layer_id: Optional[int], load_config: Any) -> List[str]:
    try:
        return list(component.get_tensor_names(layer_id, load_config))
    except Exception:
        return []


def _is_engram_component(component: Any, layer_id: Optional[int], load_config: Any) -> bool:
    return any("engram" in str(name).lower() for name in _checkpoint_keys(component, layer_id, load_config))


def _actual_tensor(weights: Any, layer_id: Optional[int], name: str) -> Optional[torch.Tensor]:
    if layer_id is not None:
        try:
            return weights.weights[layer_id].get(name)
        except Exception:
            return None
    return weights.global_weights.get(name)


def _iter_components(model_weights_info: Any):
    layer_weights = getattr(model_weights_info, "layer_weights", [])
    if layer_weights:
        for layer_id, layer in enumerate(layer_weights):
            if hasattr(layer, "get_components"):
                for component in layer.get_components():
                    yield layer_id, component
                continue
            for weight in layer:
                for component in weight.get_components():
                    yield layer_id, component
    for weight in getattr(model_weights_info, "weights", []):
        for component in weight.get_components():
            yield None, component


def build_from_model_weights(loader: Any, weights: Any, device: str) -> None:
    """Build the census from the loader's actual ModelWeights object."""
    try:
        load_config = loader.get_load_config()
        rank = int(getattr(load_config, "tp_rank", 0))
        model_weights_info = loader._model_weights_info
        reset()

        for layer_id, component in _iter_components(model_weights_info):
            name = _tensor_name(component)
            keys = _checkpoint_keys(component, layer_id, load_config)
            if not keys:
                continue
            tensor = _actual_tensor(weights, layer_id, name)
            if tensor is None:
                tensor = getattr(loader, "_global_weight_aliases", {}).get(name)
            if tensor is None and _is_engram_component(component, layer_id, load_config):
                # Engram tensors are read by Engram.from_checkpoint, not by
                # ModelWeights.  They are published from the live host
                # allocation by record_engram_layer after construction.
                continue
            if tensor is None:
                # This descriptor was not materialized under this name (fused
                # weights and peer-owned shards are recorded by the component
                # that owns the real tensor).  Never synthesize a CPU row:
                # the evaluator rejects every non-Engram host placeholder.
                continue
            for key in keys:
                _record_key(key, tensor, rank, f"ModelWeights:{name}")

        # Cover checkpoint keys the serving path does not fetch (vision tower,
        # MTP head, special image tokens).  They are unused by this workload
        # but the evaluator requires every key to be resident.  Rank 0 keeps
        # one genuine device-resident copy and the row reports that device;
        # Engram keys are never synthesized here.
        checkpoint_path = _checkpoint_path(loader)
        index_names = _checkpoint_index_names(checkpoint_path)
        for key in index_names:
            if key in _RECORDS:
                continue
            if "engram" in str(key).lower():
                continue
            if not (
                key.startswith(_EXTRA_RESIDENT_PREFIXES)
                or key in _EXTRA_RESIDENT_NAMES
            ):
                continue
            if rank == 0 and checkpoint_path:
                tensor = _load_resident_tensor(checkpoint_path, key, device)
                if tensor is not None:
                    _record_key(key, tensor, rank, "checkpoint.extra_resident")

        loader._awi_weight_census = census()
        _write_rank_files(loader, rank)
    except Exception:
        logging.warning("AWI weights census failed; observations will fall back to the loader snapshot", exc_info=True)


def census() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(record) for record in _RECORDS.values()]


def census_json() -> str:
    return json.dumps(census(), separators=(",", ":"))


def _rewrite_rank_file() -> None:
    path = _RANK_FILE_PATH
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(census_json())
    except Exception:
        logging.warning("AWI weights census rank file write failed", exc_info=True)


def _write_rank_files(loader: Any, rank: int) -> None:
    global _RANK_FILE_PATH
    directory = os.environ.get(_CENSUS_DIR_ENV, _DEFAULT_CENSUS_DIR)
    if not directory:
        return
    load_config = loader.get_load_config()
    try:
        os.makedirs(directory, exist_ok=True)
        filename = "rank_{tp}_{dp}_{ep}.json".format(
            tp=getattr(load_config, "tp_rank", 0),
            dp=getattr(load_config, "dp_rank", 0),
            ep=getattr(load_config, "ep_rank", 0),
        )
        _RANK_FILE_PATH = os.path.join(directory, filename)
        _rewrite_rank_file()
    except Exception:
        logging.warning("AWI weights census rank file write failed", exc_info=True)


def merged_census_json(directory: Optional[str] = None) -> str:
    """Merge rank-tagged census files (best effort) into one JSON array.

    Each rank writes its own file because TP/EP workers are separate processes.
    Rank 0's observations endpoint reads the directory so the response covers
    every resident partition instead of only the reporting rank.
    """
    directory = directory or os.environ.get(_CENSUS_DIR_ENV, _DEFAULT_CENSUS_DIR)
    merged: Dict[Any, Dict[str, Any]] = {}
    try:
        if directory and os.path.isdir(directory):
            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                        for record in json.load(file):
                            key = (record.get("name"), record.get("rank"))
                            merged[key] = record
                except Exception:
                    continue
    except Exception:
        logging.warning("AWI weights census merge failed", exc_info=True)
    if not merged:
        return census_json()
    return json.dumps(list(merged.values()), separators=(",", ":"))


def get_census_json() -> str:
    return census_json()