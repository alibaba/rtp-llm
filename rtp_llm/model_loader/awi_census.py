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

_CENSUS_DIR_ENV = "RTP_LLM_AWI_CENSUS_DIR"
_DEFAULT_CENSUS_DIR = "/tmp/rtp_llm_awi_census"


def reset() -> None:
    with _LOCK:
        _RECORDS.clear()


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


def _record_engram_key(key: str, rank: int, source: str) -> None:
    with _LOCK:
        _RECORDS.setdefault(
            key,
            {
                "name": key,
                "rank": rank,
                "device": "cpu",
                # The V4.1 Engram loader maps checkpoint tensors into shared,
                # CUDA-registered host pages before graph capture.
                "storage": "pinned",
                "persistent_gpu_bytes": 0,
                "materialized": True,
                "source": source,
            },
        )


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
                for key in keys:
                    _record_engram_key(key, rank, "ModelLoader:engram_host_pinned")
                continue
            for key in keys:
                _record_key(key, tensor, rank, f"ModelWeights:{name}")

        # Checkpoint keys that are neither an Engram host table nor a tensor
        # materialized by ModelWeights are still part of the checkpoint.  Keep
        # them visible with their honest (non-device) placement instead of
        # silently dropping them; normal dense checkpoints do not hit this.
        try:
            all_keys = loader.get_load_config().database.get_pretrain_tensor_names()
        except Exception:
            all_keys = []
        for key in all_keys:
            if key in _RECORDS:
                continue
            if "engram" in str(key).lower():
                _record_engram_key(key, rank, "checkpoint.engram_host_pinned")
            else:
                _record_key(key, None, rank, "checkpoint.not_materialized_by_ModelWeights")

        loader._awi_weight_census = census()
        _write_rank_files(loader, rank)
    except Exception:
        logging.warning("AWI weights census failed; observations will fall back to the loader snapshot", exc_info=True)


def census() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(record) for record in _RECORDS.values()]


def census_json() -> str:
    return json.dumps(census(), separators=(",", ":"))


def _write_rank_files(loader: Any, rank: int) -> None:
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
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as file:
            file.write(census_json())
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