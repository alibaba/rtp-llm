import glob
import json
import ntpath
import os
import re
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import torch

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_PYTORCH_INDEX = "pytorch_model.bin.index.json"
_CONSOLIDATED_RANK_RE = re.compile(r"^consolidated[._-](\d+)(?:[._-]|$)", re.IGNORECASE)
_EXCLUDED_WEIGHT_FILES = {
    "adapter_model.bin",
    "adapter_model.safetensors",
    "optimizer.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
    "training_args.bin",
}


class WeightsMapper:
    def __init__(
        self,
        exact_mapping: Optional[Dict[str, str]] = None,
        prefix_mapping: Optional[Dict[str, str]] = None,
        regex_mapping: Optional[List[Tuple[str, str]]] = None,
    ):
        self.exact_mapping = exact_mapping or {}
        self.prefix_mapping = dict(
            sorted(
                (prefix_mapping or {}).items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        self.regex_mapping = regex_mapping or []

    def map_name(self, name: str) -> str:
        if name in self.exact_mapping:
            return self.exact_mapping[name]
        for old_prefix, new_prefix in self.prefix_mapping.items():
            if name.startswith(old_prefix):
                return new_prefix + name[len(old_prefix) :]
        for pattern, replacement in self.regex_mapping:
            mapped = re.sub(pattern, replacement, name)
            if mapped != name:
                return mapped
        return name

    def apply(
        self, weights: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            yield self.map_name(name), tensor


class WeightsFilter:
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.include_patterns = [re.compile(p) for p in (include_patterns or [])]
        self.exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]

    def should_load(self, name: str) -> bool:
        if self.include_patterns and not any(
            p.search(name) for p in self.include_patterns
        ):
            return False
        return not any(p.search(name) for p in self.exclude_patterns)

    def apply(
        self, weights: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if self.should_load(name):
                yield name, tensor


def _unwrap_pytorch_state_dict(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"PyTorch checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    for key in ("state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def _load_safetensors(
    path: str,
    device: str,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device=device) as handle:
        for name in handle.keys():
            if name_filter is not None and not name_filter(name):
                continue
            yield name, handle.get_tensor(name)


def _load_pytorch(
    path: str,
    device: str,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict = _unwrap_pytorch_state_dict(payload)
    for name, tensor in state_dict.items():
        if (
            isinstance(name, str)
            and isinstance(tensor, torch.Tensor)
            and (name_filter is None or name_filter(name))
        ):
            yield name, tensor


def get_all_weights(
    ckpt_paths: List[str],
    device: str = "cpu",
    name_filter: Optional[Callable[[str], bool]] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    seen = set()
    for path in ckpt_paths:
        if path.endswith(".safetensors"):
            weights = _load_safetensors(path, device, name_filter)
        elif path.endswith((".bin", ".pt", ".pth")):
            weights = _load_pytorch(path, device, name_filter)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        for name, tensor in weights:
            if name in seen:
                raise RuntimeError(
                    f"Checkpoint tensor {name!r} appears in more than one shard"
                )
            seen.add(name)
            yield name, tensor


def _files_from_index(model_path: str, index_name: str) -> Optional[List[str]]:
    index_path = os.path.join(model_path, index_name)
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to read checkpoint index {index_path}: {exc}"
        ) from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index {index_path} has no non-empty weight_map")

    expected_suffix = ".safetensors" if index_name == _SAFETENSORS_INDEX else ".bin"
    model_root = os.path.realpath(model_path)
    files = []
    seen = set()
    for shard_name in weight_map.values():
        if not isinstance(shard_name, str) or not shard_name.endswith(expected_suffix):
            raise ValueError(
                f"Checkpoint index {index_path} contains invalid shard {shard_name!r}"
            )
        # Validate index text lexically. A symlink already present under model_root
        # is a trusted filesystem entry and may point to a content-addressed blob.
        path_parts = re.split(r"[\\/]", shard_name)
        if (
            os.path.isabs(shard_name)
            or ntpath.isabs(shard_name)
            or os.pardir in path_parts
        ):
            raise ValueError(
                f"Checkpoint index {index_path} references a path outside model directory: "
                f"{shard_name!r}"
            )

        normalized_name = os.path.normpath(shard_name)
        shard_path = os.path.join(model_root, normalized_name)
        try:
            inside_model = os.path.commonpath([model_root, shard_path]) == model_root
        except ValueError:
            inside_model = False
        if not inside_model:
            raise ValueError(
                f"Checkpoint index {index_path} references a path outside model directory: "
                f"{shard_name!r}"
            )
        if not _is_model_weight_file(normalized_name, allow_consolidated=True):
            raise ValueError(
                f"Checkpoint index {index_path} references non-model file {shard_name!r}"
            )
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"Checkpoint index {index_path} references missing shard {shard_name!r}"
            )
        shard_identity = os.path.realpath(shard_path)
        if shard_identity not in seen:
            seen.add(shard_identity)
            files.append(shard_path)
    return files


def _is_consolidated_weight_file(path: str) -> bool:
    return os.path.basename(path).lower().startswith("consolidated")


def is_rank_local_checkpoint(files: List[str]) -> bool:
    return any(
        _CONSOLIDATED_RANK_RE.match(os.path.basename(path)) is not None
        for path in files
    )


def _is_model_weight_file(path: str, allow_consolidated: bool = False) -> bool:
    name = os.path.basename(path).lower()
    if name in _EXCLUDED_WEIGHT_FILES:
        return False
    excluded_prefixes = [
        "adapter_model",
        "optimizer",
        "rng_state",
        "scheduler",
        "training_args",
    ]
    if not allow_consolidated:
        excluded_prefixes.append("consolidated")
    return not name.startswith(tuple(excluded_prefixes))


def _select_consolidated_files(
    files: List[str], tp_rank: int, tp_size: int
) -> List[str]:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (tp_rank, tp_size)
    ):
        raise TypeError("tp_rank and tp_size must be integers")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"Invalid TP partition for consolidated checkpoint: "
            f"rank={tp_rank}, size={tp_size}"
        )

    ranked_files = {}
    unranked_files = []
    for path in files:
        match = _CONSOLIDATED_RANK_RE.match(os.path.basename(path))
        if match is None:
            unranked_files.append(path)
            continue
        rank = int(match.group(1))
        if rank in ranked_files:
            raise ValueError(f"Duplicate consolidated checkpoint rank {rank}")
        ranked_files[rank] = path

    if not ranked_files:
        if len(unranked_files) != 1:
            raise ValueError("Multiple unranked consolidated checkpoints are ambiguous")
        return unranked_files
    if unranked_files:
        raise ValueError("Ranked and unranked consolidated checkpoints cannot be mixed")
    if len(ranked_files) != tp_size:
        raise ValueError(
            f"Found {len(ranked_files)} consolidated rank files, but tp_size={tp_size}; "
            "all checkpoint ranks are required"
        )

    expected_ranks = set(range(tp_size))
    if set(ranked_files) != expected_ranks:
        raise ValueError(
            f"Consolidated checkpoint ranks must be contiguous {sorted(expected_ranks)}, "
            f"got {sorted(ranked_files)}"
        )
    return [ranked_files[tp_rank]]


def discover_ckpt_files(
    model_path: str, tp_rank: int = 0, tp_size: int = 1
) -> List[str]:
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")
    for index_name in (_SAFETENSORS_INDEX, _PYTORCH_INDEX):
        indexed_files = _files_from_index(model_path, index_name)
        if indexed_files is not None:
            return indexed_files
    consolidated_by_format = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        candidates = sorted(glob.glob(os.path.join(model_path, pattern)))
        files = [path for path in candidates if _is_model_weight_file(path)]
        if files:
            return files
        consolidated = [
            path
            for path in candidates
            if _is_consolidated_weight_file(path)
            and _is_model_weight_file(path, allow_consolidated=True)
        ]
        if consolidated:
            consolidated_by_format.append(consolidated)
    if consolidated_by_format:
        return _select_consolidated_files(consolidated_by_format[0], tp_rank, tp_size)
    return []
