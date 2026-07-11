import glob
import os
import re
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import torch


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
                (prefix_mapping or {}).items(), key=lambda item: len(item[0]), reverse=True
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
        if self.include_patterns and not any(p.search(name) for p in self.include_patterns):
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


def _load_safetensors(path: str, device: str) -> Iterator[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device=device) as handle:
        for name in handle.keys():
            yield name, handle.get_tensor(name)


def _load_pytorch(path: str, device: str) -> Iterator[Tuple[str, torch.Tensor]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict = _unwrap_pytorch_state_dict(payload)
    for name, tensor in state_dict.items():
        if isinstance(name, str) and isinstance(tensor, torch.Tensor):
            yield name, tensor


def get_all_weights(
    ckpt_paths: List[str], device: str = "cpu"
) -> Iterator[Tuple[str, torch.Tensor]]:
    seen = set()
    for path in ckpt_paths:
        if path.endswith(".safetensors"):
            weights = _load_safetensors(path, device)
        elif path.endswith((".bin", ".pt")):
            weights = _load_pytorch(path, device)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        for name, tensor in weights:
            if name in seen:
                raise RuntimeError(
                    f"Checkpoint tensor {name!r} appears in more than one shard"
                )
            seen.add(name)
            yield name, tensor


def discover_ckpt_files(model_path: str) -> List[str]:
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        files = sorted(glob.glob(os.path.join(model_path, pattern)))
        if pattern == "*.bin":
            files = [
                path
                for path in files
                if "optimizer" not in os.path.basename(path).lower()
            ]
        if files:
            return files
    return []
