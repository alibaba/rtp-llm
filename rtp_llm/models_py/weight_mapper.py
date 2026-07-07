import re
from typing import Dict, Iterator, List, Optional, Tuple

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
                (prefix_mapping or {}).items(), key=lambda x: len(x[0]), reverse=True
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
            new_name = re.sub(pattern, replacement, name)
            if new_name != name:
                return new_name
        return name

    def apply(
        self, weights_iterator: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights_iterator:
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
        if self.include_patterns:
            if not any(p.search(name) for p in self.include_patterns):
                return False
        if self.exclude_patterns:
            if any(p.search(name) for p in self.exclude_patterns):
                return False
        return True

    def apply(
        self, weights_iterator: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights_iterator:
            if self.should_load(name):
                yield name, tensor


def get_all_weights(
    ckpt_paths: List[str], device: str = "cpu"
) -> Iterator[Tuple[str, torch.Tensor]]:
    for path in ckpt_paths:
        if path.endswith(".safetensors"):
            yield from _load_safetensors(path, device)
        elif path.endswith(".bin") or path.endswith(".pt"):
            yield from _load_pytorch(path, device)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")


def _load_safetensors(path: str, device: str) -> Iterator[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device=device) as f:
        for name in f.keys():
            yield name, f.get_tensor(name)


def _unwrap_pytorch_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError(f"PyTorch checkpoint payload must be a dict, got {type(payload).__name__}")
    for key in ("state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _load_pytorch(path: str, device: str) -> Iterator[Tuple[str, torch.Tensor]]:
    state_dict = _unwrap_pytorch_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            yield name, tensor


def discover_ckpt_files(model_path: str) -> List[str]:
    import glob
    import os

    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if safetensor_files:
        return safetensor_files

    bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
    if bin_files:
        bin_files = [f for f in bin_files if "optimizer" not in os.path.basename(f)]
        if bin_files:
            return bin_files

    pt_files = sorted(glob.glob(os.path.join(model_path, "*.pt")))
    if pt_files:
        return pt_files

    return []
