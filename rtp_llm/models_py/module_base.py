import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _default_build_redirect(self) -> Dict[str, Tuple[Any, int]]:
    redirect = {}
    for name, child in self.named_children():
        if hasattr(child, "shard_names") and child.shard_names:
            for idx, shard_name in enumerate(child.shard_names):
                redirect[shard_name] = (child, idx, name)
    return redirect


def _default_is_fused_target(self, child: nn.Module) -> bool:
    return (
        hasattr(child, "shard_names")
        and child.shard_names
        and hasattr(child, "load_weights")
    )


def _default_load_weights(self, weights: Any):
    """Per-tensor streaming dispatch: each tensor immediately walks the module
    tree to its leaf and copies directly.  No intermediate dict buffering, so
    peak memory stays at model-params + one ckpt tensor."""
    if isinstance(weights, dict):
        weights_iter = iter(weights.items())
    else:
        weights_iter = weights

    redirect = self._build_redirect()
    dropped: List[str] = []
    count = 0

    for full_name, tensor in weights_iter:
        count += 1

        if "." not in full_name:
            if hasattr(self, full_name):
                param = getattr(self, full_name)
                if isinstance(param, nn.Parameter):
                    param.data.copy_(tensor)
                else:
                    dropped.append(full_name)
            else:
                dropped.append(full_name)
            continue

        prefix, rest = full_name.split(".", 1)

        if prefix in redirect:
            layer, shard_id, target_name = redirect[prefix]
            layer.load_weights({f"{prefix}.{rest}": tensor})
        else:
            child = self._get_child_module(prefix)
            if child is None:
                dropped.append(full_name)
                continue

            if isinstance(child, nn.ModuleList):
                _dispatch_single_to_module_list(self, child, rest, tensor)
            elif hasattr(child, "load_weights"):
                child.load_weights({rest: tensor})
            else:
                self._assign_weight(child, rest, tensor)

    if dropped:
        cls_name = self.__class__.__name__
        sample = dropped[:10]
        more = (
            f" (+{len(dropped) - len(sample)} more)"
            if len(dropped) > len(sample)
            else ""
        )
        logger.warning(
            "[%s] %d weight(s) had no matching submodule and were dropped: %s%s",
            cls_name,
            len(dropped),
            sample,
            more,
        )


def _dispatch_single_to_module_list(
    self, module_list: nn.ModuleList, name: str, tensor: torch.Tensor
):
    if "." not in name:
        return
    idx_str, rest = name.split(".", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return
    if idx >= len(module_list):
        return
    child = module_list[idx]
    if hasattr(child, "load_weights"):
        child.load_weights({rest: tensor})
    else:
        self._assign_weight(child, rest, tensor)


def _default_get_child_module(self, name: str) -> Optional[nn.Module]:
    if hasattr(self, name):
        return getattr(self, name)
    try:
        idx = int(name)
        for attr_name, attr in self.named_children():
            if isinstance(attr, nn.ModuleList) and idx < len(attr):
                return attr[idx]
    except (ValueError, IndexError):
        pass
    return None


def _default_assign_weight(self, module: nn.Module, name: str, tensor: torch.Tensor):
    if "." in name:
        prefix, rest = name.split(".", 1)
        child = getattr(module, prefix, None)
        if child is not None:
            _default_assign_weight(self, child, rest, tensor)
    else:
        if hasattr(module, name):
            param = getattr(module, name)
            if isinstance(param, nn.Parameter):
                param.data.copy_(tensor)


def _default_groupby_prefix(
    self, weights: Iterator[Tuple[str, torch.Tensor]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    grouped = defaultdict(dict)
    for full_name, tensor in weights:
        if "." in full_name:
            prefix, rest = full_name.split(".", 1)
            grouped[prefix][rest] = tensor
        else:
            grouped["_self_"][full_name] = tensor
    return dict(grouped)


def rtp_module(cls):
    if (
        not hasattr(cls, "_build_redirect")
        or cls._build_redirect is _default_build_redirect
    ):
        cls._build_redirect = _default_build_redirect

    if (
        not hasattr(cls, "_groupby_prefix")
        or cls._groupby_prefix is _default_groupby_prefix
    ):
        cls._groupby_prefix = _default_groupby_prefix

    if (
        not hasattr(cls, "_is_fused_target")
        or cls._is_fused_target is _default_is_fused_target
    ):
        cls._is_fused_target = _default_is_fused_target

    if (
        not hasattr(cls, "_get_child_module")
        or cls._get_child_module is _default_get_child_module
    ):
        cls._get_child_module = _default_get_child_module

    if (
        not hasattr(cls, "_assign_weight")
        or cls._assign_weight is _default_assign_weight
    ):
        cls._assign_weight = _default_assign_weight

    if "load_weights" not in cls.__dict__:
        cls.load_weights = _default_load_weights

    return cls
