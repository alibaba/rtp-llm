import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ALLOWED_DROPPED_WEIGHT_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
    "rope.inv_freq",
    "inv_freq",
)


def _is_allowed_dropped_weight(name: str) -> bool:
    return any(
        name == suffix or name.endswith(f".{suffix}")
        for suffix in _ALLOWED_DROPPED_WEIGHT_SUFFIXES
    )


def _mark_loaded(param: nn.Parameter) -> None:
    setattr(param, "_rtp_weight_loaded", True)


class RtpModule(nn.Module):
    """Streaming checkpoint dispatcher for newloader model trees.

    Container modules inherit ``load_weights``. Specialized leaf modules may
    override it to implement sharding or layout conversion and must then own
    their post-load completeness checks.
    """

    def _get_child_module(self, name: str) -> Optional[nn.Module]:
        child = getattr(self, name, None)
        if isinstance(child, nn.Module):
            return child
        return None

    def _assign_weight(self, module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
        if "." in name:
            prefix, rest = name.split(".", 1)
            child = getattr(module, prefix, None)
            if not isinstance(child, nn.Module):
                return False
            return self._assign_weight(child, rest, tensor)
        param = getattr(module, name, None)
        if not isinstance(param, nn.Parameter):
            return False
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {module.__class__.__name__}.{name}: "
                f"expected {tuple(param.shape)}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        _mark_loaded(param)
        return True

    def _dispatch_to_module_list(
        self, module_list: nn.ModuleList, name: str, tensor: torch.Tensor
    ) -> bool:
        if "." not in name:
            return False
        index_text, rest = name.split(".", 1)
        try:
            index = int(index_text)
        except ValueError:
            return False
        if index < 0 or index >= len(module_list):
            return False
        child = module_list[index]
        child_loader = getattr(type(child), "load_weights", None)
        if child_loader is not None and child_loader is not RtpModule.load_weights:
            child.load_weights({rest: tensor})
            return True
        return self._dispatch(child, rest, tensor)

    def _dispatch(self, module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
        if "." not in name:
            return self._assign_weight(module, name, tensor)
        prefix, rest = name.split(".", 1)
        child = getattr(module, prefix, None)
        if isinstance(child, nn.ModuleList):
            return self._dispatch_to_module_list(child, rest, tensor)
        if not isinstance(child, nn.Module):
            return False
        child_loader = getattr(type(child), "load_weights", None)
        if child_loader is not None and child_loader is not RtpModule.load_weights:
            child.load_weights({rest: tensor})
            return True
        return self._dispatch(child, rest, tensor)

    def load_weights(self, weights: Any) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        dropped: List[str] = []
        for name, tensor in iterator:
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("Weights must be (str, torch.Tensor) pairs")
            if not self._dispatch(self, name, tensor):
                dropped.append(name)

        unexpected = [name for name in dropped if not _is_allowed_dropped_weight(name)]
        if unexpected:
            sample = unexpected[:10]
            suffix = f" (+{len(unexpected) - len(sample)} more)" if len(unexpected) > 10 else ""
            raise RuntimeError(
                f"{self.__class__.__name__} could not dispatch checkpoint tensors "
                f"{sample}{suffix}"
            )
        if dropped:
            logger.info(
                "%s ignored known non-persistent tensors: %s",
                self.__class__.__name__,
                dropped[:10],
            )

    def process_weights_after_loading(self) -> None:
        missing: List[str] = []

        def visit(module: nn.Module, prefix: str) -> None:
            for name, param in module.named_parameters(recurse=False):
                if getattr(param, "_rtp_skip_load_check", False):
                    continue
                if not getattr(param, "_rtp_weight_loaded", False):
                    missing.append(prefix + name)
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                child_loader = getattr(type(child), "load_weights", None)
                if child_loader is not None and child_loader is not RtpModule.load_weights:
                    continue
                visit(child, child_prefix)

        visit(self, "")
        if missing:
            sample = missing[:10]
            suffix = f" (+{len(missing) - len(sample)} more)" if len(missing) > 10 else ""
            raise RuntimeError(
                f"{self.__class__.__name__} is missing required checkpoint parameters: "
                f"{sample}{suffix}"
            )
