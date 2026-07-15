import logging
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ALLOWED_DROPPED_WEIGHT_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
    "rope.inv_freq",
)


def _is_allowed_dropped_weight(name: str) -> bool:
    return any(
        name == suffix or name.endswith(f".{suffix}")
        for suffix in _ALLOWED_DROPPED_WEIGHT_SUFFIXES
    )


def _mark_loaded(module: nn.Module, name: str) -> None:
    loaded = getattr(module, "_rtp_loaded_weight_names", None)
    if loaded is None:
        loaded = set()
        setattr(module, "_rtp_loaded_weight_names", loaded)
    loaded.add(name)


def collect_loaded_tensor_ids(module: nn.Module) -> set:
    loaded_tensor_ids = set()
    for current in module.modules():
        loaded_names = getattr(current, "_rtp_loaded_weight_names", set())
        for name, param in current.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            if name in loaded_names:
                loaded_tensor_ids.add(id(param))
        for name, buffer in current.named_buffers(
            recurse=False, remove_duplicate=False
        ):
            if name in loaded_names and buffer is not None:
                loaded_tensor_ids.add(id(buffer))
    return loaded_tensor_ids


def _collect_tensor_alias_groups(module: nn.Module, recurse: bool = True):
    aliases = {}
    modules = module.modules() if recurse else (module,)
    for current in modules:
        for name, tensor in current.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            aliases.setdefault(id(tensor), []).append(("parameter", current, name))
        for name, tensor in current.named_buffers(
            recurse=False, remove_duplicate=False
        ):
            if tensor is not None:
                aliases.setdefault(id(tensor), []).append(("buffer", current, name))
    return [
        registrations for registrations in aliases.values() if len(registrations) > 1
    ]


def _restore_tensor_aliases(alias_groups) -> None:
    for registrations in alias_groups:
        parameter_registrations = [
            item for item in registrations if item[0] == "parameter"
        ]
        for _, module, name in parameter_registrations:
            if not isinstance(module._parameters[name], nn.Parameter):
                raise RuntimeError(
                    f"Parameter alias {type(module).__name__}.{name} lost Parameter type"
                )
        master_kind, master_module, master_name = (
            parameter_registrations[0] if parameter_registrations else registrations[0]
        )
        master_storage = (
            master_module._parameters
            if master_kind == "parameter"
            else master_module._buffers
        )
        shared = master_storage[master_name]
        if shared is None:
            raise RuntimeError(
                f"Shared {master_kind} {master_name!r} disappeared during migration"
            )
        for kind, current, name in registrations:
            storage = current._parameters if kind == "parameter" else current._buffers
            storage[name] = shared


def _memoize_tensor_apply(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    converted: Dict[int, torch.Tensor] = {}

    def apply_once(tensor: torch.Tensor) -> torch.Tensor:
        tensor_id = id(tensor)
        if tensor_id not in converted:
            converted[tensor_id] = fn(tensor)
        return converted[tensor_id]

    return apply_once


class RtpModule(nn.Module):
    """Streaming checkpoint dispatcher for newloader model trees.

    Container modules inherit ``load_weights``. Specialized leaf modules may
    override it to implement sharding or layout conversion and must then own
    their post-load completeness checks.
    """

    def _apply(self, fn, recurse=True):
        alias_groups = _collect_tensor_alias_groups(self, recurse=recurse)
        apply_once = _memoize_tensor_apply(fn)
        result = super()._apply(apply_once, recurse=recurse)
        _restore_tensor_aliases(alias_groups)
        return result

    def _assign_weight(
        self, module: nn.Module, name: str, tensor: torch.Tensor
    ) -> bool:
        if "." in name:
            prefix, rest = name.split(".", 1)
            child = module._modules.get(prefix)
            if not isinstance(child, nn.Module):
                return False
            return self._assign_weight(child, rest, tensor)
        parameter = module._parameters.get(name)
        buffer = module._buffers.get(name)
        is_parameter = name in module._parameters and isinstance(
            parameter, nn.Parameter
        )
        is_buffer = (
            name in module._buffers
            and name not in module._non_persistent_buffers_set
            and isinstance(buffer, torch.Tensor)
        )
        if not is_parameter and not is_buffer:
            return False
        target = parameter if is_parameter else buffer
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {module.__class__.__name__}.{name}: "
                f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )
        if target.dtype != tensor.dtype and not (
            target.is_floating_point() and tensor.is_floating_point()
        ):
            raise TypeError(
                f"Dtype mismatch for {module.__class__.__name__}.{name}: "
                f"expected {target.dtype}, got {tensor.dtype}"
            )
        with torch.no_grad():
            target.copy_(tensor)
        _mark_loaded(module, name)
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
        child = module._modules.get(prefix)
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
            suffix = (
                f" (+{len(unexpected) - len(sample)} more)"
                if len(unexpected) > 10
                else ""
            )
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

    def validate_weights_loaded(self, loaded_tensor_ids=None) -> None:
        missing: List[str] = []
        if loaded_tensor_ids is None:
            loaded_tensor_ids = collect_loaded_tensor_ids(self)

        def visit(module: nn.Module, prefix: str) -> None:
            for name, param in module.named_parameters(recurse=False):
                if id(param) not in loaded_tensor_ids:
                    missing.append(prefix + name)
            for name, buffer in module.named_buffers(recurse=False):
                if name in module._non_persistent_buffers_set or buffer is None:
                    continue
                if id(buffer) not in loaded_tensor_ids:
                    missing.append(prefix + name)
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                child_loader = getattr(type(child), "load_weights", None)
                if (
                    child_loader is not None
                    and child_loader is not RtpModule.load_weights
                ):
                    continue
                visit(child, child_prefix)

        visit(self, "")
        if missing:
            sample = missing[:10]
            suffix = (
                f" (+{len(missing) - len(sample)} more)" if len(missing) > 10 else ""
            )
            raise RuntimeError(
                f"{self.__class__.__name__} is missing required checkpoint parameters: "
                f"{sample}{suffix}"
            )

    def process_weights_after_loading(self) -> None:
        """Perform device-dependent layout conversion after integrity validation."""
