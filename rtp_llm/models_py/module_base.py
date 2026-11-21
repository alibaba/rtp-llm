import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from rtp_llm.models_py.read_spec import (
    LoadedCheckpointTensor,
    ReadSpec,
    unwrap_checkpoint_tensor,
)

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


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8_")


class CheckpointLoadTarget(NamedTuple):
    """Resolved destination for one mapped checkpoint tensor."""

    module: nn.Module
    name: str
    custom_loader: bool


def copy_weight_(target: torch.Tensor, source: torch.Tensor, label: str) -> None:
    """Copy a checkpoint tensor without hiding quantized layout mistakes."""
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"Shape mismatch for {label}: expected {tuple(target.shape)}, "
            f"got {tuple(source.shape)}"
        )
    if target.dtype != source.dtype:
        compatible_floats = target.is_floating_point() and source.is_floating_point()
        if (
            not compatible_floats
            or _is_float8_dtype(target.dtype)
            or _is_float8_dtype(source.dtype)
        ):
            raise TypeError(
                f"Dtype mismatch for {label}: expected {target.dtype}, "
                f"got {source.dtype}"
            )
    with torch.no_grad():
        target.copy_(source)


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

    def validate_runtime_device(self, device: torch.device) -> None:
        """Validate device-specific runtime requirements before migration."""

    def checkpoint_weight_name_filter(
        self,
    ) -> Optional[Callable[[str], bool]]:
        """Return a pre-materialization checkpoint name filter, if needed."""
        return None

    def checkpoint_read_spec(
        self, name: str, shape: tuple[int, ...]
    ) -> Optional[ReadSpec]:
        """Return this leaf's pre-materialization slice, if it is safe.

        Container traversal is implemented by ``resolve_checkpoint_read_spec``.
        Leaves should return ``None`` for layouts that still require their
        existing full-tensor ``load_weights`` implementation.
        """
        return None

    def allow_checkpoint_read_spec(self, checkpoint_name: str) -> bool:
        """Whether backend slicing may run before this model's name transforms."""
        return True

    def _resolve_checkpoint_read_spec(
        self, module: nn.Module, name: str, shape: tuple[int, ...]
    ) -> Optional[ReadSpec]:
        _, read_spec = self._resolve_checkpoint_load_plan(module, name, shape)
        return read_spec

    @staticmethod
    def _direct_checkpoint_target(
        module: nn.Module, name: str
    ) -> Optional[CheckpointLoadTarget]:
        parameter = module._parameters.get(name)
        buffer = module._buffers.get(name)
        if isinstance(parameter, nn.Parameter):
            return CheckpointLoadTarget(module, name, False)
        if (
            isinstance(buffer, torch.Tensor)
            and name not in module._non_persistent_buffers_set
        ):
            return CheckpointLoadTarget(module, name, False)
        return None

    def _resolve_checkpoint_load_plan(
        self, module: nn.Module, name: str, shape: tuple[int, ...]
    ) -> Tuple[Optional[CheckpointLoadTarget], Optional[ReadSpec]]:
        """Resolve dispatch destination and ReadSpec in one tree traversal."""
        local_spec = None
        if isinstance(module, RtpModule):
            local_spec = module.checkpoint_read_spec(name, shape)
        if "." not in name:
            return self._direct_checkpoint_target(module, name), local_spec

        prefix, rest = name.split(".", 1)
        child = module._modules.get(prefix)
        if child is None:
            child = self._fused_shard_target(module, prefix)
            if child is not None:
                child_spec = (
                    child.checkpoint_read_spec(name, shape)
                    if isinstance(child, RtpModule)
                    else None
                )
                return (
                    CheckpointLoadTarget(child, name, True),
                    local_spec if local_spec is not None else child_spec,
                )

        if isinstance(child, nn.ModuleList):
            if "." not in rest:
                return None, local_spec
            index_text, child_name = rest.split(".", 1)
            try:
                index = int(index_text)
            except ValueError:
                return None, local_spec
            if not 0 <= index < len(child):
                return None, local_spec
            target = child[index]
            child_loader = getattr(type(target), "load_weights", None)
            if child_loader is not None and child_loader is not RtpModule.load_weights:
                child_spec = (
                    target.checkpoint_read_spec(child_name, shape)
                    if isinstance(target, RtpModule)
                    else None
                )
                return (
                    CheckpointLoadTarget(target, child_name, True),
                    local_spec if local_spec is not None else child_spec,
                )
            target_plan, child_spec = self._resolve_checkpoint_load_plan(
                target, child_name, shape
            )
            return target_plan, local_spec if local_spec is not None else child_spec

        if not isinstance(child, nn.Module):
            return None, local_spec
        child_loader = getattr(type(child), "load_weights", None)
        if child_loader is not None and child_loader is not RtpModule.load_weights:
            child_spec = (
                child.checkpoint_read_spec(rest, shape)
                if isinstance(child, RtpModule)
                else None
            )
            return (
                CheckpointLoadTarget(child, rest, True),
                local_spec if local_spec is not None else child_spec,
            )
        target_plan, child_spec = self._resolve_checkpoint_load_plan(child, rest, shape)
        return target_plan, local_spec if local_spec is not None else child_spec

    def resolve_checkpoint_load_plan(
        self, name: str, shape: tuple[int, ...]
    ) -> Tuple[Optional[CheckpointLoadTarget], Optional[ReadSpec]]:
        """Resolve a mapped name once for backend slicing and later dispatch."""
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint name must be a non-empty string")
        return self._resolve_checkpoint_load_plan(self, name, tuple(shape))

    def resolve_checkpoint_read_spec(
        self, name: str, shape: tuple[int, ...]
    ) -> Optional[ReadSpec]:
        """Resolve a mapped checkpoint name to its consumer's ``ReadSpec``."""
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint name must be a non-empty string")
        return self._resolve_checkpoint_read_spec(self, name, tuple(shape))

    def requires_staged_device_postprocess(self) -> bool:
        """Whether this leaf must postprocess immediately after device migration."""
        return False

    def checkpoint_load_complete(self) -> bool:
        """Whether this leaf has all tensors needed for staged postprocessing."""
        if self.requires_staged_device_postprocess():
            raise NotImplementedError(
                f"{type(self).__name__} must implement checkpoint_load_complete() "
                "for staged streaming"
            )
        return False

    def _apply(self, fn, recurse=True):
        alias_groups = _collect_tensor_alias_groups(self, recurse=recurse)
        apply_once = _memoize_tensor_apply(fn)
        result = super()._apply(apply_once, recurse=recurse)
        _restore_tensor_aliases(alias_groups)
        return result

    def _assign_weight(self, module: nn.Module, name: str, tensor: Any) -> bool:
        tensor, _ = unwrap_checkpoint_tensor(tensor)
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

    def _mark_weight_loaded(self, name: str) -> None:
        parameter = self._parameters.get(name)
        buffer = self._buffers.get(name)
        is_parameter = isinstance(parameter, nn.Parameter)
        is_persistent_buffer = (
            isinstance(buffer, torch.Tensor)
            and name not in self._non_persistent_buffers_set
        )
        if not is_parameter and not is_persistent_buffer:
            raise KeyError(
                f"{type(self).__name__} has no loadable parameter or persistent "
                f"buffer named {name!r}"
            )
        _mark_loaded(self, name)

    @staticmethod
    def _fused_shard_target(module: nn.Module, shard_name: str):
        matches = []
        for child in module.children():
            shard_names = getattr(child, "shard_names", ())
            if shard_name in shard_names:
                matches.append(child)
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous fused shard {shard_name!r} under "
                f"{type(module).__name__}"
            )
        return matches[0] if matches else None

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
        coordinator = getattr(self, "_rtp_staged_load_coordinator", None)
        if coordinator is not None and coordinator.handles(child):
            self._load_child_weights(child, {rest: tensor})
            return True
        child_loader = getattr(type(child), "load_weights", None)
        if child_loader is not None and child_loader is not RtpModule.load_weights:
            self._load_child_weights(child, {rest: tensor})
            return True
        return self._dispatch(child, rest, tensor)

    def _load_child_weights(self, child: nn.Module, weights: Any) -> None:
        """Load one custom child through the active streaming coordinator."""
        coordinator = getattr(self, "_rtp_staged_load_coordinator", None)
        if coordinator is None:
            child.load_weights(weights)
            return
        coordinator.load_child(child, weights)

    def _prepare_child_weight_loading(self, child: nn.Module) -> None:
        """Prepare a staged child before a custom loader writes its storage."""
        coordinator = getattr(self, "_rtp_staged_load_coordinator", None)
        if coordinator is not None:
            coordinator.prepare_child(child)

    def _finalize_child_weight_loading(self, child: nn.Module) -> None:
        """Finalize a staged child after a custom loader updates completeness."""
        coordinator = getattr(self, "_rtp_staged_load_coordinator", None)
        if coordinator is not None:
            coordinator.finalize_child_if_complete(child)

    def _dispatch(self, module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
        if module is self:
            load_plan = getattr(self, "_rtp_checkpoint_load_plan", None)
            if load_plan is not None:
                target = load_plan.get(name)
                if target is not None:
                    if target.custom_loader:
                        self._load_child_weights(target.module, {target.name: tensor})
                        return True
                    return self._assign_weight(target.module, target.name, tensor)
        if "." not in name:
            return self._assign_weight(module, name, tensor)
        prefix, rest = name.split(".", 1)
        child = module._modules.get(prefix)
        if child is None:
            child = self._fused_shard_target(module, prefix)
            if child is not None:
                self._load_child_weights(child, {name: tensor})
                return True
        if isinstance(child, nn.ModuleList):
            return self._dispatch_to_module_list(child, rest, tensor)
        if not isinstance(child, nn.Module):
            return False
        coordinator = getattr(self, "_rtp_staged_load_coordinator", None)
        if coordinator is not None and coordinator.handles(child):
            self._load_child_weights(child, {rest: tensor})
            return True
        child_loader = getattr(type(child), "load_weights", None)
        if child_loader is not None and child_loader is not RtpModule.load_weights:
            self._load_child_weights(child, {rest: tensor})
            return True
        return self._dispatch(child, rest, tensor)

    def load_weights(self, weights: Any) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        dropped: List[str] = []
        for name, tensor in iterator:
            if not isinstance(name, str) or not isinstance(
                tensor, (torch.Tensor, LoadedCheckpointTensor)
            ):
                raise TypeError(
                    "Weights must be (str, torch.Tensor or "
                    "LoadedCheckpointTensor) pairs"
                )
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
