"""Explicit target-owned parameter loans for speculative draft models.

Owners are held strongly, never registered as borrower children. Dense already
loaded inference Parameters only: packed weights need a weight/scale contract.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from rtp_llm.models_py.module_base import _mark_loaded, collect_loaded_tensor_ids


class FixedHostCheckpointStorage:
    """Leaf whose parameters are deliberately resident on CPU even in a GPU model."""

    def _apply(self, fn, recurse=True):
        return self


def _loan_device(model, name, binding, target):
    def leaf(root, path):
        prefix, _, _ = path.rpartition(".")
        return root.get_submodule(prefix) if prefix else root

    destination_host = isinstance(leaf(model, name), FixedHostCheckpointStorage)
    source_host = isinstance(
        leaf(binding.owner, binding.source_path), FixedHostCheckpointStorage
    )
    if source_host != destination_host:
        raise ValueError(
            f"Borrowed parameter {name} host residency does not match destination"
        )
    return torch.device("cpu") if destination_host else target


@dataclass(frozen=True)
class BorrowedParameter:
    owner: nn.Module
    source_path: str

    def parameter(self) -> nn.Parameter:
        return self.owner.get_parameter(self.source_path)


def aliases_from_runtime_exports(owner, exports, destination_paths):
    """Resolve existing runtime tensor aliases to real owner parameter paths."""
    parameters = dict(owner.named_parameters(remove_duplicate=False))
    aliases = {}
    for role, tensor in exports.items():
        destination = destination_paths.get(role)
        if not destination:
            raise ValueError(
                f"NewLoader model has no parameter path for borrowed role {role!r}"
            )
        source = next(
            (name for name, parameter in parameters.items() if parameter is tensor),
            None,
        )
        if source is None:
            raise ValueError(
                f"Borrowed runtime weight {role!r} must be owned by a loaded model Parameter"
            )
        if destination in aliases:
            raise ValueError(
                f"Duplicate borrowed destination parameter {destination!r}"
            )
        aliases[destination] = BorrowedParameter(owner, source)
    return aliases


def borrowed_parameter_ids(model: nn.Module) -> set[int]:
    bindings = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    return {id(binding.parameter()) for binding in bindings.values()}


def validate_borrowed_parameters(model: nn.Module, target_device=None) -> None:
    bindings = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    if not bindings:
        return
    snapshots = getattr(model, "_rtp_borrowed_parameter_snapshots", {})
    device = None if target_device is None else torch.device(target_device)
    if device is not None and device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for name, binding in bindings.items():
        current, owned = model.get_parameter(name), binding.parameter()
        original, pointer, source_device, dtype, shape = snapshots[name]
        if (
            current is not original
            or owned is not original
            or owned.data_ptr() != pointer
            or owned.device != source_device
            or owned.dtype != dtype
            or tuple(owned.shape) != shape
        ):
            raise RuntimeError(f"Borrowed parameter {name} lost owner storage identity")
        expected = _loan_device(model, name, binding, device)
        if expected is not None and (
            owned.device.type != expected.type
            or (expected.index is not None and owned.device.index != expected.index)
        ):
            raise ValueError(
                f"Borrowed parameter {name} cannot migrate owner storage to {expected}"
            )


def bind_borrowed_parameters(
    model: nn.Module, bindings: Mapping[str, BorrowedParameter], target_device: str
) -> None:
    if not bindings:
        return
    prepare = getattr(model, "prepare_borrowed_parameters", None)
    if not callable(prepare):
        raise TypeError(
            f"{type(model).__name__} must declare prepare_borrowed_parameters to validate postprocessing and runtime semantics"
        )
    device = torch.device(target_device)
    if device.type == "cuda" and device.index is None:
        # model.to("cuda") resolves to the CURRENT CUDA/HIP device, not to the
        # owner's device. Resolve it before binding, otherwise migration could
        # silently move shared owner storage to a different rank's device.
        device = torch.device("cuda", torch.cuda.current_device())
    resolved = []
    # Check all descriptors before changing any destination or owner storage.
    for name, binding in bindings.items():
        if not isinstance(binding, BorrowedParameter):
            raise TypeError(f"Invalid borrowed parameter descriptor for {name}")
        if binding.owner is model:
            raise ValueError("A borrowed parameter must have a distinct target owner")
        source = binding.parameter()
        destination = model.get_parameter(name)
        if id(source) not in collect_loaded_tensor_ids(binding.owner):
            raise ValueError(
                f"Borrowed owner parameter {binding.source_path} is not loaded"
            )
        if source.requires_grad:
            raise ValueError("Borrowed inference parameters must not require gradients")
        if source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "Quantized borrowed parameters require an explicit scale ownership contract"
            )
        if source.dtype != destination.dtype or source.shape != destination.shape:
            raise ValueError(
                f"Borrowed parameter {name} shape/dtype does not match destination"
            )
        expected = _loan_device(model, name, binding, device)
        if source.device.type != expected.type or (
            expected.index is not None and source.device.index != expected.index
        ):
            raise ValueError(
                f"Borrowed parameter {name} owner must already be on {expected}"
            )
        module_path, _, local_name = name.rpartition(".")
        leaf = model.get_submodule(module_path) if module_path else model
        owner_path, _, _ = binding.source_path.rpartition(".")
        owner_leaf = (
            binding.owner.get_submodule(owner_path) if owner_path else binding.owner
        )
        if any(
            parameter is not source
            for parameter in owner_leaf.parameters(recurse=False)
        ) or any(True for _ in owner_leaf.buffers(recurse=False)):
            raise ValueError(
                "Borrowed owner cannot carry coupled weights/scales or buffers"
            )
        for candidate in (leaf, owner_leaf):
            staged = getattr(candidate, "requires_staged_device_postprocess", None)
            if callable(staged) and staged():
                raise ValueError(
                    "Borrowed parameters cannot require staged quantization"
                )
        if any(True for _ in leaf.children()):
            raise ValueError("Borrowed parameters must belong to leaf modules")
        if any(
            parameter is not destination for parameter in leaf.parameters(recurse=False)
        ):
            raise ValueError(
                "Borrowed leaves must have a single parameter; coupled layouts need an explicit contract"
            )
        if any(True for _ in leaf.buffers(recurse=False)):
            raise ValueError("Borrowed leaves cannot carry unowned coupled buffers")
        resolved.append((leaf, local_name, source))
    prepare(dict(bindings))
    for leaf, local_name, source in resolved:
        leaf._parameters[local_name] = source
        leaf._rtp_borrowed_parameter_names = frozenset({local_name})
        _mark_loaded(leaf, local_name)
    object.__setattr__(model, "_rtp_borrowed_parameter_bindings", dict(bindings))
    object.__setattr__(
        model,
        "_rtp_borrowed_parameter_snapshots",
        {
            name: (
                binding.parameter(),
                binding.parameter().data_ptr(),
                binding.parameter().device,
                binding.parameter().dtype,
                tuple(binding.parameter().shape),
            )
            for name, binding in bindings.items()
        },
    )
    validate_borrowed_parameters(model)


def exclude_borrowed_checkpoint_weights(model, model_filter):
    bindings = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    if not bindings:
        return model_filter
    mapper = getattr(model, "WEIGHTS_MAPPER", None)

    def keep(name):
        mapped = mapper.map_name(name) if mapper is not None else name
        return mapped not in bindings and (model_filter is None or model_filter(name))

    return keep
