"""In-place runtime weight transactions shared by EPLB and online updates.

Callers must exclude inference and other writers until commit or rollback.
Tensor objects, storage addresses, shapes, dtypes and strides never change.
Validation and CPU snapshots finish before the first live write. Snapshots
remain available until a distributed caller acknowledges every rank.
"""

import copy
import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Mapping, Sequence, Union

import torch


@dataclass(frozen=True)
class TensorUpdate:
    name: str
    target: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True)
class LoRAUpdate:
    """One rank-local LoRA factor pair for a runtime weight.

    ``a`` and ``b`` use the checkpoint convention ``[rank, in]`` and
    ``[out, rank]``.  The manager intentionally operates on already
    materialized rank-local tensors; TP/EP slicing belongs to the checkpoint
    loader and must happen before an adapter is registered.
    """

    target: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float = 1.0

    def delta(self) -> torch.Tensor:
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError(f"LoRA factors for {self.target!r} must be rank-2")
        if self.a.shape[0] != self.b.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {self.target!r}: "
                f"A={tuple(self.a.shape)}, B={tuple(self.b.shape)}"
            )
        if not isinstance(self.scale, (int, float)) or isinstance(self.scale, bool):
            raise TypeError(f"LoRA scale for {self.target!r} must be numeric")
        if not torch.isfinite(torch.as_tensor(self.scale)):
            raise ValueError(f"LoRA scale for {self.target!r} must be finite")
        # Match the legacy merge path: accumulate and scale in FP32, then
        # cast the delta before adding it to the runtime base tensor.
        return (self.b.float() @ self.a.float() * float(self.scale)).to(self.a.dtype)


class RuntimeLoRAManager:
    """Atomic runtime LoRA activation for stable NewLoader storages.

    The manager is deliberately independent from checkpoint loading.  A
    caller resolves adapter names to rank-local runtime tensors, then calls
    :meth:`load`, :meth:`merge`, and :meth:`unload`.  Every transition is a
    single ``RuntimeWeightTransaction`` and recomputes from an immutable base
    snapshot, so repeated merge/unmerge cannot accumulate numerical drift.
    """

    requires_paused_engine = True

    def __init__(
        self,
        runtime_weights: Mapping[str, torch.Tensor],
        *,
        lock=None,
        validate_contract=None,
        delta_transforms=None,
    ):
        if not runtime_weights:
            raise ValueError("Runtime LoRA manager requires runtime weights")
        # The runtime must supply its forward/update lock when attaching this
        # manager to a live model. This lock alone does not pause other ranks.
        self._lock = lock if lock is not None else threading.RLock()
        self._validate_contract = validate_contract
        self._delta_transforms = dict(delta_transforms or {})
        if set(self._delta_transforms) - set(runtime_weights):
            raise ValueError("LoRA layout transforms must name existing parameters")
        if not all(callable(fn) for fn in self._delta_transforms.values()):
            raise TypeError("LoRA layout transforms must be callable")
        self._weights: Dict[str, torch.Tensor] = {}
        self._base: Dict[str, torch.Tensor] = {}
        self._signatures: Dict[str, tuple] = {}
        for name, tensor in runtime_weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Runtime weight names must be non-empty strings")
            if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
                raise TypeError(
                    f"Runtime weight {name!r} must be a materialized tensor"
                )
            if tensor.layout != torch.strided:
                raise ValueError(f"Runtime weight {name!r} must use strided layout")
            self._weights[name] = tensor
            self._signatures[name] = _signature(tensor)
        self._adapters: Dict[str, Dict[str, LoRAUpdate]] = {}
        self._active: list[str] = []

    @property
    def active_adapters(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._active)

    @property
    def loaded_adapters(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._adapters)

    def load(
        self,
        adapter_name: str,
        updates: Sequence[LoRAUpdate],
        *,
        allow_empty_stage: bool = False,
    ) -> None:
        """Register an adapter without changing inference weights."""
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError("LoRA adapter name must be non-empty")
        updates = list(updates)
        if not updates and not allow_empty_stage:
            raise ValueError("LoRA adapter must contain at least one update")
        staged: Dict[str, LoRAUpdate] = {}
        with self._lock:
            self._validate_storage()
            if adapter_name in self._adapters:
                raise ValueError(f"LoRA adapter {adapter_name!r} is already loaded")
            for update in updates:
                if not isinstance(update, LoRAUpdate):
                    raise TypeError("LoRA updates must be LoRAUpdate instances")
                if update.target in staged:
                    raise ValueError(f"Duplicate LoRA target {update.target!r}")
                target = self._weights.get(update.target)
                if target is None:
                    raise KeyError(f"Unknown runtime weight {update.target!r}")
                if target.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(
                        "LoRA merge requires an FP16, BF16 or FP32 runtime weight"
                    )
                if not isinstance(update.a, torch.Tensor) or not isinstance(
                    update.b, torch.Tensor
                ):
                    raise TypeError("LoRA factors must be tensors")
                if update.a.is_meta or update.b.is_meta:
                    raise ValueError("LoRA factors must be materialized")
                allowed = (torch.float16, torch.bfloat16, torch.float32)
                if update.a.dtype not in allowed or update.b.dtype not in allowed:
                    raise ValueError("LoRA factors must use FP16, BF16 or FP32")
                if update.a.ndim != 2 or update.b.ndim != 2 or update.a.shape[0] == 0:
                    raise ValueError(
                        "LoRA factors must be rank-2 with a positive adapter rank"
                    )
                # Own the inputs: mutating the caller's checkpoint buffers must
                # not change an adapter that has already been registered.
                frozen = LoRAUpdate(
                    update.target,
                    update.a.detach().to(device="cpu", dtype=torch.float32, copy=True),
                    update.b.detach().to(device="cpu", dtype=torch.float32, copy=True),
                    update.scale,
                )
                delta = self._encode_delta(
                    update.target, frozen.delta().to(target.dtype)
                )
                if delta.shape != target.shape or delta.dtype != target.dtype:
                    raise ValueError(
                        f"LoRA delta for {update.target!r} has shape/dtype "
                        f"{tuple(delta.shape)}/{delta.dtype}; expected "
                        f"{tuple(target.shape)}/{target.dtype}"
                    )
                if not torch.isfinite(delta).all():
                    raise ValueError(f"Non-finite LoRA delta for {update.target!r}")
                staged[update.target] = frozen
            self._adapters[adapter_name] = staged

    def unload(self, adapter_name: str) -> None:
        """Remove an adapter and deactivate it atomically if necessary."""
        with self._lock:
            if adapter_name not in self._adapters:
                raise KeyError(f"Unknown LoRA adapter {adapter_name!r}")
            if adapter_name in self._active:
                new_active = [name for name in self._active if name != adapter_name]
                self._install(new_active)
                self._active = new_active
            del self._adapters[adapter_name]

    def merge(self, adapter_names: Union[str, Sequence[str]]) -> None:
        """Activate adapters in order, replacing the current active set."""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        names = list(adapter_names)
        with self._lock:
            if len(set(names)) != len(names):
                raise ValueError("A LoRA adapter may only be activated once")
            for name in names:
                if name not in self._adapters:
                    raise KeyError(f"Unknown LoRA adapter {name!r}")
            self._install(names)
            self._active = names

    def unmerge(self) -> None:
        with self._lock:
            self._install([])
            self._active = []

    def _validate_storage(self) -> None:
        if self._validate_contract is not None:
            self._validate_contract()
        for name, tensor in self._weights.items():
            if _signature(tensor) != self._signatures[name]:
                raise RuntimeError(f"Runtime storage changed for {name!r}")

    def base_weight(self, name: str) -> torch.Tensor:
        """Return a private CPU copy for checkpoint conversion without LoRA."""
        with self._lock:
            self._validate_storage()
            if name not in self._weights:
                raise KeyError(f"Unknown runtime weight {name!r}")
            source = self._base[name] if name in self._base else self._weights[name]
            return source.detach().to(device="cpu", copy=True)

    def _deltas(self, names: Sequence[str]) -> Dict[str, torch.Tensor]:
        deltas: Dict[str, torch.Tensor] = {}
        for name in names:
            for target, update in self._adapters[name].items():
                # Round the completed FP32 product, not its input factors.
                value = self._encode_delta(
                    target, update.delta().to(self._weights[target].dtype)
                )
                deltas[target] = deltas[target] + value if target in deltas else value
        return deltas

    def _encode_delta(self, name, delta):
        transform = self._delta_transforms.get(name)
        if transform is None:
            return delta
        result = transform(delta)
        if (
            not isinstance(result, torch.Tensor)
            or result.shape != delta.shape
            or result.dtype != delta.dtype
        ):
            raise ValueError(
                f"LoRA physical layout transform changed shape/dtype for {name!r}"
            )
        if result.is_meta or not torch.isfinite(result).all():
            raise ValueError(f"Invalid LoRA physical layout result for {name!r}")
        if result.numel() and result.data_ptr() == self._weights[name].data_ptr():
            raise ValueError(
                "LoRA layout encoding must not return live parameter storage"
            )
        return result

    @staticmethod
    def _commit_updates(updates: Sequence[TensorUpdate]) -> None:
        if not updates:
            return
        transaction = RuntimeWeightTransaction(updates)
        transaction.install()
        transaction.commit()

    def _install(self, names: Sequence[str]) -> None:
        self._validate_storage()
        deltas = self._deltas(names)
        # Only active adapter targets need a base snapshot, and snapshots live
        # on CPU. Never copy or restore unrelated model parameters.
        bases = dict(self._base)
        for target in deltas:
            if target not in bases:
                bases[target] = (
                    self._weights[target].detach().to(device="cpu", copy=True)
                )
        updates = [
            TensorUpdate(
                target,
                self._weights[target],
                base + deltas[target] if target in deltas else base,
            )
            for target, base in bases.items()
        ]
        self._commit_updates(updates)
        self._base = {target: bases[target] for target in deltas}

    def update_base_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Replace rank-local base weights while retaining active adapters.

        Callers must route online updates through this method while adapters
        are active, using the same engine pause/lock as merge and unload.
        Checkpoint mapping and TP conversion must already be complete.
        """
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("Base updates require a non-empty tensor mapping")
        with self._lock:
            self._validate_storage()
            staged = {}
            for name, source in weights.items():
                if name not in self._weights:
                    raise KeyError(f"Unknown runtime weight {name!r}")
                target = self._weights[name]
                if not isinstance(source, torch.Tensor) or source.is_meta:
                    raise TypeError(
                        f"Base update {name!r} must be a materialized tensor"
                    )
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(f"Base update shape/dtype mismatch for {name!r}")
                staged[name] = source.detach().to(device="cpu", copy=True)
            deltas = self._deltas(self._active)
            self._commit_updates(
                [
                    TensorUpdate(
                        name,
                        self._weights[name],
                        base + deltas[name] if name in deltas else base,
                    )
                    for name, base in staged.items()
                ]
            )
            # Update metadata only after every live tensor was committed.
            self._base.update(
                {name: base for name, base in staged.items() if name in deltas}
            )


def collect_lora_delta_transforms(model):
    """Collect parameter-owned physical layout encoders, without legacy keys."""
    transforms = {}
    for module_name, module in model.named_modules():
        encode = getattr(module, "encode_lora_delta", None)
        if not callable(encode):
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            transforms[name] = partial(encode, parameter_name)
    return transforms


def prepare_leaf_weight_updates(
    module, weights, *, reset_fields=()
) -> list[TensorUpdate]:
    """Reuse a leaf's checkpoint TP/layout conversion on private CPU storage.

    Only explicitly supported leaves call this helper. Runtime executors and
    post-load transforms must never run on this shallow metadata copy.
    """
    if list(module.children()):
        raise ValueError("Online checkpoint updates require leaf-owned conversion")
    staged = copy.copy(module)
    staged._parameters = {
        name: (
            None
            if tensor is None
            else torch.nn.Parameter(
                tensor.detach().to(device="cpu", copy=True), requires_grad=False
            )
        )
        for name, tensor in module._parameters.items()
    }
    staged._buffers = {
        name: None if tensor is None else tensor.detach().to(device="cpu", copy=True)
        for name, tensor in module._buffers.items()
    }
    staged._rtp_loaded_weight_names = set()
    for name in reset_fields:
        value = getattr(module, name, None)
        if isinstance(value, (set, dict)):
            setattr(staged, name, type(value)())
    staged.load_weights(weights)
    changed = staged._rtp_loaded_weight_names | set(
        getattr(staged, "_loaded_parameter_shards", {})
    )
    return [
        TensorUpdate(name, module._parameters[name], staged._parameters[name])
        for name in sorted(changed)
        if name in staged._parameters
    ]


def _signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


class RuntimeWeightTransaction:
    """Prepare, install and acknowledge one rank's already-materialized weights."""

    @torch.no_grad()
    def __init__(self, updates: Iterable[TensorUpdate]):
        updates = list(updates)
        if not updates:
            raise ValueError("A runtime weight transaction must contain updates")
        names = set()
        storages = set()
        self._devices = set()
        for update in updates:
            if not isinstance(update.name, str) or not update.name:
                raise ValueError("Runtime weight names must be non-empty strings")
            if update.name in names:
                raise ValueError(f"Duplicate runtime weight {update.name!r}")
            names.add(update.name)
            target, source = update.target, update.source
            if getattr(target, "_rtp_requires_restart_for_update", False):
                raise PermissionError(
                    f"Runtime weight {update.name!r} is prepared/shared; restart required"
                )
            if not isinstance(target, torch.Tensor) or not isinstance(
                source, torch.Tensor
            ):
                raise TypeError(f"Runtime weight {update.name!r} requires tensors")
            if target.is_meta or source.is_meta:
                raise ValueError(
                    f"Runtime weight {update.name!r} cannot use meta tensors"
                )
            if target.layout != torch.strided or source.layout != torch.strided:
                raise ValueError(
                    f"Runtime weight {update.name!r} must use strided layout"
                )
            if target.device.type not in ("cpu", "cuda") or source.device.type not in (
                "cpu",
                "cuda",
            ):
                raise ValueError("Runtime updates support CPU and CUDA/ROCm tensors")
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"Runtime weight {update.name!r} mismatch: expected "
                    f"{tuple(target.shape)} {target.dtype}, got {tuple(source.shape)} {source.dtype}"
                )
            # Permit ordinary transposes, but reject broadcast/overlapping views.
            strides = sorted(
                (stride, size)
                for size, stride in zip(target.shape, target.stride())
                if size > 1
            )
            span = 1
            for stride, size in strides:
                if stride < span:
                    raise ValueError(f"Overlapping target view for {update.name!r}")
                span = stride * (size - 1) + span
            if target.numel():
                storage = (target.device, target.untyped_storage().data_ptr())
                if storage in storages:
                    raise ValueError(
                        "Aliased targets must be updated through a single owner"
                    )
                storages.add(storage)
            for tensor in (target, source):
                if tensor.device.type == "cuda":
                    self._devices.add(tensor.device)

        self._synchronize()
        self._entries = [
            (
                update.name,
                update.target,
                _signature(update.target),
                update.source.detach().to(device="cpu", copy=True),
                update.target.detach().to(device="cpu", copy=True),
            )
            for update in updates
        ]
        self._state = "prepared"

    def _synchronize(self) -> None:
        for device in sorted(self._devices, key=str):
            torch.cuda.synchronize(device)

    def _validate_storage(self) -> None:
        for name, target, signature, _, _ in self._entries:
            if _signature(target) != signature:
                raise RuntimeError(f"Runtime storage changed while updating {name!r}")

    @torch.inference_mode()
    def install(self) -> None:
        if self._state != "prepared":
            raise RuntimeError(f"Cannot install a {self._state} weight transaction")
        self._validate_storage()
        self._state = "installing"
        try:
            for _, target, _, source, _ in self._entries:
                target.copy_(source)
            self._synchronize()
        except BaseException:
            self.rollback()
            raise
        self._state = "installed"

    def commit(self) -> None:
        if self._state != "installed":
            raise RuntimeError(f"Cannot commit a {self._state} weight transaction")
        self._validate_storage()
        self._entries.clear()
        self._state = "committed"

    @torch.inference_mode()
    def rollback(self) -> None:
        if self._state == "rolled_back":
            return
        if self._state == "committed":
            raise RuntimeError("Cannot roll back committed runtime weights")
        self._validate_storage()
        if self._state != "prepared":
            for _, target, _, _, previous in self._entries:
                target.copy_(previous)
            self._synchronize()
        self._entries.clear()
        self._state = "rolled_back"
