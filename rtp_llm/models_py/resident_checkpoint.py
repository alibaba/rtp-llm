"""Resident T1 checkpoint for direct parameter owners. No checkpoint-name translation."""

import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field

import torch

from rtp_llm.resident_weights import GLOBAL_SCOPE, WeightDict


class _HeaderEntries:
    def __init__(self, facts, source):
        self.facts = facts
        self.source = source

    def __contains__(self, name):
        if self.source is not None:
            self.facts[name] = name in self.source
        return self.facts[name]

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        # Opaque arguments for _shape; no paths or file handles cross processes.
        return name, None


class ResidentFTHeader:
    """Replay only the format facts the model constructor actually consulted.

    No tensor-reading API. File selection and real FT import still use FTCheckpoint.
    """

    def __init__(self, root, facts, source=None):
        self.root = Path(root)
        self.facts = facts
        self.source = source
        self.entries = _HeaderEntries(
            facts["entries"], None if source is None else source.entries
        )
        self.index_metadata = facts["metadata"]
        self.topology = SimpleNamespace(**facts["topology"])

    @classmethod
    def recording(cls, source):
        return cls(
            source.root,
            {
                "entries": {},
                "shapes": {},
                "peer_shapes": {},
                "topology": vars(source.topology),
                "metadata": {
                    "ft_export_backend": source.index_metadata.get("ft_export_backend")
                },
            },
            source,
        )

    def _shape(self, name, unused):
        shapes = self.facts["shapes"]
        if self.source is not None:
            shapes[name] = list(self.source._shape(*self.source.entries[name]))
        return tuple(shapes[name])

    def read_rank_slice(self, name, tp_rank):
        shapes = self.facts["peer_shapes"].setdefault(str(tp_rank), {})
        if self.source is not None:
            shapes[name] = list(self.source.read_rank_slice(name, tp_rank))
        return tuple(shapes[name])


@dataclass
class _Bank:
    global_weights: dict
    weights: list = field(default_factory=list)
    consumed: set = field(default_factory=set)

    def finish_native_record(self, manager):
        manager.init_scalar(
            GLOBAL_SCOPE, "native.consumed", lambda: json.dumps(sorted(self.consumed))
        )

    def bind_executors(self, model):
        for prefix, leaf in model.named_modules():
            if callable(getattr(leaf, "bind_resident_parameters", None)):
                leaf.bind_resident_parameters(self.global_weights, prefix + ".")
            if callable(getattr(leaf, "take_executor_weight", None)) or getattr(
                leaf, "resident_consumed_parameters", ()
            ):
                leaf._resident_expert_inputs = (self, prefix + ".")


def _slots(model):
    """Canonical registrations, excluding every alias of a target-owned Parameter."""
    borrowed = getattr(model, "_rtp_borrowed_parameter_bindings", {})
    excluded = {id(model.get_parameter(path)) for path in borrowed}
    seen = set(excluded)
    slots = {}
    for prefix, leaf in model.named_modules():
        for name, tensor in (*leaf._parameters.items(), *leaf._buffers.items()):
            if (
                tensor is None
                or (name in leaf._non_persistent_buffers_set and name not in getattr(leaf, '_resident_runtime_buffers', ()))
                or id(tensor) in seen
            ):
                continue
            seen.add(id(tensor))
            slots[f"{prefix}.{name}" if prefix else name] = tensor
    return slots


class _OwnerSlots(WeightDict):
    """Switch each live registration as its underlying storage is relocated.

    Do not keep all original GPU Parameters alive until the *last* copy finishes.
    Keep detached T1 views so a later post-load set_ cannot change the record.
    """

    def __init__(self, slots):
        super().__init__(GLOBAL_SCOPE, ((k, t.detach()) for k, t in slots.items()))
        self.destinations = slots

    def __setitem__(self, name, value):
        if self.destinations is not None:
            self.destinations[name].data = value
        super().__setitem__(name, value)


def _loaded_fields(model):
    return {
        prefix: sorted(names)
        for prefix, leaf in model.named_modules()
        if (names := getattr(leaf, "_rtp_loaded_weight_names", None))
    }


def _replay_only():
    raise RuntimeError("Resident checkpoint replay unexpectedly requested a cold value")


def record(model, manager, method):
    """Called after checkpoint validation and migration, before derived kernels are built."""
    slots = _OwnerSlots(_slots(model))
    bank = _Bank(slots)
    try:
        manager.move_to_vmm(bank)
    finally:
        slots.destinations = None
    manager.init_scalar(GLOBAL_SCOPE, "native.backend", lambda: method.value)
    manager.init_scalar(
        GLOBAL_SCOPE,
        "native.loaded_fields",
        lambda: json.dumps(_loaded_fields(model), sort_keys=True),
    )
    bank.bind_executors(model)
    model._resident_bank = bank


def restore(model, manager, tensors, device, *, construction=None):
    """Bind imported views to the *same* Parameter objects fresh construction registered."""
    prepare = getattr(model, "prepare_resident_checkpoint_restore", None)
    if prepare is not None:
        prepare(construction, {entry.name: entry.tensor for entry in tensors})
    slots = _slots(model)
    values = {}
    for entry in tensors:
        if (
            entry.scope != GLOBAL_SCOPE
            or entry.name not in slots
            or entry.name in values
        ):
            raise ValueError(
                f"Unexpected resident checkpoint entry {entry.scope}/{entry.name}"
            )
        dest, value = slots[entry.name], entry.tensor
        if value.dtype != dest.dtype or value.shape != dest.shape:
            raise ValueError(f"Resident shape or dtype mismatch for {entry.name}")
        if value.device.type != "cpu" and value.device != torch.device(device):
            raise ValueError(
                f"Resident checkpoint entry {entry.name} belongs to {value.device}"
            )
        values[entry.name] = value
    consumed = set(
        json.loads(manager.init_scalar(GLOBAL_SCOPE, "native.consumed", _replay_only))
    )
    eligible = {
        f"{prefix}.{name}"
        for prefix, leaf in model.named_modules()
        if callable(getattr(leaf, "take_executor_weight", None))
        for name in leaf._parameters
        if name.startswith(("shared_", "routed_"))
    }
    eligible.update(
        f"{prefix}.{name}"
        for prefix, leaf in model.named_modules()
        for name in getattr(leaf, "resident_consumed_parameters", ())
        if name in leaf._parameters
    )
    if not consumed <= eligible:
        raise ValueError(
            "Resident record consumed tensors outside the expert parameter owner"
        )
    if slots.keys() - values.keys() - consumed:
        raise ValueError(
            f"Missing resident checkpoint entries: {sorted(slots.keys() - values.keys())[:8]}"
        )
    fields = json.loads(
        manager.init_scalar(GLOBAL_SCOPE, "native.loaded_fields", _replay_only)
    )
    leaves = dict(model.named_modules())
    for prefix, names in fields.items():
        leaf = leaves[prefix]
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise ValueError(f"Invalid resident load bookkeeping for {prefix!r}")
        leaf._rtp_loaded_weight_names = set(names)
    for name, tensor in slots.items():
        tensor.data = (
            values[name]
            if name in values
            else torch.empty(0, device=device, dtype=tensor.dtype)
        )
    bank = _Bank(values, consumed=consumed)
    manager.bind(bank)
    bank.bind_executors(model)
    model._resident_bank = bank
    for leaf in model.modules():
        validate_input = getattr(leaf, "finish_resident_checkpoint_restore", None)
        if validate_input is not None:
            validate_input()
