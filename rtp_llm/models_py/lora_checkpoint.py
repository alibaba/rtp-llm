"""Read PEFT LoRA factors using checkpoint names, independently of OldLoader."""

import json
import math
import re
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file

from rtp_llm.models_py.runtime_weight_update import (
    LoRAUpdate,
    RuntimeLoRAManager,
    collect_lora_delta_transforms,
)


def validate_attention_lora_checkpoint_target(name: str, model_name: str) -> None:
    """Bound MoE startup adapters to the attention targets supported by main."""
    parts = name.split(".")
    if not (
        len(parts) == 5
        and parts[0] == "layers"
        and parts[1].isdigit()
        and parts[2] == "self_attn"
        and parts[4] == "weight"
        and parts[3] in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        raise NotImplementedError(
            f"{model_name} startup LoRA supports attention projections only, got {name!r}"
        )


def model_lora_checkpoint_updates(model, loader, path: str) -> list[LoRAUpdate]:
    """Resolve PEFT factors through the model's existing TP/fusion owners."""
    module_names = {module: name for name, module in model.named_modules()}
    validate_target = getattr(model, "validate_lora_checkpoint_target", None)
    owns_target = getattr(model, "owns_lora_checkpoint_target", None)
    grouped = {}
    for update in read_lora_checkpoint(path):
        mapped = loader._mapped_checkpoint_name(model, update.target)
        if callable(validate_target):
            validate_target(mapped)
        if callable(owns_target) and not owns_target(mapped):
            continue
        target, _ = model.resolve_checkpoint_load_plan(
            mapped, (update.b.shape[0], update.a.shape[1])
        )
        if target is None or not callable(
            getattr(target.module, "prepare_lora_factors", None)
        ):
            raise NotImplementedError(
                f"No NewLoader LoRA factor mapping for {update.target!r}"
            )
        if getattr(target.module, "_rtp_borrowed_parameter_names", frozenset()):
            raise ValueError(
                f"Cannot apply draft LoRA to borrowed parameter {update.target!r}"
            )
        a, b = target.module.prepare_lora_factors(target.name, update.a, update.b)
        module_name = module_names[target.module]
        name = f"{module_name}.weight" if module_name else "weight"
        grouped.setdefault(name, []).append((target.name, a, b, update.scale))
    updates = []
    for name, pieces in grouped.items():
        if len({key for key, _, _, _ in pieces}) != len(pieces):
            raise ValueError(f"Duplicate mapped LoRA projection for {name!r}")
        if len(pieces) > 1 and any(key == "weight" for key, _, _, _ in pieces):
            raise ValueError(f"Cannot mix fused and per-projection LoRA for {name!r}")
        # Disjoint Q/K/V or gate/up factors share a fused parameter; concatenate
        # their ranks without materializing full checkpoint-sized dense deltas.
        if len({scale for _, _, _, scale in pieces}) != 1:
            raise ValueError(f"Inconsistent LoRA projection scales for {name!r}")
        updates.append(
            LoRAUpdate(
                name,
                torch.cat([a for _, a, _, _ in pieces], dim=0),
                torch.cat([b for _, _, b, _ in pieces], dim=1),
                pieces[0][3],
            )
        )
    return updates


def merge_startup_lora_checkpoint(model, loader, adapter_name: str, path: str) -> None:
    """Merge before model publication, without promising online weight updates.

    Call only on an unpublished model. The temporary manager validates every
    factor before its single atomic merge; no lifecycle manager is attached to
    the serving model, and its base snapshots are released after startup.
    """
    _merge_runtime_updates(
        model, adapter_name, model_lora_checkpoint_updates(model, loader, path)
    )


def prepare_startup_lora_merge(model, loader, adapter_name, path):
    """Stage factors before FST can finalize its first complete INT8 leaf."""
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError("LoRA adapter name must be non-empty")
    runtime_updates = []
    for update in model_lora_checkpoint_updates(model, loader, path):
        leaf = model.get_submodule(update.target.rpartition(".")[0])
        stage = getattr(getattr(leaf, "quant_method", None), "stage_startup_lora", None)
        if stage is None:
            runtime_updates.append(update)
        else:
            stage(leaf, update)
    # Unquantized leaves can have a device-specific layout. Retain their
    # existing post-layout transform and atomic merge after loader finalization.
    return lambda: _merge_runtime_updates(model, adapter_name, runtime_updates)


def _merge_runtime_updates(model, adapter_name, updates):
    if not updates:
        # Every target was staged before quantization or lives on another rank.
        return
    manager = RuntimeLoRAManager(
        dict(model.named_parameters()),
        delta_transforms=collect_lora_delta_transforms(model),
    )
    manager.load(adapter_name, updates)
    manager.merge(adapter_name)


class RankLocalCheckpointAdapters:
    """Load PEFT checkpoints onto the direct model's existing parameter owners.

    Registration does not change inference. Activation changes shared model
    storage, not the adapter for a single request. A serving caller must pause
    inference on all participating ranks and coordinate success or recovery;
    the manager's local lock cannot provide that distributed guarantee.
    """

    def __init__(self, model, loader, *, startup_adapters=(), admin_lock=None):
        self._model = model
        self._loader = loader
        self._startup = frozenset(startup_adapters)
        self._lock = threading.RLock() if admin_lock is None else admin_lock
        self._manager = None

    @property
    def active_adapters(self):
        with self._lock:
            return () if self._manager is None else self._manager.active_adapters

    @property
    def loaded_adapters(self):
        with self._lock:
            return () if self._manager is None else self._manager.loaded_adapters

    @contextmanager
    def base_update_guard(self):
        """Exclude adapter transitions for the entire caller's base write."""
        with self._lock:
            if self._startup or self.active_adapters:
                raise RuntimeError(
                    "Cannot update base weights with merged LoRA adapters"
                )
            yield

    def load(self, name, path):
        with self._lock:
            if name in self._startup:
                raise ValueError(
                    f"Adapter {name!r} is already merged into the startup base"
                )
            updates = model_lora_checkpoint_updates(self._model, self._loader, path)
            if self._manager is None:
                params = dict(self._model.named_parameters())

                def validate_bindings():
                    # An EPLB switch may replace an nn.Parameter entirely;
                    # unchanged storage on the old object is not sufficient.
                    live = dict(self._model.named_parameters())
                    if live.keys() != params.keys() or any(
                        live[k] is not v for k, v in params.items()
                    ):
                        raise RuntimeError("Direct model parameter bindings changed")

                manager = RuntimeLoRAManager(
                    params,
                    lock=self._lock,
                    validate_contract=validate_bindings,
                    delta_transforms=collect_lora_delta_transforms(self._model),
                )
                manager.load(name, updates, allow_empty_stage=True)
                self._manager = manager
            else:
                self._manager.load(name, updates, allow_empty_stage=True)

    def _transition(self, op, pause_engine, *args):
        if not callable(pause_engine):
            raise TypeError("A caller-provided engine quiescence context is required")
        # Enter engine quiescence before the manager lock on every transition.
        with pause_engine(), self._lock:
            if self._manager is None:
                raise ValueError("No rank-local adapters are registered")
            getattr(self._manager, op)(*args)
            if not self._manager.loaded_adapters:
                self._manager = None

    def merge(self, names, *, pause_engine):
        self._transition("merge", pause_engine, names)

    def unmerge(self, *, pause_engine):
        self._transition("unmerge", pause_engine)

    def unload(self, name, *, pause_engine):
        self._transition("unload", pause_engine, name)


def prepare_lora_checkpoint_factors(a, b, input_size, output_size, prefix=""):
    """Own FP32 checkpoint factors before module-specific TP/fusion mapping.

    The legacy merge path reads factors in FP32, accumulates their product in
    FP32, and casts only the completed delta to the base weight dtype. Casting
    factors to FP16/BF16 first loses information that accumulation cannot recover.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("LoRA factors must be tensors")
    allowed = (torch.float16, torch.bfloat16, torch.float32)
    if a.dtype not in allowed or b.dtype not in allowed:
        raise ValueError("LoRA factors must use FP16, BF16 or FP32")
    if a.is_meta or b.is_meta:
        raise ValueError("LoRA factors must be materialized")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] <= 0 or a.shape[0] != b.shape[1]:
        raise ValueError("LoRA factors must be rank-2 with matching positive rank")
    if a.shape[1] != input_size or b.shape[0] != output_size:
        raise ValueError(
            f"{prefix} LoRA checkpoint shape mismatch: "
            f"A={tuple(a.shape)}, B={tuple(b.shape)}, expected in/out={input_size}/{output_size}"
        )
    return (
        a.detach().to(device="cpu", dtype=torch.float32, copy=True),
        b.detach().to(device="cpu", dtype=torch.float32, copy=True),
    )


def read_lora_checkpoint(path: str) -> list[LoRAUpdate]:
    root = Path(path)
    with (root / "adapter_config.json").open() as stream:
        config = json.load(stream)
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
    ):
        raise ValueError("LoRA alpha must be finite and numeric")
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("Expected a LoRA adapter checkpoint")
    for option in (
        "use_dora",
        "fan_in_fan_out",
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "lora_bias",
    ):
        if config.get(option):
            raise NotImplementedError(
                f"LoRA checkpoint option {option!r} has not been adapted"
            )
    if config.get("bias", "none") != "none":
        raise NotImplementedError("LoRA checkpoint bias updates have not been adapted")
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora", False) else rank)
    candidates = [
        root / f"adapter_model.{suffix}"
        for suffix in ("safetensors", "bin", "pth", "pt")
    ]
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one adapter_model safetensors/bin/pth/pt file"
        )
    checkpoint = candidates[0]
    if checkpoint.suffix == ".safetensors":
        tensors = load_file(str(checkpoint), device="cpu")
    else:
        tensors = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("LoRA checkpoint must contain a non-empty tensor mapping")
    pairs = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str):
            raise ValueError("LoRA tensor names must be strings")
        name = name.removeprefix("base_model.model.")
        match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.[^.]+)?\.weight", name)
        if match is None:
            raise ValueError(f"Unexpected LoRA checkpoint tensor {name!r}")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"LoRA factor {name!r} must be a rank-2 tensor")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported LoRA factor dtype for {name!r}")
        target, factor = match.groups()
        pair = pairs.setdefault(target + ".weight", {})
        if factor in pair:
            raise ValueError(f"Duplicate LoRA factor for {target!r}")
        pair[factor] = tensor
    updates = []
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA A/B pair for {name!r}")
        a, b = pair["A"], pair["B"]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"LoRA rank mismatch for {name!r}")
        updates.append(LoRAUpdate(name, a, b, scale))
    return updates
