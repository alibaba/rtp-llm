import inspect
import logging
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

import rtp_llm.models_py.weight_mapper as weight_mapper
from rtp_llm.models_py.module_base import RtpModule, collect_loaded_tensor_ids
from rtp_llm.models_py.registry import get_model_class, list_models

logger = logging.getLogger(__name__)


def _validate_runtime_device(device: str, label: str) -> None:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {device!r}") from exc
    if parsed.type == "meta":
        raise ValueError(f"{label} cannot be meta; newloader requires materialized weights")


class NewLoaderLoadMethod(str, Enum):
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


@dataclass
class NewLoaderConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    load_method: NewLoaderLoadMethod = NewLoaderLoadMethod.AUTO

    def __post_init__(self) -> None:
        if isinstance(self.load_method, str):
            try:
                self.load_method = NewLoaderLoadMethod(self.load_method.strip().lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported newloader load method {self.load_method!r}") from exc
        elif not isinstance(self.load_method, NewLoaderLoadMethod):
            raise TypeError(
                f"load_method must be NewLoaderLoadMethod or str, got "
                f"{type(self.load_method).__name__}"
            )
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"Invalid TP partition: rank={self.tp_rank}, size={self.tp_size}")
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(f"Invalid EP partition: rank={self.ep_rank}, size={self.ep_size}")
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("compute_dtype must be a torch.dtype")
        _validate_runtime_device(self.device, "device")


class NewModelLoader:
    """Create a PyModel first, stream tensors into it, then run post-load hooks."""

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[NewLoaderConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        effective_config = load_config or NewLoaderConfig()
        if device is not None:
            _validate_runtime_device(device, "device override")
            effective_config = replace(effective_config, device=device)
        self.model_config = model_config
        self.load_config = effective_config
        self.model_path = model_path
        self.device = self.load_config.device
        self._ckpt_files = None

    def _resolve_load_method(self) -> NewLoaderLoadMethod:
        configured = self.load_config.load_method
        if configured == NewLoaderLoadMethod.AUTO:
            env_method = os.environ.get("LOAD_METHOD", "").strip().lower()
            if env_method:
                try:
                    configured = NewLoaderLoadMethod(env_method)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported LOAD_METHOD environment value {env_method!r}"
                    ) from exc
            else:
                configured = NewLoaderLoadMethod.SCRATCH
            if configured == NewLoaderLoadMethod.AUTO:
                configured = NewLoaderLoadMethod.SCRATCH
        if configured == NewLoaderLoadMethod.FASTSAFETENSORS:
            raise RuntimeError(
                "fastsafetensors is not part of the newloader foundation; use scratch"
            )
        return configured

    def _model_type(self) -> str:
        value = (
            self.model_config.get("model_type", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_type", "")
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_config.model_type must be a non-empty string")
        return value.strip()

    def _resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        value = (
            self.model_config.get("model_path", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_path", "")
        )
        if not value:
            raise ValueError("model_path is required")
        return value

    def _checkpoint_files(self):
        if self._ckpt_files is None:
            self._ckpt_files = weight_mapper.discover_ckpt_files(
                self._resolved_model_path()
            )
            if not self._ckpt_files:
                raise FileNotFoundError(
                    f"No supported checkpoint files in {self._resolved_model_path()}"
                )
        return self._ckpt_files

    def _create_model(self) -> nn.Module:
        model_type = self._model_type()
        try:
            model_cls = get_model_class(model_type)
        except KeyError as exc:
            raise ValueError(
                f"Model type {model_type!r} is not registered; available={list_models()}"
            ) from exc
        model = model_cls(self.model_config, self.load_config)
        if not isinstance(model, RtpModule):
            raise TypeError(
                f"Registered model {model_cls.__name__} must inherit RtpModule to "
                "provide newloader completeness validation"
            )
        logger.info(
            "Created newloader model %s from %s",
            model_cls.__qualname__,
            inspect.getfile(model_cls),
        )
        return model

    @staticmethod
    def _validate_loaded_weights(model: nn.Module) -> None:
        loaded_tensor_ids = frozenset(collect_loaded_tensor_ids(model))
        root_validator = getattr(model, "validate_weights_loaded", None)
        if not callable(root_validator):
            raise TypeError(
                f"Registered model {type(model).__name__} has no weight completeness validator"
            )
        root_validator(loaded_tensor_ids)

        for module in model.modules():
            if module is model:
                continue
            custom_loader = getattr(type(module), "load_weights", None)
            if custom_loader is not None and custom_loader is not RtpModule.load_weights:
                validator = getattr(module, "validate_weights_loaded", None)
                if not callable(validator):
                    raise TypeError(
                        f"Custom weight loader {type(module).__name__}.load_weights() "
                        "must define validate_weights_loaded()"
                    )
                validator(loaded_tensor_ids)

    @staticmethod
    def _collect_tensor_alias_groups(model: nn.Module):
        aliases = {}
        for module in model.modules():
            for name, tensor in module.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                aliases.setdefault(id(tensor), []).append(("parameter", module, name))
            for name, tensor in module.named_buffers(
                recurse=False, remove_duplicate=False
            ):
                if tensor is not None:
                    aliases.setdefault(id(tensor), []).append(("buffer", module, name))
        return [registrations for registrations in aliases.values() if len(registrations) > 1]

    @staticmethod
    def _restore_tensor_aliases(alias_groups) -> None:
        for registrations in alias_groups:
            first_kind, first_module, first_name = registrations[0]
            first_storage = (
                first_module._parameters
                if first_kind == "parameter"
                else first_module._buffers
            )
            shared = first_storage[first_name]
            if shared is None:
                raise RuntimeError(
                    f"Shared {first_kind} {first_name!r} disappeared during migration"
                )
            for kind, module, name in registrations[1:]:
                storage = module._parameters if kind == "parameter" else module._buffers
                storage[name] = shared

    @staticmethod
    def _run_post_load_hooks(model: nn.Module) -> None:
        for module in model.modules():
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    @torch.inference_mode()
    def load(self) -> nn.Module:
        method = self._resolve_load_method()
        if method != NewLoaderLoadMethod.SCRATCH:
            raise RuntimeError(f"Resolved unsupported load method: {method}")
        model = self._create_model()
        started = time.time()
        model.load_weights(
            weight_mapper.get_all_weights(self._checkpoint_files(), device="cpu")
        )
        self._validate_loaded_weights(model)
        logger.info("Streamed checkpoint tensors in %.2fs", time.time() - started)

        alias_groups = self._collect_tensor_alias_groups(model)
        model.to(self.device)
        self._restore_tensor_aliases(alias_groups)
        self._validate_loaded_weights(model)
        self._run_post_load_hooks(model)
        return model
