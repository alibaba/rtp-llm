import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

import rtp_llm.models_py.weight_mapper as weight_mapper
import torch
import torch.nn as nn
from rtp_llm.models_py.module_base import RtpModule, collect_loaded_tensor_ids
from rtp_llm.models_py.registry import get_model_class, list_models

logger = logging.getLogger(__name__)

_EXPERT_ID_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_STACKED_EXPERT_RE = re.compile(r"(?:^|\.)experts\.(?:gate_up_proj|down_proj)(?:\.|$)")


class _ExpertRangeFilter:
    """Select only this EP rank's per-expert checkpoint tensors."""

    def __init__(self, num_experts: int, ep_size: int, ep_rank: int):
        for name, value in (
            ("num_experts", num_experts),
            ("ep_size", ep_size),
            ("ep_rank", ep_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
        if num_experts <= 0 or num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible by "
                f"ep_size={ep_size}"
            )
        experts_per_rank = num_experts // ep_size
        self.start_expert = ep_rank * experts_per_rank
        self.end_expert = self.start_expert + experts_per_rank
        self.ep_size = ep_size
        self.num_experts = num_experts

    def should_load(self, name: str) -> bool:
        match = _EXPERT_ID_RE.search(name)
        if match is not None:
            expert_id = int(match.group(1))
            if not 0 <= expert_id < self.num_experts:
                raise ValueError(
                    f"Checkpoint tensor {name!r} has expert id {expert_id} outside "
                    f"[0, {self.num_experts})"
                )
            return self.start_expert <= expert_id < self.end_expert
        if self.ep_size > 1 and _STACKED_EXPERT_RE.search(name):
            raise ValueError(
                "EP loading requires per-expert checkpoint keys; stacked all-expert "
                f"tensor {name!r} would materialize non-local experts"
            )
        return True


def _validate_runtime_device(device: str, label: str) -> None:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {device!r}") from exc
    if parsed.type == "meta":
        raise ValueError(
            f"{label} cannot be meta; newloader requires materialized weights"
        )


class NewLoaderLoadMethod(str, Enum):
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


@dataclass(frozen=True)
class NewLoaderConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    load_method: NewLoaderLoadMethod = NewLoaderLoadMethod.AUTO
    quant_config: Any = None
    parallelism_config: Any = None
    moe_config: Any = None
    fmha_config: Any = None
    device_resource_config: Any = None
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    ffn_tp_size: Optional[int] = None
    ffn_tp_rank: Optional[int] = None
    lm_head_tp_size: Optional[int] = None
    lm_head_tp_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.load_method, str):
            try:
                object.__setattr__(
                    self,
                    "load_method",
                    NewLoaderLoadMethod(self.load_method.strip().lower()),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported newloader load method {self.load_method!r}"
                ) from exc
        elif not isinstance(self.load_method, NewLoaderLoadMethod):
            raise TypeError(
                f"load_method must be NewLoaderLoadMethod or str, got "
                f"{type(self.load_method).__name__}"
            )
        for name in ("tp_size", "tp_rank", "ep_size", "ep_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"Invalid TP partition: rank={self.tp_rank}, size={self.tp_size}"
            )
        for prefix in ("attn_tp", "ffn_tp", "lm_head_tp"):
            size_name = f"{prefix}_size"
            rank_name = f"{prefix}_rank"
            size = getattr(self, size_name)
            rank = getattr(self, rank_name)
            if size is None and rank is None:
                size, rank = self.tp_size, self.tp_rank
                object.__setattr__(self, size_name, size)
                object.__setattr__(self, rank_name, rank)
            elif size is None or rank is None:
                raise ValueError(
                    f"{size_name} and {rank_name} must be configured together"
                )
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError(f"{size_name} must be an integer")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"{rank_name} must be an integer")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(
                    f"Invalid {prefix} partition: rank={rank}, size={size}"
                )
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"Invalid EP partition: rank={self.ep_rank}, size={self.ep_size}"
            )
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("compute_dtype must be a torch.dtype")
        _validate_runtime_device(self.device, "device")
        if self.parallelism_config is not None:
            for prefix in ("tp", "ep"):
                configured_size = getattr(
                    self.parallelism_config, f"{prefix}_size", None
                )
                configured_rank = getattr(
                    self.parallelism_config, f"{prefix}_rank", None
                )
                if configured_size is None and configured_rank is None:
                    continue
                expected = (
                    (self.tp_size, self.tp_rank)
                    if prefix == "tp"
                    else (self.ep_size, self.ep_rank)
                )
                if (configured_size, configured_rank) != expected:
                    raise ValueError(
                        f"parallelism_config {prefix.upper()} partition does not "
                        f"match NewLoaderConfig: parallelism="
                        f"({configured_rank}, {configured_size}) loader="
                        f"({expected[1]}, {expected[0]})"
                    )
            topology_getters = {
                "attn_tp": ("get_attn_tp_size", "get_attn_tp_rank"),
                "ffn_tp": ("get_ffn_tp_size", "get_ffn_tp_rank"),
                "lm_head_tp": ("tp_size", "tp_rank"),
            }
            for prefix, (
                size_getter_name,
                rank_getter_name,
            ) in topology_getters.items():
                size_source = getattr(self.parallelism_config, size_getter_name, None)
                rank_source = getattr(self.parallelism_config, rank_getter_name, None)
                if callable(size_source) and callable(rank_source):
                    actual = (size_source(), rank_source())
                elif size_source is not None and rank_source is not None:
                    actual = (size_source, rank_source)
                else:
                    continue
                expected = (
                    getattr(self, f"{prefix}_size"),
                    getattr(self, f"{prefix}_rank"),
                )
                if actual != expected:
                    raise ValueError(
                        f"parallelism_config {prefix} partition does not match "
                        f"NewLoaderConfig: parallelism=({actual[1]}, {actual[0]}) "
                        f"loader=({expected[1]}, {expected[0]})"
                    )


class NewModelLoader:
    """Create a PyModel first, stream tensors into it, then run post-load hooks."""

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[NewLoaderConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        effective_config = NewLoaderConfig() if load_config is None else load_config
        if not isinstance(effective_config, NewLoaderConfig):
            raise TypeError(
                f"load_config must be NewLoaderConfig, got {type(effective_config).__name__}"
            )
        if device is not None:
            _validate_runtime_device(device, "device override")
            effective_config = replace(effective_config, device=device)
        self.model_config = model_config
        self.load_config = effective_config
        self.model_path = model_path
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
                self._resolved_model_path(),
                tp_rank=self.load_config.tp_rank,
                tp_size=self.load_config.tp_size,
            )
            if not self._ckpt_files:
                raise FileNotFoundError(
                    f"No supported checkpoint files in {self._resolved_model_path()}"
                )
        return self._ckpt_files

    def _expert_name_filter(self):
        if self.load_config.ep_size == 1:
            return None
        num_experts = (
            self.model_config.get("num_experts", 0)
            if isinstance(self.model_config, dict)
            else getattr(
                self.model_config,
                "expert_num",
                getattr(self.model_config, "num_experts", 0),
            )
        )
        if isinstance(num_experts, bool) or not isinstance(num_experts, int):
            raise TypeError(
                f"model_config num_experts must be an integer, got {num_experts!r}"
            )
        if num_experts <= 0:
            raise ValueError("EP loading requires model_config.num_experts")
        return _ExpertRangeFilter(
            num_experts,
            self.load_config.ep_size,
            self.load_config.ep_rank,
        ).should_load

    def _validate_ep_checkpoint_format(self, checkpoint_files) -> None:
        if self.load_config.ep_size == 1:
            return
        unsupported = [
            path
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if unsupported:
            raise ValueError(
                "EP streaming requires safetensors checkpoints with per-expert "
                "keys; PyTorch checkpoints are deserialized as a whole before "
                f"name filtering and would materialize non-local experts: {unsupported}"
            )

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
        try:
            source = inspect.getfile(model_cls)
        except (TypeError, OSError):
            source = "<unknown>"
        logger.info(
            "Created newloader model %s from %s", model_cls.__qualname__, source
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
            if (
                custom_loader is not None
                and custom_loader is not RtpModule.load_weights
            ):
                validator = getattr(module, "validate_weights_loaded", None)
                if not callable(validator):
                    raise TypeError(
                        f"Custom weight loader {type(module).__name__}.load_weights() "
                        "must define validate_weights_loaded()"
                    )
                validator(loaded_tensor_ids)

    @staticmethod
    def _run_post_load_hooks(model: nn.Module) -> None:
        for module in model.modules():
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    @staticmethod
    def _validate_runtime_backends(model: nn.Module, device: str) -> None:
        target_device = torch.device(device)
        for module in model.modules():
            if isinstance(module, RtpModule):
                module.validate_runtime_device(target_device)

    @staticmethod
    def _migrate_staged_modules(model: nn.Module, device: str) -> None:
        """Move and compress online-quantized leaves one at a time.

        Migrating the entire model first would retain every BF16 staging weight on
        the accelerator until post-load hooks run. Processing each marked leaf
        immediately bounds the extra accelerator memory to one staging layer.
        """
        target_device = torch.device(device)
        staged_modules = [
            module
            for module in model.modules()
            if isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
        ]
        staged_set = set(staged_modules)
        tensor_registrations = {}
        for module in model.modules():
            tensors = list(
                module.named_parameters(recurse=False, remove_duplicate=False)
            ) + list(module.named_buffers(recurse=False, remove_duplicate=False))
            for name, tensor in tensors:
                if tensor is not None:
                    tensor_registrations.setdefault(id(tensor), []).append(
                        (module, name)
                    )
        for registrations in tensor_registrations.values():
            if (
                any(module in staged_set for module, _ in registrations)
                and len(registrations) > 1
            ):
                aliases = sorted(
                    f"{type(module).__name__}.{name}" for module, name in registrations
                )
                raise RuntimeError(
                    "Online-quantized staging does not support shared tensor "
                    f"aliases: {aliases}"
                )

        for module in staged_modules:
            if any(True for _ in module.children()):
                raise RuntimeError(
                    "Staged device postprocess is supported only for leaf modules, "
                    f"got {type(module).__name__}"
                )
            module.to(target_device)
            module.process_weights_after_loading()

    @torch.inference_mode()
    def load(self) -> nn.Module:
        method = self._resolve_load_method()
        if method != NewLoaderLoadMethod.SCRATCH:
            raise RuntimeError(f"Resolved unsupported load method: {method}")
        checkpoint_files = self._checkpoint_files()
        self._validate_ep_checkpoint_format(checkpoint_files)
        model = self._create_model()
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files) and not bool(
            getattr(model, "supports_rank_local_checkpoint", False)
        ):
            raise ValueError(
                f"{type(model).__name__} does not support rank-local consolidated "
                "checkpoints; use a global HF checkpoint"
            )
        started = time.time()
        model.load_weights(
            weight_mapper.get_all_weights(
                checkpoint_files,
                device="cpu",
                name_filter=self._expert_name_filter(),
            )
        )
        self._validate_loaded_weights(model)
        logger.info("Streamed checkpoint tensors in %.2fs", time.time() - started)

        self._validate_runtime_backends(model, self.load_config.device)
        self._migrate_staged_modules(model, self.load_config.device)
        model.to(self.load_config.device)
        self._validate_loaded_weights(model)
        self._run_post_load_hooks(model)
        model.eval()
        return model
