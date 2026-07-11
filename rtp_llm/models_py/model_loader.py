import inspect
import logging
import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.registry import get_model_class, list_models

logger = logging.getLogger(__name__)


class LoadMethod:
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


class LoadConfig:
    def __init__(
        self,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        quant_type: str = "none",
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_method: str = LoadMethod.AUTO,
        quant_source_config: Any = None,
        force_cpu_load_weights: bool = False,
        parallelism_config: Any = None,
        fmha_config: Any = None,
        hw_kernel_config: Any = None,
        device_resource_config: Any = None,
        moe_config: Any = None,
    ):
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"Invalid TP partition: rank={tp_rank}, size={tp_size}")
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.device = device
        self.load_method = load_method
        self.quant_source_config = quant_source_config
        self.force_cpu_load_weights = force_cpu_load_weights
        self.parallelism_config = parallelism_config
        self.fmha_config = fmha_config
        self.hw_kernel_config = hw_kernel_config
        self.device_resource_config = device_resource_config
        self.moe_config = moe_config


class NewModelLoader:
    """Create a PyModel first, stream tensors into it, then run post-load hooks."""

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[LoadConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_config = model_config
        self.load_config = load_config or LoadConfig()
        self.model_path = model_path
        self.device = device if device is not None else self.load_config.device
        self._ckpt_files = None

    def _resolve_load_method(self) -> str:
        raw_method = self.load_config.load_method or LoadMethod.AUTO
        if not isinstance(raw_method, str):
            raise TypeError(
                f"load_method must be a string, got {type(raw_method).__name__}"
            )
        configured = raw_method.strip().lower()
        if configured == LoadMethod.AUTO:
            configured = os.environ.get("LOAD_METHOD", "").strip().lower()
            configured = configured or LoadMethod.SCRATCH
            if configured == LoadMethod.AUTO:
                configured = LoadMethod.SCRATCH
        if configured not in (LoadMethod.SCRATCH, LoadMethod.FASTSAFETENSORS):
            raise ValueError(
                f"Unsupported load_method {configured!r}; expected auto or scratch"
            )
        if configured == LoadMethod.FASTSAFETENSORS:
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
        if not isinstance(model, nn.Module):
            raise TypeError(f"Registered model {model_cls!r} did not create an nn.Module")
        if not callable(getattr(model, "load_weights", None)):
            raise TypeError(f"Registered model {model_cls.__name__} has no load_weights()")
        logger.info(
            "Created newloader model %s from %s",
            model_cls.__qualname__,
            inspect.getfile(model_cls),
        )
        return model

    @staticmethod
    def _run_post_load_hooks(model: nn.Module, force_cpu: bool) -> None:
        modules = list(model.modules())
        for module in modules:
            setattr(module, "_new_loader_force_cpu_load_weights", force_cpu)
        for module in modules:
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    def load(self) -> nn.Module:
        method = self._resolve_load_method()
        if method != LoadMethod.SCRATCH:
            raise RuntimeError(f"Resolved unsupported load method: {method}")

        model = self._create_model()
        started = time.time()
        model.load_weights(
            weight_mapper.get_all_weights(self._checkpoint_files(), device="cpu")
        )
        logger.info("Streamed checkpoint tensors in %.2fs", time.time() - started)

        force_cpu = bool(self.load_config.force_cpu_load_weights)
        if force_cpu:
            self._run_post_load_hooks(model, force_cpu=True)
            model.to(self.device)
        else:
            model.to(self.device)
            self._run_post_load_hooks(model, force_cpu=False)
        return model
