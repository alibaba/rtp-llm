import inspect
import json
import logging
import os
import platform
import re
import time
from contextlib import contextmanager, nullcontext
from itertools import chain
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn

import rtp_llm.models_py.weight_mapper as weight_mapper
from rtp_llm.models_py.downstream_heads import unpack_handler_mapping
from rtp_llm.models_py.borrowed_parameters import (
    FixedHostCheckpointStorage,
    bind_borrowed_parameters,
    borrowed_parameter_ids,
    exclude_borrowed_checkpoint_weights,
    validate_borrowed_parameters,
)
from rtp_llm.models_py.expert_placement import (
    ExpertPlacement,
    expert_placement_for_layer,
)
from rtp_llm.models_py.ft_checkpoint import (
    CheckpointTopology,
    FTCheckpoint,
    validate_format_parameter_ownership,
)
from rtp_llm.models_py.module_base import (
    CheckpointLoadTarget,
    RtpModule,
    collect_loaded_tensor_ids,
    validate_hf_checkpoint_storage,
)
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor, ReadSpec
from rtp_llm.models_py.registry import get_model_class, list_models, uses_isolated_checkpoint_load

logger = logging.getLogger(__name__)
_CUSTOM_WEIGHT_PREFIX = "__custom__."

_EXPERT_ID_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_EXPERT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_STACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:.*\.)?experts)\."
    r"(?P<projection>gate_up_proj|down_proj)"
    r"(?P<suffix>(?:\..+)?)$"
)
_FASTSAFETENSORS_REQUIRED_ARGUMENTS = frozenset(
    {
        "pg",
        "hf_weights_files",
        "use_tqdm_on_load",
        "device",
        "bbuf_size_kb",
        "use_shm",
        "nogds",
    }
)
_FASTSAFETENSORS_MIN_BBUF_SIZE_KB = 16 * 1024
_FASTSAFETENSORS_MAX_BBUF_SIZE_KB = 2 * 1024 * 1024
_CHECKPOINT_CATALOG_UNSET = object()
_CHECKPOINT_LOAD_PLAN_UNSET = object()


def _require_callable_name_filter(
    model_filter: Optional[Callable[[str], bool]], label: str
) -> Optional[Callable[[str], bool]]:
    if isinstance(model_filter, nn.Module):
        raise TypeError(f"{label} must be callable or None, not an nn.Module")
    if model_filter is not None and not callable(model_filter):
        raise TypeError(f"{label} must be callable or None")
    return model_filter


class _ExpertRangeFilter:
    """Select logical expert tensors needed by this EP rank's physical slots."""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any = None,
        parallelism_config: Any = None,
        placement_overrides: Optional[Dict[int, ExpertPlacement]] = None,
    ):
        for name, value in (
            ("num_experts", num_experts),
            ("ep_size", ep_size),
            ("ep_rank", ep_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
        if num_experts <= 0 or (model_config is None and num_experts % ep_size != 0):
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible by "
                f"ep_size={ep_size}"
            )
        experts_per_rank = num_experts // ep_size
        self.start_expert = ep_rank * experts_per_rank
        self.end_expert = self.start_expert + experts_per_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = num_experts
        self._model_config = model_config
        self._parallelism_config = parallelism_config
        self._placement_overrides = dict(placement_overrides or {})
        self._placement_cache: Dict[int, ExpertPlacement] = {}
        self.checkpoint_to_runtime_layer = None

    @staticmethod
    def _layer_index(name: str) -> int:
        match = _EXPERT_LAYER_RE.search(name)
        return int(match.group(1)) if match is not None else 0

    def _placement(self, name: str) -> Optional[ExpertPlacement]:
        if self._model_config is None:
            return None
        layer_index = self._layer_index(name)
        if self.checkpoint_to_runtime_layer is not None:
            layer_index = self.checkpoint_to_runtime_layer(layer_index)
        override = self._placement_overrides.get(layer_index)
        if override is not None:
            return override
        placement = self._placement_cache.get(layer_index)
        if placement is None:
            placement = expert_placement_for_layer(
                self._model_config,
                layer_index=layer_index,
                logical_expert_count=self.num_experts,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                parallelism_config=self._parallelism_config,
            )
            self._placement_cache[layer_index] = placement
        return placement

    def expert_id(self, name: str) -> Optional[int]:
        match = _EXPERT_ID_RE.search(name)
        if match is None:
            return None
        expert_id = int(match.group(1))
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"Checkpoint tensor {name!r} has expert id {expert_id} outside "
                f"[0, {self.num_experts})"
            )
        return expert_id

    def target_ep_ranks(self, name: str) -> Optional[Tuple[int, ...]]:
        expert_id = self.expert_id(name)
        if expert_id is None:
            return None
        placement = self._placement(name)
        if placement is not None:
            return placement.target_ep_ranks(expert_id)
        experts_per_rank = self.num_experts // self.ep_size
        return (expert_id // experts_per_rank,)

    def target_world_ranks(
        self, name: str, world_size: int
    ) -> Optional[Tuple[int, ...]]:
        target_ep_ranks = self.target_ep_ranks(name)
        if target_ep_ranks is None:
            return None
        return tuple(
            rank for rank in range(world_size) if rank % self.ep_size in target_ep_ranks
        )

    def should_load(self, name: str) -> bool:
        expert_id = self.expert_id(name)
        if expert_id is not None:
            placement = self._placement(name)
            if placement is not None:
                return bool(placement.local_indices(expert_id))
            return self.start_expert <= expert_id < self.end_expert
        if self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name):
            raise ValueError(
                "EP loading requires per-expert checkpoint keys; stacked all-expert "
                f"tensor {name!r} would materialize non-local experts"
            )
        return True

    def handles_safetensor(self, name: str) -> bool:
        return self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name) is not None

    def expand_safetensor(
        self, name: str, shape: Tuple[int, ...]
    ) -> List[Tuple[str, int]]:
        return [
            (read_spec.output_name, read_spec.slices[0])
            for read_spec in self.stacked_read_specs(name, shape)
            if read_spec.load
        ]

    def stacked_read_specs(
        self, name: str, shape: Tuple[int, ...]
    ) -> Tuple[ReadSpec, ...]:
        """Plan one stacked tensor as independently routed logical experts.

        FST first scatters the canonical contiguous expert ranges described by
        ``shard_rank``. It then sends only rows whose physical placement adds
        another target rank. Scratch applies the same specs as direct
        safetensors slices, so neither backend materializes the full all-expert
        tensor on every rank.
        """
        match = _STACKED_EXPERT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Tensor {name!r} is not a stacked expert tensor")
        if not shape or shape[0] != self.num_experts:
            raise ValueError(
                f"Stacked expert tensor {name!r} must contain "
                f"{self.num_experts} experts, got shape {shape}"
            )
        placement = self._placement(name)
        suffix = match.group("suffix") or ".weight"
        prefix = match.group("prefix")
        projection = match.group("projection")
        experts_per_rank = self.num_experts // self.ep_size
        specs = []
        for expert_id in range(self.num_experts):
            target_ranks = (
                placement.target_ep_ranks(expert_id)
                if placement is not None
                else (expert_id // experts_per_rank,)
            )
            specs.append(
                ReadSpec(
                    source_name=name,
                    output_name=f"{prefix}.{expert_id}.{projection}{suffix}",
                    source_shape=tuple(shape),
                    load=self.ep_rank in target_ranks,
                    slices=(expert_id,) + (slice(None),) * (len(shape) - 1),
                    partition="ep_stacked_expert",
                    shard_dim=0 if self.num_experts % self.ep_size == 0 else None,
                    shard_count=self.ep_size if self.num_experts % self.ep_size == 0 else 1,
                    shard_rank=expert_id // experts_per_rank if self.num_experts % self.ep_size == 0 else 0,
                    target_ranks=target_ranks,
                )
            )
        return tuple(specs)


class _StagedFSTLoadCoordinator:
    """Quantize staged leaves as soon as their checkpoint inputs are complete."""

    def __init__(
        self,
        model: nn.Module,
        staged_modules: List[RtpModule],
        target_device: str,
    ):
        names = {module: name or "<root>" for name, module in model.named_modules()}
        self._module_names: Dict[nn.Module, str] = {
            module: names[module] for module in staged_modules
        }
        self._target_device = target_device
        self._prepared: Set[nn.Module] = set()
        self._finalized: Set[nn.Module] = set()
        self._peak_active = 0

    def handles(self, child: nn.Module) -> bool:
        return child in self._module_names

    def prepare_child(self, child: nn.Module) -> None:
        """Materialize one staged leaf before a custom parent writes into it."""
        if child not in self._module_names:
            return
        if child in self._finalized:
            raise RuntimeError(
                "Checkpoint tensor arrived after staged postprocessing completed "
                f"for {self._module_names[child]} ({type(child).__name__})"
            )
        if child not in self._prepared:
            NewModelLoader._prepare_model_for_fastsafetensors(
                child, self._target_device
            )
            self._prepared.add(child)
            self._peak_active = max(
                self._peak_active, len(self._prepared - self._finalized)
            )

    def finalize_child_if_complete(self, child: nn.Module) -> None:
        """Postprocess a prepared leaf once its logical checkpoint is complete."""
        if child not in self._module_names:
            return
        if child not in self._prepared:
            raise RuntimeError(
                "Staged checkpoint completion was reported before preparation for "
                f"{self._module_names[child]} ({type(child).__name__})"
            )
        if child.checkpoint_load_complete():
            child.process_weights_during_staged_load()
            self._finalized.add(child)

    def load_child(self, child: nn.Module, weights: Any) -> None:
        if child not in self._module_names:
            child.load_weights(weights)
            return
        if child in self._finalized:
            # Optional metadata may live in a later file than the last required
            # weight. Let its owner validate it, but never reopen packed storage
            # or postprocess the leaf twice.
            validation_only = getattr(
                getattr(child, "quant_method", None),
                "is_validation_only_checkpoint_tensor",
                None,
            )
            if callable(validation_only) and all(
                validation_only(child, name) for name in weights
            ):
                child.load_weights(weights)
                return
        self.prepare_child(child)
        child.load_weights(weights)
        self.finalize_child_if_complete(child)

    def finish(self) -> None:
        unfinished = [
            f"{self._module_names[module]} ({type(module).__name__})"
            for module in self._module_names
            if module not in self._finalized
        ]
        if unfinished:
            raise RuntimeError(
                "FST staged postprocessing did not receive a complete checkpoint "
                f"for {unfinished[:10]}"
            )
        logger.info(
            "FST online quantization finalized %d leaves; peak concurrently "
            "staged leaves=%d",
            len(self._finalized),
            self._peak_active,
        )


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

    @classmethod
    def from_environment(cls) -> "NewLoaderLoadMethod":
        value = os.environ.get("LOAD_METHOD", "").strip().lower()
        try:
            return cls(value) if value else cls.AUTO
        except ValueError as exc:
            raise ValueError(
                f"Unsupported LOAD_METHOD environment value {value!r}"
            ) from exc


@dataclass(frozen=True)
class NewLoaderConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    # The service already resolved CLI > environment. Only direct library
    # callers that omit this option should take their default from the environment.
    load_method: NewLoaderLoadMethod = field(
        default_factory=NewLoaderLoadMethod.from_environment
    )
    quant_config: Any = None
    parallelism_config: Any = None
    moe_config: Any = None
    fmha_config: Any = None
    py_hw_kernel_config: Any = None
    max_generate_batch_size: int = 0
    device_resource_config: Any = None
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    ffn_tp_size: Optional[int] = None
    ffn_tp_rank: Optional[int] = None
    lm_head_tp_size: Optional[int] = None
    lm_head_tp_rank: Optional[int] = None
    # Construction hint only: binding still validates each owner and parameter.
    borrowed_parameter_paths: Tuple[str, ...] = ()
    custom_weight_mappings: Tuple[Tuple[str, str] | Tuple[str, str, torch.dtype], ...] = ()
    custom_projection_layers: Tuple[Tuple[str, Optional[str]], ...] = ()
    # Experimental consumer bypass retains source file staging until batch
    # close and adds CPU range I/O. Do not enable by default without proving
    # the installed FST's early-release semantics and GPU peak behavior.
    experimental_fst_segmented_reads: bool = False
    # Stage checkpoint input and conversions on CPU, independently of the
    # eventual runtime device. GPU-only leaf postprocessing runs after migration.
    force_cpu_load_weights: bool = False
    # Opaque owner-validated coupled tensor references (e.g. scaled vocabulary).
    prepared_runtime_loan: Any = field(default=None, repr=False, compare=False)
    resident_construction: Any = field(default=None, repr=False, compare=False)
    reserve_runtime_mem_mb: int = 0
    moe_pure_tp_preshard: bool = False

    @classmethod
    def from_runtime(cls, parallelism_config, **options) -> "NewLoaderConfig":
        """Bind checkpoint partitions to this role's finalized service topology.

        Accelerators in the same physical TP group may hold replicated attention
        and/or FFN weights (e.g. CP prefill). The LM head still joins main's
        physical TP group; do not infer its partition from either projection.
        """
        from rtp_llm.models_py.distributed.rank_layout import RankLayout

        pc = parallelism_config
        layout = RankLayout.from_parallelism_config(pc)
        world_rank = pc.world_rank
        coord = layout.coord_of(world_rank)
        if pc.ep_size not in (1, layout.lane_stride()):
            raise ValueError("Runtime EP must be single-rank or span its PP stage")
        expected = (
            coord.pp,
            coord.dp,
            coord.tp,
            layout.ep_rank_of(world_rank, pc.ep_size),
        )
        actual = (pc.pp_rank, pc.dp_rank, pc.tp_rank, pc.ep_rank)
        if actual != expected:
            raise ValueError(
                f"Checkpoint PP/DP/TP/EP ranks {actual} disagree with "
                f"runtime world rank {world_rank}: expected {expected}"
            )
        if torch.distributed.is_initialized() and (
            torch.distributed.get_rank() != world_rank
            or torch.distributed.get_world_size() != layout.world_size()
        ):
            raise ValueError(
                "Configured checkpoint topology disagrees with distributed WORLD"
            )
        return cls(
            parallelism_config=pc,
            tp_size=pc.tp_size,
            tp_rank=pc.tp_rank,
            ep_size=pc.ep_size,
            ep_rank=pc.ep_rank,
            attn_tp_size=pc.get_attn_tp_size(),
            attn_tp_rank=pc.get_attn_tp_rank(),
            ffn_tp_size=pc.get_ffn_tp_size(),
            ffn_tp_rank=pc.get_ffn_tp_rank(),
            lm_head_tp_size=pc.tp_size,
            lm_head_tp_rank=pc.tp_rank,
            **options,
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_generate_batch_size, bool)
            or not isinstance(self.max_generate_batch_size, int)
            or self.max_generate_batch_size < 0
        ):
            raise ValueError("max_generate_batch_size must be a non-negative integer")
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
            if prefix in ("attn_tp", "ffn_tp"):
                if size not in (1, self.tp_size):
                    raise ValueError(
                        f"{size_name}={size} must be either 1 or the physical "
                        f"tp_size={self.tp_size}; independent TP subgroups are "
                        "not supported by Group.TP collectives"
                    )
                expected_rank = 0 if size == 1 else self.tp_rank
                if rank != expected_rank:
                    raise ValueError(
                        f"{rank_name}={rank} does not match the supported "
                        f"{prefix} topology: expected rank={expected_rank} for "
                        f"size={size}"
                    )
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"Invalid EP partition: rank={self.ep_rank}, size={self.ep_size}"
            )
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("compute_dtype must be a torch.dtype")
        if not isinstance(self.force_cpu_load_weights, bool):
            raise TypeError("force_cpu_load_weights must be a bool")
        if not isinstance(self.experimental_fst_segmented_reads, bool):
            raise TypeError("experimental_fst_segmented_reads must be a bool")
        if not isinstance(self.custom_weight_mappings, tuple):
            raise TypeError("custom_weight_mappings must be a tuple")
        runtime_names = set()
        checkpoint_names = set()
        for mapping in self.custom_weight_mappings:
            runtime_name, checkpoint_name, _ = unpack_handler_mapping(mapping)
            if not isinstance(runtime_name, str) or not runtime_name:
                raise TypeError("custom runtime weight names must be non-empty strings")
            if not runtime_name.startswith(_CUSTOM_WEIGHT_PREFIX):
                raise ValueError(
                    f"custom runtime weight {runtime_name!r} must start with "
                    f"{_CUSTOM_WEIGHT_PREFIX!r}"
                )
            if not isinstance(checkpoint_name, str) or not checkpoint_name:
                raise TypeError(
                    "custom checkpoint weight names must be non-empty strings"
                )
            if runtime_name in runtime_names:
                raise ValueError(f"Duplicate custom runtime weight {runtime_name!r}")
            if checkpoint_name in checkpoint_names:
                raise ValueError(
                    f"Duplicate custom checkpoint weight {checkpoint_name!r}"
                )
            runtime_names.add(runtime_name)
            checkpoint_names.add(checkpoint_name)
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
        parameter_aliases=None,
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
        self.model_path = model_path
        self.parameter_aliases = dict(parameter_aliases or {})
        paths = tuple(self.parameter_aliases)
        if paths != effective_config.borrowed_parameter_paths:
            effective_config = replace(effective_config, borrowed_parameter_paths=paths)
        self.load_config = effective_config
        self._ckpt_files = None
        self._checkpoint_catalog_cache = _CHECKPOINT_CATALOG_UNSET
        self._selected_load_method: Optional[NewLoaderLoadMethod] = None
        self._eplb_ft_checkpoint: Optional[FTCheckpoint] = None
        self._w8a8_source = None

    @property
    def loaded_method(self) -> Optional[NewLoaderLoadMethod]:
        """Backend of the last load attempt, or None until that attempt succeeds."""
        return self._selected_load_method

    def _finetune_overlay(self, checkpoint_files=None):
        from rtp_llm.models_py.finetune_overlay import FinetuneCheckpointOverlay

        directory = (
            self.model_config.get("ptuning_path", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "ptuning_path", "")
        )
        return FinetuneCheckpointOverlay.discover(
            directory,
            self._checkpoint_files() if checkpoint_files is None else checkpoint_files,
        )

    @staticmethod
    def _moe_expert_for_layer(model: nn.Module, layer_id: int) -> RtpModule:
        matches = [
            module
            for module in model.modules()
            if bool(getattr(module, "_supports_newloader_eplb", False))
            and getattr(module, "layer_idx", None) == layer_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one NewLoader MoE expert module for layer {layer_id}, "
                f"found {len(matches)}"
            )
        return matches[0]

    @torch.inference_mode()
    def load_moe_layer_weights(
        self,
        model: nn.Module,
        *,
        layer_id: int,
        physical_to_logical: List[int],
        ep_rank: int,
        ep_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Reload one physical MoE layer for a dynamic EPLB placement."""
        if getattr(model, "_ft_eplb_stats_only", False):
            raise ValueError("FT checkpoint does not declare an expert reload layout")
        if (ep_rank, ep_size) != (self.load_config.ep_rank, self.load_config.ep_size):
            raise ValueError(
                "C++ EPLB EP partition does not match NewLoader: "
                f"callback=({ep_rank}, {ep_size}) loader="
                f"({self.load_config.ep_rank}, {self.load_config.ep_size})"
            )
        live_experts = self._moe_expert_for_layer(model, layer_id)
        placement = ExpertPlacement(
            logical_expert_count=live_experts.num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            physical_to_logical=tuple(physical_to_logical),
        )
        if placement.physical_expert_count != live_experts.physical_num_experts:
            raise ValueError("Dynamic EPLB cannot change the physical expert count")

        # The callback is entered from a C++ std::thread.  CUDA's current
        # device is thread-local, so a newly-created loading thread otherwise
        # defaults to cuda:0 even when this process owns another local rank.
        # Pin the worker to the device of the live layer and use that explicit
        # device for every allocation/load below.  Using the rank-generic
        # ``load_config.device`` (usually just ``"cuda"``) can make two EP
        # ranks issue FST/NCCL operations from the same GPU and deadlock.
        live_device = live_experts.w13.device
        if live_device.type == "cuda":
            if live_device.index is None:
                raise RuntimeError(
                    "Dynamic EPLB live MoE weight must have an explicit CUDA device"
                )
            torch.cuda.set_device(live_device)
        reload_device = str(live_device)

        with torch.device("cpu"):
            reload_experts = live_experts.clone_empty_for_placement(placement)
        self._validate_runtime_backends(reload_experts, reload_device)

        if self._eplb_ft_checkpoint is not None:
            self._eplb_ft_checkpoint.reload_expert_module(live_experts, reload_experts)
            return self._finish_eplb_reload(
                live_experts, reload_experts, reload_device, layer_id, ep_rank,
                self._eplb_ft_checkpoint.read_backend,
            )

        model_filter = self._model_checkpoint_name_filter(model)
        prefix = live_experts.prefix + "."

        def target_layer_filter(name: str) -> bool:
            if model_filter is not None and not model_filter(name):
                return False
            mapped = self._mapped_checkpoint_name(model, name)
            return mapped.startswith(prefix)

        expert_filter = _ExpertRangeFilter(
            live_experts.num_experts,
            ep_size,
            ep_rank,
            self.model_config,
            self.load_config.parallelism_config,
            placement_overrides={layer_id: placement},
        )
        expert_filter.checkpoint_to_runtime_layer = getattr(model, "checkpoint_to_runtime_expert_layer", None)
        checkpoint_files = self._checkpoint_files()
        overlay = self._finetune_overlay()
        overlay_has_layer = overlay is not None and any(
            target_layer_filter(name) for name in overlay.names
        )
        selected_files = weight_mapper.select_safetensor_files(
            self._resolved_model_path(),
            checkpoint_files,
            target_layer_filter,
            self._safetensor_catalog(),
            allow_empty=overlay_has_layer,
        )
        if not selected_files and not overlay_has_layer:
            raise FileNotFoundError(
                f"No checkpoint shards contain MoE weights for layer {layer_id}"
            )
        name_filter = self._checkpoint_name_filter(target_layer_filter, expert_filter)
        base_filter = (
            overlay.base_filter(name_filter) if overlay is not None else name_filter
        )
        read_spec_resolver = self._checkpoint_read_spec_resolver(
            model, target_layer_filter, expert_filter
        )
        base_resolver = (
            overlay.base_read_spec_resolver(read_spec_resolver)
            if overlay is not None
            else read_spec_resolver
        )
        # EP1 owns every logical row. Per-key and stacked PyTorch weights are
        # dispatched by the staging owner using its *new* slot placement. No
        # safetensors slice expansion is necessary (or valid for a .bin file).
        expander = expert_filter if ep_size > 1 else None
        base_expander = (
            overlay.base_slice_expander(expander)
            if overlay is not None
            else expander
        )
        method = self._selected_load_method
        if method is None:
            method = self._select_load_method(
                self.load_config.load_method, model, selected_files, expert_filter
            )

        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors

            # FST's native range reader is rank-local and memory-bounded. Do
            # not use the whole-file loader or inference's WORLD communicator.
            weights = iter_fst_range_tensors(
                selected_files,
                catalog=self._checkpoint_catalog(),
                name_filter=base_filter,
                slice_expander=base_expander,
                read_spec_resolver=base_resolver,
            )
        else:
            weights = weight_mapper.get_all_weights(
                selected_files,
                catalog=self._checkpoint_catalog(),
                device="cpu",
                name_filter=base_filter,
                safetensor_slice_expander=base_expander,
                read_spec_resolver=base_resolver,
                **({"w8a8_source": self._w8a8_source} if self._w8a8_source is not None else {}),
            )

        if overlay is not None:
            weights = overlay.prepend(
                weights, name_filter, read_spec_resolver, expander
            )

        try:
            for name, tensor in weights:
                mapped_name = self._mapped_checkpoint_name(model, name)
                if not mapped_name.startswith(prefix):
                    raise RuntimeError(
                        f"Dynamic EPLB received non-layer tensor {name!r} "
                        f"mapped to {mapped_name!r}"
                    )
                reload_experts.load_weights({mapped_name[len(prefix) :]: tensor})
        finally:
            close_weights = getattr(weights, "close", None)
            if callable(close_weights):
                close_weights()

        return self._finish_eplb_reload(
            live_experts, reload_experts, reload_device, layer_id, ep_rank, method.value
        )

    @staticmethod
    def _finish_eplb_reload(
        live_experts, reload_experts, reload_device, layer_id, ep_rank, backend
    ):
        reload_experts.validate_weights_loaded()
        reload_experts.to(reload_device)
        result = reload_experts.prepare_eplb_reload_weights()
        # Some kernels execute transformed copies rather than the owner's
        # raw tensors. Stage those copies too, before any rank votes ready.
        result.update(live_experts.eplb_kernel_weights(result, prepare=True))
        logger.info(
            "NewLoader reloaded EPLB layer %d on EP rank %d with %s",
            layer_id,
            ep_rank,
            backend,
        )
        # Return detached CPU tensors so background-thread CUDA postprocessing
        # is complete before the all-rank ready flag is published.  The C++
        # request boundary then installs this exact runtime layout into the
        # Python model's stable device tensor addresses.
        return {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in result.items()
        }

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
            catalog = self._checkpoint_catalog()
            if catalog is not None:
                self._ckpt_files = list(catalog.shard_files)
            else:
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

    def _safetensor_catalog(self) -> Optional[weight_mapper.CheckpointCatalog]:
        catalog = self._checkpoint_catalog()
        return catalog if catalog is not None and catalog.format == "safetensors" else None

    def _checkpoint_catalog(self) -> Optional[weight_mapper.CheckpointCatalog]:
        if self._checkpoint_catalog_cache is _CHECKPOINT_CATALOG_UNSET:
            self._checkpoint_catalog_cache = weight_mapper.checkpoint_catalog(
                self._resolved_model_path()
            )
        return self._checkpoint_catalog_cache

    def _expert_filter(self):
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
        if num_experts < 0:
            raise ValueError(
                f"model_config num_experts must be non-negative, got {num_experts}"
            )
        if num_experts == 0:
            model_type = (
                self.model_config.get("model_type", "")
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "model_type", "")
            )
            moe_top_k = (
                self.model_config.get(
                    "moe_k", self.model_config.get("num_experts_per_tok", 0)
                )
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_k", 0)
            )
            moe_style = (
                self.model_config.get("moe_style", 0)
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_style", 0)
            )
            for name, value in (("moe_k", moe_top_k), ("moe_style", moe_style)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"model_config {name} must be an integer, got {value!r}"
                    )
                if value < 0:
                    raise ValueError(
                        f"model_config {name} must be non-negative, got {value}"
                    )
            if "moe" in str(model_type).lower() or moe_top_k > 0 or moe_style > 0:
                raise ValueError(
                    "EP is configured for a MoE model "
                    f"(model_type={model_type!r}, moe_k={moe_top_k}, "
                    f"moe_style={moe_style}), but "
                    "model_config reports zero experts"
                )
            logger.warning(
                "EP size %d is configured, but model_config reports zero experts; "
                "model_type=%r and zero moe_k/moe_style markers identify a dense "
                "model, so expert checkpoint filtering is disabled",
                self.load_config.ep_size,
                model_type,
            )
            return None
        return _ExpertRangeFilter(
            num_experts,
            self.load_config.ep_size,
            self.load_config.ep_rank,
            self.model_config,
            self.load_config.parallelism_config,
        )

    @staticmethod
    def _model_checkpoint_name_filter(
        model: RtpModule,
    ) -> Optional[Callable[[str], bool]]:
        return _require_callable_name_filter(
            model.checkpoint_weight_name_filter(),
            f"{type(model).__name__}.checkpoint_weight_name_filter() return value",
        )

    @staticmethod
    def _checkpoint_name_filter(
        model_filter: Optional[Callable[[str], bool]],
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> Optional[Callable[[str], bool]]:
        model_filter = _require_callable_name_filter(model_filter, "model_filter")
        if model_filter is None and expert_filter is None:
            return None

        def should_load(name: str) -> bool:
            if model_filter is not None and not model_filter(name):
                return False
            if expert_filter is None:
                return True
            if expert_filter.handles_safetensor(name):
                return True
            return expert_filter.should_load(name)

        return should_load

    @staticmethod
    def _mapped_checkpoint_name(model: RtpModule, name: str) -> str:
        mapper = getattr(model, "WEIGHTS_MAPPER", None)
        map_name = getattr(mapper, "map_name", None)
        return map_name(name) if callable(map_name) else name

    def _checkpoint_read_spec_resolver(
        self,
        model: RtpModule,
        model_filter: Optional[Callable[[str], bool]],
        expert_filter: Optional[_ExpertRangeFilter],
        checkpoint_load_plan: Optional[Dict[str, CheckpointLoadTarget]] = None,
    ) -> Callable[[str, Tuple[int, ...]], ReadSpec]:
        model_filter = _require_callable_name_filter(model_filter, "model_filter")
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = max(self.load_config.tp_size, self.load_config.ep_size)

        parallelism = self.load_config.parallelism_config
        if parallelism is not None and getattr(parallelism, "pp_size", 1) > 1:
            from rtp_llm.models_py.distributed.rank_layout import RankLayout

            layout = RankLayout.from_parallelism_config(parallelism)
            if torch.distributed.is_initialized() and world_size != layout.world_size():
                raise ValueError(
                    "Distributed WORLD does not match the PP checkpoint layout"
                )
            # Expert targets are transport-group ranks, not WORLD peers. The
            # FST adapter translates these peers to global ranks exactly once.
            # Other PP stages own different layer IDs and are not consumers.
            world_size = layout.lane_stride()

        planned_sources = {}

        def resolve(name: str, shape: Tuple[int, ...]) -> ReadSpec:
            shape = tuple(shape)
            if model_filter is not None and not model_filter(name):
                return ReadSpec.skip(name, shape)

            # A model transform can split one physical tensor into multiple
            # logical parameters (e.g. interleaved Q/gate FP8 scales). No one
            # leaf can validate its source shape or own a cached direct target.
            read_spec = None
            if model.allow_checkpoint_read_spec(name):
                mapped_name = self._mapped_checkpoint_name(model, name)
                load_target, read_spec = model.resolve_checkpoint_load_plan(
                    mapped_name, shape
                )
                if checkpoint_load_plan is not None and load_target is not None:
                    previous = planned_sources.setdefault(mapped_name, name)
                    if previous != name:
                        raise ValueError(f"Ambiguous checkpoint sources {previous!r} and {name!r} map to {mapped_name!r}")
                    checkpoint_load_plan[mapped_name] = load_target
            if read_spec is None:
                read_spec = ReadSpec.full(name, shape)
            else:
                read_spec = read_spec.for_checkpoint_name(name)

            if expert_filter is not None:
                target_ranks = expert_filter.target_world_ranks(name, world_size)
                if target_ranks is not None:
                    read_spec = read_spec.with_targets(
                        target_ranks, expert_filter.should_load(name)
                    )
            return read_spec

        return resolve

    def _validate_ep_checkpoint_format(
        self,
        checkpoint_files,
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> None:
        if expert_filter is None:
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

    def _create_model(self, *, ft_load_config=None, ft_checkpoint=None) -> nn.Module:
        model_type = self._model_type()
        try:
            model_cls = get_model_class(model_type)
        except KeyError as exc:
            raise ValueError(
                f"Model type {model_type!r} is not registered; available={list_models()}"
            ) from exc
        # A shared config name can have genuinely different native runtimes.
        # Resolve per model instance; do not cache this platform/role choice.
        resolve_runtime = getattr(model_cls, "resolve_runtime_class", None)
        if resolve_runtime is not None:
            model_cls = resolve_runtime(self.model_config)
        construct = getattr(model_cls, "from_ft_checkpoint_header", None)
        load = ft_load_config or self.load_config
        if ft_checkpoint is not None and callable(construct):
            model = construct(self.model_config, load, ft_checkpoint)
        else:
            model = model_cls(self.model_config, load)
        if not isinstance(model, RtpModule):
            raise TypeError(
                f"Registered model {model_cls.__name__} must inherit RtpModule to "
                "provide newloader completeness validation"
            )
        if self.load_config.custom_projection_layers and not getattr(
            model, "supports_handler_projection_layers", False
        ):
            raise ValueError(
                f"{type(model).__name__} cannot validate handler projection layers"
            )
        if self.load_config.custom_weight_mappings and not bool(
            getattr(model, "supports_custom_weight_mappings", False)
        ):
            raise NotImplementedError(
                f"{type(model).__name__} does not support downstream custom weights"
            )
        try:
            source = inspect.getfile(model_cls)
        except (TypeError, OSError):
            source = "<unknown>"
        logger.info(
            "Created newloader model %s from %s", model_cls.__qualname__, source
        )
        return model

    def _create_checkpoint_model(self, checkpoint, *, constructor_header=None):
        quant = self.load_config.quant_config
        adapt = getattr(quant, "for_ft_snapshot", None)
        encoded = adapt() if isinstance(checkpoint, FTCheckpoint) and callable(adapt) else quant
        if isinstance(checkpoint, FTCheckpoint):
            load = self.load_config if encoded is quant else replace(self.load_config, quant_config=encoded)
            return self._create_model(ft_load_config=load, ft_checkpoint=constructor_header if constructor_header is not None else checkpoint)
        if encoded is quant:
            return self._create_model()
        return self._create_model(ft_load_config=replace(self.load_config, quant_config=encoded))

    @staticmethod
    def _validate_loaded_weights(model: nn.Module) -> None:
        loaded_tensor_ids = frozenset(collect_loaded_tensor_ids(model))
        root_validator = getattr(model, "validate_weights_loaded", None)
        if not callable(root_validator):
            raise TypeError(
                f"Registered model {type(model).__name__} has no weight completeness validator"
            )
        root_validator(loaded_tensor_ids)

        for module_name, module in model.named_modules():
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
                try:
                    validator(loaded_tensor_ids)
                except Exception as exc:
                    context = (
                        f"Weight validation failed for {module_name} "
                        f"({type(module).__name__})"
                    )
                    if exc.args:
                        exc.args = (f"{context}: {exc.args[0]}", *exc.args[1:])
                    else:
                        exc.args = (context,)
                    raise

    @staticmethod
    def _run_post_load_hooks(model: nn.Module) -> None:
        for module in model.modules():
            if getattr(module, "_rtp_borrowed_parameter_names", ()):
                # The owning model already postprocessed this storage.
                continue
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    @staticmethod
    def _prepare_model_for_fastsafetensors(
        model: nn.Module,
        target_device: str,
        deferred_modules: Optional[List[RtpModule]] = None,
        *,
        keep_on_device: bool = False,
    ) -> None:
        """Allocate checkpoint-backed parameters without copying placeholders.

        NewLoader constructs the model on CPU so capability checks and AUTO
        selection do not consume accelerator memory.  A normal ``model.to``
        would then copy every uninitialized parameter to the accelerator just
        before fastsafetensors overwrites it.  Allocate fresh parameter storage
        on the target instead, while retaining the values of buffers that may
        be derived by the constructor rather than supplied by the checkpoint.

        ``RtpModule._apply`` memoizes shared tensors and restores aliases, so
        tied parameters and parameter/buffer aliases remain intact.
        """
        device = torch.device(target_device)
        validate_borrowed_parameters(model, target_device)
        deferred_tensor_ids = set()
        for module in deferred_modules or ():
            deferred_tensor_ids.update(
                id(tensor)
                for tensor in list(module.parameters(recurse=True))
                + list(module.buffers(recurse=True))
                if tensor is not None
            )
        parameter_ids = {
            id(parameter)
            for parameter in model.parameters(recurse=True)
            if parameter is not None
        }
        borrowed_ids = borrowed_parameter_ids(model)

        def allocate_parameter_or_move_buffer(tensor: torch.Tensor) -> torch.Tensor:
            if id(tensor) in borrowed_ids:
                return tensor
            if id(tensor) in deferred_tensor_ids:
                return tensor
            if keep_on_device and tensor.device == device:
                return tensor
            if id(tensor) in parameter_ids:
                return torch.empty_like(tensor, device=device)
            return tensor.to(device=device)

        model._apply(allocate_parameter_or_move_buffer)

    @staticmethod
    def _staged_device_modules(model: nn.Module) -> List[RtpModule]:
        return [
            module
            for module in model.modules()
            if isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
        ]

    @staticmethod
    def _validate_staged_device_modules(
        model: nn.Module,
        staged_modules: List[RtpModule],
        *,
        require_load_complete: bool = False,
    ) -> None:
        if not staged_modules:
            return
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
            if require_load_complete and (
                type(module).checkpoint_load_complete
                is RtpModule.checkpoint_load_complete
            ):
                raise RuntimeError(
                    f"Staged leaf {type(module).__name__} does not implement "
                    "checkpoint_load_complete()"
                )

    @staticmethod
    def _fastsafetensors_api():
        try:
            from fastsafetensors import ParallelLoader, SingleGroup
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                "the fastsafetensors package is unavailable or failed to import: "
                f"{exc}"
            ) from exc

        for symbol_name, symbol in (
            ("ParallelLoader", ParallelLoader),
            ("SingleGroup", SingleGroup),
        ):
            if not callable(symbol):
                raise RuntimeError(f"fastsafetensors.{symbol_name} must be callable")
        if not callable(getattr(ParallelLoader, "iterate_weights", None)):
            raise RuntimeError("fastsafetensors.ParallelLoader lacks iterate_weights")
        try:
            parameters = inspect.signature(ParallelLoader.__init__).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot inspect fastsafetensors.ParallelLoader constructor"
            ) from exc
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        missing = _FASTSAFETENSORS_REQUIRED_ARGUMENTS - set(parameters)
        if missing and not accepts_kwargs:
            raise RuntimeError(
                "fastsafetensors.ParallelLoader is incompatible with NewLoader; "
                f"missing constructor arguments {sorted(missing)}"
            )
        return ParallelLoader, SingleGroup

    @staticmethod
    def _safetensor_metadata_summary(
        model_path: str,
        checkpoint_files: List[str],
        catalog: Optional[weight_mapper.CheckpointCatalog] = None,
    ) -> Tuple[List[str], List[str]]:
        """Return duplicate names and stacked expert names without reading data.

        A valid Hugging Face index is the preferred metadata source.  The
        streaming iterator still detects duplicate names while loading, so an
        index avoids a second full header walk without weakening correctness.
        Header inspection remains the fallback for unindexed checkpoints.
        """
        selected_identities = {
            os.path.realpath(os.fspath(path)) for path in checkpoint_files
        }
        index_entries = weight_mapper.safetensor_index_entries(model_path, catalog)
        if index_entries is not None:
            identity_by_shard = {
                shard_path: os.path.realpath(shard_path)
                for shard_path in dict.fromkeys(path for _, path in index_entries)
            }
            indexed_identities = set(identity_by_shard.values())
            if selected_identities.issubset(indexed_identities):
                stacked = [
                    name
                    for name, shard_path in index_entries
                    if identity_by_shard[shard_path] in selected_identities
                    and _STACKED_EXPERT_RE.fullmatch(name)
                ]
                return [], stacked

        from safetensors import safe_open

        seen = set()
        duplicates = set()
        stacked = []
        for path in checkpoint_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        duplicates.add(name)
                    seen.add(name)
                    if _STACKED_EXPERT_RE.fullmatch(name):
                        stacked.append(name)
        return sorted(duplicates), sorted(stacked)

    def _fastsafetensors_distributed_reason(
        self, expert_filter: Optional[_ExpertRangeFilter] = None
    ) -> Optional[str]:
        """Validate WORLD and rank fields before choosing an FST subgroup."""
        parallelism = self.load_config.parallelism_config
        expected_world_size = getattr(
            parallelism,
            "world_size",
            max(self.load_config.tp_size, self.load_config.ep_size),
        )
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or expected_world_size <= 0
        ):
            return f"invalid configured world_size={expected_world_size!r}"
        pp_size = getattr(parallelism, "pp_size", 1)
        if expert_filter is not None and pp_size > 1:
            from rtp_llm.models_py.distributed.rank_layout import RankLayout

            try:
                layout = RankLayout.from_parallelism_config(parallelism)
            except (AttributeError, ValueError) as error:
                return f"invalid PP stage layout for FST: {error}"
            if self.load_config.ep_size != layout.lane_stride():
                return (
                    "EP fastsafetensors loading requires ep_size to span one PP stage"
                )
        elif (
            expert_filter is not None
            and self.load_config.ep_size != expected_world_size
        ):
            return (
                "EP fastsafetensors loading requires ep_size to span WORLD: "
                f"ep_size={self.load_config.ep_size}, "
                f"world_size={expected_world_size}"
            )
        if not torch.distributed.is_initialized():
            if expected_world_size > 1:
                return (
                    "TP/EP fastsafetensors loading requires an initialized "
                    f"distributed WORLD group of size {expected_world_size}"
                )
            return None

        world_size = torch.distributed.get_world_size()
        if world_size != expected_world_size:
            return (
                "distributed WORLD group does not match the configured "
                f"topology: world_size={world_size}, expected={expected_world_size}"
            )
        world_rank = torch.distributed.get_rank()
        expected_tp_rank = world_rank % self.load_config.tp_size
        expected_ep_rank = world_rank % self.load_config.ep_size
        if (
            self.load_config.tp_rank != expected_tp_rank
            or self.load_config.ep_rank != expected_ep_rank
        ):
            return (
                "distributed WORLD rank does not match the NewLoader TP/EP "
                f"ranks: world_rank={world_rank}, loader_tp_rank="
                f"{self.load_config.tp_rank}, expected_tp_rank={expected_tp_rank}, "
                f"loader_ep_rank={self.load_config.ep_rank}, "
                f"expected_ep_rank={expected_ep_rank}"
            )
        if expert_filter is not None and pp_size > 1:
            try:
                self._fastsafetensors_stage_ep_group()
            except (RuntimeError, ValueError, KeyError, AttributeError) as error:
                return f"invalid FST EP stage process group: {error}"
        return None

    @staticmethod
    def _runtime_checkpoint_group(group):
        from rtp_llm.models_py.distributed.collective_torch import _get_group

        return _get_group(group)

    def _stage_group(self):
        from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout
        from rtp_llm.models_py.fastsafetensors_read_spec import (
            validate_ep_process_group,
        )

        pc = self.load_config.parallelism_config
        layout = RankLayout.from_parallelism_config(pc)
        stage = self._runtime_checkpoint_group(Group.STAGE)
        # The prerequisite collective covers the entire stage, including DP
        # peers, even for a dense draft which has no expert partition.
        validate_ep_process_group(
            stage,
            pc,
            layout.lane_stride(),
            layout.rank_in_group(Group.STAGE, pc.world_rank),
        )
        return stage

    def _fastsafetensors_stage_ep_group(self):
        from rtp_llm.models_py.distributed.rank_layout import Group
        from rtp_llm.models_py.fastsafetensors_read_spec import (
            validate_ep_process_group,
        )

        process_group = self._runtime_checkpoint_group(Group.STAGE)
        validate_ep_process_group(
            process_group,
            self.load_config.parallelism_config,
            self.load_config.ep_size,
            self.load_config.ep_rank,
        )
        return process_group

    def _fastsafetensors_process_group(
        self,
        SingleGroup,
        checkpoint_files: List[str],
        expert_filter: Optional[_ExpertRangeFilter],
    ):
        """Return the smallest group containing every checkpoint consumer."""
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files):
            logger.info("NewLoader fastsafetensors process group: rank-local")
            return SingleGroup()
        if not torch.distributed.is_initialized():
            logger.info("NewLoader fastsafetensors process group: single-rank")
            return SingleGroup()
        if expert_filter is not None:
            if getattr(self.load_config.parallelism_config, "pp_size", 1) > 1:
                process_group = self._fastsafetensors_stage_ep_group()
                logger.info("NewLoader fastsafetensors process group: EP PP-stage")
                return process_group
            logger.info("NewLoader fastsafetensors process group: EP WORLD")
            return torch.distributed.group.WORLD
        if self.load_config.tp_size <= 1:
            logger.info("NewLoader fastsafetensors process group: DP-local rank")
            return SingleGroup()
        world_size = torch.distributed.get_world_size()
        if world_size == self.load_config.tp_size:
            logger.info("NewLoader fastsafetensors process group: TP WORLD")
            return torch.distributed.group.WORLD
        from rtp_llm.models_py.distributed.rank_layout import Group
        from rtp_llm.models_py.fastsafetensors_read_spec import (
            validate_tp_process_group,
        )

        process_group = self._runtime_checkpoint_group(Group.TP)
        validate_tp_process_group(
            process_group, self.load_config.tp_size, self.load_config.tp_rank
        )
        group_size = torch.distributed.get_world_size(process_group)
        group_rank = torch.distributed.get_rank(process_group)
        logger.info(
            "NewLoader fastsafetensors process group: TP subgroup rank=%d size=%d",
            group_rank,
            group_size,
        )
        return process_group

    @staticmethod
    def _fastsafetensors_bbuf_size_kb(
        checkpoint_files: List[str], *, use_host_staging: bool = False
    ) -> int:
        # The nogds and SHM readers allocate this bounce buffer as pinned host
        # memory for every loader rank.  Sizing it to a multi-GiB checkpoint
        # shard can exhaust the process/container memlock budget before the
        # first read.  Keep the library's 16 MiB default for host staging;
        # shard-sized buffering is useful only for GDS.
        if use_host_staging:
            return _FASTSAFETENSORS_MIN_BBUF_SIZE_KB
        max_shard_bytes = max(os.path.getsize(path) for path in checkpoint_files)
        max_shard_kb = (max_shard_bytes + 1023) // 1024
        return min(
            _FASTSAFETENSORS_MAX_BBUF_SIZE_KB,
            max(_FASTSAFETENSORS_MIN_BBUF_SIZE_KB, max_shard_kb),
        )

    @staticmethod
    def _checkpoint_uses_local_block_device(
        checkpoint_files: List[str],
    ) -> Optional[bool]:
        """Whether any selected checkpoint resides on a visible block device."""
        if platform.system() != "Linux" or not checkpoint_files:
            return None
        sys_block_devices = "/sys/dev/block"
        if not os.path.isdir(sys_block_devices):
            return None
        try:
            devices = {
                os.stat(os.path.realpath(os.fspath(path))).st_dev
                for path in checkpoint_files
            }
        except OSError:
            return None
        return any(
            os.path.exists(
                os.path.join(
                    sys_block_devices,
                    f"{os.major(device)}:{os.minor(device)}",
                )
            )
            for device in devices
        )

    @staticmethod
    def _cuda_local_gds_prerequisites_available(
        checkpoint_files: List[str],
    ) -> bool:
        """Quickly reject local block mounts that cannot provide a GDS path.

        Local block storage GDS needs either nvidia-fs or the upstream PCI
        P2PDMA support used by CUDA 12.8+.  Non-block filesystems remain
        subject to the native cuFile registration probe because vendor/RDMA
        implementations have different prerequisites.
        """
        uses_local_block_device = NewModelLoader._checkpoint_uses_local_block_device(
            checkpoint_files
        )
        if uses_local_block_device is not True:
            return True
        if os.path.exists("/sys/module/nvidia_fs"):
            return True
        try:
            # ``/proc/kallsyms`` can contain hundreds of thousands of entries.
            # A decoded Python line loop costs seconds on large hosts, so keep
            # the capability check in one binary read/search.
            with open("/proc/kallsyms", "rb") as kallsyms:
                return b" p2pdma_pgmap_ops" in kallsyms.read()
        except OSError:
            # Fail open when the container hides kernel capability information;
            # the shard registration probe below remains the source of truth.
            return True

    @staticmethod
    def _cuda_cufile_available(
        checkpoint_files: Optional[List[str]] = None,
    ) -> bool:
        """Use fastsafetensors' native probe as the GDS source of truth."""
        if (
            checkpoint_files
            and not NewModelLoader._cuda_local_gds_prerequisites_available(
                checkpoint_files
            )
        ):
            logger.info(
                "Local checkpoint filesystem has neither nvidia-fs nor "
                "PCI P2PDMA support; skipping libcufile initialization"
            )
            return False
        try:
            from fastsafetensors import cpp as fastsafetensors_cpp
        except (ImportError, OSError, RuntimeError):
            return False
        try:
            # The native backend discovers libcufile lazily.  Querying before
            # this initialization always reports False even when CUDA ships a
            # usable libcufile.so, which silently forced every CUDA load onto
            # the much slower nogds path.
            fastsafetensors_cpp.load_nvidia_functions()
            if not fastsafetensors_cpp.is_cufile_found():
                return False
            if checkpoint_files:
                if fastsafetensors_cpp.init_gds() != 0:
                    return False
                # Probe the smallest shard first.  Tail shards are commonly
                # smaller and are also the most likely to expose filesystem or
                # direct-I/O registration constraints.  A failing checkpoint
                # then falls back before paying registration cost for every
                # multi-GiB shard.
                registration_order = sorted(
                    checkpoint_files, key=lambda path: os.path.getsize(path)
                )
                for path in registration_order:
                    try:
                        handle = fastsafetensors_cpp.gds_file_handle(
                            os.fspath(path), False, True
                        )
                        del handle
                    except (OSError, RuntimeError) as exc:
                        logger.warning(
                            "libcufile cannot register checkpoint shard %s; "
                            "NewLoader fastsafetensors is falling back to the "
                            "CUDA nogds path: %s",
                            path,
                            exc,
                        )
                        return False
            return True
        except (AttributeError, OSError, RuntimeError):
            return False

    @staticmethod
    def _accelerator_backend_name() -> str:
        """Identify CUDA-compatible runtimes without importing the device package."""
        if torch.version.hip is not None:
            return "ROCm"
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return "Ppu"
        return "Cuda"

    @staticmethod
    def _fastsafetensors_use_nogds(
        device: torch.device,
        checkpoint_files: Optional[List[str]] = None,
    ) -> bool:
        if device.type == "cpu":
            return True
        if os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1":
            return True
        if (
            device.type == "cuda"
            and torch.version.hip is None
            and not NewModelLoader._cuda_cufile_available(checkpoint_files)
        ):
            logger.warning(
                "libcufile.so is unavailable or cannot register the selected "
                "checkpoint shards; NewLoader fastsafetensors is falling back "
                "to the CUDA nogds path"
            )
            return True
        return False

    @staticmethod
    def _fastsafetensors_backend_options(
        device: torch.device,
        checkpoint_files: Optional[List[str]] = None,
    ) -> Tuple[bool, bool]:
        """Return ``(nogds, use_shm)`` for the active accelerator backend.

        ``SafeTensorsFileLoader`` checks ``use_shm`` before ``nogds`` and GDS.
        Therefore CUDA GDS must set both flags to false; enabling ``use_shm``
        would silently select the shared-memory reader instead of GDS.
        """
        force_nogds = os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1"
        if force_nogds:
            return True, False
        if device.type == "cuda" and NewModelLoader._accelerator_backend_name() in (
            "Ppu",
            "ROCm",
        ):
            # These readers transfer from shared host memory through their
            # own native extension. Never probe an NVIDIA driver on PPU.
            return False, True
        force_shm = (
            device.type == "cuda"
            and torch.version.hip is None
            and os.environ.get("FASTSAFETENSORS_USE_SHM", "0") == "1"
        )
        if force_shm:
            return False, True
        use_nogds = NewModelLoader._fastsafetensors_use_nogds(device, checkpoint_files)
        return use_nogds, False

    @staticmethod
    def _model_storage_bytes(model: nn.Module) -> int:
        seen = set()
        total = 0
        tensors = list(
            model.named_parameters(recurse=True, remove_duplicate=False)
        ) + list(model.named_buffers(recurse=True, remove_duplicate=False))
        for _, tensor in tensors:
            if tensor is None or id(tensor) in seen:
                continue
            if tensor.is_meta:
                raise RuntimeError(
                    "model contains meta tensors that cannot be sized for automatic "
                    "fastsafetensors selection"
                )
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def fst_staging_required_bytes(model_bytes, max_shard_bytes, bbuf_bytes):
        """The destination parameters and up to three staging buffers overlap."""
        return model_bytes + 3 * max(max_shard_bytes, bbuf_bytes)

    def _fixed_host_key_filter(self, model):
        prefixes = tuple(
            name + "." if name else ""
            for name, module in model.named_modules()
            if isinstance(module, FixedHostCheckpointStorage)
        )
        if prefixes:
            return lambda key: self._mapped_checkpoint_name(model, key).startswith(prefixes)
        return None

    def _fastsafetensors_prerequisite_reason(
        self,
        model: nn.Module,
        checkpoint_files: List[str],
        *,
        automatic: bool,
        expert_filter: Optional[_ExpertRangeFilter] = None,
        overlay_only: bool = False,
    ) -> Optional[str]:
        if not checkpoint_files and not overlay_only:
            return "checkpoint shard selection is empty"
        non_safetensors = [
            os.fspath(path)
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if non_safetensors:
            return f"checkpoint contains non-safetensors files {non_safetensors}"
        rank_local_checkpoint = weight_mapper.is_rank_local_checkpoint(checkpoint_files)
        if rank_local_checkpoint and not bool(
            getattr(model, "supports_rank_local_checkpoint", False)
        ):
            return (
                f"{type(model).__name__} does not support rank-local consolidated "
                "checkpoints"
            )
        if self.load_config.force_cpu_load_weights:
            if automatic:
                # AUTO chooses direct FST for accelerator staging only.
                return (
                    "forced CPU staging uses scratch unless FST is explicitly requested"
                )
            from rtp_llm.models_py.fst_range_reader import native_cpu_range_api

            try:
                native_cpu_range_api()
                duplicates, _ = self._safetensor_metadata_summary(
                    self._resolved_model_path(),
                    checkpoint_files,
                    self._safetensor_catalog(),
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                return f"cannot stage checkpoint with the native FST CPU reader: {exc}"
            if duplicates:
                return f"checkpoint tensor names are duplicated: {duplicates[:10]}"
            return None
        staged_modules = self._staged_device_modules(model)
        try:
            self._validate_staged_device_modules(
                model, staged_modules, require_load_complete=True
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return f"staged device postprocessing is incompatible with FST: {exc}"

        target_device = torch.device(self.load_config.device)
        if target_device.type not in ("cpu", "cuda"):
            return f"runtime device {target_device} is not supported"
        if automatic and target_device.type == "cpu":
            return (
                "the runtime device is CPU, where direct accelerator loading "
                "has no benefit"
            )
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                return f"runtime device {target_device} is unavailable"
            # CUDA, ROCm and PPU expose their accelerator through torch.cuda.
            # Let the installed reader, checkpoint, participating ranks and
            # available memory decide eligibility, not the host ISA or card label.
            backend = self._accelerator_backend_name()
            if backend in ("Ppu", "ROCm"):
                from rtp_llm.models_py.fst_host_backend import fst_host_backend_reason

                use_nogds, use_shm = self._fastsafetensors_backend_options(
                    target_device, checkpoint_files
                )
                reason = fst_host_backend_reason(
                    backend, use_shm=use_shm, use_nogds=use_nogds
                )
                if reason is not None:
                    return reason
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            device_count = torch.cuda.device_count()
            if not 0 <= device_index < device_count:
                return (
                    f"runtime device index {device_index} is outside the available "
                    f"CUDA device range [0, {device_count})"
                )

        isolated_load = bool(getattr(model, "fastsafetensors_load_isolated", False))
        if isolated_load:
            if (
                self.load_config.tp_size != 1
                or self.load_config.tp_rank != 0
                or self.load_config.ep_size != 1
                or self.load_config.ep_rank != 0
                or expert_filter is not None
            ):
                return (
                    "isolated fastsafetensors loading requires TP=1, EP=1, "
                    "rank zero, and no expert checkpoint routing"
                )
        else:
            distributed_reason = self._fastsafetensors_distributed_reason(expert_filter)
            if distributed_reason is not None:
                return distributed_reason

        try:
            ParallelLoader, _ = self._fastsafetensors_api()
        except RuntimeError as exc:
            return str(exc)
        try:
            duplicates, stacked_experts = self._safetensor_metadata_summary(
                self._resolved_model_path(),
                checkpoint_files,
                self._safetensor_catalog(),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return f"failed to inspect safetensors metadata: {exc}"
        if duplicates:
            sample = duplicates[:10]
            suffix = (
                f" (+{len(duplicates) - len(sample)} more)"
                if len(duplicates) > 10
                else ""
            )
            return f"checkpoint tensor names are duplicated: {sample}{suffix}"
        if expert_filter is not None and stacked_experts:
            expected_world_size = max(
                self.load_config.tp_size, self.load_config.ep_size
            )
            if self.load_config.ep_size != expected_world_size:
                return (
                    "stacked all-expert FST loading requires ep_size to equal "
                    "the WORLD size so get_sharded(dim=0) maps one expert range "
                    "to each EP rank"
                )
            if not hasattr(ParallelLoader, "_consume_single_batch"):
                return (
                    "the installed fastsafetensors ParallelLoader does not expose "
                    "the batch hook required for stacked all-expert sharding"
                )
        if automatic:
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            try:
                use_nogds, use_shm = self._fastsafetensors_backend_options(
                    target_device, checkpoint_files
                )
                free_bytes, _ = torch.cuda.mem_get_info(device_index)
                model_bytes = self._model_storage_bytes(model)
                staging_files = checkpoint_files
                fixed_host = self._fixed_host_key_filter(model)
                if fixed_host is not None:
                    from safetensors import safe_open

                    # Pure host shards are read by range and never staged on
                    # the GPU. Keep mixed files in the estimate: their non-host
                    # ranges still need headroom even on the independent reader.
                    staging_files = []
                    for path in checkpoint_files:
                        with safe_open(path, framework="pt", device="cpu") as shard:
                            if any(not fixed_host(key) for key in shard.keys()):
                                staging_files.append(path)
                max_shard_bytes = max(
                    (os.path.getsize(path) for path in staging_files), default=0
                )
                bbuf_bytes = 0
                if staging_files:
                    bbuf_bytes = 1024 * self._fastsafetensors_bbuf_size_kb(
                        staging_files, use_host_staging=use_nogds or use_shm
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                return f"cannot determine fastsafetensors memory safety: {exc}"
            required_bytes = self.fst_staging_required_bytes(
                model_bytes, max_shard_bytes, bbuf_bytes
            )
            logger.info(
                "NewLoader fastsafetensors memory check: free=%.2fGiB, "
                "model=%.2fGiB, max_shard=%.2fGiB, bbuf=%.2fGiB, "
                "required=%.2fGiB",
                free_bytes / (1024**3),
                model_bytes / (1024**3),
                max_shard_bytes / (1024**3),
                bbuf_bytes / (1024**3),
                required_bytes / (1024**3),
            )
            if free_bytes <= required_bytes:
                return (
                    f"insufficient accelerator memory: free={free_bytes} bytes, "
                    f"required>{required_bytes} bytes"
                )
        return None

    def _synchronize_load_failure(
        self,
        local_reason: Optional[str],
        *,
        format_route=None,
    ) -> Optional[str]:
        """Agree amongst the consumers actually taking part in this load."""
        if not torch.distributed.is_initialized():
            return local_reason
        pc = self.load_config.parallelism_config
        # PP draft models may be loaded by the tail stage only. Never enter a
        # WORLD collective from that stage; the other stages will not join.
        is_draft = (
            self.model_config.get("is_mtp", False)
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "is_mtp", False)
        )
        group = (
            self._stage_group() if is_draft and getattr(pc, "pp_size", 1) > 1 else None
        )
        participants = torch.distributed.get_world_size(group)
        if participants <= 1:
            return local_reason
        reasons: List[Optional[str]] = [None] * participants
        try:
            if format_route is None:
                torch.distributed.all_gather_object(reasons, local_reason, group=group)
            else:
                torch.distributed.all_gather_object(reasons, format_route, group=group)
                # Compared below so that a disagreement is not reported as
                # a broken process group.
        except (RuntimeError, TypeError, ValueError) as exc:
            # No shared decision exists. Only a completed vote may select scratch.
            raise RuntimeError(
                "cannot synchronize checkpoint load status across ranks"
            ) from exc
        if format_route is not None:
            if len(set(reasons)) > 1:
                raise RuntimeError(f"inconsistent checkpoint formats across ranks: {reasons}")
            return None
        failures = [
            f"rank {rank}: {reason}"
            for rank, reason in enumerate(reasons)
            if reason is not None
        ]
        if not failures:
            return None
        sample = failures[:4]
        suffix = (
            f" (+{len(failures) - len(sample)} more)"
            if len(failures) > len(sample)
            else ""
        )
        return f"distributed prerequisite failure: {'; '.join(sample)}{suffix}"

    @contextmanager
    def _checkpoint_preflight(self, model, requested, *, isolated=False):
        # Scratch and isolated owners do not participate in the FST vote.
        if (
            requested == NewLoaderLoadMethod.SCRATCH
            or isolated
            or getattr(model, "fastsafetensors_load_isolated", False)
            or not torch.distributed.is_initialized()
        ):
            yield
            return
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        reason = self._synchronize_load_failure(
            f"{type(error).__name__}: {error}" if error is not None else None
        )
        if reason is not None:
            raise RuntimeError(f"Checkpoint preflight failed: {reason}") from error

    def _select_load_method(
        self,
        requested: NewLoaderLoadMethod,
        model: nn.Module,
        checkpoint_files: List[str],
        expert_filter: Optional[_ExpertRangeFilter] = None,
        *,
        overlay_only: bool = False,
    ) -> NewLoaderLoadMethod:
        if requested == NewLoaderLoadMethod.SCRATCH:
            return requested
        automatic = requested == NewLoaderLoadMethod.AUTO
        reason = self._fastsafetensors_prerequisite_reason(
            model,
            checkpoint_files,
            automatic=automatic,
            expert_filter=expert_filter,
            overlay_only=overlay_only,
        )
        # Most distributed loads must make one consistent backend decision,
        # including replicated TP=1/DP ranks.  A small number of auxiliary
        # modules are intentionally instantiated on only one rank, however;
        # those modules use fastsafetensors' SingleGroup and must not enter a
        # WORLD collective whose other ranks will never call this loader.
        isolated_load = bool(getattr(model, "fastsafetensors_load_isolated", False))
        if not isolated_load:
            reason = self._synchronize_load_failure(reason)
        if reason is None:
            selected = NewLoaderLoadMethod.FASTSAFETENSORS
            logger.info(
                "NewLoader selected fastsafetensors (%s)",
                "automatic" if automatic else "explicit",
            )
            return selected
        if automatic:
            logger.info(
                "NewLoader automatic fastsafetensors selection fell back to "
                "scratch: %s",
                reason,
            )
            return NewLoaderLoadMethod.SCRATCH
        raise RuntimeError(f"NewLoader fastsafetensors prerequisite failed: {reason}")

    def _arm_streamed_scratch(self, model, method):
        # The reader is still scratch. Only V4 owners with checkpoint-backed
        # device parameters opt in; no change to AUTO/FST platform eligibility.
        if not (
            method == NewLoaderLoadMethod.SCRATCH
            and getattr(model, "arm_cuda_streamed_scratch", False)
            and not self.load_config.force_cpu_load_weights
            and torch.device(self.load_config.device).type == "cuda"
            and self._accelerator_backend_name() == "Cuda"
            and platform.machine().lower() in ("aarch64", "arm64")
        ):
            return False
        return self._full_world_scratch_owner(model)

    def _full_world_scratch_owner(self, model):
        # The allocation/dispatch acknowledgements use WORLD. Unlike the
        # two-rank P/D services, auxiliary owners and PP drafts can be built
        # by a single rank or stage. Do not introduce a new collective there.
        pc = self.load_config.parallelism_config
        if getattr(model, "fastsafetensors_load_isolated", False) or getattr(
            pc, "pp_size", 1
        ) != 1:
            return False
        if torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            if world > 1 and getattr(pc, "world_size", None) != world:
                return False
        return True

    def _ppu_streamed_scratch(self, model, method):
        # Qwen-7B P/D otherwise materializes a complete CPU destination on
        # every rank, then transfers it in a second, whole-model _apply.
        # Its dense checkpoint leaves can copy directly from each CPU shard
        # into final parameter storage. Do not enable for other formats or
        # for operators that own a host/online-quantization staging step.
        return (
            getattr(model, "ppu_dense_streamed_scratch", False)
            and method == NewLoaderLoadMethod.SCRATCH
            and not self.load_config.force_cpu_load_weights
            and (
                self.load_config.quant_config is None
                or getattr(self.load_config.quant_config, "quant_type", None) == "none"
            )
            and not self.load_config.custom_weight_mappings
            and not self.load_config.custom_projection_layers
            and self.load_config.compute_dtype in (torch.float16, torch.bfloat16)
            and torch.device(self.load_config.device).type == "cuda"
            and self._accelerator_backend_name() == "Ppu"
            and self._full_world_scratch_owner(model)
        )

    @contextmanager
    def _scratch_stage(self, phase):
        # All scratch reads below are rank-local. Agree at the boundary before
        # any rank enters the runtime post-load hooks with a partial model.
        error = None
        try:
            yield
        except Exception as exc:
            error = exc
        reason = None if error is None else f"{type(error).__name__}: {error}"
        failure = self._synchronize_load_failure(reason)
        if failure is not None:
            raise RuntimeError(f"streamed scratch {phase} failed: {failure}") from error

    def _prepare_streamed_scratch(self, model):
        device = torch.device(self.load_config.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._staged_device_modules(model):
            raise ValueError("streamed scratch requires final checkpoint destinations")
        loans = borrowed_parameter_ids(model)
        seen = set(loans)
        size = 0
        for tensor in chain(model.parameters(), model.buffers()):
            if id(tensor) not in seen and tensor.device != device:
                size += tensor.numel() * tensor.element_size()
                seen.add(id(tensor))
        reserve_mb = self.load_config.reserve_runtime_mem_mb
        if type(reserve_mb) is not int or reserve_mb < 0:
            raise ValueError(f"invalid runtime memory reservation: {reserve_mb!r} MiB")
        reserve = reserve_mb * 1024**2
        free, _ = torch.cuda.mem_get_info(device)
        # Caching allocator blocks belong to this process, not other jobs.
        usable = (
            free
            + torch.cuda.memory_reserved(device)
            - torch.cuda.memory_allocated(device)
        )
        if usable < size + reserve:
            raise RuntimeError(
                f"insufficient device budget on {device}: available={usable}, "
                f"checkpoint destination={size}, runtime reserve={reserve}"
            )
        self._prepare_model_for_fastsafetensors(model, str(device), keep_on_device=True)
        logger.info(
            "NewLoader scratch destination: %s (%d bytes); rank-local CPU reader, "
            "no full host destination; runtime reserve=%d",
            device,
            size,
            reserve,
        )

    def _normal_fst_read_spec_resolver(self, resolver):
        """Keep proven full-read FP4 behavior unless explicitly opted in."""
        if resolver is None or self.load_config.experimental_fst_segmented_reads:
            return resolver
        logged = False

        def resolve(name, shape):
            nonlocal logged
            spec = resolver(name, shape)
            if spec.segments is None or not spec.load:
                return spec
            if spec.partition != "nvfp4_fused":
                raise RuntimeError(
                    "Normal FST segmented reads are experimental and disabled; "
                    "cannot silently materialize an arbitrary segmented layout"
                )
            if not logged:
                logger.info(
                    "NVFP4 normal FST retains the existing full-tensor consumer path; "
                    "segmented CPU range bypass is disabled pending source-buffer "
                    "early-release and GPU peak validation"
                )
                logged = True
            # NVFP4 load_weights explicitly accepts a full fused checkpoint
            # and performs each projection's correct TP/GQA split itself.
            return replace(spec, segments=None, partition="replicated")

        return resolve

    def _fastsafetensors_weights_iterator(
        self,
        checkpoint_files: List[str],
        device: str,
        name_filter: Optional[Callable[[str], bool]],
        safetensor_slice_expander: Optional[_ExpertRangeFilter],
        read_spec_resolver: Optional[Callable[[str, Tuple[int, ...]], ReadSpec]] = None,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream safetensors directly to the model's runtime device.

        The yielded tensor is consumed synchronously by ``RtpModule.load_weights``
        before the iterator advances. Compatible FST versions execute ReadSpecs
        in their consumer so TP shards and EP expert ranges are materialized
        before recursive model dispatch.
        """
        read_spec_resolver = self._normal_fst_read_spec_resolver(read_spec_resolver)
        ParallelLoader, SingleGroup = self._fastsafetensors_api()
        process_group = self._fastsafetensors_process_group(
            SingleGroup, checkpoint_files, safetensor_slice_expander
        )

        target_device = torch.device(device)
        loader_device = str(target_device)
        if target_device.type == "cuda" and target_device.index is None:
            loader_device = f"cuda:{torch.cuda.current_device()}"
        use_nogds, use_shm = NewModelLoader._fastsafetensors_backend_options(
            target_device, checkpoint_files
        )
        from rtp_llm.models_py.v41_fst_file_pool import v41_file_staging_loader

        ParallelLoader = v41_file_staging_loader(
            ParallelLoader,
            self._model_type(),
            torch.device(loader_device),
            cuda_backend=self._accelerator_backend_name(),
            use_shm=use_shm,
        )
        bbuf_size_kb = NewModelLoader._fastsafetensors_bbuf_size_kb(
            checkpoint_files, use_host_staging=use_nogds or use_shm
        )
        logger.info(
            "NewLoader fastsafetensors backend: nogds=%s, use_shm=%s, "
            "bbuf_size=%.2fMiB",
            use_nogds,
            use_shm,
            bbuf_size_kb / 1024,
        )
        loader_kwargs = dict(
            pg=process_group,
            hf_weights_files=sorted(os.fspath(path) for path in checkpoint_files),
            use_tqdm_on_load=True,
            device=loader_device,
            bbuf_size_kb=bbuf_size_kb,
            use_shm=use_shm,
            nogds=use_nogds,
        )
        loader_handles_read_specs = False
        from rtp_llm.models_py.fst_range_reader import (
            build_segmented_fst_reader,
            validate_fst_full_tensor_fallback,
        )

        if read_spec_resolver is not None and hasattr(
            ParallelLoader, "_consume_single_batch"
        ):
            try:
                from fastsafetensors.parallel_loader import TimingContext

                from rtp_llm.models_py.fastsafetensors_read_spec import (
                    build_read_spec_parallel_loader,
                )

                read_spec_loader = build_read_spec_parallel_loader(
                    ParallelLoader, TimingContext
                )
            except (ImportError, TypeError, ValueError) as exc:
                if safetensor_slice_expander is not None:
                    raise RuntimeError(
                        "fastsafetensors does not expose compatible internals "
                        "required for EP stacked expert sharding"
                    ) from exc
                logger.warning(
                    "fastsafetensors does not expose compatible ReadSpec "
                    "internals; applying slices after tensor materialization"
                )
                validate_fst_full_tensor_fallback(
                    checkpoint_files, read_spec_resolver, name_filter
                )
                loader = ParallelLoader(**loader_kwargs)
            else:
                # Construction errors propagate: never retry allocation/I/O
                # failures by silently switching to a full-tensor loader.
                loader = read_spec_loader(
                    read_spec_resolver=read_spec_resolver,
                    safetensor_slice_expander=safetensor_slice_expander,
                    segmented_tensor_reader=build_segmented_fst_reader(
                        checkpoint_files
                    ),
                    name_filter=name_filter,
                    **loader_kwargs,
                )
                loader_handles_read_specs = True
        else:
            validate_fst_full_tensor_fallback(
                checkpoint_files, read_spec_resolver, name_filter
            )
            loader = ParallelLoader(**loader_kwargs)
        seen = set()
        try:
            for name, checkpoint_value in loader.iterate_weights():
                if isinstance(checkpoint_value, LoadedCheckpointTensor):
                    tensor_value = checkpoint_value.tensor
                else:
                    tensor_value = checkpoint_value
                if read_spec_resolver is not None and not loader_handles_read_specs:
                    read_spec = read_spec_resolver(name, tuple(tensor_value.shape))
                    if not read_spec.load:
                        continue
                    if read_spec.segments is not None:
                        raise RuntimeError(
                            "Segmented FST loading requires ReadSpec consumer integration; full tensor fallback is forbidden"
                        )
                    checkpoint_value = read_spec.materialize(tensor_value)
                if name_filter is not None and not name_filter(name):
                    continue
                if (
                    safetensor_slice_expander is not None
                    and safetensor_slice_expander.handles_safetensor(name)
                ):
                    raise RuntimeError(
                        "EP fastsafetensors backend failed to shard and expand "
                        f"stacked all-expert tensor {name!r}"
                    )
                if name in seen:
                    raise RuntimeError(
                        f"Checkpoint tensor {name!r} appears more than once during "
                        "fastsafetensors loading"
                    )
                seen.add(name)
                yield name, checkpoint_value
        finally:
            # Ordered FST owns a native reader per producer. Closing only the
            # legacy first-reader alias leaks the rest on this ABI.
            close = getattr(loader, "close", None)
            if not callable(close):
                close = getattr(getattr(loader, "loader", None), "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_runtime_backends(model: nn.Module, device: str) -> None:
        target_device = torch.device(device)
        for module in model.modules():
            if isinstance(module, RtpModule):
                module.validate_runtime_device(target_device)

    @staticmethod
    def _notify_checkpoint_source(
        model: nn.Module, method: NewLoaderLoadMethod
    ) -> None:
        """Publish the resolved reader for auxiliary model-owned checkpoint I/O.

        This is a selection notification, not proof of a successful model load.
        Host-owned leaves (for example V4.1 Engram) must use the actual reader,
        not infer it from the requested AUTO policy or fall back independently.
        """
        notify = getattr(model, "set_checkpoint_source", None)
        if notify is not None:
            notify(method.value)

    @staticmethod
    def _migrate_staged_modules(model: nn.Module, device: str, *, defer_executor=False) -> None:
        """Move and compress online-quantized leaves one at a time.

        Migrating the entire model first would retain every BF16 staging weight on
        the accelerator until post-load hooks run. Processing each marked leaf
        immediately bounds the extra accelerator memory to one staging layer.
        """
        target_device = torch.device(device)
        staged_modules = NewModelLoader._staged_device_modules(model)
        NewModelLoader._validate_staged_device_modules(model, staged_modules)
        for module in staged_modules:
            module.to(target_device)
            stage = getattr(module, "process_weights_during_staged_load", None)
            if defer_executor and callable(stage):
                stage()
            else:
                module.process_weights_after_loading()

    def _finalize_host_staged_modules(self, model, *, defer_executor=False):
        if self.load_config.force_cpu_load_weights:
            # Honor a requested host quantizer, not merely host checkpoint I/O.
            for module in self._staged_device_modules(model):
                prepare = getattr(
                    getattr(module, "quant_method", None), "prepare_weights_on_cpu", None
                )
                if callable(prepare):
                    prepare(module)
        self._migrate_staged_modules(model, self.load_config.device, defer_executor=defer_executor)

    def _prepare_checkpoint(self, requested_method, startup_lora):
        meta_checkpoint = None
        if self._model_type() == "llama":
            from rtp_llm.models_py.meta_checkpoint import MetaLlamaCheckpoint

            meta_checkpoint = MetaLlamaCheckpoint.open(self._resolved_model_path())
        checkpoint_files = (
            meta_checkpoint.files
            if meta_checkpoint is not None
            else self._checkpoint_files()
        )
        expert_filter = self._expert_filter()
        self._validate_ep_checkpoint_format(checkpoint_files, expert_filter)
        parallel = self.load_config.parallelism_config
        ft_checkpoint = meta_checkpoint or FTCheckpoint.open(
            self._resolved_model_path(),
            CheckpointTopology(
                tp_size=self.load_config.tp_size,
                tp_rank=self.load_config.tp_rank,
                ep_size=self.load_config.ep_size,
                ep_rank=self.load_config.ep_rank,
                dp_size=getattr(parallel, "dp_size", 1),
                dp_rank=getattr(parallel, "dp_rank", 0),
            ),
            num_layers=(
                self.model_config.get(
                    "num_layers", self.model_config.get("num_hidden_layers")
                )
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "num_layers", None)
            ),
        )
        runtime_quant = self.load_config.quant_config
        if (
            getattr(runtime_quant, "quant_type", None) == "W8A8_INT8_PER_CHANNEL_COMPRESSED"
            and getattr(runtime_quant, "source_config", None) is not None
            and not runtime_quant.source_config.is_quanted()
        ):
            from rtp_llm.models_py.quant_methods.w8a8_load_source import W8A8LoadSource

            if ft_checkpoint is not None or weight_mapper.is_rank_local_checkpoint(checkpoint_files):
                raise ValueError("load-time W8A8 requires a global floating HF checkpoint")
            if requested_method not in (NewLoaderLoadMethod.AUTO, NewLoaderLoadMethod.SCRATCH):
                raise ValueError("load-time W8A8 supports LOAD_METHOD=scratch or auto only")
            lora_infos = (
                self.model_config.get("lora_infos") if isinstance(self.model_config, dict)
                else getattr(self.model_config, "lora_infos", None)
            )
            if startup_lora is not None or lora_infos:
                raise ValueError("load-time W8A8 does not support LoRA")
            self._w8a8_source = W8A8LoadSource.from_files(
                checkpoint_files, runtime_quant, self._checkpoint_catalog()
            )
            self._w8a8_source.validate_overlay(self._finetune_overlay(checkpoint_files))
            if not getattr(get_model_class(self._model_type()), "supports_load_time_w8a8", False):
                raise ValueError(f"{self._model_type()} does not support load-time W8A8")
        return checkpoint_files, expert_filter, meta_checkpoint, ft_checkpoint

    def _record_resident_owner(self, model, resident, method, startup_lora, *, format_route="hf"):
        from rtp_llm.models_py import resident_checkpoint
        from rtp_llm.resident_weights import GLOBAL_SCOPE

        describe = getattr(model, "resident_constructor_state", None)
        if describe is None or (startup_lora is not None and not getattr(model, "resident_after_post_load", False)):
            raise RuntimeError(f"No resident checkpoint schema for {type(model).__name__} with startup_lora={startup_lora is not None}")
        constructor = json.dumps(describe(), sort_keys=True)
        header = getattr(model, "_resident_ft_header_facts", None)
        resident_checkpoint.record(model, resident, method)
        if format_route == "ft":
            resident.init_scalar(GLOBAL_SCOPE, "native.ft_header", lambda: json.dumps(header))
        resident.init_scalar(GLOBAL_SCOPE, "native.format", lambda: format_route)
        resident.init_scalar(GLOBAL_SCOPE, "native.constructor", lambda: constructor)

    @torch.inference_mode()
    def load(self, *, startup_lora: Optional[Tuple[str, str]] = None, prepare_loans=None, resident=None, restored=None) -> nn.Module:
        # Only a complete load publishes a backend, including on retry.
        self._selected_load_method = None
        self._eplb_ft_checkpoint = None
        self._w8a8_source = None
        if self.load_config.resident_construction is not None:
            self.load_config = replace(self.load_config, resident_construction=None)
        requested_method = self.load_config.load_method
        parallel = self.load_config.parallelism_config
        # Participation is an owner-class contract, independent of whether a
        # rank can open its local mount or construct its parameter tree.
        self._ckpt_files = None
        self._checkpoint_catalog_cache = _CHECKPOINT_CATALOG_UNSET
        distributed_load = requested_method != NewLoaderLoadMethod.SCRATCH and torch.distributed.is_initialized()
        owner = lookup_error = None
        isolated = False
        if distributed_load:
            model_type = self._model_type()
            isolated = uses_isolated_checkpoint_load(model_type)
            try:
                owner = get_model_class(model_type)
            except Exception as exc:
                lookup_error = exc
            else:
                isolated = isolated or getattr(owner, "fastsafetensors_load_isolated", False)
        with self._checkpoint_preflight(owner, requested_method, isolated=isolated):
            if lookup_error is not None:
                raise lookup_error
            if prepare_loans is not None:
                aliases, prepared = prepare_loans()
                self.parameter_aliases = dict(aliases)
                self.load_config = replace(self.load_config, borrowed_parameter_paths=tuple(aliases),
                                           prepared_runtime_loan=prepared)
            if restored is None:
                checkpoint_files, expert_filter, meta_checkpoint, ft_checkpoint = self._prepare_checkpoint(requested_method, startup_lora)
        if resident is not None and restored is not None:
            from rtp_llm.models_py import resident_checkpoint as rc
            from rtp_llm.resident_weights import GLOBAL_SCOPE

            with self._checkpoint_preflight(owner, requested_method, isolated=isolated):
                self.load_config = replace(self.load_config, resident_construction=json.loads(
                    resident.init_scalar(GLOBAL_SCOPE, "native.constructor", rc._replay_only)
                ))
                method = NewLoaderLoadMethod(resident.init_scalar(GLOBAL_SCOPE, "native.backend", rc._replay_only))
                if requested_method not in (NewLoaderLoadMethod.AUTO, method):
                    raise ValueError("Resident weight record belongs to a different checkpoint backend")
                route = resident.init_scalar(GLOBAL_SCOPE, "native.format", rc._replay_only)
                if route not in ("hf", "ft", "meta"):
                    raise ValueError(f"Unknown resident checkpoint format {route!r}")
                load = self.load_config
                adapt = getattr(load.quant_config, "for_ft_snapshot", None)
                if route == "ft" and callable(adapt):
                    load = replace(load, quant_config=adapt())
                header = None
                if route == "ft":
                    facts = json.loads(resident.init_scalar(GLOBAL_SCOPE, "native.ft_header", rc._replay_only))
                    header = rc.ResidentFTHeader(self._resolved_model_path(), facts)
                with torch.device("cpu"):
                    model = self._create_model(ft_load_config=load, ft_checkpoint=header)
                self._validate_runtime_backends(model, self.load_config.device)
                self._notify_checkpoint_source(model, method)
                bind_borrowed_parameters(model, self.parameter_aliases, self.load_config.device)
                rc.restore(model, resident, restored, self.load_config.device,
                           construction=self.load_config.resident_construction)
                model.to(self.load_config.device)
                self._validate_loaded_weights(model)
                if not getattr(model, "resident_after_post_load", False):
                    self._run_post_load_hooks(model)
                validate_borrowed_parameters(model)
                model.eval()
            self._selected_load_method = method
            return model
        if distributed_load and not isolated:
            route = ("meta" if meta_checkpoint is not None else "ft" if ft_checkpoint is not None
                     else "w8a8" if self._w8a8_source is not None else "hf")
            self._synchronize_load_failure(None, format_route=(
                route, requested_method.value, self.load_config.force_cpu_load_weights
            ))
        with self._checkpoint_preflight(owner, requested_method), torch.device("cpu"):
            if resident is not None and isinstance(ft_checkpoint, FTCheckpoint):
                from rtp_llm.models_py.resident_checkpoint import ResidentFTHeader

                header = ResidentFTHeader.recording(ft_checkpoint)
                model = self._create_checkpoint_model(ft_checkpoint, constructor_header=header)
                model._resident_ft_header_facts = header.facts
            else:
                model = self._create_checkpoint_model(ft_checkpoint)
        if expert_filter is not None:
            expert_filter.checkpoint_to_runtime_layer = getattr(model, "checkpoint_to_runtime_expert_layer", None)
        if ft_checkpoint is not None:
            with self._checkpoint_preflight(model, requested_method):
                # Format adapters own their import and return before the HF/FST
                # borrowed-parameter filters. Do not let them mutate target loans.
                bind_borrowed_parameters(model, self.parameter_aliases, self.load_config.device)
                validate_format_parameter_ownership(model, ft=meta_checkpoint is None)
                overlay = self._finetune_overlay(checkpoint_files)
                if overlay is not None:
                    ft_checkpoint.apply_overlay(overlay)
                if getattr(parallel, "pp_size", 1) != 1:
                    if meta_checkpoint is not None or not callable(
                        getattr(model, "layer_ids", None)
                    ):
                        raise ValueError(
                            "This format adapter does not declare PP stage ownership"
                        )
                    # Main's FT writer exports all layers even when deployment
                    # later uses PP. Use the constructed model's range, including
                    # the full-range exception for speculative drafts.
                    ft_checkpoint.select_stage(model.layer_ids(), model.config.num_layers)
                if requested_method == NewLoaderLoadMethod.FASTSAFETENSORS:
                    if meta_checkpoint is not None and any(
                        not path.endswith(".safetensors") for path in checkpoint_files
                    ):
                        raise ValueError(
                            "FST requires safetensors; Meta .pth archives require scratch"
                        )
                    ft_checkpoint.read_backend = "fastsafetensors"
                importer = getattr(
                    model,
                    "load_meta_checkpoint" if meta_checkpoint else "load_ft_checkpoint",
                    None,
                )
                if importer is None:
                    raise ValueError(
                        f"{type(model).__name__} has no checkpoint format adapter"
                    )
                if self.load_config.custom_weight_mappings and (
                    meta_checkpoint is not None or not getattr(model, "supports_ft_handler_heads", False)
                ):
                    raise ValueError(
                        "Custom HF weight mappings cannot be applied to FT runtime tensors"
                    )
                self._validate_runtime_backends(model, self.load_config.device)
                self._notify_checkpoint_source(
                    model,
                    (
                        NewLoaderLoadMethod.FASTSAFETENSORS
                        if ft_checkpoint.read_backend == "fastsafetensors"
                        else NewLoaderLoadMethod.SCRATCH
                    ),
                )
                importer(ft_checkpoint)
                finish_meta_lora = None
                if meta_checkpoint is not None and startup_lora is not None:
                    from rtp_llm.models_py.lora_checkpoint import prepare_startup_lora_merge

                    finish_meta_lora = prepare_startup_lora_merge(model, self, *startup_lora)
                ft_rebalance = False
                if isinstance(ft_checkpoint, FTCheckpoint):
                    routed = [
                        leaf for leaf in model.modules()
                        if callable(getattr(leaf, "bind_eplb_stats", None))
                    ]
                    if any(getattr(leaf, "_dynamic_eplb_enabled", False) for leaf in routed):
                        from rtp_llm.models_py.eplb_controller import (
                            stage_supports_weight_install,
                        )

                        ft_rebalance = stage_supports_weight_install(model, routed)
                    if ft_rebalance:
                        for leaf in routed:
                            if getattr(leaf, "_ft_expert_binding_layout", None) is None:
                                cfg = getattr(self.model_config, "eplb_config", None)
                                mode = getattr(cfg, "eplb_mode", None)
                                if getattr(mode, "name", mode) == "STATS":
                                    # This FT importer can serve and count tokens, but
                                    # cannot reconstruct another EP rank's source rows.
                                    model._ft_eplb_stats_only = True
                                    ft_rebalance = False
                                    break
                                raise ValueError(
                                    f"FT dynamic EPLB lacks a reload layout for {leaf.prefix}"
                                )
                self._validate_loaded_weights(model)
                if meta_checkpoint is not None:
                    # Raw Meta tensors, unlike FT snapshots, still need the same
                    # bounded layer-at-a-time conversion as a floating HF checkpoint.
                    self._finalize_host_staged_modules(model, defer_executor=resident is not None)
                model.to(self.load_config.device)
                self._validate_loaded_weights(model)
                after_hooks = resident is not None and getattr(model, "resident_after_post_load", False)
                if resident is not None and not after_hooks:
                    ft_method = NewLoaderLoadMethod(ft_checkpoint.read_backend)
                    self._record_resident_owner(model, resident, ft_method, startup_lora,
                                                format_route="meta" if meta_checkpoint else "ft")
                self._run_post_load_hooks(model)
                validate_borrowed_parameters(model, self.load_config.device)
                if finish_meta_lora is not None:
                    finish_meta_lora()
                elif startup_lora is not None:
                    # Format adapters may import already packed runtime parameters.
                    from rtp_llm.models_py.lora_checkpoint import (
                        merge_startup_lora_checkpoint,
                    )

                    merge_startup_lora_checkpoint(model, self, *startup_lora)
                if resident is not None and after_hooks:
                    self._record_resident_owner(model, resident, NewLoaderLoadMethod(ft_checkpoint.read_backend),
                                                startup_lora, format_route="meta" if meta_checkpoint else "ft")
                model.eval()
            if ft_rebalance:
                # Keep the stage's immutable catalog, not a second model.
                # Keep it even in STATS: an install-capable stage can switch modes.
                self._eplb_ft_checkpoint = ft_checkpoint
            self._selected_load_method = (
                NewLoaderLoadMethod.FASTSAFETENSORS
                if ft_checkpoint.read_backend == "fastsafetensors"
                else NewLoaderLoadMethod.SCRATCH
            )
            logger.info(
                "NewLoader imported %s checkpoint with %s; "
                "CPU-owned payload is copied to %s (not direct GPU/GDS)",
                "Meta consolidated" if meta_checkpoint else "rank-local FT",
                ft_checkpoint.read_backend,
                self.load_config.device,
            )
            return model
        with self._checkpoint_preflight(model, requested_method):
            # A draft's owner can disagree on just one rank. Bind inside the
            # existing participant vote, before any peer selects an FST backend.
            bind_borrowed_parameters(model, self.parameter_aliases, self.load_config.device)
            validate_hf_checkpoint_storage(model)
            finish_lora = None
            if startup_lora is not None:
                from rtp_llm.models_py.lora_checkpoint import prepare_startup_lora_merge

                finish_lora = prepare_startup_lora_merge(model, self, *startup_lora)
            overlay = self._finetune_overlay()
            if weight_mapper.is_rank_local_checkpoint(checkpoint_files) and not bool(
                getattr(model, "supports_rank_local_checkpoint", False)
            ):
                raise ValueError(
                    f"{type(model).__name__} does not support rank-local consolidated "
                    "checkpoints; use a global HF checkpoint"
                )
            self._validate_runtime_backends(model, self.load_config.device)
            model_filter = self._model_checkpoint_name_filter(model)
            model_filter = exclude_borrowed_checkpoint_weights(model, model_filter)
            name_filter = self._checkpoint_name_filter(model_filter, expert_filter)
            base_filter = (
                overlay.base_filter(name_filter) if overlay is not None else name_filter
            )
            checkpoint_load_plan: Dict[str, CheckpointLoadTarget] = {}
            read_spec_resolver = self._checkpoint_read_spec_resolver(
                model, model_filter, expert_filter, checkpoint_load_plan
            )
            base_resolver = (
                overlay.base_read_spec_resolver(read_spec_resolver)
                if overlay is not None
                else read_spec_resolver
            )
            base_expander = (
                overlay.base_slice_expander(expert_filter)
                if overlay is not None
                else expert_filter
            )
            selected_files = weight_mapper.select_safetensor_files(
                self._resolved_model_path(),
                checkpoint_files,
                model_filter,
                self._safetensor_catalog(),
                allow_empty=overlay is not None and any(
                    name_filter is None or name_filter(key) for key in overlay.names
                ),
            )
        if len(selected_files) != len(checkpoint_files):
            logger.info(
                "Model checkpoint filter selected %d/%d shards",
                len(selected_files),
                len(checkpoint_files),
            )
        method = (
            NewLoaderLoadMethod.SCRATCH if self._w8a8_source is not None
            else self._select_load_method(
                requested_method, model, selected_files, expert_filter,
                overlay_only=not selected_files and overlay is not None,
            )
        )
        with self._checkpoint_preflight(model, requested_method):
            self._notify_checkpoint_source(model, method)
        logger.info(
            "NewLoader load method: requested=%s, selected=%s",
            requested_method.value,
            method.value,
        )
        started = time.time()
        staged_coordinator = None
        coordinator_modules: List[RtpModule] = []
        accelerator_fst = (
            method == NewLoaderLoadMethod.FASTSAFETENSORS
            and not self.load_config.force_cpu_load_weights
        )
        streamed_scratch = self._arm_streamed_scratch(
            model, method
        ) or self._ppu_streamed_scratch(model, method)
        if streamed_scratch:
            with self._scratch_stage("destination allocation"):
                self._prepare_streamed_scratch(model)
        if accelerator_fst:
            with self._checkpoint_preflight(model, requested_method):
                staged_modules = self._staged_device_modules(model)
                self._validate_staged_device_modules(
                    model, staged_modules, require_load_complete=True
                )
                self._prepare_model_for_fastsafetensors(
                    model,
                    self.load_config.device,
                    deferred_modules=staged_modules,
                )
                if staged_modules:
                    staged_coordinator = _StagedFSTLoadCoordinator(
                        model, staged_modules, self.load_config.device
                    )
                    coordinator_modules = [
                        module
                        for module in model.modules()
                        if isinstance(module, RtpModule)
                    ]
                    for module in coordinator_modules:
                        module._rtp_staged_load_coordinator = staged_coordinator
        if accelerator_fst:
            from rtp_llm.models_py.fst_file_plan import plan_fst_files
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors

            _, single_group = self._fastsafetensors_api()
            process_group = self._fastsafetensors_process_group(
                single_group, checkpoint_files, base_expander
            )
            shared, independent = plan_fst_files(
                selected_files, self._resolved_model_path(), process_group,
                overlay.base_filter(model_filter) if overlay is not None else model_filter,
                self._fixed_host_key_filter(model),
                catalog=self._checkpoint_catalog(), resolver=base_resolver, expander=base_expander,
            )
            if independent:
                logger.info("NewLoader FST: %d collective files; %d use independent native CPU range reads",
                            len(shared),len(independent))
            def iter_planned_weights():
                # yield from forwards close() to the active native reader on
                # dispatch failure. Unstarted readers never acquire resources.
                if shared:
                    yield from self._fastsafetensors_weights_iterator(
                        shared, self.load_config.device, base_filter, base_expander, base_resolver
                    )
                if independent:
                    yield from iter_fst_range_tensors(
                        independent, catalog=self._checkpoint_catalog(), name_filter=base_filter,
                        slice_expander=base_expander, read_spec_resolver=base_resolver,
                    )

            weights = iter_planned_weights()
        elif method == NewLoaderLoadMethod.FASTSAFETENSORS:
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors

            logger.info(
                "NewLoader: native FST CPU staging, runtime=%s", self.load_config.device
            )
            weights = iter_fst_range_tensors(
                selected_files,
                catalog=self._checkpoint_catalog(),
                name_filter=base_filter,
                slice_expander=base_expander,
                read_spec_resolver=base_resolver,
            )
        else:
            scratch_expert_reader = None
            if (
                self.load_config.moe_pure_tp_preshard
                and self.load_config.tp_size > 1
                and self.load_config.ep_size == 1
                and getattr(self.load_config.parallelism_config, "dp_size", 1) == 1
                and startup_lora is None
                and getattr(model, "supports_scratch_moe_tp_preshard", False)
            ):
                from rtp_llm.models_py.scratch_moe_tp import make_reader

                scratch_expert_reader = make_reader(
                    lambda key: checkpoint_load_plan.get(self._mapped_checkpoint_name(model, key))
                )
            weights = weight_mapper.get_all_weights(
                selected_files,
                catalog=self._checkpoint_catalog(),
                advise_scratch_cache=True,
                scratch_expert_reader=scratch_expert_reader,
                device="cpu",
                name_filter=base_filter,
                safetensor_slice_expander=base_expander,
                read_spec_resolver=base_resolver,
                **({"w8a8_source": self._w8a8_source} if self._w8a8_source is not None else {}),
            )
        if overlay is not None:
            weights = overlay.prepend(
                weights, name_filter, read_spec_resolver, expert_filter
            )
        previous_load_plan = getattr(
            model, "_rtp_checkpoint_load_plan", _CHECKPOINT_LOAD_PLAN_UNSET
        )
        model._rtp_checkpoint_load_plan = checkpoint_load_plan
        dispatch_context = (
            self._scratch_stage("checkpoint dispatch") if streamed_scratch else nullcontext()
        )
        def checked_weights():
            sources = {}
            for name, tensor in weights:
                mapped = self._mapped_checkpoint_name(model, name)
                previous = sources.setdefault(mapped, name)
                if previous != name:
                    raise ValueError(f"Ambiguous checkpoint sources {previous!r} and {name!r} map to {mapped!r}")
                yield name, tensor

        with self._checkpoint_preflight(model, requested_method), dispatch_context:
            try:
                model.load_weights(checked_weights())
                if staged_coordinator is not None:
                    staged_coordinator.finish()
            finally:
                if previous_load_plan is _CHECKPOINT_LOAD_PLAN_UNSET:
                    delattr(model, "_rtp_checkpoint_load_plan")
                else:
                    model._rtp_checkpoint_load_plan = previous_load_plan
                if staged_coordinator is not None:
                    for module in coordinator_modules:
                        delattr(module, "_rtp_staged_load_coordinator")
                close_weights = getattr(weights, "close", None)
                if callable(close_weights):
                    close_weights()
            self._validate_loaded_weights(model)
        logger.info(
            "Streamed checkpoint tensors with %s in %.2fs",
            method.value,
            time.time() - started,
        )

        with self._checkpoint_preflight(model, requested_method):
            if not accelerator_fst:
                self._finalize_host_staged_modules(model, defer_executor=resident is not None)
            validate_borrowed_parameters(model, self.load_config.device)
            model.to(self.load_config.device)
            self._validate_loaded_weights(model)
            after_hooks = resident is not None and getattr(model, "resident_after_post_load", False)
            if resident is not None and not after_hooks:
                self._record_resident_owner(model, resident, method, startup_lora)
            self._run_post_load_hooks(model)
            if finish_lora is not None:
                finish_lora()
            if resident is not None and after_hooks:
                self._record_resident_owner(model, resident, method, startup_lora)
            validate_borrowed_parameters(model)
            model.eval()
        self._selected_load_method = method
        return model
