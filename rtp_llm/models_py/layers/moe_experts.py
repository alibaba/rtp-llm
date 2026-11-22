"""Base class for stacked-expert MoE modules with EP/TP support.

Provides common logic for:
  - EP parameter computation (num_local_experts, expert id remapping)
  - TP slicing of per-expert projections
  - Buffer allocation for w13 (up+gate fused) and w2 (down)
  - Streaming weight dispatch from per-expert HF checkpoint tensors
  - Delegation to registered MoE quantization methods
  - FusedMoe construction via FusedMoeFactory

Quantized MoE families are implemented by registered QuantizeMethodBase
instances; this base class keeps only the unquantized fallback.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.expert_placement import (
    ExpertPlacement,
    expert_placement_for_layer,
)
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.moe_layout import prepare_moe_weight_layout
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.runtime_weight_update import (
    RuntimeWeightTransaction,
    TensorUpdate,
)



class BaseMoEExperts(RtpModule):
    """Stacked-expert MLP backed by FusedMoeFactory.

    w13 layout: [E, 2*M_tp, H]
        first half ([:M_tp])  = up_proj
        second half ([M_tp:]) = gate_proj
    w2 layout:  [E, H, M_tp]
        down_proj

    The runtime contract stores ``up_proj`` before ``gate_proj`` in ``w13``.

    Quantized MoE loading is delegated to registered methods (for example
    Fp8MoEMethod and W4A8Int4MoEMethod). This class owns the shared EP/TP
    routing and the unquantized fallback only.
    """

    # Subclasses should list the HF projection names they expect.
    PROJ_NAMES: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    _supports_newloader_eplb = True
    # The parent MoE owns this separate leaf; executor selection must know
    # whether shared-expert output is gated before choosing a fused strategy.
    has_shared_expert_gate = False

    # Quant type string -> family. Registered MoE quant methods own the actual
    # quantized buffer allocation/dispatch/postprocess; this map is used for
    # diagnostics and for fail-fast if a quant family is not registered.
    _BASE_QUANT_MAP = {
        "awq": "awq",
        "gptq": "gptq",
        "INT8": "weight_only_int8",
        "W8A8_INT8_PER_CHANNEL_COMPRESSED": "w8a8_int8_per_channel",
        # already-quantized FP8 ckpts (weight is fp8 + ckpt has weight_scale)
        "FP8_PER_TENSOR_COMPRESSED": "fp8_per_tensor",
        "FP8_DYNAMIC_PER_TENSOR": "fp8_per_tensor",
        "fp8": "fp8_per_tensor",
        "FP8_PER_BLOCK": "fp8_per_block",
        "fp8_block": "fp8_per_block",
        "FP8_PER_CHANNEL_COMPRESSED": "fp8_per_channel",
        "fp8_per_channel": "fp8_per_channel",
        "FP8_PER_CHANNEL_QUARK": "fp8_per_channel",
        # BF16/FP16 ckpts that get quantized at load time
        "fp8_online": "fp8_per_tensor_online",
        "fp8_block_online": "fp8_per_block_online",
        "fp8_per_channel_online": "fp8_per_channel_online",
        "W4A8_INT4_PER_CHANNEL": "w4a8_int4_per_channel_online",
        "W4A8_INT4_PER_CHANNEL_COMPRESSED": "w4a8_int4_per_channel_compressed",
        "modelopt_fp4": "nvfp4",
        "modelopt_fp4_online": "nvfp4_online",
        "mxfp4_quark": "mxfp4_quark",
        "QuarkMXFP4": "mxfp4_quark",
    }
    _EXTRA_QUANT_MAP: Dict[str, str] = {}
    _FP8_BLOCK_SIZE = 128

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        params_dtype: torch.dtype,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        layer_idx: int,
        prefix: Optional[str] = None,
    ):
        super().__init__()
        self.prefix = prefix or f"model.layers.{layer_idx}.mlp.experts"

        dimensions = {
            "num_experts": num_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
            "tp_size": tp_size,
            "ep_size": ep_size,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        ranks = (
            ("tp_rank", tp_rank, "tp_size", tp_size),
            ("ep_rank", ep_rank, "ep_size", ep_size),
        )
        for name, rank, size_name, size in ranks:
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 0 <= rank < size
            ):
                raise ValueError(
                    f"{name} must satisfy 0 <= {name} < {size_name}, "
                    f"got rank={rank!r}, size={size}"
                )

        activation_type = model_config.activation_type
        if activation_type != "SiGLU":
            from rtp_llm.ops import ActivationType

            if activation_type != ActivationType.Swiglu:
                raise ValueError(
                    f"Expected SiGLU MoE activation, got {activation_type!r}"
                )
        # The fused executor API accepts its canonical string spelling.  Model
        # configs may carry the equivalent enum, so normalize at construction.
        self.activation_type = "SiGLU"

        # Resolve quant family from quant_config. Ignored layers must use the
        # unquantized method and must not require quantization auxiliary tensors.
        ignored_by_quant_config = bool(
            quant_config is not None
            and quant_config.is_moe_layer_ignored(self, self.prefix)
        )
        raw_qt = (
            "none"
            if ignored_by_quant_config
            else (
                getattr(quant_config, "quant_type", "none")
                if quant_config is not None
                else "none"
            )
        )
        full_map = {**self._BASE_QUANT_MAP, **self._EXTRA_QUANT_MAP}
        self._quant_family = full_map.get(raw_qt, "none")

        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}) "
                "to avoid silently dropping experts during EP loading."
            )
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})"
            )

        # Shard expert intermediate dimensions only in pure TP mode. In EP or
        # DP topologies each rank owns a local expert subset and keeps it full.
        dp_size = getattr(parallelism_config, "dp_size", 1)
        dp_rank = getattr(parallelism_config, "dp_rank", 0)
        if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
            raise ValueError(f"dp_size must be a positive integer, got {dp_size!r}")
        if (
            isinstance(dp_rank, bool)
            or not isinstance(dp_rank, int)
            or not 0 <= dp_rank < dp_size
        ):
            raise ValueError(f"Invalid DP partition: rank={dp_rank}, size={dp_size}")
        self.moe_expert_tp_size = tp_size if ep_size == 1 and dp_size == 1 else 1
        self.moe_expert_tp_rank = tp_rank if self.moe_expert_tp_size > 1 else 0
        if moe_intermediate_size % self.moe_expert_tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size {moe_intermediate_size} not divisible "
                f"by moe_expert_tp_size {self.moe_expert_tp_size}"
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.moe_inter = moe_intermediate_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.layer_idx = layer_idx

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.moe_inter_tp = moe_intermediate_size // self.moe_expert_tp_size
        self._params_dtype = params_dtype

        # EP buffers are indexed by physical expert slots. Checkpoint ids and
        # gate outputs remain logical ids until the load/forward boundaries.
        self._expert_placement = expert_placement_for_layer(
            model_config,
            layer_index=layer_idx,
            logical_expert_count=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            parallelism_config=parallelism_config,
        )
        self.physical_num_experts = self._expert_placement.physical_expert_count
        self._ep_start_expert = self._expert_placement.local_physical_start
        self.num_local_experts = self._expert_placement.physical_experts_per_rank

        eplb_config = getattr(model_config, "eplb_config", None)
        enable_eplb = getattr(eplb_config, "enable_eplb", False)
        if callable(enable_eplb):
            enable_eplb = enable_eplb()
        if not isinstance(enable_eplb, bool):
            raise TypeError("eplb_config.enable_eplb must be a bool")
        self._dynamic_eplb_enabled = enable_eplb
        # The legacy runtime keeps the initial redundant placement on its
        # canonical logical slots.  Replica routing starts only after a fully
        # loaded EPLB plan has been installed atomically.
        self._eplb_plan_installed = False
        self._eplb_install_transaction = None
        self._eplb_previous_runtime_state = None
        logic_expert_cnt, log2phy = self._routing_tensors(self._expert_placement)
        self.register_buffer(
            "_eplb_logic_expert_cnt", logic_expert_cnt, persistent=False
        )
        self.register_buffer("_eplb_log2phy", log2phy, persistent=False)
        # This scalar is read by captured device code; a Python bool would freeze
        # the pre-install branch in every graph captured during warmup.
        self.register_buffer("_eplb_active", torch.tensor(False), persistent=False)
        # Explicit phy2log files need not place a logical expert in its
        # canonical slot. Keep the legacy choice where valid; otherwise select
        # a slot that actually contains that expert from the loaded placement.
        logical_per_rank = self.num_experts // self.ep_size
        initial_map = self._expert_placement.physical_to_logical
        first = {}
        for slot, logical in enumerate(initial_map):
            first.setdefault(logical, slot)
        routes = []
        for logical in range(self.num_experts):
            canonical = (
                logical // logical_per_rank
            ) * self.num_local_experts + logical % logical_per_rank
            routes.append(
                canonical if initial_map[canonical] == logical else first[logical]
            )
        self.register_buffer(
            "_initial_expert_route",
            torch.tensor(routes, dtype=torch.int64),
            persistent=False,
        )
        self._eplb_log_stats: Optional[torch.Tensor] = None
        self._eplb_gpu_loads: Optional[torch.Tensor] = None

        self._model_config = model_config
        self._parallelism_config = parallelism_config
        self._moe_config = moe_config
        self._quant_config = quant_config or QuantizationConfig("none")
        self._ignored_by_quant_config = ignored_by_quant_config
        self._effective_model_quant_config = (
            None
            if ignored_by_quant_config
            else getattr(model_config, "quant_config", None)
        )
        self._validate_quant_config_contract(raw_qt)

        # Unified dispatch (symmetric with LinearBase): registered MoE quant
        # methods own quantized loading. The built-in fallback is kept only for
        # unquantized weights when no method is registered.
        self.quant_method = self._quant_config.get_moe_quant_method(self, self.prefix)

        self._init_buffers(params_dtype)

        self.fused_moe: Optional[nn.Module] = None
        self._loaded_count = 0
        self._expected_count = self.num_local_experts * len(self.PROJ_NAMES)
        self._loaded_keys = set()
        self._loaded_aux_keys = set()
        self._seen_checkpoint_names = set()
        self._post_load_done = False
        self._runtime_layout_prepared = False
        self._runtime_layout_prepared_keys = set()
        self._runtime_weights_dict: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _routing_tensors(
        placement: ExpertPlacement,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_replicas = (
            placement.physical_expert_count - placement.logical_expert_count + 1
        )
        counts = torch.zeros(placement.logical_expert_count, dtype=torch.int32)
        log2phy = torch.full(
            (placement.logical_expert_count, max_replicas),
            -1,
            dtype=torch.int32,
        )
        for physical_id, logical_id in enumerate(placement.physical_to_logical):
            replica_id = int(counts[logical_id].item())
            log2phy[logical_id, replica_id] = physical_id
            counts[logical_id] += 1
        return counts.contiguous(), log2phy.contiguous()

    def clone_empty_for_placement(self, placement: ExpertPlacement) -> "BaseMoEExperts":
        """Create an unloaded one-layer expert module for an EPLB checkpoint reload."""
        if placement.logical_expert_count != self.num_experts:
            raise ValueError("EPLB placement logical expert count changed")
        if placement.ep_size != self.ep_size or placement.ep_rank != self.ep_rank:
            raise ValueError("EPLB placement EP partition changed")
        if placement.physical_expert_count != self.physical_num_experts:
            raise ValueError("EPLB placement physical expert count changed")
        clone = type(self)(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.moe_inter,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            params_dtype=self._params_dtype,
            model_config=self._model_config,
            parallelism_config=self._parallelism_config,
            moe_config=self._moe_config,
            quant_config=self._quant_config,
            layer_idx=self.layer_idx,
            prefix=self.prefix,
        )
        clone._expert_placement = placement
        clone.has_shared_expert_gate = self.has_shared_expert_gate
        return clone

    def _validate_quant_config_contract(self, runtime_method: str) -> None:
        """Keep loading layout and fused-MoE executor selection consistent."""
        if self._ignored_by_quant_config:
            return
        source_config = self._effective_model_quant_config
        if runtime_method == "none":
            if source_config is not None:
                raise ValueError(
                    f"MoE layer {self.prefix} has a model quantization config but "
                    "the newloader runtime method is unquantized"
                )
            return
        if source_config is None:
            raise ValueError(
                f"MoE layer {self.prefix} selected runtime quantization "
                f"{runtime_method!r} without model_config.quant_config"
            )

        source_method = moe_checkpoint_method_key(source_config)
        if source_method == getattr(self._quant_config, "ft_encoded_online_source", None):
            # FT stored the actual output of OldLoader's online quantizer; its
            # parameters must already match the prequantized owner exactly.
            source_method = source_method.removesuffix("_online")
        source_family = {
            **self._BASE_QUANT_MAP,
            **self._EXTRA_QUANT_MAP,
            "FP8": "fp8_per_tensor",
            "FP8_DYNAMIC_PER_TENSOR": "fp8_per_tensor",
        }.get(source_method)
        if source_family is None:
            raise ValueError(
                f"MoE layer {self.prefix} does not support model quantization "
                f"method {source_method!r}; disable NewLoader or register its "
                "runtime family"
            )
        if source_family != self._quant_family:
            raise ValueError(
                f"MoE layer {self.prefix} quantization mismatch: runtime "
                f"method {runtime_method!r} resolves to {self._quant_family!r}, "
                f"but model config method {source_method!r} resolves to "
                f"{source_family!r}"
            )

    # ------------------------------------------------------------------ #
    #  Buffer allocation — override in subclass for quantization
    # ------------------------------------------------------------------ #

    def _init_buffers(self, params_dtype: torch.dtype):
        """Allocate w13/w2 parameter buffers based on quantization family."""
        self.quant_method.create_weights(
            self,
            self.num_local_experts,
            self.hidden_size,
            self.moe_inter_tp,
            params_dtype,
        )

    # ------------------------------------------------------------------ #
    #  EP: expert id remapping
    # ------------------------------------------------------------------ #

    def _remap_expert_ids(self, global_expert_id: int) -> Tuple[int, ...]:
        """Map one logical checkpoint id to every matching local physical slot."""
        return self._expert_placement.local_indices(global_expert_id)

    def checkpoint_read_spec(self, name, shape):
        if self.quant_method.owns_checkpoint_loading:
            return self.quant_method.checkpoint_read_spec(self, name, shape)
        return super().checkpoint_read_spec(name, shape)

    def _ensure_projection_not_loaded(self, local_id: int, projection: str) -> None:
        key = (local_id, projection)
        if key in self._loaded_keys:
            raise RuntimeError(
                f"Duplicate MoE projection for local expert {local_id}: "
                f"{projection}. Checkpoint aliases must not provide the same "
                "logical weight more than once."
            )

    def _record_projection_loaded(self, local_id: int, projection: str) -> None:
        self._ensure_projection_not_loaded(local_id, projection)
        self._loaded_keys.add((local_id, projection))
        self._loaded_count += 1

    # ------------------------------------------------------------------ #
    #  Weight loading: streaming dispatch
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Parse per-expert HF weight names and dispatch to copy helpers.

        Expected name format: ``{expert_id}.{proj_name}.{param_name}``
        e.g. ``65.gate_proj.weight``, ``65.gate_proj.weight_scale``.
        """
        weights = self.quant_method.normalize_checkpoint_weights(weights)
        if self.quant_method.owns_checkpoint_loading:
            self.quant_method.load_weights(self, weights)
            return
        for name, tensor in weights.items():
            if name in self._seen_checkpoint_names:
                raise RuntimeError(
                    f"Duplicate MoE checkpoint tensor {self.prefix}.{name}"
                )
            self._seen_checkpoint_names.add(name)
            # Stacked (fused-experts) ckpt format: a single 3D tensor holds all
            # experts, e.g. Qwen3-VL-MoE ships ``experts.gate_up_proj`` [E,H,2M]
            # and ``experts.down_proj`` [E,M,H]. Split per-expert and reuse the
            # per-expert copy helpers (which do TP-slice + EP remap).
            parts = name.split(".")
            base = parts[0]
            if base in ("gate_up_proj", "down_proj"):
                if self.ep_size > 1:
                    raise ValueError(
                        "Stacked all-expert checkpoints are not supported with "
                        f"ep_size={self.ep_size}; use per-expert checkpoint keys "
                        "so each rank materializes only its local experts"
                    )
                if len(parts) not in (1, 2):
                    raise RuntimeError(
                        f"Unsupported stacked MoE checkpoint tensor "
                        f"{self.prefix}.{name}"
                    )
                parameter_name = parts[1] if len(parts) == 2 else "weight"
                if tensor.dim() < 1 or tensor.shape[0] != self.num_experts:
                    raise ValueError(
                        f"stacked {base}.{parameter_name} contains "
                        f"{tensor.shape[0] if tensor.dim() else 0} experts; "
                        f"expected {self.num_experts}"
                    )
                if parameter_name == "weight":
                    if tensor.dim() != 3:
                        raise ValueError(
                            f"stacked {base}.weight must be three-dimensional, "
                            f"got {tuple(tensor.shape)}"
                        )
                    if getattr(
                        self.quant_method, "uses_packed_weights", False
                    ) or getattr(self.quant_method, "owns_expert_dispatch", False):
                        self._load_stacked_expert_weights(base, parameter_name, tensor)
                    else:
                        self._load_stacked_experts(base, tensor)
                elif parameter_name == "weight_packed":
                    self._load_stacked_expert_weights(base, parameter_name, tensor)
                elif parameter_name in self._required_aux_param_names():
                    self._load_stacked_expert_aux(base, parameter_name, tensor)
                else:
                    raise RuntimeError(
                        f"Unsupported stacked MoE parameter " f"{self.prefix}.{name}"
                    )
                continue
            if len(parts) != 3:
                raise RuntimeError(
                    f"Unsupported MoE checkpoint tensor {self.prefix}.{name}"
                )
            try:
                global_expert_id = int(parts[0])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid expert id in checkpoint tensor {self.prefix}.{name}"
                ) from exc

            proj = parts[1]
            param_name = parts[2]
            if proj not in self.PROJ_NAMES and proj != "gate_up_proj":
                raise RuntimeError(
                    f"Unsupported MoE projection {proj!r} in {self.prefix}.{name}"
                )
            if global_expert_id < 0 or global_expert_id >= self.num_experts:
                raise RuntimeError(
                    f"Expert id {global_expert_id} is outside [0, "
                    f"{self.num_experts}) in {self.prefix}.{name}"
                )

            local_expert_ids = self._remap_expert_ids(global_expert_id)
            if not local_expert_ids:
                continue

            for local_expert_id in local_expert_ids:
                self._load_per_expert_tensor(
                    name, tensor, local_expert_id, proj, param_name
                )

    def _load_per_expert_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        local_expert_id: int,
        proj: str,
        param_name: str,
    ) -> None:
        """Copy one logical tensor into one local physical expert slot."""

        if param_name in ("weight", "weight_packed"):
            projections = (
                ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
            )
            for projection in projections:
                self._ensure_projection_not_loaded(local_expert_id, projection)

        if self.quant_method.dispatch_weight(
            self, local_expert_id, proj, param_name, tensor
        ):
            if proj == "gate_up_proj":
                self._record_projection_loaded(local_expert_id, "gate_proj")
                self._record_projection_loaded(local_expert_id, "up_proj")
            elif proj in self.PROJ_NAMES:
                self._record_projection_loaded(local_expert_id, proj)
            return

        if param_name == "weight":
            if proj == "gate_up_proj":
                self._dispatch_gate_up_weight(local_expert_id, tensor)
                return
            self._dispatch_weight(local_expert_id, proj, tensor)
            return

        if proj == "gate_up_proj":
            if param_name == "weight_packed":
                self._dispatch_gate_up_quant_weight(local_expert_id, param_name, tensor)
            elif param_name in self._required_aux_param_names():
                self._dispatch_gate_up_scale(local_expert_id, param_name, tensor)
            else:
                raise RuntimeError(
                    f"Unexpected gate_up_proj auxiliary tensor: "
                    f"expert={local_expert_id} param={param_name!r}"
                )
            return

        self._dispatch_scale(local_expert_id, proj, param_name, tensor)

    def _load_stacked_expert_aux(
        self, base: str, param_name: str, tensor: torch.Tensor
    ):
        for local_id, g in enumerate(
            self._expert_placement.local_physical_to_logical
        ):
            if base == "gate_up_proj":
                self._dispatch_gate_up_scale(local_id, param_name, tensor[g])
            else:
                self._dispatch_scale(local_id, "down_proj", param_name, tensor[g])

    def _load_stacked_expert_weights(
        self, base: str, param_name: str, tensor: torch.Tensor
    ) -> None:
        for local_id, global_id in enumerate(
            self._expert_placement.local_physical_to_logical
        ):
            if base == "gate_up_proj":
                self._dispatch_gate_up_quant_weight(
                    local_id, param_name, tensor[global_id]
                )
                continue
            self._ensure_projection_not_loaded(local_id, "down_proj")
            handled = self.quant_method.dispatch_weight(
                self,
                local_id,
                "down_proj",
                param_name,
                tensor[global_id],
            )
            if not handled:
                raise RuntimeError(
                    f"MoE quant method did not handle {base}.{param_name}"
                )
            self._record_projection_loaded(local_id, "down_proj")

    def _load_stacked_experts(self, base: str, tensor: torch.Tensor):
        """Load a stacked fused-experts tensor (one 3D tensor for all experts).

        Supported layouts are ``gate_up_proj`` in either ``[E,H,2M]`` or
        ``[E,2M,H]`` form and ``down_proj`` in either ``[E,M,H]`` or
        ``[E,H,M]`` form. The copy helpers apply TP slicing after EP selection.
        """
        E = tensor.shape[0]
        if E != self.num_experts:
            raise ValueError(
                f"stacked {base} contains {E} experts; expected {self.num_experts}"
            )
        if base == "gate_up_proj":
            M = self.moe_inter
            if self.hidden_size == 2 * M:
                raise ValueError(
                    "stacked gate_up_proj layout is ambiguous because hidden_size "
                    "equals 2 * moe_intermediate_size; use per-expert projection "
                    "names instead"
                )
            for local_id, g in enumerate(
                self._expert_placement.local_physical_to_logical
            ):
                self._ensure_projection_not_loaded(local_id, "gate_proj")
                self._ensure_projection_not_loaded(local_id, "up_proj")
                e = tensor[g]  # [H, 2M]
                if e.shape == (self.hidden_size, 2 * M):
                    # Qwen3-VL-MoE layout: [H, 2M] = [gate | up].
                    gate_e = e[:, :M].t().contiguous()  # [M, H]
                    up_e = e[:, M:].t().contiguous()  # [M, H]
                elif e.shape == (2 * M, self.hidden_size):
                    # Qwen3.5 BF16 layout: [2M, H] = [gate | up].
                    gate_e = e[:M, :].contiguous()
                    up_e = e[M:, :].contiguous()
                else:
                    raise ValueError(
                        f"stacked gate_up_proj expert {g} shape {tuple(e.shape)} "
                        f"does not match [H,2M]=({self.hidden_size},{2 * M}) "
                        f"or [2M,H]=({2 * M},{self.hidden_size})"
                    )
                self._copy_gate_or_up(local_id, gate_e, gate=True)
                self._copy_gate_or_up(local_id, up_e, gate=False)
                self._record_projection_loaded(local_id, "gate_proj")
                self._record_projection_loaded(local_id, "up_proj")
        else:  # down_proj
            if self.hidden_size == self.moe_inter:
                raise ValueError(
                    "stacked down_proj layout is ambiguous because hidden_size "
                    "equals moe_intermediate_size; use per-expert projection "
                    "names instead"
                )
            for local_id, g in enumerate(
                self._expert_placement.local_physical_to_logical
            ):
                self._ensure_projection_not_loaded(local_id, "down_proj")
                e = tensor[g]
                if e.shape == (self.moe_inter, self.hidden_size):
                    d = e.t().contiguous()  # [M, H] -> [H, M]
                elif e.shape == (self.hidden_size, self.moe_inter):
                    d = e.contiguous()
                else:
                    raise ValueError(
                        f"stacked down_proj expert {g} shape {tuple(e.shape)} "
                        f"does not match [M,H]=({self.moe_inter},{self.hidden_size}) "
                        f"or [H,M]=({self.hidden_size},{self.moe_inter})"
                    )
                self._copy_down(local_id, d)
                self._record_projection_loaded(local_id, "down_proj")

    def _split_gate_up_rows(
        self, param_name: str, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = self.moe_inter
        if tensor.shape[0] == 2 * M:
            return tensor[:M].contiguous(), tensor[M:].contiguous()
        if tensor.dim() >= 2 and tensor.shape[-2] == 2 * M:
            return (
                tensor.narrow(-2, 0, M).contiguous(),
                tensor.narrow(-2, M, M).contiguous(),
            )
        raise ValueError(
            f"gate_up_proj.{param_name} shape {tuple(tensor.shape)} cannot be "
            f"split into gate/up rows with moe_intermediate_size={M}"
        )

    def _dispatch_gate_up_quant_weight(
        self, local_id: int, param_name: str, tensor: torch.Tensor
    ):
        self._ensure_projection_not_loaded(local_id, "gate_proj")
        self._ensure_projection_not_loaded(local_id, "up_proj")
        gate, up = self._split_gate_up_rows(param_name, tensor)
        gate_done = self.quant_method.dispatch_weight(
            self, local_id, "gate_proj", param_name, gate
        )
        up_done = self.quant_method.dispatch_weight(
            self, local_id, "up_proj", param_name, up
        )
        if not gate_done or not up_done:
            raise RuntimeError(
                f"MoE quant method did not handle gate_up_proj.{param_name}"
            )
        self._record_projection_loaded(local_id, "gate_proj")
        self._record_projection_loaded(local_id, "up_proj")

    def _split_gate_up_aux_tensor(
        self, param_name: str, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._quant_family == "nvfp4" and param_name in (
            "weight_scale_2",
            "input_scale",
        ):
            flat = tensor.reshape(-1)
            if flat.numel() == 1:
                return flat, flat
            if flat.numel() == 2:
                return flat[:1], flat[1:]
        if (
            param_name in ("weight_scale", "input_scale")
            and self._quant_family == "fp8_per_tensor"
        ):
            flat = tensor.float().reshape(-1)
            if flat.numel() == 1:
                return flat[0:1], flat[0:1]
            if flat.numel() == 2:
                return flat[0:1], flat[1:2]

        if param_name != "weight_scale_inv":
            try:
                return self._split_gate_up_rows(param_name, tensor)
            except ValueError:
                pass

        if param_name == "weight_scale_inv" and (
            self._quant_family == "fp8_per_block"
            or getattr(self.quant_method, "source_fp8_block_size", None) == 128
        ):
            block_size = getattr(
                self,
                "_fp8_moe_weight_block_size",
                [self._FP8_BLOCK_SIZE, self._FP8_BLOCK_SIZE],
            )
            global_blocks = (self.moe_inter + block_size[0] - 1) // block_size[0]
            if tensor.shape[0] == 2 * global_blocks:
                return (
                    tensor[:global_blocks].contiguous(),
                    tensor[global_blocks:].contiguous(),
                )
            if tensor.dim() >= 2 and tensor.shape[-1] == 2 * global_blocks:
                gate = tensor.narrow(-1, 0, global_blocks).contiguous()
                up = tensor.narrow(-1, global_blocks, global_blocks).contiguous()
                if gate.dim() == 2:
                    gate = gate.t().contiguous()
                    up = up.t().contiguous()
                return gate, up

        raise ValueError(
            f"gate_up_proj.{param_name} shape "
            f"{tuple(tensor.shape)} cannot be split into gate/up scales for "
            f"quant family {self._quant_family!r}"
        )

    def _dispatch_gate_up_scale(
        self, local_id: int, param_name: str, tensor: torch.Tensor
    ):
        gate, up = self._split_gate_up_aux_tensor(param_name, tensor)
        self._dispatch_scale(local_id, "gate_proj", param_name, gate)
        self._dispatch_scale(local_id, "up_proj", param_name, up)

    def _dispatch_gate_up_weight(self, local_id: int, tensor: torch.Tensor):
        self._ensure_projection_not_loaded(local_id, "gate_proj")
        self._ensure_projection_not_loaded(local_id, "up_proj")
        M = self.moe_inter
        if self.hidden_size == 2 * M:
            raise ValueError(
                "gate_up_proj.weight layout is ambiguous because hidden_size "
                "equals 2 * moe_intermediate_size; use separate gate_proj and "
                "up_proj checkpoint keys"
            )
        if tensor.shape == (self.hidden_size, 2 * M):
            gate = tensor[:, :M].t().contiguous()
            up = tensor[:, M:].t().contiguous()
        elif tensor.shape == (2 * M, self.hidden_size):
            gate = tensor[:M, :].contiguous()
            up = tensor[M:, :].contiguous()
        else:
            raise ValueError(
                f"expert {local_id} gate_up_proj.weight shape {tuple(tensor.shape)} "
                f"does not match [H,2M]=({self.hidden_size},{2 * M}) "
                f"or [2M,H]=({2 * M},{self.hidden_size})"
            )
        self._copy_gate_or_up(local_id, gate, gate=True)
        self._copy_gate_or_up(local_id, up, gate=False)
        self._record_projection_loaded(local_id, "gate_proj")
        self._record_projection_loaded(local_id, "up_proj")

    def _dispatch_weight(self, local_id: int, proj: str, tensor: torch.Tensor):
        """Route a weight tensor to the correct buffer position."""
        self._ensure_projection_not_loaded(local_id, proj)
        if proj == "gate_proj":
            self._copy_gate_or_up(local_id, tensor, gate=True)
            self._record_projection_loaded(local_id, "gate_proj")
        elif proj == "up_proj":
            self._copy_gate_or_up(local_id, tensor, gate=False)
            self._record_projection_loaded(local_id, "up_proj")
        elif proj == "down_proj":
            if tensor.shape == (self.hidden_size, self.moe_inter):
                tensor = tensor.contiguous()
            elif tensor.shape == (self.moe_inter, self.hidden_size):
                tensor = tensor.t().contiguous()
            else:
                raise ValueError(
                    f"expert {local_id} down_proj.weight shape "
                    f"{tuple(tensor.shape)} does not match [H,M]="
                    f"({self.hidden_size},{self.moe_inter}) or [M,H]="
                    f"({self.moe_inter},{self.hidden_size})"
                )
            self._copy_down(local_id, tensor)
            self._record_projection_loaded(local_id, "down_proj")
        else:
            raise RuntimeError(f"Unsupported MoE projection {proj!r}")

    def _dispatch_scale(
        self, local_id: int, proj: str, param_name: str, tensor: torch.Tensor
    ):
        """Route a scale/meta tensor through the registered quant method."""
        required = self._is_required_aux_param(proj, param_name)
        key = (local_id, proj, param_name)
        if required and key in self._loaded_aux_keys:
            raise RuntimeError(
                f"Duplicate MoE auxiliary tensor for local expert {local_id}: "
                f"{proj}.{param_name}"
            )
        handled = self.quant_method.dispatch_scale(
            self, local_id, proj, param_name, tensor
        )
        if not handled:
            raise RuntimeError(
                f"Unexpected MoE auxiliary tensor: expert={local_id} "
                f"projection={proj!r} parameter={param_name!r}"
            )
        if proj in self.PROJ_NAMES and required:
            self._loaded_aux_keys.add(key)

    # ------------------------------------------------------------------ #
    #  Weight copy helpers (TP slicing + buffer write)
    # ------------------------------------------------------------------ #

    def _copy_gate_or_up(self, expert_id: int, tensor: torch.Tensor, gate: bool):
        """TP-slice a gate/up projection and write into w13 buffer.

        up_proj   → first  half of w13 (rows [0:M_tp])
        gate_proj → second half of w13 (rows [M_tp:2*M_tp])
        """
        tp_rows = self.w13.shape[1] // 2
        full_rows = tp_rows * self.moe_expert_tp_size
        if tensor.shape[0] != full_rows:
            raise ValueError(
                f"expert {expert_id} {'gate' if gate else 'up'}_proj.weight "
                f"dim-0 {tensor.shape[0]} != expected {full_rows}"
            )
        start = self.moe_expert_tp_rank * tp_rows
        sliced = tensor.narrow(0, start, tp_rows).contiguous()
        row_start = tp_rows if gate else 0
        target = self.w13.data[expert_id, row_start : row_start + tp_rows]
        if sliced.shape != target.shape:
            raise ValueError(
                f"expert {expert_id} {'gate' if gate else 'up'}_proj.weight "
                f"shape {tuple(tensor.shape)} is incompatible with local target "
                f"{tuple(target.shape)}"
            )
        copy_weight_(
            target,
            sliced,
            f"expert {expert_id} {'gate' if gate else 'up'}_proj.weight",
        )

    def _copy_down(self, expert_id: int, tensor: torch.Tensor):
        """TP-slice a down projection and write into w2 buffer."""
        tp_cols = self.w2.shape[2]
        full_cols = tp_cols * self.moe_expert_tp_size
        if tensor.shape[1] != full_cols:
            raise ValueError(
                f"expert {expert_id} down_proj.weight dim-1 {tensor.shape[1]} "
                f"!= expected {full_cols}"
            )
        start = self.moe_expert_tp_rank * tp_cols
        sliced = tensor.narrow(1, start, tp_cols).contiguous()
        target = self.w2.data[expert_id]
        if sliced.shape != target.shape:
            raise ValueError(
                f"expert {expert_id} down_proj.weight shape {tuple(tensor.shape)} "
                f"is incompatible with local target {tuple(target.shape)}"
            )
        copy_weight_(target, sliced, f"expert {expert_id} down_proj.weight")

    def _required_aux_param_names(self):
        return tuple(self.quant_method.required_aux_parameters())

    def _is_required_aux_param(self, proj: str, param_name: str) -> bool:
        return param_name in self._required_aux_param_names()

    def _check_load_complete(self):
        expected = {
            (expert_id, proj)
            for expert_id in range(self.num_local_experts)
            for proj in self.PROJ_NAMES
        }
        missing = sorted(expected - self._loaded_keys)
        if missing:
            sample = missing[:10]
            more = (
                f" (+{len(missing) - len(sample)} more)"
                if len(missing) > len(sample)
                else ""
            )
            raise RuntimeError(
                f"BaseMoEExperts layer {self.layer_idx} loaded {self._loaded_count}/"
                f"{self._expected_count} expert projection weights; missing {sample}{more}. "
                "Refusing to build fused MoE with uninitialized torch.empty buffers."
            )

        aux_names = self._required_aux_param_names()
        if not aux_names:
            return
        expected_aux = {
            (expert_id, proj, aux_name)
            for expert_id in range(self.num_local_experts)
            for proj in self.PROJ_NAMES
            for aux_name in aux_names
        }
        missing_aux = sorted(expected_aux - self._loaded_aux_keys)
        if missing_aux:
            sample = missing_aux[:10]
            more = (
                f" (+{len(missing_aux) - len(sample)} more)"
                if len(missing_aux) > len(sample)
                else ""
            )
            raise RuntimeError(
                f"BaseMoEExperts layer {self.layer_idx} missing required MoE "
                f"auxiliary tensors {sample}{more}; loaded={sorted(self._loaded_aux_keys)[:10]}. "
                "Refusing to build fused MoE with incomplete quantization scales."
            )

    # ------------------------------------------------------------------ #
    #  Post-load processing
    # ------------------------------------------------------------------ #

    def process_weights_after_loading(self):
        """Called after all weights are loaded.

        Runs registered quant-method post-load work after verifying all expert
        projection weights were loaded.
        """
        if self._post_load_done:
            return
        self._check_load_complete()
        if self.w13.device.type == "cpu":
            # CPU checkpoint staging is not a fused-MoE execution backend.
            # Leave runtime layouts and the completion flag untouched so a
            # later move to the accelerator can still finalize these weights.
            # In particular, do not probe CUDA strategies during a CPU load
            # in a ROCm build whose native library has no CUDA router symbols.
            return
        self.quant_method.process_weights_after_loading(self)
        self._maybe_build_fused_moe()
        self._post_load_done = True

    def prepare_eplb_reload_weights(self) -> Dict[str, torch.Tensor]:
        """Finalize a temporary reload module without constructing an executor."""
        if self._post_load_done:
            raise RuntimeError("EPLB reload weights were already finalized")
        self._check_load_complete()
        self.quant_method.process_weights_after_loading(self)
        weights = self._build_weights_dict()
        self._runtime_weights_dict = dict(weights)
        self._post_load_done = True
        return dict(weights)

    def checkpoint_load_complete(self) -> bool:
        expected = {
            (expert_id, proj)
            for expert_id in range(self.num_local_experts)
            for proj in self.PROJ_NAMES
        }
        if not expected.issubset(self._loaded_keys):
            return False
        aux_names = self._required_aux_param_names()
        if not aux_names:
            return True
        expected_aux = {
            (expert_id, proj, aux_name)
            for expert_id in range(self.num_local_experts)
            for proj in self.PROJ_NAMES
            for aux_name in aux_names
        }
        return expected_aux.issubset(self._loaded_aux_keys)

    def validate_weights_loaded(self, loaded_tensor_ids=None) -> None:
        self._check_load_complete()

    def validate_runtime_device(self, device: torch.device) -> None:
        self.quant_method.validate_runtime_device(device)

    def requires_staged_device_postprocess(self) -> bool:
        return bool(self.quant_method.requires_staged_device_postprocess)

    def _build_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Build the weight dict for FusedMoeFactory.

        Registered quant methods add any extra runtime tensors such as FP8 or
        W4A8 scales.
        """
        from rtp_llm.utils.model_weight import W

        weights_dict: Dict[str, torch.Tensor] = {
            W.moe_w1: self.w13.data,
            W.moe_w2: self.w2.data,
        }
        self.quant_method.add_weight_tensors(self, weights_dict)
        if not self._runtime_layout_prepared:
            for name, projection, is_scale in (
                (W.moe_w1, "gate_up", False),
                (W.moe_w2, "down", False),
                (W.moe_s1, "gate_up", True),
                (W.moe_s2, "down", True),
            ):
                tensor = weights_dict.get(name)
                if tensor is None or name in self._runtime_layout_prepared_keys:
                    continue
                shuffled = prepare_moe_weight_layout(
                    tensor, projection=projection, is_scale=is_scale
                )
                # ROCm layout conversion returns a new allocation.  Keeping the
                # registered pre-shuffle tensor alive would retain nearly two
                # copies of every expert weight for the lifetime of the model.
                # Make the runtime layout the module-owned storage, matching the
                # old loader's replace-on-postprocess behavior.
                self._replace_moe_runtime_storage(name, shuffled)
                weights_dict[name] = shuffled
                self._runtime_layout_prepared_keys.add(name)
                # Release each source before preparing the next projection,
                # not after all conversions (which doubles peak expert memory).
                # The key also makes retries safe after a later conversion fails.
                del tensor
            self._runtime_layout_prepared = True
        return weights_dict

    def _replace_moe_runtime_storage(self, name: str, tensor: torch.Tensor) -> None:
        from rtp_llm.utils.model_weight import W

        attribute = {
            W.moe_w1: "w13",
            W.moe_w2: "w2",
            W.moe_s1: "w13_scale",
            W.moe_s2: "w2_scale",
        }.get(name)
        if attribute is None or not hasattr(self, attribute):
            return
        parameter = self._parameters.get(attribute)
        if parameter is not None:
            parameter.data = tensor
        else:
            # Registered buffers remain registered when assigned through
            # Module.__setattr__.
            setattr(self, attribute, tensor)

    def _configure_factory_adapter(self, adapter):
        # Topk IDs handed to the executor are in the physical namespace.
        if adapter.physical_expert_num != self.physical_num_experts:
            raise ValueError("NewLoader placement and FusedMoE physical counts differ")
        adapter.expert_num = self.physical_num_experts
        adapter.has_shared_expert_gate = self.has_shared_expert_gate

    def _maybe_build_fused_moe(self):
        if self.fused_moe is not None:
            return
        build_owned = getattr(self.quant_method, "build_moe_executor", None)
        if callable(build_owned):
            executor = build_owned(self)
            if not callable(executor):
                raise TypeError("Quantized MoE builder must return an executable backend")
            self.fused_moe = executor
            return
        from rtp_llm.models_py.modules import FusedMoeFactory
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )

        # MoEConfigAdapter.quant_config must be the CONFIG-side QuantizationConfig
        # (rtp_llm.config.quant_config, exposing is_quanted()/get_method()), which
        # the strategy resolver and the DeepEP wrapper rely on. self._quant_config
        # is the models_py runtime QuantizationConfig (only quant_type) used to
        # pick the expert quant family — passing it here makes the DeepEP
        # low-latency path crash on quant_config.is_quanted(). Use the model's
        # config-side quant_config instead.
        adapter = MoEConfigAdapter(
            model_config=self._model_config,
            parallelism_config=self._parallelism_config,
            moe_config=self._moe_config,
            quant_config=self._effective_model_quant_config,
            enable_cuda_graph=self._enable_cuda_graph(),
        )
        self._configure_factory_adapter(adapter)
        weights_dict = self._build_weights_dict()
        self._runtime_weights_dict = dict(weights_dict)
        self.fused_moe = FusedMoeFactory().create_fused_moe(adapter, weights_dict)
        if self._dynamic_eplb_enabled and self.ep_size > 1:
            # Enable the matching pair before capture, never on the first update.
            executor = getattr(self.fused_moe, "fused_experts", None)
            router = getattr(self.fused_moe, "router", None)
            if hasattr(executor, "use_deterministic_eplb_gather") and hasattr(
                router, "use_deterministic_eplb_gather"
            ):
                executor.use_deterministic_eplb_gather = True
                router.use_deterministic_eplb_gather = True

    def is_cpu_checkpoint_staging(self) -> bool:
        """Loaded CPU parameters without accelerator layout/executor readiness.

        This state may be exported by BaseModel for CPU checkpoint inspection,
        but is not a fused-MoE/EPLB runtime view. A temporary reload explicitly
        finalized by prepare_eplb_reload_weights is not a staging-only module.
        """
        return (
            self.checkpoint_load_complete()
            and not self._post_load_done
            and self.fused_moe is None
            and self._runtime_weights_dict is None
            and not self._runtime_layout_prepared
            and not self._runtime_layout_prepared_keys
            and all(parameter.device.type == "cpu" for parameter in self.parameters())
        )

    def eplb_runtime_weight_view(self) -> Dict[str, torch.Tensor]:
        from rtp_llm.utils.model_weight import W

        if self._runtime_weights_dict is None:
            raise RuntimeError(
                f"MoE layer {self.layer_idx} runtime weights are not initialized"
            )
        result = dict(self._runtime_weights_dict)
        if self._dynamic_eplb_enabled:
            result[W.logic_expert_cnt] = self._eplb_logic_expert_cnt
            result[W.log2phy] = self._eplb_log2phy
        return result

    def validate_eplb_reload_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Validate a reloaded layer completely before any rank installs it."""
        runtime = self.eplb_runtime_weight_view()
        names = self._eplb_weight_names()
        unexpected = set(weights).difference(names)
        if unexpected:
            raise ValueError(f"Undeclared EPLB reload tensors: {sorted(unexpected)}")
        for name in names:
            target = runtime.get(name)
            source = weights.get(name)
            if target is None:
                if source is not None and source.numel() != 0:
                    raise ValueError(f"Unexpected EPLB runtime tensor {name!r}")
                continue
            if source is None:
                raise ValueError(f"Missing EPLB runtime tensor {name!r}")
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"EPLB tensor {name!r} mismatch: expected "
                    f"{tuple(target.shape)} {target.dtype}, got "
                    f"{tuple(source.shape)} {source.dtype}"
                )

    def bind_eplb_stats(self, log_stats: torch.Tensor, gpu_loads: torch.Tensor) -> None:
        expected_log = (self.num_experts,)
        expected_gpu = (self.ep_size,)
        if tuple(log_stats.shape) != expected_log or log_stats.dtype != torch.int32:
            raise ValueError(
                f"EPLB logical stats must be int32 {expected_log}, got "
                f"{tuple(log_stats.shape)} {log_stats.dtype}"
            )
        if tuple(gpu_loads.shape) != expected_gpu or gpu_loads.dtype != torch.int32:
            raise ValueError(
                f"EPLB GPU loads must be int32 {expected_gpu}, got "
                f"{tuple(gpu_loads.shape)} {gpu_loads.dtype}"
            )
        if log_stats.device != self.w13.device or gpu_loads.device != self.w13.device:
            raise ValueError("EPLB stats and MoE weights must be on the same device")
        self._eplb_log_stats = log_stats
        self._eplb_gpu_loads = gpu_loads

    def _eplb_weight_names(self):
        from rtp_llm.utils.model_weight import W

        return (
            W.moe_w1,
            W.moe_w2,
            W.moe_s1,
            W.moe_s2,
            *self.quant_method.eplb_aux_weight_names(),
        )

    def _record_eplb_stats(
        self, logical_ids: torch.Tensor, physical_ids: torch.Tensor
    ) -> None:
        if self._eplb_log_stats is None or self._eplb_gpu_loads is None:
            return
        flat_logical = logical_ids.reshape(-1).to(torch.int64)
        flat_physical = physical_ids.reshape(-1).to(torch.int64)
        valid = (flat_logical >= 0) & (flat_logical < self.num_experts)
        safe_logical = flat_logical.clamp(0, self.num_experts - 1)
        self._eplb_log_stats.scatter_add_(0, safe_logical, valid.to(dtype=torch.int32))
        physical_per_rank = self.physical_num_experts // self.ep_size
        safe_physical = flat_physical.clamp(0, self.physical_num_experts - 1)
        owners = torch.div(safe_physical, physical_per_rank, rounding_mode="floor")
        self._eplb_gpu_loads.scatter_add_(0, owners, valid.to(dtype=torch.int32))

    def apply_eplb_plan(
        self,
        phy2log: torch.Tensor,
        logic_expert_cnt: torch.Tensor,
        log2phy: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        *,
        defer_commit: bool = False,
    ) -> None:
        """Install a completely loaded plan while preserving captured addresses."""
        from rtp_llm.utils.model_weight import W

        if not self._dynamic_eplb_enabled:
            raise RuntimeError(
                f"MoE layer {self.layer_idx} was not initialized for EPLB"
            )
        if self._eplb_install_transaction is not None:
            raise RuntimeError(
                "Previous EPLB install still awaits all-rank acknowledgement"
            )
        if phy2log.dtype != torch.int32:
            raise TypeError("EPLB phy2log must use torch.int32")
        expected_phy2log = (self.physical_num_experts,)
        if tuple(phy2log.shape) != expected_phy2log:
            raise ValueError(
                f"EPLB phy2log must have shape {expected_phy2log}, got "
                f"{tuple(phy2log.shape)}"
            )
        placement = ExpertPlacement(
            logical_expert_count=self.num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            physical_to_logical=tuple(int(value) for value in phy2log.cpu().tolist()),
        )
        expected_logic = tuple(self._eplb_logic_expert_cnt.shape)
        expected_log2phy = tuple(self._eplb_log2phy.shape)
        if tuple(logic_expert_cnt.shape) != expected_logic:
            raise ValueError(
                f"EPLB logic_expert_cnt must have shape {expected_logic}, got "
                f"{tuple(logic_expert_cnt.shape)}"
            )
        if tuple(log2phy.shape) != expected_log2phy:
            raise ValueError(
                f"EPLB log2phy must have shape {expected_log2phy}, got "
                f"{tuple(log2phy.shape)}"
            )
        if logic_expert_cnt.dtype != torch.int32 or log2phy.dtype != torch.int32:
            raise TypeError("EPLB routing tensors must use torch.int32")
        expected_counts, expected_log2phy = self._routing_tensors(placement)
        if not torch.equal(logic_expert_cnt.cpu(), expected_counts) or not torch.equal(
            log2phy.cpu(), expected_log2phy
        ):
            raise ValueError("EPLB routing tensors do not match phy2log")
        self.validate_eplb_reload_weights(weights)
        runtime = self.eplb_runtime_weight_view()
        updates = [
            TensorUpdate(name, runtime[name], weights[name])
            for name in self._eplb_weight_names()
            if name in runtime
        ]
        updates.extend(
            [
                TensorUpdate(
                    W.logic_expert_cnt, self._eplb_logic_expert_cnt, logic_expert_cnt
                ),
                TensorUpdate(W.log2phy, self._eplb_log2phy, log2phy),
                TensorUpdate(
                    "plan_active", self._eplb_active, torch.ones_like(self._eplb_active)
                ),
            ]
        )
        transaction = RuntimeWeightTransaction(updates)
        self._eplb_install_transaction = transaction
        self._eplb_previous_runtime_state = (
            self._expert_placement,
            self._eplb_plan_installed,
        )
        try:
            transaction.install()
            self._expert_placement = placement
            self._eplb_plan_installed = True
            if not defer_commit:
                self.finalize_eplb_install(commit=True)
        except BaseException:
            self.finalize_eplb_install(commit=False)
            raise

    def finalize_eplb_install(self, *, commit: bool) -> None:
        """Acknowledge all ranks, or restore weights and routing on every rank."""
        transaction = self._eplb_install_transaction
        if transaction is None:
            return
        if commit:
            transaction.commit()
        else:
            transaction.rollback()
            self._expert_placement, self._eplb_plan_installed = (
                self._eplb_previous_runtime_state
            )
        self._eplb_install_transaction = None
        self._eplb_previous_runtime_state = None

    def _enable_cuda_graph(self) -> bool:
        hw_kernel_config = getattr(self._quant_config, "hw_kernel_config", None)
        enabled = getattr(hw_kernel_config, "enable_cuda_graph", False)
        if not isinstance(enabled, bool):
            raise TypeError("hw_kernel_config.enable_cuda_graph must be a bool")
        return enabled

    def _map_logical_to_physical(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Map logical ids to the legacy initial physical execution slots.

        The legacy Python MoE path keeps routing over the logical expert count.
        Each EP rank therefore executes its first ``logical / ep_size`` local
        slots; statically allocated redundant slots are loaded but stay idle
        until an EPLB plan explicitly selects them.  Preserve that initial
        routing while expressing the ids in the physical expert namespace used
        by NewLoader's fused MoE.
        """
        if not self._dynamic_eplb_enabled and self._expert_placement.is_identity:
            return topk_ids
        logical_ids = topk_ids.to(dtype=torch.int64)
        valid = (logical_ids >= 0) & (logical_ids < self.num_experts)
        safe_logical_ids = logical_ids.clamp(0, self.num_experts - 1)
        physical_ids = self._initial_expert_route[safe_logical_ids]
        if self._dynamic_eplb_enabled:
            canonical_ids = physical_ids
            flat_logical = safe_logical_ids.reshape(-1)
            replica_counts = self._eplb_logic_expert_cnt.index_select(
                0, flat_logical
            ).to(torch.int64)
            occurrence = torch.arange(
                flat_logical.numel(), device=flat_logical.device, dtype=torch.int64
            )
            # Every EP rank must resolve the same logical decision to the same
            # global physical expert.  The pure-TP router then masks that id to
            # its owning rank before the cross-rank reduction.  A rank-dependent
            # offset here makes different ranks execute different replicas for
            # one token, changing the summed MoE result after a plan install.
            replica_ids = torch.remainder(occurrence, replica_counts)
            physical_ids = self._eplb_log2phy[flat_logical, replica_ids].reshape_as(
                logical_ids
            )
            physical_ids = torch.where(self._eplb_active, physical_ids, canonical_ids)
        mapped = torch.where(valid, physical_ids, logical_ids)
        if self._dynamic_eplb_enabled:
            self._record_eplb_stats(logical_ids, mapped)
        return mapped.to(dtype=topk_ids.dtype)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        if self.fused_moe is None:
            raise RuntimeError(
                "Fused MoE executor is not initialized; complete weight loading "
                "and post-load processing before reading topk_ids_dtype"
            )
        return self.fused_moe.topk_ids_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        skip_tp_allreduce: bool = False,
    ) -> torch.Tensor:
        if self.fused_moe is None:
            raise RuntimeError(
                "Fused MoE executor is not initialized; complete weight loading "
                "and post-load processing before forward"
            )
        if hidden_states.dim() != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"MoE hidden states must have shape [tokens, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        expected_topk_shape = (hidden_states.shape[0], self.top_k)
        if tuple(topk_weights.shape) != expected_topk_shape:
            raise ValueError(
                f"MoE topk_weights must have shape {expected_topk_shape}, got "
                f"{tuple(topk_weights.shape)}"
            )
        if tuple(topk_ids.shape) != expected_topk_shape:
            raise ValueError(
                f"MoE topk_ids must have shape {expected_topk_shape}, got "
                f"{tuple(topk_ids.shape)}"
            )
        if not topk_weights.is_floating_point():
            raise TypeError("MoE topk_weights must be floating point")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"MoE topk_ids must be integer, got {topk_ids.dtype}")
        devices = {hidden_states.device, topk_weights.device, topk_ids.device}
        if len(devices) != 1:
            raise ValueError(
                "MoE hidden states, topk weights, and topk ids must share a device"
            )
        physical_topk_ids = self._map_logical_to_physical(topk_ids)
        extra_finalize_args = None
        if self._dynamic_eplb_enabled:
            extra_finalize_args = {
                "eplb_logical_topk_ids": topk_ids,
                "eplb_logical_experts_per_rank": self.num_experts // self.ep_size,
            }
        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=physical_topk_ids,
            activation=self.activation_type,
            extra_finalize_args=extra_finalize_args,
            **({"skip_tp_allreduce": True} if skip_tp_allreduce else {}),
        )
