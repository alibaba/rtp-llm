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

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from rtp_llm.device import get_current_device
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


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

    # Quant type string -> family. Registered MoE quant methods own the actual
    # quantized buffer allocation/dispatch/postprocess; this map is used for
    # diagnostics and for fail-fast if a quant family is not registered.
    _BASE_QUANT_MAP = {
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
        if layer_idx == 0:
            logger.info(
                "[BaseMoEExperts] layer_idx=%d resolved raw_qt=%r -> "
                "_quant_family=%r (num_experts=%d, hidden=%d, "
                "moe_inter=%d, tp=%d, ep=%d)",
                layer_idx,
                raw_qt,
                self._quant_family,
                num_experts,
                hidden_size,
                moe_intermediate_size,
                tp_size,
                ep_size,
            )

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

        # EP: compute local expert range
        if ep_size > 1:
            experts_per_rank = num_experts // ep_size
            self._ep_start_expert = ep_rank * experts_per_rank
            self.num_local_experts = experts_per_rank
        else:
            self._ep_start_expert = 0
            self.num_local_experts = num_experts

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
        if layer_idx == 0:
            logger.info(
                "[BaseMoEExperts] layer_idx=0 quant_method=%s "
                "(all expert layouts use method delegation)",
                type(self.quant_method).__name__,
            )

        self._init_buffers(params_dtype)

        self.fused_moe: Optional[nn.Module] = None
        self._loaded_count = 0
        self._expected_count = self.num_local_experts * len(self.PROJ_NAMES)
        self._loaded_keys = set()
        self._loaded_aux_keys = set()
        self._seen_checkpoint_names = set()
        self._post_load_done = False

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

        key_getter = getattr(source_config, "get_runtime_method_key", None)
        source_method = key_getter() if callable(key_getter) else ""
        if not source_method:
            key_getter = getattr(source_config, "get_method", None)
            source_method = key_getter() if callable(key_getter) else ""
        source_family = {
            **self._BASE_QUANT_MAP,
            **self._EXTRA_QUANT_MAP,
            "FP8": "fp8_per_tensor",
            "FP8_DYNAMIC_PER_TENSOR": "fp8_per_tensor",
        }.get(source_method)
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

    def _remap_expert_id(self, global_expert_id: int) -> Optional[int]:
        """Map global expert id to local buffer index.  Returns None if out of range."""
        local_id = global_expert_id - self._ep_start_expert
        if 0 <= local_id < self.num_local_experts:
            return local_id
        return None

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
                    self._load_stacked_experts(base, tensor)
                elif parameter_name == "weight_packed":
                    self._load_stacked_expert_weights(base, parameter_name, tensor)
                elif parameter_name in ("weight_scale", "weight_scale_inv"):
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

            local_expert_id = self._remap_expert_id(global_expert_id)
            if local_expert_id is None:
                continue

            if param_name in ("weight", "weight_packed"):
                projections = (
                    ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
                )
                for projection in projections:
                    self._ensure_projection_not_loaded(local_expert_id, projection)

            if self.quant_method.dispatch_weight(
                self, local_expert_id, proj, param_name, tensor
            ):
                if proj in self.PROJ_NAMES:
                    self._record_projection_loaded(local_expert_id, proj)
                continue

            if param_name == "weight":
                if proj == "gate_up_proj":
                    self._dispatch_gate_up_weight(local_expert_id, tensor)
                    continue
                # Log first weight tensor we see, for diagnostics.
                if self.layer_idx == 0 and not getattr(
                    self, "_logged_first_weight", False
                ):
                    logger.info(
                        "[BaseMoEExperts] layer_idx=0 first weight: "
                        "name=%r proj=%r dtype=%s shape=%s buf_dtype=%s "
                        "buf_shape=%s _quant_family=%r",
                        name,
                        proj,
                        tensor.dtype,
                        tuple(tensor.shape),
                        self.w13.dtype if proj != "down_proj" else self.w2.dtype,
                        tuple(self.w13.shape if proj != "down_proj" else self.w2.shape),
                        self._quant_family,
                    )
                    self._logged_first_weight = True
                self._dispatch_weight(local_expert_id, proj, tensor)
            else:
                if proj == "gate_up_proj":
                    if param_name == "weight_packed":
                        self._dispatch_gate_up_quant_weight(
                            local_expert_id, param_name, tensor
                        )
                    elif param_name in ("weight_scale", "weight_scale_inv"):
                        self._dispatch_gate_up_scale(
                            local_expert_id, param_name, tensor
                        )
                    else:
                        raise RuntimeError(
                            f"Unexpected gate_up_proj auxiliary tensor: "
                            f"expert={local_expert_id} param={param_name!r}"
                        )
                    continue
                # Log first scale tensor we see, for diagnostics.
                if self.layer_idx == 0 and not getattr(
                    self, "_logged_first_scale", False
                ):
                    logger.info(
                        "[BaseMoEExperts] layer_idx=0 first non-weight tensor: "
                        "name=%r proj=%r param=%r dtype=%s shape=%s "
                        "_quant_family=%r",
                        name,
                        proj,
                        param_name,
                        tensor.dtype,
                        tuple(tensor.shape),
                        self._quant_family,
                    )
                    self._logged_first_scale = True
                self._dispatch_scale(local_expert_id, proj, param_name, tensor)

    def _load_stacked_expert_aux(
        self, base: str, param_name: str, tensor: torch.Tensor
    ):
        for g in range(tensor.shape[0]):
            local_id = self._remap_expert_id(g)
            if local_id is None:
                continue
            if base == "gate_up_proj":
                self._dispatch_gate_up_scale(local_id, param_name, tensor[g])
            else:
                self._dispatch_scale(local_id, "down_proj", param_name, tensor[g])

    def _load_stacked_expert_weights(
        self, base: str, param_name: str, tensor: torch.Tensor
    ) -> None:
        for global_id in range(self.num_experts):
            local_id = self._remap_expert_id(global_id)
            if local_id is None:
                continue
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
            for g in range(E):
                local_id = self._remap_expert_id(g)
                if local_id is None:
                    continue
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
            for g in range(E):
                local_id = self._remap_expert_id(g)
                if local_id is None:
                    continue
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
        if param_name == "weight_scale" and self._quant_family == "fp8_per_tensor":
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

        if param_name == "weight_scale_inv" and self._quant_family == "fp8_per_block":
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
        target.copy_(sliced)

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
        target.copy_(sliced)

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
        self.quant_method.process_weights_after_loading(self)
        self._maybe_build_fused_moe()
        self._post_load_done = True

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
        weights_dict: Dict[str, torch.Tensor] = {
            W.moe_w1: self.w13.data,
            W.moe_w2: self.w2.data,
        }
        self.quant_method.add_weight_tensors(self, weights_dict)
        runtime_device = getattr(self._model_config, "exported_device", None)
        if runtime_device is None:
            runtime_device = get_current_device()
        if self.layer_idx == 0 and not getattr(
            self, "_logged_moe_runtime_device", False
        ):
            logger.info(
                "[BaseMoEExperts] layer_idx=0 using runtime_device=%s for newloader MoE postprocess",
                type(runtime_device).__name__ if runtime_device is not None else None,
            )
            self._logged_moe_runtime_device = True
        if runtime_device is not None:
            for name in (W.moe_w1, W.moe_w2, W.moe_s1, W.moe_s2):
                tensor = weights_dict.get(name)
                if tensor is None:
                    continue
                # Build only the runtime layout required by the fused kernel.
                # Per-tensor FP8 scales are 1D and do not encode gate/up rows.
                if tensor.dim() == 1:
                    continue
                weights_dict[name] = runtime_device.shuffle_moe_weight(
                    tensor, self._model_config.data_type, name
                )
        return weights_dict

    def _maybe_build_fused_moe(self):
        if self.fused_moe is not None:
            return
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
        weights_dict = self._build_weights_dict()
        self.fused_moe = FusedMoeFactory().create_fused_moe(adapter, weights_dict)

    def _enable_cuda_graph(self) -> bool:
        hw_kernel_config = getattr(self._quant_config, "hw_kernel_config", None)
        enabled = getattr(hw_kernel_config, "enable_cuda_graph", False)
        if not isinstance(enabled, bool):
            raise TypeError("hw_kernel_config.enable_cuda_graph must be a bool")
        return enabled

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
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
        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )
