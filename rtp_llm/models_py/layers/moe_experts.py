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
from rtp_llm.models_py.modules import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class BaseMoEExperts(nn.Module):
    """Stacked-expert MLP backed by FusedMoeFactory.

    w13 layout: [E, 2*M_tp, H]
        first half ([:M_tp])  = up_proj
        second half ([M_tp:]) = gate_proj
    w2 layout:  [E, H, M_tp]
        down_proj

    This matches Qwen3.5 old loader's stack_moe_w1 input order:
    CkptWeightInfo lists up_proj first and gate_proj second, so the fused
    tensor is concat([up, gate], dim=1).

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

        # Resolve quant family from quant_config. Ignored layers must use the
        # unquantized method and must not require quantization auxiliary tensors.
        ignored_by_quant_config = bool(
            quant_config is not None and quant_config.is_layer_ignored(self.prefix)
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

        if ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {ep_size}")
        if ep_rank < 0 or ep_rank >= ep_size:
            raise ValueError(
                f"ep_rank must satisfy 0 <= ep_rank < ep_size, got "
                f"ep_rank={ep_rank}, ep_size={ep_size}"
            )
        if num_experts < 0:
            raise ValueError(f"num_experts must be non-negative, got {num_experts}")
        if num_experts != 0 and num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}) "
                "to avoid silently dropping experts during EP loading."
            )

        # Match old loader's moe_pure_tp_mode: only shard expert intermediate
        # dimensions when there is no EP/DP split. In EP all-gather topologies
        # each rank owns a local expert subset and must keep those experts full.
        dp_size = getattr(parallelism_config, "dp_size", 1)
        self.moe_expert_tp_size = tp_size if ep_size == 1 and dp_size == 1 else 1
        self.moe_expert_tp_rank = tp_rank if self.moe_expert_tp_size > 1 else 0
        if moe_intermediate_size % self.moe_expert_tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size {moe_intermediate_size} not divisible "
                f"by moe_expert_tp_size {self.moe_expert_tp_size}"
            )

        self.num_experts = num_experts
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
        self._quant_config = quant_config
        self._ignored_by_quant_config = ignored_by_quant_config
        self._effective_model_quant_config = (
            None if ignored_by_quant_config else getattr(model_config, "quant_config", None)
        )

        # Unified dispatch (symmetric with LinearBase): registered MoE quant
        # methods own quantized loading. The built-in fallback is kept only for
        # unquantized weights when no method is registered.
        self.quant_method = (
            quant_config.get_quant_method(self, self.prefix)
            if quant_config is not None
            else None
        )
        if self.quant_method is None and self._quant_family != "none":
            raise ValueError(
                f"MoE quant family {self._quant_family!r} is not registered in "
                "the newloader MoE quant-method registry; refusing to use the "
                "legacy built-in FP8 fallback path."
            )
        if layer_idx == 0:
            logger.info(
                "[BaseMoEExperts] layer_idx=0 quant_method=%s "
                "(None=unquantized fallback; otherwise method delegation)",
                type(self.quant_method).__name__ if self.quant_method else None,
            )

        self._init_buffers(params_dtype)

        self.fused_moe: Optional[nn.Module] = None
        self._loaded_count = 0
        self._expected_count = self.num_local_experts * len(self.PROJ_NAMES)
        self._loaded_keys = set()
        self._loaded_aux_keys = set()

    # ------------------------------------------------------------------ #
    #  Buffer allocation — override in subclass for quantization
    # ------------------------------------------------------------------ #

    def _init_buffers(self, params_dtype: torch.dtype):
        """Allocate w13/w2 parameter buffers based on quantization family."""
        if self.quant_method is not None:
            # 委托给统一派发拿到的 MoE 量化方法（Step B 起 fp8 走这条）。
            self.quant_method.create_weights(
                self,
                self.num_local_experts,
                self.hidden_size,
                self.moe_inter_tp,
                params_dtype,
            )
            return
        E = self.num_local_experts
        M_tp = self.moe_inter_tp
        H = self.hidden_size
        qf = self._quant_family
        if qf != "none":
            raise ValueError(
                f"MoE quant family {qf!r} must be handled by a registered "
                "QuantizeMethodBase implementation."
            )

        # Unquantized BF16/FP16 fallback. Quantized MoE families are handled by
        # registered methods such as Fp8MoEMethod/W4A8Int4MoEMethod; keeping a
        # second built-in quantized implementation here would make the two paths
        # drift silently.
        self.w13 = nn.Parameter(
            torch.empty(E, 2 * M_tp, H, dtype=params_dtype),
            requires_grad=False,
        )
        self.w2 = nn.Parameter(
            torch.empty(E, H, M_tp, dtype=params_dtype),
            requires_grad=False,
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

    # ------------------------------------------------------------------ #
    #  Weight loading: streaming dispatch
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Parse per-expert HF weight names and dispatch to copy helpers.

        Expected name format: ``{expert_id}.{proj_name}.{param_name}``
        e.g. ``65.gate_proj.weight``, ``65.gate_proj.weight_scale``.
        """
        for name, tensor in weights.items():
            # Stacked (fused-experts) ckpt format: a single 3D tensor holds all
            # experts, e.g. Qwen3-VL-MoE ships ``experts.gate_up_proj`` [E,H,2M]
            # and ``experts.down_proj`` [E,M,H]. Split per-expert and reuse the
            # per-expert copy helpers (which do TP-slice + EP remap).
            parts = name.split(".")
            base = parts[0]
            if base in ("gate_up_proj", "down_proj") and hasattr(tensor, "dim"):
                if tensor.dim() == 3 and len(parts) == 1:
                    self._load_stacked_experts(base, tensor)
                    continue
                if len(parts) == 2 and tensor.shape[0] == self.num_experts:
                    self._load_stacked_expert_aux(base, parts[1], tensor)
                    continue
            if len(parts) < 3:
                continue
            try:
                global_expert_id = int(parts[0])
            except ValueError:
                continue

            local_expert_id = self._remap_expert_id(global_expert_id)
            if local_expert_id is None:
                continue

            proj = parts[1]
            param_name = parts[2]

            if self.quant_method is not None and self.quant_method.dispatch_weight(
                self, local_expert_id, proj, param_name, tensor
            ):
                self._loaded_count += 1
                if proj in self.PROJ_NAMES:
                    self._loaded_keys.add((local_expert_id, proj))
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
                        self._dispatch_gate_up_scale(local_expert_id, param_name, tensor)
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

    def _load_stacked_expert_aux(self, base: str, param_name: str, tensor: torch.Tensor):
        for g in range(tensor.shape[0]):
            local_id = self._remap_expert_id(g)
            if local_id is None:
                continue
            if base == "gate_up_proj":
                self._dispatch_gate_up_scale(local_id, param_name, tensor[g])
            else:
                self._dispatch_scale(local_id, "down_proj", param_name, tensor[g])

    def _load_stacked_experts(self, base: str, tensor: torch.Tensor):
        """Load a stacked fused-experts tensor (one 3D tensor for all experts).

        Layouts (HF Qwen3-VL-MoE, [E, in, out]):
          * ``gate_up_proj``: [E, H, 2M]，最后一维 2M = [gate(M) | up(M)]。
            每个专家转置 + 切 gate/up，得到 per-expert [M, H]，复用 ``_copy_gate_or_up``
            (Qwen3.5 split expert 旧 loader 的 w13 顺序是 up 在前、gate 在后)。
          * ``down_proj``: [E, M, H]，转置成 [H, M] 复用 ``_copy_down``。
        ``_copy_*`` 内部负责 TP 切分；``_remap_expert_id`` 负责 EP 选本地专家。
        """
        E = tensor.shape[0]
        if base == "gate_up_proj":
            M = self.moe_inter
            for g in range(E):
                local_id = self._remap_expert_id(g)
                if local_id is None:
                    continue
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
                self._loaded_count += 2
                self._loaded_keys.add((local_id, "gate_proj"))
                self._loaded_keys.add((local_id, "up_proj"))
        else:  # down_proj
            for g in range(E):
                local_id = self._remap_expert_id(g)
                if local_id is None:
                    continue
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
                self._loaded_count += 1
                self._loaded_keys.add((local_id, "down_proj"))

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
        gate, up = self._split_gate_up_rows(param_name, tensor)
        if self.quant_method is None:
            raise RuntimeError(
                f"Unexpected gate_up_proj.{param_name} without a MoE quant method"
            )
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
        self._loaded_count += 2
        self._loaded_keys.add((local_id, "gate_proj"))
        self._loaded_keys.add((local_id, "up_proj"))

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
            nb = getattr(self, "_n_scale_blocks_per_proj", None)
            if nb is not None and tensor.shape[0] == 2 * nb:
                return tensor[:nb].contiguous(), tensor[nb:].contiguous()

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
        M = self.moe_inter
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
        self._loaded_count += 2
        self._loaded_keys.add((local_id, "gate_proj"))
        self._loaded_keys.add((local_id, "up_proj"))

    def _dispatch_weight(self, local_id: int, proj: str, tensor: torch.Tensor):
        """Route a weight tensor to the correct buffer position."""
        if proj == "gate_proj":
            self._copy_gate_or_up(local_id, tensor, gate=True)
            self._loaded_count += 1
            self._loaded_keys.add((local_id, "gate_proj"))
        elif proj == "up_proj":
            self._copy_gate_or_up(local_id, tensor, gate=False)
            self._loaded_count += 1
            self._loaded_keys.add((local_id, "up_proj"))
        elif proj == "down_proj":
            if tensor.shape == (self.moe_inter, self.hidden_size):
                tensor = tensor.t().contiguous()
            self._copy_down(local_id, tensor)
            self._loaded_count += 1
            self._loaded_keys.add((local_id, "down_proj"))

    def _dispatch_scale(
        self, local_id: int, proj: str, param_name: str, tensor: torch.Tensor
    ):
        """Route a scale/meta tensor through the registered quant method."""
        if self.quant_method is not None:
            self.quant_method.dispatch_scale(self, local_id, proj, param_name, tensor)
            if proj in self.PROJ_NAMES and self._is_required_aux_param(proj, param_name):
                self._loaded_aux_keys.add((local_id, proj, param_name))
            return
        raise RuntimeError(
            f"Unexpected MoE auxiliary tensor for unquantized fallback: "
            f"expert={local_id} proj={proj!r} param={param_name!r}. Quantized "
            "MoE checkpoints must be handled by a registered quant method."
        )

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
        self.w13.data[expert_id, row_start : row_start + tp_rows].copy_(sliced)

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
        self.w2.data[expert_id].copy_(sliced)


    def _required_aux_param_names(self):
        qf = self._quant_family
        if qf in ("fp8_per_tensor", "fp8_per_channel"):
            return ("weight_scale",)
        if qf == "fp8_per_block":
            return ("weight_scale_inv",)
        if qf == "w4a8_int4_per_channel_compressed":
            return ("weight_scale",)
        # Online quantization creates scales during post-load; unquantized has no aux tensors.
        return ()

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
            more = f" (+{len(missing) - len(sample)} more)" if len(missing) > len(sample) else ""
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
        self._check_load_complete()
        if self.quant_method is not None:
            self.quant_method.process_weights_after_loading(self)
            self._maybe_build_fused_moe()
            return
        self._maybe_build_fused_moe()

    def _build_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Build the weight dict for FusedMoeFactory.

        Registered quant methods add any extra runtime tensors such as FP8 or
        W4A8 scales.
        """
        weights_dict: Dict[str, torch.Tensor] = {
            W.moe_w1: self.w13.data,
            W.moe_w2: self.w2.data,
        }
        if self.quant_method is not None:
            self.quant_method.add_weight_tensors(self, weights_dict)
        runtime_device = getattr(self._model_config, "exported_device", None)
        if runtime_device is None:
            runtime_device = get_current_device()
        if self.layer_idx == 0 and not getattr(self, "_logged_moe_runtime_device", False):
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
                # Newloader builds executor weights directly from PyModel tensors.
                # Apply the runtime MoE layout transform here so ROCm AITER gets
                # gate/up order and preshuffled weights without using the legacy
                # loader's AtomicWeight path. Per-tensor FP8 scales are 1D and
                # do not encode gate/up rows.
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
            enable_cuda_graph=False,
        )
        weights_dict = self._build_weights_dict()
        self.fused_moe = FusedMoeFactory().create_fused_moe(adapter, weights_dict)

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
            self._maybe_build_fused_moe()
        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )
