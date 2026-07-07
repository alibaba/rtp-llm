"""Base class for stacked-expert MoE modules with EP/TP support.

Provides common logic for:
  - EP parameter computation (num_local_experts, expert id remapping)
  - TP slicing of per-expert projections
  - Buffer allocation for w13 (up+gate fused) and w2 (down)
  - Streaming weight dispatch from per-expert HF checkpoint tensors
  - FP8 quantization support (per-tensor, per-channel, per-block)
  - FusedMoe construction via FusedMoeFactory

Subclasses can extend with model-specific quantization (FP4, W4A8, etc.)
by overriding _init_buffers, _dispatch_scale, process_weights_after_loading,
and _build_weights_dict.
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

    Built-in FP8 quantization support:
      - fp8_per_tensor: one scalar scale per expert per projection
      - fp8_per_channel: one scale per output row of each projection
      - fp8_per_block: one scale per 128x128 block (DeepSeek-style)

    Subclasses can extend with model-specific quantization (FP4, W4A8)
    by overriding the relevant methods.
    """

    # Subclasses should list the HF projection names they expect.
    PROJ_NAMES: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    # Quant type string -> family.  Subclasses can extend via _EXTRA_QUANT_MAP.
    #
    # "*_online" families load BF16/FP16 weights from the ckpt and quantize at
    # the end of loading (see _online_quantize_*); the others assume the ckpt
    # is already quantized and ships per-projection weight_scale tensors that
    # the load_weights stream will dispatch.
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
    ):
        super().__init__()

        # Resolve quant family from quant_config.
        raw_qt = (
            getattr(quant_config, "quant_type", "none")
            if quant_config is not None
            else "none"
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

        # 统一派发（与 LinearBase 对称）:按层类型从 quant_config 取 MoE 量化方法。
        # 拿到 FusedMoEMethodBase → 委托给它（见 _init_buffers / _dispatch_scale /
        # process_weights_after_loading / _build_weights_dict 里的 if 分支）;
        # 拿到 None（当前 fp8/unquantized 尚未注册 MoE 方法）→ 走内置 _quant_family
        # 逻辑,行为与改动前完全一致。
        self.quant_method = (
            quant_config.get_quant_method(self) if quant_config is not None else None
        )
        if layer_idx == 0:
            logger.info(
                "[BaseMoEExperts] layer_idx=0 quant_method=%s "
                "(None=走内置 _quant_family;否则走 method 委托)",
                type(self.quant_method).__name__ if self.quant_method else None,
            )

        self._init_buffers(params_dtype)

        self.fused_moe: Optional[nn.Module] = None
        self._loaded_count = 0
        self._expected_count = self.num_local_experts * len(self.PROJ_NAMES)

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

        if qf == "fp8_per_tensor":
            dt = torch.float8_e4m3fn
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            # Temporary per-projection scales; fused in process_weights_after_loading.
            self.register_buffer("_gate_scales", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("_up_scales", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("_down_scales", torch.zeros(E, dtype=torch.float32))
            # Fused scales: one scalar per expert.
            self.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
            self.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))

        elif qf == "fp8_per_channel":
            dt = torch.float8_e4m3fn
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            # Temporary per-channel scales per projection.
            self.register_buffer(
                "_gate_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            self.register_buffer(
                "_up_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            self.register_buffer(
                "_down_ch_scales", torch.zeros(E, H, dtype=torch.float32)
            )
            # Fused scales: per output row.
            self.register_buffer(
                "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
            )
            self.register_buffer("w2_scale", torch.zeros(E, H, dtype=torch.float32))

        elif qf == "fp8_per_block":
            dt = torch.float8_e4m3fn
            BS = self._FP8_BLOCK_SIZE
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            nb = (M_tp + BS - 1) // BS
            self._n_scale_blocks_per_proj = nb
            self.register_buffer(
                "w13_scale",
                torch.zeros(E, 2 * nb, (H + BS - 1) // BS, dtype=torch.float32),
            )
            self.register_buffer(
                "w2_scale",
                torch.zeros(E, (H + BS - 1) // BS, nb, dtype=torch.float32),
            )

        elif qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            # BF16/FP16 ckpt + load-time quantization. The buffer must hold the
            # source dtype so streamed tensors land losslessly; quantization
            # happens in process_weights_after_loading once every shard is in.
            self.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=params_dtype),
                requires_grad=False,
            )
            self.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=params_dtype),
                requires_grad=False,
            )
            # Scales registered now so they exist after the dtype swap below;
            # shapes match the corresponding already-quantized family.
            if qf == "fp8_per_tensor_online":
                self.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
                self.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))
            elif qf == "fp8_per_channel_online":
                self.register_buffer(
                    "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
                )
                self.register_buffer("w2_scale", torch.zeros(E, H, dtype=torch.float32))
            else:  # fp8_per_block_online
                BS = self._FP8_BLOCK_SIZE
                nb = (M_tp + BS - 1) // BS
                self._n_scale_blocks_per_proj = nb
                self.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, (H + BS - 1) // BS, dtype=torch.float32),
                )
                self.register_buffer(
                    "w2_scale",
                    torch.zeros(E, (H + BS - 1) // BS, nb, dtype=torch.float32),
                )

        else:
            # Unquantized (BF16/FP16) or unknown — subclass handles.
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
            base = name.split(".")[0]
            if (
                base in ("gate_up_proj", "down_proj")
                and hasattr(tensor, "dim")
                and tensor.dim() == 3
            ):
                self._load_stacked_experts(base, tensor)
                continue
            parts = name.split(".")
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
                continue

            if param_name == "weight":
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

    def _dispatch_weight(self, local_id: int, proj: str, tensor: torch.Tensor):
        """Route a weight tensor to the correct buffer position."""
        if proj == "gate_proj":
            self._copy_gate_or_up(local_id, tensor, gate=True)
            self._loaded_count += 1
        elif proj == "up_proj":
            self._copy_gate_or_up(local_id, tensor, gate=False)
            self._loaded_count += 1
        elif proj == "down_proj":
            self._copy_down(local_id, tensor)
            self._loaded_count += 1

    def _dispatch_scale(
        self, local_id: int, proj: str, param_name: str, tensor: torch.Tensor
    ):
        """Route a scale/meta tensor based on quantization family.

        Handles FP8 per-tensor, per-channel, per-block natively.
        Override in subclass for additional quantization types.
        """
        if self.quant_method is not None:
            self.quant_method.dispatch_scale(self, local_id, proj, param_name, tensor)
            return
        qf = self._quant_family

        # Online families allocate BF16 buffers and quantize at post-load. If
        # the ckpt actually ships per-projection weight_scale tensors, the FP8
        # values were *already* quantized — copying them into a BF16 buffer
        # without applying the ckpt's weight_scale produces silently corrupt
        # weights (output is garbled). Fail loudly so the user re-runs with
        # QUANTIZATION=FP8 (already-quanted path) instead of FP8_DYNAMIC_*
        # (online path).
        if param_name in ("weight_scale", "weight_scale_inv") and qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            raise RuntimeError(
                f"[BaseMoEExperts] Got {param_name!r} for proj={proj!r} "
                f"while in online quant family {qf!r}; this means the "
                f"checkpoint is pre-quantized FP8 but QUANTIZATION env "
                f"selected the online (BF16->FP8 at load) path. Either:\n"
                f"  - set QUANTIZATION=FP8 (already-quantized path), or\n"
                f"  - ensure the ckpt's config.json has a "
                f"`quantization_config` block so load_from_ckpt() can pick "
                f"the right is_quanted=True config automatically."
            )

        if param_name == "weight_scale" and qf == "fp8_per_tensor":
            scale_val = tensor.float().squeeze().item()
            if proj == "gate_proj":
                self._gate_scales[local_id] = scale_val
            elif proj == "up_proj":
                self._up_scales[local_id] = scale_val
            elif proj == "down_proj":
                self._down_scales[local_id] = scale_val

        elif param_name == "weight_scale" and qf == "fp8_per_channel":
            self._copy_per_channel_scale(local_id, proj, tensor)

        elif param_name == "weight_scale_inv" and qf == "fp8_per_block":
            self._copy_block_scale(local_id, proj, tensor)

    # ------------------------------------------------------------------ #
    #  FP8 scale copy helpers
    # ------------------------------------------------------------------ #

    def _copy_per_channel_scale(self, expert_id: int, proj: str, tensor: torch.Tensor):
        """Copy per-channel (per-row) FP8 scale for one expert projection.

        ckpt scale shape: [N] or [N, 1] where N is the full (un-TP-sliced)
        output dimension.  We slice to the local TP shard.
        """
        scale = tensor.float().squeeze()
        if proj == "gate_proj":
            start = self.tp_rank * self.moe_inter_tp
            self._gate_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, self.moe_inter_tp)
            )
        elif proj == "up_proj":
            start = self.tp_rank * self.moe_inter_tp
            self._up_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, self.moe_inter_tp)
            )
        elif proj == "down_proj":
            # down_proj output dim is hidden_size, not TP-sharded on rows.
            self._down_ch_scales.data[expert_id].copy_(scale)

    def _copy_block_scale(self, expert_id: int, proj: str, tensor: torch.Tensor):
        """Copy per-block (128x128) FP8 scale for one expert projection.

        ckpt scale shape: [ceil(N/BS), ceil(K/BS)].  We TP-slice along the
        N dimension for gate/up, and along K for down.

        w13_scale layout: [E, 2*nb, Hb] — first nb rows is up, second nb is gate.
        """
        BS = self._FP8_BLOCK_SIZE
        nb = self._n_scale_blocks_per_proj
        if proj in ("gate_proj", "up_proj"):
            start_block = (self.moe_expert_tp_rank * self.moe_inter_tp) // BS
            sliced = tensor.narrow(0, start_block, nb).contiguous()
            row_start = nb if proj == "gate_proj" else 0
            self.w13_scale.data[expert_id, row_start : row_start + nb].copy_(sliced)
        elif proj == "down_proj":
            start_block = (self.moe_expert_tp_rank * self.moe_inter_tp) // BS
            sliced = tensor.narrow(1, start_block, nb).contiguous()
            self.w2_scale.data[expert_id].copy_(sliced)

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

    # ------------------------------------------------------------------ #
    #  Post-load processing
    # ------------------------------------------------------------------ #

    def process_weights_after_loading(self):
        """Called after all weights are loaded.

        Handles FP8 scale fusion natively.  Override in subclass for
        additional quantization types; always call super() at the end.
        """
        if self.quant_method is not None:
            self.quant_method.process_weights_after_loading(self)
            self._maybe_build_fused_moe()
            return
        qf = self._quant_family
        if qf == "fp8_per_tensor":
            self._fuse_fp8_per_tensor_scales()
        elif qf == "fp8_per_channel":
            self._fuse_fp8_per_channel_scales()
        elif qf == "fp8_per_tensor_online":
            self._online_quantize_per_tensor()
        elif qf == "fp8_per_channel_online":
            self._online_quantize_per_channel()
        elif qf == "fp8_per_block_online":
            self._online_quantize_per_block()
        # fp8_per_block (already-quantized) needs no post-processing.

        if (
            self.layer_idx == 0
            and qf
            in (
                "fp8_per_tensor",
                "fp8_per_tensor_online",
            )
            and hasattr(self, "w13_scale")
        ):
            try:
                w13_abs = self.w13.data.float().abs()
                w2_abs = self.w2.data.float().abs()
                logger.info(
                    "[BaseMoEExperts] layer_idx=0 post-load FP8 stats: "
                    "qf=%r w13.dtype=%s w13.shape=%s "
                    "w13_scale[:4]=%s w13_scale.min=%.3e w13_scale.max=%.3e "
                    "w2_scale[:4]=%s "
                    "w13.abs_max=%.3e w13.abs_mean=%.3e "
                    "w2.abs_max=%.3e w2.abs_mean=%.3e",
                    qf,
                    self.w13.dtype,
                    tuple(self.w13.shape),
                    self.w13_scale.flatten()[:4].cpu().tolist(),
                    float(self.w13_scale.min().item()),
                    float(self.w13_scale.max().item()),
                    self.w2_scale.flatten()[:4].cpu().tolist(),
                    float(w13_abs.max().item()),
                    float(w13_abs.mean().item()),
                    float(w2_abs.max().item()),
                    float(w2_abs.mean().item()),
                )
            except Exception as exc:
                logger.warning("[BaseMoEExperts] post-load stats log failed: %s", exc)

        self._maybe_build_fused_moe()

    # ------------------------------------------------------------------ #
    #  FP8 scale fusion helpers
    # ------------------------------------------------------------------ #

    def _fuse_fp8_per_tensor_scales(self):
        """Fuse per-projection scales into a single w13_scale per expert.

        The python cutlass executor (fp8_grouped_gemm_ptpc) expects one fp32
        scalar per expert for w13 — gate and up must share a single scale.
        The ckpt provides separate per-tensor scales for gate and up, so we
        must merge them.

        Approach: dequantize each FP8 half back to fp32 in the original
        BF16 magnitude (fp8 * ckpt_scale), stack them, then re-run a single
        per-tensor quant over the [2*M_tp, H] block. This is bit-equivalent
        to "load ckpt as BF16 and run online per-tensor quant" — same path
        the dense online code uses — and avoids the precision loss of the
        previous max+rescale approach (which double-quantized the
        smaller-scale half through fp8 -> fp32 -> fp8).

        w13 layout: [E, 2*M_tp, H] — first half is up_proj, second half is gate_proj.
        w2 has only down_proj, so a single dequant + re-quant suffices.
        """
        from rtp_llm.models_py.quant_methods.fp8 import _resolve_per_tensor_quant

        per_tensor_quant = _resolve_per_tensor_quant()

        M_tp = self.moe_inter_tp
        device = self.w13.data.device
        new_w13 = torch.empty_like(self.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(self.w2.data, dtype=torch.float8_e4m3fn)

        for e in range(self.num_local_experts):
            up_s = float(self._up_scales[e].item())
            gate_s = float(self._gate_scales[e].item())
            down_s = float(self._down_scales[e].item())

            # Dequantize each half to fp32 with its own ckpt scale, restoring
            # the magnitude of the original BF16 weight.
            up_bf16 = self.w13.data[e, :M_tp].float() * up_s
            gate_bf16 = self.w13.data[e, M_tp:].float() * gate_s
            # Stack back as a single [2*M_tp, H] block in [up; gate] order.
            w13_bf16 = torch.cat([up_bf16, gate_bf16], dim=0).contiguous()

            qw13, sc13 = per_tensor_quant(w13_bf16)
            new_w13[e].copy_(qw13)
            self.w13_scale[e] = sc13.view(-1)[0]

            # w2: dequantize then re-quantize. Even though there's no
            # multi-projection merge, doing this keeps the path symmetric and
            # ensures w2 scale lands in the same [E] fp32 tensor we registered.
            w2_bf16 = self.w2.data[e].float() * down_s
            qw2, sc2 = per_tensor_quant(w2_bf16.contiguous())
            new_w2[e].copy_(qw2)
            self.w2_scale[e] = sc2.view(-1)[0]

        self.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        self.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

        del self._gate_scales
        del self._up_scales
        del self._down_scales

    def _fuse_fp8_per_channel_scales(self):
        """Concatenate per-channel scales into fused w13_scale / w2_scale.

        w13_scale layout: [E, 2*M_tp] — first M_tp is up_proj, second is gate_proj.
        This matches w13 weight layout: [up, gate].
        w2_scale layout: [E, H] — down_proj per-channel scale.
        """
        for e in range(self.num_local_experts):
            self.w13_scale.data[e, : self.moe_inter_tp] = self._up_ch_scales[e]
            self.w13_scale.data[e, self.moe_inter_tp :] = self._gate_ch_scales[e]
            self.w2_scale.data[e] = self._down_ch_scales[e]

        del self._gate_ch_scales
        del self._up_ch_scales
        del self._down_ch_scales

    # ------------------------------------------------------------------ #
    #  Online (BF16/FP16 ckpt -> FP8) quantization helpers
    # ------------------------------------------------------------------ #

    _FP8_E4M3_MAX: float = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    _FP8_MIN_SCALE: float = 1.0 / (448.0 * 512.0)  # matches static_fp8_quant_weight

    def _online_quantize_per_tensor(self):
        """BF16 -> FP8 per-expert per-tensor quantization.

        Mirrors the old loader's `LoadQuantDynamicPerTensorFp8Weight` MoE
        path (dynamic_fp8_quant_weight.py): one per-tensor scale per expert
        per stacked-projection block. The whole [2*M_tp, H] up||gate block
        shares a single fp32 scale = max(|w13[e]|) / fp8_max — same as the
        old loader's `quantize_weight_to_fp8(kernel_tensor[i, :, :])`. This
        is what the cutlass executor expects (w13_scale shape [E]).
        """
        from rtp_llm.models_py.quant_methods.fp8 import _resolve_per_tensor_quant

        per_tensor_quant = _resolve_per_tensor_quant()

        E = self.num_local_experts
        device = self.w13.data.device
        new_w13 = torch.empty_like(self.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(self.w2.data, dtype=torch.float8_e4m3fn)

        for e in range(E):
            # w13[e] is [2*M_tp, H] BF16/FP16 — already 2D, kernel-friendly.
            qw, sc = per_tensor_quant(self.w13.data[e].contiguous())
            new_w13[e].copy_(qw)
            self.w13_scale[e] = sc.view(-1)[0]

            qw, sc = per_tensor_quant(self.w2.data[e].contiguous())
            new_w2[e].copy_(qw)
            self.w2_scale[e] = sc.view(-1)[0]

        # Rebind as fp8 Parameters; the original BF16 buffer is freed when the
        # Parameter slot is overwritten.
        self.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        self.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_quantize_per_channel(self):
        """BF16 -> FP8 per-expert per-channel (per-output-row) quantization."""
        E = self.num_local_experts
        device = self.w13.data.device
        new_w13 = torch.empty_like(self.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(self.w2.data, dtype=torch.float8_e4m3fn)
        fp8_max = self._FP8_E4M3_MAX

        for e in range(E):
            # w13: [2*M_tp, H], scale per row.
            w13_e = self.w13.data[e].float()
            row_max = w13_e.abs().amax(dim=1)  # [2*M_tp]
            row_scale = (row_max / fp8_max).clamp_min(self._FP8_MIN_SCALE)
            new_w13[e] = (w13_e / row_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            self.w13_scale.data[e] = row_scale

            # w2: [H, M_tp], scale per row (output channel = hidden dim).
            w2_e = self.w2.data[e].float()
            row_max = w2_e.abs().amax(dim=1)  # [H]
            row_scale = (row_max / fp8_max).clamp_min(self._FP8_MIN_SCALE)
            new_w2[e] = (w2_e / row_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            self.w2_scale.data[e] = row_scale

        self.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        self.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_quantize_per_block(self):
        """BF16 -> FP8 per-expert 128x128 block quantization.

        Layout of block scales matches the already-quantized path:
          w13_scale: [E, 2*nb_M, nb_H] — block row i corresponds to weight
                     rows [i*BS : (i+1)*BS].
          w2_scale:  [E, nb_H, nb_M].
        """
        E = self.num_local_experts
        BS = self._FP8_BLOCK_SIZE
        device = self.w13.data.device
        new_w13 = torch.empty_like(self.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(self.w2.data, dtype=torch.float8_e4m3fn)
        fp8_max = self._FP8_E4M3_MAX

        rows13, cols13 = self.w13.data.shape[1], self.w13.data.shape[2]
        rows2, cols2 = self.w2.data.shape[1], self.w2.data.shape[2]
        nb_r13 = (rows13 + BS - 1) // BS
        nb_c13 = (cols13 + BS - 1) // BS
        nb_r2 = (rows2 + BS - 1) // BS
        nb_c2 = (cols2 + BS - 1) // BS

        for e in range(E):
            for bi in range(nb_r13):
                r0, r1 = bi * BS, min((bi + 1) * BS, rows13)
                for bj in range(nb_c13):
                    c0, c1 = bj * BS, min((bj + 1) * BS, cols13)
                    block = self.w13.data[e, r0:r1, c0:c1].float()
                    scale = max(self._FP8_MIN_SCALE, block.abs().max().item() / fp8_max)
                    new_w13[e, r0:r1, c0:c1] = (block / scale).to(torch.float8_e4m3fn)
                    self.w13_scale.data[e, bi, bj] = scale
            for bi in range(nb_r2):
                r0, r1 = bi * BS, min((bi + 1) * BS, rows2)
                for bj in range(nb_c2):
                    c0, c1 = bj * BS, min((bj + 1) * BS, cols2)
                    block = self.w2.data[e, r0:r1, c0:c1].float()
                    scale = max(self._FP8_MIN_SCALE, block.abs().max().item() / fp8_max)
                    new_w2[e, r0:r1, c0:c1] = (block / scale).to(torch.float8_e4m3fn)
                    self.w2_scale.data[e, bi, bj] = scale

        self.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        self.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _build_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Build the weight dict for FusedMoeFactory.

        Automatically includes FP8 scale tensors when using FP8 quantization.
        Override in subclass to add additional tensors (e.g. FP4 scale_2).
        """
        weights_dict: Dict[str, torch.Tensor] = {
            W.moe_w1: self.w13.data,
            W.moe_w2: self.w2.data,
        }
        if self.quant_method is not None:
            self.quant_method.add_weight_tensors(self, weights_dict)
        elif self._quant_family in (
            "fp8_per_tensor",
            "fp8_per_channel",
            "fp8_per_block",
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            weights_dict[W.moe_s1] = self.w13_scale
            weights_dict[W.moe_s2] = self.w2_scale
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
            quant_config=getattr(self._model_config, "quant_config", None),
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
