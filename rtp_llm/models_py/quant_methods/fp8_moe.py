"""FP8 MoE 量化方法（vLLM 风格,从 BaseMoEExperts 内置 fp8 逻辑迁出）。

覆盖全部 fp8 子族(已量化:per_tensor/per_channel/per_block;在线 BF16->FP8:三者的
*_online)。逻辑与 `BaseMoEExperts` 内置 fp8 路径**逐字一致**(只把 ``self.`` 换成
``layer.``),按 ``layer._quant_family`` 分支:
  - ``create_weights``  ← `_init_buffers` 的 fp8/online 分支
  - ``dispatch_scale``  ← `_dispatch_scale` + `_copy_per_channel_scale` + `_copy_block_scale`
  - ``process_weights_after_loading`` ← `_fuse_fp8_*` / `_online_quantize_*`
  - ``add_weight_tensors`` ← `_build_weights_dict` 的 fp8 scale 注入
前向仍由 `BaseMoEExperts.forward` 经 fused_moe 完成。

验证状态:per_tensor 已端到端验证;per_channel/per_block/online 为忠实搬移,各需对应
ckpt 验证。回退某子族:从下面 `@register_moe_quant_method` 移除对应 key → 走内置。
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

# 与 BaseMoEExperts 一致的 fp8 常量（也可经 layer 取，这里就近定义保持自洽）。
_FP8_E4M3_MAX: float = 448.0
_FP8_MIN_SCALE: float = 1.0 / (448.0 * 512.0)


def _runtime_fp8_dtype() -> torch.dtype:
    try:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
            get_rocm_fp8_dtype,
        )

        return get_rocm_fp8_dtype()
    except Exception:
        return torch.float8_e4m3fn


@register_moe_quant_method(
    # per_tensor 子族
    "fp8",
    "FP8_PER_TENSOR_COMPRESSED",
    "FP8_DYNAMIC_PER_TENSOR",
    # per_block 子族
    "FP8_PER_BLOCK",
    "fp8_block",
    # per_channel 子族
    "FP8_PER_CHANNEL_COMPRESSED",
    "fp8_per_channel",
    "FP8_PER_CHANNEL_QUARK",
    # 在线 BF16->FP8 子族
    "fp8_online",
    "fp8_block_online",
    "fp8_per_channel_online",
)
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def _weight_block_size(self, layer) -> List[int]:
        return list(
            getattr(
                getattr(layer, "_quant_config", None),
                "weight_block_size",
                getattr(self.quant_config, "weight_block_size", [128, 128]),
            )
        )

    def _requant_block_to_runtime_fp8(
        self, weight: torch.Tensor, scale: torch.Tensor, block_size: List[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runtime_dtype = _runtime_fp8_dtype()
        if weight.dtype == runtime_dtype:
            return weight.contiguous(), scale.contiguous()

        block_n, block_k = block_size
        e, out_dim, in_dim = weight.shape
        if out_dim % block_n != 0 or in_dim % block_k != 0:
            raise ValueError(
                f"MoE FP8 block requant requires divisible dims, got "
                f"weight={tuple(weight.shape)} block={block_size}"
            )

        out_blocks = out_dim // block_n
        in_blocks = in_dim // block_k
        deq = weight.float().reshape(e, out_blocks, block_n, in_blocks, block_k)
        deq = deq * scale.float().reshape(e, out_blocks, 1, in_blocks, 1)
        fp8_max = float(torch.finfo(runtime_dtype).max)
        new_scale = (
            deq.abs().amax(dim=(2, 4), keepdim=True).clamp_min(_FP8_MIN_SCALE)
            / fp8_max
        )
        requant = (deq / new_scale).to(runtime_dtype).reshape(e, out_dim, in_dim)
        new_scale = new_scale.squeeze(2).squeeze(-1).to(torch.float32)
        return requant.contiguous(), new_scale.contiguous()

    # ------------------------------------------------------------------ #
    #  create_weights ← _init_buffers 的 fp8/online 分支
    # ------------------------------------------------------------------ #
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        E, M_tp, H = num_experts, intermediate_size, hidden_size
        qf = layer._quant_family
        dt = torch.float8_e4m3fn
        BS = layer._FP8_BLOCK_SIZE
        block_n, block_k = self._weight_block_size(layer)

        if qf == "fp8_per_tensor":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer("_gate_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_up_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_down_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))

        elif qf == "fp8_per_channel":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer(
                "_gate_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_up_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_down_ch_scales", torch.zeros(E, H, dtype=torch.float32)
            )
            layer.register_buffer(
                "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
            )
            layer.register_buffer("w2_scale", torch.zeros(E, H, dtype=torch.float32))

        elif qf == "fp8_per_block":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            nb = self._ceil_div(M_tp, block_n)
            hb = self._ceil_div(H, block_k)
            h_nb = self._ceil_div(H, block_n)
            k_nb = self._ceil_div(M_tp, block_k)
            layer._n_scale_blocks_per_proj = nb
            layer._k_scale_blocks_per_proj = k_nb
            layer._fp8_moe_weight_block_size = [block_n, block_k]
            if [block_n, block_k] == [BS, BS]:
                layer.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, hb, dtype=torch.float32),
                )
                layer.register_buffer(
                    "w2_scale",
                    torch.zeros(E, h_nb, k_nb, dtype=torch.float32),
                )
            else:
                # Source MXFP8 scales can be much larger than the execution
                # format. Keep them as plain CPU tensors so model.to(cuda)
                # does not move all source scales before post-load requant.
                layer.w13_scale = torch.zeros(E, 2 * nb, hb, dtype=torch.float32)
                layer.w2_scale = torch.zeros(E, h_nb, k_nb, dtype=torch.float32)

        elif qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            # 在线量化:buffer 先存源 dtype，process_weights 时再量化到 fp8。
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=params_dtype), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=params_dtype), requires_grad=False
            )
            if qf == "fp8_per_tensor_online":
                layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
                layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))
            elif qf == "fp8_per_channel_online":
                layer.register_buffer(
                    "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
                )
                layer.register_buffer(
                    "w2_scale", torch.zeros(E, H, dtype=torch.float32)
                )
            else:  # fp8_per_block_online
                nb = self._ceil_div(M_tp, block_n)
                hb = self._ceil_div(H, block_k)
                h_nb = self._ceil_div(H, block_n)
                k_nb = self._ceil_div(M_tp, block_k)
                layer._n_scale_blocks_per_proj = nb
                layer._k_scale_blocks_per_proj = k_nb
                layer._fp8_moe_weight_block_size = [block_n, block_k]
                layer.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, hb, dtype=torch.float32),
                )
                layer.register_buffer(
                    "w2_scale",
                    torch.zeros(E, h_nb, k_nb, dtype=torch.float32),
                )
        else:
            raise ValueError(f"Fp8MoEMethod 不支持的 quant_family: {qf!r}")

    # ------------------------------------------------------------------ #
    #  dispatch_scale ← _dispatch_scale + 两个 copy 助手
    # ------------------------------------------------------------------ #
    def dispatch_scale(self, layer, local_id, proj, param_name, tensor):
        qf = layer._quant_family

        # online 家族不该收到已量化 ckpt 的 weight_scale，否则是 ckpt 与 QUANTIZATION 不匹配。
        if param_name in ("weight_scale", "weight_scale_inv") and qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            raise RuntimeError(
                f"[Fp8MoEMethod] Got {param_name!r} for proj={proj!r} while in "
                f"online quant family {qf!r}; ckpt 已是预量化 FP8 但 QUANTIZATION 选了"
                f"在线路径。请改用 QUANTIZATION=FP8（已量化路径），或确保 ckpt 的"
                f" config.json 带 quantization_config 让 load_from_ckpt 自动选对。"
            )

        if param_name == "weight_scale" and qf == "fp8_per_tensor":
            scale_val = tensor.float().squeeze().item()
            if proj == "gate_proj":
                layer._gate_scales[local_id] = scale_val
            elif proj == "up_proj":
                layer._up_scales[local_id] = scale_val
            elif proj == "down_proj":
                layer._down_scales[local_id] = scale_val

        elif param_name == "weight_scale" and qf == "fp8_per_channel":
            self._copy_per_channel_scale(layer, local_id, proj, tensor)

        elif param_name == "weight_scale_inv" and qf == "fp8_per_block":
            self._copy_block_scale(layer, local_id, proj, tensor)

    def _copy_per_channel_scale(self, layer, expert_id, proj, tensor):
        scale = tensor.float().squeeze()
        if proj == "gate_proj":
            start = layer.moe_expert_tp_rank * layer.moe_inter_tp
            layer._gate_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "up_proj":
            start = layer.moe_expert_tp_rank * layer.moe_inter_tp
            layer._up_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "down_proj":
            layer._down_ch_scales.data[expert_id].copy_(scale)

    def _copy_block_scale(self, layer, expert_id, proj, tensor):
        block_n, block_k = getattr(
            layer, "_fp8_moe_weight_block_size", [layer._FP8_BLOCK_SIZE] * 2
        )
        nb = layer._n_scale_blocks_per_proj
        if proj in ("gate_proj", "up_proj"):
            start_block = (layer.moe_expert_tp_rank * layer.moe_inter_tp) // block_n
            sliced = tensor.narrow(0, start_block, nb).contiguous()
            row_start = nb if proj == "gate_proj" else 0
            layer.w13_scale.data[expert_id, row_start : row_start + nb].copy_(sliced)
        elif proj == "down_proj":
            start_block = (layer.moe_expert_tp_rank * layer.moe_inter_tp) // block_k
            k_nb = getattr(layer, "_k_scale_blocks_per_proj", nb)
            sliced = tensor.narrow(1, start_block, k_nb).contiguous()
            layer.w2_scale.data[expert_id].copy_(sliced)

    # ------------------------------------------------------------------ #
    #  process_weights_after_loading ← _fuse_* / _online_quantize_*
    # ------------------------------------------------------------------ #
    def process_weights_after_loading(self, layer):
        qf = layer._quant_family
        if qf == "fp8_per_tensor":
            self._fuse_per_tensor(layer)
        elif qf == "fp8_per_channel":
            self._fuse_per_channel(layer)
        elif qf == "fp8_per_tensor_online":
            self._online_per_tensor(layer)
        elif qf == "fp8_per_channel_online":
            self._online_per_channel(layer)
        elif qf == "fp8_per_block_online":
            self._online_per_block(layer)
        elif qf == "fp8_per_block":
            self._requant_per_block_if_needed(layer)

    def _fuse_per_tensor(self, layer):
        M_tp = layer.moe_inter_tp

        for e in range(layer.num_local_experts):
            up_s = float(layer._up_scales[e].item())
            gate_s = float(layer._gate_scales[e].item())
            down_s = float(layer._down_scales[e].item())
            max_s = max(up_s, gate_s)

            # Keep old-loader compatibility with StaticFp8QuantWeight._postprocess:
            # its shard loop never advances `start`, so both conditional rescale
            # attempts operate on the first half of moe_w1.
            for shard_s in (up_s, gate_s):
                if shard_s != max_s:
                    half = layer.w13.data[e, :M_tp].to(torch.float16) * shard_s
                    layer.w13.data[e, :M_tp] = (half / max_s).to(_runtime_fp8_dtype())
            layer.w13_scale[e] = max_s
            layer.w2_scale[e] = down_s

        del layer._gate_scales
        del layer._up_scales
        del layer._down_scales

    def _fuse_per_channel(self, layer):
        for e in range(layer.num_local_experts):
            layer.w13_scale.data[e, : layer.moe_inter_tp] = layer._up_ch_scales[e]
            layer.w13_scale.data[e, layer.moe_inter_tp :] = layer._gate_ch_scales[e]
            layer.w2_scale.data[e] = layer._down_ch_scales[e]
        del layer._gate_ch_scales
        del layer._up_ch_scales
        del layer._down_ch_scales

    def _online_per_tensor(self, layer):
        from rtp_llm.models_py.quant_methods.fp8 import (
            _resolve_per_tensor_quant,
            cpu_per_tensor_quant_like_legacy,
        )

        per_tensor_quant = _resolve_per_tensor_quant()
        E = layer.num_local_experts
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=_runtime_fp8_dtype())
        new_w2 = torch.empty_like(layer.w2.data, dtype=_runtime_fp8_dtype())
        force_cpu_quant = bool(
            getattr(layer, "_new_loader_force_cpu_load_weights", False)
        )

        for e in range(E):
            quant = (
                cpu_per_tensor_quant_like_legacy
                if force_cpu_quant
                else per_tensor_quant
            )
            qw, sc = quant(layer.w13.data[e].contiguous())
            new_w13[e].copy_(qw)
            layer.w13_scale[e] = sc.view(-1)[0]

            qw, sc = quant(layer.w2.data[e].contiguous())
            new_w2[e].copy_(qw)
            layer.w2_scale[e] = sc.view(-1)[0]

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_channel(self, layer):
        E = layer.num_local_experts
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=_runtime_fp8_dtype())
        new_w2 = torch.empty_like(layer.w2.data, dtype=_runtime_fp8_dtype())
        fp8_max = _FP8_E4M3_MAX

        for e in range(E):
            w13_e = layer.w13.data[e].float()
            row_max = w13_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(_FP8_MIN_SCALE)
            new_w13[e] = (w13_e / row_scale.unsqueeze(1)).to(_runtime_fp8_dtype())
            layer.w13_scale.data[e] = row_scale

            w2_e = layer.w2.data[e].float()
            row_max = w2_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(_FP8_MIN_SCALE)
            new_w2[e] = (w2_e / row_scale.unsqueeze(1)).to(_runtime_fp8_dtype())
            layer.w2_scale.data[e] = row_scale

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_block(self, layer):
        BS = layer._FP8_BLOCK_SIZE
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            per_block_cast_to_fp8 as legacy_per_block_cast_to_fp8,
        )

        # Use the same vectorized loader-side quantizer as linear online
        # fp8_block. The old Python block loop synchronized once per 128x128
        # block via .item(), making large MoE checkpoints exceed smoke timeout.
        new_w13, w13_scale = legacy_per_block_cast_to_fp8(
            layer.w13.data.contiguous(), BS
        )
        new_w2, w2_scale = legacy_per_block_cast_to_fp8(layer.w2.data.contiguous(), BS)

        layer.w13 = nn.Parameter(new_w13.contiguous(), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.contiguous(), requires_grad=False)
        layer.w13_scale.data.copy_(w13_scale.contiguous())
        layer.w2_scale.data.copy_(w2_scale.contiguous())

    def _requant_per_block_if_needed(self, layer):
        block_size = getattr(
            layer, "_fp8_moe_weight_block_size", [layer._FP8_BLOCK_SIZE] * 2
        )
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )

        if list(block_size) == [layer._FP8_BLOCK_SIZE, layer._FP8_BLOCK_SIZE]:
            runtime_dtype = _runtime_fp8_dtype()
            if layer.w13.dtype != runtime_dtype or layer.w2.dtype != runtime_dtype:
                layer.w13.data, layer.w13_scale = self._requant_block_to_runtime_fp8(
                    layer.w13.data.contiguous(),
                    layer.w13_scale.to(layer.w13.device, non_blocking=True).contiguous(),
                    list(block_size),
                )
                layer.w2.data, layer.w2_scale = self._requant_block_to_runtime_fp8(
                    layer.w2.data.contiguous(),
                    layer.w2_scale.to(layer.w2.device, non_blocking=True).contiguous(),
                    list(block_size),
                )
                logger.info(
                    "[Fp8MoEMethod] requantized MoE fp8 block weights to runtime dtype: "
                    "dtype=%s w13=%s w13_scale=%s w2=%s w2_scale=%s",
                    runtime_dtype,
                    tuple(layer.w13.shape),
                    tuple(layer.w13_scale.shape),
                    tuple(layer.w2.shape),
                    tuple(layer.w2_scale.shape),
                )

            if not is_deep_gemm_e8m0_used():
                return

            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                requant_weight_ue8m0,
            )

            layer.w13.data, layer.w13_scale = requant_weight_ue8m0(
                layer.w13.data.contiguous(),
                layer.w13_scale.to(layer.w13.device, non_blocking=True).contiguous(),
            )
            layer.w2.data, layer.w2_scale = requant_weight_ue8m0(
                layer.w2.data.contiguous(),
                layer.w2_scale.to(layer.w2.device, non_blocking=True).contiguous(),
            )
            logger.info(
                "[Fp8MoEMethod] requantized MoE fp8 block scales to UE8M0: "
                "w13=%s w13_scale=%s/%s w2=%s w2_scale=%s/%s",
                tuple(layer.w13.shape),
                layer.w13_scale.dtype,
                tuple(layer.w13_scale.shape),
                tuple(layer.w2.shape),
                layer.w2_scale.dtype,
                tuple(layer.w2_scale.shape),
            )
            return

        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
            block_quant_dequant,
            per_block_cast_to_fp8,
        )

        def requant_weight_fp8_block_float(weight, weight_scale):
            weight_dequant = block_quant_dequant(
                weight,
                weight_scale,
                list(block_size),
                torch.bfloat16,
            )
            return per_block_cast_to_fp8(weight_dequant, use_ue8m0=False)

        src_w13_scale = layer.w13_scale
        src_w2_scale = layer.w2_scale
        new_w13_scale = []
        new_w2_scale = []
        device = layer.w13.device
        for e in range(layer.num_local_experts):
            w, s = requant_weight_fp8_block_float(
                layer.w13.data[e].contiguous(),
                src_w13_scale[e].to(device, non_blocking=True).contiguous(),
            )
            layer.w13.data[e].copy_(w)
            new_w13_scale.append(s)
            del w
            w, s = requant_weight_fp8_block_float(
                layer.w2.data[e].contiguous(),
                src_w2_scale[e].to(device, non_blocking=True).contiguous(),
            )
            layer.w2.data[e].copy_(w)
            new_w2_scale.append(s)
            del w

        layer.w13_scale = torch.stack(new_w13_scale, dim=0).contiguous()
        layer.w2_scale = torch.stack(new_w2_scale, dim=0).contiguous()
        logger.info(
            "[Fp8MoEMethod] requantized MoE fp8 block scales from %s to [128, 128]: "
            "w13=%s w13_scale=%s w2=%s w2_scale=%s",
            block_size,
            tuple(layer.w13.shape),
            tuple(layer.w13_scale.shape),
            tuple(layer.w2.shape),
            tuple(layer.w2_scale.shape),
        )

    # ------------------------------------------------------------------ #
    #  add_weight_tensors ← _build_weights_dict 的 fp8 scale 注入
    # ------------------------------------------------------------------ #
    def add_weight_tensors(self, layer, weights_dict: Dict[str, Any]) -> None:
        weights_dict[W.moe_s1] = layer.w13_scale
        weights_dict[W.moe_s2] = layer.w2_scale
