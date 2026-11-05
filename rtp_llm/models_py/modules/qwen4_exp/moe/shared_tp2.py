"""Static Qwen4 BF16 shared-FFN TP2 partition; reuse existing GEMMs/reducer.

Opt-in is constructor-only and restart-only. The caller must establish identical
rank configuration/row order before enabling it. This helper does not negotiate
rank policy, mutate source weights, or add any forward/capture collective.
"""

import os

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear

_ENV = "RTP_QWEN4_BF16_SHARED_TP2"


class _Qwen4SharedTp2Up(CudaF16Linear):
    """Restart-only local NK pack, executed by the installed TinyGEMM API.

    Construction is outside capture. Eager warmup must load/compile the existing
    FlashInfer kernel before any Graph capture; capture support is a GPU gate.
    A process must retain its one selected CUDA device and this exact weight
    owner. Unsupported forward inputs raise; they never select cuBLAS silently.
    """

    def __init__(self, weight: torch.Tensor):
        if not (
            weight.is_cuda
            and weight.dtype == torch.bfloat16
            and not weight.requires_grad
            and tuple(weight.shape) == (640, 2560)
            and tuple(weight.stride()) == (2560, 1)
            and weight.is_contiguous()
            and weight.storage_offset() == 0
            and weight.untyped_storage().nbytes() == 3276800
            and weight.data_ptr() % 16 == 0
            and weight.device.index == torch.cuda.current_device()
            and torch.cuda.get_device_capability(weight.device) == (12, 0)
            and not torch.cuda.is_current_stream_capturing()
        ):
            raise ValueError(
                "Qwen4 shared TP2 TinyGEMM requires the qualified SM120 NK pack"
            )
        # Parent accepts KN. The two transpose views allocate no new storage;
        # publish the exact owned NK Tensor rather than another Tensor view.
        super().__init__(weight.t())
        self.weight = weight
        self._packed_weight = weight
        self._device = weight.device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not (
            weight is self._packed_weight
            and self.bias is None
            and weight.device == self._device
            and weight.dtype == torch.bfloat16
            and not weight.requires_grad
            and tuple(weight.shape) == (640, 2560)
            and tuple(weight.stride()) == (2560, 1)
            and weight.storage_offset() == 0
            and weight.untyped_storage().nbytes() == 3276800
        ):
            raise RuntimeError(
                "Qwen4 shared TP2 TinyGEMM weight changed; rebuild the model/Graphs"
            )
        if not (
            isinstance(input, torch.Tensor)
            and input.is_cuda
            and input.device == self._device
            and input.device.index == torch.cuda.current_device()
            and input.dtype == torch.bfloat16
            and not input.requires_grad
            and input.ndim == 2
            and input.shape[0] > 0
            and input.shape[1] == 2560
            and tuple(input.stride()) == (2560, 1)
            and input.is_contiguous()
            and input.data_ptr() % 16 == 0
        ):
            raise ValueError(
                "Qwen4 shared TP2 TinyGEMM requires nonempty CUDA BF16 contiguous [M,2560]"
            )
        from flashinfer.gemm import tinygemm_bf16

        output = torch.empty(
            (input.shape[0], 640), dtype=input.dtype, device=input.device
        )
        tinygemm_bf16(input, weight, output, bias=None, use_pdl=False)
        return output


def _eligible(layer):
    from rtp_llm.models_py.modules.base import FusedSiluAndMul, SigmoidGateScaleAdd
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
        CutlassNvfp4Executor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )
    from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
    from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
    from rtp_llm.ops import ActivationType

    if type(layer) is not Qwen4MoeLayer:
        return False
    pc, config = layer.parallelism_config, layer.config
    shared, fused = layer.shared_expert, layer.fused_moe
    if not (
        str(getattr(pc, "role_type", "")).rsplit(".", 1)[-1] == "DECODE"
        and not getattr(config, "is_mtp", True)
        and getattr(config, "_qwen4_full_fp8_nvfp4", False)
        and getattr(config, "_qwen4_dense_storage", None) == "source_bf16"
        and config.hidden_size == 2560
        and config.inter_size == 640
        and pc.tp_size == pc.ep_size == pc.world_size == pc.local_world_size == 2
        and pc.dp_size == pc.pp_size == 1
        and pc.ffn_sp_size == 2
        and pc.get_attn_tp_size() == 2
        and pc.get_ffn_tp_size() == layer.ffn_tp_size == 1
        and pc.get_ffn_tp_rank() == 0
        and pc.tp_rank in (0, 1)
        and pc.tp_rank == pc.ep_rank == pc.world_rank
        and not config.eplb_config.enable_eplb()
        and layer.fake_balance_expert is None
        and not layer.use_ep_shared_allreduce
        and not layer.use_unified_tp_allreduce
        and getattr(layer, "_shared_stream", None) is None
        and getattr(layer, "_reused_shared_stream", None) is None
        and type(shared) is DenseMLP
        and shared.parallelism_config is pc
        and shared.is_gated
        and shared.activation_type == ActivationType.Swiglu
        and type(shared.act_fn) is FusedSiluAndMul
        and type(layer.sigmoid_gate_scale_add) is SigmoidGateScaleAdd
        and type(fused) is FusedMoe
        and not fused.includes_shared_expert
        and type(fused.router) is PureTpRouterNvfp4Cutlass
        and fused.router.supports_skip_tp_allreduce
        and fused.router.tp_collective_size == 2
        and not fused.router.do_recompute_topk
        and fused.router.ep_size == 2
        and fused.router.ep_rank == pc.ep_rank
        and type(fused.fused_experts) is CutlassNvfp4Executor
        and fused.fused_experts.tp_size == 1
        and fused.fused_experts.tp_rank == 0
        and fused.fused_experts.ep_size == 2
        and fused.fused_experts.ep_rank == pc.ep_rank
        and fused.expert_num == fused.router.expert_num == config.expert_num
        and config.expert_num > 0
        and config.expert_num % 2 == 0
        and fused.fused_experts.w1.shape == (config.expert_num // 2, 1280, 160)
        and fused.fused_experts.w2.shape == (config.expert_num // 2, 2560, 40)
        and fused.fused_experts.w1.dtype == fused.fused_experts.w2.dtype == torch.int64
        and not getattr(layer, "_qwen4_shared_tp2", False)
        and torch.version.hip is None
    ):
        return False
    projections = (shared.up_proj, shared.down_proj, layer.shared_expert_gate)
    if not all(type(proj) is CudaF16Linear for proj in projections):
        return False
    device = shared.up_proj.weight.device
    for projection, shape, stride, nbytes in zip(
        projections,
        ((1280, 2560), (2560, 640), (1, 2560)),
        ((1, 1280), (1, 2560), (2560, 1)),
        (6553600, 3276800, 5120),
    ):
        weight = projection.weight
        if not (
            projection.bias is None
            and weight.is_cuda
            and weight.device == device
            and weight.dtype == torch.bfloat16
            and not weight.requires_grad
            and tuple(weight.shape) == shape
            and tuple(weight.stride()) == stride
            and weight.storage_offset() == 0
            and weight.untyped_storage().nbytes() == nbytes
            and "forward" not in projection.__dict__
        ):
            return False
    return torch.cuda.get_device_capability(device) == (12, 0) and not (
        torch.cuda.is_current_stream_capturing()
    )


def maybe_prepare_bf16_shared_tp2(layer):
    """Return False without mutation for default/off-domain construction.

    Unsupported local metadata is a local fallback, NOT a cross-rank agreement.
    Enable only after the two-rank qualification gate. Packing/allocation errors
    propagate and fail construction; an already running model cannot hot-toggle.
    Original source updates are unsupported: rebuild weights/modules/Graphs.
    """
    if os.getenv(_ENV, "0") != "1":
        return False
    if not _eligible(layer):
        return False

    shared = layer.shared_expert
    begin = layer.parallelism_config.tp_rank * 320
    end = begin + 320
    # Source NK has gate640 then up640. Pack their local rows in NK order.
    # Reuse installed TinyGEMM with this same owner; no global math flags change.
    with torch.no_grad():
        local_up = torch.cat(
            (
                shared.up_proj.weight[begin:end],
                shared.up_proj.weight[640 + begin : 640 + end],
            ),
            dim=0,
        ).contiguous()  # NK[640,2560], stride[2560,1]
    local_down = shared.down_proj.weight[:, begin:end]  # original owner/view
    # Complete the only copy before publishing a consistent local state.
    torch.cuda.current_stream(local_up.device).synchronize()
    up_projection = _Qwen4SharedTp2Up(local_up)
    local_up._rtp_requires_restart_for_update = True
    local_down._rtp_requires_restart_for_update = True
    shared.up_proj = up_projection
    shared.down_proj.weight = local_down
    layer.use_unified_tp_allreduce = True
    layer._qwen4_shared_tp2 = True
    return True
