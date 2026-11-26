"""Packed affine integer projection for CUDA and ROCm.

Decode only the current tile in the shared Triton kernel. Pinned ROCm Aiter's
A16W4 MoE is MXFP4 with E8M0 scales, not this asymmetric integer format.
"""

import importlib
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupwiseLayout:
    method: str
    bits: int
    group_size: int
    input_size: int
    output_size: int
    gptq_v2: bool = False

    @classmethod
    def parse(cls, value):
        if isinstance(value, dict):
            value = {"gptq_v2": False, **value}
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError(
                "Groupwise runtime requires explicit local parameter layout"
            )
        result = cls(**value)
        if not isinstance(result.gptq_v2, bool) or (
            result.gptq_v2 and result.method != "gptq"
        ):
            raise ValueError("GPTQ v2 zero points require a GPTQ checkpoint")
        if result.method not in ("awq", "gptq"):
            raise ValueError("Groupwise runtime method must be awq or gptq")
        for name in ("bits", "group_size", "input_size", "output_size"):
            if type(getattr(result, name)) is not int:
                raise TypeError(f"Groupwise {name} must be an integer")
        if result.bits not in (4, 8) or min(result.input_size, result.output_size) <= 0:
            raise ValueError("Groupwise runtime requires positive shapes and 4/8 bits")
        group = result.input_size if result.group_size == -1 else result.group_size
        if group <= 0 or result.input_size % group:
            raise ValueError("Groupwise local K must be divisible by group_size")
        pack = 32 // result.bits
        if result.output_size % pack or (
            result.method == "gptq" and result.input_size % pack
        ):
            raise ValueError("Groupwise packed shape is not word aligned")
        if result.bits == 4 and result.input_size % 2:
            raise ValueError("Native W4A16 requires pairs of input channels")
        if result.method == "awq" and result.output_size % 8:
            raise ValueError("AWQ output width must preserve its eight-channel order")
        return result

    @property
    def group(self):
        return self.input_size if self.group_size == -1 else self.group_size


@dataclass(frozen=True)
class GroupwiseRuntimeState:
    weight: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    layout: GroupwiseLayout
    runtime_bits: int
    promotion_reason: str | None

    @property
    def metadata(self):
        return dict(
            layout=(
                "cuda_wna16_v1"
                if torch.version.hip is None
                else "rocm_wna16_v1"
            ),
            method=self.layout.method,
            source_bits=self.layout.bits,
            runtime_bits=self.runtime_bits,
            promotion_reason=self.promotion_reason,
            group_size=self.layout.group,
            input_size=self.layout.input_size,
            output_size=self.layout.output_size,
        )


def _unpack_last(words, bits):
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=words.device)
    return ((words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(
        *words.shape[:-1], -1
    )


def _undo_awq_order(values):
    return (
        values.reshape(*values.shape[:-1], -1, 2, 4)
        .transpose(-1, -2)
        .reshape(values.shape)
    )


@torch.no_grad()
def prepare_groupwise_state(qweight, qzeros, scales, layout):
    """Pure integer repacking, usable on CPU for exact checkpoint regressions.

    GPTQ v1 can have an effective zero point of 16 (stored nibble 15 + 1). A
    four-bit zero-point ABI cannot represent it. Select the native W8A16
    branch for that projection, retaining integer values and recording the 2x
    weight-storage promotion. Never clip, wrap, or dequantize to float weights.
    """
    layout = GroupwiseLayout.parse(layout)
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("Groupwise checkpoint weights and zeros must be int32 words")
    if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Groupwise scales must be FP16, BF16 or FP32")
    if qweight.device != qzeros.device or qweight.device != scales.device:
        raise ValueError("Groupwise checkpoint tensors must share one device")
    if (
        qweight.ndim not in (2, 3)
        or qzeros.ndim != qweight.ndim
        or scales.ndim != qweight.ndim
    ):
        raise ValueError(
            "Groupwise checkpoint must be dense matrices or stacked experts"
        )
    dense = qweight.ndim == 2
    if dense:
        qweight, qzeros, scales = (
            qweight.unsqueeze(0),
            qzeros.unsqueeze(0),
            scales.unsqueeze(0),
        )
    experts, k, n = qweight.shape[0], layout.input_size, layout.output_size
    if experts <= 0:
        raise ValueError("Groupwise runtime requires at least one local expert")
    pack, groups = 32 // layout.bits, k // layout.group
    expected_weight = (
        (experts, k, n // pack) if layout.method == "awq" else (experts, k // pack, n)
    )
    if (
        tuple(qweight.shape) != expected_weight
        or tuple(qzeros.shape) != (experts, groups, n // pack)
        or tuple(scales.shape) != (experts, groups, n)
    ):
        raise ValueError(
            "Groupwise checkpoint shapes disagree with the explicit local layout"
        )
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Groupwise scales must be finite and positive")
    zero = _unpack_last(qzeros, layout.bits).to(torch.int16)
    if layout.method == "awq":
        zero = _undo_awq_order(zero)
    elif not layout.gptq_v2:
        zero = zero + 1
    promoted = layout.bits == 4 and bool((zero > 15).any())
    runtime_bits = 8 if promoted else layout.bits
    weight = torch.empty(
        (experts, n, k // (2 if runtime_bits == 4 else 1)),
        dtype=torch.uint8,
        device=qweight.device,
    )
    # Bound unpacking scratch to one expert and 128 K rows, not all experts.
    for expert in range(experts):
        for begin in range(0, k, 128):
            end = min(begin + 128, k)
            if layout.method == "awq":
                values = _undo_awq_order(
                    _unpack_last(qweight[expert, begin:end], layout.bits)
                ).T
            else:
                values = _unpack_last(
                    qweight[expert, begin // pack : end // pack].T, layout.bits
                )
            values = values.to(torch.uint8)
            if runtime_bits == 4:
                weight[expert, :, begin // 2 : end // 2] = values[:, 0::2] | (
                    values[:, 1::2] << 4
                )
            else:
                weight[expert, :, begin:end] = values
    zero = zero.transpose(1, 2).contiguous()
    if runtime_bits == 4:
        if n % 2:
            raise ValueError("Native W4A16 zero points require an even output width")
        zero = (zero[:, 0::2] | (zero[:, 1::2] << 4)).to(torch.uint8)
    scale = scales.transpose(1, 2).contiguous().clone()
    reason = "effective_zero_exceeds_uint4" if promoted else None
    return GroupwiseRuntimeState(weight, scale, zero, layout, runtime_bits, reason)


def validate_runtime_device(device, method, bits):
    if method not in ("awq", "gptq") or type(bits) is not int or bits not in (4, 8):
        raise ValueError("AWQ/GPTQ runtime requires an explicit 4/8-bit method")
    validate_integer_device(device)


def validate_integer_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            "AWQ/GPTQ serving requires a native accelerator backend; CPU supports checkpoint staging only"
        )
    if torch.version.hip is None:
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability(device)[0] < 8
        ):
            raise RuntimeError("CUDA integer projections require a CUDA SM80+ device")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm AWQ/GPTQ accelerator is unavailable")
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        arch = arch.split(":", 1)[0]
        if arch not in ("gfx942", "gfx950"):
            raise RuntimeError(
                f"ROCm integer projections require gfx942 or gfx950, got {arch}"
            )
    try:
        backend = importlib.import_module(
            "rtp_llm.models_py.quant_methods.groupwise_triton"
        )
    except ImportError as exc:
        raise RuntimeError("Triton integer projection backend is unavailable") from exc
    if not callable(getattr(backend, "project", None)):
        raise RuntimeError("Triton integer projection backend requires callable project")


def _validate_parameters(layer, parameter_names):
    for name in parameter_names:
        parameter = getattr(layer, name)
        if not isinstance(parameter, torch.nn.Parameter) or parameter.requires_grad:
            raise TypeError(
                "Groupwise runtime requires inference-only module parameters"
            )


def _install_state(layer, parameter_names, state):
    # Commit only after all projections have been prepared. Replace the actual
    # module parameters; do not keep a second GPU copy of packed source words.
    _validate_parameters(layer, parameter_names)
    for role, name in zip(("weight", "zero", "scale"), parameter_names):
        getattr(layer, name).data = getattr(state, role)
    if state.promotion_reason:
        logger.warning(
            "AWQ/GPTQ %s uses native W8A16 integer storage for effective zero >15; runtime weight storage is 2x W4 (no float dequantization)",
            getattr(layer, "prefix", type(layer).__name__),
        )
    return GroupwiseRuntimeState(
        getattr(layer, parameter_names[0]),
        getattr(layer, parameter_names[2]),
        getattr(layer, parameter_names[1]),
        state.layout,
        state.runtime_bits,
        state.promotion_reason,
    )


def finalize_linear(layer):
    if hasattr(layer, "_groupwise_runtime_metadata"):
        return
    if layer.qweight.device.type == "cpu":
        return
    layout = GroupwiseLayout.parse(layer._groupwise_layout)
    validate_runtime_device(layer.qweight.device, layout.method, layout.bits)
    state = prepare_groupwise_state(
        layer.qweight, layer.qzeros, layer.scales, layer._groupwise_layout
    )
    layer._groupwise_runtime = _install_state(
        layer, ("qweight", "qzeros", "scales"), state
    )
    layer._groupwise_runtime_metadata = state.metadata


def finalize_moe(layer):
    if hasattr(layer, "_groupwise_runtime_metadata"):
        return
    if layer.w13.device.type == "cpu":
        return
    states = {}
    for name in ("w13", "w2"):
        _validate_parameters(layer, (name, name + "_qzeros", name + "_scales"))
        layout = GroupwiseLayout.parse(layer._groupwise_layout[name])
        validate_runtime_device(getattr(layer, name).device, layout.method, layout.bits)
        states[name] = prepare_groupwise_state(
            getattr(layer, name),
            getattr(layer, name + "_qzeros"),
            getattr(layer, name + "_scales"),
            layer._groupwise_layout[name],
        )
    for name, state in states.items():
        states[name] = _install_state(
            layer, (name, name + "_qzeros", name + "_scales"), state
        )
    layer._groupwise_runtime = states
    layer._groupwise_runtime_metadata = {
        name: state.metadata for name, state in states.items()
    }


def _projection(
    state,
    x,
    topk_ids,
    topk_weights,
    *,
    input_top_k,
    multiply_weights,
):
    if (
        x.device.type != "cuda"
        or x.device != state.weight.device
        or x.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Grouped integer GEMM requires colocated FP16/BF16 activations"
        )
    if x.ndim != 2 or x.shape[1] != state.layout.input_size:
        raise ValueError("Grouped integer GEMM activation shape mismatch")
    from rtp_llm.models_py.quant_methods.groupwise_triton import project

    return project(state, x, topk_ids, topk_weights, input_top_k, multiply_weights)


def _validate_routes(topk_ids, experts):
    torch._assert_async(
        ((topk_ids >= -1) & (topk_ids < experts)).all(),
        "Groupwise router supplied an invalid local expert ID",
    )


def linear(layer, x, bias=None):
    state = getattr(layer, "_groupwise_runtime", None)
    if not isinstance(state, GroupwiseRuntimeState):
        raise RuntimeError("Groupwise linear has no finalized native runtime state")
    original = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return x.new_empty((*original, state.layout.output_size))
    ids = torch.zeros((flat.shape[0], 1), dtype=torch.int32, device=x.device)
    weights = torch.ones(ids.shape, dtype=torch.float32, device=x.device)
    output = _projection(
        state,
        flat,
        ids,
        weights,
        input_top_k=1,
        multiply_weights=False,
    ).reshape(*original, state.layout.output_size)
    return output if bias is None else output + bias


def build_moe_executor(layer):
    """Use the existing FusedMoe shell with TP, stage NCCL or normal DeepEP.

    Only the packed expert GEMM executor is new. No EP collective, process group,
    or communication buffer is reimplemented here.
    """
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        CombineForwardPayload,
        FusedMoe,
        FusedMoeExpertExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
    from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
        ConditionChecker,
    )
    from rtp_llm.utils.util import to_torch_dtype

    platform = "cuda" if torch.version.hip is None else "rocm"
    routers = (
        "rtp_llm.models_py.modules.factory.fused_moe.impl." + platform + ".routers."
    )

    states = getattr(layer, "_groupwise_runtime", None)
    if not isinstance(states, dict) or set(states) != {"w13", "w2"}:
        raise RuntimeError("Grouped MoE has no finalized native expert states")
    w13, w2 = states["w13"], states["w2"]
    if (
        w13.weight.shape[0] != w2.weight.shape[0]
        or w13.layout.output_size != 2 * w2.layout.input_size
        or w2.layout.output_size != w13.layout.input_size
    ):
        raise ValueError("Grouped MoE gate/up/down layouts do not form a SwiGLU expert")
    config = MoEConfigAdapter(
        model_config=layer._model_config,
        parallelism_config=layer._parallelism_config,
        moe_config=layer._moe_config,
        quant_config=layer._effective_model_quant_config,
        enable_cuda_graph=layer._enable_cuda_graph(),
    )
    if config.physical_expert_num != layer.physical_num_experts:
        raise ValueError(
            "Grouped MoE placement and router physical expert counts differ"
        )
    if config.n_local_experts != w13.weight.shape[0]:
        raise ValueError(
            "Grouped MoE native storage does not match the router's local expert count"
        )
    config.expert_num = layer.physical_num_experts
    quant = FusedMoEQuantConfig()  # W-only: router must pass FP16/BF16 unchanged.
    checker = ConditionChecker(f"groupwise_{platform}_router")
    if config.ep_size == 1 or (platform == "cuda" and config.moe_config.use_all_gather):
        pure = importlib.import_module(routers + "pure_tp_router")
        if platform == "cuda":
            router_type = pure.PureTpRouterWeightOnly
            router_type.check_conditions(checker, config)
        else:
            router_type = pure.PureTpRouterNoQuant
            pure.PureTpRouterBase.check_conditions(checker, config)
    elif platform == "cuda" and to_torch_dtype(config.data_type) == torch.float16:
        # Normal DeepEP combines BF16 only. For FP16 keep the precision of
        # both the integer GEMM outputs and their reduction using native NCCL.
        stage = importlib.import_module(routers + "pure_dp_router")
        router_type = stage.StageRouterWeightOnlyFp16
        router_type.check_conditions(checker, config)
    else:
        normal = importlib.import_module(routers + "deepep_normal_router")
        router_type = (
            normal.DeepepNormalRouterWeightOnly
            if platform == "cuda"
            else normal.DeepepNormalRouter
        )
        router_type.check_conditions(checker, config)
    if not checker.all_passed():
        failed = ", ".join(expr for _, expr, ok, _ in checker.conditions if not ok)
        raise RuntimeError(
            f"Weight-only MoE cannot use {router_type.__name__} on {platform}: {failed}"
        )
    router = router_type(config, quant)

    class IntegerExpertExecutor(FusedMoeExpertExecutor):
        def __init__(self):
            super().__init__(config, quant, {})
            self.states = states

        @classmethod
        def executor_type(cls):
            return ExecutorType.FUSED_MOE

        @property
        def topk_ids_dtype(self):
            return torch.int32

        def execute(
            self,
            payload,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        ):
            if (
                activation.lower() not in ("silu", "siglu", "swiglu")
                or expert_map is not None
                or a2_scale is not None
                or extra_expert_args
            ):
                raise ValueError(
                    "Grouped integer MoE expects SwiGLU, physical expert IDs and unquantized activations"
                )
            if payload.expert_x_scale is not None:
                raise ValueError(
                    "Grouped W-only MoE router must not quantize activations"
                )
            x, ids, weights = (
                payload.expert_x,
                payload.expert_topk_ids,
                payload.expert_topk_weights,
            )
            if (
                ids is None
                or weights is None
                or ids.ndim != 2
                or weights.shape != ids.shape
                or x.shape[0] != ids.shape[0]
            ):
                raise ValueError("Grouped MoE router payload is incomplete")
            if x.shape[0] == 0:
                return CombineForwardPayload(
                    x.new_empty((0, w2.layout.output_size)),
                    router_context=payload.router_context,
                )
            ids = ids.to(torch.int32).contiguous()
            weights = weights.float().contiguous()
            # Both EP transports deliver rank-local IDs and -1 holes. With
            # unsharded experts (EP=1), global and local IDs are identical.
            _validate_routes(ids, w13.weight.shape[0])
            first = _projection(
                w13,
                x,
                ids,
                weights,
                input_top_k=ids.shape[1],
                multiply_weights=apply_router_weight_on_input,
            )
            up, gate = first.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate) * up).reshape(
                -1, w2.layout.input_size
            )
            second = _projection(
                w2,
                activated,
                ids,
                weights,
                input_top_k=1,
                multiply_weights=not apply_router_weight_on_input,
            )
            return CombineForwardPayload(
                second.sum(dim=1), router_context=payload.router_context
            )

    return FusedMoe(
        router,
        IntegerExpertExecutor(),
        config.expert_num,
        strategy_name=platform + "_groupwise_integer",
    )
