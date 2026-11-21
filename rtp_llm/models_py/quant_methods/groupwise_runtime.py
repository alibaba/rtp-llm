"""AWQ/GPTQ runtime bridge to Aiter's native grouped integer GEMM.

No floating weight matrix is materialized. Checkpoint integer words are
repacked into the native kernel ABI; dequantization happens inside that GEMM.
CUDA mixed-GEMM and NVFP4 layouts are not interchangeable with this ABI.
"""

import importlib
import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupwiseLayout:
    method: str
    bits: int
    group_size: int
    input_size: int
    output_size: int

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError(
                "Groupwise runtime requires explicit local parameter layout"
            )
        result = cls(**value)
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
            layout="aiter_wna16_v1",
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

    GPTQ can have an effective zero point of 16 (stored nibble 15 + 1). A
    four-bit zero-point ABI cannot represent it. Select Aiter's existing W8A16
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
    else:
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


@lru_cache(maxsize=1)
def _load_backend():
    try:
        module = importlib.import_module("aiter.ops.triton.moe.moe_op")
        sorter_module = importlib.import_module(
            "aiter.ops.triton.moe.moe_align_block_size"
        )
        language = importlib.import_module("triton.language")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "ROCm AWQ/GPTQ requires importable Aiter grouped integer GEMM and routing kernels"
        ) from error
    gemm = getattr(module, "fused_moe", None)
    sorter = getattr(sorter_module, "moe_align_block_size_triton", None)
    required = {
        "A",
        "B",
        "C",
        "B_scale",
        "B_zp",
        "topk_weights",
        "topk_ids",
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "block_shape",
        "config",
    }
    if not callable(gemm) or not required.issubset(inspect.signature(gemm).parameters):
        raise RuntimeError("Aiter fused_moe lacks the grouped AWQ/GPTQ integer ABI")
    kernel = getattr(module, "_fused_moe_kernel_gptq_awq", None)
    if kernel is None or not hasattr(kernel, "__getitem__") or not callable(sorter):
        raise RuntimeError(
            "Aiter grouped AWQ/GPTQ kernel or token-alignment symbol is missing"
        )
    return gemm, sorter, language


def validate_runtime_device(device, method, bits):
    if method not in ("awq", "gptq") or type(bits) is not int or bits not in (4, 8):
        raise ValueError("AWQ/GPTQ runtime requires an explicit 4/8-bit method")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            "AWQ/GPTQ serving requires a native accelerator backend; CPU supports checkpoint staging only"
        )
    if torch.version.hip is None:
        raise RuntimeError(
            "CUDA AWQ/GPTQ native grouped integer GEMM is not available in this build; NVFP4 Marlin and W4A8 are not equivalent backends"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm AWQ/GPTQ accelerator is unavailable")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if not any(part in arch for part in ("gfx942", "gfx950")):
        raise RuntimeError(
            f"Aiter grouped AWQ/GPTQ requires a verified eight-XCD architecture, got {arch!r}"
        )
    _load_backend()


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
    sorted_ids,
    expert_ids,
    count,
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
    gemm, _, language = _load_backend()
    output = torch.empty(
        (*topk_ids.shape, state.layout.output_size), dtype=x.dtype, device=x.device
    )
    gemm(
        A=x.contiguous(),
        B=state.weight,
        C=output,
        A_scale=None,
        B_scale=state.scale,
        B_zp=state.zero,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=count,
        mul_routed_weight=multiply_weights,
        top_k=input_top_k,
        compute_type=(
            language.float16 if x.dtype == torch.float16 else language.bfloat16
        ),
        use_fp8_w8a8=False,
        use_int8_w8a16=state.runtime_bits == 8,
        use_int4_w4a16=state.runtime_bits == 4,
        block_shape=[0, state.layout.group],
        config={
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
        },
    )
    return output


def _route(topk_ids, experts):
    _, sorter, _ = _load_backend()
    # Native align's histogram does not accept -1. Put inactive routes into one
    # extra bucket, then use GEMM's explicit expert=-1 zero-output contract.
    torch._assert_async(
        ((topk_ids >= -1) & (topk_ids < experts)).all(),
        "Groupwise router supplied an invalid local expert ID",
    )
    safe_ids = torch.where(topk_ids < 0, experts, topk_ids).to(torch.int32).contiguous()
    capacity = ((topk_ids.numel() + (experts + 1) * 15 + 15) // 16) * 16
    sorted_ids = torch.full(
        (capacity,), topk_ids.numel(), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.full(
        (capacity // 16,), -1, dtype=torch.int32, device=topk_ids.device
    )
    count = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
    sorter(safe_ids, experts + 1, 16, sorted_ids, expert_ids, count)
    expert_ids = torch.where(expert_ids == experts, -1, expert_ids)
    return sorted_ids, expert_ids, count


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
        *_route(ids, 1),
        input_top_k=1,
        multiply_weights=False,
    ).reshape(*original, state.layout.output_size)
    return output if bias is None else output + bias


def build_moe_executor(layer):
    """Use the existing FusedMoe shell, TP reduction and DeepEP normal router.

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
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.deepep_normal_router import (
        DeepepNormalRouter,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
        PureTpRouterBase,
        PureTpRouterNoQuant,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
        ConditionChecker,
    )

    if getattr(layer, "_dynamic_eplb_enabled", False):
        raise RuntimeError(
            "Grouped AWQ/GPTQ native EPLB runtime-state transaction is not yet implemented; source-only parameter replacement is unsafe"
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
    checker = ConditionChecker("groupwise_rocm_router")
    if config.ep_size == 1:
        PureTpRouterBase.check_conditions(checker, config)
        router_type = PureTpRouterNoQuant
    else:
        DeepepNormalRouter.check_conditions(checker, config)
        router_type = DeepepNormalRouter
    if not checker.all_passed():
        raise RuntimeError(
            "Grouped AWQ/GPTQ requires the existing pure-TP passthrough or supported DeepEP normal router; low-latency/MoRI routing is not adapted"
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
                activation not in ("silu", "swiglu")
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
            # DeepEP normal returns local physical IDs, including -1 holes;
            # pure TP owns every expert, so global and local IDs are identical.
            routing = _route(ids, w13.weight.shape[0])
            first = _projection(
                w13,
                x,
                ids,
                weights,
                *routing,
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
                *routing,
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
        strategy_name="rocm_aiter_groupwise_integer",
    )
