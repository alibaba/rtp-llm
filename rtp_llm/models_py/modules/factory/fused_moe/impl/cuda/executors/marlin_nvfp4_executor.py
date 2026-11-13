"""Qwen4 weight adaptation and factory admission for NVFP4 Marlin."""

import torch

from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
    Nvfp4MarlinMoE,
    load_library,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.resident_weights import resident_init_tensors
from rtp_llm.utils.model_weight import W

_MARLIN_NAMES = (
    W.moe_w1,
    W.moe_w2,
    W.moe_s1,
    W.moe_s2,
    W.moe_w1_s2,
    W.moe_w2_s2,
)


def _linear_scales(swizzled, experts, rows, cols):
    return (
        swizzled.view(torch.float8_e4m3fn)
        .reshape(experts, rows // 128, cols // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows, cols)
    )


def _prepare_marlin_weights(weights, ep_size, ep_rank):
    q1 = weights[W.moe_w1].contiguous().view(torch.uint8)
    q2 = weights[W.moe_w2].contiguous().view(torch.uint8)
    experts, both_n, packed_k = q1.shape
    n, k = both_n // 2, packed_k * 2
    sf1 = _linear_scales(weights[W.moe_s1], experts, both_n, k // 16)
    sf2 = _linear_scales(weights[W.moe_s2], experts, k, n // 16)
    g1, g2 = weights[W.moe_w1_s2], weights[W.moe_w2_s2]
    for scale in (sf1, sf2, g1, g2):
        if not bool(torch.isfinite(scale.float()).all()) or bool(
            (scale.float() < 0).any()
        ):
            raise ValueError("Marlin NVFP4 requires finite nonnegative scales")
    moe = Nvfp4MarlinMoE(q1, sf1, g1, q2, sf2, g2, ep_size, ep_rank)
    for name in _MARLIN_NAMES:
        weights.pop(name)
    return moe.w1, moe.w2, moe.s1, moe.s2, moe.g1, moe.g2


class Qwen4DecodeMarlinExecutor(FusedMoeExpertExecutor):
    """Adapt prepared Qwen4 NVFP4 weights to the common Marlin kernel.

    Factory admission remains limited to qwen4_exp and qwen4_exp_mtp; other
    models require separate validation before their admission is enabled.
    """

    @classmethod
    def executor_type(cls):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

        return ExecutorType.CUTLASS_NVFP4

    @property
    def topk_ids_dtype(self):
        return torch.int32

    @property
    def local_num_experts(self):
        return self.moe.experts

    def __init__(self, native):
        super().__init__(native.config, native.quant_config, native.weights)
        self.ep_size, self.ep_rank = native.ep_size, native.ep_rank
        self.tp_size, self.tp_rank = native.tp_size, native.tp_rank
        self._initialize_weights()
        # Callers retaining native views must release their original allocations too.
        native.w1 = self.weights[W.moe_w1].view(torch.int64)
        native.w2 = self.weights[W.moe_w2].view(torch.int64)
        native.block_scales = (
            self.weights[W.moe_s1].view(torch.int32),
            self.weights[W.moe_s2].view(torch.int32),
        )

    def _initialize_weights(self):
        # A Qwen compute dictionary borrows storage from the loader's WeightDict.
        # Record at that owner, then rebind both dictionaries to the returned storage.
        owner = getattr(self.weights, "_storage_owner", self.weights)
        key = f"qwen4.marlin_nvfp4.v1:ep{self.ep_size}:rank{self.ep_rank}"
        prior = getattr(owner.get(W.moe_w1), "_qwen4_marlin_layout", None)
        if prior is None:
            w1, w2, s1, s2, g1, g2 = resident_init_tensors(
                owner,
                transform_key=key,
                output_names=tuple(
                    f"{key}.{name}" for name in ("w1", "w2", "s1", "s2", "g1", "g2")
                ),
                consumed_names=_MARLIN_NAMES,
                builder=lambda: _prepare_marlin_weights(
                    self.weights, self.ep_size, self.ep_rank
                ),
            )
            experts, k_tiles, packed_n = w1.shape
            k, n = k_tiles * 16, packed_n // 4
            # Preserve canonical geometry for the C++ converter; the bytes are now packed.
            canonical = (
                w1.view(torch.uint8).reshape(experts, 2 * n, k // 2),
                w2.view(torch.uint8).reshape(experts, k, n // 2),
                s1.reshape(experts, 2 * n, k // 16),
                s2.reshape(experts, k, n // 16),
                g1,
                g2,
            )
            for name, tensor in zip(_MARLIN_NAMES, canonical):
                self.weights[name] = tensor
                owner[name] = tensor
            # This process-local guard is recreated after replay. It owns no tensors.
            canonical[0]._qwen4_marlin_layout = (
                key,
                tuple(tensor.data_ptr() for tensor in canonical),
            )
        else:
            canonical = tuple(owner[name] for name in _MARLIN_NAMES)
            if prior != (key, tuple(tensor.data_ptr() for tensor in canonical)):
                raise ValueError("Changing packed Qwen MoE weights requires restart")
            for name, tensor in zip(_MARLIN_NAMES, canonical):
                self.weights[name] = tensor
            experts, both_n, packed_k = canonical[0].shape
            n, k = both_n // 2, packed_k * 2
            w1 = canonical[0].view(torch.int32).reshape(experts, k // 16, 4 * n)
            w2 = canonical[1].view(torch.int32).reshape(experts, n // 16, 2 * k)
            s1 = canonical[2].view(torch.float8_e4m3fn).reshape(experts, k // 16, 2 * n)
            s2 = canonical[3].view(torch.float8_e4m3fn).reshape(experts, n // 16, k)
            g1, g2 = canonical[4:]

        load_library()
        moe = object.__new__(Nvfp4MarlinMoE)
        moe.experts, moe.n, moe.k = experts, n, k
        moe._one_row_sm120 = (k, n) == (2560, 640) and torch.cuda.get_device_capability(
            w1.device
        ) == (12, 0)
        moe.first = experts * self.ep_rank
        moe.w1, moe.w2, moe.s1, moe.s2, moe.g1, moe.g2 = w1, w2, s1, s2, g1, g2
        moe.workspaces = {}
        self.moe = moe

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        if activation.lower() not in ("silu", "swiglu", "siglu"):
            raise ValueError("Marlin NVFP4 requires SwiGLU")
        if (
            apply_router_weight_on_input
            or payload.expert_ids_are_local
            or expert_map is not None
        ):
            raise ValueError(
                "Marlin NVFP4 requires global IDs and output router weighting"
            )
        x = payload.expert_x
        if x.numel() == 0:
            output = torch.empty_like(x)
        else:
            output = self.moe(
                x,
                payload.expert_topk_ids.to(torch.int32),
                payload.expert_topk_weights.to(torch.float32),
            )
        return CombineForwardPayload(fused_expert_output=output)


class Qwen4MarlinExecutor(Qwen4DecodeMarlinExecutor):
    """Explicit Marlin selection, currently restricted to Qwen4 models.

    Common placement fixes the dependency direction; it does not expand
    admission beyond qwen4_exp and qwen4_exp_mtp without model validation.
    """

    @classmethod
    def executor_type(cls):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

        return ExecutorType.MARLIN_NVFP4

    @classmethod
    def check_conditions(cls, checker, config):
        from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant = config.model_config.quant_config
        checker.check(resolver.is_bf16(config))
        checker.check(resolver.get_quant_method(config) == "modelopt_fp4")
        checker.check(config.model_config.model_type in ("qwen4_exp", "qwen4_exp_mtp"))
        checker.check(
            quant is not None
            and (
                (
                    getattr(quant, "model_recipe", None) == QWEN4_FP8_NVFP4_RECIPE
                    and not quant.is_quanted()
                )
                or (
                    getattr(config.model_config, "_qwen4_nvfp4_source", False)
                    and quant.is_quanted()
                    and quant.bits == 4
                    and quant.group_size() == 16
                )
            )
        )
        checker.check(Nvfp4MarlinMoE.is_supported())
        checker.check(config.tp_size == config.ep_size)
        checker.check(
            config.ep_size > 1 or config.parallelism_config.get_ffn_tp_size() == 1
        )
        checker.check(config.moe_config.fp4_moe_op in ("auto", "marlin"))
        checker.check(not config.has_redundant_experts)
        checker.check(not config.model_config.eplb_config.enable_eplb())

    def __init__(self, config, quant_config, weights):
        FusedMoeExpertExecutor.__init__(self, config, quant_config, weights)
        self.ep_size, self.ep_rank = config.ep_size, config.ep_rank
        pc = getattr(config, "parallelism_config", None)
        pure_tp = pc is not None and self.ep_size == 1
        self.tp_size = pc.get_ffn_tp_size() if pure_tp else 1
        self.tp_rank = pc.get_ffn_tp_rank() if pure_tp else 0
        self._initialize_weights()
