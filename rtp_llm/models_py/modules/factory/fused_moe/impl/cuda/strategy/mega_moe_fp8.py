"""Explicit opt-in strategies for BF16 activations and FP8 block-quantized weights."""

from dataclasses import replace

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class MegaMoeFp8Router(Fp8Fp4Router):
    """Pass routing tensors, or raw gate scores, to the fused FP8 executor."""

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        checker.check(config.ep_size > 1)

    @property
    def supports_gate_pack(self) -> bool:
        return True

    def _token_slice(self, num_tokens):
        # Attention TP ranks hold the same tokens. Give each EP source rank a
        # disjoint interval, just as the DeepEP normal router does. Under CP,
        # the adapter exposes tp_size=1 and inputs are already local.
        tokens_per_rank = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        start = min(tokens_per_rank * self.config.tp_rank, num_tokens)
        return slice(start, min(start + tokens_per_rank, num_tokens))

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        if self.config.tp_size > 1:
            token_slice = self._token_slice(a1.size(0))
            a1 = a1[token_slice]
            topk_weights = topk_weights[token_slice]
            topk_ids = topk_ids[token_slice]
        return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)

    def prepare_gate_pack(self, a1, gate_payload):
        if self.config.tp_size > 1:
            token_slice = self._token_slice(a1.size(0))
            a1 = a1[token_slice]
            gate_payload = replace(
                gate_payload,
                scores=gate_payload.scores[token_slice],
                input_ids=(
                    gate_payload.input_ids[token_slice]
                    if gate_payload.input_ids is not None
                    else None
                ),
            )
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            gate_payload=gate_payload,
        )

    def finalize(
        self,
        payload,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        extra_finalize_args,
    ):
        output = payload.fused_expert_output
        if self.config.tp_size == 1:
            return output
        if extra_finalize_args is None:
            raise ValueError("MegaMoE TP requires the original token count")
        num_tokens = extra_finalize_args["original_num_tokens"]
        if num_tokens == 0:
            # The executor has still participated in EP dispatch/combine.
            return output
        tokens_per_rank = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        if output.size(0) < tokens_per_rank:
            padded = output.new_zeros((tokens_per_rank, output.size(1)))
            padded[: output.size(0)].copy_(output)
            output = padded
        # Each row is already summed across its routed experts by MegaMoE.
        # Gather token intervals; an all-reduce would multiply the result.
        return all_gather(output.contiguous(), group=Group.TP)[:num_tokens]


class _CudaMegaMoeFp8Strategy(MoeStrategy):
    supported_moe_quant_method = "FP8_PER_BLOCK"
    requires_shared = False

    @classmethod
    def get_executor_class(cls):
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_strategy == cls.strategy_name)
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        if cls.requires_shared:
            checker.check(getattr(config, "n_shared_experts", 0) == 1)
            checker.check(bool(getattr(config, "has_shared_expert_gate", False)))

    def get_attributes(self):
        return StrategyAttributes(
            router_class=MegaMoeFp8Router,
            executor_class=self.get_executor_class(),
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
            ),
        )


class CudaMegaMoeFp8Strategy(_CudaMegaMoeFp8Strategy):
    strategy_name = "mega_moe_fp8"

    @classmethod
    def get_executor_class(cls):
        return MegaMoeFp8Executor


class CudaMegaMoeFp8SEStrategy(_CudaMegaMoeFp8Strategy):
    strategy_name = "mega_moe_fp8_se"
    requires_shared = True

    @classmethod
    def get_executor_class(cls):
        return MegaMoeFp8SEExecutor
