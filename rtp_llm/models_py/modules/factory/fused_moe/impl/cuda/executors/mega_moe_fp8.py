"""Opt-in FP8 MegaMoE executor. Static weights are packed during construction."""

import logging

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
    expand_fp8_scale,
    prepare_mega_moe_fp8_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.group import (
    get_validated_world_ep_group,
)
from rtp_llm.utils.model_weight import W

_BUFFER_CACHE = {}


def mega_moe_fp8_capacity(config):
    # A single request may exceed the scheduler's batch token budget.
    # MoEConfigAdapter stores sequence length in its underlying model_config.
    model_config = getattr(config, "model_config", None)
    context_length = int(getattr(model_config, "max_seq_len", 0))
    tokens = max(config.max_tokens_per_rank, context_length)
    return ((tokens + 255) // 256) * 256


def mega_moe_fp8_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from deep_gemm import mega_fp8
    except ImportError:
        return False
    return all(
        callable(getattr(mega_fp8, name, None))
        for name in (
            "fp8_fp8_mega_moe",
            "transform_weights_for_mega_moe_fp8",
            "get_symm_buffer_for_mega_moe_fp8",
        )
    )


class MegaMoeFp8Executor(FusedMoeExpertExecutor):
    execute_empty_inputs = True  # Every EP rank must participate in the collective.

    @classmethod
    def executor_type(cls):
        return ExecutorType.MEGA_MOE_FP8

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        checker.check(MoeConfigResolver().is_bf16(config))
        checker.check(config.ep_size > 1)
        checker.check(config.tp_size == 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(not config.has_redundant_experts)
        checker.check(not config.enable_cuda_graph)
        checker.check(config.swiglu_limit == 0.0)
        checker.check(config.hidden_size % 128 == 0)
        checker.check(config.moe_inter_dim % 128 == 0)
        checker.check(config.expert_num % config.ep_size == 0)
        checker.check(mega_moe_fp8_available())

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        if not mega_moe_fp8_available():
            raise RuntimeError(
                "mega_moe_fp8 requires SM10x and DeepGEMM mega_fp8 support"
            )
        self.group = get_validated_world_ep_group(config, dist)
        self.capacity = mega_moe_fp8_capacity(config)
        self.l1, self.l2 = prepare_mega_moe_fp8_weights(
            weights[W.moe_w1],
            weights[W.moe_s1],
            weights[W.moe_w2],
            weights[W.moe_s2],
            config.moe_inter_dim,
            config.moe_w1_layout,
        )
        self._weight_ready = torch.cuda.Event()
        self._weight_ready.record(torch.cuda.current_stream())
        self._ready_streams = set()
        logging.info(
            "MegaMoE FP8 weights prepared during model construction: experts=%d, "
            "hidden=%d, intermediate=%d, capacity=%d, storage_reused=True",
            config.n_local_experts,
            config.hidden_size,
            config.moe_inter_dim,
            self.capacity,
        )

    def _buffer(self):
        from deep_gemm import mega_fp8

        c = self.config
        key = (
            self.group,
            torch.cuda.current_device(),
            c.expert_num,
            self.capacity,
            c.moe_k,
            c.hidden_size,
            c.moe_inter_dim,
        )
        if key not in _BUFFER_CACHE:
            sym = mega_fp8.get_symm_buffer_for_mega_moe_fp8(
                self.group,
                c.expert_num,
                self.capacity,
                c.moe_k,
                c.hidden_size,
                c.moe_inter_dim,
                use_fp8_dispatch=True,
            )
            _BUFFER_CACHE[key] = [sym, None, torch.cuda.Event()]
        state = _BUFFER_CACHE[key]
        stream = torch.cuda.current_stream()
        if state[1] is not None and state[1] != stream:
            state[2].record(state[1])
            stream.wait_event(state[2])
        state[1] = stream
        return state[0]

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        from deep_gemm import mega_fp8

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterBase,
        )

        if activation.lower() not in ("silu", "siglu", "swiglu"):
            raise ValueError("mega_moe_fp8 supports SwiGLU only")
        if (
            expert_map is not None
            or a2_scale is not None
            or apply_router_weight_on_input
            or extra_expert_args
        ):
            raise ValueError(
                "mega_moe_fp8 does not support expert remapping or externally scaled activations"
            )
        x, ids, weights = (
            payload.expert_x,
            payload.expert_topk_ids,
            payload.expert_topk_weights,
        )
        n, h = x.shape
        if not x.is_cuda or x.dtype != torch.bfloat16 or h != self.config.hidden_size:
            raise ValueError(
                "mega_moe_fp8 requires CUDA BF16 activations with the configured hidden size"
            )
        if n > self.capacity:
            raise ValueError(
                f"MegaMoE token count {n} exceeds configured capacity {self.capacity}"
            )
        if (
            ids is None
            or weights is None
            or ids.shape != weights.shape
            or tuple(ids.shape) != (n, self.config.moe_k)
        ):
            raise ValueError("MegaMoE routing shape does not match token count/top-k")
        stream = torch.cuda.current_stream()
        if stream not in self._ready_streams:
            stream.wait_event(self._weight_ready)
            self._ready_streams.add(stream)
        sym = self._buffer()
        if n:
            # Preserve the existing 128-element input quantization exactly.
            q, sf = DeepepNormalRouterBase._do_quant_fp8_per_block(None, x)
            sym.x[:n].copy_(q)
            sym.x_sf[:n].copy_(expand_fp8_scale(sf, n, h))
            sym.topk_idx[:n].copy_(ids)
            sym.topk_weights[:n].copy_(weights)
        output = torch.empty_like(x)
        mega_fp8.fp8_fp8_mega_moe(
            output,
            self.l1,
            self.l2,
            sym,
            recipe=(1, 1, 32),
            weight_recipe=(1, 32),
            fast_math=False,
        )
        return CombineForwardPayload(fused_expert_output=output)
