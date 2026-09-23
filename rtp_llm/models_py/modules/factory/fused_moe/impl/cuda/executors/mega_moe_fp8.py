"""DeepGEMM ``fp8_fp8_mega_moe`` routed-expert executor.

Mirrors ``mega_moe.py`` (fp8_fp4): CUDA-graph-safe buffers, fused input /
gate packing, and an explicit ``mega_moe_fp8`` strategy. Shared-expert fusion
lives on ``mega_moe_fp8_se``.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
    _mega_output_capacity,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_fp8_buf,
    _get_or_create_mega_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
    prepare_mega_moe_fp8_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.group import (
    get_validated_world_ep_group,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    mega_moe_jit_warmup_enabled,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)
from rtp_llm.utils.model_weight import W


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


class MegaMoeFp8Executor(MegaMoeExecutor):
    execute_empty_inputs = True  # Every EP rank must participate in the collective.
    gated_shared_expert_requested = False
    _num_shared_experts = 0
    gate_pack_score_func = "softmax"

    @classmethod
    def executor_type(cls):
        return ExecutorType.MEGA_MOE_FP8

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        checker.check(MoeConfigResolver().is_bf16(config))
        checker.check(config.ep_size > 1)
        checker.check(config.tp_size >= 1 and config.world_size % config.tp_size == 0)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(not config.has_redundant_experts)
        checker.check(config.swiglu_limit == 0.0)
        checker.check(config.hidden_size % 128 == 0)
        checker.check(config.moe_inter_dim % 128 == 0)
        checker.check(config.expert_num % config.ep_size == 0)
        checker.check(mega_moe_fp8_available())

    def setup_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if not mega_moe_fp8_available():
            raise RuntimeError(
                "mega_moe_fp8 requires SM10x and DeepGEMM mega_fp8 support"
            )
        self.uses_shared_expert_gates = False
        self.shared_l1 = self.shared_l2 = None
        config = self.cfg
        self.l1, self.l2 = prepare_mega_moe_fp8_weights(
            weights[W.moe_w1],
            weights[W.moe_s1],
            weights[W.moe_w2],
            weights[W.moe_s2],
            config.moe_inter_dim,
            config.moe_w1_layout,
        )
        # Routed weights are owned by this backend after packing. Drop the
        # checkpoint scales from the shared layer dictionary, otherwise the
        # model/weight manager keep their GPU storage alive alongside l1/l2.
        # Only release after both conversions succeed; backend changes already
        # require a model reload because weight storage is repacked in place.
        del weights[W.moe_s1], weights[W.moe_s2]
        self._mega_l1_w = self.l1[0]
        self._mega_group = get_validated_world_ep_group(config, dist)
        self._mega_buf = _get_or_create_mega_fp8_buf(
            group=self._mega_group,
            num_experts=config.expert_num,
            num_max_tokens_per_rank=max(config.max_tokens_per_rank, 1),
            num_topk=config.moe_k,
            hidden=config.hidden_size,
            intermediate_hidden=config.moe_inter_dim,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        self._mega_y = _get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, config.max_tokens_per_rank),
            config.hidden_size,
            torch.bfloat16,
            self.l1[0].device,
        )
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()
        logging.info(
            "MegaMoE FP8 weights prepared during model construction: experts=%d, "
            "hidden=%d, intermediate=%d, max_tokens_per_rank=%d, shared=%d",
            config.n_local_experts,
            config.hidden_size,
            config.moe_inter_dim,
            config.max_tokens_per_rank,
            self._num_shared_experts,
        )

    @property
    def supports_gate_pack(self) -> bool:
        return True

    def _warmup_gate_payloads(
        self,
        tokens: int,
        scores: torch.Tensor,
        device: torch.device,
    ) -> list[ExpertGatePayload]:
        cfg = self.cfg
        return [
            ExpertGatePayload(
                scores=scores,
                topk=cfg.n_activated_experts,
                score_func=self.gate_pack_score_func,
                route_scale=cfg.route_scale,
            )
        ]

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        if not mega_moe_jit_warmup_enabled() or not token_counts:
            return
        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(token_counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        scores = torch.zeros(
            (max_tokens, cfg.n_routed_experts),
            dtype=torch.bfloat16,
            device=device,
        )
        for token_count in token_counts:
            if dist.is_initialized():
                dist.barrier()
            for payload in self._warmup_gate_payloads(
                token_count, scores[:token_count], device
            ):
                self.forward_gate_pack(x[:token_count], payload)
                torch.cuda.synchronize(device)
        if dist.is_initialized():
            dist.barrier()

    def _validate_capacity(self, tokens: int) -> None:
        if tokens > self._mega_buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE FP8 input tokens={tokens} exceeds "
                f"num_max_tokens_per_rank={self._mega_buf.num_max_tokens_per_rank}"
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE FP8 output buffer rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={tokens}"
            )

    def _launch(self, y: torch.Tensor, tokens: int, device: torch.device, **kwargs):
        from deep_gemm import mega_fp8

        self._maybe_pre_kernel_barrier(tokens)
        sync_cuda_graph_warmup_ranks(
            f"moe.mega_moe_fp8.layer{getattr(self.config, 'layer_id', -1)}"
            f".tokens{tokens}.before_deepgemm",
            device,
        )
        mega_fp8.fp8_fp8_mega_moe(
            y,
            self.l1,
            self.l2,
            self._mega_buf,
            recipe=(1, 1, 32),
            weight_recipe=(1, 32),
            activation="swiglu",
            fast_math=False,
            **kwargs,
        )

    def forward_gate_pack(self, x, gate_payload):
        if not self.supports_gate_pack:
            raise RuntimeError(
                "MegaMoE FP8 fused gate packing requires the fused packer"
            )
        tokens = x.size(0)
        self._validate_capacity(tokens)
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        buf = self._mega_buf
        fused_pack_mega_moe_gate_inputs(
            x,
            gate_payload.scores,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
            topk=gate_payload.topk,
            score_func=gate_payload.score_func,
            route_scale=gate_payload.route_scale,
            norm_eps=gate_payload.norm_eps,
            bias=gate_payload.bias,
            input_ids=gate_payload.input_ids,
            tid2eid=gate_payload.tid2eid,
        )
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y, buf.topk_weights[:tokens], buf.topk_idx[:tokens]

    def forward(self, x, weights, indices):
        """Return BF16 routed output; participate even for local T=0."""
        tokens = x.size(0)
        buf = self._mega_buf
        self._validate_capacity(tokens)
        self._input_packer.pack(x, weights, indices, buf, tokens)
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y
