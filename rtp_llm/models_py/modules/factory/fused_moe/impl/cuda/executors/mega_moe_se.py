"""DeepGEMM MegaMoE executor with FP8 shared experts fused in-kernel.

The installed DeepGEMM API uses the ordinary ``fp8_fp4_mega_moe`` symbol with
optional shared weights. The public ``mega_moe_se`` strategy owns independent
buffer/packer/warmup state from the routed-only ``mega_moe`` path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
    _get_or_create_mega_se_buf,
    _get_or_create_mega_se_output,
    _mega_moe_se_available,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_input_packer import (
    get_mega_moe_se_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_se_jit_token_counts,
    mega_moe_se_jit_warmup_enabled,
    parse_mega_moe_se_jit_warmup_tokens_override,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)

from .fp8_fp4_base import Fp8Fp4ExecutorBase
from .mega_moe import (
    MegaMoeExecutor,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _get_validated_world_ep_group,
    _mega_output_capacity,
    _restore_tmpdir,
)

_MEGA_MOE_SE_JIT_WARMED_KEYS: set[tuple] = set()
_ROUTED_RECIPE = (1, 1, FP4_BLOCK)
_SHARED_RECIPE = (1, 128, 128)
_MMA_TYPE = "fp8xfp4"


class MegaMoeSEExecutor(MegaMoeExecutor):
    """Run routed FP4 experts and the replicated FP8 shared expert together."""

    includes_shared_expert = True

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        Fp8Fp4ExecutorBase.check_conditions(checker, config)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)
        checker.check(_mega_moe_se_available())

    def setup_weights(self, layer_weights: Dict) -> None:
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        w13 = layer_weights.pop(W.moe_w1)
        s13_raw = layer_weights.pop(W.moe_s1)
        device = w13.device
        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        w2 = layer_weights.pop(W.moe_w2)
        s2_raw = layer_weights.pop(W.moe_s2)
        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()
        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        self._setup_shared_expert_weights(layer_weights, deep_gemm, W, D, inter)

        group = _get_validated_world_ep_group(cfg, dist)
        self._mega_group = group
        self._mega_buf = _get_or_create_mega_se_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            num_shared_experts=cfg.n_shared_experts,
            activation="swiglu",
        )
        self._mega_y = _get_or_create_mega_se_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            D,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_se_input_packer()
        self._maybe_warmup_jit_once()

    def _setup_shared_expert_weights(self, layer_weights, deep_gemm, W, D, inter):
        w13_fp8 = layer_weights.pop(W.ffn_w13)
        w13_scale = layer_weights.pop(W.ffn_s13)
        w2_fp8 = layer_weights.pop(W.ffn_w2)
        w2_scale = layer_weights.pop(W.ffn_s2)

        shared_inter = inter * self.cfg.n_shared_experts
        expected_w13 = (2 * shared_inter, D)
        expected_w2 = (D, shared_inter)
        if tuple(w13_fp8.shape) != expected_w13:
            raise RuntimeError(
                "MegaMoESE shared w13 weight shape mismatch: "
                f"got {tuple(w13_fp8.shape)}, expected {expected_w13}"
            )
        if tuple(w2_fp8.shape) != expected_w2:
            raise RuntimeError(
                "MegaMoESE shared w2 weight shape mismatch: "
                f"got {tuple(w2_fp8.shape)}, expected {expected_w2}"
            )
        if w13_fp8.dtype != torch.float8_e4m3fn or w2_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "MegaMoESE shared weights must be float8_e4m3fn; "
                f"got w13={w13_fp8.dtype}, w2={w2_fp8.dtype}"
            )

        w13_sf_int = self._shared_expert_sf_to_int(
            deep_gemm, w13_scale, 2 * shared_inter, D
        )
        w2_sf_int = self._shared_expert_sf_to_int(deep_gemm, w2_scale, D, shared_inter)
        del w13_scale, w2_scale
        (se_l1_w, se_l1_sf), (se_l2_w, se_l2_sf) = (
            deep_gemm.transform_weights_for_mega_moe(
                (w13_fp8.contiguous(), w13_sf_int),
                (w2_fp8.contiguous(), w2_sf_int),
            )
        )
        del w13_fp8, w13_sf_int, w2_fp8, w2_sf_int
        torch.cuda.empty_cache()
        self._se_l1_w = se_l1_w
        self._se_l1_sf = se_l1_sf
        self._se_l2_w = se_l2_w
        self._se_l2_sf = se_l2_sf

    @staticmethod
    def _shared_expert_sf_to_int(deep_gemm, scale, mn, k):
        if scale.dtype == torch.int32:
            return scale
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(
                "MegaMoESE expected shared UE8M0 scale, " f"got {scale.dtype}"
            )
        return deep_gemm.transform_sf_into_required_layout(
            scale.float(), mn, k, _SHARED_RECIPE[1:], num_groups=None
        )

    def _block_m(self, tokens: int) -> int:
        import deep_gemm

        return int(
            deep_gemm.get_block_m_for_mega_moe(
                self.cfg.ep_size,
                self.cfg.n_routed_experts,
                self._mega_buf.num_max_tokens_per_rank,
                int(tokens),
                self.cfg.n_activated_experts,
                _MMA_TYPE,
            )
        )

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_mega_moe_se_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens_per_rank)
        return generate_mega_moe_se_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=cfg.moe_inter_dim,
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
            include_cap=bool(getattr(cfg, "warmup_include_capacity", False)),
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_se_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoESE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return
        warmup_key = (
            "mega_moe_se",
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_shared_experts,
            int(cfg.max_tokens_per_rank),
            cfg.swiglu_limit,
            num_sms,
            _SHARED_RECIPE,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_SE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[MegaMoESE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d shared=%d num_sms=%d shared_recipe=%s",
                cfg.layer_id,
                format_token_counts(token_counts),
                cfg.max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_shared_experts,
                num_sms,
                _SHARED_RECIPE,
            )
        tmpdir, previous_tmpdir = _activate_mega_moe_rank_nvcc_tmpdir(rank)
        try:
            if rank == 0:
                logging.info("[MegaMoESE] rank-local nvcc TMPDIR=%s", tmpdir)
            self.warmup_jit(token_counts)
        finally:
            _restore_tmpdir(previous_tmpdir)
        _MEGA_MOE_SE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[MegaMoESE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    def _validate_capacity(self, tokens: int) -> None:
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE SE input tokens={tokens} exceeds "
                f"num_max_tokens_per_rank={buf.num_max_tokens_per_rank}. "
                "Raise the startup MoE token budget."
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE SE output rows={self._mega_y.size(0)} are smaller "
                f"than input tokens={tokens}; aligned capacity is inconsistent"
            )

    def _launch(self, y: torch.Tensor, tokens: int, device: torch.device) -> None:
        import deep_gemm

        self._maybe_pre_kernel_barrier(tokens)
        sync_cuda_graph_warmup_ranks(
            f"moe.mega_moe_se.layer{self.cfg.layer_id}.before_deepgemm",
            device,
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_l1_weights=(self._se_l1_w, self._se_l1_sf),
            shared_l2_weights=(self._se_l2_w, self._se_l2_sf),
            recipe=_ROUTED_RECIPE,
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
            shared_recipe=_SHARED_RECIPE,
        )

    def forward(self, x, weights, indices):
        """Return BF16 routed+shared output; participate even for local T=0."""

        tokens = x.size(0)
        self._validate_capacity(tokens)
        block_m = self._block_m(tokens)
        self._input_packer.pack(x, weights, indices, self._mega_buf, tokens, block_m)
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y

    def forward_gate_pack(
        self,
        x: torch.Tensor,
        gate_payload: ExpertGatePayload,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route, pack, and stage shared-expert scales for MegaMoE-SE."""

        if not self.supports_gate_pack:
            raise RuntimeError(
                "MegaMoE-SE fused gate packing requires the fused packer"
            )
        tokens = x.size(0)
        self._validate_capacity(tokens)
        block_m = self._block_m(tokens)
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack import (
            stage_mega_moe_se_shared_l1_scales,
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
        stage_mega_moe_se_shared_l1_scales(
            buf.x_sf[:tokens],
            buf.shared_l1_acts_sf,
            tokens,
            block_m,
        )
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y, buf.topk_weights[:tokens], buf.topk_idx[:tokens]


__all__ = ["MegaMoeSEExecutor"]
