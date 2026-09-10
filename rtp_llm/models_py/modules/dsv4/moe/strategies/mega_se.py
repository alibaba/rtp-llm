"""DeepGEMM Mega MoE strategy with the FP8 shared expert fused in-kernel.

The installed DeepGEMM API uses the ordinary ``fp8_fp4_mega_moe`` symbol with
optional shared weights.  This strategy is enabled by default and can be
disabled via ``DSV4_USE_MEGA_MOE_SE=0``.  It owns independent
buffer/packer/warmup state from the routed-only Mega path.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch
import torch.nn.functional as F

from ..._profiler import record_function_range
from ...quant_layouts import FP4_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from ..mega_se_buf import (
    _get_or_create_mega_se_buf,
    _get_or_create_mega_se_output,
    _mega_moe_se_enabled,
)
from ..mega_se_input_packer import get_mega_moe_se_input_packer
from ..mega_se_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_se_jit_token_counts,
    mega_moe_se_jit_warmup_enabled,
    parse_mega_moe_se_jit_warmup_tokens_override,
)
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, register_strategy
from .mega import (
    MegaMoEStrategy,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _gate_pack_input_packer_env_allows,
    _mega_output_capacity,
    _restore_tmpdir,
)

_MEGA_MOE_SE_JIT_WARMED_KEYS: set[tuple] = set()
_MEGA_SE_GATE_PACK_KERNELS = None
_MEGA_SE_GATE_PACK_KERNELS_UNAVAILABLE = False
_ROUTED_RECIPE = (1, 1, FP4_BLOCK)
_SHARED_RECIPE = (1, 128, 128)
_MMA_TYPE = "fp8xfp4"


def _get_mega_se_gate_pack_kernels():
    global _MEGA_SE_GATE_PACK_KERNELS, _MEGA_SE_GATE_PACK_KERNELS_UNAVAILABLE
    if _MEGA_SE_GATE_PACK_KERNELS_UNAVAILABLE:
        return None
    if _MEGA_SE_GATE_PACK_KERNELS is not None:
        return _MEGA_SE_GATE_PACK_KERNELS
    try:
        from .._mega_se_gate_pack_triton import (
            fused_mega_moe_se_gate_pack_hash,
            fused_mega_moe_se_gate_pack_nonhash,
            fused_mega_moe_se_gate_pack_supported,
            triton,
        )
    except Exception:
        _MEGA_SE_GATE_PACK_KERNELS_UNAVAILABLE = True
        return None
    if triton is None:
        _MEGA_SE_GATE_PACK_KERNELS_UNAVAILABLE = True
        return None
    _MEGA_SE_GATE_PACK_KERNELS = (
        fused_mega_moe_se_gate_pack_nonhash,
        fused_mega_moe_se_gate_pack_hash,
        fused_mega_moe_se_gate_pack_supported,
    )
    return _MEGA_SE_GATE_PACK_KERNELS


@register_strategy
class MegaMoEStrategySE(MegaMoEStrategy):
    """Run routed FP4 experts and the replicated FP8 shared expert together."""

    name = "mega_se"
    routed_includes_shared = True

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and _mega_moe_se_enabled()

    def setup_weights(self, layer_weights: Dict) -> None:
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        # Routed FP4 L1 (gate + up), identical layout to ordinary Mega.
        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device
        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w13[:, :inter].copy_(st_w1_w)
        s13_raw[:, :inter].copy_(st_w1_s)
        w13[:, inter:].copy_(st_w3_w)
        s13_raw[:, inter:].copy_(st_w3_s)
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        # Routed FP4 L2; keep the same memory-serialised setup as ordinary Mega.
        st_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        st_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w2.copy_(st_w2_w)
        s2_raw.copy_(st_w2_s)
        del st_w2_w, st_w2_s
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

        assert dist.is_initialized(), (
            "Mega MoE SE requires torch.distributed initialised; "
            "capability selection should have gated this earlier"
        )
        group = dist.group.WORLD
        self._mega_group = group
        self._mega_buf = _get_or_create_mega_se_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
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
        w13_fp8 = layer_weights.pop(W.v4_shared_w13_w)
        w13_scale = layer_weights.pop(W.v4_shared_w13_s)
        w2_fp8 = layer_weights.pop(W.v4_shared_w2_w)
        w2_scale = layer_weights.pop(W.v4_shared_w2_s)

        expected_w13 = (2 * inter, D)
        expected_w2 = (D, inter)
        if tuple(w13_fp8.shape) != expected_w13:
            raise RuntimeError(
                "MegaMoE-SE shared w13 weight shape mismatch: "
                f"got {tuple(w13_fp8.shape)}, expected {expected_w13}"
            )
        if tuple(w2_fp8.shape) != expected_w2:
            raise RuntimeError(
                "MegaMoE-SE shared w2 weight shape mismatch: "
                f"got {tuple(w2_fp8.shape)}, expected {expected_w2}"
            )
        if w13_fp8.dtype != torch.float8_e4m3fn or w2_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "MegaMoE-SE shared weights must be float8_e4m3fn; "
                f"got w13={w13_fp8.dtype}, w2={w2_fp8.dtype}"
            )

        w13_sf_int = self._shared_expert_sf_to_int(deep_gemm, w13_scale, 2 * inter, D)
        w2_sf_int = self._shared_expert_sf_to_int(deep_gemm, w2_scale, D, inter)
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
                "MegaMoE-SE expected shared UE8M0 scale, " f"got {scale.dtype}"
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
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_se_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE-SE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return
        warmup_key = (
            "mega_se",
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            int(cfg.max_tokens_per_rank),
            cfg.swiglu_limit,
            num_sms,
            _SHARED_RECIPE,
            tuple(token_counts),
            bool(getattr(self, "_gate_pack_warmup_enabled", False)),
            (
                float(getattr(self, "_gate_pack_route_scale", 1.0))
                if getattr(self, "_gate_pack_warmup_enabled", False)
                else None
            ),
        )
        if warmup_key in _MEGA_MOE_SE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE-SE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d num_sms=%d shared_recipe=%s",
                cfg.layer_id,
                format_token_counts(token_counts),
                cfg.max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                num_sms,
                _SHARED_RECIPE,
            )
        tmpdir, previous_tmpdir = _activate_mega_moe_rank_nvcc_tmpdir(rank)
        try:
            if rank == 0:
                logging.info("[DSV4 MegaMoE-SE] rank-local nvcc TMPDIR=%s", tmpdir)
            self.warmup_jit(token_counts)
        finally:
            _restore_tmpdir(previous_tmpdir)
        _MEGA_MOE_SE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE-SE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    def _warmup_gate_pack_jit(self, token_counts: list[int]) -> None:
        if not getattr(self, "_gate_pack_warmup_enabled", False):
            return
        kernels = _get_mega_se_gate_pack_kernels()
        if kernels is None:
            return
        pack_nonhash, pack_hash, _ = kernels
        counts = [int(t) for t in token_counts if int(t) > 0]
        if not counts:
            return

        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        scores = torch.zeros(
            (max_tokens, cfg.n_routed_experts),
            dtype=torch.bfloat16,
            device=device,
        )
        bias = torch.zeros((cfg.n_routed_experts,), dtype=torch.float32, device=device)
        input_ids = torch.zeros((max_tokens,), dtype=torch.long, device=device)
        tid2eid = (
            torch.arange(cfg.n_activated_experts, dtype=torch.long, device=device)
            .view(1, -1)
            .contiguous()
        )
        buf = self._mega_buf
        route_scale = float(getattr(self, "_gate_pack_route_scale", 1.0))
        for token_count in counts:
            block_m = self._block_m(token_count)
            pack_nonhash(
                x[:token_count],
                scores[:token_count],
                bias,
                buf.x[:token_count],
                buf.x_sf[:token_count],
                buf.shared_l1_acts_sf,
                buf.topk_idx[:token_count],
                buf.topk_weights[:token_count],
                block_m=block_m,
                route_scale=route_scale,
                norm_eps=1.0e-12,
            )
            pack_hash(
                x[:token_count],
                scores[:token_count],
                input_ids[:token_count],
                tid2eid,
                buf.x[:token_count],
                buf.x_sf[:token_count],
                buf.shared_l1_acts_sf,
                buf.topk_idx[:token_count],
                buf.topk_weights[:token_count],
                block_m=block_m,
                route_scale=route_scale,
                norm_eps=1.0e-12,
            )
            torch.cuda.synchronize(device)

    def can_use_gate_pack_static(self, gate) -> bool:
        return (
            os.environ.get("DSV4_GATE_FUSED", "1") != "0"
            and os.environ.get("DSV4_GATE_FP32", "0") != "1"
            and gate.score_func == "sqrtsoftplus"
            and 1 <= int(gate.topk) <= 32
            and self.cfg.dim % 128 == 0
            and _gate_pack_input_packer_env_allows()
            and _get_mega_se_gate_pack_kernels() is not None
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
            f"dsv4.mega_moe_se.layer{self.cfg.layer_id}.before_deepgemm",
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

    def forward_prepacked(self, tokens: int, device: torch.device) -> torch.Tensor:
        """Run MegaMoE-SE after the CUDA extension populated its input buffer."""
        tokens = int(tokens)
        self._validate_capacity(tokens)
        y = self._mega_y[:tokens]
        # A zero-token rank still enters the collective launch. This is
        # required when EP/DP routing leaves a rank with no local tokens.
        self._launch(y, tokens, device)
        return y

    def forward_with_gate_pack(self, x, gate, input_ids):
        kernels = _get_mega_se_gate_pack_kernels()
        if kernels is None:
            raise RuntimeError(
                "MegaMoE-SE gate-pack was selected but kernels are unavailable"
            )
        pack_nonhash, pack_hash, _ = kernels
        tokens = x.size(0)
        self._validate_capacity(tokens)
        block_m = self._block_m(tokens)
        buf = self._mega_buf
        y = self._mega_y[:tokens]

        with record_function_range("dsv4.moe.gate_linear_bf16"):
            scores_bf16 = F.linear(x, gate._weight_bf16())
        with record_function_range("dsv4.moe.mega_se_gate_pack"):
            if gate.hash:
                assert input_ids is not None
                pack_hash(
                    x,
                    scores_bf16.contiguous(),
                    input_ids.reshape(-1).contiguous(),
                    gate.tid2eid.contiguous(),
                    buf.x[:tokens],
                    buf.x_sf[:tokens],
                    buf.shared_l1_acts_sf,
                    buf.topk_idx[:tokens],
                    buf.topk_weights[:tokens],
                    block_m=block_m,
                    route_scale=float(gate.route_scale),
                    norm_eps=1.0e-12,
                )
            else:
                assert gate.bias is not None
                pack_nonhash(
                    x,
                    scores_bf16.contiguous(),
                    gate.bias.contiguous(),
                    buf.x[:tokens],
                    buf.x_sf[:tokens],
                    buf.shared_l1_acts_sf,
                    buf.topk_idx[:tokens],
                    buf.topk_weights[:tokens],
                    block_m=block_m,
                    route_scale=float(gate.route_scale),
                    norm_eps=1.0e-12,
                )
        self._launch(y, tokens, x.device)
        return y


__all__ = ["MegaMoEStrategySE"]
