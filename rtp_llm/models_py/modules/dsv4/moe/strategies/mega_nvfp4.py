"""DeepGEMM ``nvfp4_nvfp4_mega_moe`` strategy for DeepSeek-V4."""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch

from ..._profiler import record_function_range
from ...quant_layouts import prepare_fp4_weight_scale_for_deepgemm
from ..mega_nvfp4_buf import (
    _get_or_create_mega_nvfp4_buf,
    _get_or_create_mega_nvfp4_output,
    _mega_moe_nvfp4_enabled,
)
from ..mega_nvfp4_input_packer import get_mega_nvfp4_input_packer
from ..mega_nvfp4_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_nvfp4_jit_token_counts,
    mega_moe_nvfp4_jit_warmup_enabled,
    parse_mega_moe_nvfp4_jit_warmup_tokens_override,
)
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

NVFP4_BLOCK = 16
_MEGA_NVFP4_JIT_WARMED_KEYS: set[tuple] = set()
_NVFP4_NVCC_TMPDIR_ENV = "DSV4_MEGA_MOE_NVFP4_NVCC_TMPDIR"
_NVFP4_PRE_KERNEL_BARRIER_ENV = "DSV4_MEGA_MOE_NVFP4_PRE_KERNEL_BARRIER"


def _output_capacity(buf, requested: int) -> int:
    return max(int(requested), int(buf.num_max_tokens_per_rank), 1)


def _rank_nvcc_tmpdir(rank: int) -> str:
    base = (
        os.environ.get(_NVFP4_NVCC_TMPDIR_ENV)
        or os.environ.get("DG_JIT_CACHE_DIR")
        or os.environ.get("TRITON_CACHE_DIR")
        or "/tmp"
    )
    return os.path.join(base, "rtp_llm_dsv4_mega_moe_nvfp4_nvcc", f"rank_{rank}")


@register_strategy
class MegaNVFP4MoEStrategy(RoutedExpertsStrategy):
    """EP routed experts with NVFP4 L1 dispatch and MXFP4 L2 weights."""

    name = "mega_nvfp4"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and _mega_moe_nvfp4_enabled()

    def setup_weights(self, layer_weights: Dict) -> None:
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        num_local_experts = cfg.n_local_experts
        hidden = cfg.dim
        intermediate = cfg.moe_inter_dim

        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device
        expected_l1_weight = (num_local_experts, intermediate, hidden // 2)
        expected_l1_scale = (num_local_experts, intermediate, hidden // 64)
        if st_w1_w.dtype != torch.int8 or tuple(st_w1_w.shape) != expected_l1_weight:
            raise TypeError(
                f"NVFP4 L1 weight must be int8 {expected_l1_weight}, got "
                f"{st_w1_w.dtype} {tuple(st_w1_w.shape)}"
            )
        for name, tensor in (("w1.scale", st_w1_s), ("w3.scale", st_w3_s)):
            if tensor.dtype != torch.int32 or tuple(tensor.shape) != expected_l1_scale:
                raise TypeError(
                    f"NVFP4 {name} must be packed int32 {expected_l1_scale}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )

        w13 = torch.empty(
            (num_local_experts, 2 * intermediate, hidden // 2),
            dtype=torch.int8,
            device=device,
        )
        s13_raw = torch.empty(
            (num_local_experts, 2 * intermediate, hidden // 64),
            dtype=torch.int32,
            device=device,
        )
        w13[:, :intermediate].copy_(st_w1_w)
        w13[:, intermediate:].copy_(st_w3_w)
        s13_raw[:, :intermediate].copy_(st_w1_s)
        s13_raw[:, intermediate:].copy_(st_w3_s)
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13_int = deep_gemm.transform_sf_into_required_layout(
            s13_raw,
            2 * intermediate,
            hidden,
            (1, NVFP4_BLOCK),
            num_local_experts,
        )
        del s13_raw
        torch.cuda.empty_cache()

        st_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        st_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        expected_l2_weight = (num_local_experts, hidden, intermediate // 2)
        expected_l2_scale = (num_local_experts, hidden, intermediate // 32)
        if st_w2_w.dtype != torch.int8 or tuple(st_w2_w.shape) != expected_l2_weight:
            raise TypeError(
                f"NVFP4 Mega L2 weight must be int8 {expected_l2_weight}, got "
                f"{st_w2_w.dtype} {tuple(st_w2_w.shape)}"
            )
        if (
            st_w2_s.dtype != torch.float8_e8m0fnu
            or tuple(st_w2_s.shape) != expected_l2_scale
        ):
            raise TypeError(
                f"NVFP4 Mega L2 scale must be float8_e8m0fnu {expected_l2_scale}, "
                f"got {st_w2_s.dtype} {tuple(st_w2_s.shape)}"
            )
        w2 = torch.empty(expected_l2_weight, dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            expected_l2_scale, dtype=torch.float8_e8m0fnu, device=device
        )
        w2.copy_(st_w2_w)
        s2_raw.copy_(st_w2_s)
        del st_w2_w, st_w2_s
        s2_int = prepare_fp4_weight_scale_for_deepgemm(
            s2_raw, hidden, intermediate, num_local_experts
        )
        del s2_raw
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_nvfp4(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()
        self._mega_nvfp4_l1_w = l1_w
        self._mega_nvfp4_l1_sf = l1_sf
        self._mega_nvfp4_l2_w = l2_w
        self._mega_nvfp4_l2_sf = l2_sf

        assert dist.is_initialized(), "NVFP4 Mega MoE requires torch.distributed"
        self._mega_nvfp4_group = dist.group.WORLD
        self._mega_nvfp4_buf = _get_or_create_mega_nvfp4_buf(
            group=self._mega_nvfp4_group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=hidden,
            intermediate_hidden=intermediate,
            activation="swiglu",
        )
        self._mega_nvfp4_y = _get_or_create_mega_nvfp4_output(
            _output_capacity(self._mega_nvfp4_buf, cfg.max_tokens_per_rank),
            hidden,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_nvfp4_input_packer()
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        override = parse_mega_moe_nvfp4_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, self.cfg.max_tokens_per_rank)
        cfg = self.cfg
        return generate_mega_moe_nvfp4_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=cfg.moe_inter_dim,
            num_sms=num_sms,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_nvfp4_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 MegaMoE JIT warmup cannot run during capture")

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        key = (
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.max_tokens_per_rank,
            cfg.swiglu_limit,
            num_sms,
            tuple(token_counts),
        )
        if not token_counts or key in _MEGA_NVFP4_JIT_WARMED_KEYS:
            return
        rank = dist.get_rank(self._mega_nvfp4_group)
        tmpdir = _rank_nvcc_tmpdir(rank)
        os.makedirs(tmpdir, exist_ok=True)
        previous_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = tmpdir
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE NVFP4] JIT warmup layer=%d tokens=[%s] TMPDIR=%s",
                cfg.layer_id,
                format_token_counts(token_counts),
                tmpdir,
            )
        try:
            self.warmup_jit(token_counts)
        finally:
            if previous_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_tmpdir
        _MEGA_NVFP4_JIT_WARMED_KEYS.add(key)

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_nvfp4_l1_w.device
        maximum = max(token_counts)
        x = torch.zeros((maximum, cfg.dim), dtype=torch.bfloat16, device=device)
        weights = torch.full(
            (maximum, cfg.n_activated_experts),
            1.0 / cfg.n_activated_experts,
            dtype=torch.float32,
            device=device,
        )
        local_ids = cfg.local_expert_start + torch.arange(
            cfg.n_activated_experts, dtype=torch.int64, device=device
        ) % max(cfg.n_local_experts, 1)
        indices = local_ids.view(1, -1).expand(maximum, -1).contiguous()
        for tokens in token_counts:
            dist.barrier(group=self._mega_nvfp4_group)
            self.forward(x[:tokens], weights[:tokens], indices[:tokens])
            torch.cuda.synchronize(device)
        dist.barrier(group=self._mega_nvfp4_group)

    def forward(self, x, weights, indices):  # type: ignore[override]
        import deep_gemm

        tokens = x.size(0)
        buf = self._mega_nvfp4_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"NVFP4 Mega MoE input tokens={tokens} exceeds buffer capacity="
                f"{buf.num_max_tokens_per_rank}"
            )
        if tokens > self._mega_nvfp4_y.size(0):
            raise RuntimeError(
                f"NVFP4 Mega MoE output rows={self._mega_nvfp4_y.size(0)} "
                f"is smaller than tokens={tokens}"
            )
        with record_function_range("dsv4.moe.mega_nvfp4_pack"):
            self._input_packer.pack(x, weights, indices, buf, tokens)
        self._maybe_pre_kernel_barrier(tokens)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe_nvfp4.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )
        y = self._mega_nvfp4_y[:tokens]
        with record_function_range("dsv4.moe.mega_nvfp4"):
            deep_gemm.nvfp4_nvfp4_mega_moe(
                y,
                (self._mega_nvfp4_l1_w, self._mega_nvfp4_l1_sf),
                (self._mega_nvfp4_l2_w, self._mega_nvfp4_l2_sf),
                buf,
                recipe=(1, 1, NVFP4_BLOCK),
                activation="swiglu",
                activation_clamp=(
                    self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
                ),
                fast_math=True,
                assume_all_topk_valid=False,
            )
        return y

    def _maybe_pre_kernel_barrier(self, tokens: int) -> None:
        if os.environ.get(_NVFP4_PRE_KERNEL_BARRIER_ENV, "0") != "1":
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_NVFP4_PRE_KERNEL_BARRIER_ENV}=1 is incompatible with capture"
            )
        import torch.distributed as dist

        torch.cuda.synchronize(self._mega_nvfp4_l1_w.device)
        logging.info(
            "[DSV4 MegaMoE NVFP4] pre-kernel barrier layer=%d rank=%d tokens=%d",
            self.cfg.layer_id,
            dist.get_rank(self._mega_nvfp4_group),
            tokens,
        )
        dist.barrier(group=self._mega_nvfp4_group)
