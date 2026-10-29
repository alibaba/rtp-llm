"""DeepGEMM MegaMoE executor with FP8 shared experts fused in-kernel.

The installed DeepGEMM API uses the ordinary ``fp8_fp4_mega_moe`` symbol with
optional shared weights. The public ``mega_moe_se`` strategy owns independent
buffer/packer/warmup state from the routed-only ``mega_moe`` path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from rtp_llm.config.mega_moe import FP8_MEGA_MOE_STRATEGIES

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    mega_shared_expert_block32_enabled,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    resolve_mega_num_sms,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fused_buffer import (
    get_or_create_mega_buf_fused,
    make_shared_mid_workspace,
    transform_shared_expert_fp8_for_fused,
)
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
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)
from rtp_llm.utils.model_weight import W

from .fp8_fp4_base import Fp8Fp4ExecutorBase
from .mega_moe import (
    MegaMoeExecutor,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _get_validated_world_ep_group,
    _mega_intermediate_size,
    _mega_output_capacity,
    _restore_tmpdir,
)

_MEGA_MOE_SE_JIT_WARMED_KEYS: set[tuple] = set()
_ROUTED_RECIPE = (1, 1, FP4_BLOCK)
# Routed experts are FP4 (K block 32); the shared expert stays FP8 and picks
# the recipe that matches its checkpoint quantization.
_SHARED_RECIPES = {32: (1, 1, 32), 128: (1, 128, 128)}
_MMA_TYPE = "fp8xfp4"


class MegaMoeSEExecutor(MegaMoeExecutor):
    """Run routed FP4 experts and the replicated FP8 shared expert together."""

    includes_shared_expert = True
    _defer_buffer_until_shared_weights = True
    _defer_jit_until_shared_weights = True

    @property
    def _shared_recipe(self) -> tuple[int, int, int]:
        return _SHARED_RECIPES[self.cfg.shared_fp8_block_size]

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        Fp8Fp4ExecutorBase.check_conditions(checker, config)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)
        if getattr(config, "mega_moe_policy", None) is not None:
            if getattr(config, "moe_strategy", "mega_moe_se") not in FP8_MEGA_MOE_STRATEGIES:
                checker.check(_mega_moe_se_available(allow_single_rank=True))
            return
        checker.check(config.shared_fp8_block_size in _SHARED_RECIPES)
        # The fused activation/combine rounds differently from the standalone
        # BF16 shared expert, so group-32 needs an explicit decode-only opt-in.
        checker.check(
            config.shared_fp8_block_size != 32
            or (config.is_decode_role and mega_shared_expert_block32_enabled())
        )
        checker.check(_mega_moe_se_available())

    def setup_weights(self, layer_weights: Dict) -> None:
        if self._prepared_policy is not None:
            return super().setup_weights(layer_weights)

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        D = cfg.dim
        inter = _mega_intermediate_size(cfg)
        if inter != cfg.moe_inter_dim:
            raise RuntimeError(
                "MegaMoE-SE requires DeepGEMM support for the checkpoint "
                f"intermediate size {cfg.moe_inter_dim}; the routed padding "
                f"path ({inter}) has no shared-expert weights. Use "
                "MOE_STRATEGY=mega_moe to run the routed padding fallback "
                "with a standalone shared expert."
            )
        device = self._transform_routed_weights(layer_weights, deep_gemm, inter)

        self._setup_shared_expert_weights(layer_weights, deep_gemm, D, inter)

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

    def _setup_shared_expert_weights(self, layer_weights, deep_gemm, D, inter):
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
            deep_gemm, w13_scale, 2 * shared_inter, D, self.cfg.shared_fp8_block_size
        )
        w2_sf_int = self._shared_expert_sf_to_int(
            deep_gemm, w2_scale, D, shared_inter, self.cfg.shared_fp8_block_size
        )
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
    def _shared_expert_sf_to_int(deep_gemm, scale, mn, k, block_size):
        if block_size not in _SHARED_RECIPES:
            raise ValueError(f"MegaMoESE unsupported shared block size {block_size}")
        if scale.dtype == torch.int32:
            if block_size == 32 and (
                tuple(scale.shape) != (mn, (k + 127) // 128) or scale.stride(0) != 1
            ):
                raise ValueError("MegaMoESE K32 packed scales must use MN-major layout")
            return scale
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(
                "MegaMoESE expected shared UE8M0 scale, " f"got {scale.dtype}"
            )
        expected_shape = (
            (mn + block_size - 1) // block_size,
            (k + block_size - 1) // block_size,
        )
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"MegaMoESE shared scale shape {tuple(scale.shape)} does not "
                f"match {expected_shape} for block size {block_size}"
            )
        if block_size == 32:
            if k % 32:
                raise ValueError(
                    "MegaMoESE K32 shared weights require K divisible by 32"
                )
            scale_1x32 = scale.float().repeat_interleave(32, dim=0)[:mn].contiguous()
            return deep_gemm.transform_sf_into_required_layout(
                scale_1x32.unsqueeze(0), mn, k, (1, 32), 1
            ).squeeze(0)
        return deep_gemm.transform_sf_into_required_layout(
            scale.float(), mn, k, _SHARED_RECIPES[block_size][1:], num_groups=None
        )

    def _buffer_kwargs(self):
        kwargs = super()._buffer_kwargs()
        kwargs["num_shared_experts"] = self._num_shared_experts
        if not self._fp8_routed:
            kwargs.pop("use_fp8_dispatch")
            kwargs["mma_type"] = _MMA_TYPE
        return kwargs

    def _allocate_mega_buffer(self):
        if self.cfg.n_activated_experts > 31:
            raise ValueError(
                "MegaMoE SE requires topk <= 31 because shared output uses one combine slot"
            )
        if self._fp8_routed:
            buf = super()._allocate_mega_buffer()
        else:
            kwargs = self._buffer_kwargs()
            kwargs.pop("mma_type")
            buf = _get_or_create_mega_se_buf(**kwargs)
        actual = getattr(buf, "num_shared_experts", None)
        if actual is not None and int(actual) != self._num_shared_experts:
            raise RuntimeError("DeepGEMM MegaMoE SE buffer/shared weight mismatch")
        return buf

    def _allocate_mega_output(self):
        if self._fp8_routed:
            return super()._allocate_mega_output()
        return _get_or_create_mega_se_output(
            _mega_output_capacity(self._mega_buf, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.bfloat16,
            self._mega_l1_w.device,
        )

    def _block_m(self, tokens: int) -> int:
        import deep_gemm

        args = (
            self.cfg.ep_size,
            self.cfg.n_routed_experts,
            self._mega_buf.num_max_tokens_per_rank,
            int(tokens),
            self.cfg.n_activated_experts,
        )
        if self._fp8_routed:
            return int(deep_gemm.get_block_m_for_mega_moe_fp8(*args))
        return int(deep_gemm.get_block_m_for_mega_moe(*args, _MMA_TYPE))

    def _stage_kernel_inputs(self, tokens, *, inputs_prepacked):
        stage_shared_fp8_input_scales(
            self._mega_buf.x_sf,
            self._mega_buf.shared_l1_acts_sf,
            tokens,
            self._block_m(tokens),
        )

    def _resolve_jit_warmup_token_counts(
        self, num_sms: int, deep_gemm=None
    ) -> list[int]:
        if self._prepared_policy is not None:
            return super()._resolve_jit_warmup_token_counts(num_sms, deep_gemm)
        cfg = self.cfg
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_mega_moe_se_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens_per_rank)
        get_block_m = getattr(deep_gemm, "get_block_m_for_mega_moe", None)
        block_m_resolver = None
        if callable(get_block_m):
            block_m_resolver = lambda tokens: get_block_m(
                cfg.ep_size,
                cfg.n_routed_experts,
                max_tokens_per_rank,
                tokens,
                cfg.n_activated_experts,
                _MMA_TYPE,
            )
        return generate_mega_moe_se_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=_mega_intermediate_size(cfg),
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
            include_cap=cfg.warmup_include_capacity,
            block_m_resolver=block_m_resolver,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if self._prepared_policy is not None:
            return super()._maybe_warmup_jit_once()
        if not mega_moe_se_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoESE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = resolve_mega_num_sms(deep_gemm, self._mega_l1_w.device)
        token_counts = self._resolve_jit_warmup_token_counts(num_sms, deep_gemm)
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
            cfg.route_scale,
            num_sms,
            self._shared_recipe,
            tuple(token_counts),
            self.supports_gate_pack,
        )
        if warmup_key in _MEGA_MOE_SE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[MegaMoESE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d shared=%d route_scale=%s num_sms=%d "
                "shared_recipe=%s",
                cfg.layer_id,
                format_token_counts(token_counts),
                cfg.max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_shared_experts,
                cfg.route_scale,
                num_sms,
                self._shared_recipe,
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
        if self._prepared_policy is not None:
            return super()._launch(y, tokens, device)

        import deep_gemm

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
            shared_recipe=self._shared_recipe,
        )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        kwargs = (
            {"assume_all_topk_valid": True}
            if self._fp8_routed
            else {"shared_recipe": _SHARED_RECIPE}
        )
        getattr(deep_gemm, self._kernel_name)(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=_ROUTED_RECIPE,
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
            **kwargs,
        )

    def forward(self, x, weights, indices):
        """Return BF16 routed+shared output; participate even for local T=0."""

        if self._prepared_policy is not None:
            return super().forward(x, weights, indices)

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


class MegaMoeFusedBackend(MegaMoeExecutor):
    """Adapter for the legacy fused kernel's separate shared-workspace ABI."""

    includes_shared_expert = True
    _fp4_transform_name = "transform_weights_for_mega_moe_fused"
    _defer_jit_until_shared_weights = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        super().check_conditions(checker, config)
        try:
            import deep_gemm
        except ImportError:
            checker.check(False)
            return

        checker.check(hasattr(deep_gemm, cls._fp4_transform_name))
        checker.check(
            hasattr(deep_gemm, "transform_shared_expert_weights_for_mega_moe_fused")
        )

    def _allocate_mega_buffer(self):
        return get_or_create_mega_buf_fused(**self._buffer_kwargs())

    def setup_shared_expert_from_fp8(
        self,
        w1_w: torch.Tensor,  # [2*inter, dim] float8_e4m3fn (gate||up stacked on N)
        w1_s: torch.Tensor,  # float32 / e8m0 block SF, or packed int32
        w2_w: torch.Tensor,  # [dim, inter] float8_e4m3fn
        w2_s: torch.Tensor,  # float32 / e8m0 block SF, or packed int32
    ) -> None:
        """Setup pre-quantized FP8 per-block shared-expert weights for fused MegaMoE."""
        cfg = self.cfg
        (l1_w, l1_sf), (l2_w, l2_sf) = transform_shared_expert_fp8_for_fused(
            w1_w,
            w1_s,
            w2_w,
            w2_s,
            dim=cfg.dim,
            inter=cfg.moe_inter_dim,
        )
        self._shared_l1_w = l1_w
        self._shared_l1_sf = l1_sf
        self._shared_l2_w = l2_w
        self._shared_l2_sf = l2_sf
        self._setup_shared_expert_workspace()

    def _setup_shared_expert_workspace(self) -> None:
        if self._mega_buf is None:
            raise RuntimeError(
                "setup routed MegaMoE weights before shared expert weights"
            )
        cfg = self.cfg
        device = self._mega_l1_w.device
        capacity = _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank)
        self._shared_mid_fp8, self._shared_mid_sf = make_shared_mid_workspace(
            capacity,
            cfg.moe_inter_dim,
            device,
        )

    def _check_shared_expert_ready(self) -> None:
        if (
            self._shared_l1_w is None
            or self._shared_l1_sf is None
            or self._shared_l2_w is None
            or self._shared_l2_sf is None
            or self._shared_mid_fp8 is None
            or self._shared_mid_sf is None
        ):
            raise RuntimeError("shared expert FP8 weights/workspace are not set up")

    def _validate_capacity(self, tokens):
        super()._validate_capacity(tokens)
        if tokens > self._shared_mid_fp8.size(0):
            raise RuntimeError(
                f"MegaMoE shared workspace rows={self._shared_mid_fp8.size(0)} "
                f"is smaller than input tokens={tokens}"
            )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        deep_gemm.fp8_fp4_mega_moe_fused(
            y,
            self._shared_l1_w,
            self._shared_l1_sf,
            self._shared_l2_w,
            self._shared_l2_sf,
            self._shared_mid_fp8,
            self._shared_mid_sf,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
        )


__all__ = ["MegaMoeSEExecutor", "MegaMoeFusedBackend"]
