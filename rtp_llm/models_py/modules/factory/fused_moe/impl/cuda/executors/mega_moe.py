"""DeepGEMM fused expert executor for FP4 and FP8 weight recipes.

EP > 1 only. The Mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM +
combine into one kernel backed by a PyTorch symmetric-memory buffer for
NVLink communication. Requires SM100, PyTorch ≥ 2.9 (symmetric_memory),
DeepGEMM ≥ 2.5, and an initialised process group.

Selected by the fused-MoE factory when ``ep_size > 1`` and the Mega kernel is
available.
"""

from __future__ import annotations

import copy
import logging
import os
from functools import lru_cache
from typing import Any, Dict

import torch

from rtp_llm.config.mega_moe import FP8_MEGA_MOE_STRATEGIES
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_buf,
    _get_or_create_mega_output,
    _mega_moe_available,
    estimate_mega_moe_symm_buffer_bytes,
    get_or_create_mega_clone_buffer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
    resolve_mega_num_sms,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.topology import (
    validated_world_ep_group as _get_validated_world_ep_group,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)
from rtp_llm.utils.model_weight import W

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

_MEGA_MOE_JIT_WARMED_KEYS: set[tuple] = set()
_MEGA_MOE_NVCC_TMPDIR_ENV = "MEGA_MOE_NVCC_TMPDIR"
_MEGA_INTERMEDIATE_ALIGNMENT = 16 * FP4_BLOCK
_MMA_TYPE = "fp8xfp4"


@lru_cache(maxsize=None)
def _native_mega_intermediate_supported(
    group_size: int, num_experts: int, num_topk: int, hidden: int, intermediate: int
) -> bool:
    return (
        estimate_mega_moe_symm_buffer_bytes(
            group_size, num_experts, 384, num_topk, hidden, intermediate
        )
        is not None
    )


def _mega_intermediate_size(cfg) -> int:
    """Intermediate width the Mega kernel and its resident weights use.

    DeepGEMM bounds the symmetric buffer by a 384-token staging capacity. When
    the checkpoint intermediate exceeds what fits, the routed FP4 buffers are
    padded up to ``16 * FP4_BLOCK`` so the kernel sees a legal tile.
    """
    intermediate = int(cfg.moe_inter_dim)
    # Aligned widths never pad; unaligned ones pad only when DeepGEMM cannot
    # size a buffer for the native width.
    if intermediate % _MEGA_INTERMEDIATE_ALIGNMENT == 0 or (
        _native_mega_intermediate_supported(
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
            cfg.dim,
            intermediate,
        )
    ):
        return intermediate
    return (
        (intermediate + _MEGA_INTERMEDIATE_ALIGNMENT - 1)
        // _MEGA_INTERMEDIATE_ALIGNMENT
        * _MEGA_INTERMEDIATE_ALIGNMENT
    )


def _mega_output_capacity(buf, requested_capacity: int) -> int:
    """Output rows must cover DeepGEMM's internally aligned token capacity."""
    capacity = max(int(requested_capacity), 1)
    aligned_capacity = getattr(buf, "num_max_tokens_per_rank", None)
    if aligned_capacity is not None:
        capacity = max(capacity, int(aligned_capacity))
    return capacity


def _mega_moe_rank_nvcc_tmpdir(rank: int) -> str:
    from rtp_llm.models_py.utils.deep_gemm_scratch import rank_nvcc_tmpdir

    return rank_nvcc_tmpdir(
        rank, "rtp_llm_mega_moe_nvcc", os.environ.get(_MEGA_MOE_NVCC_TMPDIR_ENV)
    )


def _activate_mega_moe_rank_nvcc_tmpdir(rank: int) -> tuple[str, str | None]:
    """Use a rank-local nvcc temp dir during MegaMoE warmup compilation."""

    previous_tmpdir = os.environ.get("TMPDIR")
    tmpdir = _mega_moe_rank_nvcc_tmpdir(rank)
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except Exception:
        tmpdir = os.path.join(
            "/tmp",
            "rtp_llm_mega_moe_nvcc",
            f"rank_{int(rank)}",
        )
        os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    return tmpdir, previous_tmpdir


def _restore_tmpdir(previous_tmpdir: str | None) -> None:
    if previous_tmpdir is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = previous_tmpdir




class MegaMoeExecutor(Fp8Fp4ExecutorBase):
    execute_empty_inputs = True
    _fp4_transform_name = "transform_weights_for_mega_moe"

    _defer_buffer_until_shared_weights = False
    _defer_jit_until_shared_weights = False

    @staticmethod
    def _kernel_spec(config):
        strategy = getattr(config, "moe_strategy", "mega_moe")
        if strategy in FP8_MEGA_MOE_STRATEGIES:
            return "fp8_fp8_mega_moe", "get_symm_buffer_for_mega_moe_fp8"
        if strategy == "mega_moe_fused":
            return "fp8_fp4_mega_moe_fused", "get_symm_buffer_for_mega_moe_fused"
        return "fp8_fp4_mega_moe", "get_symm_buffer_for_mega_moe"

    @property
    def _fp8_routed(self):
        return getattr(self.cfg, "moe_strategy", "mega_moe") in FP8_MEGA_MOE_STRATEGIES

    @property
    def _kernel_name(self):
        return self._kernel_spec(self.cfg)[0]

    @property
    def _buffer_factory(self):
        return self._kernel_spec(self.cfg)[1]

    def __init__(self, config, quant_config, weights):
        self._mega_l1_w = self._mega_l1_sf = None
        self._mega_l2_w = self._mega_l2_sf = None
        self._shared_l1_w = self._shared_l1_sf = None
        self._shared_l2_w = self._shared_l2_sf = None
        self._shared_mid_fp8 = self._shared_mid_sf = None
        self._mega_buf = self._mega_y = self._mega_group = None
        self._input_packer = None
        self._num_shared_experts = 0
        super().__init__(config, quant_config, weights)

    @property
    def _prepared_policy(self):
        return getattr(self.cfg, "mega_moe_policy", None)

    @property
    def _activation_clamp(self):
        if self._prepared_policy is not None:
            return self._prepared_policy.activation_clamp
        limit = self.cfg.swiglu_limit
        return limit if limit > 0 else None

    @property
    def _fast_math(self):
        return self._prepared_policy.fast_math if self._prepared_policy else True

    def _new_input_packer(self):
        return get_mega_moe_input_packer(prepared=self._prepared_policy is not None)

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        execute = super().execute
        args = (
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )
        if getattr(self.cfg, "mega_moe_policy", None) is None:
            return execute(payload, *args)
        from .mega_moe_runtime import execute_prepared_mega_moe

        return execute_prepared_mega_moe(self, payload, execute, *args)

    @property
    def supports_gate_pack(self) -> bool:
        return (
            self._prepared_policy is None
            and self._input_packer.name == "fused"
            and 1 <= self.cfg.n_activated_experts <= 32
        )

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        prepared = getattr(config, "mega_moe_policy", None) is not None
        if prepared:
            checker.check(config.ep_size >= 1)
            checker.check(not config.has_redundant_experts)
        else:
            super().check_conditions(checker, config)
            checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        if prepared:
            kernel, allocator = cls._kernel_spec(config)
            checker.check(
                _mega_moe_available(
                    kernel_name=kernel,
                    allocator_name=allocator,
                    allow_single_rank=True,
                )
            )
        else:
            checker.check(_mega_moe_available())

    def setup_weights(self, layer_weights: Dict) -> None:
        """Keep the MegaMoE-resident weights and register the symm-mem buffer."""
        import deep_gemm
        import torch.distributed as dist

        if self._prepared_policy is not None:
            from .mega_moe_runtime import setup_prepared_mega_moe

            setup_prepared_mega_moe(self, layer_weights)
            return

        cfg = self.cfg
        D = cfg.dim
        inter = _mega_intermediate_size(cfg)
        device = self._transform_routed_weights(layer_weights, deep_gemm, inter)

        self._setup_buffer_and_warmup()

    def setup_weights_from_fp4(
        self,
        w1_w: torch.Tensor,  # [E_local, 2*inter, dim//2] int8 (FP4 packed, gate+up)
        w1_s: torch.Tensor,  # [E_local, 2*inter, dim//FP4_BLOCK] float8_e8m0fnu
        w2_w: torch.Tensor,  # [E_local, dim, inter//2] int8 (FP4 packed)
        w2_s: torch.Tensor,  # [E_local, dim, inter//FP4_BLOCK] float8_e8m0fnu
        w1_layout: str = "gate_up",
    ) -> None:
        """Setup weights from pre-quantized FP4 format (same as DSv4 checkpoint).

        If your checkpoint already has FP4 weights (int8 packed + UE8M0 scales),
        use this method directly.
        """
        from rtp_llm.utils.model_weight import W

        self._setup_fp4_weights(
            {W.moe_w1: w1_w, W.moe_s1: w1_s, W.moe_w2: w2_w, W.moe_s2: w2_s},
            w1_layout,
        )

    def _setup_fp4_weights(self, layer_weights, w1_layout):
        """Consume each checkpoint stack before allocating runtime workspaces.

        Taking ownership through the mapping keeps native FP4 loading from
        retaining every original weight/scale tensor on a caller's argument
        stack while symmetric arenas and JIT workspaces are allocated.
        """
        if self._fp8_routed:
            raise ValueError("FP8 MegaMoE only accepts FP8 per-block weights")
        import deep_gemm

        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.weights import (
            _interleave_stacked_up_gate,
        )
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        E = cfg.n_local_experts
        inter = cfg.moe_inter_dim

        w1_w = layer_weights.pop(W.moe_w1)
        if w1_w.size(-2) != 2 * inter:
            raise ValueError(f"W.moe_w1 has {w1_w.size(-2)} rows, expected {2 * inter}")
        if w1_layout not in ("gate_up", "up_gate"):
            raise ValueError(f"Unsupported FP4 W1 layout: {w1_layout}")
        # Release each raw scale stack before transforming the next one.
        s13_int = prepare_fp4_weight_scale_for_deepgemm(
            layer_weights.pop(W.moe_s1), 2 * inter, D, E
        )
        torch.cuda.empty_cache()
        w2_w = layer_weights.pop(W.moe_w2)
        s2_int = prepare_fp4_weight_scale_for_deepgemm(
            layer_weights.pop(W.moe_s2), D, inter, E
        )
        torch.cuda.empty_cache()

        # Apply mega MoE transform: L1 gate/up interleave + UTCCP transpose SF.
        # RTP checkpoints store L1 as [up | gate].  Transforming that layout
        # directly avoids split, restack, and torch.stack copies of the full
        # per-layer expert tensor during model startup.
        if (
            w1_layout == "up_gate"
            and self._fp4_transform_name == "transform_weights_for_mega_moe"
        ):
            from deep_gemm.mega import _transpose_sf_for_utccp

            l1_w = _interleave_stacked_up_gate(w1_w)
            l1_sf = _transpose_sf_for_utccp(_interleave_stacked_up_gate(s13_int))
            l2_w = w2_w
            l2_sf = _transpose_sf_for_utccp(s2_int)
        else:
            if w1_layout == "up_gate":
                w1_w = torch.cat(w1_w.chunk(2, dim=1)[::-1], dim=1)
                s13_int = torch.cat(s13_int.chunk(2, dim=1)[::-1], dim=1)
            (l1_w, l1_sf), (l2_w, l2_sf) = getattr(deep_gemm, self._fp4_transform_name)(
                (w1_w, s13_int),
                (w2_w, s2_int),
            )
        del w1_w, w2_w, s13_int, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        self._setup_buffer_and_warmup()

    def setup_shared_expert_from_fp8(self, w1_w, w1_s, w2_w, w2_s):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights import (
            prepare_shared_expert_weights,
        )

        weights = prepare_shared_expert_weights(
            self.cfg,
            w1_w,
            w1_s,
            w2_w,
            w2_s,
            fp8_routed=self._fp8_routed,
            allow_requant=self._prepared_policy is not None,
            expected_num_experts=(
                None if self._prepared_policy is not None else self.cfg.n_shared_experts
            ),
        )
        self._shared_l1_w, self._shared_l1_sf = weights.l1
        self._shared_l2_w, self._shared_l2_sf = weights.l2
        self._num_shared_experts = weights.num_experts
        self._setup_buffer_and_warmup()

    def _check_shared_expert_ready(self):
        if any(
            getattr(self, name, None) is None
            for name in (
                "_shared_l1_w",
                "_shared_l1_sf",
                "_shared_l2_w",
                "_shared_l2_sf",
            )
        ):
            raise RuntimeError("MegaMoE shared expert weights are not set up")

    def maybe_warmup_fused_shared_jit_once(self):
        self._check_shared_expert_ready()
        self._maybe_warmup_jit_once()

    def _buffer_kwargs(self):
        cfg = self.cfg
        return dict(
            group=self._mega_group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=(
                cfg.moe_inter_dim if self._prepared_policy else _mega_intermediate_size(cfg)
            ),
            use_fp8_dispatch=True,
            activation="swiglu",
        )

    def _allocate_mega_buffer(self):
        if self._fp8_routed:
            from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_buffer import (
                get_or_create_mega_buf_fp8,
            )

            return get_or_create_mega_buf_fp8(**self._buffer_kwargs())
        return _get_or_create_mega_buf(**self._buffer_kwargs())

    def _allocate_mega_output(self):
        return _get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.bfloat16,
            self._mega_l1_w.device,
        )

    def _setup_buffer_and_warmup(self) -> None:
        """Allocate one aligned collective arena and warm the selected recipe."""
        if self._defer_buffer_until_shared_weights and self._num_shared_experts == 0:
            return
        import torch.distributed as dist

        self._mega_group = _get_validated_world_ep_group(self.cfg, dist)
        self._mega_buf = self._allocate_mega_buffer()
        self._mega_y = self._allocate_mega_output()
        self._input_packer = self._new_input_packer()
        if not self._defer_jit_until_shared_weights:
            self._maybe_warmup_jit_once()

    def setup_weights_from_fp8(self, *args, **kwargs) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.weights import (
            transform_fp8_mega_weights,
            transform_fp8_to_fp4_mega_weights,
        )

        transform = (
            transform_fp8_mega_weights
            if self._fp8_routed
            else transform_fp8_to_fp4_mega_weights
        )
        if not self._fp8_routed:
            kwargs["transform_name"] = self._fp4_transform_name
        l1, l2 = transform(self.cfg, *args, **kwargs)
        self._mega_l1_w, self._mega_l1_sf = l1
        self._mega_l2_w, self._mega_l2_sf = l2
        self._setup_buffer_and_warmup()

    def _transform_routed_weights(
        self, layer_weights: Dict, deep_gemm, inter: int
    ) -> torch.device:
        """Stack EP-local routed-expert SFs into the int32 UTCCP-transposed
        layout ``fp8_fp4_mega_moe`` expects.

        Routed weights arrive as already-EP-sliced stacks (the loader handles
        rank slicing), each shaped ``[E_local, ...]``. We pop them so the only references
        kept alive are the kernel-consumable l1/l2 buffers below.

        Mega MoE expects, per expert:
          L1 w [2*inter, dim//2] int8 (gate | up rows concatenated)
          L1 sf [2*inter, ...] int32  (post-``transform_sf_into_required_layout``
            + ``transform_weights_for_mega_moe``: gate/up interleaved gran=8
            along N, SF UTCCP-transposed)
          L2 w [dim, inter//2] int8
          L2 sf [dim, ...] int32

        When ``inter`` is padded above the checkpoint width, the tail of L1 and
        L2 is zero-filled with neutral UE8M0 scales, so the extra columns
        contribute exactly zero to the fused SwiGLU.

        Memory: serialise L1 → L2 with ``del`` + ``empty_cache()`` between
        stages. Pre-allocating both fp32 SF stacks at once also keeps the
        transform's temporary allocations alive. Splitting keeps the live set
        to at most one input stack.
        """
        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        source_inter = cfg.moe_inter_dim

        w13 = layer_weights.pop(W.moe_w1)
        s13_raw = layer_weights.pop(W.moe_s1)
        w13, s13_raw = normalize_moe_w13_gate_up(
            w13,
            s13_raw,
            source_inter,
            cfg.moe_w1_layout,
        )
        device = w13.device
        if inter != source_inter:
            padded_w13 = torch.zeros(
                (E, 2 * inter, D // 2), dtype=torch.int8, device=device
            )
            padded_s13 = torch.empty(
                (E, 2 * inter, D // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            padded_s13.view(torch.uint8).fill_(127)
            padded_w13[:, :source_inter].copy_(w13[:, :source_inter])
            padded_s13[:, :source_inter].copy_(s13_raw[:, :source_inter])
            padded_w13[:, inter : inter + source_inter].copy_(w13[:, source_inter:])
            padded_s13[:, inter : inter + source_inter].copy_(s13_raw[:, source_inter:])
            del w13, s13_raw
            w13, s13_raw = padded_w13, padded_s13
        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        w2 = layer_weights.pop(W.moe_w2)
        s2_raw = layer_weights.pop(W.moe_s2)
        if inter != source_inter:
            padded_w2 = torch.zeros((E, D, inter // 2), dtype=torch.int8, device=device)
            padded_s2 = torch.empty(
                (E, D, inter // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            padded_s2.view(torch.uint8).fill_(127)
            padded_w2[:, :, : source_inter // 2].copy_(w2)
            padded_s2[:, :, : source_inter // FP4_BLOCK].copy_(s2_raw)
            del w2, s2_raw
            w2, s2_raw = padded_w2, padded_s2
        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        # Mega MoE transform: L1 gate/up interleave (gran=8 along N) +
        # both SFs UTCCP-transposed. Drop inputs immediately after.
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()

        # Stash as plain attributes (not Parameters — the kernel reads
        # raw int8/int32 buffers with no autograd).  Original stacked
        # fp32 SFs are dropped now that the int layout has been derived.
        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        return device

    def _resolve_jit_warmup_token_counts(
        self, num_sms: int, deep_gemm=None
    ) -> list[int]:
        cfg = self.cfg
        # Use the logical model/runtime token cap, not DeepGEMM's internally
        # aligned buffer capacity.  The JIT key is driven by request-visible T
        # buckets; the aligned capacity only needs to be large enough to hold
        # those representatives.
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = (
            parse_mega_moe_jit_warmup_tokens_override()
            if self._prepared_policy is None
            else None
        )
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
        return generate_mega_moe_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=_mega_intermediate_size(cfg),
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
            include_cap=self._prepared_policy is not None or cfg.warmup_include_capacity,
            block_m_resolver=block_m_resolver,
        )

    def _jit_warmup_enabled(self) -> bool:
        return self._prepared_policy is not None or mega_moe_jit_warmup_enabled()

    def _maybe_warmup_jit_once(self) -> None:
        if not self._jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = resolve_mega_num_sms(deep_gemm, self._mega_l1_w.device)
        token_counts = self._resolve_jit_warmup_token_counts(num_sms, deep_gemm)
        if not token_counts:
            return

        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        warmup_key = (
            type(self).__name__,
            self._kernel_name,
            id(self._mega_group),
            str(self._mega_l1_w.device),
            self._num_shared_experts,
            self._prepared_policy is not None,
            self._activation_clamp,
            self._fast_math,
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            max_tokens_per_rank,
            getattr(cfg, "route_scale", 1.0),
            num_sms,
            tuple(token_counts),
            self.supports_gate_pack,
        )
        if warmup_key in _MEGA_MOE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[MegaMoE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d route_scale=%g num_sms=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                getattr(cfg, "route_scale", 1.0),
                num_sms,
            )
        tmpdir, previous_tmpdir = _activate_mega_moe_rank_nvcc_tmpdir(rank)
        try:
            if rank == 0:
                logging.info("[MegaMoE] rank-local nvcc TMPDIR=%s", tmpdir)
            self.warmup_jit(token_counts)
        finally:
            _restore_tmpdir(previous_tmpdir)
        _MEGA_MOE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[MegaMoE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        """Compile MegaMoE JIT buckets with synthetic rank-local tokens."""
        if not self._jit_warmup_enabled():
            return
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(token_counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        weights = torch.zeros(
            (max_tokens, cfg.n_activated_experts),
            dtype=torch.float32,
            device=device,
        )
        local_expert_ids = cfg.local_expert_start + torch.arange(
            cfg.n_activated_experts, dtype=torch.long, device=device
        ) % max(cfg.n_local_experts, 1)
        indices = local_expert_ids.view(1, -1).expand(max_tokens, -1).contiguous()
        if self.supports_gate_pack:
            scores = torch.zeros(
                (max_tokens, cfg.n_routed_experts),
                dtype=torch.bfloat16,
                device=device,
            )
            bias = torch.zeros(
                cfg.n_routed_experts,
                dtype=torch.float32,
                device=device,
            )
            input_ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
            tid2eid = local_expert_ids.view(1, -1).contiguous()

        for token_count in token_counts:
            dist.barrier()
            self.forward(
                x[:token_count],
                weights[:token_count],
                indices[:token_count],
            )
            torch.cuda.synchronize(device)
            if self.supports_gate_pack:
                self.forward_gate_pack(
                    x[:token_count],
                    ExpertGatePayload(
                        scores=scores[:token_count],
                        topk=cfg.n_activated_experts,
                        score_func="sqrtsoftplus",
                        route_scale=cfg.route_scale,
                        bias=bias,
                    ),
                )
                torch.cuda.synchronize(device)
                self.forward_gate_pack(
                    x[:token_count],
                    ExpertGatePayload(
                        scores=scores[:token_count],
                        topk=cfg.n_activated_experts,
                        score_func="sqrtsoftplus",
                        route_scale=cfg.route_scale,
                        input_ids=input_ids[:token_count],
                        tid2eid=tid2eid,
                    ),
                )
                torch.cuda.synchronize(device)
        dist.barrier()

    def _validate_capacity(self, tokens: int) -> None:
        if self.includes_shared_expert:
            self._check_shared_expert_ready()
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE input tokens={tokens} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE output buffer rows={self._mega_y.size(0)} is smaller "
                f"than input tokens={tokens}. This indicates inconsistent aligned "
                "MegaMoE buffer sizing."
            )

    def _launch(self, y: torch.Tensor, tokens: int, device: torch.device) -> None:
        sync_cuda_graph_warmup_ranks(
            f"moe.mega_moe.layer{self.cfg.layer_id}.before_deepgemm",
            device,
        )
        self._run_kernel(y, tokens)

    def _run_kernel(self, y: torch.Tensor, tokens: int) -> None:
        import deep_gemm

        kwargs = {"assume_all_topk_valid": True} if self._fp8_routed else {}
        getattr(deep_gemm, self._kernel_name)(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
            **kwargs,
        )

    def forward(self, x, weights, indices) -> torch.Tensor:
        return self._forward_impl(x, weights, indices, inputs_prepacked=False)

    def _pack_inputs(self, x, weights, indices, tokens: int) -> None:
        self._input_packer.pack(x, weights, indices, self._mega_buf, tokens)

    def _stage_kernel_inputs(self, tokens: int, *, inputs_prepacked: bool) -> None:
        """Recipe hook for shared-expert activation scales."""

    def _forward_impl(self, x, weights, indices, *, inputs_prepacked: bool):
        tokens = int(x.size(0))
        self._validate_capacity(tokens)
        if not inputs_prepacked:
            if weights is None or indices is None:
                raise ValueError("weights and indices are required before packing")
            self._pack_inputs(x, weights, indices, tokens)
        self._stage_kernel_inputs(tokens, inputs_prepacked=inputs_prepacked)
        # Zero-token EP ranks must still enter the collective as expert hosts.
        # Skipping a rank loses remote expert contributions and hangs its peers.
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y

    def prepacked_input_views(self, tokens: int):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            planned_moe_tokens,
        )

        tokens = int(tokens)
        buf = self._mega_buf
        capacity = int(buf.num_max_tokens_per_rank)
        planned = planned_moe_tokens()
        if planned is not None and planned > capacity:
            raise ValueError(
                "prepacked MegaMoE inputs cannot use a multi-chunk EP plan"
            )
        if tokens <= 0 or tokens > capacity:
            raise ValueError(f"tokens must be in [1,{capacity}], got {tokens}")
        self._validate_capacity(tokens)
        return (
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
        )

    def forward_prepacked(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, None, None, inputs_prepacked=True)

    def _clone_mega_buffer(self):
        return get_or_create_mega_clone_buffer(
            self._mega_buf, self._buffer_factory, **self._buffer_kwargs()
        )

    def clone_for_cuda_graph(self, *, share_mega_buf: bool = False):
        """Share immutable weights and isolate output/workspace storage.

        Sharing the EP arena requires ordered captures/replays across all peers;
        these clones must not execute concurrently on independent streams.
        """
        clone = copy.copy(self)
        clone._parameters = self._parameters.copy()
        clone._buffers = self._buffers.copy()
        clone._modules = self._modules.copy()
        if share_mega_buf:
            if self._mega_buf is None or self._mega_group is None:
                raise RuntimeError("Shared MegaMoE buffer must be initialized")
            clone._mega_buf = self._mega_buf
        else:
            clone._mega_buf = self._clone_mega_buffer()
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = self._new_input_packer()
        for name in ("_shared_mid_fp8", "_shared_mid_sf"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(clone, name, torch.empty_like(value))
        return clone

    def forward_gate_pack(
        self,
        x: torch.Tensor,
        gate_payload: ExpertGatePayload,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route and pack directly into the MegaMoE symmetric buffer."""

        if not self.supports_gate_pack:
            raise RuntimeError("MegaMoE fused gate packing requires the fused packer")
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
        self._stage_kernel_inputs(tokens, inputs_prepacked=True)
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y, buf.topk_weights[:tokens], buf.topk_idx[:tokens]
