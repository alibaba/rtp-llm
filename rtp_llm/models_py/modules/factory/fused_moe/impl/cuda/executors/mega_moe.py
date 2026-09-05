"""DeepGEMM ``fp8_fp4_mega_moe`` fused expert executor.

EP > 1 only. The Mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM +
combine into one kernel backed by a PyTorch symmetric-memory buffer for
NVLink communication. Requires SM100, PyTorch ≥ 2.9 (symmetric_memory),
DeepGEMM ≥ 2.5, and an initialised process group.

Selected by the fused-MoE factory when ``ep_size > 1`` and the Mega kernel is
available.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    strict_fused_moe_enabled,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_buf,
    _get_or_create_mega_output,
    _mega_moe_available,
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
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)

from .fp8_fp4_base import Fp8Fp4ExecutorBase

_MEGA_MOE_JIT_WARMED_KEYS: set[tuple] = set()
_MEGA_MOE_NVCC_TMPDIR_ENV = "MEGA_MOE_NVCC_TMPDIR"
_PRE_KERNEL_BARRIER_ENV = "MEGA_MOE_PRE_KERNEL_BARRIER"
_PRE_KERNEL_BARRIER_VERBOSE_ENV = "MEGA_MOE_PRE_KERNEL_BARRIER_VERBOSE"
_PRE_KERNEL_BARRIER_LOGGED_KEYS: set[tuple[int, int]] = set()


def _mega_output_capacity(buf, requested_capacity: int) -> int:
    """Output rows must cover DeepGEMM's internally aligned token capacity."""
    capacity = max(int(requested_capacity), 1)
    aligned_capacity = getattr(buf, "num_max_tokens_per_rank", None)
    if aligned_capacity is not None:
        capacity = max(capacity, int(aligned_capacity))
    return capacity


def _mega_moe_rank_nvcc_tmpdir(rank: int) -> str:
    base_dir = (
        os.environ.get(_MEGA_MOE_NVCC_TMPDIR_ENV)
        or os.environ.get("DG_JIT_CACHE_DIR")
        or os.environ.get("TRITON_CACHE_DIR")
        or "/tmp"
    )
    return os.path.join(
        base_dir,
        "rtp_llm_mega_moe_nvcc",
        f"rank_{int(rank)}",
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


def _pre_kernel_barrier_enabled() -> bool:
    return os.environ.get(_PRE_KERNEL_BARRIER_ENV, "0") == "1"


def _pre_kernel_barrier_verbose_enabled() -> bool:
    return os.environ.get(_PRE_KERNEL_BARRIER_VERBOSE_ENV, "0") == "1"


def _log_pre_kernel_barrier(
    phase: str,
    layer_id: int,
    rank: int,
    world_size: int,
    tokens: int,
    device: torch.device,
) -> None:
    if _pre_kernel_barrier_verbose_enabled():
        logging.info(
            "[MegaMoE] pre-kernel barrier %s: layer=%d rank=%d/%d "
            "tokens=%d device=%s",
            phase,
            layer_id,
            rank,
            world_size,
            tokens,
            device,
        )
        return

    if phase != "enter":
        return
    key = (layer_id, rank)
    if key in _PRE_KERNEL_BARRIER_LOGGED_KEYS:
        return
    _PRE_KERNEL_BARRIER_LOGGED_KEYS.add(key)
    logging.info(
        "[MegaMoE] pre-kernel barrier enabled: layer=%d rank=%d/%d "
        "tokens=%d device=%s; set %s=1 to log every barrier",
        layer_id,
        rank,
        world_size,
        tokens,
        device,
        _PRE_KERNEL_BARRIER_VERBOSE_ENV,
    )


def _get_validated_world_ep_group(cfg, dist):
    if not dist.is_initialized():
        raise RuntimeError("MegaMoE requires torch.distributed to be initialized")
    group = dist.group.WORLD
    actual_size = dist.get_world_size(group)
    actual_rank = dist.get_rank(group)
    if actual_size != cfg.ep_size or actual_rank != cfg.ep_rank:
        raise RuntimeError(
            "MegaMoE currently requires the EP group to equal WORLD: "
            f"runtime WORLD is rank {actual_rank}/{actual_size}, but "
            f"configuration EP is rank {cfg.ep_rank}/{cfg.ep_size}"
        )
    return group


class MegaMoeExecutor(Fp8Fp4ExecutorBase):
    execute_empty_inputs = True

    @property
    def supports_gate_pack(self) -> bool:
        return self._input_packer.name == "fused"

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(_mega_moe_available())

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-local routed-expert SFs into the int32 UTCCP-transposed
        layout ``fp8_fp4_mega_moe`` expects, then register the symm-mem
        dispatch buffer.

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

        Memory: serialise L1 → L2 with ``del`` + ``empty_cache()`` between
        stages. Pre-allocating both fp32 SF stacks at once also keeps the
        transform's temporary allocations alive. Splitting keeps the live set
        to at most one input stack.
        """
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

        # (4) Allocate the symmetric-memory buffer.  Uses
        # ``torch.distributed.group.WORLD`` because our DP+EP layout has
        # ``ep_size == world_size`` — every rank holds a distinct 64/256
        # slice.  ``num_max_tokens_per_rank`` caps per-rank token count
        # fed into the MoE; bounded from ``max_tokens_per_rank`` (plumbed
        # from the runtime token budget). The library aligns this internally
        # using ``get_token_alignment_for_mega_moe()``.
        group = _get_validated_world_ep_group(cfg, dist)
        self._mega_group = group
        # Symm buffer is single-layer staging — share one across all
        # MoE layers via the module-level cache (see _get_or_create_mega_buf).
        self._mega_buf = _get_or_create_mega_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        # Single-layer staging output. All MoE layers execute sequentially, so one
        # process-local buffer is enough and avoids O(layers) persistent memory.
        self._mega_y = _get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            D,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
        # Use the logical model/runtime token cap, not DeepGEMM's internally
        # aligned buffer capacity.  The JIT key is driven by request-visible T
        # buckets; the aligned capacity only needs to be large enough to hold
        # those representatives.
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_mega_moe_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens_per_rank)
        return generate_mega_moe_jit_token_counts(
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
        if not mega_moe_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return

        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        warmup_key = (
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            max_tokens_per_rank,
            cfg.swiglu_limit,
            num_sms,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[MegaMoE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d num_sms=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
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
        if not mega_moe_jit_warmup_enabled():
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

        for token_count in token_counts:
            dist.barrier()
            self.forward(
                x[:token_count],
                weights[:token_count],
                indices[:token_count],
            )
            torch.cuda.synchronize(device)
        dist.barrier()

    def _validate_capacity(self, tokens: int) -> None:
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
        import deep_gemm

        self._maybe_pre_kernel_barrier(tokens)
        sync_cuda_graph_warmup_ranks(
            f"moe.mega_moe.layer{self.cfg.layer_id}.before_deepgemm",
            device,
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )

    def forward(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in BF16.  The MoE epilogue
        owns the final routed+shared cast.
        """
        T = x.size(0)
        buf = self._mega_buf
        self._validate_capacity(T)

        # ``deep_gemm.fp8_fp4_mega_moe`` is a peer-symmetric NVLink collective:
        # every rank in ``buf.group`` MUST enter the kernel together. The kernel
        # is symmetric — each rank both *dispatches* its ``T`` local tokens to
        # peers' experts AND *hosts* its local-expert slice to compute peers'
        # tokens routed to it.  Skipping a rank with ``T == 0`` (e.g. an EP/CP
        # rank that holds no input tokens for a given batch shape) does two
        # things, both bad:
        #   1. Strands the routed-expert work that peers dispatched to its
        #      local experts -> peers see zero contribution from those experts
        #      (silent wrong output).
        #   2. Triggers NVLink barrier timeout in the surviving peers
        #      (``deep_gemm/include/deep_gemm/comm/barrier.cuh:72``,
        #      ``DG_DEVICE_ASSERT(false and "NVLink barrier timeout")``) ->
        #      kernel-side ``asm("trap;")`` -> SIGTRAP after 30 s.  The trap is
        #      what surfaces as the prod ``CUDA_ERROR_LAUNCH_FAILED`` (719)
        #      cascading from ``sm100_fp8_fp4_mega_moe_impl``.
        # Therefore: always pack and always call the kernel, even when local
        # ``T == 0``.  ``pack`` becomes a no-op (zero-row slices), ``y[:0]``
        # signals ``num_tokens=0`` so this rank's dispatch loop iterates zero
        # times, and the rank still participates as expert host.  No control
        # flow depending on a GPU-side scalar -> CUDA-graph-capture safe.
        self._input_packer.pack(x, weights, indices, buf, T)
        y = self._mega_y[:T]
        self._launch(y, T, x.device)
        return y

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
        y = self._mega_y[:tokens]
        self._launch(y, tokens, x.device)
        return y, buf.topk_weights[:tokens], buf.topk_idx[:tokens]

    def _maybe_pre_kernel_barrier(self, tokens: int) -> None:
        """Optional host-side rendezvous before the DeepGEMM MegaMoE kernel.

        This is a diagnostic guard for cases where one rank does not enter the
        peer-symmetric DeepGEMM kernel in time.  It intentionally synchronizes
        the current stream first so the barrier represents "RTP-side pack is
        done and this rank is ready to launch MegaMoE".
        """
        if not _pre_kernel_barrier_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_PRE_KERNEL_BARRIER_ENV}=1 is incompatible with CUDA graph "
                "capture"
            )

        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                f"{_PRE_KERNEL_BARRIER_ENV}=1 requires torch.distributed "
                "to be initialized"
            )

        cfg = self.cfg
        group = getattr(self, "_mega_group", dist.group.WORLD)
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        device = self._mega_l1_w.device
        _log_pre_kernel_barrier("enter", cfg.layer_id, rank, world_size, tokens, device)

        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.current_stream().synchronize()
                try:
                    dist.barrier(
                        group=group,
                        device_ids=[torch.cuda.current_device()],
                    )
                except TypeError:
                    dist.barrier(group=group)
        else:
            dist.barrier(group=group)

        _log_pre_kernel_barrier("leave", cfg.layer_id, rank, world_size, tokens, device)
