"""MegaMoEStrategy: DeepGEMM ``fp8_fp4_mega_moe`` symm-mem fused kernel.

EP > 1 only. The Mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM +
combine into one kernel backed by a PyTorch symmetric-memory buffer for
NVLink communication. Requires SM100, PyTorch ≥ 2.9 (symmetric_memory),
DeepGEMM ≥ 2.5, and an initialised process group.

Wired into ``MoE`` via ``select_strategy`` when ep_size > 1 and Mega is
available. Direct port of the pre-refactor ``_setup_mega_moe`` +
``_routed_experts_mega_moe`` methods.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, Optional

import torch

from ..._profiler import record_function_range
from ...quant_layouts import FP4_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from ..input_packer import get_mega_moe_input_packer
from ..mega_buf import (
    _get_or_create_mega_buf,
    _get_or_create_mega_output,
    _mega_moe_enabled,
    estimate_mega_moe_symm_buffer_bytes,
)
from ..mega_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
    resolve_mega_num_sms,
)
from ..shared_expert import strict_fused_moe_enabled
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

_MEGA_MOE_JIT_WARMED_KEYS: set[tuple] = set()
_MEGA_MOE_NVCC_TMPDIR_ENV = "DSV4_MEGA_MOE_NVCC_TMPDIR"
_PRE_KERNEL_BARRIER_ENV = "DSV4_MEGA_MOE_PRE_KERNEL_BARRIER"
_PRE_KERNEL_BARRIER_VERBOSE_ENV = "DSV4_MEGA_MOE_PRE_KERNEL_BARRIER_VERBOSE"
_PRE_KERNEL_BARRIER_LOGGED_KEYS: set[tuple[int, int]] = set()
_GATE_PACK_KERNELS = None
_GATE_PACK_KERNELS_UNAVAILABLE = False


def _get_gate_pack_kernels():
    global _GATE_PACK_KERNELS, _GATE_PACK_KERNELS_UNAVAILABLE
    if _GATE_PACK_KERNELS_UNAVAILABLE:
        return None
    if _GATE_PACK_KERNELS is not None:
        return _GATE_PACK_KERNELS
    try:
        from .._mega_gate_pack_triton import (
            fused_mega_moe_gate_pack_hash,
            fused_mega_moe_gate_pack_nonhash,
            fused_mega_moe_gate_pack_supported,
            triton,
        )
    except Exception:
        _GATE_PACK_KERNELS_UNAVAILABLE = True
        return None
    if triton is None:
        _GATE_PACK_KERNELS_UNAVAILABLE = True
        return None
    _GATE_PACK_KERNELS = (
        fused_mega_moe_gate_pack_nonhash,
        fused_mega_moe_gate_pack_hash,
        fused_mega_moe_gate_pack_supported,
    )
    return _GATE_PACK_KERNELS


def _gate_pack_input_packer_env_allows() -> bool:
    mode = os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER", "fused").strip().lower()
    impl = (
        os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER_IMPL", "optimized").strip().lower()
    )
    return mode in ("auto", "fused") and impl == "optimized"


def _mega_output_capacity(buf, requested_capacity: int) -> int:
    """Output rows must cover DeepGEMM's internally aligned token capacity."""
    capacity = max(int(requested_capacity), 1)
    aligned_capacity = getattr(buf, "num_max_tokens_per_rank", None)
    if aligned_capacity is not None:
        capacity = max(capacity, int(aligned_capacity))
    return capacity


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


def _mega_intermediate_size(cfg: MoeCfg) -> int:
    intermediate = int(cfg.moe_inter_dim)
    alignment = 16 * FP4_BLOCK
    if intermediate % alignment == 0 or _native_mega_intermediate_supported(
        cfg.ep_size,
        cfg.n_routed_experts,
        cfg.n_activated_experts,
        cfg.dim,
        intermediate,
    ):
        return intermediate
    return (intermediate + alignment - 1) // alignment * alignment


def _mega_moe_rank_nvcc_tmpdir(rank: int) -> str:
    from rtp_llm.models_py.utils.deep_gemm_scratch import rank_nvcc_tmpdir

    return rank_nvcc_tmpdir(
        rank, "rtp_llm_dsv4_mega_moe_nvcc", os.environ.get(_MEGA_MOE_NVCC_TMPDIR_ENV)
    )


def _activate_mega_moe_rank_nvcc_tmpdir(rank: int) -> tuple[str, str | None]:
    """Use a rank-local nvcc temp dir during MegaMoE warmup compilation."""
    previous_tmpdir = os.environ.get("TMPDIR")
    tmpdir = _mega_moe_rank_nvcc_tmpdir(rank)
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except Exception:
        tmpdir = os.path.join("/tmp", "rtp_llm_dsv4_mega_moe_nvcc", f"rank_{int(rank)}")
        os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    return (tmpdir, previous_tmpdir)


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
            "[DSV4 MegaMoE] pre-kernel barrier %s: layer=%d rank=%d/%d tokens=%d device=%s",
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
        "[DSV4 MegaMoE] pre-kernel barrier enabled: layer=%d rank=%d/%d tokens=%d device=%s; set %s=1 to log every barrier",
        layer_id,
        rank,
        world_size,
        tokens,
        device,
        _PRE_KERNEL_BARRIER_VERBOSE_ENV,
    )


@register_strategy
class MegaMoEStrategy(RoutedExpertsStrategy):
    name = "mega"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and _mega_moe_enabled()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-local routed-expert SFs into the int32 UTCCP-transposed
        layout ``fp8_fp4_mega_moe`` expects, then register the symm-mem
        dispatch buffer.

        Routed weights arrive as already-EP-sliced stacks (loader handles
        the rank slicing): ``layer_weights[W.v4_routed_w{1,2,3}_{w,s}]``
        each shaped ``[E_local, ...]``. We pop them so the only references
        kept alive are the kernel-consumable l1/l2 buffers below.

        Mega MoE expects, per expert:
          L1 w [2*inter, dim//2] int8 (gate | up rows concatenated)
          L1 sf [2*inter, ...] int32  (post-``transform_sf_into_required_layout``
            + ``transform_weights_for_mega_moe``: gate/up interleaved gran=8
            along N, SF UTCCP-transposed)
          L2 w [dim, inter//2] int8
          L2 sf [dim, ...] int32

        Memory: serialise L1 → L2 with ``del`` + ``empty_cache()`` between
        stages. Pre-allocating both fp32 SF stacks at once (and feeding
        the live tuple into ``transform_weights_for_mega_moe`` whose internal
        interleave allocates another ~size(w13)+size(w2) transient) OOMs
        268 GB on V4-Pro cp4. Splitting keeps the live set ≤ one stack.
        """
        self._apply_routed_weight_transform(layer_weights)
        device = self._mega_l1_w.device
        self._mega_runtime_device = device

    def setup_runtime(self) -> None:
        """Allocate shared dispatch/output workspaces and run optional JIT warmup."""
        import torch.distributed as dist

        cfg = self.cfg
        D = cfg.dim
        inter = _mega_intermediate_size(cfg)
        device = self._mega_runtime_device
        assert (
            dist.is_initialized()
        ), "Mega MoE requires torch.distributed initialised; _mega_moe_available() should have gated this earlier"
        group = dist.group.WORLD
        self._mega_group = group
        self._mega_buf_kwargs = dict(
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        self._mega_out_hidden = D
        self._mega_out_capacity_tokens = cfg.max_tokens_per_rank
        self._mega_out_device = device
        self._mega_buf = None
        self._mega_y = None
        self._ensure_mega_buffers()
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def _apply_routed_weight_transform(self, layer_weights: Dict) -> None:
        """Transform checkpoint weights into the resident kernel layout."""
        import deep_gemm

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        source_inter = cfg.moe_inter_dim
        inter = _mega_intermediate_size(cfg)
        padded = inter != source_inter
        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device
        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device
        )
        if padded:
            w13.zero_()
            s13_raw.view(torch.uint8).fill_(127)
        w13[:, :source_inter].copy_(st_w1_w)
        s13_raw[:, :source_inter].copy_(st_w1_s)
        w13[:, inter : inter + source_inter].copy_(st_w3_w)
        s13_raw[:, inter : inter + source_inter].copy_(st_w3_s)
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()
        st_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        st_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device
        )
        if padded:
            w2.zero_()
            s2_raw.view(torch.uint8).fill_(127)
        w2[:, :, : source_inter // 2].copy_(st_w2_w)
        s2_raw[:, :, : source_inter // FP4_BLOCK].copy_(st_w2_s)
        del st_w2_w, st_w2_s
        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int), (w2, s2_int)
        )
        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()

    def _ensure_mega_buffers(self) -> None:
        """Initialize runtime buffers once and preserve their addresses across graph replay."""
        if self._mega_buf is None:
            self._mega_buf = _get_or_create_mega_buf(
                group=self._mega_group, **self._mega_buf_kwargs
            )
        if self._mega_y is None:
            self._mega_y = _get_or_create_mega_output(
                _mega_output_capacity(self._mega_buf, self._mega_out_capacity_tokens),
                self._mega_out_hidden,
                torch.bfloat16,
                self._mega_out_device,
            )

    def _resolve_jit_warmup_token_counts(
        self, num_sms: int, deep_gemm=None
    ) -> list[int]:
        cfg = self.cfg
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_mega_moe_jit_warmup_tokens_override()
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
                "fp8xfp4",
            )
        return generate_mega_moe_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=_mega_intermediate_size(cfg),
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
            block_m_resolver=block_m_resolver,
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
        num_sms = resolve_mega_num_sms(
            deep_gemm, getattr(self, "_mega_runtime_device", None)
        )
        token_counts = self._resolve_jit_warmup_token_counts(num_sms, deep_gemm)
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
            bool(getattr(self, "_gate_pack_warmup_enabled", False)),
            (
                float(getattr(self, "_gate_pack_route_scale", 1.0))
                if getattr(self, "_gate_pack_warmup_enabled", False)
                else None
            ),
        )
        if warmup_key in _MEGA_MOE_JIT_WARMED_KEYS:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE] JIT warmup start: layer=%d tokens=[%s] max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d intermediate=%d num_sms=%d",
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
                logging.info("[DSV4 MegaMoE] rank-local nvcc TMPDIR=%s", tmpdir)
            self.warmup_jit(token_counts)
        finally:
            _restore_tmpdir(previous_tmpdir)
        _MEGA_MOE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE] JIT warmup done: layer=%d tokens=[%s]",
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
            (max_tokens, cfg.n_activated_experts), dtype=torch.float32, device=device
        )
        local_expert_ids = cfg.local_expert_start + torch.arange(
            cfg.n_activated_experts, dtype=torch.long, device=device
        ) % max(cfg.n_local_experts, 1)
        indices = local_expert_ids.view(1, -1).expand(max_tokens, -1).contiguous()
        for token_count in token_counts:
            dist.barrier()
            self.forward(x[:token_count], weights[:token_count], indices[:token_count])
            torch.cuda.synchronize(device)
        dist.barrier()
        self._warmup_gate_pack_jit(token_counts)

    def _warmup_gate_pack_jit(self, token_counts: list[int]) -> None:
        if not getattr(self, "_gate_pack_warmup_enabled", False):
            return
        kernels = _get_gate_pack_kernels()
        if kernels is None:
            return
        fused_mega_moe_gate_pack_nonhash, fused_mega_moe_gate_pack_hash, _ = kernels
        counts = [int(t) for t in token_counts if int(t) > 0]
        if not counts:
            return
        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        scores = torch.zeros(
            (max_tokens, cfg.n_routed_experts), dtype=torch.bfloat16, device=device
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
            fused_mega_moe_gate_pack_nonhash(
                x[:token_count],
                scores[:token_count],
                bias,
                buf.x[:token_count],
                buf.x_sf[:token_count],
                buf.topk_idx[:token_count],
                buf.topk_weights[:token_count],
                route_scale=route_scale,
                norm_eps=1e-12,
            )
            fused_mega_moe_gate_pack_hash(
                x[:token_count],
                scores[:token_count],
                input_ids[:token_count],
                tid2eid,
                buf.x[:token_count],
                buf.x_sf[:token_count],
                buf.topk_idx[:token_count],
                buf.topk_weights[:token_count],
                route_scale=route_scale,
                norm_eps=1e-12,
            )
            torch.cuda.synchronize(device)

    def can_use_gate_pack_static(self, gate) -> bool:
        return (
            os.environ.get("DSV4_GATE_FUSED", "1") != "0"
            and os.environ.get("DSV4_GATE_FP32", "0") != "1"
            and (gate.score_func == "sqrtsoftplus")
            and (1 <= int(gate.topk) <= 32)
            and (self.cfg.dim % 128 == 0)
            and _gate_pack_input_packer_env_allows()
            and (_get_gate_pack_kernels() is not None)
        )

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in BF16.  The MoE epilogue
        owns the final routed+shared cast.
        """
        import deep_gemm

        if self._mega_buf is None or self._mega_y is None:
            self._ensure_mega_buffers()
        T = x.size(0)
        buf = self._mega_buf
        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe.layer{self.cfg.layer_id}.before_deepgemm", x.device
        )
        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y

    def forward_with_gate_pack(
        self, x: torch.Tensor, gate, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Run MegaMoE with router gate + input pack fused together."""
        kernels = _get_gate_pack_kernels()
        fused_mega_moe_gate_pack_nonhash, fused_mega_moe_gate_pack_hash, _ = kernels
        if self._mega_buf is None or self._mega_y is None:
            self._ensure_mega_buffers()
        T = x.size(0)
        buf = self._mega_buf
        y = self._mega_y[:T]
        import deep_gemm

        with record_function_range("dsv4.moe.gate_linear_bf16"):
            scores_bf16 = gate._project_scores(x, gate._weight_bf16())
        with record_function_range("dsv4.moe.mega_gate_pack"):
            if gate.hash:
                fused_mega_moe_gate_pack_hash(
                    x,
                    scores_bf16.contiguous(),
                    input_ids.reshape(-1).contiguous(),
                    gate.tid2eid.contiguous(),
                    buf.x[:T],
                    buf.x_sf[:T],
                    buf.topk_idx[:T],
                    buf.topk_weights[:T],
                    route_scale=float(gate.route_scale),
                    norm_eps=1e-12,
                )
            else:
                fused_mega_moe_gate_pack_nonhash(
                    x,
                    scores_bf16.contiguous(),
                    gate.bias.contiguous(),
                    buf.x[:T],
                    buf.x_sf[:T],
                    buf.topk_idx[:T],
                    buf.topk_weights[:T],
                    route_scale=float(gate.route_scale),
                    norm_eps=1e-12,
                )
        self._maybe_pre_kernel_barrier(T)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe.layer{self.cfg.layer_id}.before_deepgemm", x.device
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y

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
                f"{_PRE_KERNEL_BARRIER_ENV}=1 is incompatible with CUDA graph capture"
            )
        import torch.distributed as dist

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
                    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
                except TypeError:
                    dist.barrier(group=group)
        else:
            dist.barrier(group=group)
        _log_pre_kernel_barrier("leave", cfg.layer_id, rank, world_size, tokens, device)
