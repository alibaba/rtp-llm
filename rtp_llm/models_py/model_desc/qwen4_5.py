"""Qwen 4.5-Flash (Qwen4_5ForCausalLM) PyTorch / Triton B300 Model Description.

Implements the complete 40-layer hybrid architecture verified against the
production ``1109-bf16-re-mxfp4-ple-fp8`` checkpoint:
  - 4-stream Hyper-Connections with ``hc_prev_mix=True``, per-branch GemmaRMSNorm+bias,
    Layer 0 head up-projection ``[5120, 128]``, Layer 39 ``emits_latent=False``, and
    final ``model.hyper_connection_mixer`` (no ``model.norm``).
  - Layer 1 (1-based ``ple_layer_ids=[2]``) Per-Layer N-gram Embedding (PLE) with
    128-shard FP8 E4M3 storage (CPU-UVA or GPU EP-sharded).
  - 3:1 hybrid attention: 30 Gated DeltaNet (GDN) layers + 10 MQA-1 dense Full
    Attention layers with YoCo KV/GDN sharing (Layer 19 -> 5 FullAttn consumers;
    Layer 20 -> 14 GDN consumers with O(1) Decode SSM state readout).
  - Layer 0 Dense MLP (12800) + Layers 1..39 Latent MoE (5120->2560->640 MXFP4
    routed experts top-10->2560->5120 + Shared Expert 1280), SiTU-GLU(4, 25),
    Expert Parallelism AllReduce in 2560-dim latent space.
  - Per-sublayer depthwise causal ShortConv (k=4, residual=True) on attention
    and FFN outputs, plus FFN output clamping to ``[-1000.0, 1000.0]``.
"""

from __future__ import annotations

import inspect
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from rtp_llm.models_py.modules.qwen4_5.attention import Qwen4_5Attention
from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
    get_qwen4_5_fast_all_reduce,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    GDNLayerState,
    Qwen4_5ArchConfig,
    Qwen4_5RuntimeState,
    SharedGDNInputs,
)
from rtp_llm.models_py.modules.qwen4_5.gdn import Qwen4_5GatedDeltaNet
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import Qwen4_5HyperConnection
from rtp_llm.models_py.modules.qwen4_5.moe import Qwen4_5DenseMLP, Qwen4_5LatentMoE
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import MXFP4_BLOCK_SIZE
from rtp_llm.models_py.modules.qwen4_5.ple import (
    DEFAULT_PLE_WEIGHT_SCALE,
    Qwen4_5PLELayer,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import Qwen4_5ShortConv
from rtp_llm.models_py.modules.qwen4_5.yoco import YoCoSharingPlan


_SUPPORTED_PARALLEL_MODES = ("tp4_ep4", "dp4_ep4", "dp2_tp2_ep4", "sp4_tp4_ep4")
_SUBGROUP_CACHE_LOCK = threading.Lock()
_SUBGROUP_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _resolve_tp_group_safe(tp_size: int) -> Any:
    if tp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from rtp_llm.models_py.distributed import collective_torch

        if getattr(collective_torch, "_initialized", False):
            from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

            return _get_group(Group.TP)
    except Exception:
        pass
    if dist.get_world_size() == tp_size:
        return dist.group.WORLD
    return None


def _resolve_dp_group_safe(dp_size: int) -> Any:
    if dp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from rtp_llm.models_py.distributed import collective_torch

        if getattr(collective_torch, "_initialized", False):
            from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

            return _get_group(Group.DP)
    except Exception:
        pass
    if dist.get_world_size() == dp_size:
        return dist.group.WORLD
    return None


@dataclass(frozen=True)
class Qwen4_5ParallelTopology:
    """Resolved multi-dimensional parallelism topology for Target and DFlash2."""

    parallel_mode: str
    world_size: int
    world_rank: int
    tp_size: int
    tp_rank: int
    tp_group: Any
    dp_size: int
    dp_rank: int
    dp_group: Any
    ep_size: int
    ep_rank: int
    ep_group: Any
    sp_size: int
    sp_rank: int
    sp_group: Any
    dp_attn: bool


def _get_or_create_2d_subgroups(
    world_size: int, world_rank: int, dp_size: int, tp_size: int
) -> Tuple[Any, Any]:
    """Deterministically construct DP & TP ProcessGroups on an initialized WORLD."""
    backend = dist.get_backend(dist.group.WORLD)
    cache_key = (id(dist.group.WORLD), str(backend), world_size, dp_size, tp_size)
    with _SUBGROUP_CACHE_LOCK:
        cached = _SUBGROUP_CACHE.get(cache_key)
        if cached is not None:
            return cached["tp_group"], cached["dp_group"]

        my_tp_group: Any = dist.group.WORLD if tp_size == world_size else None
        my_dp_group: Any = dist.group.WORLD if dp_size == world_size else None
        if 1 < tp_size < world_size:
            for dp_r in range(dp_size):
                ranks = [dp_r * tp_size + t for t in range(tp_size)]
                grp = dist.new_group(ranks=ranks, backend=backend)
                if world_rank in ranks:
                    my_tp_group = grp
            dist.barrier(group=dist.group.WORLD)
        if 1 < dp_size < world_size:
            for tp_r in range(tp_size):
                ranks = [d * tp_size + tp_r for d in range(dp_size)]
                grp = dist.new_group(ranks=ranks, backend=backend)
                if world_rank in ranks:
                    my_dp_group = grp
            dist.barrier(group=dist.group.WORLD)

        _SUBGROUP_CACHE[cache_key] = {"tp_group": my_tp_group, "dp_group": my_dp_group}
        return my_tp_group, my_dp_group


def resolve_qwen4_5_parallel_topology(
    *,
    parallel_mode: Optional[str] = None,
    tp_size: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Any = None,
    ep_size: Optional[int] = None,
    ep_rank: Optional[int] = None,
    ep_group: Any = None,
    dp_size: Optional[int] = None,
    dp_rank: Optional[int] = None,
    dp_group: Any = None,
    sp_size: Optional[int] = None,
    sp_rank: Optional[int] = None,
    sp_group: Any = None,
    dp_attn: Optional[bool] = None,
) -> Qwen4_5ParallelTopology:
    """Resolve TP/DP/EP/SP topology from explicit args or ``parallel_mode`` preset."""
    dist_init = dist.is_available() and dist.is_initialized()
    dist_ws = int(dist.get_world_size()) if dist_init else 1
    dist_rk = int(dist.get_rank()) if dist_init else 0

    mode = str(parallel_mode).strip().lower() if parallel_mode else ""
    if mode and mode not in _SUPPORTED_PARALLEL_MODES:
        raise ValueError(
            f"Unsupported parallel_mode={parallel_mode!r}; expected one of {_SUPPORTED_PARALLEL_MODES}"
        )

    if mode == "tp4_ep4":
        tp_s = max(1, int(tp_size if tp_size is not None else (dist_ws if dist_ws > 1 else 4)))
        ep_s = max(1, int(ep_size if ep_size is not None else tp_s))
        dp_s = max(1, int(dp_size if dp_size is not None else 1))
        sp_s = max(1, int(sp_size if sp_size is not None else 1))
        tp_r = int(tp_rank if tp_rank is not None else (dist_rk % tp_s if dist_init else 0))
        ep_r = int(ep_rank if ep_rank is not None else (dist_rk % ep_s if dist_init else 0))
        dp_r = int(dp_rank if dp_rank is not None else 0)
        sp_r = int(sp_rank if sp_rank is not None else 0)
        eff_dp_attn = bool(dp_attn) if dp_attn is not None else (dp_s > 1)
    elif mode == "dp4_ep4":
        dp_s = max(1, int(dp_size if dp_size is not None else (dist_ws if dist_ws > 1 else 4)))
        tp_s = max(1, int(tp_size if tp_size is not None else 1))
        ep_s = max(1, int(ep_size if ep_size is not None else dp_s))
        sp_s = max(1, int(sp_size if sp_size is not None else 1))
        dp_r = int(dp_rank if dp_rank is not None else (dist_rk % dp_s if dist_init else 0))
        ep_r = int(ep_rank if ep_rank is not None else (dist_rk % ep_s if dist_init else 0))
        tp_r = int(tp_rank if tp_rank is not None else 0)
        sp_r = int(sp_rank if sp_rank is not None else 0)
        eff_dp_attn = bool(dp_attn) if dp_attn is not None else True
    elif mode == "dp2_tp2_ep4":
        tp_s = max(1, int(tp_size if tp_size is not None else 2))
        dp_s = max(1, int(dp_size if dp_size is not None else (max(1, dist_ws // tp_s) if dist_ws > 1 else 2)))
        ep_s = max(1, int(ep_size if ep_size is not None else dp_s * tp_s))
        sp_s = max(1, int(sp_size if sp_size is not None else 1))
        if dist_init and dist_ws >= dp_s * tp_s:
            def_dp_r, def_tp_r = divmod(dist_rk % (dp_s * tp_s), tp_s)
        else:
            def_dp_r, def_tp_r = 0, 0
        dp_r = int(dp_rank if dp_rank is not None else def_dp_r)
        tp_r = int(tp_rank if tp_rank is not None else def_tp_r)
        ep_r = int(ep_rank if ep_rank is not None else (dist_rk % ep_s if dist_init else (dp_r * tp_s + tp_r)))
        sp_r = int(sp_rank if sp_rank is not None else 0)
        eff_dp_attn = bool(dp_attn) if dp_attn is not None else True
    elif mode == "sp4_tp4_ep4":
        tp_s = max(1, int(tp_size if tp_size is not None else (dist_ws if dist_ws > 1 else 4)))
        sp_s = max(1, int(sp_size if sp_size is not None else tp_s))
        ep_s = max(1, int(ep_size if ep_size is not None else tp_s))
        dp_s = max(1, int(dp_size if dp_size is not None else 1))
        tp_r = int(tp_rank if tp_rank is not None else (dist_rk % tp_s if dist_init else 0))
        sp_r = int(sp_rank if sp_rank is not None else (dist_rk % sp_s if dist_init else 0))
        ep_r = int(ep_rank if ep_rank is not None else (dist_rk % ep_s if dist_init else 0))
        dp_r = int(dp_rank if dp_rank is not None else 0)
        eff_dp_attn = bool(dp_attn) if dp_attn is not None else False
    else:
        tp_s = max(1, int(tp_size if tp_size is not None else 1))
        ep_s = max(1, int(ep_size if ep_size is not None else 1))
        dp_s = max(1, int(dp_size if dp_size is not None else 1))
        sp_s = max(1, int(sp_size if sp_size is not None else 1))
        tp_r = int(tp_rank if tp_rank is not None else 0)
        ep_r = int(ep_rank if ep_rank is not None else 0)
        dp_r = int(dp_rank if dp_rank is not None else 0)
        sp_r = int(sp_rank if sp_rank is not None else 0)
        eff_dp_attn = bool(dp_attn) if dp_attn is not None else (dp_s > 1)
        if not mode:
            if sp_s > 1 and tp_s > 1:
                mode = "sp4_tp4_ep4"
            elif dp_s > 1 and tp_s > 1:
                mode = "dp2_tp2_ep4"
            elif dp_s > 1:
                mode = "dp4_ep4"
            else:
                mode = "tp4_ep4"

    eff_ws = max(1, dist_ws if dist_init else max(dp_s * tp_s, ep_s, sp_s))
    eff_wr = dist_rk if dist_init else (dp_r * tp_s + tp_r)

    if dist_init:
        if ep_group is None and ep_s > 1 and dist_ws == ep_s:
            ep_group = dist.group.WORLD
        if tp_group is None and tp_s > 1:
            tp_group = _resolve_tp_group_safe(tp_s)
        if dp_group is None and dp_s > 1:
            dp_group = _resolve_dp_group_safe(dp_s)
        if (tp_s > 1 and tp_group is None) or (dp_s > 1 and dp_group is None):
            if dp_s * tp_s == dist_ws:
                sg_tp, sg_dp = _get_or_create_2d_subgroups(dist_ws, dist_rk, dp_s, tp_s)
                if tp_group is None and tp_s > 1:
                    tp_group = sg_tp
                if dp_group is None and dp_s > 1:
                    dp_group = sg_dp
        if sp_group is None and sp_s > 1:
            if tp_s == sp_s and tp_group is not None:
                sp_group = tp_group
            elif dist_ws == sp_s:
                sp_group = dist.group.WORLD

    return Qwen4_5ParallelTopology(
        parallel_mode=mode,
        world_size=eff_ws,
        world_rank=eff_wr,
        tp_size=tp_s,
        tp_rank=tp_r,
        tp_group=tp_group,
        dp_size=dp_s,
        dp_rank=dp_r,
        dp_group=dp_group,
        ep_size=ep_s,
        ep_rank=ep_r,
        ep_group=ep_group,
        sp_size=sp_s,
        sp_rank=sp_r,
        sp_group=sp_group,
        dp_attn=eff_dp_attn,
    )


class _CpuAllToAllState:
    def __init__(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        splits: List[int],
        local_tokens: int,
    ) -> None:
        self.mode = "cpu_gloo_gather"
        self.hidden = hidden
        self.hidden_fp8 = None
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.splits = splits
        self.local_tokens = local_tokens


class _CpuAllToAllCommShim:
    """Pure-PyTorch AllToAll shim matching Qwen4_5AllToAllComm for Gloo/CPU unit tests."""

    def __init__(self, group: Any) -> None:
        self.group = group
        self.world_size = int(dist.get_world_size(group)) if group is not None else 1
        self.rank = int(dist.get_rank(group)) if group is not None else 0

    def dispatch(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: Optional[int] = None,
        equal_tokens: Optional[bool] = None,
        quantize_fp8: bool = False,
    ) -> _CpuAllToAllState:
        del num_experts, quantize_fp8
        dev = hidden.device
        m_loc = int(hidden.shape[0])
        if self.world_size <= 1 or self.group is None:
            return _CpuAllToAllState(hidden, topk_ids, topk_weights, [m_loc], m_loc)
        if equal_tokens:
            splits = [m_loc] * self.world_size
        else:
            cnts = [torch.empty((1,), dtype=torch.int64, device="cpu") for _ in range(self.world_size)]
            dist.all_gather(cnts, torch.tensor([m_loc], dtype=torch.int64, device="cpu"), group=self.group)
            splits = [int(c.item()) for c in cnts]
        max_m = max(splits) if splits else 0
        if max_m == 0:
            return _CpuAllToAllState(hidden, topk_ids, topk_weights, splits, 0)
        h_cpu = hidden.detach().to(device="cpu").contiguous()
        i_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        w_cpu = topk_weights.detach().to(device="cpu", dtype=torch.float32).contiguous()
        pad = max_m - m_loc
        h_pad = torch.cat([h_cpu, torch.zeros((pad, h_cpu.shape[1]), dtype=h_cpu.dtype)], dim=0) if pad > 0 else h_cpu
        i_pad = torch.cat([i_cpu, torch.full((pad, i_cpu.shape[1]), -1, dtype=torch.int64)], dim=0) if pad > 0 else i_cpu
        w_pad = torch.cat([w_cpu, torch.zeros((pad, w_cpu.shape[1]), dtype=torch.float32)], dim=0) if pad > 0 else w_cpu
        gh = [torch.empty_like(h_pad) for _ in range(self.world_size)]
        gi = [torch.empty_like(i_pad) for _ in range(self.world_size)]
        gw = [torch.empty_like(w_pad) for _ in range(self.world_size)]
        dist.all_gather(gh, h_pad, group=self.group)
        dist.all_gather(gi, i_pad, group=self.group)
        dist.all_gather(gw, w_pad, group=self.group)
        cat_h = torch.cat([t[:s] for t, s in zip(gh, splits) if s > 0], dim=0).to(dev)
        cat_i = torch.cat([t[:s] for t, s in zip(gi, splits) if s > 0], dim=0).to(dev)
        cat_w = torch.cat([t[:s] for t, s in zip(gw, splits) if s > 0], dim=0).to(dev)
        return _CpuAllToAllState(cat_h, cat_i, cat_w, splits, m_loc)

    def combine(
        self,
        ep_output: torch.Tensor,
        state: _CpuAllToAllState,
        *,
        use_symm_p2p: Optional[bool] = None,
    ) -> torch.Tensor:
        del use_symm_p2p
        if self.world_size <= 1 or self.group is None:
            return ep_output
        if sum(state.splits) == 0:
            return ep_output[:0]
        dev = ep_output.device
        buf = ep_output.detach().to(device="cpu").contiguous()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self.group)
        off = sum(state.splits[: self.rank])
        return buf[off : off + state.local_tokens].to(dev)

    def _ensure_tbo_resources(self) -> None:
        return None

    def dispatch_async(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_experts: Optional[int] = None,
        equal_tokens: Optional[bool] = None,
        quantize_fp8: bool = False,
        comm_stream: Any = None,
        slot: int = 0,
    ) -> _CpuAllToAllState:
        del comm_stream, slot
        return self.dispatch(
            hidden,
            topk_ids,
            topk_weights,
            num_experts=num_experts,
            equal_tokens=equal_tokens,
            quantize_fp8=quantize_fp8,
        )

    def wait_dispatch(self, handle: _CpuAllToAllState) -> _CpuAllToAllState:
        return handle

    def combine_async(
        self,
        ep_output: torch.Tensor,
        state: _CpuAllToAllState,
        *,
        use_symm_p2p: Optional[bool] = None,
        comm_stream: Any = None,
        slot: int = 0,
    ) -> torch.Tensor:
        del comm_stream, slot
        return self.combine(ep_output, state, use_symm_p2p=use_symm_p2p)

    def wait_combine(self, handle: torch.Tensor) -> torch.Tensor:
        return handle


def resolve_qwen4_5_all_to_all(
    ep_group: Any,
    *,
    device: Optional[torch.device],
    dtype: torch.dtype,
    hidden_dim: int,
    top_k: int,
) -> Any:
    if ep_group is None or not dist.is_available() or not dist.is_initialized():
        return None
    if int(dist.get_world_size(ep_group)) <= 1:
        return None
    dev = device if device is not None else torch.device("cpu")
    backend = str(dist.get_backend(ep_group)).lower()
    if dev.type == "cuda" and backend != "gloo":
        from rtp_llm.models_py.modules.qwen4_5.alltoall_comm import get_qwen4_5_all_to_all

        return get_qwen4_5_all_to_all(
            ep_group,
            device=dev,
            dtype=dtype,
            hidden_dim=hidden_dim,
            top_k=top_k,
            single_node=True,
        )
    return _CpuAllToAllCommShim(ep_group)


def _resolve_dense_linear_mode(override: Optional[str], stored: Optional[str]) -> str:
    if override is not None:
        return str(override).strip().lower()
    if stored is not None:
        return str(stored).strip().lower()
    return os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "bf16").strip().lower()


def _wrap_linear_with_sm100_fp8(
    mod: nn.Module,
    eff_mode: str,
    *,
    allow_decode_scaled_mm: Optional[bool] = None,
    prepare_deepgemm_fp8: Optional[bool] = None,
    prepare_mxfp8: Optional[bool] = None,
) -> nn.Module:
    from rtp_llm.models_py.modules.qwen4_5.fp8_linear import SM100FP8Linear

    if isinstance(mod, SM100FP8Linear):
        mod.mode = eff_mode
        mod._prepare_w4a8_flag = bool(eff_mode == "w4a8")
        if allow_decode_scaled_mm is not None and hasattr(mod, "_allow_decode_scaled_mm"):
            mod._allow_decode_scaled_mm = bool(allow_decode_scaled_mm)
        if prepare_mxfp8 is not None and hasattr(mod, "_prepare_mxfp8_flag"):
            mod._prepare_mxfp8_flag = bool(prepare_mxfp8)
            if not prepare_mxfp8:
                mod.w_mxfp8 = None
                mod.w_mxfp8_sc = None
        if prepare_deepgemm_fp8 is not None and hasattr(mod, "_prepare_dg_fp8_flag"):
            if hasattr(mod, "_explicit_dg_fp8_flag"):
                mod._explicit_dg_fp8_flag = bool(prepare_deepgemm_fp8)
            mod._prepare_dg_fp8_flag = bool(prepare_deepgemm_fp8)
            if not prepare_deepgemm_fp8:
                mod.w_dg_fp8 = None
                mod.w_dg_sc = None
        elif (
            prepare_mxfp8 is not None
            and getattr(mod, "_explicit_dg_fp8_flag", None) is None
            and hasattr(mod, "_prepare_dg_fp8_flag")
        ):
            mod._prepare_dg_fp8_flag = bool(prepare_mxfp8)
            if not prepare_mxfp8:
                mod.w_dg_fp8 = None
                mod.w_dg_sc = None
        if eff_mode != "bf16" and mod.weight is not None and mod.weight.is_cuda:
            mod.prepare_weights()
        else:
            mod._refresh_fast_dispatch()
        return mod
    if eff_mode == "bf16" or not isinstance(mod, nn.Linear) or not mod.weight.is_cuda:
        return mod
    raw_extra: Dict[str, Any] = {
        "allow_decode_scaled_mm": (
            bool(allow_decode_scaled_mm) if allow_decode_scaled_mm is not None else False
        )
    }
    if prepare_deepgemm_fp8 is not None:
        raw_extra["prepare_deepgemm_fp8"] = bool(prepare_deepgemm_fp8)
    extra_kw = _filter_init_kwargs(SM100FP8Linear.__init__, raw_extra)
    inst = SM100FP8Linear(
        in_features=mod.in_features,
        out_features=mod.out_features,
        bias=(mod.bias is not None),
        mode=eff_mode,
        keep_bf16=False,
        prepare_trtllm_ll=True,
        prepare_mxfp8=(True if prepare_mxfp8 is None else bool(prepare_mxfp8)),
        prepare_w4a8=(eff_mode == "w4a8"),
        dtype=mod.weight.dtype,
        device=mod.weight.device,
        **extra_kw,
    )
    inst.weight = mod.weight
    if mod.bias is not None:
        inst.bias = mod.bias
    inst.prepare_weights()
    return inst


def _filter_init_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(cls).parameters
    except Exception:
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


class Qwen4_5ProfileRecorder:
    """Lightweight CUDA-event category timer (zero overhead when unused)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._pairs: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self._pairs.clear()

    @contextmanager
    def record(self, category: str) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pairs.append((category, start, end))

    def summarize_ms(self) -> Dict[str, float]:
        if not self._pairs:
            return {}
        torch.cuda.synchronize()
        totals: Dict[str, float] = {}
        for cat, s, e in self._pairs:
            totals[cat] = totals.get(cat, 0.0) + float(s.elapsed_time(e))
        return totals


def run_latent_moe_tbo_tokens(
    mlp: nn.Module,
    hidden: torch.Tensor,
    *,
    use_triton_mxfp4: bool = True,
    equal_tokens: Optional[bool] = None,
) -> torch.Tensor:
    """Run 3-stage 2-microbatch MoE TBO across flattened token dim when supported."""
    last_dim = int(hidden.shape[-1]) if hidden.ndim > 0 else 0
    total_tokens = (
        int(hidden.shape[0])
        if hidden.ndim == 2
        else (int(hidden.numel() // last_dim) if last_dim > 0 else 0)
    )
    if (
        total_tokens >= 2
        and hasattr(mlp, "forward_route_and_dispatch")
        and int(getattr(mlp, "ep_size", 1)) > 1
        and bool(getattr(mlp, "dp_attn", False))
        and getattr(mlp, "all_to_all_comm", None) is not None
    ):
        flat = hidden if hidden.ndim == 2 else hidden.reshape(total_tokens, last_dim)
        t0 = total_tokens // 2
        m0, m1 = flat[:t0], flat[t0:]
        if not torch.cuda.is_current_stream_capturing() and hasattr(
            mlp.all_to_all_comm, "_ensure_tbo_resources"
        ):
            mlp.all_to_all_comm._ensure_tbo_resources()
        ctx0 = mlp.forward_route_and_dispatch(
            m0,
            use_triton_mxfp4=use_triton_mxfp4,
            equal_tokens=equal_tokens,
            slot=0,
        )
        ctx1 = mlp.forward_route_and_dispatch(
            m1,
            use_triton_mxfp4=use_triton_mxfp4,
            equal_tokens=equal_tokens,
            slot=1,
        )
        ctx0 = mlp.forward_routed_compute_and_combine(ctx0, slot=0)
        ctx1 = mlp.forward_routed_compute_and_combine(ctx1, slot=1)
        o0 = mlp.forward_finish_combine(ctx0)
        o1 = mlp.forward_finish_combine(ctx1)
        out = torch.cat((o0, o1), dim=0)
        return out if hidden.ndim == 2 else out.reshape(hidden.shape)
    return mlp(hidden, use_triton_mxfp4=use_triton_mxfp4)


class Qwen4_5DecoderLayer(nn.Module):
    """Single decoder layer of Qwen 4.5-Flash (GDN or FullAttn + Dense or LatentMoE)."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        layer_idx: int,
        sharing_plan: YoCoSharingPlan,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Any = None,
        fast_all_reduce: Any = None,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
        kv_dtype: str = "bf16",
        fp8_indexer_kv: Optional[bool] = None,
        all_to_all_comm: Any = None,
        dp_equal_tokens: Optional[bool] = None,
        use_symm_p2p_combine: Optional[bool] = None,
        dense_linear_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.is_consumer = sharing_plan.is_consumer(layer_idx)
        self.is_source = sharing_plan.is_source(layer_idx)
        self.source_layer_idx = sharing_plan.source_for(layer_idx)
        self.kv_cache_owner = sharing_plan.kv_cache_owner(layer_idx)
        self.ep_size = max(1, int(ep_size or 1))
        self.ep_group = ep_group
        self.tp_size = max(1, int(tp_size or 1))
        self.tp_rank = int(tp_rank or 0)
        self.tp_group = tp_group
        self.fast_all_reduce = fast_all_reduce
        self.sp_size = max(1, int(sp_size or 1))
        self.sp_rank = int(sp_rank or 0)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)
        self.kv_dtype = str(kv_dtype)
        self._dense_linear_mode = dense_linear_mode

        sp_dp_kw = {"sp_size": self.sp_size, "sp_rank": self.sp_rank, "sp_group": self.sp_group, "dp_attn": self.dp_attn}
        is_first_layer = layer_idx == 0
        is_last_layer = layer_idx == (config.num_hidden_layers - 1)

        # 1. Optional PLE module (1-based ple_layer_ids -> 0-based layer_idx)
        self.has_ple = (layer_idx + 1) in config.ple_layer_ids
        if self.has_ple:
            self.ple: Optional[Qwen4_5PLELayer] = Qwen4_5PLELayer(
                hidden_size=config.hidden_size,
                hc_count=config.hc_count,
                ple_embed_dim=config.ple_embed_dim,
                ngram_size=config.ngram_size,
                heads_per_ngram=config.heads_per_ngram,
                conv_kernel_size=config.ple_conv_kernel_size,
                split_ngram_parts=config.split_ngram_parts,
                eos_token_id=config.eos_token_id,
                eps=config.rms_norm_eps,
                ngram_vocab_size_base=config.ngram_vocab_size_base,
                dtype=dtype,
                device=device,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
                **_filter_init_kwargs(Qwen4_5PLELayer.__init__, sp_dp_kw),
            )
        else:
            self.ple = None

        # 2. Attention Hyper-Connection
        self.attn_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            prev_mix=config.hc_prev_mix,
            is_head=is_first_layer,
            emits_latent=True,
            dtype=dtype,
            device=device,
            **_filter_init_kwargs(Qwen4_5HyperConnection.__init__, sp_dp_kw),
        )

        # 3. Attention block (GDN or dense MQA-1 Full Attention / QSA)
        if self.layer_type == "linear_attention":
            self.linear_attn: Optional[Qwen4_5GatedDeltaNet] = (
                Qwen4_5GatedDeltaNet.from_config(
                    config,
                    layer_idx,
                    sharing_plan=sharing_plan,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    tp_group=tp_group,
                    fast_all_reduce=fast_all_reduce,
                    **_filter_init_kwargs(Qwen4_5GatedDeltaNet.from_config, sp_dp_kw),
                ).to(device=device, dtype=dtype)
            )
            self.self_attn: Optional[Qwen4_5Attention] = None
            # Keep A_log in FP32 as required by GDN gating spec
            if self.linear_attn.A_log is not None:
                self.linear_attn.A_log.data = self.linear_attn.A_log.data.float()
        elif self.layer_type == "full_attention":
            eff_fp8_idx = (
                bool(fp8_indexer_kv)
                if fp8_indexer_kv is not None
                else self.kv_dtype.lower().startswith("fp8")
            )
            attn_extra = {
                **sp_dp_kw,
                "fp8_indexer_kv": eff_fp8_idx,
                "kv_cache_dtype": self.kv_dtype,
            }
            self.self_attn = Qwen4_5Attention.from_config(
                config,
                layer_idx,
                sharing_plan=sharing_plan,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
                **_filter_init_kwargs(Qwen4_5Attention.from_config, attn_extra),
            ).to(device=device, dtype=dtype)
            self.linear_attn = None
            # Keep RoPE inv_freq in FP32
            self.self_attn.inv_freq = self.self_attn.inv_freq.float()
        else:
            raise ValueError(f"Unsupported layer_type={self.layer_type!r} at layer {layer_idx}")

        # 4. Optional Attention output ShortConv (k=4, residual=True)
        attn_site = (
            "linear_attn_output_sconv"
            if self.layer_type == "linear_attention"
            else "full_attn_output_sconv"
        )
        if config.has_output_sconv(attn_site, is_consumer=self.is_consumer):
            self.attn_output_sconv: Optional[Qwen4_5ShortConv] = Qwen4_5ShortConv(
                channels=config.hidden_size,
                kernel_size=config.linear_conv_kernel_dim,
                residual=True,
                use_activation=config.output_sconv_use_activation,
                dtype=dtype,
                device=device,
                **_filter_init_kwargs(Qwen4_5ShortConv.__init__, sp_dp_kw),
            )
        else:
            self.attn_output_sconv = None

        # 5. MLP Hyper-Connection (last layer omits input_mix_weight_down -> emits_latent=False)
        self.mlp_hyper_connection = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            prev_mix=config.hc_prev_mix,
            is_head=False,
            emits_latent=not is_last_layer,
            dtype=dtype,
            device=device,
            **_filter_init_kwargs(Qwen4_5HyperConnection.__init__, sp_dp_kw),
        )

        # 6. FFN block: first_k_dense_replace Dense MLP, remaining Latent MoE
        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            moe_extra = _filter_init_kwargs(
                Qwen4_5LatentMoE.from_config,
                {
                    "all_to_all_comm": all_to_all_comm,
                    "dp_equal_tokens": dp_equal_tokens,
                    "use_symm_p2p_combine": use_symm_p2p_combine,
                    "dense_linear_mode": dense_linear_mode,
                },
            )
            self.mlp: Union[Qwen4_5DenseMLP, Qwen4_5LatentMoE] = Qwen4_5LatentMoE.from_config(
                config,
                use_triton_mxfp4=use_triton_mxfp4,
                compute_mode=compute_mode,
                dtype=dtype,
                device=device,
                init_mxfp4_weights=init_mxfp4_weights,
                ep_size=ep_size,
                ep_rank=ep_rank,
                ep_group=ep_group,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
                **moe_extra,
            )
        else:
            dense_extra = _filter_init_kwargs(
                Qwen4_5DenseMLP.from_config,
                {"dense_linear_mode": dense_linear_mode},
            )
            self.mlp = Qwen4_5DenseMLP.from_config(
                config,
                is_shared_expert=False,
                use_triton=use_triton_mxfp4,
                dtype=dtype,
                device=device,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=tp_group,
                fast_all_reduce=fast_all_reduce,
                **dense_extra,
            )

        # 7. Optional FFN output ShortConv (k=4, residual=True)
        if config.has_output_sconv("ffn_output_sconv", is_consumer=self.is_consumer):
            self.ffn_output_sconv: Optional[Qwen4_5ShortConv] = Qwen4_5ShortConv(
                channels=config.hidden_size,
                kernel_size=config.linear_conv_kernel_dim,
                residual=True,
                use_activation=config.output_sconv_use_activation,
                dtype=dtype,
                device=device,
                **_filter_init_kwargs(Qwen4_5ShortConv.__init__, sp_dp_kw),
            )
        else:
            self.ffn_output_sconv = None
        self._attn_sconv_has_clip = (
            self.attn_output_sconv is not None
            and "clip_value" in inspect.signature(self.attn_output_sconv.forward).parameters
            and "clip_value" in inspect.signature(self.attn_output_sconv.forward_ref).parameters
        )
        self._ffn_sconv_has_clip = (
            self.ffn_output_sconv is not None
            and "clip_value" in inspect.signature(self.ffn_output_sconv.forward).parameters
            and "clip_value" in inspect.signature(self.ffn_output_sconv.forward_ref).parameters
        )

    @staticmethod
    def _run_sconv_with_clip(
        sconv: Optional[Qwen4_5ShortConv],
        x: torch.Tensor,
        sc_state: Optional[Any],
        clip_val: Optional[float],
        has_clip: bool,
        *,
        use_reference: bool,
        profiler: Optional[Qwen4_5ProfileRecorder],
    ) -> torch.Tensor:
        """Run output ShortConv and fuse symmetric clamping when supported."""
        if sconv is not None:
            if clip_val is not None and has_clip:
                if profiler is not None:
                    with profiler.record("short_conv"):
                        return sconv(x, state=sc_state, clip_value=clip_val)
                if use_reference:
                    return sconv.forward_ref(x, state=sc_state, clip_value=clip_val)
                return sconv(x, state=sc_state, clip_value=clip_val)
            if profiler is not None:
                with profiler.record("short_conv"):
                    x = sconv(x, state=sc_state)
            elif use_reference:
                x = sconv.forward_ref(x, state=sc_state)
            else:
                x = sconv(x, state=sc_state)
        if clip_val is not None:
            x = x.clamp(-clip_val, clip_val)
        return x

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        transpose_mxfp4: bool = False,
        dense_linear_mode: Optional[str] = None,
    ) -> None:
        eff_mode = _resolve_dense_linear_mode(dense_linear_mode, self._dense_linear_mode)
        if dense_linear_mode is not None:
            self._dense_linear_mode = dense_linear_mode
        if isinstance(self.mlp, Qwen4_5LatentMoE):
            self.mlp.prepare_weights(
                transpose_mxfp4=transpose_mxfp4,
                fuse_shared_gate_up=True,
                dense_linear_mode=eff_mode,
            )
        else:
            self.mlp.prepare_weights(fuse_gate_up=True, dense_linear_mode=eff_mode)
        def _wrap_fn(mod: nn.Module) -> nn.Module:
            return _wrap_linear_with_sm100_fp8(mod, eff_mode)

        if self.linear_attn is not None:
            gdn_dg = eff_mode == "deepgemm_fp8"
            gdn_mx = eff_mode == "mxfp8"
            self.linear_attn.in_proj_qkvz = _wrap_linear_with_sm100_fp8(
                self.linear_attn.in_proj_qkvz,
                eff_mode,
                prepare_deepgemm_fp8=gdn_dg,
                prepare_mxfp8=gdn_mx,
            )
            self.linear_attn.out_proj = _wrap_linear_with_sm100_fp8(
                self.linear_attn.out_proj,
                eff_mode,
                prepare_deepgemm_fp8=gdn_dg,
                prepare_mxfp8=gdn_mx,
            )
        if self.self_attn is not None:
            prev_attn_fused = getattr(self.self_attn, "qkv_idx_proj", None)
            fused_attn = (
                self.self_attn.prepare_fused_projections(_wrap_fn)
                if hasattr(self.self_attn, "prepare_fused_projections")
                else None
            )
            if fused_attn is not None:
                if fused_attn is prev_attn_fused and not isinstance(fused_attn, nn.Linear):
                    self.self_attn.qkv_idx_proj = _wrap_fn(fused_attn)
            else:
                self.self_attn.q_proj = _wrap_fn(self.self_attn.q_proj)
            self.self_attn.o_proj = _wrap_fn(self.self_attn.o_proj)

    def forward_pre_mlp(
        self,
        hyper_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        prev_latent: Optional[torch.Tensor],
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]],
        shared_gdn_by_source: Dict[int, SharedGDNInputs],
        shared_qsa_by_source: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        source_prefix_ssm: Optional[Dict[int, torch.Tensor]] = None,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run PLE, attention HC, attention core, attn ShortConv, and MLP HC mix."""
        cfg = self.config
        lid = self.layer_idx
        mods = self._modules
        ple = mods.get("ple")
        attn_hc = mods["attn_hyper_connection"]
        linear_attn = mods.get("linear_attn")
        self_attn = mods.get("self_attn")
        attn_sconv = mods.get("attn_output_sconv")
        mlp_hc = mods["mlp_hyper_connection"]

        # Step A: PLE injection (Layer 1 only) — flush deferred HC combine before PLE
        if ple is not None:
            if deferred is not None:
                if profiler is not None:
                    with profiler.record("hc"):
                        hyper_hidden = attn_hc.combine(
                            deferred[0], hyper_hidden, deferred[1]
                        )
                elif use_reference:
                    hyper_hidden = attn_hc.combine_ref(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                else:
                    hyper_hidden = attn_hc.combine(
                        deferred[0], hyper_hidden, deferred[1]
                    )
                deferred = None

            ple_state = state.ple.get(lid) if state is not None else None
            if profiler is not None:
                with profiler.record("ple"):
                    hyper_hidden = ple(hyper_hidden, input_ids, state=ple_state)
            elif use_reference:
                hyper_hidden = ple.forward_ref(hyper_hidden, input_ids, state=ple_state)
            else:
                hyper_hidden = ple(hyper_hidden, input_ids, state=ple_state)

        # Step B: Attention Hyper-Connection (fused combine+norm+mix when deferred exists)
        if profiler is not None:
            with profiler.record("hc"):
                if deferred is None:
                    attn_in, hyper_hidden, attn_inj, prev_latent = (
                        attn_hc.mix_with_latent(hyper_hidden, prev_latent)
                    )
                else:
                    attn_in, hyper_hidden, attn_inj, prev_latent = (
                        attn_hc.combine_and_mix_with_latent(
                            deferred[0], hyper_hidden, deferred[1], prev_latent
                        )
                    )
        elif use_reference:
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    attn_hc.mix_with_latent_ref(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    attn_hc.combine_and_mix_with_latent_ref(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
        else:
            if deferred is None:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    attn_hc.mix_with_latent(hyper_hidden, prev_latent)
                )
            else:
                attn_in, hyper_hidden, attn_inj, prev_latent = (
                    attn_hc.combine_and_mix_with_latent(
                        deferred[0], hyper_hidden, deferred[1], prev_latent
                    )
                )
        assert attn_inj is not None

        # Step C: Attention core (GDN or Full Attention)
        if linear_attn is not None:
            gdn_state = state.gdn.get(lid) if state is not None else None
            if (
                self.is_source
                and source_prefix_ssm is not None
                and gdn_state is not None
                and gdn_state.ssm_state is not None
            ):
                source_prefix_ssm[lid] = gdn_state.ssm_state.clone()
            yoco_on = (
                cfg.enable_yoco_decode
                if enable_yoco_decode is None
                else bool(enable_yoco_decode)
            )
            gdn_call_state = gdn_state
            if (
                self.is_consumer
                and input_ids.shape[1] > 1
                and yoco_on
                and gdn_state is not None
                and gdn_state.ssm_state is None
                and source_prefix_ssm is not None
                and self.source_layer_idx in source_prefix_ssm
            ):
                # Reuse pre-suffix source SSM transiently so chunked/suffix prefill preserves prefix context
                gdn_call_state = GDNLayerState(
                    conv_state=gdn_state.conv_state,
                    ssm_state=source_prefix_ssm[self.source_layer_idx].clone(),  # type: ignore[index]
                )
            shared_in = (
                shared_gdn_by_source[self.source_layer_idx]
                if (self.is_consumer and self.source_layer_idx is not None)
                else None
            )
            if profiler is not None:
                cat = "gdn_consumer" if self.is_consumer else "gdn_normal"
                with profiler.record(cat):
                    attn_out, published = linear_attn(
                        attn_in,
                        state=gdn_call_state,
                        shared=shared_in,
                        enable_yoco_decode=enable_yoco_decode,
                    )
            else:
                attn_out, published = linear_attn(
                    attn_in,
                    state=gdn_call_state,
                    shared=shared_in,
                    enable_yoco_decode=enable_yoco_decode,
                )
            if published is not None:
                shared_gdn_by_source[lid] = published
        else:
            assert self_attn is not None
            kv_cache = state.full_attn_kv.get(self.kv_cache_owner) if state is not None else None
            k_sc = state.k_sconv.get(lid) if (state is not None and not self.is_consumer) else None
            v_sc = state.v_sconv.get(lid) if (state is not None and not self.is_consumer) else None
            if self_attn.indexer is not None:
                qsa_kwargs: Dict[str, Any] = {}
                if not self.is_consumer:
                    raw_prev = state.qsa_raw_keys.get(lid) if state is not None else None
                    past_raw = (
                        raw_prev if (raw_prev is not None and raw_prev.shape[1] > 0) else None
                    )
                    src_k_norm = self_attn.indexer.k_norm_gamma

                    def _record_raw(
                        raw_k: torch.Tensor,
                        _past: Optional[torch.Tensor] = past_raw,
                        _lid: int = lid,
                        _k_norm: torch.Tensor = src_k_norm,
                    ) -> None:
                        full_raw = (
                            torch.cat([_past, raw_k], dim=1) if _past is not None else raw_k
                        )
                        if state is not None:
                            state.qsa_raw_keys[_lid] = full_raw
                        if shared_qsa_by_source is not None:
                            shared_qsa_by_source[_lid] = (full_raw, _k_norm)

                    qsa_kwargs["qsa_past_raw_keys"] = past_raw
                    qsa_kwargs["qsa_on_raw_keys"] = _record_raw
                else:
                    src_lid = self.kv_cache_owner
                    src_raw = None
                    src_k_norm = None
                    if shared_qsa_by_source is not None and src_lid in shared_qsa_by_source:
                        src_raw, src_k_norm = shared_qsa_by_source[src_lid]
                    elif state is not None and src_lid in state.qsa_raw_keys:
                        src_raw = state.qsa_raw_keys[src_lid]
                    qsa_kwargs["qsa_source_raw_keys"] = src_raw
                    qsa_kwargs["qsa_source_k_norm_gamma"] = src_k_norm
                if profiler is not None:
                    cat = "full_attn_consumer" if self.is_consumer else "full_attn_producer"
                    with profiler.record(cat):
                        attn_out = self_attn(
                            attn_in,
                            positions=positions,
                            kv_cache=kv_cache,
                            k_sconv_state=k_sc,
                            v_sconv_state=v_sc,
                            seq_offset=seq_offset,
                            **qsa_kwargs,
                        )
                else:
                    attn_out = self_attn(
                        attn_in,
                        positions=positions,
                        kv_cache=kv_cache,
                        k_sconv_state=k_sc,
                        v_sconv_state=v_sc,
                        seq_offset=seq_offset,
                        **qsa_kwargs,
                    )
            elif profiler is not None:
                cat = "full_attn_consumer" if self.is_consumer else "full_attn_producer"
                with profiler.record(cat):
                    attn_out = self_attn(
                        attn_in,
                        positions=positions,
                        kv_cache=kv_cache,
                        k_sconv_state=k_sc,
                        v_sconv_state=v_sc,
                        seq_offset=seq_offset,
                    )
            else:
                attn_out = self_attn(
                    attn_in,
                    positions=positions,
                    kv_cache=kv_cache,
                    k_sconv_state=k_sc,
                    v_sconv_state=v_sc,
                    seq_offset=seq_offset,
                )

        # Step D: Attention output ShortConv (+ optional fused clip)
        if attn_sconv is not None:
            attn_sc_state = state.attn_output_sconv.get(lid) if state is not None else None
            attn_out = self._run_sconv_with_clip(
                attn_sconv,
                attn_out,
                attn_sc_state,
                cfg.clip_attn_output,
                self._attn_sconv_has_clip,
                use_reference=use_reference,
                profiler=profiler,
            )
        elif cfg.clip_attn_output is not None:
            attn_out = attn_out.clamp(-cfg.clip_attn_output, cfg.clip_attn_output)

        # Step E: MLP Hyper-Connection (fused combine of attn_out + norm + mix)
        if profiler is not None:
            with profiler.record("hc"):
                mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                    mlp_hc.combine_and_mix_with_latent(
                        attn_out, hyper_hidden, attn_inj, prev_latent
                    )
                )
        elif use_reference:
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                mlp_hc.combine_and_mix_with_latent_ref(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
        else:
            mlp_in, hyper_hidden, mlp_inj, prev_latent = (
                mlp_hc.combine_and_mix_with_latent(
                    attn_out, hyper_hidden, attn_inj, prev_latent
                )
            )
        assert mlp_inj is not None
        return mlp_in, hyper_hidden, mlp_inj, prev_latent

    def _run_mlp(
        self,
        mlp_in: torch.Tensor,
        *,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        enable_moe_tbo: bool = False,
        tbo_equal_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        mlp = self._modules["mlp"]
        if isinstance(mlp, Qwen4_5LatentMoE):
            if enable_moe_tbo:
                return self.forward_mlp_tbo(
                    mlp_in,
                    use_reference=use_reference,
                    profiler=profiler,
                    equal_tokens=tbo_equal_tokens,
                )
            if profiler is not None:
                with profiler.record("latent_moe"):
                    return mlp(mlp_in, use_triton_mxfp4=not use_reference)
            return mlp(mlp_in, use_triton_mxfp4=not use_reference)
        if profiler is not None:
            with profiler.record("dense_mlp"):
                return mlp(mlp_in, use_triton=not use_reference)
        return mlp(mlp_in, use_triton=not use_reference)

    def forward_mlp_tbo(
        self,
        mlp_in: torch.Tensor,
        *,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        equal_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run 3-stage MoE TBO by splitting flattened tokens into 2 microbatches."""
        mlp = self._modules["mlp"]
        if not isinstance(mlp, Qwen4_5LatentMoE):
            return self._run_mlp(mlp_in, use_reference=use_reference, profiler=profiler)
        if profiler is not None:
            with profiler.record("latent_moe"):
                return run_latent_moe_tbo_tokens(
                    mlp,
                    mlp_in,
                    use_triton_mxfp4=not use_reference,
                    equal_tokens=equal_tokens,
                )
        return run_latent_moe_tbo_tokens(
            mlp,
            mlp_in,
            use_triton_mxfp4=not use_reference,
            equal_tokens=equal_tokens,
        )

    def forward_post_mlp(
        self,
        mlp_out: torch.Tensor,
        hyper_hidden: torch.Tensor,
        prev_latent: Optional[torch.Tensor],
        mlp_inj: torch.Tensor,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Run FFN output ShortConv + clamping and emit deferred branch tuple."""
        cfg = self.config
        ffn_sc = self._modules.get("ffn_output_sconv")
        if ffn_sc is not None:
            ffn_sc_state = state.ffn_output_sconv.get(self.layer_idx) if state is not None else None
            mlp_out = self._run_sconv_with_clip(
                ffn_sc,
                mlp_out,
                ffn_sc_state,
                cfg.clip_mlp_output,
                self._ffn_sconv_has_clip,
                use_reference=use_reference,
                profiler=profiler,
            )
        elif cfg.clip_mlp_output is not None:
            mlp_out = mlp_out.clamp(-cfg.clip_mlp_output, cfg.clip_mlp_output)
        return hyper_hidden, prev_latent, (mlp_out, mlp_inj)

    def forward(
        self,
        hyper_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        prev_latent: Optional[torch.Tensor],
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]],
        shared_gdn_by_source: Dict[int, SharedGDNInputs],
        shared_qsa_by_source: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        source_prefix_ssm: Optional[Dict[int, torch.Tensor]] = None,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        enable_moe_tbo: bool = False,
        tbo_equal_tokens: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Run one full decoder layer returning ``(hyper_hidden, prev_latent, deferred)``."""
        mlp_in, hyper_hidden, mlp_inj, prev_latent = self.forward_pre_mlp(
            hyper_hidden,
            input_ids,
            prev_latent=prev_latent,
            deferred=deferred,
            shared_gdn_by_source=shared_gdn_by_source,
            shared_qsa_by_source=shared_qsa_by_source,
            source_prefix_ssm=source_prefix_ssm,
            state=state,
            positions=positions,
            seq_offset=seq_offset,
            enable_yoco_decode=enable_yoco_decode,
            use_reference=use_reference,
            profiler=profiler,
        )
        mlp_out = self._run_mlp(
            mlp_in,
            use_reference=use_reference,
            profiler=profiler,
            enable_moe_tbo=enable_moe_tbo,
            tbo_equal_tokens=tbo_equal_tokens,
        )
        return self.forward_post_mlp(
            mlp_out,
            hyper_hidden,
            prev_latent,
            mlp_inj,
            state=state,
            use_reference=use_reference,
            profiler=profiler,
        )


class Qwen4_5Model(nn.Module):
    """Full Qwen 4.5-Flash backbone (embeddings + N decoder layers + final HC mixer)."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        parallel_mode: Optional[str] = None,
        ep_size: Optional[int] = None,
        ep_rank: Optional[int] = None,
        ep_group: Any = None,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Any = None,
        dp_size: Optional[int] = None,
        dp_rank: Optional[int] = None,
        dp_group: Any = None,
        sp_size: Optional[int] = None,
        sp_rank: Optional[int] = None,
        sp_group: Any = None,
        dp_attn: Optional[bool] = None,
        kv_dtype: str = "bf16",
        fp8_indexer_kv: Optional[bool] = None,
        all_to_all_comm: Any = None,
        dp_equal_tokens: Optional[bool] = None,
        use_symm_p2p_combine: Optional[bool] = None,
        dense_linear_mode: Optional[str] = None,
        fast_all_reduce: Any = None,
        fast_ar_max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        topo = resolve_qwen4_5_parallel_topology(
            parallel_mode=parallel_mode,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group=dp_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            dp_attn=dp_attn,
        )
        self.parallel_topology = topo
        self.parallel_mode = topo.parallel_mode
        self.tp_size = topo.tp_size
        self.tp_rank = topo.tp_rank
        self.tp_group = topo.tp_group
        self.ep_size = topo.ep_size
        self.ep_rank = topo.ep_rank
        self.ep_group = topo.ep_group
        self.dp_size = topo.dp_size
        self.dp_rank = topo.dp_rank
        self.dp_group = topo.dp_group
        self.sp_size = topo.sp_size
        self.sp_rank = topo.sp_rank
        self.sp_group = topo.sp_group
        self.dp_attn = topo.dp_attn
        self.kv_dtype = str(kv_dtype)
        self._dense_linear_mode = dense_linear_mode
        if not (0 <= self.tp_rank < self.tp_size):
            raise ValueError(f"tp_rank={self.tp_rank} out of range for tp_size={self.tp_size}")

        dev = torch.device(device) if device is not None else None
        if (
            fast_all_reduce is None
            and self.tp_size > 1
            and self.tp_group is not None
            and dev is not None
            and dev.type == "cuda"
            and dist.is_available()
            and dist.is_initialized()
        ):
            ar_kwargs: Dict[str, Any] = {
                "device": dev,
                "dtype": dtype,
                "hidden_dims": (int(config.hidden_size), int(config.routed_expert_hidden_size)),
            }
            if fast_ar_max_tokens is not None:
                ar_kwargs["max_tokens"] = int(fast_ar_max_tokens)
            fast_all_reduce = get_qwen4_5_fast_all_reduce(
                self.tp_group,
                **ar_kwargs,
            )
        self.fast_all_reduce = fast_all_reduce

        if all_to_all_comm is None and self.ep_size > 1 and self.dp_attn and self.ep_group is not None:
            all_to_all_comm = resolve_qwen4_5_all_to_all(
                self.ep_group,
                device=dev,
                dtype=dtype,
                hidden_dim=int(config.routed_expert_hidden_size),
                top_k=int(config.num_experts_per_tok),
            )
        self.all_to_all_comm = all_to_all_comm
        self.sharing_plan = YoCoSharingPlan.from_config(config)

        # Target embed_tokens stays replicated [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                Qwen4_5DecoderLayer(
                    config,
                    layer_idx=i,
                    sharing_plan=self.sharing_plan,
                    use_triton_mxfp4=use_triton_mxfp4,
                    compute_mode=compute_mode,
                    dtype=dtype,
                    device=device,
                    init_mxfp4_weights=init_mxfp4_weights,
                    ep_size=self.ep_size,
                    ep_rank=self.ep_rank,
                    ep_group=self.ep_group,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                    tp_group=self.tp_group,
                    fast_all_reduce=fast_all_reduce,
                    sp_size=self.sp_size,
                    sp_rank=self.sp_rank,
                    sp_group=self.sp_group,
                    dp_attn=self.dp_attn,
                    kv_dtype=self.kv_dtype,
                    fp8_indexer_kv=fp8_indexer_kv,
                    all_to_all_comm=all_to_all_comm,
                    dp_equal_tokens=dp_equal_tokens,
                    use_symm_p2p_combine=use_symm_p2p_combine,
                    dense_linear_mode=dense_linear_mode,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Final mixer collapses 4-stream HC -> single hidden_size vector (NO model.norm!)
        self.hyper_connection_mixer = Qwen4_5HyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=False,
            prev_mix=False,
            is_head=False,
            emits_latent=True,
            dtype=dtype,
            device=device,
            **_filter_init_kwargs(
                Qwen4_5HyperConnection.__init__,
                {"sp_size": self.sp_size, "sp_rank": self.sp_rank, "sp_group": self.sp_group, "dp_attn": self.dp_attn},
            ),
        )
        # Freeze inference parameters so Qwen4_5HyperConnection / PLE hit cached fused weights.
        self.requires_grad_(False)

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        transpose_mxfp4: bool = False,
        dense_linear_mode: Optional[str] = None,
    ) -> None:
        if dense_linear_mode is not None:
            self._dense_linear_mode = dense_linear_mode
        self.requires_grad_(False)
        layers_mod = self._modules["layers"]
        layer_iter = layers_mod._modules.values() if isinstance(layers_mod, nn.ModuleList) else layers_mod
        for layer in layer_iter:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            layer.prepare_weights(
                transpose_mxfp4=transpose_mxfp4,
                dense_linear_mode=dense_linear_mode,
            )
            for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                if hc.emits_latent and hc.use_combine and hasattr(hc, "_get_fused_down_inject_weight"):
                    hc._get_fused_down_inject_weight()
        if (
            self.all_to_all_comm is not None
            and not torch.cuda.is_current_stream_capturing()
            and hasattr(self.all_to_all_comm, "_ensure_tbo_resources")
        ):
            self.all_to_all_comm._ensure_tbo_resources()

    def create_runtime_state(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Qwen4_5RuntimeState:
        """Allocate minimal YoCo-aware KV cache, GDN SSM, ShortConv, and PLE states."""
        dev = device if device is not None else self.embed_tokens.weight.device
        dt = dtype if dtype is not None else self.embed_tokens.weight.dtype
        yoco_on = (
            self.config.enable_yoco_decode
            if enable_yoco_decode is None
            else bool(enable_yoco_decode)
        )
        state = Qwen4_5RuntimeState(batch_size=batch_size, max_seq_len=max_seq_len)
        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            if layer.ple is not None:
                state.ple[lid] = layer.ple.alloc_state(batch_size, device=dev, dtype=dt)
            if layer.linear_attn is not None:
                state.gdn[lid] = layer.linear_attn.create_state(
                    batch_size,
                    device=dev,
                    dtype=dt,
                    allocate_consumer_ssm=not yoco_on,
                )
            elif layer.self_attn is not None and not layer.is_consumer:
                kv = layer.self_attn.create_kv_cache(
                    batch_size, max_seq_len, device=dev, dtype=dt
                )
                if kv is not None:
                    state.full_attn_kv[lid] = kv
                k_sc, v_sc = layer.self_attn.create_sconv_states(
                    batch_size, device=dev, dtype=dt
                )
                if k_sc is not None:
                    state.k_sconv[lid] = k_sc
                if v_sc is not None:
                    state.v_sconv[lid] = v_sc
                if layer.self_attn.indexer is not None:
                    state.qsa_raw_keys[lid] = torch.empty(
                        (batch_size, 0, self.config.indexer_head_dim),
                        dtype=dt,
                        device=dev,
                    )
            if layer.attn_output_sconv is not None:
                state.attn_output_sconv[lid] = layer.attn_output_sconv.alloc_state(
                    batch_size, device=dev, dtype=dt
                )
            if layer.ffn_output_sconv is not None:
                state.ffn_output_sconv[lid] = layer.ffn_output_sconv.alloc_state(
                    batch_size, device=dev, dtype=dt
                )
        return state

    @torch.no_grad()
    def init_synthetic_weights(
        self,
        *,
        seed: int = 42,
        ple_mode: str = "gpu_sharded",
        ple_shard_rows: Optional[int] = None,
        ple_fp8: bool = True,
    ) -> None:
        """Populate well-conditioned synthetic weights deterministically across EP ranks.

        All replicated parameters use identical RNG streams across ranks, while
        MXFP4 routed experts and PLE shards are seeded per global index so
        ``EP=1``, ``EP=4``, and ``EP=8`` materialize the exact same mathematical
        model.
        """
        cfg = self.config
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        def _gen(offset: int) -> torch.Generator:
            g = torch.Generator(device=device)
            g.manual_seed(seed + offset)
            return g

        # Global embedding + final HC mixer
        self.embed_tokens.weight.normal_(0.0, 0.02, generator=_gen(1))
        for idx, p in enumerate(self.hyper_connection_mixer.parameters()):
            if p.ndim == 1:
                p.zero_()
            else:
                p.normal_(0.0, 0.02, generator=_gen(100 + idx))

        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            lid = layer.layer_idx
            base_seed = 10_000 + lid * 1_000

            def _slice_for_tp_param(p_name: str, full_t: torch.Tensor) -> torch.Tensor:
                if self.tp_size <= 1:
                    return full_t
                tp_s = self.tp_size
                tp_r = self.tp_rank
                if p_name == "self_attn.q_proj.weight":
                    local = full_t.shape[0] // tp_s
                    return full_t[tp_r * local : (tp_r + 1) * local]
                if p_name in ("self_attn.o_proj.weight", "mlp.down_proj.weight", "linear_attn.out_proj.weight"):
                    local = full_t.shape[1] // tp_s
                    return full_t[:, tp_r * local : (tp_r + 1) * local]
                if p_name in (
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "linear_attn.in_proj_qkvz.weight",
                    "linear_attn.in_proj_ba.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                ):
                    local = full_t.shape[0] // tp_s
                    return full_t[tp_r * local : (tp_r + 1) * local]
                if p_name == "linear_attn.conv1d.weight":
                    if layer.is_consumer:
                        local = full_t.shape[0] // tp_s
                        return full_t[tp_r * local : (tp_r + 1) * local]
                    full_k = cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    full_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim
                    loc_k = full_k // tp_s
                    loc_v = full_v // tp_s
                    q_seg, k_seg, v_seg = torch.split(full_t, [full_k, full_k, full_v], dim=0)
                    return torch.cat(
                        [
                            q_seg[tp_r * loc_k : (tp_r + 1) * loc_k],
                            k_seg[tp_r * loc_k : (tp_r + 1) * loc_k],
                            v_seg[tp_r * loc_v : (tp_r + 1) * loc_v],
                        ],
                        dim=0,
                    )
                return full_t

            def _full_shape_for_tp_param(p_name: str, local_shape: Tuple[int, ...]) -> Tuple[int, ...]:
                if self.tp_size <= 1:
                    return local_shape
                tp_s = self.tp_size
                if p_name == "self_attn.q_proj.weight":
                    return (local_shape[0] * tp_s, *local_shape[1:])
                if p_name in ("self_attn.o_proj.weight", "mlp.down_proj.weight", "linear_attn.out_proj.weight"):
                    return (local_shape[0], local_shape[1] * tp_s, *local_shape[2:])
                if p_name in (
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "linear_attn.in_proj_qkvz.weight",
                    "linear_attn.in_proj_ba.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.conv1d.weight",
                ):
                    return (local_shape[0] * tp_s, *local_shape[1:])
                return local_shape

            # Deterministic small-variance init (sliced identically across TP ranks)
            for p_idx, (name, param) in enumerate(layer.named_parameters()):
                g = _gen(base_seed + p_idx)
                full_shape = _full_shape_for_tp_param(name, tuple(param.shape))
                buf = (
                    param
                    if full_shape == tuple(param.shape)
                    else torch.empty(full_shape, dtype=param.dtype, device=param.device)
                )
                if name.endswith("A_log"):
                    # Match BF16 storage precision of real 1109 checkpoint (cast to FP32 at runtime)
                    buf.uniform_(-2.0, -0.5, generator=g)
                    buf.copy_(buf.to(torch.bfloat16).float())
                elif name.endswith("dt_bias"):
                    buf.uniform_(-1.0, 0.5, generator=g)
                elif "expert_bias" in name:
                    buf.uniform_(-0.05, 0.05, generator=g)
                elif buf.ndim == 1:
                    if "norm" in name and "linear_attn" in name:
                        buf.fill_(1.0)
                    else:
                        buf.zero_()
                elif buf.ndim == 3:
                    # Depthwise ShortConv / GDN conv1d weights
                    buf.normal_(0.0, 0.05, generator=g)
                elif name == "mlp.gate.weight":
                    # Realistic O(1) router logit scale so top-k routing is well-conditioned
                    std = 2.0 / math.sqrt(max(full_shape[-1], 1))
                    buf.normal_(0.0, std, generator=g)
                else:
                    std = min(0.02, 1.0 / math.sqrt(max(full_shape[-1], 1)))
                    buf.normal_(0.0, std, generator=g)
                if buf is not param:
                    param.copy_(_slice_for_tp_param(name, buf))

            # Deterministic per-expert MXFP4 routed weights (exact across EP sizes)
            if isinstance(layer.mlp, Qwen4_5LatentMoE):
                moe = layer.mlp
                assert (
                    moe.w13_u8 is not None
                    and moe.w13_scale is not None
                    and moe.w2_u8 is not None
                    and moe.w2_scale is not None
                )
                # Populate each local expert from its global expert id so EP1==EP4==EP8.
                # Exponent 121..123 keeps activations well-scaled across 40 layers.
                for local_e in range(moe.local_num_experts):
                    global_e = moe.ep_start + local_e
                    ge = _gen(500_000 + lid * 10_000 + global_e)
                    moe.w13_u8[local_e].random_(0, 256, generator=ge)
                    moe.w13_scale[local_e].random_(121, 124, generator=ge)
                    moe.w2_u8[local_e].random_(0, 256, generator=ge)
                    moe.w2_scale[local_e].random_(121, 124, generator=ge)

            # Deterministic PLE n-gram table shards
            if layer.ple is not None:
                ple = layer.ple
                total_shards = cfg.split_ngram_parts
                if ple_shard_rows is None:
                    align = math.lcm(
                        max(1, int(cfg.make_ngram_vocab_size_divisible_by)),
                        int(total_shards),
                    )
                    padded_rows = ((ple.total_vocab_size + align - 1) // align) * align
                    shard_rows = padded_rows // total_shards
                else:
                    shard_rows = int(ple_shard_rows)

                table_dtype = torch.float8_e4m3fn if ple_fp8 else dtype
                ple_group_size = self.tp_size if self.dp_attn else self.ep_size
                ple_group_rank = self.tp_rank if self.dp_attn else self.ep_rank
                ple_group = self.tp_group if self.dp_attn else self.ep_group
                if ple_mode == "gpu_sharded" and ple_group_size > 1:
                    if total_shards % ple_group_size != 0:
                        raise ValueError(
                            f"split_ngram_parts={total_shards} not divisible by ple_group_size={ple_group_size}"
                        )
                    local_shards = total_shards // ple_group_size
                    shard_offset = ple_group_rank * local_shards
                    shards: List[torch.Tensor] = []
                    for s_local in range(local_shards):
                        s_global = shard_offset + s_local
                        gs = _gen(900_000 + lid * 1_000 + s_global)
                        if ple_fp8:
                            # Valid finite FP8 E4M3 codes (exclude NaN 0x7f/0xff)
                            raw = torch.randint(
                                0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            )
                            sign = torch.randint(
                                0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            ) << 7
                            raw |= sign
                            del sign
                            shards.append(raw.view(torch.float8_e4m3fn))
                        else:
                            shards.append(
                                torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device).normal_(
                                    0.0, 0.02, generator=gs
                                )
                            )
                    ple.set_ngram_table(
                        shards,
                        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                        shard_offset=shard_offset,
                        total_shards=total_shards,
                        ep_group=ple_group,
                    )
                elif ple_mode in ("gpu_sharded", "gpu_full"):
                    shards = []
                    for s_global in range(total_shards):
                        gs = _gen(900_000 + lid * 1_000 + s_global)
                        if ple_fp8:
                            raw = torch.randint(
                                0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            )
                            sign = torch.randint(
                                0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, device=device, generator=gs
                            ) << 7
                            raw |= sign
                            del sign
                            shards.append(raw.view(torch.float8_e4m3fn))
                        else:
                            shards.append(
                                torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device).normal_(
                                    0.0, 0.02, generator=gs
                                )
                            )
                    ple.set_ngram_table(shards, weight_scale=DEFAULT_PLE_WEIGHT_SCALE)
                elif ple_mode == "cpu_uva":
                    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
                        alloc_shared_host_ple_shards,
                        finalize_shared_host_ple_shards,
                        resolve_ple_share_group,
                    )

                    share_group = resolve_ple_share_group(self)
                    cpu_gen = torch.Generator(device="cpu")
                    if share_group is not None:
                        shard_specs = [(table_dtype, (shard_rows, ple.head_dim)) for _ in range(total_shards)]
                        shared_buf = alloc_shared_host_ple_shards(
                            shard_specs,
                            group=share_group,
                            device=device,
                        )
                        node_size = len(shared_buf.same_host_ranks)
                        host_local_rank = shared_buf.host_local_rank
                        for s_global in range(total_shards):
                            if s_global % node_size != host_local_rank:
                                continue
                            cpu_gen.manual_seed(seed + 900_000 + lid * 1_000 + s_global)
                            dst_shard = shared_buf.shard_bufs[s_global]
                            if ple_fp8:
                                raw = torch.randint(
                                    0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                                )
                                sign = torch.randint(
                                    0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                                ) << 7
                                raw |= sign
                                dst_shard.copy_(raw.view(torch.float8_e4m3fn))
                            else:
                                tmp_bf = torch.empty((shard_rows, ple.head_dim), dtype=dtype).normal_(
                                    0.0, 0.02, generator=cpu_gen
                                )
                                dst_shard.copy_(tmp_bf)
                        pinned_ok = finalize_shared_host_ple_shards(shared_buf)
                        ple.set_ngram_table(
                            shared_buf.shard_bufs,
                            weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                            pin_cpu=False,
                            enable_cpu_uva=pinned_ok,
                            ep_group=None,
                        )
                    else:
                        shards = []
                        for s_global in range(total_shards):
                            cpu_gen.manual_seed(seed + 900_000 + lid * 1_000 + s_global)
                            if ple_fp8:
                                raw = torch.randint(
                                    0, 120, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                                )
                                sign = torch.randint(
                                    0, 2, (shard_rows, ple.head_dim), dtype=torch.uint8, generator=cpu_gen
                                ) << 7
                                raw |= sign
                                del sign
                                shards.append(raw.view(torch.float8_e4m3fn))
                            else:
                                shards.append(
                                    torch.empty((shard_rows, ple.head_dim), dtype=dtype).normal_(
                                        0.0, 0.02, generator=cpu_gen
                                    )
                                )
                        ple.set_ngram_table(
                            shards,
                            weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                            pin_cpu=True,
                            enable_cpu_uva=True,
                        )
                else:
                    raise ValueError(f"Unsupported ple_mode={ple_mode!r}")

    def _embed_hyper_streams(
        self,
        input_ids: torch.Tensor,
        *,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> torch.Tensor:
        embed_tokens = self._modules["embed_tokens"]
        if profiler is not None:
            with profiler.record("embed"):
                emb = embed_tokens(input_ids)
                return emb.repeat(1, 1, self.config.hc_count)
        emb = embed_tokens(input_ids)
        return emb.repeat(1, 1, self.config.hc_count)

    def _finalize_hyper_streams(
        self,
        hyper_hidden: torch.Tensor,
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> torch.Tensor:
        assert deferred is not None
        hm = self._modules["hyper_connection_mixer"]
        if profiler is not None:
            with profiler.record("final_hc"):
                hidden, _, _, _ = hm.combine_and_mix_with_latent(
                    deferred[0], hyper_hidden, deferred[1], prev_latent=None
                )
        elif use_reference:
            hidden, _, _, _ = hm.combine_and_mix_with_latent_ref(
                deferred[0], hyper_hidden, deferred[1], prev_latent=None
            )
        else:
            hidden, _, _, _ = hm.combine_and_mix_with_latent(
                deferred[0], hyper_hidden, deferred[1], prev_latent=None
            )
        if state is not None:
            state.seq_len += seq_len
        return hidden

    def forward_tbo(
        self,
        microbatches: Sequence[torch.Tensor],
        *,
        states: Optional[Sequence[Optional[Qwen4_5RuntimeState]]] = None,
        positions: Optional[Sequence[Optional[torch.Tensor]]] = None,
        seq_offsets: Optional[Sequence[Optional[int]]] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        equal_tokens: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 3-stage Two-Batch Overlap (TBO) forward across two microbatches."""
        if len(microbatches) != 2:
            raise ValueError(f"Expected exactly 2 microbatches for TBO, got {len(microbatches)}")
        ids0, ids1 = microbatches[0], microbatches[1]
        if ids0.ndim != 2 or ids1.ndim != 2:
            raise ValueError(
                f"Expected [B, T] microbatches, got {tuple(ids0.shape)} and {tuple(ids1.shape)}"
            )
        st0, st1 = (states[0], states[1]) if states is not None else (None, None)
        pos0, pos1 = (positions[0], positions[1]) if positions is not None else (None, None)
        off0, off1 = (seq_offsets[0], seq_offsets[1]) if seq_offsets is not None else (None, None)

        if (
            self.all_to_all_comm is not None
            and not torch.cuda.is_current_stream_capturing()
            and hasattr(self.all_to_all_comm, "_ensure_tbo_resources")
        ):
            self.all_to_all_comm._ensure_tbo_resources()

        hyp0 = self._embed_hyper_streams(ids0, profiler=profiler)
        hyp1 = self._embed_hyper_streams(ids1, profiler=profiler)
        lat0: Optional[torch.Tensor] = None
        lat1: Optional[torch.Tensor] = None
        def0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        def1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        gdn0: Dict[int, SharedGDNInputs] = {}
        gdn1: Dict[int, SharedGDNInputs] = {}
        qsa0: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        qsa1: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        seq0, seq1 = int(ids0.shape[1]), int(ids1.shape[1])
        past0 = st0.seq_len if st0 is not None else 0
        past1 = st1.seq_len if st1 is not None else 0
        ssm0: Optional[Dict[int, torch.Tensor]] = {} if (st0 is not None and seq0 > 1 and past0 > 0) else None
        ssm1: Optional[Dict[int, torch.Tensor]] = {} if (st1 is not None and seq1 > 1 and past1 > 0) else None

        precomputed_mb0: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = None
        num_layers = len(self.layers)
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, Qwen4_5DecoderLayer)
            if isinstance(layer.mlp, Qwen4_5LatentMoE):
                if precomputed_mb0 is not None:
                    mlp_in0, hyp0, inj0, lat0 = precomputed_mb0
                    precomputed_mb0 = None
                else:
                    mlp_in0, hyp0, inj0, lat0 = layer.forward_pre_mlp(
                        hyp0,
                        ids0,
                        prev_latent=lat0,
                        deferred=def0,
                        shared_gdn_by_source=gdn0,
                        shared_qsa_by_source=qsa0,
                        source_prefix_ssm=ssm0,
                        state=st0,
                        positions=pos0,
                        seq_offset=off0,
                        enable_yoco_decode=enable_yoco_decode,
                        use_reference=use_reference,
                        profiler=profiler,
                    )
                ctx0 = layer.mlp.forward_route_and_dispatch(
                    mlp_in0,
                    use_triton_mxfp4=not use_reference,
                    equal_tokens=equal_tokens,
                    slot=0,
                )
                mlp_in1, hyp1, inj1, lat1 = layer.forward_pre_mlp(
                    hyp1,
                    ids1,
                    prev_latent=lat1,
                    deferred=def1,
                    shared_gdn_by_source=gdn1,
                    shared_qsa_by_source=qsa1,
                    source_prefix_ssm=ssm1,
                    state=st1,
                    positions=pos1,
                    seq_offset=off1,
                    enable_yoco_decode=enable_yoco_decode,
                    use_reference=use_reference,
                    profiler=profiler,
                )
                ctx1 = layer.mlp.forward_route_and_dispatch(
                    mlp_in1,
                    use_triton_mxfp4=not use_reference,
                    equal_tokens=equal_tokens,
                    slot=1,
                )
                ctx0 = layer.mlp.forward_routed_compute_and_combine(ctx0, slot=0)
                ctx1 = layer.mlp.forward_routed_compute_and_combine(ctx1, slot=1)
                out0 = layer.mlp.forward_finish_combine(ctx0)
                hyp0, lat0, def0 = layer.forward_post_mlp(
                    out0,
                    hyp0,
                    lat0,
                    inj0,
                    state=st0,
                    use_reference=use_reference,
                    profiler=profiler,
                )
                # When tp_size == 1 (pure dp_ep), next_layer.forward_pre_mlp has no TP
                # collectives and can overlap Layer L mb1 combine on comm_stream cleanly.
                next_layer = (
                    self.layers[layer_idx + 1] if (layer_idx + 1 < num_layers) else None
                )
                if (
                    self.tp_size == 1
                    and isinstance(next_layer, Qwen4_5DecoderLayer)
                    and isinstance(next_layer.mlp, Qwen4_5LatentMoE)
                ):
                    precomputed_mb0 = next_layer.forward_pre_mlp(
                        hyp0,
                        ids0,
                        prev_latent=lat0,
                        deferred=def0,
                        shared_gdn_by_source=gdn0,
                        shared_qsa_by_source=qsa0,
                        source_prefix_ssm=ssm0,
                        state=st0,
                        positions=pos0,
                        seq_offset=off0,
                        enable_yoco_decode=enable_yoco_decode,
                        use_reference=use_reference,
                        profiler=profiler,
                    )
                out1 = layer.mlp.forward_finish_combine(ctx1)
                hyp1, lat1, def1 = layer.forward_post_mlp(
                    out1,
                    hyp1,
                    lat1,
                    inj1,
                    state=st1,
                    use_reference=use_reference,
                    profiler=profiler,
                )
            else:
                hyp0, lat0, def0 = layer(
                    hyp0,
                    ids0,
                    prev_latent=lat0,
                    deferred=def0,
                    shared_gdn_by_source=gdn0,
                    shared_qsa_by_source=qsa0,
                    source_prefix_ssm=ssm0,
                    state=st0,
                    positions=pos0,
                    seq_offset=off0,
                    enable_yoco_decode=enable_yoco_decode,
                    use_reference=use_reference,
                    profiler=profiler,
                )
                hyp1, lat1, def1 = layer(
                    hyp1,
                    ids1,
                    prev_latent=lat1,
                    deferred=def1,
                    shared_gdn_by_source=gdn1,
                    shared_qsa_by_source=qsa1,
                    source_prefix_ssm=ssm1,
                    state=st1,
                    positions=pos1,
                    seq_offset=off1,
                    enable_yoco_decode=enable_yoco_decode,
                    use_reference=use_reference,
                    profiler=profiler,
                )

        hid0 = self._finalize_hyper_streams(
            hyp0, def0, seq0, state=st0, use_reference=use_reference, profiler=profiler
        )
        hid1 = self._finalize_hyper_streams(
            hyp1, def1, seq1, state=st1, use_reference=use_reference, profiler=profiler
        )
        return hid0, hid1

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        enable_tbo: bool = False,
        tbo_equal_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run full backbone forward returning collapsed ``[B, T, hidden_size]``."""
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids of shape [B, T], got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if enable_tbo and state is None and bsz >= 2:
            b0 = bsz // 2
            pos_pair = (positions[:b0], positions[b0:]) if positions is not None else None
            h0, h1 = self.forward_tbo(
                (input_ids[:b0], input_ids[b0:]),
                positions=pos_pair,
                seq_offsets=(seq_offset, seq_offset),
                enable_yoco_decode=enable_yoco_decode,
                use_reference=use_reference,
                profiler=profiler,
                equal_tokens=tbo_equal_tokens,
            )
            return torch.cat([h0, h1], dim=0)

        hyper_hidden = self._embed_hyper_streams(input_ids, profiler=profiler)
        prev_latent: Optional[torch.Tensor] = None
        deferred: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        shared_gdn_by_source: Dict[int, SharedGDNInputs] = {}
        shared_qsa_by_source: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        past_len = state.seq_len if state is not None else 0
        source_prefix_ssm: Optional[Dict[int, torch.Tensor]] = (
            {} if (state is not None and seq_len > 1 and past_len > 0) else None
        )

        for layer in self.layers:
            assert isinstance(layer, Qwen4_5DecoderLayer)
            hyper_hidden, prev_latent, deferred = layer(
                hyper_hidden,
                input_ids,
                prev_latent=prev_latent,
                deferred=deferred,
                shared_gdn_by_source=shared_gdn_by_source,
                shared_qsa_by_source=shared_qsa_by_source,
                source_prefix_ssm=source_prefix_ssm,
                state=state,
                positions=positions,
                seq_offset=seq_offset,
                enable_yoco_decode=enable_yoco_decode,
                use_reference=use_reference,
                profiler=profiler,
                enable_moe_tbo=enable_tbo,
                tbo_equal_tokens=tbo_equal_tokens,
            )

        return self._finalize_hyper_streams(
            hyper_hidden,
            deferred,
            seq_len,
            state=state,
            use_reference=use_reference,
            profiler=profiler,
        )


class Qwen4_5ForCausalLM(nn.Module):
    """Causal LM wrapper around ``Qwen4_5Model`` with vocab-parallel ``lm_head``."""

    def __init__(
        self,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        parallel_mode: Optional[str] = None,
        ep_size: Optional[int] = None,
        ep_rank: Optional[int] = None,
        ep_group: Any = None,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Any = None,
        dp_size: Optional[int] = None,
        dp_rank: Optional[int] = None,
        dp_group: Any = None,
        sp_size: Optional[int] = None,
        sp_rank: Optional[int] = None,
        sp_group: Any = None,
        dp_attn: Optional[bool] = None,
        kv_dtype: str = "bf16",
        fp8_indexer_kv: Optional[bool] = None,
        all_to_all_comm: Any = None,
        dp_equal_tokens: Optional[bool] = None,
        use_symm_p2p_combine: Optional[bool] = None,
        dense_linear_mode: Optional[str] = None,
        fast_all_reduce: Any = None,
        fast_ar_max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen4_5Model(
            config,
            use_triton_mxfp4=use_triton_mxfp4,
            compute_mode=compute_mode,
            dtype=dtype,
            device=device,
            init_mxfp4_weights=init_mxfp4_weights,
            parallel_mode=parallel_mode,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group=dp_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            dp_attn=dp_attn,
            kv_dtype=kv_dtype,
            fp8_indexer_kv=fp8_indexer_kv,
            all_to_all_comm=all_to_all_comm,
            dp_equal_tokens=dp_equal_tokens,
            use_symm_p2p_combine=use_symm_p2p_combine,
            dense_linear_mode=dense_linear_mode,
            fast_all_reduce=fast_all_reduce,
            fast_ar_max_tokens=fast_ar_max_tokens,
        )
        self.parallel_topology = self.model.parallel_topology
        self.parallel_mode = self.model.parallel_mode
        self.tp_size = self.model.tp_size
        self.tp_rank = self.model.tp_rank
        self.tp_group = self.model.tp_group
        self.ep_size = self.model.ep_size
        self.ep_rank = self.model.ep_rank
        self.ep_group = self.model.ep_group
        self.dp_size = self.model.dp_size
        self.dp_rank = self.model.dp_rank
        self.dp_group = self.model.dp_group
        self.sp_size = self.model.sp_size
        self.sp_rank = self.model.sp_rank
        self.sp_group = self.model.sp_group
        self.dp_attn = self.model.dp_attn
        self.kv_dtype = self.model.kv_dtype
        self.fast_all_reduce = self.model.fast_all_reduce
        self._dense_linear_mode = dense_linear_mode
        if config.vocab_size % self.tp_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be divisible by tp_size ({self.tp_size})"
            )
        from rtp_llm.models_py.modules.qwen4_5.fp8_linear import SM100FP8Linear

        self.local_vocab_size = config.vocab_size // self.tp_size
        init_lm_mode = _resolve_dense_linear_mode(dense_linear_mode, None)
        lm_extra_kw = _filter_init_kwargs(
            SM100FP8Linear.__init__,
            {"allow_decode_scaled_mm": True},
        )
        self.lm_head: nn.Module = SM100FP8Linear(
            in_features=config.hidden_size,
            out_features=self.local_vocab_size,
            bias=False,
            mode=init_lm_mode,
            keep_bf16=True,
            prepare_trtllm_ll=True,
            prepare_mxfp8=True,
            prepare_w4a8=(init_lm_mode == "w4a8"),
            dtype=dtype,
            device=device,
            **lm_extra_kw,
        )
        self.requires_grad_(False)

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        transpose_mxfp4: bool = False,
        dense_linear_mode: Optional[str] = None,
    ) -> None:
        eff_mode = _resolve_dense_linear_mode(dense_linear_mode, self._dense_linear_mode)
        if dense_linear_mode is not None:
            self._dense_linear_mode = dense_linear_mode
        self.model.prepare_weights(
            transpose_mxfp4=transpose_mxfp4,
            dense_linear_mode=dense_linear_mode,
        )
        self.lm_head = _wrap_linear_with_sm100_fp8(
            self.lm_head, eff_mode, allow_decode_scaled_mm=True
        )

    def create_runtime_state(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_yoco_decode: Optional[bool] = None,
    ) -> Qwen4_5RuntimeState:
        return self.model.create_runtime_state(
            batch_size,
            max_seq_len,
            device=device,
            dtype=dtype,
            enable_yoco_decode=enable_yoco_decode,
        )

    @torch.no_grad()
    def init_synthetic_weights(
        self,
        *,
        seed: int = 42,
        ple_mode: str = "gpu_sharded",
        ple_shard_rows: Optional[int] = None,
        ple_fp8: bool = True,
    ) -> None:
        self.model.init_synthetic_weights(
            seed=seed,
            ple_mode=ple_mode,
            ple_shard_rows=ple_shard_rows,
            ple_fp8=ple_fp8,
        )
        g = torch.Generator(device=self.lm_head.weight.device)
        g.manual_seed(seed + 7_777_777)
        if self.tp_size > 1:
            full_lm = torch.empty(
                (self.config.vocab_size, self.config.hidden_size),
                dtype=self.lm_head.weight.dtype,
                device=self.lm_head.weight.device,
            )
            full_lm.normal_(0.0, 0.02, generator=g)
            s0 = self.tp_rank * self.local_vocab_size
            self.lm_head.weight.copy_(full_lm[s0 : s0 + self.local_vocab_size])
        else:
            self.lm_head.weight.normal_(0.0, 0.02, generator=g)

    def _project_logits(
        self,
        hidden: torch.Tensor,
        *,
        logits_to_keep: Optional[int] = None,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
    ) -> torch.Tensor:
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        lm_head = self._modules["lm_head"]
        if profiler is not None:
            with profiler.record("lm_head"):
                logits = lm_head(hidden)
        else:
            logits = lm_head(hidden)
        if (
            self.tp_size > 1
            and self.tp_group is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            if self.fast_all_reduce is not None and self.fast_all_reduce.group is self.tp_group:
                logits = self.fast_all_reduce.all_gather(logits, dim=-1)
            else:
                gathered = [torch.empty_like(logits) for _ in range(self.tp_size)]
                dist.all_gather(gathered, logits.contiguous(), group=self.tp_group)
                logits = torch.cat(gathered, dim=-1)
        return logits

    def forward_tbo(
        self,
        microbatches: Sequence[torch.Tensor],
        *,
        states: Optional[Sequence[Optional[Qwen4_5RuntimeState]]] = None,
        positions: Optional[Sequence[Optional[torch.Tensor]]] = None,
        seq_offsets: Optional[Sequence[Optional[int]]] = None,
        enable_yoco_decode: Optional[bool] = None,
        logits_to_keep: Optional[int] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        equal_tokens: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 3-stage TBO forward on two microbatches and gather vocab-sharded logits."""
        hid0, hid1 = self.model.forward_tbo(
            microbatches,
            states=states,
            positions=positions,
            seq_offsets=seq_offsets,
            enable_yoco_decode=enable_yoco_decode,
            use_reference=use_reference,
            profiler=profiler,
            equal_tokens=equal_tokens,
        )
        keep_tail = int(logits_to_keep) if (logits_to_keep is not None and logits_to_keep > 0) else 0
        if keep_tail > 0:
            hid0_k = hid0[:, -keep_tail:, :]
            hid1_k = hid1[:, -keep_tail:, :]
        else:
            hid0_k, hid1_k = hid0, hid1
        if hid0_k.shape[1:] == hid1_k.shape[1:]:
            b0 = int(hid0_k.shape[0])
            logits_cat = self._project_logits(
                torch.cat((hid0_k, hid1_k), dim=0),
                logits_to_keep=None,
                profiler=profiler,
            )
            return logits_cat[:b0], logits_cat[b0:]
        return (
            self._project_logits(hid0_k, logits_to_keep=None, profiler=profiler),
            self._project_logits(hid1_k, logits_to_keep=None, profiler=profiler),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: Optional[Qwen4_5RuntimeState] = None,
        positions: Optional[torch.Tensor] = None,
        seq_offset: Optional[int] = None,
        enable_yoco_decode: Optional[bool] = None,
        logits_to_keep: Optional[int] = None,
        use_reference: bool = False,
        profiler: Optional[Qwen4_5ProfileRecorder] = None,
        enable_tbo: bool = False,
        tbo_equal_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids,
            state=state,
            positions=positions,
            seq_offset=seq_offset,
            enable_yoco_decode=enable_yoco_decode,
            use_reference=use_reference,
            profiler=profiler,
            enable_tbo=enable_tbo,
            tbo_equal_tokens=tbo_equal_tokens,
        )
        return self._project_logits(hidden, logits_to_keep=logits_to_keep, profiler=profiler)


__all__ = [
    "run_latent_moe_tbo_tokens",
    "Qwen4_5DecoderLayer",
    "Qwen4_5ForCausalLM",
    "Qwen4_5Model",
    "Qwen4_5PagedStateBridge",
    "Qwen4_5ProfileRecorder",
    "Qwen4_5RTPModel",
]

def __getattr__(name: str) -> Any:
    if name in ("Qwen4_5RTPModel", "Qwen4_5PagedStateBridge"):
        from rtp_llm.models_py.model_desc import qwen4_5_runtime
        return getattr(qwen4_5_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
