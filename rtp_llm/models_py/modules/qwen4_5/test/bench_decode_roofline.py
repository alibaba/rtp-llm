#!/usr/bin/env python3
"""Standalone 4-GPU Decode Roofline & Parity Benchmark for Qwen 4.5-Flash (5L Target Verify + 5L DFlash2).

Executes directly on Whale D (4x NVIDIA L20A SM100) via `torchrun --nproc_per_node=4`:
- Loads full-width 5L QSA-Sparse Target + 5L DFlash2 Draft weights from local safetensors
- Pre-populates paged Target & DFlash2 KV/OPAQUE_STATE caches at `avg_ctx = 20,480`
- Runs CUDA-Graph-captured `DFlash2 Draft (query_width=8) + CandidateSelector + Target Verify (V=8)` cycles
- Computes empirical hardware ceilings & per-operator / whole-network MFU% & MBU%
- Captures Rank 0 Kineto Top-15 kernels, saves/checks BF16 reference parity, and updates `leaderboard_decode.json`
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import math
import os
import statistics
import sys
import time
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_5.tools.roofline_utils import (
    PARALLEL_MODES,
    DFlash2ModelSpec,
    HardwareRoofline,
    calibrate_l20a_roofline,
    compute_qwen45_workload_metrics,
)


def _resolve_parallel_layout(
    parallel_mode: str,
    world_rank: int,
    world_size: int,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Return ``(dp_size, dp_rank, tp_size, tp_rank, ep_size, ep_rank, sp_size, sp_rank)``."""
    pm = str(parallel_mode).strip().lower()
    if pm == "dp4_ep4":
        return world_size, world_rank, 1, 0, world_size, world_rank, 1, 0
    if pm in ("dp2_tp2_ep4", "dp2_tpsp2_ep4"):
        tp = 2 if world_size >= 2 else 1
        dp = max(1, world_size // tp)
        dp_r = world_rank // tp
        tp_r = world_rank % tp
        sp = tp if pm == "dp2_tpsp2_ep4" else 1
        sp_r = tp_r if sp > 1 else 0
        return dp, dp_r, tp, tp_r, world_size, world_rank, sp, sp_r
    if pm == "sp4_tp4_ep4":
        return 1, 0, world_size, world_rank, world_size, world_rank, world_size, world_rank
    return 1, 0, world_size, world_rank, world_size, world_rank, 1, 0


def _resolve_global_ctx_list(
    batch_size: int,
    *,
    avg_ctx: int,
    skew_preset: str,
    ctx_list_arg: Sequence[int],
    skew_layout: str = "rank_skewed",
) -> List[int]:
    """Build per-request prefix context lengths for uniform or skewed Decode batches."""
    if batch_size <= 0:
        return []
    if ctx_list_arg:
        base = [max(1, int(c)) for c in ctx_list_arg]
        ctxs = [base[i % len(base)] for i in range(batch_size)]
        if skew_layout == "rank_skewed" and len(ctxs) > len(base):
            ctxs.sort(reverse=True)
        return ctxs
    preset = str(skew_preset or "").strip().lower()
    if not preset:
        return [int(avg_ctx)] * batch_size
    if preset == "40k_2k":
        long_ctx, short_ctx = 40960, 2048
        n_long = max(1, batch_size // 4)
    elif preset == "avg20k_skew":
        if batch_size < 4:
            return [int(avg_ctx)] * batch_size
        # 1/4 @ 75776 + 3/4 @ 2048 -> exact mean 20,480 (37x context skew, multiple of 64)
        long_ctx, short_ctx = 75776, 2048
        n_long = batch_size // 4
    elif preset == "straggler_40k_2k":
        long_ctx, short_ctx = 40960, 2048
        n_long = 1
    else:
        raise ValueError(f"Unsupported skew_preset={skew_preset!r}")
    if skew_layout == "interleaved" and preset in ("40k_2k", "avg20k_skew"):
        return [long_ctx if (i % 4 == 0) else short_ctx for i in range(batch_size)]
    return [long_ctx] * n_long + [short_ctx] * (batch_size - n_long)


def _compute_dp_kv_imbalance(global_ctxs: Sequence[int], dp_size: int) -> Tuple[int, int, float]:
    """Compute min/max per-DP-rank KV prefix tokens and max/min imbalance ratio."""
    if not global_ctxs:
        return 0, 0, 1.0
    if dp_size <= 1:
        tot = int(sum(global_ctxs))
        return tot, tot, 1.0
    bs_local = len(global_ctxs) // dp_size
    rank_sums = [int(sum(global_ctxs[r * bs_local : (r + 1) * bs_local])) for r in range(dp_size)]
    mn = min(rank_sums)
    mx = max(rank_sums)
    return mn, mx, round(mx / max(1, mn), 2)


def load_or_calibrate_roofline(
    device: torch.device,
    world_size: int,
    world_grp: Any,
    *,
    roofline_cache_json: Optional[Path] = None,
) -> HardwareRoofline:
    """Load cached L20A roofline JSON or calibrate live across all ranks."""
    has_cache = bool(
        roofline_cache_json is not None and roofline_cache_json.is_file()
    )
    flag = torch.tensor([1 if has_cache else 0], dtype=torch.int32, device=device)
    if dist.is_initialized() and world_size > 1:
        dist.broadcast(flag, src=0, group=world_grp)
    if int(flag.item()) == 1 and roofline_cache_json is not None:
        try:
            cached = HardwareRoofline.load_json(roofline_cache_json)
            if int(cached.world_size) == world_size:
                return cached
        except Exception:
            pass
    return calibrate_l20a_roofline(
        device=device,
        json_path=roofline_cache_json,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Fast Sharded 5L Weight Loaders (Target QSA + DFlash2)
# ---------------------------------------------------------------------------


def _slice_dspark_tensor(
    wm_name: str,
    raw_list: List[torch.Tensor],
    process_fun: Any,
    target_dt: torch.dtype,
    tp_size: int,
    tp_rank: int,
    spec: Any,
) -> torch.Tensor:
    """Slice and transform one AtomicWeight for (tp_size, tp_rank) without loading peer shards."""
    from rtp_llm.utils.model_weight import W
    from rtp_llm.models.qwen4_5.qwen4_5_dspark import DFLASH2_W

    t_eval = process_fun([t.to(dtype=target_dt) for t in raw_list])
    if tp_size <= 1:
        return t_eval.contiguous()

    # Keep W.embedding replicated [V, H] so _embed_tokens skips CG all_gather.
    if wm_name == W.embedding:
        return t_eval.contiguous()

    if wm_name == W.lm_head:
        vocab_size = int(spec.vocab_size)
        per_rank_pad8 = ((vocab_size + tp_size - 1) // tp_size + 7) // 8 * 8
        v0 = min(tp_rank * per_rank_pad8, vocab_size)
        v1 = min((tp_rank + 1) * per_rank_pad8, vocab_size)
        sl = t_eval[v0:v1]
        if int(sl.shape[0]) < per_rank_pad8:
            sl = F.pad(sl, (0, 0, 0, per_rank_pad8 - int(sl.shape[0])))
        return sl.contiguous()

    if wm_name == W.attn_qkv_w:
        q_dim = int(spec.num_attention_heads) * int(spec.head_dim)
        kv_dim = int(spec.num_key_value_heads) * int(spec.head_dim)
        # merge_qkv_hf returned [H, q_dim + 2 * kv_dim]
        q_loc = q_dim // tp_size
        q_part = t_eval[:, tp_rank * q_loc : (tp_rank + 1) * q_loc]
        kv_part = t_eval[:, q_dim : q_dim + 2 * kv_dim]
        return torch.cat([q_part, kv_part], dim=1).contiguous()

    if wm_name == W.attn_o_w:
        # transpose returned [q_dim, H]
        q_dim = int(spec.num_attention_heads) * int(spec.head_dim)
        q_loc = q_dim // tp_size
        return t_eval[tp_rank * q_loc : (tp_rank + 1) * q_loc, :].contiguous()

    if wm_name == DFLASH2_W.attn_output_gate:
        # raw weight [q_dim, H]
        q_dim = int(spec.num_attention_heads) * int(spec.head_dim)
        q_loc = q_dim // tp_size
        return t_eval[tp_rank * q_loc : (tp_rank + 1) * q_loc, :].contiguous()

    return t_eval.contiguous()


def load_dspark_5l_sharded(
    ckpt_dir: Path,
    *,
    parallelism_config: Any,
    device: torch.device,
    parallel_mode: str = "tp4_ep4",
    dp_size: Optional[int] = None,
    dp_rank: Optional[int] = None,
    dp_group: Any = None,
    tp_group: Any = None,
    ep_size: Optional[int] = None,
    ep_rank: Optional[int] = None,
    ep_group: Any = None,
    sp_size: Optional[int] = None,
    sp_rank: Optional[int] = None,
    sp_group: Any = None,
    max_generate_batch_size: int = 128,
    dflash_kv_dtype: str = "bf16",
    dflash_attn_backend: str = "auto",
    dense_linear_mode: Optional[str] = None,
    enable_tbo: bool = False,
) -> Tuple[Any, Any, float]:
    """Load 5L full-width DFlash2 weights onto `device` with EP+TP sharding."""
    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.models.qwen4_5.qwen4_5_dspark import Qwen4_5DSpark, Qwen4_5DSparkWeight
    from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import Qwen4_5DSparkModel
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import SafetensorsShardReader

    t0 = time.perf_counter()
    cfg = Qwen4_5DSpark._create_config(str(ckpt_dir))
    spec = cfg._dflash2_spec
    tp_size = max(1, int(getattr(parallelism_config, "tp_size", 1) or 1))
    tp_rank = int(getattr(parallelism_config, "tp_rank", 0) or 0)

    with SafetensorsShardReader(ckpt_dir) as reader:
        wb = Qwen4_5DSparkWeight(cfg, parallelism_config=parallelism_config)
        wb._process_meta({}, list(reader.weight_map.keys()))
        winfo = wb._get_weight_info()

        mw = ModelWeights(cfg.num_layers, device=device, dtype=torch.bfloat16)

        def _load_atomic(aw: Any) -> torch.Tensor:
            target_dt = getattr(aw, "data_type", None) or torch.bfloat16
            raw_list = [reader.get_tensor(ck.name) for ck in aw.weights]
            out = _slice_dspark_tensor(
                aw.name,
                raw_list,
                aw.process_fun,
                target_dt,
                tp_size,
                tp_rank,
                spec,
            )
            return out.to(device=device, dtype=target_dt).contiguous()

        for gw in winfo.weights:
            mw.set_global_weight(gw.name, _load_atomic(gw))

        for lid, lws in enumerate(winfo.layer_weights):
            for lw in lws:
                mw.set_layer_weight(lid, lw.name, _load_atomic(lw))

    sig_params = inspect.signature(Qwen4_5DSparkModel.__init__).parameters
    extra_kw: Dict[str, Any] = {}
    if "parallel_mode" in sig_params:
        extra_kw["parallel_mode"] = parallel_mode
        extra_kw["dp_equal_tokens"] = True
    topo_opts = {
        "dp_size": dp_size,
        "dp_rank": dp_rank,
        "dp_group": dp_group,
        "tp_group": tp_group,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "ep_group": ep_group,
        "sp_size": sp_size,
        "sp_rank": sp_rank,
        "sp_group": sp_group,
    }
    for k, v in topo_opts.items():
        if k in sig_params and v is not None:
            extra_kw[k] = v
    if "dflash_kv_dtype" in sig_params:
        extra_kw["dflash_kv_dtype"] = dflash_kv_dtype
    if "dflash_attn_backend" in sig_params:
        extra_kw["dflash_attn_backend"] = dflash_attn_backend
    if "dense_linear_mode" in sig_params and dense_linear_mode:
        extra_kw["dense_linear_mode"] = dense_linear_mode
    if "enable_tbo" in sig_params:
        extra_kw["enable_tbo"] = bool(enable_tbo)

    model = Qwen4_5DSparkModel(
        cfg,
        parallelism_config,
        mw,
        max_generate_batch_size=max(128, int(max_generate_batch_size)),
        **extra_kw,
    )
    del mw
    torch.cuda.synchronize(device)
    return cfg, model, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 20K Context Paged KV + OPAQUE_STATE Populator & Cycle Runner
# ---------------------------------------------------------------------------


def build_20k_target_kv_cache(
    arch: Any,
    *,
    max_batch: int,
    avg_ctx: int,
    verify_width: int,
    tokens_per_block: int,
    tp_size: int,
    parallel_mode: str = "tp4_ep4",
    kv_dtype: str = "bf16",
    device: torch.device,
    max_total_blocks: Optional[int] = None,
) -> Any:
    from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
        FULL_ATTN_KV_TAG,
        INDEXER_KV_TAG,
        INDEXER_STATE_TAG,
        build_qwen4_5_kv_cache_plan,
    )
    from rtp_llm.models_py.model_desc.qwen4_5_runtime import LocalKVCache

    plan_sig = inspect.signature(build_qwen4_5_kv_cache_plan).parameters
    plan_kw: Dict[str, Any] = {"enable_yoco_decode": True, "tp_size": tp_size}
    if "parallel_mode" in plan_sig:
        plan_kw["parallel_mode"] = parallel_mode
    if "kv_dtype" in plan_sig:
        plan_kw["kv_dtype"] = "FP8" if str(kv_dtype).lower().startswith("fp8") else kv_dtype
    plan = build_qwen4_5_kv_cache_plan(arch, **plan_kw)
    blocks_per_req = (avg_ctx + verify_width + 4 + tokens_per_block - 1) // tokens_per_block + 1
    num_full_blocks = (
        int(max_total_blocks)
        if max_total_blocks is not None
        else (max_batch * blocks_per_req + 64)
    )
    num_opaque_blocks = max_batch + 32

    num_blocks_per_tag: Dict[str, int] = {
        FULL_ATTN_KV_TAG: num_full_blocks,
        INDEXER_KV_TAG: num_full_blocks,
        INDEXER_STATE_TAG: num_opaque_blocks,
    }
    for lp in plan.layers:
        for reg in lp.opaque_regions:
            num_blocks_per_tag.setdefault(reg.tag, num_opaque_blocks)
        if lp.layer_type == "linear_attention":
            num_blocks_per_tag.setdefault("linear", num_opaque_blocks)
            num_blocks_per_tag.setdefault("gdn_consumer_q_conv", num_opaque_blocks)
    is_fp8_kv = str(kv_dtype).lower().startswith("fp8")
    fp8_dt = getattr(torch, "float8_e4m3fn", None)
    lkv_sig = inspect.signature(LocalKVCache.__init__).parameters
    lkv_kw: Dict[str, Any] = {
        "tokens_per_block": tokens_per_block,
        "num_blocks_per_tag": num_blocks_per_tag,
        "device": device,
        "dtype": torch.bfloat16,
    }
    if "kv_dtype" in lkv_sig and is_fp8_kv and fp8_dt is not None:
        lkv_kw["kv_dtype"] = fp8_dt
        if "allocate_fp8_kv_scale" in lkv_sig:
            lkv_kw["allocate_fp8_kv_scale"] = True
    kv_cache = LocalKVCache(plan, **lkv_kw)

    # Populate deterministic max_batch-invariant data (repeating fixed 32-block template)
    def _fill_kv_periodic(t_buf: torch.Tensor, seed_val: int) -> None:
        if t_buf.ndim < 2:
            return
        blk_axis = 0 if t_buf.shape[0] >= 32 else (1 if t_buf.shape[1] >= 32 else 0)
        n_blk = int(t_buf.shape[blk_axis])
        if n_blk <= 0:
            return
        pat_n = min(32, n_blk)
        pat_shape = list(t_buf.shape)
        pat_shape[blk_axis] = pat_n
        g_cpu = torch.Generator(device="cpu").manual_seed(seed_val)
        pat = (torch.randn(pat_shape, dtype=torch.float32, generator=g_cpu) * 0.05).to(
            device=t_buf.device, dtype=t_buf.dtype
        )
        for start in range(0, n_blk, pat_n):
            end = min(n_blk, start + pat_n)
            cnt = end - start
            if blk_axis == 0:
                t_buf[start:end].copy_(pat[:cnt])
            else:
                t_buf[:, start:end].copy_(pat[:, :cnt])

    for lid, tag_map in enumerate(kv_cache.layer_caches):
        for tag, lc in tag_map.items():
            if tag == FULL_ATTN_KV_TAG:
                _fill_kv_periodic(lc.kv_cache_base, 1337 + lid * 17)
                if is_fp8_kv and fp8_dt is not None and lc.kv_cache_base.dtype != fp8_dt:
                    lc.kv_cache_base = (lc.kv_cache_base.float() / 0.5).clamp(-448.0, 448.0).to(fp8_dt)
                    if lc.kv_cache_base.ndim == 5:
                        # 4D HND contiguous scale matching sparse_paged_fmha.py:107 [N, 2, H_kv, B_tok]
                        nb, _, b_tok, n_kvh, _ = lc.kv_cache_base.shape
                        lc.kv_scale_base = torch.full(
                            (nb, 2, n_kvh, b_tok), 0.5, dtype=torch.float32, device=device
                        )
                    else:
                        lc.kv_scale_base = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
            elif tag == INDEXER_KV_TAG:
                _fill_kv_periodic(lc.kv_cache_base.view(torch.bfloat16), 1937 + lid * 17)
    return kv_cache


def build_20k_dflash_kv_cache(
    spec: Any,
    *,
    max_batch: int,
    avg_ctx: int,
    verify_width: int,
    tokens_per_block: int,
    dflash_kv_dtype: str = "bf16",
    device: torch.device,
    max_total_blocks: Optional[int] = None,
) -> Tuple[Any, List[torch.Tensor]]:
    from rtp_llm.models_py.model_desc.qwen4_5_runtime import LocalLayerKVCache

    blocks_per_req = (avg_ctx + verify_width + 4 + tokens_per_block - 1) // tokens_per_block + 1
    total_blocks = (
        int(max_total_blocks)
        if max_total_blocks is not None
        else (max_batch * blocks_per_req + 64)
    )
    is_fp8 = str(dflash_kv_dtype).lower().startswith("fp8")
    fp8_dt = getattr(torch, "float8_e4m3fn", None) if is_fp8 else None
    kv_dt = fp8_dt if fp8_dt is not None else torch.bfloat16
    kv_scale = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device) if fp8_dt is not None else None

    kv_bufs = [
        torch.empty(
            (total_blocks, 2, int(spec.num_key_value_heads), tokens_per_block, int(spec.head_dim)),
            dtype=kv_dt,
            device=device,
        )
        for _ in range(int(spec.num_hidden_layers))
    ]
    pat_n = min(32, total_blocks)
    for lid, buf in enumerate(kv_bufs):
        g_cpu = torch.Generator(device="cpu").manual_seed(2448 + lid * 17)
        pat_bf16 = (
            torch.randn(
                (pat_n, 2, int(spec.num_key_value_heads), tokens_per_block, int(spec.head_dim)),
                dtype=torch.float32,
                generator=g_cpu,
            )
            * 0.05
        ).to(device=device, dtype=torch.bfloat16)
        pat = (
            (pat_bf16.float() / 0.5).clamp(-448.0, 448.0).to(kv_dt)
            if fp8_dt is not None
            else pat_bf16
        )
        for start in range(0, total_blocks, pat_n):
            end = min(total_blocks, start + pat_n)
            buf[start:end].copy_(pat[: end - start])

    class _SimpleDFlashKVCache:
        def get_kernel_seq_size_per_block(self, tag: str = "full") -> int:
            return tokens_per_block

        def get_layer_cache(self, layer_idx: int) -> Any:
            return LocalLayerKVCache(
                kv_cache_base=kv_bufs[layer_idx],
                seq_size_per_block=tokens_per_block,
                layer_id=layer_idx,
                tag="full",
                kv_scale_base=kv_scale,
            )

        def get_layer_cache_groups(self, layer_idx: int) -> List[Any]:
            return [self.get_layer_cache(layer_idx)]

    return _SimpleDFlashKVCache(), kv_bufs


def _build_verify_and_draft_inputs(
    *,
    target_kv: Any,
    dflash_kv_bufs: List[torch.Tensor],
    spec: Any,
    batch_size: int,
    avg_ctx: int,
    verify_width: int,
    tokens_per_block: int,
    device: torch.device,
    seed: int = 42,
    dp_rank: int = 0,
    dp_size: int = 1,
    local_ctxs: Optional[Sequence[int]] = None,
) -> Tuple[Any, Any, torch.Tensor, torch.Tensor]:
    from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import FULL_ATTN_KV_TAG, INDEXER_KV_TAG
    from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
        LocalPyModelInputs,
        make_tagged_attention_inputs,
    )
    from rtp_llm.ops import PyAttentionInputs, PyModelInputs

    ctxs = [int(c) for c in local_ctxs] if local_ctxs is not None else [int(avg_ctx)] * batch_size
    if len(ctxs) != batch_size:
        raise ValueError(f"local_ctxs len={len(ctxs)} != batch_size={batch_size}")

    # Assign contiguous non-overlapping physical block IDs per request across [0..batch_size-1]
    req_block_ids: List[List[int]] = []
    cur_blk = 1
    for c in ctxs:
        req_blks = (c + verify_width + 4 + tokens_per_block - 1) // tokens_per_block + 1
        req_block_ids.append(list(range(cur_blk, cur_blk + req_blks)))
        cur_blk += req_blks
    max_req_blk = max((len(r) for r in req_block_ids), default=1)

    token_tags = {FULL_ATTN_KV_TAG, INDEXER_KV_TAG}
    tables_by_tag: Dict[str, List[List[int]]] = {}
    for tag in target_kv.group_tags():
        if tag in token_tags:
            tables_by_tag[tag] = [list(r) for r in req_block_ids]
        else:
            tables_by_tag[tag] = [[1 + b] for b in range(batch_size)]

    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    global_bs = batch_size * max(1, dp_size)
    v_toks_all = torch.randint(10, 10000, (global_bs, verify_width), dtype=torch.int32, generator=g_cpu, device="cpu")
    v_toks = v_toks_all[dp_rank * batch_size : (dp_rank + 1) * batch_size].to(device)
    verify_attn = make_tagged_attention_inputs(
        tables_by_tag,
        is_prefill=True,
        input_lengths=[verify_width] * batch_size,
        sequence_lengths=[],
        prefix_lengths=ctxs,
        device=device,
    )
    for grp in verify_attn.values():
        grp.is_target_verify = True
        grp.prefix_lengths = grp.prefix_lengths.to(device=device, dtype=torch.int32)
    verify_in = LocalPyModelInputs(input_ids=v_toks.reshape(-1), attention_inputs=verify_attn)

    # DFlash2 propose inputs (query_width = verify_width = 8)
    d_bt = torch.zeros((batch_size, max_req_blk), dtype=torch.int32, device=device)
    for b_idx, r_ids in enumerate(req_block_ids):
        d_bt[b_idx, : len(r_ids)] = torch.tensor(r_ids, dtype=torch.int32, device=device)
    qw = int(spec.query_width)
    pref_t = torch.tensor(ctxs, dtype=torch.int32, device=device)
    inp_t = torch.full((batch_size,), qw, dtype=torch.int32, device=device)
    cu_t = torch.arange(0, (batch_size + 1) * qw, qw, dtype=torch.int32, device=device)

    d_attn = PyAttentionInputs()
    d_attn.is_prefill = True
    d_attn.is_target_verify = False
    d_attn.total_tokens = int(batch_size * qw)
    d_attn.prefix_lengths = pref_t
    d_attn.input_lengths = inp_t
    d_attn.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cpu")
    d_attn.cu_seqlens = cu_t
    d_attn.cu_seqlens_device = cu_t
    d_attn.kv_cache_block_id = d_bt.cpu()
    d_attn.kv_cache_kernel_block_id = d_attn.kv_cache_block_id
    d_attn.kv_cache_block_id_device = d_bt
    d_attn.kv_cache_kernel_block_id_device = d_bt

    anchor_all = torch.randint(10, 10000, (global_bs,), dtype=torch.int32, generator=g_cpu, device="cpu")
    anchor_ids = anchor_all[dp_rank * batch_size : (dp_rank + 1) * batch_size].to(device)
    d_toks = torch.full((batch_size, qw), int(spec.mask_token_id), dtype=torch.int32, device=device)
    d_toks[:, 0] = anchor_ids
    d_in = types.SimpleNamespace(
        input_ids=d_toks.reshape(-1),
        input_hiddens=torch.empty(0, dtype=torch.bfloat16, device=device),
        attention_inputs=d_attn,
        attention_inputs_by_tag={"full": d_attn},
    )

    # DFlash2 commit inputs (1 accepted anchor token per request -> scatters feature KV at pos=avg_ctx-1)
    c_attn = PyAttentionInputs()
    c_attn.is_prefill = True
    c_attn.is_target_verify = False
    c_attn.total_tokens = int(batch_size)
    c_attn.prefix_lengths = torch.tensor([max(0, c - 1) for c in ctxs], dtype=torch.int32, device=device)
    c_attn.input_lengths = torch.ones((batch_size,), dtype=torch.int32, device=device)
    c_attn.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cpu")
    c_cu = torch.arange(0, batch_size + 1, 1, dtype=torch.int32, device=device)
    c_attn.cu_seqlens = c_cu
    c_attn.cu_seqlens_device = c_cu
    c_attn.kv_cache_block_id = d_attn.kv_cache_block_id
    c_attn.kv_cache_kernel_block_id = d_attn.kv_cache_kernel_block_id
    c_attn.kv_cache_block_id_device = d_bt
    c_attn.kv_cache_kernel_block_id_device = d_bt
    g_commit = torch.Generator(device="cpu").manual_seed(seed + 100 + global_bs)
    d_commit_in = types.SimpleNamespace(
        input_ids=anchor_ids.clone(),
        input_hiddens=torch.randn(
            (batch_size, int(spec.fc_in_features)),
            dtype=torch.bfloat16,
            generator=g_commit,
            device="cpu",
        ).to(device) * 0.05,
        attention_inputs=c_attn,
        attention_inputs_by_tag={"full": c_attn},
    )
    return verify_in, d_in, d_commit_in, anchor_ids


# ---------------------------------------------------------------------------
# Kineto Parsing
# ---------------------------------------------------------------------------


def _kernel_bucket(name: str) -> str:
    nl = name.lower()
    if any(k in nl for k in ("mxfp4", "deep_gemm", "nvjet", "cutlass", "cublas", "sgemm", "gemm")):
        return "gemm_and_moe"
    if any(k in nl for k in ("_gdn_", "dwconv1d", "grouped_conv", "causal_conv")):
        return "gdn_sconv"
    if any(k in nl for k in ("_qsa_", "_sparse_", "_dflash2_", "flashinfer", "rope")) and "sampling" not in nl and "allreduce" not in nl:
        return "attn_qsa_dflash2_rope"
    if any(k in nl for k in ("nccl", "allreduce", "oneshot", "twoshot", "lamport", "reduce_scatter", "allgather", "alltoall")):
        return "tp_ep_comm"
    if "softmax" in nl:
        return "softmax"
    if any(k in nl for k in ("sampling", "topk", "argmax", "radix")):
        return "sampling_topk"
    if any(k in nl for k in ("_hc_", "_ple_", "_situ", "rmsnorm", "rms_norm", "quant", "silu", "gelu")):
        return "norm_act_quant_hc"
    if any(k in nl for k in ("copy", "catarray", "fillfunctor", "binaryfunctor", "unaryfunctor", "indexselect", "repeat", "elementwise")):
        return "elementwise_copy_idx"
    return "other_triton_or_misc"


@contextlib.contextmanager
def _kineto_decode_forward_scopes(target_rtp: Any, dflash_model: Any):
    """Attach temporary non-invasive ``rf:<bucket>`` record_function scopes for eager Kineto capture."""
    from torch.autograd.profiler import record_function

    handles: List[Any] = []
    patches: List[Tuple[Any, str, Any, bool]] = []
    rf_stacks: Dict[int, List[Any]] = defaultdict(list)

    def _push_rf(mod: Any, _inputs: Any, bucket: str) -> None:
        ctx = record_function(f"rf:{bucket}")
        ctx.__enter__()
        rf_stacks[id(mod)].append(ctx)

    def _pop_rf(mod: Any, _inputs: Any, _output: Any) -> None:
        st = rf_stacks.get(id(mod))
        if st:
            st.pop().__exit__(None, None, None)

    def _hook_mod(mod: Any, bucket: str) -> None:
        if mod is None:
            return
        handles.append(mod.register_forward_pre_hook(lambda m, inp, b=bucket: _push_rf(m, inp, b)))
        handles.append(mod.register_forward_hook(_pop_rf))

    def _wrap_method(obj: Any, attr: str, bucket: str) -> None:
        if obj is None or not hasattr(obj, attr):
            return
        had_own = attr in getattr(obj, "__dict__", {})
        orig = getattr(obj, attr)
        patches.append((obj, attr, orig, had_own))

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            with record_function(f"rf:{bucket}"):
                return orig(*args, **kwargs)

        setattr(obj, attr, _wrapped)

    t_model = getattr(target_rtp, "model", None)
    if t_model is not None:
        _hook_mod(getattr(t_model, "embed_tokens", None), "embed_ple")
        hcm = getattr(t_model, "hyper_connection_mixer", None)
        for meth in ("mix_with_latent", "combine_and_mix_with_latent", "mix_from_normed", "combine"):
            _wrap_method(hcm, meth, "hc_and_output_sconv")
        for lyr in getattr(t_model, "layers", ()):
            _hook_mod(getattr(lyr, "ple", None), "embed_ple")
            for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc_mod = getattr(lyr, hc_name, None)
                for meth in ("mix_with_latent", "combine_and_mix_with_latent", "mix_from_normed", "combine"):
                    _wrap_method(hc_mod, meth, "hc_and_output_sconv")
            _hook_mod(getattr(lyr, "attn_output_sconv", None), "hc_and_output_sconv")
            _hook_mod(getattr(lyr, "ffn_output_sconv", None), "hc_and_output_sconv")

            lin = getattr(lyr, "linear_attn", None)
            if lin is not None:
                _wrap_method(lin, "forward", "gdn_core")
                for sub_attr in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                    _hook_mod(getattr(lin, sub_attr, None), "gdn_proj")

            sa = getattr(lyr, "self_attn", None)
            if sa is not None:
                _wrap_method(sa, "forward", "full_attn_qsa_core")
                for sub_attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _hook_mod(getattr(sa, sub_attr, None), "full_attn_proj")
                _wrap_method(getattr(sa, "indexer", None), "project", "full_attn_proj")

            mlp = getattr(lyr, "mlp", None)
            if mlp is not None:
                is_moe = hasattr(mlp, "latent_down_proj")
                _hook_mod(mlp, "moe_routed" if is_moe else "dense_mlp")
                if is_moe:
                    for sub_attr in (
                        "latent_down_proj",
                        "latent_gate_proj",
                        "latent_up_proj",
                        "latent_scale_gate",
                        "gate",
                        "shared_expert",
                        "shared_expert_gate",
                    ):
                        _hook_mod(getattr(mlp, sub_attr, None), "moe_latent_shared")

    _wrap_method(target_rtp, "compute_logits", "lm_head")

    if dflash_model is not None:
        _wrap_method(dflash_model, "forward_commit", "dflash2_attn_dense")
        _wrap_method(dflash_model, "forward_propose", "dflash2_attn_dense")
        _wrap_method(dflash_model, "propose_with_candidate_selector", "dflash2_attn_dense")
        d_core = getattr(dflash_model, "model", dflash_model)
        for lyr in getattr(d_core, "layers", ()):
            mlp = getattr(lyr, "mlp", None)
            if mlp is not None and (hasattr(mlp, "experts") or hasattr(mlp, "gate_up_weight")):
                _hook_mod(mlp, "dflash2_moe_routed")
                for sub_attr in (
                    "latent_down_proj",
                    "latent_gate_proj",
                    "latent_up_proj",
                    "latent_scale_gate",
                    "gate",
                    "shared_expert",
                    "shared_expert_gate",
                ):
                    _hook_mod(getattr(mlp, sub_attr, None), "dflash2_attn_dense")

    try:
        yield
    finally:
        for obj, attr, orig, had_own in reversed(patches):
            if had_own:
                setattr(obj, attr, orig)
            elif attr in getattr(obj, "__dict__", {}):
                delattr(obj, attr)
        for h in handles:
            h.remove()
        for st in rf_stacks.values():
            while st:
                st.pop().__exit__(None, None, None)


def _classify_decode_kernel(name: str, rf_bucket: str = "", phase_name: str = "") -> str:
    nl = name.lower()
    if "alltoall" in nl:
        return "comm_alltoall"
    if any(k in nl for k in ("nccl", "allreduce", "allgather", "oneshot", "twoshot", "lamport")):
        return "comm_allreduce"
    if "_dflash2_" in nl or "dflash" in nl:
        return "dflash2_attn_dense"
    if any(k in nl for k in ("_qsa_", "_sparse_", "mbtopk", "digitcounts")):
        return "full_attn_qsa_core"
    if any(k in nl for k in ("_gdn_", "_yoco_gdn_", "dwconv1d", "causal_dwconv1d")):
        return "gdn_core"
    if any(k in nl for k in ("_hc_", "hyper_connection", "sconv")):
        return "hc_and_output_sconv"
    if "_ple_" in nl:
        return "embed_ple"
    is_moe_kern = any(k in nl for k in ("mxfp4", "fp8_fp4", "deep_gemm", "m_grouped", "group_gemm", "_situ_glu", "sigmoid_topk", "sbtopk"))
    if phase_name == "dflash_propose":
        if rf_bucket == "dflash2_moe_routed" or is_moe_kern:
            return "dflash2_moe_routed"
        return "dflash2_attn_dense"
    if phase_name == "dflash_selector":
        return "dflash2_attn_dense"
    if is_moe_kern:
        return "moe_routed"
    if rf_bucket:
        return rf_bucket
    return _kernel_bucket(name)


def parse_torch_profiler_kineto(trace_path: Path, top_k: int = 15) -> Dict[str, Any]:
    raw = json.loads(trace_path.read_text(encoding="utf-8"))
    events = raw.get("traceEvents", [])
    ker: Dict[str, List[Any]] = defaultdict(lambda: [0.0, 0, None, None])
    buckets: Dict[str, List[Any]] = defaultdict(lambda: [0.0, 0])
    # Map CPU user_annotation spans (smallest enclosing span) -> cuda_runtime correlation -> CUDA kernels
    phase_names = ("dflash_propose", "dflash_selector", "target_verify", "target_logits_commit")
    ann_spans: List[Tuple[float, float, float, str]] = []
    rf_spans: List[Tuple[float, float, float, str]] = []
    runtime_calls: List[Tuple[float, int]] = []
    for ev in events:
        cat = ev.get("cat")
        if cat == "user_annotation":
            ename = str(ev.get("name", ""))
            ts = float(ev.get("ts", 0.0))
            dur = float(ev.get("dur", 0.0))
            if ename in phase_names:
                ann_spans.append((ts, ts + dur, dur, ename))
            elif ename.startswith("rf:"):
                rf_spans.append((ts, ts + dur, dur, ename[3:]))
        elif cat == "cuda_runtime":
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                runtime_calls.append((float(ev.get("ts", 0.0)), int(corr)))

    def _match_smallest(spans: List[Tuple[float, float, float, str]]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if not spans or not runtime_calls:
            return out
        for rts, corr in runtime_calls:
            best_dur = float("inf")
            best_name = ""
            for s0, s1, dur, sname in spans:
                if s0 <= rts <= s1 and dur < best_dur:
                    best_dur = dur
                    best_name = sname
            if best_name:
                out[corr] = best_name
        return out

    corr_to_phase = _match_smallest(ann_spans)
    corr_to_rf = _match_smallest(rf_spans)

    scopes: Dict[str, List[Any]] = defaultdict(lambda: [0.0, 0])
    rf_ops: Dict[str, List[Any]] = defaultdict(lambda: [0.0, 0])
    for ev in events:
        if ev.get("cat") == "kernel":
            n = str(ev.get("name", ""))
            dur_ms = float(ev.get("dur", 0.0)) / 1000.0
            args = ev.get("args") or {}
            st = ker[n]
            st[0] += dur_ms
            st[1] += 1
            if st[2] is None:
                st[2] = args.get("grid")
            if st[3] is None:
                st[3] = args.get("block")
            b = _kernel_bucket(n)
            buckets[b][0] += dur_ms
            buckets[b][1] += 1
            corr = args.get("correlation")
            if corr is not None:
                c_int = int(corr)
                sname = corr_to_phase.get(c_int, "unscoped")
                rname = _classify_decode_kernel(n, corr_to_rf.get(c_int, ""), sname)
            else:
                sname = "unscoped"
                rname = _classify_decode_kernel(n, "", "unscoped")
            scopes[sname][0] += dur_ms
            scopes[sname][1] += 1
            rf_ops[rname][0] += dur_ms
            rf_ops[rname][1] += 1
    tot = max(1e-9, sum(v[0] for v in ker.values()))
    top_rows = [
        {
            "rank": idx,
            "name": n[:120],
            "total_ms": round(v[0], 3),
            "pct": round(v[0] * 100.0 / tot, 2),
            "calls": v[1],
            "mean_us": round(v[0] * 1000.0 / max(1, v[1]), 2),
            "grid": v[2],
            "block": v[3],
        }
        for idx, (n, v) in enumerate(sorted(ker.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k], 1)
    ]
    bucket_rows = {
        b: {"total_ms": round(v[0], 3), "pct": round(v[0] * 100.0 / tot, 2), "calls": v[1]}
        for b, v in sorted(buckets.items(), key=lambda kv: kv[1][0], reverse=True)
    }
    scope_rows = {
        s: {"total_ms": round(v[0], 3), "pct": round(v[0] * 100.0 / tot, 2), "calls": v[1]}
        for s, v in sorted(scopes.items(), key=lambda kv: kv[1][0], reverse=True)
    }
    rf_op_rows = {
        r: {"total_ms": round(v[0], 3), "pct": round(v[0] * 100.0 / tot, 2), "calls": v[1]}
        for r, v in sorted(rf_ops.items(), key=lambda kv: kv[1][0], reverse=True)
    }
    return {
        "trace_file": str(trace_path),
        "total_cuda_kernel_ms": round(tot, 3),
        "launches": sum(v[1] for v in ker.values()),
        "unique_kernels": len(ker),
        "top_kernels": top_rows,
        "buckets": bucket_rows,
        "scopes": scope_rows,
        "rf_ops": rf_op_rows,
    }


# ---------------------------------------------------------------------------
# Benchmark Driver
# ---------------------------------------------------------------------------


def _apply_dtype_mode_env(dtype_mode: str) -> None:
    if dtype_mode == "w4a8":
        os.environ["QWEN4_5_MXFP4_DECODE_BACKEND"] = "sm100_winner"
    elif dtype_mode == "triton":
        os.environ["QWEN4_5_MXFP4_BACKEND"] = "triton"
        os.environ.pop("QWEN4_5_MXFP4_DECODE_BACKEND", None)
    elif dtype_mode == "auto":
        os.environ.pop("QWEN4_5_MXFP4_DECODE_BACKEND", None)
        os.environ.pop("QWEN4_5_MXFP4_BACKEND", None)


def _compute_parity(test_t: torch.Tensor, ref_t: torch.Tensor) -> Dict[str, Any]:
    tf = test_t.float().reshape(-1)
    rf = ref_t.float().reshape(-1)
    cos = float(F.cosine_similarity(tf.unsqueeze(0), rf.unsqueeze(0)).item())
    max_abs = float((tf - rf).abs().max().item())
    top1 = float((test_t.float().argmax(dim=-1) == ref_t.float().argmax(dim=-1)).float().mean().item())
    res: Dict[str, Any] = {"cosine": round(cos, 7), "max_abs": round(max_abs, 6), "top1_agreement": round(top1, 6)}
    if test_t.ndim >= 2 and test_t.shape[0] > 0:
        t2 = test_t.float().reshape(test_t.shape[0], -1)
        r2 = ref_t.float().reshape(ref_t.shape[0], -1)
        row_cos = F.cosine_similarity(t2, r2, dim=-1)
        res["per_tok_min"] = round(float(row_cos.min().item()), 7)
        res["per_tok_p10"] = round(float(torch.quantile(row_cos, 0.10).item()), 7)
        res["frac_ge_0999"] = round(float((row_cos >= 0.999).float().mean().item()), 6)
    return res


def _stabilize_moe_routers(
    model: Any, step: float, seed: int = 2026
) -> List[Tuple[Any, bool, Optional[torch.Tensor]]]:
    """Apply deterministic per-expert router bias step (gap >= step) and return restore handles."""
    saved: List[Tuple[Any, bool, Optional[torch.Tensor]]] = []
    if step <= 0.0:
        return saved
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", ())
    for lid, lyr in enumerate(layers):
        mlp = getattr(lyr, "mlp", None)
        if mlp is None or not hasattr(mlp, "num_experts"):
            continue
        num_exp = int(getattr(mlp, "num_experts", 0))
        if num_exp <= 0:
            continue
        dev = mlp.gate.weight.device if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight") else torch.device("cpu")
        g_cpu = torch.Generator(device="cpu").manual_seed(seed + lid * 131)
        perm = torch.randperm(num_exp, generator=g_cpu, dtype=torch.int64)
        bias_vals = (perm.to(torch.float32) * float(step)).to(dev)
        cur_bias = getattr(mlp, "expert_bias", None)
        if cur_bias is not None and isinstance(cur_bias, torch.Tensor) and tuple(cur_bias.shape) == (num_exp,):
            saved.append((mlp, True, cur_bias.data.clone()))
            cur_bias.data.copy_(bias_vals.to(dtype=cur_bias.dtype))
        else:
            saved.append((mlp, False, None))
            mlp.register_parameter("expert_bias", torch.nn.Parameter(bias_vals, requires_grad=False))
    return saved


def _restore_moe_routers(saved: List[Tuple[Any, bool, Optional[torch.Tensor]]]) -> None:
    for mlp, had_param, prev_data in saved:
        if had_param and prev_data is not None:
            mlp.expert_bias.data.copy_(prev_data)
        else:
            mlp.register_parameter("expert_bias", None)


def _extract_tensor(out: Any) -> Optional[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _run_decode_parity_probe(
    *,
    target_rtp: Any,
    dflash_model: Any,
    target_kv: Any,
    dflash_kv_bufs: List[torch.Tensor],
    d_spec: Any,
    p_bs: int,
    dp_size: int,
    dp_rank: int,
    avg_ctx: int,
    verify_width: int,
    tokens_per_block: int,
    device: torch.device,
    world_rank: int,
    save_ref: bool,
    ref_entry: Optional[Dict[str, Any]],
    parity_min_cos: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run deterministic 1-step parity probe capturing Target + DFlash2 layer/attn snapshots & E2E outputs."""
    from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import FULL_ATTN_KV_TAG, INDEXER_KV_TAG

    p_bs_local = p_bs // dp_size
    for tag_map in target_kv.layer_caches:
        for tag, lc in tag_map.items():
            if tag not in (FULL_ATTN_KV_TAG, INDEXER_KV_TAG) and hasattr(lc, "kv_cache_base"):
                lc.kv_cache_base.zero_()

    p_v_in, p_d_in, _p_dc_in, _p_anch = _build_verify_and_draft_inputs(
        target_kv=target_kv,
        dflash_kv_bufs=dflash_kv_bufs,
        spec=d_spec,
        batch_size=p_bs_local,
        avg_ctx=avg_ctx,
        verify_width=verify_width,
        tokens_per_block=tokens_per_block,
        device=device,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    snaps_raw: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []

    def _rec_hook(name: str, cap_dim: int = 1024):
        def _fn(_mod: Any, _inp: Any, out: Any) -> None:
            t = _extract_tensor(out)
            if t is not None and t.ndim >= 2 and name not in snaps_raw:
                t2 = t.reshape(-1, t.shape[-1])
                snaps_raw[name] = t2[: min(8, t2.shape[0]), : min(cap_dim, t2.shape[1])].detach().cpu().float()
        return _fn

    if world_rank == 0:
        t_core = getattr(target_rtp, "model", target_rtp)
        for lid, lyr in enumerate(getattr(t_core, "layers", ())):
            hooks.append(lyr.register_forward_hook(_rec_hook(f"target_l{lid}_mod")))
            attn_sub = getattr(lyr, "self_attn", None) or getattr(lyr, "linear_attn", None)
            if attn_sub is not None:
                hooks.append(attn_sub.register_forward_hook(_rec_hook(f"target_l{lid}_attn")))
            mlp_sub = getattr(lyr, "mlp", None)
            if mlp_sub is not None:
                hooks.append(mlp_sub.register_forward_hook(_rec_hook(f"target_l{lid}_mlp")))
        d_core = getattr(dflash_model, "model", dflash_model)
        for lid, lyr in enumerate(getattr(d_core, "layers", ())):
            hooks.append(lyr.register_forward_hook(_rec_hook(f"dflash_l{lid}")))
            attn_sub = getattr(lyr, "self_attn", None)
            if attn_sub is not None:
                hooks.append(attn_sub.register_forward_hook(_rec_hook(f"dflash_l{lid}_attn")))
            mlp_sub = getattr(lyr, "mlp", None)
            if mlp_sub is not None:
                hooks.append(mlp_sub.register_forward_hook(_rec_hook(f"dflash_l{lid}_mlp")))

    try:
        p_fmha_t = target_rtp.prepare_fmha_impl(p_v_in, is_cuda_graph=False)
        p_fmha_d = dflash_model.prepare_fmha_impl(p_d_in, is_cuda_graph=False)
        p_zero_acc = torch.zeros((p_bs_local,), dtype=torch.int32, device=device)
        p_d_out = dflash_model.forward_propose(p_d_in, fmha_impl=p_fmha_d)
        p_t_out = target_rtp.forward(p_v_in, fmha_impl=p_fmha_t)
        p_logits = target_rtp.compute_logits(p_t_out.hidden_states)
        target_rtp.prepare_speculative_target_commit(p_zero_acc)
        target_rtp.finish_speculative_target_commit(False)
        target_rtp.finalize_speculative_target_commit()
        torch.cuda.synchronize(device)
    finally:
        for h in hooks:
            h.remove()

    saved_rec: Optional[Dict[str, Any]] = None
    par_res: Optional[Dict[str, Any]] = None
    if world_rank == 0:
        aux_h = getattr(p_t_out, "mtp_target_hidden_states", None)
        t_layers = getattr(getattr(target_rtp, "model", target_rtp), "layers", ())
        n_t_layers = len(t_layers)
        if aux_h is not None and n_t_layers > 0 and aux_h.shape[-1] % n_t_layers == 0:
            aux_3d = aux_h.view(aux_h.shape[0], n_t_layers, -1)
            for lid in range(n_t_layers):
                t_slice = aux_3d[: min(8, aux_3d.shape[0]), lid, :1024].detach().cpu().float()
                snaps_raw[f"target_l{lid}"] = t_slice
        tl_sample = p_logits[: min(8, p_logits.shape[0]), :4096].cpu().float()
        dh_sample = p_d_out.hidden_states[: min(8, p_d_out.hidden_states.shape[0]), :1024].cpu().float()
        if save_ref:
            saved_rec = {
                "target_logits_sample": tl_sample.tolist(),
                "dflash_hidden_sample": dh_sample.tolist(),
                "snapshots": {k: v.tolist() for k, v in snaps_raw.items()},
            }
        if ref_entry is not None:
            r_tl = torch.tensor(ref_entry["target_logits_sample"], dtype=torch.float32)
            r_dh = torch.tensor(ref_entry["dflash_hidden_sample"], dtype=torch.float32)
            tl_par = _compute_parity(tl_sample[: r_tl.shape[0], : r_tl.shape[1]], r_tl)
            dh_par = _compute_parity(dh_sample[: r_dh.shape[0], : r_dh.shape[1]], r_dh)
            snap_pars: Dict[str, Any] = {}
            ref_snaps = ref_entry.get("snapshots", {})
            for sk, sval in snaps_raw.items():
                if sk in ref_snaps:
                    rt = torch.tensor(ref_snaps[sk], dtype=torch.float32)
                    n_r = min(int(sval.shape[0]), int(rt.shape[0]))
                    n_c = min(int(sval.shape[1]), int(rt.shape[1]))
                    snap_pars[sk] = _compute_parity(sval[:n_r, :n_c], rt[:n_r, :n_c])
            min_e2e = min(tl_par["cosine"], dh_par["cosine"])
            min_snap = min((v["cosine"] for v in snap_pars.values()), default=None)
            passed = bool(min_e2e >= parity_min_cos and (min_snap is None or min_snap >= parity_min_cos))
            par_res = {
                "target_logits": tl_par,
                "dflash_hidden": dh_par,
                "dflash_l0_attn_cos": snap_pars.get("dflash_l0_attn", {}).get("cosine"),
                "target_l0_attn_cos": snap_pars.get("target_l0_attn", {}).get("cosine"),
                "snap_count": len(snap_pars),
                "min_snap_cos": min_snap,
                "min_e2e_cos": round(min_e2e, 7),
                "passed": passed,
                "snapshots": snap_pars,
            }
    del p_v_in, p_d_in, _p_dc_in, _p_anch, p_fmha_t, p_fmha_d, p_d_out, p_t_out, p_logits
    torch.cuda.empty_cache()
    return saved_rec, par_res


def _update_leaderboard(
    path: Path,
    *,
    round_tag: str,
    parallel_mode: str,
    dtype_mode: str,
    avg_ctx: int,
    hw: HardwareRoofline,
    rows: List[Dict[str, Any]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    existing: Dict[str, Any] = {"entries": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if "entries" not in existing or not isinstance(existing["entries"], list):
                existing = {"entries": []}
        except Exception:
            existing = {"entries": []}

    meta = dict(extra_meta or {})
    entry = {
        "round": round_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "parallel_mode": parallel_mode,
        "dtype_mode": dtype_mode,
        "avg_ctx": avg_ctx,
        **meta,
        "hw_roofline": hw.to_dict(),
        "b1": next((r for r in rows if r["batch_size"] == 1), None),
        "b8": next((r for r in rows if r["batch_size"] == 8), None),
        "b64": next((r for r in rows if r["batch_size"] == 64), None),
        "peak_tok_s": max((r["raw_cycle_tok_s"] for r in rows), default=0.0),
        "rows": rows,
    }
    def _same_cfg(e: Dict[str, Any]) -> bool:
        if not (
            e.get("round") == round_tag
            and e.get("parallel_mode") == parallel_mode
            and e.get("dtype_mode") == dtype_mode
            and e.get("avg_ctx") == avg_ctx
        ):
            return False
        for k in (
            "kv_dtype",
            "dflash_kv_dtype",
            "dflash_attn_backend",
            "dense_linear_mode",
            "ple_mode",
            "router_margin",
            "skew_preset",
            "skew_layout",
        ):
            if e.get(k) != meta.get(k):
                return False
        if bool(e.get("enable_tbo", False)) != bool(meta.get("enable_tbo", False)):
            return False
        return True

    existing["entries"] = [e for e in existing["entries"] if not _same_cfg(e)]
    existing["entries"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone 4-GPU Decode Roofline Benchmark (Qwen 4.5-Flash)")
    parser.add_argument("--target-ckpt", default="/home/admin/lhc_qwen45_opt/models/Qwen4.5-Flash-5L-fullwidth-qsa-mock")
    parser.add_argument("--dflash-ckpt", default="/home/admin/lhc_qwen45_opt/models/Qwen4.5-Flash-5L-fullwidth-dflash-mock")
    parser.add_argument(
        "--parallel-mode",
        choices=("tp4_ep4", "dp4_ep4", "dp2_tp2_ep4", "sp4_tp4_ep4", "dp2_tpsp2_ep4"),
        default="tp4_ep4",
    )
    parser.add_argument("--dtype-mode", choices=("w4a8", "auto", "triton", "bf16", "fp8", "nvfp4"), default="w4a8")
    parser.add_argument("--avg-ctx", type=int, default=20480)
    parser.add_argument(
        "--ctx-list",
        default="",
        help="Optional comma-separated per-request prefix lengths (e.g. 40960,2048,2048,2048)",
    )
    parser.add_argument(
        "--skew-preset",
        choices=("", "40k_2k", "avg20k_skew", "straggler_40k_2k"),
        default="",
        help="Optional skewed-context Decode preset (1x40K+3x2K, avg20K extreme skew, or single 40K straggler)",
    )
    parser.add_argument(
        "--skew-layout",
        choices=("rank_skewed", "interleaved"),
        default="rank_skewed",
        help="Request ordering across DP ranks when --skew-preset or --ctx-list is active",
    )
    parser.add_argument("--verify-width", type=int, default=8)
    parser.add_argument("--batches", default="1,4,8,16,32,64,96,128")
    parser.add_argument("--tokens-per-block", type=int, default=64)
    parser.add_argument("--ple-mode", choices=("auto", "gpu_sharded", "cpu_uva", "shm", "mmap"), default="gpu_sharded")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--cuda-graph", dest="cuda_graph", action="store_true", default=True)
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    parser.add_argument("--include-selector", dest="include_selector", action="store_true", default=True)
    parser.add_argument("--no-selector", dest="include_selector", action="store_false")
    parser.add_argument("--kineto-batches", default="1,8")
    parser.add_argument("--traces-dir", default="/home/admin/lhc_qwen45_opt/traces")
    parser.add_argument("--round-tag", default="R0")
    parser.add_argument("--out-json", default="/home/admin/lhc_qwen45_opt/logs/decode_r0_roofline.json")
    parser.add_argument("--leaderboard-json", default="/home/admin/lhc_qwen45_opt/logs/leaderboard_decode.json")
    parser.add_argument("--roofline-cache", default="/home/admin/lhc_qwen45_opt/logs/l20a_roofline.json")
    parser.add_argument("--dense-linear-mode", default="", help="Optional QWEN4_5_DENSE_LINEAR_MODE override (e.g. bf16, auto, fp8_mxfp4)")
    parser.add_argument("--kv-dtype", choices=("bf16", "fp8_e4m3"), default="bf16")
    parser.add_argument("--dflash-kv-dtype", default="", help="Optional DFlash2 KV cache dtype (defaults to --kv-dtype)")
    parser.add_argument("--dflash-attn-backend", choices=("auto", "flashinfer", "triton"), default="auto")
    parser.add_argument("--indexer-kv-dtype", choices=("bf16", "fp8_e4m3"), default="bf16")
    parser.add_argument("--op-kineto", dest="op_kineto", action="store_true", default=True, help="Capture 1 eager op-scoped Kineto trace per kineto batch")
    parser.add_argument("--no-op-kineto", dest="op_kineto", action="store_false")
    parser.add_argument("--ref-out", default="")
    parser.add_argument("--ref-in", default="")
    parser.add_argument("--verify-parity", dest="verify_parity", default="", help="Alias for --ref-in")
    parser.add_argument("--router-margin", type=float, default=0.0, help="Persistent per-expert router bias step during timing & parity (0.0 disables)")
    parser.add_argument("--parity-router-margin", type=float, default=2.0, help="Per-expert router bias step applied during parity probe (>=1.0 eliminates 1-ULP expert flips)")
    parser.add_argument("--parity-min-cos", type=float, default=0.999)
    parser.add_argument("--fail-on-parity", action="store_true", default=False)
    parser.add_argument(
        "--enable-tbo",
        dest="enable_tbo",
        action="store_true",
        default=None,
        help="Enable 3-stage EP TBO in Target Verify & DFlash2 (default: follow QWEN4_5_RUNTIME_MOE_TBO)",
    )
    parser.add_argument("--no-enable-tbo", dest="enable_tbo", action="store_false")
    args = parser.parse_args()
    if args.verify_parity and not args.ref_in:
        args.ref_in = args.verify_parity
    if args.enable_tbo is None:
        args.enable_tbo = os.environ.get("QWEN4_5_RUNTIME_MOE_TBO", "0") == "1"
    os.environ["QWEN4_5_RUNTIME_MOE_TBO"] = "1" if args.enable_tbo else "0"

    _apply_dtype_mode_env(args.dtype_mode)
    if args.dense_linear_mode:
        os.environ["QWEN4_5_DENSE_LINEAR_MODE"] = args.dense_linear_mode
    eff_dflash_kv = args.dflash_kv_dtype or args.kv_dtype

    world_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from rtp_llm.config.collective_config import CollectiveConfig
    from rtp_llm.config.qwen4_5_config import Qwen4_5ModelConfig
    from rtp_llm.models_py.distributed.collective_torch import (
        Group,
        _get_group,
        init_distributed_environment,
    )
    from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
    from rtp_llm.models_py.model_desc.qwen4_5_runtime import Qwen4_5RTPModel
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import load_qwen4_5_from_checkpoint
    from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
    from rtp_llm.ops import NcclCommConfig, ParallelismConfig

    pc = ParallelismConfig()
    pc.world_size = world_size
    pc.world_rank = world_rank
    pc.local_rank = local_rank
    pc.local_world_size = world_size
    pc.pp_size = 1
    pc.pp_rank = 0
    pc.ffn_tp_size = 1
    pc.ffn_tp_rank = 0
    dp_size, dp_rank, tp_size, tp_rank, ep_size, ep_rank, sp_size, sp_rank = _resolve_parallel_layout(
        args.parallel_mode, world_rank, world_size
    )
    pc.tp_size = tp_size
    pc.tp_rank = tp_rank
    pc.dp_size = dp_size
    pc.dp_rank = dp_rank
    pc.ep_size = ep_size
    pc.ep_rank = ep_rank
    for attr, val in (("sp_size", sp_size), ("sp_rank", sp_rank), ("parallel_mode", args.parallel_mode)):
        try:
            setattr(pc, attr, val)
        except Exception:
            pass

    nccl_cfg = NcclCommConfig(nccl_ip=master_addr)
    init_distributed_environment(
        parallelism_config=pc,
        nccl_comm_config=nccl_cfg,
        nccl_init_port=master_port,
        backend="nccl",
        collective_config=CollectiveConfig(),
    )

    def _safe_get_group(grp_key: Any) -> Any:
        try:
            return _get_group(grp_key)
        except Exception:
            return None

    world_grp = _get_group(Group.WORLD)
    tp_grp = _safe_get_group(Group.TP) if pc.tp_size > 1 else None
    if tp_grp is None and pc.tp_size == world_size and world_size > 1:
        tp_grp = world_grp
    dp_grp = _safe_get_group(Group.DP) if pc.dp_size > 1 else None
    sp_grp = tp_grp if sp_size > 1 else None
    ep_grp = world_grp
    for attr, val in (("tp_group", tp_grp), ("dp_group", dp_grp), ("sp_group", sp_grp), ("ep_group", ep_grp)):
        if val is not None:
            try:
                setattr(pc, attr, val)
            except Exception:
                pass

    hw = load_or_calibrate_roofline(
        device,
        world_size,
        world_grp,
        roofline_cache_json=Path(args.roofline_cache) if args.roofline_cache else None,
    )
    if world_rank == 0:
        print(f"[ROOFLINE_HW] {json.dumps(hw.to_dict())}", flush=True)

    # 1. Load 5L full-width QSA Target + 5L full-width DFlash2 Draft
    t_load0 = time.perf_counter()
    raw_t_cfg = json.loads((Path(args.target_ckpt) / "config.json").read_text(encoding="utf-8"))
    t_arch = Qwen4_5ArchConfig.from_dict(raw_t_cfg)
    raw_batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    dp_size = max(1, int(pc.dp_size))
    batches = [b for b in raw_batches if b % dp_size == 0]
    if not batches:
        batches = [dp_size]
    max_local_batch = max(b // dp_size for b in batches)

    ctx_list_arg = [int(x.strip()) for x in args.ctx_list.split(",") if x.strip()]
    global_ctxs_by_bs: Dict[int, List[int]] = {
        bs: _resolve_global_ctx_list(
            bs,
            avg_ctx=args.avg_ctx,
            skew_preset=args.skew_preset,
            ctx_list_arg=ctx_list_arg,
            skew_layout=args.skew_layout,
        )
        for bs in batches
    }
    max_global_ctx = max(
        [args.avg_ctx] + [c for ctxs in global_ctxs_by_bs.values() for c in ctxs]
    )
    tpb = int(args.tokens_per_block)
    max_local_blocks = 0
    for bs, g_ctxs in global_ctxs_by_bs.items():
        bs_loc = bs // dp_size
        loc_ctxs = g_ctxs[pc.dp_rank * bs_loc : (pc.dp_rank + 1) * bs_loc]
        needed = sum((c + args.verify_width + 4 + tpb - 1) // tpb + 1 for c in loc_ctxs)
        max_local_blocks = max(max_local_blocks, needed)
    if args.ref_out or (args.ref_in and Path(args.ref_in).is_file()):
        uni_blocks = max_local_batch * ((args.avg_ctx + args.verify_width + 4 + tpb - 1) // tpb + 1)
        max_local_blocks = max(max_local_blocks, uni_blocks)
    max_total_blocks = max_local_blocks + 64

    lm_sig = inspect.signature(Qwen4_5ForCausalLM.__init__).parameters
    lm_kw: Dict[str, Any] = dict(
        use_triton_mxfp4=(args.dtype_mode != "bf16"),
        dtype=torch.bfloat16,
        device=device,
        ep_size=pc.ep_size,
        ep_rank=pc.ep_rank,
        ep_group=ep_grp,
        tp_size=pc.tp_size,
        tp_rank=pc.tp_rank,
    )
    for k, v in (
        ("tp_group", tp_grp),
        ("dp_size", dp_size),
        ("dp_rank", pc.dp_rank),
        ("dp_group", dp_grp),
        ("sp_size", sp_size),
        ("sp_rank", sp_rank),
        ("sp_group", sp_grp),
    ):
        if k in lm_sig and v is not None:
            lm_kw[k] = v
    if "parallel_mode" in lm_sig:
        lm_kw["parallel_mode"] = args.parallel_mode
        lm_kw["dp_equal_tokens"] = True
    if "kv_dtype" in lm_sig:
        lm_kw["kv_dtype"] = args.kv_dtype
    if "fp8_indexer_kv" in lm_sig and args.indexer_kv_dtype == "fp8_e4m3":
        lm_kw["fp8_indexer_kv"] = True
    if "dense_linear_mode" in lm_sig and args.dense_linear_mode:
        lm_kw["dense_linear_mode"] = args.dense_linear_mode

    target_lm = Qwen4_5ForCausalLM(t_arch, **lm_kw)
    load_qwen4_5_from_checkpoint(
        target_lm,
        args.target_ckpt,
        ple_mode=args.ple_mode,
        num_workers=8,
        verbose=(world_rank == 0),
    )
    if args.router_margin > 0.0:
        _stabilize_moe_routers(target_lm, args.router_margin, seed=2026)
    if hasattr(target_lm, "prepare_weights"):
        prep_sig = inspect.signature(target_lm.prepare_weights).parameters
        if "dense_linear_mode" in prep_sig and args.dense_linear_mode:
            target_lm.prepare_weights(dense_linear_mode=args.dense_linear_mode)
        else:
            target_lm.prepare_weights()

    mcfg = Qwen4_5ModelConfig(arch_config=t_arch)
    mcfg.max_seq_len = max(int(getattr(mcfg, "max_seq_len", 8192)), max_global_ctx + args.verify_width + 64)
    mcfg.capture_aux_hidden_layer_ids = [0, 1, 2, 3, 4]
    mcfg.is_mtp = False
    target_kv = build_20k_target_kv_cache(
        t_arch,
        max_batch=max_local_batch,
        avg_ctx=args.avg_ctx,
        verify_width=args.verify_width,
        tokens_per_block=args.tokens_per_block,
        tp_size=pc.tp_size,
        parallel_mode=args.parallel_mode,
        kv_dtype=args.kv_dtype,
        device=device,
        max_total_blocks=max_total_blocks,
    )
    target_rtp = Qwen4_5RTPModel.from_causal_lm(
        target_lm,
        model_config=mcfg,
        parallelism_config=pc,
        kv_cache=target_kv,
        max_generate_batch_size=max_local_batch,
    )

    d_cfg, dflash_model, d_load_s = load_dspark_5l_sharded(
        Path(args.dflash_ckpt),
        parallelism_config=pc,
        device=device,
        parallel_mode=args.parallel_mode,
        dp_size=dp_size,
        dp_rank=pc.dp_rank,
        dp_group=dp_grp,
        tp_group=tp_grp,
        ep_size=pc.ep_size,
        ep_rank=pc.ep_rank,
        ep_group=ep_grp,
        sp_size=sp_size,
        sp_rank=sp_rank,
        sp_group=sp_grp,
        max_generate_batch_size=max_local_batch,
        dflash_kv_dtype=eff_dflash_kv,
        dflash_attn_backend=args.dflash_attn_backend,
        dense_linear_mode=(args.dense_linear_mode or None),
        enable_tbo=bool(args.enable_tbo),
    )
    dflash_kv, dflash_kv_bufs = build_20k_dflash_kv_cache(
        d_cfg._dflash2_spec,
        max_batch=max_local_batch,
        avg_ctx=args.avg_ctx,
        verify_width=args.verify_width,
        tokens_per_block=args.tokens_per_block,
        dflash_kv_dtype=eff_dflash_kv,
        device=device,
        max_total_blocks=max_total_blocks,
    )
    dflash_model.kv_cache = dflash_kv
    if args.router_margin > 0.0:
        _stabilize_moe_routers(dflash_model, args.router_margin, seed=3026)
    dist.barrier(group=world_grp)

    if world_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(
            f"[LOADED] total={time.perf_counter() - t_load0:.2f}s (dflash={d_load_s:.2f}s), "
            f"rank0 allocated={mem_gb:.2f} GiB",
            flush=True,
        )

    ref_in_data = json.loads(Path(args.ref_in).read_text(encoding="utf-8")) if (args.ref_in and Path(args.ref_in).is_file()) else None
    ref_out_records: Dict[str, Any] = {}
    kineto_batch_set = {int(x.strip()) for x in args.kineto_batches.split(",") if x.strip()}
    kineto_results: Dict[str, Any] = {}

    parity_by_bs: Dict[int, Dict[str, Any]] = {}
    if args.ref_out or ref_in_data is not None:
        eff_parity_margin = args.router_margin if args.router_margin > 0.0 else float(args.parity_router_margin)
        if ref_in_data is not None and not args.ref_out and args.router_margin <= 0.0:
            meta_margin = (ref_in_data.get("_meta") or {}).get("router_margin")
            if meta_margin is None:
                for rv in ref_in_data.values():
                    if isinstance(rv, dict) and "router_margin" in rv:
                        meta_margin = rv["router_margin"]
                        break
            if meta_margin is None:
                if world_rank == 0:
                    print(
                        f"[PARITY WARN] Reference lacks router_margin metadata (legacy unstabilized ref); "
                        f"aligning parity probe router_margin=0.0 instead of {eff_parity_margin}",
                        flush=True,
                    )
                eff_parity_margin = 0.0
            elif float(meta_margin) != eff_parity_margin:
                if world_rank == 0:
                    print(
                        f"[PARITY INFO] Aligning parity probe router_margin={float(meta_margin)} to match reference",
                        flush=True,
                    )
                eff_parity_margin = float(meta_margin)
        if args.ref_out and world_rank == 0:
            ref_out_records["_meta"] = {"router_margin": eff_parity_margin, "dtype_mode": args.dtype_mode}
        saved_t_routers = _stabilize_moe_routers(target_lm, eff_parity_margin, seed=2026) if (args.router_margin <= 0.0 and eff_parity_margin > 0.0) else []
        saved_d_routers = _stabilize_moe_routers(dflash_model, eff_parity_margin, seed=3026) if (args.router_margin <= 0.0 and eff_parity_margin > 0.0) else []
        try:
            parity_batches = [b for b in (1, 8, 64) if b in batches]
            for p_bs in parity_batches:
                pkey = f"b{p_bs}"
                r_entry = ref_in_data.get(pkey) if (ref_in_data is not None and pkey in ref_in_data) else None
                saved_rec, par_res = _run_decode_parity_probe(
                    target_rtp=target_rtp,
                    dflash_model=dflash_model,
                    target_kv=target_kv,
                    dflash_kv_bufs=dflash_kv_bufs,
                    d_spec=d_cfg._dflash2_spec,
                    p_bs=p_bs,
                    dp_size=dp_size,
                    dp_rank=pc.dp_rank,
                    avg_ctx=args.avg_ctx,
                    verify_width=args.verify_width,
                    tokens_per_block=args.tokens_per_block,
                    device=device,
                    world_rank=world_rank,
                    save_ref=bool(args.ref_out),
                    ref_entry=r_entry,
                    parity_min_cos=float(args.parity_min_cos),
                )
                if world_rank == 0:
                    if saved_rec is not None:
                        saved_rec["router_margin"] = eff_parity_margin
                        ref_out_records[pkey] = saved_rec
                    if par_res is not None:
                        par_res["router_margin"] = eff_parity_margin
                        parity_by_bs[p_bs] = par_res
                        print(
                            f"[PARITY B={p_bs} margin={eff_parity_margin}] pass={par_res['passed']} "
                            f"t_cos={par_res['target_logits']['cosine']:.6f} "
                            f"d_cos={par_res['dflash_hidden']['cosine']:.6f} "
                            f"d_l0_attn={par_res.get('dflash_l0_attn_cos')} "
                            f"min_snap={par_res.get('min_snap_cos')} ({par_res.get('snap_count')} snaps)",
                            flush=True,
                        )
                        if not par_res["passed"] and par_res.get("snapshots"):
                            worst = sorted(par_res["snapshots"].items(), key=lambda kv: kv[1]["cosine"])[:4]
                            print(f"  [PARITY WORST B={p_bs}] " + ", ".join(f"{k}={v['cosine']:.6f}" for k, v in worst), flush=True)
        finally:
            if saved_t_routers:
                _restore_moe_routers(saved_t_routers)
            if saved_d_routers:
                _restore_moe_routers(saved_d_routers)
        if args.fail_on_parity and parity_by_bs:
            parity_ok_t = torch.tensor(
                [1 if all(bool(v.get("passed")) for v in parity_by_bs.values()) else 0],
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(parity_ok_t, src=0, group=world_grp)
            if int(parity_ok_t.item()) == 0:
                raise RuntimeError(f"Parity gate failed (<{args.parity_min_cos}): {parity_by_bs}")

    bench_rows: List[Dict[str, Any]] = []

    for bs in batches:
        bs_local = bs // dp_size
        global_ctxs = global_ctxs_by_bs[bs]
        local_ctxs = global_ctxs[pc.dp_rank * bs_local : (pc.dp_rank + 1) * bs_local]
        eff_avg_ctx_f = sum(global_ctxs) / float(max(1, len(global_ctxs)))
        eff_avg_ctx = int(round(eff_avg_ctx_f))
        ctx_min = min(global_ctxs)
        ctx_max = max(global_ctxs)
        dp_kv_min, dp_kv_max, dp_kv_imb = _compute_dp_kv_imbalance(global_ctxs, dp_size)
        torch.cuda.empty_cache()
        dist.barrier(group=world_grp)
        try:
            verify_in, draft_in, d_commit_in, anchor_ids = _build_verify_and_draft_inputs(
                target_kv=target_kv,
                dflash_kv_bufs=dflash_kv_bufs,
                spec=d_cfg._dflash2_spec,
                batch_size=bs_local,
                avg_ctx=eff_avg_ctx,
                verify_width=args.verify_width,
                tokens_per_block=args.tokens_per_block,
                device=device,
                dp_rank=pc.dp_rank,
                dp_size=dp_size,
                local_ctxs=local_ctxs,
            )

            # Prepare FMHA handles
            fmha_t = target_rtp.prepare_fmha_impl(verify_in, is_cuda_graph=args.cuda_graph)
            fmha_d = dflash_model.prepare_fmha_impl(draft_in, is_cuda_graph=args.cuda_graph)
            assert fmha_t is not None and fmha_d is not None

            # Eager warmup (both models)
            zero_acc = torch.zeros((bs_local,), dtype=torch.int32, device=device)
            t_out_eager = target_rtp.forward(verify_in, fmha_impl=fmha_t)
            target_rtp.prepare_speculative_target_commit(zero_acc)
            target_rtp.finish_speculative_target_commit(False)
            target_rtp.finalize_speculative_target_commit()
            if args.cuda_graph:
                target_rtp.on_cuda_graph_warmup(verify_in)
            aux_h = t_out_eager.mtp_target_hidden_states
            assert aux_h is not None
            d_commit_in.input_hiddens.copy_(aux_h.reshape(bs_local, args.verify_width, -1)[:, 0, :])
            dflash_model.forward_commit(d_commit_in)
            d_out_eager = dflash_model.forward_propose(draft_in, fmha_impl=fmha_d)

            g_target: Optional[torch.cuda.CUDAGraph] = None
            g_draft: Optional[torch.cuda.CUDAGraph] = None
            t_out_cg = t_out_eager
            d_out_cg = d_out_eager

            if args.cuda_graph:
                fmha_t.prepare_cuda_graph(verify_in.attention_inputs)
                g_target = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_target):
                    t_out_cg = target_rtp.forward(verify_in, fmha_impl=fmha_t)
                target_rtp.on_cuda_graph_capture(bs_local, verify_in)

                fmha_d.prepare_cuda_graph(draft_in.attention_inputs_by_tag)
                g_draft = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_draft):
                    d_out_cg = dflash_model.forward_propose(draft_in, fmha_impl=fmha_d)
                torch.cuda.synchronize(device)

            def _run_one_cycle(record_events: bool = False, use_cg: bool = True) -> Tuple[float, float, float, float]:
                if record_events:
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e2 = torch.cuda.Event(enable_timing=True)
                    e3 = torch.cuda.Event(enable_timing=True)
                    e4 = torch.cuda.Event(enable_timing=True)
                    e0.record()

                # Phase 1: DFlash2 Propose (CG or Eager)
                with torch.profiler.record_function("dflash_propose"):
                    if use_cg and g_draft is not None:
                        fmha_d.prepare_cuda_graph(draft_in.attention_inputs_by_tag)
                        g_draft.replay()
                        d_out_cur = d_out_cg
                    else:
                        d_out_cur = dflash_model.forward_propose(draft_in, fmha_impl=fmha_d)
                if record_events:
                    e1.record()

                # Phase 2: DFlash2 Candidate Selector (VocabParallel TopK + Markov walk)
                with torch.profiler.record_function("dflash_selector"):
                    if args.include_selector:
                        _toks, _cands, _edges = dflash_model.propose_with_candidate_selector(
                            d_out_cur.hidden_states, anchor_ids
                        )
                if record_events:
                    e2.record()

                # Phase 3: Target Verify V=8 (CG or Eager)
                with torch.profiler.record_function("target_verify"):
                    if use_cg and g_target is not None:
                        fmha_t.prepare_cuda_graph(verify_in.attention_inputs)
                        g_target.replay()
                        target_rtp.on_cuda_graph_replay(bs_local, verify_in)
                        t_out_cur = t_out_cg
                    else:
                        t_out_cur = target_rtp.forward(verify_in, fmha_impl=fmha_t)
                if record_events:
                    e3.record()

                # Phase 4: Target Logits + Speculative Tx Rollback + DFlash2 Feature KV Commit
                with torch.profiler.record_function("target_logits_commit"):
                    v_logits = target_rtp.compute_logits(t_out_cur.hidden_states)
                    target_rtp.prepare_speculative_target_commit(zero_acc)
                    target_rtp.finish_speculative_target_commit(False)
                    target_rtp.finalize_speculative_target_commit()
                    d_commit_in.input_hiddens.copy_(
                        t_out_cur.mtp_target_hidden_states.reshape(bs_local, args.verify_width, -1)[:, 0, :]
                    )
                    dflash_model.forward_commit(d_commit_in)
                if record_events:
                    e4.record()
                    torch.cuda.synchronize(device)
                    return (
                        float(e0.elapsed_time(e1)),
                        float(e1.elapsed_time(e2)),
                        float(e2.elapsed_time(e3)),
                        float(e3.elapsed_time(e4)),
                    )
                return 0.0, 0.0, 0.0, 0.0

            # Warmup cycles
            for _ in range(args.warmup_iters):
                _run_one_cycle(record_events=False)
            torch.cuda.synchronize(device)
            dist.barrier(group=world_grp)

            parity_info: Optional[Dict[str, Any]] = parity_by_bs.get(bs)

            # Timed benchmark cycles
            dp_ms, ds_ms, tv_ms, tl_ms, tot_ms = [], [], [], [], []
            for _ in range(args.bench_iters):
                dist.barrier(group=world_grp)
                a, b_t, c, d_t = _run_one_cycle(record_events=True)
                dp_ms.append(a)
                ds_ms.append(b_t)
                tv_ms.append(c)
                tl_ms.append(d_t)
                tot_ms.append(a + b_t + c + d_t)

            # Optional Rank 0 Kineto trace capture
            if bs in kineto_batch_set:
                trace_dir = Path(args.traces_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                skew_sfx = f"_{args.skew_preset}" if args.skew_preset else ("_customctx" if ctx_list_arg else "")
                trace_path = trace_dir / f"decode_{args.round_tag}_{args.parallel_mode}_{args.dtype_mode}_ctx{eff_avg_ctx}{skew_sfx}_b{bs}.json"
                if world_rank == 0:
                    print(f"[KINETO B={bs}] capturing 4 cycles -> {trace_path}", flush=True)
                dist.barrier(group=world_grp)
                prof_ctx = (
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False,
                    )
                    if world_rank == 0
                    else contextlib.nullcontext()
                )
                with prof_ctx as prof:
                    for _ in range(4):
                        _run_one_cycle(record_events=False)
                    torch.cuda.synchronize(device)
                if world_rank == 0 and prof is not None:
                    prof.export_chrome_trace(str(trace_path))
                    k_res = parse_torch_profiler_kineto(trace_path, top_k=15)
                    kineto_results[f"b{bs}"] = k_res
                    print(f"[KINETO B={bs}] scopes={k_res.get('scopes')}", flush=True)
                dist.barrier(group=world_grp)

                if args.op_kineto:
                    op_trace_path = trace_dir / f"decode_{args.round_tag}_{args.parallel_mode}_{args.dtype_mode}_ctx{eff_avg_ctx}{skew_sfx}_b{bs}_ops.json"
                    op_prof_ctx = (
                        torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=False,
                        )
                        if world_rank == 0
                        else contextlib.nullcontext()
                    )
                    op_scope_ctx = _kineto_decode_forward_scopes(target_rtp, dflash_model) if world_rank == 0 else contextlib.nullcontext()
                    with op_prof_ctx as op_prof:
                        with op_scope_ctx:
                            _run_one_cycle(record_events=False, use_cg=False)
                        torch.cuda.synchronize(device)
                    if world_rank == 0 and op_prof is not None:
                        op_prof.export_chrome_trace(str(op_trace_path))
                        op_parsed = parse_torch_profiler_kineto(op_trace_path, top_k=15)
                        kineto_results[f"b{bs}"]["eager_rf_ops"] = op_parsed.get("rf_ops", {})
                        kineto_results[f"b{bs}"]["eager_trace_file"] = str(op_trace_path)
                    dist.barrier(group=world_grp)

            mean_dp = statistics.mean(dp_ms)
            mean_ds = statistics.mean(ds_ms)
            mean_tv = statistics.mean(tv_ms)
            mean_tl = statistics.mean(tl_ms)
            mean_tot = statistics.mean(tot_ms)
            p50_tot = sorted(tot_ms)[len(tot_ms) // 2]

            rf_parallel_mode = (
                args.parallel_mode
                if args.parallel_mode in PARALLEL_MODES
                else ("dp2_tp2_ep4" if args.parallel_mode == "dp2_tpsp2_ep4" else args.parallel_mode)
            )
            rf = compute_qwen45_workload_metrics(
                t_arch,
                batch_size=bs,
                seq_len_per_step=args.verify_width,
                avg_ctx=eff_avg_ctx,
                parallel_mode=rf_parallel_mode,
                dtype_mode=(
                    "mxfp4"
                    if args.dtype_mode in ("w4a8", "auto", "triton")
                    else args.dtype_mode
                ),
                kv_dtype=args.kv_dtype,
                dflash_kv_dtype=(args.dflash_kv_dtype or args.kv_dtype),
                indexer_kv_dtype=args.indexer_kv_dtype,
                world_size=world_size,
                is_prefill=False,
                prefix_len=eff_avg_ctx,
                logits_tokens=bs * args.verify_width,
                include_dflash2=True,
                dflash_spec=DFlash2ModelSpec(),
                dflash_tokens_per_step=bs * args.verify_width,
                roofline=hw,
                elapsed_ms=mean_tot,
                op_times_ms={"lm_head": mean_tl},
            ).to_dict()

            row = {
                "batch_size": bs,
                "verify_width": args.verify_width,
                "avg_ctx": eff_avg_ctx,
                "ctx_min": ctx_min,
                "ctx_max": ctx_max,
                "dp_kv_min_tokens": dp_kv_min,
                "dp_kv_max_tokens": dp_kv_max,
                "dp_kv_imbalance": dp_kv_imb,
                "cycle_mean_ms": round(mean_tot, 3),
                "cycle_p50_ms": round(p50_tot, 3),
                "dflash_propose_ms": round(mean_dp, 3),
                "dflash_selector_ms": round(mean_ds, 3),
                "target_verify_ms": round(mean_tv, 3),
                "target_logits_commit_ms": round(mean_tl, 3),
                "raw_cycle_tok_s": round(bs * 1000.0 / max(mean_tot, 1e-6), 2),
                "raw_verified_tok_s": round(bs * args.verify_width * 1000.0 / max(mean_tot, 1e-6), 2),
                "roofline": rf,
                "parity": parity_info,
            }
            bench_rows.append(row)
            if world_rank == 0:
                skew_note = (
                    f" [skew {ctx_min}..{ctx_max}, dp_imb={dp_kv_imb:.2f}x]"
                    if ctx_min != ctx_max
                    else ""
                )
                print(
                    f"[B={bs:3d} ctx={eff_avg_ctx}] cycle={row['cycle_mean_ms']:6.3f}ms "
                    f"(draft={row['dflash_propose_ms']:.2f}+sel={row['dflash_selector_ms']:.2f}"
                    f"+verify={row['target_verify_ms']:.2f}+lm={row['target_logits_commit_ms']:.2f}) | "
                    f"raw_cyc={row['raw_cycle_tok_s']:7.1f} tok/s (verified={row['raw_verified_tok_s']:8.1f}) | "
                    f"MFU_bf16={rf.get('mfu_bf16_pct', 0.0):5.2f}% MFU_aware={rf.get('mfu_aware_pct', 0.0):5.2f}% "
                    f"MBU={rf.get('mbu_pct', 0.0):5.2f}% SOL={rf.get('sol_efficiency_pct', 0.0):5.2f}% "
                    f"(HBM floor={rf.get('ideal_hbm_ms', 0.0):.2f}ms){skew_note}",
                    flush=True,
                )

            del g_target, g_draft, verify_in, draft_in, d_commit_in, fmha_t, fmha_d, t_out_eager, d_out_eager
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as exc:
            if world_rank == 0:
                print(f"[B={bs:3d}] OOM: {exc}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            break

    if world_rank == 0:
        if args.ref_out and ref_out_records:
            p = Path(args.ref_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(ref_out_records) + "\n", encoding="utf-8")

        rep_avg_ctx = (
            int(round(sum(r["avg_ctx"] for r in bench_rows) / len(bench_rows)))
            if bench_rows
            else args.avg_ctx
        )
        extra_meta = {
            "ple_mode": args.ple_mode,
            "dense_linear_mode": args.dense_linear_mode or "bf16",
            "kv_dtype": args.kv_dtype,
            "dflash_kv_dtype": eff_dflash_kv,
            "dflash_attn_backend": args.dflash_attn_backend,
            "indexer_kv_dtype": args.indexer_kv_dtype,
            "router_margin": float(args.router_margin),
            "enable_tbo": bool(args.enable_tbo),
            "sp_size": int(sp_size),
            "skew_preset": args.skew_preset,
            "skew_layout": args.skew_layout,
            "ctx_list": ctx_list_arg,
        }
        report = {
            "round": args.round_tag,
            "parallel_mode": args.parallel_mode,
            "dtype_mode": args.dtype_mode,
            "avg_ctx": rep_avg_ctx,
            "verify_width": args.verify_width,
            **extra_meta,
            "hw_roofline": hw.to_dict(),
            "results": bench_rows,
            "kineto": kineto_results,
        }
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if args.leaderboard_json:
            _update_leaderboard(
                Path(args.leaderboard_json),
                round_tag=args.round_tag,
                parallel_mode=args.parallel_mode,
                dtype_mode=args.dtype_mode,
                avg_ctx=rep_avg_ctx,
                hw=hw,
                rows=bench_rows,
                extra_meta=extra_meta,
            )

    dist.barrier(group=world_grp)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
