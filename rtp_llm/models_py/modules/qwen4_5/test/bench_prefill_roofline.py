#!/usr/bin/env python3
"""Standalone 4-GPU Prefill Roofline Benchmark (`torchrun --nproc_per_node=4`) for Qwen4.5-Flash 5L.

Measures on 4x L20A (or configurable world_size) loading the 5L full-width QSA-Sparse mock:
  1. 0% Cache Hit (`T=20,480`) optimal scheduled token budget sweep (`M_budget=4K, 8K, 16K, 32K`)
     across both chunked 20K TTFT and single-step ragged saturation (`M=4K, 8K, 16K, 20K, 32K`).
  2. 90% Cache Hit (`18,432` cached prefix + `2,048` suffix x `B=1, 4, 8, 16`) with genuinely
     populated block-boundary KV / QSA Indexer / GDN SSM+Conv / ShortConv / PLE states.
  3. Non-inflated Per-Operator and Whole-Model MFU% & MBU% scorecard via `roofline_utils.py`.
  4. Rank-0 PyTorch Kineto Top-15 kernel & operator breakdown on 0%-hit and 90%-hit Prefill.
  5. Pluggable parallelisms (`tp4_ep4`, `dp4_ep4`, `dp2_tp2_ep4`, `sp4_tp4_ep4`) and frozen
     BF16 reference outputs + exact 64-token block-boundary snapshot states (`--save-ref` /
     `--verify-parity`) for cross-wave accuracy gates (`cos_sim >= 0.999` BF16).
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import inspect
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.config.qwen4_5_config import Qwen4_5ModelConfig, parse_qwen4_5_config_json
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan
from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
    LocalKVCache,
    LocalPyModelInputs,
    Qwen4_5RTPModel,
    build_block_tables_for_requests,
    make_tagged_attention_inputs,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import load_qwen4_5_from_checkpoint
from rtp_llm.models_py.modules.qwen4_5.tools.roofline_utils import (
    DTYPE_PRESETS,
    PARALLEL_MODES,
    HardwareRoofline,
    calibrate_l20a_roofline,
    compute_qwen45_workload_metrics,
)


DEFAULT_CKPT_PATH = "/home/admin/lhc_qwen45_opt/models/Qwen4.5-Flash-5L-fullwidth-qsa-mock"


# ---------------------------------------------------------------------------
# Distributed Subgroups & Pluggable Parallel Model Builder
# ---------------------------------------------------------------------------


def _build_parallel_groups(
    parallel_mode: str,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Create DP / TP / SP / EP subgroups and ParallelismConfig for pluggable 4-GPU execution."""
    pm = parallel_mode.lower()
    if pm not in PARALLEL_MODES:
        raise ValueError(f"Unsupported parallel_mode={parallel_mode!r}; expected {tuple(PARALLEL_MODES)}")
    dp, tp, ep, sp = PARALLEL_MODES[pm]
    if world_size == 1:
        return {
            "parallel_mode": "tp1_ep1",
            "dp_size": 1,
            "dp_rank": 0,
            "dp_group": None,
            "tp_size": 1,
            "tp_rank": 0,
            "tp_group": None,
            "ep_size": 1,
            "ep_rank": 0,
            "ep_group": None,
            "sp_size": 1,
            "sp_rank": 0,
            "sp_group": None,
            "pc": None,
        }
    if dp * tp != world_size or ep > world_size:
        raise ValueError(
            f"parallel_mode={pm} (dp={dp}, tp={tp}, ep={ep}) incompatible with world_size={world_size}"
        )

    tp_rank = rank % tp
    dp_rank = rank // tp
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    pc = None
    tp_group = None
    dp_group = None
    world_group = None
    try:
        from rtp_llm.config.collective_config import CollectiveConfig
        from rtp_llm.models_py.distributed.collective_torch import (
            Group,
            _get_group,
            init_distributed_environment,
        )
        from rtp_llm.ops import NcclCommConfig, ParallelismConfig

        pc = ParallelismConfig()
        pc.world_size = world_size
        pc.world_rank = rank
        pc.local_rank = local_rank
        pc.local_world_size = world_size
        pc.tp_size = tp
        pc.tp_rank = tp_rank
        pc.ep_size = ep
        pc.ep_rank = rank % ep
        pc.dp_size = dp
        pc.dp_rank = dp_rank
        pc.pp_size = 1
        pc.pp_rank = 0
        pc.ffn_tp_size = 1
        pc.ffn_tp_rank = 0
        # Attach optional SP/mode hints cleanly if not C++ slots
        for attr, val in (("sp_size", sp), ("sp_rank", tp_rank if sp > 1 else 0), ("parallel_mode", pm)):
            try:
                setattr(pc, attr, val)
            except Exception:
                pass

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        nccl_cfg = NcclCommConfig(nccl_ip=master_addr)
        init_distributed_environment(
            parallelism_config=pc,
            nccl_comm_config=nccl_cfg,
            nccl_init_port=master_port,
            backend="nccl",
            collective_config=CollectiveConfig(),
        )
        world_group = _get_group(Group.WORLD)
        tp_group = _get_group(Group.TP) if tp > 1 else None
        dp_group = _get_group(Group.DP) if dp > 1 else None
    except Exception:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_group = dist.group.WORLD
        if tp == world_size:
            tp_group = world_group
        elif tp > 1:
            for d in range(dp):
                ranks = list(range(d * tp, (d + 1) * tp))
                g = dist.new_group(ranks=ranks)
                if rank in ranks:
                    tp_group = g
        if dp == world_size:
            dp_group = world_group
        elif dp > 1:
            for t in range(tp):
                ranks = [d * tp + t for d in range(dp)]
                g = dist.new_group(ranks=ranks)
                if rank in ranks:
                    dp_group = g

    sp_group = tp_group if sp > 1 else None
    sp_rank = tp_rank if sp > 1 else 0
    return {
        "parallel_mode": pm,
        "dp_size": dp,
        "dp_rank": dp_rank,
        "dp_group": dp_group,
        "tp_size": tp,
        "tp_rank": tp_rank,
        "tp_group": tp_group,
        "ep_size": ep,
        "ep_rank": rank % ep,
        "ep_group": world_group if ep > 1 else None,
        "sp_size": sp,
        "sp_rank": sp_rank,
        "sp_group": sp_group,
        "pc": pc,
    }


def _filter_supported_kwargs(fn: Any, candidates: Mapping[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


def build_prefill_model_and_kv(
    ckpt_path: str,
    *,
    pg: Mapping[str, Any],
    dtype_mode: str,
    kv_dtype: str,
    device: torch.device,
    max_kv_tokens: int,
    tokens_per_block: int = 64,
    num_load_workers: int = 8,
    ple_mode: str = "gpu_sharded",
    dense_linear_mode: Optional[str] = None,
    fp8_indexer_kv: Optional[bool] = None,
) -> Tuple[Any, LocalKVCache, Qwen4_5RTPModel]:
    """Load 5L full-width QSA checkpoint across 4 GPUs and wrap in Qwen4_5RTPModel."""
    cfg_raw = json.loads((Path(ckpt_path) / "config.json").read_text(encoding="utf-8"))
    arch = parse_qwen4_5_config_json(cfg_raw, strict=False)
    dm = dtype_mode.lower()
    if dm not in DTYPE_PRESETS:
        raise ValueError(f"Unsupported dtype_mode={dtype_mode!r}; expected {tuple(DTYPE_PRESETS)}")

    compute_mode = "fp8" if dm in ("fp8", "fp8_mxfp4", "w8a8_mxfp4", "fp8_fp4") else "bf16"
    eff_dense_mode = (
        str(dense_linear_mode).strip().lower()
        if dense_linear_mode is not None and str(dense_linear_mode).strip() != ""
        else ("auto" if dm in ("fp8", "fp8_mxfp4", "w8a8_mxfp4", "fp8_fp4") else "bf16")
    )
    kv_low = str(kv_dtype).strip().lower()
    is_fp8_kv = kv_low.startswith("fp8") or kv_low == "f8_e4m3"
    eff_fp8_idx = is_fp8_kv if fp8_indexer_kv is None else bool(fp8_indexer_kv)
    eff_ple_mode = str(ple_mode).strip().lower()

    model_parallel_mode = None if pg["parallel_mode"] == "tp1_ep1" else pg["parallel_mode"]
    causal_kwargs = _filter_supported_kwargs(
        Qwen4_5ForCausalLM.__init__,
        {
            "use_triton_mxfp4": (dm != "bf16"),
            "compute_mode": compute_mode,
            "dense_linear_mode": eff_dense_mode,
            "kv_dtype": "fp8_e4m3" if is_fp8_kv else "bf16",
            "fp8_indexer_kv": eff_fp8_idx,
            "dtype": torch.bfloat16,
            "device": device,
            "init_mxfp4_weights": True,
            "parallel_mode": model_parallel_mode,
            "dp_attn": bool(pg["dp_size"] > 1),
            "dp_size": pg["dp_size"],
            "dp_rank": pg["dp_rank"],
            "dp_group": pg["dp_group"],
            "tp_size": pg["tp_size"],
            "tp_rank": pg["tp_rank"],
            "tp_group": pg["tp_group"],
            "ep_size": pg["ep_size"],
            "ep_rank": pg["ep_rank"],
            "ep_group": pg["ep_group"],
            "sp_size": pg["sp_size"],
            "sp_rank": pg["sp_rank"],
            "sp_group": pg["sp_group"],
        },
    )
    causal_lm = Qwen4_5ForCausalLM(arch, **causal_kwargs)
    load_kwargs = _filter_supported_kwargs(
        load_qwen4_5_from_checkpoint,
        {"ple_mode": eff_ple_mode, "num_workers": num_load_workers},
    )
    load_qwen4_5_from_checkpoint(causal_lm, ckpt_path, **load_kwargs)
    prep_kwargs = _filter_supported_kwargs(
        causal_lm.prepare_weights,
        {"dense_linear_mode": eff_dense_mode},
    )
    causal_lm.prepare_weights(**prep_kwargs)

    plan_kv_dt = "F8_E4M3" if is_fp8_kv else "BF16"
    try:
        plan_kwargs = _filter_supported_kwargs(
            build_qwen4_5_kv_cache_plan,
            {"tp_size": pg["tp_size"], "parallel_mode": model_parallel_mode, "kv_dtype": plan_kv_dt},
        )
        plan = build_qwen4_5_kv_cache_plan(arch, **plan_kwargs)
    except ValueError:
        plan_kwargs = _filter_supported_kwargs(
            build_qwen4_5_kv_cache_plan,
            {"tp_size": pg["tp_size"], "parallel_mode": model_parallel_mode, "kv_dtype": "FP8" if is_fp8_kv else "BF16"},
        )
        plan = build_qwen4_5_kv_cache_plan(arch, **plan_kwargs)

    total_blocks = max(64, (max_kv_tokens + tokens_per_block - 1) // tokens_per_block + 16)
    num_blocks_per_tag: Dict[str, int] = {}
    for lp in plan.layers:
        if lp.allocates_kv_cache:
            num_blocks_per_tag["full"] = total_blocks
            if lp.has_qsa_indexer_cache:
                num_blocks_per_tag["indexer_kv"] = total_blocks
                num_blocks_per_tag["indexer_state"] = total_blocks
        if lp.allocates_ssm_state:
            num_blocks_per_tag["linear"] = total_blocks
        if getattr(lp, "has_gdn_conv", False):
            num_blocks_per_tag["gdn_consumer_q_conv"] = total_blocks
        for reg in lp.opaque_regions:
            num_blocks_per_tag[reg.tag] = total_blocks

    fp8_dt = getattr(torch, "float8_e4m3fn", None) if is_fp8_kv else None
    if is_fp8_kv and fp8_dt is None:
        raise RuntimeError("torch.float8_e4m3fn is required for fp8_e4m3 KV cache")
    kv_extra = _filter_supported_kwargs(
        LocalKVCache.__init__,
        {
            "kv_dtype": fp8_dt if is_fp8_kv else torch.bfloat16,
            "allocate_fp8_kv_scale": is_fp8_kv,
        },
    )
    kv_cache = LocalKVCache(
        plan,
        tokens_per_block=tokens_per_block,
        num_blocks_per_tag=num_blocks_per_tag,
        device=device,
        dtype=torch.bfloat16,
        **kv_extra,
    )
    if is_fp8_kv:
        assert fp8_dt is not None
        for lp in plan.layers:
            lid = lp.layer_idx
            tag_map = kv_cache.layer_caches[lid]
            if "full" in tag_map:
                lc = tag_map["full"]
                nb = int(lc.kv_cache_base.shape[0])
                if lc.kv_cache_base.dtype != fp8_dt:
                    lc.kv_cache_base = torch.zeros_like(lc.kv_cache_base, dtype=fp8_dt)
                if lc.kv_scale_base is None:
                    lc.kv_scale_base = torch.ones(
                        (nb, 2, int(lp.num_kv_heads), int(tokens_per_block)),
                        dtype=torch.float32,
                        device=device,
                    )
    mcfg = Qwen4_5ModelConfig(arch_config=arch)
    mcfg.max_seq_len = max(int(getattr(mcfg, "max_seq_len", 8192)), max_kv_tokens + 64)
    mcfg.is_mtp = False
    rtp_kwargs = _filter_supported_kwargs(
        Qwen4_5RTPModel.from_causal_lm,
        {
            "model_config": mcfg,
            "parallelism_config": pg.get("pc"),
            "kv_cache": kv_cache,
            "parallel_mode": model_parallel_mode,
            "dp_size": pg["dp_size"],
            "dp_rank": pg["dp_rank"],
            "dp_group": pg["dp_group"],
            "sp_size": pg["sp_size"],
            "sp_rank": pg["sp_rank"],
            "sp_group": pg["sp_group"],
        },
    )
    rtp_model = Qwen4_5RTPModel.from_causal_lm(causal_lm, **rtp_kwargs)
    return arch, kv_cache, rtp_model


# ---------------------------------------------------------------------------
# Helpers: DP Request Partitioning, Timing, Block Reset, Kineto Attribution
# ---------------------------------------------------------------------------


def _local_batch_for_rank(global_batch: int, pg: Mapping[str, Any]) -> Tuple[int, int]:
    """Return `(local_batch, start_req_idx)` for this rank under DP or TP."""
    dp = int(pg["dp_size"])
    dp_rank = int(pg["dp_rank"])
    if dp <= 1:
        return global_batch, 0
    base, rem = divmod(global_batch, dp)
    local_b = base + (1 if dp_rank < rem else 0)
    start_idx = dp_rank * base + min(dp_rank, rem)
    return local_b, start_idx


def _make_deterministic_tokens(
    seq_lens: Sequence[int],
    *,
    vocab_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    chunks = [
        torch.randint(10, min(vocab_size, 240000), (ln,), generator=gen, dtype=torch.int64)
        for ln in seq_lens
    ]
    if not chunks:
        return torch.empty((0,), dtype=torch.int64, device=device)
    return torch.cat(chunks, dim=0).to(device=device)


def _sync_max_ms(local_ms: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return local_ms
    t = torch.tensor([local_ms], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _timed_step_ms(
    fn: Any,
    *,
    device: torch.device,
    warmup: int = 2,
    iters: int = 8,
    pre_iter_fn: Any = None,
) -> Tuple[float, float]:
    """Run `fn()` with warmup/iters and return `(mean_rank_max_ms, p50_rank_max_ms)`."""
    for _ in range(warmup):
        if pre_iter_fn is not None:
            pre_iter_fn()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        fn()
        torch.cuda.synchronize(device)

    per_iter_ms: List[float] = []
    for _ in range(iters):
        if pre_iter_fn is not None:
            pre_iter_fn()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize(device)
        per_iter_ms.append(_sync_max_ms(float(ev0.elapsed_time(ev1)), device))

    sorted_ms = sorted(per_iter_ms)
    p50 = sorted_ms[len(sorted_ms) // 2]
    return sum(per_iter_ms) / len(per_iter_ms), p50


def _reset_kv_allocator(kv_cache: LocalKVCache) -> None:
    kv_cache._next_free_block = {t: 1 for t in kv_cache._tags_present}


def _run_prefill_pass(
    rtp_model: Qwen4_5RTPModel,
    kv_cache: LocalKVCache,
    tables: Dict[str, List[List[int]]],
    flat_ids: torch.Tensor,
    input_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    *,
    device: torch.device,
    compute_last_logits: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run one `Qwen4_5RTPModel.forward` prefill step plus optional last-token `compute_logits`."""
    if not input_lengths or int(flat_ids.numel()) == 0:
        empty_h = torch.empty((0, rtp_model.arch_config.hidden_size), dtype=torch.bfloat16, device=device)
        return empty_h, None

    seq_lengths = [p + l for p, l in zip(prefix_lengths, input_lengths)]
    attn = make_tagged_attention_inputs(
        tables,
        is_prefill=True,
        input_lengths=list(input_lengths),
        sequence_lengths=seq_lengths,
        prefix_lengths=list(prefix_lengths),
        device=device,
    )
    out = rtp_model.forward(LocalPyModelInputs(input_ids=flat_ids, attention_inputs=attn))
    hidden = out.hidden_states
    logits = None
    if compute_last_logits:
        ends: List[int] = []
        acc = 0
        for ln in input_lengths:
            acc += ln
            ends.append(acc - 1)
        last_h = hidden[torch.tensor(ends, dtype=torch.int64, device=device)]
        logits = rtp_model.compute_logits(last_h)
    return hidden, logits


# ---------------------------------------------------------------------------
# Rank-0 PyTorch Profiler (Kineto) Analysis -> Top-15 + Operator Attribution
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _kineto_forward_scopes(rtp_model: Any):
    """Temporarily attach non-invasive ``record_function`` scopes during Rank-0 Kineto capture.

    Only installed for the single Kineto capture iteration so timed runs keep 0 overhead.
    """
    if rtp_model is None:
        yield
        return

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

    model = getattr(rtp_model, "model", None)
    if model is not None:
        _hook_mod(getattr(model, "embed_tokens", None), "embed_ple")
        hcm = getattr(model, "hyper_connection_mixer", None)
        for meth in ("mix_with_latent", "combine_and_mix_with_latent", "mix_from_normed", "combine"):
            _wrap_method(hcm, meth, "hc_and_output_sconv")

        for lyr in getattr(model, "layers", ()):
            ple = getattr(lyr, "ple", None)
            _hook_mod(ple, "embed_ple")
            _wrap_method(ple, "forward_packed_prefill_with_hc", "embed_ple")

            for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc_mod = getattr(lyr, hc_name, None)
                for meth in ("mix_with_latent", "combine_and_mix_with_latent", "mix_from_normed", "combine"):
                    _wrap_method(hc_mod, meth, "hc_and_output_sconv")
            _hook_mod(getattr(lyr, "attn_output_sconv", None), "hc_and_output_sconv")
            _hook_mod(getattr(lyr, "ffn_output_sconv", None), "hc_and_output_sconv")

            lin = getattr(lyr, "linear_attn", None)
            if lin is not None:
                _wrap_method(lin, "forward_packed_prefill", "gdn_core")
                for sub_attr in ("in_proj_qkvz", "in_proj_ba", "in_proj_qkvz_ba", "out_proj"):
                    _hook_mod(getattr(lin, sub_attr, None), "gdn_proj")

            sa = getattr(lyr, "self_attn", None)
            if sa is not None:
                for sub_attr in ("q_proj", "k_proj", "v_proj", "qkv_idx_proj", "o_proj"):
                    _hook_mod(getattr(sa, sub_attr, None), "full_attn_proj")
                _wrap_method(sa, "project_qkv_and_indexer", "full_attn_proj")
                idx_mod = getattr(sa, "indexer", None)
                _wrap_method(idx_mod, "project", "full_attn_proj")

            mlp = getattr(lyr, "mlp", None)
            if mlp is not None:
                is_moe = hasattr(mlp, "latent_down_proj")
                # Kernel-name rules catch all routed MXFP4/DeepGEMM/SiTU/TopK ops as "moe_routed";
                # tagging LatentMoE outer scope as "moe_latent_shared" captures fused down+gate GEMMs too.
                _hook_mod(mlp, "moe_latent_shared" if is_moe else "dense_mlp")

    _wrap_method(rtp_model, "compute_logits", "lm_head")

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


def _classify_prefill_kernel(
    name: str,
    cpu_parent: str,
    input_shapes: Sequence[Any],
    tp_size: int,
    rf_bucket: str = "",
) -> str:
    nl = name.lower()
    if any(k in nl for k in ("_qsa_", "_sparse_", "sparse_prefill", "mbtopk", "digitcounts")):
        return "full_attn_qsa_core"
    if any(k in nl for k in ("_gdn_", "dwconv1d", "causal_dwconv1d", "rms_norm_gated")):
        return "gdn_core"
    if any(k in nl for k in ("_hc_", "hyper_connection", "sconv", "shortconv")):
        return "hc_and_output_sconv"
    if any(k in nl for k in ("_ple_", "ple_gate")):
        return "embed_ple"
    if "alltoall" in nl:
        return "comm_alltoall"
    if any(k in nl for k in ("nccl", "allreduce", "allgather", "oneshot", "twoshot", "lamport")):
        return "comm_allreduce"
    # DeepGEMM FP8xFP8 (1x128) dense linear ops also instantiate the
    # `sm100_fp8_fp4_gemm_1d1d_impl` template with `(float_e4m3_t, float_e4m3_t)`
    # and `128u, 128u, 128u` (without `m_grouped` or `e2m1`); exclude those so
    # `rf_bucket` attributes them to `gdn_proj`/`full_attn_proj`/`dense_mlp`/`moe_latent_shared`.
    is_dg_fp8_dense = (
        "deep_gemm" in nl
        and "m_grouped" not in nl
        and "e2m1" not in nl
        and (
            "float_e4m3_t, cutlass::float_e4m3_t" in nl
            or ("128u, 128u, 128u" in nl and "32u, 32u, 32u" not in nl)
        )
    )
    if (
        not is_dg_fp8_dense
        and any(
            k in nl
            for k in (
                "_mxfp4_",
                "fp8_fp4",
                "m_grouped",
                "group_gemm",
                "groupproblemshape",
                "_situ_glu",
                "_situ_and_mul",
                "_scatter_fp8",
                "_gather_slot",
                "sigmoid_topk",
                "sbtopk",
                "radixsortkv",
            )
        )
    ):
        return "moe_routed"

    if rf_bucket:
        return rf_bucket

    # Fallback heuristic attribution when unannotated
    if any(k in nl for k in ("nvjet", "cutlass", "cublas", "sgemm", "gemm", "deep_gemm")):
        flat_dims = [int(x) for sub in input_shapes if isinstance(sub, (list, tuple)) for x in sub if isinstance(x, int)]
        n_dim = flat_dims[-1] if len(flat_dims) >= 4 else 0
        k_dim = flat_dims[1] if len(flat_dims) >= 2 else 0
        local_vocab = 248320 // max(1, tp_size)
        if local_vocab in flat_dims or 248320 in flat_dims:
            return "lm_head"
        if 20480 in flat_dims and k_dim == 5120:
            return "embed_ple"
        if 256 in flat_dims or 1152 in flat_dims or (4096 in flat_dims and 5120 in flat_dims):
            return "full_attn_proj"
        if 12800 in flat_dims or (3200 in flat_dims and 5120 in flat_dims):
            return "dense_mlp"
        if 2560 in flat_dims or 1280 in flat_dims or 640 in flat_dims:
            return "moe_latent_shared"
        if 5120 in flat_dims and (2048 in flat_dims or 10240 in flat_dims or 64 in flat_dims):
            return "gdn_proj"
        if n_dim == 5120 and k_dim == 5120:
            return "gdn_proj"
        return "dense_gemm_other"

    if any(k in nl for k in ("copy", "fillfunctor", "binaryfunctor", "unaryfunctor", "catarray", "indexselect", "elementwise", "reduce_kernel")):
        return "elementwise_and_copy"
    return "other_triton_or_misc"


def capture_prefill_kineto(
    run_fn: Any,
    *,
    trace_path: Optional[str | Path],
    tp_size: int,
    device: torch.device,
    rtp_model: Any = None,
) -> Dict[str, Any]:
    """Capture a single prefill iteration on Rank 0 with PyTorch Kineto & return Top-15 + buckets."""
    # Warm 1 step cleanly before profiling
    run_fn()
    torch.cuda.synchronize(device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    if not is_rank0:
        run_fn()
        torch.cuda.synchronize(device)
        return {}

    from torch.profiler import ProfilerActivity, profile

    tmp_trace = Path(trace_path) if trace_path else Path("/tmp/r0_prefill_kineto_tmp.json")
    tmp_trace.parent.mkdir(parents=True, exist_ok=True)
    with _kineto_forward_scopes(rtp_model):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            run_fn()
            torch.cuda.synchronize(device)
    prof.export_chrome_trace(str(tmp_trace))

    raw_events = json.loads(tmp_trace.read_text(encoding="utf-8")).get("traceEvents", [])
    if not trace_path and tmp_trace.exists():
        tmp_trace.unlink(missing_ok=True)

    # Correlate CUDA kernels -> launching cpu_op / user_annotation rf:<bucket> spans
    cpu_ops: List[Tuple[float, float, int, str, Any]] = []
    rf_spans: List[Tuple[float, float, int, str]] = []
    rt_by_corr: Dict[int, Tuple[float, int]] = {}
    for e in raw_events:
        cat = e.get("cat")
        if cat in ("cpu_op", "user_annotation"):
            ts = float(e.get("ts", 0.0))
            dur = float(e.get("dur", 0.0))
            tid = int(e.get("tid", 0) or 0)
            nm = str(e.get("name", ""))
            if nm.startswith("rf:"):
                rf_spans.append((ts, ts + dur, tid, nm[3:]))
            elif cat == "cpu_op":
                args = e.get("args") or {}
                dims = args.get("Input Dims") or args.get("dims") or []
                cpu_ops.append((ts, ts + dur, tid, nm, dims))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                rt_by_corr[int(corr)] = (float(e.get("ts", 0.0)), int(e.get("tid", 0) or 0))
    cpu_ops.sort(key=lambda x: x[0])
    rf_spans.sort(key=lambda x: x[0])

    cpu_starts = [x[0] for x in cpu_ops]
    rf_starts = [x[0] for x in rf_spans]

    def _find_cpu_parent(rt_ts: float, rt_tid: int) -> Tuple[str, Any, str]:
        idx = bisect.bisect_right(cpu_starts, rt_ts) - 1
        best_name = ""
        best_dims: Any = []
        best_span = float("inf")
        for j in range(max(0, idx - 64), idx + 1):
            s, end_t, tid, nm, dm = cpu_ops[j]
            if tid == rt_tid and s <= rt_ts <= end_t:
                span = end_t - s
                if span < best_span:
                    best_span = span
                    best_name = nm
                    best_dims = dm

        ridx = bisect.bisect_right(rf_starts, rt_ts) - 1
        best_rf = ""
        best_rspan = float("inf")
        for j in range(max(0, ridx - 64), ridx + 1):
            s, end_t, tid, bkt = rf_spans[j]
            if tid == rt_tid and s <= rt_ts <= end_t:
                span = end_t - s
                if span < best_rspan:
                    best_rspan = span
                    best_rf = bkt
        return best_name, best_dims, best_rf

    ker_stats: Dict[str, List[Any]] = defaultdict(lambda: [0.0, 0, ""])
    bucket_ms: Dict[str, float] = defaultdict(float)
    total_ms = 0.0
    total_launches = 0

    for e in raw_events:
        if e.get("cat") == "kernel":
            dur_ms = float(e.get("dur", 0.0)) / 1000.0
            if dur_ms <= 0.0:
                continue
            name = str(e.get("name", ""))
            corr = (e.get("args") or {}).get("correlation")
            cpu_name, shapes, rf_bkt = ("", [], "")
            if corr is not None and int(corr) in rt_by_corr:
                rt_ts, rt_tid = rt_by_corr[int(corr)]
                cpu_name, shapes, rf_bkt = _find_cpu_parent(rt_ts, rt_tid)
            bkt = _classify_prefill_kernel(name, cpu_name, shapes, tp_size, rf_bucket=rf_bkt)
            ker = ker_stats[name]
            ker[0] += dur_ms
            ker[1] += 1
            ker[2] = bkt
            bucket_ms[bkt] += dur_ms
            total_ms += dur_ms
            total_launches += 1

    top15 = [
        {
            "rank": idx,
            "name": name,
            "ms": round(st[0], 4),
            "pct": round((st[0] / max(total_ms, 1e-9)) * 100.0, 2),
            "calls": st[1],
            "mean_us": round((st[0] * 1000.0) / max(st[1], 1), 2),
            "bucket": st[2],
        }
        for idx, (name, st) in enumerate(
            sorted(ker_stats.items(), key=lambda kv: kv[1][0], reverse=True)[:15], 1
        )
    ]
    buckets = {
        k: {"ms": round(v, 4), "pct": round((v / max(total_ms, 1e-9)) * 100.0, 2)}
        for k, v in sorted(bucket_ms.items(), key=lambda kv: kv[1], reverse=True)
    }
    return {
        "trace_file": str(trace_path) if trace_path else None,
        "total_cuda_ms": round(total_ms, 4),
        "total_launches": total_launches,
        "unique_kernels": len(ker_stats),
        "top_15": top15,
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Frozen BF16 Reference & Parity Gate (including Exact 64-tok Snapshot States)
# ---------------------------------------------------------------------------


def _gather_tp_tensor(t: torch.Tensor, dim: int, pg: Mapping[str, Any]) -> torch.Tensor:
    tp = int(pg["tp_size"])
    tp_grp = pg["tp_group"]
    if tp <= 1 or tp_grp is None:
        return t.detach().cpu()
    chunks = [torch.empty_like(t) for _ in range(tp)]
    dist.all_gather(chunks, t.contiguous(), group=tp_grp)
    return torch.cat(chunks, dim=dim).detach().cpu()


def _extract_64tok_snapshots(
    kv_cache: LocalKVCache,
    tables: Mapping[str, List[List[int]]],
    pg: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Gather canonical block-0 (`0..64` tokens) KV and opaque snapshot tensors to CPU."""
    snaps: Dict[str, torch.Tensor] = {}
    for lid, tag_map in enumerate(kv_cache.layer_caches):
        for tag, lc in tag_map.items():
            bid = int(tables[tag][0][0])
            row = lc.kv_cache_base[bid : bid + 1].detach()
            if tag in ("linear", "gdn_ssm_state", "gdn_conv_state", "gdn_consumer_q_conv") and int(pg["tp_size"]) > 1:
                row = _gather_tp_tensor(row, dim=-1, pg=pg)
            else:
                row = row.cpu()
            snaps[f"L{lid}:{tag}"] = row
    return snaps


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


def _per_tok_cos_stats(cur: torch.Tensor, ref: torch.Tensor) -> Dict[str, Any]:
    if cur.ndim != 2 or cur.shape != ref.shape or cur.shape[0] <= 0:
        return {}
    per_tok = F.cosine_similarity(cur.float(), ref.float(), dim=-1)
    return {
        "per_tok_cos_min": round(float(per_tok.min().item()), 6),
        "per_tok_cos_p10": round(float(torch.quantile(per_tok, 0.10).item()), 6),
        "per_tok_cos_median": round(float(per_tok.median().item()), 6),
        "frac_ge_0999": round(float((per_tok >= 0.999).float().mean().item()), 4),
    }


@contextlib.contextmanager
def _temporary_stabilized_router(
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    *,
    base_bias: float = 4.0,
    step: float = 0.25,
):
    """Temporarily set deterministic expert_bias margins across EP shards, restoring on exit."""
    model = getattr(rtp_model, "model", None)
    saved: List[Tuple[Any, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for layer in getattr(model, "layers", ()):
                if not getattr(layer, "is_moe", False):
                    continue
                mlp = getattr(layer, "mlp", None)
                eb = getattr(mlp, "expert_bias", None)
                if mlp is None or eb is None:
                    continue
                saved.append((eb, eb.detach().clone()))
                eb.zero_()
                n_exp = int(getattr(mlp, "num_experts", eb.numel()))
                top_k = max(1, min(int(getattr(mlp, "top_k", 10)), n_exp))
                shard_cnt = 4 if (n_exp >= 4 and n_exp % 4 == 0) else max(1, int(pg.get("ep_size", 1)))
                shard = max(1, n_exp // shard_cnt)
                lid = int(getattr(layer, "layer_idx", 0))
                used: set[int] = set()
                for slot in range(top_k):
                    r = slot % shard_cnt
                    k_within = slot // shard_cnt
                    e_id = r * shard + ((lid * 7 + k_within * 13) % shard)
                    while e_id in used:
                        e_id = (e_id + 1) % n_exp
                    used.add(e_id)
                    eb[e_id] = float(base_bias) + float(step) * float(top_k - slot)
        yield
    finally:
        with torch.no_grad():
            for eb, orig in saved:
                eb.copy_(orig)


@contextlib.contextmanager
def _temporary_mxfp4_ref_backend(rtp_model: Qwen4_5RTPModel):
    """Temporarily switch MoE and weight-backed dense FP8 layers on ``rtp_model`` to BF16/ref."""
    model = getattr(rtp_model, "model", None)
    prev_env = os.environ.get("QWEN4_5_MXFP4_BACKEND")
    prev_dense_env = os.environ.get("QWEN4_5_DENSE_LINEAR_MODE")
    saved_mlp: List[Tuple[Any, bool, Any, Any]] = []
    saved_fp8: List[Tuple[Any, Any]] = []
    saved_dense_attr: List[Tuple[Any, Any]] = []
    set_dense_env = False
    try:
        os.environ["QWEN4_5_MXFP4_BACKEND"] = "ref"
        all_mods: List[Any] = []
        seen_ids = set()
        for root in (rtp_model, model):
            if root is None or not hasattr(root, "modules"):
                continue
            for m in root.modules():
                if id(m) not in seen_ids:
                    seen_ids.add(id(m))
                    all_mods.append(m)

        has_weightless_fp8 = False
        for m in all_mods:
            if hasattr(m, "mode") and hasattr(m, "_refresh_fast_dispatch"):
                if getattr(m, "weight", None) is not None:
                    saved_fp8.append((m, m.mode))
                    m.mode = "bf16"
                    m._refresh_fast_dispatch()
                else:
                    has_weightless_fp8 = True

        if not has_weightless_fp8:
            os.environ["QWEN4_5_DENSE_LINEAR_MODE"] = "bf16"
            set_dense_env = True
            for m in all_mods:
                if hasattr(m, "dense_linear_mode"):
                    saved_dense_attr.append((m, m.dense_linear_mode))
                    m.dense_linear_mode = "bf16"

        for layer in getattr(model, "layers", ()):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            se = getattr(mlp, "shared_expert", None)
            saved_mlp.append(
                (
                    mlp,
                    bool(getattr(mlp, "use_triton_mxfp4", True)),
                    se,
                    getattr(se, "use_triton", None) if se is not None else None,
                )
            )
            if hasattr(mlp, "use_triton_mxfp4"):
                mlp.use_triton_mxfp4 = False
            if se is not None and hasattr(se, "use_triton"):
                se.use_triton = False
        yield
    finally:
        if prev_env is None:
            os.environ.pop("QWEN4_5_MXFP4_BACKEND", None)
        else:
            os.environ["QWEN4_5_MXFP4_BACKEND"] = prev_env
        if set_dense_env:
            if prev_dense_env is None:
                os.environ.pop("QWEN4_5_DENSE_LINEAR_MODE", None)
            else:
                os.environ["QWEN4_5_DENSE_LINEAR_MODE"] = prev_dense_env
        for m, orig_dlm in saved_dense_attr:
            m.dense_linear_mode = orig_dlm
        for m, orig_mode in saved_fp8:
            m.mode = orig_mode
            m._refresh_fast_dispatch()
        for mlp, orig_m, se, orig_se in saved_mlp:
            if hasattr(mlp, "use_triton_mxfp4"):
                mlp.use_triton_mxfp4 = orig_m
            if se is not None and orig_se is not None:
                se.use_triton = orig_se


@contextlib.contextmanager
def _capture_layer_mlp_hooks(
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    expected_tokens: int,
):
    """Capture `L{lid}:mlp_out` snapshots cleanly for Case 2 and detach handles in `finally`."""
    layer_snaps: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []
    model = getattr(rtp_model, "model", None)
    sp_size = int(pg.get("sp_size", 1))
    sp_group = pg.get("sp_group")
    try:
        for idx, layer in enumerate(getattr(model, "layers", ())):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            lid = int(getattr(layer, "layer_idx", idx))

            def _make_hook(layer_id: int):
                def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                    t = out[0] if isinstance(out, tuple) else out
                    if not isinstance(t, torch.Tensor):
                        return
                    if sp_size > 1 and sp_group is not None and int(t.shape[0]) < expected_tokens:
                        chunks = [torch.empty_like(t) for _ in range(sp_size)]
                        dist.all_gather(chunks, t.contiguous(), group=sp_group)
                        t = torch.cat(chunks, dim=0)
                    layer_snaps[f"L{layer_id}:mlp_out"] = t[:expected_tokens].detach().cpu()

                return _hook

            handles.append(mlp.register_forward_hook(_make_hook(lid)))
        yield layer_snaps
    finally:
        for h in handles:
            h.remove()


def _run_parity_cases(
    arch: Any,
    kv_cache: LocalKVCache,
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute deterministic 0%-2K, 64-tok boundary + 24 snaps, and 90%-hit 18K+2K passes."""
    local_b = 1
    tpb = kv_cache.tokens_per_block

    # Case 1: 0% Hit T=2048
    _reset_kv_allocator(kv_cache)
    tok_0 = _make_deterministic_tokens([2048], vocab_size=arch.vocab_size, seed=101, device=device)
    tbl_0 = build_block_tables_for_requests(kv_cache, [2048]) if local_b else {t: [] for t in kv_cache.group_tags()}
    h_0, log_0 = _run_prefill_pass(
        rtp_model,
        kv_cache,
        tbl_0,
        tok_0 if local_b else torch.empty((0,), dtype=torch.int64, device=device),
        [2048] if local_b else [],
        [0] if local_b else [],
        device=device,
    )

    # Case 2: Exact 64-token block boundary snapshot + continuation (0..64 -> 64..128)
    _reset_kv_allocator(kv_cache)
    tok_64a = _make_deterministic_tokens([tpb], vocab_size=arch.vocab_size, seed=202, device=device)
    tok_64b = _make_deterministic_tokens([tpb], vocab_size=arch.vocab_size, seed=203, device=device)
    tbl_64 = build_block_tables_for_requests(kv_cache, [2 * tpb]) if local_b else {t: [] for t in kv_cache.group_tags()}
    with _capture_layer_mlp_hooks(rtp_model, pg, tpb) as mlp_snaps:
        _run_prefill_pass(
            rtp_model,
            kv_cache,
            tbl_64,
            tok_64a if local_b else torch.empty((0,), dtype=torch.int64, device=device),
            [tpb] if local_b else [],
            [0] if local_b else [],
            device=device,
            compute_last_logits=False,
        )
    snap_64 = _extract_64tok_snapshots(kv_cache, tbl_64, pg) if local_b else {}
    snap_64.update(mlp_snaps)
    h_64b, log_64b = _run_prefill_pass(
        rtp_model,
        kv_cache,
        tbl_64,
        tok_64b if local_b else torch.empty((0,), dtype=torch.int64, device=device),
        [tpb] if local_b else [],
        [tpb] if local_b else [],
        device=device,
    )

    # Case 3: 90% Cache Hit (prefix=18432 + suffix=2048)
    _reset_kv_allocator(kv_cache)
    tok_pfx = _make_deterministic_tokens([18432], vocab_size=arch.vocab_size, seed=303, device=device)
    tok_sfx = _make_deterministic_tokens([2048], vocab_size=arch.vocab_size, seed=304, device=device)
    tbl_90 = build_block_tables_for_requests(kv_cache, [20480]) if local_b else {t: [] for t in kv_cache.group_tags()}
    off = 0
    for chk in (8192, 8192, 2048):
        _run_prefill_pass(
            rtp_model,
            kv_cache,
            tbl_90,
            tok_pfx[off : off + chk] if local_b else torch.empty((0,), dtype=torch.int64, device=device),
            [chk] if local_b else [],
            [off] if local_b else [],
            device=device,
            compute_last_logits=False,
        )
        off += chk
    h_90, log_90 = _run_prefill_pass(
        rtp_model,
        kv_cache,
        tbl_90,
        tok_sfx if local_b else torch.empty((0,), dtype=torch.int64, device=device),
        [2048] if local_b else [],
        [18432] if local_b else [],
        device=device,
    )

    return {
        "h_0pct": h_0.detach().cpu(),
        "logits_0pct": log_0.detach().cpu() if log_0 is not None else None,
        "h_64b": h_64b.detach().cpu(),
        "logits_64b": log_64b.detach().cpu() if log_64b is not None else None,
        "snap_64": snap_64,
        "h_90pct": h_90.detach().cpu(),
        "logits_90pct": log_90.detach().cpu() if log_90 is not None else None,
    }


def _compare_parity_payload(
    cur: Mapping[str, Any],
    ref: Mapping[str, Any],
    *,
    min_cos: float,
    kv_min_cos: Optional[float] = None,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for name, c_h_key, c_l_key, r_h_key, r_l_key in (
        ("prefill_0pct_2048", "h_0pct", "logits_0pct", "h_0pct", "logits_0pct"),
        ("prefill_64tok_cont", "h_64b", "logits_64b", "h_64b", "logits_64b"),
        ("prefill_90pct_18k_2k", "h_90pct", "logits_90pct", "h_90pct", "logits_90pct"),
    ):
        cur_h = cur[c_h_key]
        rh = ref[r_h_key]
        cos_h = _cosine(cur_h, rh)
        max_h = float((cur_h.float() - rh.float()).abs().max().item())
        top1_ok = True
        cos_l = None
        cur_l = cur.get(c_l_key)
        rl = ref.get(r_l_key)
        if cur_l is not None and rl is not None:
            cos_l = _cosine(cur_l, rl)
            top1_ok = int(cur_l.argmax(dim=-1).item()) == int(rl.argmax(dim=-1).item())
        case_entry: Dict[str, Any] = {
            "hidden_cos": round(cos_h, 6),
            "hidden_max_abs": round(max_h, 5),
            "logits_cos": round(cos_l, 6) if cos_l is not None else None,
            "top1_match": top1_ok,
            "pass": cos_h >= min_cos and top1_ok,
        }
        case_entry.update(_per_tok_cos_stats(cur_h, rh))
        checks[name] = case_entry

    snap_diffs: Dict[str, Any] = {}
    ref_snaps = ref.get("snap_64", {})
    for skey, cur_s in cur.get("snap_64", {}).items():
        ref_s = ref_snaps.get(skey)
        if ref_s is None or cur_s.shape != ref_s.shape:
            continue
        is_kv_buf = skey.endswith(":full") or skey.endswith(":indexer_state")
        eff_thr = kv_min_cos if (kv_min_cos is not None and is_kv_buf) else min_cos
        if skey.endswith(":indexer_kv") and cur_s.dtype == torch.uint8 and ref_s.dtype == torch.uint8:
            scos = _cosine(cur_s.contiguous().view(torch.bfloat16), ref_s.contiguous().view(torch.bfloat16))
            snap_diffs[skey] = {"cos": round(scos, 6), "threshold": eff_thr, "pass": scos >= eff_thr}
        elif cur_s.dtype in (torch.int32, torch.int64, torch.uint8):
            eq = bool(torch.equal(cur_s, ref_s))
            snap_diffs[skey] = {"exact_equal": eq, "pass": eq}
        else:
            scos = _cosine(cur_s, ref_s)
            snap_diffs[skey] = {"cos": round(scos, 6), "threshold": eff_thr, "pass": scos >= eff_thr}

    all_ok = all(v["pass"] for v in checks.values()) and all(v["pass"] for v in snap_diffs.values())
    return {
        "all_pass": all_ok,
        "min_cos": min_cos,
        "kv_min_cos": kv_min_cos if kv_min_cos is not None else min_cos,
        "num_snapshots": len(snap_diffs),
        "cases": checks,
        "snapshots": snap_diffs,
    }


def run_parity_suite(
    arch: Any,
    kv_cache: LocalKVCache,
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    *,
    device: torch.device,
    save_ref_path: Optional[str] = None,
    verify_ref_path: Optional[str] = None,
    min_cos: float = 0.999,
    kv_min_cos: Optional[float] = None,
    stabilize_router: bool = False,
    verify_inproc_ref: bool = False,
    strict_raw_parity: bool = False,
) -> Dict[str, Any]:
    """Run BF16/candidate parity with 24+ layer/KV snapshots, stabilized-router & in-proc ref gates."""
    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    res: Dict[str, Any] = {
        "saved": None,
        "verified": None,
        "stabilized_verified": None,
        "inproc_ref_verified": None,
        "raw_diagnostic": None,
    }

    cur_raw: Optional[Dict[str, Any]] = None
    cur_stab: Optional[Dict[str, Any]] = None

    if save_ref_path:
        cur_raw = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
        with _temporary_stabilized_router(rtp_model, pg):
            cur_stab = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
        if is_rank0:
            payload = dict(cur_raw)
            payload["stabilized"] = dict(cur_stab)
            payload["is_stabilized_bundle"] = True
            p = Path(save_ref_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, p)
            res["saved"] = str(p)

    if verify_ref_path or verify_inproc_ref:
        ref = torch.load(verify_ref_path, map_location="cpu", weights_only=False) if verify_ref_path else None
        raw_diag: Optional[Dict[str, Any]] = None
        stab_cmp: Optional[Dict[str, Any]] = None
        inproc_cmp: Optional[Dict[str, Any]] = None
        has_stab_ref = False

        if ref is not None:
            if cur_raw is None:
                cur_raw = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
            if is_rank0 and "h_0pct" in ref:
                raw_diag = _compare_parity_payload(cur_raw, ref, min_cos=min_cos, kv_min_cos=kv_min_cos)
                res["raw_diagnostic"] = raw_diag

            stab_ref: Optional[Mapping[str, Any]] = None
            if isinstance(ref.get("stabilized"), Mapping):
                stab_ref = ref["stabilized"]
                has_stab_ref = True
            elif stabilize_router or bool(ref.get("is_stabilized", False)):
                stab_ref = ref
                has_stab_ref = True

            if stab_ref is not None:
                if cur_stab is None:
                    with _temporary_stabilized_router(rtp_model, pg):
                        cur_stab = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
                if is_rank0:
                    stab_cmp = _compare_parity_payload(cur_stab, stab_ref, min_cos=min_cos, kv_min_cos=kv_min_cos)
                    res["stabilized_verified"] = stab_cmp

        need_inproc = bool(verify_inproc_ref) or (ref is not None and not has_stab_ref)
        if need_inproc:
            if ref is not None and not has_stab_ref:
                res["legacy_raw_ref_fallback_inproc"] = True
            with _temporary_stabilized_router(rtp_model, pg):
                if cur_stab is None:
                    cur_stab = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
                with _temporary_mxfp4_ref_backend(rtp_model):
                    inproc_ref = _run_parity_cases(arch, kv_cache, rtp_model, pg, device=device)
            if is_rank0:
                inproc_cmp = _compare_parity_payload(cur_stab, inproc_ref, min_cos=min_cos, kv_min_cos=kv_min_cos)
                res["inproc_ref_verified"] = inproc_cmp

        if is_rank0:
            gate_checks: List[Tuple[str, Dict[str, Any]]] = []
            if stab_cmp is not None:
                gate_checks.append(("stabilized_ref", stab_cmp))
            if inproc_cmp is not None:
                gate_checks.append(("inproc_ref", inproc_cmp))
            if strict_raw_parity and raw_diag is not None:
                gate_checks.append(("raw_strict", raw_diag))
            all_ok = bool(gate_checks) and all(c["all_pass"] for _, c in gate_checks)
            summary: Dict[str, Any] = {
                "all_pass": all_ok,
                "min_cos": min_cos,
                "kv_min_cos": kv_min_cos if kv_min_cos is not None else min_cos,
                "active_gates": [k for k, _ in gate_checks],
                "num_snapshots": max((c["num_snapshots"] for _, c in gate_checks), default=0),
            }
            primary = stab_cmp if stab_cmp is not None else inproc_cmp
            if primary is not None:
                summary["cases"] = primary["cases"]
                summary["snapshots"] = primary["snapshots"]
            res["verified"] = summary
            if not all_ok:
                fail_payload = {
                    "summary": summary,
                    "stabilized_verified": stab_cmp,
                    "inproc_ref_verified": inproc_cmp,
                    "raw_diagnostic": raw_diag,
                }
                raise AssertionError(f"Parity gate FAILED: {json.dumps(fail_payload, indent=2)}")
    return res


# ---------------------------------------------------------------------------
# Benchmark Runners: 0%-Hit Budget Sweep & 90%-Hit Batch Sweep
# ---------------------------------------------------------------------------


def run_prefill_0pct_sweeps(
    arch: Any,
    kv_cache: LocalKVCache,
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    roofline: HardwareRoofline,
    *,
    dtype_mode: str,
    kv_dtype: str = "bf16",
    budgets: Sequence[int],
    total_prompt_len: int,
    warmup: int,
    iters: int,
    device: torch.device,
    kineto_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run 0%-cache-hit `T=20,480` chunked budget sweep and single-step packed token sweep."""
    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    dp_size = int(pg["dp_size"])
    b_chunk = dp_size
    local_b = 1
    toks_20k = _make_deterministic_tokens(
        [total_prompt_len], vocab_size=arch.vocab_size, seed=777 + int(pg["dp_rank"]), device=device
    )

    # 1. Chunked 20,480 TTFT across scheduled token budgets (4K, 8K, 16K, 32K cluster tokens)
    chunked_rows: List[Dict[str, Any]] = []
    best_budget = None
    best_ttft_ms = float("inf")
    for budget in budgets:
        _reset_kv_allocator(kv_cache)
        tbl = build_block_tables_for_requests(kv_cache, [total_prompt_len])

        # Build chunk schedule (divide cluster budget across DP ranks when dp_size > 1)
        per_rank_budget = max(512, budget // dp_size)
        chunks: List[Tuple[int, int]] = []
        off = 0
        while off < total_prompt_len:
            clen = min(per_rank_budget, total_prompt_len - off)
            chunks.append((off, clen))
            off += clen

        def _run_chunked() -> None:
            for idx, (pfx, clen) in enumerate(chunks):
                _run_prefill_pass(
                    rtp_model,
                    kv_cache,
                    tbl,
                    toks_20k[pfx : pfx + clen],
                    [clen],
                    [pfx],
                    device=device,
                    compute_last_logits=(idx == len(chunks) - 1),
                )

        try:
            torch.cuda.reset_peak_memory_stats(device)
            mean_ms, p50_ms = _timed_step_ms(_run_chunked, device=device, warmup=warmup, iters=iters)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            wl = compute_qwen45_workload_metrics(
                arch,
                batch_size=b_chunk,
                seq_len_per_step=total_prompt_len,
                avg_ctx=total_prompt_len,
                parallel_mode=pg["parallel_mode"],
                dtype_mode=dtype_mode,
                kv_dtype=kv_dtype,
                world_size=roofline.world_size,
                is_prefill=True,
                prefix_len=0,
                roofline=roofline,
                elapsed_ms=mean_ms,
            ).to_dict()
            row = {
                "budget": budget,
                "batch_size": b_chunk,
                "num_chunks": len(chunks),
                "chunk_lens": [c[1] for c in chunks],
                "mean_ttft_ms": round(mean_ms, 3),
                "p50_ttft_ms": round(p50_ms, 3),
                "prefill_tok_s": wl.get("tokens_per_s"),
                "mfu_bf16_pct": wl.get("mfu_bf16_pct"),
                "mfu_aware_pct": wl.get("mfu_aware_pct"),
                "mbu_pct": wl.get("mbu_pct"),
                "sol_efficiency_pct": wl.get("sol_efficiency_pct"),
                "peak_alloc_gib": round(peak_gib, 2),
                "workload": wl,
            }
            if mean_ms < best_ttft_ms:
                best_ttft_ms = mean_ms
                best_budget = budget
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            row = {"budget": budget, "oom": str(exc)}
        chunked_rows.append(row)
        if is_rank0:
            print(f"[0%-HIT CHUNKED {b_chunk}x{total_prompt_len} M_budget={budget}] {row.get('mean_ttft_ms', 'OOM')}ms | tok/s={row.get('prefill_tok_s')} | MFU_bf16={row.get('mfu_bf16_pct')}% MBU={row.get('mbu_pct')}%", flush=True)

    # 2. Single-step packed token budget sweep (M = 4096, 8192, 16384, 20480, 32768)
    packed_rows: List[Dict[str, Any]] = []
    kineto_0pct: Dict[str, Any] = {}
    for m_step in budgets:
        if dp_size > 1:
            b_step = dp_size
        else:
            b_step = 2 if m_step > total_prompt_len else 1
        t_step = m_step // b_step
        lb, _ = _local_batch_for_rank(b_step, pg)
        _reset_kv_allocator(kv_cache)
        tbl = build_block_tables_for_requests(kv_cache, [t_step] * lb)
        tok_s = _make_deterministic_tokens([t_step] * lb, vocab_size=arch.vocab_size, seed=888 + m_step + int(pg["dp_rank"]), device=device)

        def _run_packed() -> None:
            _run_prefill_pass(
                rtp_model,
                kv_cache,
                tbl,
                tok_s,
                [t_step] * lb if lb else [],
                [0] * lb if lb else [],
                device=device,
            )

        try:
            torch.cuda.reset_peak_memory_stats(device)
            mean_ms, p50_ms = _timed_step_ms(_run_packed, device=device, warmup=warmup, iters=iters)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            k_res: Dict[str, Any] = {}
            if m_step == min(total_prompt_len, max(b for b in budgets if b <= total_prompt_len) if any(b <= total_prompt_len for b in budgets) else total_prompt_len) or m_step == best_budget:
                tr = (kineto_dir / f"r0_prefill_0pct_m{m_step}.json") if kineto_dir else None
                k_res = capture_prefill_kineto(_run_packed, trace_path=tr, tp_size=pg["tp_size"], device=device, rtp_model=rtp_model)
                if k_res:
                    kineto_0pct[f"m{m_step}"] = k_res

            op_times = {k: v["ms"] for k, v in k_res.get("buckets", {}).items()} if k_res else None
            wl = compute_qwen45_workload_metrics(
                arch,
                batch_size=b_step,
                seq_len_per_step=t_step,
                avg_ctx=t_step,
                parallel_mode=pg["parallel_mode"],
                dtype_mode=dtype_mode,
                kv_dtype=kv_dtype,
                world_size=roofline.world_size,
                is_prefill=True,
                prefix_len=0,
                roofline=roofline,
                elapsed_ms=mean_ms,
                op_times_ms=op_times,
            ).to_dict()
            prow = {
                "m_tokens": m_step,
                "batch_size": b_step,
                "seq_len": t_step,
                "mean_ms": round(mean_ms, 3),
                "p50_ms": round(p50_ms, 3),
                "tok_s": wl.get("tokens_per_s"),
                "mfu_bf16_pct": wl.get("mfu_bf16_pct"),
                "mfu_aware_pct": wl.get("mfu_aware_pct"),
                "mbu_pct": wl.get("mbu_pct"),
                "sol_efficiency_pct": wl.get("sol_efficiency_pct"),
                "peak_alloc_gib": round(peak_gib, 2),
                "workload": wl,
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            prow = {"m_tokens": m_step, "oom": str(exc)}
        packed_rows.append(prow)
        if is_rank0:
            print(f"[0%-HIT PACKED M={m_step} ({b_step}x{t_step})] {prow.get('mean_ms', 'OOM')}ms | tok/s={prow.get('tok_s')} | MFU_bf16={prow.get('mfu_bf16_pct')}% MBU={prow.get('mbu_pct')}%", flush=True)

    return {
        "optimal_budget_tokens": best_budget,
        "chunked_20k": chunked_rows,
        "packed_budget_sweep": packed_rows,
        "kineto_0pct": kineto_0pct,
    }


def run_prefill_90pct_sweep(
    arch: Any,
    kv_cache: LocalKVCache,
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    roofline: HardwareRoofline,
    *,
    dtype_mode: str,
    kv_dtype: str = "bf16",
    prefix_len: int,
    suffix_len: int,
    batches: Sequence[int],
    warmup: int,
    iters: int,
    device: torch.device,
    kineto_dir: Optional[Path],
) -> Dict[str, Any]:
    """Populate genuine `prefix_len` cache blocks once, then benchmark `suffix_len x B` sweeps."""
    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    max_b = max(batches)
    max_local_b, _ = _local_batch_for_rank(max_b, pg)
    total_seq = prefix_len + suffix_len

    # Allocate max_local_b independent sequences so there is zero cross-request KV aliasing
    _reset_kv_allocator(kv_cache)
    master_tbl = (
        build_block_tables_for_requests(kv_cache, [total_seq] * max_local_b)
        if max_local_b > 0
        else {t: [] for t in kv_cache.group_tags()}
    )

    # Populate prefix_len (e.g. 18,432 tokens) once per request in clean 8192-tok chunks
    if max_local_b > 0:
        if is_rank0:
            print(f"[90%-HIT SETUP] Populating {max_local_b}x {prefix_len}-tok independent prefix caches...", flush=True)
        pfx_toks_all = [
            _make_deterministic_tokens([prefix_len], vocab_size=arch.vocab_size, seed=1000 + r, device=device)
            for r in range(max_local_b)
        ]
        off = 0
        while off < prefix_len:
            chk = min(8192, prefix_len - off)
            # Run sub-batches of <= 4 requests at a time so setup memory stays low
            for b0 in range(0, max_local_b, 4):
                b1 = min(max_local_b, b0 + 4)
                sub_tbl = {t: rows[b0:b1] for t, rows in master_tbl.items()}
                sub_ids = torch.cat([pfx_toks_all[r][off : off + chk] for r in range(b0, b1)], dim=0)
                _run_prefill_pass(
                    rtp_model,
                    kv_cache,
                    sub_tbl,
                    sub_ids,
                    [chk] * (b1 - b0),
                    [off] * (b1 - b0),
                    device=device,
                    compute_last_logits=False,
                )
            off += chk
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    rows: List[Dict[str, Any]] = []
    kineto_90pct: Dict[str, Any] = {}
    dp_size = int(pg["dp_size"])
    for b_global in batches:
        if dp_size > 1 and (b_global < dp_size or b_global % dp_size != 0):
            srow = {"batch_size": b_global, "skipped": f"requires batch_size divisible by dp_size={dp_size}"}
            rows.append(srow)
            if is_rank0:
                print(f"[90%-HIT {prefix_len}+{suffix_len} B={b_global}] SKIPPED ({srow['skipped']})", flush=True)
            continue
        lb, start_idx = _local_batch_for_rank(b_global, pg)
        sub_tbl = {t: rows[:lb] for t, rows in master_tbl.items()}
        sfx_ids = _make_deterministic_tokens(
            [suffix_len] * lb, vocab_size=arch.vocab_size, seed=2000 + start_idx + b_global, device=device
        )

        def _run_suffix() -> None:
            _run_prefill_pass(
                rtp_model,
                kv_cache,
                sub_tbl,
                sfx_ids,
                [suffix_len] * lb if lb else [],
                [prefix_len] * lb if lb else [],
                device=device,
            )

        try:
            torch.cuda.reset_peak_memory_stats(device)
            mean_ms, p50_ms = _timed_step_ms(_run_suffix, device=device, warmup=warmup, iters=iters)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            k_res: Dict[str, Any] = {}
            if b_global in (4, 16) or b_global == batches[-1]:
                tr = (kineto_dir / f"r0_prefill_90pct_b{b_global}.json") if kineto_dir else None
                k_res = capture_prefill_kineto(_run_suffix, trace_path=tr, tp_size=pg["tp_size"], device=device, rtp_model=rtp_model)
                if k_res:
                    kineto_90pct[f"b{b_global}"] = k_res

            op_times = {k: v["ms"] for k, v in k_res.get("buckets", {}).items()} if k_res else None
            wl = compute_qwen45_workload_metrics(
                arch,
                batch_size=b_global,
                seq_len_per_step=suffix_len,
                avg_ctx=total_seq,
                parallel_mode=pg["parallel_mode"],
                dtype_mode=dtype_mode,
                kv_dtype=kv_dtype,
                world_size=roofline.world_size,
                is_prefill=True,
                prefix_len=prefix_len,
                roofline=roofline,
                elapsed_ms=mean_ms,
                op_times_ms=op_times,
            ).to_dict()
            row = {
                "batch_size": b_global,
                "prefix_len": prefix_len,
                "suffix_len": suffix_len,
                "total_new_tokens": b_global * suffix_len,
                "mean_ms": round(mean_ms, 3),
                "p50_ms": round(p50_ms, 3),
                "suffix_tok_s": wl.get("tokens_per_s"),
                "effective_ctx_tok_s": round((b_global * total_seq) / (mean_ms * 1e-3), 2),
                "mfu_bf16_pct": wl.get("mfu_bf16_pct"),
                "mfu_aware_pct": wl.get("mfu_aware_pct"),
                "mbu_pct": wl.get("mbu_pct"),
                "sol_efficiency_pct": wl.get("sol_efficiency_pct"),
                "peak_alloc_gib": round(peak_gib, 2),
                "workload": wl,
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            row = {"batch_size": b_global, "oom": str(exc)}
        rows.append(row)
        if is_rank0:
            print(
                f"[90%-HIT {prefix_len}+{suffix_len} B={b_global}] "
                f"{row.get('mean_ms', 'OOM')}ms | sfx_tok/s={row.get('suffix_tok_s')} | "
                f"MFU_bf16={row.get('mfu_bf16_pct')}% MBU={row.get('mbu_pct')}%",
                flush=True,
            )

    return {"rows": rows, "kineto_90pct": kineto_90pct}


def run_tbo_prefill_benchmark(
    arch: Any,
    rtp_model: Qwen4_5RTPModel,
    pg: Mapping[str, Any],
    hr: HardwareRoofline,
    *,
    dtype_mode: str,
    kv_dtype: str,
    seq_len: int = 2048,
    global_batches: Sequence[int] = (8, 16),
    warmup: int = 2,
    iters: int = 6,
    device: torch.device,
    kineto_dir: Optional[Path] = None,
    min_cos: float = 0.999,
) -> Dict[str, Any]:
    """Benchmark Qwen4_5Model Two-Batch Overlap (enable_tbo=False vs True) sharing loaded weights."""
    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    core = getattr(rtp_model, "model", None)
    if core is None or not hasattr(core, "forward_tbo"):
        return {
            "paged_runtime_tbo_supported": False,
            "model_forward_tbo_supported": False,
            "rows": [],
        }

    dp_size = max(1, int(pg.get("dp_size", 1)))
    dp_rank = int(pg.get("dp_rank", 0))
    rows: List[Dict[str, Any]] = []

    def _reset_rt_state(st: Any) -> None:
        if st is None:
            return
        st.seq_len = 0
        for kv in getattr(st, "full_attn_kv", {}).values():
            kv.seq_len = 0
        for g in getattr(st, "gdn", {}).values():
            if getattr(g, "ssm_state", None) is not None:
                g.ssm_state.zero_()
            if getattr(g, "conv_state", None) is not None:
                g.conv_state.zero_()
        for attr in ("k_sconv", "v_sconv", "attn_output_sconv", "ffn_output_sconv"):
            for sc in getattr(st, attr, {}).values():
                if getattr(sc, "conv_state", None) is not None:
                    sc.conv_state.zero_()
        for p in getattr(st, "ple", {}).values():
            if getattr(p, "conv_state", None) is not None:
                p.conv_state.zero_()
            if getattr(p, "ngram_ctx", None) is not None:
                p.ngram_ctx.zero_()
        rk = getattr(st, "qsa_raw_keys", None)
        if isinstance(rk, dict):
            for lid in list(rk.keys()):
                rk[lid] = rk[lid][:, :0]

    # Stabilized-router numerical parity between TBO Off and TBO On before timing.
    # Under W4A8, B=2 vs B=1 crosses MoE/Triton M-tiling boundaries (even TBO-off B=2[:1] vs B=1
    # differs at ~0.992), so we verify BOTH active same-shape microbatch parity (tbo0 vs non_tbo_b1_0,
    # tbo1 vs non_tbo_b1_1) AND full-batch B=2 parity under _temporary_mxfp4_ref_backend.
    probe_seq = min(int(seq_len), 128)
    probe_g = torch.Generator(device=device).manual_seed(9100 + dp_rank)
    probe_ids = torch.randint(0, arch.vocab_size, (2, probe_seq), generator=probe_g, dtype=torch.int64, device=device)
    has_state_api = hasattr(core, "create_runtime_state")
    def _mk_state(bs: int):
        return core.create_runtime_state(bs, probe_seq + 8, device=device, dtype=torch.bfloat16) if has_state_api else None
    with _temporary_stabilized_router(rtp_model, pg):
        with torch.no_grad():
            st_p_off, st_p0, st_p1 = _mk_state(2), _mk_state(1), _mk_state(1)
            st_s0, st_s1 = _mk_state(1), _mk_state(1)
            h_off = core(probe_ids, state=st_p_off, enable_tbo=False)
            h_s0 = core(probe_ids[:1], state=st_s0, enable_tbo=False)
            h_s1 = core(probe_ids[1:], state=st_s1, enable_tbo=False)
            if has_state_api:
                h0_p, h1_p = core.forward_tbo(
                    (probe_ids[:1], probe_ids[1:]),
                    states=(st_p0, st_p1),
                    equal_tokens=True,
                )
                h_on = torch.cat([h0_p, h1_p], dim=0)
            else:
                h_on = core(probe_ids, enable_tbo=True, tbo_equal_tokens=True)
                h0_p, h1_p = h_on[:1], h_on[1:]
            del st_p_off, st_p0, st_p1, st_s0, st_s1
        with _temporary_mxfp4_ref_backend(rtp_model):
            with torch.no_grad():
                st_r_off, st_r0, st_r1 = _mk_state(2), _mk_state(1), _mk_state(1)
                h_off_ref = core(probe_ids, state=st_r_off, enable_tbo=False)
                if has_state_api:
                    h0_r, h1_r = core.forward_tbo(
                        (probe_ids[:1], probe_ids[1:]),
                        states=(st_r0, st_r1),
                        equal_tokens=True,
                    )
                    h_on_ref = torch.cat([h0_r, h1_r], dim=0)
                else:
                    h_on_ref = core(probe_ids, enable_tbo=True, tbo_equal_tokens=True)
                del st_r_off, st_r0, st_r1
    if is_rank0:
        tbo_mb0_cos = _cosine(h0_p.detach().float().cpu(), h_s0.detach().float().cpu())
        tbo_mb1_cos = _cosine(h1_p.detach().float().cpu(), h_s1.detach().float().cpu())
        tbo_full_active_cos = _cosine(h_on.detach().float().cpu(), h_off.detach().float().cpu())
        tbo_full_ref_cos = _cosine(h_on_ref.detach().float().cpu(), h_off_ref.detach().float().cpu())
        tbo_cos = min(tbo_mb0_cos, tbo_mb1_cos, tbo_full_ref_cos)
        print(
            f"[tbo_parity] mb0_cos={tbo_mb0_cos:.6f} mb1_cos={tbo_mb1_cos:.6f} "
            f"full_active_cos={tbo_full_active_cos:.6f} full_ref_cos={tbo_full_ref_cos:.6f} "
            f"min_gate={tbo_cos:.6f}",
            flush=True,
        )
    else:
        tbo_cos = 1.0
    if is_rank0 and tbo_cos < min_cos:
        raise AssertionError(f"TBO parity FAILED: cos={tbo_cos:.6f} < {min_cos}")

    valid_batches = [
        b for b in global_batches
        if b % dp_size == 0 and (b // dp_size) >= 2 and (b // dp_size) % 2 == 0
    ]
    for b_global in valid_batches:
        local_b = b_global // dp_size
        b0 = local_b // 2
        g = torch.Generator(device=device).manual_seed(9200 + dp_rank + b_global)
        ids = torch.randint(0, arch.vocab_size, (local_b, seq_len), generator=g, dtype=torch.int64, device=device)
        mb0, mb1 = ids[:b0], ids[b0:]
        st_off = core.create_runtime_state(local_b, seq_len + 8, device=device, dtype=torch.bfloat16) if has_state_api else None
        st0 = core.create_runtime_state(b0, seq_len + 8, device=device, dtype=torch.bfloat16) if has_state_api else None
        st1 = core.create_runtime_state(local_b - b0, seq_len + 8, device=device, dtype=torch.bfloat16) if has_state_api else None

        def _run_tbo_off() -> torch.Tensor:
            return core(ids, state=st_off, enable_tbo=False)

        def _run_tbo_on() -> Tuple[torch.Tensor, torch.Tensor]:
            if has_state_api:
                return core.forward_tbo((mb0, mb1), states=(st0, st1), equal_tokens=True)
            return core(ids, enable_tbo=True, tbo_equal_tokens=True)

        try:
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                off_mean, off_p50 = _timed_step_ms(
                    _run_tbo_off,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                    pre_iter_fn=(lambda: _reset_rt_state(st_off)) if has_state_api else None,
                )
                on_mean, on_p50 = _timed_step_ms(
                    _run_tbo_on,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                    pre_iter_fn=(lambda: (_reset_rt_state(st0), _reset_rt_state(st1))) if has_state_api else None,
                )
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            k_off: Dict[str, Any] = {}
            k_on: Dict[str, Any] = {}
            if kineto_dir is not None and b_global == valid_batches[-1]:
                with torch.no_grad():
                    def _k_off_fn():
                        _reset_rt_state(st_off)
                        return _run_tbo_off()

                    def _k_on_fn():
                        _reset_rt_state(st0)
                        _reset_rt_state(st1)
                        return _run_tbo_on()

                    k_off = capture_prefill_kineto(
                        _k_off_fn,
                        trace_path=kineto_dir / f"prefill_tbo_off_{pg['parallel_mode']}_b{b_global}.json",
                        tp_size=int(pg["tp_size"]),
                        device=device,
                        rtp_model=rtp_model,
                    )
                    k_on = capture_prefill_kineto(
                        _k_on_fn,
                        trace_path=kineto_dir / f"prefill_tbo_on_{pg['parallel_mode']}_b{b_global}.json",
                        tp_size=int(pg["tp_size"]),
                        device=device,
                        rtp_model=rtp_model,
                    )
            op_off = {k: v["ms"] for k, v in k_off.get("buckets", {}).items()} if k_off else None
            op_on = {k: v["ms"] for k, v in k_on.get("buckets", {}).items()} if k_on else None
            wl_off = compute_qwen45_workload_metrics(
                arch,
                batch_size=b_global,
                seq_len_per_step=seq_len,
                avg_ctx=seq_len,
                parallel_mode=pg["parallel_mode"],
                dtype_mode=dtype_mode,
                kv_dtype=kv_dtype,
                world_size=hr.world_size,
                is_prefill=True,
                prefix_len=0,
                roofline=hr,
                elapsed_ms=off_mean,
                op_times_ms=op_off,
            ).to_dict()
            wl_on = compute_qwen45_workload_metrics(
                arch,
                batch_size=b_global,
                seq_len_per_step=seq_len,
                avg_ctx=seq_len,
                parallel_mode=pg["parallel_mode"],
                dtype_mode=dtype_mode,
                kv_dtype=kv_dtype,
                world_size=hr.world_size,
                is_prefill=True,
                prefix_len=0,
                roofline=hr,
                elapsed_ms=on_mean,
                op_times_ms=op_on,
            ).to_dict()
            tot_tok = b_global * seq_len
            row: Dict[str, Any] = {
                "batch_size": b_global,
                "local_batch": local_b,
                "seq_len": seq_len,
                "total_tokens": tot_tok,
                "tbo_off_mean_ms": round(off_mean, 3),
                "tbo_off_p50_ms": round(off_p50, 3),
                "tbo_off_tok_s": round(tot_tok / (off_mean * 1e-3), 2),
                "tbo_on_mean_ms": round(on_mean, 3),
                "tbo_on_p50_ms": round(on_p50, 3),
                "tbo_on_tok_s": round(tot_tok / (on_mean * 1e-3), 2),
                "speedup_mean": round(off_mean / max(on_mean, 1e-6), 4),
                "speedup_p50": round(off_p50 / max(on_p50, 1e-6), 4),
                "mfu_bf16_off_pct": wl_off.get("mfu_bf16_pct"),
                "mfu_bf16_on_pct": wl_on.get("mfu_bf16_pct"),
                "mbu_off_pct": wl_off.get("mbu_pct"),
                "mbu_on_pct": wl_on.get("mbu_pct"),
                "peak_alloc_gib": round(peak_gib, 2),
                "workload_off": wl_off,
                "workload_on": wl_on,
            }
            if k_off:
                row["kineto_tbo_off"] = k_off
            if k_on:
                row["kineto_tbo_on"] = k_on
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            row = {"batch_size": b_global, "oom": str(exc)}
        finally:
            del st_off, st0, st1
            torch.cuda.empty_cache()
        rows.append(row)
        if is_rank0:
            print(
                f"[TBO B={b_global} T={seq_len}] "
                f"off={row.get('tbo_off_mean_ms', 'OOM')}ms -> on={row.get('tbo_on_mean_ms', 'OOM')}ms "
                f"({row.get('speedup_mean', '-')}x)",
                flush=True,
            )

    return {
        "paged_runtime_tbo_supported": False,
        "model_forward_tbo_supported": True,
        "tbo_stabilized_cos": round(tbo_cos, 6),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone 4-GPU Prefill Roofline Benchmark for Qwen4.5-Flash 5L")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--parallel-mode", default="tp4_ep4", choices=tuple(PARALLEL_MODES))
    parser.add_argument("--dtype-mode", default="mxfp4", choices=tuple(DTYPE_PRESETS))
    parser.add_argument(
        "--kv-dtype",
        default="BF16",
        choices=("BF16", "FP8", "F8_E4M3", "FP8_E4M3", "bf16", "fp8_e4m3", "fp8"),
    )
    parser.add_argument(
        "--ple-mode",
        default="gpu_sharded",
        choices=("auto", "gpu_sharded", "cpu_uva", "shm", "mmap"),
    )
    parser.add_argument(
        "--dense-linear-mode",
        default="",
        choices=("", "auto", "bf16", "scaled_mm", "fp8_mxfp4"),
    )
    parser.add_argument("--fp8-indexer-kv", default="auto", choices=("auto", "on", "off"))
    parser.add_argument("--roofline-json", default="/home/admin/lhc_qwen45_opt/logs/l20a_roofline.json")
    parser.add_argument("--calibrate-if-missing", action="store_true", default=True)
    parser.add_argument("--prompt-len-0pct", type=int, default=20480)
    parser.add_argument("--budgets-0pct", default="4096,8192,16384,20480,32768")
    parser.add_argument("--prefix-len-90pct", type=int, default=18432)
    parser.add_argument("--suffix-len-90pct", type=int, default=2048)
    parser.add_argument("--batches-90pct", default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--kineto-dir", default="/home/admin/lhc_qwen45_opt/traces")
    parser.add_argument("--save-ref", default="")
    parser.add_argument("--save-stabilized-ref", default="")
    parser.add_argument("--verify-parity", default="")
    parser.add_argument("--verify-stabilized-ref", default="")
    parser.add_argument("--stabilize-router-parity", action="store_true", default=False)
    parser.add_argument("--verify-inproc-ref", action="store_true", default=False)
    parser.add_argument("--strict-raw-parity", action="store_true", default=False)
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--kv-min-cos", type=float, default=None)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--skip-0pct", action="store_true")
    parser.add_argument("--skip-90pct", action="store_true")
    parser.add_argument("--benchmark-tbo", action="store_true", default=False)
    parser.add_argument("--tbo-seq-len", type=int, default=2048)
    parser.add_argument("--tbo-batches", default="8,16")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    pg = _build_parallel_groups(args.parallel_mode, rank, world_size)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if Path(args.roofline_json).is_file():
        hr = HardwareRoofline.load_json(args.roofline_json)
    elif args.calibrate_if_missing:
        hr = calibrate_l20a_roofline(device=device, json_path=args.roofline_json)
    else:
        raise FileNotFoundError(f"Missing --roofline-json: {args.roofline_json}")

    batches_90 = _parse_int_list(args.batches_90pct)
    budgets_0 = _parse_int_list(args.budgets_0pct)
    max_90_local, _ = _local_batch_for_rank(max(batches_90 or [1]), pg)
    max_kv_tokens = max(
        args.prompt_len_0pct,
        max(budgets_0 or [20480]),
        max_90_local * (args.prefix_len_90pct + args.suffix_len_90pct),
    )
    kv_low = str(args.kv_dtype).strip().lower()
    is_fp8_kv = kv_low.startswith("fp8") or kv_low == "f8_e4m3"
    kv_dt = "fp8_e4m3" if is_fp8_kv else "BF16"
    fp8_idx_arg: Optional[bool] = None
    if args.fp8_indexer_kv == "on":
        fp8_idx_arg = True
    elif args.fp8_indexer_kv == "off":
        fp8_idx_arg = False

    arch, kv_cache, rtp_model = build_prefill_model_and_kv(
        args.ckpt_path,
        pg=pg,
        dtype_mode=args.dtype_mode,
        kv_dtype=kv_dt,
        device=device,
        max_kv_tokens=max_kv_tokens,
        ple_mode=args.ple_mode,
        dense_linear_mode=args.dense_linear_mode or None,
        fp8_indexer_kv=fp8_idx_arg,
    )

    save_ref = args.save_ref or args.save_stabilized_ref or None
    verify_ref = args.verify_parity or args.verify_stabilized_ref or None
    use_stab = bool(args.stabilize_router_parity or args.verify_stabilized_ref or args.save_stabilized_ref)

    parity_res: Dict[str, Any] = {}
    if save_ref or verify_ref or args.verify_inproc_ref:
        parity_res = run_parity_suite(
            arch,
            kv_cache,
            rtp_model,
            pg,
            device=device,
            save_ref_path=save_ref,
            verify_ref_path=verify_ref,
            min_cos=args.min_cos,
            kv_min_cos=args.kv_min_cos,
            stabilize_router=use_stab,
            verify_inproc_ref=args.verify_inproc_ref,
            strict_raw_parity=args.strict_raw_parity,
        )

    k_dir = Path(args.kineto_dir) if args.kineto_dir else None
    sweep_0: Dict[str, Any] = {}
    if not args.skip_0pct and budgets_0:
        sweep_0 = run_prefill_0pct_sweeps(
            arch,
            kv_cache,
            rtp_model,
            pg,
            hr,
            dtype_mode=args.dtype_mode,
            kv_dtype=kv_dt,
            budgets=budgets_0,
            total_prompt_len=args.prompt_len_0pct,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            kineto_dir=k_dir,
        )

    sweep_90: Dict[str, Any] = {}
    if not args.skip_90pct and batches_90:
        sweep_90 = run_prefill_90pct_sweep(
            arch,
            kv_cache,
            rtp_model,
            pg,
            hr,
            dtype_mode=args.dtype_mode,
            kv_dtype=kv_dt,
            prefix_len=args.prefix_len_90pct,
            suffix_len=args.suffix_len_90pct,
            batches=batches_90,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            kineto_dir=k_dir,
        )

    tbo_res: Dict[str, Any] = {}
    tbo_batches = _parse_int_list(args.tbo_batches)
    if args.benchmark_tbo and tbo_batches:
        tbo_res = run_tbo_prefill_benchmark(
            arch,
            rtp_model,
            pg,
            hr,
            dtype_mode=args.dtype_mode,
            kv_dtype=kv_dt,
            seq_len=args.tbo_seq_len,
            global_batches=tbo_batches,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            kineto_dir=k_dir,
            min_cos=args.min_cos,
        )

    if rank == 0:
        report = {
            "parallel_mode": pg["parallel_mode"],
            "dtype_mode": args.dtype_mode,
            "kv_dtype": kv_dt,
            "ple_mode": args.ple_mode,
            "dense_linear_mode": args.dense_linear_mode or "default",
            "fp8_indexer_kv": args.fp8_indexer_kv,
            "world_size": world_size,
            "hardware_roofline": hr.to_dict(),
            "parity": parity_res,
            "prefill_0pct": sweep_0,
            "prefill_90pct": sweep_90,
            "tbo_probe": tbo_res,
        }
        if args.out_json:
            op = Path(args.out_json)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(report, indent=2))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
