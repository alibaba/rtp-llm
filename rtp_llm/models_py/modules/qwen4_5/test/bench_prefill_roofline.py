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
) -> Tuple[Any, LocalKVCache, Qwen4_5RTPModel]:
    """Load 5L full-width QSA checkpoint across 4 GPUs and wrap in Qwen4_5RTPModel."""
    cfg_raw = json.loads((Path(ckpt_path) / "config.json").read_text(encoding="utf-8"))
    arch = parse_qwen4_5_config_json(cfg_raw, strict=False)
    dm = dtype_mode.lower()
    if dm not in DTYPE_PRESETS:
        raise ValueError(f"Unsupported dtype_mode={dtype_mode!r}; expected {tuple(DTYPE_PRESETS)}")

    compute_mode = "fp8" if dm in ("fp8", "fp8_mxfp4", "w8a8_mxfp4", "fp8_fp4") else "bf16"
    dense_linear_mode = "auto" if dm in ("fp8", "fp8_mxfp4", "w8a8_mxfp4", "fp8_fp4") else "bf16"
    causal_kwargs = _filter_supported_kwargs(
        Qwen4_5ForCausalLM.__init__,
        {
            "use_triton_mxfp4": (dm != "bf16"),
            "compute_mode": compute_mode,
            "dense_linear_mode": dense_linear_mode,
            "kv_dtype": kv_dtype.lower(),
            "fp8_indexer_kv": kv_dtype.lower().startswith("fp8"),
            "dtype": torch.bfloat16,
            "device": device,
            "init_mxfp4_weights": True,
            "parallel_mode": pg["parallel_mode"],
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
    load_qwen4_5_from_checkpoint(
        causal_lm,
        ckpt_path,
        ple_mode="gpu_sharded",
        num_workers=num_load_workers,
    )
    prep_kwargs = _filter_supported_kwargs(
        causal_lm.prepare_weights,
        {"dense_linear_mode": dense_linear_mode},
    )
    causal_lm.prepare_weights(**prep_kwargs)

    plan_kwargs = _filter_supported_kwargs(
        build_qwen4_5_kv_cache_plan,
        {"tp_size": pg["tp_size"], "parallel_mode": pg["parallel_mode"], "kv_dtype": kv_dtype},
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

    kv_cache = LocalKVCache(
        plan,
        tokens_per_block=tokens_per_block,
        num_blocks_per_tag=num_blocks_per_tag,
        device=device,
        dtype=torch.bfloat16,
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
            "parallel_mode": pg["parallel_mode"],
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
                for sub_attr in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                    _hook_mod(getattr(lin, sub_attr, None), "gdn_proj")

            sa = getattr(lyr, "self_attn", None)
            if sa is not None:
                for sub_attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    _hook_mod(getattr(sa, sub_attr, None), "full_attn_proj")
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
    if any(
        k in nl
        for k in (
            "_mxfp4_",
            "fp8_fp4",
            "deep_gemm",
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
    ):
        return "moe_routed"

    if rf_bucket:
        return rf_bucket

    # Fallback heuristic attribution when unannotated
    if any(k in nl for k in ("nvjet", "cutlass", "cublas", "sgemm", "gemm")):
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
) -> Dict[str, Any]:
    """Run deterministic BF16/candidate parity & 64-token boundary snapshot suite."""
    is_rank0 = not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
    # Every DP rank must participate symmetrically in PLE/EP collectives even on B=1 parity checks.
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

    res: Dict[str, Any] = {"saved": None, "verified": None}
    if save_ref_path and is_rank0 and local_b > 0:
        payload = {
            "h_0pct": h_0.detach().cpu(),
            "logits_0pct": log_0.detach().cpu() if log_0 is not None else None,
            "h_64b": h_64b.detach().cpu(),
            "logits_64b": log_64b.detach().cpu() if log_64b is not None else None,
            "snap_64": snap_64,
            "h_90pct": h_90.detach().cpu(),
            "logits_90pct": log_90.detach().cpu() if log_90 is not None else None,
        }
        p = Path(save_ref_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)
        res["saved"] = str(p)

    if verify_ref_path and is_rank0 and local_b > 0:
        ref = torch.load(verify_ref_path, map_location="cpu", weights_only=False)
        checks = {}
        for name, cur_h, cur_l, r_h_key, r_l_key in (
            ("prefill_0pct_2048", h_0, log_0, "h_0pct", "logits_0pct"),
            ("prefill_64tok_cont", h_64b, log_64b, "h_64b", "logits_64b"),
            ("prefill_90pct_18k_2k", h_90, log_90, "h_90pct", "logits_90pct"),
        ):
            rh = ref[r_h_key]
            cos_h = _cosine(cur_h.cpu(), rh)
            max_h = float((cur_h.cpu().float() - rh.float()).abs().max().item())
            top1_ok = True
            cos_l = None
            if cur_l is not None and ref.get(r_l_key) is not None:
                rl = ref[r_l_key]
                cos_l = _cosine(cur_l.cpu(), rl)
                top1_ok = int(cur_l.cpu().argmax(dim=-1).item()) == int(rl.argmax(dim=-1).item())
            checks[name] = {
                "hidden_cos": round(cos_h, 6),
                "hidden_max_abs": round(max_h, 5),
                "logits_cos": round(cos_l, 6) if cos_l is not None else None,
                "top1_match": top1_ok,
                "pass": cos_h >= min_cos and top1_ok,
            }
        # Also verify 64-tok boundary snapshots
        snap_diffs = {}
        for skey, cur_s in snap_64.items():
            ref_s = ref["snap_64"].get(skey)
            if ref_s is not None and cur_s.shape == ref_s.shape:
                if cur_s.dtype in (torch.int32, torch.int64, torch.uint8):
                    eq = bool(torch.equal(cur_s, ref_s))
                    snap_diffs[skey] = {"exact_equal": eq, "pass": eq}
                else:
                    scos = _cosine(cur_s, ref_s)
                    snap_diffs[skey] = {"cos": round(scos, 6), "pass": scos >= min_cos}
        all_ok = all(v["pass"] for v in checks.values()) and all(v["pass"] for v in snap_diffs.values())
        res["verified"] = {"all_pass": all_ok, "min_cos": min_cos, "cases": checks, "snapshots": snap_diffs}
        if not all_ok:
            raise AssertionError(f"Parity gate FAILED: {json.dumps(res['verified'], indent=2)}")
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
    parser.add_argument("--kv-dtype", default="BF16", choices=("BF16", "FP8", "F8_E4M3", "bf16", "fp8_e4m3"))
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
    parser.add_argument("--verify-parity", default="")
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--skip-0pct", action="store_true")
    parser.add_argument("--skip-90pct", action="store_true")
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
    kv_dt = "FP8" if args.kv_dtype.lower().startswith("fp8") or args.kv_dtype.lower() == "f8_e4m3" else "BF16"

    arch, kv_cache, rtp_model = build_prefill_model_and_kv(
        args.ckpt_path,
        pg=pg,
        dtype_mode=args.dtype_mode,
        kv_dtype=kv_dt,
        device=device,
        max_kv_tokens=max_kv_tokens,
    )

    parity_res: Dict[str, Any] = {}
    if args.save_ref or args.verify_parity:
        parity_res = run_parity_suite(
            arch,
            kv_cache,
            rtp_model,
            pg,
            device=device,
            save_ref_path=args.save_ref or None,
            verify_ref_path=args.verify_parity or None,
            min_cos=args.min_cos,
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

    if rank == 0:
        report = {
            "parallel_mode": pg["parallel_mode"],
            "dtype_mode": args.dtype_mode,
            "kv_dtype": kv_dt,
            "world_size": world_size,
            "hardware_roofline": hr.to_dict(),
            "parity": parity_res,
            "prefill_0pct": sweep_0,
            "prefill_90pct": sweep_90,
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
