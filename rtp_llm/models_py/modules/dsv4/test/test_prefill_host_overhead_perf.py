"""Host-side prefill overhead microbenchmark for DSV4 FP8 (W4-2).

Measures per-layer CPU host enqueue latency, end-to-end wall latency,
PyTorch ``aten::empty*`` / ``cudaStreamIsCapturing`` counts via
``torch.profiler``, top ATen allocator callers, and Triton kernel Python
launch dispatch latency across 16 genuine DSV4 FP8 layers (SWA + CSA + HCA
with TileLang mHC and noop MoE).
"""

from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import block as _block_mod
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.prefill import forward as _prefill_forward
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.utils.model_weight import W

_SM103_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10
)


class _NoopMoE(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.strategy_name = "noop"

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        is_decode_forward: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del input_ids, is_decode_forward, kwargs
        return hidden_states


class _DummyLayerCache:
    def __init__(self, base: torch.Tensor, tokens_per_block: int) -> None:
        self.kv_cache_base = base
        self.seq_size_per_block = tokens_per_block


class _DummyKVCacheMgr:
    def __init__(self, layers_by_tag: Dict[str, _DummyLayerCache], tpb: int) -> None:
        self._layers = layers_by_tag
        self._tpb = tpb
        self.group_tags = list(layers_by_tag.keys())

    def get_layer_cache(self, layer_id: int, tag: str) -> _DummyLayerCache:
        del layer_id
        return self._layers[tag]

    def get_seq_size_per_block(self, tag: str) -> int:
        del tag
        return self._tpb

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        del tag
        return self._tpb


def _fp8_weight_pair(
    rows: int, cols: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros((rows, cols), dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    # raw UE8M0 bytes = 0x7f -> 1.0f
    s = torch.full(
        (rows // 128, cols // 128), 0x7F, dtype=torch.uint8, device=device
    ).view(torch.float8_e8m0fnu)
    return w, s


def _build_shared_weights(args: V4Args, device: torch.device) -> SimpleNamespace:
    dim = args.dim
    n_heads = args.n_heads
    q_lora_rank = args.q_lora_rank
    head_dim = args.head_dim
    o_groups = args.o_groups
    o_lora_rank = args.o_lora_rank
    hc_mult = args.hc_mult
    hc_mix_count = (2 + hc_mult) * hc_mult  # 24

    def _base_layer_dict() -> Dict[Any, torch.Tensor]:
        wq_a_w, wq_a_s = _fp8_weight_pair(q_lora_rank, dim, device)
        wq_b_w, wq_b_s = _fp8_weight_pair(n_heads * head_dim, q_lora_rank, device)
        wkv_w, wkv_s = _fp8_weight_pair(head_dim, dim, device)
        wo_a_w, wo_a_s = _fp8_weight_pair(
            o_groups * o_lora_rank, (n_heads * head_dim) // o_groups, device
        )
        wo_b_w, wo_b_s = _fp8_weight_pair(dim, o_groups * o_lora_rank, device)
        return {
            W.v4_attn_wq_a_w: wq_a_w,
            W.v4_attn_wq_a_s: wq_a_s,
            W.v4_attn_wq_b_w: wq_b_w,
            W.v4_attn_wq_b_s: wq_b_s,
            W.v4_attn_wkv_w: wkv_w,
            W.v4_attn_wkv_s: wkv_s,
            W.v4_attn_wo_a_w: wo_a_w,
            W.v4_attn_wo_a_s: wo_a_s,
            W.v4_attn_wo_b_w: wo_b_w,
            W.v4_attn_wo_b_s: wo_b_s,
            W.v4_attn_q_norm: torch.ones(
                (q_lora_rank,), dtype=torch.bfloat16, device=device
            ),
            W.v4_attn_kv_norm: torch.ones(
                (head_dim,), dtype=torch.bfloat16, device=device
            ),
            W.v4_attn_sink: torch.zeros((n_heads,), dtype=torch.float32, device=device),
            W.v4_attn_norm: torch.ones((dim,), dtype=torch.bfloat16, device=device),
            W.v4_ffn_norm: torch.ones((dim,), dtype=torch.bfloat16, device=device),
            W.v4_hc_attn_fn: torch.zeros(
                (hc_mix_count, hc_mult * dim), dtype=torch.float32, device=device
            ),
            W.v4_hc_attn_base: torch.zeros(
                (hc_mix_count,), dtype=torch.float32, device=device
            ),
            W.v4_hc_attn_scale: torch.full(
                (3,), 0.5, dtype=torch.float32, device=device
            ),
            W.v4_hc_ffn_fn: torch.zeros(
                (hc_mix_count, hc_mult * dim), dtype=torch.float32, device=device
            ),
            W.v4_hc_ffn_base: torch.zeros(
                (hc_mix_count,), dtype=torch.float32, device=device
            ),
            W.v4_hc_ffn_scale: torch.full(
                (3,), 0.5, dtype=torch.float32, device=device
            ),
        }

    base_weights = _base_layer_dict()
    by_ratio: Dict[int, Dict[Any, torch.Tensor]] = {}
    by_ratio[0] = dict(base_weights)

    for ratio in (4, 128):
        coff = 2 if ratio == 4 else 1
        d = dict(base_weights)
        d.update(
            {
                W.v4_compressor_ape: torch.zeros(
                    (ratio, coff * head_dim), dtype=torch.float32, device=device
                ),
                W.v4_compressor_wkv: torch.zeros(
                    (coff * head_dim, dim), dtype=torch.bfloat16, device=device
                ),
                W.v4_compressor_wgate: torch.zeros(
                    (coff * head_dim, dim), dtype=torch.bfloat16, device=device
                ),
                W.v4_compressor_norm: torch.ones(
                    (head_dim,), dtype=torch.bfloat16, device=device
                ),
            }
        )
        if ratio == 4:
            iw_b, is_b = _fp8_weight_pair(
                args.index_n_heads * args.index_head_dim, q_lora_rank, device
            )
            d.update(
                {
                    W.v4_indexer_wq_b_w: iw_b,
                    W.v4_indexer_wq_b_s: is_b,
                    W.v4_indexer_weights_proj_w: torch.zeros(
                        (args.index_n_heads, dim), dtype=torch.bfloat16, device=device
                    ),
                    W.v4_indexer_compressor_ape: torch.zeros(
                        (4, 2 * args.index_head_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    W.v4_indexer_compressor_wkv: torch.zeros(
                        (2 * args.index_head_dim, dim),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    W.v4_indexer_compressor_wgate: torch.zeros(
                        (2 * args.index_head_dim, dim),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    W.v4_indexer_compressor_norm: torch.ones(
                        (args.index_head_dim,), dtype=torch.bfloat16, device=device
                    ),
                }
            )
        by_ratio[ratio] = d

    global_weights = {
        W.embedding: torch.zeros(
            (args.vocab_size, dim), dtype=torch.bfloat16, device=device
        ),
        W.final_ln_gamma: torch.ones((dim,), dtype=torch.bfloat16, device=device),
        W.v4_hc_head_fn: torch.zeros(
            (hc_mult, hc_mult * dim), dtype=torch.float32, device=device
        ),
        W.v4_hc_head_base: torch.zeros((hc_mult,), dtype=torch.float32, device=device),
        W.v4_hc_head_scale: torch.full((1,), 0.5, dtype=torch.float32, device=device),
    }
    weights_per_layer = [
        by_ratio[int(r)] for r in args.compress_ratios[: args.n_layers]
    ]
    return SimpleNamespace(global_weights=global_weights, weights=weights_per_layer)


def _build_dummy_kv_pools(
    num_blocks: int,
    block_size: int,
    head_dim: int,
    idx_hd: int,
    max_batch: int,
    device: torch.device,
) -> Tuple[_DummyKVCacheMgr, Dict[str, torch.Tensor]]:
    layers: Dict[str, _DummyLayerCache] = {}
    for tag, ratio in ((SWA_KV, 1), (CSA_KV, 4), (HCA_KV, 128)):
        eb = max(1, block_size // ratio)
        raw = torch.zeros((num_blocks, eb * 584), dtype=torch.uint8, device=device)
        layers[tag] = _DummyLayerCache(raw, block_size)

    idx_eb = max(1, block_size // 4)
    idx_raw = torch.zeros((num_blocks, idx_eb, 132), dtype=torch.uint8, device=device)
    idx_raw[:, :, 128:132].view(torch.float32).fill_(1.0)
    layers[INDEXER_KV] = _DummyLayerCache(idx_raw.view(num_blocks, -1), block_size)

    layers[CSA_STATE] = _DummyLayerCache(
        torch.zeros(
            (num_blocks, block_size * 4 * head_dim), dtype=torch.float32, device=device
        ),
        block_size,
    )
    layers[HCA_STATE] = _DummyLayerCache(
        torch.zeros(
            (num_blocks, block_size * 2 * head_dim), dtype=torch.float32, device=device
        ),
        block_size,
    )
    layers[INDEXER_STATE] = _DummyLayerCache(
        torch.zeros(
            (num_blocks, block_size * 4 * idx_hd), dtype=torch.float32, device=device
        ),
        block_size,
    )

    bt = (
        torch.arange(1, num_blocks, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(max_batch, -1)
        .contiguous()
    )
    bt_by_type = {tag: bt for tag in layers}
    return _DummyKVCacheMgr(layers, block_size), bt_by_type


def _make_prefill_case(
    in_lens: List[int],
    pf_lens: List[int],
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SimpleNamespace]:
    total_tokens = sum(in_lens)
    input_ids = (
        torch.arange(total_tokens, dtype=torch.int32, device=device) % vocab_size
    )
    pos_list: List[int] = []
    cu = [0]
    for s, p in zip(in_lens, pf_lens):
        pos_list.extend(range(p, p + s))
        cu.append(cu[-1] + s)
    positions = torch.tensor(pos_list, dtype=torch.int64, device=device)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=torch.device("cpu"))
    host_in = torch.tensor(in_lens, dtype=torch.int32, device=torch.device("cpu"))
    host_pf = torch.tensor(pf_lens, dtype=torch.int32, device=torch.device("cpu"))
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        cache_store_inputs=None,
        input_lengths=host_in,
        prefix_lengths=host_pf,
        input_lengths_device=None,
        prefix_lengths_device=None,
    )
    return input_ids, positions, cu_seqlens, attn_inputs


def _profile_one_forward(
    v4: V4Transformer,
    kv_cache: _DummyKVCacheMgr,
    bt_by_type: Dict[str, torch.Tensor],
    case: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SimpleNamespace],
    n_layers: int,
) -> Dict[str, Any]:
    input_ids, positions, cu_seqlens, attn_inputs = case
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        with_stack=True,
        record_shapes=False,
    ) as prof:
        with torch.inference_mode():
            _ = _prefill_forward.forward_layers(
                v4,
                kv_cache=kv_cache,
                input_ids=input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                block_tables_by_type=bt_by_type,
                attn_inputs=attn_inputs,
            )
    torch.cuda.synchronize()

    counts = {
        "aten_empty": 0,
        "aten_empty_like": 0,
        "aten_empty_strided": 0,
        "aten_zeros": 0,
        "aten_full": 0,
        "aten_arange": 0,
        "cudaStreamIsCapturing": 0,
        "cudaLaunchKernel": 0,
        "cuLaunchKernelEx": 0,
    }
    cpu_ops: Dict[str, Tuple[int, float]] = {}
    empty_stacks: Dict[str, int] = {}
    capturing_stacks: Dict[str, int] = {}
    malloc_stacks: Dict[str, Tuple[int, float]] = {}

    for ev in prof.events():
        name = ev.name
        if name == "aten::empty":
            counts["aten_empty"] += 1
        elif name == "aten::empty_like":
            counts["aten_empty_like"] += 1
        elif name == "aten::empty_strided":
            counts["aten_empty_strided"] += 1
        elif name == "aten::zeros":
            counts["aten_zeros"] += 1
        elif name == "aten::full":
            counts["aten_full"] += 1
        elif name == "aten::arange":
            counts["aten_arange"] += 1
        elif "cudaStreamIsCapturing" in name:
            counts["cudaStreamIsCapturing"] += 1
        elif "cudaLaunchKernel" in name and "Ex" not in name:
            counts["cudaLaunchKernel"] += 1
        elif "cuLaunchKernelEx" in name or "cudaLaunchKernelEx" in name:
            counts["cuLaunchKernelEx"] += 1

    def _fmt_stack(stack: Any) -> str:
        if not stack:
            return "<no-stack>"
        rtp_frames = [f.strip() for f in stack if "rtp_llm" in f]
        frames = rtp_frames[:3] if rtp_frames else [f.strip() for f in stack[:3]]
        return " <- ".join(frames)

    for avg in prof.key_averages(group_by_stack_n=8):
        name = avg.key
        stack_str = _fmt_stack(getattr(avg, "stack", None))
        cnt = int(avg.count)
        self_us = float(avg.self_cpu_time_total)

        if not name.startswith("forward(layer="):
            prev_c, prev_us = cpu_ops.get(name, (0, 0.0))
            cpu_ops[name] = (prev_c + cnt, prev_us + self_us)

        if name in (
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::zeros",
            "aten::full",
            "aten::arange",
        ):
            key = f"{name} @ {stack_str}"
            empty_stacks[key] = empty_stacks.get(key, 0) + cnt
        elif "cudaStreamIsCapturing" in name:
            capturing_stacks[stack_str] = capturing_stacks.get(stack_str, 0) + cnt
        elif "cudaMalloc" in name:
            prev_c, prev_us = malloc_stacks.get(stack_str, (0, 0.0))
            malloc_stacks[stack_str] = (prev_c + cnt, prev_us + self_us)

    total_empty = (
        counts["aten_empty"]
        + counts["aten_empty_like"]
        + counts["aten_empty_strided"]
        + counts["aten_zeros"]
        + counts["aten_full"]
    )
    top_cpu = sorted(cpu_ops.items(), key=lambda kv: kv[1][1], reverse=True)[:25]
    top_callers = sorted(empty_stacks.items(), key=lambda kv: kv[1], reverse=True)[:25]
    top_capturing = sorted(
        capturing_stacks.items(), key=lambda kv: kv[1], reverse=True
    )[:10]
    top_malloc = sorted(malloc_stacks.items(), key=lambda kv: kv[1][1], reverse=True)[
        :10
    ]
    return {
        "counts": counts,
        "total_empty": total_empty,
        "total_empty_per_layer": total_empty / n_layers,
        "capturing_per_layer": counts["cudaStreamIsCapturing"] / n_layers,
        "top_cpu_ops": [(k, v[0], round(v[1] / 1000.0, 3)) for k, v in top_cpu],
        "top_empty_callers": top_callers,
        "top_capturing_callers": top_capturing,
        "top_malloc_callers": [
            (k, v[0], round(v[1] / 1000.0, 3)) for k, v in top_malloc
        ],
    }


def _bench_triton_launch_us(device: torch.device, iters: int = 500) -> Dict[str, float]:
    from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
        quantize_and_insert_k_cache,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
        combine_topk_swa_indices,
    )

    T = 512
    K = 512
    W_sz = 128
    k_in = torch.randn((T, 512), dtype=torch.bfloat16, device=device)
    k_cache = torch.empty((4, 256, 584), dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(T, dtype=torch.int64, device=device)

    topk = (
        torch.arange(K, dtype=torch.int32, device=device)
        .view(1, K)
        .expand(T, K)
        .contiguous()
    )
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([256 + T], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([256 + T], dtype=torch.int32, device=device)

    for _ in range(10):
        quantize_and_insert_k_cache(k_in, k_cache, slot_mapping)
        combine_topk_swa_indices(
            topk,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size=W_sz,
            compress_ratio=4,
            topk=K,
            M=1024,
            N=512,
            flash_mla_indices=True,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        quantize_and_insert_k_cache(k_in, k_cache, slot_mapping)
    t1 = time.perf_counter()
    torch.cuda.synchronize()

    t2 = time.perf_counter()
    for _ in range(iters):
        combine_topk_swa_indices(
            topk,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size=W_sz,
            compress_ratio=4,
            topk=K,
            M=1024,
            N=512,
            flash_mla_indices=True,
        )
    t3 = time.perf_counter()
    torch.cuda.synchronize()

    comb_topk = ((K + W_sz + 127) // 128) * 128
    out_indices = torch.empty((T, 1, comb_topk), dtype=torch.int32, device=device)
    out_lens = torch.empty((T,), dtype=torch.int32, device=device)
    for _ in range(10):
        combine_topk_swa_indices(
            topk,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size=W_sz,
            compress_ratio=4,
            topk=K,
            M=1024,
            N=512,
            flash_mla_indices=True,
            out_indices=out_indices,
            out_lens=out_lens,
        )
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    for _ in range(iters):
        combine_topk_swa_indices(
            topk,
            query_start_loc,
            seq_lens,
            gather_lens,
            window_size=W_sz,
            compress_ratio=4,
            topk=K,
            M=1024,
            N=512,
            flash_mla_indices=True,
            out_indices=out_indices,
            out_lens=out_lens,
        )
    t5 = time.perf_counter()
    torch.cuda.synchronize()

    return {
        "quantize_and_insert_k_cache_host_us": round((t1 - t0) * 1e6 / iters, 2),
        "combine_topk_host_us": round((t3 - t2) * 1e6 / iters, 2),
        "combine_topk_prealloc_host_us": round((t5 - t4) * 1e6 / iters, 2),
    }


@unittest.skipUnless(_SM103_AVAILABLE, "Requires SM103 GPU")
class TestPrefillHostOverheadPerf(unittest.TestCase):
    def test_prefill_host_overhead_benchmark(self) -> None:
        device = torch.device("cuda:0")
        n_layers = 16
        args = V4Args(
            vocab_size=256,
            dim=4096,
            n_heads=64,
            n_layers=n_layers,
            n_mtp_layers=0,
            q_lora_rank=1024,
            head_dim=512,
            rope_head_dim=64,
            o_groups=8,
            o_lora_rank=1024,
            window_size=128,
            compress_ratios=[0, 0] + [4, 128] * 7,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            max_batch_size=4,
            max_seq_len=8192,
            fp8_kv_cache=True,
            skip_python_lm_head=True,
        )
        mw = _build_shared_weights(args, device)

        with (
            patch.object(_block_mod, "ChunkedFp8Fp4MoeLayer", _NoopMoE),
            patch.object(_block_mod, "adapt_split_moe_weights", lambda *a, **kw: None),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.mega_support.mega_decode_unavailable_reason",
                return_value="perf benchmark prefill-only",
            ),
        ):
            v4 = V4Transformer(args, mw).to(device)
        for layer in v4.layers:
            layer.attn.init_rope_cache(device)
        v4._bind_prefill_workspace_dims(
            q_rows=2048,
            q_dim=args.n_heads * args.head_dim,
            full_rows=0,
            main_w=0,
            idx_w=0,
        )

        kv_cache, bt_by_type = _build_dummy_kv_pools(
            num_blocks=64,
            block_size=256,
            head_dim=args.head_dim,
            idx_hd=args.index_head_dim,
            max_batch=4,
            device=device,
        )

        workloads = {
            "B1_T1024_sp0": _make_prefill_case([1024], [0], args.vocab_size, device),
            "B1_T1024_sp256": _make_prefill_case(
                [1024], [256], args.vocab_size, device
            ),
            "B2_T1024_mixed": _make_prefill_case(
                [512, 512], [0, 256], args.vocab_size, device
            ),
        }

        warmup_iters = 5
        bench_iters = 20
        timing_summary: Dict[str, Dict[str, float]] = {}
        profile_summary: Dict[str, Any] = {}

        with torch.inference_mode():
            for name, case in workloads.items():
                input_ids, positions, cu_seqlens, attn_inputs = case
                for _ in range(warmup_iters):
                    _ = _prefill_forward.forward_layers(
                        v4,
                        kv_cache=kv_cache,
                        input_ids=input_ids,
                        positions=positions,
                        cu_seqlens=cu_seqlens,
                        block_tables_by_type=bt_by_type,
                        attn_inputs=attn_inputs,
                    )
                torch.cuda.synchronize()

                # Measure unprofiled CPU host-enqueue time (Python + ATen CPU overhead)
                # separate from GPU sync wall time.
                t_enqueue_total = 0.0
                t_wall_start = time.perf_counter()
                for _ in range(bench_iters):
                    # Sync before each iteration to measure clean CPU dispatch vs stream backpressure
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = _prefill_forward.forward_layers(
                        v4,
                        kv_cache=kv_cache,
                        input_ids=input_ids,
                        positions=positions,
                        cu_seqlens=cu_seqlens,
                        block_tables_by_type=bt_by_type,
                        attn_inputs=attn_inputs,
                    )
                    t1 = time.perf_counter()
                    t_enqueue_total += t1 - t0
                torch.cuda.synchronize()
                t_wall_total = time.perf_counter() - t_wall_start

                host_ms_per_iter = (t_enqueue_total * 1000.0) / bench_iters
                wall_ms_per_iter = (t_wall_total * 1000.0) / bench_iters
                timing_summary[name] = {
                    "host_enqueue_ms": round(host_ms_per_iter, 3),
                    "host_enqueue_ms_per_layer": round(host_ms_per_iter / n_layers, 3),
                    "wall_ms": round(wall_ms_per_iter, 3),
                    "wall_ms_per_layer": round(wall_ms_per_iter / n_layers, 3),
                }
                profile_summary[name] = _profile_one_forward(
                    v4, kv_cache, bt_by_type, case, n_layers
                )

        triton_us = _bench_triton_launch_us(device)
        report = {
            "n_layers": n_layers,
            "timings": timing_summary,
            "profile": profile_summary,
            "triton_launch_us": triton_us,
        }
        print("\n=== DSV4 Prefill Host Overhead Benchmark ===")
        print(f"METRICS_JSON={json.dumps(report, sort_keys=True)}")
        for name, t in timing_summary.items():
            p = profile_summary[name]
            print(
                f"[{name}] host_ms/layer={t['host_enqueue_ms_per_layer']:.3f}ms "
                f"wall_ms/layer={t['wall_ms_per_layer']:.3f}ms "
                f"empty/layer={p['total_empty_per_layer']:.2f} "
                f"capturing/layer={p['capturing_per_layer']:.2f} "
                f"empty_total={p['total_empty']} "
                f"counts={p['counts']}"
            )
            print("  top_cpu_ops:")
            for op in p["top_cpu_ops"]:
                print(f"    {op}")
            print("  top_empty_callers:")
            for caller in p["top_empty_callers"]:
                print(f"    {caller}")
            print(f"  top_capturing_callers={p['top_capturing_callers']}")
            print(f"  top_malloc_callers={p['top_malloc_callers']}")
        print(f"[triton_launch_us] {triton_us}")
        self.assertGreater(len(timing_summary), 0)
        for name, p in profile_summary.items():
            self.assertLess(
                p["total_empty_per_layer"],
                25.0,
                f"{name} exceeded expected aten::empty/layer budget",
            )


if __name__ == "__main__":
    unittest.main()
