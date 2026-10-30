"""Eight-rank production Indexer forward: synchronous vs auxiliary K, bit exact.

Projection modules are identity/linear fixtures; RoPE/Hadamard/FP8 cache writes,
CP assembly, DeepGEMM logits and TopK are the actual production operators.
This is a correctness test, not a performance benchmark.
"""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_case(rank, lengths, owner, fused, zero_queries, side):
    from rtp_llm.models_py.distributed import pynccl_cp
    from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
    from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
        build_actual_local_cu_kv_seqlens,
        build_indexer_cp_chunk_plan,
    )
    from rtp_llm.models_py.modules.hybrid.indexer import Indexer

    n = sum(lengths)
    size = dist.get_world_size()
    rows = (n + size - 1) // size
    device = torch.device("cuda", rank)
    lens = torch.tensor(lengths, dtype=torch.int64)
    plan = build_indexer_cp_chunk_plan(
        NS(cp_size=size, cp_rank=rank), lens, 64, device, owner_block_size=owner
    )
    cu = build_actual_local_cu_kv_seqlens(plan)
    owned = [[i for i in range(v) if i // owner % size == rank] for v in lengths]
    pages = [max(1, (len(x) + 63) // 64) for x in owned]
    table = torch.full((len(lengths), max(pages)), -1, dtype=torch.int32)
    slots = torch.full((n,), -1, dtype=torch.int64)
    page_offset = 1
    offset = 0
    for b, (count, positions) in enumerate(zip(pages, owned)):
        table[b, :count] = torch.arange(page_offset, page_offset + count)
        for j, pos in enumerate(positions):
            slots[offset + pos] = (page_offset + j // 64) * 64 + j % 64
        offset += lengths[b]
        page_offset += count
    table = table.to(device)
    slots = slots.to(device)
    dst = []
    actual_start = pad_start = 0
    for a, p in zip(
        plan.per_req_actual_local_kv_lens.cpu().tolist(),
        plan.per_req_local_kv_lens.cpu().tolist(),
    ):
        dst.extend(range(pad_start, pad_start + a))
        pad_start += p
        actual_start += a
    src = torch.full((plan.total_local_T,), -1, dtype=torch.int64, device=device)
    if dst:
        src[torch.tensor(dst, device=device)] = torch.arange(len(dst), device=device)
    local_ids = torch.arange(min(rows, 8), device=device, dtype=torch.int64)
    if zero_queries and rank == size - 1:
        local_ids = local_ids[:0]
    # Valid causal ranges for the selected queries; padding rows are excluded.
    local_ids = local_ids[(rank * rows + local_ids) < n]
    global_ids = (rank * rows + local_ids).cpu().tolist()
    starts = []
    ends = []
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    for g in global_ids:
        b = next(
            i for i in range(len(lengths)) if cumulative[i] <= g < cumulative[i + 1]
        )
        starts.append(cumulative[b])
        ends.append(g + 1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    cp = NS(
        kv_cache_sharded=True,
        cp_size=size,
        cp_rank=rank,
        total_kv_len=n,
        full_rope_pos_ids=torch.arange(rows, device=device, dtype=torch.int32),
        sharded_slot_mapping=slots,
        kv_restore_unpad_indices=torch.arange(n, device=device, dtype=torch.int64),
        cu_kv_seqlens_global=i32(cumulative),
        indexer_cp_plan=plan,
        indexer_cp_local_cu=cu,
        indexer_copy_dst_idx=torch.tensor(dst, dtype=torch.int64, device=device),
        indexer_src_for_padded=src,
        total_local_ids=local_ids,
        total_local_ids_is_identity=False,
        precomputed_ks=i32(starts),
        precomputed_ke=i32(ends),
        precomputed_lengths=i32([e - s for s, e in zip(starts, ends)]),
        precomputed_topk_off=i32([0] * len(starts)),
    )
    attn = NS(is_prefill=True, kv_cache_block_id_device=table)
    fmha = NS(prefill_max_kv_len=max(lengths))
    model = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(model)
    model.index_n_heads = 32
    model.index_head_dim = 128
    model.index_topk = 2048
    model._kv_owner_tokens_per_block = owner
    model.parallelism_config = NS(prefill_cp_config=NS(is_enabled=lambda: True))
    model._fuse_logits_head_gate = False
    model.softmax_scale = 128**-0.5
    model.weights_scale = 32**-0.5
    model.wq_b = lambda x, **kw: x.clone()
    model.wk = lambda x, **kw: x.clone()
    model.k_norm = lambda x: x
    torch.manual_seed(9 + rank)
    model.weights_proj = torch.nn.Linear(
        128, 32, bias=False, device=device, dtype=torch.float32
    )
    angles = (
        torch.arange(rows, device=device)[:, None].float()
        * torch.arange(1, 33, device=device)[None, :]
        / 100
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    model.indexer_op = IndexerOp(32, 128, 2048, 64, cos_sin_cache=cos_sin)
    hidden = torch.randn(rows, 128, device=device, dtype=torch.bfloat16)
    q = torch.randn(rows, 32 * 128, device=device, dtype=torch.bfloat16)
    observed = []
    logits_observed = []
    from rtp_llm.models_py.modules.base.cuda import indexer_op as indexer_module

    original_logits = indexer_module.deep_gemm.fp8_mqa_logits

    def observe_logits(*args, **kwargs):
        result = original_logits(*args, **kwargs)
        q_input, _, gate, ks, ke = args[:5]
        cols = torch.arange(result.shape[1], device=result.device)[None, :]
        valid = (cols >= ks[:, None]) & (cols < ke[:, None])
        logits_observed.append(
            (q_input.clone(), gate.clone(), result.masked_fill(~valid, 0).clone())
        )
        return result

    original = model.indexer_op.prepare_cp_k

    def observe(*args, **kwargs):
        values = original(*args, **kwargs)
        observed.append(tuple(t.clone() for t in values))
        return values

    model.indexer_op.prepare_cp_k = observe
    results = []
    caches = []
    with patch.dict(
        os.environ, {"DSV4_CP_PREFILL_INDEXER_FUSED_QUANT": str(int(fused))}
    ):
        for async_mode in (False, False, True, True):
            cache = NS(
                kv_scale_base=torch.zeros(
                    page_offset, 64, 132, dtype=torch.uint8, device=device
                )
            )
            # A new live KV gather on its separate stream exercises isolation.
            kv_side = pynccl_cp.warmup_kv_overlap(dist.group.WORLD)
            with torch.cuda.stream(kv_side):
                peer = torch.full((64, 576), rank, dtype=torch.bfloat16, device=device)
                peers = pynccl_cp.all_gather_kv_overlap(peer, dist.group.WORLD)
            with patch.object(
                pynccl_cp,
                "indexer_overlap_stream",
                return_value=side if async_mode else None,
            ):
                with patch.object(
                    indexer_module.deep_gemm, "fp8_mqa_logits", observe_logits
                ):
                    result = model.forward(hidden, q, cache, fmha, attn, False, cp)
            # Exercise auxiliary allocator reuse before main-stream logits finish.
            with torch.cuda.stream(side):
                junk = torch.empty((n, 128), dtype=torch.float8_e4m3fn, device=device)
                junk.view(torch.uint8).fill_(255)
            torch.cuda.synchronize()
            expected = torch.arange(
                size, device=device, dtype=torch.bfloat16
            ).repeat_interleave(64)
            torch.testing.assert_close(peers[:, 0], expected, rtol=0, atol=0)
            results.append(result)
            caches.append(cache.kv_scale_base)
    diagnostics = []
    for i in (1, 2, 3):
        diagnostics.append(
            dict(
                comparison=i,
                async_mode=i >= 2,
                cache_exact=torch.equal(caches[0], caches[i]),
                k_exact=[
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(observed[0], observed[i])
                ],
                topk_order_exact=torch.equal(results[0], results[i]),
                topk_set_exact=torch.equal(
                    results[0].sort(dim=-1).values, results[i].sort(dim=-1).values
                ),
            )
        )
    Path(
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"],
        f"diagnostic_rank{rank}_n{n}_fused{fused}_empty{zero_queries}.json",
    ).write_text(json.dumps(diagnostics, indent=2))
    for i in (1, 2, 3):
        torch.testing.assert_close(caches[0], caches[i], rtol=0, atol=0)
        for a, b in zip(observed[0], observed[i]):
            torch.testing.assert_close(
                a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
            )
    # TopK order also varies on synchronous repeats. Require identical selected
    # sets AND bit-identical numerical inputs/logits; production sorting stays.
    if logits_observed:
        assert len(logits_observed) == 4
        for values in logits_observed[1:]:
            for expected, actual in zip(logits_observed[0], values):
                torch.testing.assert_close(
                    expected.view(torch.uint8), actual.view(torch.uint8), rtol=0, atol=0
                )
    for i in (1, 2, 3):
        torch.testing.assert_close(
            results[0].sort(dim=-1).values,
            results[i].sort(dim=-1).values,
            rtol=0,
            atol=0,
        )
    return dict(
        lengths=lengths,
        owner=owner,
        fused=fused,
        zero_queries=zero_queries,
        cache_k_q_gate_logits_bit_exact=True,
        topk_set_exact=True,
        topk_order_exact=all(d["topk_order_exact"] for d in diagnostics),
    )


def worker(rank, port, world_size):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    from rtp_llm.models_py.distributed import collective_torch, pynccl_cp
    from rtp_llm.models_py.distributed.symm_mem import init_symm_mem_communicator

    collective_torch._group_map[collective_torch.Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = NS(
        tp_size=world_size, dp_size=1, world_size=world_size
    )
    assert init_symm_mem_communicator(dist.group.WORLD) is not None
    pynccl_cp.warmup(dist.group.WORLD)
    pynccl_cp.warmup_kv_overlap(dist.group.WORLD)
    side = pynccl_cp.indexer_overlap_stream(
        dist.group.WORLD, torch.device("cuda", rank)
    )
    assert side is not None
    results = []
    for lengths in ([1, 0, 513, 4097], [64], [513, 1025]):
        for fused in (True, False):
            for zero_queries in (True, False):
                results.append(
                    run_case(
                        rank, lengths, 512 if fused else 64, fused, zero_queries, side
                    )
                )
    Path(
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], f"indexer_rank{rank}.json"
    ).write_text(json.dumps(results, indent=2))
    dist.barrier()
    pynccl_cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(os.environ.get("INDEXER_TEST_WORLD_SIZE", "8"))
    assert world_size in (4, 8)
    assert (
        torch.cuda.device_count() == world_size
    ), "requires exactly allocated test GPUs"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port, world_size), nprocs=world_size, join=True)
