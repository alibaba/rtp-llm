"""Constructed-input DSA contracts. Run with torchrun on eight selected GPUs.

These check cache coordinates, distributed equivalence and Graph replay, not
checkpoint/model accuracy. --tp-size 4 creates two independent Decode owners.
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.base.cuda.bf16_indexer_op import Bf16DraftIndexerOp
from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams as BaseDsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, compute_ops


def assert_topk_members(actual, expected):
    # Selection is exact; candidate order is not an attention contract.
    torch.testing.assert_close(
        actual.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )


class DsaPageRRParams(BaseDsaPageRRParams):
    """Explicit model/Graph workspace contract for constructed fixtures."""

    def __init__(self, attention, parallel, **kwargs):
        owner = DsaWorkspaceOwner(tp_size=parallel.tp_size)
        value = torch.zeros(512, 128, dtype=torch.bfloat16, device="cuda")
        value[:128].copy_(torch.eye(128, dtype=torch.bfloat16, device="cuda"))
        owner.value_weights[0] = value[None].expand(96, -1, -1).contiguous()
        super().__init__(
            attention, parallel, workspace_owner=owner, is_cuda_graph=True, **kwargs
        )

    def prepare(self, inputs, forbid_realloc=False):
        super().prepare(inputs, forbid_realloc=forbid_realloc)
        self.prepare_execution("indexer")
        self.prepare_execution("sfa")


def force_independent_kv_chunks(params, *, force_sfa=False):
    # Tighten each complete budget just below its full-history requirement.
    # No fixed token capacity is introduced in the production path.
    owner = params.workspace_owner
    owner.budgets["indexer"] = params.indexer_execution.required_bytes - 1
    for phase in owner.phases.values():
        phase.release()  # Before any capture; a fixture may rebuild its plan.
    params.indexer_execution = params.sfa_execution = None
    index = params.prepare_execution("indexer")
    sfa = params.prepare_execution("sfa")
    if force_sfa:
        # Use short-Q/long-history for this case: a tiny first Prefill can
        # need more memory for FP32 running latent than for all its KV.
        owner.budgets["sfa"] = sfa.required_bytes - 1
        sfa.workspace.release()
        params.sfa_execution = None
        sfa = params.prepare_execution("sfa")
    assert index.plan.chunked
    assert sfa.plan.chunked == force_sfa


def check_score_kernel():
    torch.manual_seed(3101)
    q = (torch.randn(1, 3, 32, 128, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(1, 2053, 128, device="cuda") * 0.1).bfloat16()
    gates = torch.randn(1, 3, 32, device="cuda") * 0.01
    positions = torch.tensor([[16, 2047, 2052]], dtype=torch.int32, device="cuda")
    # One representative JIT/warmup group; no measured performance here.
    v4_indexer_score(q, k, gates, positions)
    actual = v4_indexer_score(q, k, gates, positions)
    reference = (
        torch.einsum("bqhd,bkd->bqhk", q.float(), k.float()).relu() * gates[..., None]
    ).sum(2)
    mask = (
        torch.arange(k.shape[1], device="cuda")[None, None, :] <= positions[..., None]
    )
    reference = reference.masked_fill(~mask, -float("inf"))
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-4)


def check_topk_mask_and_ties(op):
    rows, width = 257, 2053
    score = (torch.arange(rows * width, device="cuda").view(rows, width) % 7).float()
    limits = torch.arange(rows, device="cuda") % width
    limits[128:130] = torch.tensor([2048, width], device="cuda")
    score.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= limits[:, None], -float("inf")
    )
    positions = torch.arange(width, dtype=torch.int32, device="cuda")
    expected = score.argsort(dim=-1, descending=True, stable=True)[:, :2048].int()
    expected.masked_fill_(~torch.isfinite(score.gather(-1, expected.long())), -1)
    assert_topk_members(op._select(score, positions), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op._select(score, positions)
    score.fill_(-float("inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert (captured == -1).all().item()


def check_score_dcp(op, tp, rank):
    # Feed constructed complete-head scores through the current common KV
    # executor. Do not restore the removed private _full_score_dcp shortcut.
    from unittest.mock import patch
    for length in (17, 2048, 4097, 65537):
        width = math.ceil(length / (tp * 1024)) * 1024
        local = torch.arange(width, dtype=torch.int32, device="cuda")
        positions = ((local // 1024) * tp + rank) * 1024 + local % 1024
        limits = torch.tensor([5, length, length - 1], device="cuda")
        score = torch.ones(3, width, device="cuda").masked_fill(
            positions[None, :] >= limits[:, None], -float("inf"))
        global_positions = torch.cat([
            ((local // 1024) * tp + peer) * 1024 + local % 1024
            for peer in range(tp)])
        buffers = {}
        def workspace(name, shape, dtype):
            key = (name,tuple(shape),dtype)
            if key not in buffers:
                if torch.cuda.is_current_stream_capturing():
                    raise AssertionError('DCP fixture workspace grew inside capture')
                buffers[key] = torch.empty(shape, dtype=dtype, device="cuda")
            return buffers[key]
        execution = SimpleNamespace(batch=3,queries=1,rows=3,selection_rows=math.ceil(3/tp),
            fp8_indexer=False,blocks=[SimpleNamespace(width=width,dcp_positions=global_positions)],
            begin=lambda:workspace,finish=lambda:None)
        params = SimpleNamespace(shard_size=tp,shard_rank=rank,page_size=1024,
            kernel_page_size=128,is_prefill=False,indexer_execution=execution,
            positions_d=torch.arange(3,dtype=torch.int32,device="cuda"))
        query = torch.zeros((3,32,128),dtype=torch.bfloat16,device="cuda")
        weights = torch.ones((3,32),dtype=torch.float32,device="cuda")
        def evaluate():
            with patch.object(op,'cache_view',return_value=None),patch.object(op,'_score_decode',return_value=score):
                return op.compute_topk(query,weights,None,params,None,None)
        actual = evaluate()
        expected = torch.full_like(actual, -1)
        for row, limit in enumerate(limits.tolist()):
            count = min(limit,2048)
            expected[row,:count] = torch.arange(count,dtype=torch.int32,device="cuda")
        assert_topk_members(actual,expected)
    graph = torch.cuda.CUDAGraph()
    evaluate()
    dist.barrier()
    with torch.cuda.graph(graph):
        captured = evaluate()
    score.fill_(-float("inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert (captured == -1).all().item()


def check_sparse_attention(parallel):
    tp, rank = parallel.tp_size, parallel.tp_rank
    page, kernel, length, batch, queries = 1024, 128, 4097, 2, 4
    local_tokens = math.ceil(length / (page * tp)) * page
    columns = local_tokens // kernel
    table = torch.arange(1, batch * columns + 1, dtype=torch.int32, device="cuda").view(
        batch, columns
    )
    inputs = SimpleNamespace(
        is_prefill=False,
        is_target_verify=True,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.full((batch,), queries, dtype=torch.int32),
        physical_token_count=batch * queries,
        prefix_lengths_device=torch.full(
            (batch,), length - queries, dtype=torch.int32, device="cuda"
        ),
        sequence_lengths_plus_1_device=torch.full(
            (batch,), length, dtype=torch.int32, device="cuda"
        ),
        kv_cache_kernel_block_id_device=table,
    )
    attention = SimpleNamespace(tokens_per_block=page, kernel_tokens_per_block=kernel)
    params = DsaPageRRParams(attention, parallel)
    params.prepare(inputs)
    torch.manual_seed(3102 + parallel.dp_rank)
    global_kv = (torch.randn(batch, length, 576, device="cuda") * 0.1).bfloat16()
    full_query = (torch.randn(batch * queries, 96, 576, device="cuda") * 0.1).bfloat16()
    local_query = full_query[
        :, rank * (96 // tp) : (rank + 1) * (96 // tp)
    ].contiguous()
    local = torch.arange(local_tokens, device="cuda")
    global_pos = ((local // page) * tp + rank) * page + local % page
    valid = global_pos < length
    raw_cache = torch.zeros(
        batch * columns + 1, kernel, 576, dtype=torch.bfloat16, device="cuda"
    )
    for request in range(batch):
        packed = torch.zeros(local_tokens, 576, dtype=torch.bfloat16, device="cuda")
        packed[valid] = global_kv[request, global_pos[valid]]
        raw_cache[table[request].long()] = packed.view(columns, kernel, 576)
    indices = torch.full((batch * queries, 2048), -1, dtype=torch.int32, device="cuda")
    for row, pos in enumerate(params.positions_d.tolist()):
        # Row zero's selected keys live on one owner; other owners are empty.
        chosen = torch.arange(
            17 if row == 0 else 2048, dtype=torch.int32, device="cuda"
        )
        indices[row, : chosen.numel()] = chosen
    indices[-1].fill_(-1)  # Entirely empty query must return finite zero.
    sparse = PageRRSparseMlaOp(
        96 // tp,
        512,
        64,
        128,
        kernel,
        1.0,
        2048,
        parallelism_config=parallel,
        workspace_owner=params.workspace_owner,
    )
    sparse.plan(params, table, inputs)
    cache = raw_cache.view(-1, 1, 576)
    sparse.forward(local_query, cache, indices)  # one complete warmup group
    actual = sparse.forward(local_query, cache, indices)
    expected = torch.zeros_like(local_query[..., :128])
    for row in range(batch * queries):
        selected = indices[row][indices[row] >= 0].long()
        if selected.numel():
            kv = global_kv[row // queries, selected].float()
            score = full_query[row].float() @ kv.T / math.sqrt(192)
            output = torch.softmax(score, -1) @ kv[:, :512]
            expected[row] = output[
                rank * (96 // tp) : (rank + 1) * (96 // tp), :128
            ].bfloat16()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.015)
    assert torch.isfinite(actual).all().item()
    # Metadata stays fixed and candidate slots are recomputed before replay.
    pointers = (params.positions_d.data_ptr(), params.slot_mapping.data_ptr())
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        captured = sparse.forward(local_query, cache, indices)
    inputs.prefix_lengths_device.add_(1)
    params.prepare(inputs, forbid_realloc=True)
    assert pointers == (params.positions_d.data_ptr(), params.slot_mapping.data_ptr())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=2e-4, rtol=0.015)
    # Query CP scatter/restore preserves global token order and head shards.
    scattered = sparse.communicator.scatter_queries(local_query)
    restored = sparse.communicator.restore_queries(scattered[..., :512])
    torch.testing.assert_close(restored, local_query[..., :512])


def check_prefill(op, parallel):
    """Exercise actual owner writes, both KV AllGathers and ragged query CP."""
    tp, rank = parallel.tp_size, parallel.tp_rank
    logical = 17 + 2053
    physical = math.ceil(logical / tp) * tp
    lengths = [17, 2053] + ([physical - logical] if physical > logical else [])
    offsets = [0]
    padding = []
    for request, length in enumerate(lengths):
        padding.extend([request * max(lengths) - offsets[-1]] * length)
        offsets.append(offsets[-1] + length)
    columns = 8
    table = torch.arange(
        1, len(lengths) * columns + 1, device="cuda", dtype=torch.int32
    ).view(len(lengths), columns)
    if physical > logical:
        table[-1].zero_()
    inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.tensor(lengths, dtype=torch.int32),
        prefix_lengths=torch.zeros(len(lengths), dtype=torch.int32),
        input_lengths_device=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        prefix_lengths_device=torch.zeros(
            len(lengths), device="cuda", dtype=torch.int32
        ),
        cu_seqlens_device=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        padding_offset=torch.tensor(padding, device="cuda", dtype=torch.int32),
        kv_cache_kernel_block_id_device=table,
        logical_token_count=logical,
        physical_token_count=physical,
    )
    params = DsaPageRRParams(
        SimpleNamespace(tokens_per_block=1024, kernel_tokens_per_block=128), parallel
    )
    params.prepare(inputs)
    force_independent_kv_chunks(params)
    assert "running" not in params.sfa_execution.shapes
    assert params.query_indices.numel() == physical // tp
    torch.manual_seed(3103 + parallel.dp_rank)
    query = (torch.randn(physical, 32, 128, device="cuda") * 0.1).bfloat16()
    keys = (torch.randn(physical, 128, device="cuda") * 0.1).bfloat16()
    gates = torch.rand(physical, 32, 1, device="cuda") * 0.01
    index_cache = SimpleNamespace(
        kv_cache_base=torch.zeros(
            len(lengths) * columns + 1, 128, 256, dtype=torch.uint8, device="cuda"
        )
    )
    op.encode_q_k(query, keys, index_cache, params, inputs, params)
    selected = op.compute_topk(query, gates, index_cache, params, inputs, params)
    counts = (selected >= 0).sum(-1)
    expected_counts = (params.query_positions + 1).clamp(0, 2048)
    torch.testing.assert_close(counts, expected_counts.to(counts.dtype))
    assert ((selected < 0) | (selected <= params.query_positions[:, None])).all().item()
    assert not (params.slot_mapping[logical:] >= 0).any().item()
    # Independent unchunked scores use original keys before PageRR packing.
    valid_rows = (params.query_positions >= 0).nonzero().flatten()
    for local_row in valid_rows[[0, -1]].tolist():
        global_row = params.query_indices[local_row].item()
        request = params.query_requests[local_row].item()
        offset = sum(lengths[:request])
        count = params.kv_lens_host[request]
        score = v4_indexer_score(
            query[global_row : global_row + 1].unsqueeze(0).contiguous(),
            keys[offset : offset + count].unsqueeze(0).contiguous(),
            gates[global_row : global_row + 1, :, 0].unsqueeze(0).contiguous(),
            params.query_positions[local_row : local_row + 1].view(1, 1),
        )[0]
        expected = torch.full((1, 2048), -1, dtype=torch.int32, device="cuda")
        chosen = score.argsort(dim=-1, descending=True, stable=True)[
            :, : min(count, 2048)
        ]
        chosen = chosen.masked_fill(~torch.isfinite(score.gather(-1, chosen)), -1)
        expected[:, : chosen.shape[1]].copy_(chosen)
        torch.testing.assert_close(selected[local_row : local_row + 1], expected)
    full_kv = (torch.randn(physical, 576, device="cuda") * 0.1).bfloat16()
    raw = torch.zeros(
        len(lengths) * columns + 1, 128, 576, dtype=torch.bfloat16, device="cuda"
    )
    compute_ops.concat_and_cache_mla(
        full_kv[:, :512], full_kv[:, 512:], raw, params.slot_mapping, "auto", op.scale
    )
    local_query = (torch.randn(physical, 96 // tp, 576, device="cuda") * 0.1).bfloat16()
    sparse = PageRRSparseMlaOp(
        96 // tp,
        512,
        64,
        128,
        128,
        1.0,
        2048,
        parallelism_config=parallel,
        workspace_owner=params.workspace_owner,
    )
    sparse.plan(params, table, inputs)
    sparse.forward(local_query, raw.view(-1, 1, 576), selected)
    actual = sparse.forward(local_query, raw.view(-1, 1, 576), selected)
    # Restore selected indices to request-major rows for an independent oracle.
    full_indices = collective.all_gather(selected, collective.Group.TP)
    for row in (0, 16, 17, logical - 1):
        chosen = full_indices[row][full_indices[row] >= 0].long()
        request_offset = 0 if row < 17 else 17
        kv = full_kv[request_offset + chosen].float()
        score = local_query[row].float() @ kv.T / math.sqrt(192)
        expected = (torch.softmax(score, -1) @ kv[:, :128]).bfloat16()
        torch.testing.assert_close(actual[row], expected, atol=2e-4, rtol=0.015)
    assert torch.isfinite(actual).all().item()
    assert not actual[logical:].count_nonzero().item()


def check_cached_prefill_and_decode(op, parallel):
    """Short Q/long cached history, request isolation and A2A query padding."""
    tp, rank, page, kernel = parallel.tp_size, parallel.tp_rank, 1024, 128
    lengths, prefixes = [17, 131], [65520, 8191]
    logical = sum(lengths)
    physical = math.ceil(logical / tp) * tp
    if physical > logical:
        lengths.append(physical - logical)
        prefixes.append(0)
    extents = [q + p for q, p in zip(lengths, prefixes)]
    columns = math.ceil(max(extents) / (page * tp)) * (page // kernel)
    table = torch.arange(
        1, len(lengths) * columns + 1, device="cuda", dtype=torch.int32
    ).view(len(lengths), columns)
    if physical > logical:
        table[-1].zero_()
    offsets, padding = [0], []
    for request, length in enumerate(lengths):
        padding.extend([request * max(lengths) - offsets[-1]] * length)
        offsets.append(offsets[-1] + length)
    inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.tensor(lengths, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
        input_lengths_device=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        prefix_lengths_device=torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
        cu_seqlens_device=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        padding_offset=torch.tensor(padding, dtype=torch.int32, device="cuda"),
        kv_cache_kernel_block_id_device=table,
        logical_token_count=logical,
        physical_token_count=physical,
    )
    attention = SimpleNamespace(tokens_per_block=page, kernel_tokens_per_block=kernel)
    params = DsaPageRRParams(attention, parallel)
    params.prepare(inputs)
    torch.manual_seed(3105 + parallel.dp_rank)
    raw = torch.zeros(
        len(lengths) * columns + 1, kernel, 128, dtype=torch.bfloat16, device="cuda"
    )
    local = torch.arange(columns * kernel, device="cuda")
    global_pos = ((local // page) * tp + rank) * page + local % page
    current_keys = []
    for request, extent in enumerate(extents):
        history = (torch.randn(extent, 128, device="cuda") * 0.1).bfloat16()
        if request < 2:
            packed = torch.zeros(
                columns * kernel, 128, dtype=torch.bfloat16, device="cuda"
            )
            valid = global_pos < extent
            packed[valid] = history[global_pos[valid]]
            raw[table[request].long()] = packed.view(columns, kernel, 128)
        current_keys.append(history[prefixes[request] :])
    cache = SimpleNamespace(kv_cache_base=raw.view(torch.uint8))
    query = (torch.randn(physical, 32, 128, device="cuda") * 0.1).bfloat16()
    gates = torch.rand(physical, 32, 1, device="cuda") * 0.01
    op.encode_q_k(query, torch.cat(current_keys), cache, params, inputs, params)
    unchunked = op.compute_topk(query, gates, cache, params, inputs, params).clone()
    force_independent_kv_chunks(params, force_sfa=True)
    actual = op.compute_topk(query, gates, cache, params, inputs, params)
    assert_topk_members(actual, unchunked)
    reference = Bf16DraftIndexerOp(
        SimpleNamespace(
            indexer_use_fp8=False, index_rope_dim=0, index_head_dim=128, index_topk=2048
        )
    )
    assert_topk_members(
        actual, reference.compute_topk(query, gates, cache, params, inputs, params)
    )
    counts = (actual >= 0).sum(-1)
    torch.testing.assert_close(
        counts, (params.query_positions + 1).clamp(0, 2048).to(counts.dtype)
    )

    # Force genuine multi-block SFA on long cached history while preserving
    # the entire CP-local query batch, including its ragged padding rows.
    full_kv = [(torch.randn(n, 576, device="cuda") * 0.1).bfloat16() for n in extents]
    latent_cache = torch.zeros(
        len(lengths) * columns + 1, kernel, 576, dtype=torch.bfloat16, device="cuda"
    )
    for request, extent in enumerate(extents[:2]):
        packed = torch.zeros(columns * kernel, 576, dtype=torch.bfloat16, device="cuda")
        valid = global_pos < extent
        packed[valid] = full_kv[request][global_pos[valid]]
        latent_cache[table[request].long()] = packed.view(columns, kernel, 576)
    local_query = (torch.randn(physical, 96 // tp, 576, device="cuda") * 0.1).bfloat16()
    sparse = PageRRSparseMlaOp(
        96 // tp,
        512,
        64,
        128,
        128,
        1.0,
        2048,
        parallelism_config=parallel,
        workspace_owner=params.workspace_owner,
    )
    sparse.plan(params, table, inputs)
    from unittest.mock import patch

    with patch.object(torch, "bmm", wraps=torch.bmm) as value_up:
        output = sparse.forward(local_query, latent_cache.view(-1, 1, 576), actual)
        assert value_up.call_count == 1
    full_indices = collective.all_gather(actual, collective.Group.TP)
    for row in (0, 16, 17, logical - 1):
        request = 0 if row < 17 else 1
        chosen = full_indices[row][full_indices[row] >= 0].long()
        history = full_kv[request][chosen].float()
        scores = local_query[row].float() @ history.T / math.sqrt(192)
        expected = (scores.softmax(-1) @ history[:, :128]).bfloat16()
        torch.testing.assert_close(output[row], expected, atol=0.0005, rtol=0.03)
    assert not output[logical:].count_nonzero().item()
    assert params.sfa_execution.shapes["running"][0][-1] == 512

    batch, queries, extent = 35, 4, 4097
    columns = math.ceil(extent / (page * tp)) * (page // kernel)
    table = torch.arange(1, batch * columns + 1, device="cuda", dtype=torch.int32).view(
        batch, columns
    )
    inputs = SimpleNamespace(
        is_prefill=False,
        is_target_verify=True,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.full((batch,), queries, dtype=torch.int32),
        physical_token_count=batch * queries,
        prefix_lengths_device=torch.full(
            (batch,), extent - queries, dtype=torch.int32, device="cuda"
        ),
        sequence_lengths_plus_1_device=torch.full(
            (batch,), extent, dtype=torch.int32, device="cuda"
        ),
        kv_cache_kernel_block_id_device=table,
    )
    params = DsaPageRRParams(attention, parallel)
    params.prepare(inputs)
    cache = SimpleNamespace(
        kv_cache_base=(
            torch.randn(batch * columns + 1, kernel, 128, device="cuda") * 0.1
        )
        .bfloat16()
        .view(torch.uint8)
    )
    query = (torch.randn(batch * queries, 32, 128, device="cuda") * 0.1).bfloat16()
    gates = torch.rand(batch * queries, 32, 1, device="cuda") * 0.01

    actual = op.compute_topk(query, gates, cache, params, inputs, params)
    assert_topk_members(
        actual, reference.compute_topk(query, gates, cache, params, inputs, params)
    )


def check_mtp_candidate_overwrite(op, parallel):
    """Synthetic accept lengths 0..3 hide rejected slots and overwrite them.

    This uses the real Indexer/SFA cache writers and Graph replay. The native
    stream-processor test separately checks how accepted lengths are selected;
    this fixture supplies those lengths rather than changing model sampling.
    """
    tp, rank, batch, width = parallel.tp_size, parallel.tp_rank, 4, 4
    page, kernel = 1024, 128
    prefixes = torch.tensor([1022, 1023, 8191, 65535], dtype=torch.int32, device="cuda")
    accepts = torch.arange(1, 5, dtype=torch.int32, device="cuda")
    columns = math.ceil(65540 / (page * tp)) * (page // kernel)
    table = torch.arange(1, batch * columns + 1, dtype=torch.int32, device="cuda").view(
        batch, columns
    )
    # Graph inputs reserve global-length padding around compact owner pages.
    padded_table = torch.zeros(
        batch, 69 * (page // kernel), dtype=torch.int32, device="cuda"
    )
    padded_table[:, :columns].copy_(table)
    attention = SimpleNamespace(tokens_per_block=page, kernel_tokens_per_block=kernel)
    verify_inputs = SimpleNamespace(
        is_prefill=False,
        is_target_verify=True,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.full((batch,), width, dtype=torch.int32),
        physical_token_count=batch * width,
        prefix_lengths_device=prefixes.clone(),
        sequence_lengths_plus_1_device=prefixes + width,
        kv_cache_kernel_block_id_device=padded_table,
    )
    verify = DsaPageRRParams(attention, parallel, max_seq_len=69632)
    verify.prepare(verify_inputs)
    assert verify.block_table.shape[1] == columns
    index_cache = SimpleNamespace(
        kv_cache_base=torch.zeros(
            batch * columns + 1, kernel, 256, dtype=torch.uint8, device="cuda"
        )
    )
    latent = torch.zeros(
        batch * columns + 1, kernel, 576, dtype=torch.bfloat16, device="cuda"
    )
    # A separate draft pool must remain untouched by target verification.
    draft_index = torch.zeros_like(index_cache.kv_cache_base)
    draft_latent = torch.zeros_like(latent)
    candidate_values = torch.arange(
        1, batch * width + 1, device="cuda", dtype=torch.float32
    ).bfloat16()
    candidate_keys = candidate_values[:, None].expand(-1, 128).contiguous()
    candidate_kv = candidate_values[:, None].expand(-1, 576).contiguous()
    candidate_q = torch.ones(
        batch * width, 32, 128, device="cuda", dtype=torch.bfloat16
    )
    op.encode_q_k(
        candidate_q, candidate_keys, index_cache, verify, verify_inputs, verify
    )
    compute_ops.concat_and_cache_mla(
        candidate_kv[:, :512],
        candidate_kv[:, 512:],
        latent,
        verify.slot_mapping,
        "auto",
        op.scale,
    )

    next_position = prefixes + accepts
    inputs = SimpleNamespace(
        is_prefill=False,
        is_target_verify=False,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=torch.ones(batch, dtype=torch.int32),
        physical_token_count=batch,
        prefix_lengths_device=next_position.clone(),
        sequence_lengths_plus_1_device=next_position + 1,
        kv_cache_kernel_block_id_device=padded_table,
    )
    params = DsaPageRRParams(attention, parallel, max_seq_len=69632)
    params.prepare(inputs)
    assert params.block_table.shape[1] == columns
    query = torch.ones(batch, 32, 128, dtype=torch.bfloat16, device="cuda")
    gates = torch.ones(batch, 32, 1, device="cuda", dtype=torch.float32) / 32
    replacement = (
        torch.arange(batch, device="cuda", dtype=torch.float32) + 32
    ).bfloat16()
    replacement_keys = replacement[:, None].expand(-1, 128).contiguous()
    replacement_kv = replacement[:, None].expand(-1, 576).contiguous()
    sparse_query = torch.zeros(
        batch, 96 // tp, 576, dtype=torch.bfloat16, device="cuda"
    )
    sparse = PageRRSparseMlaOp(
        96 // tp,
        512,
        64,
        128,
        kernel,
        1.0,
        2048,
        parallelism_config=parallel,
        workspace_owner=params.workspace_owner,
    )
    sparse.plan(params, table, inputs)

    def execute():
        op.encode_q_k(query, replacement_keys, index_cache, params, inputs, params)
        compute_ops.concat_and_cache_mla(
            replacement_kv[:, :512],
            replacement_kv[:, 512:],
            latent,
            params.slot_mapping,
            "auto",
            op.scale,
        )
        selected = op.compute_topk(query, gates, index_cache, params, inputs, params)
        return selected, sparse.forward(sparse_query, latent.view(-1, 1, 576), selected)

    execute()  # one representative warmup; no measured performance
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        selected, output = execute()
    pointers = (params.positions_d.data_ptr(), params.slot_mapping.data_ptr())
    params.prepare(inputs, forbid_realloc=True)
    assert pointers == (params.positions_d.data_ptr(), params.slot_mapping.data_ptr())
    graph.replay()
    torch.cuda.synchronize()
    reference_params = DsaPageRRParams(attention, parallel)
    reference_params.prepare(inputs)
    unbounded = op.compute_topk(
        query, gates, index_cache, reference_params, inputs, reference_params
    )
    torch.testing.assert_close(selected, unbounded)
    for row, (prefix, accepted, position) in enumerate(
        zip(prefixes.tolist(), accepts.tolist(), next_position.tolist())
    ):
        chosen = selected[row][selected[row] >= 0].long()
        assert chosen.numel() == min(position + 1, 2048)
        assert (chosen <= position).all().item()  # rejected candidates remain invisible
        for offset in range(accepted):
            assert (chosen == prefix + offset).any().item()
        assert (chosen == position).any().item()
        values = (
            candidate_values[row * width : row * width + accepted].float().sum()
            + replacement[row].float()
        )
        expected = torch.full_like(output[row], (values / chosen.numel()).item())
        torch.testing.assert_close(output[row], expected, atol=2e-4, rtol=0.015)
        # Ownership translation must overwrite the first rejected candidate,
        # preserving the accepted prefix even across a physical page boundary.
        slot = params.slot_mapping[row].item()
        if slot >= 0:
            torch.testing.assert_close(latent.view(-1, 576)[slot], replacement_kv[row])
            torch.testing.assert_close(
                op.cache_view(index_cache, kernel).view(-1, 128)[slot],
                replacement_keys[row],
            )
    assert not draft_index.count_nonzero().item()
    assert not draft_latent.count_nonzero().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    world_rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel = ParallelismConfig()
    parallel.tp_size, parallel.dp_size, parallel.ep_size = (
        args.tp_size,
        8 // args.tp_size,
        8,
    )
    parallel.world_size = parallel.local_world_size = 8
    parallel.world_rank = parallel.local_rank = world_rank
    parallel.ffn_sp_size = args.tp_size
    parallel.prefill_cp_config.kv_cache_sharded = True
    nccl = NcclCommConfig()
    nccl.nccl_ip = os.environ["MASTER_ADDR"]
    collective.init_distributed_environment(
        parallel, nccl, int(os.environ["MASTER_PORT"]), disable_custom_all_reduce=True
    )
    op = Bf16DraftIndexerOp(
        SimpleNamespace(
            indexer_use_fp8=False, index_rope_dim=0, index_head_dim=128, index_topk=2048
        )
    )
    stages = (
        ("score", lambda: check_score_kernel()),
        ("topk_ties_graph", lambda: check_topk_mask_and_ties(op)),
        (
            "score_dcp_graph",
            lambda: check_score_dcp(op, parallel.tp_size, parallel.tp_rank),
        ),
        ("sfa_dcp_graph", lambda: check_sparse_attention(parallel)),
        ("forced_prefill_kv_chunks", lambda: check_prefill(op, parallel)),
        (
            "cached_prefill_decode",
            lambda: check_cached_prefill_and_decode(op, parallel),
        ),
        (
            "mtp_candidate_overwrite_graph",
            lambda: check_mtp_candidate_overwrite(op, parallel),
        ),
    )
    for name, check in stages:
        print(
            json.dumps({"rank": world_rank, "stage": name, "state": "start"}),
            flush=True,
        )
        check()
        dist.barrier()
        print(
            json.dumps({"rank": world_rank, "stage": name, "state": "passed"}),
            flush=True,
        )
    dist.barrier()
    print(
        json.dumps(
            {
                "world_rank": world_rank,
                "tp": parallel.tp_size,
                "dp_owner": parallel.dp_rank,
                "dsa_contract": "passed",
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
