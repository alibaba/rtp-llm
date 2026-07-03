"""DSV4 CP distributed attention correctness baseline.

This file is the module-level oracle for the future symmetric-buffer
``dsv4_cp_distributed_prefill_attention`` op.  It intentionally does not depend
on the full 800GB model.  Instead it builds deterministic HCA/CSA attention
fixtures that preserve the production semantics the fused op must match:

* CP size 8 and rank-local query rows.
* Varlen batch with mixed prefix/input lengths.
* HCA dense compressed history plus SWA window.
* CSA indexer score/topk plus SWA window.
* Attention sink in the softmax denominator.

The pure PyTorch oracle runs by default and pins the expected semantics.  Once
the custom op is wired into ``rtp_llm_ops.dsv4_cp_distributed_prefill_attention``,
the distributed candidate tests can be enabled under an 8-rank torchrun/Bazel
environment and will compare rank-local op outputs against this oracle.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.distributed_attention_buffer import (
    Dsv4CpAttentionBufferSpec,
    get_or_create_dsv4_cp_attention_buffer,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_slots_to_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
    WorkspaceMeta,
)


SWA_HEAD_DIM = 512
SWA_ENTRY_BYTES = 584
SWA_BLOCK_SIZE = 256
COMPRESSOR_STATE_BLOCK_SIZE = 16
COMPRESSOR_KV_BLOCK_SIZE = 16
INDEXER_ENTRY_BYTES = 132
INDEXER_HEAD_DIM = 128


@dataclass(frozen=True)
class AttentionCase:
    name: str
    compress_ratio: int
    prefix_lengths: tuple[int, ...]
    input_lengths: tuple[int, ...]
    window_size: int
    compressed_topk: int
    n_heads: int = 4
    head_dim: int = 16
    index_heads: int = 4
    index_dim: int = 8
    cp_size: int = 8


@dataclass
class AttentionFixture:
    case: AttentionCase
    q: torch.Tensor
    kv_by_req: list[torch.Tensor]
    indexer_q: torch.Tensor
    indexer_k_by_req: list[torch.Tensor]
    attn_sink: torch.Tensor
    req_id_per_token: torch.Tensor
    position_ids: torch.Tensor


def _make_case_fixture(
    case: AttentionCase,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> AttentionFixture:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260630 + int(case.compress_ratio))
    total_q = sum(case.input_lengths)

    q = torch.randn(
        total_q,
        case.n_heads,
        case.head_dim,
        generator=gen,
        dtype=dtype,
    ).to(device)
    indexer_q = torch.randn(
        total_q,
        case.index_heads,
        case.index_dim,
        generator=gen,
        dtype=dtype,
    ).to(device)
    attn_sink = (
        torch.randn(case.n_heads, generator=gen, dtype=torch.float32).to(device) * 0.1
    )

    kv_by_req: list[torch.Tensor] = []
    indexer_k_by_req: list[torch.Tensor] = []
    req_ids: list[int] = []
    positions: list[int] = []
    for req, (prefix_len, input_len) in enumerate(
        zip(case.prefix_lengths, case.input_lengths)
    ):
        total_len = prefix_len + input_len
        kv = torch.randn(
            total_len,
            case.n_heads,
            case.head_dim,
            generator=gen,
            dtype=dtype,
        ).to(device)
        indexer_k = torch.randn(
            max(total_len // max(case.compress_ratio, 1), 1),
            case.index_heads,
            case.index_dim,
            generator=gen,
            dtype=dtype,
        ).to(device)
        kv_by_req.append(kv)
        indexer_k_by_req.append(indexer_k)
        req_ids.extend([req] * input_len)
        positions.extend(range(prefix_len, prefix_len + input_len))

    return AttentionFixture(
        case=case,
        q=q,
        kv_by_req=kv_by_req,
        indexer_q=indexer_q,
        indexer_k_by_req=indexer_k_by_req,
        attn_sink=attn_sink,
        req_id_per_token=torch.tensor(req_ids, dtype=torch.long, device=device),
        position_ids=torch.tensor(positions, dtype=torch.long, device=device),
    )


def _compressed_source_positions(q_pos: int, ratio: int) -> list[int]:
    # Compressed KV entry c corresponds to the boundary token (c + 1) * ratio - 1.
    return [(c + 1) * ratio - 1 for c in range((q_pos + 1) // ratio)]


def _hca_compressed_indices(q_pos: int, ratio: int, topk: int) -> list[int]:
    return list(range(min((q_pos + 1) // ratio, topk)))


def _csa_compressed_indices(
    fixture: AttentionFixture,
    row: int,
    req: int,
    q_pos: int,
) -> list[int]:
    case = fixture.case
    valid = (q_pos + 1) // case.compress_ratio
    valid = min(valid, fixture.indexer_k_by_req[req].shape[0])
    if valid <= 0:
        return []
    q = fixture.indexer_q[row]  # [index_heads, index_dim]
    k = fixture.indexer_k_by_req[req][:valid]  # [valid, index_heads, index_dim]
    scores = torch.einsum("hd,nhd->n", q.float(), k.float())
    k_eff = min(case.compressed_topk, valid)
    # Stable sort gives deterministic tie behavior for the oracle.
    order = torch.argsort(scores, descending=True, stable=True)[:k_eff]
    return [int(x) for x in order.cpu().tolist()]


def _attention_over_keys(
    q_row: torch.Tensor,
    keys: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    if keys.numel() == 0:
        return torch.zeros_like(q_row)
    scale = q_row.shape[-1] ** -0.5
    logits = torch.einsum("hd,nhd->hn", q_row.float(), keys.float()) * scale
    sink = attn_sink.float().view(-1, 1)
    lse = torch.logsumexp(torch.cat([logits, sink], dim=1), dim=1)
    prob = torch.exp(logits - lse.view(-1, 1))
    return torch.einsum("hn,nhd->hd", prob, keys.float()).to(q_row.dtype)


def reference_attention(fixture: AttentionFixture) -> torch.Tensor:
    """Full semantic oracle for the fused CP attention op."""
    case = fixture.case
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]

        if case.compress_ratio == 128:
            compressed_indices = _hca_compressed_indices(
                q_pos, case.compress_ratio, case.compressed_topk
            )
        else:
            compressed_indices = _csa_compressed_indices(fixture, row, req, q_pos)

        key_rows: list[torch.Tensor] = []
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c in compressed_indices:
            if c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])

        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def _pack_flat_indexer_cache(k: torch.Tensor) -> torch.Tensor:
    """Pack flat INDEXER K rows into the production 132B layout."""
    assert k.dim() == 2
    N, D = int(k.shape[0]), int(k.shape[1])
    assert D <= INDEXER_HEAD_DIM
    cache = torch.zeros(
        N,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=k.device,
    )
    scale = torch.ones(N, dtype=torch.float32, device=k.device)
    cache[:, :D].copy_(k.to(torch.float8_e4m3fn).view(torch.uint8))
    cache[:, INDEXER_HEAD_DIM : INDEXER_HEAD_DIM + 4].copy_(
        scale.view(torch.uint8).view(N, 4)
    )
    return cache.contiguous()


def _pack_paged_indexer_pool(
    k: torch.Tensor,
    slots: torch.Tensor,
    *,
    block_size: int = COMPRESSOR_KV_BLOCK_SIZE,
) -> torch.Tensor:
    assert k.dim() == 2
    N, D = int(k.shape[0]), int(k.shape[1])
    assert D <= INDEXER_HEAD_DIM
    max_slot = int(slots.clamp(min=0).max().item()) if int(slots.numel()) > 0 else 0
    num_blocks = max(1, max_slot // int(block_size) + 1)
    pool = torch.empty(
        num_blocks,
        int(block_size),
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=k.device,
    )
    pool.fill_(0x5A)
    k_bytes = k.to(torch.float8_e4m3fn).view(torch.uint8)
    one_scale = torch.ones(1, dtype=torch.float32, device=k.device).view(torch.uint8)
    for row in range(N):
        slot = int(slots[row].item())
        if slot < 0:
            continue
        block = slot // int(block_size)
        off = slot % int(block_size)
        raw = pool[block].reshape(-1)
        raw[off * INDEXER_HEAD_DIM : off * INDEXER_HEAD_DIM + D].copy_(
            k_bytes[row, :D]
        )
        scale_start = int(block_size) * INDEXER_HEAD_DIM + off * 4
        raw[scale_start : scale_start + 4].copy_(one_scale)
    return pool.contiguous()


def _unpack_flat_indexer_cache(cache: torch.Tensor, dim: int) -> torch.Tensor:
    k_fp8 = cache[:, :dim].contiguous().view(torch.float8_e4m3fn).float()
    scale = cache[:, INDEXER_HEAD_DIM : INDEXER_HEAD_DIM + 4].contiguous()
    scale = scale.view(torch.float32).view(-1, 1)
    return k_fp8 * scale


def _unpack_paged_indexer_pool(
    pool: torch.Tensor,
    slots: torch.Tensor,
    dim: int,
    *,
    block_size: int = COMPRESSOR_KV_BLOCK_SIZE,
) -> torch.Tensor:
    out = torch.empty(int(slots.numel()), int(dim), dtype=torch.float32, device=pool.device)
    raw = pool.reshape(int(pool.shape[0]), -1)
    for row in range(int(slots.numel())):
        slot = int(slots[row].item())
        block = slot // int(block_size)
        off = slot % int(block_size)
        token = raw[block, off * INDEXER_HEAD_DIM : off * INDEXER_HEAD_DIM + int(dim)]
        scale_bytes = raw[
            block,
            int(block_size) * INDEXER_HEAD_DIM + off * 4 : int(block_size) * INDEXER_HEAD_DIM + off * 4 + 4,
        ]
        out[row].copy_(token.contiguous().view(torch.float8_e4m3fn).float() * scale_bytes.contiguous().view(torch.float32))
    return out


def _pack_flat_model1_cache(k: torch.Tensor) -> torch.Tensor:
    assert k.dim() == 2
    N, D = int(k.shape[0]), int(k.shape[1])
    assert D <= SWA_HEAD_DIM
    padded = torch.zeros(N, SWA_HEAD_DIM, dtype=torch.bfloat16, device=k.device)
    padded[:, :D].copy_(k.to(torch.bfloat16))
    slots = torch.arange(N, dtype=torch.long, device=k.device)
    cache = _alloc_swa_cache(max(N - 1, 0), k.device)
    cache.fill_(0x5A)
    quantize_and_insert_k_cache(padded.contiguous(), cache, slots)
    return cache.view(-1, SWA_ENTRY_BYTES)[:N].contiguous()


def _unpack_flat_model1_cache(cache: torch.Tensor, dim: int) -> torch.Tensor:
    N = int(cache.shape[0])
    out = torch.empty(N, dim, dtype=torch.float32, device=cache.device)
    for d in range(dim):
        if d < SWA_HEAD_DIM - 64:
            encoded = cache[:, 576 + d // 64].to(torch.float32)
            scale = torch.pow(2.0, encoded - 127.0)
            vals = cache[:, d].contiguous().view(torch.float8_e4m3fn).float()
            out[:, d] = vals * scale
        else:
            off = 448 + (d - 448) * 2
            out[:, d] = (
                cache[:, off : off + 2]
                .contiguous()
                .view(torch.bfloat16)
                .float()
                .reshape(N)
            )
    return out


def reference_attention_with_fp8_indexer(
    fixture: AttentionFixture,
    indexer_k_cache: torch.Tensor,
    weights: torch.Tensor,
    cu_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference matching the op's optional FP8 INDEXER_KV score/topk path."""
    case = fixture.case
    k_flat = _unpack_flat_indexer_cache(indexer_k_cache, case.index_dim)
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]
        start = int(cu_lens[req].item())
        end = int(cu_lens[req + 1].item())
        valid = min((q_pos + 1) // case.compress_ratio, end - start)
        compressed_indices: list[int] = []
        if valid > 0:
            q = fixture.indexer_q[row].float()
            k = k_flat[start : start + valid].float()
            w = weights[row].float()
            logits = torch.einsum("hd,nd->hn", q, k).clamp_min_(0.0)
            scores = (logits * w.view(-1, 1)).sum(dim=0)
            order = torch.argsort(scores, descending=True, stable=True)
            compressed_indices = [
                int(x) for x in order[: min(case.compressed_topk, valid)].cpu().tolist()
            ]

        key_rows: list[torch.Tensor] = []
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c in compressed_indices:
            if c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])

        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_packed_cmp_cache(
    fixture: AttentionFixture,
    cmp_cache: torch.Tensor,
    cu_lens: torch.Tensor,
) -> torch.Tensor:
    case = fixture.case
    cmp_k = _unpack_flat_model1_cache(cmp_cache, case.head_dim)
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]
        start = int(cu_lens[req].item())
        end = int(cu_lens[req + 1].item())
        valid = min((q_pos + 1) // case.compress_ratio, end - start)
        if case.compress_ratio == 128:
            compressed_indices = list(range(min(valid, case.compressed_topk)))
        else:
            compressed_indices = _csa_compressed_indices(fixture, row, req, q_pos)

        key_rows: list[torch.Tensor] = []
        for c in compressed_indices:
            if 0 <= c < valid:
                shared = cmp_k[start + c].to(kv.dtype)
                key_rows.append(shared.view(1, -1).expand(case.n_heads, -1))
        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])
        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_indexer_k_rows(
    fixture: AttentionFixture,
    k_flat: torch.Tensor,
    weights: torch.Tensor,
    cu_lens: torch.Tensor,
) -> torch.Tensor:
    case = fixture.case
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]
        start = int(cu_lens[req].item())
        end = int(cu_lens[req + 1].item())
        valid = min((q_pos + 1) // case.compress_ratio, end - start)
        compressed_indices: list[int] = []
        if valid > 0:
            q = fixture.indexer_q[row].float()
            k = k_flat[start : start + valid].float()
            w = weights[row].float()
            logits = torch.einsum("hd,nd->hn", q, k).clamp_min_(0.0)
            scores = (logits * w.view(-1, 1)).sum(dim=0)
            order = torch.argsort(scores, descending=True, stable=True)
            compressed_indices = [
                int(x) for x in order[: min(case.compressed_topk, valid)].cpu().tolist()
            ]

        key_rows: list[torch.Tensor] = []
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c in compressed_indices:
            if c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])

        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_explicit_csa_topk(
    fixture: AttentionFixture,
    topk: torch.Tensor,
) -> torch.Tensor:
    case = fixture.case
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]

        key_rows: list[torch.Tensor] = []
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c_t in topk[row]:
            c = int(c_t.item())
            if c >= 0 and c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])
        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_cmp_k_rows(
    fixture: AttentionFixture,
    cmp_k_by_req: list[torch.Tensor],
) -> torch.Tensor:
    case = fixture.case
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        kv = fixture.kv_by_req[req]
        cmp_k = cmp_k_by_req[req]
        valid = min((q_pos + 1) // case.compress_ratio, int(cmp_k.shape[0]))
        if case.compress_ratio == 128:
            compressed_indices = list(range(min(valid, case.compressed_topk)))
        else:
            compressed_indices = _csa_compressed_indices(fixture, row, req, q_pos)

        key_rows: list[torch.Tensor] = []
        for c in compressed_indices:
            if 0 <= c < valid:
                shared = cmp_k[c].to(kv.dtype)
                key_rows.append(shared.view(1, -1).expand(case.n_heads, -1))
        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            key_rows.append(kv[pos])
        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_packed_swa_cache(
    fixture: AttentionFixture,
    swa_cache: torch.Tensor,
    cu_lens: torch.Tensor,
) -> torch.Tensor:
    case = fixture.case
    swa_k = _unpack_flat_model1_cache(swa_cache, case.head_dim)
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        prefix_len = int(fixture.case.prefix_lengths[req])
        kv = fixture.kv_by_req[req]

        key_rows: list[torch.Tensor] = []
        if case.compress_ratio == 128:
            compressed_indices = _hca_compressed_indices(
                q_pos, case.compress_ratio, case.compressed_topk
            )
        else:
            compressed_indices = _csa_compressed_indices(fixture, row, req, q_pos)
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c in compressed_indices:
            if c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            if pos < prefix_len:
                shared = swa_k[int(cu_lens[req].item()) + pos].to(kv.dtype)
                key_rows.append(shared.view(1, -1).expand(case.n_heads, -1))
            else:
                key_rows.append(kv[pos])
        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def reference_attention_with_swa_suffix_rows(
    fixture: AttentionFixture,
    swa_suffix_by_req: list[torch.Tensor],
    gather_lens: torch.Tensor,
) -> torch.Tensor:
    case = fixture.case
    outputs: list[torch.Tensor] = []
    for row in range(fixture.q.shape[0]):
        req = int(fixture.req_id_per_token[row].item())
        q_pos = int(fixture.position_ids[row].item())
        prefix_len = int(fixture.case.prefix_lengths[req])
        kv = fixture.kv_by_req[req]

        key_rows: list[torch.Tensor] = []
        if case.compress_ratio == 128:
            compressed_indices = _hca_compressed_indices(
                q_pos, case.compress_ratio, case.compressed_topk
            )
        else:
            compressed_indices = _csa_compressed_indices(fixture, row, req, q_pos)
        boundary_positions = _compressed_source_positions(q_pos, case.compress_ratio)
        for c in compressed_indices:
            if c < len(boundary_positions) and boundary_positions[c] < kv.shape[0]:
                key_rows.append(kv[boundary_positions[c]])

        P = int(gather_lens[req].item())
        start = prefix_len - P
        suffix = swa_suffix_by_req[req]
        swa_start = max(0, q_pos - case.window_size + 1)
        for pos in range(swa_start, min(q_pos + 1, kv.shape[0])):
            if pos < prefix_len:
                col = pos - start
                shared = suffix[col].to(kv.dtype)
                key_rows.append(shared.view(1, -1).expand(case.n_heads, -1))
            else:
                key_rows.append(kv[pos])
        keys = torch.stack(key_rows, dim=0) if key_rows else kv[:0]
        outputs.append(_attention_over_keys(fixture.q[row], keys, fixture.attn_sink))
    return torch.stack(outputs, dim=0)


def rank_local_rows(total_rows: int, cp_rank: int, cp_size: int) -> torch.Tensor:
    # The op consumes explicit global positions, so this oracle only needs a
    # deterministic rank-local partition for module-level validation.
    if int(cp_rank) >= int(total_rows):
        return torch.empty((0,), dtype=torch.long)
    return torch.arange(cp_rank, total_rows, cp_size, dtype=torch.long)


def _pad_for_candidate_op(
    fixture: AttentionFixture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    case = fixture.case
    B = len(fixture.kv_by_req)
    max_kv_len = max(kv.shape[0] for kv in fixture.kv_by_req)
    max_index_len = max(k.shape[0] for k in fixture.indexer_k_by_req)

    kv = torch.zeros(
        B,
        max_kv_len,
        case.n_heads,
        case.head_dim,
        dtype=fixture.q.dtype,
        device=fixture.q.device,
    )
    indexer_k = torch.zeros(
        B,
        max_index_len,
        case.index_heads,
        case.index_dim,
        dtype=fixture.q.dtype,
        device=fixture.q.device,
    )
    for req, src in enumerate(fixture.kv_by_req):
        kv[req, : src.shape[0]].copy_(src)
    for req, src in enumerate(fixture.indexer_k_by_req):
        indexer_k[req, : src.shape[0]].copy_(src)

    prefix_lengths = torch.tensor(
        fixture.case.prefix_lengths, dtype=torch.long, device=fixture.q.device
    )
    input_lengths = torch.tensor(
        fixture.case.input_lengths, dtype=torch.long, device=fixture.q.device
    )
    return kv.contiguous(), indexer_k.contiguous(), prefix_lengths, input_lengths


def _benchmark(fn, *, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(iters, 1)


def _candidate_op():
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
    except Exception:
        return None
    return getattr(rtp_llm_ops, "dsv4_cp_distributed_prefill_attention", None)


def _assert_attention_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bfloat16 or expected.dtype == torch.bfloat16:
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def _splitk_actual_tiling(
    total_keys: int,
    requested_keys_per_block: int,
) -> tuple[int, int]:
    keys = max(int(total_keys), 1)
    keys_per_block = max(int(requested_keys_per_block), 64)
    if keys >= 2048:
        keys_per_block = max(keys_per_block, 256)
    elif keys >= 1024:
        keys_per_block = max(keys_per_block, 128)
    key_blocks = (keys + keys_per_block - 1) // keys_per_block
    if key_blocks == 8:
        keys_per_block = max(keys_per_block, (keys + 6) // 7)
        keys_per_block = ((keys_per_block + 7) // 8) * 8
        key_blocks = (keys + keys_per_block - 1) // keys_per_block
    return keys_per_block, max(key_blocks, 1)


def _csa_topk_pair_better(lhs: tuple[float, int], rhs: tuple[float, int]) -> bool:
    return lhs[0] > rhs[0] or (lhs[0] == rhs[0] and lhs[1] < rhs[1])


def _csa_streaming_tile_topk(scores: list[float], k: int, tile: int = 8) -> list[int]:
    selected: list[tuple[float, int]] = []
    for base in range(0, len(scores), tile):
        tile_pairs = [(scores[idx], idx) for idx in range(base, min(base + tile, len(scores)))]
        if len(selected) < k:
            fill_count = min(k - len(selected), len(tile_pairs))
            selected.extend(tile_pairs[:fill_count])
            selected.sort(key=lambda x: (x[0], -x[1]))
            tile_pairs = tile_pairs[fill_count:]
        if not tile_pairs:
            continue
        worst = selected[0]
        tile_best = max(tile_pairs, key=lambda x: (x[0], -x[1]))
        if not _csa_topk_pair_better(tile_best, worst):
            continue
        worst_order = sorted(range(len(selected)), key=lambda pos: (selected[pos][0], -selected[pos][1]))
        tile_order = sorted(tile_pairs, key=lambda x: (-x[0], x[1]))
        for worst_pos, candidate in zip(worst_order, tile_order):
            if _csa_topk_pair_better(candidate, selected[worst_pos]):
                selected[worst_pos] = candidate
            else:
                break
        selected.sort(key=lambda x: (x[0], -x[1]))
    return [idx for _, idx in sorted(selected, key=lambda x: (-x[0], x[1]))]


def _rank_local_4d_chunk(src: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    local_len = (int(src.size(1)) + int(cp_size) - 1) // int(cp_size)
    out = torch.zeros(
        src.size(0),
        local_len,
        src.size(2),
        src.size(3),
        dtype=src.dtype,
        device=src.device,
    )
    start = int(rank) * local_len
    end = min(start + local_len, int(src.size(1)))
    if start < end:
        out[:, : end - start].copy_(src[:, start:end])
    return out


def _rank_local_2d_chunk(src: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    local_len = (int(src.size(0)) + int(cp_size) - 1) // int(cp_size)
    out = torch.zeros(
        local_len,
        src.size(1),
        dtype=src.dtype,
        device=src.device,
    )
    start = int(rank) * local_len
    end = min(start + local_len, int(src.size(0)))
    if start < end:
        out[: end - start].copy_(src[start:end])
    return out


def _rank_local_1d_chunk(
    src: torch.Tensor,
    rank: int,
    cp_size: int,
    *,
    pad_value: int,
) -> torch.Tensor:
    local_len = (int(src.size(0)) + int(cp_size) - 1) // int(cp_size)
    out = torch.full(
        (local_len,),
        int(pad_value),
        dtype=src.dtype,
        device=src.device,
    )
    start = int(rank) * local_len
    end = min(start + local_len, int(src.size(0)))
    if start < end:
        out[: end - start].copy_(src[start:end])
    return out


def _alloc_swa_cache(max_slot: int, device: torch.device) -> torch.Tensor:
    num_blocks = max(1, int(max_slot) // SWA_BLOCK_SIZE + 1)
    return torch.empty(
        num_blocks,
        SWA_BLOCK_SIZE,
        SWA_ENTRY_BYTES,
        dtype=torch.uint8,
        device=device,
    )


def _make_swa_fixture(
    num_tokens: int,
    device: torch.device,
    *,
    skipped: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260701 + int(num_tokens))
    swa_k = (
        torch.randn(
            num_tokens,
            SWA_HEAD_DIM,
            generator=gen,
            dtype=torch.bfloat16,
        )
        * 0.05
    ).to(device)
    slot_mapping = torch.arange(
        SWA_BLOCK_SIZE,
        SWA_BLOCK_SIZE + num_tokens,
        dtype=torch.long,
        device=device,
    )
    for idx in skipped:
        if 0 <= int(idx) < num_tokens:
            slot_mapping[int(idx)] = -1
    max_slot = int(slot_mapping.clamp(min=0).max().item())
    ref_cache = _alloc_swa_cache(max_slot, device)
    actual_cache = torch.empty_like(ref_cache)
    ref_cache.fill_(0x5A)
    actual_cache.fill_(0x5A)
    return swa_k.contiguous(), slot_mapping.contiguous(), ref_cache, actual_cache


def _alloc_compressor_state_cache(
    max_slot: int,
    device: torch.device,
    *,
    block_size: int,
    head_size: int,
) -> torch.Tensor:
    num_blocks = max(1, int(max_slot) // int(block_size) + 1)
    return torch.empty(
        num_blocks,
        int(block_size),
        int(head_size) * 2,
        dtype=torch.float32,
        device=device,
    )


def _make_compressor_state_fixture(
    num_tokens: int,
    head_size: int,
    compress_ratio: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    skipped: tuple[int, ...] = (),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260702 + int(num_tokens) * 17 + int(head_size))
    kv = (
        torch.randn(
            num_tokens,
            head_size,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.05
    ).to(dtype=dtype, device=device)
    score = (
        torch.randn(
            num_tokens,
            head_size,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.05
    ).to(dtype=dtype, device=device)
    ape = (
        torch.randn(
            compress_ratio,
            head_size,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.01
    ).to(device)
    positions = torch.arange(
        7,
        7 + int(num_tokens),
        dtype=torch.long,
        device=device,
    )
    slots = torch.arange(
        COMPRESSOR_STATE_BLOCK_SIZE,
        COMPRESSOR_STATE_BLOCK_SIZE + int(num_tokens),
        dtype=torch.long,
        device=device,
    )
    for idx in skipped:
        if 0 <= int(idx) < int(num_tokens):
            slots[int(idx)] = -1
    max_slot = int(slots.clamp(min=0).max().item())
    ref_cache = _alloc_compressor_state_cache(
        max_slot,
        device,
        block_size=COMPRESSOR_STATE_BLOCK_SIZE,
        head_size=head_size,
    )
    actual_cache = torch.empty_like(ref_cache)
    ref_cache.fill_(-123.0)
    actual_cache.fill_(-123.0)
    return (
        kv.contiguous(),
        score.contiguous(),
        ape.contiguous(),
        positions.contiguous(),
        slots.contiguous(),
        ref_cache,
        actual_cache,
    )


def _alloc_compressor_kv_cache(
    max_slot: int,
    device: torch.device,
    *,
    block_size: int,
    entry_bytes: int,
) -> torch.Tensor:
    num_blocks = max(1, int(max_slot) // int(block_size) + 1)
    return torch.empty(
        num_blocks,
        int(block_size),
        int(entry_bytes),
        dtype=torch.uint8,
        device=device,
    )


def _make_raw_compressor_kv_write_fixture(
    *,
    num_tokens: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    entry_bytes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    width = int(head_dim) * (2 if overlap else 1)
    base = torch.linspace(
        -0.25,
        0.25,
        int(head_dim),
        dtype=torch.float32,
        device=device,
    ).view(1, int(head_dim))
    if overlap:
        row = torch.cat([base, base], dim=1)
    else:
        row = base
    kv = row.repeat(int(num_tokens), 1).to(torch.bfloat16).contiguous()
    score = torch.zeros(
        int(num_tokens),
        width,
        dtype=torch.bfloat16,
        device=device,
    )
    ape = torch.zeros(
        int(compress_ratio),
        width,
        dtype=torch.float32,
        device=device,
    )
    positions = torch.arange(int(num_tokens), dtype=torch.long, device=device)
    state_slots = torch.arange(
        COMPRESSOR_STATE_BLOCK_SIZE,
        COMPRESSOR_STATE_BLOCK_SIZE + int(num_tokens),
        dtype=torch.long,
        device=device,
    )
    state_cache = _alloc_compressor_state_cache(
        int(state_slots.max().item()),
        device,
        block_size=COMPRESSOR_STATE_BLOCK_SIZE,
        head_size=width,
    )
    state_cache.fill_(-123.0)
    token_to_req = torch.zeros(int(num_tokens), dtype=torch.int32, device=device)
    state_block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    norm_weight = torch.ones(int(head_dim), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(
        int(num_tokens) + int(compress_ratio) + 1,
        int(rope_head_dim),
        dtype=torch.float32,
        device=device,
    )
    half_rope = int(rope_head_dim) // 2
    if half_rope > 0:
        cos_sin_cache[:, :half_rope] = 1.0
    kv_slots = torch.full((int(num_tokens),), -1, dtype=torch.long, device=device)
    kv_slots[int(num_tokens) - 1] = 0
    ref_kv_cache = _alloc_compressor_kv_cache(
        0,
        device,
        block_size=COMPRESSOR_KV_BLOCK_SIZE,
        entry_bytes=entry_bytes,
    )
    actual_kv_cache = torch.empty_like(ref_kv_cache)
    ref_kv_cache.fill_(0x5A)
    actual_kv_cache.fill_(0x5A)
    return {
        "kv": kv,
        "score": score.contiguous(),
        "ape": ape.contiguous(),
        "positions": positions.contiguous(),
        "state_slots": state_slots.contiguous(),
        "state_cache": state_cache,
        "token_to_req": token_to_req.contiguous(),
        "state_block_table": state_block_table.contiguous(),
        "norm_weight": norm_weight.contiguous(),
        "cos_sin_cache": cos_sin_cache.contiguous(),
        "kv_slots": kv_slots.contiguous(),
        "ref_kv_cache": ref_kv_cache,
        "actual_kv_cache": actual_kv_cache,
    }


def _make_cache_read_compressor_kv_write_fixture(
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    entry_bytes: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    width = int(head_dim) * (2 if overlap else 1)
    window_count = int(compress_ratio) * (2 if overlap else 1)
    base = torch.linspace(
        -0.2,
        0.3,
        int(head_dim),
        dtype=torch.float32,
        device=device,
    ).view(1, int(head_dim))
    if overlap:
        row = torch.cat([base, base + 0.125], dim=1)
    else:
        row = base
    kv = row.repeat(window_count, 1).to(torch.bfloat16).contiguous()
    score = torch.zeros(window_count, width, dtype=torch.bfloat16, device=device)
    ape = torch.zeros(int(compress_ratio), width, dtype=torch.float32, device=device)
    positions = torch.arange(window_count, dtype=torch.long, device=device)
    state_slots = torch.full(
        (window_count,),
        -1,
        dtype=torch.long,
        device=device,
    )
    state_cache = _alloc_compressor_state_cache(
        window_count,
        device,
        block_size=window_count,
        head_size=width,
    )
    state_cache.fill_(-123.0)
    for pos in range(window_count):
        row_offset = pos % window_count
        state_cache[1, row_offset, :width].copy_(kv[pos].float())
        state_cache[1, row_offset, width : 2 * width].copy_(score[pos].float())
    token_to_req = torch.zeros(window_count, dtype=torch.int32, device=device)
    state_block_table = torch.ones((1, 1), dtype=torch.int32, device=device)
    norm_weight = torch.ones(int(head_dim), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(
        window_count + int(compress_ratio) + 1,
        int(rope_head_dim),
        dtype=torch.float32,
        device=device,
    )
    half_rope = int(rope_head_dim) // 2
    if half_rope > 0:
        cos_sin_cache[:, :half_rope] = 1.0
    kv_slots = torch.full((window_count,), -1, dtype=torch.long, device=device)
    kv_slots[window_count - 1] = 0
    ref_kv_cache = _alloc_compressor_kv_cache(
        0,
        device,
        block_size=COMPRESSOR_KV_BLOCK_SIZE,
        entry_bytes=entry_bytes,
    )
    actual_kv_cache = torch.empty_like(ref_kv_cache)
    ref_kv_cache.fill_(0x5A)
    actual_kv_cache.fill_(0x5A)
    return {
        "kv": kv,
        "score": score.contiguous(),
        "ape": ape.contiguous(),
        "positions": positions.contiguous(),
        "state_slots": state_slots.contiguous(),
        "state_cache": state_cache,
        "token_to_req": token_to_req.contiguous(),
        "state_block_table": state_block_table.contiguous(),
        "norm_weight": norm_weight.contiguous(),
        "cos_sin_cache": cos_sin_cache.contiguous(),
        "kv_slots": kv_slots.contiguous(),
        "ref_kv_cache": ref_kv_cache,
        "actual_kv_cache": actual_kv_cache,
        "state_tokens_per_block": window_count,
    }


def _make_varlen_raw_compressor_kv_write_fixture(
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    entry_bytes: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    width = int(head_dim) * (2 if overlap else 1)
    per_req_tokens = int(compress_ratio) * (2 if overlap else 1)
    req0_positions = torch.arange(0, per_req_tokens, dtype=torch.long, device=device)
    req1_start = int(compress_ratio) * 25
    req1_positions = torch.arange(
        req1_start,
        req1_start + per_req_tokens,
        dtype=torch.long,
        device=device,
    )
    positions = torch.cat([req0_positions, req1_positions], dim=0).contiguous()
    num_tokens = int(positions.numel())
    base = torch.linspace(
        -0.35,
        0.35,
        int(head_dim),
        dtype=torch.float32,
        device=device,
    )
    req0_row = base
    req1_row = base.flip(0) + 0.0625
    if overlap:
        req0_wide = torch.cat([req0_row, req0_row + 0.03125], dim=0)
        req1_wide = torch.cat([req1_row, req1_row - 0.03125], dim=0)
    else:
        req0_wide = req0_row
        req1_wide = req1_row
    kv = torch.vstack(
        [
            req0_wide.repeat(per_req_tokens, 1),
            req1_wide.repeat(per_req_tokens, 1),
        ]
    ).to(torch.bfloat16).contiguous()
    score = torch.zeros(num_tokens, width, dtype=torch.bfloat16, device=device)
    ape = torch.zeros(int(compress_ratio), width, dtype=torch.float32, device=device)
    state_slots = torch.arange(
        COMPRESSOR_STATE_BLOCK_SIZE,
        COMPRESSOR_STATE_BLOCK_SIZE + num_tokens,
        dtype=torch.long,
        device=device,
    )
    state_cache = _alloc_compressor_state_cache(
        int(state_slots.max().item()),
        device,
        block_size=COMPRESSOR_STATE_BLOCK_SIZE,
        head_size=width,
    )
    state_cache.fill_(-123.0)
    token_to_req = torch.cat(
        [
            torch.zeros(per_req_tokens, dtype=torch.int32, device=device),
            torch.ones(per_req_tokens, dtype=torch.int32, device=device),
        ],
        dim=0,
    ).contiguous()
    state_block_table = torch.zeros((2, 1), dtype=torch.int32, device=device)
    norm_weight = torch.ones(int(head_dim), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(
        int(req1_positions.max().item()) + int(compress_ratio) + 1,
        int(rope_head_dim),
        dtype=torch.float32,
        device=device,
    )
    half_rope = int(rope_head_dim) // 2
    if half_rope > 0:
        cos_sin_cache[:, :half_rope] = 1.0
    kv_slots = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
    kv_slots[per_req_tokens - 1] = 0
    kv_slots[num_tokens - 1] = 1
    seq_start_per_req = torch.tensor(
        [0, req1_start],
        dtype=torch.int32,
        device=device,
    )
    cu_seq_per_req = torch.tensor(
        [0, per_req_tokens, num_tokens],
        dtype=torch.int32,
        device=device,
    )
    ref_kv_cache = _alloc_compressor_kv_cache(
        1,
        device,
        block_size=COMPRESSOR_KV_BLOCK_SIZE,
        entry_bytes=entry_bytes,
    )
    actual_kv_cache = torch.empty_like(ref_kv_cache)
    ref_kv_cache.fill_(0x5A)
    actual_kv_cache.fill_(0x5A)
    return {
        "kv": kv,
        "score": score.contiguous(),
        "ape": ape.contiguous(),
        "positions": positions,
        "state_slots": state_slots.contiguous(),
        "state_cache": state_cache,
        "token_to_req": token_to_req,
        "state_block_table": state_block_table.contiguous(),
        "norm_weight": norm_weight.contiguous(),
        "cos_sin_cache": cos_sin_cache.contiguous(),
        "kv_slots": kv_slots.contiguous(),
        "seq_start_per_req": seq_start_per_req.contiguous(),
        "cu_seq_per_req": cu_seq_per_req.contiguous(),
        "ref_kv_cache": ref_kv_cache,
        "actual_kv_cache": actual_kv_cache,
        "state_tokens_per_block": COMPRESSOR_STATE_BLOCK_SIZE,
    }


class DistributedAttentionOracleTest(unittest.TestCase):
    def _run_oracle_case(self, case: AttentionCase) -> None:
        fixture = _make_case_fixture(case, torch.device("cpu"))
        full = reference_attention(fixture)
        self.assertEqual(
            tuple(full.shape),
            (sum(case.input_lengths), case.n_heads, case.head_dim),
        )

        pieces = []
        rows_all = []
        for rank in range(case.cp_size):
            rows = rank_local_rows(full.shape[0], rank, case.cp_size)
            rows_all.append(rows)
            pieces.append(full.index_select(0, rows))
        restored = torch.empty_like(full)
        for rows, part in zip(rows_all, pieces):
            restored.index_copy_(0, rows, part)
        torch.testing.assert_close(restored, full, rtol=0.0, atol=0.0)

    def test_hca_var_batch_cp8_oracle(self) -> None:
        self._run_oracle_case(
            AttentionCase(
                name="hca_var_batch",
                compress_ratio=128,
                prefix_lengths=(0, 127, 260),
                input_lengths=(9, 7, 5),
                window_size=16,
                compressed_topk=8,
            )
        )

    def test_csa_var_batch_cp8_oracle(self) -> None:
        self._run_oracle_case(
            AttentionCase(
                name="csa_var_batch",
                compress_ratio=4,
                prefix_lengths=(0, 15, 31),
                input_lengths=(9, 6, 5),
                window_size=16,
                compressed_topk=4,
            )
        )

    def test_reference_perf_record_is_available(self) -> None:
        case = AttentionCase(
            name="csa_perf_record",
            compress_ratio=4,
            prefix_lengths=(16, 32, 48, 64),
            input_lengths=(16, 12, 10, 8),
            window_size=16,
            compressed_topk=4,
            n_heads=8,
            head_dim=16,
        )
        fixture = _make_case_fixture(case, torch.device("cpu"))
        ms = _benchmark(lambda: reference_attention(fixture), warmup=1, iters=2)
        self.assertGreater(ms, 0.0)
        if os.environ.get("DSV4_DIST_ATTN_PRINT_PERF", "0") == "1":
            print(json.dumps({"case": case.name, "reference_ms": ms}, sort_keys=True))

    def test_splitk_actual_tiling_avoids_single_tail_wave(self) -> None:
        self.assertEqual(_splitk_actual_tiling(512, 64), (80, 7))
        self.assertEqual(_splitk_actual_tiling(1024, 128), (152, 7))
        self.assertEqual(_splitk_actual_tiling(2048, 256), (296, 7))

        self.assertEqual(_splitk_actual_tiling(511, 64), (80, 7))
        self.assertEqual(_splitk_actual_tiling(513, 64), (64, 9))
        self.assertEqual(_splitk_actual_tiling(897, 64), (64, 15))
        self.assertEqual(_splitk_actual_tiling(1025, 128), (128, 9))
        self.assertEqual(_splitk_actual_tiling(2049, 256), (256, 9))

    def test_csa_streaming_tile_topk_matches_stable_oracle(self) -> None:
        cases = [
            ([0.0, 0.5, 0.5, 0.25, 0.75, 0.75, 0.1, 0.2, 0.75], 3, 8),
            ([float(i % 5) for i in range(23)], 7, 8),
            ([1.0] * 17, 5, 8),
            ([0.1, 0.2, 0.3, 0.4, 9.0, 8.0, 7.0, 6.0, 9.0, 8.5], 4, 8),
            ([float((i * 7) % 11) for i in range(37)], 9, 4),
            ([float(i % 3) for i in range(19)], 11, 16),
        ]
        for scores, k, tile in cases:
            expected = [
                idx
                for _, idx in sorted(
                    [(score, idx) for idx, score in enumerate(scores)],
                    key=lambda x: (-x[0], x[1]),
                )[:k]
            ]
            self.assertEqual(_csa_streaming_tile_topk(scores, k, tile), expected)


class DistributedAttentionCandidateTest(unittest.TestCase):
    def _run_candidate_case(
        self, case: AttentionCase, dtype: torch.dtype = torch.float32
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=dtype)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                indexer_k,
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_hca_rank_local_allclose(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="hca_candidate",
                compress_ratio=128,
                prefix_lengths=(0, 127, 260),
                input_lengths=(9, 7, 5),
                window_size=16,
                compressed_topk=8,
            )
        )

    def test_candidate_cuda_hca_cold_no_compressed_history_allclose(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="hca_cold_no_compressed_history",
                compress_ratio=128,
                prefix_lengths=(0,),
                input_lengths=(15,),
                window_size=128,
                compressed_topk=0,
                n_heads=4,
                head_dim=32,
                index_heads=1,
                index_dim=1,
            )
        )

    def test_candidate_cuda_hca_ncu_large_profile_fixture(self) -> None:
        if os.environ.get("DSV4_DIST_ATTN_NCU_LARGE", "0") != "1":
            self.skipTest("large profile fixture is enabled only for NCU runs")
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        device = torch.device("cuda")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260719)
        T, H, D = 512, 16, 128
        q = torch.randn(T, H, D, generator=gen, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        kv = torch.randn(1, T, 1, D, generator=gen, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        indexer_q = torch.zeros(T, 1, 1, dtype=torch.bfloat16, device=device)
        indexer_k = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=device)
        actual = op(
            q.contiguous(),
            kv.contiguous(),
            indexer_q,
            indexer_k,
            torch.zeros(H, dtype=torch.float32, device=device),
            torch.zeros(T, dtype=torch.long, device=device),
            torch.arange(T, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.tensor([T], dtype=torch.long, device=device),
            torch.arange(T, dtype=torch.long, device=device),
            128,
            64,
            4,
        )
        torch.cuda.synchronize()
        self.assertEqual(tuple(actual.shape), (T, H, D))
        self.assertTrue(torch.isfinite(actual.float()).all())

    def test_candidate_cuda_csa_rank_local_allclose(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="csa_candidate",
                compress_ratio=4,
                prefix_lengths=(0, 15, 31),
                input_lengths=(9, 6, 5),
                window_size=16,
                compressed_topk=4,
            )
        )

    def test_candidate_cuda_hca_bf16_rank_local_allclose(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="hca_candidate_bf16",
                compress_ratio=128,
                prefix_lengths=(0, 127, 260),
                input_lengths=(9, 7, 5),
                window_size=16,
                compressed_topk=8,
            ),
            dtype=torch.bfloat16,
        )

    def test_candidate_cuda_hca_production_grouped_mqa_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_production_grouped_mqa",
            compress_ratio=128,
            prefix_lengths=(0, 130, 260),
            input_lengths=(3, 2, 1),
            window_size=5,
            compressed_topk=3,
            n_heads=128,
            head_dim=512,
            index_heads=1,
            index_dim=1,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        mqa_kv_by_req = [kv[:, :1, :].contiguous() for kv in fixture.kv_by_req]
        fixture.kv_by_req = [
            kv.expand(-1, case.n_heads, -1).contiguous() for kv in mqa_kv_by_req
        ]
        expected = reference_attention(fixture)

        B = len(mqa_kv_by_req)
        max_kv_len = max(int(kv.shape[0]) for kv in mqa_kv_by_req)
        kv = torch.zeros(
            B,
            max_kv_len,
            1,
            case.head_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        for req, src in enumerate(mqa_kv_by_req):
            kv[req, : src.shape[0]].copy_(src)
        indexer_k = torch.zeros(
            B,
            1,
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        prefix_lengths = torch.tensor(
            case.prefix_lengths, dtype=torch.long, device=fixture.q.device
        )
        input_lengths = torch.tensor(
            case.input_lengths, dtype=torch.long, device=fixture.q.device
        )

        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv.contiguous(),
                fixture.indexer_q.contiguous(),
                indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_hca_kh_gt_one_does_not_use_grouped_mqa(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="hca_kh_gt_one_grouped_guard",
                compress_ratio=128,
                prefix_lengths=(0, 130),
                input_lengths=(3, 2),
                window_size=5,
                compressed_topk=3,
                n_heads=8,
                head_dim=512,
                index_heads=1,
                index_dim=1,
            ),
            dtype=torch.bfloat16,
        )

    def test_candidate_cuda_split_k_gate_requires_symmetric_backend(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="splitk_gate_requires_symm",
            compress_ratio=128,
            prefix_lengths=(0,),
            input_lengths=(1,),
            window_size=1,
            compressed_topk=0,
            n_heads=128,
            head_dim=512,
            index_heads=1,
            index_dim=1,
            cp_size=1,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        mqa_kv = fixture.kv_by_req[0][:, :1, :].contiguous()
        rows = torch.arange(1, dtype=torch.long, device=fixture.q.device)
        old_value = os.environ.get("DSV4_CP_ATTENTION_MEGA_SPLITK")
        os.environ["DSV4_CP_ATTENTION_MEGA_SPLITK"] = "1"
        try:
            with self.assertRaisesRegex(RuntimeError, "requires symmetric backend"):
                op(
                    fixture.q.contiguous(),
                    mqa_kv.view(1, 1, 1, case.head_dim).contiguous(),
                    fixture.indexer_q.contiguous(),
                    torch.zeros(1, 1, 1, 1, dtype=fixture.q.dtype, device=fixture.q.device),
                    fixture.attn_sink.contiguous(),
                    fixture.req_id_per_token.contiguous(),
                    fixture.position_ids.contiguous(),
                    torch.tensor(case.prefix_lengths, dtype=torch.long, device=fixture.q.device),
                    torch.tensor(case.input_lengths, dtype=torch.long, device=fixture.q.device),
                    rows,
                    case.compress_ratio,
                    case.window_size,
                    case.compressed_topk,
                )
        finally:
            if old_value is None:
                os.environ.pop("DSV4_CP_ATTENTION_MEGA_SPLITK", None)
            else:
                os.environ["DSV4_CP_ATTENTION_MEGA_SPLITK"] = old_value

    def test_candidate_cuda_hca_mqa_fresh_k_restore_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_mqa_restore",
            compress_ratio=128,
            prefix_lengths=(0, 0),
            input_lengths=(3, 2),
            window_size=4,
            compressed_topk=1,
            n_heads=3,
            head_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        # Production DSv4 prefill has MQA K: one shared K head consumed by all Q
        # heads. Make the oracle fixture match that contract, then pass only the
        # single K head through the op.
        for req, kv in enumerate(fixture.kv_by_req):
            fixture.kv_by_req[req] = kv[:, :1, :].expand(-1, case.n_heads, -1).contiguous()
        expected = reference_attention(fixture)
        _, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        fresh_global = torch.cat(
            [fixture.kv_by_req[req][: case.input_lengths[req], :1, :] for req in range(len(case.input_lengths))],
            dim=0,
        ).contiguous()
        restore = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long, device=fresh_global.device)
        gathered = torch.empty_like(fresh_global)
        gathered.index_copy_(0, restore, fresh_global)
        kv_cu_lens = torch.tensor([0, 3, 5], dtype=torch.long, device=fresh_global.device)
        rows = torch.arange(expected.shape[0], dtype=torch.long, device=fresh_global.device)

        actual = op(
            fixture.q.contiguous(),
            gathered.unsqueeze(0).contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            kv_unpad_restore=restore.contiguous(),
            kv_cu_lens=kv_cu_lens,
        )
        _assert_attention_close(actual, expected)

    def test_candidate_cuda_csa_bf16_rank_local_allclose(self) -> None:
        self._run_candidate_case(
            AttentionCase(
                name="csa_candidate_bf16",
                compress_ratio=4,
                prefix_lengths=(0, 15, 31),
                input_lengths=(9, 6, 5),
                window_size=16,
                compressed_topk=4,
            ),
            dtype=torch.bfloat16,
        )

    def test_candidate_cuda_csa_fp8_indexer_cache_topk_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_fp8_indexer_cache",
            compress_ratio=4,
            prefix_lengths=(0, 8),
            input_lengths=(8, 8),
            window_size=6,
            compressed_topk=2,
            n_heads=2,
            head_dim=8,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260703)
        k_by_req = []
        cu = [0]
        for req, indexer_k_ref in enumerate(fixture.indexer_k_by_req):
            rows = int(indexer_k_ref.shape[0])
            k_req = (
                torch.randn(rows, case.index_dim, generator=gen, dtype=torch.float32)
                * 0.25
                + float(req) * 0.05
            ).to(device=fixture.q.device)
            k_by_req.append(k_req)
            cu.append(cu[-1] + rows)
        k_flat = torch.cat(k_by_req, dim=0).contiguous()
        k_cache = _pack_flat_indexer_cache(k_flat)
        weights = (
            torch.randn(
                fixture.q.shape[0],
                case.index_heads,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.2
            + 1.0
        ).to(device=fixture.q.device).contiguous()
        cu_lens = torch.tensor(cu, dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_fp8_indexer(
            fixture,
            k_cache,
            weights,
            cu_lens,
        )
        dummy_indexer_k = torch.zeros(
            len(case.input_lengths),
            max(int(x.shape[0]) for x in fixture.indexer_k_by_req),
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_cache=k_cache,
                csa_indexer_weights=weights,
                csa_indexer_cu_lens=cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_fp8_indexer_tie_break_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_fp8_indexer_tie_break",
            compress_ratio=4,
            prefix_lengths=(32,),
            input_lengths=(4,),
            window_size=4,
            compressed_topk=3,
            n_heads=8,
            head_dim=16,
            index_heads=64,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        # Every visible compressed candidate gets the same positive score for
        # every query row. Correct tie-break must therefore keep smaller
        # candidate indices.
        flat_k_rows = (case.prefix_lengths[0] + case.input_lengths[0]) // case.compress_ratio
        k_flat = torch.ones(
            flat_k_rows,
            case.index_dim,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        k_cache = _pack_flat_indexer_cache(k_flat)
        fixture.indexer_q.fill_(1.0 / float(case.index_dim))
        weights = torch.ones(
            fixture.q.shape[0],
            case.index_heads,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        cu_lens = torch.tensor([0, flat_k_rows], dtype=torch.long, device=fixture.q.device)

        expected_topk = torch.full(
            (fixture.q.shape[0], case.compressed_topk),
            -1,
            dtype=torch.int32,
            device=fixture.q.device,
        )
        for row in range(fixture.q.shape[0]):
            q_pos = int(fixture.position_ids[row].item())
            visible = min((q_pos + 1) // case.compress_ratio, flat_k_rows)
            count = min(visible, case.compressed_topk)
            if count > 0:
                expected_topk[row, :count] = torch.arange(
                    count,
                    dtype=torch.int32,
                    device=fixture.q.device,
                )
        expected = reference_attention_with_explicit_csa_topk(fixture, expected_topk)

        dummy_indexer_k = torch.zeros(
            1,
            flat_k_rows,
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_cache=k_cache,
                csa_indexer_weights=weights,
                csa_indexer_cu_lens=cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_fp8_indexer_matrix_tile_replacement_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_fp8_indexer_matrix_tile_replacement",
            compress_ratio=4,
            prefix_lengths=(64,),
            input_lengths=(4,),
            window_size=4,
            compressed_topk=3,
            n_heads=8,
            head_dim=16,
            index_heads=64,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        flat_k_rows = (case.prefix_lengths[0] + case.input_lengths[0]) // case.compress_ratio
        k_flat = torch.zeros(
            flat_k_rows,
            case.index_dim,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        # Later candidates score higher, so the topK builder must replace the
        # initial fill after multiple 8-candidate matrix-scorer tiles.
        for c in range(flat_k_rows):
            k_flat[c].fill_(float(c + 1) / float(flat_k_rows))
        k_cache = _pack_flat_indexer_cache(k_flat)
        fixture.indexer_q.fill_(1.0 / float(case.index_dim))
        weights = torch.ones(
            fixture.q.shape[0],
            case.index_heads,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        cu_lens = torch.tensor([0, flat_k_rows], dtype=torch.long, device=fixture.q.device)

        expected = reference_attention_with_fp8_indexer(
            fixture,
            k_cache,
            weights,
            cu_lens,
        )

        dummy_indexer_k = torch.zeros(
            1,
            flat_k_rows,
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_cache=k_cache,
                csa_indexer_weights=weights,
                csa_indexer_cu_lens=cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_fp8_indexer_matrix_tile_cross_tile_tie_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_fp8_indexer_matrix_tile_cross_tile_tie",
            compress_ratio=4,
            prefix_lengths=(64,),
            input_lengths=(4,),
            window_size=4,
            compressed_topk=3,
            n_heads=8,
            head_dim=16,
            index_heads=64,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        flat_k_rows = (case.prefix_lengths[0] + case.input_lengths[0]) // case.compress_ratio
        k_flat = torch.ones(
            flat_k_rows,
            case.index_dim,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        # The topK boundary crosses the 8-candidate tile edge. Candidate 7 and
        # 8 have equal scores, so the exact tie-break must keep smaller idx 7.
        scores = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.90, 1.00, 1.00, 0.95]
        for c, value in enumerate(scores):
            k_flat[c].fill_(value)
        k_cache = _pack_flat_indexer_cache(k_flat)
        fixture.indexer_q.fill_(1.0 / float(case.index_dim))
        weights = torch.ones(
            fixture.q.shape[0],
            case.index_heads,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        cu_lens = torch.tensor([0, flat_k_rows], dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_fp8_indexer(
            fixture,
            k_cache,
            weights,
            cu_lens,
        )

        dummy_indexer_k = torch.zeros(
            1,
            flat_k_rows,
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_cache=k_cache,
                csa_indexer_weights=weights,
                csa_indexer_cu_lens=cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_paged_indexer_cache_topk_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_paged_indexer_cache",
            compress_ratio=4,
            prefix_lengths=(0, 8),
            input_lengths=(8, 8),
            window_size=6,
            compressed_topk=4,
            n_heads=2,
            head_dim=8,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260717)
        k_by_req = []
        slots = []
        block_table_rows = []
        next_block = 2
        for req, indexer_k_ref in enumerate(fixture.indexer_k_by_req):
            rows = int(indexer_k_ref.shape[0])
            k_req = (
                torch.randn(rows, case.index_dim, generator=gen, dtype=torch.float32)
                * 0.25
                + float(req) * 0.05
            ).to(device=fixture.q.device)
            k_by_req.append(k_req)
            block_table_rows.append([next_block])
            slots.extend([next_block * COMPRESSOR_KV_BLOCK_SIZE + i for i in range(rows)])
            next_block += 2
        k_flat = torch.cat(k_by_req, dim=0).contiguous()
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=fixture.q.device)
        k_pool = _pack_paged_indexer_pool(k_flat, slot_tensor)
        k_pool_flat = _unpack_paged_indexer_pool(k_pool, slot_tensor, case.index_dim)
        weights = (
            torch.randn(
                fixture.q.shape[0],
                case.index_heads,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.2
            + 1.0
        ).to(device=fixture.q.device).contiguous()
        cu = [0]
        for k_req in k_by_req:
            cu.append(cu[-1] + int(k_req.shape[0]))
        cu_lens = torch.tensor(cu, dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_indexer_k_rows(
            fixture,
            k_pool_flat,
            weights,
            cu_lens,
        )
        block_table = torch.tensor(
            block_table_rows,
            dtype=torch.int32,
            device=fixture.q.device,
        )
        seq_lens = torch.tensor(
            [int(x.shape[0]) for x in k_by_req],
            dtype=torch.int32,
            device=fixture.q.device,
        )
        dummy_indexer_k = torch.zeros(
            len(case.input_lengths),
            max(int(x.shape[0]) for x in fixture.indexer_k_by_req),
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_pool=k_pool,
                csa_indexer_weights=weights,
                csa_indexer_block_table=block_table,
                csa_indexer_seq_lens=seq_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_paged_indexer_matrix_tile_replacement_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_paged_indexer_matrix_tile_replacement",
            compress_ratio=4,
            prefix_lengths=(64, 64),
            input_lengths=(4, 4),
            window_size=4,
            compressed_topk=3,
            n_heads=8,
            head_dim=16,
            index_heads=64,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)

        k_by_req = []
        slots = []
        block_table_rows = []
        for req in range(len(case.input_lengths)):
            rows = (case.prefix_lengths[req] + case.input_lengths[req]) // case.compress_ratio
            k_req = torch.zeros(
                rows,
                case.index_dim,
                dtype=torch.float32,
                device=fixture.q.device,
            )
            for c in range(rows):
                k_req[c].fill_(float(c + 1 + req) / float(rows + req + 1))
            k_by_req.append(k_req)
            first_block = 2 + req * 3
            block_table_rows.append([first_block, first_block + 1])
            slots.extend(
                [
                    first_block * COMPRESSOR_KV_BLOCK_SIZE + c
                    for c in range(min(COMPRESSOR_KV_BLOCK_SIZE, rows))
                ]
            )
            slots.extend(
                [
                    (first_block + 1) * COMPRESSOR_KV_BLOCK_SIZE + c
                    for c in range(max(0, rows - COMPRESSOR_KV_BLOCK_SIZE))
                ]
            )
        k_flat = torch.cat(k_by_req, dim=0).contiguous()
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=fixture.q.device)
        k_pool = _pack_paged_indexer_pool(k_flat, slot_tensor)
        k_pool_flat = _unpack_paged_indexer_pool(k_pool, slot_tensor, case.index_dim)
        weights = torch.ones(
            fixture.q.shape[0],
            case.index_heads,
            dtype=torch.float32,
            device=fixture.q.device,
        )
        cu = [0]
        for k_req in k_by_req:
            cu.append(cu[-1] + int(k_req.shape[0]))
        cu_lens = torch.tensor(cu, dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_indexer_k_rows(
            fixture,
            k_pool_flat,
            weights,
            cu_lens,
        )
        block_table = torch.tensor(
            block_table_rows,
            dtype=torch.int32,
            device=fixture.q.device,
        )
        seq_lens = torch.tensor(
            [int(x.shape[0]) for x in k_by_req],
            dtype=torch.int32,
            device=fixture.q.device,
        )
        dummy_indexer_k = torch.zeros(
            len(case.input_lengths),
            max(int(x.shape[0]) for x in fixture.indexer_k_by_req),
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                dummy_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                csa_indexer_k_pool=k_pool,
                csa_indexer_weights=weights,
                csa_indexer_block_table=block_table,
                csa_indexer_seq_lens=seq_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_csa_qfp8_folded_indexer_topk_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
            indexer_q_fp8_quant_fold,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
            fp8_mqa_indexer_score,
            has_fp8_mqa_logits,
        )

        if not has_fp8_mqa_logits():
            self.skipTest("deep_gemm.fp8_mqa_logits is not available")

        case = AttentionCase(
            name="csa_qfp8_folded_indexer",
            compress_ratio=4,
            prefix_lengths=(60, 60),
            input_lengths=(4, 4),
            window_size=4,
            compressed_topk=4,
            n_heads=2,
            head_dim=8,
            index_heads=32,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        kv, _, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260718)

        k_by_req = []
        slots = []
        block_table_rows = []
        next_block = 2
        for req in range(len(case.input_lengths)):
            rows = (case.prefix_lengths[req] + case.input_lengths[req]) // case.compress_ratio
            k_req = (
                torch.randn(rows, case.index_dim, generator=gen, dtype=torch.float32)
                * 0.25
                + float(req) * 0.05
            ).to(device=fixture.q.device)
            k_by_req.append(k_req)
            block_table_rows.append([next_block])
            slots.extend([next_block * COMPRESSOR_KV_BLOCK_SIZE + i for i in range(rows)])
            next_block += 2
        k_flat = torch.cat(k_by_req, dim=0).contiguous()
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=fixture.q.device)
        k_pool = _pack_paged_indexer_pool(k_flat, slot_tensor)
        block_table = torch.tensor(
            block_table_rows,
            dtype=torch.int32,
            device=fixture.q.device,
        )
        seq_lens = torch.tensor(
            [int(x.shape[0]) for x in k_by_req],
            dtype=torch.int32,
            device=fixture.q.device,
        )
        cu = torch.zeros(len(k_by_req) + 1, dtype=torch.int32, device=fixture.q.device)
        cu[1:] = torch.cumsum(seq_lens.to(torch.int64), dim=0).to(torch.int32)

        raw_weights = (
            torch.randn(
                fixture.q.shape[0],
                case.index_heads,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.2
            + 1.0
        ).to(device=fixture.q.device, dtype=torch.bfloat16)
        q_fp8, w_fold = indexer_q_fp8_quant_fold(
            fixture.indexer_q.unsqueeze(0).contiguous(),
            raw_weights.unsqueeze(0).contiguous(),
        )
        q_score = q_fp8.view(fixture.q.shape[0], case.index_heads, case.index_dim)
        w_score = w_fold.view(fixture.q.shape[0], case.index_heads)
        k_quant = k_flat.to(torch.float8_e4m3fn).contiguous()
        k_scale = torch.ones(k_flat.shape[0], dtype=torch.float32, device=fixture.q.device)
        req_ids = fixture.req_id_per_token.to(torch.long)
        ks = cu.to(torch.int64).index_select(0, req_ids).to(torch.int32)
        per_req_lens = seq_lens.to(torch.int64).index_select(0, req_ids)
        visible = ((fixture.position_ids.to(torch.int64) + 1) // case.compress_ratio).clamp_max(
            per_req_lens
        )
        ke = (ks.to(torch.int64) + visible).to(torch.int32)
        logits = fp8_mqa_indexer_score(
            q_score,
            w_score,
            k_quant,
            k_scale,
            ks,
            ke,
            clean_logits=False,
        )
        col = torch.arange(k_flat.shape[0], device=fixture.q.device, dtype=torch.int32).view(1, -1)
        masked = torch.where(
            (col >= ks.view(-1, 1)) & (col < ke.view(-1, 1)),
            logits,
            torch.full_like(logits, float("-inf")),
        )
        topk_global = masked.topk(case.compressed_topk, dim=-1).indices.to(torch.int32)
        topk_local = topk_global - ks.view(-1, 1)
        k_arange = torch.arange(case.compressed_topk, device=fixture.q.device).view(1, -1)
        topk_local = torch.where(
            k_arange < visible.view(-1, 1),
            topk_local,
            torch.full_like(topk_local, -1),
        )
        expected = reference_attention_with_explicit_csa_topk(fixture, topk_local)

        dummy_indexer_k = torch.zeros(
            len(case.input_lengths),
            1,
            case.index_heads,
            case.index_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        actual = op(
            fixture.q.contiguous(),
            kv,
            q_fp8.float().to(dtype=fixture.q.dtype).view(
                fixture.q.shape[0], case.index_heads, case.index_dim
            ).contiguous(),
            dummy_indexer_k.contiguous(),
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            torch.arange(fixture.q.shape[0], dtype=torch.long, device=fixture.q.device),
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            csa_indexer_k_pool=k_pool,
            csa_indexer_weights=w_score.contiguous(),
            csa_indexer_block_table=block_table,
            csa_indexer_seq_lens=seq_lens,
        )
        _assert_attention_close(actual, expected)

    def test_candidate_cuda_hca_packed_compressed_k_attention_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_packed_cmp_attention",
            compress_ratio=128,
            prefix_lengths=(128,),
            input_lengths=(4,),
            window_size=1,
            compressed_topk=1,
            n_heads=2,
            head_dim=16,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        fixture.kv_by_req[0][127].zero_()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260704)
        cmp_k = (
            torch.randn(1, case.head_dim, generator=gen, dtype=torch.float32) * 0.2
            + 0.35
        ).to(fixture.q.device)
        cmp_cache = _pack_flat_model1_cache(cmp_k)
        cmp_cu_lens = torch.tensor([0, 1], dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_packed_cmp_cache(
            fixture,
            cmp_cache,
            cmp_cu_lens,
        )
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                indexer_k,
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                attention_cmp_k_cache=cmp_cache,
                attention_cmp_cu_lens=cmp_cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_hca_paged_compressed_k_attention_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_paged_cmp_attention",
            compress_ratio=128,
            prefix_lengths=(128,),
            input_lengths=(4,),
            window_size=1,
            compressed_topk=1,
            n_heads=2,
            head_dim=16,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        fixture.kv_by_req[0][127].zero_()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260714)
        cmp_k = (
            torch.randn(1, case.head_dim, generator=gen, dtype=torch.float32) * 0.2
            + 0.35
        ).to(fixture.q.device)
        cmp_pool = _alloc_swa_cache(2 * SWA_BLOCK_SIZE, fixture.q.device)
        cmp_pool.fill_(0x5A)
        cmp_slot = torch.tensor([2 * SWA_BLOCK_SIZE], dtype=torch.long, device=fixture.q.device)
        cmp_k_padded = torch.zeros(1, SWA_HEAD_DIM, dtype=torch.bfloat16, device=fixture.q.device)
        cmp_k_padded[:, : case.head_dim].copy_(cmp_k.to(torch.bfloat16))
        quantize_and_insert_k_cache(cmp_k_padded.contiguous(), cmp_pool, cmp_slot)
        cmp_deq = dequantize_slots_to_bf16(cmp_pool, cmp_slot)[:, : case.head_dim]
        expected = reference_attention_with_cmp_k_rows(fixture, [cmp_deq])
        cmp_block_table = torch.tensor([[2]], dtype=torch.int32, device=fixture.q.device)
        cmp_seq_lens = torch.tensor([1], dtype=torch.int32, device=fixture.q.device)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                indexer_k,
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                attention_cmp_k_pool=cmp_pool,
                attention_cmp_block_table=cmp_block_table,
                attention_cmp_seq_lens=cmp_seq_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_swa_prefix_packed_attention_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_packed_swa_prefix",
            compress_ratio=128,
            prefix_lengths=(3,),
            input_lengths=(2,),
            window_size=3,
            compressed_topk=1,
            n_heads=2,
            head_dim=16,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        # Prefix BF16 K is deliberately wrong; the op must read prefix SWA
        # keys from the 584B cache and only use BF16 for fresh prefill keys.
        fixture.kv_by_req[0][:3].zero_()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260705)
        swa_prefix = (
            torch.randn(3, case.head_dim, generator=gen, dtype=torch.float32) * 0.15
            + 0.25
        ).to(fixture.q.device)
        swa_cache = _pack_flat_model1_cache(swa_prefix)
        swa_cu_lens = torch.tensor([0, 3], dtype=torch.long, device=fixture.q.device)
        expected = reference_attention_with_packed_swa_cache(
            fixture,
            swa_cache,
            swa_cu_lens,
        )
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                indexer_k,
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                attention_swa_k_cache=swa_cache,
                attention_swa_cu_lens=swa_cu_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_swa_prefix_paged_attention_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_paged_swa_prefix",
            compress_ratio=128,
            prefix_lengths=(3,),
            input_lengths=(2,),
            window_size=3,
            compressed_topk=1,
            n_heads=2,
            head_dim=16,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        fixture.kv_by_req[0][:3].zero_()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260715)
        swa_prefix = (
            torch.randn(3, case.head_dim, generator=gen, dtype=torch.float32) * 0.15
            + 0.25
        ).to(fixture.q.device)
        swa_pool = _alloc_swa_cache(3 * SWA_BLOCK_SIZE + 7, fixture.q.device)
        swa_pool.fill_(0x5A)
        swa_slots = torch.tensor(
            [[3 * SWA_BLOCK_SIZE + 5, 3 * SWA_BLOCK_SIZE + 7]],
            dtype=torch.long,
            device=fixture.q.device,
        )
        swa_k_padded = torch.zeros(2, SWA_HEAD_DIM, dtype=torch.bfloat16, device=fixture.q.device)
        swa_k_padded[:, : case.head_dim].copy_(swa_prefix[1:3].to(torch.bfloat16))
        quantize_and_insert_k_cache(
            swa_k_padded.contiguous(),
            swa_pool,
            swa_slots.reshape(-1).contiguous(),
        )
        swa_gather_lens = torch.tensor([2], dtype=torch.int32, device=fixture.q.device)
        swa_deq = dequantize_slots_to_bf16(swa_pool, swa_slots.reshape(-1))[
            :, : case.head_dim
        ].view(1, 2, case.head_dim)
        expected = reference_attention_with_swa_suffix_rows(
            fixture,
            [swa_deq[0]],
            swa_gather_lens,
        )
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        restored = torch.empty_like(expected)
        for rank in range(case.cp_size):
            rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
                device=fixture.q.device
            )
            actual = op(
                fixture.q.contiguous(),
                kv,
                fixture.indexer_q.contiguous(),
                indexer_k,
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                attention_swa_k_pool=swa_pool,
                attention_swa_slot_mapping=swa_slots.contiguous(),
                attention_swa_gather_lens=swa_gather_lens,
            )
            _assert_attention_close(actual, expected.index_select(0, rows))
            restored.index_copy_(0, rows, actual)
        _assert_attention_close(restored, expected)

    def test_candidate_cuda_swa_prefix_paged_with_fresh_only_kv_allclose(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_paged_swa_prefix_fresh_only",
            compress_ratio=128,
            prefix_lengths=(6,),
            input_lengths=(3,),
            window_size=6,
            compressed_topk=0,
            n_heads=2,
            head_dim=16,
            index_heads=1,
            index_dim=1,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.float32)
        fixture.kv_by_req[0][: case.prefix_lengths[0]].zero_()
        fresh_shared = fixture.kv_by_req[0][
            case.prefix_lengths[0] : case.prefix_lengths[0] + case.input_lengths[0],
            :1,
            :,
        ].expand(-1, case.n_heads, -1).contiguous()
        fixture.kv_by_req[0][
            case.prefix_lengths[0] : case.prefix_lengths[0] + case.input_lengths[0]
        ].copy_(fresh_shared)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260720)
        swa_prefix = (
            torch.randn(
                case.prefix_lengths[0],
                case.head_dim,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.15
            + 0.25
        ).to(fixture.q.device)
        gather_len = min(case.prefix_lengths[0], case.window_size - 1)
        suffix = swa_prefix[-gather_len:]
        swa_pool = _alloc_swa_cache(4 * SWA_BLOCK_SIZE + gather_len, fixture.q.device)
        swa_pool.fill_(0x5A)
        swa_slots = (
            torch.arange(
                4 * SWA_BLOCK_SIZE,
                4 * SWA_BLOCK_SIZE + gather_len,
                dtype=torch.long,
                device=fixture.q.device,
            )
            .view(1, gather_len)
            .contiguous()
        )
        swa_k_padded = torch.zeros(
            gather_len,
            SWA_HEAD_DIM,
            dtype=torch.bfloat16,
            device=fixture.q.device,
        )
        swa_k_padded[:, : case.head_dim].copy_(suffix.to(torch.bfloat16))
        quantize_and_insert_k_cache(
            swa_k_padded.contiguous(),
            swa_pool,
            swa_slots.reshape(-1).contiguous(),
        )
        swa_deq = dequantize_slots_to_bf16(swa_pool, swa_slots.reshape(-1))[
            :, : case.head_dim
        ].view(1, gather_len, case.head_dim)
        gather_lens = torch.tensor(
            [gather_len], dtype=torch.int32, device=fixture.q.device
        )
        expected = reference_attention_with_swa_suffix_rows(
            fixture,
            [swa_deq[0]],
            gather_lens,
        )
        _, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        fresh_kv = fixture.kv_by_req[0][
            case.prefix_lengths[0] : case.prefix_lengths[0] + case.input_lengths[0],
            :1,
            :,
        ].contiguous()
        rows = torch.arange(expected.shape[0], dtype=torch.long, device=fixture.q.device)
        actual = op(
            fixture.q.contiguous(),
            fresh_kv.unsqueeze(0).contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k[:, :1].contiguous(),
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            attention_swa_k_pool=swa_pool,
            attention_swa_slot_mapping=swa_slots,
            attention_swa_gather_lens=gather_lens,
            kv_unpad_restore=torch.arange(
                case.input_lengths[0], dtype=torch.long, device=fixture.q.device
            ),
            kv_cu_lens=torch.tensor(
                [0, case.input_lengths[0]], dtype=torch.long, device=fixture.q.device
            ),
        )
        _assert_attention_close(actual, expected)

    def test_candidate_cuda_hca_python_hook_allclose_identity_output(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention hook requires CUDA")
        if _candidate_op() is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV

        device = torch.device("cuda")
        T, H, D = 128, 2, SWA_HEAD_DIM
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260716)
        q = torch.randn(T, H, D, generator=gen, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        fresh_k = (
            torch.randn(T, D, generator=gen, dtype=torch.float32) * 0.04
        ).to(device=device, dtype=torch.bfloat16)
        data = _make_raw_compressor_kv_write_fixture(
            num_tokens=T,
            head_dim=D,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            entry_bytes=SWA_ENTRY_BYTES,
            device=device,
        )

        ref_state = data["state_cache"].clone()
        ref_kv_cache = data["ref_kv_cache"].clone()
        run_save_partial_states(
            data["kv"],
            data["score"],
            data["ape"],
            data["positions"],
            ref_state,
            data["state_slots"],
            compress_ratio=128,
        )
        run_fused_compress_kv_write(
            ref_state,
            data["token_to_req"],
            data["positions"],
            data["state_slots"],
            data["state_block_table"],
            data["norm_weight"],
            1.0e-6,
            data["cos_sin_cache"],
            ref_kv_cache,
            data["kv_slots"],
            data["kv"],
            data["score"],
            data["ape"],
            0,
            disable_raw_path=False,
            head_dim=D,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )
        cmp_k = dequantize_slots_to_bf16(
            ref_kv_cache,
            torch.tensor([0], dtype=torch.long, device=device),
        )[:, :D]
        expected_rows = []
        sink = torch.zeros(H, dtype=torch.float32, device=device)
        for row in range(T):
            keys = [fresh_k[row].view(1, D).expand(H, D)]
            if row == T - 1:
                keys.insert(0, cmp_k[0].view(1, D).expand(H, D))
            expected_rows.append(
                _attention_over_keys(q[row], torch.stack(keys, dim=0), sink)
            )
        expected = torch.stack(expected_rows, dim=0).to(torch.bfloat16)

        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio = 128
        layer.window_size = 1
        layer.head_dim = D
        layer.rope_head_dim = 64
        layer.n_heads = H
        layer.dim = H * D
        layer.n_groups = 1
        layer.layer_id = 0
        layer.attn_sink = sink
        layer.tp_size = 1
        layer._prefill_output_proj = lambda o, freqs: o
        layer._prefill_output_all_reduce = lambda out: None

        state_cache = data["state_cache"].clone()
        actual_kv_cache = data["actual_kv_cache"].clone()
        swa_cache = _alloc_swa_cache(T - 1, device)
        swa_cache.fill_(0x5A)

        compressor = SimpleNamespace(
            ape=data["ape"],
            norm=SimpleNamespace(weight=data["norm_weight"]),
            norm_eps=1.0e-6,
            head_dim=D,
            rope_head_dim=64,
            overlap=False,
            _state_pool_3d=state_cache,
            _kv_pool_view=actual_kv_cache,
            _state_block_table=data["state_block_table"],
            _state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            _ensure_cos_sin_cache=lambda device_arg: data["cos_sin_cache"],
        )
        layer.compressor = compressor
        layer._pool_view_3d_fp8 = lambda attn_type: actual_kv_cache if attn_type == HCA_KV else swa_cache
        layer._build_cp_distributed_swa_slot_mapping = lambda common_arg: torch.arange(
            T, dtype=torch.long, device=device
        )

        cp_ctx = SimpleNamespace(
            cp_size=1,
            cp_rank=0,
            chunk_length=T,
            seq_len_full=T,
            unpad_restore=torch.arange(T, dtype=torch.long, device=device),
            global_positions=torch.arange(T, dtype=torch.long, device=device),
            req_id_per_token=torch.zeros(T, dtype=torch.long, device=device),
            prefix_lengths=torch.tensor([0], dtype=torch.long, device=device),
            input_lengths_global=torch.tensor([T], dtype=torch.int32, device=device),
            cu_seqlens_global=torch.tensor([0, T], dtype=torch.long, device=device),
            prefix_length=0,
            kv_cache_sharded=False,
        )
        common = PrefillMeta(
            seqlen=T,
            seqlen_full=T,
            rd=0,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=True,
            freqs_cis=torch.empty(T, 1, device=device),
            topk_idxs=torch.empty(0, dtype=torch.int32, device=device),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([T], dtype=torch.long, device=device),
            batch_size=1,
            input_lengths=torch.tensor([T], dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor([0], dtype=torch.long, device=device),
            position_ids=torch.arange(T, dtype=torch.long, device=device),
            req_id_per_token=torch.zeros(T, dtype=torch.long, device=device),
            swa_meta=SimpleNamespace(
                slot_mapping=torch.arange(T, dtype=torch.long, device=device)
            ),
        )
        qkv = PrefillQKV(
            qr=torch.empty(T, 1, dtype=torch.bfloat16, device=device),
            q=q,
            kv_full=None,
            swa_k_local=fresh_k,
        )
        compressor_meta = SimpleNamespace(
            positions=data["positions"],
            state_slots=data["state_slots"],
            kv_slots=data["kv_slots"],
            token_to_req=data["token_to_req"],
            seq_start_per_req=torch.tensor([0], dtype=torch.int32, device=device),
            cu_seq_per_req=torch.tensor([0, T], dtype=torch.int32, device=device),
        )
        workspace_meta = WorkspaceMeta(
            M=1,
            N=1,
            swa_eb=SWA_BLOCK_SIZE,
            cmp_eb=COMPRESSOR_KV_BLOCK_SIZE,
            swa_bt_int32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            cmp_bt_int32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            swa_seq_lens=torch.tensor([T], dtype=torch.int32, device=device),
            cmp_seq_lens=torch.tensor([1], dtype=torch.int32, device=device),
            swa_gather_lens=torch.tensor([1], dtype=torch.int32, device=device),
            swa_cache_seq_lens=torch.tensor([0], dtype=torch.int32, device=device),
            swa_cache_gather_lens=torch.tensor([0], dtype=torch.int32, device=device),
            qsl=torch.tensor([0, T], dtype=torch.int32, device=device),
            dense_cmp_topk=torch.zeros(T, 1, dtype=torch.int32, device=device),
            new_k_slot_in_flat=torch.arange(T, dtype=torch.long, device=device),
            cmp_reader=None,
            use_cp_raw_q_merge=False,
            swa_cache_slot_mapping=torch.empty(1, 0, dtype=torch.long, device=device),
            swa_slot_mapping=None,
        )

        class _FakeBuffer:
            spec = SimpleNamespace(per_rank_bytes=0)

            def op_kwargs(self, *, cp_rank: int) -> dict:
                return {
                    "cp_rank": int(cp_rank),
                    "cp_size": 1,
                    "comm_ptr": 0,
                    "buffer_handle": -1,
                    "signal_handle": -1,
                    "per_rank_buffer_bytes": 0,
                    "rank_offsets": [],
                }

        layer._ensure_cp_distributed_attention_buffer = lambda qkv_arg, common_arg: _FakeBuffer()
        actual = layer._forward_prefill_cp_distributed_attention(
            qkv,
            common,
            workspace_meta,
            None,
            compressor_meta,
            (data["kv"], data["score"]),
        )
        torch.cuda.synchronize()
        _assert_attention_close(actual, expected)
        self.assertTrue(torch.equal(actual_kv_cache, ref_kv_cache))

    def test_candidate_cuda_csa_python_hook_allclose_identity_output(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention hook requires CUDA")
        if _candidate_op() is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, INDEXER_KV, SWA_KV

        device = torch.device("cuda")
        T, H, D = 4, 2, SWA_HEAD_DIM
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260720)
        q = torch.randn(T, H, D, generator=gen, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        fresh_k = (
            torch.randn(T, D, generator=gen, dtype=torch.float32) * 0.04
        ).to(device=device, dtype=torch.bfloat16)

        main = _make_raw_compressor_kv_write_fixture(
            num_tokens=T,
            head_dim=D,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            entry_bytes=SWA_ENTRY_BYTES,
            device=device,
        )
        nested = _make_raw_compressor_kv_write_fixture(
            num_tokens=T,
            head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            entry_bytes=INDEXER_ENTRY_BYTES,
            device=device,
        )

        ref_main_state = main["state_cache"].clone()
        ref_main_kv_cache = main["ref_kv_cache"].clone()
        run_save_partial_states(
            main["kv"],
            main["score"],
            main["ape"],
            main["positions"],
            ref_main_state,
            main["state_slots"],
            compress_ratio=4,
        )
        run_fused_compress_kv_write(
            ref_main_state,
            main["token_to_req"],
            main["positions"],
            main["state_slots"],
            main["state_block_table"],
            main["norm_weight"],
            1.0e-6,
            main["cos_sin_cache"],
            ref_main_kv_cache,
            main["kv_slots"],
            main["kv"],
            main["score"],
            main["ape"],
            0,
            disable_raw_path=False,
            head_dim=D,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )
        ref_nested_state = nested["state_cache"].clone()
        ref_nested_kv_cache = nested["ref_kv_cache"].clone()
        run_save_partial_states(
            nested["kv"],
            nested["score"],
            nested["ape"],
            nested["positions"],
            ref_nested_state,
            nested["state_slots"],
            compress_ratio=4,
        )
        run_fused_compress_kv_write(
            ref_nested_state,
            nested["token_to_req"],
            nested["positions"],
            nested["state_slots"],
            nested["state_block_table"],
            nested["norm_weight"],
            1.0e-6,
            nested["cos_sin_cache"],
            ref_nested_kv_cache,
            nested["kv_slots"],
            nested["kv"],
            nested["score"],
            nested["ape"],
            0,
            disable_raw_path=False,
            head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )

        cmp_k = dequantize_slots_to_bf16(
            ref_main_kv_cache,
            torch.tensor([0], dtype=torch.long, device=device),
        )[:, :D]
        expected_rows = []
        sink = torch.zeros(H, dtype=torch.float32, device=device)
        for row in range(T):
            keys = [fresh_k[row].view(1, D).expand(H, D)]
            if row == T - 1:
                keys.insert(0, cmp_k[0].view(1, D).expand(H, D))
            expected_rows.append(
                _attention_over_keys(q[row], torch.stack(keys, dim=0), sink)
            )
        expected = torch.stack(expected_rows, dim=0).to(torch.bfloat16)

        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio = 4
        layer.window_size = 1
        layer.head_dim = D
        layer.rope_head_dim = 64
        layer.n_heads = H
        layer.dim = H * D
        layer.n_groups = 1
        layer.layer_id = 0
        layer.attn_sink = sink
        layer.tp_size = 1
        layer._prefill_output_proj = lambda o, freqs: o
        layer._prefill_output_all_reduce = lambda out: None

        actual_main_kv_cache = main["actual_kv_cache"].clone()
        actual_nested_kv_cache = nested["actual_kv_cache"].clone()
        main_state = main["state_cache"].clone()
        nested_state = nested["state_cache"].clone()
        swa_cache = _alloc_swa_cache(T - 1, device)
        swa_cache.fill_(0x5A)

        main_compressor = SimpleNamespace(
            ape=main["ape"],
            norm=SimpleNamespace(weight=main["norm_weight"]),
            norm_eps=1.0e-6,
            head_dim=D,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            _state_pool_3d=main_state,
            _kv_pool_view=actual_main_kv_cache,
            _state_block_table=main["state_block_table"],
            _state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            _ensure_cos_sin_cache=lambda device_arg: main["cos_sin_cache"],
        )
        nested_compressor = SimpleNamespace(
            ape=nested["ape"],
            norm=SimpleNamespace(weight=nested["norm_weight"]),
            norm_eps=1.0e-6,
            head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            _state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            _ensure_cos_sin_cache=lambda device_arg: nested["cos_sin_cache"],
        )
        layer.compressor = main_compressor
        indexer_stub = SimpleNamespace(
            _kv_pool_view=actual_nested_kv_cache,
            _kv_block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
            _kv_eb=COMPRESSOR_KV_BLOCK_SIZE,
            _kv_tokens_per_block=COMPRESSOR_KV_BLOCK_SIZE,
            _kv_owner_tokens_per_block=COMPRESSOR_KV_BLOCK_SIZE,
            _state_pool_3d=nested_state,
            _state_block_table=nested["state_block_table"],
            _state_eb=COMPRESSOR_STATE_BLOCK_SIZE,
            _state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            compressor=nested_compressor,
        )
        indexer_stub._propagate_pool_to_nested = lambda: (
            setattr(nested_compressor, "_kv_pool_view", indexer_stub._kv_pool_view),
            setattr(nested_compressor, "_kv_block_table", indexer_stub._kv_block_table),
            setattr(nested_compressor, "_kv_eb", indexer_stub._kv_eb),
            setattr(
                nested_compressor,
                "_kv_tokens_per_block",
                indexer_stub._kv_tokens_per_block,
            ),
            setattr(
                nested_compressor,
                "_kv_owner_tokens_per_block",
                indexer_stub._kv_owner_tokens_per_block,
            ),
            setattr(nested_compressor, "_state_pool_3d", indexer_stub._state_pool_3d),
            setattr(
                nested_compressor,
                "_state_block_table",
                indexer_stub._state_block_table,
            ),
            setattr(nested_compressor, "_state_eb", indexer_stub._state_eb),
            setattr(
                nested_compressor,
                "_state_tokens_per_block",
                indexer_stub._state_tokens_per_block,
            ),
        )
        layer.indexer = indexer_stub
        layer._pool_view_3d_fp8 = (
            lambda attn_type: actual_main_kv_cache
            if attn_type == CSA_KV
            else actual_nested_kv_cache
            if attn_type == INDEXER_KV
            else swa_cache
            if attn_type == SWA_KV
            else None
        )
        layer._build_cp_distributed_swa_slot_mapping = lambda common_arg: torch.arange(
            T, dtype=torch.long, device=device
        )

        cp_ctx = SimpleNamespace(
            cp_size=1,
            cp_rank=0,
            chunk_length=T,
            seq_len_full=T,
            unpad_restore=torch.arange(T, dtype=torch.long, device=device),
            global_positions=torch.arange(T, dtype=torch.long, device=device),
            req_id_per_token=torch.zeros(T, dtype=torch.long, device=device),
            prefix_lengths=torch.tensor([0], dtype=torch.long, device=device),
            input_lengths_global=torch.tensor([T], dtype=torch.int32, device=device),
            cu_seqlens_global=torch.tensor([0, T], dtype=torch.long, device=device),
            prefix_length=0,
            kv_cache_sharded=False,
        )
        indexer_meta = SimpleNamespace(
            block_table_i32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            cu_kv_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=device),
            compressor_meta=SimpleNamespace(
                positions=nested["positions"],
                state_slots=nested["state_slots"],
                kv_slots=nested["kv_slots"],
                token_to_req=nested["token_to_req"],
                seq_start_per_req=torch.tensor([0], dtype=torch.int32, device=device),
                cu_seq_per_req=torch.tensor([0, T], dtype=torch.int32, device=device),
            ),
        )
        common = PrefillMeta(
            seqlen=T,
            seqlen_full=T,
            rd=0,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=True,
            freqs_cis=torch.empty(T, 1, device=device),
            topk_idxs=torch.empty(0, dtype=torch.int32, device=device),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([T], dtype=torch.long, device=device),
            batch_size=1,
            input_lengths=torch.tensor([T], dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor([0], dtype=torch.long, device=device),
            position_ids=torch.arange(T, dtype=torch.long, device=device),
            req_id_per_token=torch.zeros(T, dtype=torch.long, device=device),
            swa_meta=SimpleNamespace(
                slot_mapping=torch.arange(T, dtype=torch.long, device=device)
            ),
            csa_meta=SimpleNamespace(
                indexer_meta=indexer_meta,
                compressor_meta=SimpleNamespace(
                    positions=main["positions"],
                    state_slots=main["state_slots"],
                    kv_slots=main["kv_slots"],
                    token_to_req=main["token_to_req"],
                    seq_start_per_req=torch.tensor([0], dtype=torch.int32, device=device),
                    cu_seq_per_req=torch.tensor([0, T], dtype=torch.int32, device=device),
                ),
                workspace_meta=None,
            ),
        )
        qkv = PrefillQKV(
            qr=torch.empty(T, 1, dtype=torch.bfloat16, device=device),
            q=q,
            kv_full=None,
            swa_k_local=fresh_k,
        )
        workspace_meta = WorkspaceMeta(
            M=1,
            N=1,
            swa_eb=SWA_BLOCK_SIZE,
            cmp_eb=COMPRESSOR_KV_BLOCK_SIZE,
            swa_bt_int32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            cmp_bt_int32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            swa_seq_lens=torch.tensor([T], dtype=torch.int32, device=device),
            cmp_seq_lens=torch.tensor([1], dtype=torch.int32, device=device),
            swa_gather_lens=torch.tensor([1], dtype=torch.int32, device=device),
            swa_cache_seq_lens=torch.tensor([0], dtype=torch.int32, device=device),
            swa_cache_gather_lens=torch.tensor([0], dtype=torch.int32, device=device),
            qsl=torch.tensor([0, T], dtype=torch.int32, device=device),
            dense_cmp_topk=None,
            new_k_slot_in_flat=torch.arange(T, dtype=torch.long, device=device),
            cmp_reader=None,
            use_cp_raw_q_merge=False,
            swa_cache_slot_mapping=torch.empty(1, 0, dtype=torch.long, device=device),
            swa_slot_mapping=None,
        )

        class _FakeBuffer:
            spec = SimpleNamespace(per_rank_bytes=0)

            def op_kwargs(self, *, cp_rank: int) -> dict:
                return {
                    "cp_rank": int(cp_rank),
                    "cp_size": 1,
                    "comm_ptr": 0,
                    "buffer_handle": -1,
                    "signal_handle": -1,
                    "per_rank_buffer_bytes": 0,
                    "rank_offsets": [],
                }

        layer._ensure_cp_distributed_attention_buffer = lambda qkv_arg, common_arg: _FakeBuffer()
        indexer_payload = SimpleNamespace(
            indexer_q=torch.zeros(T, 1, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device=device),
            weights=torch.ones(T, 1, dtype=torch.float32, device=device),
            compressor_kv=nested["kv"],
            compressor_score=nested["score"],
        )
        actual = layer._forward_prefill_cp_distributed_attention(
            qkv,
            common,
            workspace_meta,
            None,
            common.csa_meta.compressor_meta,
            (main["kv"], main["score"]),
            indexer_payload,
        )
        torch.cuda.synchronize()
        _assert_attention_close(actual, expected)
        self.assertTrue(torch.equal(actual_main_kv_cache, ref_main_kv_cache))
        self.assertTrue(torch.equal(actual_nested_kv_cache, ref_nested_kv_cache))

    def test_candidate_cuda_swa_cache_write_byte_equal_to_triton(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_swa_write",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = rank_local_rows(expected.shape[0], 0, case.cp_size).to(
            device=fixture.q.device
        )
        swa_k, slot_mapping, ref_cache, actual_cache = _make_swa_fixture(
            17, torch.device("cuda"), skipped=(2, 11)
        )
        quantize_and_insert_k_cache(swa_k, ref_cache, slot_mapping)
        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            swa_k=swa_k,
            swa_k_cache=actual_cache,
            swa_slot_mapping=slot_mapping,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual_cache, ref_cache))

    def test_candidate_cuda_compressor_state_write_equal_to_triton(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_compressor_state_write",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = rank_local_rows(expected.shape[0], 0, case.cp_size).to(
            device=fixture.q.device
        )

        (
            compressor_kv,
            compressor_score,
            compressor_ape,
            compressor_positions,
            compressor_slots,
            ref_state,
            actual_state,
        ) = _make_compressor_state_fixture(
            19,
            32,
            4,
            torch.device("cuda"),
            dtype=torch.bfloat16,
            skipped=(3, 17),
        )
        run_save_partial_states(
            compressor_kv,
            compressor_score,
            compressor_ape,
            compressor_positions,
            ref_state,
            compressor_slots,
            compress_ratio=4,
        )
        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            compressor_kv=compressor_kv,
            compressor_score=compressor_score,
            compressor_ape=compressor_ape,
            compressor_positions=compressor_positions,
            compressor_state_cache=actual_state,
            compressor_state_slots=compressor_slots,
            compressor_ratio=4,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_state, ref_state, rtol=0.0, atol=0.0)

    def _run_compressor_kv_write_case(
        self,
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        overlap: bool,
        entry_bytes: int,
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_compressor_kv_write",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = rank_local_rows(expected.shape[0], 0, case.cp_size).to(
            device=fixture.q.device
        )

        num_tokens = int(compress_ratio)
        data = _make_raw_compressor_kv_write_fixture(
            num_tokens=num_tokens,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            entry_bytes=entry_bytes,
            device=torch.device("cuda"),
        )
        ref_state = data["state_cache"].clone()
        run_save_partial_states(
            data["kv"],
            data["score"],
            data["ape"],
            data["positions"],
            ref_state,
            data["state_slots"],
            compress_ratio=compress_ratio,
        )
        run_fused_compress_kv_write(
            ref_state,
            data["token_to_req"],
            data["positions"],
            data["state_slots"],
            data["state_block_table"],
            data["norm_weight"],
            1.0e-6,
            data["cos_sin_cache"],
            data["ref_kv_cache"],
            data["kv_slots"],
            data["kv"],
            data["score"],
            data["ape"],
            0,
            disable_raw_path=False,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )
        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            compressor_kv=data["kv"],
            compressor_score=data["score"],
            compressor_ape=data["ape"],
            compressor_positions=data["positions"],
            compressor_state_cache=data["state_cache"],
            compressor_state_slots=data["state_slots"],
            compressor_ratio=compress_ratio,
            compressor_token_to_req=data["token_to_req"],
            compressor_state_block_table=data["state_block_table"],
            compressor_norm_weight=data["norm_weight"],
            compressor_cos_sin_cache=data["cos_sin_cache"],
            compressor_kv_cache=data["actual_kv_cache"],
            compressor_kv_slots=data["kv_slots"],
            compressor_seq_start=0,
            compressor_disable_raw_path=False,
            compressor_rms_norm_eps=1.0e-6,
            compressor_head_dim=head_dim,
            compressor_rope_head_dim=rope_head_dim,
            compressor_overlap=overlap,
            compressor_state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(data["actual_kv_cache"], data["ref_kv_cache"]))

    def test_candidate_cuda_hca_compressor_kv_write_byte_equal_to_triton(self) -> None:
        self._run_compressor_kv_write_case(
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            entry_bytes=584,
        )

    def test_candidate_cuda_indexer_compressor_kv_write_byte_equal_to_triton(
        self,
    ) -> None:
        self._run_compressor_kv_write_case(
            head_dim=128,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            entry_bytes=132,
        )

    def test_candidate_cuda_csa_dual_compressor_kv_write_byte_equal_to_triton(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="csa_dual_compressor_write",
            compress_ratio=4,
            prefix_lengths=(0,),
            input_lengths=(4,),
            window_size=4,
            compressed_topk=1,
            n_heads=2,
            head_dim=8,
            index_heads=2,
            index_dim=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = torch.arange(expected.shape[0], dtype=torch.long, device=fixture.q.device)

        nested = _make_raw_compressor_kv_write_fixture(
            num_tokens=4,
            head_dim=128,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            entry_bytes=132,
            device=torch.device("cuda"),
        )
        main = _make_raw_compressor_kv_write_fixture(
            num_tokens=4,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            entry_bytes=584,
            device=torch.device("cuda"),
        )

        for data, head_dim, entry_bytes in ((nested, 128, 132), (main, 512, 584)):
            ref_state = data["state_cache"].clone()
            run_save_partial_states(
                data["kv"],
                data["score"],
                data["ape"],
                data["positions"],
                ref_state,
                data["state_slots"],
                compress_ratio=4,
            )
            run_fused_compress_kv_write(
                ref_state,
                data["token_to_req"],
                data["positions"],
                data["state_slots"],
                data["state_block_table"],
                data["norm_weight"],
                1.0e-6,
                data["cos_sin_cache"],
                data["ref_kv_cache"],
                data["kv_slots"],
                data["kv"],
                data["score"],
                data["ape"],
                0,
                disable_raw_path=False,
                head_dim=head_dim,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            )
            self.assertEqual(int(data["ref_kv_cache"].shape[-1]), entry_bytes)

        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            csa_indexer_compressor_kv=nested["kv"],
            csa_indexer_compressor_score=nested["score"],
            csa_indexer_compressor_ape=nested["ape"],
            csa_indexer_compressor_positions=nested["positions"],
            csa_indexer_compressor_state_cache=nested["state_cache"],
            csa_indexer_compressor_state_slots=nested["state_slots"],
            csa_indexer_compressor_ratio=4,
            csa_indexer_compressor_token_to_req=nested["token_to_req"],
            csa_indexer_compressor_state_block_table=nested["state_block_table"],
            csa_indexer_compressor_norm_weight=nested["norm_weight"],
            csa_indexer_compressor_cos_sin_cache=nested["cos_sin_cache"],
            csa_indexer_compressor_kv_cache=nested["actual_kv_cache"],
            csa_indexer_compressor_kv_slots=nested["kv_slots"],
            csa_indexer_compressor_seq_start=0,
            csa_indexer_compressor_disable_raw_path=False,
            csa_indexer_compressor_rms_norm_eps=1.0e-6,
            csa_indexer_compressor_head_dim=128,
            csa_indexer_compressor_rope_head_dim=64,
            csa_indexer_compressor_overlap=True,
            csa_indexer_compressor_state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
            compressor_kv=main["kv"],
            compressor_score=main["score"],
            compressor_ape=main["ape"],
            compressor_positions=main["positions"],
            compressor_state_cache=main["state_cache"],
            compressor_state_slots=main["state_slots"],
            compressor_ratio=4,
            compressor_token_to_req=main["token_to_req"],
            compressor_state_block_table=main["state_block_table"],
            compressor_norm_weight=main["norm_weight"],
            compressor_cos_sin_cache=main["cos_sin_cache"],
            compressor_kv_cache=main["actual_kv_cache"],
            compressor_kv_slots=main["kv_slots"],
            compressor_seq_start=0,
            compressor_disable_raw_path=False,
            compressor_rms_norm_eps=1.0e-6,
            compressor_head_dim=512,
            compressor_rope_head_dim=64,
            compressor_overlap=True,
            compressor_state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )
        _assert_attention_close(actual, expected)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(nested["actual_kv_cache"], nested["ref_kv_cache"]))
        self.assertTrue(torch.equal(main["actual_kv_cache"], main["ref_kv_cache"]))

    def _run_compressor_kv_writer_fixture(
        self,
        data: dict[str, torch.Tensor | int],
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        overlap: bool,
        disable_raw_path: bool,
        use_varlen_raw: bool = False,
        use_unpad_restore_raw: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        op = _candidate_op()
        if op is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        case = AttentionCase(
            name="hca_compressor_kv_writer_fixture",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = rank_local_rows(expected.shape[0], 0, case.cp_size).to(
            device=fixture.q.device
        )
        ref_state = data["state_cache"].clone()
        if not disable_raw_path:
            run_save_partial_states(
                data["kv"],
                data["score"],
                data["ape"],
                data["positions"],
                ref_state,
                data["state_slots"],
                compress_ratio=compress_ratio,
            )
        kwargs = {}
        if use_varlen_raw:
            kwargs.update(
                {
                    "seq_start_per_req": data["seq_start_per_req"],
                    "cu_seq_per_req": data["cu_seq_per_req"],
                }
            )
        run_fused_compress_kv_write(
            ref_state,
            data["token_to_req"],
            data["positions"],
            data["state_slots"],
            data["state_block_table"],
            data["norm_weight"],
            1.0e-6,
            data["cos_sin_cache"],
            data["ref_kv_cache"],
            data["kv_slots"],
            data["kv"],
            data["score"],
            data["ape"],
            0,
            disable_raw_path=disable_raw_path,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            state_tokens_per_block=int(data["state_tokens_per_block"]),
            **kwargs,
        )
        op_kv = data["kv"]
        op_score = data["score"]
        unpad_restore = None
        if use_unpad_restore_raw:
            perm = torch.tensor(
                [3, 0, 5, 1, 7, 2, 4, 6, 11, 8, 13, 9, 15, 10, 12, 14],
                dtype=torch.long,
                device=op_kv.device,
            )
            self.assertEqual(int(perm.numel()), int(op_kv.shape[0]))
            shuffled_kv = torch.empty_like(op_kv)
            shuffled_score = torch.empty_like(op_score)
            shuffled_kv.index_copy_(0, perm, op_kv)
            shuffled_score.index_copy_(0, perm, op_score)
            op_kv = shuffled_kv.contiguous()
            op_score = shuffled_score.contiguous()
            unpad_restore = perm.contiguous()

        op_kwargs = {
            "compressor_kv": op_kv,
            "compressor_score": op_score,
            "compressor_ape": data["ape"],
            "compressor_positions": data["positions"],
            "compressor_state_cache": data["state_cache"],
            "compressor_state_slots": data["state_slots"],
            "compressor_ratio": compress_ratio,
            "compressor_token_to_req": data["token_to_req"],
            "compressor_state_block_table": data["state_block_table"],
            "compressor_norm_weight": data["norm_weight"],
            "compressor_cos_sin_cache": data["cos_sin_cache"],
            "compressor_kv_cache": data["actual_kv_cache"],
            "compressor_kv_slots": data["kv_slots"],
            "compressor_seq_start": 0,
            "compressor_disable_raw_path": disable_raw_path,
            "compressor_rms_norm_eps": 1.0e-6,
            "compressor_head_dim": head_dim,
            "compressor_rope_head_dim": rope_head_dim,
            "compressor_overlap": overlap,
            "compressor_state_tokens_per_block": int(data["state_tokens_per_block"]),
        }
        if use_varlen_raw:
            op_kwargs.update(
                {
                    "compressor_seq_start_per_req": data["seq_start_per_req"],
                    "compressor_cu_seq_per_req": data["cu_seq_per_req"],
                }
            )
        if use_unpad_restore_raw:
            self.assertIsNotNone(unpad_restore)
            op_kwargs["compressor_unpad_restore"] = unpad_restore
        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **op_kwargs,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(data["actual_kv_cache"], data["ref_kv_cache"]))

    def test_candidate_cuda_hca_compressor_kv_cache_read_byte_equal_to_triton(
        self,
    ) -> None:
        self._run_compressor_kv_writer_fixture(
            _make_cache_read_compressor_kv_write_fixture(
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=128,
                overlap=False,
                entry_bytes=584,
                device=torch.device("cuda"),
            ),
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            disable_raw_path=True,
        )

    def test_candidate_cuda_indexer_compressor_kv_varlen_raw_byte_equal_to_triton(
        self,
    ) -> None:
        self._run_compressor_kv_writer_fixture(
            _make_varlen_raw_compressor_kv_write_fixture(
                head_dim=128,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                entry_bytes=132,
                device=torch.device("cuda"),
            ),
            head_dim=128,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            disable_raw_path=False,
            use_varlen_raw=True,
        )

    def test_candidate_cuda_indexer_compressor_kv_varlen_raw_restore_byte_equal_to_triton(
        self,
    ) -> None:
        self._run_compressor_kv_writer_fixture(
            _make_varlen_raw_compressor_kv_write_fixture(
                head_dim=128,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                entry_bytes=132,
                device=torch.device("cuda"),
            ),
            head_dim=128,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            disable_raw_path=False,
            use_varlen_raw=True,
            use_unpad_restore_raw=True,
        )


class DistributedAttentionTorchrunCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("candidate distributed attention op requires CUDA")
        if _candidate_op() is None:
            self.skipTest(
                "rtp_llm_ops.dsv4_cp_distributed_prefill_attention is not built yet"
            )
        if (
            torch.distributed.is_available()
            and not torch.distributed.is_initialized()
            and "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
        ):
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend="nccl")
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            self.skipTest("run under an initialized 8-rank distributed environment")
        if torch.distributed.get_world_size() != 8:
            self.skipTest("candidate baseline requires world_size=8")
        local_rank = int(os.environ.get("LOCAL_RANK", torch.distributed.get_rank()))
        torch.cuda.set_device(local_rank)

    def test_torchrun_rank_local_contract(self) -> None:
        """Distributed smoke for rank-local rows once run under torchrun."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="csa_torchrun_candidate",
            compress_ratio=4,
            prefix_lengths=(0, 15, 31),
            input_lengths=(9, 6, 5),
            window_size=16,
            compressed_topk=4,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"))
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )
        actual = op(
            fixture.q.contiguous(),
            kv,
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))

    def test_torchrun_symm_mem_protocol_contract(self) -> None:
        """Production ABI smoke: op consumes symmetric-memory handles on all ranks."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="hca_torchrun_symm_mem",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )
        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(rank_consistent_token_cap, 1),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=int(local_kv.numel() * local_kv.element_size()),
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        actual = op(
            fixture.q.contiguous(),
            local_kv.contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **buffer.op_kwargs(cp_rank=rank),
        )
        _assert_attention_close(actual, expected.index_select(0, rows))

    def test_torchrun_symm_mem_csa_indexer_payload_contract(self) -> None:
        """CSA smoke: KV and indexer-K payloads are both gathered inside op."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="csa_torchrun_symm_mem",
            compress_ratio=4,
            prefix_lengths=(0, 15, 31),
            input_lengths=(9, 6, 5),
            window_size=16,
            compressed_topk=4,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        local_indexer_k = _rank_local_4d_chunk(indexer_k, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )
        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(rank_consistent_token_cap, 1),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=max(
                int(local_kv.numel() * local_kv.element_size()),
                int(local_indexer_k.numel() * local_indexer_k.element_size()),
            ),
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        actual = op(
            fixture.q.contiguous(),
            local_kv.contiguous(),
            fixture.indexer_q.contiguous(),
            local_indexer_k.contiguous(),
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **buffer.op_kwargs(cp_rank=rank),
        )
        _assert_attention_close(actual, expected.index_select(0, rows))

    def _run_hca_grouped_mqa_long_key_case(
        self, *, splitk_env_value: Optional[str]
    ) -> None:
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="hca_torchrun_splitk_grouped_mqa",
            compress_ratio=128,
            prefix_lengths=(0, 640, 1408),
            input_lengths=(3, 2, 1),
            window_size=504,
            compressed_topk=8,
            n_heads=128,
            head_dim=512,
            index_heads=1,
            index_dim=1,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        mqa_kv_by_req = [kv[:, :1, :].contiguous() for kv in fixture.kv_by_req]
        fixture.kv_by_req = [
            kv.expand(-1, case.n_heads, -1).contiguous() for kv in mqa_kv_by_req
        ]
        expected = reference_attention(fixture)
        B = len(mqa_kv_by_req)
        max_kv_len = max(int(kv.shape[0]) for kv in mqa_kv_by_req)
        kv = torch.zeros(
            B,
            max_kv_len,
            1,
            case.head_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        for req, src in enumerate(mqa_kv_by_req):
            kv[req, : src.shape[0]].copy_(src)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        prefix_lengths = torch.tensor(
            case.prefix_lengths, dtype=torch.long, device=fixture.q.device
        )
        input_lengths = torch.tensor(
            case.input_lengths, dtype=torch.long, device=fixture.q.device
        )
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )
        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(rank_consistent_token_cap, 1),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=4 * 1024 * 1024,
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        env_name = "DSV4_CP_ATTENTION_MEGA_SPLITK"
        old_value = os.environ.get(env_name)
        if splitk_env_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = splitk_env_value
        try:
            actual = op(
                fixture.q.contiguous(),
                local_kv.contiguous(),
                fixture.indexer_q.contiguous(),
                torch.zeros(B, 1, 1, 1, dtype=fixture.q.dtype, device=fixture.q.device),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                **buffer.op_kwargs(cp_rank=rank),
            )
        finally:
            if old_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old_value
        _assert_attention_close(actual, expected.index_select(0, rows))

    def test_torchrun_symm_mem_splitk_hca_grouped_mqa_allclose(self) -> None:
        """Explicit split-K smoke: production grouped MQA path stays allclose."""
        self._run_hca_grouped_mqa_long_key_case(splitk_env_value="1")

    def test_torchrun_symm_mem_auto_splitk_hca_grouped_mqa_allclose(self) -> None:
        """Default long-key grouped MQA path auto-selects in-kernel split-K."""
        self._run_hca_grouped_mqa_long_key_case(splitk_env_value=None)

    def _run_csa_grouped_mqa_case(
        self,
        *,
        window_size: int,
        compressed_topk: int,
        splitk_env_value: Optional[str],
        scratch_bytes_per_rank: int,
    ) -> None:
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        prefix_lengths = (64, 509, 640) if window_size >= 507 else (64, 97, 130)
        case = AttentionCase(
            name="csa_torchrun_splitk_grouped_mqa",
            compress_ratio=4,
            prefix_lengths=prefix_lengths,
            input_lengths=(3, 2, 1),
            window_size=window_size,
            compressed_topk=compressed_topk,
            n_heads=128,
            head_dim=512,
            index_heads=64,
            index_dim=128,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        mqa_kv_by_req = [kv[:, :1, :].contiguous() for kv in fixture.kv_by_req]
        fixture.kv_by_req = [
            kv.expand(-1, case.n_heads, -1).contiguous() for kv in mqa_kv_by_req
        ]
        expected = reference_attention(fixture)
        B = len(mqa_kv_by_req)
        max_kv_len = max(int(kv.shape[0]) for kv in mqa_kv_by_req)
        kv = torch.zeros(
            B,
            max_kv_len,
            1,
            case.head_dim,
            dtype=fixture.q.dtype,
            device=fixture.q.device,
        )
        for req, src in enumerate(mqa_kv_by_req):
            kv[req, : src.shape[0]].copy_(src)
        _, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        local_indexer_k = _rank_local_4d_chunk(indexer_k, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )
        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(rank_consistent_token_cap, 1),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=scratch_bytes_per_rank,
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        env_name = "DSV4_CP_ATTENTION_MEGA_SPLITK"
        old_value = os.environ.get(env_name)
        if splitk_env_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = splitk_env_value
        try:
            actual = op(
                fixture.q.contiguous(),
                local_kv.contiguous(),
                fixture.indexer_q.contiguous(),
                local_indexer_k.contiguous(),
                fixture.attn_sink.contiguous(),
                fixture.req_id_per_token.contiguous(),
                fixture.position_ids.contiguous(),
                prefix_lengths,
                input_lengths,
                rows,
                case.compress_ratio,
                case.window_size,
                case.compressed_topk,
                **buffer.op_kwargs(cp_rank=rank),
            )
        finally:
            if old_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old_value
        _assert_attention_close(actual, expected.index_select(0, rows))

    def test_torchrun_symm_mem_splitk_csa_grouped_mqa_allclose(self) -> None:
        """Short-key CSA smoke: grouped MQA path supports topK and empty ranks."""
        self._run_csa_grouped_mqa_case(
            window_size=7,
            compressed_topk=5,
            splitk_env_value="1",
            scratch_bytes_per_rank=8 * 1024 * 1024,
        )

    def test_torchrun_symm_mem_auto_splitk_csa_grouped_mqa_allclose(self) -> None:
        """Long-key CSA smoke: default grouped MQA path auto-selects split-K."""
        self._run_csa_grouped_mqa_case(
            window_size=507,
            compressed_topk=5,
            splitk_env_value=None,
            scratch_bytes_per_rank=16 * 1024 * 1024,
        )

    def test_torchrun_symm_mem_swa_payload_cache_write_contract(self) -> None:
        """SWA smoke: rank-local fresh-K payload is gathered and cached in op."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="hca_torchrun_swa_write",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )

        full_swa_tokens = 21
        swa_k_full, _, _, _ = _make_swa_fixture(full_swa_tokens, torch.device("cuda"))
        local_swa_k = _rank_local_2d_chunk(swa_k_full, rank, case.cp_size)
        local_len = int(local_swa_k.shape[0])
        gathered_swa_ref = torch.cat(
            [
                _rank_local_2d_chunk(swa_k_full, peer, case.cp_size)
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        slot_mapping = torch.full(
            (local_len * case.cp_size,),
            -1,
            dtype=torch.long,
            device=fixture.q.device,
        )
        for peer in range(case.cp_size):
            for i in range(local_len):
                src_idx = peer * local_len + i
                if src_idx < full_swa_tokens:
                    slot_mapping[peer * local_len + i] = SWA_BLOCK_SIZE + src_idx
        ref_cache = _alloc_swa_cache(
            int(slot_mapping.clamp(min=0).max().item()), fixture.q.device
        )
        actual_cache = torch.empty_like(ref_cache)
        ref_cache.fill_(0x5A)
        actual_cache.fill_(0x5A)
        quantize_and_insert_k_cache(gathered_swa_ref, ref_cache, slot_mapping)

        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(rank_consistent_token_cap, local_len, 1),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=max(
                int(local_kv.numel() * local_kv.element_size()),
                int(local_swa_k.numel() * local_swa_k.element_size()),
            ),
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        op_kwargs = buffer.op_kwargs(cp_rank=rank)
        op_kwargs.update(
            {
                "swa_k": local_swa_k.contiguous(),
                "swa_k_cache": actual_cache,
                "swa_slot_mapping": slot_mapping.contiguous(),
            }
        )
        actual = op(
            fixture.q.contiguous(),
            local_kv.contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **op_kwargs,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual_cache, ref_cache))

    def test_torchrun_symm_mem_compressor_state_write_contract(self) -> None:
        """Compressor STATE smoke: rank-local kv/score are gathered and cached in op."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="hca_torchrun_compressor_state",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )

        (
            compressor_kv_full,
            compressor_score_full,
            compressor_ape,
            compressor_positions_full,
            compressor_slots_full,
            ref_state,
            actual_state,
        ) = _make_compressor_state_fixture(
            23,
            32,
            4,
            torch.device("cuda"),
            dtype=torch.bfloat16,
            skipped=(5, 21),
        )
        local_compressor_kv = _rank_local_2d_chunk(
            compressor_kv_full, rank, case.cp_size
        )
        local_compressor_score = _rank_local_2d_chunk(
            compressor_score_full, rank, case.cp_size
        )
        gathered_compressor_kv = torch.cat(
            [
                _rank_local_2d_chunk(compressor_kv_full, peer, case.cp_size)
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_compressor_score = torch.cat(
            [
                _rank_local_2d_chunk(compressor_score_full, peer, case.cp_size)
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_compressor_positions = torch.cat(
            [
                _rank_local_1d_chunk(
                    compressor_positions_full,
                    peer,
                    case.cp_size,
                    pad_value=0,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_compressor_slots = torch.cat(
            [
                _rank_local_1d_chunk(
                    compressor_slots_full,
                    peer,
                    case.cp_size,
                    pad_value=-1,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        run_save_partial_states(
            gathered_compressor_kv,
            gathered_compressor_score,
            compressor_ape,
            gathered_compressor_positions,
            ref_state,
            gathered_compressor_slots,
            compress_ratio=4,
        )

        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(
                rank_consistent_token_cap,
                int(local_compressor_kv.shape[0]),
                1,
            ),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=max(
                int(local_kv.numel() * local_kv.element_size()),
                int(local_compressor_kv.numel() * local_compressor_kv.element_size()),
                int(
                    local_compressor_score.numel()
                    * local_compressor_score.element_size()
                ),
            ),
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        op_kwargs = buffer.op_kwargs(cp_rank=rank)
        op_kwargs.update(
            {
                "compressor_kv": local_compressor_kv.contiguous(),
                "compressor_score": local_compressor_score.contiguous(),
                "compressor_ape": compressor_ape,
                "compressor_positions": gathered_compressor_positions,
                "compressor_state_cache": actual_state,
                "compressor_state_slots": gathered_compressor_slots,
                "compressor_ratio": 4,
            }
        )
        actual = op(
            fixture.q.contiguous(),
            local_kv.contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **op_kwargs,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_state, ref_state, rtol=0.0, atol=0.0)

    def test_torchrun_symm_mem_hca_compressor_kv_write_contract(self) -> None:
        """HCA compressed-KV smoke: in-op gather + 584B writer match Triton."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        case = AttentionCase(
            name="hca_torchrun_compressor_kv",
            compress_ratio=128,
            prefix_lengths=(0, 127, 260),
            input_lengths=(9, 7, 5),
            window_size=16,
            compressed_topk=8,
        )
        fixture = _make_case_fixture(case, torch.device("cuda"), dtype=torch.bfloat16)
        expected = reference_attention(fixture)
        kv, indexer_k, prefix_lengths, input_lengths = _pad_for_candidate_op(fixture)
        local_kv = _rank_local_4d_chunk(kv, rank, case.cp_size)
        rows = rank_local_rows(expected.shape[0], rank, case.cp_size).to(
            device=fixture.q.device
        )

        data = _make_raw_compressor_kv_write_fixture(
            num_tokens=128,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            entry_bytes=584,
            device=torch.device("cuda"),
        )
        local_compressor_kv = _rank_local_2d_chunk(data["kv"], rank, case.cp_size)
        local_compressor_score = _rank_local_2d_chunk(
            data["score"], rank, case.cp_size
        )
        gathered_positions = torch.cat(
            [
                _rank_local_1d_chunk(
                    data["positions"],
                    peer,
                    case.cp_size,
                    pad_value=0,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_state_slots = torch.cat(
            [
                _rank_local_1d_chunk(
                    data["state_slots"],
                    peer,
                    case.cp_size,
                    pad_value=-1,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_token_to_req = torch.cat(
            [
                _rank_local_1d_chunk(
                    data["token_to_req"],
                    peer,
                    case.cp_size,
                    pad_value=0,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_kv_slots = torch.cat(
            [
                _rank_local_1d_chunk(
                    data["kv_slots"],
                    peer,
                    case.cp_size,
                    pad_value=-1,
                )
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_compressor_kv = torch.cat(
            [
                _rank_local_2d_chunk(data["kv"], peer, case.cp_size)
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()
        gathered_compressor_score = torch.cat(
            [
                _rank_local_2d_chunk(data["score"], peer, case.cp_size)
                for peer in range(case.cp_size)
            ],
            dim=0,
        ).contiguous()

        ref_state = data["state_cache"].clone()
        run_save_partial_states(
            gathered_compressor_kv,
            gathered_compressor_score,
            data["ape"],
            gathered_positions,
            ref_state,
            gathered_state_slots,
            compress_ratio=128,
        )
        run_fused_compress_kv_write(
            ref_state,
            gathered_token_to_req,
            gathered_positions,
            gathered_state_slots,
            data["state_block_table"],
            data["norm_weight"],
            1.0e-6,
            data["cos_sin_cache"],
            data["ref_kv_cache"],
            gathered_kv_slots,
            gathered_compressor_kv,
            gathered_compressor_score,
            data["ape"],
            0,
            disable_raw_path=False,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            state_tokens_per_block=COMPRESSOR_STATE_BLOCK_SIZE,
        )

        rank_consistent_token_cap = (
            int(expected.shape[0]) + case.cp_size - 1
        ) // case.cp_size
        spec = Dsv4CpAttentionBufferSpec(
            cp_size=case.cp_size,
            max_tokens_per_rank=max(
                rank_consistent_token_cap,
                int(local_compressor_kv.shape[0]),
                1,
            ),
            batch_cap=len(case.input_lengths),
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=max(
                int(local_kv.numel() * local_kv.element_size()),
                int(local_compressor_kv.numel() * local_compressor_kv.element_size()),
                int(
                    local_compressor_score.numel()
                    * local_compressor_score.element_size()
                ),
            ),
        )
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        op_kwargs = buffer.op_kwargs(cp_rank=rank)
        op_kwargs.update(
            {
                "compressor_kv": local_compressor_kv.contiguous(),
                "compressor_score": local_compressor_score.contiguous(),
                "compressor_ape": data["ape"],
                "compressor_positions": gathered_positions,
                "compressor_state_cache": data["state_cache"],
                "compressor_state_slots": gathered_state_slots,
                "compressor_ratio": 128,
                "compressor_token_to_req": gathered_token_to_req,
                "compressor_state_block_table": data["state_block_table"],
                "compressor_norm_weight": data["norm_weight"],
                "compressor_cos_sin_cache": data["cos_sin_cache"],
                "compressor_kv_cache": data["actual_kv_cache"],
                "compressor_kv_slots": gathered_kv_slots,
                "compressor_seq_start": 0,
                "compressor_disable_raw_path": False,
                "compressor_rms_norm_eps": 1.0e-6,
                "compressor_head_dim": 512,
                "compressor_rope_head_dim": 64,
                "compressor_overlap": False,
                "compressor_state_tokens_per_block": COMPRESSOR_STATE_BLOCK_SIZE,
            }
        )
        actual = op(
            fixture.q.contiguous(),
            local_kv.contiguous(),
            fixture.indexer_q.contiguous(),
            indexer_k,
            fixture.attn_sink.contiguous(),
            fixture.req_id_per_token.contiguous(),
            fixture.position_ids.contiguous(),
            prefix_lengths,
            input_lengths,
            rows,
            case.compress_ratio,
            case.window_size,
            case.compressed_topk,
            **op_kwargs,
        )
        _assert_attention_close(actual, expected.index_select(0, rows))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(data["actual_kv_cache"], data["ref_kv_cache"]))

    def test_torchrun_symm_mem_hca_e2e_layer0_shape_contract(self) -> None:
        """Production layer-0 HCA shape smoke with all HCA side effects enabled."""
        op = _candidate_op()
        self.assertIsNotNone(op)
        rank = torch.distributed.get_rank()
        cp_size = 8
        chunk = int(os.environ.get("DSV4_DIST_ATTN_TEST_LAYER0_CHUNK", "2"))
        total_tokens = int(
            os.environ.get("DSV4_DIST_ATTN_TEST_LAYER0_TOTAL_TOKENS", str(cp_size * chunk - 1))
        )
        self.assertGreater(chunk, 0)
        self.assertGreater(total_tokens, 0)
        self.assertLessEqual(total_tokens, cp_size * chunk)
        n_heads = 128
        head_dim = 512
        device = torch.device("cuda")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260701)
        q_full = (
            torch.randn(
                cp_size * chunk,
                n_heads,
                head_dim,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.01
        ).to(torch.bfloat16).to(device)
        kv_full = (
            torch.randn(
                cp_size * chunk,
                head_dim,
                generator=gen,
                dtype=torch.float32,
            )
            * 0.01
        ).to(torch.bfloat16).to(device)
        local_q = q_full[rank * chunk : (rank + 1) * chunk].contiguous()
        local_kv = kv_full[rank * chunk : (rank + 1) * chunk].contiguous()

        positions = torch.arange(
            rank * chunk,
            (rank + 1) * chunk,
            dtype=torch.long,
            device=device,
        )
        req_ids = torch.zeros(chunk, dtype=torch.long, device=device)
        prefix_lengths = torch.zeros(1, dtype=torch.long, device=device)
        input_lengths = torch.tensor([total_tokens], dtype=torch.long, device=device)
        kv_cu_lens = torch.tensor([0, total_tokens], dtype=torch.long, device=device)
        if os.environ.get("DSV4_DIST_ATTN_TEST_SERVICE_METADATA", "0") == "1":
            if total_tokens == 15:
                kv_unpad_restore = torch.tensor(
                    [0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3],
                    dtype=torch.long,
                    device=device,
                )
            else:
                even = torch.arange(0, total_tokens, 2, dtype=torch.long, device=device)
                odd = torch.arange(
                    total_tokens - 1 if (total_tokens & 1) == 0 else total_tokens - 2,
                    0,
                    -2,
                    dtype=torch.long,
                    device=device,
                )
                kv_unpad_restore = torch.cat([even, odd], dim=0)
        else:
            kv_unpad_restore = torch.arange(total_tokens, dtype=torch.long, device=device)
        local_rows = torch.arange(chunk, dtype=torch.long, device=device)

        if os.environ.get("DSV4_DIST_ATTN_TEST_SERVICE_METADATA", "0") == "1":
            if cp_size * chunk == 16:
                full_slots = torch.tensor(
                    [132, -1, 133, 146, 134, 145, 135, 144, 136, 143, 137, 142, 138, 141, 139, 140],
                    dtype=torch.long,
                    device=device,
                )
            else:
                full_slots = torch.arange(cp_size * chunk, dtype=torch.long, device=device) + 132
                full_slots[torch.arange(1, cp_size * chunk, 17, device=device)] = -1
        else:
            full_slots = torch.full(
                (cp_size * chunk,),
                -1,
                dtype=torch.long,
                device=device,
            )
            full_slots[:total_tokens] = torch.arange(total_tokens, dtype=torch.long, device=device)
        swa_pool = torch.empty(
            32,
            132,
            SWA_ENTRY_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        swa_pool.fill_(0x5A)
        cmp_pool = torch.empty(
            128,
            1,
            SWA_ENTRY_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        cmp_pool.fill_(0x5A)
        state_cache = torch.empty(
            200,
            132,
            head_dim * 2,
            dtype=torch.float32,
            device=device,
        )
        state_cache.fill_(-123.0)
        if os.environ.get("DSV4_DIST_ATTN_TEST_SERVICE_METADATA", "0") == "1" and cp_size * chunk == 16:
            positions_padded = torch.tensor(
                [0, 0, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8],
                dtype=torch.long,
                device=device,
            )
        elif os.environ.get("DSV4_DIST_ATTN_TEST_SERVICE_METADATA", "0") == "1":
            positions_padded = torch.remainder(
                torch.arange(cp_size * chunk, dtype=torch.long, device=device) * 17,
                max(total_tokens, 1),
            )
        else:
            positions_padded = torch.arange(cp_size * chunk, dtype=torch.long, device=device)
        state_slots = full_slots.clone()
        token_to_req = torch.zeros(cp_size * chunk, dtype=torch.int32, device=device)
        kv_slots = torch.full((cp_size * chunk,), -1, dtype=torch.long, device=device)
        state_block_table_width = max(4, (total_tokens + 131) // 132 + 1)
        if os.environ.get("DSV4_DIST_ATTN_TEST_SERVICE_METADATA", "0") == "1":
            state_block_table = (
                torch.arange(
                    4,
                    4 + state_block_table_width,
                    dtype=torch.int32,
                    device=device,
                )
                .view(1, -1)
                .contiguous()
            )
        else:
            state_block_table = torch.zeros(1, state_block_table_width, dtype=torch.int32, device=device)
        norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
        cos_sin_cache = torch.zeros(256, 64, dtype=torch.float32, device=device)
        cos_sin_cache[:, :32] = 1.0
        ape = torch.zeros(128, head_dim, dtype=torch.float32, device=device)

        spec = Dsv4CpAttentionBufferSpec(
            cp_size=cp_size,
            max_tokens_per_rank=chunk,
            batch_cap=1,
            swa_bytes_per_token=1024,
            scratch_bytes_per_rank=max(
                int(local_kv.numel() * local_kv.element_size()),
                int(local_q.numel() * local_q.element_size()),
                4 * 1024 * 1024,
            ),
        )
        dummy_symm_buffer = None
        dummy_symm_handle = None
        if os.environ.get("DSV4_DIST_ATTN_TEST_PREALLOC_SYMM", "0") == "1":
            import torch.distributed._symmetric_memory as symm_mem

            dummy_symm_buffer = symm_mem.empty(
                1024 * 1024,
                dtype=torch.int8,
                device=device,
            )
            dummy_symm_handle = symm_mem.rendezvous(
                dummy_symm_buffer,
                group=torch.distributed.group.WORLD,
            )
            dummy_symm_buffer.zero_()
            torch.distributed.barrier()
            torch.cuda.synchronize()
        buffer = get_or_create_dsv4_cp_attention_buffer(
            group=torch.distributed.group.WORLD,
            cp_rank=rank,
            spec=spec,
        )
        actual = op(
            local_q,
            local_kv.view(1, chunk, 1, head_dim).contiguous(),
            local_q[:, :1, :1].contiguous(),
            torch.empty(1, 1, 1, 1, dtype=local_q.dtype, device=device),
            torch.zeros(n_heads, dtype=torch.float32, device=device),
            req_ids,
            positions,
            prefix_lengths,
            input_lengths,
            local_rows,
            128,
            512,
            4,
            **buffer.op_kwargs(cp_rank=rank),
            swa_k=local_kv.contiguous(),
            swa_k_cache=swa_pool,
            swa_slot_mapping=full_slots.contiguous(),
            compressor_kv=local_kv.contiguous(),
            compressor_score=torch.zeros_like(local_kv).contiguous(),
            compressor_ape=ape,
            compressor_positions=positions_padded.contiguous(),
            compressor_state_cache=state_cache,
            compressor_state_slots=state_slots.contiguous(),
            compressor_ratio=128,
            compressor_token_to_req=token_to_req.contiguous(),
            compressor_state_block_table=state_block_table,
            compressor_norm_weight=norm_weight,
            compressor_cos_sin_cache=cos_sin_cache,
            compressor_kv_cache=cmp_pool,
            compressor_kv_slots=kv_slots.contiguous(),
            compressor_seq_start=0,
            compressor_disable_raw_path=False,
            compressor_rms_norm_eps=1.0e-6,
            compressor_head_dim=head_dim,
            compressor_rope_head_dim=64,
            compressor_overlap=False,
            compressor_state_tokens_per_block=132,
            compressor_seq_start_per_req=torch.zeros(1, dtype=torch.int32, device=device),
            compressor_cu_seq_per_req=torch.tensor([0, total_tokens], dtype=torch.int32, device=device),
            compressor_unpad_restore=kv_unpad_restore,
            attention_cmp_k_pool=cmp_pool,
            attention_cmp_block_table=state_block_table,
            attention_cmp_seq_lens=torch.zeros(1, dtype=torch.int32, device=device),
            attention_swa_k_pool=swa_pool,
            attention_swa_slot_mapping=full_slots.view(1, -1).contiguous(),
            attention_swa_gather_lens=torch.zeros(1, dtype=torch.int32, device=device),
            kv_unpad_restore=kv_unpad_restore,
            kv_cu_lens=kv_cu_lens,
        )
        self.assertEqual(tuple(actual.shape), (chunk, n_heads, head_dim))
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(actual.float()).all().item())


if __name__ == "__main__":
    unittest.main()
