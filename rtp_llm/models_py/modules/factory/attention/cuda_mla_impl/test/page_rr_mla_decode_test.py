"""Page-RR Decode production pipeline against a replicated mathematical reference.

DCP_TEST_WORLD_SIZE selects 4/8/16 ranks. A smaller-rank pass validates the
same contracts but does not imply that a larger communication topology ran.
"""

import json
import gc
import logging
import multiprocessing as mp
import os
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_mla_impl
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import mla_dcp_comm
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_dcp_comm import (
    MlaDcpCommunicator,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_decode import (
    PageRRMlaDecodeImpl,
    PageRRMlaDecodeOp,
)
from rtp_llm.ops import (DecodeCPMLABackend, DecodeCPMLAFusionMode,
                         KvCacheDataType, NcclCommConfig, ParallelismConfig, RoleType)
from rtp_llm.server.server_args.util import str2_decode_cp_mla_fusion_mode, str2_decode_cp_mla_a2a_backend
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.model_weight import W


def _cycle_prefixes(batch, values):
    return [values[index % len(values)] for index in range(batch)]


# Every case runs metadata preparation, the real distributed attention path,
# a replicated mathematical reference, and mutable CUDA Graph replay.
_CASES = (
    (1, 1, [31]),
    (2, 1, [127, 128]),
    (8, 1, [1023, 1024, 1025, 4095, 4096, 4097, 32767, 32768]),
    (32, 1, _cycle_prefixes(32, [63, 64, 65, 127, 128, 129])),
    (128, 1, _cycle_prefixes(128, [31, 32, 33, 127, 128, 129])),
    (512, 1, _cycle_prefixes(512, [31, 32, 33, 127, 128, 129])),
    (1, 2, [32767]),
    (8, 2, _cycle_prefixes(8, [4095, 4096, 4097, 32767])),
    (256, 2, _cycle_prefixes(256, [4095, 4096])),
    (32, 4, _cycle_prefixes(32, [65535, 65536, 65537, 131071])),
    (128, 4, _cycle_prefixes(128, [1023, 1024, 1025, 4095, 4096, 4097])),
    (8, 8, _cycle_prefixes(8, [131071, 131072, 131073])),
    (64, 8, _cycle_prefixes(64, [4095, 4096, 4097])),
    (16, 16, _cycle_prefixes(16, [262143, 262144, 262145])),
    (32, 16, _cycle_prefixes(32, [1048575, 1048576, 1048577])),
)


def _cases(size, backend):
    if "DCP_TEST_CASES" in os.environ:
        return json.loads(os.environ["DCP_TEST_CASES"])
    if backend == DecodeCPMLABackend.TOKENSPEED:
        # Keep the existing backend's regression matrix. The additional FP8
        # short-prefix counterexample also fails the unmodified TS 0.2.3 path;
        # an explicit case table can reproduce it without weakening its oracle.
        return _CASES
    cases = list(_CASES)
    cases.extend([
        {"B": 4, "Q": 4, "prefixes": [0, 31, 32, 16384],
         "q_scale": 0.01, "kv_scale": 0.02},
        (2, 33, [0, 1]),
        (2, 3, [127, 128]),
        (2, 5, [4, 33]),
        (1, 65, [63]),
        (4, 4, [8191, 8192, 524287, 524288]),
        (4, 4, [4095, 4096, 16383, 16384]),
        (12, 4, _cycle_prefixes(12, [4096, 16384])),
        (13, 4, _cycle_prefixes(13, [4096, 16384])),
        # Retiring a fused workspace must not unload CUDA code during a later
        # capture. Keep both cases in this same real factory/forward workflow.
        {"B": 16, "Q": 4, "prefixes": [8192] * 16, "fusion_mode": "FUSED"},
        {"B": 1, "Q": 4, "prefixes": [8192], "fusion_mode": "UNFUSED", "collect_during_capture": True},
        (2, 32, [0, 1]),
        (2, 64, [0, 1]),
        (2, 128, [127, 128]),
        (128, 8, _cycle_prefixes(128, [1023, 1024, 1025])),
        {"B": 4, "Q": 1, "prefixes": [31, 32, 129, 599],
         "history": [[31, 32, 129, 599], [32, 31, 129, 599], [0, 39, 129, 599]]},
    ])
    pages = (2, 4, 8, 16, 32, 64, 128)
    boundaries = [31, 32, 33, 127, 128, 129, 128 * size - 1, 128 * size, 128 * size + 1]
    for index, heads in enumerate(range(size, 129, size)):
        cases.append({"B": 128, "Q": 1, "prefixes": _cycle_prefixes(128, boundaries),
                      "heads": heads, "kernel_page": pages[index % len(pages)]})
    for page in pages:
        cases.append({"B": 16, "Q": 4, "kernel_page": page,
                      "prefixes": _cycle_prefixes(16, [page - 1, page, page + 1, 511, 512, 513])})
    if 100 % size == 0:
        cases.append({"B": 2, "Q": 33, "prefixes": [0, 1], "heads": 100, "kernel_page": 4})
        # 2100 token/head rows leave a tail in the eight-row local reducer.
        cases.append({"B": 21, "Q": 1, "heads": 100,
                      "prefixes": _cycle_prefixes(21, [0, 1, 16384]), "strided_cache": True})
    cases.extend([
        {"B": 8, "Q": 4, "prefixes": [0, 1, 127, 128, 8192, 16384, 16383, 4095],
         "strided_cache": True},
        {"B": 16, "Q": 4, "prefixes": _cycle_prefixes(16, [127, 128, 129]),
         "heads": 128, "kernel_page": 8},
        {"B": 16, "Q": 4, "prefixes": _cycle_prefixes(16, [1023, 1024, 1025, 4095, 4096, 4097]),
         "page": 1024, "kernel_page": 128},
        {"B": 16, "Q": 4, "prefixes": _cycle_prefixes(16, [0, 1, 127, 128, 4095, 4096]),
         "kernel_page": 128, "strided_cache": True},
        {"B": 2, "Q": 33, "prefixes": [0, 1], "kernel_page": 4, "strided_cache": True},
        # Exercise both grid.y's 65535 limit and >2^31-element Q/wire offsets.
        {"B": 512, "Q": 128, "prefixes": [4096] * 512, "heads": 128, "kernel_page": 128},
        # BF16 packing counts 64-bit words, so crossing its signed offset
        # boundary requires a larger tensor than the element-offset case.
        {"B": 1024, "Q": 128, "prefixes": [4096] * 1024, "heads": 128, "kernel_page": 128},
        {"B": 2, "Q": 4, "prefixes": [126, 127], "dominant_last_key": True},
        {"B": 2, "Q": 4, "prefixes": [16383, 16384], "dominant_last_key": True,
         "strided_cache": True},
        # Different model capacities can produce the same table shape across
        # kernel pages. Each PAGE specialization must be ready before forward.
        {"B": 16, "Q": 4, "prefixes": _cycle_prefixes(16, [0, 1, 31, 128]),
         "page": 1024, "kernel_page": 128, "model_max_seq_len": 5120,
         "forbid_pack_compile": True},
        {"B": 16, "Q": 4, "prefixes": _cycle_prefixes(16, [0, 1, 31, 128]),
         "page": 1024, "kernel_page": 64, "model_max_seq_len": 1024,
         "forbid_pack_compile": True},
    ])
    for mode in ("AUTO", "FUSED", "UNFUSED"):
        for batch, lengths in (
            (1, [0]),
            (1, [8192]),
            (13, _cycle_prefixes(13, [0, 1, 8192, 16384])),
        ):
            cases.append({"B": batch, "Q": 4, "prefixes": lengths,
                          "fusion_mode": mode})
    # Exercise probability rounding after an early outlier in long KV and
    # distinguish two concentrated keys, alongside the ideal-math diagnostic.
    cases.extend([
        {"B": 13, "Q": 4, "prefixes": [8188] * 13, "early_score_outlier": 2.0},
        {"B": 13, "Q": 4, "prefixes": [1048572] * 13, "early_score_outlier": 4.0},
        {"B": 13, "Q": 4, "prefixes": [131068] * 13, "early_score_outlier": 4.0,
         "early_score_outlier_values": (-2.0, 2.0)},
        {"B": 1, "Q": 4, "prefixes": [131068], "early_score_outlier": 4.0,
         "early_score_outlier_values": (-2.0, 2.0)},
    ])
    # Forced UNFUSED also reaches the packed wire offsets at large T.
    cases.append({"B": 1024, "Q": 128, "prefixes": [4096] * 1024,
                  "heads": 128, "kernel_page": 128, "fusion_mode": "UNFUSED"})
    # Same real factory/forward/Graph workflow, including transitions between
    # transports on one workspace and tail payloads outside AUTO's policy.
    for selector in ("AUTO", "NCCL", "CUSTOM"):
        for batch, queries, heads, auto_transport in (
            (1, 4, 96, "nccl"), (2, 4, 96, "pull"),
            (1, 1, 40, "nccl"), (129, 4, 96, "nccl"),
        ):
            cases.append({"B": batch, "Q": queries, "heads": heads,
                          "prefixes": _cycle_prefixes(batch, [0, 1, 8192]),
                          "independent_requests": True,
                          "fusion_mode": "UNFUSED", "a2a_backend": selector,
                          "forbid_copy_compile": selector == "CUSTOM",
                          "a2a_transport": ("pull" if selector == "CUSTOM" else
                                            "nccl" if selector == "NCCL" else auto_transport)})
        cases.append({"B": 13, "Q": 4, "prefixes": [8192] * 13,
                      "independent_requests": True,
                      "fusion_mode": "FUSED", "a2a_backend": selector,
                      "a2a_transport": "producer"})
    return cases


def _rotate(value, positions, cos_sin):
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    while cos.ndim < value.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x, y = value.float().chunk(2, dim=-1)
    # Native RoPE contracts the cos product into a float32 FMA. Preserve that
    # rounding before BF16/FP8 conversion; separate multiply/add can cross a
    # BF16 rounding boundary, which is observable in the exact cache check.
    return torch.cat(
        (torch.addcmul(-y * sin, x, cos), torch.addcmul(x * sin, y, cos)), dim=-1
    ).to(value.dtype)


def _fixture(
    rank,
    size,
    queries,
    fp8,
    generation=0,
    *,
    draft=False,
    q_replicated=False,
    prefix_lengths=None,
    page=128,
    kernel_page=64,
    heads=96,
    independent_requests=False,
    poison_new_pages=False,
    strided_cache=False,
    device_index=None,
    dominant_last_key=False,
    early_score_outlier=0.0,
    early_score_outlier_values=(0.0, 0.0),
    kv_size=None,
    kv_rank=None,
    request_ids=None,
    q_scale=1.0,
    kv_scale=1.0,
    cache_capacity_tokens=None,
    graph_capacity_tokens=None,
    projection_gate=True,
):
    torch.manual_seed(1701 + generation)
    device = torch.device("cuda", rank if device_index is None else device_index)
    cache_size = size if kv_size is None else kv_size
    cache_rank = rank if kv_rank is None else kv_rank
    latent, rope, nope, value = 512, 64, 128, 128
    # The short request crosses into a 128:1 split during replay; the other
    # spans a whole CP period. This distinguishes LSE weighting from averaging.
    prefixes = (
        list(prefix_lengths)
        if prefix_lengths is not None
        else [page - 2 + generation, page * size + 3 + generation]
    )
    batch, local_heads = len(prefixes), heads // size
    if independent_requests:
        # Equal lengths must not hide a request-index error. Large stress cases
        # may still share reference content to bound oracle memory consumption.
        unique_prefixes = list(prefixes)
        prefix_groups = list(range(batch))
    else:
        unique_prefixes = list(dict.fromkeys(prefixes))
        prefix_to_group = {prefix: group for group, prefix in enumerate(unique_prefixes)}
        prefix_groups = [prefix_to_group[prefix] for prefix in prefixes]
    group_count = len(unique_prefixes)
    group_ids = torch.tensor(prefix_groups, dtype=torch.long, device=device)
    lengths = [p + queries for p in prefixes]
    tokens = batch * queries
    cache_length = max(max(lengths), cache_capacity_tokens or 0)
    max_len = cache_length + 16
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    # K3's dense_e4m3_v1 production contract uses unit Q/KV scales. Extra
    # non-unit cases remain explicit; they must not replace that contract.
    q_scale, kv_scale = (q_scale, kv_scale) if fp8 else (1.0, 1.0)
    q_groups = (
        torch.randn(group_count, queries, heads, nope + rope, device=device) * 0.2
    ).bfloat16()
    ckv_groups = (
        torch.randn(group_count, queries, latent, device=device) * 0.2
    ).bfloat16()
    k_pe_groups = (
        torch.randn(group_count, queries, rope, device=device) * 0.2
    ).bfloat16()
    torch.manual_seed(3301)
    kc = (torch.randn(heads, nope, latent, device=device) * 0.02).bfloat16()
    vc = (torch.randn(heads, latent, value, device=device) * 0.02).bfloat16()
    if dominant_last_key:
        # A query's final speculative key dominates once visible. Earlier
        # queries must exclude it, even across Page-RR ownership boundaries.
        q_groups.zero_()
        q_groups[..., :16] = 4
        k_pe_groups.zero_()
        ckv_groups[:, -1, :16] = 4
        kc.zero_()
        diagonal = torch.arange(16, device=device)
        kc[:, diagonal, diagonal] = 1
    if early_score_outlier:
        assert min(prefixes) >= 2
        q_groups.zero_()
        q_groups[..., :16] = early_score_outlier
        k_pe_groups.zero_()
        ckv_groups.zero_()
        ckv_groups[..., 16] = 2
        kc.zero_()
        diagonal = torch.arange(16, device=device)
        kc[:, diagonal, diagonal] = 1
        vc.zero_()
        vc[:, 16, 0] = 1
    angles = (
        torch.arange(max_len, device=device)[:, None]
        * (torch.arange(rope // 2, device=device)[None, :] + 1)
        / 97
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    group_positions = torch.tensor(
        [[prefix + query for query in range(queries)] for prefix in unique_prefixes],
        dtype=torch.int32,
        device=device,
    )
    q_rotated_groups = q_groups.clone()
    q_rotated_groups[..., nope:] = _rotate(
        q_groups[..., nope:].reshape(group_count * queries, heads, rope),
        group_positions.reshape(-1).long(),
        cos_sin,
    ).view(group_count, queries, heads, rope)
    k_rotated_groups = _rotate(
        k_pe_groups.reshape(group_count * queries, rope),
        group_positions.reshape(-1).long(),
        cos_sin,
    ).view(group_count, queries, rope)
    canonical = (
        torch.randn(group_count, max_len, latent + rope, device=device) * 0.2
    ).bfloat16()
    if early_score_outlier:
        # Valid finite Q/K/V: early high logits, then many small-probability
        # keys with nonzero values. Random zero-mean data hides their loss.
        canonical.zero_()
        canonical[..., 16] = 2
        canonical[:, 0, 16] = early_score_outlier_values[0]
        canonical[:, 1, 16] = early_score_outlier_values[1]
        canonical[:, :2, :16] = 4
        canonical[:, 1, 0] = 3.75
    for group, prefix in enumerate(unique_prefixes):
        canonical[group, prefix : prefix + queries] = torch.cat(
            (ckv_groups[group], k_rotated_groups[group]), dim=-1
        )
    encoded = (
        (canonical.float() / kv_scale).clamp(-448, 448).to(dtype) if fp8 else canonical
    )
    if request_ids is not None:
        # Generate the same global requests before selecting a DP owner's rows.
        # The RNG sequence and reference data must not depend on topology.
        assert independent_requests and request_ids
        prefixes = [prefixes[index] for index in request_ids]
        prefix_groups = [prefix_groups[index] for index in request_ids]
        group_ids = torch.tensor(prefix_groups, dtype=torch.long, device=device)
        batch, tokens = len(prefixes), len(prefixes) * queries
        lengths = [prefix + queries for prefix in prefixes]
    q_all = q_groups[group_ids].reshape(tokens, heads, nope + rope)
    q_rotated = q_rotated_groups[group_ids].reshape(tokens, heads, nope + rope)
    ckv = ckv_groups[group_ids].reshape(tokens, latent)
    k_pe = k_pe_groups[group_ids].reshape(tokens, rope)
    # K3 normalizes the latent KV into its own output, but K PE remains a
    # slice of the Q-latent/KV/RoPE (and optionally gate) projection buffer.
    k_pe_offset = 1536 + latent
    projection_width = k_pe_offset + rope + (local_heads * value if projection_gate else 0)
    k_projection = torch.zeros((tokens, projection_width), device=device, dtype=k_pe.dtype)
    k_projection[:, k_pe_offset:k_pe_offset + rope].copy_(k_pe)
    k_pe = k_projection[:, k_pe_offset:k_pe_offset + rope]
    positions = group_positions[group_ids].reshape(tokens)

    # Real kernel tables are padded views and may select a native draft FULL
    # group distinct from the singular view last selected by a KDA layer.
    width = (cache_length + page * cache_size - 1) // (page * cache_size) * (page // kernel_page)
    table_width = width
    if graph_capacity_tokens is not None:
        # CudaGraphRunner reserves global model capacity plus MTP proposals;
        # only live local pages below are backed by real KV allocations.
        table_width = ((graph_capacity_tokens + page - 1) // page + queries - 1) * (page // kernel_page)
        assert table_width >= width
    table_storage = torch.full(
        (3, batch, table_width + 3), -1, dtype=torch.int32, device=device
    )
    group_id = 2 if draft else 0
    table = table_storage[group_id, :, :table_width]
    table.zero_()
    page_count = batch * width
    logical_pages = torch.arange(page_count, device=device).view(batch, width)
    physical_pages = 1 + (page_count - 1 - logical_pages + generation) % page_count
    table[:, :width].copy_(physical_pages.to(torch.int32))
    local_pages = torch.arange(width, device=device)
    pages_per_owner = page // kernel_page
    global_starts = ((local_pages // pages_per_owner) * cache_size + cache_rank) * page + (
        local_pages % pages_per_owner
    ) * kernel_page
    global_positions = global_starts[:, None] + torch.arange(kernel_page, device=device)
    safe_positions = global_positions.clamp_max(max_len - 1)
    packed_pages = encoded[group_ids[:, None, None], safe_positions[None]]
    valid = (
        global_positions[None] < torch.tensor(prefixes, device=device)[:, None, None]
    )
    packed_pages = torch.where(
        valid[..., None], packed_pages, torch.zeros((), dtype=dtype, device=device)
    )
    cache = SimpleNamespace()
    cache_shape = (page_count + 1, kernel_page, latent + rope)
    if strided_cache:
        # Model a view into a larger pool with independent page/token strides
        # and a nonzero, TMA-aligned storage offset.
        entry_stride = latent + rope + 32
        block_stride = kernel_page * entry_stride + 128
        offset = 64
        storage = torch.full((offset + cache_shape[0] * block_stride + 64,),
                             -7, dtype=dtype, device=device)
        raw_cache = storage.as_strided(cache_shape, (block_stride, entry_stride, 1), offset)
        raw_cache.zero_()
        guard = torch.ones(storage.numel(), dtype=torch.bool, device=device)
        # The native new-page clear owns the entire entry stride, including
        # token padding. Only pool prefix/suffix and inter-page gaps are guards.
        guard.as_strided((cache_shape[0], kernel_page * entry_stride),
                         (block_stride, 1), offset).fill_(False)
        cache.guard_storage, cache.guard_mask = storage, guard
    else:
        raw_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    cache.kv_cache_base = raw_cache
    raw_cache[physical_pages.reshape(-1)] = packed_pages.reshape(
        page_count, kernel_page, latent + rope
    )
    new_pages = torch.empty(0, dtype=torch.long, device=device)
    if poison_new_pages:
        prefix_tensor = torch.tensor(prefixes, device=device)[:, None]
        new_page_mask = (global_starts[None] >= prefix_tensor) & (
            global_starts[None] < prefix_tensor + queries
        )
        new_pages = physical_pages[new_page_mask].long()
        raw_cache[new_pages] = float("nan")
    fields = dict(
        is_prefill=queries > 1,
        is_target_verify=queries > 1 and not draft,
        is_mtp_draft_update=queries > 1 and draft,
        is_cuda_graph=False,
        total_tokens=tokens,
        input_lengths=torch.full((batch,), queries, dtype=torch.int32, device=device),
        input_lengths_host=torch.full((batch,), queries, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32, device=device),
        prefix_lengths_host=torch.tensor(prefixes, dtype=torch.int32),
        sequence_lengths=torch.tensor(prefixes, dtype=torch.int32, device=device),
        sequence_lengths_plus_1_d=torch.tensor(
            [p + 1 for p in prefixes], dtype=torch.int32, device=device
        ),
        kv_cache_kernel_block_id_device_by_group=[
            table_storage[g, :, :table_width] for g in range(3)
        ],
        kv_cache_kernel_block_id_device=table_storage[1, :, :table_width],
        # Native Graph keeps the CPU map in this field, without a _host mirror.
        kv_cache_layer_to_group=torch.tensor(
            [2] if draft else [1, 0], dtype=torch.int32
        ).pin_memory(),
        cache_store_inputs=None,
    )
    inputs = PyAttentionInputs()
    for name, field in fields.items():
        setattr(inputs, name, field)
    config = SimpleNamespace(
        head_num=local_heads,
        kv_lora_rank=latent,
        rope_head_dim=rope,
        nope_head_dim=nope,
        kernel_tokens_per_block=kernel_page,
        tokens_per_block=page,
        softmax_extra_scale=1.0,
        use_mla=True,
        indexer_topk=2048,
        is_sparse=False,
        mla_fp8_compute=fp8,
        mla_fp8_q_scale=q_scale,
        mla_fp8_kv_scale=kv_scale,
        kv_cache_dtype=KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE,
        rope_config=SimpleNamespace(is_neox_style=True),
    )
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    weights = [
        {
            W.mla_kc: kc if q_replicated else kc[head_slice],
            W.mla_vc: vc[head_slice],
        }
    ]
    if not draft:
        weights.insert(0, {})  # The target starts with LINEAR, not FULL MLA.
    absorbed = torch.bmm(q_rotated[..., :nope].transpose(0, 1), kc).transpose(0, 1)
    absorbed = torch.cat((absorbed, q_rotated[..., nope:]), dim=-1)
    representative_rows = torch.tensor(
        list(range(batch)) if independent_requests else [prefixes.index(prefix) for prefix in unique_prefixes],
        dtype=torch.long,
        device=device,
    )
    local_q = absorbed.view(batch, queries, heads, latent + rope)[
        representative_rows, :, head_slice
    ]
    del absorbed
    if fp8:
        # Quantization is elementwise: select the oracle's request/head rows
        # first, avoiding a full replicated FP32 temporary in address tests.
        # The production fixture inputs and their full sizes are unchanged.
        local_q = (local_q.float() / q_scale).clamp(-448, 448).to(dtype)
    oracle_groups = representative_rows.numel()
    oracle_kv = encoded[group_ids[representative_rows]]
    oracle_prefixes = list(prefixes) if independent_requests else unique_prefixes
    scores = torch.bmm(
        local_q.reshape(oracle_groups, queries * local_heads, latent + rope).float(),
        oracle_kv.float().transpose(1, 2),
    ).view(oracle_groups, queries, local_heads, max_len)
    scores.mul_((nope + rope) ** -0.5 * q_scale * kv_scale)
    causal_lengths = torch.tensor(
        [
            [prefix + query + 1 for query in range(queries)]
            for prefix in oracle_prefixes
        ],
        dtype=torch.int64,
        device=device,
    )
    causal_mask = torch.arange(max_len, device=device).view(1, 1, 1, -1)
    scores.masked_fill_(causal_mask >= causal_lengths[:, :, None, None], -float("inf"))
    if cache_size == 1:
        merged = torch.bmm(
            scores.softmax(-1).reshape(oracle_groups, queries * local_heads, max_len),
            oracle_kv[..., :latent].float(),
        ).view(oracle_groups, queries, local_heads, latent)
        merged.mul_(kv_scale)
    else:
        # RTP transports each owner's local context as BF16 before the
        # cross-rank LSE merge. Preserve that rounding in the independent
        # mathematical oracle; a single global softmax misses this boundary.
        owner = (torch.arange(max_len, device=device) // page) % cache_size
        contexts, logs = [], []
        values = oracle_kv[..., :latent].float()
        for peer in range(cache_size):
            peer_scores = scores.masked_fill(owner != peer, -float("inf"))
            log = peer_scores.logsumexp(-1)
            safe_log = torch.where(torch.isfinite(log), log, 0.)
            probability = torch.exp(peer_scores - safe_log[..., None])
            context = torch.bmm(probability.reshape(oracle_groups, queries * local_heads, max_len), values)
            contexts.append((context.view(oracle_groups, queries, local_heads, latent) * kv_scale).bfloat16())
            logs.append(log)
        weight = torch.stack(logs).softmax(0)
        merged = (torch.stack(contexts).float() * weight[..., None]).sum(0)
    merged_by_head = (
        merged.bfloat16()
        .permute(2, 0, 1, 3)
        .reshape(local_heads, oracle_groups * queries, latent)
    )
    expected_groups = torch.bmm(merged_by_head, vc[head_slice]).permute(1, 0, 2)
    expected = expected_groups.view(oracle_groups, queries, local_heads, value)[
        torch.arange(batch, device=device) if independent_requests else group_ids
    ].reshape(tokens, local_heads, value)
    return SimpleNamespace(
        config=config,
        inputs=inputs,
        weights=weights,
        cos_sin=cos_sin,
        q=(q_all if q_replicated else q_all[:, head_slice]).contiguous(),
        ckv=ckv,
        k_pe=k_pe,
        k_projection=k_projection,
        k_pe_offset=k_pe_offset,
        cache=cache,
        expected=expected,
        positions=positions,
        canonical=encoded,
        prefix_groups=prefix_groups,
        prefixes=prefixes,
        dtype=dtype,
        layer_id=0 if draft else 1,
        group_id=group_id,
        new_pages=new_pages,
        probability_reference_inputs=SimpleNamespace(
            query=local_q,
            canonical_groups=group_ids[representative_rows].tolist(),
            prefixes=oracle_prefixes,
            output_rows=list(range(batch)) if independent_requests else group_ids.tolist(),
            softmax_scale=(nope + rope) ** -0.5 * q_scale * kv_scale,
            output_scale=kv_scale,
            cache_size=cache_size,
        ),
    )


def _clone_query_inputs(fixture):
    # clone(view) compacts K PE and would erase the production stride being
    # tested. Clone the projection first, then recover its mutable RoPE view.
    projection = fixture.k_projection.clone()
    k_pe = projection[:, fixture.k_pe_offset:fixture.k_pe_offset + 64]
    assert k_pe.stride() == fixture.k_pe.stride()
    assert k_pe.storage_offset() == fixture.k_pe.storage_offset()
    return fixture.q.clone(), k_pe


def _reference_local_prefix_tokens(position, rank, size, page=128):
    # Count owned complete pages and the optional partial page. This avoids
    # scanning a million Python integers for every query of the 1M fixtures.
    pages, tail = divmod(position + 1, page)
    owned_pages = (pages + size - 1 - rank) // size
    return owned_pages * page + (tail if pages % size == rank else 0)


def _probability_split_reference(query, keys, visible, scale):
    """Logical-array reference for upstream's input-dtype, K128 online P.

    This models probability rounding, not producer addresses or loaded pages.
    The ideal FP32 mathematical result remains separately in fixture.expected.
    """
    queries, heads, _ = query.shape
    if keys.shape[0] == 0:
        return (torch.zeros((queries, heads, 512), device=query.device),
                torch.full((queries, heads), -torch.inf, device=query.device))
    scores = torch.matmul(query.float().reshape(queries * heads, -1), keys.float().T)
    scores = scores.view(queries, heads, -1) * scale
    mask = torch.arange(keys.shape[0], device=query.device)[None, None, :] >= visible[:, None, None]
    scores = scores.masked_fill(mask, -torch.inf)
    padded = torch.nn.functional.pad(scores, (0, (-keys.shape[0]) % 128), value=-torch.inf)
    tiles = padded.view(queries, heads, -1, 128)
    running_max = tiles.amax(-1).cummax(-1).values
    running_max = torch.where(torch.isfinite(running_max), running_max, 0.)
    last_max = running_max[..., -1]
    probability = torch.exp(tiles - running_max[..., None])
    rounded = probability.to(query.dtype).float()
    effective = rounded * torch.exp(running_max - last_max[..., None])[..., None]
    effective = effective.flatten(-2)[..., :keys.shape[0]]
    denominator = torch.exp(scores - last_max[..., None]).sum(-1)
    numerator = torch.matmul(effective, keys[:, :512].float())
    context = numerator / torch.where(denominator > 0, denominator, 1.)[..., None]
    return context, torch.log(denominator) + last_max


def _probability_reference_output(fixture, backend):
    """Retain upstream P precision and the selected path's wire rounding."""
    if backend is None:
        return fixture.expected
    fp32_peer = backend.fused and backend.splits > 1
    key = (backend.splits, fp32_peer)
    if not hasattr(fixture, "probability_expected"):
        fixture.probability_expected = {}
    if key not in fixture.probability_expected:
        ref = fixture.probability_reference_inputs
        queries, heads = ref.query.shape[1:3]
        groups = []

        def merge(contexts, logs):
            logs = torch.stack(logs)
            maximum = logs.max(0).values
            maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
            weights = torch.exp(logs - maximum)
            denominator = weights.sum(0)
            value = (torch.stack(contexts).float() * weights[..., None]).sum(0)
            value /= torch.where(denominator > 0, denominator, 1.)[..., None]
            return value, torch.logsumexp(logs, 0)

        for group, canonical_group in enumerate(ref.canonical_groups):
            prefix = ref.prefixes[group]
            positions = torch.arange(prefix + queries, device=ref.query.device)
            owners = (positions // fixture.config.tokens_per_block) % ref.cache_size
            contexts, logs = [], []
            for owner in range(ref.cache_size):
                keys = fixture.canonical[canonical_group, :prefix + queries][owners == owner]
                visible = torch.tensor([
                    _reference_local_prefix_tokens(prefix + q, owner, ref.cache_size,
                                                   fixture.config.tokens_per_block)
                    for q in range(queries)
                ], device=ref.query.device)
                # Use the already-selected split count, while independently
                # constructing each owner's logical key sequence and cutoff.
                tiles = (keys.shape[0] + 127) // 128
                width = ((tiles + backend.splits - 1) // backend.splits) * 128
                parts, part_logs = [], []
                for part in range(backend.splits):
                    begin, end = min(part * width, keys.shape[0]), min((part + 1) * width, keys.shape[0])
                    value, log = _probability_split_reference(
                        ref.query[group], keys[begin:end],
                        (visible - begin).clamp(0, end - begin), ref.softmax_scale,
                    )
                    parts.append(value * ref.output_scale)
                    part_logs.append(log)
                if fp32_peer:
                    contexts.extend(parts)
                    logs.extend(part_logs)
                else:
                    value, log = merge(parts, part_logs)
                    contexts.append(value.bfloat16())
                    logs.append(log)
            groups.append(merge(contexts, logs)[0].bfloat16())
        merged = torch.stack(groups).permute(2, 0, 1, 3).reshape(heads, -1, 512)
        values = torch.bmm(merged, fixture.weights[fixture.layer_id][W.mla_vc]).permute(1, 0, 2)
        expected = values.view(len(groups), queries, heads, -1)[ref.output_rows].reshape(
            len(fixture.prefixes) * queries, heads, -1)
        fixture.probability_expected[key] = expected
    return fixture.probability_expected[key]


def _assert_result(fixture, impl, output, rank, size, expected=None):
    atol = 2e-3 if fixture.dtype == torch.float8_e4m3fn else 1e-3
    if expected is None:
        expected = _probability_reference_output(fixture, impl.fmha_impl.fia2a_backend)
    torch.testing.assert_close(output, expected, atol=atol, rtol=0.015)
    if rank == 0:
        dtype_name = "FP8" if fixture.dtype == torch.float8_e4m3fn else "BF16"
        delta = (output.float() - fixture.expected.float()).abs()
        print(f"DCP {dtype_name}_MATH_DIAGNOSTIC max_abs={float(delta.max())} "
              f"outside_original_tolerance={int((delta > atol + .015 * fixture.expected.float().abs()).sum())}", flush=True)
    metadata = impl.fmha_params
    torch.testing.assert_close(metadata.positions_d, fixture.positions, rtol=0, atol=0)
    expected_lengths = [
        _reference_local_prefix_tokens(p, rank, size, fixture.config.tokens_per_block)
        for p in fixture.positions.tolist()
    ]
    assert metadata.local_causal_lens.flatten().tolist() == expected_lengths
    slots = metadata.slot_mapping.tolist()
    owned_slots, groups, positions = [], [], []
    for row, position in enumerate(fixture.positions.tolist()):
        if (position // fixture.config.tokens_per_block) % size != rank:
            assert slots[row] == -1
        else:
            assert slots[row] >= 0
            batch = row // (len(slots) // len(fixture.prefixes))
            owned_slots.append(slots[row])
            groups.append(fixture.prefix_groups[batch])
            positions.append(position)
    if owned_slots:
        device = fixture.cache.kv_cache_base.device
        ids = lambda values: torch.tensor(values, dtype=torch.long, device=device)
        slots_d = ids(owned_slots)
        page_size = fixture.config.kernel_tokens_per_block
        actual = fixture.cache.kv_cache_base[slots_d // page_size, slots_d % page_size]
        expected = fixture.canonical[ids(groups), ids(positions)]
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0,
            atol=0 if fixture.dtype == torch.float8_e4m3fn else 1e-6,
        )
    if fixture.new_pages.numel():
        assert torch.isfinite(fixture.cache.kv_cache_base[fixture.new_pages].float()).all(), "New-page tails were not cleared before KV writes"
    if hasattr(fixture.cache, "guard_mask"):
        guarded = fixture.cache.guard_storage.float()[fixture.cache.guard_mask]
        assert torch.all(guarded == -7), "KV writes modified storage outside the page spans"


def _worker(rank, size, port, failed):
    torch.cuda.set_device(rank)
    dp_size = int(os.environ.get("DCP_TEST_DP_SIZE", "1"))
    assert size % dp_size == 0
    tp_size, tp_rank = size // dp_size, rank % (size // dp_size)
    owner = rank // tp_size
    parallelism = ParallelismConfig()
    parallelism.world_rank = parallelism.local_rank = rank
    parallelism.world_size = parallelism.local_world_size = size
    parallelism.tp_rank, parallelism.tp_size = tp_rank, tp_size
    parallelism.dp_rank, parallelism.dp_size = owner, dp_size
    parallelism.role_type = RoleType.DECODE
    parallelism.decode_cp_kv_cache_sharded = True
    init_distributed_environment(
        parallelism, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=180
    )
    graph = None
    try:
        from rtp_llm.server.server_args.server_args import setup_args

        with patch("sys.argv", ["page_rr_mla_decode_test"]):
            fmha_config = setup_args().fmha_config
        if os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1":
            logging.getLogger().setLevel(logging.INFO)
        if (os.environ.get("DCP_TEST_GRAPH_LIFECYCLE_ONLY") == "1"
                or os.environ.get("DCP_TEST_DRAFT_GRAPH_ONLY") == "1"):
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.page_rr_mla_graph_lifecycle import run_lifecycle

            run_lifecycle(rank, size, parallelism, fmha_config, _fixture, _assert_result,
                          _probability_reference_output)
            return
        cases = _cases(tp_size, fmha_config.decode_cp_mla_backend)
        total_cases = len(cases)
        geometry_plans = {}
        configured_fusion_mode = fmha_config.decode_cp_mla_fusion_mode
        configured_a2a_backend = fmha_config.decode_cp_mla_a2a_backend
        for q_replicated in (False, True):
            parallelism.decode_cp_q_replicated = q_replicated
            for fp8 in (False, True):
                for case_index, case in enumerate(cases, start=1):
                    if isinstance(case, dict) and "by_owner" in case:
                        assert len(case["by_owner"]) == dp_size
                        case = case["by_owner"][owner]
                    if isinstance(case, dict):
                        batch, queries, prefixes = case["B"], case["Q"], case["prefixes"]
                        history = case.get("history", [prefixes] * 3)
                    else:
                        batch, queries, prefixes = case
                        history = [prefixes] * 3
                        case = {}
                    fmha_config.decode_cp_mla_fusion_mode = (
                        str2_decode_cp_mla_fusion_mode(case["fusion_mode"])
                        if "fusion_mode" in case else configured_fusion_mode
                    )
                    fmha_config.decode_cp_mla_a2a_backend = (
                        str2_decode_cp_mla_a2a_backend(case["a2a_backend"])
                        if "a2a_backend" in case else configured_a2a_backend
                    )
                    # The broad matrix includes the 1M boundary on both sides.
                    # Keep its model capacity fixed at 2M; production-capacity
                    # regressions explicitly select 1M or the serving limit.
                    model_max_seq_len = case.get("model_max_seq_len", int(
                        os.environ.get("DCP_TEST_MODEL_MAX_SEQ_LEN", str(2 * 1024 * 1024))
                    ))
                    live_capacity = max(max(p) for p in history) + queries
                    assert 0 < live_capacity <= model_max_seq_len
                    options = dict(
                        heads=case.get("heads", int(os.environ.get("DCP_TEST_HEADS", "96"))),
                        kernel_page=case.get("kernel_page", int(os.environ.get("DCP_TEST_KERNEL_PAGE", "64"))),
                        page=case.get("page", 128),
                        independent_requests=case.get("independent_requests", batch * live_capacity <= 1048576),
                        poison_new_pages=True,
                        strided_cache=case.get("strided_cache", False),
                        device_index=rank,
                        dominant_last_key=case.get("dominant_last_key", False),
                        early_score_outlier=case.get("early_score_outlier", 0.0),
                        early_score_outlier_values=case.get("early_score_outlier_values", (0.0, 0.0)),
                        q_scale=case.get("q_scale", 1.0),
                        kv_scale=case.get("kv_scale", 1.0),
                        cache_capacity_tokens=live_capacity,
                        graph_capacity_tokens=model_max_seq_len,
                        projection_gate=case.get("projection_gate", True),
                    )
                    if tp_rank == 0:
                        print(
                            f"DCP START ranks={size} owner={owner} TP={tp_size} q_replicated={q_replicated} "
                            f"dtype={'FP8' if fp8 else 'BF16'} "
                            f"case={case_index}/{total_cases} "
                            f"batch={batch} q={queries}",
                            flush=True,
                        )
                    fixture = _fixture(
                        tp_rank,
                        tp_size,
                        queries,
                        fp8,
                        draft=not fp8,
                        q_replicated=q_replicated,
                        prefix_lengths=history[0],
                        **options,
                    )
                    if queries == 1:
                        # Match the runner's synthetic capture descriptor;
                        # replay below switches to live plus-one lengths.
                        fixture.inputs.sequence_lengths_plus_1_d.zero_()
                    allocated_before_prepare = torch.cuda.memory_allocated()
                    impl = get_mla_impl(
                        fixture.config,
                        SimpleNamespace(
                            weights=fixture.weights,
                            get_global_weight=lambda key: fixture.cos_sin,
                        ),
                        fixture.inputs,
                        fmha_config=fmha_config,
                        max_seq_len=model_max_seq_len,
                        parallelism_config=parallelism,
                        is_cuda_graph=True,
                    )
                    assert isinstance(impl, PageRRMlaDecodeImpl)
                    if "max_prepare_allocation_bytes" in case:
                        allocated = torch.cuda.memory_allocated() - allocated_before_prepare
                        print(f"DCP PREPARE_MEMORY rank={rank} bytes={allocated} "
                              f"B={batch} Q={queries} model={model_max_seq_len}", flush=True)
                        assert allocated <= case["max_prepare_allocation_bytes"], (
                            f"factory allocated {allocated} bytes for a short live KV; "
                            f"model={model_max_seq_len}, B={batch}, Q={queries}"
                        )
                    assert (impl.fmha_impl.fia2a_backend is not None) == (
                        fmha_config.decode_cp_mla_backend == DecodeCPMLABackend.FIA2A
                    )
                    if rank == 0:
                        print(
                            f"DCP READY ranks={size} q_replicated={q_replicated} "
                            f"dtype={'FP8' if fp8 else 'BF16'} "
                            f"case={case_index}/{total_cases} backend=page_rr",
                            flush=True,
                        )
                    comm = mla_dcp_comm.get_mla_dcp(
                        fixture.config, fixture.q.device, fixture.dtype
                    )
                    assert comm.backend == "a2a"
                    assert (
                        impl.fmha_params.cache_group_id == fixture.group_id
                    ), f"Page-RR selected group {impl.fmha_params.cache_group_id}, expected {fixture.group_id}"
                    q, k_pe = _clone_query_inputs(fixture)
                    compilation_guard = nullcontext()
                    if fp8 and case.get("forbid_pack_compile", False):
                        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a.ops import _pack

                        compilation_guard = patch.object(
                            _pack, "compile",
                            side_effect=AssertionError("FIA2A pack must be compiled in prepare"),
                        )
                    elif case.get("forbid_copy_compile", False):
                        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a.ops import _a2a_pull
                        compilation_guard = patch.object(
                            _a2a_pull, "compile",
                            side_effect=AssertionError("FIA2A A2A copy must be compiled in prepare"),
                        )
                    route_dir = os.environ.get("DCP_TEST_ROUTE_TRACE_DIR")
                    capture_route = bool(route_dir and q_replicated and "fusion_mode" in case)
                    route_profiler = (
                        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
                        if capture_route else nullcontext()
                    )
                    with compilation_guard, route_profiler as profile:
                        output = impl.forward(
                            q, fixture.ckv, k_pe, fixture.cache, fixture.layer_id,
                        )
                        if capture_route:
                            torch.cuda.synchronize()
                    _assert_result(fixture, impl, output, tp_rank, tp_size)
                    if capture_route:
                        path = Path(route_dir) / f"case{case_index}-fp8{int(fp8)}-rank{rank}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        profile.export_chrome_trace(str(path))
                        kernels = [event["name"] for event in json.loads(path.read_text())["traceEvents"]
                                   if event.get("cat") == "kernel"]
                        backend = impl.fmha_impl.fia2a_backend
                        barriers = sum("PeerBarrier" in name for name in kernels)
                        nccl = any("nccl" in name.lower() for name in kernels)
                        local_merge = any("merge_local_splits" in name for name in kernels)
                        pull = backend.a2a_buffers is not None
                        assert barriers == (2 if backend.fused or pull else 0), (path, kernels)
                        assert nccl == (not backend.fused and not pull), (path, kernels)
                        assert any("_a2a_pull" in name for name in kernels) == pull, (path, kernels)
                        assert local_merge == (not backend.fused and backend.splits > 1), (path, kernels)
                        print(f"DCP GPU ROUTE PASS rank={rank} requested={backend.requested_mode.name} "
                              f"mode={backend.mode.name} S={backend.splits} trace={path}", flush=True)
                    if impl.fmha_impl.fia2a_backend is not None and "mode" in case:
                        assert impl.fmha_impl.fia2a_backend.mode == str2_decode_cp_mla_fusion_mode(case["mode"].upper())
                    backend = impl.fmha_impl.fia2a_backend
                    if backend is not None:
                        if "a2a_transport" in case:
                            transport = ("producer" if backend.fused else
                                         "pull" if backend.a2a_buffers is not None else "nccl")
                            assert transport == case["a2a_transport"], (case, transport)
                        plan = (backend.splits, backend.mode)
                        geometry = (fixture.dtype, batch, queries, backend.heads, backend.page_size)
                        # Model capacity and live KV do not change query work.
                        assert geometry_plans.setdefault(geometry, backend.splits) == backend.splits
                        if backend.requested_mode == DecodeCPMLAFusionMode.AUTO:
                            assert backend.fused == (backend.splits == 1)
                        else:
                            assert backend.mode == backend.requested_mode
                    if tp_rank == 0 and backend is not None:
                        print(f"DCP FIA2A EXECUTED mode={backend.mode.name.lower()} S={backend.splits} "
                              f"requested_mode={backend.requested_mode.name} "
                              f"a2a_transport={'pull' if backend.a2a_buffers is not None else 'nccl' if not backend.fused else 'producer'} "
                              f"owner={owner} TP={tp_size} B={batch} Q={queries} H={backend.heads} P={backend.page_size} "
                              f"model_max_seq_len={model_max_seq_len} actual_max_kv={max(prefixes)+queries} "
                              f"kv_stride={fixture.cache.kv_cache_base.stride()} "
                              f"kv_offset={fixture.cache.kv_cache_base.storage_offset()}", flush=True)
                    q, k_pe = _clone_query_inputs(fixture)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            impl.forward(
                                q, fixture.ckv, k_pe, fixture.cache, fixture.layer_id
                            )
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        if case.get("collect_during_capture", False):
                            gc.collect()
                        output = impl.forward(
                            q, fixture.ckv, k_pe, fixture.cache, fixture.layer_id
                        )
                    addresses = tuple(
                        x.data_ptr()
                        for x in (
                            impl.fmha_params.positions_d,
                            impl.fmha_params.slot_mapping,
                            impl.fmha_params.local_causal_lens,
                            (impl.fmha_params.query_block_tables
                             if impl.fmha_params.query_block_tables is not None
                             else impl.fmha_params.block_tables),
                        )
                        if x is not None
                    )
                    for generation in (1, 2):
                        live = _fixture(
                            tp_rank,
                            tp_size,
                            queries,
                            fp8,
                            generation,
                            draft=not fp8,
                            q_replicated=q_replicated,
                            prefix_lengths=history[generation],
                            **options,
                        )
                        # Weights/RoPE stay fixed; mutate live tensors at
                        # their captured addresses, including physical IDs.
                        q.copy_(live.q)
                        k_pe.copy_(live.k_pe)
                        fixture.ckv.copy_(live.ckv)
                        fixture.cache.kv_cache_base.copy_(live.cache.kv_cache_base)
                        for field in (
                            "prefix_lengths",
                            "sequence_lengths",
                            "sequence_lengths_plus_1_d",
                        ):
                            getattr(fixture.inputs, field).copy_(
                                getattr(live.inputs, field)
                            )
                        groups = fixture.inputs.kv_cache_kernel_block_id_device_by_group
                        updated = live.inputs.kv_cache_kernel_block_id_device_by_group[live.group_id]
                        if case.get("replace_table_on_replay", False):
                            groups[fixture.group_id] = updated
                            fixture.inputs.kv_cache_kernel_block_id_device_by_group = groups
                        else:
                            groups[fixture.group_id].copy_(updated)
                        impl.prepare_cuda_graph(fixture.inputs)
                        allocated = torch.cuda.memory_allocated()
                        graph.replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_allocated() == allocated
                        live.cache = fixture.cache
                        _assert_result(live, impl, output, tp_rank, tp_size)
                        if backend is not None:
                            assert (backend.splits, backend.mode) == plan
                        assert addresses == tuple(
                            x.data_ptr()
                            for x in (
                                impl.fmha_params.positions_d,
                                impl.fmha_params.slot_mapping,
                                impl.fmha_params.local_causal_lens,
                                (impl.fmha_params.query_block_tables
                                 if impl.fmha_params.query_block_tables is not None
                                 else impl.fmha_params.block_tables),
                            )
                            if x is not None
                        )
                    graph.reset()
                    graph = None
                    torch.cuda.synchronize()
                    if tp_rank == 0:
                        print(
                            f"DCP PASS ranks={size} owner={owner} TP={tp_size} backend={fmha_config.decode_cp_mla_backend.name} q_replicated={q_replicated} "
                            f"dtype={fixture.dtype} batch={batch} q={queries}",
                            flush=True,
                        )

        fixture = _fixture(tp_rank, tp_size, 1, False, device_index=rank)
        a2a = MlaDcpCommunicator(96 // tp_size, 576, 512, fixture.dtype, fixture.q.device)
        assert a2a.backend == "a2a"
        try:
            MlaDcpCommunicator(96 // tp_size, 576, 512, torch.float32, fixture.q.device)
        except ValueError as error:
            assert "BF16/E4M3" in str(error)
        else:
            raise AssertionError(
                "unsupported communication dtype must fail during initialization"
            )
        if rank == 0:
            print(
                "DCP PASS fixed A2A backend and lazy communicator contract", flush=True
            )
    except BaseException:
        # Report the first error before any distributed teardown can wait.
        import traceback

        traceback.print_exc()
        failed.set()
        raise
    finally:
        # NCCL finalization waits for captured graph callbacks to be released.
        # Destroy the graph while its communicator is still alive.
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
        mla_dcp_comm._communicators.clear()
        destroy_distributed_environment()


class PageRRMlaDecodeTest(unittest.TestCase):
    def test_prefix_reference_matches_token_ownership(self):
        for size in (2, 4, 8, 16):
            counts = [0] * size
            for position in range(128 * size * 3 + 19):
                counts[(position // 128) % size] += 1
                for rank in range(size):
                    self.assertEqual(
                        _reference_local_prefix_tokens(position, rank, size),
                        counts[rank],
                    )

    def test_production_page_rr_decode(self):
        size = int(os.environ.get("DCP_TEST_WORLD_SIZE", "8"))
        self.assertIn(size, (2, 4, 8, 16))
        self.assertGreaterEqual(
            torch.cuda.device_count(),
            size,
            "DCP test must not pass by skipping missing GPUs",
        )
        mp.set_start_method("spawn", force=True)
        ports, locks = PortManager().get_consecutive_ports(1)
        failed = mp.Event()
        processes = [
            mp.Process(
                target=_worker, args=(rank, size, ports[0], failed), name=f"dcp-rank-{rank}"
            )
            for rank in range(size)
        ]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + int(
                os.environ.get("DCP_TEST_WORKER_TIMEOUT", "3000")
            )
            pending = list(processes)
            while pending:
                self.assertFalse(failed.is_set(), "worker failed before distributed cleanup; see its first traceback")
                for process in pending[:]:
                    if process.exitcode is not None:
                        process.join()
                        self.assertEqual(process.exitcode, 0, process.name)
                        pending.remove(process)
                self.assertLess(
                    time.monotonic(), deadline,
                    f"Workers timed out: {[process.name for process in pending]}",
                )
                if pending:
                    time.sleep(0.1)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                if process.pid is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


class PageRREmptyPartialTest(unittest.TestCase):
    def setUp(self):
        # Load the compatibility bridge lazily: spawn must not import TokenSpeed
        # while unpickling the distributed worker.
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
            _load_tokenspeed_mla,
        )

        self.assertTrue(_load_tokenspeed_mla())
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            tokenspeed_mla_page_rr,
        )

        self.tokenspeed = tokenspeed_mla_page_rr
        self.query = torch.ones((1, 2, 2, 6), dtype=torch.bfloat16, device="cuda")
        self.cache = self.query.new_ones(1, 4, 6)
        self.workspace = torch.empty(4096, dtype=torch.int8, device="cuda")
        self.tables = torch.zeros((2, 1), dtype=torch.int32, device="cuda")
        self.lengths = self.tables.new_tensor([[0, 1]])
        backend = patch.object(
            self.tokenspeed.backend,
            "tokenspeed_mla_decode",
            side_effect=self._dirty_backend,
        )
        backend.start()
        self.addCleanup(backend.stop)

    def _dirty_backend(self, **kwargs):
        output = kwargs["out"].fill_(7)
        lse = torch.full(output.shape[:3], 11, dtype=torch.float32, device="cuda")
        self.dirty = output, lse
        return output, lse

    def test_standalone_api_normalizes_dirty_empty_rows(self):
        output, lse = self.tokenspeed.tokenspeed_mla_page_rr_decode(
            self.query,
            self.cache,
            self.workspace,
            4,
            2,
            self.tables,
            self.lengths,
            4,
            0.5,
        )
        self.assertTrue((output[0, 0] == 0).all())
        self.assertTrue(torch.isneginf(lse[0, 0]).all())
        self.assertTrue((output[0, 1] == 7).all())
        self.assertTrue((lse[0, 1] == 11).all())

    def test_page_rr_op_masks_dirty_empty_rows_without_cleanup_on_graph_replay(self):
        packed = self.query.new_empty(1, 2, 2, 6)

        def combine(partial, lse, lengths):
            mla_dcp_comm._pack_a2a[(2, 2)](
                partial,
                lse,
                lengths,
                packed,
                packed.view(torch.float32),
                2,
                heads=2,
                local_heads=2,
                dim=4,
                block=8,
            )
            return packed[0, :, :, :4].permute(1, 0, 2).contiguous()

        op = object.__new__(PageRRMlaDecodeOp)
        op.num_heads = op.all_heads = op.query_heads = 2
        op.q_replicated = True
        op.kv_lora_rank, op.qk_rope_head_dim = 4, 2
        op.fp8_compute, op.bmm1_scale = False, 0.5
        op.communicator = SimpleNamespace(combine=combine)
        op._workspace = self.workspace
        op.metadata = SimpleNamespace(
            local_causal_lens=self.lengths,
            query_block_tables=self.tables,
            kernel_page_size=4,
        )
        op.weights = [
            {
                W.mla_kc: self.query.new_zeros(2, 2, 4),
                W.mla_vc: self.query.new_ones(2, 4, 3),
            }
        ]
        query = self.query.new_ones(2, 2, 2)
        cache = SimpleNamespace(kv_cache_base=self.cache)
        # A real PageRR forward must delegate empty masking to the packing
        # kernel. Both output and LSE from the backend are deliberately dirty.
        with patch.object(
            self.tokenspeed,
            "_set_empty_partial_identity",
            side_effect=AssertionError("redundant empty cleanup launched"),
        ):
            op.forward(query, query, cache, 0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = op.forward(query, query, cache, 0)

        for _ in range(2):
            packed.fill_(23)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue((self.dirty[0][0] == 7).all())
            self.assertTrue((self.dirty[1][0] == 11).all())
            self.assertTrue((packed[0, 0, :, :4] == 0).all())
            self.assertTrue(
                torch.isneginf(packed.view(torch.float32)[0, 0, :, 2]).all()
            )
            self.assertTrue((packed[0, 1, :, :4] == 7).all())
            self.assertTrue((packed.view(torch.float32)[0, 1, :, 2] == 11).all())
            self.assertTrue((output[0] == 0).all())
            self.assertTrue((output[1] == 28).all())


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
