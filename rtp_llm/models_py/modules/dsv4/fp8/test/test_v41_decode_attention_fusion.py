"""Focused V4.1 routing, selection, and CUDA Graph integration contracts.

The paged scorer is replaced by the original FP32 score over the actual
grouped FP8 cache. This isolates attention routing/candidate semantics from
DeepGEMM numerical tests, and also runs on GPUs that cannot execute DeepGEMM.
"""

import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    AttentionV41FP8,
    fp8_roundtrip,
    rope_only,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import paged_topk_translator


def _keys_from_packed(pool, table, capacity, logical_entries):
    blocks, physical, _ = pool.shape
    raw = pool.view(blocks, -1)
    quantized = raw[:, : physical * 128].contiguous().view(torch.float8_e4m3fn)
    scales = raw[:, physical * 128 :].contiguous().view(torch.float32)
    quantized = quantized.view(blocks, physical, 128).float()
    columns = torch.arange(capacity, device=pool.device)
    ids = table[:, columns // logical_entries].long()
    offsets = columns % logical_entries
    return quantized[ids, offsets] * scales[ids, offsets, None]


def _original_paged_score(q, weights, freqs, pool, table, lengths, **kwargs):
    """Old arithmetic with the paged helper's causal -inf output contract."""
    capacity = kwargs["max_ctx_len"]
    b, s, h, d = q.shape
    keys = _keys_from_packed(pool, table, capacity, kwargs["logical_entries_per_block"])
    query = fp8_roundtrip(rope_only(q.reshape(b * s, h, d).clone(), freqs, 64))
    folded = weights.reshape(b * s, h)
    scores = []
    for request in range(b):
        start = request * s
        logits = torch.einsum(
            "thd,kd->thk", query[start : start + s], keys[request]
        ).relu_()
        scores.append((logits * folded[start : start + s, :, None]).sum(1))
    result = torch.cat(scores)
    columns = torch.arange(capacity, device=q.device)
    return result.masked_fill(columns[None] >= lengths.reshape(-1, 1), -torch.inf)


def _fixture(device, ratio, candidate=False):
    torch.manual_seed(410 + ratio)
    batch, span, capacity, physical = 2, 6, 768, 128
    logical = 128 // ratio
    pages = capacity // logical
    table = (
        torch.arange(1, 2 * pages + 1, dtype=torch.int32, device=device)
        .view(batch, pages)
        .flip(1)
        .contiguous()
    )
    blocks = 2 * pages + 1
    pool = torch.empty(blocks, physical, 132, dtype=torch.uint8, device=device)
    values = torch.randn(blocks, physical, 128, device=device)
    scales = torch.rand(blocks, physical, device=device) + 0.25
    if ratio == 2:
        # Padding is deliberately nonzero and vastly larger than real K.
        values[:, logical:].fill_(128)
        scales[:, logical:].fill_(1024)
    raw = pool.view(blocks, -1)
    raw[:, : physical * 128].copy_(
        values.to(torch.float8_e4m3fn).view(torch.uint8).reshape(blocks, -1)
    )
    raw[:, physical * 128 :].copy_(scales.view(torch.uint8).reshape(blocks, -1))
    owner = 20 if ratio == 1 else 2
    packed = indexer.DecodeIndexerKeys(pool, table, capacity, logical)
    state = {"global": {owner: packed}, "topk": {}}
    angles = torch.rand(2048, 32, device=device) * 6.28
    attn = SimpleNamespace(
        is_index_source=True,
        layer_id=owner,
        kv_source_layer_id=owner,
        index_source_layer_id=owner,
        compress_ratio=ratio,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        rope_head_dim=64,
        index_wq=torch.randn(4096, 16, device=device, dtype=torch.bfloat16) * 0.125,
        index_weights=torch.randn(32, 8, device=device, dtype=torch.bfloat16),
        freqs_cis=torch.polar(torch.ones_like(angles), angles),
        v41_config={
            "candidate_source_layer_id": 20 if candidate else -1,
            "candidate_topk_blocks": 8 if candidate else 0,
            "candidate_block_size": 8 if candidate else 0,
        },
        _shared_attention=state,
        _lin=lambda weight, x: F.linear(x, weight),
    )
    x = torch.randn(batch, span, 8, device=device, dtype=torch.bfloat16)
    qr = torch.randn(batch, span, 16, device=device, dtype=torch.bfloat16)
    # Request zero exercises short-context ascending order and ratio2 zero
    # visibility. Request one exercises top-k and the newest candidate block.
    positions = (
        torch.tensor([0, 700 * ratio], device=device)[:, None]
        + torch.arange(span, device=device)
    ).flatten()
    return attn, packed, x, qr, positions


def _select(attn, x, qr, positions):
    return AttentionV41FP8._select_indices_decode(attn, x, qr, positions)


def _reference(attn, packed, x, qr, positions, candidates=None):
    old = SimpleNamespace(**vars(attn))
    keys = _keys_from_packed(
        packed.pool,
        packed.block_table,
        packed.capacity,
        packed.logical_entries_per_block,
    )
    old._shared_attention = {"global": {attn.kv_source_layer_id: keys}, "topk": {}}
    if candidates is not None:
        old._shared_attention["candidates"] = candidates.clone()
    result = _select(old, x, qr, positions)
    return result, old._shared_attention.get("candidates")


class V41DecodeAttentionFusionCPU(unittest.TestCase):
    def test_paged_source_routing_per_token_lengths_and_short_context(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                attn, packed, x, qr, positions = _fixture("cpu", ratio)
                # A later index source must still use the KV owner's cache.
                attn.layer_id += 4
                attn._shared_attention["global"][attn.layer_id] = object()
                with patch.object(
                    indexer, "score_decode_indexer", side_effect=_original_paged_score
                ) as scorer:
                    got = _select(attn, x, qr, positions)
                expected, _ = _reference(attn, packed, x, qr, positions)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                args = scorer.call_args.args
                self.assertIs(args[3], packed.pool)
                self.assertIs(args[4], packed.block_table)
                torch.testing.assert_close(
                    args[5], ((positions + 1) // ratio).view(2, 6), rtol=0, atol=0
                )
                self.assertEqual(
                    scorer.call_args.kwargs["logical_entries_per_block"], 128 // ratio
                )
                for row in range(6):
                    visible = int((positions[row] + 1) // ratio)
                    torch.testing.assert_close(
                        got[row, :visible],
                        torch.arange(visible, dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue((got[row, visible:] == -1).all())
                attn.is_index_source = False
                attn.index_source_layer_id = attn.layer_id
                with patch.object(indexer, "score_decode_indexer") as unused:
                    self.assertIs(_select(attn, x, qr, positions), got)
                    unused.assert_not_called()

    def test_candidate_source_consumer_and_latest_block_are_preserved(self):
        attn, packed, x, qr, positions = _fixture("cpu", 1, candidate=True)
        with patch.object(
            indexer, "score_decode_indexer", side_effect=_original_paged_score
        ):
            source = _select(attn, x, qr, positions)
            candidates = attn._shared_attention["candidates"].clone()
            expected_source, expected_candidates = _reference(
                attn, packed, x, qr, positions
            )
            torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
            torch.testing.assert_close(candidates, expected_candidates, rtol=0, atol=0)
            newest = positions // 8
            self.assertTrue((candidates == newest[:, None]).any(-1).all())
            attn.layer_id = 24
            consumer = _select(attn, x, qr, positions)
            expected, _ = _reference(attn, packed, x, qr, positions, candidates)
            torch.testing.assert_close(consumer, expected, rtol=0, atol=0)
            for row in range(6, 12):
                valid = consumer[row][consumer[row] >= 0]
                self.assertTrue(torch.isin(valid // 8, candidates[row]).all())

    def test_unexpected_support_change_is_not_silently_fallback(self):
        attn, _, x, qr, positions = _fixture("cpu", 2)
        with patch.object(indexer, "score_decode_indexer", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "support changed"):
                _select(attn, x, qr, positions)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41DecodeAttentionFusionCUDA(unittest.TestCase):
    def test_actual_slot_gate_and_padded_pool_fallback(self):
        for ratio in (1, 2):
            for padded in (False, True):
                with self.subTest(ratio=ratio, padded=padded):
                    logical = 128 // ratio
                    entries = logical * (2 if padded else 1)
                    region = CSA_KV if ratio == 2 else HCA_KV
                    pool = torch.empty(
                        4, entries, 584, dtype=torch.uint8, device="cuda"
                    )
                    table = torch.tensor(
                        [[1, 0, 3], [2, 3, 0]], dtype=torch.int32, device="cuda"
                    )
                    selected = (
                        torch.tensor(
                            [-1, -2, 0, logical - 1, logical, logical + 1, 3 * logical],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        .repeat(12, 74)[:, :512]
                        .contiguous()
                    )
                    requests = torch.arange(
                        2, device="cuda", dtype=torch.int32
                    ).repeat_interleave(6)
                    attn = SimpleNamespace(
                        compress_ratio=ratio,
                        _cp_ctx=None,
                        _kv_cache=SimpleNamespace(
                            kernel_seq_size_per_block=128, seq_size_per_block=256
                        ),
                        _block_tables_by_type={region: table},
                        _global_region=lambda: region,
                        _source_pool=lambda _: pool,
                        _source_entries=lambda _, p: p.shape[1],
                    )
                    attn._slots = MethodType(AttentionV41FP8._slots, attn)
                    with patch.dict(os.environ, {"DSV41_FUSED_DECODE_SLOTS": "0"}):
                        expected = AttentionV41FP8._decode_global_slots(
                            attn, selected, requests
                        )
                    with patch.dict(
                        os.environ, {"DSV41_FUSED_DECODE_SLOTS": "1"}
                    ), patch.object(
                        paged_topk_translator,
                        "translate_local_to_global_slots",
                        wraps=paged_topk_translator.translate_local_to_global_slots,
                    ) as fused:
                        got = AttentionV41FP8._decode_global_slots(
                            attn, selected, requests
                        )
                        self.assertEqual(fused.call_count, int(not padded))
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_selection_graph_updates_positions_and_owner_cache(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                attn, packed, x, qr, positions = _fixture(
                    "cuda", ratio, candidate=ratio == 1
                )
                with patch.object(
                    indexer, "score_decode_indexer", side_effect=_original_paged_score
                ):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            _select(attn, x, qr, positions)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        got = _select(attn, x, qr, positions)
                    for _ in range(2):
                        positions.add_(ratio)
                        x.normal_()
                        # Change actual grouped FP8 K bytes, leaving scales
                        # and the padded half untouched.
                        raw = packed.pool.view(packed.pool.shape[0], -1)
                        raw[:, : packed.logical_entries_per_block * 128].zero_()
                        graph.replay()
                        expected, candidates = _reference(
                            attn, packed, x, qr, positions
                        )
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                        if candidates is not None:
                            torch.testing.assert_close(
                                attn._shared_attention["candidates"],
                                candidates,
                                rtol=0,
                                atol=0,
                            )


if __name__ == "__main__":
    unittest.main()
