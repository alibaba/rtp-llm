"""Focused V4.1 routing, selection, and CUDA Graph integration contracts.

The paged scorer is replaced by the original FP32 score over the actual
packed FP4 cache. This isolates attention routing/candidate semantics from
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
from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import _fp4_rows_torch
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    AttentionV41FP8,
    rope_only,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import paged_topk_translator


def _decode_fp4_rows(payload, sf):
    """Decode packed e2m1 payload + packed-UE8M0 int32 scales to fp32."""
    rows = payload.shape[0]
    flat = payload.view(torch.uint8)
    low = flat & 15
    high = flat >> 4
    codes = torch.stack((low, high), dim=-1).reshape(rows, -1)
    magnitude = codes & 7
    normal = torch.exp2((magnitude >> 1).float() - 1.0) * (
        1.0 + (magnitude & 1).float() * 0.5
    )
    values = torch.where(magnitude < 2, magnitude.float() * 0.5, normal)
    values = torch.where(codes.ge(8), -values, values)
    exponent = sf.view(torch.uint8).view(rows, 4).float() - 127.0
    scale = torch.exp2(exponent)
    return (values.reshape(rows, 4, 32) * scale[:, :, None]).reshape(rows, 128)


def _keys_from_packed(pool, table, capacity, logical_entries):
    """Dequantize the planar FP4 INDEX_K pool with independent byte logic."""
    blocks, physical, _ = pool.shape
    raw = pool.view(blocks, -1)
    payload = (
        raw[:, : physical * 64]
        .view(blocks, physical, 64)
        .view(torch.int8)
        .reshape(-1, 64)
    )
    sf = (
        raw[:, physical * 64 :]
        .view(blocks, physical, 4)
        .view(torch.int32)
        .reshape(-1)
    )
    columns = torch.arange(capacity, device=pool.device)
    ids = table[:, columns // logical_entries].long()
    offsets = columns % logical_entries
    slots = (ids * physical + offsets).reshape(-1)
    return _decode_fp4_rows(payload[slots], sf[slots]).view(
        table.shape[0], capacity, 128
    )


def _original_paged_score(q, weights, freqs, pool, table, lengths, **kwargs):
    """Old arithmetic with the paged helper's causal -inf output contract."""
    capacity = kwargs["max_ctx_len"]
    b, s, h, d = q.shape
    keys = _keys_from_packed(pool, table, capacity, kwargs["logical_entries_per_block"])
    query = _fp4_rows_torch(rope_only(q.reshape(b * s, h, d).clone(), freqs, 64))[
        2
    ].view(b * s, h, d)
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
    pool = torch.zeros(blocks, physical, 68, dtype=torch.uint8, device=device)
    keys = torch.randn(blocks, physical, 128, device=device)
    if ratio == 2:
        # Padding is deliberately nonzero and vastly larger than real K.
        keys[:, logical:].fill_(6.0)
    payload, sf, _ = _fp4_rows_torch(keys.reshape(-1, 128))
    raw = pool.view(blocks, -1)
    raw[:, : physical * 64] = payload.view(blocks, physical, 64).reshape(blocks, -1)
    raw[:, physical * 64 :] = sf.view(torch.uint8).view(blocks, physical, 4).reshape(
        blocks, -1
    )
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


# Last paged logits returned by the patched scorer. The tensor is the graph
# pool buffer on CUDA, so after a replay it holds the replayed logits.
_LAST_PAGED_LOGITS = []


def _recorded_paged_score(*args, **kwargs):
    result = _original_paged_score(*args, **kwargs)
    _LAST_PAGED_LOGITS.append(result)
    return result


def _assert_selection_equivalent(case, got, expected, positions, ratio):
    """Compare selections against the torch-path reference.

    Rows within index_topk keep the bitwise ascending contract (dense
    shortcut). Rows beyond index_topk compare as valid top-k selections of
    the same logits: the radix-select TopK op's documented contract
    (test_topk_v3.py) leaves output order unspecified and tie members are
    kernel's choice, so indices compare through their gathered value
    multisets (falling back to set equality without recorded logits).
    """
    long_rows = ((positions + 1) // ratio) > 512
    torch.testing.assert_close(
        got[~long_rows], expected[~long_rows], rtol=0, atol=0
    )
    logits = _LAST_PAGED_LOGITS[-1] if _LAST_PAGED_LOGITS else None
    for row in long_rows.nonzero().flatten().tolist():
        g = got[row][got[row] >= 0]
        e = expected[row][expected[row] >= 0]
        case.assertEqual(g.numel(), e.numel())
        if logits is not None:
            torch.testing.assert_close(
                logits[row].gather(0, g.long()).sort().values,
                logits[row].gather(0, e.long()).sort().values,
                rtol=0,
                atol=0,
            )
        else:
            case.assertEqual(set(g.tolist()), set(e.tolist()))


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
    def setUp(self):
        _LAST_PAGED_LOGITS.clear()

    def test_paged_source_routing_per_token_lengths_and_short_context(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                attn, packed, x, qr, positions = _fixture("cpu", ratio)
                # A later index source must still use the KV owner's cache.
                attn.layer_id += 4
                attn._shared_attention["global"][attn.layer_id] = object()
                with patch.object(
                    indexer, "score_decode_indexer", side_effect=_recorded_paged_score
                ) as scorer:
                    got = _select(attn, x, qr, positions)
                expected, _ = _reference(attn, packed, x, qr, positions)
                _assert_selection_equivalent(
                    self, got, expected, positions, ratio
                )
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
            indexer, "score_decode_indexer", side_effect=_recorded_paged_score
        ):
            source = _select(attn, x, qr, positions)
            candidates = attn._shared_attention["candidates"].clone()
            expected_source, expected_candidates = _reference(
                attn, packed, x, qr, positions
            )
            _assert_selection_equivalent(
                self, source, expected_source, positions, 1
            )
            torch.testing.assert_close(candidates, expected_candidates, rtol=0, atol=0)
            newest = positions // 8
            self.assertTrue((candidates == newest[:, None]).any(-1).all())
            attn.layer_id = 24
            consumer = _select(attn, x, qr, positions)
            expected, _ = _reference(attn, packed, x, qr, positions, candidates)
            _assert_selection_equivalent(
                self, consumer, expected, positions, 1
            )
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
    def setUp(self):
        _LAST_PAGED_LOGITS.clear()

    def test_actual_slot_gate_and_padded_pool_fallback(self):
        for ratio in (1, 2):
            for padded in (False, True):
                with self.subTest(ratio=ratio, padded=padded):
                    logical = 128 // ratio
                    entries = logical * (2 if padded else 1)
                    region = CSA_KV if ratio == 2 else HCA_KV
                    pool = torch.empty(
                        4, entries, 288, dtype=torch.uint8, device="cuda"
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
                    indexer, "score_decode_indexer", side_effect=_recorded_paged_score
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
                        # Change actual packed FP4 K payload bytes, leaving
                        # scales and the padded half untouched.
                        raw = packed.pool.view(packed.pool.shape[0], -1)
                        raw[:, : packed.logical_entries_per_block * 64].zero_()
                        graph.replay()
                        expected, candidates = _reference(
                            attn, packed, x, qr, positions
                        )
                        _assert_selection_equivalent(
                    self, got, expected, positions, ratio
                )
                        if candidates is not None:
                            torch.testing.assert_close(
                                attn._shared_attention["candidates"],
                                candidates,
                                rtol=0,
                                atol=0,
                            )


def _pack_swa_fp8_pool(keys, entries):
    """Pack ``[N, 512]`` BF16 keys into a TMA-padded planar 584B SWA pool.

    Mirrors ``_swa_kv_insert_triton``: per block a data plane
    (``entries * 576`` bytes: 448 fp8 NoPE + 128 bf16 RoPE per token) then a
    scale plane (``entries * 8`` UE8M0 bytes). Returns the 4D FlashMLA view
    and the dequantized keys the kernel will effectively see.
    """
    blocks = (keys.shape[0] + entries - 1) // entries
    stride = ((entries * 584 + 575) // 576) * 576
    backing = torch.zeros(blocks, stride, dtype=torch.uint8, device=keys.device)
    pool = backing.as_strided((blocks, entries, 1, 584), (stride, 584, 584, 1))
    fp8_max = 448.0
    nope = keys[:, :448].float().view(-1, 7, 64)
    amax = nope.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.exp2(exponent)
    fp8 = torch.clamp(nope / scale[:, :, None], -fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    scale_bytes = torch.zeros(keys.shape[0], 8, dtype=torch.uint8, device=keys.device)
    scale_bytes[:, :7] = torch.clamp(exponent + 127, 0, 255).to(torch.uint8)
    dequant = torch.cat(
        (
            (fp8.float() * scale[:, :, None]).reshape(-1, 448),
            keys[:, 448:].float(),
        ),
        dim=-1,
    )
    for pos in range(keys.shape[0]):
        block, offset = divmod(pos, entries)
        backing[
            block, offset * 576 : offset * 576 + 448
        ] = fp8.reshape(-1, 448)[pos].view(torch.uint8)
        backing[
            block, offset * 576 + 448 : offset * 576 + 576
        ] = keys[pos, 448:].contiguous().view(torch.uint8)
        backing[
            block, entries * 576 + offset * 8 : entries * 576 + offset * 8 + 8
        ] = scale_bytes[pos]
    return pool.squeeze(2), dequant


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41Fp4DualDecodeAttentionCUDA(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100 CUDA required")
        try:
            import flash_mla  # noqa: F401
        except Exception as error:
            self.skipTest(f"flash_mla not importable: {error}")

    def test_merged_output_matches_dual_softmax_oracle(self):
        from flash_mla import get_mla_metadata

        from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_decode_attn import (
            fp4_dual_decode_attention,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
            dequantize_k_cache_slots_fp4,
            quantize_and_insert_k_cache_fp4,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        torch.manual_seed(2331)
        device = "cuda"
        heads, head_dim = 64, 512
        batch, span = 3, 1
        window, topk = 64, 96
        swa_entries, swa_blocks = 64, 3
        global_entries, global_blocks = 64, 3
        scale = head_dim**-0.5

        swa_keys = (
            torch.randn(swa_blocks * swa_entries, head_dim, device=device) * 0.3
        ).to(torch.bfloat16)
        swa_pool, swa_deq = _pack_swa_fp8_pool(swa_keys, swa_entries)
        global_pool = torch.zeros(
            global_blocks, global_entries, 288, dtype=torch.uint8, device=device
        )
        global_keys = (
            torch.randn(global_blocks * global_entries, head_dim, device=device) * 0.3
        ).to(torch.bfloat16)
        all_slots = torch.arange(
            global_blocks * global_entries, device=device, dtype=torch.int64
        )
        quantize_and_insert_k_cache_fp4(global_keys, global_pool, all_slots)
        global_deq = dequantize_k_cache_slots_fp4(
            global_pool, all_slots, out_dtype=torch.float32
        )

        q = (torch.randn(batch, span, heads, head_dim, device=device) * 0.3).to(
            torch.bfloat16
        )
        sink = (torch.randn(heads, device=device) * 0.5).float()
        swa_topk = torch.zeros(batch, span, window, dtype=torch.int32, device=device)
        global_topk = torch.zeros(batch, span, topk, dtype=torch.int32, device=device)
        # Row 0: both pools full; row 1: empty SWA; row 2: empty GLOBAL.
        swa_topk[0] = (
            torch.arange(window, device=device, dtype=torch.int32)
            % (swa_blocks * swa_entries)
        )
        swa_topk[1] = -1
        swa_topk[2, 0, :32] = torch.arange(32, device=device, dtype=torch.int32)
        swa_topk[2, 0, 32:] = -1
        global_topk[0] = (
            torch.arange(topk, device=device, dtype=torch.int32)
            % (global_blocks * global_entries)
        )
        global_topk[1, 0, :48] = (
            torch.arange(48, device=device, dtype=torch.int32) + global_entries
        )
        global_topk[1, 0, 48:] = -1
        global_topk[2] = -1

        sched_meta, _ = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=batch * span * heads,
            topk=window,
            num_heads_q=heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        op = SparseAttnV4DecodeFp8Op(
            n_heads=heads, head_dim=head_dim, softmax_scale=scale
        )
        out = fp4_dual_decode_attention(
            q=q,
            swa_pool_3d=swa_pool,
            global_pool_3d=global_pool,
            attn_sink=sink,
            swa_topk_3d=swa_topk,
            global_topk_3d=global_topk,
            swa_block_table=None,
            sched_meta=sched_meta,
            fp8_op=op,
        )
        torch.cuda.synchronize()

        swa_rows = [
            swa_topk[0][swa_topk[0] >= 0].long(),
            swa_topk[1][swa_topk[1] >= 0].long(),
            swa_topk[2][swa_topk[2] >= 0].long(),
        ]
        global_rows = [
            global_topk[0][global_topk[0] >= 0].long(),
            global_topk[1][global_topk[1] >= 0].long(),
            global_topk[2][global_topk[2] >= 0].long(),
        ]
        for b in range(batch):
            keys_s = swa_deq[swa_rows[b]]
            keys_g = global_deq[global_rows[b]]
            scores = torch.cat(
                (
                    q[b, 0].float() @ keys_s.T,
                    q[b, 0].float() @ keys_g.T,
                ),
                dim=-1,
            ) * scale
            reference = torch.maximum(scores.max(dim=-1).values, sink)
            weights = torch.exp(scores - reference[:, None])
            denominator = weights.sum(-1) + torch.exp(sink - reference)
            numerator = weights @ torch.cat((keys_s, keys_g), dim=0)
            expected = numerator / denominator[:, None]
            torch.testing.assert_close(
                out[b, 0].float(), expected, rtol=0.03, atol=0.01
            )


if __name__ == "__main__":
    unittest.main()
