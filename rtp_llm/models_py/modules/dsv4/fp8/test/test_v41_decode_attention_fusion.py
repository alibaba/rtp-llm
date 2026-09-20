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
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8, rope_only
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
    sf = raw[:, physical * 64 :].view(blocks, physical, 4).view(torch.int32).reshape(-1)
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
    raw[:, physical * 64 :] = (
        sf.view(torch.uint8).view(blocks, physical, 4).reshape(blocks, -1)
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
    torch.testing.assert_close(got[~long_rows], expected[~long_rows], rtol=0, atol=0)
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
                _assert_selection_equivalent(self, got, expected, positions, ratio)
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
            _assert_selection_equivalent(self, source, expected_source, positions, 1)
            torch.testing.assert_close(candidates, expected_candidates, rtol=0, atol=0)
            newest = positions // 8
            self.assertTrue((candidates == newest[:, None]).any(-1).all())
            attn.layer_id = 24
            consumer = _select(attn, x, qr, positions)
            expected, _ = _reference(attn, packed, x, qr, positions, candidates)
            _assert_selection_equivalent(self, consumer, expected, positions, 1)
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
                            # Candidate order is unused: consumers only read
                            # block membership. Keep the selected set exact.
                            torch.testing.assert_close(
                                attn._shared_attention["candidates"].sort(-1).values,
                                candidates.sort(-1).values,
                                rtol=0,
                                atol=0,
                            )


def _pack_swa_fp8_pool(keys, entries):
    """Independent PyTorch encoder for V4.1's all-FP8, group-32 SWA.

    A physical page contains ``entries * 512`` payload bytes, then
    ``entries * 16`` UE8M0 scales, then padding to a 512-byte page stride.
    The nominal tensor shape stays ``[blocks, entries, 528]``. Return the
    actual FP32 dequantized keys as the independent attention oracle input.
    """
    rows = keys.shape[0]
    blocks = (rows + entries - 1) // entries
    stride = ((entries * 528 + 511) // 512) * 512
    backing = torch.full((blocks, stride), 0x5A, dtype=torch.uint8, device=keys.device)
    pool = backing.as_strided((blocks, entries, 528), (stride, 528, 1))
    groups = keys.float().reshape(rows, 16, 32)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0))
    scale = torch.exp2(exponent)
    quantized = (
        (groups / scale[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    encoded = (exponent + 127).clamp(0, 255).to(torch.uint8)
    payload_rows = backing[:, : entries * 512].view(blocks, entries, 512)
    scale_rows = backing[:, entries * 512 : entries * 528].view(blocks, entries, 16)
    positions = torch.arange(rows, device=keys.device)
    payload_rows[positions // entries, positions % entries] = quantized.view(
        torch.uint8
    ).reshape(rows, 512)
    scale_rows[positions // entries, positions % entries] = encoded
    dequant = (quantized.float() * scale[:, :, None]).reshape(rows, 512)
    return pool, dequant


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41Fp4DualDecodeAttentionCUDA(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100 CUDA required")
        try:
            import flash_mla  # noqa: F401
        except ImportError as error:
            self.skipTest(f"flash_mla not importable: {error}")

    @staticmethod
    def _metadata(case):
        from flash_mla import get_mla_metadata

        batch, span, heads, _ = case.q.shape
        metadata, _ = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=batch * span * heads,
            topk=case.swa_topk.shape[-1],
            num_heads_q=heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        return metadata

    @staticmethod
    def _set_slots(case, phase):
        """Vary valid addresses and visibility without changing captured shapes."""
        batch, span = case.q.shape[:2]
        swa_count = case.swa_pool.shape[0] * case.swa_pool.shape[1]
        global_count = case.global_pool.shape[0] * case.global_pool.shape[1]
        swa_columns = torch.arange(case.swa_topk.shape[-1], device=case.q.device)
        global_columns = torch.arange(case.global_topk.shape[-1], device=case.q.device)
        for b in range(batch):
            for s in range(span):
                mode = (b * span + s + phase) % 4
                swa_slots = (swa_columns + 19 * b + 3 * s + 17 * phase) % swa_count
                global_slots = (
                    global_columns + 43 * b + 5 * s + 37 * phase
                ) % global_count
                # Preserve interior holes, rather than testing only valid prefixes.
                swa_slots = torch.where((swa_columns + phase) % 17 == 0, -1, swa_slots)
                global_slots = torch.where(
                    (global_columns + phase) % 29 == 0, -1, global_slots
                )
                if mode in (2, 3):
                    swa_slots.fill_(-1)
                elif mode == 1:
                    swa_slots[37:] = -1
                if mode in (1, 3):
                    global_slots.fill_(-1)
                elif mode == 2:
                    global_slots[73:] = -1
                case.swa_topk[b, s].copy_(swa_slots)
                case.global_topk[b, s].copy_(global_slots)

    @classmethod
    def _case(cls, batch, span, global_entries):
        from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
            quantize_and_insert_k_cache_fp4,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        torch.manual_seed(2331 + global_entries)
        heads, dimension, window, topk = 64, 512, 128, 512
        swa_entries, swa_blocks, global_blocks = 136, batch + 1, 12
        swa_keys = (
            torch.randn(swa_blocks * swa_entries, dimension, device="cuda") * 0.3
        ).to(torch.bfloat16)
        swa_pool, swa_dequant = _pack_swa_fp8_pool(swa_keys, swa_entries)
        global_pool = torch.zeros(
            global_blocks, global_entries, 288, dtype=torch.uint8, device="cuda"
        )
        global_keys = (
            torch.randn(global_blocks * global_entries, dimension, device="cuda") * 0.3
        ).to(torch.bfloat16)
        quantize_and_insert_k_cache_fp4(
            global_keys,
            global_pool,
            torch.arange(global_keys.shape[0], device="cuda", dtype=torch.int64),
        )
        case = SimpleNamespace(
            q=(torch.randn(batch, span, heads, dimension, device="cuda") * 0.3).to(
                torch.bfloat16
            ),
            sink=torch.linspace(-5, 5, heads, device="cuda"),
            swa_pool=swa_pool,
            swa_dequant=swa_dequant,
            global_pool=global_pool,
            swa_topk=torch.empty(batch, span, window, dtype=torch.int32, device="cuda"),
            global_topk=torch.empty(
                batch, span, topk, dtype=torch.int32, device="cuda"
            ),
            op=SparseAttnV4DecodeFp8Op(heads, dimension, dimension**-0.5),
        )
        cls._set_slots(case, 0)
        return case

    @staticmethod
    def _decode(case, metadata):
        from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_decode_attn import (
            fp4_dual_decode_attention,
        )

        return fp4_dual_decode_attention(
            q=case.q,
            swa_pool_3d=case.swa_pool,
            global_pool_3d=case.global_pool,
            attn_sink=case.sink,
            swa_topk_3d=case.swa_topk,
            global_topk_3d=case.global_topk,
            swa_block_table=None,
            sched_meta=metadata,
            fp8_op=case.op,
        )

    @staticmethod
    def _oracle(case):
        # Decode the actual GLOBAL bytes independently of the production codec.
        # Run this only after native decode/replay so its casts/matmuls cannot
        # initialize or hide failures in the first native invocation.
        blocks, entries, _ = case.global_pool.shape
        raw = case.global_pool.view(blocks, -1)
        packed = raw[:, : entries * 256].reshape(-1, 256)
        scale = raw[:, entries * 256 : entries * 288].contiguous()
        scale = scale.view(torch.float8_e4m3fn).float().reshape(-1, 32)
        codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 512)
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=case.q.device
        )
        values = levels[(codes & 7).long()]
        values = torch.where(codes >= 8, -values, values)
        global_dequant = values * scale.repeat_interleave(16, -1)
        expected = torch.empty_like(case.q, dtype=torch.float32)
        for b in range(case.q.shape[0]):
            for s in range(case.q.shape[1]):
                swa_ids = case.swa_topk[b, s]
                global_ids = case.global_topk[b, s]
                keys = torch.cat(
                    (
                        case.swa_dequant[swa_ids[swa_ids >= 0].long()],
                        global_dequant[global_ids[global_ids >= 0].long()],
                    ),
                    0,
                )
                if keys.shape[0] == 0:
                    # A finite learned sink gives a finite denominator and zero V.
                    expected[b, s].zero_()
                    continue
                scores = (case.q[b, s].float() @ keys.T) * (512**-0.5)
                maximum = torch.maximum(scores.amax(-1), case.sink)
                weights = torch.exp(scores - maximum[:, None])
                denominator = weights.sum(-1) + torch.exp(case.sink - maximum)
                expected[b, s] = (weights @ keys) / denominator[:, None]
        return expected

    def _assert_oracle(self, case, out):
        expected = self._oracle(case)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, case.q.shape)
        self.assertTrue(bool(torch.isfinite(out).all()))
        torch.testing.assert_close(out.float(), expected, rtol=0.03, atol=0.01)
        empty = (case.swa_topk < 0).all(-1) & (case.global_topk < 0).all(-1)
        torch.testing.assert_close(
            out[empty], torch.zeros_like(out[empty]), rtol=0, atol=0
        )

    @torch.inference_mode()
    def test_merged_output_matches_dual_softmax_oracle(self):
        for entries in (64, 128):
            with self.subTest(global_entries=entries):
                case = self._case(4, 1, entries)
                self.assertEqual(case.swa_pool.stride(0) % 512, 0)
                self.assertGreater(case.swa_pool.stride(0), 136 * 528)
                output = self._decode(case, self._metadata(case))
                torch.cuda.synchronize()
                self._assert_oracle(case, output)

    @torch.inference_mode()
    def test_cuda_graph_replay_changes_queries_slots_and_sink(self):
        for entries in (64, 128):
            with self.subTest(global_entries=entries):
                case = self._case(4, 6, entries)
                warmup_meta = self._metadata(case)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._decode(case, warmup_meta)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                # Fresh metadata ensures any scheduler initialization belongs
                # to capture, matching the production metadata owner's contract.
                capture_meta = self._metadata(case)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._decode(case, capture_meta)
                graph.replay()
                torch.cuda.synchronize()
                self._assert_oracle(case, output)
                previous = output.clone()
                for phase in (1, 2):
                    case.q.normal_(0, 0.3)
                    case.sink.add_(0.75)
                    self._set_slots(case, phase)
                    graph.replay()
                    torch.cuda.synchronize()
                    self._assert_oracle(case, output)
                    self.assertFalse(torch.equal(output, previous))
                    previous.copy_(output)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41DecodeTopKFusionCUDA(unittest.TestCase):
    """Independent old-PyTorch oracles for the decode selection helpers.

    Candidate membership must be exact, including cutoff ties. Token TopK has
    an unordered contract: compare selected score multisets and uniqueness,
    while keeping the old short-context ascending indices exactly.
    """

    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_topk

        self.impl = _v41_decode_topk
        enabled = patch.dict(
            os.environ, {"DSV41_FUSED_DECODE_TOPK": "1", "DSV4_TOPK_V3": "1"}
        )
        enabled.start()
        self.addCleanup(enabled.stop)

    @staticmethod
    def _old_candidates(logits, lengths, block_size=8, topk_blocks=2048):
        """The old per-request S=6 pool/pin/sorted-TopK/invalid chain."""
        rows, width = logits.shape
        nblocks = (width + block_size - 1) // block_size
        parts = []
        for start in range(0, rows, 6):
            values = (
                F.pad(
                    logits[start : start + 6],
                    (0, nblocks * block_size - width),
                    value=-torch.inf,
                )
                .view(-1, nblocks, block_size)
                .amax(-1)
            )
            visible = lengths[start : start + 6]
            newest = (visible.long() - 1).clamp_min(0) // block_size
            values.scatter_(
                1,
                newest[:, None],
                torch.where(visible > 0, torch.inf, -torch.inf)[:, None],
            )
            scores, ids = values.topk(min(topk_blocks, nblocks), sorted=True)
            parts.append(torch.where(scores > -torch.inf, ids, -1).int())
        candidates = torch.cat(parts)
        # Extra sentinel column means -1 candidates cannot alias real block 0.
        flags = torch.zeros(
            (rows, nblocks + 1), dtype=torch.uint8, device=logits.device
        )
        flags.scatter_(1, torch.where(candidates >= 0, candidates, nblocks).long(), 1)
        return candidates, flags[:, :nblocks]

    def _assert_candidates(self, logits, lengths, candidates, flags, block_size=8):
        expected, expected_flags = self._old_candidates(logits, lengths, block_size)
        self.assertEqual(candidates.dtype, torch.int32)
        self.assertEqual(flags.dtype, torch.uint8)
        torch.testing.assert_close(
            candidates.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(flags, expected_flags, rtol=0, atol=0)
        for row, visible in enumerate(lengths.cpu().tolist()):
            ids = candidates[row][candidates[row] >= 0]
            self.assertEqual(ids.numel(), ids.unique().numel())
            if visible:
                self.assertTrue(bool((ids == (visible - 1) // block_size).any()))
            else:
                self.assertEqual(ids.numel(), 0)
        return expected_flags

    def _assert_tokens(self, logits, lengths, result, topk=512):
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(tuple(result.shape), (logits.shape[0], topk))
        for row, visible in enumerate(lengths.cpu().tolist()):
            selected = result[row]
            if visible <= topk:
                expected = torch.arange(topk, device=logits.device, dtype=torch.int32)
                expected.masked_fill_(expected >= visible, -1)
                torch.testing.assert_close(selected, expected, rtol=0, atol=0)
                continue
            values = logits[row, :visible].topk(topk, sorted=True).values
            expected_values = values[values.isfinite()].sort().values
            ids = selected[selected >= 0].long()
            self.assertEqual(ids.numel(), ids.unique().numel())
            self.assertTrue(bool((ids < visible).all()))
            self.assertTrue(bool(((selected == -1) | (selected >= 0)).all()))
            torch.testing.assert_close(
                logits[row, ids].sort().values, expected_values, rtol=0, atol=0
            )

    def test_candidate_real_capacity_batch24_preserves_ties_and_flags(self):
        torch.manual_seed(4102048)
        rows, width = 24, 524288
        logits = torch.rand(rows, width, dtype=torch.float32, device="cuda")
        lengths = torch.tensor(
            [
                0,
                1,
                7,
                8,
                9,
                511,
                512,
                513,
                16383,
                16384,
                16385,
                32767,
                32768,
                32769,
                65535,
                131071,
                262143,
                524281,
                524287,
                524288,
                17001,
                24000,
                33003,
                510003,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        # All-zero cutoff ties span more blocks than K. The latest block must
        # still be included, and batching 4x6 -> 24 must keep the same set.
        logits[12].zero_()
        logits[16].fill_(1.0)
        logits[16, : 1023 * 8] = 3.0
        logits[16, 1023 * 8 : 8192 * 8] = 2.0
        logits[18].fill_(-torch.inf)
        logits[19, 7 * 8] = torch.inf
        logits[19, 11 * 8] = torch.nan
        logits[19, 23 * 8 : 24 * 8] = -torch.inf
        # NaN in the latest partial block is overwritten by the +inf pin.
        logits[23, 510002] = torch.nan
        columns = torch.arange(width, device="cuda")
        logits.masked_fill_(columns[None] >= lengths[:, None], -torch.inf)
        self.assertTrue(self.impl.is_supported(logits, lengths, 512))
        candidates, flags = self.impl.select_candidates(logits, lengths, 8, 2048)
        expected_flags = self._assert_candidates(logits, lengths, candidates, flags)
        actual = logits.clone()
        self.impl.mask_candidates(actual, flags, 8)
        expected = logits.masked_fill(
            ~expected_flags[:, columns // 8].bool(), -torch.inf
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(int(flags[18].sum()), 1)
        self.assertEqual(int(flags[19, 7]), 1)
        self.assertEqual(int(flags[19, 11]), 0)

    def test_candidate_padded_row_stride_and_partial_blocks(self):
        for width in (4099, 16391):
            with self.subTest(width=width):
                storage = torch.full((6, width + 13), 12345.0, device="cuda")
                logits = storage[:, :width]
                columns = torch.arange(width, device="cuda")
                logits.copy_((columns % 31).float()[None])
                lengths = torch.tensor(
                    [0, 1, 7, 8, width - 1, width], dtype=torch.int32, device="cuda"
                )
                logits.masked_fill_(columns[None] >= lengths[:, None], -torch.inf)
                self.assertFalse(logits.is_contiguous())
                self.assertTrue(self.impl.is_supported(logits, lengths, 512))
                candidates, flags = self.impl.select_candidates(
                    logits, lengths, 8, 2048
                )
                expected_flags = self._assert_candidates(
                    logits, lengths, candidates, flags
                )
                expected = logits.masked_fill(
                    ~expected_flags[:, columns // 8].bool(), -torch.inf
                )
                self.impl.mask_candidates(logits, flags, 8)
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                self.assertTrue(bool((storage[:, width:] == 12345.0).all()))

    def test_token_finite_filter_short_order_and_padded_row_stride(self):
        rows, width, topk = 24, 65536, 512
        storage = torch.full((rows, width + 17), 12345.0, device="cuda")
        logits = storage[:, :width]
        columns = torch.arange(width, device="cuda")
        logits.copy_(columns.float()[None])
        lengths = torch.tensor(
            [0, 1, 511, 512, 513, 600, 4096, 10001, 16383, 32769, 65535, 65536] * 2,
            device="cuda",
            dtype=torch.int32,
        )
        # Short rows retain all causal indices even for nonfinite score data.
        logits[:4].fill_(torch.nan)
        logits[4].fill_(-torch.inf)
        logits[4, :500] = torch.arange(500, device="cuda").float()
        logits[5].fill_(-torch.inf)
        logits[5, 80:90] = torch.arange(10, device="cuda").float()
        logits[5, 0:3] = torch.nan
        logits[5, 10] = torch.inf
        logits[6].fill_(torch.nan)
        logits[7].zero_()
        logits.masked_fill_(columns[None] >= lengths[:, None], -torch.inf)
        self.assertTrue(self.impl.is_supported(logits, lengths, topk))
        result = self.impl.select_tokens(logits, lengths, topk)
        self._assert_tokens(logits, lengths, result, topk)
        self.assertEqual(int((result[4] >= 0).sum()), 500)
        self.assertEqual(int((result[5] >= 0).sum()), 10)
        self.assertTrue(bool((result[6] == -1).all()))
        self.assertTrue(bool((storage[:, width:] == 12345.0).all()))

    def test_graph_replay_updates_candidates_masks_tokens_and_device_lengths(self):
        rows, width = 24, 32777
        columns = torch.arange(width, device="cuda")
        static_logits = torch.empty(rows, width, device="cuda")
        static_lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
        static_logits.copy_(columns.float()[None])

        def run():
            ids, flags = self.impl.select_candidates(
                static_logits, static_lengths, 8, 2048
            )
            masked = static_logits.clone()
            self.impl.mask_candidates(masked, flags, 8)
            return (
                ids,
                flags,
                masked,
                self.impl.select_tokens(masked, static_lengths, 512),
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for phase in range(3):
            lengths = torch.tensor(
                (
                    [0, 1, 7, 512, 513, 16385] * 4
                    if phase == 0
                    else (
                        [width, width - 1, 17001, 20003, 8, 0] * 4
                        if phase == 1
                        else [0] * rows
                    )
                ),
                dtype=torch.int32,
                device="cuda",
            )
            data = (
                ((columns + phase * 97) % 4093).float()[None].expand(rows, -1).clone()
            )
            if phase == 1:
                data[0, 11 * 8] = torch.nan
                data[1, 23 * 8] = torch.inf
            data.masked_fill_(columns[None] >= lengths[:, None], -torch.inf)
            static_logits.copy_(data)
            static_lengths.copy_(lengths)
            graph.replay()
            torch.cuda.synchronize()
            ids, flags, masked, tokens = captured
            expected_flags = self._assert_candidates(data, lengths, ids, flags)
            expected = data.masked_fill(
                ~expected_flags[:, columns // 8].bool(), -torch.inf
            )
            torch.testing.assert_close(masked, expected, rtol=0, atol=0, equal_nan=True)
            self._assert_tokens(expected, lengths, tokens)

    def test_support_gate_rejects_disabled_dtype_layout_and_short_capacity(self):
        logits = torch.empty(2, 1024, device="cuda")
        lengths = torch.tensor([8, 900], dtype=torch.int32, device="cuda")
        self.assertTrue(self.impl.is_supported(logits, lengths, 512))
        with patch.dict(os.environ, {"DSV41_FUSED_DECODE_TOPK": "0"}):
            self.assertFalse(self.impl.is_supported(logits, lengths, 512))
        with patch.dict(os.environ, {"DSV4_TOPK_V3": "0"}):
            self.assertFalse(self.impl.is_supported(logits, lengths, 512))
        for values, lens, count in (
            (logits.to(torch.bfloat16), lengths, 512),
            (logits[:, ::2], lengths, 512),
            (logits[:1].expand(2, -1), lengths, 512),
            (logits[:, :511], lengths, 512),
            (logits, lengths.long(), 512),
            (logits, lengths.cpu(), 512),
            (logits, lengths, 513),
            (logits[:0], lengths[:0], 512),
        ):
            self.assertFalse(self.impl.is_supported(values, lens, count))


if __name__ == "__main__":
    unittest.main()
