"""N5 two-phase boundary for the single-launch Q/K RoPE candidate."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import (
    GRAPH_BATCHED_WO_A_ENV,
    _try_batched_wo_a,
)
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _compress_pairs, _rope_tail
from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import is_supported, qk_rope
from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import (
    try_compress_csa,
    try_rope_,
    try_rotate_q_and_write_swa,
    try_write_csa_state,
    try_write_packed_kv,
)


# Exact scalar operands from immutable N5/L0-cpu-edges-b05.json. Each edge is
# mapped to a distinct complex pair so all16 also fit a single Decode row.
_ROPE_EDGES = (
    (19, 46, 1.140625, -2.421875, -0.9046522974967957, 0.42615044116973877, 0, 0.00021457672119140625),
    (12, 5, 2.234375, -0.62890625, -0.2709486782550812, 0.9625937938690186, 0, -0.000019669532775878906),
    (9, 5, -3.1875, 2.65625, 0.6216099262237549, 0.7833269238471985, 1, -0.84765625),
    (10, 49, 0.71875, -0.9140625, 0.7901318669319153, -0.6129369139671326, 0, 0.007659912109375),
    (22, 28, -0.9140625, -0.1357421875, -0.9891483187675476, 0.14692047238349915, 1, -0.000025391578674316406),
    (29, 4, 3.15625, -1.28125, 0.9261506199836731, 0.37715381383895874, 1, 0.0037689208984375),
    (12, 62, -3.984375, 3.1875, 0.7808259725570679, 0.6247486472129822, 1, -0.0003490447998046875),
    (6, 62, 1.65625, -2.171875, 0.8054140210151672, 0.5927127003669739, 1, -0.76953125),
    (22, 7, 3.90625, -1.6953125, 0.9999995231628418, 0.000999772921204567, 1, -1.6953125),
    (26, 59, -2.046875, 2.1875, 0.6806167364120483, 0.7326396703720093, 1, -0.01080322265625),
    (26, 42, 0.466796875, -0.318359375, 0.5648878216743469, -0.825167715549469, 0, 0.00098419189453125),
    (3, 26, -0.8125, 0.00372314453125, 0.9999895691871643, 0.004565374460071325, 1, 0.000013768672943115234),
    (17, 27, -0.48046875, 0.55859375, 0.9914709329605103, 0.13032807409763336, 1, 0.490234375),
    (17, 22, -1.109375, 0.1328125, 0.9918851852416992, 0.12713700532913208, 1, -0.00933837890625),
    (25, 50, 5.96875, -4.6875, 0.8293553590774536, 0.5587214827537537, 1, -0.55078125),
    (2, 19, 3.3125, -0.92578125, 0.953084409236908, 0.3027046024799347, 1, 0.12060546875),
)


def _assert_rope_edges(test, tokens):
    q = torch.full((tokens, 64, 512), 0.5, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((tokens, 512), 0.25, device="cuda", dtype=torch.bfloat16)
    parts = torch.zeros(tokens, 32, 2, device="cuda", dtype=torch.float32)
    parts[..., 0] = 1
    targets = []
    for pair, (position, head, real, imag, cosine, sine, component, expected) in enumerate(_ROPE_EDGES):
        row = position if tokens == 32 else 0
        dim = 448 + 2 * pair
        values = torch.tensor([real, imag], device="cuda", dtype=torch.bfloat16)
        q[row, head, dim:dim + 2] = values
        kv[row, dim:dim + 2] = values
        parts[row, pair] = torch.tensor([cosine, sine], device="cuda", dtype=torch.float32)
        targets.append((row, head, dim + component, expected))
    freqs = torch.view_as_complex(parts).contiguous()
    saved = q.clone(), kv.clone(), freqs.clone()
    actual_q, actual_kv = qk_rope(q, kv, freqs)
    ref_q, ref_kv = _rope_tail(q, freqs), _rope_tail(kv, freqs)
    torch.cuda.synchronize()
    test.assertTrue(torch.equal(actual_q, ref_q))
    test.assertTrue(torch.equal(actual_kv, ref_kv))
    for row, head, dim, expected in targets:
        test.assertEqual(float(ref_q[row, head, dim]), expected)
        test.assertEqual(float(ref_kv[row, dim]), expected)
    for actual, before in zip((q, kv, freqs), saved):
        test.assertTrue(torch.equal(actual, before))
    test.assertEqual(len({t.data_ptr() for t in (q, kv, actual_q, actual_kv)}), 4)
    for tag in ("swa_kv", "global_kv_2"):
        test.assertTrue(torch.equal(encode_kv(actual_kv, pool_tag=tag), encode_kv(ref_kv, pool_tag=tag)))


class SharedFusedOpsDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_and_decode(self):
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(5105)
        for tokens in (32, 1):
            q = torch.randn(tokens, 10, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
            phase = torch.randn(tokens, 32, device="cuda")
            freqs = torch.polar(torch.ones_like(phase), phase).contiguous()
            saved = (q.clone(), kv.clone(), freqs.clone())
            self.assertTrue(is_supported(q, kv, freqs))
            actual_q, actual_kv = qk_rope(q, kv, freqs)
            torch.cuda.synchronize()
            ref_q, ref_kv = _rope_tail(q, freqs), _rope_tail(kv, freqs)
            # Accepted PPU FP32 complex contraction and one BF16 landing:
            # exact equality is the contract, not a fitted tolerance.
            self.assertTrue(torch.equal(actual_q, ref_q))
            self.assertTrue(torch.equal(actual_kv, ref_kv))
            for actual, expected in zip((q, kv, freqs), saved):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len({t.data_ptr() for t in (q, kv, actual_q, actual_kv)}), 4)
            for tag in ("swa_kv", "global_kv_2"):
                self.assertTrue(torch.equal(
                    encode_kv(actual_kv, pool_tag=tag), encode_kv(ref_kv, pool_tag=tag)
                ))
            self.assertFalse(is_supported(q[..., ::2], kv, freqs))
            with self.assertRaises(ValueError):
                qk_rope(q[..., ::2], kv, freqs)
        for tokens in (32, 1):
            _assert_rope_edges(self, tokens)


class GraphSwaWriteDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_replay_matches_bf16_rope_and_padded_native_pool_bytes(self):
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(8193)
        entries, blocks, entry_bytes = 136, 4, 528
        for rows in (1, 8):
            with self.subTest(rows=rows):
                q_input = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
                q = torch.empty_like(q_input)
                kv = torch.randn(rows, 512, device=q.device, dtype=q.dtype)
                angle = torch.rand(rows, 32, device=q.device)
                freqs = torch.polar(torch.ones_like(angle), angle)
                slots = torch.full((rows,), -1, device=q.device, dtype=torch.int64)
                valid_mask = torch.ones(rows, device=q.device, dtype=torch.bool)
                pool = torch.full(
                    (blocks, entries * entry_bytes + 64),
                    0xA5,
                    device=q.device,
                    dtype=torch.uint8,
                )
                cache = SimpleNamespace(
                    pool=lambda owner, tag: pool,
                    entries={"swa_kv": entries},
                    _table_tag=lambda owner, tag: tag,
                )
                inputs = (
                    torch.randn(rows, 64, 512, device=q.device, dtype=q.dtype),
                    torch.randn(rows, 512, device=q.device, dtype=q.dtype),
                    torch.randn(rows, 10, 128, device=q.device, dtype=q.dtype),
                    torch.randn(rows, 128, device=q.device, dtype=q.dtype),
                )
                rotated = tuple(torch.empty_like(value) for value in inputs)

                def step():
                    q.copy_(q_input)
                    self.assertIs(
                        try_rotate_q_and_write_swa(q, kv, freqs, cache, 0, slots), q
                    )
                    for index, (dst, src) in enumerate(zip(rotated, inputs)):
                        dst.copy_(src)
                        self.assertIs(
                            try_rope_(
                                dst, freqs, inverse=index == 0,
                                valid_mask=valid_mask if index == 0 else None,
                            ),
                            dst,
                        )

                # Compile both the slot writer and the layout variants before capture.
                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    step()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step()

                for replay in (0, 1):
                    q_input.normal_()
                    kv.normal_()
                    for value in inputs:
                        value.normal_()
                    valid_mask.copy_(torch.arange(rows, device=q.device) % 2 == replay)
                    inputs[0][~valid_mask] = float("nan")
                    angle.uniform_(-3.14, 3.14)
                    freqs.copy_(torch.polar(torch.ones_like(angle), angle))
                    kv[0, :32].zero_()
                    kv[0, 32:64].fill_(448.0)
                    kv[0, 64:96].fill_(-896.0)
                    kv[0, 96:128].fill_(torch.finfo(kv.dtype).max)
                    if replay:
                        kv[0, 128] = float("nan")
                        kv[0, 160] = float("inf")
                    writes = [
                        entries + replay,
                        2 * entries + 135 - replay,
                        -1,
                        0,
                        blocks * entries,
                        3 * entries + replay,
                        -9,
                        1,
                    ][:rows]
                    slots.copy_(torch.tensor(writes, device=q.device, dtype=slots.dtype))
                    pool.fill_(0xA5)
                    expected = torch.full_like(pool, 0xA5)
                    expected_q = _rope_tail(q_input, freqs)
                    packed = encode_kv(_rope_tail(kv, freqs), pool_tag="swa_kv")
                    for index, slot in enumerate(writes):
                        block, offset = divmod(slot, entries)
                        if 0 < block < blocks:
                            start = offset * entry_bytes
                            expected[block, start : start + entry_bytes] = packed[index]
                    masked = torch.where(
                        valid_mask[:, None, None], inputs[0], torch.zeros_like(inputs[0])
                    )
                    expected_rotated = [_rope_tail(masked, freqs, inverse=True)]
                    expected_rotated.extend(_rope_tail(src, freqs) for src in inputs[1:])
                    self.assertIs(try_rope_(masked, freqs, inverse=True), masked)
                    saved_kv = kv.view(torch.int16).clone()
                    saved_freqs = freqs.clone()
                    graph.replay()
                    self.assertTrue(torch.equal(q, expected_q))
                    self.assertTrue(torch.equal(kv.view(torch.int16), saved_kv))
                    self.assertTrue(torch.equal(freqs, saved_freqs))
                    for actual, reference in zip(rotated, expected_rotated):
                        self.assertTrue(torch.equal(actual, reference))
                    torch.testing.assert_close(
                        rotated[0].view(torch.int16), masked.view(torch.int16), rtol=0, atol=0
                    )
                    if not torch.equal(pool, expected):
                        block, byte = torch.nonzero(pool != expected)[0].tolist()
                        self.fail(
                            f"Replay {replay} block {block} byte {byte}: "
                            f"got {int(pool[block, byte])}, "
                            f"expected {int(expected[block, byte])}"
                        )

    @torch.inference_mode()
    def test_replay_matches_shared_swa_write_slots(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import GraphPackedAttention

        device = torch.device("cuda", torch.cuda.current_device())
        entries, span, blocks = 130, 256, 5
        domain = "decoder_swa_kv"
        table = torch.tensor([2, 0, 4, blocks, -1, 3], device=device, dtype=torch.int32)
        table = table.repeat(8, 1)
        pool = torch.empty((blocks, 1), device=device)
        cache = NativeGraphCache.__new__(NativeGraphCache)
        cache.device = device
        cache._framework_tags = {(owner, "swa_kv"): domain for owner in range(2)}
        cache._pools = {(owner, "swa_kv"): pool for owner in range(2)}
        cache.entries, cache.spans, cache.tables = (
            {domain: entries}, {domain: span}, {domain: table}
        )
        packed = GraphPackedAttention.__new__(GraphPackedAttention)
        packed.cache = cache
        packed.offsets = torch.arange(128, device=device)
        packed.swa_domains = {owner: (domain, blocks) for owner in range(2)}
        positions = torch.tensor([0, 129, 130, 256, 517, 774, 1200, -1], device=device)
        valid = torch.ones_like(positions, dtype=torch.bool)

        def step():
            shared = {}
            writes = packed.swa_write_slots(0, positions, valid, shared)
            self.assertIs(packed.swa_write_slots(1, positions, valid, shared), writes)
            return writes, packed._swa_indices(1, positions, valid, shared)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            writes, (reads, lengths) = step()

        for replay in (0, 1):
            if replay:
                positions.add_(span)
                table.copy_(table.flip(-1))
                valid[1::2] = False
            graph.replay()
            end = positions + 1
            reference_lengths = torch.where(valid, end.clamp_max(128), 0)
            raw = (end - 128).clamp_min(0)[:, None] + packed.offsets[None]
            raw = torch.where(packed.offsets[None] < reference_lengths[:, None], raw, -1)
            expected_reads = cache.slots(1, "swa_kv", raw).int()[:, None]
            expected_writes = cache.slots(1, "swa_kv", torch.where(valid, positions, -1))
            self.assertTrue(torch.equal(reads, expected_reads))
            self.assertTrue(torch.equal(lengths, reference_lengths.int()))
            self.assertTrue(torch.equal(writes, expected_writes))

    @torch.inference_mode()
    def test_replay_matches_csa_pooling_rounding_and_ring_state(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache

        torch.manual_seed(8195)
        device = torch.device("cuda", torch.cuda.current_device())
        for rows in (1, 8, 16):
            with self.subTest(rows=rows):
                cache = NativeGraphCache.__new__(NativeGraphCache)
                cache.device = device
                cache.geometry = SimpleNamespace(ratio=lambda owner: 2)
                cache.entries = {"csa_state": 4, "global_kv_2": 4, "indexer_kv": 8}
                cache.spans = dict.fromkeys(cache.entries, 8)
                cache._framework_tags = {(0, tag): tag for tag in cache.entries}
                row_blocks = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
                cache.tables = {
                    tag: row_blocks[:, None].repeat(1, 4) for tag in cache.entries
                }
                pool = torch.empty(rows + 1, 4 * 1024 + 32, device=device)
                cache._pools = {
                    (0, "csa_state"): pool,
                    (0, "global_kv_2"): torch.empty(rows + 1, 1, device=device),
                    (0, "indexer_kv"): torch.empty(rows + 1, 1, device=device),
                }
                # Row 1 reads the prior generation from the slot row 0 wraps
                # into, while both rows still write to distinct current slots.
                if rows > 1:
                    cache.tables["csa_state"][1, 0] = 1
                    cache.tables["csa_state"][4, 1] = 0
                    cache.tables["global_kv_2"][5, 2] = -1
                values = torch.empty(rows, 528, device=device)[:, :512]
                scores = torch.empty(rows, 544, device=device)[:, :512]
                gamma = (torch.rand(512, device=device) + 0.5).bfloat16()
                phase = torch.rand(24, 32, device=device)
                freqs = torch.polar(torch.ones_like(phase), phase)
                positions = torch.arange(rows, dtype=torch.long, device=device) % 2
                valid = torch.ones(rows, dtype=torch.bool, device=device)
                eps = 1e-6

                def step():
                    shared = {}
                    pooled = try_compress_csa(
                        values, scores, None, eps, freqs, positions, valid, cache, 0, shared
                    )
                    actual = try_compress_csa(
                        values, scores, gamma, eps, freqs, positions, valid, cache, 0, shared
                    )
                    self.assertIsNotNone(pooled)
                    self.assertIsNotNone(actual)
                    self.assertIs(pooled[1], actual[1])
                    self.assertTrue(try_write_csa_state(values, scores, cache, 0, actual[1]))
                    return pooled[0], actual

                pool.zero_()
                values.zero_()
                scores.zero_()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    pooled, (normalized, current, main, index, rope) = step()

                for replay, pos in enumerate((
                    [4, 1, 7, 8, 9, 17, 33, 100],
                    [5, 3, 4, 7, 1, 9, -1, 20],
                    [33, 4, 0, 9, 31, 15, 4, 17],
                )):
                    positions.copy_(torch.tensor(pos * 2, device=device)[:rows])
                    valid.fill_(True)
                    if replay == 2:
                        valid[::3] = False
                    values.normal_()
                    scores.normal_()
                    pool.normal_()
                    pool[0].fill_(float("nan"))
                    expected_pool = pool.clone()
                    previous_position = positions - 1
                    commit = valid & ((positions & 1) == 1)
                    raw = torch.where(commit, previous_position, -1)
                    current_reference = cache.slots(0, "csa_state", torch.where(valid, positions, -1))
                    previous = cache.read_rows(0, "csa_state", cache.slots(0, "csa_state", raw))
                    paired_values = torch.stack((previous[:, :512], values), dim=1)
                    paired_scores = torch.stack((previous[:, 512:], scores), dim=1)
                    pooled_reference = (paired_values * paired_scores.softmax(1)).sum(1).bfloat16()
                    norm_reference = _compress_pairs(paired_values, paired_scores, gamma, eps)
                    for row, slot in enumerate(current_reference.tolist()):
                        if slot >= 0:
                            block, entry = divmod(slot, cache.entries["csa_state"])
                            start = entry * 1024
                            expected_pool[block, start : start + 512] = values[row]
                            expected_pool[block, start + 512 : start + 1024] = scores[row]
                    graph.replay()
                    torch.testing.assert_close(
                        pooled.view(torch.int16), pooled_reference.view(torch.int16), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        normalized.view(torch.int16), norm_reference.view(torch.int16), rtol=0, atol=0
                    )
                    self.assertTrue(torch.equal(current, current_reference))
                    self.assertTrue(torch.equal(main, cache.slots(0, "global_kv_2", raw)))
                    self.assertTrue(torch.equal(index, cache.slots(0, "indexer_kv", raw)))
                    self.assertTrue(torch.equal(rope, freqs[previous_position.clamp(0, 23)]))
                    self.assertTrue(torch.equal(pool.view(torch.int32), expected_pool.view(torch.int32)))

    @torch.inference_mode()
    def test_replay_matches_fp4_and_csa_reference_bytes(self):
        torch.manual_seed(8194)
        entries, blocks = 136, 4
        formats = {
            "global_kv_1": (512, 288),
            "global_kv_2": (512, 288),
            "indexer_kv": (128, 68),
        }
        for rows, csa_dtype in (
            (1, torch.bfloat16),
            (8, torch.bfloat16),
            (1, torch.float32),
            (8, torch.float32),
        ):
            with self.subTest(rows=rows, csa_dtype=csa_dtype):
                slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
                keys = {
                    tag: torch.empty(rows, dim + 16, device="cuda", dtype=torch.bfloat16)[
                        :, :dim
                    ]
                    for tag, (dim, _) in formats.items()
                }
                values = torch.empty(rows, 528, device="cuda", dtype=csa_dtype)[:, :512]
                scores = torch.empty(rows, 544, device="cuda", dtype=csa_dtype)[:, :512]
                backing = {
                    tag: torch.full(
                        (blocks, entries * width + 64),
                        0xA5,
                        device="cuda",
                        dtype=torch.uint8,
                    )
                    for tag, (_, width) in formats.items()
                }
                backing["csa_state"] = torch.full(
                    (blocks, entries * 1024 + 64), -3.0, device="cuda", dtype=torch.float32
                )
                pools = {tag: pool[:, :-32] for tag, pool in backing.items()}
                cache = SimpleNamespace(
                    pool=lambda owner, tag: pools[tag],
                    entries=dict.fromkeys(pools, entries),
                    _table_tag=lambda owner, tag: tag,
                )

                def step():
                    for tag, key in keys.items():
                        self.assertTrue(try_write_packed_kv(key, cache, 0, tag, slots))
                    self.assertTrue(try_write_csa_state(values, scores, cache, 0, slots))

                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    step()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step()

                ties = torch.tensor(
                    [-6, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25,
                     -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for replay in (0, 1):
                    writes = [
                        entries + replay,
                        2 * entries + 135 - replay,
                        -1,
                        0,
                        blocks * entries,
                        3 * entries + replay,
                        -9,
                        1,
                    ][:rows]
                    slots.copy_(torch.tensor(writes, device="cuda", dtype=slots.dtype))
                    references = {}
                    for tag, key in keys.items():
                        key.normal_()
                        key[0, :16] = ties
                        key[0, 16:32] = ties * 0.5625
                        key[0, 32:64].zero_()
                        if replay:
                            key[0, 64] = float("nan")
                            key[0, 96] = -float("inf")
                        references[tag] = encode_kv(key, pool_tag=tag)
                        backing[tag].fill_(0xA5)
                    values.normal_()
                    scores.normal_()
                    if csa_dtype == torch.float32 and replay:
                        # Preserve signed zero, subnormal, infinity and NaN payload.
                        values[0, :4].view(torch.int32).copy_(torch.tensor(
                            [-0x80000000, 1, 0x7F800000, 0x7FC12345],
                            device="cuda",
                            dtype=torch.int32,
                        ))
                        scores[0, :4].view(torch.int32).copy_(torch.tensor(
                            [0, -0x7FFFFFFF, -0x00800000, -0x003EDCBB],
                            device="cuda",
                            dtype=torch.int32,
                        ))
                    references["csa_state"] = torch.cat((values, scores), dim=1).float()
                    backing["csa_state"].fill_(-3.0)
                    graph.replay()

                    for tag, reference in references.items():
                        fill = -3.0 if tag == "csa_state" else 0xA5
                        expected = torch.full_like(backing[tag], fill)
                        for index, slot in enumerate(writes):
                            block, offset = divmod(slot, entries)
                            if not 0 < block < blocks:
                                continue
                            if tag == "indexer_kv":
                                payload = offset * 64
                                scale = entries * 64 + offset * 4
                                expected[block, payload : payload + 64] = reference[index, :64]
                                expected[block, scale : scale + 4] = reference[index, 64:]
                            else:
                                width = reference.shape[1]
                                start = offset * width
                                expected[block, start : start + width] = reference[index]
                        actual = backing[tag]
                        if tag == "csa_state":
                            actual = actual.view(torch.int32)
                            expected = expected.view(torch.int32)
                        if not torch.equal(actual, expected):
                            block, column = torch.nonzero(actual != expected)[0].tolist()
                            self.fail(
                                f"{tag} replay {replay} block {block} offset {column}: "
                                f"got {actual[block, column].item()}, "
                                f"expected {expected[block, column].item()}"
                            )


class GraphOutputProjectionDevice(unittest.TestCase):
    @torch.inference_mode()
    @mock.patch.dict(os.environ, {GRAPH_BATCHED_WO_A_ENV: "1"})
    def test_batched_wo_a_replay_and_rounding(self):
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(8274)
        grouped_weight = torch.randn(
            (8, 1024, 4096), device="cuda", dtype=torch.bfloat16
        ).mul_(0.01)
        # Exercise exact ties around 1.0 and distinguish the group lanes.
        grouped_weight[:, :2].zero_()
        grouped_weight[:, 0, :2] = 1
        grouped_weight[:, 1, 2] = torch.arange(1, 9, device="cuda")
        weight = grouped_weight.flatten(0, 1)
        reference_weight = grouped_weight.cpu().double().transpose(1, 2)
        for rows in (1, 8):
            with self.subTest(rows=rows):
                output = torch.randn(
                    (rows, 8, 4096), device="cuda", dtype=torch.bfloat16
                )
                self.assertIsNone(_try_batched_wo_a(output[..., ::2], weight))
                with mock.patch.dict(os.environ, {GRAPH_BATCHED_WO_A_ENV: "0"}):
                    self.assertIsNone(_try_batched_wo_a(output, weight))

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.assertIsNotNone(_try_batched_wo_a(output, weight))
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = _try_batched_wo_a(output, weight)
                self.assertIsNotNone(actual)
                self.assertEqual(actual.shape, (rows, 8, 1024))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.flatten(1).data_ptr(), actual.data_ptr())

                for replay, midpoint in enumerate((1 / 256, 3 / 256)):
                    output.normal_()
                    output[0, :, 0] = 1
                    output[0, :, 1] = midpoint
                    output[0, :, 2] = replay + 1
                    if rows > 1:
                        output[-1].zero_()
                    exact = torch.bmm(
                        output.cpu().double().transpose(0, 1), reference_weight
                    ).transpose(0, 1)
                    rounded = exact.to(torch.bfloat16).float()
                    legacy = torch.empty_like(actual)
                    for group in range(8):
                        torch.mm(
                            output[:, group],
                            grouped_weight[group].T,
                            out=legacy[:, group],
                        )
                    graph.replay()
                    result = actual.cpu().float()
                    legacy_result = legacy.cpu().float()
                    # FP32 reductions may differ near a BF16 midpoint.
                    # The absolute term covers small cancellation residuals.
                    for projected, reference in (
                        (result, rounded),
                        (legacy_result, rounded),
                        (result, legacy_result),
                    ):
                        torch.testing.assert_close(
                            projected, reference, rtol=2**-7, atol=2**-15
                        )
                    torch.testing.assert_close(
                        result[0, :, :2], rounded[0, :, :2], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        legacy_result[0, :, :2], rounded[0, :, :2], rtol=0, atol=0
                    )

                if rows == 8:
                    self._assert_replay_without_projection_copies(graph)

    def _assert_replay_without_projection_copies(self, graph):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            graph.replay()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batched-wo-a.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [event["name"].lower() for event in events if event.get("cat") == "kernel"]
        self.assertEqual(sum("gemm" in name for name in kernels), 1, kernels)
        self.assertFalse(any("copy_kernel" in name or "catarray" in name for name in kernels))
        self.assertFalse(any(event.get("cat") == "gpu_memcpy" for event in events))


if __name__ == "__main__":
    unittest.main()
