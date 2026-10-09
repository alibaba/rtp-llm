"""Opt-in sink division: numerical edges, cache invariants and component graphs."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_attention_triton as paged,
    _v41_prefill_attention_sm120 as prefill,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    dense_oracle,
    make_layer,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_sm120 import (
    oracle,
    pool,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_owner_sm120 import (
    decode_metadata,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
    compressor_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention


class SparsePrecisionSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41193)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def test_scoped_stage_selection_and_exception_restoration(self):
        original_prefill, original_decode = prefill._sparse_prefill, paged._attention
        with sparse_sink_div_rn(True, decode=False):
            outer = prefill._sparse_prefill
            self.assertIsNot(outer, original_prefill)
            self.assertIs(paged._attention, original_decode)
            with sparse_sink_div_rn(False):
                self.assertIs(prefill._sparse_prefill, outer)
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with sparse_sink_div_rn(True):
                    self.assertIsNot(paged._attention, original_decode)
                    raise RuntimeError("restore")
            self.assertIs(prefill._sparse_prefill, outer)
            self.assertIs(paged._attention, original_decode)
        self.assertIs(prefill._sparse_prefill, original_prefill)
        self.assertIs(paged._attention, original_decode)
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "SM120"):
                with sparse_sink_div_rn(True):
                    self.fail("Non-SM120 diagnostic admitted")
            with sparse_sink_div_rn(False):
                pass
        self.assertIs(prefill._sparse_prefill, original_prefill)
        self.assertIs(paged._attention, original_decode)

    @torch.inference_mode()
    def test_prefill_holes_lengths_and_extreme_sinks(self):
        with sparse_sink_div_rn(True):
            for heads, width in ((1, 17), (4, 128), (64, 640)):
                with self.subTest(heads=heads, width=width):
                    q = torch.randn(7, heads, 512, device="cuda", dtype=torch.bfloat16)
                    kv = torch.randn(769, 1, 512, device="cuda", dtype=torch.bfloat16)
                    ids = torch.randint(
                        769, (7, 1, width), device="cuda", dtype=torch.int32
                    )
                    ids[:, :, 3::5] = -1
                    ids[1].fill_(-1)
                    lengths = torch.tensor(
                        [0, width, 7, width // 2, width - 1, width, width],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    # Out-of-range entries outside a row's live length are inert.
                    ids[2, :, 7:] = 800
                    sinks = torch.linspace(-2, 3, heads, device="cuda")
                    inputs = (q, kv, ids, lengths, sinks)
                    before = [value.clone() for value in inputs]
                    actual = prefill.sparse_prefill_attention(*inputs)
                    torch.testing.assert_close(
                        actual.cpu(), dense_oracle(*inputs), rtol=0.01, atol=0.004
                    )
                    self.assertEqual(torch.count_nonzero(actual[:2]).item(), 0)
                    for value, saved in zip(inputs, before):
                        torch.testing.assert_close(value, saved, rtol=0, atol=0)
            q = torch.zeros(3, 4, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.full((33, 1, 512), 0.25, device="cuda", dtype=torch.bfloat16)
            ids = torch.arange(33, device="cuda", dtype=torch.int64)
            ids = ids[None, None].expand(3, 1, -1).contiguous()
            ids[1].fill_(-1)
            lengths = torch.tensor([0, 33, 33], device="cuda", dtype=torch.int64)
            sinks = torch.tensor([-1000, 1000, -100, 100], device="cuda").float()
            actual = prefill.sparse_prefill_attention(q, kv, ids, lengths, sinks)
            self.assertTrue(bool(actual.isfinite().all()))
            self.assertEqual(torch.count_nonzero(actual[:2]).item(), 0)
            torch.testing.assert_close(
                actual.cpu(), dense_oracle(q, kv, ids, lengths, sinks), rtol=0, atol=0
            )
            empty = prefill.sparse_prefill_attention(
                q[:0], kv, ids[:0], lengths[:0], sinks
            )
            self.assertEqual(empty.shape, q[:0].shape)

    @torch.inference_mode()
    def test_paged_masked_rows_and_extreme_sinks(self):
        with sparse_sink_div_rn(True):
            for gen in (0, 1):
                for tag in (None, "csa_kv", "hca_kv"):
                    with self.subTest(gen=gen, tag=tag):
                        swa, _, backing = pool("swa_kv", gen)
                        sw = torch.randint(
                            swa.layout.entries,
                            4 * swa.layout.entries,
                            (3, 128),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        sw[0].fill_(-1)
                        sw[1, ::3] = -1
                        sw[2].fill_(4 * swa.layout.entries + 1)
                        sources = [(swa, sw[:2, None])]
                        originals = [(backing, backing.clone())]
                        global_cache = gs = None
                        if tag:
                            global_cache, _, raw = pool(tag, gen)
                            gs = torch.randint(
                                global_cache.layout.entries,
                                4 * global_cache.layout.entries,
                                (3, 512),
                                device="cuda",
                                dtype=torch.int32,
                            )
                            gs[0].fill_(-1)
                            gs[1, ::5] = -1
                            gs[2].fill_(-2)
                            sources.append((global_cache, gs[:2, None]))
                            originals.append((raw, raw.clone()))
                        q = torch.randn(3, 4, 512, device="cuda", dtype=torch.bfloat16)
                        q[2].fill_(torch.nan)
                        sinks = torch.tensor(
                            [-1000, 1000, -1, 2], device="cuda", dtype=torch.float32
                        )
                        valid = torch.tensor([True, True, False], device="cuda")
                        actual = V41SparseAttention()(
                            q,
                            sinks,
                            swa,
                            sw,
                            valid_rows=valid,
                            global_cache=global_cache,
                            global_slots=gs,
                        )
                        torch.testing.assert_close(
                            actual[:2, None].cpu(),
                            oracle(q[:2, None], sinks, sources),
                            rtol=0.01,
                            atol=0.004,
                        )
                        self.assertTrue(bool(actual.isfinite().all()))
                        self.assertEqual(torch.count_nonzero(actual[[0, 2]]).item(), 0)
                        for raw, saved in originals:
                            torch.testing.assert_close(raw, saved, rtol=0, atol=0)

    @torch.inference_mode()
    def test_changed_input_component_graph_matches_eager_bitwise(self):
        for gen, tag in ((0, "csa_kv"), (1, "hca_kv")):
            with self.subTest(gen=gen, tag=tag), sparse_sink_div_rn(True):
                swa, _, swa_raw = pool("swa_kv", gen)
                global_cache, _, global_raw = pool(tag, gen)
                q = torch.randn(3, 4, 512, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(137, 1, 512, device="cuda", dtype=torch.bfloat16)
                ids = torch.full((3, 1, 129), -1, device="cuda", dtype=torch.int32)
                lengths = torch.zeros(3, device="cuda", dtype=torch.int32)
                sinks = torch.linspace(-2, 2, 4, device="cuda")
                sw = torch.full((3, 128), -1, device="cuda", dtype=torch.int32)
                gs = torch.full((3, 512), -1, device="cuda", dtype=torch.int32)
                valid = torch.ones(3, device="cuda", dtype=torch.bool)
                attention = V41SparseAttention()

                def run():
                    return (
                        prefill.sparse_prefill_attention(q, kv, ids, lengths, sinks),
                        attention(
                            q,
                            sinks,
                            swa,
                            sw,
                            valid_rows=valid,
                            global_cache=global_cache,
                            global_slots=gs,
                        ),
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = run()
                previous = None
                for iteration in range(3):
                    q.normal_()
                    kv.normal_()
                    sinks.copy_(
                        torch.tensor(
                            [-1000, 1000, iteration, -iteration], device="cuda"
                        )
                    )
                    ids.random_(137)
                    ids[iteration, :, ::3] = -1
                    lengths.copy_(torch.tensor([0, 65, 129], device="cuda"))
                    sw.random_(swa.layout.entries, 4 * swa.layout.entries)
                    gs.random_(
                        global_cache.layout.entries, 4 * global_cache.layout.entries
                    )
                    sw[iteration, ::3] = -1
                    gs[iteration, ::5] = -1
                    valid.fill_(True)
                    valid[iteration] = False
                    for cache in (swa, global_cache):
                        slots = torch.arange(
                            cache.layout.entries,
                            4 * cache.layout.entries,
                            device="cuda",
                        )
                        cache.write(
                            torch.randn(
                                len(slots), 512, device="cuda", dtype=torch.bfloat16
                            ),
                            slots,
                        )
                    inputs = (
                        q,
                        kv,
                        ids,
                        lengths,
                        sinks,
                        sw,
                        gs,
                        valid,
                        swa_raw,
                        global_raw,
                    )
                    saved = [value.clone() for value in inputs]
                    graph.replay()
                    expected = run()
                    for actual, reference in zip(outputs, expected):
                        self.assertTrue(bool(actual.isfinite().all()))
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    for value, before in zip(inputs, saved):
                        torch.testing.assert_close(value, before, rtol=0, atol=0)
                    if previous is not None:
                        self.assertFalse(torch.equal(outputs[0], previous))
                    previous = outputs[0].clone()

    @torch.inference_mode()
    def test_owner_prefill_and_eight_decode_cache_bytes_unchanged(self):
        for ratio in (0, 1, 2):
            for gen in (0, 1):
                with self.subTest(ratio=ratio, gen=gen):
                    source = 20 if ratio == 1 else 2 if ratio == 2 else 0
                    owners, caches = [], []
                    for _ in range(2):
                        torch.manual_seed(41194)
                        owner = make_layer(source, ratio, gen=gen)
                        owners.append(owner)
                        caches.append(NativeTaggedCache([owner], gen=gen))
                    starts, tables = [3, 125], None
                    prompt = torch.randn(
                        sum(starts), 512, device="cuda", dtype=torch.bfloat16
                    )
                    for index, (owner, cache) in enumerate(zip(owners, caches)):
                        with compressor_bf16(
                            True, fused=True, hca_projection=True
                        ), sparse_sink_div_rn(index == 1):
                            positions, current_tables, _ = prepare(
                                owner, prompt, cache, starts, tables=tables
                            )
                            output = owner(
                                prompt.clone(), positions, cache, current_tables
                            )
                        tables = current_tables
                        self.assertTrue(bool(output.isfinite().all()))
                    for key in caches[0].storage:
                        torch.testing.assert_close(
                            caches[0].storage[key],
                            caches[1].storage[key],
                            rtol=0,
                            atol=0,
                        )
                    for step in range(8):
                        x = torch.randn(2, 1, 512, device="cuda", dtype=torch.bfloat16)
                        selected = []
                        for index, (owner, cache) in enumerate(zip(owners, caches)):
                            metadata = decode_metadata(owner, cache, starts, tables)
                            with compressor_bf16(
                                True, fused=True, hca_projection=True
                            ), sparse_sink_div_rn(index == 1):
                                output = owner.forward_decode(
                                    x.clone(), metadata, cache
                                )
                            self.assertTrue(bool(output.isfinite().all()))
                            self.assertIsNone(owner._kv_cache)
                            self.assertIsNone(owner._block_tables_by_type)
                            if ratio:
                                selected.append(
                                    owner._shared_attention["topk"][source].clone()
                                )
                        if ratio:
                            torch.testing.assert_close(*selected, rtol=0, atol=0)
                        for key in caches[0].storage:
                            torch.testing.assert_close(
                                caches[0].storage[key],
                                caches[1].storage[key],
                                rtol=0,
                                atol=0,
                                msg=f"ratio={ratio} gen={gen} step={step} pool={key}",
                            )
                        starts = [start + 1 for start in starts]


if __name__ == "__main__":
    unittest.main()
