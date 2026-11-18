"""CP8 flat-layout bytes/ownership and real native mixed-request scoring.

CPU tests double only device/stream/upload admission and the scoring boundary.
Device tests retain actual MXFP4 bytes, BF16 weights, both native providers,
token selection and the formal M1024/K65536 query shape.
"""

from contextlib import ExitStack
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_indexer_layout as layout
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_paged_indexer as paged
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_paged_indexer import (
    UPLOAD, _cpu_upload, _device_values, _fixture,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import (
    assert_equivalent, reference,
)


def _rows(device="cpu", *, values=None):
    state, bank, kwargs = _fixture(
        (65536,) * 8, 63488, (512,) * 8, ratio=2, source=2,
        device=device, values=values,
    )
    cp, _ = _context(0, (4096,) * 8, (126976,) * 8, device)
    state.context, state.compacted, state.shared = cp, False, {}
    state._cp_metadata_stream = layout._stream_key(cp.global_positions.device), None
    rows = paged.prepare_rows(
        state, bank, ratio=2, request_slices=kwargs["request_slices"],
        positions=cp.global_positions, request_ids=cp.req_id_per_token,
    )
    if rows is None:
        raise AssertionError("formal CP8 query rows were not admitted")
    return rows


class CP8IndexerLayoutContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ))
        os.environ.pop(layout.ENV, None)
        self.stack.enter_context(patch.object(paged, "_device_supported", return_value=True))
        self.stack.enter_context(patch.object(paged, "_deep_gemm", return_value=object()))
        self.stack.enter_context(patch.object(layout, "_stream_key", return_value=(torch.device("cpu"), 7)))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_operator_eligible", return_value=True,
        ))
        self.stack.enter_context(patch(UPLOAD, side_effect=_cpu_upload))

    def test_CP8_planes_request_boundaries_and_single_slot_lifetime(self):
        rows = _rows()
        bank = rows.bank
        for source in (2, 8, 14):
            self.assertTrue(layout.eligible(replace(rows, bank=replace(bank, source=source))))
        for source in (0, 6, 12, 20):
            self.assertFalse(layout.eligible(replace(rows, bank=replace(bank, source=source))))
        with torch.inference_mode():
            packed = layout.prepare(rows.slice(slice(0, 1024)))
        self.assertTrue(packed.valid())
        self.assertFalse(packed.payload.is_inference())
        self.assertFalse(packed.scales.is_inference())
        self.assertEqual((packed.payload.shape, packed.scales.shape), ((524288, 64), (524288,)))
        self.assertTrue(packed.payload.is_contiguous() and packed.scales.is_contiguous())
        self.assertEqual(packed.nbytes, 35651584)
        storage = bank.pages.view(bank.pages.shape[0], 256 * 68)
        for request in range(8):
            for logical in (0, 255, 256, 63487, 63488, 65535):
                page = bank.request_table_host[request][logical // 256]
                entry = logical % 256
                actual = request * packed.request_stride + logical
                self.assertTrue(torch.equal(
                    packed.payload[actual].view(torch.uint8),
                    storage[page, entry * 64:(entry + 1) * 64],
                ))
                expected_sf = storage[page, 256 * 64 + entry * 4:256 * 64 + (entry + 1) * 4]
                self.assertTrue(torch.equal(packed.scales[actual:actual + 1].view(torch.uint8), expected_sf))
        self.assertIs(layout.prepare(rows.slice(slice(1024, 2048))), packed)
        stats = bank.cache._cp_indexer_layout_stats
        self.assertEqual((stats["packs"], stats["copies"], stats["writeBytes"]), (1, 2, 35651584))
        self.assertEqual(stats["prefixDuplicateBytes"], 7 * 63488 * 68)
        source_keys = tuple(paged._source_tensor_key(v) for v in (bank.pages, bank.request_table))
        for source in (8, 14):
            next_bank = replace(bank, source=source, epoch=bank.epoch + source)
            bank.cache._valid_shared_indexer = lambda value, active=next_bank: (
                value is active and source_keys == tuple(paged._source_tensor_key(v) for v in (value.pages, value.request_table))
            )
            second = layout.prepare(replace(rows, bank=next_bank).slice(slice(0, 1024)))
            self.assertIs(bank.cache._cp_indexer_flat_layout, second)
            self.assertFalse(packed.valid())
        self.assertEqual(stats["packs"], 3)
        self.assertEqual(stats["copies"], 6)
        self.assertEqual(stats["writeBytes"], 106954752)
        self.assertEqual(stats["retainedBytes"], 35651584)
        self.assertEqual(stats["peakRetainedBytes"], 35651584)

    def test_CP8_offset_semantics_mutation_and_unsupported_no_work(self):
        rows = _rows()
        with patch.object(torch.Tensor, "index_select", side_effect=AssertionError("unsupported packing")):
            with patch.dict(os.environ, {layout.ENV: "0"}):
                self.assertIsNone(layout.prepare(rows))
            self.assertIsNone(layout.prepare(rows.slice(slice(510, 518))))
            self.assertFalse(layout.eligible(replace(rows, ratio=1)))
            rows.state.compacted = True
            self.assertIsNone(layout.prepare(rows))
            rows.state.compacted = False
        tile = rows.slice(slice(508, 516))
        q = torch.zeros(8, 32, 64, dtype=torch.int8)
        sf = torch.zeros(8, 32, dtype=torch.int32)
        weights = torch.ones(8, 32, dtype=torch.bfloat16)
        calls = []

        def operation(query, keys, head_weights, starts, ends, *, max_seqlen_k):
            calls.append((query, keys))
            self.assertTrue(torch.equal(starts, torch.tensor((0,) * 4 + (65536,) * 4, dtype=torch.int32)))
            self.assertTrue(torch.equal(ends - starts, tile.visible))
            self.assertEqual(max_seqlen_k, 65536)
            self.assertIs(query[0], q)
            self.assertEqual(keys[0].shape, (524288, 64))
            return torch.zeros(8, 65536, dtype=torch.bfloat16)

        self.assertEqual(layout.score(q, sf, weights, tile, operation).shape, (8, 65536))
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows.bank.cache._cp_indexer_layout_stats["denseCalls"], 1)
        with self.assertRaisesRegex(RuntimeError, "no logits after launch"):
            layout.score(q, sf, weights, tile, lambda *args, **kwargs: None)
        self.assertEqual(rows.bank.cache._cp_indexer_layout_stats["denseCalls"], 1)
        slot = rows.bank.cache._cp_indexer_flat_layout
        slot.payload.add_(0)
        with self.assertRaisesRegex(RuntimeError, "stale CP flat"):
            layout.prepare(tile)

    def test_CP8_same_stream_query_cannot_rebaseline_mutated_source_or_rows(self):
        for field in ("pages", "table", "visible"):
            rows = _rows()
            value = {"pages": rows.bank.pages, "table": rows.bank.request_table,
                     "visible": rows.visible}[field]

            def stream(_device):
                value.add_(0)
                return torch.device("cpu"), 7

            with patch.object(layout, "_stream_key", side_effect=stream), patch.object(
                torch.Tensor, "index_select", side_effect=AssertionError("changed source packing")
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during flat layout admission"):
                    layout.prepare(rows)
            self.assertIsNone(getattr(rows.bank.cache, "_cp_indexer_flat_layout", None))
        rows = _rows().slice(slice(0, 4))
        layout.prepare(rows)
        q = torch.zeros(4, 32, 64, dtype=torch.int8)
        sf = torch.zeros(4, 32, dtype=torch.int32)
        weights = torch.ones(4, 32, dtype=torch.bfloat16)

        def stream(_device):
            q.add_(0)
            return torch.device("cpu"), 7

        with patch.object(layout, "_stream_key", side_effect=stream):
            with self.assertRaisesRegex(RuntimeError, "changed before native scoring"):
                layout.score(q, sf, weights, rows, lambda *a, **k: self.fail("mutated native inputs"))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class CP8IndexerLayoutDevice(unittest.TestCase):
    def test_CP8_actual_M1024_K65536_flat_provider_logits_and_selection(self):
        with torch.no_grad(), patch(
            "rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_operator_eligible", return_value=True,
        ), patch.dict(os.environ, {token.ENV: "1"}):
            self.assertTrue(_quant.is_ppu())
            self.assertTrue(callable(getattr(paged._deep_gemm(), "fp8_fp4_mqa_logits", None)))
            for nonfinite in (False, True):
                values = _device_values((65536,) * 8, 63488, nonfinite=nonfinite)
                rows = _rows("cuda", values=values).slice(slice(0, 1024))
                query = torch.randn(1024, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
                q, sf = _quant.quantize_indexer_q(query)
                weights = torch.randn(1024, 32, device="cuda", dtype=torch.bfloat16) * 0.01
                with patch.dict(os.environ, {layout.ENV: "0"}):
                    expected = paged.score_dense(q, sf, weights, rows)
                with patch.dict(os.environ, {layout.ENV: "1"}):
                    # Warm up packing/native flat compilation before observing
                    # the real backend; neither scorer nor its math is doubled.
                    paged.score_dense(q, sf, weights, rows)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        actual = paged.score_dense(q, sf, weights, rows)
                        torch.cuda.synchronize()
                names = [event.name for event in prof.events()]
                self.assertTrue(any("attention_mqa_logits_fp4" in name for name in names), names[:20])
                mask = torch.arange(65536, device="cuda")[None] < rows.visible[:, None]
                torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0, equal_nan=True)
                if not nonfinite:
                    self.assertTrue(torch.equal(actual[mask].view(torch.uint8), expected[mask].view(torch.uint8)))
                original = expected.masked_fill(~mask, -torch.inf)
                selected = token.try_select_tokens(actual, rows.visible, disposable=True)
                self.assertIsNotNone(selected)
                assert_equivalent(self, selected, reference(original, rows.visible), original, rows.visible)
                self.assertEqual(rows.bank.cache._cp_indexer_layout_stats["packs"], 1)
                self.assertEqual(rows.bank.cache._cp_indexer_layout_stats["copies"], 2)
                del actual, expected, original, selected, query, q, sf, rows, values, weights


if __name__ == "__main__":
    unittest.main()
