"""Actual selected-row FlashMLA execution, bounded allocation and Graph replay."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.flashmla import (
    PlanarGlobalBinding,
    PlanarSwaBinding,
    _finalize_native,
    _pack_selected_rows,
    flashmla_attention,
    flashmla_compact_attention,
    to_planar,
)
from test_compact_reader import _fixture, _ints, _probe_attention_output_buffers
from test_flashmla import FlashMLAGpuTest


class FlashMLACompactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FlashMLAGpuTest.setUpClass()

    def setUp(self):
        # Native byte-staging comparisons require the same native arithmetic.
        # The guard tests below explicitly select and verify the mixed reader.
        native = patch.dict(os.environ, {"DSV41_FLASHMLA_PRECISION_GUARD": "0"})
        native.start()
        self.addCleanup(native.stop)

    def _case(self):
        swa, global_kv = FlashMLAGpuTest.bindings(self)
        query, requests, positions, floors, selected, sinks = FlashMLAGpuTest.data(self)
        return (query, requests, positions, floors, swa, sinks), global_kv, selected

    def _reference(self, arguments, global_kv, selected):
        query, requests, positions, floors, swa, sinks = arguments
        return flashmla_attention(
            query,
            requests,
            positions,
            floors,
            PlanarSwaBinding.from_compact(swa),
            sinks,
            global_kv=PlanarGlobalBinding.from_compact(global_kv),
            global_indices=selected,
        )

    def _same(self, left, right):
        for field in ("output", "lse", "status"):
            torch.testing.assert_close(
                getattr(left, field), getattr(right, field), rtol=0, atol=0
            )

    def test_wrong_destination_type_is_rejected_before_write(self):
        pages, _ = _fixture(CacheRegion.SWA)
        output = replace(pages, data=pages.data.clone())
        before = output.data.clone()
        with self.assertRaises(TypeError):
            to_planar(pages, out=output)
        torch.testing.assert_close(output.data, before, rtol=0, atol=0)

    def test_raw_bytes_and_padding_are_exact_with_padded_source_stride(self):
        for region, entries, slots in (
            (CacheRegion.SWA, 136, 128),
            (CacheRegion.GLOBAL, 53, 512),
        ):
            with self.subTest(region=region):
                pages, _ = _fixture(region, entries=entries)
                physical = torch.full(
                    (2, 1, slots), -1, dtype=torch.int32, device="cuda"
                )
                physical[:, 0, :4] = _ints(
                    [[entries, 2 * entries + 3, 3 * entries, entries + 1]] * 2
                )
                packed, remapped = _pack_selected_rows(pages, physical)
                row_bytes, payload = (
                    (528, 512) if region == CacheRegion.SWA else (288, 256)
                )
                source = pages.data.cpu().view(4, entries, row_bytes)
                target = packed.data.cpu()
                for row in range(2):
                    for slot in range(4):
                        page, offset = divmod(int(physical[row, 0, slot]), entries)
                        actual = torch.cat(
                            (
                                target[row + 1, slot * payload : (slot + 1) * payload],
                                target[
                                    row + 1,
                                    slots * payload
                                    + slot * (row_bytes - payload) : slots * payload
                                    + (slot + 1) * (row_bytes - payload),
                                ],
                            )
                        )
                        torch.testing.assert_close(
                            actual, source[page, offset], rtol=0, atol=0
                        )
                        self.assertEqual(
                            int(remapped[row, 0, slot]), (row + 1) * slots + slot
                        )
                self.assertTrue((remapped[:, :, 4:] == -1).all().item())
                self.assertEqual(torch.count_nonzero(target[0]).item(), 0)
                self.assertEqual(
                    torch.count_nonzero(target[:, slots * row_bytes :]).item(), 0
                )

    def test_native_output_and_status_match_full_planar_for_both_ratios(self):
        arguments, global_kv, selected = self._case()
        for ratio in (1, 2):
            global_kv = replace(global_kv, compress_ratio=ratio)
            for invalid in (False, True):
                with self.subTest(ratio=ratio, invalid=invalid):
                    table = global_kv.page_table.clone()
                    if invalid:
                        table[0, 0] = 0
                    binding = replace(global_kv, page_table=table)
                    actual = flashmla_compact_attention(
                        *arguments, global_kv=binding, global_indices=selected
                    )
                    self._same(actual, self._reference(arguments, binding, selected))

    def test_storage_does_not_scale_with_allocated_pool_capacity(self):
        arguments, global_kv, selected = self._case()
        expected = self._reference(arguments, global_kv, selected)
        observations = []
        for capacity in (4, 8193):
            storage = torch.zeros(
                (capacity, global_kv.pages.data.stride(0)),
                dtype=torch.uint8,
                device="cuda",
            )
            storage[:4, : global_kv.pages.data.shape[1]].copy_(global_kv.pages.data)
            binding = replace(global_kv, pages=replace(global_kv.pages, data=storage))
            flashmla_compact_attention(
                *arguments, global_kv=binding, global_indices=selected
            )
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            actual = flashmla_compact_attention(
                *arguments, global_kv=binding, global_indices=selected
            )
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            self._same(actual, expected)
            self.assertLess(peak, 16 * 1024 * 1024)
            observations.append({"capacity_pages": capacity, "peak_extra_bytes": peak})
            del actual, storage, binding
        self.assertLessEqual(
            abs(
                observations[1]["peak_extra_bytes"]
                - observations[0]["peak_extra_bytes"]
            ),
            1024 * 1024,
        )
        report = {
            "observations": observations,
            "query_rows": 6,
            "global_slots": 512,
            "scope": "component allocation probe, not real 1M generation",
        }
        folder = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "selected_rows_memory.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report), flush=True)

    def test_changed_ids_lengths_candidates_and_bytes_under_one_graph(self):
        arguments, global_kv, selected = self._case()
        query, requests, positions, floors, swa, _ = arguments
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            flashmla_compact_attention(
                *arguments, global_kv=global_kv, global_indices=selected
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = flashmla_compact_attention(
                *arguments, global_kv=global_kv, global_indices=selected
            )
        pointers = (
            actual.output.data_ptr(),
            actual.lse.data_ptr(),
            actual.status.data_ptr(),
        )
        for iteration in range(6):
            query.mul_(-1)
            global_kv.page_table.copy_(global_kv.page_table.roll(1, dims=0))
            swa.page_ids.copy_(swa.page_ids.roll(1))
            positions[1] = 3 + iteration
            swa.valid_ends[0] = 200 + iteration
            selected[1, 0] = iteration % 2
            # Copy complete encoded rows, preserving the exact format and scales.
            global_kv.pages.data[1].copy_(global_kv.pages.data[2])
            graph.replay()
            torch.cuda.synchronize()
            self._same(actual, self._reference(arguments, global_kv, selected))
            self.assertEqual(
                pointers,
                (
                    actual.output.data_ptr(),
                    actual.lse.data_ptr(),
                    actual.status.data_ptr(),
                ),
            )

    def test_output_aliases_and_disjoint_graph_buffers(self):
        _probe_attention_output_buffers(self, flashmla_compact_attention)

    def test_fast_staging_matches_legacy_with_graph_metadata_and_byte_changes(self):
        arguments, global_kv, selected = self._case()
        query, requests, positions, floors, swa, _ = arguments

        def operation():
            return flashmla_compact_attention(
                *arguments, global_kv=global_kv, global_indices=selected
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict(os.environ, {"DSV41_FLASHMLA_FAST_STAGING": "1"}):
            with torch.cuda.stream(stream):
                operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = operation()
        addresses = [
            tensor.data_ptr() for tensor in (actual.output, actual.lse, actual.status)
        ]
        for iteration in range(6):
            query.mul_(-1)
            global_kv.page_table.copy_(global_kv.page_table.roll(1, dims=0))
            swa.page_ids.copy_(swa.page_ids.roll(1))
            positions[1] = 3 + iteration
            swa.valid_ends[0] = 200 + iteration
            selected[1, 0] = iteration % 2
            global_kv.pages.data[1].copy_(global_kv.pages.data[2])
            if iteration == 5:
                positions[0] = -1
                query[0].fill_(float("nan"))
                global_kv.pages.data[0].fill_(255)
            with patch.dict(os.environ, {"DSV41_FLASHMLA_FAST_STAGING": "0"}):
                expected = operation()
            graph.replay()
            torch.cuda.synchronize()
            self._same(actual, expected)
            self.assertEqual(
                addresses,
                [
                    tensor.data_ptr()
                    for tensor in (actual.output, actual.lse, actual.status)
                ],
            )

    def test_fused_finalize_masks_special_values_and_refreshes_graph(self):
        for rows, heads in ((0, 64), (1, 64), (7, 128), (128, 64)):
            with self.subTest(rows=rows, heads=heads):
                native = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
                native_lse = torch.randn(rows, heads, device="cuda")
                sinks = torch.randn(heads, device="cuda")
                sinks[:4] = torch.tensor([float("inf"), -float("inf"), float("nan"), 0], device="cuda")
                if rows:
                    native_lse[:, :4] = sinks[:4]
                metadata = SimpleNamespace(
                    query_valid=torch.ones(rows, device="cuda", dtype=torch.bool),
                    have_kv=torch.ones(rows, device="cuda", dtype=torch.bool),
                    status=torch.zeros(rows, device="cuda", dtype=torch.int32),
                )
                actual = (torch.empty_like(native), torch.empty_like(native_lse),
                          torch.empty(rows, heads, device="cuda", dtype=torch.int32))
                expected = tuple(torch.empty_like(value) for value in actual)

                def operation(outputs):
                    _finalize_native(native, native_lse, sinks, metadata, *outputs)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with patch.dict(os.environ, {"DSV41_FLASHMLA_FUSED_FINALIZE": "1"}):
                    with torch.cuda.stream(stream):
                        operation(actual)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        operation(actual)
                pointers = [value.data_ptr() for value in actual]
                for iteration in range(4):
                    native.neg_()
                    if rows:
                        metadata.query_valid[0] = iteration != 1
                        metadata.have_kv[-1] = iteration not in (1, 2)
                        metadata.status[0] = iteration
                        if iteration == 1:
                            native[0].fill_(float("nan"))
                        else:
                            native[0].fill_(iteration)
                    with patch.dict(os.environ, {"DSV41_FLASHMLA_FUSED_FINALIZE": "0"}):
                        operation(expected)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=1e-6, equal_nan=True)
                    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
                    self.assertEqual(pointers, [value.data_ptr() for value in actual])

    def test_fused_finalize_actual_native_matches_legacy(self):
        arguments, global_kv, selected = self._case()

        def operation():
            return flashmla_compact_attention(
                *arguments, global_kv=global_kv, global_indices=selected
            )

        for staging in ("0", "1"):
            with patch.dict(os.environ, {"DSV41_FLASHMLA_FAST_STAGING": staging,
                                         "DSV41_FLASHMLA_FUSED_FINALIZE": "0"}):
                expected = operation()
            with patch.dict(os.environ, {"DSV41_FLASHMLA_FAST_STAGING": staging,
                                         "DSV41_FLASHMLA_FUSED_FINALIZE": "1"}):
                actual = operation()
            torch.testing.assert_close(actual.output, expected.output, rtol=0, atol=0)
            torch.testing.assert_close(actual.lse, expected.lse, rtol=0, atol=1e-6)
            torch.testing.assert_close(actual.status, expected.status, rtol=0, atol=0)

    def test_precision_guard_mixed_rows_refresh_under_one_graph(self):
        self._check_precision_guard_graph()

    def test_precision_guard_paired_heads_refresh_under_one_graph(self):
        self._check_precision_guard_graph(rows_repeat=2, optimized=True)

    def _check_precision_guard_graph(self, rows_repeat=1, optimized=False):
        from rtp_llm.models_py.modules.dsv41.compact_reader import compact_attention
        from rtp_llm.models_py.modules.dsv41.flashmla import build_indices

        arguments, global_kv, selected = self._case()
        query, requests, positions, floors, swa, sinks = arguments
        if rows_repeat != 1:
            query = query.repeat(rows_repeat, 1, 1)
            requests = requests.repeat(rows_repeat)
            positions = positions.repeat(rows_repeat)
            floors = floors.repeat(rows_repeat)
            selected = selected.repeat(rows_repeat, 1)
            arguments = query, requests, positions, floors, swa, sinks
        global_kv = replace(global_kv, page_table=global_kv.page_table.repeat(1, 3))
        positions.fill_(2048)
        floors.fill_(1921)
        swa.valid_starts.fill_(1793)
        swa.valid_ends.fill_(2049)
        positions[-1] = -1
        floors[-1] = 0
        full = torch.arange(512, device=query.device, dtype=torch.int32)
        selected.copy_(full[None].expand_as(selected))
        selected[1, 1:] = -1
        selected[2, 63:] = -1

        def operation(reader=flashmla_compact_attention):
            return reader(*arguments, global_kv=global_kv, global_indices=selected)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict(
            os.environ,
            {
                "DSV41_FLASHMLA_FAST_STAGING": "1",
                "DSV41_FLASHMLA_PRECISION_GUARD": "1",
                "DSV41_FLASHMLA_SKIP_EMPTY_TILES": str(int(optimized)),
            },
        ):
            with torch.cuda.stream(stream):
                operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = operation()
        pointers = [
            tensor.data_ptr() for tensor in (actual.output, actual.lse, actual.status)
        ]
        for iteration in range(6):
            selected.copy_(full[None].expand_as(selected))
            selected[iteration % 4, 1:] = -1
            selected[(iteration + 1) % 4, 63:] = -1
            selected[(iteration + 2) % 4, 511:] = -1
            query.mul_(-1)
            swa.page_ids.copy_(swa.page_ids.roll(1))
            global_kv.pages.data[1].copy_(global_kv.pages.data[2])
            if iteration == 5:
                query[-1].fill_(float("nan"))
                global_kv.pages.data[0].fill_(255)
            with patch.dict(os.environ, {"DSV41_FLASHMLA_PRECISION_GUARD": "0"}):
                native = operation()
            fallback = operation(compact_attention)
            metadata = build_indices(
                requests,
                positions,
                floors,
                swa,
                global_kv=global_kv,
                global_indices=selected,
            )
            partial = metadata.extra_lengths < 512
            graph.replay()
            torch.cuda.synchronize()
            for field in ("output", "lse", "status"):
                left, right = getattr(native, field), getattr(fallback, field)
                mask = partial.view(-1, *([1] * (left.ndim - 1)))
                expected = torch.where(mask, right, left)
                torch.testing.assert_close(
                    getattr(actual, field), expected, rtol=0, atol=0
                )
            self.assertEqual(
                pointers,
                [
                    tensor.data_ptr()
                    for tensor in (actual.output, actual.lse, actual.status)
                ],
            )

    def test_precision_guard_preserves_output_alias_validation(self):
        with patch.dict(os.environ, {"DSV41_FLASHMLA_PRECISION_GUARD": "1"}):
            _probe_attention_output_buffers(self, flashmla_compact_attention)

    def test_precision_guard_static_fallback_keeps_input_validation(self):
        arguments, global_kv, selected = self._case()
        query, requests, positions, floors, swa, sinks = arguments
        with patch.dict(os.environ, {"DSV41_FLASHMLA_PRECISION_GUARD": "1"}):
            for offset in (1, 2, 3):
                for invalid in (
                    arguments[offset].float(),
                    arguments[offset][:-1],
                    arguments[offset].cpu(),
                    arguments[offset][:, None].repeat(1, 2)[:, 0],
                ):
                    changed = list(arguments)
                    changed[offset] = invalid
                    with self.subTest(offset=offset, dtype=invalid.dtype, shape=invalid.shape):
                        with self.assertRaises(ValueError):
                            flashmla_compact_attention(*changed)
            with self.assertRaises(ValueError):
                flashmla_compact_attention(*arguments, global_indices=selected)
            for invalid in (
                None, selected[0], selected[:1], selected.float(), selected[:, :511],
                selected.cpu(), torch.cat((selected, selected[:, :1]), dim=1),
            ):
                with self.subTest(indices=None if invalid is None else (invalid.shape, invalid.dtype)):
                    with self.assertRaises(ValueError):
                        flashmla_compact_attention(
                            *arguments, global_kv=global_kv, global_indices=invalid
                        )
            for invalid_swa in (
                replace(swa, valid_ends=swa.valid_ends.float()),
                replace(swa, pages=replace(swa.pages, entries_per_page=127)),
            ):
                with self.assertRaises(ValueError):
                    flashmla_compact_attention(query, requests, positions, floors, invalid_swa, sinks)
            with self.assertRaises(ValueError):
                flashmla_compact_attention(*arguments, output=query)
            with self.assertRaises(ValueError):
                flashmla_compact_attention(
                    *arguments,
                    lse=torch.empty(query.shape[:2], device=query.device, dtype=torch.bfloat16),
                )

    def test_precision_guard_static_fallback_graph_keeps_status_and_skips_indices(self):
        from rtp_llm.models_py.modules.dsv41.compact_reader import compact_attention

        for capacity in (None, 0, 1, 511):
            arguments, global_kv, selected = self._case()
            query, requests, positions, floors, swa, _ = arguments
            kwargs = {} if capacity is None else {
                "global_kv": global_kv,
                "global_indices": selected[:, :capacity].contiguous(),
            }
            saved = [tensor.clone() for tensor in (requests, positions, floors, swa.page_ids)]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with patch.dict(os.environ, {"DSV41_FLASHMLA_PRECISION_GUARD": "1"}), patch(
                "rtp_llm.models_py.modules.dsv41.flashmla.build_indices",
                side_effect=AssertionError("static fallback must not build native indices"),
            ):
                with torch.cuda.stream(stream):
                    flashmla_compact_attention(*arguments, **kwargs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = flashmla_compact_attention(*arguments, **kwargs)
            pointers = [tensor.data_ptr() for tensor in (actual.output, actual.lse, actual.status)]
            for fault, expected_status in (
                ("request", 2), ("position", 2), ("negative_position", 2),
                ("floor", 2), ("page", 1), ("padding", 0), ("recovered", 0),
            ):
                for tensor, value in zip((requests, positions, floors, swa.page_ids), saved):
                    tensor.copy_(value)
                if fault == "request":
                    requests[0] = swa.page_ids.numel()
                elif fault == "position":
                    positions[0] = 1048576
                elif fault == "negative_position":
                    positions[0] = -2
                elif fault == "floor":
                    floors[0] = positions[0] + 1
                elif fault == "page":
                    swa.page_ids[requests[0]] = 0
                elif fault == "padding":
                    positions[0] = -1
                expected = compact_attention(*arguments, **kwargs)
                graph.replay()
                torch.cuda.synchronize()
                self._same(actual, expected)
                self.assertTrue((actual.status[0] == expected_status).all().item())
                self.assertEqual(
                    pointers,
                    [tensor.data_ptr() for tensor in (actual.output, actual.lse, actual.status)],
                )


if __name__ == "__main__":
    unittest.main()
