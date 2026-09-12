"""Actual selected-row FlashMLA execution, bounded allocation and Graph replay."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.flashmla import (
    PlanarGlobalBinding,
    PlanarSwaBinding,
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


if __name__ == "__main__":
    unittest.main()
