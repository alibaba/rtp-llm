"""CPU addressing contracts and GPU kernel parity for opt-in CP compact prefill."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.msa_cp_compact import (
    _get_compact_geometry,
    build_source_metadata,
    materialize_pages,
    restore_idx_pages,
    selected_page_map,
    selected_page_map_reference,
    validate_compact_mode,
)


class CompactCpMetadataTest(unittest.TestCase):
    def test_source_rows_and_suffix_offsets(self):
        metadata = build_source_metadata(
            [128, 256],
            torch.tensor([133, 259]),
            512,
            128,
            torch.tensor([0, 4, 5]),
            torch.tensor([2, 0, 1]),
        )
        prefix, lens, offsets, rows = metadata
        self.assertEqual(prefix.tolist(), [128, 256])
        self.assertEqual(lens.tolist(), [133, 259])
        self.assertEqual(offsets.tolist(), [0, 5])
        self.assertEqual(rows.tolist(), [2, -1, -1, -1, 0, 1, -1, -1])
        self.assertTrue(all(t.dtype == torch.int64 for t in metadata))

    def test_selected_request_pages_and_empty_selection(self):
        table = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32)
        topk = torch.tensor([[[0, -1], [1, -1]], [[1, -1], [2, -1]]])
        ids, compact = selected_page_map(topk, table, [0, 1, 2])
        self.assertEqual(ids.tolist(), [0, 1, 5, 6])
        self.assertEqual(compact.tolist(), [[1, 2, 0], [0, 3, 4]])
        self.assertEqual(compact.stride(0) % 4, 0)
        ids, compact = selected_page_map(torch.full_like(topk, -1), table, [0, 1, 2])
        self.assertEqual(ids.numel(), 0)
        self.assertFalse(compact.any())

    def test_overlap_is_rejected(self):
        validate_compact_mode(False, False)
        for packed, prefix in ((True, False), (False, True), (True, True)):
            with self.assertRaisesRegex(ValueError, "requires"):
                validate_compact_mode(packed, prefix)

    def test_alignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "page aligned"):
            build_source_metadata(
                [3], torch.tensor([7]), 256, 128, torch.tensor([]).long(), None
            )


class CompactGeometryCacheTest(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "num_kv_heads": 2,
            "qo_segment_lens": torch.tensor([3, 2], dtype=torch.int32),
            "seqused_k": torch.tensor([259, 258], dtype=torch.int32),
            "kv_segment_lens": torch.tensor([384, 384], dtype=torch.int32),
            "_chunk_meta": object(),
        }
        self.page_map = torch.arange(6, dtype=torch.int32)
        self.patch = patch(
            "rtp_llm.models_py.triton_kernels.sparse_msa.prefill."
            "topk_bt_fused._build_chunk_meta",
            side_effect=lambda plan, page_map, *args: {"pages": page_map.clone()},
        )
        self.builder = self.patch.start()
        self.addCleanup(self.patch.stop)

    def geometry(
        self, plan=None, page_map=None, chunk_size=4, partial_dtype=torch.bfloat16
    ):
        return _get_compact_geometry(
            self.plan if plan is None else plan,
            self.page_map if page_map is None else page_map,
            16,
            128,
            chunk_size,
            32,
            128,
            partial_dtype,
            torch.device("cpu"),
        )

    def test_repeated_layers_reuse_and_keep_legacy_isolated(self):
        legacy = self.plan["_chunk_meta"]
        first, boundaries = self.geometry()
        second, reused_boundaries = self.geometry()
        self.assertIs(first, second)
        self.assertIs(boundaries, reused_boundaries)
        self.assertEqual(boundaries, [[0, 3, 4], [0, 1]])
        self.assertEqual(first["query_segments"][0].tolist(), [0, 0, 0, 1])
        self.assertEqual(first["query_segments"][1].tolist(), [0])
        self.assertIs(first["query_segments"], second["query_segments"])
        self.assertEqual(self.builder.call_count, 1)
        self.assertIs(self.plan["_chunk_meta"], legacy)

    def test_replacement_and_inplace_map_changes_invalidate(self):
        first, _ = self.geometry()
        self.page_map = self.page_map.clone()
        second, _ = self.geometry()
        self.assertIsNot(first, second)
        self.page_map.add_(7)
        third, _ = self.geometry()
        self.assertIsNot(second, third)
        self.assertTrue(torch.equal(third["pages"], self.page_map))
        self.assertEqual(self.builder.call_count, 3)

    def test_length_and_chunk_geometry_changes_invalidate(self):
        previous, _ = self.geometry()
        for name in ("qo_segment_lens", "seqused_k", "kv_segment_lens"):
            self.plan[name].add_(1)
            current, _ = self.geometry()
            self.assertIsNot(previous, current)
            previous = current
        changed_chunk, _ = self.geometry(chunk_size=8)
        self.assertIsNot(previous, changed_chunk)
        changed_partial, _ = self.geometry(
            chunk_size=8, partial_dtype=torch.float8_e4m3fn
        )
        self.assertIsNot(changed_chunk, changed_partial)

    def test_copied_plan_cannot_reuse_other_forward_cache(self):
        first, _ = self.geometry()
        copied_plan = dict(self.plan)
        second, _ = self.geometry(plan=copied_plan)
        self.assertIsNot(first, second)
        self.assertIs(self.geometry()[0], first)
        self.assertEqual(self.builder.call_count, 2)

    def test_selection_is_recomputed_with_reused_geometry(self):
        first, _ = self.geometry()
        table = self.page_map.view(2, 3)
        selection_a = torch.tensor([[[0], [1]], [[0], [1]]])
        selection_b = torch.tensor([[[2], [0]], [[2], [0]]])
        ids_a, _ = selected_page_map(selection_a, table, [0, 1, 2])
        ids_b, _ = selected_page_map(selection_b, table, [0, 1, 2])
        self.assertFalse(torch.equal(ids_a, ids_b))
        self.assertIs(self.geometry()[0], first)
        self.assertEqual(self.builder.call_count, 1)

    def test_versionless_inference_map_bypasses_cache(self):
        with torch.inference_mode():
            page_map = self.page_map.clone()
        first, _ = self.geometry(page_map=page_map)
        with torch.inference_mode():
            page_map.add_(1)
        second, _ = self.geometry(page_map=page_map)
        self.assertIsNot(first, second)
        self.assertTrue(torch.equal(second["pages"], page_map))
        self.assertNotIn("_compact_geometry", self.plan)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompactCpKernelTest(unittest.TestCase):
    def test_bitmap_selection_matches_reference(self):
        device = torch.device("cuda", 0)
        # Nonmultiple-of-four width exercises aligned rows. Segments 0/1 are
        # one request but select different logical pages; row masks must remain
        # exact instead of filling all globally selected aliases into both rows.
        storage = torch.full((3, 584), -1, dtype=torch.int32, device=device)
        table = storage[:, :581]
        table[0] = torch.arange(581, device=device)
        table[1] = table[0]
        table[2] = torch.arange(1024, 1605, device=device)
        boundaries = [0, 7, 19, 31]
        cases = {}
        cases["padding"] = torch.full((4, 31, 16), -1, dtype=torch.int32, device=device)
        cases["hotspot"] = torch.zeros_like(cases["padding"])
        sparse = cases["padding"].clone()
        for head in range(4):
            sparse[head, :7, :3] = torch.tensor([head, 200 + head, 580], device=device)
            sparse[head, 7:19, :2] = torch.tensor(
                [100 + head, 400 + head], device=device
            )
            sparse[head, 19:, :2] = torch.tensor([head, 580], device=device)
        cases["sparse_heads_partial_tail"] = sparse
        cases["all_pages"] = (
            torch.arange(4 * 31 * 32, device=device)
            .reshape(4, 31, 32)
            .remainder(581)
            .int()
        )
        for name, topk in cases.items():
            with self.subTest(case=name):
                expected_ids, expected_table = selected_page_map_reference(
                    topk, table, boundaries
                )
                actual_ids, actual_table = selected_page_map(
                    topk, table, boundaries, page_capacity=2048
                )
                self.assertTrue(torch.equal(actual_ids, expected_ids))
                self.assertTrue(torch.equal(actual_table, expected_table))
                self.assertEqual(actual_table.stride(0) % 4, 0)
        # Consecutive calls deliberately change selection on the same table.
        for topk in (sparse, cases["hotspot"], cases["padding"], sparse):
            actual = selected_page_map(topk, table, boundaries, page_capacity=2048)
            expected = selected_page_map_reference(topk, table, boundaries)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_bitmap_empty_query_and_noncontiguous_topk(self):
        device = torch.device("cuda", 0)
        table = torch.tensor(
            [[0, 3, 7], [16, 20, 31]], device=device, dtype=torch.int32
        )
        for topk, boundaries in (
            (torch.empty(2, 0, 4, dtype=torch.int32, device=device), [0, 0, 0]),
            (
                torch.arange(2 * 6 * 4, device=device)
                .reshape(2, 6, 4)
                .remainder(3)[:, ::2],
                [0, 1, 3],
            ),
        ):
            actual = selected_page_map(topk, table, boundaries, page_capacity=32)
            expected = selected_page_map_reference(topk, table, boundaries)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def _inputs(self, prefix_dtype):
        device = torch.device("cuda", 0)
        page, heads, dim, ni = 128, 2, 128, 64
        nk = heads * dim
        # Original BF16 suffix values intentionally not representable in E4M3.
        packed = torch.arange(8 * (2 * nk + ni), device=device).reshape(8, -1)
        packed = (packed.float().remainder(31) / 128 + 1).to(torch.bfloat16)
        unpad = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3], device=device)
        main = torch.randn(3, 2, heads, page, dim, device=device).to(prefix_dtype)
        dst = torch.tensor([0, 4, 5], device=device)
        rows = torch.tensor([2, 0, 1], device=device)
        lengths = torch.tensor([133, 259], device=device)
        metadata = build_source_metadata([128, 256], lengths, 512, page, dst, rows)
        selected = torch.tensor([0, 1, 4, 5, 6], device=device)
        return packed, unpad, main, metadata, selected, dst, rows

    def test_selected_prefix_and_original_suffix_with_tail(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                packed, unpad, main, meta, selected, _, _ = self._inputs(dtype)
                k, v = materialize_pages(
                    selected, main, packed, unpad, meta, 512, 128, 2, 128, 64
                )
                expected_k = torch.zeros_like(k)
                expected_v = torch.zeros_like(v)
                for output, scratch_page in enumerate(selected.tolist(), 1):
                    request, logical_page = divmod(scratch_page, 4)
                    prefix, length = int(meta[0][request]), int(meta[1][request])
                    if logical_page * 128 < prefix:
                        row = int(meta[3][scratch_page])
                        expected_k[output] = main[row, 0].to(torch.bfloat16)
                        expected_v[output] = main[row, 1].to(torch.bfloat16)
                    else:
                        for token in range(128):
                            position = logical_page * 128 + token
                            if position < length:
                                suffix_token = int(meta[2][request]) + position - prefix
                                row = int(unpad[suffix_token])
                                expected_k[output, :, token] = packed[row, :256].view(
                                    2, 128
                                )
                                expected_v[output, :, token] = packed[
                                    row, 256:512
                                ].view(2, 128)
                self.assertTrue(torch.equal(k, expected_k))
                self.assertTrue(torch.equal(v, expected_v))

    def test_cold_suffix_only_and_empty_selection(self):
        packed, unpad, main, _, _, _, _ = self._inputs(torch.float8_e4m3fn)
        device = packed.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        meta = build_source_metadata(
            [0], torch.tensor([8], device=device), 256, 128, empty, None
        )
        selected = torch.tensor([0], device=device)
        k, v = materialize_pages(
            selected, main[:0], packed, unpad, meta, 256, 128, 2, 128, 64
        )
        self.assertTrue(
            torch.equal(
                k[1, :, :8], packed[unpad, :256].view(8, 2, 128).transpose(0, 1)
            )
        )
        self.assertFalse(k[1, :, 8:].any() or v[1, :, 8:].any())
        k, v = materialize_pages(
            empty, main[:0], packed, unpad, meta, 256, 128, 2, 128, 64
        )
        self.assertEqual(k.shape[0], 1)
        self.assertFalse(k.any() or v.any())

    def test_compact_attention_matches_full_pages(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("native sparse FMHA requires SM100")
        import sys

        import nvidia_cutlass_dsl

        python_packages = os.path.join(
            nvidia_cutlass_dsl.__path__[0], "python_packages"
        )
        if os.path.isdir(python_packages) and python_packages not in sys.path:
            sys.path.insert(0, python_packages)
        import fmha_sm100  # noqa: F401

        from rtp_llm.models_py.modules.hybrid.msa_cp_compact import (
            run_compact_attention,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            topk_bt_fused as tbf,
        )

        packed, unpad, main, meta, selected, _, _ = self._inputs(torch.float8_e4m3fn)
        device = packed.device
        full_k, full_v = materialize_pages(
            torch.arange(8, device=device),
            main,
            packed,
            unpad,
            meta,
            512,
            128,
            2,
            128,
            64,
        )
        q = torch.randn(8, 32, 128, dtype=torch.bfloat16, device=device)
        topk_idx = torch.full((2, 8, 16), -1, dtype=torch.int32, device=device)
        topk_idx[:, :5, :2] = torch.tensor([0, 1], device=device)
        topk_idx[:, 5:, :3] = torch.tensor([0, 1, 2], device=device)
        page_ids = torch.tensor([0, 1, 4, 5, 6], device=device, dtype=torch.int32)
        for partial_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(partial_dtype=partial_dtype):
                plan = dict(
                    num_kv_heads=2,
                    qo_segment_lens=torch.tensor([5, 3], dtype=torch.int32),
                    seqused_k=torch.tensor(
                        [133, 259], dtype=torch.int32, device=device
                    ),
                    kv_segment_lens=torch.tensor([133, 259], dtype=torch.int32),
                    causal=True,
                    partial_dtype=partial_dtype,
                    usable_SM_count=-1,
                )
                reference = tbf._sparse_attn_chunked(
                    q,
                    full_k,
                    full_v,
                    topk_idx,
                    page_ids + 1,
                    dict(plan),
                    16,
                    128,
                    128**-0.5,
                    6,
                )
                with patch.dict(os.environ, {"M3_SPARSE_ATTN_CHUNK_SIZE": "6"}):
                    actual = run_compact_attention(
                        q,
                        topk_idx,
                        page_ids,
                        plan,
                        main,
                        packed,
                        unpad,
                        meta,
                        512,
                        128,
                        2,
                        128,
                        64,
                        16,
                    )
                self.assertTrue(torch.equal(actual, reference))

    def test_idx_restore_rank_major_rows(self):
        packed, _, _, _, _, dst, rows = self._inputs(torch.bfloat16)
        idx = torch.arange(3 * 128 * 64, device=packed.device).reshape(3, 128, 64)
        idx = idx.to(torch.bfloat16)
        scratch = torch.full(
            (1024, 1, 64), -7, device=packed.device, dtype=torch.bfloat16
        )
        expected = scratch.clone().view(8, 128, 64)
        expected[dst] = idx[rows]
        restore_idx_pages(idx, dst, scratch, rows)
        self.assertTrue(torch.equal(scratch.view_as(expected), expected))

    def test_persistent_idx_writer_without_main_scratch(self):
        from rtp_llm.models_py.modules.hybrid.msa_attention import _fused_cp_paged_write

        packed, unpad, _, meta, _, _, _ = self._inputs(torch.bfloat16)
        device = packed.device
        write_slots = torch.tensor(
            [128, 129, 130, 131, 132, 768, 769, 770], device=device
        )
        slot_mapping = torch.tensor([0, 1, -1, 3, 4, 128, 129, -1], device=device)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                base = torch.zeros(4, 2, 2, 128, 128, dtype=dtype, device=device)
                scale = torch.zeros(512, 64, dtype=torch.bfloat16, device=device)
                idx = torch.full((1024, 1, 64), -7, dtype=torch.bfloat16, device=device)
                base_ref, scale_ref, idx_ref = base.clone(), scale.clone(), idx.clone()
                k = torch.empty(8, 2, 128, 128, dtype=torch.bfloat16, device=device)
                v = torch.empty_like(k)
                _fused_cp_paged_write(
                    packed,
                    unpad,
                    write_slots,
                    slot_mapping,
                    k,
                    v,
                    idx_ref,
                    base_ref,
                    scale_ref,
                    meta[1],
                    512,
                    256,
                    64,
                    2,
                    128,
                    128,
                    scratch_is_paged=True,
                )
                _fused_cp_paged_write(
                    packed,
                    unpad,
                    write_slots,
                    slot_mapping,
                    None,
                    None,
                    idx,
                    base,
                    scale,
                    meta[1],
                    512,
                    256,
                    64,
                    2,
                    128,
                    128,
                    scratch_is_paged=True,
                    write_main_scratch=False,
                )
                self.assertTrue(torch.equal(base.float(), base_ref.float()))
                self.assertTrue(torch.equal(scale, scale_ref))
                self.assertTrue(torch.equal(idx, idx_ref))


if __name__ == "__main__":
    unittest.main()
