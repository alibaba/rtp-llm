import json
import os
import unittest
from unittest.mock import patch

import torch
import triton

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    quantize_main_index_rows_to_planes,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import topk_bt_fused as op
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.nvfp4_q8_index_score import (
    _prefill_score_kernel,
    q8kv4_prefill_index_score,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
    PrefillScoreHostMetadata,
)


def cpu_oracle(q, packed, scales, cu, lengths, prefixes, offsets, table, blocks):
    """Independent byte decode and CPU FP32 dot; no production dequant helper."""
    q = q.float().cpu()
    packed = packed.cpu()
    scale = scales.view(torch.uint8).cpu()
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    k = torch.empty(packed.shape[0], 128, 128)
    for page in range(packed.shape[0]):
        b = packed[page, 0]
        raw = torch.stack((b & 15, b >> 4), -1).reshape(128, 128).long()
        linear = torch.empty(128, 8, dtype=torch.uint8)
        for token in range(128):
            for group in range(8):
                linear[token, group] = scale[
                    page, 0, group // 4, token % 32, token // 32, group % 4
                ]
        s = linear.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
        # Native ABI dequant is E2M1 -> half multiply -> E4M3 satfinite.
        k[page] = (
            (lut[raw].half() * s.half())
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .float()
        )
    out = torch.full((q.shape[1], q.shape[0], blocks), float("-inf"))
    for seg in range(len(lengths)):
        for row in range(cu[seg], cu[seg + 1]):
            for block in range(blocks):
                index = offsets[seg] + block
                if index >= offsets[seg + 1] or block * 128 >= lengths[seg]:
                    continue
                page = table[index]
                if not 0 <= page < len(k):
                    continue
                visible = min(
                    128,
                    lengths[seg] - block * 128,
                    prefixes[seg] + row - cu[seg] + 1 - block * 128,
                )
                if visible > 0:
                    out[:, row, block] = (q[row] @ k[page, :visible].T).max(-1).values
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQ8KV4PrefillIndexScore(unittest.TestCase):
    def test_query_address_crosses_int32_element_boundary(self):
        q = torch.empty_strided(
            (129, 4, 128),
            (2**24, 128, 1),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        q.copy_(torch.ones(129, 4, 128, device="cuda").to(torch.float8_e4m3fn))
        packed = torch.full((2, 1, 128, 64), 0x22, device="cuda", dtype=torch.uint8)
        scales = torch.ones((2, 1, 2, 32, 4, 4), device="cuda").to(torch.float8_e4m3fn)
        cu = torch.tensor([0, 129], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([256], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([127], device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        pages = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        out = torch.empty(4, 129, 2, device="cuda")
        for tile in (32, 128):

            def run():
                q8kv4_prefill_index_score(
                    q,
                    packed,
                    scales,
                    cu,
                    lengths,
                    prefixes,
                    offsets,
                    pages,
                    out,
                    max_seqlen_q=129,
                    tile_q=tile,
                )

            run()
            torch.testing.assert_close(
                out[:, :, 0], torch.full_like(out[:, :, 0], 128), atol=0, rtol=0
            )
            self.assertTrue(bool(torch.isneginf(out[:, 0, 1]).all()))
            torch.testing.assert_close(
                out[:, 1:, 1], torch.full_like(out[:, 1:, 1], 128), atol=0, rtol=0
            )
            expected = out.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            out.fill_(100)
            graph.replay()
            torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_dynamic_geometry_reuses_compiled_variant(self):
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            nvfp4_q8_index_score as score_op,
        )

        dynamic = {
            "TOTAL_Q",
            "MAX_PAGES",
            "PHYSICAL_PAGES",
            "TABLE_SIZE",
            "OSTRIDE_H",
            "OSTRIDE_Q",
        }
        kernel = score_op._prefill_two_page_score_kernel
        params = {param.name: param for param in kernel.params}
        self.assertEqual(set(kernel.do_not_specialize), dynamic)
        for name in dynamic:
            self.assertFalse(params[name].is_constexpr)
            self.assertTrue(params[name].do_not_specialize)

        torch.manual_seed(2026100553)
        previous = None
        for total, blocks, pages in ((258, 3, 7), (274, 5, 13), (300, 7, 17)):
            q = torch.randn(total, 4, 128, device="cuda").to(torch.float8_e4m3fn)
            packed = torch.randint(
                0, 256, (pages, 1, 128, 64), device="cuda", dtype=torch.uint8
            )
            scales = torch.full(
                (pages, 1, 2, 32, 4, 4),
                0.5,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            cu = torch.tensor([0, 129, total], device="cuda", dtype=torch.int32)
            lengths = torch.full(
                (2,), blocks * 128 - 1, device="cuda", dtype=torch.int32
            )
            prefix = lengths - torch.tensor(
                [129, total - 129], device="cuda", dtype=torch.int32
            )
            offsets = torch.tensor(
                [0, blocks, blocks * 2], device="cuda", dtype=torch.int32
            )
            table = torch.arange(blocks * 2, device="cuda", dtype=torch.int32)
            table[1] = -1
            buffers = [
                torch.full((4, total + 7, blocks + 3), 123.0, device="cuda")
                for _ in range(2)
            ]
            expected, actual = (buffer[:, :total, :blocks] for buffer in buffers)
            # Independent one-page reduction preserves the original dot shape.
            _prefill_score_kernel[(triton.cdiv(max(129, total - 129), 128), 8, blocks)](
                q,
                packed,
                scales.view(torch.uint8),
                cu,
                lengths,
                prefix,
                offsets,
                table,
                expected,
                total,
                4,
                blocks,
                pages,
                table.numel(),
                q.stride(0),
                q.stride(1),
                packed.stride(0),
                scales.stride(0),
                expected.stride(0),
                expected.stride(1),
                128,
                num_warps=4,
            )
            q8kv4_prefill_index_score(
                q,
                packed,
                scales,
                cu,
                lengths,
                prefix,
                offsets,
                table,
                actual,
                max_seqlen_q=max(129, total - 129),
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for buffer in buffers:
                self.assertTrue(torch.all(buffer[:, total:, :] == 123.0))
                self.assertTrue(torch.all(buffer[:, :total, blocks:] == 123.0))
            variants = len(kernel.device_caches[torch.cuda.current_device()][0])
            if previous is not None:
                self.assertEqual(variants, previous)
            previous = variants

    def test_two_page_production_topk_cross_segment_chunks(self):
        torch.manual_seed(2026100127)
        pages, total = 5, 274
        k = torch.randn(pages * 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        data = [
            torch.empty(pages, 1, 128, 64, device="cuda", dtype=torch.uint8)
            for _ in range(3)
        ]
        scales = [
            torch.empty(pages, 1, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
            for _ in range(3)
        ]
        quantize_main_index_rows_to_planes(
            k,
            k,
            k,
            torch.arange(pages * 128, device="cuda"),
            data[0],
            scales[0],
            data[1],
            scales[1],
            data[2],
            scales[2],
        )
        mma = scales[2].view(pages, 1, 2, 32, 4, 4)
        q = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 137, total], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([639, 639], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([502, 502], device="cuda", dtype=torch.int32)
        table = torch.tensor(
            [4, 2, 0, 3, 1, 1, 3, 4, 0, 2], device="cuda", dtype=torch.int32
        )
        host = PrefillScoreHostMetadata(
            query_lens=(137, 137),
            seq_lens=(639, 639),
            prefix_lens=(502, 502),
            slot_ids=(0, 1),
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            nvfp4_q8_index_score as score_op,
        )

        def old_score(
            q8,
            packed,
            scale,
            cuq,
            lens,
            prefix,
            offsets,
            indices,
            out,
            *,
            max_seqlen_q,
            tile_q=None
        ):
            tile = tile_q or (128 if max_seqlen_q >= 128 else 32)
            _prefill_score_kernel[
                (triton.cdiv(max_seqlen_q, tile), (cuq.numel() - 1) * 4, out.shape[2])
            ](
                q8,
                packed,
                scale.view(torch.uint8),
                cuq,
                lens,
                prefix,
                offsets,
                indices,
                out,
                q8.shape[0],
                4,
                out.shape[2],
                pages,
                indices.numel(),
                q8.stride(0),
                q8.stride(1),
                packed.stride(0),
                scale.stride(0),
                out.stride(0),
                out.stride(1),
                tile,
                num_warps=4,
            )
            return out

        def run():
            return op.flash_prefill_topk_to_block_tables_fp4(
                q,
                data[2],
                mma,
                cu,
                lengths,
                prefixes,
                max_seqlen_q=137,
                max_seqlen_k=639,
                block_size_k=128,
                topk=2,
                local_blocks=0,
                num_pages=pages,
                index_score_plan={"_fp4_host_metadata": host},
                kv_indices=table,
                emit_block_table=False,
            )[2]

        # 129-row chunks exercise both the new long-segment dispatch and the
        # short-segment tail after a chunk crosses the request boundary.
        launches = []
        original_kernel = score_op._prefill_two_page_score_kernel

        class KernelSpy:
            def __getitem__(self, grid):
                launches.append(grid)
                return original_kernel[grid]

        with patch.dict(os.environ, {"M3_MSA_INDEX_SCORE_CHUNK_ROWS": "129"}):
            with patch.object(
                score_op, "q8kv4_prefill_index_score", side_effect=old_score
            ):
                expected = run()
            with patch.object(score_op, "_prefill_two_page_score_kernel", KernelSpy()):
                actual = run()
                self.assertTrue(launches)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_two_page_dynamic_ownership_padded_graph(self):
        """Long-query production wrapper versus the original page reduction."""
        torch.manual_seed(2026100126)
        pages, total = 5, 274
        k = torch.randn(pages * 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        data = [
            torch.empty(pages, 1, 128, 64, device="cuda", dtype=torch.uint8)
            for _ in range(3)
        ]
        scales = [
            torch.empty(pages, 1, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
            for _ in range(3)
        ]
        quantize_main_index_rows_to_planes(
            k,
            k,
            k,
            torch.arange(pages * 128, device="cuda"),
            data[0],
            scales[0],
            data[1],
            scales[1],
            data[2],
            scales[2],
        )
        mma = scales[2].view(pages, 1, 2, 32, 4, 4)
        # Poison a future row: masking must suppress its NaN score.
        mma[-1, 0, :, 31, 3, :].view(torch.uint8).fill_(127)
        q = torch.randn(total, 4, 128, device="cuda").to(torch.float8_e4m3fn)
        cu = torch.zeros(6, device="cuda", dtype=torch.int32)
        offsets = torch.zeros_like(cu)
        lengths = torch.zeros(5, device="cuda", dtype=torch.int32)
        prefixes = torch.zeros_like(lengths)
        table = torch.full((25,), -1, device="cuda", dtype=torch.int32)
        backing = [torch.full((4, total, 8), 12345.0, device="cuda") for _ in range(2)]
        outputs = [x[:, :, :pages] for x in backing]

        def metadata(widths, runs):
            cumulative, page_cumulative, flat = [0], [0], []
            lens, prefix = [], []
            for width, run in zip(widths, runs):
                cumulative.append(cumulative[-1] + width)
                page_cumulative.append(page_cumulative[-1] + run)
                flat.extend(list(range(min(run, pages))) + [-1] * max(0, run - pages))
                length = min(run, pages) * 128 - 1
                lens.append(length)
                prefix.append(length - width)
            for dst, values in (
                (cu, cumulative),
                (offsets, page_cumulative),
                (table, flat),
                (lengths, lens),
                (prefixes, prefix),
            ):
                dst.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))

        def baseline():
            out = outputs[0]
            _prefill_score_kernel[(triton.cdiv(129, 128), 5 * 4, pages)](
                q,
                data[2],
                mma.view(torch.uint8),
                cu,
                lengths,
                prefixes,
                offsets,
                table,
                out,
                total,
                4,
                pages,
                pages,
                table.numel(),
                q.stride(0),
                q.stride(1),
                data[2].stride(0),
                mma.stride(0),
                out.stride(0),
                out.stride(1),
                128,
                num_warps=4,
            )

        def candidate():
            return q8kv4_prefill_index_score(
                q,
                data[2],
                mma,
                cu,
                lengths,
                prefixes,
                offsets,
                table,
                outputs[1],
                max_seqlen_q=129,
            )

        metadata([1, 0, 17, 127, 129], [5] * 5)
        baseline()
        self.assertIs(candidate(), outputs[1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate()
        for widths, runs in (
            ([1, 0, 17, 127, 129], [5] * 5),
            ([129, 127, 0, 17, 1], [4, 5, 3, 5, 8]),
            ([17, 129, 1, 0, 127], [3, 5, 5, 8, 4]),
        ):
            metadata(widths, runs)
            baseline()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertFalse(torch.isnan(outputs[1]).any().item())
            for buffer in backing:
                self.assertTrue(torch.all(buffer[:, :, pages:] == 12345.0).item())
        before = torch.cuda.memory_allocated()
        candidate()
        torch.cuda.synchronize()
        self.assertEqual(before, torch.cuda.memory_allocated())

    def test_writer_permutation_ragged_graph(self):
        torch.manual_seed(1001)
        pages = 7
        k = torch.randn(pages * 128, 4, 128, device="cuda", dtype=torch.bfloat16)
        ik = torch.randn(pages * 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        kp = torch.empty(pages, 4, 128, 64, device="cuda", dtype=torch.uint8)
        vp = torch.empty_like(kp)
        ks = torch.empty(pages, 4, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
        vs = torch.empty_like(ks)
        ip = torch.empty(pages, 1, 128, 64, device="cuda", dtype=torch.uint8)
        isc = torch.empty(pages, 1, 128, 8, device="cuda", dtype=torch.float8_e4m3fn)
        quantize_main_index_rows_to_planes(
            k, k, ik, torch.arange(pages * 128, device="cuda"), kp, ks, vp, vs, ip, isc
        )
        isc = isc.view(pages, 1, 2, 32, 4, 4)
        q = torch.randn(75, 4, 128, device="cuda").to(torch.float8_e4m3fn)
        cu = torch.tensor([0, 37, 37, 75], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([259, 0, 181], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([222, 0, 143], device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0, 3, 3, 5], device="cuda", dtype=torch.int32)
        table = torch.tensor([5, 1, 4, 2, 6], device="cuda", dtype=torch.int32)
        out = torch.empty(4, 75, 4, device="cuda")
        tiles = (16, 32, 64, 128)

        def run(tile=128):
            return q8kv4_prefill_index_score(
                q,
                ip,
                isc,
                cu,
                lengths,
                prefixes,
                offsets,
                table,
                out,
                max_seqlen_q=75,
                tile_q=tile,
            )

        def check():
            ref = cpu_oracle(
                q,
                ip,
                isc,
                cu.tolist(),
                lengths.tolist(),
                prefixes.tolist(),
                offsets.tolist(),
                table.tolist(),
                4,
            )
            torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-6)
            return float(
                (out.cpu()[torch.isfinite(ref)] - ref[torch.isfinite(ref)]).abs().max()
            )

        errors = []
        for tile in tiles:
            run(tile)
            errors.append(check())
        graphs = {}
        for tile in tiles:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(tile)
            graphs[tile] = graph
        for step in range(3):
            q.copy_(torch.randn_like(q.float()).to(q.dtype))
            ik.normal_()
            quantize_main_index_rows_to_planes(
                k,
                k,
                ik,
                torch.arange(pages * 128, device="cuda"),
                kp,
                ks,
                vp,
                vs,
                ip,
                isc.view(pages, 1, 128, 8),
            )
            table.copy_(table.roll(1))
            lengths.copy_(
                torch.tensor(
                    [129 + step, 0, 180 - step], device="cuda", dtype=torch.int32
                )
            )
            prefixes.copy_(
                torch.tensor(
                    [90 + step, 0, 139 - step], device="cuda", dtype=torch.int32
                )
            )
            cu.copy_(
                torch.tensor(
                    [0, 40 - step, 40 - step, 75], device="cuda", dtype=torch.int32
                )
            )
            for graph in graphs.values():
                graph.replay()
                errors.append(check())
        lengths.zero_()
        for graph in graphs.values():
            out.fill_(123)
            graph.replay()
            self.assertTrue(torch.isneginf(out).all().item())

        # Exercise the production chunk builder and RTP TopK consumer, including
        # a segment split across chunks and a permuted physical page table.
        cu.copy_(torch.tensor([0, 37, 37, 75], device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor([259, 0, 181], device="cuda", dtype=torch.int32))
        prefixes.copy_(torch.tensor([222, 0, 143], device="cuda", dtype=torch.int32))
        q_bf16 = torch.randn(75, 4, 128, device="cuda", dtype=torch.bfloat16)
        q8 = q_bf16.to(torch.float8_e4m3fn)
        score_ref = cpu_oracle(
            q8,
            ip,
            isc,
            cu.tolist(),
            lengths.tolist(),
            prefixes.tolist(),
            offsets.tolist(),
            table.tolist(),
            4,
        )
        host = PrefillScoreHostMetadata(
            query_lens=(37, 0, 38),
            seq_lens=(259, 0, 181),
            prefix_lens=(222, 0, 143),
            slot_ids=(0, 1, 2),
        )
        q34_cu = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        q34_len = torch.tensor([259], device="cuda", dtype=torch.int32)
        q34_prefix = torch.tensor([256], device="cuda", dtype=torch.int32)
        q34_pages = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        q34_score = torch.empty(4, 3, 3, device="cuda", dtype=torch.float32)
        q8kv4_prefill_index_score(
            q8[34:37],
            ip,
            isc,
            q34_cu,
            q34_len,
            q34_prefix,
            q34_pages,
            table[:3],
            q34_score,
            max_seqlen_q=3,
        )
        q34_ref = cpu_oracle(
            q8[34:37], ip, isc, [0, 3], [259], [256], [0, 3], table[:3].tolist(), 3
        )
        torch.testing.assert_close(q34_score.cpu(), q34_ref, rtol=1e-5, atol=1e-4)
        with patch.dict(os.environ, {"M3_MSA_INDEX_SCORE_CHUNK_ROWS": "17"}):
            _, _, selected = op.flash_prefill_topk_to_block_tables_fp4(
                q_bf16,
                ip,
                isc,
                cu,
                lengths,
                prefixes,
                max_seqlen_q=38,
                max_seqlen_k=259,
                block_size_k=128,
                topk=2,
                local_blocks=0,
                num_pages=pages,
                index_score_plan={"_fp4_host_metadata": host},
                kv_indices=table,
                emit_block_table=False,
            )
        expected = score_ref.topk(2, dim=-1).indices.to(
            device="cuda", dtype=torch.int32
        )
        actual_sets = torch.sort(selected, dim=-1).values
        expected_sets = torch.sort(expected, dim=-1).values
        torch.testing.assert_close(actual_sets, expected_sets, rtol=0, atol=0)
        # A CP producer-marked plan may reuse metadata across layers. Q/score/
        # TopK still execute, and rebuilding the table must invalidate the plan.
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import score_chunk

        shared_plan = {"_fp4_host_metadata": host}
        score_chunk.publish_fp4_prefill_metadata_table(shared_plan, table)
        with patch.dict(os.environ, {"M3_MSA_INDEX_SCORE_CHUNK_ROWS": "17"}):
            with patch.object(
                score_chunk,
                "build_prefill_score_chunks",
                wraps=score_chunk.build_prefill_score_chunks,
            ) as builder:
                for _ in range(2):
                    _, _, cached_topk = op.flash_prefill_topk_to_block_tables_fp4(
                        q_bf16,
                        ip,
                        isc,
                        cu,
                        lengths,
                        prefixes,
                        max_seqlen_q=38,
                        max_seqlen_k=259,
                        block_size_k=128,
                        topk=2,
                        local_blocks=0,
                        num_pages=pages,
                        index_score_plan=shared_plan,
                        kv_indices=table,
                        emit_block_table=False,
                    )
                    torch.testing.assert_close(cached_topk, selected, rtol=0, atol=0)
                self.assertEqual(builder.call_count, 1)
                changed_table = table.roll(1).contiguous()
                score_chunk.publish_fp4_prefill_metadata_table(
                    shared_plan, changed_table
                )
                _, _, changed_topk = op.flash_prefill_topk_to_block_tables_fp4(
                    q_bf16,
                    ip,
                    isc,
                    cu,
                    lengths,
                    prefixes,
                    max_seqlen_q=38,
                    max_seqlen_k=259,
                    block_size_k=128,
                    topk=2,
                    local_blocks=0,
                    num_pages=pages,
                    index_score_plan=shared_plan,
                    kv_indices=changed_table,
                    emit_block_table=False,
                )
                self.assertEqual(builder.call_count, 2)
                changed_score = cpu_oracle(
                    q8,
                    ip,
                    isc,
                    cu.tolist(),
                    lengths.tolist(),
                    prefixes.tolist(),
                    offsets.tolist(),
                    changed_table.tolist(),
                    4,
                )
                expected_changed = changed_score.topk(2, dim=-1).indices.to(
                    device="cuda", dtype=torch.int32
                )
                torch.testing.assert_close(
                    torch.sort(changed_topk, dim=-1).values,
                    torch.sort(expected_changed, dim=-1).values,
                    rtol=0,
                    atol=0,
                )
        # Same-size persistent output launch: no allocator allocation.
        before = torch.cuda.memory_allocated()
        run()
        torch.cuda.synchronize()
        self.assertEqual(before, torch.cuda.memory_allocated())
        ms = []
        lengths.copy_(torch.tensor([259, 0, 181], device="cuda", dtype=torch.int32))
        for tile in tiles:
            run(tile)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(30):
                run(tile)
            end.record()
            end.synchronize()
            ms.append(start.elapsed_time(end) / 30)
        report = {
            "cpu_fp32_oracle_max_abs": max(errors),
            "tile_q_ms": dict(zip(tiles, ms)),
            "geometry": {
                "physical_pages": 7,
                "q_tokens": 75,
                "heads": 4,
                "dim": 128,
                "segments": 3,
                "maxpages": 4,
            },
            "graph_updates": 3,
            "production_topk_chunk_rows": 17,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "boundary": "Tiny operator/TopK integration correctness case; not production performance or model quality.",
        }
        print(json.dumps(report), flush=True)
        path = os.environ.get("Q8KV4_PREFILL_TEST_JSON")
        if path:
            with open(path, "w") as f:
                json.dump(report, f, indent=2)

    def test_empty_and_validation(self):
        q = torch.empty(0, 4, 128, device="cuda", dtype=torch.float8_e4m3fn)
        p = torch.empty(0, 1, 128, 64, device="cuda", dtype=torch.uint8)
        s = torch.empty(0, 1, 2, 32, 4, 4, device="cuda", dtype=torch.float8_e4m3fn)
        cu = torch.zeros(1, device="cuda", dtype=torch.int32)
        empty = cu[:0]
        out = torch.empty(4, 0, 0, device="cuda")
        self.assertIs(
            q8kv4_prefill_index_score(
                q, p, s, cu, empty, empty, cu, empty, out, max_seqlen_q=0
            ),
            out,
        )
        with self.assertRaises(ValueError):
            q8kv4_prefill_index_score(
                q.float(), p, s, cu, empty, empty, cu, empty, out, max_seqlen_q=0
            )


if __name__ == "__main__":
    unittest.main()
