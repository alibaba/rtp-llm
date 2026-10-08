"""Runtime-shape helper regressions against independent tensor references."""

import unittest

import torch
import triton

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
    native_q8_index_score as op,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NativeIndexRuntimeShapeTest(unittest.TestCase):
    def test_compact_map_rebuild_and_graph_replay(self):
        device = torch.device("cuda:0")
        pages, entries = 31, 263
        packed = torch.randint(0, 256, (pages, 8448), device=device, dtype=torch.uint8)
        scales = torch.full((pages, 1280), 56, device=device, dtype=torch.uint8)
        table = torch.arange(entries, device=device, dtype=torch.int32) % pages
        table[0], table[-1] = -1, pages
        mapping = torch.empty(pages + 1, device=device, dtype=torch.int32)
        page_list = torch.empty(pages, device=device, dtype=torch.int32)
        count = torch.empty(1, device=device, dtype=torch.int32)
        safe = torch.empty_like(table)
        compact = torch.zeros(
            (pages + 1, 128, 128), device=device, dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        full = torch.empty_like(compact)

        def run():
            mapping.fill_(-1)
            count.zero_()
            grid = (triton.cdiv(entries, 256),)
            op._claim_index_pages[grid](
                table, mapping, page_list, count, entries, pages, 256
            )
            op._stage_compact_index_pages[(pages,)](
                packed,
                scales,
                compact,
                page_list,
                count,
                8448,
                1280,
                pages,
                num_warps=4,
            )
            op._remap_index_pages[grid](
                table, mapping, safe, entries, pages, pages, 256
            )

        def check():
            op._stage_index_pages[(pages,)](
                packed, scales, full, pages, 8448, 1280, num_warps=4
            )
            full[pages].zero_()
            expected = torch.where((table >= 0) & (table < pages), table, pages)
            self.assertTrue(
                torch.equal(
                    compact[safe.long()].view(torch.uint8),
                    full[expected.long()].view(torch.uint8),
                )
            )
            self.assertEqual(
                count.item(), table[(table >= 0) & (table < pages)].unique().numel()
            )

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for mode in ("changed", "invalid", "one_page"):
            if mode == "changed":
                table.copy_((table * 13 + 7) % pages)
                packed.bitwise_xor_(255)
            elif mode == "invalid":
                table.fill_(-1)
            else:
                table.fill_(0)
            graph.replay()
            check()

    def test_shape_changes_preserve_packed_reader_and_score_layout(self):
        device = torch.device("cuda:0")
        torch.manual_seed(20261007)
        magnitudes = torch.tensor(
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
            ],
            device=device,
        )
        for pages, rows, blocks, padding in (
            (16, 63, 17, 0),
            (17, 80, 26, 16),
            (19, 114, 44, 0),
            (23, 182, 80, 16),
        ):
            with self.subTest(pages=pages, rows=rows, blocks=blocks, padding=padding):
                ks, ss = 8192 + padding, 1024 + padding
                packed = torch.randint(
                    0, 256, (pages, ks), device=device, dtype=torch.uint8
                )
                # All positive finite E4M3 encodings, including subnormals and
                # saturation at scale=448. NaN encodings are not writer output.
                scales = torch.randint(
                    1, 127, (pages, ss), device=device, dtype=torch.uint8
                )
                scales[:, 0], scales[:, 1] = 1, 126
                output = torch.empty(
                    (pages, 128, 128), device=device, dtype=torch.float8_e4m3fn
                )
                op._stage_index_pages[(pages,)](
                    packed, scales, output, pages, ks, ss, num_warps=4
                )
                token = torch.arange(128, device=device)[:, None]
                group = torch.arange(8, device=device)[None, :]
                scale_offset = (
                    (group // 4) * 512
                    + (token % 32) * 16
                    + (token // 32) * 4
                    + group % 4
                )
                scale = (
                    scales[:, scale_offset]
                    .contiguous()
                    .view(torch.float8_e4m3fn)
                    .to(torch.float16)
                )
                bits = packed[:, :8192].reshape(pages, 128, 64)
                codes = (
                    torch.stack((bits & 15, bits >> 4), dim=-1)
                    .reshape(pages, 128, 128)
                    .long()
                )
                expected = (
                    (
                        magnitudes[codes].to(torch.float16)
                        * scale.repeat_interleave(16, dim=-1)
                    )
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )

                table = torch.arange(blocks, device=device, dtype=torch.int32) % pages
                table[0], table[-1] = -1, pages
                safe = torch.empty_like(table)
                op._safe_index_pages[(triton.cdiv(blocks, 256),)](
                    table, safe, blocks, pages, 256, num_warps=4
                )
                self.assertTrue(
                    torch.equal(
                        safe, torch.where((table >= 0) & (table < pages), table, pages)
                    )
                )

                tiles = triton.cdiv(blocks, 128) * 128
                source = torch.randn((4, tiles, rows), device=device)
                destination = torch.full(
                    (4, rows, blocks + padding), 12345.0, device=device
                )
                vec = lambda values: torch.tensor(
                    values, dtype=torch.int32, device=device
                )
                cu, lens, prefix, offsets = (
                    vec([0, rows]),
                    vec([blocks * 128 - 3]),
                    vec([7]),
                    vec([0, blocks]),
                )
                grid = (triton.cdiv(rows, 32), 1, 4 * triton.cdiv(blocks, 32))
                op._copy_index_scores[grid](
                    source,
                    destination,
                    cu,
                    lens,
                    prefix,
                    offsets,
                    table,
                    rows,
                    blocks,
                    pages,
                    *source.stride(),
                    *destination.stride()[:2],
                    num_warps=4
                )
                logical = torch.arange(blocks, device=device)[None, :]
                visible = (
                    (logical * 128 <= 7 + torch.arange(rows, device=device)[:, None])
                    & (logical * 128 < blocks * 128 - 3)
                    & (table[None, :] >= 0)
                    & (table[None, :] < pages)
                )
                reference = (
                    source[:, :blocks, :]
                    .permute(0, 2, 1)
                    .masked_fill(~visible, float("-inf"))
                )
                self.assertTrue(torch.equal(destination[:, :, :blocks], reference))
                self.assertTrue(torch.all(destination[:, :, blocks:] == 12345.0))


if __name__ == "__main__":
    unittest.main()
