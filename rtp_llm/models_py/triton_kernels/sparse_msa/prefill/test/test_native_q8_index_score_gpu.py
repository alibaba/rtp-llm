"""Runtime-shape helper regressions against independent tensor references."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
    native_q8_index_score as op,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NativeIndexRuntimeShapeTest(unittest.TestCase):
    def test_real_onlyscore_immutable_matches_mutable_scores_and_topk(self):
        """Actual fmha_sm100 planner/OnlyScore and production TopK, no mocks."""
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import topk_bt_fused as wrapper

        device = torch.device("cuda:0")
        pages, rows, blocks = 64, 4096, 64
        vec = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
        chunks = []
        for index, prefix in enumerate((4096, 2048)):
            table = torch.arange(blocks, device=device, dtype=torch.int32)
            table = (table * (index + 3)) % pages
            table[1], table[2], table[-1] = table[0], -1, pages
            chunks.append(SimpleNamespace(
                q_start=index * rows, q_end=(index + 1) * rows,
                host_metadata=SimpleNamespace(query_lens=(rows,), seq_lens=(8192,), prefix_lens=(prefix,)),
                cu_seqlens=vec([0, rows]), seq_lens=vec([8192]), prefix_lens=vec([prefix]),
                kv_indices=table, max_seqlen_q=rows, max_seqlen_k=8192,
            ))
        chunks = tuple(chunks)
        cached = op.NativeIndexWorkspace(chunks, pages, 4, device, compact=True, immutable_tables=True)
        mutable = op.NativeIndexWorkspace(chunks, pages, 4, device, compact=True)
        table_values = tuple(t.clone() for t in cached.safe_tables)
        table_ptrs = tuple(t.data_ptr() for t in cached.safe_tables)
        torch.manual_seed(20261009)
        packed = torch.randint(0, 256, (pages, 1, 128, 64), device=device, dtype=torch.uint8)
        scales = torch.full((pages, 1, 2, 32, 4, 4), 40, device=device, dtype=torch.uint8)
        q = (torch.randn(rows, 4, 128, device=device) * 0.1).to(torch.float8_e4m3fn)
        offsets = vec([0, blocks])

        def topk(score, chunk):
            bt, lengths, indices = wrapper._allocate_topk_outputs(rows, 4, 16, device, False, overwrite_topk=True)
            wrapper._launch_topk_to_block_table(
                rows, 1, 4, blocks, score, bt, lengths, indices, 1, 128,
                chunk.cu_seqlens, chunk.cu_seqlens, chunk.prefix_lens,
                16, 0, 0, pages, *score.stride(), *bt.stride(), *indices.stride(),
                NKV=4, MASK_INIT=False, MASK_LOCAL=False,
                EMIT_BLOCK_TABLE=False, EMIT_TOPK_IDX=True,
            )
            return indices

        previous = None
        for layer in range(2):
            if layer:
                packed.bitwise_xor_(255)
                scales.fill_(48)
            cached.stage(packed, scales)
            mutable.stage(packed, scales)
            layer_scores = []
            for index, chunk in enumerate(chunks):
                outputs = []
                for workspace in (cached, mutable):
                    output = torch.empty((4, rows, blocks), device=device)
                    outputs.append(workspace.score(index, q, offsets, output))
                self.assertTrue(torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32)))
                self.assertTrue(torch.isfinite(outputs[0]).any())
                self.assertTrue(torch.equal(topk(outputs[0], chunk), topk(outputs[1], chunk)))
                self.assertTrue(torch.isneginf(outputs[0][:, :, 2]).all())
                self.assertTrue(torch.isneginf(outputs[0][:, :, -1]).all())
                layer_scores.append(outputs[0])
            if previous is not None:
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(previous, layer_scores)))
            previous = layer_scores
            self.assertEqual(table_ptrs, tuple(t.data_ptr() for t in cached.safe_tables))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(table_values, cached.safe_tables)))

    def test_immutable_workspace_staging_matches_mutable_across_layers(self):
        """Real constructor/ID/staging kernels; OnlyScore itself is mocked."""
        device = torch.device("cuda:0")
        pages, rows = 7, 2048
        vec = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
        chunks = tuple(
            SimpleNamespace(
                q_start=index * rows, q_end=(index + 1) * rows,
                host_metadata=SimpleNamespace(
                    query_lens=(rows,), seq_lens=(len(ids) * 128,), prefix_lens=(0,),
                ),
                cu_seqlens=vec([0, rows]), seq_lens=vec([len(ids) * 128]),
                prefix_lens=vec([0]), kv_indices=vec(ids),
                max_seqlen_q=rows, max_seqlen_k=len(ids) * 128,
            )
            for index, ids in enumerate(([1, 1, -1, pages], [6, 0, 6]))
        )

        def planner(*args, **kwargs):
            return {"max_k_tiles": 128, "orig_num_qo_heads": 4,
                    "MM-SA-Nv": False, "num_kv_splits": 1}

        def score(*args, **kwargs):
            return None, kwargs["max_score"]

        with patch.object(op, "_native_api", return_value=(planner, score)):
            cached = op.NativeIndexWorkspace(chunks, pages, 4, device, compact=True, immutable_tables=True)
            mutable = op.NativeIndexWorkspace(chunks, pages, 4, device, compact=True)
        table_ptrs = tuple(t.data_ptr() for t in cached.safe_tables)
        table_values = tuple(t.clone() for t in cached.safe_tables)
        packed_storage = torch.randint(0, 256, (pages, 8448), device=device, dtype=torch.uint8)
        scale_storage = torch.full((pages, 1280), 56, device=device, dtype=torch.uint8)
        packed = torch.as_strided(packed_storage, (pages, 1, 128, 64), (8448, 8192, 64, 1))
        scales = torch.as_strided(scale_storage, (pages, 1, 2, 32, 4, 4), (1280, 1024, 512, 16, 4, 1))
        q = torch.zeros((rows, 4, 128), device=device, dtype=torch.float8_e4m3fn)
        full = torch.zeros((pages + 1, 1, 128, 128), device=device, dtype=torch.float8_e4m3fn)
        previous = None
        with patch.object(cached, "_prepare_table", side_effect=AssertionError("immutable IDs rebuilt")):
            for layer in range(2):
                if layer:
                    packed_storage.bitwise_xor_(255)
                    scale_storage.fill_(64)
                op._stage_index_pages[(pages,)](
                    packed, scales, full, pages, packed.stride(0), scales.stride(0), num_warps=4,
                )
                cached.stage(packed, scales)
                mutable.stage(packed, scales)
                layer_readouts = []
                for index, chunk in enumerate(chunks):
                    output = torch.empty((4, rows, chunk.kv_indices.numel()), device=device)
                    offsets = vec([0, chunk.kv_indices.numel()])
                    readouts = []
                    for workspace in (cached, mutable):
                        workspace.score(index, q, offsets, output)
                        readouts.append(workspace.staged[workspace.safe_tables[index].long()].view(torch.uint8))
                    table = chunk.kv_indices
                    ids = torch.where((table >= 0) & (table < pages), table, pages)
                    expected = full[ids.long()].view(torch.uint8)
                    self.assertTrue(torch.equal(readouts[0], expected))
                    self.assertTrue(torch.equal(readouts[1], expected))
                    self.assertTrue(torch.equal(cached.staged[cached.stage_pages].view(torch.uint8), torch.zeros_like(full[pages].view(torch.uint8))))
                    layer_readouts.append(readouts[0])
                if previous is not None:
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(previous, layer_readouts)))
                previous = layer_readouts
                self.assertEqual(table_ptrs, tuple(t.data_ptr() for t in cached.safe_tables))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(table_values, cached.safe_tables)))

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
