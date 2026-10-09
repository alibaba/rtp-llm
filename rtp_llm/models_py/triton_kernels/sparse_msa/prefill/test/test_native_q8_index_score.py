"""CPU-only geometry/cache ABI and mocked OnlyScore ownership tests."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
    native_q8_index_score as op,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import topk_bt_fused as wrapper
from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.score_chunk import (
    PrefillScoreHostMetadata,
)


def chunk(rows=4096, length=4096, prefix=0, start=0):
    pages = (length + 127) // 128
    return SimpleNamespace(
        q_start=start,
        q_end=start + rows,
        host_metadata=SimpleNamespace(
            query_lens=(rows,), seq_lens=(length,), prefix_lens=(prefix,)
        ),
        cu_seqlens=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens=torch.tensor([length], dtype=torch.int32),
        prefix_lens=torch.tensor([prefix], dtype=torch.int32),
        kv_indices=torch.arange(pages, dtype=torch.int32),
        max_seqlen_q=rows,
        max_seqlen_k=length,
    )


class LaunchMock:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        return self.function


class NativeIndexHostTest(unittest.TestCase):
    def supported(self, chunks, pages=32, heads=4, total=4096, rows=4096, blocks=32):
        return op.supported_native_index_workspace(
            chunks, pages, heads, total, rows, blocks
        )

    def test_eligibility_limits(self):
        c = chunk()
        self.assertTrue(self.supported((c,)))
        self.assertFalse(self.supported((c,), heads=8))
        self.assertFalse(self.supported((c,), total=4095))
        self.assertFalse(self.supported((c,), rows=16385))
        self.assertFalse(self.supported((c,), blocks=1025))
        self.assertFalse(self.supported((c,), pages=0))
        self.assertFalse(self.supported(()))

    def test_missing_host_metadata_never_reads_device_values(self):
        c = chunk()
        c.host_metadata = None
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("device sync")
        ), patch.object(
            torch.Tensor, "item", side_effect=AssertionError("device sync")
        ), patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("device sync")
        ):
            self.assertFalse(self.supported((c,)))

    def test_budget_rounds_native_tiles_and_counts_all_safe_tables(self):
        chunks = (chunk(4096, 16512), chunk(2048, 8192, start=4096))
        expected = 65 * 16384 + 4 * 256 * 4096 * 4 + (129 + 64) * 4
        self.assertEqual(op.native_index_workspace_bytes(chunks, 64, 4), expected)
        self.assertTrue(self.supported(chunks, pages=64, total=6144, blocks=129))

    def test_physical_pool_highwater_rejects_over_budget(self):
        c = chunk()
        overhead = op.native_index_workspace_bytes((c,), 1, 4) - 2 * 16384
        largest_pages = (op._WORKSPACE_LIMIT - overhead) // 16384 - 1
        self.assertTrue(self.supported((c,), pages=largest_pages))
        self.assertFalse(self.supported((c,), pages=largest_pages + 1))

    def test_compact_admission_bounds_without_reading_page_contents(self):
        c = chunk(rows=16384, length=90000)
        args = ((c,), 28160, 4, 16384, 16384, 704)
        self.assertFalse(op.supported_native_index_workspace(*args))
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("sync")
        ), patch.object(torch.Tensor, "tolist", side_effect=AssertionError("sync")):
            self.assertTrue(op.supported_native_index_workspace(*args, compact=True))
        self.assertLess(
            op.native_index_workspace_bytes((c,), 28160, 4, compact=True),
            op._WORKSPACE_LIMIT,
        )
        # Compaction changes staging capacity only, never expands the planner's
        # validated logical-page/long-context geometry.
        c = chunk(rows=16384, length=1048576)
        self.assertFalse(
            op.supported_native_index_workspace(
                (c,), 8192, 4, 16384, 16384, 8192, compact=True
            )
        )

    def test_largest_accepted_chunk_and_logical_page_limits(self):
        c = chunk(16384, 131072)
        self.assertTrue(
            self.supported((c,), pages=1024, total=16384, rows=16384, blocks=1024)
        )
        self.assertFalse(
            self.supported(
                (chunk(16385, 131072),), total=16385, rows=16385, blocks=1024
            )
        )

    def test_partition_and_causal_metadata_are_checked(self):
        self.assertFalse(self.supported((chunk(start=1),)))
        self.assertFalse(self.supported((chunk(prefix=-1),)))
        c = chunk()
        c.max_seqlen_k = 4097
        self.assertFalse(self.supported((c,)))
        c = chunk()
        c.kv_indices = c.kv_indices.to(torch.int64)
        self.assertFalse(self.supported((c,)))
        c = chunk()
        c.host_metadata.prefix_lens = (2**31,)
        self.assertFalse(self.supported((c,)))

    def test_cp_tail_padding_and_fully_padded_segment(self):
        c = chunk(rows=4096, length=74011, prefix=70852)
        self.assertTrue(self.supported((c,), blocks=579))
        tail = chunk(rows=932, length=74011, prefix=73084)
        self.assertEqual(op._chunk_geometry(tail)[0], 932)
        tail.host_metadata.prefix_lens = (74016,)
        self.assertEqual(op._chunk_geometry(tail)[0], 932)

    def test_strided_page_abi_and_e4m3_byte_view(self):
        packed = torch.empty_strided(
            (3, 1, 128, 64), (16384, 8192, 64, 1), dtype=torch.uint8
        )
        scales = torch.empty_strided(
            (3, 1, 2, 32, 4, 4), (8192, 1024, 512, 16, 4, 1), dtype=torch.float8_e4m3fn
        )
        op.validate_native_index_cache(packed, scales, 3)
        self.assertEqual(scales.view(torch.uint8).stride(), scales.stride())
        op.validate_native_index_cache(packed, scales.view(torch.uint8), 3)
        with self.assertRaises(ValueError):
            op.validate_native_index_cache(packed, scales.to(torch.float16), 3)
        with self.assertRaises(ValueError):
            op.validate_native_index_cache(packed[:, :, :, ::2], scales, 3)
        with self.assertRaises(ValueError):
            op.validate_native_index_cache(packed, scales.transpose(-1, -2), 3)

    def test_planner_inputs_are_cpu_and_private_contract_is_checked(self):
        seen = []

        def planner(q, k, heads, **kwargs):
            seen.append((q.device.type, k.device.type, kwargs["qo_offset"].device.type))
            self.assertEqual(kwargs["num_kv_splits"], 1)
            self.assertTrue(kwargs["causal"])
            return {
                "max_k_tiles": 128,
                "orig_num_qo_heads": heads,
                "MM-SA-Nv": False,
                "num_kv_splits": 1,
            }

        op._build_native_plan(chunk(), 4, torch.device("cuda:0"), planner)
        self.assertEqual(seen, [("cpu", "cpu", "cpu")])
        with self.assertRaises(RuntimeError):
            op._build_native_plan(
                chunk(length=16512), 4, torch.device("cuda:0"), planner
            )

    def test_mock_score_refreshes_safe_ids_and_checks_output_ownership(self):
        # Exercise Python launch/ownership contracts on CPU; no Triton/CUDA launch.
        c = chunk(rows=2, length=128)
        workspace = op.NativeIndexWorkspace.__new__(op.NativeIndexWorkspace)
        workspace.chunks = (c,)
        workspace._geometry = (op._chunk_geometry(c),)
        workspace.device = torch.device("cpu")
        workspace.pages, workspace.heads = 1, 4
        workspace.compact = False
        workspace.immutable_tables = False
        workspace.plans = ({"max_k_tiles": 128},)
        workspace.native_score = torch.empty(4 * 128 * 2, dtype=torch.float32)
        workspace.staged = torch.zeros((2, 1, 128, 128), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        workspace.safe_tables = (torch.empty_like(c.kv_indices),)
        workspace._staged = True
        seen = []

        def safe(table, target, n, pages, b, **kwargs):
            target.copy_(torch.where((table >= 0) & (table < pages), table, pages))

        def score(q, k, v, plan, **kwargs):
            seen.append(kwargs["kv_indices"].clone())
            self.assertFalse(kwargs["output_o"])
            self.assertEqual(
                (kwargs["sm_scale"], kwargs["q_scale"], kwargs["k_scale"]),
                (1.0, 1.0, 1.0),
            )
            self.assertTrue(torch.isneginf(kwargs["max_score"]).all())
            return None, kwargs["max_score"]

        workspace._score = score
        q8 = torch.zeros((2, 4, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
        output = torch.empty((4, 2, 1))
        offsets = torch.tensor([0, 1], dtype=torch.int32)
        with patch.object(op, "_safe_index_pages", LaunchMock(safe)), patch.object(
            op, "_copy_index_scores", LaunchMock(lambda *args, **kwargs: None)
        ):
            self.assertIs(workspace.score(0, q8, offsets, output), output)
            c.kv_indices[0] = -1
            workspace.score(0, q8, offsets, output)
            c.kv_indices[0] = 2
            workspace.score(0, q8, offsets, output)
            self.assertEqual([v.tolist() for v in seen], [[0], [1], [1]])
            workspace._score = lambda *args, **kwargs: (
                None,
                kwargs["max_score"].clone(),
            )
            with self.assertRaises(RuntimeError):
                workspace.score(0, q8, offsets, output)

    def test_immutable_compact_ids_reuse_but_layer_keys_refresh(self):
        chunks = (chunk(rows=2, length=256), chunk(rows=2, length=128, start=2))
        chunks[0].kv_indices.fill_(1)  # Repeated references share one staged page.
        chunks[1].kv_indices.fill_(-1)  # Empty valid set maps to the zero sentinel.
        workspace = op.NativeIndexWorkspace.__new__(op.NativeIndexWorkspace)
        workspace.chunks = chunks
        workspace._geometry = tuple(op._chunk_geometry(c) for c in chunks)
        workspace.device = torch.device("cpu")
        workspace.pages, workspace.heads, workspace.stage_pages = 2, 4, 2
        workspace.compact, workspace.immutable_tables = True, True
        workspace.plans = ({"max_k_tiles": 128},) * 2
        workspace.native_score = torch.empty(4 * 128 * 2)
        workspace.staged = torch.zeros((3, 1, 128, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
        workspace.safe_tables = tuple(torch.empty_like(c.kv_indices) for c in chunks)
        workspace.page_map = torch.empty(3, dtype=torch.int32)
        workspace.page_lists = tuple(torch.empty(c.kv_indices.numel(), dtype=torch.int32) for c in chunks)
        workspace.page_counts = tuple(torch.empty(1, dtype=torch.int32) for _ in chunks)
        workspace._staged = True
        seen_keys = []

        def claim(table, mapping, listing, count, n, pages, b, **kwargs):
            ids = table[(table >= 0) & (table < pages)].unique()
            listing[:ids.numel()].copy_(ids)
            count.fill_(ids.numel())
            mapping[ids.long()] = torch.arange(ids.numel(), dtype=torch.int32)

        def remap(table, mapping, safe, n, pages, cap, b, **kwargs):
            safe.copy_(torch.where((table >= 0) & (table < pages), mapping[table.clamp(0, pages).long()], cap))

        def stage(packed, scales, output, listing, count, *args, **kwargs):
            seen_keys.append((packed[0].item(), listing[:count.item()].tolist()))

        def score(*args, **kwargs):
            self.assertTrue(torch.isneginf(kwargs["max_score"]).all())
            return None, kwargs["max_score"]

        workspace._score = score
        with patch.object(op, "_claim_index_pages", LaunchMock(claim)), patch.object(
            op, "_remap_index_pages", LaunchMock(remap)
        ):
            for index in range(2):
                workspace._prepare_table(index)
        tables = tuple(t.clone() for t in workspace.safe_tables)
        self.assertEqual([t.tolist() for t in tables], [[0, 0], [2]])
        with patch.object(op, "_stage_compact_index_pages", LaunchMock(stage)), patch.object(
            op, "_copy_index_scores", LaunchMock(lambda *args, **kwargs: None)
        ), patch.object(workspace, "_prepare_table", side_effect=AssertionError("cached IDs rebuilt")):
            for layer in (17, 29):
                workspace._packed = torch.tensor([layer], dtype=torch.uint8)
                workspace._scales = torch.tensor([56], dtype=torch.uint8)
                for index, c in enumerate(chunks):
                    q = torch.zeros((2, 4, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
                    offsets = torch.tensor([0, c.kv_indices.numel()], dtype=torch.int32)
                    workspace.score(index, q, offsets, torch.empty(4, 2, c.kv_indices.numel()))
        self.assertEqual(seen_keys, [(17, [1]), (17, []), (29, [1]), (29, [])])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(tables, workspace.safe_tables)))

    def test_immutable_compact_budget_counts_each_chunk_list_and_count(self):
        chunks = (chunk(rows=2048, length=256), chunk(rows=2048, length=128, start=2048))
        mutable = op.native_index_workspace_bytes(chunks, 31, 4, compact=True)
        cached = op.native_index_workspace_bytes(chunks, 31, 4, compact=True, immutable_tables=True)
        self.assertEqual(cached - mutable, (3 + 2 - 2 - 1) * 4)

    def test_dispatch_releases_old_epoch_before_allocating_native_workspace(self):
        c = chunk(length=128)
        q = torch.zeros((4096, 4, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
        packed = torch.zeros((1, 1, 128, 64), dtype=torch.uint8)
        scales = torch.zeros((1, 1, 2, 32, 4, 4), dtype=torch.uint8)
        pages = torch.tensor([0], dtype=torch.int32)
        offsets = torch.tensor([0, 1], dtype=torch.int32)
        plan = {
            "_fp4_host_metadata": PrefillScoreHostMetadata((4096,), (128,), (0,), (0,))
        }

        class Workspace:
            pages, heads = 1, 4
            device = torch.device("cpu")
            chunks = ()

            def stage(self, *args):
                pass

            def score(self, index, query, page_offsets, output):
                output.zero_()

        old = Workspace()
        old_ref = weakref.ref(old)
        plan["_native_q8_index_workspace"] = old
        del old
        created = []
        chunk_tuple = (c,)

        def construct(*args, **kwargs):
            self.assertIsNone(old_ref(), "old epoch still retained during allocation")
            value = Workspace()
            value.chunks = args[0]
            value.compact = kwargs["compact"]
            value.immutable_tables = kwargs["immutable_tables"]
            created.append(value)
            return value

        # These helpers are imported inside the function: patch their modules.
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            nvfp4_q8_index_score,
            score_chunk,
        )

        with patch.object(
            score_chunk,
            "prepare_fp4_prefill_score_chunks",
            return_value=(chunk_tuple, (offsets,)),
        ), patch.object(
            op, "NativeIndexWorkspace", side_effect=construct
        ), patch.object(
            op, "supported_native_index_workspace", return_value=True
        ), patch.object(
            wrapper, "_launch_topk_to_block_table"
        ):
            wrapper.flash_prefill_topk_to_block_tables_fp4(
                q,
                packed,
                scales,
                c.cu_seqlens,
                c.seq_lens,
                c.prefix_lens,
                4096,
                128,
                128,
                16,
                1,
                index_score_plan=plan,
                kv_indices=pages,
            )
            self.assertFalse(created[0].immutable_tables)
            epoch = object()
            plan["_fp4_metadata_table"] = (epoch, pages)
            plan["_fp4_prepared_chunks"] = (epoch, pages, None, (chunk_tuple, (offsets,)))
            def forward():
                wrapper.flash_prefill_topk_to_block_tables_fp4(
                    q, packed, scales, c.cu_seqlens, c.seq_lens,
                    c.prefix_lens, 4096, 128, 128, 16, 1,
                    index_score_plan=plan, kv_indices=pages,
                )
            forward()
            self.assertTrue(created[-1].immutable_tables)
            self.assertEqual(len(created), 2)
            forward()
            self.assertEqual(len(created), 2)
            epoch = object()
            plan["_fp4_metadata_table"] = (epoch, pages)
            plan["_fp4_prepared_chunks"] = (epoch, pages, None, (chunk_tuple, (offsets,)))
            forward()
            self.assertEqual(len(created), 3)
        self.assertIs(plan["_native_q8_index_workspace"], created[-1])
        # Legacy chunk preparation can manufacture host metadata by a GPU
        # readback; that must not admit the new native planner without a producer.
        plan.pop("_fp4_host_metadata")
        with patch.object(
            score_chunk,
            "prepare_fp4_prefill_score_chunks",
            return_value=((c,), (offsets,)),
        ), patch.object(op, "supported_native_index_workspace") as admit, patch.object(
            nvfp4_q8_index_score, "q8kv4_prefill_index_score"
        ) as direct, patch.object(
            wrapper, "_launch_topk_to_block_table"
        ):
            wrapper.flash_prefill_topk_to_block_tables_fp4(
                q,
                packed,
                scales,
                c.cu_seqlens,
                c.seq_lens,
                c.prefix_lens,
                4096,
                128,
                128,
                16,
                1,
                index_score_plan=plan,
                kv_indices=pages,
            )
        admit.assert_not_called()
        direct.assert_called_once()
        self.assertNotIn("_native_q8_index_workspace", plan)


if __name__ == "__main__":
    unittest.main()
