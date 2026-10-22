"""FIA2A planning and adapter contracts without launching GPU kernels."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.ops import DecodeCPMLAA2ABackend as A2A, DecodeCPMLAFusionMode as Fusion
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import page_rr_fia2a_backend as module
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a import workspace as workspace_module


class FIA2ABackendTest(unittest.TestCase):
    def setUp(self):
        self.kernel = MagicMock()
        self.pack = MagicMock()
        self.patches = [
            patch.object(module, "_compile_producer", return_value=self.kernel),
            patch.object(module, "_pack", self.pack),
            patch.object(module, "_merge_splits_serial", MagicMock()),
            patch.object(module, "merge_local_splits", MagicMock()),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "synchronize"),
            patch.object(module.dist, "barrier"),
            patch.object(module.tvm_ffi, "use_torch_stream", MagicMock()),
            patch.dict(os.environ, {"KIMI_K3_SMOKE_EVIDENCE": "0"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def backend(self, *, world=4, fusion=Fusion.AUTO, a2a=A2A.NCCL, token_major=True):
        result = object.__new__(module.PageRRFIA2AMLABackend)
        result.requested_mode, result.a2a_backend = fusion, a2a
        result.token_major, result.heads, result.page_size = token_major, 24, 128
        result.dtype, result.sm_count, result._shape = torch.bfloat16, 148, None
        result.local_control = torch.zeros(1, dtype=torch.int64)
        result.communicator = SimpleNamespace(
            heads=24, local_heads=24 // world, size=world, rank=world - 1,
            device=torch.device("cpu"), group=object(), combine=MagicMock(),
        )

        def buffers(rows, splits, fused):
            size = world if fused else 1
            return SimpleNamespace(
                capacity=rows * splits,
                output=torch.empty(size, rows * splits, 24 // size, 512,
                                   dtype=torch.float32 if splits > 1 else torch.bfloat16),
                lse=torch.empty(size, rows * splits, 24 // size),
                pointers=torch.zeros(size, 2, dtype=torch.int64),
                output_ptrs=(), control_peers=torch.zeros(size, dtype=torch.int64),
            )

        result.workspace = SimpleNamespace(
            prepared=set(), acquire=MagicMock(side_effect=buffers),
            acquire_all_to_all=MagicMock(side_effect=lambda rows: buffers(rows, 1, False)),
        )
        return result

    @staticmethod
    def metadata(batch=1, queries=4):
        return SimpleNamespace(
            local_causal_lens=torch.ones(batch, queries, dtype=torch.int32),
            block_tables=torch.ones(batch, 2, dtype=torch.int32),
            query_block_tables=torch.ones(batch * queries, 2, dtype=torch.int32),
        )

    def test_auto_and_forced_modes_preserve_split_geometry(self):
        for world in (4, 8):
            for mode in (Fusion.AUTO, Fusion.FUSED, Fusion.UNFUSED):
                for batch in (1, 128):
                    with self.subTest(world=world, mode=mode, batch=batch):
                        backend = self.backend(world=world, fusion=mode)
                        backend.prepare(self.metadata(batch), 0.1, 1.0)
                        self.assertEqual(backend.splits, 32 if batch == 1 else 1)
                        self.assertEqual(backend.fused, mode == Fusion.FUSED or (
                            mode == Fusion.AUTO and batch == 128
                        ))

    def test_ragged_update_selects_owner_table_even_when_row_counts_match(self):
        metadata = self.metadata(batch=2, queries=1)
        metadata.query_block_tables = torch.tensor([[17, 18], [17, 18]], dtype=torch.int32)
        backend = self.backend()
        backend.prepare(metadata, 0.1, 1.0, per_query_table=True)
        self.assertIs(backend.table, metadata.query_block_tables)
        self.assertIsNot(backend.table, metadata.block_tables)

    def test_replicated_query_forward_does_not_pack_or_gather(self):
        backend = self.backend(fusion=Fusion.FUSED)
        backend.prepare(self.metadata(), 0.1, 1.0)
        query = torch.empty(4, 24, 576, dtype=torch.bfloat16)
        cache = torch.empty(1, 128, 576, dtype=torch.bfloat16)
        output = backend.forward(query, cache)
        self.assertIs(output, backend.output)
        self.pack.warmup.assert_not_called()
        self.pack.__getitem__.assert_not_called()
        backend.communicator.combine.assert_not_called()
        self.kernel.assert_called_once()

    def test_custom_merge_consumes_direct_symmetric_sources(self):
        backend = self.backend(fusion=Fusion.UNFUSED, a2a=A2A.CUSTOM)
        backend.prepare(self.metadata(), 0.1, 1.0)
        backend.forward(torch.empty(4, 24, 576), torch.empty(1, 128, 576))
        self.assertIsNotNone(backend.a2a_buffers)
        self.assertEqual(backend.output.data_ptr(), backend.a2a_buffers.output.data_ptr())
        self.assertEqual(backend.output_lse.data_ptr(), backend.a2a_buffers.lse.data_ptr())
        self.assertIs(backend.communicator.combine.call_args.kwargs["a2a_buffers"], backend.a2a_buffers)

    def test_head_sharded_query_keeps_the_prepared_reorder(self):
        backend = self.backend(fusion=Fusion.FUSED, token_major=False)
        backend.prepare(self.metadata(), 0.1, 1.0)
        backend.forward(torch.empty(24, 4, 576, dtype=torch.bfloat16),
                        torch.empty(1, 128, 576, dtype=torch.bfloat16))
        self.pack.warmup.assert_called_once()
        self.pack.__getitem__.return_value.assert_called_once()

    def test_shape_growth_is_rejected_inside_capture(self):
        backend = self.backend()
        backend.prepare(self.metadata(), 0.1, 1.0)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "warmed"):
                backend.prepare(self.metadata(batch=2), 0.1, 1.0)

    def test_workspace_growth_keeps_captured_generation(self):
        communicator = self.backend().communicator
        arena = workspace_module.FIA2AWorkspace(communicator)
        with patch.object(workspace_module, "_OutputBuffers", side_effect=lambda h, capacity, *args: SimpleNamespace(capacity=capacity)):
            captured = arena.acquire(4, 1, True)
            grown = arena.acquire(33, 1, True)
            self.assertIsNot(captured, grown)
            self.assertEqual(captured.capacity, 16)
            self.assertEqual(grown.capacity, 64)
            self.assertIs(arena.acquire(32, 1, True), grown)


class FIA2AMetadataGraphTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA metadata/Graph regression")
    def test_request_table_keeps_capture_pointer_on_replay(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import PageRRMlaMetadata

        metadata = PageRRMlaMetadata(4096, 128, 4, 0, expand_query_table=False)
        original = torch.arange(1, 129, device="cuda", dtype=torch.int32).reshape(2, 64)
        inputs = SimpleNamespace(
            is_target_verify=True, is_mtp_draft_update=False, is_cuda_graph=False,
            input_lengths=torch.ones(2, dtype=torch.int32), physical_token_count=8,
            prefix_lengths_device=torch.tensor([0, 4096], device="cuda", dtype=torch.int32),
            kv_cache_kernel_block_id_device=original,
        )
        # Only metadata kernels execute here; attention and peer communication
        # have their own multi-rank numerical acceptance requirements.
        metadata.prepare_decode(inputs)
        self.assertIsNone(metadata.query_block_tables)
        pointer = metadata.block_tables.data_ptr()
        replacement = original + 4
        inputs.kv_cache_kernel_block_id_device = replacement
        metadata.prepare_decode(inputs, forbid_realloc=True)
        self.assertEqual(metadata.block_tables.data_ptr(), pointer)
        torch.testing.assert_close(metadata.block_tables, replacement, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata.prepare_decode(inputs, forbid_realloc=True)
        replacement.add_(4)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(metadata.block_tables.data_ptr(), pointer)
        torch.testing.assert_close(metadata.block_tables, replacement, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
