"""Cross-stream plan lifetime, changed inputs, and multi-stream graph replay."""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.decode import sparse_plan_overlap as overlap
from rtp_llm.models_py.modules.dsv41.fp8.test import test_v41_sparse_decode as cases


class SparsePlanOverlapCPU(unittest.TestCase):
    def test_default_launch_and_explicit_disable(self):
        shared = {"candidates": Mock()}
        source = SimpleNamespace(layer_id=20, kv_source_layer_id=20,
                                 compress_ratio=1, _shared_attention=shared,
                                 v41_config={"candidate_source_layer_id": 20,
                                             "candidate_block_size": 8})
        with patch.dict(os.environ), patch.object(
            sparse, "is_supported", return_value=False
        ) as supported:
            os.environ.pop("DSV41_OVERLAP_SPARSE_PLAN", None)
            overlap.start_plan(source, Mock(), Mock(), query_len=6)
            supported.assert_called_once_with(shared["candidates"])
            supported.reset_mock()
            os.environ["DSV41_OVERLAP_SPARSE_PLAN"] = "0"
            overlap.start_plan(source, Mock(), Mock(), query_len=6)
            supported.assert_not_called()

    def test_full_fast_return_preserves_output(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode import full_overlap
        from rtp_llm.models_py.modules.dsv41.fp8.decode import rope_metadata

        x = torch.empty(1, 1, 8)
        positions = torch.zeros(1, dtype=torch.int64)
        metadata = SimpleNamespace(position_ids=positions, swa_global_slots=positions)
        mega = Mock()
        mega.supports_input.return_value = True
        mega.supports_decode.return_value = True
        mega.supports_pool.return_value = True
        layer = SimpleNamespace(
            _begin_forward=Mock(), _shared_attention={}, freqs_cis=Mock(),
            _pool_view_3d_fp8=Mock(), _mega_attention=mega,
            compress_ratio=1, index_topk=512,
            _source_pool=Mock(), _global_region=Mock(),
        )
        output = torch.empty_like(x)
        with patch.object(rope_metadata, "decode_rope_metadata",
                          return_value=(positions, Mock())), \
             patch.object(full_overlap, "try_full_decode", return_value=output):
            actual = AttentionV41FP8._forward_decode_body(layer, x, metadata)
        self.assertIs(actual, output)

    def test_wait_once_and_record_output_lifetimes(self):
        current, auxiliary = Mock(), Mock()
        outputs = sparse.SparseDecodePlan(*(Mock() for _ in range(5)))
        work = overlap.SparsePlanWork(current, auxiliary, (Mock(),), outputs)
        self.assertIs(work.wait(), outputs)
        self.assertIs(work.wait(), outputs)
        current.wait_stream.assert_called_once_with(auxiliary)
        for tensor in outputs:
            tensor.record_stream.assert_called_once_with(current)
        self.assertEqual(work.inputs, ())

    def test_interrupted_forward_joins_before_dropping_plan(self):
        work = Mock()
        shared = {"layers": {0: None}, "decode_sparse_plan_ready": work}
        AttentionV41FP8._begin_forward(SimpleNamespace(layer_id=0, _shared_attention=shared))
        work.wait.assert_called_once()
        self.assertNotIn("decode_sparse_plan_ready", shared)
        self.assertNotIn("decode_sparse_plan", shared)

    def test_no_local_consumer_means_no_fork(self):
        layer = SimpleNamespace(layer_id=20, _shared_attention={"layers": {20: None}})
        self.assertFalse(overlap.has_consumer(layer))

    def test_candidate_launch_orders_producer_and_publishes_work_immediately(self):
        candidates = torch.empty(2, 2048, dtype=torch.int32)
        table = torch.empty(2, 256, dtype=torch.int32)
        lengths = torch.empty(2, dtype=torch.int32)
        flags = torch.empty_like(candidates)
        shared = {"candidates": candidates, "candidate_mask": Mock()}
        source = SimpleNamespace(
            layer_id=20, kv_source_layer_id=20, compress_ratio=1,
            v41_config={"candidate_source_layer_id": 20, "candidate_block_size": 8},
            _shared_attention=shared,
        )
        keys = SimpleNamespace(pool=SimpleNamespace(shape=(1, 128)),
                               block_table=table, logical_entries_per_block=128)
        current, auxiliary, producer = Mock(), Mock(), Mock()
        plan = sparse.SparseDecodePlan(*(Mock() for _ in range(5)))
        with patch.dict(os.environ, {"DSV41_OVERLAP_SPARSE_PLAN": "1"}), \
             patch.dict(overlap._PLAN_STREAMS, {0: auxiliary}, clear=True), \
             patch.object(overlap, "has_consumer", return_value=True), \
             patch.object(sparse, "is_supported", return_value=True), \
             patch.object(sparse, "_plan_candidate_flags", return_value=flags), \
             patch.object(torch.cuda, "current_stream", return_value=current), \
             patch.object(torch.cuda, "device", return_value=nullcontext()), \
             patch.object(torch.cuda, "stream", return_value=nullcontext()), \
             patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch("rtp_llm.models_py.modules.dsv41.fp8.attention._cuda_device_index",
                   return_value=0), \
             patch.object(sparse, "make_plan", return_value=plan) as planner:
            overlap.start_plan(source, keys, lengths, query_len=6, producer_stream=producer)
            self.assertEqual(auxiliary.wait_stream.call_args_list,
                             [call(current), call(producer)])
            planner.assert_called_once_with(
                candidates, lengths, table, query_len=6,
                candidate_mask=shared["candidate_mask"],
            )
            work = shared["decode_sparse_plan_ready"]
            self.assertIs(work.plan, plan)
            current.wait_stream.assert_not_called()
            # Publication is once per forward; consumers join only when needed.
            overlap.start_plan(source, keys, lengths, query_len=6, producer_stream=producer)
            self.assertEqual(planner.call_count, 1)
            overlap.wait_plan(shared)
            self.assertIs(shared["decode_sparse_plan"], plan)
            current.wait_stream.assert_called_once_with(auxiliary)
            self.assertEqual(work.inputs, ())
            self.assertNotIn("decode_sparse_plan_ready", shared)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparsePlanOverlapCUDA(unittest.TestCase):
    def test_eager_and_graph_refresh_all_four_consumers(self):
        if not sparse.is_supported(torch.empty(1, 2048, device="cuda", dtype=torch.int32)):
            self.skipTest("SM10x and native paged Sparse MQA required")
        torch.manual_seed(7301)
        batch, query_len, capacity = 2, 3, 32768
        rows = batch * query_len
        table, pool = cases.SparseDecodeCUDA._paged_case(self, batch, capacity)
        keys = indexer.DecodeIndexerKeys(pool, table, capacity, 128)
        shared = {"layers": {0: None}}
        first = SimpleNamespace(layer_id=0, _shared_attention=shared)
        config = dict(candidate_source_layer_id=20, candidate_topk_blocks=2048,
                      candidate_block_size=8)
        source_layer = SimpleNamespace(layer_id=20, kv_source_layer_id=20,
                                       compress_ratio=1, v41_config=config,
                                       _shared_attention=shared)
        positions = torch.tensor([20000, 20001, 20002, 30000, 30001, 30002], device="cuda")
        source = torch.randn(rows, capacity, device="cuda")
        producer = torch.cuda.Stream()
        x = torch.randn(batch, query_len, 32, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(batch, query_len, 16, device="cuda", dtype=torch.bfloat16)
        frequencies = torch.polar(torch.ones(capacity, 32, device="cuda"),
                                  torch.randn(capacity, 32, device="cuda"))
        layers = []
        for layer_id in (24, 28, 32, 36):
            layer = AttentionV41FP8.__new__(AttentionV41FP8)
            torch.nn.Module.__init__(layer)
            layer.layer_id = layer.index_source_layer_id = layer_id
            layer.kv_source_layer_id = 20
            layer.is_index_source = layer.decode_sparse_mqa = True
            layer.compress_ratio = 1
            layer.index_n_heads, layer.index_head_dim, layer.index_topk = 32, 128, 512
            layer.rope_head_dim = 64
            layer._shared_attention, layer.v41_config = shared, config
            layer.index_wq = torch.nn.Linear(16, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
            layer.index_weights = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            layer.freqs_cis = frequencies
            layers.append(layer)
            shared["layers"][layer_id] = layer

        def forward(expect_async=True):
            AttentionV41FP8._begin_forward(first)
            shared["global"] = {20: keys}
            lengths = (positions + 1).int()
            # Exercise the real FULL candidate producer, which returns before
            # the auxiliary stream's candidate/flag writes have completed.
            from rtp_llm.models_py.modules.dsv41.fp8.decode.full_overlap import FullDecodeWork

            current = torch.cuda.current_stream()
            work = FullDecodeWork(current, current, producer)
            candidates, flags = work.select_candidates(source, lengths, 8, 2048)
            shared["candidates"] = candidates
            shared["candidate_mask"] = (candidates, flags, 8)
            overlap.start_plan(
                source_layer, keys, lengths, query_len=query_len,
                producer_stream=producer,
            )
            self.assertEqual("decode_sparse_plan_ready" in shared, expect_async)
            # Independent main-stream work between publication and consumption.
            scratch = torch.mm(qr.reshape(rows, 16).float(), qr.reshape(rows, 16).float().T)
            work.finish()
            self.assertEqual("decode_sparse_plan_ready" in shared, expect_async)
            outputs = [layer._select_indices_decode(x, qr, positions) for layer in layers]
            self.assertNotIn("decode_sparse_plan_ready", shared)
            return outputs, candidates, shared["decode_sparse_plan"], scratch

        def check(result):
            outputs, candidates, plan, _ = result
            reference = sparse.make_plan(candidates, positions + 1, table, query_len=query_len)
            for actual, expected in zip(plan[:4], reference[:4]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for layer, actual in zip(layers, outputs):
                q = layer.index_wq(qr).view(batch, query_len, 32, 128)
                weights = torch.nn.functional.linear(x, layer.index_weights)
                expected = sparse.select_decode(q, weights, frequencies, positions, pool,
                                                reference, rope_head_dim=64)
                torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values,
                                           rtol=0, atol=0)

        with torch.no_grad(), patch.dict(os.environ):
            # Exercise the production default, not an explicit opt-in.
            os.environ.pop("DSV41_OVERLAP_SPARSE_PLAN", None)
            with patch.object(sparse, "make_plan", wraps=sparse.make_plan) as planner:
                eager = forward()
                self.assertEqual(planner.call_count, 1)
            check(eager)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                forward()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = forward()
            for delta in (1, 17, 3):
                positions.add_(delta)
                source.normal_()
                qr.normal_()
                table.copy_(table.roll(1, dims=1))
                # Reload-like weight updates must not reuse a previous plan/output.
                for layer in layers:
                    layer.index_wq.weight.normal_()
                    layer.index_weights.normal_()
                if delta == 17:
                    source[:, :28000] = torch.nan
                elif delta == 3:
                    source.fill_(-torch.inf)
                graph.replay()
                check(captured)
            # Resume eager execution after graph use and replace forward-local buffers.
            source.normal_()
            check(forward())
            # No stream initialization/JIT from an unwarmed capture: serial fallback.
            # The original serial path also requires DG workspace warm-up on
            # its actual capture stream (the reference above used default).
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream), patch.dict(
                os.environ, {"DSV41_OVERLAP_SPARSE_PLAN": "0"}
            ):
                forward(expect_async=False)
            torch.cuda.current_stream().wait_stream(capture_stream)
            shared["decode_sparse_plan_warmed"].clear()
            cold = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold, stream=capture_stream):
                fallback = forward(expect_async=False)
            positions.add_(1)
            source.normal_()
            cold.replay()
            check(fallback)


if __name__ == "__main__":
    unittest.main()
