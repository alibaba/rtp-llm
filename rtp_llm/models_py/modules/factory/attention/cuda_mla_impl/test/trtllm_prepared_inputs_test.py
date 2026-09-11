"""Real TRT prepared-input GPU tests, not a complete CMP/model precision test.

Uses the production converter, TRT op, and CMP prepare/consume bridge. The CMP
stream test simulates Q/KV/TopK producers; it does not execute RTP-kernel's CMP
prologue, Indexer, RoPE, Wvc, MoE, or an entire model. Inputs are already-rotated
absorbed BF16 Q and RTP's existing 656-byte paged cache.

Run only on an explicitly reserved SM100/103 GPU, for example:
  CUDA_VISIBLE_DEVICES=5 /opt/conda310/bin/python3 trtllm_prepared_inputs_test.py
All clones, output assertions and host synchronization are outside captured
forwards. Boundary-rejection tests deliberately assert a Python exception during
capture. These are correctness tests, not latency measurements.
"""

import importlib
import unittest
from types import SimpleNamespace

import torch
from trtllm_sparse_decode_test import PackedFixture, load_backend_class


class TrtllmPreparedInputsGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
            (10, 0),
            (10, 3),
        ):
            raise unittest.SkipTest("Prepared-input tests require an SM100/103 GPU")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        cls.Op = load_backend_class()

    def tearDown(self):
        torch.cuda.synchronize()
        super().tearDown()

    @staticmethod
    def inputs(fixture):
        return {
            "q": fixture.q,
            "cache": fixture.cache,
            "topk": fixture.topk,
            "seq_lens": fixture.seq_lens,
            "table": fixture.table,
            "request_ids": fixture.request_ids,
        }

    @classmethod
    def snapshot(cls, fixture):
        return {name: value.clone() for name, value in cls.inputs(fixture).items()}

    @staticmethod
    def restore(destination, source):
        for name, value in destination.items():
            value.copy_(source[name])

    def assert_bytes_equal(self, actual, expected, label=""):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            ),
            label,
        )

    def assert_inputs_equal(self, fixture, expected):
        for name, actual in self.inputs(fixture).items():
            self.assert_bytes_equal(actual, expected[name], name)

    def assert_scratch_equal(self, prepared_op, direct_op):
        self.assertNotEqual(
            prepared_op.selected_kv.data_ptr(), direct_op.selected_kv.data_ptr()
        )
        self.assertNotEqual(
            prepared_op.workspace_buffer.data_ptr(),
            direct_op.workspace_buffer.data_ptr(),
        )
        for name in (
            "q_fp8",
            "source_indices",
            "physical_indices",
            "valid_counts",
            "trt_seq_lens",
        ):
            self.assert_bytes_equal(
                getattr(prepared_op, name), getattr(direct_op, name), name
            )
        # Unreachable KV tails are intentionally unspecified by the public ABI.
        for row, count in enumerate(direct_op.valid_counts.cpu().tolist()):
            slots = max(count, 1)
            self.assert_bytes_equal(
                prepared_op.selected_kv[row, :slots],
                direct_op.selected_kv[row, :slots],
                f"selected_kv row={row} count={count}",
            )

    @staticmethod
    def prepare(fixture, op, *, four_dim=False, fp8_storage=False):
        cache = fixture.cache
        if fp8_storage:
            cache = cache.view(torch.float8_e4m3fn)
        if four_dim:
            cache = cache.unsqueeze(2)
        topk = fixture.topk.unsqueeze(1) if four_dim else fixture.topk
        return op.prepare(fixture.q, cache, topk, layer_id=3)

    def split_forward(self, fixture, op, **kwargs):
        return op.forward_prepared(self.prepare(fixture, op, **kwargs))

    def test_direct_and_prepared_are_bitwise_equal_h64_q1_q4_q6(self):
        for queries in (1, 4, 6):
            with self.subTest(queries=queries):
                fixture = PackedFixture(batch=2, heads=64, queries=queries)
                counts = [
                    0 if row == 0 else 65 + row % 3 for row in range(fixture.rows)
                ]
                fixture.set_counts(counts)
                fixture.q[0].fill_(float("nan"))
                fixture.topk[1, :8] = 17  # Keep duplicates, not a set of indices.
                original = self.snapshot(fixture)
                direct_op = fixture.new_op()
                prepared_op = fixture.new_op()
                expected = fixture.forward(direct_op).clone()
                for four_dim, fp8_storage in ((False, False), (True, True)):
                    with self.subTest(four_dim=four_dim, fp8_storage=fp8_storage):
                        actual = self.split_forward(
                            fixture,
                            prepared_op,
                            four_dim=four_dim,
                            fp8_storage=fp8_storage,
                        ).clone()
                        self.assert_bytes_equal(actual, expected, "latent attention")
                        self.assert_scratch_equal(prepared_op, direct_op)
                        self.assert_inputs_equal(fixture, original)
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                        self.assert_bytes_equal(actual[0], torch.zeros_like(actual[0]))

    def test_graph_refreshes_same_storage_live_empty_live_h64_q1_q4_q6(self):
        for queries in (1, 4, 6):
            with self.subTest(queries=queries):
                fixture = PackedFixture(batch=2, heads=64, queries=queries)
                fixture.set_counts([2048] * fixture.rows)
                saved = self.snapshot(fixture)
                inputs = self.inputs(fixture)
                pointers = {name: value.data_ptr() for name, value in inputs.items()}
                graph_op = fixture.new_op(cuda_graph=True)
                direct_op = fixture.new_op()
                for _ in range(3):
                    self.split_forward(fixture, graph_op)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.split_forward(fixture, graph_op)

                previous = None
                initial = None
                stages = (
                    "initial",
                    "q",
                    "cache",
                    "topk",
                    "lengths",
                    "table",
                    "request_ids",
                    "empty",
                    "live_again",
                )
                for stage in stages:
                    with self.subTest(stage=stage):
                        if stage == "q":
                            fixture.q.mul_(-2.0)
                        elif stage == "cache":
                            fixture.cache[:, :, 528:].zero_()
                        elif stage == "topk":
                            fixture.topk[:, ::3] = 17
                        elif stage == "lengths":
                            fixture.seq_lens.fill_(64)
                        elif stage == "table":
                            fixture.table.copy_(fixture.table.roll(1, dims=1))
                        elif stage == "request_ids":
                            fixture.request_ids.copy_(fixture.request_ids.flip(0))
                        elif stage == "empty":
                            fixture.topk.fill_(-1)
                            fixture.q.fill_(float("nan"))
                        elif stage == "live_again":
                            self.restore(inputs, saved)
                        before = self.snapshot(fixture)
                        graph.replay()
                        actual = output.clone()
                        expected = fixture.forward(direct_op).clone()
                        self.assert_bytes_equal(actual, expected, stage)
                        self.assert_scratch_equal(graph_op, direct_op)
                        self.assert_inputs_equal(fixture, before)
                        self.assertEqual(
                            pointers,
                            {name: value.data_ptr() for name, value in inputs.items()},
                        )
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                        if stage == "empty":
                            self.assert_bytes_equal(actual, torch.zeros_like(actual))
                            self.assertEqual(
                                graph_op.valid_counts.cpu().tolist(), [0] * fixture.rows
                            )
                        if previous is not None:
                            self.assertFalse(
                                torch.equal(actual, previous),
                                f"Replay ignored {stage} update",
                            )
                        if stage == "initial":
                            initial = actual.clone()
                        elif stage == "live_again":
                            self.assert_bytes_equal(actual, initial, "restored inputs")
                        previous = actual

    def test_cmp_prepare_on_side_stream_join_then_consume_is_captured(self):
        # Real bridge methods, simulated producers: no full CMP-model claim.
        cmp_module = importlib.import_module(
            "rtp_llm.models_py.modules.hybrid.glm5_cmp"
        )
        cmp = object.__new__(cmp_module.Glm5Cmp)
        cmp.layer_idx = 3
        fixture = PackedFixture(batch=2, heads=64, queries=4)
        fixture.set_counts([65] * fixture.rows)
        sources = self.snapshot(fixture)
        saved = {name: value.clone() for name, value in sources.items()}
        destinations = self.inputs(fixture)
        op = fixture.new_op(cuda_graph=True)
        direct_op = fixture.new_op()
        outer = SimpleNamespace(weights={}, fmha_params=fixture.params, fmha_impl=op)
        layer_cache = SimpleNamespace(kv_cache_base=fixture.cache)
        producer = torch.cuda.Stream(device=fixture.device)
        caller_ready, producer_done = torch.cuda.Event(), torch.cuda.Event()

        def joined_forward():
            caller = torch.cuda.current_stream()
            caller_ready.record(caller)
            with torch.cuda.stream(producer):
                producer.wait_event(caller_ready)
                self.restore(destinations, sources)
                token = cmp._prepare_sparse_mla_inputs(
                    outer, fixture.q, fixture.cache, fixture.topk
                )
                producer_done.record(producer)
            caller.wait_event(producer_done)
            return cmp.sparse_mla(token, fixture.topk, outer, layer_cache)

        for _ in range(3):
            joined_forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = joined_forward()
        initial = None
        for stage in ("live", "empty", "live_again"):
            with self.subTest(stage=stage):
                if stage == "empty":
                    sources["q"].fill_(float("nan"))
                    sources["topk"].fill_(-1)
                    sources["seq_lens"].zero_()
                    sources["cache"][:, :, 528:].zero_()
                    sources["table"].copy_(sources["table"].roll(1, dims=1))
                elif stage == "live_again":
                    self.restore(sources, saved)
                    sources["q"].mul_(1.75)
                    sources["topk"][:, ::3] = 17
                    sources["seq_lens"].fill_(64)
                    sources["table"].copy_(sources["table"].roll(2, dims=1))
                # Only source storage changes: replay must run both producer
                # copies and conversion, not consume a prior eager preparation.
                graph.replay()
                actual = output.clone()
                self.assert_inputs_equal(fixture, sources)
                expected = fixture.forward(direct_op).clone()
                self.assert_bytes_equal(actual, expected, stage)
                self.assert_scratch_equal(op, direct_op)
                self.assert_inputs_equal(fixture, sources)
                if stage == "empty":
                    self.assert_bytes_equal(actual, torch.zeros_like(actual))
                elif stage == "live":
                    initial = actual.clone()
                else:
                    self.assertFalse(torch.equal(actual, initial))

    def test_rejects_wrong_owner_stale_duplicate_and_replanned_tokens(self):
        fixture = PackedFixture(batch=1, heads=64)
        op, other = fixture.new_op(), fixture.new_op()
        token = self.prepare(fixture, op)
        with self.assertRaisesRegex(RuntimeError, "prepared inputs"):
            other.forward_prepared(token)
        # A rejected wrong-owner call must not invalidate the correct owner.
        actual = op.forward_prepared(token).clone()
        self.assert_bytes_equal(actual, fixture.forward(other).clone())
        with self.assertRaisesRegex(RuntimeError, "prepared inputs"):
            op.forward_prepared(token)

        stale = self.prepare(fixture, op)
        current = self.prepare(fixture, op)
        with self.assertRaisesRegex(RuntimeError, "prepared inputs"):
            op.forward_prepared(stale)
        op.forward_prepared(current)

        token = self.prepare(fixture, op)
        fixture.forward(op)  # The ordinary entry point also overwrites scratch.
        with self.assertRaisesRegex(RuntimeError, "prepared inputs"):
            op.forward_prepared(token)

        token = self.prepare(fixture, op)
        op.plan(fixture.params, fixture.table, fixture.attn_inputs)
        with self.assertRaisesRegex(RuntimeError, "prepared inputs"):
            op.forward_prepared(token)
        self.assert_bytes_equal(
            self.split_forward(fixture, op).clone(), fixture.forward(other).clone()
        )

    def test_rejects_eager_token_consumed_inside_capture(self):
        fixture = PackedFixture(batch=1, heads=64)
        op = fixture.new_op(cuda_graph=True)
        for _ in range(3):
            self.split_forward(fixture, op)
        token = self.prepare(fixture, op)
        marker = torch.zeros(1, device=fixture.device)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            marker.add_(1)  # Keep a nonempty, valid capture after Python rejection.
            with self.assertRaisesRegex(RuntimeError, "capture boundary"):
                op.forward_prepared(token)
        # Rejection is host-side and must not leave CUDA capture invalidated.
        graph.replay()
        op.forward_prepared(token)

    def test_rejects_captured_token_consumed_eagerly(self):
        fixture = PackedFixture(batch=1, heads=64)
        op = fixture.new_op(cuda_graph=True)
        for _ in range(3):
            self.split_forward(fixture, op)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            token = self.prepare(fixture, op)
        # Intentionally capture only the producer for this negative test. The
        # normal graph tests above always capture producer and consumer together.
        with self.assertRaisesRegex(RuntimeError, "capture boundary"):
            op.forward_prepared(token)
        self.split_forward(fixture, op)  # Fresh paired use remains valid.


if __name__ == "__main__":
    unittest.main()
