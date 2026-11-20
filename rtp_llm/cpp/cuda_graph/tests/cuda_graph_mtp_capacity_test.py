"""Exercise native draft graph layouts independently of hidden-state width."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.cpp.cuda_graph.tests import cuda_graph_tagged_cache_test as fixtures
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_STATE_TAG
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    DraftPrefillGraphReplay,
    draft_graph_rows,
)
from rtp_llm.ops.compute_ops import GraphReplayMetadataLease, PyModelOutputs


class PackedDraftCapacityModel:
    def __init__(self, hc_mult=4):
        self.shapes = set()
        self.hc_mult = hc_mult

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        batch = metadata.input_lengths_device.numel()
        self.shapes.add((rows, batch))
        hidden = inputs.input_hiddens.reshape(rows, self.hc_mult, 4).sum(1)
        return PyModelOutputs(hidden + inputs.input_ids.to(hidden.dtype)[:, None])


class DraftCapacityModel(PackedDraftCapacityModel):
    def __init__(self, num_tokens_per_request=4, hc_mult=4):
        super().__init__(hc_mult)
        self.num_tokens_per_request = num_tokens_per_request

    def cuda_graph_draft_prefill_uses_request_lanes(self):
        return True

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        assert (
            rows == metadata.input_lengths_device.numel() * self.num_tokens_per_request
        )
        outputs = super().forward(inputs, fmha_impl)
        _, _, _, positions, _ = draft_graph_rows(metadata, rows)
        return PyModelOutputs(
            outputs.hidden_states + positions.to(outputs.hidden_states.dtype)[:, None]
        )


class MetadataLeaseDraftModel(DraftCapacityModel):
    def __init__(self):
        super().__init__()
        self.delegate = Mock()
        self.replays = []

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        replay = DraftPrefillGraphReplay(
            {"full": self.delegate},
            [("full", 128, 128, False), (INDEXER_STATE_TAG, 128, 128, False)],
            1,
            inputs.input_ids.numel(),
            SimpleNamespace(
                get_seq_size_per_block=lambda tag: 128,
                get_kernel_seq_size_per_block=lambda tag: 128,
            ),
        )
        self.replays.append(replay)
        return replay


class MtpGraphCapacityTest(unittest.TestCase):
    def test_request_capacity_shrink_and_grow_with_variable_acceptance(self):
        # HC width and speculative request lanes are independent dimensions.
        for lanes, hc_mult in ((1, 4), (2, 4), (3, 1), (4, 4)):
            with self.subTest(lanes=lanes, hc_mult=hc_mult):
                self._check_request_capacity(lanes, hc_mult)

    def _check_request_capacity(self, lanes, hc_mult):
        torch.cuda.set_device(0)
        model = DraftCapacityModel(lanes, hc_mult)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, lanes, hc_mult)
        self.assertEqual(
            model.shapes, {(lanes * batch, batch) for batch in range(1, 9)}
        )

        # The same graph keys are revisited after both larger and smaller
        # batches. Even one-token requests retain the configured request lanes.
        for lengths in (
            [lanes] * 8,
            [1],
            [lanes],
            [1] * 8,
            [lanes, 1, max(1, lanes - 1)],
            [1, 1],
            [min(2, lanes)],
            [max(1, lanes - 1)] * 7,
            [1],
        ):
            with self.subTest(lengths=lengths):
                rows = sum(lengths)
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                # A single-group graph uses the anchor ABI, not a one-entry tag map.
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                inputs.input_hiddens = (
                    torch.arange(rows * hc_mult * 4, device="cuda", dtype=torch.float32)
                    .remainder(11)
                    .reshape(rows, hc_mult * 4)
                    .to(torch.bfloat16)
                )
                metadata = get_primary_attention_inputs(inputs)
                metadata.prefix_lengths.fill_(7)
                metadata.context_total_kv_length = rows + 7 * len(lengths)
                cu_kv = [0]
                for length in lengths:
                    cu_kv.append(cu_kv[-1] + 7 + length)
                metadata.cu_kv_seqlens_device.copy_(
                    torch.tensor(cu_kv, device="cuda", dtype=torch.int32)
                )
                positions = torch.tensor(
                    [7 + offset for length in lengths for offset in range(length)],
                    device="cuda",
                    dtype=torch.int32,
                )
                inputs.combo_position_ids = positions
                expected = inputs.input_hiddens.reshape(rows, hc_mult, 4).sum(1)
                expected = (
                    expected
                    + (positions + inputs.input_ids).to(expected.dtype)[:, None]
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), lanes * len(lengths))
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

        overflow = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 9)
        overflow.attention_inputs = get_primary_attention_inputs(overflow)
        self.assertFalse(runner.canRun(overflow))

    def test_undeclared_layout_keeps_packed_metadata_with_hc(self):
        torch.cuda.set_device(0)
        model = PackedDraftCapacityModel(hc_mult=4)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, 3, 4)
        self.assertEqual(model.shapes, {(3 * batch, 8) for batch in range(1, 9)})
        for lengths in ([1] * 8, [3], [3, 1, 2]):
            with self.subTest(lengths=lengths):
                rows = sum(lengths)
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                inputs.input_hiddens = torch.ones(
                    (rows, 16), dtype=torch.bfloat16, device="cuda"
                )
                inputs.combo_position_ids = torch.tensor(
                    [offset for length in lengths for offset in range(length)],
                    dtype=torch.int32,
                    device="cuda",
                )
                expected = 4 + inputs.input_ids.to(torch.bfloat16)[:, None].expand(
                    -1, 4
                )
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_nonpositive_request_lanes_rejected_before_capture(self):
        for lanes in (0, -1):
            with self.subTest(lanes=lanes):
                with self.assertRaisesRegex(RuntimeError, "int32 token/padding"):
                    CudaGraphRunner().init_mtp_draft_prefill(
                        DraftCapacityModel(), 8, 4, lanes, 4
                    )


class SourceKernelMtpGraphCapacityTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.kernels.cuda.qsa import draft_metadata_triton

        original = draft_metadata_triton.replay_invalid_from_sources

        def require_kernel(*args, **kwargs):
            result = original(*args, **kwargs)
            self.assertIsNotNone(result, "source kernel must run, not fall back")
            return result

        required = patch.object(
            draft_metadata_triton, "replay_invalid_from_sources", require_kernel
        )
        required.start()
        self.addCleanup(required.stop)

    def test_source_kernel_uses_independent_reusable_snapshots_without_item(
        self,
    ):
        from rtp_llm.models_py.kernels.cuda.qsa import draft_metadata_triton

        runner, model = self._metadata_lease_runner()
        snapshots, leases, gpu_items, uploads = [], [], [], []
        original = draft_metadata_triton.replay_invalid_from_sources
        native_matches = GraphReplayMetadataLease.matches
        item, copy = torch.Tensor.item, torch.Tensor.copy_

        def inspect_source(
            lengths,
            prefixes,
            captured_lengths,
            captured_prefixes,
            table,
            transported,
            cu,
            **kwargs,
        ):
            self.assertNotEqual(lengths.data_ptr(), captured_lengths.data_ptr())
            self.assertNotEqual(prefixes.data_ptr(), captured_prefixes.data_ptr())
            snapshots.append((lengths.numel(), lengths.data_ptr(), prefixes.data_ptr()))
            return original(
                lengths,
                prefixes,
                captured_lengths,
                captured_prefixes,
                table,
                transported,
                cu,
                **kwargs,
            )

        def observe(lease, groups):
            leases.append(lease)
            return native_matches(lease, groups)

        def checked_item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                gpu_items.append(1)
            return item(tensor, *args, **kwargs)

        def checked_copy(tensor, source, *args, **kwargs):
            if tensor.is_cuda and not source.is_cuda and source.is_pinned():
                uploads.append(1)
            return copy(tensor, source, *args, **kwargs)

        for lengths in ([4] * 8, [1], [4, 1, 3], [1, 1], [4] * 8):
            inputs = self._metadata_lease_inputs(lengths)
            with (
                patch.object(
                    draft_metadata_triton, "replay_invalid_from_sources", inspect_source
                ),
                patch.object(GraphReplayMetadataLease, "matches", observe),
                patch.object(torch.Tensor, "item", checked_item),
                patch.object(torch.Tensor, "copy_", checked_copy),
            ):
                self.assertTrue(runner.prepare(inputs))
            self.assertIs(leases[-1].device_validation_invalid, False)
            self._assert_metadata_lease_output(runner, inputs)
        self.assertTrue(snapshots)
        for capacity in {row[0] for row in snapshots}:
            self.assertEqual(
                len({row[1:] for row in snapshots if row[0] == capacity}), 1
            )
        self.assertEqual(gpu_items, [])
        self.assertEqual(uploads, [])

    def test_source_kernel_interrupted_enqueue_recovers_on_other_stream(self):
        from rtp_llm.models_py.kernels.cuda.qsa import draft_metadata_triton

        runner, model = self._metadata_lease_runner()
        inputs = self._metadata_lease_inputs([2, 1])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        original = draft_metadata_triton.replay_invalid_from_sources

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected early verdict interruption")

        before = model.delegate.prepare_cuda_graph.call_count
        with (
            torch.cuda.stream(stream),
            patch.object(
                draft_metadata_triton, "replay_invalid_from_sources", interrupted
            ),
        ):
            # Delay the earlier fills/copies too. Waiting only at attention
            # preparation is too late if forward first overwrites token data.
            torch.cuda._sleep(200_000_000)
            with self.assertRaisesRegex(RuntimeError, "early verdict interruption"):
                runner.prepare(inputs)
        self.assertEqual(model.delegate.prepare_cuda_graph.call_count, before)
        # Deliberately no caller wait on the interrupted stream. The runner's
        # failure fence must protect both snapshots and shared capture storage.
        fresh = self._metadata_lease_inputs([1, 3], prefix=11)
        self._assert_metadata_lease_output(runner, fresh)

    def test_source_kernel_retains_original_source_until_failure_fence(self):
        import gc

        from rtp_llm.models_py.kernels.cuda.qsa import draft_metadata_triton

        runner, _ = self._metadata_lease_runner()
        inputs = self._metadata_lease_inputs([2, 1])
        source = inputs.attention_inputs["full"].input_lengths
        source_lifetime = torch._C._WeakTensorRef(source)
        del source
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Keep the already-enqueued source D2D readers pending when the
            # CPU exception unwinds and the caller drops its whole input.
            torch.cuda._sleep(20_000_000)
            with patch.object(
                draft_metadata_triton,
                "replay_invalid_from_sources",
                side_effect=RuntimeError("injected before verdict return"),
            ):
                with self.assertRaisesRegex(RuntimeError, "before verdict return"):
                    runner.prepare(inputs)
        del inputs
        gc.collect()
        retained_before_fence = not source_lifetime.expired()
        # Drain before reporting a failing control so no borrowed source read
        # outlives the test, even on the deliberately unfixed implementation.
        stream.synchronize()
        self.assertTrue(
            retained_before_fence,
            "producer TensorImpl released before failed-prepare fence",
        )
        self._assert_metadata_lease_output(
            runner, self._metadata_lease_inputs([1, 3], prefix=11)
        )
        gc.collect()
        self.assertTrue(
            source_lifetime.expired(),
            "retired producer source retained after safe slot reuse",
        )

    def _metadata_lease_runner(self):
        torch.cuda.set_device(0)
        model = MetadataLeaseDraftModel()
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, 4, 4, ["full", INDEXER_STATE_TAG])
        return runner, model

    def _metadata_lease_inputs(self, lengths, *, prefix=7, device_first=True):
        inputs = fixtures._build_prefill_inputs(
            ["full", INDEXER_STATE_TAG],
            {"full": 1, INDEXER_STATE_TAG: 3},
            lengths,
        )
        source_lengths = torch.tensor(lengths, dtype=torch.int32)
        source_prefixes = torch.full_like(source_lengths, prefix)
        if device_first:
            source_lengths = source_lengths.cuda()
            source_prefixes = source_prefixes.cuda()
        else:
            source_lengths = source_lengths.pin_memory()
            source_prefixes = source_prefixes.pin_memory()
        cu_kv = [0]
        for length in lengths:
            cu_kv.append(cu_kv[-1] + prefix + length)
        groups = inputs.attention_inputs
        for group in groups.values():
            group.input_lengths = source_lengths
            group.prefix_lengths = source_prefixes
            group.cu_kv_seqlens_device = torch.tensor(
                cu_kv, device="cuda", dtype=torch.int32
            )
            group.context_total_kv_length = cu_kv[-1]
        inputs.attention_inputs = groups
        rows = sum(lengths)
        inputs.input_hiddens = (
            torch.arange(rows * 16, device="cuda")
            .remainder(11)
            .reshape(rows, 16)
            .to(torch.bfloat16)
        )
        inputs.combo_position_ids = torch.tensor(
            [prefix + i for length in lengths for i in range(length)],
            device="cuda",
            dtype=torch.int32,
        )
        return inputs

    def _assert_metadata_lease_output(self, runner, inputs):
        self.assertTrue(runner.canRun(inputs))
        actual = runner.forward(inputs).hidden_states.clone()
        torch.cuda.synchronize()
        rows = inputs.input_ids.numel()
        expected = inputs.input_hiddens.reshape(rows, 4, 4).sum(1)
        expected += (inputs.input_ids + inputs.combo_position_ids).to(expected.dtype)[
            :, None
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_source_kernel_rejects_corrupt_transport_before_delegate(self):
        from rtp_llm.models_py.kernels.cuda.qsa import draft_metadata_triton

        runner, model = self._metadata_lease_runner()
        original = draft_metadata_triton.replay_invalid_from_sources
        for defect in ("position", "cu", "state"):
            with self.subTest(defect=defect):

                def corrupt(*args, **kwargs):
                    # Inject after Base-RoPE canonicalization, into the actual
                    # captured buffers that the independent kernel validates.
                    if defect == "position":
                        args[5].data[0] += 1
                    elif defect == "cu":
                        args[6].data[1] += 1
                    else:
                        args[4].data[0, 0] = 0
                    return original(*args, **kwargs)

                before = model.delegate.prepare_cuda_graph.call_count
                with patch.object(
                    draft_metadata_triton, "replay_invalid_from_sources", corrupt
                ):
                    with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
                        runner.prepare(self._metadata_lease_inputs([2, 1]))
                self.assertEqual(model.delegate.prepare_cuda_graph.call_count, before)
                self._assert_metadata_lease_output(
                    runner, self._metadata_lease_inputs([3, 1], prefix=11)
                )

    def test_native_device_verdict_host_sources_keep_reference_path(self):
        runner, model = self._metadata_lease_runner()
        gpu_items = []
        item = torch.Tensor.item

        def checked_item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                gpu_items.append(1)
            return item(tensor, *args, **kwargs)

        inputs = self._metadata_lease_inputs([3, 1], device_first=False)
        with patch.object(torch.Tensor, "item", checked_item):
            self.assertTrue(runner.prepare(inputs))
        self.assertEqual(gpu_items, [1])
        self._assert_metadata_lease_output(runner, inputs)

    def test_native_device_verdict_readonly_result_does_not_extend_lease_lifetime(self):
        runner, model = self._metadata_lease_runner()
        observed = []
        native_matches = GraphReplayMetadataLease.matches

        def observe(lease, groups):
            observed.append((lease, groups))
            return native_matches(lease, groups)

        with patch.object(GraphReplayMetadataLease, "matches", observe):
            self.assertTrue(runner.prepare(self._metadata_lease_inputs([2, 1])))
        lease, groups = observed[-1]
        self.assertIs(lease.device_validation_invalid, False)
        with self.assertRaises(AttributeError):
            lease.device_validation_invalid = False
        self.assertFalse(lease.matches(groups))
        self.assertFalse(lease.consume(groups))
        before = model.delegate.prepare_cuda_graph.call_count
        with self.assertRaisesRegex(RuntimeError, "stale metadata lease"):
            model.replays[-1].prepare_cuda_graph(groups, metadata_lease=lease)
        self.assertEqual(model.delegate.prepare_cuda_graph.call_count, before)


if __name__ == "__main__":
    unittest.main()
