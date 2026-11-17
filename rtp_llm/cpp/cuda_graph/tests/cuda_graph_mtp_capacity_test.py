"""Exercise native draft graph layouts independently of hidden-state width."""

import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests import cuda_graph_tagged_cache_test as fixtures
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    draft_graph_rows,
)
from rtp_llm.ops.compute_ops import PyModelOutputs


class PackedDraftCapacityModel:
    def __init__(self, hc_mult=4):
        self.shapes = set()
        self.inputs_by_batch = {}
        self.hc_mult = hc_mult

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        batch = metadata.input_lengths_device.numel()
        self.shapes.add((rows, batch))
        self.inputs_by_batch[batch] = inputs
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


class MtpGraphCapacityTest(unittest.TestCase):
    def test_request_capacity_shrink_and_grow_with_variable_acceptance(self):
        # HC width and speculative request lanes are independent dimensions.
        for lanes, hc_mult in ((1, 4), (2, 4), (3, 1), (4, 4)):
            with self.subTest(lanes=lanes, hc_mult=hc_mult):
                self._check_request_capacity(lanes, hc_mult, [1, 2, 4, 8])

    def test_default_request_buckets_pad_intermediate_batches(self):
        self._check_request_capacity(3, 4, [])

    def _check_request_capacity(self, lanes, hc_mult, configured_buckets):
        torch.cuda.set_device(0)
        model = DraftCapacityModel(lanes, hc_mult)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, lanes, hc_mult, configured_buckets)
        buckets = (configured_buckets or [1, 8]) if torch.version.hip else list(range(1, 9))
        self.assertEqual(model.shapes, {(lanes * batch, batch) for batch in buckets})

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
            [1] * 5,
            [lanes] * 6,
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
                selected_batch = next(
                    batch for batch in buckets if batch >= len(lengths)
                )
                self.assertEqual(
                    runner.getCurrentRealGraphSize(), lanes * selected_batch
                )
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                captured = model.inputs_by_batch[selected_batch]
                captured_metadata = get_primary_attention_inputs(captured)
                padding = selected_batch - len(lengths)
                self.assertEqual(
                    captured_metadata.input_lengths_device.tolist(),
                    lengths + [0] * padding,
                )
                self.assertEqual(
                    captured_metadata.prefix_lengths_device.tolist(),
                    [7] * len(lengths) + [0] * padding,
                )
                expected_cu = [0]
                for length in lengths:
                    expected_cu.append(expected_cu[-1] + length)
                self.assertEqual(
                    captured_metadata.cu_seqlens_device.tolist(),
                    expected_cu + [rows] * padding,
                )
                self.assertEqual(
                    captured_metadata.kv_cache_kernel_block_id_device[len(lengths) :]
                    .count_nonzero()
                    .item(),
                    0,
                )
                self.assertEqual(captured.input_ids[rows:].count_nonzero().item(), 0)
                self.assertEqual(
                    captured.input_hiddens[rows:].count_nonzero().item(), 0
                )

        overflow = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 9)
        overflow.attention_inputs = get_primary_attention_inputs(overflow)
        self.assertFalse(runner.canRun(overflow))

    def test_undeclared_layout_keeps_packed_metadata_with_hc(self):
        torch.cuda.set_device(0)
        model = PackedDraftCapacityModel(hc_mult=4)
        runner = CudaGraphRunner()
        # A packed backend has not declared request-lane padding support; a
        # sparse decode configuration must not change its capture geometry.
        runner.init_mtp_draft_prefill(model, 8, 4, 3, 4, [1, 4, 8])
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
        with self.assertRaisesRegex(
            RuntimeError, "tokens per request must be positive"
        ):
            CudaGraphRunner().init_mtp_draft_prefill(DraftCapacityModel(), 8, 4, 0, 4)

    def test_request_bucket_outside_capacity_rejected_before_capture(self):
        for buckets in ([0, 8], [1, 9]):
            with self.subTest(buckets=buckets):
                with self.assertRaisesRegex(RuntimeError, "request bucket .* must be"):
                    CudaGraphRunner().init_mtp_draft_prefill(
                        DraftCapacityModel(3, 4), 8, 4, 3, 4, buckets
                    )


if __name__ == "__main__":
    unittest.main()
