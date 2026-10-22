"""Exercise the native HC draft graph with variable request and packed lengths."""

import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests import cuda_graph_tagged_cache_test as fixtures
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    draft_graph_rows,
)
from rtp_llm.ops.compute_ops import PyModelOutputs


class DraftCapacityModel:
    def __init__(self):
        self.shapes = set()

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        batch = metadata.input_lengths_device.numel()
        assert rows == batch * 4
        self.shapes.add((rows, batch))
        _, _, _, positions, _ = draft_graph_rows(metadata, rows)
        hidden = inputs.input_hiddens.reshape(rows, 4, 4).sum(1)
        output = hidden + (positions + inputs.input_ids).to(hidden.dtype)[:, None]
        return PyModelOutputs(output)


class MtpGraphCapacityTest(unittest.TestCase):
    def test_request_capacity_shrink_and_grow_with_variable_acceptance(self):
        torch.cuda.set_device(0)
        model = DraftCapacityModel()
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4)
        self.assertEqual(model.shapes, {(4 * batch, batch) for batch in range(1, 9)})

        # The same graph keys are revisited after both larger and smaller
        # batches. Eight one-token requests must select 32 lanes, not 8.
        for lengths in (
            [4] * 8,
            [1],
            [4],
            [1] * 8,
            [4, 1, 3],
            [1, 1],
            [2],
            [3] * 7,
            [1],
        ):
            with self.subTest(lengths=lengths):
                rows = sum(lengths)
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                # A single-group graph uses the anchor ABI, not a one-entry tag map.
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                inputs.input_hiddens = (
                    torch.arange(rows * 16, device="cuda", dtype=torch.float32)
                    .remainder(11)
                    .reshape(rows, 16)
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
                expected = inputs.input_hiddens.reshape(rows, 4, 4).sum(1)
                expected = (
                    expected
                    + (positions + inputs.input_ids).to(expected.dtype)[:, None]
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 4 * len(lengths))
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

        overflow = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 9)
        overflow.attention_inputs = get_primary_attention_inputs(overflow)
        self.assertFalse(runner.canRun(overflow))


if __name__ == "__main__":
    unittest.main()
