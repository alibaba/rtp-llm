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


class DraftHostPageTableModel(PackedDraftCapacityModel):
    def __init__(self):
        super().__init__()
        self.page_tables = {}

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return self

    def prepare_cuda_graph(self, inputs_by_tag):
        self.page_tables = {
            tag: inputs.kv_cache_kernel_block_id.clone()
            for tag, inputs in inputs_by_tag.items()
        }


class MtpGraphCapacityTest(unittest.TestCase):
    def test_draft_prepare_refreshes_strided_host_page_tables(self):
        tags = ["full", "aux"]
        model = DraftHostPageTableModel()
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 4, 4, 3, 4, tags, 4)

        # Reuse capture buffers with changing widths and physical row pitches.
        # The last iteration exercises the generic copy for a strided column view.
        for width, aux_col_stride in [(4, 1), (2, 1), (6, 2)]:
            with self.subTest(width=width, aux_col_stride=aux_col_stride):
                inputs = fixtures._build_prefill_inputs(
                    tags, {"full": 1, "aux": 2}, [2, 1]
                )
                inputs.combo_position_ids = torch.tensor(
                    [0, 1, 0], device="cuda", dtype=torch.int32
                )
                inputs.input_hiddens = torch.ones(
                    (3, 16), device="cuda", dtype=torch.bfloat16
                )
                groups = inputs.attention_inputs
                expected = {}
                for offset, (tag, attention) in enumerate(groups.items()):
                    col_stride = aux_col_stride if tag == "aux" else 1
                    pitch = width * col_stride + 3
                    values = torch.arange(2 * pitch, dtype=torch.int32).reshape(
                        2, pitch
                    )
                    values += 1 + offset * 100
                    host_table = values[:, : width * col_stride : col_stride]
                    table = values.cuda()[:, : width * col_stride : col_stride]
                    attention.kv_cache_kernel_block_id = table
                    attention.kv_cache_kernel_block_id_device = table.contiguous()
                    expected[tag] = torch.zeros((4, 4), dtype=torch.int32)
                    live_width = min(width, 4)
                    expected[tag][:2, :live_width] = host_table[:, :live_width]
                inputs.attention_inputs = groups

                self.assertTrue(runner.prepare(inputs))
                for tag in tags:
                    torch.testing.assert_close(model.page_tables[tag], expected[tag])
                runner.forward(inputs)

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


if __name__ == "__main__":
    unittest.main()
