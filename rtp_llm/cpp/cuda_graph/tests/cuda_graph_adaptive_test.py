import unittest

import torch
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


class Attention:
    def prepare_cuda_graph(self, inputs):
        pass


class ShapeRecordingModel:
    def __init__(self):
        self.captured_rows = []

    def prepare_fmha_impl(self, inputs, *args):
        return Attention()

    def forward(self, inputs, attn=None):
        rows = inputs.input_hiddens.shape[0]
        if torch.cuda.is_current_stream_capturing():
            self.captured_rows.append(rows)
        result = PyModelOutputs()
        result.hidden_states = inputs.input_hiddens + rows
        return result


def make_inputs(lengths, value):
    rows = sum(lengths)
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    inputs.input_hiddens = torch.full(
        (rows, 8), value, device="cuda", dtype=torch.float16
    )
    attn = PyAttentionInputs()
    attn.is_prefill = True
    attn.is_target_verify = True
    attn.input_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    attn.prefix_lengths = torch.full_like(attn.input_lengths, 8)
    attn.sequence_lengths = torch.empty(0, device="cuda", dtype=torch.int32)
    attn.cu_seqlens_host = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    )
    attn.cu_seqlens = attn.cu_seqlens_host.cuda()
    attn.cu_kv_seqlens = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            (attn.input_lengths + 8).cumsum(0).int(),
        )
    )
    attn.kv_cache_kernel_block_id_host = torch.ones(
        (len(lengths), 1), dtype=torch.int32
    )
    attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id_host.cuda()
    attn.kv_cache_block_id_host = attn.kv_cache_kernel_block_id_host
    attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
    attn.padding_offset = torch.zeros(rows, device="cuda", dtype=torch.int32)
    attn.total_tokens = rows
    attn.context_total_kv_length = rows + 8 * len(lengths)
    attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))
    inputs.attention_inputs = attn
    return inputs


class AdaptiveGraphTest(unittest.TestCase):
    def test_short_graph_and_ragged_replay(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = ShapeRecordingModel()
                runner = CudaGraphRunner()
                runner.init_decode(model, 8, 64, 64, 64, [1, 4], 6, enabled)
                self.assertEqual(len(model.captured_rows), 12 if enabled else 2)
                for value, lengths in enumerate(
                    ([4], [1, 3], [6, 1, 2], [1], [6], [4]), 1
                ):
                    inputs = make_inputs(lengths, value)
                    self.assertTrue(runner.canRun(inputs))
                    result = runner.forward(inputs).hidden_states.clone()
                    batch = 1 if len(lengths) == 1 else 4
                    capacity = batch * (
                        (sum(lengths) + batch - 1) // batch if enabled else 6
                    )
                    torch.testing.assert_close(
                        result[: sum(lengths)],
                        torch.full_like(inputs.input_hiddens, value + capacity),
                        rtol=0,
                        atol=0,
                    )
                del runner


if __name__ == "__main__":
    unittest.main()
