"""Exercise Engram history copies through the real C++ CUDA graph runner."""

import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.cuda_graph_test_utils import SyntheticCudaGraphModel
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs


class HistoryModel(SyntheticCudaGraphModel):
    def cuda_graph_engram_window_size(self):
        return 4

    def forward(self, inputs, fmha_impl=None):
        windows = inputs.engram_token_windows
        if windows.device != inputs.input_ids.device:
            raise AssertionError("The graph runner must supply device token history")
        # -1 padding contributes zero. The batch-wide signal makes stale rows
        # observable even though the runner returns only live output rows.
        rows = (windows.float() + 1).sum(-1, keepdim=True)
        signal = (rows + rows.sum()) / 128
        return PyModelOutputs(signal.expand(-1, self.hidden_size).half().contiguous())


class TestCudaGraphEngramHistory(unittest.TestCase):
    def _inputs(self, batch_size, q_len, offset):
        inputs = PyModelInputs()
        n_tokens = batch_size * q_len
        inputs.input_ids = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.zeros(
            (n_tokens, 8), dtype=torch.float16, device="cuda"
        )
        windows = torch.arange(n_tokens * 4, dtype=torch.int32).view(n_tokens, 4)
        windows = windows + offset
        windows[0, 2:] = -1
        inputs.engram_token_windows = windows.cuda()

        attn = PyAttentionInputs()
        attn.is_prefill = q_len > 1
        attn.is_target_verify = q_len > 1
        attn.input_lengths = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device="cuda"
        )
        attn.sequence_lengths = torch.full(
            (batch_size,), 10, dtype=torch.int32
        ).pin_memory()
        if q_len > 1:
            attn.prefix_lengths = attn.sequence_lengths.cuda()
        attn.sequence_lengths_plus_1_d = attn.sequence_lengths.cuda() + q_len
        attn.decode_cu_seqlens_d = (
            torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * q_len
        )
        attn.cu_seqlens_host = (
            torch.arange(batch_size + 1, dtype=torch.int32).pin_memory() * q_len
        )
        attn.cu_seqlens = attn.cu_seqlens_host.cuda()
        attn.cu_kv_seqlens = (
            attn.cu_seqlens
            + torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * 10
        )
        block_ids = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device="cuda"
        ).view(-1, 1)
        attn.kv_cache_kernel_block_id_device = block_ids
        attn.kv_cache_kernel_block_id_host = block_ids.cpu().pin_memory()
        attn.kv_cache_block_id_device = block_ids
        attn.kv_cache_block_id_host = attn.kv_cache_kernel_block_id_host
        attn.padding_offset = torch.zeros(n_tokens, dtype=torch.int32, device="cuda")
        attn.total_tokens = n_tokens
        attn.context_total_kv_length = batch_size * (10 + q_len)
        inputs.attention_inputs = attn
        return inputs

    def _check_replays(self, q_len):
        model = HistoryModel(8, torch.float16, "cuda:0")
        runner = CudaGraphRunner()
        runner.init_decode(model, 8, 64, 64, 64, [2, 4], q_len)
        # Revisit one capture with fewer live rows, then change both the bucket
        # and device source allocation. D2D staging must update the stable
        # graph buffer; neither history nor inactive rows may stick.
        for batch_size, offset in (
            (4, 11),
            (3, 101),
            (1, 201),
            (2, 301),
            (1, 401),
        ):
            with self.subTest(q_len=q_len, batch_size=batch_size, offset=offset):
                inputs = self._inputs(batch_size, q_len, offset)
                self.assertTrue(runner.canRun(inputs))
                actual = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                expected = model(inputs).hidden_states
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_single_token_decode_history_refresh(self):
        self._check_replays(q_len=1)

    def test_dspark_target_verify_history_refresh(self):
        self._check_replays(q_len=6)


if __name__ == "__main__":
    unittest.main()
