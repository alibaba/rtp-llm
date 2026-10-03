import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)


class _NoopAttention:
    def prepare_cuda_graph(self, _attention_inputs):
        return None


class _RaggedEchoModel:
    """Small capturable model used to test graph geometry, not attention math."""

    def prepare_fmha_impl(self, _inputs, _capture):
        return _NoopAttention()

    def forward(self, inputs, _attention=None):
        if not inputs.attention_inputs.is_ragged_target_verify:
            raise AssertionError(
                "ragged target-verify flag was lost during capture/replay"
            )
        return PyModelOutputs(inputs.input_hiddens * 2)


class TestCudaGraphRaggedTargetVerify(unittest.TestCase):
    hidden_size = 32
    max_seq_len = 128
    tokens_per_block = 128
    compact_width = 5

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.runner = CudaGraphRunner()
        cls.runner.init_decode(
            _RaggedEchoModel(),
            cls.hidden_size,
            cls.max_seq_len,
            cls.tokens_per_block,
            cls.tokens_per_block,
            [2, 4],
            "bf16",
            cls.compact_width,
            True,
            True,
            True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.runner = None
        torch.cuda.synchronize()

    def _inputs(self, verify_lengths):
        batch_size = len(verify_lengths)
        total_tokens = sum(verify_lengths)
        lengths_cpu = torch.tensor(verify_lengths, dtype=torch.int32)
        cu_cpu = torch.cat(
            [torch.zeros(1, dtype=torch.int32), lengths_cpu.cumsum(0)]
        ).pin_memory()

        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(total_tokens, dtype=torch.int32, device="cuda")
        torch.manual_seed(total_tokens)
        inputs.input_hiddens = torch.randn(
            total_tokens,
            self.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        attn = PyAttentionInputs()
        attn.input_lengths = lengths_cpu.cuda()
        attn.prefix_lengths = torch.full(
            (batch_size,), 8, dtype=torch.int32, device="cuda"
        )
        attn.sequence_lengths = torch.full(
            (batch_size,), 8, dtype=torch.int32, device="cuda"
        )
        attn.sequence_lengths_plus_1_d = attn.sequence_lengths + attn.input_lengths
        attn.cu_seqlens_host = cu_cpu
        attn.cu_seqlens = cu_cpu.cuda()
        attn.cu_kv_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                (attn.prefix_lengths + attn.input_lengths).cumsum(0),
            ]
        )
        attn.decode_cu_seqlens_d = attn.cu_seqlens

        # max_kv_blocks = ceil(128 / 128) + (compact_width - 1) = 5.
        block_ids = torch.arange(
            1, batch_size * self.compact_width + 1, dtype=torch.int32, device="cuda"
        ).view(batch_size, self.compact_width)
        attn.kv_cache_kernel_block_id_device = block_ids
        attn.kv_cache_kernel_block_id_host = block_ids.cpu().pin_memory()
        attn.kv_cache_block_id_device = block_ids
        attn.kv_cache_block_id_host = attn.kv_cache_kernel_block_id_host

        attn.padding_offset = torch.zeros(total_tokens, dtype=torch.int32)
        attn.is_prefill = True
        attn.is_target_verify = True
        attn.is_ragged_target_verify = True
        attn.context_total_kv_length = total_tokens
        attn.total_tokens = total_tokens
        attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        inputs.attention_inputs = attn
        return inputs

    def _replay_and_check(self, verify_lengths):
        inputs = self._inputs(verify_lengths)
        expected = inputs.input_hiddens * 2
        self.assertTrue(self.runner.canRun(inputs))
        output = self.runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(output.hidden_states.shape[0], sum(verify_lengths))
        torch.testing.assert_close(output.hidden_states, expected)

    def test_same_total_different_request_lengths_replay(self):
        self._replay_and_check([5, 5, 5, 5])
        self._replay_and_check([1, 3, 8, 8])
        self._replay_and_check([8, 8, 3, 1])

    def test_missing_exact_batch_bucket_falls_back(self):
        self.assertFalse(self.runner.canRun(self._inputs([5, 5, 5])))

    def test_total_token_mismatch_falls_back(self):
        self.assertFalse(self.runner.canRun(self._inputs([4, 5, 5, 5])))


if __name__ == "__main__":
    unittest.main()
