"""Real-device backend selection regressions for CUDA13 Blackwell."""

import os
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
    XQAAttnOp,
    get_typemeta,
)


class KernelCapabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.geteuid() != 0
        assert torch.cuda.is_available()
        assert torch.version.cuda.startswith("13.")
        assert torch.cuda.get_device_capability()[0] == 10

    def config(self, head_dim):
        config = AttentionConfigs()
        config.head_num = 64
        config.kv_head_num = 8
        config.size_per_head = head_dim
        config.tokens_per_block = config.kernel_tokens_per_block = 128
        config.is_causal = True
        config.use_mla = False
        config.q_scaling = config.softmax_extra_scale = 1.0
        return config

    def inputs(self, prefill, query_length, prefix=0, cache=True):
        inputs = PyAttentionInputs()
        inputs.is_prefill = prefill
        inputs.dtype = get_typemeta(torch.empty(0, dtype=torch.bfloat16))
        inputs.input_lengths = torch.tensor([query_length], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([prefix], dtype=torch.int32)
        inputs.sequence_lengths = torch.tensor([7], dtype=torch.int32)
        if cache:
            inputs.kv_cache_kernel_block_id_device = torch.zeros(
                (1, 1), dtype=torch.int32, device="cuda"
            )
        return inputs

    def test_gen_rejects_unavailable_head_dimension(self):
        config = self.config(64)
        for prefill, query_length in ((False, 1), (True, 6), (True, 128)):
            with self.subTest(prefill=prefill, query_length=query_length):
                inputs = self.inputs(prefill, query_length)
                self.assertFalse(FlashInferTRTLLMDecodeOp(config).support(inputs))
                self.assertFalse(FlashInferTRTLLMPrefillOp(config).support(inputs))

    def test_gen_requires_cache_and_valid_query_rows(self):
        config = self.config(128)
        inputs = self.inputs(True, 6)
        self.assertTrue(FlashInferTRTLLMDecodeOp(config).support(inputs))
        self.assertTrue(FlashInferTRTLLMPrefillOp(config).support(inputs))
        inputs = self.inputs(True, 6, cache=False)
        self.assertFalse(FlashInferTRTLLMDecodeOp(config).support(inputs))
        self.assertFalse(FlashInferTRTLLMPrefillOp(config).support(inputs))
        inputs = self.inputs(True, 6)
        inputs.input_lengths = torch.empty(0, dtype=torch.int32)
        self.assertFalse(FlashInferTRTLLMDecodeOp(config).support(inputs))

    def test_native_trt_probe_matches_bundled_architecture(self):
        config = self.config(128)
        # Bundled SM100 TRT V2 entries are asymmetric 192/128 and 576/512;
        # this symmetric 128/128 shape is absent on both tested architectures.
        # The supported GEN 128 path is independently exercised above and by
        # trtllm_gen_test's real kernel cases.
        supported = False
        self.assertEqual(TRTAttnOp(config).support(self.inputs(True, 16)), supported)
        self.assertEqual(
            TRTPagedAttnOp(config).support(self.inputs(True, 16, prefix=16)),
            supported,
        )

    def test_sm90_xqa_is_not_selected_on_blackwell(self):
        self.assertFalse(XQAAttnOp(self.config(64)).support(self.inputs(False, 1)))


if __name__ == "__main__":
    unittest.main()
