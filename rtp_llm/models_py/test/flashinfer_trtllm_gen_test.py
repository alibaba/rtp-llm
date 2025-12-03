import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main
from unittest.mock import PropertyMock, patch

import torch

sys.path.append("/home/zw193905/RTP-LLM/github-opensource")
# CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(str(CUR_PATH), "../../../"))
device = torch.device(f"cuda")

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.mha.flashinfer_trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

class FlashInferPythonMHATest(TestCase):
    NUM_TOKENS = [7]
    HIDDEN_SIZES = [2048]
    PAGE_SIZE = [64]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)
        # add_setter_to_pybind_property(PyAttentionInputs, 'cu_seqlens')
        # add_setter_to_pybind_property(PyAttentionInputs, 'cu_kv_seqlens')
        # add_setter_to_pybind_property(PyAttentionInputs, 'kv_cache_block_id_device')

        self.num_pages = 1024
        self.page_size = 64
        self.head_dim = 128
        self.num_kv_heads = 8
        self.num_heads = 96
        self.k_cache = torch.randn(
            self.num_pages,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            dtype=torch.float16,
            device="cuda:0",
        ).to(torch.float8_e4m3fn)
        self.v_cache = torch.randn(
            self.num_pages,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            dtype=torch.float16,
            device="cuda:0",
        ).to(torch.float8_e4m3fn)

    def gen_attention_inputs(
        self,
        input_lengths: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None,
    ) -> PyAttentionInputs:
        assert not (input_lengths is None and sequence_lengths is None)
        attention_inputs: PyAttentionInputs = PyAttentionInputs()
        batch_size: int = 0
        max_seq_len: int = 0
        if sequence_lengths is not None:
            batch_size = len(sequence_lengths)
            attention_inputs.sequence_lengths = torch.tensor(
                sequence_lengths, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            max_seq_len = attention_inputs.sequence_lengths.max().item()
            attention_inputs.is_prefill = False
        if input_lengths is not None:
            batch_size = len(input_lengths)
            attention_inputs.input_lengths = torch.tensor(
                input_lengths, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            attention_inputs.is_prefill = True
            cu_seqlens = torch.zeros(
                len(input_lengths) + 1, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            cu_seqlens[1:] = attention_inputs.input_lengths.cumsum(0)
            # import pdb
            # pdb.set_trace()
            attention_inputs.cu_seqlens = cu_seqlens
            attention_inputs.cu_kv_seqlens = cu_seqlens
            max_seq_len = attention_inputs.input_lengths.max().item()
        max_block_size = max_seq_len // self.page_size + 1
        assert batch_size * max_block_size < self.page_size
        block_tables = (
            torch.arange(
                batch_size * max_block_size,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            .view(batch_size, max_block_size)
            .pin_memory()
        )
        attention_inputs.kv_cache_block_id_device = block_tables
        return attention_inputs

    def test_run_flashinfer_prefill_test(self):
        input_lengths = [2, 3, 10, 12]
        num_tokens = sum(input_lengths)

        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim

        attn_inputs = self.gen_attention_inputs(input_lengths=input_lengths)
        q = torch.randn(
            [num_tokens, config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache
        op = FlashInferTRTLLMPrefillOp(config)
        input_params = op.prepare(attn_inputs)
        # impl = FlashInferPythonPrefillImpl(config, attn_inputs)
        # print('kk:', flush=True)
        out = op.forward(q, kv_cache, input_params)
        print(out)
        # self.assertTrue(torch.allclose(out, out_ref, atol=1, rtol=1))

    def test_run_flashinfer_decode_test(self):
        sequence_lengths = [2, 3, 10, 12]
        num_tokens = len(sequence_lengths)
        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim

        attn_inputs = self.gen_attention_inputs(sequence_lengths=sequence_lengths)
        q = torch.randn(
            [num_tokens, config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )
        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache
        print("gg:", flush=True)
        op = FlashInferTRTLLMDecodeOp(config)
        input_params = op.prepare(attn_inputs)
        # impl = FlashInferPythonPrefillImpl(config, attn_inputs)
        # print('kk:', flush=True)
        out = op.forward(q, kv_cache, input_params)
        print(out)
        # impl = FlashInferPythonDecodeImpl(config, attn_inputs)
        # out = impl.forward(q, kv_cache)
        # print(out)


if __name__ == "__main__":
    main()
