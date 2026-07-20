import logging
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferRaggedPrefillSm120(BaseAttentionTest):
    def test_non_causal_prefill_without_kv_block_table(self):
        """Test encoder-only ragged prefill with no paged KV block table."""
        logging.info("\n=== Testing non-causal prefill without KV block table ===")

        sequence_lengths = [10, 20]
        batch_size = len(sequence_lengths)
        config = self._create_config(
            head_num=8,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
        )
        config.attn_configs.is_causal = False

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size,
            sequence_lengths,
            config.seq_size_per_block,
            with_kv_cache_block_ids=False,
        )

        attn_op = PyFlashinferPrefillAttnOp(config.attn_configs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        self.assertTrue(attn_op.support(attn_inputs))
        params = attn_op.prepare(attn_inputs)
        self.assertIsNotNone(params)

        total_tokens = sum(sequence_lengths)
        hidden_size_q = config.size_per_head * config.head_num
        hidden_size_k = config.size_per_head * config.head_num_kv
        hidden_size_v = config.size_per_head * config.head_num_kv

        qkv = torch.randn(
            total_tokens,
            hidden_size_q + hidden_size_k + hidden_size_v,
            dtype=torch.float16,
            device=self.device,
        )
        q_flat, k_flat, v_flat = torch.split(
            qkv,
            [hidden_size_q, hidden_size_k, hidden_size_v],
            dim=-1,
        )
        q = q_flat.reshape(total_tokens, config.head_num, config.size_per_head)
        k = k_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)

        output = attn_op.forward(qkv, None)
        ref_output = compute_flashinfer_prefill_reference(
            q, k, v, attn_inputs.cu_seqlens_device, causal=False
        )

        compare_tensors(
            output,
            ref_output,
            rtol=1e-2,
            atol=1e-2,
            name="Non-causal prefill output without KV block table",
        )


if __name__ == "__main__":
    unittest.main()
