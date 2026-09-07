"""Single-pass UQI attention versus an independent FP32 dense reference."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi import BertUqiAttention
from rtp_llm.ops import AttentionConfigs


class BertUqiAttentionTest(unittest.TestCase):
    def test_ragged_masks_and_wrapper_reuse(self):
        self.assertTrue(torch.cuda.is_available(), "requires CUDA")
        cases = [
            ([12], [(4, 3)]),
            ([9, 7, 15], [(3, 4), None, (1, 5)]),
            ([64, 31, 48, 17], [(55, 4), (2, 8), None, (7, 6)]),
            ([8, 5], [None, None]),
            ([12], [(0, 3)]),
        ]
        for dtype in (torch.float16, torch.bfloat16):
            config = AttentionConfigs()
            config.dtype = dtype
            config.head_num = config.kv_head_num = 4
            config.size_per_head = 64
            config.is_causal = config.need_rope_kv_cache = False
            config.q_scaling = 1.0
            config.softmax_extra_scale = 0.73
            attention = BertUqiAttention(config)
            wrapper = attention.op.prefill_wrapper
            for lengths, spans in cases:
                with self.subTest(dtype=dtype, lengths=lengths):
                    masks = []
                    for length, span in zip(lengths, spans):
                        profile = torch.zeros(length, dtype=torch.bool)
                        if span is not None:
                            start, count = span
                            profile[start : start + count] = True
                        masks.append(profile[:, None] | ~profile[None, :])
                    inputs = SimpleNamespace(
                        input_lengths=torch.tensor(lengths, dtype=torch.int32),
                        cu_seqlens_device=torch.tensor(
                            [0] + lengths, dtype=torch.int32, device="cuda"
                        ).cumsum(0, dtype=torch.int32),
                        bert_uqi_mask=torch.cat([mask.flatten() for mask in masks]).pin_memory(),
                    )
                    torch.manual_seed(sum(lengths))
                    qkv = torch.randn(sum(lengths), 3 * 4 * 64, dtype=dtype, device="cuda") / 3
                    q, k, v = [x.reshape(-1, 4, 64).float() for x in qkv.split(4 * 64, -1)]
                    reference, offset = [], 0
                    for length, mask in zip(lengths, masks):
                        end = offset + length
                        logits = torch.einsum("qhd,khd->hqk", q[offset:end], k[offset:end])
                        logits *= 0.73 / math.sqrt(64)
                        logits.masked_fill_(~mask.to("cuda")[None], float("-inf"))
                        reference.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v[offset:end]))
                        offset = end
                    attention.prepare(inputs, torch.device("cuda"))
                    self.assertIs(attention.op.prefill_wrapper, wrapper)
                    with patch.object(wrapper, "run", wraps=wrapper.run) as run:
                        output = attention.forward(qkv, None)
                        self.assertEqual(run.call_count, 1)
                    torch.testing.assert_close(
                        output.float(), torch.cat(reference),
                        atol=2e-2 if dtype == torch.bfloat16 else 3e-3, rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
