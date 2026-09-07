import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.utils.bert_user_profile import (
    build_bert_uqi_flashinfer_mask,
    derive_bert_uqi_segment_ids,
)
from rtp_llm.utils.sm120_fp8_backend import resolve_sm120_fp8_backend
from rtp_llm.utils.tensor_utils import bert_dual_head_scores


class BertUserProfileTest(unittest.TestCase):
    def test_ragged_segments_include_vision_in_qi(self):
        ids = torch.tensor([101, 102, 2, 17, 102, -1, -2, 101, 102, -1])
        cu = torch.tensor([0, 7, 10])
        seg = derive_bert_uqi_segment_ids(ids, cu)
        self.assertEqual(seg.tolist(), [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        mask = build_bert_uqi_flashinfer_mask(seg, cu)
        expected = torch.ones(7, 7, dtype=torch.bool)
        expected[[0, 1, 5, 6], 2:5] = False
        torch.testing.assert_close(mask[:49].reshape(7, 7), expected)
        self.assertTrue(mask[49:].all())

    def test_custom_marker_missing_sep_and_empty_batch(self):
        ids = torch.tensor([101, 3, 9, 101, 102])
        cu = torch.tensor([0, 3, 3, 5])
        seg = derive_bert_uqi_segment_ids(ids, cu, b_start_token_id=3)
        self.assertEqual(seg.tolist(), [0, 1, 1, 0, 0])
        self.assertEqual(build_bert_uqi_flashinfer_mask(seg, cu).numel(), 13)
        empty = derive_bert_uqi_segment_ids(ids[:0], torch.tensor([0]))
        self.assertEqual(
            build_bert_uqi_flashinfer_mask(empty, torch.tensor([0])).numel(), 0
        )

    def test_random_ragged_masks_match_explicit_visibility(self):
        generator = torch.Generator().manual_seed(19)
        for _ in range(30):
            lengths = torch.randint(1, 16, (5,), generator=generator)
            cu = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
            ids = torch.randint(10, 90, (int(cu[-1]),), generator=generator)
            expected_segments = []
            for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
                expected = [0] * (end - start)
                if end - start >= 4:
                    ids[start + 1] = 2
                    ids[end - 2] = 102
                    expected[1 : end - start - 1] = [1] * (end - start - 2)
                expected_segments.extend(expected)
            seg = derive_bert_uqi_segment_ids(ids, cu)
            self.assertEqual(seg.tolist(), expected_segments)
            expected_mask = []
            for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
                for i in range(start, end):
                    for j in range(start, end):
                        expected_mask.append(
                            expected_segments[i] == 1 or expected_segments[j] == 0
                        )
            self.assertEqual(
                build_bert_uqi_flashinfer_mask(seg, cu).tolist(), expected_mask
            )

    def _check_attention_and_heads(self, device):
        torch.manual_seed(7)
        ids = torch.tensor([101, 11, 102, 2, 25, 102, -1], device=device)
        cu = torch.tensor([0, 7], device=device)
        seg = derive_bert_uqi_segment_ids(ids, cu)
        mask = build_bert_uqi_flashinfer_mask(seg, cu).reshape(7, 7)
        hidden = torch.randn(7, 16, device=device)
        changed = hidden.clone()
        changed[3:6] += torch.randn(3, 16, device=device) * 3

        def encode(x):
            for _ in range(3):
                attention = F.scaled_dot_product_attention(
                    x[None, None], x[None, None], x[None, None], attn_mask=mask
                )[0, 0]
                x = F.layer_norm(x + attention, (16,))
            return x

        left, right = encode(hidden), encode(changed)
        torch.testing.assert_close(left[seg == 0], right[seg == 0], rtol=0, atol=0)
        self.assertGreater((left[3] - right[3]).abs().max().item(), 0.01)
        qi, uqi = torch.nn.Linear(16, 4).to(device), torch.nn.Linear(16, 4).to(device)
        scores = bert_dual_head_scores(
            left, ids, torch.tensor([7], device=device), qi, uqi
        )
        other = bert_dual_head_scores(
            right, ids, torch.tensor([7], device=device), qi, uqi
        )
        self.assertEqual(scores.shape, (1, 8))
        torch.testing.assert_close(scores[:, :4], other[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(
            scores.reshape(2, 4).sum(-1), torch.ones(2, device=device)
        )
        torch.testing.assert_close(scores[:, 4:], torch.softmax(uqi(left[3:4]), -1))

    def test_attention_isolation_and_dual_heads_cpu(self):
        self._check_attention_and_heads("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_attention_isolation_and_dual_heads_cuda(self):
        self._check_attention_and_heads("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_flashinfer_custom_mask_matches_torch(self):
        try:
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        except ImportError:
            self.skipTest("FlashInfer unavailable")
        ids = torch.tensor([101, 11, 102, 2, 25, 102, -1, 101, 102], device="cuda")
        cu = torch.tensor([0, 7, 9], dtype=torch.int32, device="cuda")
        mask = build_bert_uqi_flashinfer_mask(derive_bert_uqi_segment_ids(ids, cu), cu)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        )
        wrapper.plan(
            cu,
            cu,
            2,
            2,
            64,
            custom_mask=mask,
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        q, k, v = [
            torch.randn(9, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        out = wrapper.run(q, k, v)
        for start, end, offset in [(0, 7, 0), (7, 9, 49)]:
            ref = F.scaled_dot_product_attention(
                q[start:end].float().transpose(0, 1),
                k[start:end].float().transpose(0, 1),
                v[start:end].float().transpose(0, 1),
                attn_mask=mask[offset : offset + (end - start) ** 2].reshape(
                    end - start, end - start
                ),
            ).transpose(0, 1)
            torch.testing.assert_close(
                out[start:end].float(), ref, rtol=0.02, atol=0.02
            )

    def test_dual_head_ragged_pooling_and_missing_marker(self):
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        ids = torch.tensor([101, 2, 102, 101, 17, 102])
        qi, uqi = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        scores = bert_dual_head_scores(hidden, ids, torch.tensor([3, 3]), qi, uqi)
        torch.testing.assert_close(scores[:, :4], torch.softmax(qi(hidden[[0, 3]]), -1))
        torch.testing.assert_close(
            scores[:, 4:], torch.softmax(uqi(hidden[[1, 3]]), -1)
        )

    def test_default_backend_preserves_deepgemm(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_sm120_fp8_backend(), "deepgemm")


if __name__ == "__main__":
    unittest.main()
