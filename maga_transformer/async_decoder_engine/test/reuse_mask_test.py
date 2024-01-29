import torch
from typing import Any, List
from unittest import TestCase, main

from maga_transformer.async_decoder_engine.normal_model_executor import NormalModelExecutor

class ReuseMaskTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _create_fake_attention_mask(self, input_lengths: List[int]):
        batch_size = len(input_lengths)
        max_input_length = max(input_lengths)
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device='cpu')\
            .tril().unsqueeze(0)
        # attention_mask = ~attention_mask
        attention_mask = attention_mask.tile(batch_size, 1, 1).half()
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
        return attention_mask

    def test_simple(self):
        input_lengths = [3]
        reuse_lengths = [5]
        full_attn_mask = self._create_fake_attention_mask(input_lengths)
        final_mask = NormalModelExecutor.append_reuse_mask(full_attn_mask, input_lengths, reuse_lengths)
        self.assertEqual(final_mask.shape[1], 3)
        self.assertEqual(final_mask.shape[2], 8)
        self.assertEqual(final_mask.tolist(),
                         [[[1., 1., 1., 1., 1., 1., 0., 0.], 
                           [1., 1., 1., 1., 1., 1., 1., 0.], 
                           [1., 1., 1., 1., 1., 1., 1., 1.]]])

    def test_batch(self):
        input_lengths = [5, 2]
        reuse_lengths = [0, 5]
        full_attn_mask = self._create_fake_attention_mask(input_lengths)
        final_mask = NormalModelExecutor.append_reuse_mask(full_attn_mask, input_lengths, reuse_lengths)
        self.assertEqual(final_mask.shape[1], 5)
        self.assertEqual(final_mask.shape[2], 10)
        self.assertEqual(final_mask.tolist(), 
                         [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                          [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        print(final_mask.tolist())

if __name__ == '__main__':
    main()