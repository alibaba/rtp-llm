import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.ops import fused_rope_kvcache_op as op


class PositionCompatibilityTest(unittest.TestCase):

    def setUp(self):
        op._prefill_position_argument.cache_clear()

    def tearDown(self):
        op._prefill_position_argument.cache_clear()

    def forward(self, kernel, positions):
        rope = SimpleNamespace(**{
            k: 0 for k in [
                'style', 'dim', 'base', 'scale', 'factor1', 'factor2', 'max_pos',
                'extrapolation_factor', 'mscale', 'offset', 'index_factor',
                'mrope_dim1', 'mrope_dim2', 'mrope_dim3',
            ]
        })
        config = SimpleNamespace(
            rope_config=rope, max_seq_len=32, head_num=2, kv_head_num=2,
            size_per_head=8, kernel_tokens_per_block=16, use_logn_attn=False,
        )
        params = SimpleNamespace(
            position_ids=positions, cu_seqlens=torch.tensor([0, 3]), max_seq_len=3,
            kv_cache_offset=None, kv_cache_offset_h=None, padding_offset=None,
            prefix_lengths=torch.tensor([0]), max_prefix_length=0,
        )
        qkv = torch.zeros(3, 48)
        with (
            patch.object(op, '_get_fused_rope_kvcache',
                         return_value=SimpleNamespace(prefill_fused_rope_kvcache=kernel)),
            patch.object(op, 'get_rope_cache_once', return_value=None),
            patch.object(op, 'check_rope_cache', return_value=False),
        ):
            result = op.FusedRopeKVCachePrefillOpQKVOut(config).forward(qkv, None, params)
        self.assertIs(result, qkv)

    def test_new_api_preserves_multiaxis_positions(self):
        ids = torch.arange(9).reshape(3, 3)

        def kernel(qkv, *args, position_ids=None, **kwargs):
            self.assertIs(position_ids, ids)
            self.assertNotIn('cp_position_ids', kwargs)
            return qkv
        self.forward(kernel, ids)

    def test_old_api_preserves_1d_positions(self):
        ids = torch.arange(3)

        def kernel(qkv, *args, cp_position_ids=None, **kwargs):
            self.assertIs(cp_position_ids, ids)
            self.assertNotIn('position_ids', kwargs)
            return qkv
        self.forward(kernel, ids)

    def test_old_api_preserves_none(self):

        def kernel(qkv, *args, cp_position_ids=None, **kwargs):
            self.assertIsNone(cp_position_ids)
            self.assertNotIn('position_ids', kwargs)
            return qkv
        self.forward(kernel, None)

    def test_old_api_rejects_multiaxis_before_kernel(self):

        def kernel(qkv, *args, cp_position_ids=None, **kwargs):
            self.fail('Unsupported kernel must not execute')
        with self.assertRaisesRegex(RuntimeError, 'multi-axis'):
            self.forward(kernel, torch.zeros(3, 3))

    def test_signature_checked_once(self):

        def kernel(position_ids=None):
            pass
        with (
            patch.object(op, '_get_fused_rope_kvcache',
                         return_value=SimpleNamespace(prefill_fused_rope_kvcache=kernel)),
            patch.object(op.inspect, 'signature', wraps=inspect.signature) as sig,
        ):
            for _ in range(4):
                self.assertEqual(op._prefill_position_kwargs(None), {'position_ids': None})
            self.assertEqual(sig.call_count, 1)

    def test_unknown_api_fails_explicitly(self):

        def kernel(qkv):
            pass
        with patch.object(
            op, '_get_fused_rope_kvcache',
            return_value=SimpleNamespace(prefill_fused_rope_kvcache=kernel),
        ):
            with self.assertRaisesRegex(RuntimeError, 'no supported position'):
                op._prefill_position_kwargs(None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
