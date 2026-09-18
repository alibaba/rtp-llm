import unittest

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCHead, DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import get_shared_expert_executor


class DelayedHCTest(unittest.TestCase):
    @staticmethod
    def unit(pre_bias):
        base = torch.zeros(24)
        base[:4] = torch.tensor(pre_bias)
        return DelayedHCUnit(
            torch.zeros(24, 12),
            base,
            torch.ones(3),
            dim=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

    def test_entry_uses_identity_and_next_sublayer_uses_previous_mix(self):
        first = self.unit([-3.0, -1.0, 1.0, 3.0])
        second = self.unit([9.0, 9.0, 9.0, 9.0])
        second.set_previous(first)
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        first_input, post, comb = first.pre(x)
        torch.testing.assert_close(first_input, x[:, 0])
        residual = first.post(torch.ones(2, 3), x, post, comb)
        second_input, _, _ = second.pre(residual)
        expected = (residual * first.pre_mix_out.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(second_input, expected)
        own_mix_readout = (residual * second.pre_mix_out.unsqueeze(-1)).sum(-2)
        self.assertFalse(torch.allclose(second_input, own_mix_readout))

    def test_head_consumes_final_ffn_mix_for_batched_verify(self):
        final = self.unit([-2.0, 0.0, 2.0, 4.0])
        head = DelayedHCHead(final)
        x = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3)
        _, post, comb = final.pre(x)
        residual = final.post(torch.ones(2, 3, 3), x, post, comb)
        expected = (residual * final.pre_mix_out.unsqueeze(-1)).sum(-2)
        torch.testing.assert_close(head.head(residual), expected)
        self.assertEqual(tuple(head.head(residual).shape), (2, 3, 3))
        self.assertEqual(len(list(head.parameters())), 0)

    def test_new_request_replaces_previous_token_layout(self):
        first, second = self.unit([0.0] * 4), self.unit([2.0] * 4)
        second.set_previous(first)
        for tokens in (7, 1, 3):
            x = torch.ones(tokens, 4, 3)
            first.pre(x)
            y, _, _ = second.pre(x)
            torch.testing.assert_close(y, torch.full((tokens, 3), 2.000004))

    def test_loader_scale_column_is_normalized_before_execution(self):
        expected = self.unit([-3.0, -1.0, 1.0, 3.0])
        loaded = DelayedHCUnit(
            expected.fn,
            expected.base,
            expected.scale.unsqueeze(-1),
            dim=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        self.assertEqual(tuple(loaded.scale.shape), (3,))
        self.assertEqual(loaded.scale.data_ptr(), expected.scale.data_ptr())
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        for actual, reference in zip(loaded.pre(x), expected.pre(x)):
            torch.testing.assert_close(actual, reference)

    def test_mxfp8_shared_executor_keeps_native_linears(self):
        executor = get_shared_expert_executor(native_mxfp8=True)
        shared = torch.nn.Linear(3, 3, bias=False)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        executor.prepare(shared)
        executor.start(shared, x)
        torch.testing.assert_close(executor.finish(), shared(x))
        self.assertIsNone(executor._fast_path)
        executor.start(shared, x[:0])
        self.assertEqual(tuple(executor.finish().shape), (0, 3))


if __name__ == "__main__":
    unittest.main()
