"""Non-divisible TP layouts must preserve routing and shared-expert summation."""

import unittest

import torch
from rtp_llm.models_py.distributed import tp_token_shard as m


class TokenShardTest(unittest.TestCase):
    def test_token_order_padding_and_shared_tp_sum(self):
        for tokens in [0, 1, 7, 8, 9, 31, 32, 33, 129, 16512 * 8 + 1]:
            with self.subTest(tokens=tokens):
                x = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
                ids = (
                    torch.arange(tokens * 2, dtype=torch.int64).reshape(tokens, 2) % 17
                )
                weights = torch.arange(tokens * 2, dtype=torch.float32).reshape(
                    tokens, 2
                ) / max(tokens, 1)

                def routed(h, w, i):
                    return h * ((i + 1) * w).sum(-1, keepdim=True)

                expected = routed(x, weights, ids)
                shards = [
                    m.slice_routed_tokens(x, weights, ids, r, 8) for r in range(8)
                ]
                self.assertEqual(len({s[0].shape[0] for s in shards}), 1)
                outputs = [routed(*s[:3]) for s in shards]
                gathered = torch.cat(outputs, dim=0)
                for out in outputs:
                    torch.testing.assert_close(
                        m.gather_routed_tokens(out, tokens, lambda _: gathered),
                        expected,
                        rtol=0,
                        atol=0,
                    )
                # Shared experts process the full token set on each TP rank,
                # then sum their distinct weight partitions exactly once.
                shared = sum(x * (r + 1) for r in range(8))
                torch.testing.assert_close(
                    gathered[:tokens] + shared, expected + x * 36, rtol=0, atol=0
                )
                self.assertTrue(
                    torch.equal(gathered[tokens:], torch.zeros_like(gathered[tokens:]))
                )

    def test_invalid_layout(self):
        x = torch.ones(2, 4)
        w = torch.ones(2, 2)
        i = torch.ones(2, 2, dtype=torch.int64)
        for rank, size in [(-1, 8), (8, 8), (0, 0)]:
            with self.assertRaises(ValueError):
                m.slice_routed_tokens(x, w, i, rank, size)
        with self.assertRaises(ValueError):
            m.slice_routed_tokens(x, w[:1], i, 0, 8)


if __name__ == "__main__":
    unittest.main()
