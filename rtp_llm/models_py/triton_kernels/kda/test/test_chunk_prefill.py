"""Shared cuLA Prefill: ragged prefixes, page checkpoints and virtual rows."""

import math
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    cula_kda_paged_prefill,
)


def recurrent_reference(q, k, v, gate, beta, initial, a_log, dt_bias, prefix):
    q = F.normalize(q.float(), dim=-1) / math.sqrt(q.shape[-1])
    k = F.normalize(k.float(), dim=-1)
    state = initial.float().clone()
    beta = beta.float().sigmoid()
    output, checkpoints = [], {}
    for token in range(q.shape[0]):
        decay = -a_log.float().exp()[:, None] * F.softplus(
            gate[token].float() + dt_bias.float().reshape(gate.shape[1:])
        )
        state *= decay.exp().unsqueeze(-1)
        predicted = (state * k[token, :, :, None]).sum(-2)
        correction = (v[token].float() - predicted) * beta[token].float()[:, None]
        state += k[token, :, :, None] * correction[:, None, :]
        output.append((q[token, :, :, None] * state).sum(-2))
        if (prefix + token + 1) % 64 == 0 or token + 1 == q.shape[0]:
            checkpoints[token + 1] = state.clone()
    return torch.stack(output), state, checkpoints


class ChunkPrefillTest(unittest.TestCase):
    def assert_kda_matches(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        similarity = F.cosine_similarity(
            actual.flatten().float()[None], expected.flatten().float()[None]
        ).item()
        # Preserve the existing KDA operator tests' cosine threshold.
        self.assertGreater(similarity, 0.99)

    def test_ragged_prefix_checkpoints_match_reference(self):
        torch.manual_seed(53)
        lengths, prefixes, heads, dim = (64, 65, 129, 7), (0, 31, 128, 0), 16, 128
        tokens = sum(lengths)
        q, k, v, gate = [
            torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        beta = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
        a_log = torch.randn(heads, device="cuda") * 0.1
        dt_bias = torch.randn(heads * dim, device="cuda") * 0.01
        initial = torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.02
        initial[0].zero_()
        offsets = [0, 64, 129, 258, tokens]
        table = [[1, -1, -1, -1, -1], [2, 3, -1, -1, -1],
                 [4, 5, 6, 7, 8], [0, 0, 0, 0, 0]]
        cache = torch.full((10, heads, dim, dim), float("nan"), device="cuda")
        cache[2].copy_(initial[1])
        cache[5].copy_(initial[2])
        before = cache.clone()
        output = cula_kda_paged_prefill(
            q, k, v, gate, beta, a_log, dt_bias, None, cache,
            offsets, list(prefixes), table, 64,
        )
        references = []
        for row, (begin, end) in enumerate(zip(offsets[:3], offsets[1:4])):
            ref = recurrent_reference(q[begin:end], k[begin:end], v[begin:end],
                                      gate[begin:end], beta[begin:end], initial[row],
                                      a_log, dt_bias, prefixes[row])
            references.append(ref)
            self.assert_kda_matches(output[begin:end], ref[0])
        self.assertEqual(torch.count_nonzero(output[offsets[3]:]).item(), 0)

        # Every full page and each partial tail must publish the correct state.
        written = set()
        for row, length in enumerate(lengths[:3]):
            start = 0
            while start < length:
                position = prefixes[row] + start
                end = min(length, start + 64 - position % 64)
                block = table[row][position // 64]
                self.assert_kda_matches(cache[block], references[row][2][end])
                written.add(block)
                start = end
        untouched = [block for block in range(cache.shape[0]) if block not in written]
        torch.testing.assert_close(cache[untouched], before[untouched],
                                   rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
