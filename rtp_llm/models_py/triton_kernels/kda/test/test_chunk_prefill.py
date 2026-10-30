"""Generic KDA chunk/cache regression; does not import the cuLA package."""

import math
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.kda.chunk import chunk_kda


def recurrent_reference(q, k, v, gate, beta, initial, a_log, dt_bias):
    q = F.normalize(q.float(), dim=-1) / math.sqrt(q.shape[-1])
    k = F.normalize(k.float(), dim=-1)
    state = initial.float().clone()
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
        if (token + 1) % 64 == 0 or token + 1 == q.shape[0]:
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
        lengths, heads, dim = (64, 65, 129), 2, 128
        tokens = sum(lengths)
        q, k, v, gate = [
            torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        beta = torch.randn(tokens, heads, device="cuda").sigmoid().bfloat16()
        a_log = torch.randn(heads, device="cuda") * 0.1
        dt_bias = torch.randn(heads * dim, device="cuda") * 0.01
        initial = torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.02
        offsets = [0, 64, 129, tokens]
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        output, final, intermediate = chunk_kda(
            q[None], k[None], v[None], gate[None], beta[None],
            initial_state=initial, output_final_state=True, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            return_intermediate_states=True, A_log=a_log, dt_bias=dt_bias,
        )
        references = []
        for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            ref = recurrent_reference(q[begin:end], k[begin:end], v[begin:end],
                                      gate[begin:end], beta[begin:end], initial[row],
                                      a_log, dt_bias)
            references.append(ref)
            self.assert_kda_matches(output[0, begin:end], ref[0])
            self.assert_kda_matches(final[row], ref[1])

        # Every full page and each partial tail must publish the correct state.
        table = torch.tensor([[1, -1, -1], [2, 3, -1], [4, 5, 6]],
                             dtype=torch.int32, device="cuda")
        cache = torch.full((8, heads, dim, dim), float("nan"), device="cuda")
        store_ssm_state_to_block_map(
            intermediate.float(), final.float(),
            torch.zeros(3, dtype=torch.int32, device="cuda"), cu, table,
            cache, seq_size_per_block=64, chunk_size=64,
        )
        for row, length in enumerate(lengths):
            for page, end in enumerate(range(64, length + 64, 64)):
                self.assert_kda_matches(cache[table[row, page]],
                                        references[row][2][min(end, length)])
        self.assertTrue(torch.isnan(cache[0]).all().item())
        self.assertTrue(torch.isnan(cache[7]).all().item())
        prefix = torch.tensor([64, 64, 128], dtype=torch.int32, device="cuda")
        restored = torch.empty_like(initial)
        load_initial_state_from_block_map(prefix, table, cache, restored, 64)
        for row, length in enumerate((64, 64, 128)):
            self.assert_kda_matches(restored[row], references[row][2][length])


if __name__ == "__main__":
    unittest.main()
