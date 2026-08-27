import os
import unittest

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)


class Glm5KdaOpsTest(unittest.TestCase):
    HEADS = 4
    HEAD_DIM = 128

    def setUp(self):
        torch.manual_seed(7)
        shape = (1, 8, self.HEADS, self.HEAD_DIM)
        self.q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.beta = torch.randn(
            shape[:-1], device="cuda", dtype=torch.bfloat16
        ).sigmoid()
        self.a_log = torch.randn(
            self.HEADS, device="cuda", dtype=torch.float32
        ).mul_(0.1)
        self.dt_bias = torch.randn(
            self.HEADS, self.HEAD_DIM, device="cuda", dtype=torch.float32
        ).mul_(0.01)

    @staticmethod
    def _cosine(actual, expected):
        return F.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).item()

    def _reference(self):
        q = F.normalize(self.q.float(), dim=-1).mul_(self.HEAD_DIM**-0.5)
        k = F.normalize(self.k.float(), dim=-1)
        v = self.v.float()
        gate = -torch.exp(self.a_log).view(1, 1, self.HEADS, 1) * F.softplus(
            self.g.float() + self.dt_bias.view(1, 1, self.HEADS, self.HEAD_DIM)
        )
        state = torch.zeros(
            1,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        outputs = []
        for token in range(self.q.shape[1]):
            state.mul_(torch.exp(gate[:, token]).unsqueeze(-1))
            key = k[:, token].unsqueeze(-1)
            value = v[:, token].unsqueeze(-2)
            correction = (state * key).sum(dim=-2, keepdim=True)
            value = (value - correction) * self.beta[:, token, :, None, None]
            state.add_(key * value)
            outputs.append(
                torch.matmul(q[:, token].unsqueeze(-2), state).squeeze(-2)
            )
        return torch.stack(outputs, dim=1), state

    def test_gate_matches_torch(self):
        actual = fused_kda_gate(
            self.g.flatten(0, 1),
            self.a_log,
            dt_bias=self.dt_bias,
        )
        expected = -torch.exp(self.a_log).view(1, self.HEADS, 1) * F.softplus(
            self.g.flatten(0, 1).float() + self.dt_bias
        )
        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertTrue((actual <= 0).all())

    def test_prefill_and_decode_match_torch(self):
        expected, expected_state = self._reference()
        initial_state = torch.zeros_like(expected_state)
        actual, actual_state = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
        )
        self.assertGreater(self._cosine(actual, expected), 0.98)
        self.assertGreater(self._cosine(actual_state, expected_state), 0.98)

        decode, _ = fused_recurrent_kda(
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=self._reference_prefix_state(),
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
        self.assertGreater(self._cosine(decode, expected[:, -1:]), 0.98)

    def _reference_prefix_state(self):
        q, k, v, g, beta = self.q, self.k, self.v, self.g, self.beta
        self.q, self.k, self.v, self.g, self.beta = (
            q[:, :-1],
            k[:, :-1],
            v[:, :-1],
            g[:, :-1],
            beta[:, :-1],
        )
        try:
            return self._reference()[1]
        finally:
            self.q, self.k, self.v, self.g, self.beta = q, k, v, g, beta


if __name__ == "__main__":
    unittest.main()
