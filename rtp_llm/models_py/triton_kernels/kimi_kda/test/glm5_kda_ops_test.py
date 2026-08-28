import os
import unittest

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from rtp_llm.models_py.triton_kernels.fla import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


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
        self.a_log = torch.randn(self.HEADS, device="cuda", dtype=torch.float32).mul_(
            0.1
        )
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

    def _reference(self, lower_bound=None):
        q = F.normalize(self.q.float(), dim=-1).mul_(self.HEAD_DIM**-0.5)
        k = F.normalize(self.k.float(), dim=-1)
        v = self.v.float()
        gate_input = self.g.float() + self.dt_bias.view(1, 1, self.HEADS, self.HEAD_DIM)
        a = torch.exp(self.a_log).view(1, 1, self.HEADS, 1)
        gate = (
            lower_bound * torch.sigmoid(a * gate_input)
            if lower_bound is not None
            else -a * F.softplus(gate_input)
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
            outputs.append(torch.matmul(q[:, token].unsqueeze(-2), state).squeeze(-2))
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

    def test_bounded_prefill_state_matches_decode(self):
        lower_bound = -5.0
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": lower_bound,
        }
        full_output, _ = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            safe_gate=True,
            **common,
        )
        _, prefix_state = chunk_kda(
            self.q[:, :-1],
            self.k[:, :-1],
            self.v[:, :-1],
            self.g[:, :-1],
            self.beta[:, :-1],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            safe_gate=True,
            **common,
        )
        decode_output, _ = fused_recurrent_kda(
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=prefix_state,
            **common,
        )

        self.assertGreater(self._cosine(decode_output, full_output[:, -1:]), 0.999)

    def test_bounded_paged_prefill_state_matches_decode(self):
        lower_bound = -5.0
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": lower_bound,
        }
        prefix_length = self.q.shape[1] - 1
        _, prefix_state, intermediate_states = chunk_kda(
            self.q[:, :prefix_length],
            self.k[:, :prefix_length],
            self.v[:, :prefix_length],
            self.g[:, :prefix_length],
            self.beta[:, :prefix_length],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            cu_seqlens=torch.tensor(
                [0, prefix_length], device="cuda", dtype=torch.int32
            ),
            safe_gate=True,
            return_intermediate_states=True,
            **common,
        )
        page_size = 64
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        paged_state = torch.zeros(
            2,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        store_ssm_state_to_block_map(
            intermediate_states.float(),
            prefix_state,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, prefix_length], device="cuda", dtype=torch.int32),
            block_map,
            paged_state,
            page_size,
            chunk_size=64,
        )
        torch.testing.assert_close(paged_state[1], prefix_state[0])

        decode_args = (
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
        )
        expected, _ = fused_recurrent_kda(
            *decode_args,
            initial_state=prefix_state.clone(),
            **common,
        )
        actual, _ = fused_recurrent_kda(
            *decode_args,
            initial_state=paged_state,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=torch.tensor(
                [prefix_length + 1], device="cuda", dtype=torch.int32
            ),
            **common,
        )

        self.assertGreater(self._cosine(actual, expected), 0.999)

    def test_paged_prefill_conv_state_matches_decode(self):
        token_count = 9
        prefix_length = token_count - 2
        channels = self.HEAD_DIM
        inputs = torch.randn(token_count, channels, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        conv_cache = torch.zeros(3, 3, channels, device="cuda", dtype=torch.bfloat16)
        prefix_cu_seqlens = torch.tensor(
            [0, prefix_length], device="cuda", dtype=torch.int32
        )
        causal_conv1d_fn(
            inputs[:prefix_length].transpose(0, 1),
            weight,
            None,
            conv_cache.transpose(1, 2),
            prefix_cu_seqlens,
            block_map,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        )
        torch.testing.assert_close(
            conv_cache[1], inputs[prefix_length - 3 : prefix_length]
        )

        first_actual = causal_conv1d_update(
            inputs[prefix_length : prefix_length + 1]
            .reshape(1, 1, channels)
            .transpose(1, 2),
            conv_cache.transpose(1, 2),
            weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [prefix_length + 1], device="cuda", dtype=torch.int32
            ),
        )
        torch.testing.assert_close(
            conv_cache[1], inputs[prefix_length - 2 : prefix_length + 1]
        )
        second_actual = causal_conv1d_update(
            inputs[-1:].reshape(1, 1, channels).transpose(1, 2),
            conv_cache.transpose(1, 2),
            weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
        )
        expected = causal_conv1d_fn(
            inputs.transpose(0, 1),
            weight,
            None,
            None,
            torch.tensor([0, token_count], device="cuda", dtype=torch.int32),
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        )

        torch.testing.assert_close(first_actual[0, :, 0], expected[:, -2])
        torch.testing.assert_close(second_actual[0, :, 0], expected[:, -1])

    def test_combined_paged_cache_prefill_then_decode(self):
        token_count = 8
        prefix_length = token_count - 1
        channels = 3 * self.HEADS * self.HEAD_DIM
        inputs = torch.randn(token_count, channels, device="cuda", dtype=torch.bfloat16)
        conv_weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        converter = LinearCacheConverter(
            local_num_v_heads=self.HEADS,
            head_v_dim=self.HEAD_DIM,
            head_k_dim=self.HEAD_DIM,
            ssm_state_dtype=torch.float32,
            linear_conv_kernel_dim=4,
            qkv_size=channels,
            conv_state_dtype=torch.bfloat16,
        )
        cache = torch.zeros(
            2,
            converter.block_size_bytes // 2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ssm_cache = converter.get_ssm_state_tensor(cache)
        conv_cache = converter.get_conv_state_tensor(cache)
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        prefix_cu = torch.tensor([0, prefix_length], device="cuda", dtype=torch.int32)

        prefix_qkv = causal_conv1d_fn(
            inputs[:prefix_length].transpose(0, 1),
            conv_weight,
            None,
            conv_cache.transpose(1, 2),
            prefix_cu,
            block_map,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        ).transpose(0, 1)
        prefix_q, prefix_k, prefix_v = (
            part.view(1, prefix_length, self.HEADS, self.HEAD_DIM).contiguous()
            for part in prefix_qkv.split(channels // 3, dim=-1)
        )
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": -5.0,
        }
        _, prefix_state, intermediate_states = chunk_kda(
            prefix_q,
            prefix_k,
            prefix_v,
            self.g[:, :prefix_length],
            self.beta[:, :prefix_length],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            cu_seqlens=prefix_cu,
            safe_gate=True,
            return_intermediate_states=True,
            **common,
        )
        store_ssm_state_to_block_map(
            intermediate_states.float(),
            prefix_state.float(),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            prefix_cu,
            block_map,
            ssm_cache,
            64,
            chunk_size=64,
        )
        torch.testing.assert_close(ssm_cache[1], prefix_state[0])

        decode_qkv = causal_conv1d_update(
            inputs[-1:].reshape(1, 1, channels).transpose(1, 2),
            conv_cache.transpose(1, 2),
            conv_weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
        ).transpose(1, 2)
        decode_q, decode_k, decode_v = (
            part.view(1, 1, self.HEADS, self.HEAD_DIM).contiguous()
            for part in decode_qkv.split(channels // 3, dim=-1)
        )
        reference_decode, _ = fused_recurrent_kda(
            decode_q,
            decode_k,
            decode_v,
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=prefix_state.clone(),
            **common,
        )
        actual, _ = fused_recurrent_kda(
            decode_q,
            decode_k,
            decode_v,
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=ssm_cache,
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
            **common,
        )

        full_qkv = causal_conv1d_fn(
            inputs.transpose(0, 1),
            conv_weight,
            None,
            None,
            torch.tensor([0, token_count], device="cuda", dtype=torch.int32),
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        ).transpose(0, 1)
        full_q, full_k, full_v = (
            part.view(1, token_count, self.HEADS, self.HEAD_DIM).contiguous()
            for part in full_qkv.split(channels // 3, dim=-1)
        )
        torch.testing.assert_close(decode_qkv[0, 0], full_qkv[-1])
        expected, _ = chunk_kda(
            full_q,
            full_k,
            full_v,
            self.g,
            self.beta,
            output_final_state=False,
            safe_gate=True,
            **common,
        )
        self.assertGreater(self._cosine(reference_decode, expected[:, -1:]), 0.999)
        self.assertGreater(self._cosine(actual, expected[:, -1:]), 0.999)

    def test_bounded_prefill_matches_torch(self):
        lower_bound = -5.0
        expected, expected_state = self._reference(lower_bound)
        actual, actual_state = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=torch.zeros_like(expected_state),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
            safe_gate=True,
            lower_bound=lower_bound,
        )

        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertGreater(self._cosine(actual_state, expected_state), 0.999)

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
