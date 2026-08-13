import unittest

import torch

from rtp_llm.models_py.triton_kernels.fla import (
    fused_recurrent_gated_delta_rule,
)


def recurrent_reference(q, k, v, g, beta, scale, initial_state):
    head_group_size = v.shape[2] // q.shape[2]
    q = q.float().repeat_interleave(head_group_size, dim=2)
    k = k.float().repeat_interleave(head_group_size, dim=2)
    q = torch.nn.functional.normalize(q, p=2, dim=-1) * scale
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    v = v.float()
    g = g.float()
    beta = beta.float()
    state = initial_state.float().clone()
    outputs = []
    states = []

    for token_idx in range(q.shape[1]):
        state = state * g[:, token_idx].exp()[..., None, None]
        value = v[:, token_idx] - torch.einsum(
            "bhkv,bhk->bhv", state, k[:, token_idx]
        )
        value = value * beta[:, token_idx][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, token_idx], value)
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q[:, token_idx], state)
        )
        states.append(state.clone())

    return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


class FusedRecurrentMultitokenTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_multitoken_output_and_states(self):
        torch.manual_seed(20260813)
        device = "cuda"
        batch_size, num_tokens = 1, 7
        num_key_heads, num_value_heads = 16, 32
        key_dim, value_dim = 128, 128
        dtype = torch.bfloat16
        scale = key_dim**-0.5

        q = torch.randn(
            batch_size,
            num_tokens,
            num_key_heads,
            key_dim,
            device=device,
            dtype=dtype,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch_size,
            num_tokens,
            num_value_heads,
            value_dim,
            device=device,
            dtype=dtype,
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(
                batch_size,
                num_tokens,
                num_value_heads,
                device=device,
                dtype=torch.float32,
            )
        )
        beta = torch.sigmoid(
            torch.randn(
                batch_size,
                num_tokens,
                num_value_heads,
                device=device,
                dtype=dtype,
            )
        )
        initial_state = torch.randn(
            batch_size,
            num_value_heads,
            key_dim,
            value_dim,
            device=device,
            dtype=torch.float32,
        ) * 0.01
        seq_size_per_block = 128
        sequence_lengths = torch.tensor([129], device=device, dtype=torch.int32)
        block_map = torch.arange(
            1, num_tokens + 3, device=device, dtype=torch.int32
        ).unsqueeze(0)
        state_cache = torch.zeros(
            num_tokens + 3,
            num_value_heads,
            key_dim,
            value_dim,
            device=device,
            dtype=torch.float32,
        )
        state_cache[1].copy_(initial_state[0])

        expected_output, expected_states = recurrent_reference(
            q, k, v, g, beta, scale, initial_state
        )
        actual_output, _ = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state_cache,
            inplace_final_state=True,
            block_map=block_map,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=sequence_lengths,
            use_qk_l2norm_in_kernel=True,
        )
        actual_states = torch.stack(
            [state_cache[int(block_map[0, 1 + i])] for i in range(num_tokens)],
            dim=0,
        ).unsqueeze(0)

        torch.testing.assert_close(
            actual_output.float(),
            expected_output,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            actual_states.float(),
            expected_states,
            rtol=3e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
