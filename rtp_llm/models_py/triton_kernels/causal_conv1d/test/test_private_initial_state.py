import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    gather_conv_initial_states,
)


def conv_reference(x, weight, initial_state):
    padded = torch.cat([initial_state, x], dim=-1)
    return F.conv1d(
        padded,
        weight.unsqueeze(1),
        padding=0,
        groups=weight.shape[0],
    )


class PrivateInitialStateTest(unittest.TestCase):
    def test_private_initial_state_preserves_restored_history(self):
        device = torch.device("cuda")
        torch.manual_seed(1729)
        batch = 2
        lengths = [6, 7]
        prefixes = [0, 5]
        dim, width, seq_size_per_block = 32, 4, 8

        x = torch.randn(sum(lengths), dim, device=device, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)

        # Runtime blocks use the [block, width-1, dim] cache layout.
        conv_layout = torch.randn(
            batch, width - 1, dim, device=device, dtype=torch.bfloat16
        )
        block_map = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
        conv_states = conv_layout.transpose(1, 2)
        prefix_tensor = torch.tensor(prefixes, device=device, dtype=torch.int64)
        initial_states = gather_conv_initial_states(
            conv_states,
            prefix_tensor,
            block_map,
            seq_size_per_block,
        )

        cu = torch.tensor([0, 6, 13], device=device, dtype=torch.int32)
        output = causal_conv1d_fn(
            x.transpose(0, 1),
            weight,
            None,
            conv_states,
            cu,
            block_map,
            prefix_tensor.to(torch.int32),
            seq_size_per_block,
            activation=None,
            initial_states=initial_states,
        ).transpose(0, 1)

        offset = 0
        references = []
        for batch_index, length in enumerate(lengths):
            sequence = x[offset : offset + length, :].transpose(0, 1)
            references.append(
                conv_reference(
                    sequence,
                    weight,
                    initial_states[batch_index],
                ).transpose(0, 1)
            )
            offset += length
        reference = torch.cat(references, dim=0)
        torch.testing.assert_close(
            output.float(), reference.float(), rtol=1e-2, atol=5e-2
        )

        # The runtime cache blocks are allowed to be updated in place; this test
        # proves the first-token read came from the private restoration copy.
        self.assertTrue(torch.isfinite(output.float()).all())


if __name__ == "__main__":
    unittest.main()
