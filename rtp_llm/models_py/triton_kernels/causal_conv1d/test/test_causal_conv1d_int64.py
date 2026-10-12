"""Real GPU addresses above INT32_MAX and ragged, cached prefill pressure."""

import itertools
import unittest

import torch
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
)


class CausalConvInt64Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        print("GPU:", torch.cuda.get_device_name())

    def test_prefill_large_channel_stride_and_padding(self):
        stride = (1 << 30) + 128
        storage = torch.empty(2 * stride + 1, device="cuda", dtype=torch.bfloat16)
        x = storage.as_strided((3, 1), (stride, 1))
        values = torch.tensor([[1], [2], [3]], device="cuda", dtype=x.dtype)
        x.copy_(values)
        weight = torch.zeros((3, 4), device="cuda")
        weight[:, -1] = 1
        cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        metadata = CausalConv1dMetadata(
            torch.tensor([0, -1], device="cuda", dtype=torch.int32),
            torch.tensor([0, -1], device="cuda", dtype=torch.int32),
            2,
        )
        actual = causal_conv1d_fn(
            x,
            weight,
            None,
            None,
            cu,
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            256,
            activation=None,
            metadata=metadata,
        )
        torch.testing.assert_close(actual, values, rtol=0, atol=0)
        torch.cuda.synchronize()

    def test_decode_large_batch_stride(self):
        stride, dim = (1 << 30) + 128, 4
        storage = torch.empty(2 * stride + dim, device="cuda", dtype=torch.bfloat16)
        x = storage.as_strided((3, dim), (stride, 1))
        values = torch.arange(12, device="cuda").reshape(3, dim).bfloat16()
        x.copy_(values)
        state = torch.zeros((4, 3, dim), device="cuda", dtype=x.dtype)
        weight = torch.zeros((dim, 4), device="cuda")
        weight[:, -1] = 1
        mapping = torch.arange(1, 4, device="cuda", dtype=torch.int32).view(3, 1)
        actual = causal_conv1d_update(
            x,
            state.transpose(1, 2),
            weight,
            block_map=mapping,
            sequence_lengths=torch.full((3,), 2, device="cuda", dtype=torch.int32),
            seq_size_per_block=256,
        )
        torch.testing.assert_close(actual, values, rtol=0, atol=0)
        torch.testing.assert_close(state[1:, -1], values, rtol=0, atol=0)
        torch.cuda.synchronize()

    def test_ragged_prefill_large_state_arena(self):
        torch.manual_seed(531011)
        batch, dim, pages = 64, 384, 133
        # Shared SSM+conv arenas expose large physical strides even for small history views.
        stride = (1 << 24) + 2048
        storage = torch.empty(
            (pages - 1) * stride + 3 * dim, device="cuda", dtype=torch.bfloat16
        )
        state = storage.as_strided((pages, dim, 3), (stride, 1, dim))
        self.assertGreater(132 * stride, (1 << 31) - 1)
        state.copy_(
            (torch.randint(-16, 17, state.shape, device="cuda") / 16).bfloat16()
        )
        lengths = ([1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 418, 419] * 6)[:batch]
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        x = (
            (torch.randint(-16, 17, (sum(lengths), dim), device="cuda") / 16)
            .bfloat16()
            .T
        )
        weights = (
            torch.tensor([0.125, -0.25, 0.5, 1], device="cuda")
            .expand(dim, -1)
            .contiguous()
        )
        mapping = torch.full((batch, 3), -1, device="cuda", dtype=torch.int32)
        mapping[:, 0] = 132
        mapping[:, 1:] = (
            torch.randperm(128, device="cuda", dtype=torch.int32).reshape(batch, 2) + 1
        )
        prefix = torch.full((batch,), 256, device="cuda", dtype=torch.int32)
        dense_state = state.clone()
        expected, cursor = [], 0
        for length in lengths:
            history = torch.cat(
                (dense_state[132], x[:, cursor : cursor + length]), dim=1
            ).float()
            output = sum(
                history[:, tap : tap + length] * weights[:, tap : tap + 1]
                for tap in range(4)
            )
            expected.append(output.bfloat16())
            cursor += length
        expected = torch.cat(expected, dim=1)
        for iteration in range(8):
            actual = causal_conv1d_fn(
                x,
                weights,
                None,
                state,
                cu,
                mapping,
                prefix,
                256,
                activation=None,
                output_groups=3,
            )
            contiguous = causal_conv1d_fn(
                x,
                weights,
                None,
                dense_state,
                cu,
                mapping,
                prefix,
                256,
                activation=None,
                output_groups=3,
            )
            torch.testing.assert_close(actual, contiguous, rtol=0, atol=0)
            unpacked = actual.permute(1, 0, 2).reshape(sum(lengths), dim).T
            torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
            torch.testing.assert_close(state, dense_state, rtol=0, atol=0)
            torch.cuda.synchronize()
        print("Ragged pressure: 64 sequences x 8 rounds, state offset", 132 * stride)


if __name__ == "__main__":
    unittest.main()
