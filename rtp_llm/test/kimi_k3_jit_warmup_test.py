"""The directed replay compilation must leave live request state untouched."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup import (
    _inactive_replay_inputs,
)
from rtp_llm.models_py.modules.kimi_k3.native_mla_ops import gate_sigmoid_mul
from rtp_llm.models_py.triton_kernels.linear_replay import (
    finalize_linear_replay,
    linear_serial_replay,
)


class KimiK3JitWarmupTest(unittest.TestCase):
    def test_mla_gate_preserves_bf16_sigmoid_materialization(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(53)
        for rows in (1, 7, 129):
            with self.subTest(rows=rows):
                values = torch.randn((rows, 257), device="cuda", dtype=torch.bfloat16)[:, 1:]
                gates = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16)[:, ::2]
                actual = gate_sigmoid_mul(values, gates)
                expected = values * gates.sigmoid()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_inactive_production_shape_replay_does_not_change_pool(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        slots, blocks, capacity, dim, width = 7, 11, 4, 128, 4
        channels = 3 * dim

        def tensor(shape, dtype=torch.float32):
            return torch.ones(shape, device=device, dtype=dtype)

        cache = SimpleNamespace(
            k=tensor((slots, capacity, 1, dim)),
            u=tensor((slots, capacity, 1, dim)),
            g=tensor((slots, capacity, 1, dim)),
            conv_inputs=tensor((slots, capacity, channels), torch.bfloat16),
            slot_generations=tensor((slots,), torch.int64),
            log_epochs=tensor((slots,), torch.int64),
            valid_counts=tensor((slots,), torch.int32),
            error_flags=torch.zeros((slots,), device=device, dtype=torch.int32),
        )
        state = tensor((blocks, 1, dim, dim))
        conv = tensor((blocks, width - 1, channels), torch.bfloat16)
        inputs = _inactive_replay_inputs(2, device)
        tracked = (state, conv, *(getattr(cache, name) for name in vars(cache)))
        before = tuple(value.clone() for value in tracked)

        for steps in range(1, capacity + 1):
            qkv = tensor((steps, channels), torch.bfloat16)
            q, k, v = qkv.split((dim, dim, dim), dim=-1)
            result = linear_serial_replay(
                q, k, v,
                tensor((steps, dim), torch.bfloat16),
                tensor((steps, 1), torch.bfloat16),
                tensor((channels, width), torch.bfloat16),
                tensor((1,)), tensor((dim,)),
                state, conv, cache, inputs,
                group_id=2, vector_gate=True, state_v_first=False,
                lower_bound=-5.0,
            )
            finalize_linear_replay(cache, inputs, steps)
            self.assertEqual(tuple(result.shape), (steps, 1, dim))
            self.assertEqual(torch.count_nonzero(result).item(), 0)
        torch.cuda.synchronize(device)
        for actual, expected in zip(tracked, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
