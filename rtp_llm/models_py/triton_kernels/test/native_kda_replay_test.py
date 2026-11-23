"""Compare native and Triton replay while switching routes on live leases."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated
from rtp_llm.models_py.triton_kernels.k3_native_verify import K3NativeVerify
from rtp_llm.models_py.triton_kernels.linear_replay import LinearReplay


class NativeKdaReplayTest(unittest.TestCase):
    def test_cuda_graph_with_banked_replay(self):
        self.check_replay(batch=8, heads=48, capture=True)

    def test_routes_share_accepted_prefix_history(self):
        self.check_replay(batch=8, heads=12, mixed_routes=True)

    def test_other_full_verify_widths(self):
        for steps in (3, 5, 8):
            with self.subTest(steps=steps):
                self.check_replay(batch=8, heads=12, steps=steps)

    def check_replay(self, batch, heads, *, steps=4, capture=False, mixed_routes=False):
        device = torch.device("cuda")
        self.assertEqual(torch.cuda.get_device_capability(device)[0], 10)
        generator = torch.Generator(device=device).manual_seed(8811)
        dim = 128
        width, channels = heads * dim, 3 * heads * dim
        logical_width = 4 * width + dim + heads
        lower_bound = -5.0

        def sample(shape, dtype=torch.bfloat16):
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)

        def i32(shape, value=0):
            return torch.full(shape, value, device=device, dtype=torch.int32)

        pages = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
        records = pages.flip(0) - 1
        records[0] = -1
        slots = batch + 1
        context = SimpleNamespace(
            slot_ids=records,
            prev_accept_lengths=i32((batch,)),
            history_valid_lengths=i32((batch,)),
            history_log_banks=i32((batch,)),
            init_kinds=i32((batch,), 1),
            # Pitched group rows need not be aligned to 16 bytes.
            state_read_block_ids=i32((2, batch + 1), -1)[:, :batch],
            active_block_ids=i32((2, batch + 1), -1)[:, :batch],
        )
        context.init_kinds[-1] = 2
        context.state_read_block_ids[1] = pages
        context.state_read_block_ids[1, -1] = -1
        context.active_block_ids[1] = pages + batch
        initial_state = sample((2 * batch + 1, heads, dim, dim), torch.float32) * 0.02
        initial_conv = sample((2 * batch + 1, 3, channels))
        conv_weight = sample((channels, 4), torch.float32) * 0.2
        forget_weight = (sample((width, dim)) * 3).round().clamp(-8, 8) / 64
        a_log = sample((heads,), torch.float32) * 0.1
        dt_bias = sample((width,), torch.float32) * 0.1
        norm = FusedRMSNormGated(sample((dim,)).sigmoid() + 0.5, 1e-6)

        def cache():
            shape = (steps, heads, dim)
            return SimpleNamespace(
                k=torch.zeros((2 * slots, *shape), device=device),
                u=torch.zeros((slots, *shape), device=device),
                g=torch.zeros((2 * slots, *shape), device=device),
                conv_inputs=torch.zeros(
                    (slots, steps, channels), device=device, dtype=torch.bfloat16
                ),
            )

        def reference(kv):
            return LinearReplay(
                conv_weight,
                a_log,
                dt_bias,
                kv.state,
                kv.conv,
                kv.logs,
                group_id=1,
                vector_gate=True,
                state_v_first=False,
                lower_bound=lower_bound,
            )

        def pools():
            return SimpleNamespace(
                state=initial_state.clone(), conv=initial_conv.clone(), logs=cache()
            )

        expected, actual = pools(), pools()
        triton_expected, triton_actual = reference(expected), reference(actual)
        decoder = SimpleNamespace(
            conv_weights=conv_weight,
            alog=a_log,
            dt_bias=dt_bias,
            gate_lower_bound=lower_bound,
            _get_conv_states=lambda _: actual.conv,
            _get_ssm_states=lambda _: actual.state,
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=actual.state, linear_replay=actual.logs, group_id=1
        )
        packed = sample((batch * steps, logical_width))
        native = K3NativeVerify(decoder, forget_weight, norm)
        self.assertTrue(native.bind(kv_cache))

        def run_triton(kernel, projected):
            qkv, gate, fa, beta = projected.split((channels, width, dim, heads), dim=-1)
            value = kernel(qkv, F.linear(fa, forget_weight), beta, context)
            return norm(value, gate.view(-1, heads, dim)).view(-1, width)

        if capture:
            saved_slots = records.clone()
            records.fill_(-1)
            native(packed, kv_cache, context)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = native(packed, kv_cache, context)
            records.copy_(saved_slots)

        previous_steps = steps
        for turn in range(8):
            width_this_turn = steps - 1 if mixed_routes and turn % 2 else steps
            if turn:
                context.prev_accept_lengths.copy_((pages + turn) % previous_steps + 1)
                context.history_valid_lengths.fill_(previous_steps)
                context.history_log_banks.bitwise_xor_(1)
                context.init_kinds.zero_()
                context.state_read_block_ids.copy_(context.active_block_ids)
                context.active_block_ids[1] = pages + (turn % 3 == 0) * batch
            packed.copy_(sample(packed.shape))
            projected = packed[: batch * width_this_turn]
            ref = run_triton(triton_expected, projected)
            if capture:
                graph.replay()
                out = captured
            else:
                out = native(projected, kv_cache, context)
                if width_this_turn < steps:
                    self.assertIsNone(out)
                    out = run_triton(triton_actual, projected)
                else:
                    self.assertIsNotNone(out)
            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.005)
            live_pages = context.active_block_ids[1, 1:].long()
            torch.testing.assert_close(
                actual.state[live_pages],
                expected.state[live_pages],
                rtol=2e-4,
                atol=2e-5,
            )
            torch.testing.assert_close(
                actual.conv[live_pages], expected.conv[live_pages], rtol=0, atol=0
            )
            live_records = records[1:].long()
            write_bank = (
                live_records + (1 - context.history_log_banks[1:].long()) * slots
            )
            for name in ("k", "u", "g", "conv_inputs"):
                index = write_bank if name in ("k", "g") else live_records
                a, e = getattr(actual.logs, name), getattr(expected.logs, name)
                torch.testing.assert_close(
                    a[index, :width_this_turn],
                    e[index, :width_this_turn],
                    rtol=0.02,
                    atol=0.002,
                )
            previous_steps = width_this_turn


if __name__ == "__main__":
    unittest.main()
