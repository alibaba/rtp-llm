import unittest

import torch

from rtp_llm.models_py.modules.hybrid import glm5_cmp_mxfp8 as producers
from rtp_llm.models_py.modules.hybrid.test.glm5_cmp_mxfp8_ref import (
    _assert_k,
    _assert_q,
)


def _q_inputs(rows):
    generator = torch.Generator(device="cuda").manual_seed(813 + rows)
    projected = torch.randn(
        rows, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    projected[0, :128].zero_()
    if rows > 1:
        projected[1] *= 64
    trig = torch.zeros(8, 64, device="cuda")
    trig[::2, :32] = 1
    trig[1::2, 32:] = -1
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) % 8
    gates = torch.randn(rows, 32, device="cuda", generator=generator)
    outputs = (
        torch.empty(rows, 32, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(rows, 32, device="cuda", dtype=torch.int32),
        torch.empty(rows, 32, device="cuda", dtype=torch.bfloat16),
    )
    return projected, trig, positions, gates, outputs


def _k_inputs(rows):
    pattern = torch.tensor([-3, -1, 1, 3], device="cuda").repeat(32).float()
    activation = torch.zeros(rows, 6144, device="cuda")
    activation[:, :128] = pattern
    activation[:, 128:256] = pattern.flip(0) / 2
    weight = torch.zeros(128, 6144, device="cuda")
    weight[:, :128] = torch.eye(128, device="cuda")
    weight[:, 128:256] = torch.eye(128, device="cuda") * 0.25
    aligned_rows = (rows + 3) // 4 * 4
    activation_sf = torch.full(
        (12, aligned_rows), 0x7F7F7F7F, device="cuda", dtype=torch.int32
    ).T[:rows]
    weight_sf = torch.full((12, 128), 0x7F7F7F7F, device="cuda", dtype=torch.int32).T
    norm_weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    norm_bias = torch.zeros_like(norm_weight)
    trig = torch.zeros(8, 64, device="cuda")
    trig[::2, :32] = 1
    trig[1::2, 32:] = -1
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) % 8
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    if rows > 1:
        slots[0] = -1
        slots[-1] = 128
    cache = torch.full((2, 64, 1, 132), 0xA5, device="cuda", dtype=torch.uint8)
    return (
        activation.to(torch.float8_e4m3fn),
        activation_sf,
        weight.to(torch.float8_e4m3fn),
        weight_sf,
        norm_weight,
        norm_bias,
        trig,
        positions,
        slots,
        cache,
    )


class Glm5CmpMxfp8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability()[0] == 10
        producers.prepare()

    def test_q_bytes_scales_and_unfolded_gate(self):
        for rows in (1, 17, 33):
            for pdl in (False, True):
                with self.subTest(rows=rows, pdl=pdl):
                    projected, trig, positions, gates, outputs = _q_inputs(rows)
                    producers.indexer_q_rope_quant(
                        projected, trig, positions, gates, out=outputs, pdl=pdl
                    )
                    _assert_q(projected, trig, positions, gates, outputs)

    def test_q_graph_replay_updates_inputs_and_metadata(self):
        for pdl in (False, True):
            with self.subTest(pdl=pdl):
                projected, trig, positions, gates, outputs = _q_inputs(17)
                event = torch.cuda.Event()
                event.record()
                producers.indexer_q_rope_quant(
                    projected, trig, positions, gates, out=outputs, pdl=pdl
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                pointers = tuple(t.data_ptr() for t in outputs)
                with torch.cuda.graph(graph):
                    producers.indexer_q_rope_quant(
                        projected,
                        trig,
                        positions,
                        gates,
                        out=outputs,
                        notify_event=event,
                        pdl=pdl,
                    )
                for iteration in range(3):
                    projected.add_(0.25)
                    trig.neg_()
                    positions.copy_((positions + 1) % 8)
                    gates.add_(iteration + 1)
                    graph.replay()
                    _assert_q(projected, trig, positions, gates, outputs)
                    assert tuple(t.data_ptr() for t in outputs) == pointers

    def test_q_rejects_uninitialized_event(self):
        projected, trig, positions, gates, outputs = _q_inputs(1)
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            producers.indexer_q_rope_quant(
                projected,
                trig,
                positions,
                gates,
                out=outputs,
                notify_event=torch.cuda.Event(),
            )

    def test_q_nontrivial_rotations(self):
        for pdl in (False, True):
            with self.subTest(pdl=pdl):
                projected, trig, positions, gates, outputs = _q_inputs(17)
                angles = torch.linspace(-2.75, 2.75, 8 * 32, device="cuda").view(8, 32)
                trig[:, :32].copy_(angles.cos())
                trig[:, 32:].copy_(angles.sin())
                producers.indexer_q_rope_quant(
                    projected, trig, positions, gates, out=outputs, pdl=pdl
                )
                _assert_q(projected, trig, positions, gates, outputs)

    def test_k_fused_projection_and_paged_cache_bytes(self):
        for rows in (1, 17, 33):
            for pdl in (False, True):
                with self.subTest(rows=rows, pdl=pdl):
                    inputs = _k_inputs(rows)
                    initial_cache = inputs[-1].clone()
                    producers.indexer_k_cache(*inputs, epsilon=1e-5, pdl=pdl)
                    _assert_k(inputs, initial_cache, 1e-5)

    def test_k_graph_replay_updates_inputs_positions_and_slots(self):
        for pdl in (False, True):
            with self.subTest(pdl=pdl):
                inputs = _k_inputs(17)
                producers.indexer_k_cache(*inputs, epsilon=1e-5, pdl=pdl)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    producers.indexer_k_cache(*inputs, epsilon=1e-5, pdl=pdl)
                for iteration in range(3):
                    (
                        activation,
                        activation_sf,
                        weight,
                        weight_sf,
                        norm_weight,
                        norm_bias,
                        trig,
                        positions,
                        slots,
                        cache,
                    ) = inputs
                    activation.copy_(activation.float().neg().to(torch.float8_e4m3fn))
                    weight.copy_(weight.float().mul(0.5).to(torch.float8_e4m3fn))
                    activation_sf.fill_(0x80808080 - 0x100000000)
                    weight_sf.fill_(0x7E7E7E7E)
                    norm_weight.fill_(iteration + 1)
                    norm_bias.fill_(0.25 * iteration)
                    trig.neg_()
                    positions.copy_((positions + 1) % 8)
                    slots.copy_(
                        (torch.arange(17, device="cuda") + iteration * 32).long()
                    )
                    initial_cache = cache.clone()
                    graph.replay()
                    _assert_k(inputs, initial_cache, 1e-5)

    def test_k_random_dense_projection_and_rotations(self):
        for pdl in (False, True):
            with self.subTest(pdl=pdl):
                inputs = _k_inputs(5)
                (
                    activation,
                    _,
                    weight,
                    _,
                    norm_weight,
                    norm_bias,
                    trig,
                    _,
                    slots,
                    cache,
                ) = inputs
                generator = torch.Generator(device="cuda").manual_seed(927)
                activation.copy_(
                    torch.randn(
                        activation.shape, device="cuda", generator=generator
                    ).to(activation.dtype)
                )
                weight.copy_(
                    torch.randn(weight.shape, device="cuda", generator=generator).to(
                        weight.dtype
                    )
                )
                norm_weight.copy_(
                    0.5 + torch.rand(128, device="cuda", generator=generator)
                )
                norm_bias.copy_(
                    torch.randn(128, device="cuda", generator=generator) * 0.1
                )
                angles = torch.linspace(-2.75, 2.75, 8 * 32, device="cuda").view(8, 32)
                trig[:, :32].copy_(angles.cos())
                trig[:, 32:].copy_(angles.sin())
                slots.copy_(torch.arange(5, device="cuda", dtype=torch.int64) * 17)
                initial_cache = cache.clone()
                producers.indexer_k_cache(*inputs, epsilon=1e-5, pdl=pdl)
                _assert_k(inputs, initial_cache, 1e-5)


if __name__ == "__main__":
    unittest.main()
