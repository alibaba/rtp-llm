"""Real SM120 group32 FP4 expert regression with an independent CPU oracle.

Exercises provider masked/contiguous kernels and local expert-shard sums.
It does not qualify multi-rank EP communication, CUDA graphs or service startup.
"""

import unittest
from importlib.metadata import version

import torch
import torch.nn.functional as F
from rtp_llm.models_py.modules.dsv41.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv41.v41_routed import V41RoutedExperts


def _weights(experts=4, dim=128, inter=256):
    generator = torch.Generator().manual_seed(41340 + experts)
    tensors = {}
    for name, shape in (
        ("w13_w", (experts, 2 * inter, dim // 2)),
        ("w2_w", (experts, dim, inter // 2)),
        ("w13_s", (experts, 2 * inter, dim // 32)),
        ("w2_s", (experts, dim, inter // 32)),
    ):
        low, high = (0, 256) if name.endswith("_w") else (119, 124)
        tensors[name] = torch.randint(
            low, high, shape, dtype=torch.uint8, generator=generator
        )
    return tensors


def _quantize_reference(x):
    grouped = x.float().reshape(x.shape[0], -1, 32)
    # Independent checkpoint equation: FP32 reciprocal multiplication,
    # bitwise ceil-to-power-of-two scale, then E4M3 rounding.
    raw = grouped.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scales = (exponent << 23).view(torch.float32)
    values = (grouped / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.float(), scales


def _projection_reference(x, packed, encoded_scales):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    weights = levels[(codes & 7).long()]
    weights = torch.where((codes & 8) != 0, -weights, weights)
    scale_b = torch.exp2(encoded_scales.float() - 127)
    values, scale_a = _quantize_reference(x)
    result = torch.zeros((x.shape[0], packed.shape[0]), dtype=torch.float32)
    for group in range(values.shape[1]):
        begin = 32 * group
        result.add_(
            (values[:, group] @ weights[:, begin : begin + 32].T)
            * scale_a[:, group, None]
            * scale_b[None, :, group]
        )
    return result.bfloat16()


def _local_reference(x, routes, indices, tensors, start, valid_rows):
    output = torch.zeros(x.shape, dtype=torch.float32)
    for expert in range(tensors["w2_w"].shape[0]):
        for row in range(x.shape[0]):
            if not valid_rows[row]:
                continue
            for slot in range(indices.shape[1]):
                if indices[row, slot] != start + expert or routes[row, slot] == 0:
                    continue
                gate_up = _projection_reference(
                    x[row : row + 1],
                    tensors["w13_w"][expert],
                    tensors["w13_s"][expert],
                )
                gate, up = gate_up.float().chunk(2, dim=-1)
                hidden = (
                    routes[row, slot]
                    * (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0))
                ).bfloat16()
                output[row].add_(
                    _projection_reference(
                        hidden,
                        tensors["w2_w"][expert],
                        tensors["w2_s"][expert],
                    )[0].float()
                )
    return output


class Sm120V41MoeTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            torch.cuda.is_available(), "SM120 expert regression requires CUDA"
        )
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.assertIn(
            "ee6161b", version("deep-gemm"), "requires the pinned SM120 FP4 provider"
        )

    @staticmethod
    def _local(tensors, start):
        return V41LocalMoE(
            V41RoutedExperts({key: value.cuda() for key, value in tensors.items()}),
            expert_start_id=start,
            max_tokens=16,
            masked_token_limit=16,
            max_topk=3,
        )

    def _check(self, actual, expected):
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # Same two-GEMM quantization tolerance as the source SM120 acceptance.
        torch.testing.assert_close(actual.cpu(), expected, rtol=2.5e-2, atol=1.5e-2)

    def test_masked_and_contiguous_local_routes_match_independent_reference(self):
        tensors = _weights(experts=3)
        local = self._local(tensors, start=4)
        x = torch.randn(5, 128, generator=torch.Generator().manual_seed(413)).bfloat16()
        ids = torch.tensor([[6, 4, -1], [5, 6, 4], [3, 8, -1], [5, -1, 4], [4, 5, 6]])
        routes = torch.tensor(
            [
                [0.37, 0.25, float("nan")],
                [1.5, 0.3, 0.1],
                [1.0, 1.0, float("nan")],
                [0.0, float("nan"), 0.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor([True, True, True, True, False])
        x[2:] = float("nan")
        expected = _local_reference(x, routes, ids, tensors, 4, valid)
        self.assertGreater(float(expected[:2].abs().max()), 0.0)
        for layout in ("masked", "contiguous"):
            with self.subTest(layout=layout):
                actual = local(
                    x.cuda(),
                    routes.cuda(),
                    ids.cuda(),
                    valid_rows=valid.cuda(),
                    layout=layout,
                )
                self._check(actual, expected)
                self.assertTrue(bool((actual[2:] == 0).all()))

    def test_local_shard_partials_combine_in_fp32_without_second_route_weight(self):
        tensors = _weights(experts=4)
        x = torch.randn(3, 128, generator=torch.Generator().manual_seed(414)).bfloat16()
        ids = torch.tensor([[13, 10, 11], [12, 11, -1], [10, 12, 13]])
        routes = torch.tensor(
            [[0.37, 1.5, 0.2], [0.6, 0.25, 0.0], [0.3, 0.7, 1.2]], dtype=torch.float32
        )
        expected = _local_reference(
            x, routes, ids, tensors, 10, torch.ones(3, dtype=torch.bool)
        )
        partials = []
        for begin in (0, 2):
            shard = {
                key: value[begin : begin + 2].contiguous()
                for key, value in tensors.items()
            }
            partial = self._local(shard, start=10 + begin)(
                x.cuda(), routes.cuda(), ids.cuda(), layout="contiguous"
            )
            self.assertEqual(partial.dtype, torch.float32)
            partials.append(partial)
        # Both shards run on this GPU: this checks arithmetic, not NCCL/EP transport.
        self._check(partials[0] + partials[1], expected)

    def test_no_local_routes_and_empty_rows_are_exactly_zero(self):
        local = self._local(_weights(experts=2), start=20)
        for layout in ("masked", "contiguous"):
            with self.subTest(layout=layout):
                x = torch.full(
                    (3, 128), float("nan"), device="cuda", dtype=torch.bfloat16
                )
                indices = torch.tensor([[1, -1]], device="cuda").repeat(3, 1)
                weights = torch.tensor([[1.0, float("nan")]], device="cuda").repeat(
                    3, 1
                )
                output = local(x, weights, indices, layout=layout)
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(bool((output == 0).all()))
                empty = local(x[:0], weights[:0], indices[:0], layout=layout)
                self.assertEqual(empty.shape, (0, 128))
                self.assertEqual(empty.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
