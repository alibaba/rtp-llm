"""Whole-packet equality, including inactive routes, padding and spare capacity."""

import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_return import V41ReturnLayout, pack_return


class V41PacketFusionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41725)

    def test_dispatch_entire_allocation(self):
        for rows in (0, 1, 3, 31, 127, 129, 1021, 1024, 4093, 4096):
            for width in (32, 160, 5120):
                layout = V41DispatchLayout(width, 6, 384)
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                weights = torch.rand(rows, 6, device="cuda")
                ids = torch.arange(rows * 6, device="cuda").view(rows, 6) % 384
                valid = torch.arange(rows, device="cuda") % 7 != 0
                weights[::5, 2] = 0
                ids[::5, 2] = 2**32
                weights[::11] = 0
                x[::11] = float("nan")
                x[~valid] = float("nan")
                weights[~valid] = float("nan")
                for capacity in (rows, rows + 5):
                    with self.subTest(rows=rows, width=width, capacity=capacity):
                        expected = pack_dispatch(
                            x,
                            weights,
                            ids,
                            valid,
                            layout,
                            out=layout.allocate(capacity, device=x.device).fill_(33),
                        )
                        actual = pack_dispatch(
                            x,
                            weights,
                            ids,
                            valid,
                            layout,
                            out=layout.allocate(capacity, device=x.device).fill_(77),
                            fused=True,
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_return_rounding_and_strided_input(self):
        for rows in (0, 1, 3, 129, 1021, 1024, 4096):
            for width in (32, 160, 5120):
                storage = torch.randn(rows, width * 2, device="cuda")
                partial = storage[:, ::2]
                if rows:
                    partial[0, :8] = torch.tensor(
                        [-0.0, 0, 1e-8, -1e-8, 1.00390625, -1.00390625, 448, -448],
                        device="cuda",
                    )
                for mode in ("fp32", "mxfp8"):
                    with self.subTest(rows=rows, width=width, mode=mode):
                        layout = V41ReturnLayout(width, mode)
                        expected = pack_return(partial, layout)
                        actual = pack_return(partial, layout, fused=True)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_captured_wire_packets(self):
        directory = os.environ.get("DSV41_MOE_FIXTURES")
        if not directory:
            self.skipTest("Real model fixture path not supplied")
        paths = sorted(Path(directory).glob("rank*-layer*-chunk*.pt"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            tensors = torch.load(path, weights_only=True, map_location="cpu")["tensors"]
            partial = tensors["local_partial"].cuda()
            expected_return = tensors["wire1_send"].cuda()
            actual_return = pack_return(
                partial, V41ReturnLayout(5120, "mxfp8"), fused=True
            )
            torch.testing.assert_close(actual_return, expected_return, rtol=0, atol=0)
            layout = V41DispatchLayout(5120, 6, 384)
            x = tensors["moe_input"].cuda()
            packet = tensors["wire0_send"][: x.shape[0]].cuda().contiguous()
            _, _, weights, ids = layout.views(packet)
            actual_dispatch = pack_dispatch(
                x,
                weights.contiguous(),
                ids.contiguous(),
                torch.ones(x.shape[0], dtype=torch.bool, device=x.device),
                layout,
                fused=True,
            )
            torch.testing.assert_close(actual_dispatch, packet, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
