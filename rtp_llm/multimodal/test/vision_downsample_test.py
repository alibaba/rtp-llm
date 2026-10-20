"""Exact byte/stride/ownership tests for packed vision downsampling."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41 import vision_downsample_triton as downsample


def original_columns(x, grids, ratio):
    columns = []
    for part, (height, width) in zip(x.split([h * w for h, w in grids]), grids):
        image = part.view(height, width, -1).permute(2, 0, 1)
        image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
        columns.append(F.unfold(image.unsqueeze(0), ratio, stride=ratio).squeeze(0))
    return columns


class DownsampleContractTest(unittest.TestCase):
    def test_cpu_and_empty_are_unsupported(self):
        for shape, grids in (((0, 1024), []), ((30, 1024), [(3, 5)] * 2)):
            with torch.inference_mode():
                self.assertIsNone(
                    downsample.apply_vision_downsample(torch.zeros(shape), grids, 3)
                )

    def test_forward_propagates_supported_launch_errors(self):
        with patch.object(downsample, "is_supported", return_value=True), patch.object(
            downsample.torch, "empty", side_effect=RuntimeError("allocation failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                downsample.apply_vision_downsample(
                    torch.empty(30, 1024), [(3, 5)] * 2, 3
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DownsampleCudaTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x specialization")
        self.guard = torch.inference_mode()
        self.guard.__enter__()
        self.addCleanup(self.guard.__exit__, None, None, None)
        torch.manual_seed(20261020)

    def compare(self, batch, height, width):
        grids = [(height, width)] * batch
        # Random BF16 bit patterns exercise signed zero, subnormals and NaN bits.
        x = (
            torch.randint(
                0,
                65536,
                (batch * height * width, 1024),
                dtype=torch.int32,
                device="cuda",
            )
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        actual = downsample.apply_vision_downsample(x, grids, 3)
        self.assertIsNotNone(actual)
        reference = original_columns(x, grids, 3)
        for got, want in zip(actual, reference):
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.stride(), want.stride())
            self.assertEqual(got.t().stride(), want.t().stride())
            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
            self.assertNotEqual(
                got.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
            )
        return x, actual

    def test_ragged_edges_prime_batches_and_large_offsets(self):
        for batch, height, width in (
            (2, 1, 1),
            (3, 3, 5),
            (5, 7, 11),
            (25, 75, 112),
            (63, 75, 112),
            (126, 75, 112),
        ):
            with self.subTest(batch=batch, height=height, width=width):
                self.compare(batch, height, width)

    def test_unsupported_shapes_and_grad_keep_original_path(self):
        x = torch.zeros((30, 1024), dtype=torch.bfloat16, device="cuda")
        for value, grids, ratio in (
            (x, [(3, 10)], 3),
            (x, [(3, 5), (5, 3)], 3),
            (x.float(), [(3, 5)] * 2, 3),
            (x, [(3, 5)] * 2, 2),
            (x[:, ::2], [(3, 5)] * 2, 3),
            (x, [(2, 5)] * 2, 3),
        ):
            self.assertIsNone(downsample.apply_vision_downsample(value, grids, ratio))
        with torch.inference_mode(False), torch.enable_grad():
            self.assertIsNone(downsample.apply_vision_downsample(x, [(3, 5)] * 2, 3))

    def test_w1_layout_output_and_graph_replay(self):
        grids = [(7, 11)] * 3
        x = torch.randn((231, 1024), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((128, 9216), dtype=torch.bfloat16, device="cuda")
        # Explicit non-default stream warmup before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            downsample.apply_vision_downsample(x, grids, 3)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = downsample.apply_vision_downsample(x, grids, 3)
        graph.replay()
        reference = original_columns(x, grids, 3)
        for got, want in zip(output, reference):
            self.assertTrue(
                torch.equal(F.linear(got.t(), weight), F.linear(want.t(), weight))
            )
        x.neg_()
        graph.replay()
        for got, want in zip(output, original_columns(x, grids, 3)):
            self.assertTrue(torch.equal(got, want))
        self.assertTrue(
            all(
                output[i].data_ptr() != output[j].data_ptr()
                for i in range(3)
                for j in range(i)
            )
        )


if __name__ == "__main__":
    unittest.main()
