"""Preserve source GLM53 fused mHC row, ownership and replay contracts."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.hc.factory import build_hc_unit
from rtp_llm.models_py.modules.glm53_mhc import fused_post_pre


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGlm53MhcFusedBatchInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    def inputs(self, count, hidden=4096):
        torch.manual_seed(5328 + hidden)
        values = (
            torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(count, 4, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(count, 4, 1, device="cuda"),
            torch.randn(count, 4, 4, device="cuda"),
        )
        weights = (
            torch.randn(24, 4 * hidden, device="cuda") * 0.01,
            torch.randn(3, device="cuda") * 0.1,
            torch.randn(24, device="cuda") * 0.1,
        )
        return values, weights

    def run_hc(self, values, weights):
        unit = build_hc_unit(
            weights[0],
            weights[2],
            weights[1],
            dim=values[0].shape[-1],
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-06,
            hc_eps=1e-06,
        )
        return fused_post_pre(*values, unit)

    def rowwise(self, values, weights):
        rows = [
            self.run_hc(tuple(x[i : i + 1].contiguous() for x in values), weights)
            for i in range(values[0].shape[0])
        ]
        return tuple(torch.cat([row[index] for row in rows]) for index in range(4))

    def assert_bits(self, actual, expected):
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @torch.inference_mode()
    def test_batch_size_order_and_ownership(self):
        values, weights = self.inputs(192)
        saved = tuple(x.clone() for x in values)
        expected = self.rowwise(values, weights)
        for count in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 192):
            with self.subTest(tokens=count):
                self.assert_bits(
                    self.run_hc(tuple((x[:count] for x in values)), weights),
                    tuple((x[:count] for x in expected)),
                )
        order = torch.randperm(192, device="cuda")
        self.assert_bits(
            self.run_hc(tuple((x.index_select(0, order) for x in values)), weights),
            tuple((x.index_select(0, order) for x in expected)),
        )
        self.assert_bits(values, saved)

    @torch.inference_mode()
    def test_larger_hidden_size(self):
        values, weights = self.inputs(32, hidden=7168)
        self.assert_bits(self.run_hc(values, weights), self.rowwise(values, weights))

    @torch.inference_mode()
    def test_cuda_graph(self):
        for count in (8, 32):
            with self.subTest(tokens=count):
                values, weights = self.inputs(count)
                expected = self.rowwise(values, weights)
                for _ in range(3):
                    self.run_hc(values, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = self.run_hc(values, weights)
                values[0].normal_()
                values[1].normal_()
                values[2].normal_()
                values[3].normal_()
                weights[0].normal_(0, 0.01)
                weights[1].normal_(0, 0.1)
                weights[2].normal_(0, 0.1)
                graph.replay()
                expected = self.rowwise(values, weights)
                self.assert_bits(actual, expected)

    @torch.inference_mode()
    def test_wrapper_supports_multi_request_verify(self):
        values, weights = self.inputs(32)
        expected = self.rowwise(values, weights)
        for batch, sequence in ((1, 32), (2, 16)):
            with self.subTest(batch=batch, sequence=sequence):
                batched = tuple(
                    (x.reshape(batch, sequence, *x.shape[1:]) for x in values)
                )
                actual = self.run_hc(batched, weights)
                self.assert_bits(tuple(x.flatten(0, 1) for x in actual), expected)

    def test_outer_grad_mode_preserves_source_inference_contract(self):
        values, weights = self.inputs(4)
        saved = tuple(x.clone() for x in values)
        with torch.inference_mode():
            expected = self.run_hc(values, weights)
        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            actual = self.run_hc(values, weights)
        self.assert_bits(actual, expected)
        self.assert_bits(values, saved)
        self.assertNotEqual(actual[0].data_ptr(), values[1].data_ptr())
