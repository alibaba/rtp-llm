"""GLM53 Prefill, Decode and MTP must preserve the one-token mHC reduction."""

import importlib
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.forward_metadata import scoped_forward_metadata


class _InitialPreModel(torch.nn.Module):
    def __init__(self, *, mtp=False):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="glm_5_mtp" if mtp else "glm5_3_flash", is_glm53_mtp=mtp
        )

    @scoped_forward_metadata
    def forward(self, inputs, residual, weights):
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_pre

        return tk_mhc_pre(
            residual,
            *weights,
            norm_eps=1e-6,
            pre_eps=1e-6,
            sinkhorn_eps=1e-6,
            sinkhorn_iters=20,
        )


def _inputs(*, prefill=False, verify=False):
    return SimpleNamespace(
        attention_inputs=SimpleNamespace(is_prefill=prefill, is_target_verify=verify)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGlm53MhcBatchInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        cls.ops = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )

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
        result = self.ops.mhc_fused_post_pre(
            *values,
            *weights,
            rms_eps=1e-6,
            mhc_pre_eps=1e-6,
            mhc_sinkhorn_eps=1e-6,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )
        self.assertIsNotNone(result)
        return result

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
    def test_initial_pre_decode_and_verify_preserve_single_tokens(self):
        for hidden in (4096, 7168):
            values, weights = self.inputs(256, hidden=hidden)
            residual = values[1]
            model = _InitialPreModel()
            single = [model(_inputs(), row, weights) for row in residual.split(1)]
            expected = tuple(torch.cat([row[k] for row in single]) for k in range(3))
            for mtp, verify in ((False, False), (False, True), (True, False)):
                model = _InitialPreModel(mtp=mtp)
                inputs = _inputs(prefill=verify, verify=verify)
                for count in (4, 8, 64, 128, 192, 256):
                    with self.subTest(
                        hidden=hidden, mtp=mtp, verify=verify, tokens=count
                    ):
                        self.assert_bits(
                            model(inputs, residual[:count], weights),
                            tuple(x[:count] for x in expected),
                        )
                order = torch.randperm(256, device="cuda")
                self.assert_bits(
                    model(inputs, residual[order], weights),
                    tuple(x[order] for x in expected),
                )

    @torch.inference_mode()
    def test_initial_pre_cuda_graph_replay(self):
        values, weights = self.inputs(256)
        residual = values[1]
        model = _InitialPreModel()
        inputs = _inputs(prefill=True, verify=True)
        for _ in range(3):
            model(inputs, residual, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = model(inputs, residual, weights)
        residual.normal_()
        graph.replay()
        expected = [model(_inputs(), row, weights) for row in residual.split(1)]
        expected = tuple(torch.cat([row[k] for row in expected]) for k in range(3))
        self.assert_bits(actual, expected)

    @torch.inference_mode()
    def test_initial_pre_prefill_token_shards_preserve_single_tokens(self):
        values, weights = self.inputs(4)
        needles = values[1]
        model = _InitialPreModel()
        single = [model(_inputs(), row, weights) for row in needles.split(1)]
        expected = tuple(torch.cat([row[k] for row in single]) for k in range(3))
        # ceil(1948/8), ceil(15*1948/8), ceil(51738/8): the real
        # short serial/concurrent and long TP8 Prefill shard lengths.
        for mtp in (False, True):
            model = _InitialPreModel(mtp=mtp)
            for count in (244, 3653, 6468):
                with self.subTest(mtp=mtp, local_tokens=count):
                    residual = torch.zeros(
                        count, 4, 4096, device="cuda", dtype=torch.bfloat16
                    )
                    positions = torch.tensor(
                        [0, count // 3, count // 2, count - 1], device="cuda"
                    )
                    residual[positions] = needles
                    actual = model(_inputs(prefill=True), residual, weights)
                    self.assert_bits(tuple(x[positions] for x in actual), expected)

    @torch.inference_mode()
    def test_initial_pre_preserves_other_models_prefill_reduction(self):
        from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_pre

        values, weights = self.inputs(256)
        residual = values[1]
        expected = tk_mhc_pre(
            residual,
            *weights,
            norm_eps=1e-6,
            pre_eps=1e-6,
            sinkhorn_eps=1e-6,
            sinkhorn_iters=20,
        )
        model = _InitialPreModel()
        model.config = SimpleNamespace(model_type="dsv4", is_glm53_mtp=False)
        self.assert_bits(model(_inputs(prefill=True), residual, weights), expected)

    @torch.inference_mode()
    def test_batch_size_order_and_ownership(self):
        values, weights = self.inputs(192)
        saved = tuple(x.clone() for x in values)
        expected = self.rowwise(values, weights)
        for count in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 192):
            with self.subTest(tokens=count):
                self.assert_bits(
                    self.run_hc(tuple(x[:count] for x in values), weights),
                    tuple(x[:count] for x in expected),
                )
        order = torch.randperm(192, device="cuda")
        self.assert_bits(
            self.run_hc(tuple(x.index_select(0, order) for x in values), weights),
            tuple(x.index_select(0, order) for x in expected),
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
                graph.replay()
                expected = self.rowwise(values, weights)
                self.assert_bits(actual, expected)

    @torch.inference_mode()
    def test_wrapper_supports_multi_request_verify(self):
        from rtp_llm.models_py.modules.dsv4.hc import mhc_tilelang

        values, weights = self.inputs(32)
        expected = self.rowwise(values, weights)
        actual = mhc_tilelang.tk_mhc_fused_post_pre(
            *(x.unsqueeze(0) for x in values),
            *weights,
            norm_eps=1e-6,
            pre_eps=1e-6,
            sinkhorn_eps=1e-6,
            sinkhorn_iters=20,
        )
        self.assertIsNotNone(actual)
        self.assert_bits(tuple(x.squeeze(0) for x in actual), expected)


if __name__ == "__main__":
    unittest.main()
