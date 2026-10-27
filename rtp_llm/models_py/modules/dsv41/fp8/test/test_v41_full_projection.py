"""FULL merged projection, strided cache writes and live graph weights."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_global as global_decode
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_full_projection as projection
from rtp_llm.models_py.modules.dsv41.fp8.compressor import _linear_bf16_bf16_fp32
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_global import (
    _case,
    _clone_case,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FullProjectionTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x required")

    def make_case(self, ratio, batch=4, span=6):
        case = _case("cuda", ratio, batch, span)
        attn = case.owner
        attn.is_kv_source = attn.is_index_source = True
        attn.tp_size = 1
        torch.manual_seed(812)
        attn.index_weights = (
            torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16) / 5120**0.5
        )
        projection.prepare(attn)
        return case

    @torch.inference_mode()
    def test_projection_preserves_precision_boundaries_and_reload_addresses(self):
        for ratio in (1, 2):
            case = self.make_case(ratio, 64)
            attn, x = case.owner, case.x.view(-1, 5120)
            ptr = attn._full_projection_weight.data_ptr()
            for step in range(2):
                values, scores, heads = projection.project(attn, case.x)
                expected_value = _linear_bf16_bf16_fp32(x, attn.global_wkv)
                expected_score = (
                    _linear_bf16_bf16_fp32(x, attn.global_wgate)
                    if ratio == 2
                    else expected_value
                )
                for actual, expected in (
                    (values, expected_value),
                    (scores, expected_score),
                ):
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                self.assertEqual(heads.dtype, torch.bfloat16)
                self.assertTrue(heads.is_contiguous())
                torch.testing.assert_close(
                    heads, F.linear(x, attn.index_weights), rtol=0.008, atol=0.008
                )
                self.assertEqual(values.stride(0), 1056 if ratio == 2 else 544)
                attn.global_wkv.mul_(0.5)
                attn.index_weights.mul_(0.5)
                projection.prepare(attn)
                self.assertEqual(ptr, attn._full_projection_weight.data_ptr())

    @torch.inference_mode()
    def test_strided_projections_preserve_all_cache_and_ring_bytes(self):
        for ratio, span in ((1, 6), (2, 6), (2, 8)):
            original = self.make_case(ratio, 4, span)
            strided = _clone_case(original)
            x = original.x.view(-1, 5120)
            values = _linear_bf16_bf16_fp32(x, original.owner.global_wkv)
            scores = (
                _linear_bf16_bf16_fp32(x, original.owner.global_wgate)
                if ratio == 2
                else values
            )
            storage = torch.cat((values, scores, torch.zeros_like(values[:, :32])), 1)
            for case, pair in (
                (original, (values, scores)),
                (strided, (storage[:, :512], storage[:, 512:1024])),
            ):
                self.assertTrue(
                    global_decode.try_produce_global(
                        case.owner,
                        case.x,
                        case.positions,
                        case.req_ids.int(),
                        case.starts.int(),
                        projected=pair,
                    )
                )
            for region, expected in original.owner._test_pools.items():
                torch.testing.assert_close(
                    strided.owner._test_pools[region], expected, rtol=0, atol=0
                )

    @torch.inference_mode()
    def test_projection_graph_uses_updated_inputs_and_reloaded_weights(self):
        for ratio in (1, 2):
            case = self.make_case(ratio)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    projection.project(case.owner, case.x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = projection.project(case.owner, case.x)
            torch.cuda.current_stream().wait_stream(stream)
            for step in range(3):
                case.x.normal_()
                case.owner.global_wkv.mul_(0.5)
                case.owner.global_wgate.mul_(0.5)
                case.owner.index_weights.mul_(0.5)
                projection.prepare(case.owner)
                graph.replay()
                expected = projection.project(case.owner, case.x)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
