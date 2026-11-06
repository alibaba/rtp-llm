"""CPU contracts for FP8 weight/scale row selection around KV padding."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock, patch
import sys
import unittest

import torch


path = Path(__file__).resolve().parents[1] / "models_py/modules/kimi_k3/natural_dsa.py"
spec = importlib.util.spec_from_file_location("natural_fp8_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Fp8NaturalProjectionTest(unittest.TestCase):
    def projection(self, packed):
        weight = torch.arange(640).remainder(32).reshape(-1, 1).expand(-1, 256).to(torch.float8_e4m3fn)
        scales = (torch.arange(640, dtype=torch.int32).reshape(1, -1).T if packed
                  else torch.arange(10, dtype=torch.float32).reshape(5, 2))
        return SimpleNamespace(weight=weight, weight_scales=scales, bias=None,
                               scale_ue8m0=packed,
                               _deepgemm_linear=SimpleNamespace(quantize_input=lambda hidden: None))

    def test_packed_scale_rows_preserve_offset_and_stride(self):
        source = self.projection(True)
        view = module.Fp8NaturalProjection(source, 128, 384, 192)
        self.assertEqual(view.logical_width, 192)
        self.assertEqual(tuple(view.weight.shape), (256, 256))
        self.assertEqual(view.weight.untyped_storage().data_ptr(), source.weight.untyped_storage().data_ptr())
        self.assertEqual(view.weight_scales.untyped_storage().data_ptr(), source.weight_scales.untyped_storage().data_ptr())
        torch.testing.assert_close(view.weight_scales, source.weight_scales[128:384], rtol=0, atol=0)
        self.assertEqual(view.weight_scales.stride(), source.weight_scales.stride())

    def test_float_scales_follow_output_blocks_and_gate_skips_padding(self):
        source = self.projection(False)
        kv = module.Fp8NaturalProjection(source, 128, 384, 192)
        gate = module.Fp8NaturalProjection(source, 384, 640, 256)
        torch.testing.assert_close(kv.weight_scales, source.weight_scales[1:3], rtol=0, atol=0)
        torch.testing.assert_close(gate.weight_scales, source.weight_scales[3:5], rtol=0, atol=0)
        self.assertEqual(gate.weight.storage_offset(), source.weight[384:].storage_offset())

    def test_unaligned_or_out_of_range_views_fail(self):
        source = self.projection(True)
        for start, end, logical in [(129, 384, 192), (128, 320, 192),
                                    (128, 384, 257), (-128, 128, 128), (0, 768, 128)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                module.Fp8NaturalProjection(source, start, end, logical)

    def test_q_and_gate_consume_one_encoding_without_requantization(self):
        source = self.projection(True)
        values = torch.ones((3, 256), dtype=torch.float8_e4m3fn)
        scales = torch.ones((3, 2), dtype=torch.int32)
        quantize = Mock(return_value=(values, scales))
        source._deepgemm_linear.quantize_input = quantize
        q = module.Fp8NaturalProjection(source, 0, 128, 128)
        gate = module.Fp8NaturalProjection(source, 384, 640, 256)
        calls = []
        wrapper = ModuleType("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper")
        def gemm(activation, weight, output, **kwargs):
            self.assertIs(activation[0], values)
            self.assertIs(activation[1], scales)
            calls.append(weight[0].storage_offset())
            output.fill_(len(calls))
        wrapper.fp8_gemm_nt = gemm
        hidden = torch.zeros((3, 256), dtype=torch.bfloat16)
        with patch.dict(sys.modules, {wrapper.__name__: wrapper}):
            encoded = q.quantize_input(hidden)
            q_out = q.forward_quantized(*encoded)
            gate_out = gate.forward_quantized(*encoded)
        quantize.assert_called_once_with(hidden)
        self.assertEqual(calls, [source.weight.storage_offset(), source.weight[384:].storage_offset()])
        self.assertEqual(tuple(q_out.shape), (3, 128))
        self.assertEqual(tuple(gate_out.shape), (3, 256))
        self.assertTrue(torch.all(q_out == 1))
        self.assertTrue(torch.all(gate_out == 2))

    def test_merged_q_gate_packs_the_correct_rows_and_calls_one_gemm(self):
        for packed in (False, True):
            source = self.projection(packed)
            values = torch.ones((3, 256), dtype=torch.float8_e4m3fn)
            scales = torch.ones((3, 2), dtype=torch.int32)
            quantize = Mock(return_value=(values, scales))
            source._deepgemm_linear.quantize_input = quantize
            q = module.Fp8NaturalProjection(source, 0, 128, 128)
            gate = module.Fp8NaturalProjection(source, 384, 640, 256)
            merged = module.NaturalQGateProjection(q, gate)
            expected_weight = torch.cat((source.weight[:128].view(torch.uint8),
                                         source.weight[384:].view(torch.uint8)), 0)
            torch.testing.assert_close(merged.weight.view(torch.uint8), expected_weight, rtol=0, atol=0)
            expected_scale = torch.cat((q.weight_scales, gate.weight_scales), 0)
            torch.testing.assert_close(merged.weight_scales, expected_scale, rtol=0, atol=0)
            if source.weight_scales.stride(0) == 1:
                self.assertEqual(merged.weight_scales.stride(0), 1)
            wrapper = ModuleType("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper")
            calls = []
            def gemm(activation, weight, output, **kwargs):
                self.assertIs(activation[0], values)
                self.assertIs(activation[1], scales)
                calls.append(weight[0])
                output[:, :128].fill_(1)
                output[:, 128:].fill_(2)
            wrapper.fp8_gemm_nt = gemm
            with patch.dict(sys.modules, {wrapper.__name__: wrapper}):
                q_out, gate_out = merged(torch.zeros((3, 256), dtype=torch.bfloat16))
            self.assertEqual(quantize.call_count, 1)
            self.assertEqual(len(calls), 1)
            self.assertTrue(torch.all(q_out == 1))
            self.assertTrue(torch.all(gate_out == 2))


if __name__ == "__main__":
    unittest.main()
