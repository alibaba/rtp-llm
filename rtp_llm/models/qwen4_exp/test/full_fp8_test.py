import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.full_fp8_weight import (
    FP8_STORAGE_SCALE_SUFFIX,
    Qwen4Fp8StorageWeight,
)
from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import (
    Qwen4Fp8Linear,
    compute_weight_view,
)
from rtp_llm.models_py.modules.qwen4_exp.vocab_embedding import Qwen4CpuVocabEmbedding
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class _Source(TensorSource):
    def load_tensor(self, name, data_type=torch.float16):
        return [torch.arange(32).reshape(4, 8).to(data_type) - 16]

    def has_tensor(self, name):
        return True

    def get_database(self):
        return None


def quantized(value):
    scale = value.float().abs().amax().clamp_min(1e-12).reshape(1) / 448
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


class FullFp8StorageTest(unittest.TestCase):
    def test_original_descriptor_split_runs_before_quantization(self):
        source = AtomicWeight(W.embedding, [CkptWeightInfo("weight")])
        source.split_func = lambda t, **kwargs: t[:, :4]
        config = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            ep_rank=0,
            dp_rank=0,
            ffn_tp_rank=0,
            ffn_tp_size=2,
            hidden_size=8,
            head_num=1,
            head_num_kv=1,
            size_per_head=8,
            moe_pure_tp_mode=False,
            bit=16,
            dp_size=1,
            ep_size=1,
            merge_lora=False,
            compute_dtype=torch.bfloat16,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda key, value, **kw: value
            ),
            use_swizzleA=False,
        )
        descriptor = Qwen4Fp8StorageWeight(source)
        loaded = descriptor.load(_Source(), None, "cpu", config)
        self.assertEqual(loaded[W.embedding].dtype, torch.float8_e4m3fn)
        self.assertEqual(loaded[W.embedding].element_size(), 1)
        scale = loaded[W.embedding + FP8_STORAGE_SCALE_SUFFIX]
        self.assertEqual(scale.numel(), 1)
        self.assertIn(
            W.embedding + FP8_STORAGE_SCALE_SUFFIX, descriptor.additional_output_names
        )
        torch.testing.assert_close(
            loaded[W.embedding].float() * scale,
            _Source().load_tensor("weight", torch.float32)[0][:, :4],
            rtol=0.08,
            atol=0.5,
        )

    def test_compute_view_preserves_fp8_inventory_and_matrices(self):
        weights = ModelWeights(1, "cpu", torch.bfloat16)
        matrix, ms = quantized(torch.randn(32, 32))
        norm, ns = quantized(torch.randn(32))
        weights.weights[0] = {
            "matrix": matrix,
            "matrix" + FP8_STORAGE_SCALE_SUFFIX: ms,
            "norm": norm,
            "norm" + FP8_STORAGE_SCALE_SUFFIX: ns,
        }
        view = compute_weight_view(weights)
        self.assertIs(view.weights[0]["matrix"], matrix)
        self.assertEqual(view.weights[0]["norm"].dtype, torch.bfloat16)
        self.assertEqual(weights.weights[0]["norm"].dtype, torch.float8_e4m3fn)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FullFp8CudaTest(unittest.TestCase):
    def test_dense_gemm_eager_side_stream_and_graph_live_inputs(self):
        torch.manual_seed(873)
        weight, scale = quantized(torch.randn(128, 256, device="cuda"))
        module = Qwen4Fp8Linear(weight, scale)
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = module(x)
        torch.cuda.current_stream().wait_stream(stream)
        expected = F.linear(x, (weight.float() * scale).to(x.dtype))
        torch.testing.assert_close(actual, expected, rtol=0.1, atol=1.5)
        for _ in range(3):
            module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.testing.assert_close(output, module(x), rtol=0, atol=0)
        self.assertIsNone(module.compute_weight)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)

    def test_fp8_host_and_device_vocab_graph(self):
        weight, scale = quantized(torch.randn(257, 320))
        ids = torch.tensor([0, 17, 256, -1], dtype=torch.int32, device="cuda")
        for cpu in (True, False):
            stored = weight.pin_memory() if cpu else weight.cuda()
            module = Qwen4CpuVocabEmbedding(
                stored, 1, torch.device("cuda", 0), weight_scale=scale
            )
            for _ in range(3):
                module(ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = module(ids)
            ids.copy_(torch.tensor([256, 3, 19, -1], dtype=torch.int32, device="cuda"))
            graph.replay()
            expected = (weight[[256, 3, 19]].float() * scale).to(torch.bfloat16).cuda()
            torch.testing.assert_close(output[:3], expected, rtol=0.008, atol=0.02)
            self.assertEqual(output[-1].count_nonzero().item(), 0)
            self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
