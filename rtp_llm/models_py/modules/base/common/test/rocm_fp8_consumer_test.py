"""Small CPU contract tests and actual HIP execution/capture regressions."""

import json
import os
import unittest

import torch

from rtp_llm.models_py.modules.base.common.rocm_fp8 import (
    convert_fn_weight,
    fn_decode_table,
    quantize_input,
    vocab_rows,
)


class StorageContractTest(unittest.TestCase):
    def test_fn_to_fnuz_preserves_finite_values_and_source(self):
        raw = torch.arange(256, dtype=torch.uint8)
        source = raw.view(torch.float8_e4m3fn)
        converted, scale = convert_fn_weight(
            source, torch.tensor([0.25]), torch.float8_e4m3fnuz
        )
        finite = torch.isfinite(source.float())
        torch.testing.assert_close(
            converted.float()[finite] * scale,
            source.float()[finite] * 0.25,
            rtol=0,
            atol=0,
        )
        self.assertEqual(raw[128].item(), 128)
        self.assertEqual(converted.view(torch.uint8)[128].item(), 0)

    def test_vocab_byte_decode_mask_and_invalid_ids(self):
        raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        weight = raw.view(torch.float8_e4m3fn)
        tokens = torch.tensor([0, 3, 8, 14, -1, 16])
        mask = torch.tensor([1, 0, 1, 1, 1, 1])
        actual = vocab_rows(
            tokens, weight, torch.tensor([0.5]), fn_decode_table("cpu"), mask
        )
        expected = weight.float().to(torch.bfloat16)[tokens.clamp(0, 15)] * 0.5
        expected[[1, 4, 5]] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(),
    "requires real ROCm GPU",
)
class HipConsumerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.metadata

        import aiter
        import triton

        print(
            "ROCM_RUNTIME "
            + json.dumps(
                {
                    "torch": torch.__version__,
                    "torch_file": torch.__file__,
                    "hip": torch.version.hip,
                    "aiter": importlib.metadata.version("amd-aiter"),
                    "aiter_file": aiter.__file__,
                    "triton": triton.__version__,
                    "triton_file": triton.__file__,
                    "device": torch.cuda.get_device_name(0),
                    "arch": torch.cuda.get_device_properties(0).gcnArchName,
                    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                }
            ),
            flush=True,
        )

    def test_zero_input_and_padding(self):
        from rtp_llm.models_py.modules.base.common.rocm_fp8 import native_fp8_dtype

        rows = torch.zeros((1, 320), device="cuda", dtype=torch.bfloat16)
        output = torch.empty(
            (16, 320), device="cuda", dtype=native_fp8_dtype(rows.device)
        )
        quantized, scale = quantize_input(rows, output=output)
        self.assertEqual(quantized.data_ptr(), output.data_ptr())
        self.assertTrue(torch.isfinite(scale).all().item())
        self.assertGreater(scale.item(), 0)
        self.assertEqual(torch.count_nonzero(quantized.view(torch.uint8)).item(), 0)

    def test_dense_fp8_actual_shapes_eager_and_replay(self):
        from rtp_llm.models_py.modules.base.common.fp8_linear import (
            Fp8Linear,
            project_fp8_pair,
        )

        torch.manual_seed(42)
        # Qwen4 HC, PLE and an aligned attention-sized projection.
        for n, k in ((10240, 320), (640, 1280), (2560, 2560)):
            weight = (torch.randn(n, k) * 0.125).to(torch.float8_e4m3fn).cuda()
            projection = Fp8Linear(weight, torch.tensor([0.25], device="cuda"))
            self.assertIsNone(projection.compute_weight)
            from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
                PreparedWeightDict,
                record_projection_consumer,
            )
            from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

            inventory_weights = PreparedWeightDict()
            inventory_weights["test"] = weight
            inventory_weights["test" + FP8_STORAGE_SCALE_SUFFIX] = torch.tensor(
                [0.25], device="cuda"
            )
            record_projection_consumer(inventory_weights, "test", projection)
            entry = inventory_weights.consumer_views["test"]
            self.assertIs(entry.storage, weight)
            self.assertIs(entry.compute, projection.weight)
            self.assertIs(entry.compute_scale, projection.scale)
            for m in (1, 4, 16, 257):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                expected = torch.nn.functional.linear(
                    x,
                    weight.cpu()
                    .float()
                    .mul(0.25)
                    .to(device="cuda", dtype=torch.bfloat16),
                )
                out = projection(x)
                self.assertTrue(torch.isfinite(out).all().item())
                error = (
                    out.float() - expected.float()
                ).norm() / expected.float().norm()
                self.assertLess(error.item(), 0.08)
                if m in (1, 16):
                    # Warm up allocator and BLAS on a side stream before capture.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            projection(x)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = projection(x)
                    x.mul_(0.5)
                    graph.replay()
                    torch.testing.assert_close(captured, projection(x), rtol=0, atol=0)
                    pair = project_fp8_pair(x, projection, projection)
                    torch.testing.assert_close(pair[0], projection(x), rtol=0, atol=0)

    def test_unaligned_shared_gate_and_fn_dequantization(self):
        from rtp_llm.models_py.modules.base.common.fp8_linear import (
            Fp8Linear,
            dequantize_parameter,
        )
        from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

        # Actual Qwen4 shared-expert gate; scalar output cannot use scaled_mm.
        torch.manual_seed(73)
        source = (torch.randn(1, 2560) * 0.125).to(torch.float8_e4m3fn)
        weight = source.cuda()
        scale = torch.tensor([0.25], device="cuda")
        reference = (source.float() * 0.25).to(torch.bfloat16).cuda()
        projection = Fp8Linear(weight, scale)
        self.assertFalse(projection.use_scaled_mm)
        self.assertIsNotNone(projection.compute_weight)
        torch.testing.assert_close(projection.compute_weight, reference, rtol=0, atol=0)
        restored = dequantize_parameter(
            {"gate": weight, "gate" + FP8_STORAGE_SCALE_SUFFIX: scale}, "gate"
        )
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            weight.cpu().view(torch.uint8), source.view(torch.uint8)
        )
        for rows in (1, 16):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(x, reference)
            torch.testing.assert_close(projection(x), expected, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    projection(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = projection(x)
            x.mul_(0.5)
            graph.replay()
            torch.testing.assert_close(
                result, torch.nn.functional.linear(x, reference), rtol=0, atol=0
            )

    def test_embedding_eager_and_changed_token_replay(self):
        from rtp_llm.models_py.modules.base.common.fp8_embedding import (
            Fp8VocabEmbedding,
        )

        weight = (
            torch.arange(256, dtype=torch.uint8)
            .reshape(16, 16)
            .view(torch.float8_e4m3fn)
            .cuda()
        )
        with torch.device("cuda"):
            embedding = Fp8VocabEmbedding(weight, 1, weight.device, torch.tensor([0.5]))
        tokens = torch.tensor([0, 3, 8, 14], device="cuda")
        for _ in range(3):
            embedding(tokens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = embedding(tokens)
        tokens.copy_(torch.tensor([1, 2, 4, 5], device="cuda"))
        graph.replay()
        expected = weight.cpu().float().to(torch.bfloat16)[tokens.cpu()] * 0.5
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def test_embedding_tp_column_order_contract(self):
        from unittest.mock import patch

        from rtp_llm.models_py.modules.base.common.fp8_embedding import (
            Fp8VocabEmbedding,
        )

        weight = torch.arange(64).reshape(16, 4).float().to(torch.float8_e4m3fn).cuda()
        embedding = Fp8VocabEmbedding(weight, 2, weight.device, torch.tensor([0.5]))
        tokens = torch.tensor([1, 4, -1], device="cuda")
        mask = torch.tensor([1, 0, 1], device="cuda")
        local = vocab_rows(
            tokens, weight, embedding.weight_scale, embedding.rocm_decode_table, mask
        )
        # Mock only communication to verify row/column order and HC repeat.
        peer = local * 2
        with patch(
            "rtp_llm.models_py.modules.base.common.fp8_embedding.all_gather",
            return_value=torch.cat((local, peer), dim=0),
        ):
            expected = torch.cat((local, peer), dim=1)
            torch.testing.assert_close(
                embedding(tokens, text_tokens_mask=mask), expected
            )
            torch.testing.assert_close(
                embedding.forward_hyper(tokens, branches=4, text_tokens_mask=mask),
                expected.repeat(1, 4),
            )

    def test_cpu_ple_eager_and_prepared_replay(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding

        weight = torch.arange(64).reshape(16, 4).float().to(torch.float8_e4m3fn)
        embedding = Qwen4ExpNGramEmbedding(
            [weight],
            torch.tensor([16], device="cuda"),
            torch.tensor([0], device="cuda"),
            torch.tensor([1, 3], device="cuda"),
            ngram_size=2,
            eos_token_id=0,
            weight_scale=torch.tensor([0.5]),
        )
        ids = torch.tensor([[[0], [3]]], device="cuda")
        expected = embedding._gather_torch(ids.cpu())
        torch.testing.assert_close(embedding._gather_local(ids).cpu(), expected)
        embedding.prepare_cpu_graph_rows(ids, 7)
        with embedding.lookup_context(True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = embedding._gather_local(ids) + 0
        pointer = embedding._cpu_graph_rows.data_ptr()
        ids.add_(1)
        embedding.prepare_cpu_graph_rows(ids, 7)
        self.assertEqual(pointer, embedding._cpu_graph_rows.data_ptr())
        graph.replay()
        torch.testing.assert_close(output.cpu(), embedding._gather_torch(ids.cpu()))
        self.assertEqual(embedding.shards[0].device.type, "cpu")
        embedding.close()


if __name__ == "__main__":
    unittest.main()
