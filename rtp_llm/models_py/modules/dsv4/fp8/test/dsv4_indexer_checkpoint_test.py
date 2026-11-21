"""Actual V4 indexer owners/checkpoint transformations, not CUDA execution."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.weight_mapper import get_all_weights


class Dsv4IndexerCheckpointTest(unittest.TestCase):
    def _layer(self):
        return IndexerFP8(
            dim=128,
            q_lora_rank=128,
            index_n_heads=2,
            index_head_dim=128,
            rope_head_dim=64,
            index_topk=2,
            compress_ratio=4,
            max_batch_size=1,
            max_seq_len=128,
        )

    def _weights(self):
        return {
            "wq_b.weight": torch.full((256, 128), 2.0, dtype=torch.float8_e4m3fn),
            "wq_b.scale": torch.tensor([[127], [128]], dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            "weights_proj.weight": torch.arange(256).reshape(2, 128).to(torch.bfloat16),
            "compressor.ape": torch.full((4, 256), 1.000123, dtype=torch.float32),
            "compressor.wkv.weight": torch.full((256, 128), 2.0, dtype=torch.bfloat16),
            "compressor.wgate.weight": torch.full(
                (256, 128), 3.0, dtype=torch.bfloat16
            ),
            "compressor.norm.weight": torch.full((128,), 4.0, dtype=torch.bfloat16),
        }

    def _load(self, weights):
        layer = self._layer()
        with tempfile.TemporaryDirectory() as folder:
            filename = os.path.join(folder, "pytorch_model.bin")
            # Pinned 26 safetensors does not support native UE8M0 serialization.
            torch.save(weights, filename)
            layer.load_weights(get_all_weights([filename]))
        NewModelLoader._validate_loaded_weights(layer)
        return layer

    def test_real_checkpoint_owns_raw_dtype_and_projection_fold_matches_old(self):
        weights = self._weights()
        layer = self._load(weights)
        self.assertEqual(len(dict(layer.named_parameters())), 7)
        for name in ("weight", "scale"):
            torch.testing.assert_close(
                getattr(layer.wq_b, name).view(torch.uint8),
                weights["wq_b." + name].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        self.assertEqual(layer.compressor.ape.dtype, torch.float32)
        torch.testing.assert_close(
            layer.compressor.ape, weights["compressor.ape"], rtol=0, atol=0
        )
        parameter = layer.weights_proj
        storage = parameter.data_ptr()
        expected = (weights["weights_proj.weight"] * (128**-0.5 * 2**-0.5)).contiguous()
        # Only the CPU-safe conversion is exercised, not the FP8 GPU executor.
        layer.process_weights_after_loading()
        self.assertIs(layer.weights_proj, parameter)
        self.assertEqual(layer.weights_proj.data_ptr(), storage)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        layer.process_weights_after_loading()
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        layer.compressor.process_weights_after_loading()
        self.assertEqual(
            layer.compressor._wkv_wgate_fused.data_ptr(),
            layer.compressor.wkv.weight.data_ptr(),
        )

    def test_missing_projection_is_rejected_before_fold(self):
        weights = self._weights()
        del weights["weights_proj.weight"]
        with self.assertRaisesRegex(RuntimeError, "weights_proj"):
            self._load(weights)

    def test_missing_native_scale_is_rejected(self):
        weights = self._weights()
        del weights["wq_b.scale"]
        with self.assertRaisesRegex(RuntimeError, "scale"):
            self._load(weights)

    def test_wrong_projection_shape_is_rejected(self):
        weights = self._weights()
        weights["weights_proj.weight"] = torch.zeros(1, 128, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "[Ss]hape"):
            self._load(weights)

    def test_cpu_execution_is_not_silently_enabled(self):
        layer = self._load(self._weights())
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            layer.validate_runtime_device(torch.device("cpu"))

    def test_loading_preflight_rejects_each_missing_score_symbol(self):
        layer = self._layer()
        required = (
            "fp8_mqa_logits",
            "fp8_paged_mqa_logits",
            "get_paged_mqa_logits_metadata",
        )
        for absent in required:
            backend = types.SimpleNamespace(
                **{name: (lambda: None) for name in required if name != absent}
            )
            with self.subTest(symbol=absent), mock.patch.object(
                torch.version, "hip", None
            ), mock.patch.object(
                indexer, "supports_mqa_logits", return_value=True
            ), mock.patch.object(
                indexer, "supports_paged_mqa_logits", return_value=True
            ), mock.patch.object(
                layer.wq_b, "validate_runtime_device"
            ), mock.patch.dict(
                "sys.modules", {"deep_gemm": backend}
            ):
                with self.assertRaisesRegex(RuntimeError, absent):
                    layer.validate_runtime_device("cuda:0")

    def test_loading_preflight_keeps_main_triton_score_route(self):
        layer = self._layer()
        with mock.patch.object(torch.version, "hip", None), mock.patch.object(
            layer.wq_b, "validate_runtime_device"
        ), mock.patch.object(
            indexer, "supports_mqa_logits", return_value=False
        ), mock.patch.object(
            indexer, "supports_paged_mqa_logits", return_value=False
        ), mock.patch.dict(
            "sys.modules", {"deep_gemm": None}
        ):
            layer.validate_runtime_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
