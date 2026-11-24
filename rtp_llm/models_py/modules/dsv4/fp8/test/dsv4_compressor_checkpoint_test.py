"""Compressor checkpoint/parameter contracts, not GPU compressor execution."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.cp import _CP_ROLE_INDEXER, _CP_ROLE_MAIN
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _CompressorCheckpointFixture(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"layers.0.attn.": ""})

    def __init__(self, model_config, load_config):
        super().__init__()
        self.compressor = CompressorFP8(
            dim=32,
            head_dim=model_config.head_dim,
            rope_head_dim=0,
            compress_ratio=model_config.ratio,
            max_batch_size=1,
            cp_role=_CP_ROLE_INDEXER if model_config.head_dim == 128 else _CP_ROLE_MAIN,
        )
        self.parameter_ids = {
            name: id(param) for name, param in self.named_parameters()
        }

    def load_weights(self, weights):
        super().load_weights(self.WEIGHTS_MAPPER.apply(weights))


class Dsv4CompressorCheckpointTest(unittest.TestCase):
    def _weights(self, head_dim=128, ratio=4):
        out_dim = head_dim * (2 if ratio == 4 else 1)
        prefix = "layers.0.attn.compressor."
        return {
            prefix + "ape": torch.full((ratio, out_dim), 1.000123, dtype=torch.float32),
            prefix + "wkv.weight": torch.full((out_dim, 32), 2.0, dtype=torch.bfloat16),
            prefix
            + "wgate.weight": torch.full((out_dim, 32), 3.0, dtype=torch.bfloat16),
            prefix + "norm.weight": torch.full((head_dim,), 4.0, dtype=torch.bfloat16),
        }

    def _load(self, weights, head_dim=128, ratio=4):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            MODEL_REGISTRY, {"dsv4_compressor_fixture": _CompressorCheckpointFixture}
        ):
            save_file(weights, os.path.join(folder, "model.safetensors"))
            return NewModelLoader(
                types.SimpleNamespace(
                    model_type="dsv4_compressor_fixture", head_dim=head_dim, ratio=ratio
                ),
                NewLoaderConfig(
                    device="cpu", compute_dtype=torch.bfloat16, load_method="scratch"
                ),
                model_path=folder,
            ).load()

    def test_real_checkpoint_fuses_into_live_parameters_for_all_layouts(self):
        for head_dim, ratio in ((128, 4), (512, 4), (512, 128)):
            with self.subTest(head_dim=head_dim, ratio=ratio):
                weights = self._weights(head_dim, ratio)
                model = self._load(weights, head_dim, ratio)
                for name, param in model.named_parameters():
                    self.assertEqual(id(param), model.parameter_ids[name])
                    expected = weights["layers.0.attn." + name]
                    torch.testing.assert_close(param, expected, rtol=0, atol=0)
                cmp = model.compressor
                self.assertEqual(cmp.ape.dtype, torch.float32)
                fused = cmp._wkv_wgate_fused
                self.assertEqual(fused.dtype, torch.bfloat16)
                self.assertEqual(cmp.wkv.weight.data_ptr(), fused.data_ptr())
                self.assertEqual(
                    cmp.wgate.weight.data_ptr(), fused[cmp.coff * head_dim :].data_ptr()
                )
                NewModelLoader._validate_loaded_weights(model)

    def test_postprocess_is_idempotent_and_same_device_to_preserves_storage(self):
        model = self._load(self._weights())
        cmp = model.compressor
        old = cmp._wkv_wgate_fused
        cmp.process_weights_after_loading()
        model.to("cpu")
        self.assertIs(cmp._wkv_wgate_fused, old)

    def test_dtype_migration_refreshes_fused_runtime_view(self):
        model = self._load(self._weights())
        cmp = model.compressor
        # Exercise a real _apply storage change without requiring a GPU.
        cmp.to(dtype=torch.float32)
        self.assertEqual(cmp._wkv_wgate_fused.dtype, torch.float32)
        self.assertEqual(cmp.wkv.weight.data_ptr(), cmp._wkv_wgate_fused.data_ptr())
        torch.testing.assert_close(
            cmp._wkv_wgate_fused[:256], torch.full((256, 32), 2.0)
        )

    def test_missing_projection_fails_before_fusing_unloaded_storage(self):
        weights = self._weights()
        del weights["layers.0.attn.compressor.wgate.weight"]
        with self.assertRaisesRegex(RuntimeError, "wgate"):
            self._load(weights)

    def test_wrong_projection_shape_is_rejected(self):
        weights = self._weights()
        weights["layers.0.attn.compressor.wkv.weight"] = torch.zeros(
            128, 32, dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(ValueError, "[Ss]hape"):
            self._load(weights)

    def test_parameter_update_is_visible_to_fused_runtime_storage(self):
        model = self._load(self._weights())
        cmp = model.compressor
        with torch.inference_mode():
            cmp.wkv.weight.fill_(5)
        torch.testing.assert_close(
            cmp._wkv_wgate_fused[:256], torch.full((256, 32), 5.0, dtype=torch.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
