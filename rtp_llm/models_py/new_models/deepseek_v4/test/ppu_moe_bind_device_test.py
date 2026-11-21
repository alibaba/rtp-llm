"""Non-skipping M890P checkpoint -> native grouped-MXFP4 executor test."""

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import prepare_fp4_weight_scale_mxfp4
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_tp_moe import PpuTPMoE
from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider
from rtp_llm.models_py.new_models.deepseek_v4.test.ppu_moe_owner_test import (
    Fixture,
    checkpoint,
    config,
)
from rtp_llm.models_py.registry import MODEL_REGISTRY


class PpuBindTest(unittest.TestCase):
    def test_bind_loaded_operands_to_native_grouped_runtime(self):
        self.assertTrue(is_ppu(), "Requires a real M890P worker")
        cfg = config(1, 0, 1)
        cfg.provider = PpuModuleProvider({})
        cfg.is_decode_role, cfg.max_tokens_per_rank = False, 32
        cfg.world_rank, cfg.world_size, cfg.moe_strategy = 0, 1, "auto"
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            MODEL_REGISTRY, {cfg.model_type: Fixture}
        ):
            save_file(checkpoint(), os.path.join(directory, "model.safetensors"))
            owner = (
                NewModelLoader(
                    cfg,
                    NewLoaderConfig(device="cuda", load_method="scratch"),
                    model_path=directory,
                )
                .load()
                .ffn
            )
        w13_ptr, w2_ptr = owner.routed_w13.data_ptr(), owner.routed_w2.data_ptr()
        expected13 = prepare_fp4_weight_scale_mxfp4(owner.routed_s13)
        expected2 = prepare_fp4_weight_scale_mxfp4(owner.routed_s2)
        runtime = owner.bind_runtime(cfg)
        self.assertIs(type(runtime), PpuTPMoE)
        self.assertIs(owner.bind_runtime(cfg), runtime)
        self.assertEqual(runtime._strategy._ppu_w13.data_ptr(), w13_ptr)
        self.assertEqual(runtime._strategy._ppu_w2.data_ptr(), w2_ptr)
        torch.testing.assert_close(runtime._strategy._ppu_s13, expected13)
        torch.testing.assert_close(runtime._strategy._ppu_s2, expected2)
        self.assertIsNone(owner.routed_w13)
        self.assertIsNone(owner.shared_s13)


if __name__ == "__main__":
    unittest.main()
