"""Tiny real checkpoint-to-HC consumer regression; CPU, not full-model smoke."""

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4HyperConnection
from rtp_llm.models_py.registry import register_model


@register_model("qwen4_hc_owner_fixture")
class _CheckpointHC(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.attn_hyper_connection = Qwen4HyperConnection(
            8,
            4,
            16,
            dtype=load_config.compute_dtype,
            fp8_storage=config.get("fp8_storage", False),
        )
        self.hyper_connection_mixer = Qwen4HyperConnection(
            8,
            4,
            16,
            with_inject=False,
            dtype=load_config.compute_dtype,
            fp8_storage=config.get("fp8_storage", False),
        )


class Qwen4HyperConnectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(248)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(os.environ, {"LOAD_METHOD": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.weights = {}
        for prefix in ("attn_hyper_connection", "hyper_connection_mixer"):
            shapes = {
                "hc_norm": (32,),
                "input_mix_weight_down": (16, 32),
                "input_mix_weight_up": (32, 16),
            }
            if prefix == "attn_hyper_connection":
                shapes["block_inject_weight"] = (4, 32)
            for name, shape in shapes.items():
                self.weights[f"{prefix}.{name}.weight"] = (
                    torch.randn(shape) * 0.03
                ).to(torch.bfloat16)

    def load(self, fp8_storage=False):
        save_file(self.weights, os.path.join(self.temp.name, "model.safetensors"))
        return NewModelLoader(
            {"model_type": "qwen4_hc_owner_fixture", "fp8_storage": fp8_storage},
            NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.bfloat16,
                load_method=NewLoaderLoadMethod.SCRATCH,
            ),
            model_path=self.temp.name,
        ).load()

    def test_real_checkpoint_binds_original_runtime_and_keeps_raw_norm(self):
        model = self.load()
        for prefix in ("attn_hyper_connection", "hyper_connection_mixer"):
            owner = getattr(model, prefix)
            tensors = [
                self.weights[f"{prefix}.{name}.weight"]
                for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
            ]
            inject = self.weights.get(f"{prefix}.block_inject_weight.weight")
            reference = GatedResidual(*tensors, inject, hc_mult=4, norm_eps=1e-6)
            x = torch.randn(5, 32, dtype=torch.bfloat16)
            for actual, expected in zip(owner(x), reference(x)):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(owner.hc_norm.weight, tensors[0], rtol=0, atol=0)
            self.assertIs(owner._runtime.norm_gamma, owner.hc_norm.weight)
            self.assertIs(owner._runtime.mix_down, owner.input_mix_weight_down.weight)
            self.assertNotIn("_runtime", dict(owner.named_children()))

    def test_fp8_storage_per_owned_tensor_and_unaligned_inject_bf16(self):
        model = self.load(True)
        owner = model.attn_hyper_connection
        for name in ("input_mix_weight_down", "input_mix_weight_up"):
            projection = getattr(owner, name)
            source = self.weights[f"attn_hyper_connection.{name}.weight"].float()
            scale = source.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
            expected = (source / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.assertTrue(
                torch.equal(
                    projection.weight.view(torch.uint8), expected.view(torch.uint8)
                )
            )
            torch.testing.assert_close(projection.storage_scale, scale, rtol=0, atol=0)
        self.assertEqual(owner.block_inject_weight.weight.dtype, torch.bfloat16)
        self.assertIsNone(owner.block_inject_weight.storage_scale)
        identities = [(id(p), p.data_ptr()) for p in owner.parameters()]
        owner.process_weights_after_loading()
        self.assertEqual(
            identities, [(id(p), p.data_ptr()) for p in owner.parameters()]
        )
        self.assertTrue(
            all(p._rtp_requires_restart_for_update for p in owner.parameters())
        )
        # Execute the original per-tensor FP8 consumer, with its CPU BF16
        # compute-copy path. This is not CUDA scaled_mm validation.
        out = owner(torch.randn(2, 32, dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(out[0]).all())

    def test_missing_and_bad_shape_fail_before_runtime_binding(self):
        del self.weights["attn_hyper_connection.hc_norm.weight"]
        with self.assertRaises(RuntimeError):
            self.load()
        self.weights["attn_hyper_connection.hc_norm.weight"] = torch.zeros(
            31, dtype=torch.bfloat16
        )
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
