"""Release-name Qwen4 MTP checkpoint-to-projection tests (CPU; not model smoke)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import requires_native_fst
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp_mtp.input_owner import Qwen4MTPInputOwner
from rtp_llm.models_py.registry import register_model


@register_model("_q4_mtp_input_test")
class _Fixture(RtpModule):
    fastsafetensors_load_isolated = True

    def __init__(self, config, load):
        super().__init__()
        self.mtp = Qwen4MTPInputOwner(16, 3, norm_eps=1e-6, fp8_storage=config.fp8)


def _weights():
    rng = torch.Generator().manual_seed(319)
    return {
        "mtp." + name + ".weight": torch.randn(shape, generator=rng).bfloat16()
        for name, shape in (
            ("fc_embedding", (16, 16)),
            ("fc_hidden", (16, 16)),
            ("pre_fc_norm_hidden", (48,)),
            ("pre_fc_norm_embedding", (16,)),
        )
    }


def _reference(values, x, h, fp8):
    def tensor(name):
        return values["mtp." + name + ".weight"]

    def norm(x, name):
        y = x.float()
        return (
            y
            * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * (1 + tensor(name).float())
        ).bfloat16()

    def fc(x, name):
        weight = tensor(name)
        if fp8:
            scale = weight.float().abs().amax().clamp_min(1e-12) / 448.0
            weight = (
                (weight.float() / scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .float()
                * scale
            ).bfloat16()
        return torch.nn.functional.linear(x, weight)

    token = fc(norm(x, "pre_fc_norm_embedding"), "fc_embedding")
    hc = fc(norm(h, "pre_fc_norm_hidden").reshape(-1, 16), "fc_hidden")
    return (token[:, None, :] + hc.reshape(-1, 3, 16)).flatten(-2)


class MTPProjectionCheckpointTest(unittest.TestCase):
    def load(self, folder, fp8, method="scratch"):
        loader = NewModelLoader(
            SimpleNamespace(model_type="_q4_mtp_input_test", fp8=fp8),
            NewLoaderConfig(
                device="cpu", compute_dtype=torch.bfloat16, load_method=method
            ),
            model_path=folder,
        )
        model = loader.load()
        self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
        return model.mtp

    def compare(self, fp8, method):
        values = _weights()
        with tempfile.TemporaryDirectory() as folder:
            save_file(values, str(Path(folder, "model.safetensors")))
            owner = self.load(folder, fp8, method)
        x = torch.arange(32).reshape(2, 16).bfloat16()
        h = torch.arange(96).reshape(2, 48).bfloat16() - 32
        torch.testing.assert_close(
            owner(x, h), _reference(values, x, h, fp8), rtol=0, atol=0
        )
        self.assertIs(owner._runtime.hidden_gamma, owner.pre_fc_norm_hidden.weight)
        self.assertIs(owner._runtime.fc_hidden, owner.fc_hidden)
        self.assertEqual(
            set("mtp." + n for n, _ in owner.named_parameters()), set(values)
        )
        self.assertEqual(
            owner.fc_hidden.weight.dtype, torch.float8_e4m3fn if fp8 else torch.bfloat16
        )
        with self.assertRaisesRegex(RuntimeError, "target hidden width"):
            owner(x, h[:, :16])

    def test_scratch_source_bf16_and_fp8_storage(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                self.compare(fp8, "scratch")

    @requires_native_fst
    def test_native_fst_with_released_stage_buffers(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                self.compare(fp8, "fastsafetensors")

    def test_malformed_or_missing_global_does_not_publish(self):
        for missing in (False, True):
            values = _weights()
            key = "mtp.pre_fc_norm_hidden.weight"
            if missing:
                del values[key]
            else:
                values[key] = values[key][:16]
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                save_file(values, str(Path(folder, "model.safetensors")))
                with self.assertRaises((RuntimeError, ValueError)):
                    self.load(folder, False)


if __name__ == "__main__":
    unittest.main()
