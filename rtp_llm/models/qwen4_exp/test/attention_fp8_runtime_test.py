"""Runtime dispatch checks for the optional Qwen attention FP8 mode."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.config.quant_config import ModelOptFp4Config


class QwenAttentionFp8RuntimeTest(unittest.TestCase):
    def test_runtime_attention_quant_config_does_not_change_moe_config(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextDecoderLayer
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer

        original = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        cfg = SimpleNamespace(quant_config=original, _qwen4_attention_fp8=True)
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        self.assertEqual(
            layer._attention_quant_config(cfg, {}).get_method(), "FP8_PER_BLOCK"
        )
        self.assertIs(cfg.quant_config, original)
        cfg._qwen4_attention_fp8 = False
        self.assertIs(layer._attention_quant_config(cfg, {}), original)
        base = Qwen3NextDecoderLayer.__new__(Qwen3NextDecoderLayer)
        cfg._qwen4_attention_fp8 = True
        self.assertIs(base._attention_quant_config(cfg, {}), original)

    def test_gdn_bf16_override_preserves_full_attention_fp8(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
        from rtp_llm.ops import HybridAttentionType

        original = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        cfg = SimpleNamespace(
            quant_config=original, _qwen4_attention_fp8=True, _qwen4_gdn_bf16=True
        )
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        layer.layer_type = HybridAttentionType.LINEAR
        self.assertIs(layer._attention_quant_config(cfg, {}), original)
        layer.layer_type = HybridAttentionType.NONE
        self.assertEqual(
            layer._attention_quant_config(cfg, {}).get_method(), "FP8_PER_BLOCK"
        )
        self.assertIs(cfg.quant_config, original)
        cfg._qwen4_gdn_bf16 = False
        layer.layer_type = HybridAttentionType.LINEAR
        self.assertEqual(
            layer._attention_quant_config(cfg, {}).get_method(), "FP8_PER_BLOCK"
        )

    def test_missing_optional_fp4_does_not_block_fp8_registration(self):
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl import cuda

        real_import = builtins.__import__

        def import_without_fp4(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fp4_linear" and level == 1:
                raise ModuleNotFoundError("cutlass test sentinel")
            return real_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            cuda.__name__ + ".registration_probe", Path(cuda.__file__)
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = cuda.__name__
        with mock.patch.object(
            builtins, "__import__", import_without_fp4
        ), mock.patch.object(LinearFactory, "register") as register, mock.patch(
            "rtp_llm.models_py.utils.arch.is_cuda", return_value=True
        ), mock.patch(
            "rtp_llm.models_py.utils.arch.get_sm", return_value=(12, 0)
        ):
            spec.loader.exec_module(module)
        names = [args[0].__name__ for args, _ in register.call_args_list]
        self.assertEqual(
            names, ["CudaF16Linear", "CudaFp8PerTensorLinear", "CudaFp8GEMMLinear"]
        )


if __name__ == "__main__":
    unittest.main()
