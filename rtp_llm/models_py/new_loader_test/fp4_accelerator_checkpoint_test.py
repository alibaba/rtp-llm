"""Dedicated SM100/gfx950 end-to-end loader tests; never run on CPU shards.

The two real reader paths must produce the same output from the same native
executor. This is loader parity, not an independent kernel accuracy oracle.
"""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import ModelOptFp4Config, MXFp4QuarkQuantConfig
from rtp_llm.models_py.layers.linear import ColumnParallelLinear, QKVParallelLinear
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.ops import MoeConfig, ParallelismConfig


class _AcceleratorRoot(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        if config.recipe == "dense":
            self.projection = ColumnParallelLinear(
                128,
                128,
                params_dtype=torch.bfloat16,
                quant_config=QuantizationConfig("modelopt_fp4"),
                prefix="projection",
            )
            return
        if config.recipe == "qkv_cache":
            self.projection = QKVParallelLinear(
                hidden_size=128,
                num_heads=8,
                num_kv_heads=4,
                head_dim=16,
                params_dtype=torch.bfloat16,
                quant_config=QuantizationConfig("modelopt_fp4"),
                prefix="projection",
                bias=False,
            )
            return
        source = (
            MXFp4QuarkQuantConfig(is_quanted=True)
            if config.recipe == "quark"
            else ModelOptFp4Config(4, 16, config.recipe != "online")
        )
        model = ModelConfig()
        model.model_type = "qwen_3_moe"
        model.data_type = "bf16"
        model.hidden_size = model.moe_inter_size = 128
        model.expert_num = 2
        model.moe_k = 1
        model.num_layers = 1
        model.activation_type = "SiGLU"
        model.quant_config = source
        parallel = ParallelismConfig()
        moe = MoeConfig()
        moe.use_all_gather = True
        moe.fp4_moe_op = "trtllm"
        self.experts = BaseMoEExperts(
            num_experts=2,
            top_k=1,
            hidden_size=128,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            moe_intermediate_size=128,
            params_dtype=torch.bfloat16,
            model_config=model,
            parallelism_config=parallel,
            moe_config=moe,
            quant_config=QuantizationConfig(moe_checkpoint_method_key(source), source),
            layer_idx=0,
            prefix="experts",
        )


def _weights(recipe):
    if recipe == "qkv_cache":
        result = {
            "projection.k_proj.k_scale": torch.tensor(77.0),
            "projection.v_proj.v_scale": torch.tensor(99.0),
        }
        for proj, rows in (("q", 128), ("k", 64), ("v", 64)):
            stem = f"projection.{proj}_proj."
            result[stem + "weight"] = torch.full((rows, 64), 17, dtype=torch.uint8)
            result[stem + "weight_scale"] = torch.ones(rows, 8).to(torch.float8_e4m3fn)
            result[stem + "weight_scale_2"] = torch.tensor(0.25)
            result[stem + "input_scale"] = torch.tensor(1.0)
        return result
    if recipe == "dense":
        return {
            "projection.weight": torch.full((128, 64), 17, dtype=torch.uint8),
            "projection.weight_scale": torch.ones(128, 8).to(torch.float8_e4m3fn),
            "projection.weight_scale_2": torch.tensor(0.25),
            "projection.input_scale": torch.tensor(1.0),
        }
    result = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stem = f"experts.{expert}.{projection}."
            if recipe == "online":
                result[stem + "weight"] = (
                    torch.arange(128 * 128).reshape(128, 128).float().sin() / 128
                ).bfloat16()
            else:
                result[stem + "weight"] = torch.full((128, 64), 17, dtype=torch.uint8)
                if recipe == "quark":
                    result[stem + "weight_scale"] = torch.full(
                        (128, 4), 127, dtype=torch.uint8
                    )
                else:
                    result[stem + "weight_scale"] = torch.ones(128, 8).to(
                        torch.float8_e4m3fn
                    )
                    result[stem + "weight_scale_2"] = torch.tensor(0.25)
                    result[stem + "input_scale"] = torch.tensor(1.0)
    return result


class Fp4AcceleratorCheckpointTest(unittest.TestCase):
    def test_real_gpu_scratch_fst_forward(self):
        platform = os.environ.get("FP4_TEST_PLATFORM")
        self.assertIn(platform, ("sm100", "gfx950"), "Use a dedicated platform target")
        self.assertTrue(
            torch.cuda.is_available(), "This GPU target must not silently skip"
        )
        if platform == "sm100":
            self.assertIsNone(torch.version.hip)
            self.assertGreaterEqual(torch.cuda.get_device_capability()[0], 10)
            recipes = ("dense", "qkv_cache", "offline", "online")
        else:
            self.assertTrue(torch.version.hip)
            self.assertEqual(
                torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx950"
            )
            recipes = ("quark",)
        with mock.patch.dict(
            MODEL_REGISTRY, {"fp4_accelerator_fixture": _AcceleratorRoot}
        ):
            for recipe in recipes:
                with self.subTest(
                    recipe=recipe
                ), tempfile.TemporaryDirectory() as directory:
                    save_file(
                        _weights(recipe), os.path.join(directory, "model.safetensors")
                    )
                    outputs = []
                    for method in ("scratch", "fastsafetensors"):
                        loader = NewModelLoader(
                            types.SimpleNamespace(
                                model_type="fp4_accelerator_fixture", recipe=recipe
                            ),
                            NewLoaderConfig(
                                device="cuda:0",
                                compute_dtype=torch.bfloat16,
                                load_method=method,
                            ),
                            model_path=directory,
                        )
                        model = loader.load()
                        self.assertEqual(loader.loaded_method.value, method)
                        if recipe == "qkv_cache":
                            self.assertEqual(
                                model.projection.weight_scale_2.item(), 0.25
                            )
                            self.assertEqual(model.projection.input_scale.item(), 1.0)
                        x = (
                            torch.arange(8 * 128, device="cuda")
                            .float()
                            .reshape(8, 128)
                            .sin()
                            / 128
                        ).bfloat16()
                        with torch.inference_mode():
                            if recipe in ("dense", "qkv_cache"):
                                output = model.projection(x)
                            else:
                                output = model.experts(
                                    x,
                                    torch.ones(8, 1, device="cuda"),
                                    (torch.arange(8, device="cuda") % 2)
                                    .reshape(8, 1)
                                    .int(),
                                )
                        self.assertTrue(torch.isfinite(output).all())
                        outputs.append(output.cpu())
                        del model
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
