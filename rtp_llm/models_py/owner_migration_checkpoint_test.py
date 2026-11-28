"""Rank-local stacked, per-expert and online-FP8 checkpoint contracts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.config.quant_config import Fp8PerChannelCompressedQuantConfig
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import register_model


@register_model("_stacked_expert_source_contract")
class _Owner(RtpModule):
    fastsafetensors_load_isolated = True

    def __init__(self, config, load):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=1,
            hidden_size=6,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=load.ep_size,
            ep_rank=load.ep_rank,
            params_dtype=torch.float32,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1, dp_rank=0, world_size=2, local_world_size=2
            ),
            moe_config=None,
            quant_config=load.quant_config,
            layer_idx=0,
        )


def _weights():
    gate = torch.arange(96).float().view(4, 4, 6) + 1
    up = gate + 100
    down = (gate - 50).transpose(1, 2).contiguous()
    return gate, up, down


def _load(entries, method, ep_size):
    config = SimpleNamespace(
        model_type="_stacked_expert_source_contract",
        num_layers=1,
        expert_num=4,
        activation_type="SiGLU",
        quant_config=None,
    )
    with tempfile.TemporaryDirectory() as d:
        save_file(entries, str(Path(d, "model.safetensors")))
        loader = NewModelLoader(
            config,
            NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.float32,
                load_method=method,
                ep_size=ep_size,
                ep_rank=ep_size - 1,
            ),
            model_path=d,
        )
        model = loader.load()
        assert loader._selected_load_method == NewLoaderLoadMethod(method)
        return model.experts


def check_stacked_routing(test):
    gate, up, down = _weights()
    sources = [
        {
            "experts.gate_up_proj": torch.cat((gate, up), dim=1),
            "experts.down_proj": down,
        },
        {
            f"experts.{i}.{proj}.weight": value[i]
            for i in range(4)
            for proj, value in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down),
            )
        },
    ]
    for entries in sources:
        for method, ep in checkpoint_readers(("scratch", 2), ("fastsafetensors", 1)):
            with test.subTest(method=method, key=next(iter(entries))):
                owned = _load(entries, method, ep)
                start = 2 if ep == 2 else 0
                torch.testing.assert_close(
                    owned.w13, torch.cat((up[start:], gate[start:]), dim=1)
                )
                torch.testing.assert_close(owned.w2, down[start:])
    bad = dict(sources[0], **{"experts.2.up_proj.weight": up[2]})
    with test.assertRaisesRegex((ValueError, RuntimeError), "[Dd]uplicate|already"):
        _load(bad, "scratch", 2)


def check_stacked_online_fp8_cpu_staging(test):
    gate, up, down = _weights()
    with tempfile.TemporaryDirectory() as d:
        source = Fp8PerChannelCompressedQuantConfig(is_quanted=False)
        config = SimpleNamespace(
            model_type="_stacked_expert_source_contract",
            num_layers=1,
            expert_num=4,
            activation_type="SiGLU",
            quant_config=source,
        )
        load = NewLoaderConfig(
            device="cpu",
            ep_size=1,
            ep_rank=0,
            quant_config=QuantizationConfig("fp8_per_channel_online", source),
        )
        path = str(Path(d, "model.safetensors"))
        save_file(
            {"gate_up_proj": torch.cat((gate, up), dim=1), "down_proj": down}, path
        )
        # This is the staging module inside accelerator loading, not a claim
        # that NewModelLoader supports CPU FP8 inference as its runtime target.
        owned = _Owner(config, load).experts
        with safe_open(path, framework="pt", device="cpu") as file:
            for name in file.keys():
                owned.load_weights({name: file.get_tensor(name)})
        owned.validate_weights_loaded()
        # This public hook must retain checkpoint bytes, not choose a runtime
        # FP8 encoding from whichever accelerator happens to be visible.
        pointers = (owned.w13.data_ptr(), owned.w2.data_ptr())
        owned.process_weights_after_loading()
        test.assertEqual((owned.w13.data_ptr(), owned.w2.data_ptr()), pointers)
        test.assertEqual(owned.w13.device.type, "cpu")
        test.assertEqual(owned.w13.dtype, torch.float32)
        test.assertEqual(owned.w2.dtype, torch.float32)
        torch.testing.assert_close(owned.w13, torch.cat((up, gate), 1), rtol=0, atol=0)
        torch.testing.assert_close(owned.w2, down, rtol=0, atol=0)
        test.assertFalse(owned._post_load_done)
        test.assertIsNone(owned.fused_moe)
        return owned


def check_stacked_online_fp8(test, runtime_device):
    from rtp_llm.models_py.quant_methods.fp8 import _runtime_fp8_dtype

    owned = check_stacked_online_fp8_cpu_staging(test)
    target = torch.device(runtime_device)
    test.assertEqual(target.type, "cuda")
    # Like ordinary load and EPLB reload, migrate before device-specific
    # quantization. The tiny fixture tests quantization, not fused-MoE execution.
    owned.to(target)
    runtime_dtype = _runtime_fp8_dtype(target)
    owned.quant_method.process_weights_after_loading(owned)
    gate, up, down = _weights()
    for weights, scale, expected in (
        (owned.w13, owned.w13_scale, torch.cat((up, gate), 1)),
        (owned.w2, owned.w2_scale, down),
    ):
        test.assertEqual(weights.dtype, runtime_dtype)
        test.assertEqual(weights.device, target)
        expected = expected.to(target)
        reference_scale = expected.abs().amax(dim=-1) / torch.finfo(runtime_dtype).max
        torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)
        reference = (expected / reference_scale.unsqueeze(-1)).to(runtime_dtype)
        test.assertTrue(
            torch.equal(weights.view(torch.uint8), reference.view(torch.uint8))
        )


class OwnerCheckpointTest(unittest.TestCase):
    test_stacked_routing = check_stacked_routing
    test_stacked_online_fp8 = check_stacked_online_fp8_cpu_staging


if __name__ == "__main__":
    unittest.main()
