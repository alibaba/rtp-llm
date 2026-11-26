"""K2.5 routed experts: real CT checkpoint ownership before CUDA repacking."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import CompressedW4A8Int4PerChannelQuantConfig
from rtp_llm.config.quant_config import QuantizationConfig as CheckpointQuantConfig
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.weight_mapper import get_all_weights


def owner(tp=1, tr=0, ep=1, er=0, source=None):
    if source is None:
        source = CompressedW4A8Int4PerChannelQuantConfig(bits=4, group_size=32)
    return BaseMoEExperts(
        num_experts=4,
        top_k=1,
        hidden_size=64,
        moe_intermediate_size=128,
        tp_size=tp,
        tp_rank=tr,
        ep_size=ep,
        ep_rank=er,
        params_dtype=torch.bfloat16,
        model_config=SimpleNamespace(
            activation_type="SiGLU", quant_config=source, num_layers=1
        ),
        parallelism_config=SimpleNamespace(
            dp_size=1, dp_rank=0, world_size=max(tp, ep), local_world_size=max(tp, ep)
        ),
        moe_config=None,
        quant_config=QuantizationConfig(
            moe_checkpoint_method_key(source), source_config=source
        ),
        layer_idx=0,
    )


class KimiCompressedExpertsTest(unittest.TestCase):
    def test_real_compressed_checkpoint_source_layout_across_rank_views(self):
        source = {}
        for expert in range(4):
            for idx, (proj, n, k) in enumerate(
                (("gate", 128, 64), ("up", 128, 64), ("down", 64, 128))
            ):
                prefix = f"{expert}.{proj}_proj."
                raw = torch.arange(n * k // 8, dtype=torch.int32).reshape(n, k // 8)
                raw = raw * 13001 + 127 * idx + expert
                source[prefix + "weight_packed"] = raw
                source[prefix + "weight_shape"] = torch.tensor([n, k])
                source[prefix + "weight_scale"] = (
                    torch.arange(n * k // 32).reshape(n, k // 32) % 11
                    + expert
                    + idx
                    + 1
                ).to(torch.bfloat16)

        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "experts.safetensors")
            save_file(source, path)
            groups = {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "group_size": 32,
                        "strategy": "group",
                    },
                    "targets": ["Linear"],
                    "input_activations": None,
                }
            }
            (Path(temp) / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "quantization_config": {
                                "quant_method": "compressed-tensors",
                                "config_groups": groups,
                            }
                        }
                    }
                )
            )
            parsed = CheckpointQuantConfig.load_from_ckpt(temp)
            self.assertIsInstance(parsed, CompressedW4A8Int4PerChannelQuantConfig)
            for read, (tp, tr, ep, er) in itertools.product(
                checkpoint_readers(get_all_weights, iter_fst_range_tensors),
                ((1, 0, 1, 0), (2, 0, 1, 0), (2, 1, 1, 0), (1, 0, 2, 1)),
            ):
                with self.subTest(reader=read.__name__, tp=tp, tr=tr, ep=ep, er=er):
                    layer = owner(tp, tr, ep, er, parsed)
                    self.assertTrue(layer.quant_method.compressed)
                    self.assertEqual(layer.quant_method.group_size, 32)
                    for name, value in read([path]):
                        layer.load_weights({name: value})
                    self.assertTrue(layer.checkpoint_load_complete())
                    for local_id in range(4 // ep):
                        expert = er * (4 // ep) + local_id
                        for proj in ("up", "gate", "down"):
                            prefix = f"{expert}.{proj}_proj."
                            axis = 1 if proj == "down" else 0
                            expected = source[prefix + "weight_packed"].view(torch.int8)
                            torch.testing.assert_close(
                                getattr(layer, f"_w4a8_{proj}_packed")[local_id],
                                expected.chunk(tp, axis)[tr],
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                getattr(layer, f"_w4a8_{proj}_scale")[local_id],
                                source[prefix + "weight_scale"].chunk(tp, axis)[tr],
                                rtol=0,
                                atol=0,
                            )

    def test_bad_orientation_and_broadcastable_scale_are_rejected(self):
        layer = owner()
        method = layer.quant_method
        for proj in ("up_proj", "gate_proj"):
            with self.assertRaisesRegex(ValueError, "weight_shape"):
                method.dispatch_scale(
                    layer, 0, proj, "weight_shape", torch.tensor([64, 128])
                )
            with self.assertRaisesRegex(ValueError, "source shape"):
                method.dispatch_scale(
                    layer, 0, proj, "weight_scale", torch.ones(128, 1)
                )
        with self.assertRaisesRegex(ValueError, "source shape"):
            method.dispatch_weight(
                layer,
                0,
                "down_proj",
                "weight_packed",
                torch.ones(1, 16, dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "down scale source shape"):
            method.dispatch_scale(layer, 0, "down_proj", "weight_scale", torch.ones(4))


if __name__ == "__main__":
    unittest.main()
