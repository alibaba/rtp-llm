"""CPU checkpoint staging only; Cutlass packing/GPU inference is not emulated."""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import (
    CompressedW4A8Int4PerChannelQuantConfig,
    W4a8Int4PerChannelQuantConfig,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.test.test_w8a8_int8_moe import _MoeModel
from rtp_llm.models_py.registry import MODEL_REGISTRY


class W4A8CheckpointTest(unittest.TestCase):
    def checkpoint(self, compressed=True):
        tensors = {}
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                stem = f"experts.{expert}.{projection}."
                if compressed:
                    # Eight nibbles per int32; this checks byte-level TP slicing.
                    tensors[stem + "weight_packed"] = (
                        torch.arange(64 * 8, dtype=torch.int32).reshape(64, 8)
                        + expert * 1024
                        + (17 if projection == "gate_proj" else 0)
                    )
                    tensors[stem + "weight_scale"] = (
                        torch.arange(128).reshape(64, 2).float() + expert + 1
                    ) / 128
                    tensors[stem + "weight_shape"] = torch.tensor([64, 64])
                else:
                    tensors[stem + "weight"] = (
                        torch.arange(4096).reshape(64, 64) / 128 + expert
                    ).to(torch.bfloat16)
        return tensors

    def load(
        self,
        tensors,
        compressed=True,
        method="scratch",
        tp=2,
        rank=1,
        ep=1,
        redundant=0,
    ):
        source = (
            CompressedW4A8Int4PerChannelQuantConfig(
                bits=4, group_size=32, is_quanted=True
            )
            if compressed
            else W4a8Int4PerChannelQuantConfig(bits=4, group_size=32, is_quanted=False)
        )
        config = types.SimpleNamespace(
            model_type="w4a8_cpu_fixture",
            expert_num=4,
            moe_k=1,
            moe_style=2,
            hidden_size=64,
            intermediate_size=64,
            activation_type="SiGLU",
            data_type="bf16",
            quant_config=source,
            num_layers=1,
            phy2log_path="",
            moe_topk_group=1,
            attn_config=types.SimpleNamespace(head_num=2),
            eplb_config=types.SimpleNamespace(
                redundant_expert=redundant,
                phy_exp_num=lambda logical: logical + redundant,
                enable_eplb=lambda: False,
            ),
        )
        with tempfile.TemporaryDirectory() as path:
            save_file(tensors, os.path.join(path, "model.safetensors"))
            with mock.patch.dict(MODEL_REGISTRY, {config.model_type: _MoeModel}):
                loader = NewModelLoader(
                    config,
                    NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        load_method=method,
                        force_cpu_load_weights=method == "fastsafetensors",
                        tp_size=tp,
                        tp_rank=rank,
                        ep_size=ep,
                        ep_rank=rank if ep > 1 else 0,
                        quant_config=QuantizationConfig(source.get_method(), source),
                    ),
                    model_path=path,
                )
                result = loader.load().experts
                self.assertEqual(loader._selected_load_method.value, method)
                self.assertFalse(result._post_load_done)
                self.assertIsNone(result.fused_moe)
                return result

    def test_real_scratch_and_native_fst_staging_tp_ep_redundant(self):
        for compressed in (False, True):
            for ep, redundant in ((1, 0), (2, 0), (2, 2)):
                for rank in range(2):
                    with self.subTest(
                        compressed=compressed, ep=ep, redundant=redundant, rank=rank
                    ):
                        values = self.checkpoint(compressed)
                        scratch = self.load(
                            values, compressed, rank=rank, ep=ep, redundant=redundant
                        )
                        fst = self.load(
                            values,
                            compressed,
                            "fastsafetensors",
                            rank=rank,
                            ep=ep,
                            redundant=redundant,
                        )
                        if compressed:
                            names = [
                                f"_w4a8_{p}_{f}"
                                for p in ("up", "gate", "down")
                                for f in ("packed", "scale")
                            ]
                        else:
                            names = ["w13", "w2"]
                        for name in names:
                            torch.testing.assert_close(
                                getattr(scratch, name),
                                getattr(fst, name),
                                rtol=0,
                                atol=0,
                            )
                        for local, logical in enumerate(
                            scratch._expert_placement.local_physical_to_logical
                        ):
                            width = scratch.moe_inter_tp
                            begin = scratch.moe_expert_tp_rank * width
                            for projection in ("up", "gate", "down"):
                                stem = f"experts.{logical}.{projection}_proj."
                                if compressed:
                                    packed = values[stem + "weight_packed"].view(
                                        torch.int8
                                    )
                                    scale = values[stem + "weight_scale"].to(
                                        torch.bfloat16
                                    )
                                    packed = (
                                        packed[:, begin // 2 : (begin + width) // 2]
                                        if projection == "down"
                                        else packed[begin : begin + width]
                                    )
                                    scale = (
                                        scale[:, begin // 32 : (begin + width) // 32]
                                        if projection == "down"
                                        else scale[begin : begin + width]
                                    )
                                    torch.testing.assert_close(
                                        getattr(scratch, f"_w4a8_{projection}_packed")[
                                            local
                                        ],
                                        packed,
                                        rtol=0,
                                        atol=0,
                                    )
                                    torch.testing.assert_close(
                                        getattr(scratch, f"_w4a8_{projection}_scale")[
                                            local
                                        ],
                                        scale,
                                        rtol=0,
                                        atol=0,
                                    )

    def test_missing_scale_wrong_shape_dtype_and_nonfinite_rejected(self):
        variants = []
        values = self.checkpoint()
        del values["experts.0.gate_proj.weight_scale"]
        variants.append(values)
        for bad in (torch.tensor([32, 128]), torch.tensor([64.0, 64.0])):
            values = self.checkpoint()
            values["experts.0.gate_proj.weight_shape"] = bad
            variants.append(values)
        values = self.checkpoint()
        values["experts.0.gate_proj.weight_packed"] = torch.zeros(64, 8)
        variants.append(values)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            values = self.checkpoint()
            values["experts.0.gate_proj.weight_scale"][0, 0] = bad
            variants.append(values)
        for values in variants:
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    with self.assertRaises((ValueError, RuntimeError, TypeError)):
                        self.load(values, method=method)

    def test_cpu_staging_cannot_publish_cuda_layout(self):
        layer = self.load(self.checkpoint())
        with self.assertRaisesRegex(RuntimeError, "CPU checkpoint staging"):
            layer.prepare_eplb_reload_weights()


if __name__ == "__main__":
    unittest.main()
