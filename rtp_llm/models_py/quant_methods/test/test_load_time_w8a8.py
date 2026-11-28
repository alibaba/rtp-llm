"""Real CPU safetensors loads against the offline/main row quantization recipe."""

import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.models_py.layers.linear import RowParallelLinear
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.test.test_w8a8_int8_moe import _MoeModel
from rtp_llm.models_py.quant_methods.w8a8_load_source import W8A8LoadSource
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.utils.w8a8_int8_quant import (
    is_recipe_weight,
    quantize_weight_per_output_channel,
    scale_name_for,
)


class _Fixture(RtpModule):
    supports_expert_parallel = True
    supports_load_time_w8a8 = True

    def __init__(self, config, load):
        super().__init__()
        block = RtpModule()
        self.layers = RtpModule()
        self.layers.add_module("0", block)
        block.mlp = RtpModule()
        block.mlp.experts = _MoeModel(config, load).experts
        block.self_attn = RtpModule()
        block.self_attn.o_proj = RowParallelLinear(
            input_size=8,
            output_size=8,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            quant_config=load.quant_config,
            prefix="layers.0.self_attn.o_proj",
            params_dtype=torch.bfloat16,
            bias=False,
        )
        block.norm = RtpModule()
        block.norm.weight = torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16))


class LoadTimeW8A8Test(unittest.TestCase):
    def values(self):
        def tensor(shape):
            v = torch.arange(torch.Size(shape).numel()).reshape(shape).float() % 17 - 8
            v[..., -1] *= 200  # global K maximum deliberately lies outside rank 0
            v.reshape(-1, shape[-1])[0].zero_()
            return v.to(torch.bfloat16)

        return {
            "layers.0.mlp.experts.gate_up_proj": tensor((4, 24, 8)),
            "layers.0.mlp.experts.down_proj": tensor((4, 8, 12)),
            "layers.0.self_attn.o_proj.weight": tensor((8, 8)),
            "layers.0.norm.weight": torch.arange(8, dtype=torch.bfloat16),
        }

    def loader(self, folder, rank=0, ep=1, online=True, method="scratch"):
        source = CompressedW8A8Int8PerChannelQuantConfig(
            is_quanted=not online, load_chunk_rows=2
        )
        cfg = NS(
            model_type="qwen35_moe",
            expert_num=4,
            moe_k=1,
            moe_style=2,
            hidden_size=8,
            intermediate_size=12,
            activation_type="SiGLU",
            data_type="bf16",
            quant_config=source,
            num_layers=1,
            phy2log_path="",
            attn_config=NS(head_num=2),
            moe_topk_group=1,
            eplb_config=NS(
                redundant_expert=0, phy_exp_num=lambda n: n, enable_eplb=lambda: False
            ),
        )
        return NewModelLoader(
            cfg,
            NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.bfloat16,
                load_method=method,
                tp_size=2,
                tp_rank=rank,
                ep_size=ep,
                ep_rank=rank if ep > 1 else 0,
                quant_config=QuantizationConfig(
                    source.get_method(), source_config=source
                ),
            ),
            model_path=folder,
        )

    def test_auto_and_scratch_match_offline_for_tp_and_ep(self):
        raw = self.values()
        offline = {}
        for key, tensor in raw.items():
            if is_recipe_weight(key):
                offline[key], offline[scale_name_for(key)] = (
                    quantize_weight_per_output_channel(tensor)
                )
            else:
                offline[key] = tensor
        with (
            tempfile.TemporaryDirectory() as a,
            tempfile.TemporaryDirectory() as b,
            patch.dict(MODEL_REGISTRY, {"qwen35_moe": _Fixture}),
        ):
            save_file(raw, os.path.join(a, "model.safetensors"))
            save_file(offline, os.path.join(b, "model.safetensors"))
            for rank in range(2):
                for ep in (1, 2):
                    with self.subTest(rank=rank, ep=ep):
                        loader = self.loader(
                            a, rank, ep, method="auto" if ep == 2 else "scratch"
                        )
                        actual = loader.load().state_dict()
                        self.assertEqual(loader._selected_load_method.value, "scratch")
                        expected = (
                            self.loader(b, rank, ep, online=False).load().state_dict()
                        )
                        self.assertEqual(actual.keys(), expected.keys())
                        for name in actual:
                            self.assertEqual(actual[name].dtype, expected[name].dtype)
                            torch.testing.assert_close(
                                actual[name], expected[name], rtol=0, atol=0
                            )

    def test_online_recipe_requires_owner_capability_not_model_name(self):
        class NoOnlineRecipe(_Fixture):
            supports_load_time_w8a8 = False

        with tempfile.TemporaryDirectory() as root, patch.dict(
            MODEL_REGISTRY, {"_another_w8a8_owner": _Fixture, "qwen35_moe": NoOnlineRecipe}
        ):
            save_file(self.values(), os.path.join(root, "model.safetensors"))
            other = self.loader(root)
            other.model_config.model_type = "_another_w8a8_owner"
            self.assertTrue(any(t.dtype == torch.int8 for t in other.load().state_dict().values()))
            with self.assertRaisesRegex(ValueError, "does not support load-time W8A8"):
                self.loader(root).load()

    def test_incompatible_sources_fail_before_model_allocation(self):
        key = "layers.0.self_attn.q_proj.weight"
        options = (
            ({key: torch.zeros(3, 4, dtype=torch.int8)}, "floating"),
            (
                {key: torch.ones(3, 4), scale_name_for(key): torch.ones(3, 1)},
                "already contains",
            ),
            ({key: torch.ones(3)}, "shape"),
            (
                {"layers.0.mlp.experts.0.down_proj.weight": torch.ones(3, 4)},
                "split experts",
            ),
            ({"layers.0.norm.weight": torch.ones(4)}, "no weights"),
        )
        for values, msg in options:
            with self.subTest(msg=msg), tempfile.TemporaryDirectory() as folder:
                save_file(values, os.path.join(folder, "model.safetensors"))
                loader = self.loader(folder)
                with patch.object(
                    loader,
                    "_create_checkpoint_model",
                    side_effect=AssertionError("too late"),
                ):
                    with self.assertRaisesRegex(ValueError, msg):
                        loader.load()

    def test_forced_fst_and_lora_fail_before_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            save_file(self.values(), os.path.join(folder, "model.safetensors"))
            for method, lora, msg in (
                ("fastsafetensors", None, "scratch or auto"),
                ("auto", ("lora", "adapter"), "LoRA"),
            ):
                loader = self.loader(folder, method=method)
                with self.assertRaisesRegex(ValueError, msg):
                    loader.load(startup_lora=lora)

    def test_ignore_and_scale_in_another_shard(self):
        from safetensors import safe_open

        key = "model.language_model.layers.0.linear_attn.out_proj.weight"
        ignored = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
        config = CompressedW8A8Int8PerChannelQuantConfig(
            is_quanted=False, ignore_patterns=["layers.{i}.mlp.shared_expert"]
        )
        runtime = QuantizationConfig(config.get_method(), source_config=config)
        with tempfile.TemporaryDirectory() as folder:
            files = [os.path.join(folder, name + ".safetensors") for name in ("a", "b")]
            save_file(
                {key: torch.ones(3, 4), ignored: torch.full((4, 3), 7.0)}, files[0]
            )
            view = W8A8LoadSource.from_files(files[:1], runtime)
            with safe_open(files[0], framework="pt", device="cpu") as handle:
                shard = view.shard(handle, None)
                self.assertNotIn(scale_name_for(ignored), list(shard.keys()))
                torch.testing.assert_close(
                    shard.get_tensor(ignored), torch.full((4, 3), 7.0)
                )
            save_file({scale_name_for(key): torch.ones(3, 1)}, files[1])
            with self.assertRaisesRegex(ValueError, "already contains"):
                W8A8LoadSource.from_files(files, runtime)

    def test_recipe_finetune_overlay_cannot_silently_use_the_base_scale(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as ptune,
        ):
            raw = self.values()
            save_file(raw, os.path.join(directory, "model.safetensors"))
            torch.save(
                {"layers.0.self_attn.o_proj.weight": torch.ones(8, 8)},
                os.path.join(ptune, "pytorch_model.bin"),
            )
            loader = self.loader(directory)
            loader.model_config.ptuning_path = ptune
            with self.assertRaisesRegex(ValueError, "finetune overlay"):
                loader.load()

    def test_full_k_bounded_reads_and_nonfinite(self):
        # Record source reads: even when the consumer asks for K[0:2], every
        # read must span full K, but never more than two output rows per read.
        from rtp_llm.models_py.quant_methods.w8a8_load_source import _Rows

        class Source:
            def __init__(self, x):
                self.x, self.cuts = x, []

            def get_shape(self):
                return self.x.shape

            def __getitem__(self, cuts):
                self.cuts.append(cuts)
                return self.x[cuts]

        x = torch.tensor([[1.0, 2.0, 3.0, 100.0]]).repeat(7, 1)
        source = Source(x)
        view = _Rows(source, "test", 2)
        q, scale = quantize_weight_per_output_channel(x)
        torch.testing.assert_close(
            view.read((slice(None), slice(0, 2)), False), q[:, :2]
        )
        torch.testing.assert_close(view.read(slice(None), True), scale)
        self.assertEqual(len(source.cuts), 4)
        self.assertTrue(
            all(r.stop - r.start <= 2 and k == slice(None) for r, k in source.cuts)
        )
        x[2, 3] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _Rows(source, "test", 2).read(slice(None), False)


if __name__ == "__main__":
    unittest.main()
