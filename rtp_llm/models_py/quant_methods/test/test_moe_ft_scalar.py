"""Legacy per-channel/per-tensor FT into NewModelLoader owners on CPU.

Staging is not serving: real public FP8 runtime preflight rejects CPU.
"""

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8DynamicPerTensorQuantConfig,
    Fp8PerTensorCompressedQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.ft_checkpoint import FTCheckpoint, CheckpointTopology
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.block_fp8_ft import block_moe_ft_bindings
from rtp_llm.models_py.quant_methods.channel_tensor_moe_ft import (
    nonblock_moe_ft_bindings,
)
from rtp_llm.models_py.quant_methods.moe_quant_key import moe_checkpoint_method_key
from rtp_llm.models_py.quant_methods.test.test_block_fp8_ft import write_ft
from rtp_llm.models_py.registry import register_model


@register_model("_test_scalar_ft_owner")
class ExpertOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=2,
            top_k=1,
            hidden_size=128,
            moe_intermediate_size=128,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.bfloat16,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1, dp_rank=0, world_size=1, local_world_size=1
            ),
            moe_config=None,
            quant_config=load.quant_config,
            layer_idx=0,
        )

    def load_ft_checkpoint(self, cp):
        fn = (
            block_moe_ft_bindings
            if self.experts._quant_family == "fp8_per_block"
            else nonblock_moe_ft_bindings
        )
        cp.load_parameters(self, fn(self.experts, "layers.0.", "experts", cp, 0))


class LegacyScalarFTTest(unittest.TestCase):
    def test_float_expert_excluded_from_online_quantizer(self):
        from dataclasses import replace

        for abi in ("cuda", "rocm"):
            ft = {}
            expected = {}
            for name, stem, rows in (
                ("w13", "intermediate_weight", 256),
                ("w2", "intermediate_weight2", 128),
            ):
                v = (torch.arange(2 * rows * 128).reshape(2, rows, 128) % 97).bfloat16()
                expected[name] = v
                if abi == "rocm":
                    if name == "w13":
                        v = torch.cat(v.chunk(2, dim=1)[::-1], dim=1)
                    v = (
                        v.reshape(2, rows // 16, 16, 128 // 32, 4, 8)
                        .permute(0, 1, 3, 4, 2, 5)
                        .contiguous()
                        .reshape(2, rows, 128)
                    )
                ft["layers.0.partial_moe_weights." + stem + ".kernel"] = v
            with tempfile.TemporaryDirectory() as root:
                write_ft(root, ft)
                source = Fp8BlockWiseQuantConfig(is_quanted=False)
                cfg, lc = self.make(root, source)
                q = QuantizationConfig(
                    "fp8_block_online", source, ignored_layers=["layers.0.mlp.experts"]
                )
                loader = NewModelLoader(
                    cfg, replace(lc, quant_config=q), model_path=root
                )
                for method in ("scratch", "fastsafetensors"):
                    cp = FTCheckpoint.open(root, CheckpointTopology())
                    cp.index_metadata["ft_export_backend"] = abi
                    cp.read_backend = method
                    model = loader._create_checkpoint_model(cp)
                    self.assertEqual(model.experts._quant_family, "none")
                    model.load_ft_checkpoint(cp)
                    loader._validate_loaded_weights(model)
                    for name, v in expected.items():
                        torch.testing.assert_close(
                            getattr(model.experts, name), v, rtol=0, atol=0
                        )

    def test_rocm_dense_sources_for_scalar_and_channel(self):
        from rtp_llm.models_py.quant_methods.fp8_ft import (
            PerChannelFP8FTProjection,
            PerTensorFP8FTProjection,
        )
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        n, k = 128, 256
        w = (torch.arange(n * k).reshape(n, k) % 53 - 26).to(torch.float8_e4m3fn)
        for swizzled in (False, True):
            encoded = (
                swizzle_tensor(w, False)
                if swizzled
                else w.reshape(n // 16, 16, k // 32, 2, 16)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
                .reshape(n, k)
            )
            for channel in (False, True):
                with tempfile.TemporaryDirectory() as root:
                    keys = "layers.0.k", "layers.0.s"
                    s = (
                        torch.arange(n).float() / 128 + 0.5
                        if channel
                        else torch.tensor([2.0])
                    )
                    write_ft(
                        root,
                        {
                            keys[0]: encoded.reshape(k, n) if channel else encoded,
                            keys[1]: s.reshape(1, n) if channel else s,
                        },
                    )
                    checkpoint = FTCheckpoint.open(root, CheckpointTopology())
                    checkpoint.index_metadata["ft_export_backend"] = "rocm"
                    for method in ("scratch", "fastsafetensors"):
                        checkpoint.read_backend = method
                        cls = (
                            PerChannelFP8FTProjection
                            if channel
                            else PerTensorFP8FTProjection
                        )
                        obj = cls(
                            SimpleNamespace(weight=w, tp_size=1, tp_rank=0),
                            checkpoint,
                            *keys,
                            "out",
                            swizzled=swizzled,
                        )
                        torch.testing.assert_close(
                            obj.weight().view(torch.uint8), w.view(torch.uint8)
                        )
                        torch.testing.assert_close(obj.scale().flatten(), s)

    def test_ep_rank_aggregate_tracks_selected_physical_rows(self):
        from safetensors.torch import save_file
        from rtp_llm.models_py.expert_placement import ExpertPlacement

        with tempfile.TemporaryDirectory() as root:
            weights = {}
            for i in (0, 1):
                base = f"rank_{i}_0_{i}.layers.0."
                data = {
                    base + "w": torch.arange(i * 2, i * 2 + 2).reshape(2, 1).float(),
                    base + "scalar": torch.tensor(2.0**i),
                }
                file = f"rank{i}.safetensors"
                save_file(data, str(Path(root) / file))
                weights.update(dict.fromkeys(data, file))
            (Path(root) / "model.safetensors.index.json").write_text(
                json.dumps(
                    dict(
                        is_ft_style_weight=True,
                        __env__params__={"TP_SIZE": "2", "EP_SIZE": "2"},
                        weight_map=weights,
                    )
                )
            )
            cp = FTCheckpoint.open(root, CheckpointTopology(tp_size=2, ep_size=2))
            placement = ExpertPlacement(4, 2, 0, (3, 1, 3, 2, 0, 1))
            for method in ("scratch", "fastsafetensors"):
                cp.read_backend = method
                torch.testing.assert_close(
                    cp.read_expert_rank_scalars(
                        "layers.0.scalar", "layers.0.w", 0, placement
                    ),
                    torch.tensor([2.0, 1.0, 2.0]),
                )

    def stage(self, loader, root, method):
        cp = FTCheckpoint.open(root, CheckpointTopology())
        cp.read_backend = method
        model = loader._create_checkpoint_model(cp)
        model.load_ft_checkpoint(cp)
        loader._validate_loaded_weights(model)
        return model.experts

    def make(self, root, source):
        config = SimpleNamespace(
            ckpt_path=root,
            model_type="_test_scalar_ft_owner",
            num_layers=1,
            expert_num=2,
            activation_type="SiGLU",
            quant_config=source,
        )
        quant = QuantizationConfig(moe_checkpoint_method_key(source), source)
        base = NewLoaderConfig(
            device="cpu", compute_dtype=torch.bfloat16, quant_config=quant
        )
        return config, base

    def fixture(self, family, static):
        ft = {}
        for i, (stem, w, rows) in enumerate(
            (("intermediate_weight", "w13", 256), ("intermediate_weight2", "w2", 128))
        ):
            pref = "layers.0.partial_moe_weights." + stem
            data = (torch.arange(2 * rows * 128).reshape(2, rows, 128) % 25 - 12).to(
                torch.float8_e4m3fn
            )
            ft[pref + ".kernel"] = data
            if family == "fp8_per_channel":
                s = torch.arange(2 * rows).reshape(2, rows, 1) / 128 + 0.25
            elif family == "fp8_per_block":
                s = (
                    torch.arange(2 * (rows // 128)).reshape(2, rows // 128, 1).float()
                    + 0.5
                )
            else:
                s = torch.tensor([0.25, 2.0]) * (i + 1)
                if w == "w2":
                    s = s.reshape(-1, 1, 1)
            ft[pref + ".weight_only_quant_scale"] = s
            if static:
                p = "layers.0." + (
                    "moe_w1_activation." if w == "w13" else "moe_w2_activation."
                )
                ft[p + "static_quant_reciprocal"] = torch.tensor(2.0 ** (i + 1))
                ft[p + "static_quant"] = torch.tensor(2.0 ** (-i - 1))
        return ft

    def test_legacy_files_and_online_origin(self):
        from dataclasses import replace

        for source, family, static in (
            (
                Fp8PerChannelCompressedQuantConfig(is_quanted=True),
                "fp8_per_channel",
                False,
            ),
            (
                Fp8PerChannelCompressedQuantConfig(is_quanted=False),
                "fp8_per_channel",
                False,
            ),
            (
                Fp8PerTensorCompressedQuantConfig(is_quanted=True, dynamic=False),
                "fp8_per_tensor",
                True,
            ),
            (
                Fp8PerTensorCompressedQuantConfig(is_quanted=True, dynamic=True),
                "fp8_per_tensor",
                False,
            ),
            (Fp8DynamicPerTensorQuantConfig(is_quanted=False), "fp8_per_tensor", False),
            (Fp8BlockWiseQuantConfig(is_quanted=False), "fp8_per_block", False),
        ):
            ft = self.fixture(family, static)
            with tempfile.TemporaryDirectory() as root:
                write_ft(root, ft)
                config, base = self.make(root, source)
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(
                        family=family,
                        online=not source.is_quanted(),
                        static=static,
                        method=method,
                    ):
                        loader = NewModelLoader(
                            config, replace(base, load_method=method), model_path=root
                        )
                        owner = self.stage(loader, root, method)
                        self.assertEqual(owner._quant_family, family)
                        owner.validate_weights_loaded()
                        for w, stem in (
                            ("w13", "intermediate_weight"),
                            ("w2", "intermediate_weight2"),
                        ):
                            p = "layers.0.partial_moe_weights." + stem
                            torch.testing.assert_close(
                                getattr(owner, w).view(torch.uint8),
                                ft[p + ".kernel"].view(torch.uint8),
                            )
                            expected = ft[p + ".weight_only_quant_scale"].reshape_as(
                                getattr(owner, w + "_scale")
                            )
                            torch.testing.assert_close(
                                getattr(owner, w + "_scale"), expected
                            )
                        if family == "fp8_per_channel":
                            torch.testing.assert_close(
                                owner._gate_ch_scales, owner.w13_scale[:, 128:]
                            )
                            torch.testing.assert_close(
                                owner._up_ch_scales, owner.w13_scale[:, :128]
                            )
                        if static:
                            torch.testing.assert_close(
                                owner.checkpoint_input_scales,
                                torch.tensor([[2.0, 2.0, 4.0]] * 2),
                            )
                            self.assertEqual(owner.w2_input_scale_inv.item(), 0.25)
                        self.assertEqual(
                            base.quant_config.quant_type,
                            moe_checkpoint_method_key(source),
                        )

    def test_inconsistent_aggregation_rejected(self):
        source = Fp8PerTensorCompressedQuantConfig(is_quanted=True, dynamic=False)
        with tempfile.TemporaryDirectory() as root:
            ft = self.fixture("fp8_per_tensor", True)
            ft["layers.0.moe_w1_activation.static_quant"] = torch.tensor(0.125)
            write_ft(root, ft)
            cfg, load = self.make(root, source)
            with self.assertRaisesRegex(ValueError, "Inconsistent reciprocal"):
                self.stage(NewModelLoader(cfg, load, model_path=root), root, "scratch")


if __name__ == "__main__":
    unittest.main()
