import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.fp4_read_spec import (
    modelopt_fp4_qkv_read_spec,
    modelopt_fp4_read_spec,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _Fp4CheckpointModel(RtpModule):
    """CPU loading fixture, deliberately without a quantized forward executor."""

    def __init__(self, model_config, load_config):
        super().__init__()
        self.dim = model_config.partition_dim
        self.tp_size = load_config.tp_size
        self.tp_rank = load_config.tp_rank
        n = 8 // self.tp_size if self.dim == 0 else 8
        k = 64 // self.tp_size if self.dim == 1 else 64
        for name, shape, dtype in (
            ("weight", (n, k // 2), torch.uint8),
            ("weight_scale", (n, k // 16), torch.float8_e4m3fn),
            ("weight_scale_2", (), torch.float32),
            ("input_scale", (1,), torch.float32),
        ):
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                ),
            )

    def checkpoint_read_spec(self, name, shape):
        return modelopt_fp4_read_spec(
            name,
            shape,
            input_size=64,
            output_size=8,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            partition_dim=self.dim,
        )


class ModelOptFp4ReadSpecTest(unittest.TestCase):
    def test_newloader_scratch_populates_rank_local_parameters(self):
        weights = {
            "weight": torch.arange(256).to(torch.uint8).reshape(8, 32),
            "weight_scale": torch.ones(8, 4).to(torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor(0.25),
            "input_scale": torch.tensor([0.5]),
        }
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            MODEL_REGISTRY, {"fp4_checkpoint_test": _Fp4CheckpointModel}
        ):
            save_file(weights, os.path.join(folder, "model.safetensors"))
            for dim in (0, 1):
                for rank in range(2):
                    with self.subTest(dim=dim, rank=rank):
                        loader = NewModelLoader(
                            types.SimpleNamespace(
                                model_type="fp4_checkpoint_test", partition_dim=dim
                            ),
                            NewLoaderConfig(
                                device="cpu",
                                load_method="scratch",
                                tp_size=2,
                                tp_rank=rank,
                            ),
                            model_path=folder,
                        )
                        model = loader.load()
                        for name, source in weights.items():
                            actual = dict(model.named_parameters())[name]
                            expected = source
                            if source.ndim == 2:
                                width = source.shape[dim] // 2
                                expected = source.narrow(
                                    dim, rank * width, width
                                ).contiguous()
                            self.assertEqual(actual.dtype, source.dtype)
                            torch.testing.assert_close(
                                actual.detach().reshape(-1).view(torch.uint8),
                                expected.reshape(-1).view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )

    def plan(self, name, shape, **overrides):
        config = dict(
            input_size=64, output_size=8, tp_size=2, tp_rank=0, partition_dim=0
        )
        config.update(overrides)
        return modelopt_fp4_read_spec(name, shape, **config)

    def test_real_safetensors_tp_shards_preserve_packed_bytes_and_scales(self):
        prefix = "model.layers.0.mlp.down_proj."
        weights = {
            prefix + "weight": torch.arange(256).to(torch.uint8).reshape(8, 32),
            prefix
            + "weight_scale": (torch.arange(32).reshape(8, 4) / 8).to(
                torch.float8_e4m3fn
            ),
            prefix + "weight_scale_2": torch.tensor(0.25),
            prefix + "input_scale": torch.tensor([0.5]),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.safetensors")
            save_file(weights, path)
            for dim in (0, 1):
                with self.subTest(partition_dim=dim), safe_open(
                    path, framework="pt", device="cpu"
                ) as reader:
                    for name, source in weights.items():
                        pieces = []
                        for rank in range(2):
                            spec = self.plan(
                                name,
                                tuple(source.shape),
                                partition_dim=dim,
                                tp_rank=rank,
                            )
                            checkpoint = (
                                reader.get_slice(name)
                                if spec.is_local_shard
                                else reader.get_tensor(name)
                            )
                            actual, applied_spec = unwrap_checkpoint_tensor(
                                spec.materialize(checkpoint)
                            )
                            if source.ndim == 2:
                                # Independently reproduce legacy sp_0 / sp_neg1 on
                                # physical checkpoint shapes, not logical FP4 sizes.
                                width = source.shape[dim] // 2
                                expected = source.narrow(
                                    dim, rank * width, width
                                ).contiguous()
                                self.assertIs(applied_spec, spec)
                                self.assertEqual(actual.dtype, source.dtype)
                                torch.testing.assert_close(
                                    actual.view(torch.uint8),
                                    expected.view(torch.uint8),
                                    rtol=0,
                                    atol=0,
                                )
                                pieces.append(actual.view(torch.uint8))
                            else:
                                self.assertIsNone(applied_spec)
                                torch.testing.assert_close(
                                    actual, source, rtol=0, atol=0
                                )
                        if source.ndim == 2:
                            torch.testing.assert_close(
                                torch.cat(pieces, dim=dim),
                                source.view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )

    def test_tp1_retains_full_read(self):
        for name, shape in (
            ("weight", (8, 32)),
            ("weight_scale", (8, 4)),
            ("weight_scale_2", ()),
            ("input_scale", (1,)),
        ):
            with self.subTest(name=name):
                self.assertFalse(self.plan(name, shape, tp_size=1).is_local_shard)

    def test_row_parallel_rejects_split_inside_scale_block(self):
        with self.assertRaisesRegex(ValueError, "scale blocks"):
            self.plan("weight", (8, 24), input_size=48, partition_dim=1)

    def test_checkpoint_layout_is_not_silently_transposed_or_reinterpreted(self):
        for name, shape in (
            ("weight", (32, 8)),
            ("weight", (8, 64)),
            ("weight_scale", (8, 32)),
        ):
            with self.subTest(name=name, shape=shape), self.assertRaisesRegex(
                ValueError, "expected checkpoint shape"
            ):
                self.plan(name, shape)

    def test_projection_scales_are_scalar_not_fused_projection_vectors(self):
        for name in ("weight_scale_2", "input_scale"):
            for shape in ((2,), (0,)):
                with self.subTest(name=name, shape=shape), self.assertRaisesRegex(
                    ValueError, "must be scalar"
                ):
                    self.plan(name, shape)

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.plan("weight_scale_inv", (8, 4))

    def test_invalid_topology_is_rejected_even_for_replicated_scales(self):
        for overrides in (
            dict(tp_size=0),
            dict(tp_size=True),
            dict(tp_rank=-1),
            dict(tp_rank=2),
            dict(tp_rank=True),
            dict(partition_dim=2),
            dict(partition_dim=True),
            dict(input_size=63),
            dict(output_size=7),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.plan("input_scale", (), **overrides)


class _Fp4QKVCheckpointModel(RtpModule):
    """Real loader fixture for packed QKV storage, without a forward backend."""

    def __init__(self, model_config, load_config):
        super().__init__()
        self.config = model_config
        self.tp_size, self.tp_rank = load_config.tp_size, load_config.tp_rank
        for projection in ("q_proj", "k_proj", "v_proj"):
            heads = (
                model_config.num_heads
                if projection == "q_proj"
                else model_config.num_kv_heads
            )
            local_heads = heads // self.tp_size if heads >= self.tp_size else 1
            rows = local_heads * model_config.head_dim
            child = RtpModule()
            for name, shape, dtype in (
                ("weight", (rows, 32), torch.uint8),
                ("weight_scale", (rows, 4), torch.float8_e4m3fn),
                ("weight_scale_2", (), torch.float32),
                ("input_scale", (1,), torch.float32),
            ):
                child.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.empty(shape, dtype=dtype), requires_grad=False
                    ),
                )
            self.add_module(projection, child)

    def resolve_checkpoint_load_plan(self, name, shape):
        target, _ = super().resolve_checkpoint_load_plan(name, shape)
        spec = modelopt_fp4_qkv_read_spec(
            name,
            shape,
            hidden_size=64,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        return target, spec


class ModelOptFp4QKVReadSpecTest(unittest.TestCase):
    def test_real_checkpoint_replicates_kv_by_contiguous_tp_rank_groups(self):
        for kv_heads, tp in ((1, 4), (2, 4), (4, 2), (8, 1)):
            config = types.SimpleNamespace(
                model_type="fp4_qkv_fixture",
                num_heads=8,
                num_kv_heads=kv_heads,
                head_dim=16,
            )
            weights = {}
            for i, projection in enumerate(("q_proj", "k_proj", "v_proj")):
                rows = (8 if i == 0 else kv_heads) * 16
                weights[projection + ".weight"] = (
                    (torch.arange(rows * 32) + 13 * i).to(torch.uint8).reshape(rows, 32)
                )
                weights[projection + ".weight_scale"] = (
                    ((torch.arange(rows * 4) % 16 + i) / 4)
                    .reshape(rows, 4)
                    .to(torch.float8_e4m3fn)
                )
                weights[projection + ".weight_scale_2"] = torch.tensor(0.25 + i)
                weights[projection + ".input_scale"] = torch.tensor([0.5 + i])
            with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
                MODEL_REGISTRY, {"fp4_qkv_fixture": _Fp4QKVCheckpointModel}
            ):
                save_file(weights, os.path.join(folder, "model.safetensors"))
                for rank in range(tp):
                    with self.subTest(kv_heads=kv_heads, tp=tp, rank=rank):
                        model = NewModelLoader(
                            config,
                            NewLoaderConfig(
                                device="cpu",
                                load_method="scratch",
                                tp_size=tp,
                                tp_rank=rank,
                            ),
                            model_path=folder,
                        ).load()
                        for name, actual in model.named_parameters():
                            source = weights[name]
                            expected = source
                            if source.ndim == 2:
                                # Head-based reference, independent of ReadSpec.
                                heads = 8 if name.startswith("q_proj.") else kv_heads
                                if heads >= tp:
                                    start, count = rank * (heads // tp), heads // tp
                                else:
                                    start, count = rank // (tp // heads), 1
                                expected = source[start * 16 : (start + count) * 16]
                            self.assertEqual(actual.dtype, source.dtype)
                            self.assertEqual(actual.shape, expected.shape)
                            torch.testing.assert_close(
                                actual.detach()
                                .contiguous()
                                .reshape(-1)
                                .view(torch.uint8),
                                expected.contiguous().reshape(-1).view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )

    def test_invalid_topology_is_checked_for_scalar_scales(self):
        defaults = dict(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=16,
            tp_size=4,
            tp_rank=0,
        )
        for overrides in (
            dict(num_heads=7),
            dict(num_kv_heads=3),
            dict(tp_rank=4),
            dict(tp_size=True),
            dict(hidden_size=63),
            dict(head_dim=0),
            dict(num_heads=12, num_kv_heads=4, tp_size=6),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                modelopt_fp4_qkv_read_spec(
                    "k_proj.input_scale", (), **{**defaults, **overrides}
                )

    def test_fused_names_and_wrong_packed_shapes_are_not_silently_accepted(self):
        defaults = dict(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=16,
            tp_size=4,
            tp_rank=0,
        )
        for name, shape in (
            ("qkv_proj.weight", (192, 32)),
            ("k_proj.weight", (32, 64)),
            ("k_proj.weight_scale", (32, 32)),
            ("k_proj.input_scale", (3,)),
        ):
            with self.subTest(name=name, shape=shape), self.assertRaises(ValueError):
                modelopt_fp4_qkv_read_spec(name, shape, **defaults)


if __name__ == "__main__":
    unittest.main()
