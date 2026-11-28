"""Real scratch/FST checkpoint ingestion, not CPU admission to GPU executors.

No backend or reader is mocked. These tests deliberately stop at parameter
validation; load(cpu) must still reject an accelerator-only execution recipe.
"""

import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers, requires_native_fst
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_loader_test import fp4_linear_test as dense_fixture
from rtp_llm.models_py.new_loader_test.fp4_moe_test import _experts, _weights
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.test.test_mxfp4_quark import (
    PREFIX,
    checkpoint_tensors,
    make_experts,
)
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _Root(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        kwargs = dict(tp_size=load_config.tp_size, tp_rank=load_config.tp_rank)
        if config.recipe == "dense":
            self.projection = dense_fixture._DenseCheckpointModel(
                types.SimpleNamespace(kind=config.kind), load_config
            ).projection
        elif config.recipe == "quark":
            self.experts = make_experts(**kwargs)
        else:
            self.experts = _experts(online=config.recipe == "online", **kwargs)


def source_weights(recipe, kind="qkv"):
    if recipe == "dense":
        return dense_fixture.Fp4LinearTest().checkpoint(
            kind, fused=kind in ("qkv", "merged")
        )
    if recipe == "quark":
        return {
            name.replace(PREFIX, "experts."): value
            for name, value in checkpoint_tensors().items()
        }
    return _weights(online=recipe == "online")


def make_config(recipe, kind="qkv", tp=1, rank=0, device="cpu", method="scratch"):
    model_config = types.SimpleNamespace(
        model_type="fp4_native_fixture", recipe=recipe, kind=kind
    )
    config = NewLoaderConfig(
        device=device,
        compute_dtype=torch.bfloat16,
        load_method=method,
        tp_size=tp,
        tp_rank=rank,
        quant_config=QuantizationConfig("modelopt_fp4"),
    )
    return model_config, config


def ingest(directory, recipe, *, kind="qkv", tp=1, rank=0, method="scratch"):
    model_config, config = make_config(recipe, kind, tp, rank)
    model = _Root(model_config, config)
    loader = NewModelLoader(model_config, config, model_path=directory)
    files = [os.path.join(directory, "model.safetensors")]
    resolver = loader._checkpoint_read_spec_resolver(model, None, None)
    if method == "fastsafetensors":
        iterator = loader._fastsafetensors_weights_iterator(
            files, "cpu", None, None, resolver
        )
    else:
        iterator = weight_mapper.get_all_weights(files, read_spec_resolver=resolver)
    # RtpModule consumes every borrowed FST tensor synchronously before next().
    model.load_weights(iterator)
    loader._validate_loaded_weights(model)
    return model


class NativeFp4CheckpointTest(unittest.TestCase):
    def test_modelopt_qkv_cache_sidecars_are_not_gemm_scales(self):
        weights = dense_fixture.Fp4LinearTest().checkpoint("qkv")
        sidecars = {
            "projection.k_proj.k_scale": torch.tensor(99.0),
            "projection.v_proj.v_scale": torch.tensor([77.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {**weights, **sidecars}, os.path.join(directory, "model.safetensors")
            )
            for tp in (1, 2):
                for rank in range(tp):
                    for method in checkpoint_readers("scratch", "fastsafetensors"):
                        with self.subTest(tp=tp, rank=rank, method=method):
                            layer = ingest(
                                directory, "dense", tp=tp, rank=rank, method=method
                            ).projection
                            self.assertEqual(layer.weight_scale_2.item(), 2.5)
                            self.assertEqual(layer.input_scale.item(), 3.25)

        model_config, config = make_config("dense")
        layer = _Root(model_config, config).projection
        layer.load_weights(
            {key.removeprefix("projection."): value for key, value in sidecars.items()}
        )
        self.assertFalse(layer._loaded_parameter_names)
        for key, tensor in sidecars.items():
            spec = layer.checkpoint_read_spec(
                key.removeprefix("projection."), tuple(tensor.shape)
            )
            self.assertFalse(spec.load)
        with self.assertRaisesRegex(ValueError, "must be scalar"):
            layer.checkpoint_read_spec("k_proj.k_scale", (2,))
        with self.assertRaisesRegex(ValueError, "Unknown NVFP4"):
            layer.checkpoint_read_spec("q_proj.k_scale", ())
        model_config, config = make_config("dense", kind="column")
        with self.assertRaisesRegex(ValueError, "Unknown NVFP4"):
            _Root(model_config, config).projection.checkpoint_read_spec(
                "k_proj.k_scale", ()
            )

    @requires_native_fst
    def test_real_scratch_and_native_fst_storage_agree(self):
        for recipe in ("dense", "offline", "online", "quark"):
            kinds = (
                ("qkv", "merged", "row", "column") if recipe == "dense" else ("qkv",)
            )
            for kind in kinds:
                for tp in (1, 2):
                    for rank in range(tp):
                        with self.subTest(
                            recipe=recipe, kind=kind, tp=tp, rank=rank
                        ), tempfile.TemporaryDirectory() as directory:
                            save_file(
                                source_weights(recipe, kind),
                                os.path.join(directory, "model.safetensors"),
                            )
                            scratch = ingest(
                                directory, recipe, kind=kind, tp=tp, rank=rank
                            )
                            fst = ingest(
                                directory,
                                recipe,
                                kind=kind,
                                tp=tp,
                                rank=rank,
                                method="fastsafetensors",
                            )
                            expected, actual = scratch.state_dict(), fst.state_dict()
                            self.assertEqual(set(expected), set(actual))
                            for name, value in expected.items():
                                self.assertEqual(value.dtype, actual[name].dtype)
                                torch.testing.assert_close(
                                    value.contiguous().reshape(-1).view(torch.uint8),
                                    actual[name]
                                    .contiguous()
                                    .reshape(-1)
                                    .view(torch.uint8),
                                )
                            if recipe != "dense":
                                self.assertFalse(fst.experts._post_load_done)
                                self.assertIsNone(fst.experts.fused_moe)

    def test_missing_scale_is_rejected_by_both_real_readers(self):
        for recipe in ("dense", "offline", "quark"):
            weights = source_weights(recipe)
            del weights[next(name for name in weights if name.endswith("weight_scale"))]
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(
                    recipe=recipe, method=method
                ), tempfile.TemporaryDirectory() as directory:
                    save_file(weights, os.path.join(directory, "model.safetensors"))
                    with self.assertRaises((ValueError, RuntimeError)):
                        ingest(directory, recipe, method=method)

    def test_invalid_bytes_are_rejected_by_both_real_readers(self):
        for recipe in ("dense", "offline", "quark"):
            weights = source_weights(recipe)
            key = next(name for name in weights if name.endswith("weight"))
            weights[key] = weights[key].float()
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(
                    recipe=recipe, method=method
                ), tempfile.TemporaryDirectory() as directory:
                    save_file(weights, os.path.join(directory, "model.safetensors"))
                    with self.assertRaises((TypeError, ValueError)):
                        ingest(directory, recipe, method=method)

    def test_full_cpu_load_does_not_claim_a_gpu_backend(self):
        for recipe in ("dense", "offline", "online", "quark"):
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(
                    recipe=recipe, method=method
                ), tempfile.TemporaryDirectory() as directory:
                    save_file(
                        source_weights(recipe),
                        os.path.join(directory, "model.safetensors"),
                    )
                    model_config, config = make_config(recipe, method=method)
                    with mock.patch.dict(MODEL_REGISTRY, {"fp4_native_fixture": _Root}):
                        loader = NewModelLoader(
                            model_config, config, model_path=directory
                        )
                        with self.assertRaisesRegex(
                            (RuntimeError, ValueError), "CUDA|NVIDIA|gfx950"
                        ):
                            loader.load()
                        self.assertIsNone(loader.loaded_method)


if __name__ == "__main__":
    unittest.main()
