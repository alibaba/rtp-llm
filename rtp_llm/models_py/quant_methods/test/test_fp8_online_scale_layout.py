"""CPU checkpoint contracts and SM90/ROCm device tests for online block FP8."""

import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.kernels.cuda import fp8_quant
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods import fp8
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def _owner(kind, tp, rank):
    root = RtpModule()
    kwargs = dict(
        tp_size=tp,
        tp_rank=rank,
        params_dtype=torch.bfloat16,
        quant_config=QuantizationConfig("fp8_block_online"),
        prefix="proj",
    )
    if kind == "qkv_partial":
        root.proj = QKVParallelLinear(
            hidden_size=1152, num_heads=6, num_kv_heads=2, head_dim=24, **kwargs
        )
    elif kind == "qkv":
        root.proj = QKVParallelLinear(
            hidden_size=1152, num_heads=4, num_kv_heads=2, head_dim=128, **kwargs
        )
    elif kind in ("merged", "merged_partial"):
        root.proj = MergedColumnParallelLinear(
            input_size=1152,
            output_size=384 if kind == "merged_partial" else 512,
            shard_names=["gate_proj", "up_proj"],
            **kwargs
        )
    elif kind == "row":
        root.proj = RowParallelLinear(input_size=1152, output_size=384, **kwargs)
    else:
        root.proj = ColumnParallelLinear(input_size=1152, output_size=384, **kwargs)
    return root


def _checkpoint(kind):
    rows = {
        "qkv": [("q_proj", 512), ("k_proj", 256), ("v_proj", 256)],
        "qkv_partial": [("q_proj", 144), ("k_proj", 48), ("v_proj", 48)],
        "merged": [("gate_proj", 256), ("up_proj", 256)],
        "merged_partial": [("gate_proj", 192), ("up_proj", 192)],
        "row": [("", 384)],
        "column": [("", 384)],
    }[kind]
    return {
        "proj."
        + (name + "." if name else "")
        + "weight": (
            torch.arange(n * 1152).reshape(n, 1152).float().sin() * (idx + 1)
        ).bfloat16()
        for idx, (name, n) in enumerate(rows)
    }


def _load(directory, kind, tp, rank, backend):
    root = _owner(kind, tp, rank)
    loader = NewModelLoader(
        SimpleNamespace(model_type="layout_fixture"),
        NewLoaderConfig(device="cpu", tp_size=tp, tp_rank=rank),
        model_path=directory,
    )
    paths = [os.path.join(directory, "model.safetensors")]
    resolver = loader._checkpoint_read_spec_resolver(root, None, None)
    iterator = (
        loader._fastsafetensors_weights_iterator(paths, "cpu", None, None, resolver)
        if backend == "fastsafetensors"
        else weight_mapper.get_all_weights(paths, read_spec_resolver=resolver)
    )
    root.load_weights(iterator)
    loader._validate_loaded_weights(root)
    return root.proj


class OnlineFp8ScaleLayoutTest(unittest.TestCase):
    def _check_partial_projection_tiles(self, device):
        for kind in ("merged_partial", "qkv_partial"):
            checkpoint = _checkpoint(kind)
            # Independent OldLoader reference: quantize the complete fused
            # source, not the loader's staged weight or its individual parts.
            source = torch.cat(list(checkpoint.values())).to(device)
            padded = torch.nn.functional.pad(source, (0, 0, 0, -len(source) % 128))
            expected, scale = fp8_quant.per_block_cast_to_fp8(
                padded.float(), use_ue8m0=False
            )
            if device == "cuda" and torch.version.hip:
                expected, scale = fp8._prepare_rocm_fp8_block_weight(expected, scale)
            with tempfile.TemporaryDirectory() as directory:
                save_file(checkpoint, os.path.join(directory, "model.safetensors"))
                for reader in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(kind=kind, reader=reader, device=device):
                        layer = _load(directory, kind, 1, 0, reader).to(device)
                        # CPU checks the real float-scale quantizer; only
                        # backend discovery is substituted on a CPU worker.
                        backend = (
                            mock.patch.object(
                                fp8,
                                "_select_fp8_runtime_backend",
                                return_value=fp8._CUDA_DEEP_GEMM,
                            )
                            if device == "cpu"
                            else nullcontext()
                        )
                        with backend:
                            layer.process_weights_after_loading()
                        self.assertEqual(layer.weight.shape, padded.shape)
                        self.assertEqual(layer._fp8_logical_output_size, len(source))
                        torch.testing.assert_close(
                            layer.weight.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            layer.weight_scale, scale, rtol=0, atol=0
                        )
                        if device == "cuda":
                            # A one-hot activation compares the real GEMM with
                            # the unquantized projection, including tail rows.
                            x = source.new_zeros(2, 3, source.shape[1])
                            x[..., 1] = 1
                            out = layer(x)
                            ref = torch.nn.functional.linear(x, source)
                            self.assertEqual(out.shape, ref.shape)
                            torch.testing.assert_close(out, ref, rtol=0.07, atol=0.002)

    def test_tp1_float_scale_tiles_cross_projection_boundaries(self):
        self._check_partial_projection_tiles("cpu")

    @unittest.skipUnless(os.getenv("TEST_FP8_FLOAT_SCALE_DEVICE"), "CPU contract suite")
    def test_tp1_float_scale_tiles_through_device_gemm(self):
        self.assertTrue(torch.cuda.is_available())
        if os.environ["TEST_FP8_FLOAT_SCALE_DEVICE"] == "sm90":
            self.assertIsNone(torch.version.hip)
            self.assertEqual(torch.cuda.get_device_capability()[0], 9)
        else:
            self.assertEqual(os.environ["TEST_FP8_FLOAT_SCALE_DEVICE"], "rocm")
            self.assertIsNotNone(torch.version.hip)
        self._check_partial_projection_tiles("cuda")

    def test_tp2_cannot_join_partial_projection_tiles(self):
        for kind in ("merged_partial", "qkv_partial"):
            for rank in (0, 1):
                for backend, ue8m0 in (
                    (fp8._CUDA_DEEP_GEMM, False),
                    (fp8._ROCM_AITER_BLOCK, False),
                    (fp8._CUDA_DEEP_GEMM, True),
                ):
                    with self.subTest(
                        kind=kind, rank=rank, backend=backend, ue8m0=ue8m0
                    ):
                        layer = _owner(kind, 2, rank).proj
                        with mock.patch.object(
                            fp8, "_select_fp8_runtime_backend", return_value=backend
                        ), mock.patch.object(
                            fp8, "is_deep_gemm_e8m0_used", return_value=ue8m0
                        ), self.assertRaisesRegex(ValueError, "align"):
                            layer.process_weights_after_loading()

    def test_checkpoint_packed_scales_retain_mn_major_storage_until_gemm(self):
        # K>512 is essential: a single packed column hides contiguous() damage.
        for kind in ("column", "row", "merged", "qkv"):
            with tempfile.TemporaryDirectory() as directory:
                save_file(
                    _checkpoint(kind), os.path.join(directory, "model.safetensors")
                )
                for backend in checkpoint_readers("scratch", "fastsafetensors"):
                    for tp in (1, 2):
                        for rank in range(tp):
                            with self.subTest(
                                kind=kind, backend=backend, tp=tp, rank=rank
                            ):
                                layer = _load(directory, kind, tp, rank, backend)
                                observed = {}

                                def gpu_requant_contract(weight):
                                    weight, _ = fp8_quant.per_block_cast_to_fp8(
                                        weight, use_ue8m0=True
                                    )
                                    n, k = weight.shape
                                    columns = (k + 511) // 512
                                    # Include a padded leading dimension, as
                                    # returned by a TMA-aligned backend layout.
                                    storage = torch.arange(
                                        columns * (n + 4), dtype=torch.int32
                                    ).reshape(columns, n + 4)
                                    packed = storage.T[:n]
                                    observed.update(
                                        weight=weight,
                                        scale=packed,
                                        values=packed.clone(),
                                    )
                                    return weight, packed

                                def activation_quant_contract(x, **kwargs):
                                    self.assertTrue(kwargs["scale_ue8m0"])
                                    return x.to(torch.float8_e4m3fn), torch.ones(
                                        x.shape[0], 1, dtype=torch.int32
                                    )

                                def gpu_gemm_contract(a, b, out, **kwargs):
                                    self.assertEqual(b[1].stride(-2), 1)
                                    self.assertEqual(
                                        b[1].stride(), observed["scale"].stride()
                                    )
                                    self.assertEqual(
                                        b[1].data_ptr(), observed["scale"].data_ptr()
                                    )
                                    torch.testing.assert_close(b[1], observed["values"])
                                    self.assertFalse(kwargs["disable_ue8m0_cast"])
                                    out.zero_()

                                with mock.patch.object(
                                    fp8,
                                    "_select_fp8_runtime_backend",
                                    return_value=fp8._CUDA_DEEP_GEMM,
                                ), mock.patch.object(
                                    fp8, "is_deep_gemm_e8m0_used", return_value=True
                                ), mock.patch.object(
                                    fp8,
                                    "quant_weight_ue8m0_packed",
                                    side_effect=gpu_requant_contract,
                                ), mock.patch.object(
                                    fp8,
                                    "_resolve_sgl_per_token_group_quant",
                                    return_value=activation_quant_contract,
                                ), mock.patch.object(
                                    fp8,
                                    "_resolve_fp8_gemm_nt",
                                    return_value=gpu_gemm_contract,
                                ):
                                    with torch.inference_mode():
                                        layer.process_weights_after_loading()
                                        self.assertEqual(
                                            layer.weight_scale.stride(-2), 1
                                        )
                                        self.assertGreater(
                                            layer.weight_scale.shape[1], 1
                                        )
                                        for tokens in (1, 7):
                                            x = torch.zeros(
                                                tokens,
                                                layer._fp8_logical_input_size,
                                                dtype=torch.bfloat16,
                                            )
                                            out = layer.quant_method.apply(layer, x)
                                            self.assertEqual(
                                                tuple(out.shape),
                                                (
                                                    tokens,
                                                    layer._fp8_logical_output_size,
                                                ),
                                            )

    def test_fp32_block_scale_storage_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                _checkpoint("column"), os.path.join(directory, "model.safetensors")
            )
            layer = _load(directory, "column", 1, 0, "scratch")
            expected_weight, expected_scale = fp8.per_block_quant_like_legacy(
                layer.weight.data, 128
            )
            with mock.patch.object(
                fp8, "_select_fp8_runtime_backend", return_value=fp8._CUDA_DEEP_GEMM
            ), mock.patch.object(
                fp8, "is_deep_gemm_e8m0_used", return_value=False
            ), mock.patch.object(
                fp8,
                "_resolve_requant_weight_ue8m0",
                side_effect=AssertionError("FP32 path must not requantize"),
            ):
                layer.process_weights_after_loading()
            self.assertEqual(layer.weight_scale.dtype, torch.float32)
            self.assertTrue(layer.weight_scale.is_contiguous())
            torch.testing.assert_close(layer.weight_scale, expected_scale)
            torch.testing.assert_close(
                layer.weight.view(torch.uint8), expected_weight.view(torch.uint8)
            )

    def test_direct_source_quantization_matches_main_and_not_round_trip(self):
        # Only CUDA's final scale packing is replaced; real source checkpoint,
        # TP dispatch and production quantization arithmetic run on CPU.
        def pack_cpu(scale, mn):
            exponents = (scale.view(torch.int32) >> 23).to(torch.int64)
            exponents = exponents.repeat_interleave(128, dim=0)[:mn]
            exponents = torch.nn.functional.pad(
                exponents, (0, (-exponents.shape[1]) % 4)
            )
            parts = exponents.reshape(mn, -1, 4)
            packed = sum(parts[..., i] << (8 * i) for i in range(4)).to(torch.int32)
            return packed.T.contiguous().T

        def main_reference(source):
            # main f559bb4 fp8_kernel.per_block_cast_to_fp8(use_ue8m0=True)
            n, k = source.shape
            blocks = source.reshape(n // 128, 128, k // 128, 128)
            amax = blocks.abs().float().amax((1, 3), keepdim=True).clamp(1e-4)
            scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
            weight = (blocks * (1.0 / scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
            return weight.reshape(n, k), scale.reshape(n // 128, k // 128)

        for kind in ("column", "row", "merged", "qkv", "merged_partial"):
            with tempfile.TemporaryDirectory() as directory:
                save_file(
                    _checkpoint(kind), os.path.join(directory, "model.safetensors")
                )
                for backend in checkpoint_readers("scratch", "fastsafetensors"):
                    # Non-128-row projection boundaries are representable
                    # for an unsharded fused checkpoint, not a partial tile
                    # from which another TP rank's source rows are absent.
                    for tp in ((1,) if kind == "merged_partial" else (1, 2)):
                        for rank in range(tp):
                            with self.subTest(
                                kind=kind, backend=backend, tp=tp, rank=rank
                            ):
                                layer = _load(directory, kind, tp, rank, backend)
                                source = torch.nn.functional.pad(
                                    layer.weight.detach(),
                                    (
                                        0,
                                        (-layer.weight.shape[1]) % 128,
                                        0,
                                        (-layer.weight.shape[0]) % 128,
                                    ),
                                )
                                expected_weight, expected_scale = main_reference(source)
                                # This source discriminates direct main from the
                                # previous two-rounding path, not just layouts.
                                first_weight, first_scale = (
                                    fp8.per_block_quant_like_legacy(source, 128)
                                )
                                restored = fp8_quant.block_quant_dequant(
                                    first_weight,
                                    first_scale,
                                    [128, 128],
                                    torch.bfloat16,
                                )
                                roundtrip_weight, _ = main_reference(restored)
                                self.assertTrue(
                                    torch.any(
                                        expected_weight.view(torch.uint8)
                                        != roundtrip_weight.view(torch.uint8)
                                    )
                                )
                                with mock.patch.object(
                                    fp8,
                                    "_select_fp8_runtime_backend",
                                    return_value=fp8._CUDA_DEEP_GEMM,
                                ), mock.patch.object(
                                    fp8, "is_deep_gemm_e8m0_used", return_value=True
                                ), mock.patch.object(
                                    fp8_quant,
                                    "_transform_scale_ue8m0",
                                    side_effect=pack_cpu,
                                ):
                                    layer.process_weights_after_loading()
                                torch.testing.assert_close(
                                    layer.weight.view(torch.uint8),
                                    expected_weight.view(torch.uint8),
                                    rtol=0,
                                    atol=0,
                                )
                                torch.testing.assert_close(
                                    layer.weight_scale,
                                    pack_cpu(expected_scale, source.shape[0]),
                                    rtol=0,
                                    atol=0,
                                )
                                self.assertEqual(layer.weight_scale.stride(-2), 1)


if __name__ == "__main__":
    unittest.main()
