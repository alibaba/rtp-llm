"""CPU checkpoint regression for the DeepGEMM packed-scale storage ABI.

Readers, owner TP/GQA dispatch and online block quantization are real. Only the
GPU requantizer/GEMM are replaced with explicit CPU contract doubles; no GPU
numerical accuracy, CUDA graph or smoke result is established by these tests.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

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
    if kind == "qkv":
        root.proj = QKVParallelLinear(
            hidden_size=1152, num_heads=4, num_kv_heads=2, head_dim=128, **kwargs
        )
    elif kind == "merged":
        root.proj = MergedColumnParallelLinear(
            input_size=1152,
            output_size=512,
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
        "merged": [("gate_proj", 256), ("up_proj", 256)],
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
    def test_checkpoint_packed_scales_retain_mn_major_storage_until_gemm(self):
        # K>512 is essential: a single packed column hides contiguous() damage.
        for kind in ("column", "row", "merged", "qkv"):
            with tempfile.TemporaryDirectory() as directory:
                save_file(
                    _checkpoint(kind), os.path.join(directory, "model.safetensors")
                )
                for backend in ("scratch", "fastsafetensors"):
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

        for kind in ("column", "row", "merged", "qkv"):
            with tempfile.TemporaryDirectory() as directory:
                save_file(
                    _checkpoint(kind), os.path.join(directory, "model.safetensors")
                )
                for backend in ("scratch", "fastsafetensors"):
                    for tp in (1, 2):
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
