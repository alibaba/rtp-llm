"""Scale and architecture-dispatch contracts; native GPU acceptance is separate."""

import gc
import json
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    expand_block32_scales,
    unpack_ue8m0_input_scales,
)
from rtp_llm.models_py.modules.dsv41.utils import (
    V41MXFP8Linear,
    release_prepared_sm90_checkpoint_weights,
)

# Checkpoint config: hidden=5120, heads=64, head_dim=512, q_rank=1280,
# o_rank=1024, o_groups=8, expert_intermediate=2304. AttentionFP8 slices
# wq_b/wo_a/wo_b by TP, but MoE/W13SharedExpert keep the shared expert whole.
_V41_TP8_DENSE_SHAPES = {
    "wq_a": (1280, 5120),
    "wkv": (512, 5120),
    "merged_wq_a_wkv": (1792, 5120),
    "wq_b": (4096, 1280),
    "wo_a_group": (1024, 4096),
    "wo_b": (5120, 1024),
    "shared_w13": (4608, 5120),
    "shared_w2": (5120, 2304),
}


class DenseScaleAndDispatchTest(unittest.TestCase):
    def setUp(self):
        backend = patch.dict(
            os.environ,
            {
                "DSV41_SM90_DENSE_BACKEND": "deepgemm",
                "DSV41_SM90_DENSE_BACKEND_MAP": "{}",
            },
        )
        backend.start()
        self.addCleanup(backend.stop)

    def test_block32_scale_expansion_preserves_exponent_bytes(self):
        scales = torch.arange(120, 144, dtype=torch.uint8).reshape(3, 8)
        actual = expand_block32_scales(scales.view(torch.float8_e8m0fnu), 65, 256)
        self.assertEqual(actual.dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            actual.view(torch.uint8),
            scales.repeat_interleave(32, dim=0)[:65],
            rtol=0,
            atol=0,
        )

    def test_invalid_scale_contract_rejected(self):
        with self.assertRaises(TypeError):
            expand_block32_scales(torch.ones(2, 8), 64, 256)
        with self.assertRaises(ValueError):
            expand_block32_scales(torch.ones(2, 8).to(torch.float8_e8m0fnu), 96, 256)

    def test_unpack_preserves_all_ue8m0_values_including_subnormal_and_nan(self):
        exponent_bytes = torch.arange(256, dtype=torch.uint8).reshape(2, 128)
        # Match RTP's column-major packed words, not just contiguous storage.
        packed = exponent_bytes.view(torch.int32).T.contiguous().T
        self.assertEqual(packed.stride(0), 1)
        actual = unpack_ue8m0_input_scales(packed)
        expected = torch.exp2(exponent_bytes.float() - 127)
        expected[-1, -1] = float("nan")
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_unpack_singleton_tma_strides_without_dtype_view_failure(self):
        for rows in (0, 1, 3, 7):
            with self.subTest(rows=rows):
                aligned = max(4, (rows + 3) // 4 * 4)
                packed = torch.empty_strided((rows, 1), (1, aligned), dtype=torch.int32)
                packed.fill_(0x7F7F7F7F)
                self.assertTrue(packed.is_contiguous())
                self.assertEqual(packed.stride(-1), aligned)
                actual = unpack_ue8m0_input_scales(packed)
                torch.testing.assert_close(actual, torch.ones(rows, 4), rtol=0, atol=0)

    def test_actual_tp8_projection_scale_geometry_without_weight_allocation(self):
        self.assertEqual(_V41_TP8_DENSE_SHAPES["wq_b"][0], 64 // 8 * 512)
        self.assertEqual(_V41_TP8_DENSE_SHAPES["wo_b"][1], 8 // 8 * 1024)
        self.assertEqual(_V41_TP8_DENSE_SHAPES["shared_w13"][0], 2 * 2304)
        for name, (n, k) in _V41_TP8_DENSE_SHAPES.items():
            with self.subTest(projection=name):
                scales = torch.empty(
                    (n // 32, k // 32), device="meta", dtype=torch.float8_e8m0fnu
                )
                expanded = expand_block32_scales(scales, n, k)
                self.assertEqual(tuple(expanded.shape), (n, k // 32))
                self.assertEqual(expanded.dtype, torch.float8_e8m0fnu)

    def test_sm90_dispatch_does_not_call_deepgemm_scale_packer(self):
        weight = SimpleNamespace(shape=(256, 512), is_cuda=True, device="cuda:0")
        scales = object()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.hip", None),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense.Sm90DeepGEMMMXFP8Linear"
            ) as adapter,
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
        ):
            linear = V41MXFP8Linear(weight, scales)
            adapter.assert_called_once_with(weight, scales)
            self.assertIs(linear.weight, adapter.return_value.weight)
            pack.assert_not_called()
            q, s, out = object(), object(), object()
            self.assertIs(
                linear.forward_quantized(q, s, out=out),
                adapter.return_value.forward_quantized.return_value,
            )
            adapter.return_value.forward_quantized.assert_called_once_with(
                q, s, out=out
            )

    def test_sm100_keeps_existing_packed_scale_path(self):
        weight = SimpleNamespace(shape=(256, 512), is_cuda=True, device="cuda:0")
        scales = object()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.hip", None),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
        ):
            linear = V41MXFP8Linear(weight, scales)
            pack.assert_called_once_with(scales)
            self.assertIsNone(linear._sm90_linear)
            self.assertIs(linear.weight, weight)
            self.assertIs(linear.weight_scales, pack.return_value)

    def test_sm100_shared_executor_ignores_sm90_overlap_setting(self):
        from rtp_llm.models_py.modules.dsv41.moe.shared_expert import (
            MXFP8SharedExpertExecutor,
        )

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.hip", None),
            patch.dict("os.environ", {"DSV4_SHARED_EXPERT_MODE": "sm90-only-value"}),
            patch(
                "rtp_llm.models_py.modules.dsv41.moe.shared_expert._find_module_cuda_device",
                return_value=torch.device("cuda:0"),
            ),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch(
                "rtp_llm.models_py.modules.dsv41.moe.shared_expert._ensure_shared_expert_stream"
            ) as create_stream,
        ):
            executor = MXFP8SharedExpertExecutor()
            executor.prepare(object())
            create_stream.assert_not_called()
            self.assertIsNone(executor._stream)

    def test_rocm_major9_keeps_existing_dense_cache_and_shared_paths(self):
        from rtp_llm.models_py.modules.dsv41.kv_cache_layout import use_v4_kv_cache
        from rtp_llm.models_py.modules.dsv41.moe.shared_expert import (
            MXFP8SharedExpertExecutor,
        )

        weight = SimpleNamespace(shape=(256, 512), is_cuda=True, device="cuda:0")
        scales = object()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.hip", "6.3.mock"),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense.Sm90DeepGEMMMXFP8Linear"
            ) as adapter,
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
            patch(
                "rtp_llm.models_py.modules.dsv41.moe.shared_expert._find_module_cuda_device",
                return_value=torch.device("cuda:0"),
            ),
            patch(
                "rtp_llm.models_py.modules.dsv41.moe.shared_expert._ensure_shared_expert_stream"
            ) as create_stream,
            patch.dict("os.environ", {"DSV4_SHARED_EXPERT_MODE": "sm90-only-value"}),
        ):
            linear = V41MXFP8Linear(weight, scales)
            self.assertIsNone(linear._sm90_linear)
            pack.assert_called_once_with(scales)
            adapter.assert_not_called()
            self.assertFalse(use_v4_kv_cache(weight.device))
            executor = MXFP8SharedExpertExecutor()
            executor.prepare(object())
            create_stream.assert_not_called()
            self.assertIsNone(executor._stream)

    def test_shared_expert_shape_uses_production_sm90_adapter(self):
        weight = SimpleNamespace(shape=(4608, 5120), is_cuda=True, device="cuda:0")
        scales = object()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.version.hip", None),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense.Sm90DeepGEMMMXFP8Linear"
            ) as adapter,
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
        ):
            linear = V41MXFP8Linear(weight, scales)
            adapter.assert_called_once_with(weight, scales)
            self.assertIs(linear._sm90_linear, adapter.return_value)
            pack.assert_not_called()
