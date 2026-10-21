"""SM90 native-g32 dense contracts; executed by the CUDA13 Bazel target."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    expand_block32_scales,
    unpack_ue8m0_input_scales,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

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
                "rtp_llm.models_py.modules.dsv41.sm90_dense.Sm90MXFP8Linear"
            ) as adapter,
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
        ):
            linear = V41MXFP8Linear(weight, scales)
            adapter.assert_called_once_with(weight, scales)
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
                "rtp_llm.models_py.modules.dsv41.sm90_dense.Sm90MXFP8Linear"
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
                "rtp_llm.models_py.modules.dsv41.sm90_dense.Sm90MXFP8Linear"
            ) as adapter,
            patch.object(V41MXFP8Linear, "pack_scales") as pack,
        ):
            linear = V41MXFP8Linear(weight, scales)
            adapter.assert_called_once_with(weight, scales)
            self.assertIs(linear._sm90_linear, adapter.return_value)
            pack.assert_not_called()


class Sm90MXFP8DenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is a dedicated SM90 target: a skipped GPU suite is not a pass.
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("test_sm90_mxfp8_dense requires a CUDA SM90 GPU")

    def setUp(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def _linear(self, n=256, k=512):
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(
            2.0,
            torch.randint(-3, 3, ((n + 31) // 32, k // 32), device="cuda").float(),
        ).to(torch.float8_e8m0fnu)
        return V41MXFP8Linear(weight, scales), weight, scales

    def _reference(self, q, packed, weight, scales):
        rows, k = q.numel() // q.shape[-1], q.shape[-1]
        # Independent integer extraction also works with singleton TMA strides.
        shifts = torch.arange(4, device=packed.device, dtype=torch.int64) * 8
        exponents = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 255).reshape(
            rows, -1
        )[:, : k // 32]
        activation_scales = torch.pow(2.0, exponents.float() - 127)
        activation = q.reshape(rows, -1, 32).float() * activation_scales.unsqueeze(-1)
        w = weight.float() * expand_block32_scales(
            scales, weight.shape[0], k
        ).float().repeat_interleave(32, dim=1)
        return (activation.reshape(rows, k) @ w.T).to(torch.bfloat16)

    def test_prequantized_group32_and_out_match_independent_reference(self):
        for n, k, shape in (
            (256, 128, (1,)),
            (256, 128, (7,)),
            (256, 512, (7,)),
            (320, 640, (3, 5)),
            (256, 512, (129,)),
        ):
            with self.subTest(n=n, k=k, shape=shape):
                linear, weight, scales = self._linear(n, k)
                self.assertFalse(linear._sm90_linear.config.use_fused_e8m0_scale)
                self.assertFalse(linear._sm90_linear.config.use_int_weight_scale)
                self.assertTrue(
                    json.loads(linear._sm90_linear.compute_config)[
                        "use_batch_invariant"
                    ]
                )
                x = torch.randn(*shape, k, device="cuda", dtype=torch.bfloat16)
                q, packed = linear._quantize_input(x)
                q_before = q.view(torch.uint8).clone()
                packed_before = packed.clone()
                out = torch.empty(*shape, n, device="cuda", dtype=torch.bfloat16)
                result = linear.forward_quantized(q, packed, out=out)
                self.assertIs(result, out)
                expected = self._reference(q, packed, weight, scales)
                torch.testing.assert_close(
                    result.reshape_as(expected), expected, rtol=0.015, atol=0.02
                )
                torch.testing.assert_close(
                    q.view(torch.uint8), q_before, rtol=0, atol=0
                )
                torch.testing.assert_close(packed, packed_before, rtol=0, atol=0)
                torch.testing.assert_close(linear(x), result, rtol=0, atol=0)

    def test_actual_tp8_dense_projection_shapes_at_decode_batch(self):
        # Four representatives cover real hidden/Q-rank/grouped-O/expert K.
        # M=3 bounds allocations; this is a numerical component test, not a
        # synthetic performance or full-model acceptance claim.
        for name in ("merged_wq_a_wkv", "wq_b", "wo_a_group", "shared_w2"):
            with self.subTest(projection=name):
                n, k = _V41_TP8_DENSE_SHAPES[name]
                linear, weight, scales = self._linear(n, k)
                x = torch.randn(3, k, dtype=torch.bfloat16, device="cuda")
                q, packed = linear._quantize_input(x)
                out = torch.empty(3, n, dtype=torch.bfloat16, device="cuda")
                self.assertIs(linear.forward_quantized(q, packed, out=out), out)
                expected = self._reference(q, packed, weight, scales)
                if name == "shared_w2":
                    # Anchor the CUDA oracle to an independent CPU BLAS result
                    # at real K, so CUDA TF32 policy cannot hide a backend error.
                    cpu_expected = self._reference(
                        q.cpu(), packed.cpu(), weight.cpu(), scales.cpu()
                    )
                    torch.testing.assert_close(
                        expected.cpu(), cpu_expected, rtol=0.015, atol=0.02
                    )
                torch.testing.assert_close(
                    out,
                    expected,
                    rtol=0.015,
                    atol=0.02,
                )
                del linear, weight, scales, x, q, packed, out

    def test_actual_shared_down_batch_invariant_ab_with_fixed_oracle(self):
        from humming.kernel.humming import HummingKernel

        rows = 3
        n, k = _V41_TP8_DENSE_SHAPES["shared_w2"]
        linear, weight, scales = self._linear(n, k)
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        q, packed = linear._quantize_input(x)
        q_before, packed_before = q.view(torch.uint8).clone(), packed.clone()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        expected = self._reference(q, packed, weight, scales)
        adapter = linear._sm90_linear
        original = adapter.compute_config
        original_config = json.loads(original)
        observations = []
        try:
            for batch_invariant in (False, True):
                # One policy knob changes; inputs, transformed weights,
                # scales, quantization and the FP32 oracle remain identical.
                candidate = dict(original_config, use_batch_invariant=batch_invariant)
                adapter.compute_config = json.dumps(candidate)
                layer_config = adapter.config.to_str()
                _, _, intervals = HummingKernel._resolve_configs(
                    layer_config, adapter.compute_config, None, q.device
                )
                self.assertIsInstance(intervals, list)
                selected = [entry for entry in intervals if entry[0] < rows <= entry[1]]
                self.assertEqual(len(selected), 1)
                low, high, selected_tuning = selected[0]
                compiled = (
                    HummingKernel.prepare_kernels(
                        layer_config, adapter.compute_config, device=q.device
                    )
                    .reshape(-1, 4)
                    .tolist()
                )
                selected_kernels = [
                    entry for entry in compiled if entry[0] < rows <= entry[1]
                ]
                self.assertEqual(len(selected_kernels), 1)
                self.assertEqual(selected_kernels[0][:2], [low, high])
                actual = linear.forward_quantized(q, packed)
                abs_error = (actual.float() - expected.float()).abs()
                observation = {
                    "M": rows,
                    "N": n,
                    "K": k,
                    "batch_invariant": batch_invariant,
                    "oracle_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "selected_m_interval": [low, high],
                    "kernel_id": selected_kernels[0][2],
                    "use_stream_k": bool(selected_tuning.get("use_stream_k", False)),
                    "block_shape": selected_tuning["block_shape"],
                    "warp_shape": selected_tuning["warp_shape"],
                    "max_abs_error": abs_error.max().item(),
                    "rmse": abs_error.square().mean().sqrt().item(),
                    "tolerance_violations": int(
                        (~torch.isclose(actual, expected, rtol=0.015, atol=0.02))
                        .sum()
                        .item()
                    ),
                }
                observations.append(observation)
                print(
                    "SM90_DENSE_POLICY_AB " + json.dumps(observation, sort_keys=True),
                    flush=True,
                )
                if batch_invariant:
                    self.assertFalse(observation["use_stream_k"])
                    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.02)
                # The historical path is diagnostic only: do not require it
                # to fail on GPUs/upstream heuristics that choose another tile.
                torch.testing.assert_close(
                    q.view(torch.uint8), q_before, rtol=0, atol=0
                )
                torch.testing.assert_close(packed, packed_before, rtol=0, atol=0)
        finally:
            adapter.compute_config = original
        self.assertEqual(len(observations), 2)

    def test_empty_rows_preserve_out_without_launch(self):
        linear, _, _ = self._linear()
        q = torch.empty(0, 512, dtype=torch.float8_e4m3fn, device="cuda")
        scales = torch.empty(0, 4, dtype=torch.int32, device="cuda")
        out = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
        with patch.object(linear._sm90_linear, "_forward") as forward:
            self.assertIs(linear.forward_quantized(q, scales, out=out), out)
            forward.assert_not_called()
        self.assertEqual(
            tuple(
                linear(torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)).shape
            ),
            (0, 256),
        )

    def test_prequantized_cuda_graph_replay(self):
        linear, weight, scales = self._linear()
        x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        q, packed = linear._quantize_input(x)
        out = torch.empty(7, 256, dtype=torch.bfloat16, device="cuda")
        # Compile the Humming specialization before graph capture.
        for _ in range(3):
            linear.forward_quantized(q, packed, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            linear.forward_quantized(q, packed, out=out)
        for multiplier in (16.0, 0.03125):
            # Change payload and *every* g32 scale after capture, keeping the
            # source allocations fixed. A stale contiguous scale copy captured
            # outside the graph must not satisfy this check.
            new_q, new_packed = linear._quantize_input(torch.randn_like(x) * multiplier)
            self.assertTrue(
                torch.all(
                    packed.contiguous().view(torch.uint8)
                    != new_packed.contiguous().view(torch.uint8)
                )
            )
            self.assertFalse(torch.equal(q.view(torch.uint8), new_q.view(torch.uint8)))
            q.copy_(new_q)
            packed.copy_(new_packed)
            graph.replay()
            torch.testing.assert_close(
                out, self._reference(q, packed, weight, scales), rtol=0.015, atol=0.02
            )
            eager = linear.forward_quantized(q, packed)
            torch.testing.assert_close(out, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
