import math
import os
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.moe_config import Fp4MoeOp
from rtp_llm.device.device_impl import CudaImpl, prepare_static_weights_for_fp4_moe
from rtp_llm.device.flashinfer_b12x_adapter import (
    DISABLE_CUDA12_9_COMPAT_ENV,
    SUPPORTED_FLASHINFER_VERSION,
    _load_b12x_symbols,
    get_b12x_kernel_tile_n,
    get_disable_cuda12_9_compat,
    relaxed_b12x_cuda_version_gate,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
    _E4M3_MIN_NORMAL,
    _ZEROED_ENERGY_LIMIT,
    _ZEROED_ENERGY_LIMIT_ENV,
    B12xFp4Executor,
    _get_zeroed_energy_limit,
    _validate_b12x_weight_shapes,
    _validate_execute_inputs,
    _validate_folded_blockscale,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class B12xWeightPreparationTest(unittest.TestCase):
    @staticmethod
    def _reference_swizzle(scale: torch.Tensor) -> torch.Tensor:
        scale_ndim = scale.ndim
        raw = scale.view(torch.uint8)
        if scale_ndim == 2:
            raw = raw.unsqueeze(0)
        batches, rows, cols = raw.shape
        padded_rows = (rows + 127) // 128 * 128
        padded_cols = (cols + 3) // 4 * 4
        output = torch.zeros((batches, padded_rows * padded_cols), dtype=torch.uint8)
        for batch in range(batches):
            for row in range(rows):
                row_block, row_in_block = divmod(row, 128)
                row_group, row_in_group = divmod(row_in_block, 32)
                for col in range(cols):
                    col_block, col_in_block = divmod(col, 4)
                    flat_index = (
                        (
                            (row_block * (padded_cols // 4) + col_block) * 32
                            + row_in_group
                        )
                        * 4
                        + row_group
                    ) * 4 + col_in_block
                    output[batch, flat_index] = raw[batch, row, col]
        output = output.reshape(batches, padded_rows, padded_cols).view(scale.dtype)
        return output[0] if scale_ndim == 2 else output

    def test_swizzle_matches_reference_on_cpu_with_padding(self):
        for dtype in (torch.uint8, torch.float8_e4m3fn):
            for shape in ((3, 5), (2, 3, 5)):
                values = torch.arange(math.prod(shape), dtype=torch.uint8).reshape(
                    shape
                )
                scale = values if dtype is torch.uint8 else values.view(dtype)
                with self.subTest(dtype=dtype, shape=shape):
                    actual = CudaImpl.swizzle_blockscale(scale)
                    expected = self._reference_swizzle(scale)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8), expected.view(torch.uint8)
                        )
                    )

    def test_b12x_preparation_uses_explicit_operator(self):
        kernel = torch.arange(16, dtype=torch.uint8).reshape(1, 4, 4)
        scale = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
        prepared_kernel, prepared_scale = prepare_static_weights_for_fp4_moe(
            Fp4MoeOp.B12X.value, W.moe_w1, W.moe_s1, kernel, scale
        )
        self.assertIs(prepared_kernel, kernel)
        self.assertTrue(
            torch.equal(
                prepared_scale.view(torch.uint8),
                self._reference_swizzle(scale).view(torch.uint8),
            )
        )

    def test_rejects_zero_graph_capacity_before_weight_preparation(self):
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = 0
        config = MoEConfigAdapter(
            model_config=ModelConfig(),
            parallelism_config=ParallelismConfig(),
            moe_config=moe_config,
            enable_cuda_graph=True,
        )
        with self.assertRaisesRegex(ValueError, "ll_num_max_token > 0"):
            B12xFp4Executor(config, Mock(), {})

    def test_rejects_unresolved_auto_operator(self):
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            prepare_static_weights_for_fp4_moe(
                Fp4MoeOp.AUTO.value,
                W.moe_w1,
                W.moe_s1,
                torch.empty((1, 4, 4), dtype=torch.uint8),
                torch.empty((1, 4, 2), dtype=torch.uint8),
            )


class B12xWeightValidationTest(unittest.TestCase):
    def _weights(self, *, experts: int = 2, hidden: int = 128, intermediate: int = 128):
        return (
            torch.empty((experts, 2 * intermediate, hidden // 2), dtype=torch.uint8),
            torch.empty((experts, hidden, intermediate // 2), dtype=torch.uint8),
            torch.empty(
                (experts, 2 * intermediate, hidden // 16),
                dtype=torch.float8_e4m3fn,
            ),
            torch.empty(
                (experts, hidden, intermediate // 16),
                dtype=torch.float8_e4m3fn,
            ),
            torch.ones(experts, dtype=torch.float32),
            torch.ones(experts, dtype=torch.float32),
        )

    def test_accepts_aligned_shapes(self):
        intermediate, hidden = _validate_b12x_weight_shapes(
            *self._weights(), num_experts=2, kernel_tile_n=128
        )
        self.assertEqual((intermediate, hidden), (128, 128))

    def test_accepts_per_expert_scale_with_singleton_dimension(self):
        weights = list(self._weights())
        weights[4] = torch.ones((2, 1))
        weights[5] = torch.ones((2, 1))
        _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_scale_without_leading_expert_dimension(self):
        weights = list(self._weights())
        weights[4] = torch.ones((1, 2))
        with self.assertRaisesRegex(ValueError, "one scalar per expert"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_non_tile_aligned_intermediate_size(self):
        with self.assertRaisesRegex(ValueError, "gate/up tile width 128"):
            _validate_b12x_weight_shapes(
                *self._weights(intermediate=64),
                num_experts=2,
                kernel_tile_n=128,
            )

    def test_rejects_w13_rows_not_aligned_to_swizzle_tile(self):
        with self.assertRaisesRegex(ValueError, "2\*intermediate_size"):
            _validate_b12x_weight_shapes(
                *self._weights(intermediate=96),
                num_experts=2,
                kernel_tile_n=96,
            )

    def test_rejects_non_aligned_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "hidden_size to be a multiple of 128"):
            _validate_b12x_weight_shapes(
                *self._weights(hidden=192),
                num_experts=2,
                kernel_tile_n=128,
            )

    def test_rejects_mismatched_blockscale_shape(self):
        weights = list(self._weights())
        weights[2] = torch.empty((2, 256, 7), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "w1 blockscale shape"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_mismatched_w2_intermediate_size(self):
        weights = list(self._weights())
        weights[1] = torch.empty((2, 128, 32), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "w2 last dim"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_unpacked_weight_dtype(self):
        weights = list(self._weights())
        weights[0] = weights[0].to(torch.float16)
        with self.assertRaisesRegex(ValueError, "packed uint8"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_non_e4m3_blockscale_dtype(self):
        weights = list(self._weights())
        weights[2] = weights[2].to(torch.float32)
        with self.assertRaisesRegex(ValueError, "torch.float8_e4m3fn"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_non_float32_weight_scale_2(self):
        weights = list(self._weights())
        weights[4] = weights[4].to(torch.float16)
        with self.assertRaisesRegex(
            ValueError, "weight_scale_2 must use torch.float32"
        ):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_mixed_weight_devices(self):
        weights = list(self._weights())
        weights[1] = torch.empty_like(weights[1], device="meta")
        with self.assertRaisesRegex(ValueError, "must share one device"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)


class B12xFoldValidationTest(unittest.TestCase):
    def test_reads_zeroed_energy_limit_override(self):
        with patch.dict(os.environ, {_ZEROED_ENERGY_LIMIT_ENV: "0.25"}):
            limit = _get_zeroed_energy_limit()
            self.assertEqual(limit, 0.25)
            _, lost_energy, _ = _validate_folded_blockscale(
                "w1", torch.tensor([1.0, 0.1]), torch.tensor([1.0, 0.0]), limit
            )
            self.assertLess(lost_energy, limit)

    def test_parses_cuda12_9_compat_switch_strictly(self):
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value), patch.dict(
                os.environ, {DISABLE_CUDA12_9_COMPAT_ENV: value}
            ):
                self.assertEqual(get_disable_cuda12_9_compat(), expected)
        for value in ("true", "yes", "2", ""):
            with self.subTest(value=value), patch.dict(
                os.environ, {DISABLE_CUDA12_9_COMPAT_ENV: value}
            ), self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                get_disable_cuda12_9_compat()

    def test_rejects_invalid_zeroed_energy_limit_override(self):
        for value in ("invalid", "nan", "-0.1", "1.1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {_ZEROED_ENERGY_LIMIT_ENV: value}
            ), self.assertRaisesRegex(ValueError, _ZEROED_ENERGY_LIMIT_ENV):
                _get_zeroed_energy_limit()

    def test_rejects_non_finite_folded_scale(self):
        with self.assertRaisesRegex(ValueError, "overflowed e4m3"):
            _validate_folded_blockscale(
                "w1", torch.ones(2), torch.tensor([1.0, float("nan")])
            )

    def test_rejects_material_underflow(self):
        with self.assertRaisesRegex(ValueError, "total scale energy"):
            _validate_folded_blockscale("w1", torch.ones(2), torch.zeros(2))

    def test_rejects_all_zero_scale_product(self):
        with self.assertRaisesRegex(ValueError, "zero total scale energy"):
            _validate_folded_blockscale("w1", torch.zeros(2), torch.zeros(2))

    def test_accepts_negligible_underflow(self):
        zeroed, lost_energy, _ = _validate_folded_blockscale(
            "w1", torch.tensor([1.0, 1e-4]), torch.tensor([1.0, 0.0])
        )
        self.assertEqual(zeroed.tolist(), [False, True])
        self.assertLess(lost_energy, 0.001)

    def test_underflow_energy_threshold(self):
        for ratio, should_raise in ((0.5, False), (2.0, True)):
            small = math.sqrt(_ZEROED_ENERGY_LIMIT * ratio)
            product = torch.tensor([1.0, small])
            folded = torch.tensor([1.0, 0.0])
            with self.subTest(ratio=ratio):
                if should_raise:
                    with self.assertRaisesRegex(ValueError, "total scale energy"):
                        _validate_folded_blockscale("w1", product, folded)
                else:
                    _, lost_energy, _ = _validate_folded_blockscale(
                        "w1", product, folded
                    )
                    self.assertLess(lost_energy, _ZEROED_ENERGY_LIMIT)

    def test_reports_nonzero_subnormal_fraction(self):
        product = torch.tensor([1.0, _E4M3_MIN_NORMAL / 2])
        folded = product.clone()
        _, _, subnormal_frac = _validate_folded_blockscale("w1", product, folded)
        self.assertEqual(subnormal_frac, 0.5)


class B12xExecuteValidationTest(unittest.TestCase):
    def setUp(self):
        self.expert_x = torch.zeros((2, 128), dtype=torch.bfloat16)
        self.topk_ids = torch.zeros((2, 2), dtype=torch.int32)
        self.topk_weights = torch.full((2, 2), 0.5)

    def _validate(self, **overrides):
        args = {
            "expert_x": self.expert_x,
            "topk_ids": self.topk_ids,
            "topk_weights": self.topk_weights,
            "expected_top_k": 2,
            "expected_hidden_size": 128,
            "activation": "silu",
            "expert_map": None,
            "a2_scale": None,
            "apply_router_weight_on_input": False,
            "extra_expert_args": None,
        }
        args.update(overrides)
        return _validate_execute_inputs(**args)

    def test_accepts_supported_inputs(self):
        expert_x, topk_ids, topk_weights = self._validate()
        self.assertIs(expert_x, self.expert_x)
        self.assertIs(topk_ids, self.topk_ids)
        self.assertIs(topk_weights, self.topk_weights)

    def test_rejects_non_bf16_activation(self):
        with self.assertRaisesRegex(ValueError, "consumes bf16"):
            self._validate(expert_x=self.expert_x.float())

    def test_rejects_non_int32_topk_ids(self):
        with self.assertRaisesRegex(ValueError, "top-k ids with dtype torch.int32"):
            self._validate(topk_ids=self.topk_ids.to(torch.int64))

    def test_rejects_non_float32_topk_weights(self):
        with self.assertRaisesRegex(
            ValueError, "top-k weights with dtype torch.float32"
        ):
            self._validate(topk_weights=self.topk_weights.to(torch.bfloat16))

    def test_rejects_mismatched_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "hidden_size=64"):
            self._validate(expected_hidden_size=64)

    def test_rejects_mismatched_router_shape(self):
        with self.assertRaisesRegex(ValueError, "top-k ids must have shape"):
            self._validate(topk_ids=self.topk_ids[:1])

    def test_rejects_preapplied_router_weights(self):
        with self.assertRaisesRegex(ValueError, "weight the output twice"):
            self._validate(apply_router_weight_on_input=True)

    def test_rejects_expert_map(self):
        with self.assertRaisesRegex(ValueError, "local-expert remapping"):
            self._validate(expert_map=torch.arange(2))

    def test_rejects_external_a2_scale(self):
        with self.assertRaisesRegex(ValueError, "external a2_scale"):
            self._validate(a2_scale=torch.ones(1))

    def test_rejects_extra_expert_args(self):
        with self.assertRaisesRegex(ValueError, "extra expert arguments"):
            self._validate(extra_expert_args={"unsupported": True})

    def test_rejects_mismatched_top_k(self):
        with self.assertRaisesRegex(ValueError, "top-k ids must have shape"):
            self._validate(expected_top_k=1)

    def test_accepts_repository_activation_spellings(self):
        for activation in ("SiGLU", "SwiGLU"):
            with self.subTest(activation=activation):
                self._validate(activation=activation)

    def test_rejects_unsupported_activation(self):
        with self.assertRaisesRegex(ValueError, "gated SiLU"):
            self._validate(activation="gelu")


class B12xFlashInferCompatibilityTest(unittest.TestCase):
    def setUp(self):
        from flashinfer.jit import cpp_ext

        self.cpp_ext = cpp_ext
        self.version_type = type(cpp_ext.get_cuda_version())

    def _probe(self, version: str) -> Mock:
        return Mock(return_value=self.version_type(version))

    def test_patches_cuda_12_9_only_inside_context(self):
        probe = self._probe("12.9")
        with patch.object(self.cpp_ext, "get_cuda_version", probe):
            with relaxed_b12x_cuda_version_gate():
                self.assertEqual(str(self.cpp_ext.get_cuda_version()), "13.0")
                self.assertIsNot(self.cpp_ext.get_cuda_version, probe)
            self.assertIs(self.cpp_ext.get_cuda_version, probe)

    def test_other_threads_observe_real_cuda_version(self):
        probe = self._probe("12.9")
        observed_versions = []
        with patch.object(self.cpp_ext, "get_cuda_version", probe):
            with relaxed_b12x_cuda_version_gate():
                worker = threading.Thread(
                    target=lambda: observed_versions.append(
                        str(self.cpp_ext.get_cuda_version())
                    )
                )
                worker.start()
                worker.join()
                self.assertEqual(str(self.cpp_ext.get_cuda_version()), "13.0")
        self.assertEqual(observed_versions, ["12.9"])

    def test_compatibility_patch_can_be_disabled(self):
        probe = self._probe("12.9")
        with patch.dict(os.environ, {DISABLE_CUDA12_9_COMPAT_ENV: "1"}), patch.object(
            self.cpp_ext, "get_cuda_version", probe
        ):
            with relaxed_b12x_cuda_version_gate():
                self.assertIs(self.cpp_ext.get_cuda_version, probe)

    def test_does_not_patch_cuda_13_or_unsupported_cuda(self):
        for version in ("13.0", "12.10", "12.8"):
            probe = self._probe(version)
            with self.subTest(version=version), patch.object(
                self.cpp_ext, "get_cuda_version", probe
            ):
                with relaxed_b12x_cuda_version_gate():
                    self.assertIs(self.cpp_ext.get_cuda_version, probe)

    def test_restores_probe_when_construction_raises(self):
        probe = self._probe("12.9")
        with patch.object(self.cpp_ext, "get_cuda_version", probe):
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                with relaxed_b12x_cuda_version_gate():
                    raise RuntimeError("construction failed")
            self.assertIs(self.cpp_ext.get_cuda_version, probe)

    def test_pinned_flashinfer_apis_exist(self):
        import flashinfer

        self.assertEqual(flashinfer.__version__, SUPPORTED_FLASHINFER_VERSION)
        self.assertGreater(get_b12x_kernel_tile_n(), 0)


if __name__ == "__main__":
    unittest.main()
