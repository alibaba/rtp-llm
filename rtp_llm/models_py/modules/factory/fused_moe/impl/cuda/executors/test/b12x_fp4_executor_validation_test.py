import importlib
import math
import threading
import unittest
from importlib import metadata
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.moe_config import (
    B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
    Fp4MoeOp,
    validate_b12x_zeroed_energy_limit,
)
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.device.b12x_fp4 import (
    _E4M3_MIN_NORMAL,
    prepare_b12x_blockscale,
    validate_b12x_checkpoint_input_scale,
    validate_folded_b12x_blockscale,
)
from rtp_llm.device.device_impl import CudaImpl, prepare_static_weights_for_fp4_moe
from rtp_llm.device.flashinfer_b12x_adapter import (
    SUPPORTED_FLASHINFER_VERSION,
    _load_b12x_symbols,
    get_b12x_kernel_tile_n,
    relaxed_b12x_cuda_version_gate,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    NVFP4_BLOCK_SIZE,
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_fp4_executor import (
    B12xFp4Executor,
    _validate_b12x_weight_shapes,
    _validate_execute_options,
    _validate_execute_payload,
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
        kernel = torch.zeros((2, 128, 64), dtype=torch.uint8)
        scale = torch.ones((2, 128, 8), dtype=torch.float8_e4m3fn)
        prepared_kernel, prepared_scale = prepare_static_weights_for_fp4_moe(
            Fp4MoeOp.B12X.value,
            W.moe_w1,
            W.moe_s1,
            kernel,
            scale,
            scale_2=torch.ones(2, dtype=torch.float32),
            b12x_zeroed_energy_limit=B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
        )
        self.assertIs(prepared_kernel, kernel)
        self.assertEqual(tuple(prepared_scale.shape), (32, 4, 1, 4, 2, 2))
        physical_scale = (
            prepared_scale.permute(5, 2, 4, 0, 1, 3).contiguous().reshape(2, 128, 8)
        )
        self.assertTrue(torch.equal(physical_scale, self._reference_swizzle(scale)))

    def test_b12x_preparation_requires_weight_scale_2(self):
        with self.assertRaisesRegex(ValueError, "requires weight_scale_2"):
            prepare_static_weights_for_fp4_moe(
                Fp4MoeOp.B12X.value,
                W.moe_w1,
                W.moe_s1,
                torch.zeros((2, 128, 64), dtype=torch.uint8),
                torch.ones((2, 128, 8), dtype=torch.float8_e4m3fn),
                b12x_zeroed_energy_limit=B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
            )

    def test_b12x_preparation_requires_explicit_energy_limit(self):
        with self.assertRaisesRegex(ValueError, "must be provided"):
            prepare_static_weights_for_fp4_moe(
                Fp4MoeOp.B12X.value,
                W.moe_w1,
                W.moe_s1,
                torch.zeros((1, 128, 64), dtype=torch.uint8),
                torch.ones((1, 128, 8), dtype=torch.float8_e4m3fn),
                scale_2=torch.ones(1, dtype=torch.float32),
            )

    def test_cuda_impl_reads_emergency_limit_from_moe_config(self):
        device = object.__new__(CudaImpl)
        device.py_env_configs = PyEnvConfigs()
        device.py_env_configs.moe_config.fp4_moe_op = Fp4MoeOp.B12X.value
        device._cache_permute_indices = {}

        kernel = torch.zeros((1, 128, 64), dtype=torch.uint8)
        blockscale = torch.full((1, 128, 8), 2.0**-9, dtype=torch.float8_e4m3fn)
        scale_2 = torch.full((1,), 0.25, dtype=torch.float32)

        with self.assertRaisesRegex(
            ValueError, "of the total scale energy from the GEMM"
        ):
            device.maybe_prepare_static_weights_for_fp4_moe(
                W.moe_w1,
                W.moe_s1,
                kernel,
                blockscale,
                scale_2=scale_2,
            )

        device.py_env_configs.moe_config.b12x_zeroed_energy_limit = 1.0
        with patch(
            "rtp_llm.device.b12x_fp4.convert_b12x_blockscale_to_mma_layout",
            side_effect=lambda scale, **_: scale,
        ):
            prepared_kernel, prepared_scale = (
                device.maybe_prepare_static_weights_for_fp4_moe(
                    W.moe_w1,
                    W.moe_s1,
                    kernel,
                    blockscale,
                    scale_2=scale_2,
                )
            )

        self.assertIs(prepared_kernel, kernel)
        self.assertTrue(torch.equal(prepared_scale, torch.zeros_like(prepared_scale)))

    def test_b12x_preparation_rejects_invalid_weight_scale_2(self):
        kernel = torch.zeros((2, 128, 64), dtype=torch.uint8)
        blockscale = CudaImpl.swizzle_blockscale(
            torch.ones((2, 128, 8), dtype=torch.float8_e4m3fn)
        )
        for scale_2 in (
            torch.ones(2, dtype=torch.float16),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, float("nan")], dtype=torch.float32),
        ):
            with self.subTest(scale_2=scale_2), self.assertRaisesRegex(
                ValueError, "weight_scale_2"
            ):
                prepare_b12x_blockscale(
                    "w1",
                    kernel,
                    blockscale,
                    scale_2,
                    None,
                    B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
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
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE],
        )
        with self.assertRaisesRegex(ValueError, "ll_num_max_token > 0"):
            B12xFp4Executor(config, quant_config, {})

    def test_rejects_ep_topology_before_weight_preparation(self):
        model_config = ModelConfig()
        model_config.expert_num = 2
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 2
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE],
        )
        with self.assertRaisesRegex(ValueError, "requires ep_size=1"):
            B12xFp4Executor(config, quant_config, {})

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
                (32, 4, 2 * intermediate // 128, 4, hidden // 64, experts),
                dtype=torch.float8_e4m3fn,
            ),
            torch.empty(
                (32, 4, hidden // 128, 4, intermediate // 64, experts),
                dtype=torch.float8_e4m3fn,
            ),
        )

    def test_accepts_aligned_shapes(self):
        intermediate, hidden = _validate_b12x_weight_shapes(
            *self._weights(), num_experts=2, kernel_tile_n=128
        )
        self.assertEqual((intermediate, hidden), (128, 128))

    def test_rejects_non_tile_aligned_intermediate_size(self):
        with self.assertRaisesRegex(ValueError, "gate/up tile width 128"):
            _validate_b12x_weight_shapes(
                *self._weights(intermediate=64),
                num_experts=2,
                kernel_tile_n=128,
            )

    def test_rejects_w13_rows_not_aligned_to_swizzle_tile(self):
        with self.assertRaisesRegex(ValueError, r"2\*intermediate_size"):
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
        weights[2] = torch.empty((32, 4, 1, 4, 2, 2), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "w1 blockscale must use"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_mismatched_w2_blockscale_shape(self):
        weights = list(self._weights())
        weights[3] = torch.empty((32, 4, 1, 4, 1, 2), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "w2 blockscale must use"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)

    def test_rejects_ep_sharded_expert_count(self):
        with self.assertRaisesRegex(ValueError, "EP-sharded weights are"):
            _validate_b12x_weight_shapes(
                *self._weights(experts=1),
                num_experts=2,
                kernel_tile_n=128,
            )

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

    def test_rejects_mixed_weight_devices(self):
        weights = list(self._weights())
        weights[1] = torch.empty_like(weights[1], device="meta")
        with self.assertRaisesRegex(ValueError, "must share one device"):
            _validate_b12x_weight_shapes(*weights, num_experts=2, kernel_tile_n=128)


class B12xCheckpointInputScaleValidationTest(unittest.TestCase):
    def test_rejects_input_scale_on_different_device(self):
        with self.assertRaisesRegex(ValueError, "input_scale must be on"):
            validate_b12x_checkpoint_input_scale(
                "w1", torch.ones(1, device="meta"), torch.device("cpu")
            )

    def test_rejects_invalid_input_scale_values(self):
        for value in (0.0, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite, strictly positive"
            ):
                validate_b12x_checkpoint_input_scale(
                    "w1", torch.tensor([value]), torch.device("cpu")
                )


class B12xConstructionReadOnlyTest(unittest.TestCase):
    def _config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.expert_num = 2
        model_config.moe_k = 2
        model_config.hidden_size = 128
        model_config.moe_inter_size = 128
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 1
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

    def _weights(self) -> dict[str, torch.Tensor]:
        return {
            W.moe_w1: torch.empty((2, 256, 64), dtype=torch.uint8),
            W.moe_w2: torch.empty((2, 128, 64), dtype=torch.uint8),
            W.moe_s1: torch.ones((32, 4, 2, 4, 2, 2), dtype=torch.float8_e4m3fn),
            W.moe_s2: torch.ones((32, 4, 1, 4, 2, 2), dtype=torch.float8_e4m3fn),
            W.moe_w1_s2: torch.ones(2, dtype=torch.float32),
            W.moe_w2_s2: torch.ones(2, dtype=torch.float32),
            W.moe_w1_i_s: torch.full((2,), 0.001, dtype=torch.float32),
            W.moe_w2_i_s: torch.full((2,), 0.002, dtype=torch.float32),
        }

    def test_executor_construction_leaves_weights_unchanged(self):
        weights = self._weights()
        original = dict(weights)
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE],
        )
        module = B12xFp4Executor.__module__
        with patch(f"{module}.get_b12x_kernel_tile_n", return_value=128), patch(
            f"{module}.create_b12x_wrappers", return_value=(Mock(), None)
        ):
            B12xFp4Executor(self._config(), quant_config, weights)

        self.assertEqual(set(weights), set(original))
        for key, tensor in original.items():
            self.assertIs(weights[key], tensor)


class B12xFoldValidationTest(unittest.TestCase):
    def test_accepts_zeroed_energy_limit_boundaries(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                self.assertEqual(validate_b12x_zeroed_energy_limit(value), value)

    def test_rejects_invalid_zeroed_energy_limit(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "b12x_zeroed_energy_limit"
            ):
                validate_b12x_zeroed_energy_limit(value)

    def test_rejects_non_finite_folded_scale(self):
        with self.assertRaisesRegex(ValueError, "overflowed e4m3"):
            validate_folded_b12x_blockscale(
                "w1",
                torch.ones(2),
                torch.tensor([1.0, float("nan")]),
                B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
            )

    def test_rejects_material_underflow(self):
        with self.assertRaisesRegex(
            ValueError, "of the total scale energy from the GEMM"
        ):
            validate_folded_b12x_blockscale(
                "w1",
                torch.ones(2),
                torch.zeros(2),
                B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
            )

    def test_rejects_all_zero_scale_product(self):
        with self.assertRaisesRegex(ValueError, "zero total scale energy"):
            validate_folded_b12x_blockscale(
                "w1",
                torch.zeros(2),
                torch.zeros(2),
                B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
            )

    def test_accepts_negligible_underflow(self):
        zeroed, lost_energy, _ = validate_folded_b12x_blockscale(
            "w1",
            torch.tensor([1.0, 1e-4]),
            torch.tensor([1.0, 0.0]),
            B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
        )
        self.assertEqual(zeroed.tolist(), [False, True])
        self.assertLess(lost_energy, 0.001)

    def test_underflow_energy_threshold(self):
        for ratio, should_raise in ((0.5, False), (2.0, True)):
            small = math.sqrt(B12X_ZEROED_ENERGY_LIMIT_DEFAULT * ratio)
            product = torch.tensor([1.0, small])
            folded = torch.tensor([1.0, 0.0])
            with self.subTest(ratio=ratio):
                if should_raise:
                    with self.assertRaisesRegex(
                        ValueError, "of the total scale energy from the GEMM"
                    ):
                        validate_folded_b12x_blockscale(
                            "w1",
                            product,
                            folded,
                            B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
                        )
                else:
                    _, lost_energy, _ = validate_folded_b12x_blockscale(
                        "w1",
                        product,
                        folded,
                        B12X_ZEROED_ENERGY_LIMIT_DEFAULT,
                    )
                    self.assertLess(lost_energy, B12X_ZEROED_ENERGY_LIMIT_DEFAULT)

    def test_reports_nonzero_subnormal_fraction(self):
        product = torch.tensor([1.0, _E4M3_MIN_NORMAL / 2])
        folded = product.clone()
        _, _, subnormal_frac = validate_folded_b12x_blockscale(
            "w1", product, folded, B12X_ZEROED_ENERGY_LIMIT_DEFAULT
        )
        self.assertEqual(subnormal_frac, 0.5)


class B12xExecuteValidationTest(unittest.TestCase):
    def setUp(self):
        self.expert_x = torch.zeros((2, 128), dtype=torch.bfloat16)
        self.topk_ids = torch.zeros((2, 2), dtype=torch.int32)
        self.topk_weights = torch.full((2, 2), 0.5)

    def _validate_payload(self, **overrides):
        expected_top_k = overrides.pop("expected_top_k", 2)
        expected_hidden_size = overrides.pop("expected_hidden_size", 128)
        payload_args = {
            "expert_x": self.expert_x,
            "expert_x_origin_dtype": torch.bfloat16,
            "expert_x_scale": None,
            "expert_topk_ids": self.topk_ids,
            "expert_topk_weights": self.topk_weights,
            "expert_ids_are_local": False,
        }
        payload_args.update(overrides)
        return _validate_execute_payload(
            ExpertForwardPayload(**payload_args),
            expected_top_k=expected_top_k,
            expected_hidden_size=expected_hidden_size,
        )

    def _validate_options(self, **overrides):
        option_args = {
            "activation": "silu",
            "expert_map": None,
            "a2_scale": None,
            "apply_router_weight_on_input": False,
            "extra_expert_args": None,
        }
        option_args.update(overrides)
        return _validate_execute_options(**option_args)

    def test_accepts_supported_inputs(self):
        self._validate_options()
        expert_x, topk_ids, topk_weights = self._validate_payload()
        self.assertIs(expert_x, self.expert_x)
        self.assertIs(topk_ids, self.topk_ids)
        self.assertIs(topk_weights, self.topk_weights)

    def test_rejects_non_bf16_activation(self):
        with self.assertRaisesRegex(ValueError, "consumes bf16"):
            self._validate_payload(expert_x=self.expert_x.float())

    def test_rejects_external_expert_x_scale(self):
        with self.assertRaisesRegex(ValueError, "external expert_x_scale"):
            self._validate_payload(expert_x_scale=torch.ones(2))

    def test_rejects_non_int32_topk_ids(self):
        with self.assertRaisesRegex(ValueError, "top-k ids with dtype torch.int32"):
            self._validate_payload(expert_topk_ids=self.topk_ids.to(torch.int64))

    def test_rejects_non_float32_topk_weights(self):
        with self.assertRaisesRegex(
            ValueError, "top-k weights with dtype torch.float32"
        ):
            self._validate_payload(
                expert_topk_weights=self.topk_weights.to(torch.bfloat16)
            )

    def test_rejects_mismatched_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "hidden_size=64"):
            self._validate_payload(expected_hidden_size=64)

    def test_rejects_mismatched_router_shape(self):
        with self.assertRaisesRegex(ValueError, "top-k ids must have shape"):
            self._validate_payload(expert_topk_ids=self.topk_ids[:1])

    def test_rejects_preapplied_router_weights(self):
        with self.assertRaisesRegex(ValueError, "weight the output twice"):
            self._validate_options(apply_router_weight_on_input=True)

    def test_rejects_expert_map(self):
        with self.assertRaisesRegex(ValueError, "local-expert remapping"):
            self._validate_options(expert_map=torch.arange(2))

    def test_rejects_local_expert_ids(self):
        with self.assertRaisesRegex(ValueError, "requires global expert ids"):
            self._validate_payload(expert_ids_are_local=True)

    def test_rejects_external_a2_scale(self):
        with self.assertRaisesRegex(ValueError, "external a2_scale"):
            self._validate_options(a2_scale=torch.ones(1))

    def test_rejects_extra_expert_args(self):
        with self.assertRaisesRegex(ValueError, "extra expert arguments"):
            self._validate_options(extra_expert_args={"unsupported": True})

    def test_rejects_mismatched_top_k(self):
        with self.assertRaisesRegex(ValueError, "top-k ids must have shape"):
            self._validate_payload(expected_top_k=1)

    def test_accepts_repository_activation_spellings(self):
        for activation in ("SiGLU", "SwiGLU"):
            with self.subTest(activation=activation):
                self._validate_options(activation=activation)

    def test_rejects_unsupported_activation(self):
        with self.assertRaisesRegex(ValueError, "gated SiLU"):
            self._validate_options(activation="gelu")


class B12xFlashInferCompatibilityTest(unittest.TestCase):
    def setUp(self):
        from flashinfer.jit import cpp_ext

        self.cpp_ext = cpp_ext
        self.version_type = cpp_ext.Version

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
        with patch.object(self.cpp_ext, "get_cuda_version", probe):
            with relaxed_b12x_cuda_version_gate(disable_cuda12_9_compat=True):
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

    def test_wrapper_does_not_bind_cuda_version_at_module_scope(self):
        wrapper_module = importlib.import_module(
            _load_b12x_symbols().wrapper.__module__
        )
        self.assertFalse(
            hasattr(wrapper_module, "get_cuda_version"),
            "B12xMoEWrapper now binds get_cuda_version at module scope; update "
            "relaxed_b12x_cuda_version_gate to patch the consumer binding",
        )

    def test_pinned_dependency_closure_is_available(self):
        self.assertEqual(metadata.version("cuda-tile"), "1.4.0")
        self.assertEqual(metadata.version("nvidia-cutlass-dsl"), "4.4.2")


if __name__ == "__main__":
    unittest.main()
