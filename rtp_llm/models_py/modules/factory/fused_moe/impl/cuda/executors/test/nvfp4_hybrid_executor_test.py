"""NVFP4 dispatch contracts and opt-in SM120 GPU regression tests."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.config.qwen4_exp_config import build_qwen4_quant_config
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor import (
    Nvfp4HybridExecutor,
    decode_tuning,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
    CudaNvfp4CutlassStrategy,
    CudaNvfp4HybridStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)

_FORCED_BACKEND = "dynamic"

MODULE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.nvfp4_hybrid_executor"


def config_for(
    role="DECODE",
    ep=2,
    hidden=2560,
    intermediate=640,
    experts=512,
    top_k=10,
    model_recipe=QWEN4_FP8_NVFP4_RECIPE,
    **tuning,
):
    quant = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=False, model_recipe=model_recipe, **tuning
    )
    pc = SimpleNamespace(
        role_type=role,
        tp_size=ep,
        ep_size=ep,
        dp_size=1,
        pp_size=1,
        get_ffn_tp_size=lambda: 1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            quant_config=quant,
            data_type=torch.bfloat16,
            eplb_config=SimpleNamespace(enable_eplb=lambda: False),
        ),
        parallelism_config=pc,
        ep_size=ep,
        tp_size=ep,
        moe_quant_method="modelopt_fp4",
        moe_strategy="nvfp4_cutlass",
        has_redundant_experts=False,
        moe_config=SimpleNamespace(
            fp4_moe_op="cutlass",
            moe_strategy="nvfp4_cutlass",
            use_all_gather=True,
            use_deepep_moe=False,
        ),
        hidden_size=hidden,
        moe_inter_dim=intermediate,
        n_local_experts=experts // ep,
        expert_num=experts,
        moe_k=top_k,
        moe_w1_layout="up_gate",
    )


class Nvfp4HybridExecutorTest(unittest.TestCase):
    def test_registry_selects_by_stage_and_conditions_without_model_name(self):
        registry = StrategyRegistry()
        baseline, hybrid = CudaNvfp4CutlassStrategy(), CudaNvfp4HybridStrategy()
        registry.register(baseline)
        registry.register(hybrid)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch(f"{MODULE}.B12xNvfp4MoE.is_available", return_value=True),
        ):
            for enabled, role, expected in [
                ("1", "DECODE", hybrid),
                ("1", "PREFILL", baseline),
                ("1", "PDFUSION", baseline),
                ("0", "DECODE", baseline),
            ]:
                with self.subTest(enabled=enabled, role=role), patch.dict(
                    "os.environ", {"RTP_NVFP4_HYBRID_MOE": enabled}
                ):
                    self.assertIs(
                        registry.get_strategy(
                            config_for(role=role, decode_b12x_max_rows=16)
                        ),
                        expected,
                    )
            with patch.dict("os.environ", {"RTP_NVFP4_HYBRID_MOE": "1"}):
                for ep, k, n, e, top_k in [
                    (1, 2048, 1024, 64, 4),
                    (4, 4096, 1536, 128, 8),
                ]:
                    self.assertIs(
                        registry.get_strategy(
                            config_for(
                                ep=ep, hidden=k, intermediate=n, experts=e, top_k=top_k
                            )
                        ),
                        baseline,
                    )
                self.assertIs(
                    registry.get_strategy(
                        config_for(intermediate=192, decode_b12x_max_rows=16)
                    ),
                    baseline,
                )
                config = config_for(decode_b12x_max_rows=16)
                self.assertIs(
                    registry.get_strategy(config_for(ep=8, decode_b12x_max_rows=16)),
                    baseline,
                )
                config.moe_w1_layout = "gate_up"
                self.assertIs(registry.get_strategy(config), baseline)
                with patch("torch.cuda.get_device_capability", return_value=(12, 1)):
                    self.assertIs(
                        registry.get_strategy(config_for(decode_b12x_max_rows=16)),
                        baseline,
                    )
            with patch.dict(os.environ):
                os.environ.pop("RTP_NVFP4_HYBRID_MOE", None)
                self.assertIs(
                    registry.get_strategy(config_for(model_recipe=None)), baseline
                )
                self.assertIs(
                    registry.get_strategy(
                        config_for(model_recipe=None, decode_b12x_max_rows=16)
                    ),
                    baseline,
                )
                config = config_for(model_recipe=QWEN4_FP8_NVFP4_RECIPE)
                self.assertIs(registry.get_strategy(config), hybrid)
                with patch(f"{MODULE}.B12xNvfp4MoE.is_available", return_value=False):
                    self.assertIs(registry.get_strategy(config), baseline)

    def test_dispatch_uses_actual_rows_and_configured_boundary(self):
        executor = Nvfp4HybridExecutor.__new__(Nvfp4HybridExecutor)
        self.assertEqual(executor.topk_ids_dtype, torch.int32)
        executor.hidden_size = 2560
        executor.top_k = 10
        executor.max_b12x_rows = 16
        executor.cutlass_profiles = [13, 37]
        executor.ep_size = 2
        executor.ep_rank = 0
        executor.tp_size = 1
        executor.tp_rank = 0
        executor.w1 = object()
        executor.w2 = object()
        executor.quant_scales = []
        executor.b12x = Mock(side_effect=lambda x, ids, weights: x + 1)
        with patch(
            MODULE + ".cutlass_fused_moe",
            side_effect=lambda **kwargs: [kwargs["input"] - 1],
        ) as cutlass:
            # Includes the threshold and its immediate neighbor, not request concurrency.
            for rows in (*range(1, 18), 32, 64):
                x = torch.zeros(rows, 2560, dtype=torch.bfloat16)
                payload = ExpertForwardPayload(
                    expert_x=x,
                    expert_topk_ids=torch.zeros(
                        rows, 10, dtype=executor.topk_ids_dtype
                    ),
                    expert_topk_weights=torch.ones(rows, 10),
                )
                output = executor.execute(
                    payload, "SiGLU", None, None, False, None
                ).fused_expert_output
                self.assertTrue(torch.equal(output, x + (1 if rows <= 16 else -1)))
                if rows <= 16:
                    ids = executor.b12x.call_args.args[1]
                else:
                    ids = cutlass.call_args.kwargs["token_selected_experts"]
                self.assertEqual(ids.dtype, torch.int32)
                self.assertEqual(ids.data_ptr(), payload.expert_topk_ids.data_ptr())
            self.assertEqual(executor.b12x.call_count, 16)
            self.assertEqual(cutlass.call_count, 3)
            self.assertEqual(cutlass.call_args.kwargs["profile_ids"], [13, 37])
            executor.max_b12x_rows = 8
            executor.cutlass_profiles = None
            payload.expert_x = torch.zeros(16, 2560, dtype=torch.bfloat16)
            payload.expert_topk_ids = torch.zeros(16, 10, dtype=torch.int64)
            payload.expert_topk_weights = torch.ones(16, 10)
            executor.execute(payload, "SiGLU", None, None, False, None)
            self.assertNotIn("profile_ids", cutlass.call_args.kwargs)
            payload.expert_ids_are_local = True
            with self.assertRaisesRegex(ValueError, "global expert IDs"):
                executor.execute(payload, "SiGLU", None, None, False, None)

    def test_b12x_adapter_derives_geometry_and_isolates_workspace_shapes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            B12xNvfp4MoE,
        )

        def launch(*, quant_mode):
            return _FORCED_BACKEND

        def weight_views(*, branches, **kwargs):
            return dict(branches=branches, **kwargs)

        def allocate(*, backend):
            raise AssertionError("Constructing weights must not allocate a row bucket")

        convert = Mock(side_effect=lambda value, **kwargs: value)
        dispatch = SimpleNamespace(
            launch_sm120_static_moe=launch,
            _get_weight_views=weight_views,
            allocate_sm120_moe_workspace=allocate,
        )
        model = object()
        pools = []
        with patch.object(B12xNvfp4MoE, "_runtime", return_value=(convert, dispatch)):
            for e, k, n, topk in [(4, 256, 256, 2), (8, 512, 384, 4)]:
                convert.reset_mock()
                w1 = torch.zeros(e, 2 * n, k // 16, dtype=torch.int64)
                w2 = torch.zeros(e, k, n // 16, dtype=torch.int64)
                scales = [
                    torch.ones(1),
                    torch.zeros(16, dtype=torch.int32),
                    torch.ones(e),
                    torch.ones(1),
                    torch.zeros(16, dtype=torch.int32),
                    torch.ones(e),
                ]
                adapter = B12xNvfp4MoE(w1, w2, scales, 1, model, topk)
                self.assertEqual(
                    (
                        adapter.num_experts,
                        adapter.hidden_size,
                        adapter.intermediate_size,
                        adapter.top_k,
                    ),
                    (e, k, n, topk),
                )
                self.assertEqual(adapter.weights["w1_fp4"].shape, (e, 2 * n, k // 2))
                self.assertEqual(adapter.weights["w2_fp4"].shape, (e, k, n // 2))
                self.assertEqual(
                    convert.call_args_list[0].kwargs,
                    dict(m=2 * n, k=k, num_groups=e, sf_vec_size=16),
                )
                self.assertEqual(
                    convert.call_args_list[1].kwargs,
                    dict(m=k, k=n, num_groups=e, sf_vec_size=16),
                )
                self.assertEqual(launch(quant_mode="nvfp4"), "dynamic")
                same_shape = B12xNvfp4MoE(w1, w2, scales, 1, model, topk)
                self.assertIs(adapter.states, same_shape.states)
                pools.append(adapter.workspace_pool)
        self.assertIsNot(pools[0], pools[1])

    def test_static_entry_does_not_use_backend_arbitration(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            _launch_static,
        )

        tensor = torch.zeros(8, 8)
        workspace = SimpleNamespace(
            **{
                name: tensor
                for name in (
                    "row_counts",
                    "token_map",
                    "route_output_scratch",
                    "packed_a_view",
                    "packed_input_scale",
                    "packed_a_flat",
                    "scale_flat",
                    "barrier_count",
                    "barrier_epoch",
                    "route_state",
                    "active_expert_count",
                    "weight_expert_ids",
                    "global_to_local_expert",
                    "virt_route_scratch",
                    "token_weights",
                )
            },
            max_rows=8,
        )
        weights = SimpleNamespace(
            **{
                name: tensor
                for name in (
                    "_w13_sf_storage",
                    "_down_sf_storage",
                    "static_w13_fp4",
                    "static_down_fp4",
                    "w1_alpha",
                    "w2_alpha",
                )
            },
            static_intermediate_size=256,
            intermediate_size=256,
            static_w13_sf_storage=None,
            source_scales=True,
        )
        compiled = Mock()
        forbidden = Mock(side_effect=AssertionError("backend arbitration reached"))
        dispatch = SimpleNamespace(
            _align_up=lambda x, a: (x + a - 1) // a * a,
            _LEVEL_TILE_N=128,
            get_num_sm=lambda device: 80,
            get_max_active_clusters=lambda n: 80,
            _lookup_mac_ladder=lambda ladder, pairs: None,
            _STATIC_MAC_LADDER=(),
            _static_merged_groups=lambda n, pairs: False,
            _static_deferred_init=lambda pairs: False,
            _get_static_kernel=Mock(return_value=(compiled, 64)),
            launch_sm120_static_moe=forbidden,
            _get_micro_kernel=forbidden,
            _get_direct_micro_kernel=forbidden,
        )
        for backend in (None, "static", "micro", "direct_micro", "dynamic"):
            dispatch._FORCED_BACKEND = backend
            output = _launch_static(
                dispatch,
                workspace=workspace,
                weights=weights,
                a=tensor,
                topk_ids=torch.zeros(2, 2, dtype=torch.int32),
                topk_weights=torch.ones(2, 2),
                input_gs=tensor,
                down_input_scale=tensor,
                scatter_output=tensor,
                num_experts=4,
                num_tokens=2,
                k=256,
                n=256,
                top_k=2,
            )
            self.assertIs(output, tensor)
            self.assertEqual(dispatch._FORCED_BACKEND, backend)
            self.assertEqual(
                dispatch._get_static_kernel.call_args.kwargs["quant_mode"], "nvfp4"
            )
            self.assertEqual(
                dispatch._get_static_kernel.call_args.kwargs["mac_override"], 64
            )
        self.assertEqual(compiled.call_count, 5)
        forbidden.assert_not_called()

    def test_adapter_keeps_nonuniform_swizzled_scale_bytes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            B12xNvfp4MoE,
        )

        def launch(**kwargs):
            return _FORCED_BACKEND

        experts, hidden, intermediate = 4, 256, 256
        values = []

        def convert(value, *, m, k, num_groups, sf_vec_size):
            # Assert the adapter passes the physical packed bytes unchanged.
            packed = values[len(seen)]
            self.assertTrue(
                torch.equal(value.view(torch.uint8), packed.view(torch.uint8))
            )
            result = value.view(num_groups, m // 128, k // 64, 32, 4, 4).permute(
                3, 4, 1, 5, 2, 0
            )
            self.assertEqual(result.data_ptr(), value.data_ptr())
            self.assertEqual(
                tuple(result.shape), (32, 4, m // 128, 4, k // 64, num_groups)
            )
            self.assertFalse(result.is_contiguous())
            seen.append(result)
            return result

        for rows, cols in ((2 * intermediate, hidden), (hidden, intermediate)):
            _, packed = B12xNvfp4GpuTest.swizzled_scales(experts, rows, cols, 9, "cpu")
            values.append(packed)
        scales = [
            torch.ones(()),
            values[0],
            torch.ones(experts),
            torch.ones(()),
            values[1],
            torch.ones(experts),
        ]
        seen = []
        dispatch = SimpleNamespace(
            launch_sm120_static_moe=launch, _get_weight_views=lambda **kw: kw
        )
        with patch.object(B12xNvfp4MoE, "_runtime", return_value=(convert, dispatch)):
            B12xNvfp4MoE(
                torch.zeros(experts, 2 * intermediate, hidden // 16, dtype=torch.int64),
                torch.zeros(experts, hidden, intermediate // 16, dtype=torch.int64),
                scales,
                0,
                object(),
                2,
            )
        self.assertEqual(len(seen), 2)

    def test_tuning_defaults_do_not_apply_qwen_tactics_to_other_models(self):
        for timeout in (-1, True, 1.5, 2**31):
            with self.assertRaisesRegex(ValueError, "decode_lsa_timeout_ms"):
                ModelOptFp4Config(
                    bits=4,
                    group_size=16,
                    is_quanted=False,
                    decode_lsa_timeout_ms=timeout,
                )
        config = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        self.assertEqual(config.decode_lsa_timeout_ms, 0)
        self.assertEqual(decode_tuning(config), (0, None))
        config.decode_cutlass_profile_ids = [13, 37]
        self.assertEqual(decode_tuning(config), (0, [13, 37]))
        requested = json.dumps(
            dict(
                method="modelopt_fp4",
                bits=4,
                group_size=16,
                is_quanted=False,
                model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                decode_b12x_max_rows=8,
                decode_cutlass_profile_ids=[13, 37],
            )
        )
        resolved = build_qwen4_quant_config(requested, source_fp8=True)
        self.assertEqual(decode_tuning(resolved), (8, [13, 37]))
        for bad in ([13], [13, 37, 1], [13, True], [-2, 37]):
            config.decode_cutlass_profile_ids = bad
            with self.assertRaises(ValueError):
                decode_tuning(config)
        config.decode_cutlass_profile_ids = None
        for rows in (0, 16):
            config.decode_b12x_max_rows = rows
            self.assertEqual(decode_tuning(config), (rows, None))
        for bad in (-1, 17, True):
            config.decode_b12x_max_rows = bad
            with self.assertRaisesRegex(ValueError, "decode_b12x_max_rows"):
                decode_tuning(config)


@unittest.skipUnless(
    os.environ.get("RTP_TEST_B12X_NVFP4_GPU") == "1",
    "requires SM120 and the pinned FlashInfer b12x runtime",
)
class B12xNvfp4GpuTest(unittest.TestCase):
    @staticmethod
    def swizzled_scales(experts, rows, cols, offset, device):
        # Logical block16 scales vary across experts, rows and K tiles.
        logical = (
            torch.arange(experts * rows * (cols // 16), device=device)
            .reshape(experts, rows, cols // 16)
            .add(offset)
            .remainder(29)
            .add(4)
            .div(128)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        packed = (
            logical.reshape(experts, rows // 128, 4, 32, cols // 64, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .view(experts, rows, cols // 16)
            .view(torch.int32)
        )
        return logical, packed

    @staticmethod
    def packed_weights(experts, hidden, intermediate, device, *, nonuniform=False):
        packed = 0x2222222222222222
        w1 = torch.full(
            (experts, 2 * intermediate, hidden // 16),
            packed,
            dtype=torch.int64,
            device=device,
        )
        w2 = torch.full(
            (experts, hidden, intermediate // 16),
            packed,
            dtype=torch.int64,
            device=device,
        )
        scale1 = torch.ones(
            (experts, 2 * intermediate, hidden // 16), device=device
        ).to(torch.float8_e4m3fn)
        scale2 = torch.ones((experts, hidden, intermediate // 16), device=device).to(
            torch.float8_e4m3fn
        )
        alpha = torch.ones(experts, device=device)
        inverse_act_scale = torch.ones((), device=device)
        if nonuniform:

            def fill_codes(weight, offset):
                expert = torch.arange(weight.shape[0], device=device)[:, None, None]
                row = torch.arange(weight.shape[1], device=device)[None, :, None]
                column = torch.arange(weight.shape[2], device=device)[None, None, :]
                weight.zero_()
                for lane in range(16):
                    codes = (
                        expert + 2 * row + column + row // intermediate + lane + offset
                    ) % 5 + 1
                    weight.bitwise_or_(codes << (4 * lane))

            fill_codes(w1, 0)
            fill_codes(w2, 2)
            alpha = torch.linspace(0.75, 1.25, experts, device=device)
            inverse_act_scale = torch.full((), 1.25, device=device)
        return (
            w1,
            w2,
            [
                inverse_act_scale,
                scale1.view(torch.int32),
                alpha,
                inverse_act_scale * 0.75 if nonuniform else inverse_act_scale,
                scale2.view(torch.int32),
                alpha.flip(0) if nonuniform else alpha,
            ],
        )

    def test_remote_routes_and_workspace_reuse_in_eager_and_graph(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            B12xNvfp4MoE,
        )

        self.assertTrue(
            B12xNvfp4MoE.is_available(),
            "Pinned runtime is mandatory for this GPU target",
        )
        _, dispatch = B12xNvfp4MoE._runtime()
        original_backend = dispatch._FORCED_BACKEND
        static = Mock(wraps=dispatch._get_static_kernel)
        forbidden = Mock(
            side_effect=AssertionError(
                "static workspace must not launch a micro backend"
            )
        )
        with patch.dict(
            dispatch.__dict__,
            {
                "_get_static_kernel": static,
                "launch_sm120_static_moe": forbidden,
                "_get_micro_kernel": forbidden,
                "_get_direct_micro_kernel": forbidden,
            },
        ):
            for ep_rank in (0, 1):
                for rows in range(1, 17):
                    with self.subTest(ep_rank=ep_rank, rows=rows):
                        before = static.call_count
                        self.check_remote_routes(ep_rank, rows)
                        print(
                            f"B12X_STATIC_CASE ep_rank={ep_rank} rows={rows}",
                            flush=True,
                        )
                        self.assertGreater(static.call_count, before)
                        forbidden.assert_not_called()
                        self.assertEqual(dispatch._FORCED_BACKEND, original_backend)

    def check_remote_routes(self, ep_rank, rows):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.b12x_nvfp4 import (
            B12xNvfp4MoE,
        )

        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        device = torch.device("cuda", torch.cuda.current_device())
        experts, hidden, intermediate, top_k = 4, 256, 256, 2
        w1, w2, scales = self.packed_weights(
            experts, hidden, intermediate, device, nonuniform=True
        )
        model = object()
        adapters = []
        layer_scales = []
        for offset in (0, 11):
            logical1, packed1 = self.swizzled_scales(
                experts, 2 * intermediate, hidden, offset, device
            )
            logical2, packed2 = self.swizzled_scales(
                experts, hidden, intermediate, offset, device
            )
            scales[1], scales[4] = packed1, packed2
            layer = B12xNvfp4MoE(w1, w2, scales, ep_rank, model, top_k)
            layer_scales.append(list(scales))
            convert, _ = layer._runtime()
            for logical, packed in ((logical1, packed1), (logical2, packed2)):
                view = convert(
                    packed.view(torch.float8_e4m3fn),
                    m=logical.shape[1],
                    k=logical.shape[2] * 16,
                    num_groups=experts,
                    sf_vec_size=16,
                )
                self.assertEqual(view.data_ptr(), packed.data_ptr())
                restored = view.view(torch.uint8).permute(5, 2, 1, 0, 4, 3)
                self.assertTrue(torch.equal(restored.reshape(logical.shape), logical))
            adapters.append(layer)
        adapter, second_layer = adapters
        self.assertIs(adapter.states, second_layer.states)

        def poison_output(launch):
            def wrapper(**kwargs):
                kwargs["scatter_output"].fill_(float("nan"))
                launch(**kwargs)

            return wrapper

        for layer in adapters:
            layer.launch = poison_output(layer.launch)
        x = (
            torch.arange(rows * hidden, device=device)
            .view(rows, hidden)
            .remainder(53)
            .add(7)
            .div(256)
            .to(torch.bfloat16)
        )
        local_ids = torch.tensor(
            [ep_rank * experts, ep_rank * experts + 1], dtype=torch.int32, device=device
        ).repeat(rows, 1)
        remote_ids = local_ids + (experts if ep_rank == 0 else -experts)
        mixed_ids = local_ids.clone()
        mixed_ids[::2].copy_(remote_ids[::2])
        mixed_slots = local_ids.clone()
        mixed_slots[:, 1].copy_(remote_ids[:, 1])
        routes = local_ids.clone()
        routing_weights = torch.full((rows, top_k), 0.5, device=device)
        expected_local = adapter(x, routes, routing_weights)
        second_expected = second_layer(x, routes, routing_weights)

        def cutlass_reference(selected, quant_scales):
            from flashinfer import ActivationType
            from flashinfer.fused_moe import cutlass_fused_moe

            reference = cutlass_fused_moe(
                input=x,
                input_sf=None,
                token_selected_experts=selected,
                token_final_scales=routing_weights,
                fc1_expert_weights=w1,
                fc2_expert_weights=w2,
                output_dtype=torch.bfloat16,
                quant_scales=quant_scales,
                tp_size=1,
                tp_rank=0,
                ep_size=2,
                ep_rank=ep_rank,
                activation_type=ActivationType.Swiglu,
                enable_pdl=False,
            )
            return reference[0] if isinstance(reference, list) else reference

        for expected, quant_scales in zip(
            (expected_local, second_expected), layer_scales
        ):
            torch.testing.assert_close(
                expected,
                cutlass_reference(local_ids, quant_scales),
                atol=0.01,
                rtol=0.02,
            )
        routes.copy_(mixed_slots)
        expected_mixed_slots = adapter(x, routes, routing_weights)
        torch.testing.assert_close(
            expected_mixed_slots,
            cutlass_reference(mixed_slots, layer_scales[0]),
            atol=0.01,
            rtol=0.02,
        )
        self.assertTrue(torch.isfinite(second_expected).all().item())
        self.assertFalse(torch.equal(expected_local, second_expected))
        self.assertTrue(torch.isfinite(expected_local).all().item())
        self.assertGreater(expected_local.abs().min().item(), 0)
        expected_remote = torch.zeros_like(expected_local)
        expected_mixed = expected_local.clone()
        expected_mixed[::2].zero_()

        for ids, expected in (
            (remote_ids, expected_remote),
            (mixed_ids, expected_mixed),
            (mixed_slots, expected_mixed_slots),
        ):
            routes.copy_(ids)
            torch.testing.assert_close(
                adapter(x, routes, routing_weights), expected, atol=0, rtol=0
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = adapter(x, routes, routing_weights)
        second_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second_graph):
            second_output = second_layer(x, local_ids, routing_weights)
        for ids, expected in (
            (local_ids, expected_local),
            (remote_ids, expected_remote),
            (mixed_ids, expected_mixed),
            (mixed_slots, expected_mixed_slots),
            (remote_ids, expected_remote),
        ):
            routes.copy_(ids)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            second_graph.replay()
            torch.testing.assert_close(second_output, second_expected, atol=0, rtol=0)

    def test_explicit_cutlass_tactics_above_b12x_limit(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        device = torch.device("cuda", torch.cuda.current_device())
        executor = Nvfp4HybridExecutor.__new__(Nvfp4HybridExecutor)
        executor.hidden_size, executor.top_k = 2560, 10
        executor.max_b12x_rows, executor.cutlass_profiles = 16, [13, 37]
        executor.tp_size, executor.tp_rank = 1, 0
        executor.ep_size, executor.ep_rank = 2, 1
        executor.b12x = Mock(side_effect=AssertionError("large M must use CUTLASS"))
        executor.w1, executor.w2, executor.quant_scales = self.packed_weights(
            256, executor.hidden_size, 640, device
        )
        rows = 32
        x = torch.full(
            (rows, executor.hidden_size), 0.125, device=device, dtype=torch.bfloat16
        )
        ids = torch.arange(256, 266, device=device, dtype=torch.int32).repeat(rows, 1)
        payload = ExpertForwardPayload(
            expert_x=x,
            expert_topk_ids=ids,
            expert_topk_weights=torch.full((rows, executor.top_k), 0.1, device=device),
        )

        def run():
            return executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output

        expected = run()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(expected).all().item())
        self.assertGreater(expected.abs().min().item(), 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        x.zero_()
        graph.replay()
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        executor.b12x.assert_not_called()


if __name__ == "__main__":
    unittest.main()
