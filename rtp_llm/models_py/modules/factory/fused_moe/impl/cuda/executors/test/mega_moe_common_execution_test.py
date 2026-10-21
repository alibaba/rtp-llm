"""Exercise the common executor with each DeepGEMM recipe and model policy."""

import os
import unittest
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.mega_moe import MegaMoeExecutionPolicy
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _MEGA_MOE_JIT_WARMED_KEYS,
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Backend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SharedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fused import (
    MegaMoeFusedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
)

PACKAGE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors"
RECIPES = (
    MegaMoeExecutor,
    MegaMoeSEExecutor,
    MegaMoeFp8Backend,
    MegaMoeFp8SharedBackend,
    MegaMoeFusedBackend,
)
SHARED_SCALES = (MegaMoeSEExecutor, MegaMoeFp8SharedBackend)


class RecordingPacker:
    name = "fused"

    def __init__(self):
        self.calls = []

    def pack(self, x, weights, ids, buf, tokens):
        self.calls.append(tokens)
        buf.x[:tokens].copy_(x)
        buf.x_sf[:tokens].fill_(7)
        buf.topk_weights[:tokens].copy_(weights)
        buf.topk_idx[:tokens].copy_(ids)


class CommonMegaMoeExecutionTest(unittest.TestCase):
    def executor(self, recipe=MegaMoeExecutor, *, prepared=True):
        cfg = SimpleNamespace(
            layer_id=3,
            dim=128,
            moe_inter_dim=16,
            n_routed_experts=4,
            n_local_experts=2,
            n_activated_experts=2,
            ep_size=2,
            ep_rank=0,
            local_expert_start=0,
            max_tokens_per_rank=8,
            warmup_include_capacity=False,
            swiglu_limit=7.0,
            route_scale=1.5,
            mega_moe_policy=MegaMoeExecutionPolicy() if prepared else None,
        )
        with patch.object(recipe, "setup_weights"):
            executor = recipe(cfg, object(), {})
        executor._mega_l1_w = executor._mega_l2_w = torch.ones(2, 2)
        executor._mega_l1_sf = executor._mega_l2_sf = torch.ones(2, 2)
        executor._shared_l1_w = executor._shared_l2_w = torch.ones(2, 2)
        executor._shared_l1_sf = executor._shared_l2_sf = torch.ones(2, 2)
        executor._shared_mid_fp8 = executor._shared_mid_sf = torch.ones(8, 16)
        executor._num_shared_experts = int(executor.includes_shared_expert)
        executor._mega_group = object()
        executor._mega_buf = SimpleNamespace(
            num_max_tokens_per_rank=8,
            x=torch.empty(8, 128, dtype=torch.bfloat16),
            x_sf=torch.full((8, 1), -1, dtype=torch.int32),
            topk_idx=torch.empty(8, 2, dtype=torch.long),
            topk_weights=torch.empty(8, 2),
            shared_l1_acts_sf=torch.full((256, 1), -1, dtype=torch.int32),
        )
        executor._mega_y = torch.zeros(8, 128, dtype=torch.bfloat16)
        executor._input_packer = RecordingPacker()
        return executor

    def launch_context(self, executor):
        stack = ExitStack()
        kernel = stack.enter_context(
            patch("deep_gemm." + executor._kernel_name, create=True)
        )
        stack.enter_context(
            patch("deep_gemm.get_block_m_for_mega_moe", return_value=16, create=True)
        )
        stack.enter_context(
            patch(
                "deep_gemm.get_block_m_for_mega_moe_fp8", return_value=16, create=True
            )
        )
        stack.enter_context(patch.object(executor, "_maybe_pre_kernel_barrier"))
        stack.enter_context(patch(PACKAGE + ".mega_moe.sync_cuda_graph_warmup_ranks"))
        return stack, kernel

    def test_zero_and_nonzero_tokens_enter_every_recipe_collective(self):
        for recipe in RECIPES:
            for tokens in (0, 3):
                with self.subTest(recipe=recipe.__name__, tokens=tokens):
                    executor = self.executor(recipe)
                    x = torch.ones(tokens, 128, dtype=torch.bfloat16)
                    context, kernel = self.launch_context(executor)
                    with context:
                        result = executor(
                            x,
                            torch.ones(tokens, 2),
                            torch.zeros(tokens, 2, dtype=torch.long),
                        )
                    self.assertEqual(result.shape, x.shape)
                    kernel.assert_called_once()
                    self.assertEqual(kernel.call_args.args[0].shape[0], tokens)
                    self.assertEqual(executor._input_packer.calls, [tokens])
                    if recipe in SHARED_SCALES:
                        sf = executor._mega_buf.shared_l1_acts_sf
                        if tokens == 0:
                            self.assertEqual(torch.count_nonzero(sf).item(), 0)
                        else:
                            expected = torch.zeros_like(sf[:128])
                            expected.index_fill_(
                                0, shared_fp8_scale_row_indices(tokens, 16), 7
                            )
                            torch.testing.assert_close(sf[:128], expected)

    def test_prepacked_inputs_skip_packing_and_stage_shared_scales(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe.__name__):
                executor = self.executor(recipe)
                x, sf, ids, weights = executor.prepacked_input_views(3)
                self.assertEqual(x.data_ptr(), executor._mega_buf.x.data_ptr())
                sf.fill_(11)
                ids.fill_(1)
                weights.fill_(0.5)
                context, kernel = self.launch_context(executor)
                with context:
                    result = executor.forward_prepacked(
                        torch.zeros(3, 128, dtype=torch.bfloat16)
                    )
                kernel.assert_called_once()
                self.assertEqual(result.shape, (3, 128))
                self.assertEqual(executor._input_packer.calls, [])
                self.assertTrue(torch.all(ids == 1))
                if recipe in SHARED_SCALES:
                    expected = torch.zeros(128, 1, dtype=torch.int32)
                    expected.index_fill_(0, shared_fp8_scale_row_indices(3, 16), 11)
                    torch.testing.assert_close(
                        executor._mega_buf.shared_l1_acts_sf[:128], expected
                    )

    def test_capacity_fails_before_packing_or_entering_kernel(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe.__name__):
                executor = self.executor(recipe)
                with patch.object(executor, "_launch") as launch:
                    with self.assertRaisesRegex(RuntimeError, "input tokens=9 exceeds"):
                        executor(
                            torch.zeros(9, 128),
                            torch.ones(9, 2),
                            torch.zeros(9, 2, dtype=torch.long),
                        )
                launch.assert_not_called()
                self.assertEqual(executor._input_packer.calls, [])
                executor._mega_y = executor._mega_y[:2]
                with self.assertRaisesRegex(RuntimeError, "output buffer rows=2"):
                    executor.forward_prepacked(torch.zeros(3, 128))

    def test_recipe_arguments_preserve_model_numerical_policy(self):
        for recipe in RECIPES:
            for prepared in (False, True):
                with self.subTest(recipe=recipe.__name__, prepared=prepared):
                    executor = self.executor(recipe, prepared=prepared)
                    context, kernel = self.launch_context(executor)
                    with context:
                        executor.forward_prepacked(torch.zeros(1, 128))
                    kwargs = kernel.call_args.kwargs
                    self.assertEqual(kwargs["recipe"], (1, 1, 32))
                    self.assertEqual(
                        kwargs["activation_clamp"], None if prepared else 7.0
                    )
                    self.assertEqual(kwargs["fast_math"], not prepared)
                    if recipe is MegaMoeSEExecutor:
                        self.assertEqual(kwargs["shared_recipe"], (1, 128, 128))
                    if recipe in SHARED_SCALES:
                        self.assertIs(
                            kwargs["shared_l1_weights"][0], executor._shared_l1_w
                        )
                    if recipe in (MegaMoeFp8Backend, MegaMoeFp8SharedBackend):
                        self.assertTrue(kwargs["assume_all_topk_valid"])

    def test_warmup_cache_separates_policy_recipe_and_collective_group(self):
        ordinary = self.executor(prepared=False)
        prepared = self.executor()
        prepared._mega_group = ordinary._mega_group
        another_layer = self.executor()
        another_layer._mega_group = prepared._mega_group
        another_group = self.executor()
        fp8 = self.executor(MegaMoeFp8Backend)
        fp8._mega_group = prepared._mega_group
        executors = (ordinary, prepared, another_layer, another_group, fp8)
        with ExitStack() as stack:
            _MEGA_MOE_JIT_WARMED_KEYS.clear()
            self.addCleanup(_MEGA_MOE_JIT_WARMED_KEYS.clear)
            stack.enter_context(patch.dict(os.environ, {"MODEL_WARM_UP": "1"}))
            stack.enter_context(
                patch("torch.cuda.is_current_stream_capturing", return_value=False)
            )
            stack.enter_context(patch("deep_gemm.get_num_sms", return_value=132))
            stack.enter_context(
                patch("torch.distributed.is_initialized", return_value=False)
            )
            for executor in executors:
                executor._resolve_jit_warmup_token_counts = Mock(return_value=[1, 8])
                executor.warmup_jit = Mock()
                executor._maybe_warmup_jit_once()
        self.assertEqual([e.warmup_jit.call_count for e in executors], [1, 1, 0, 1, 1])

    def test_fp8_accepts_strided_native_ue8m0_scale_views(self):
        executor = self.executor(MegaMoeFp8Backend)
        stacked = torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2)
        up_scale, gate_scale = stacked.view(torch.float8_e8m0fnu).chunk(2, dim=1)
        self.assertFalse(up_scale.is_contiguous())
        self.assertFalse(gate_scale.is_contiguous())
        prepared_scales = []

        def prepare(scale, *args):
            prepared_scales.append(scale.view(torch.uint8).clone())
            return scale.view(torch.uint8).to(torch.int32)

        with patch(
            PACKAGE + ".mega_moe_fp8.prepare_fp8_weight_scale_for_deepgemm", new=prepare
        ), patch(
            "deep_gemm.transform_weights_for_mega_moe_fp8",
            side_effect=lambda l1, l2: (l1, l2),
            create=True,
        ), patch.object(
            executor, "_setup_buffer_and_warmup"
        ):
            executor.setup_weights_from_fp8(
                torch.empty(2, 16, 128, dtype=torch.float8_e4m3fn),
                gate_scale,
                torch.empty(2, 128, 16, dtype=torch.float8_e4m3fn),
                gate_scale,
                torch.empty(2, 16, 128, dtype=torch.float8_e4m3fn),
                up_scale,
            )
        expected = torch.cat((stacked[:, 2:], stacked[:, :2]), dim=1)
        torch.testing.assert_close(prepared_scales[0], expected)

    def test_native_setup_releases_checkpoint_stacks_before_arena_allocation(self):
        from rtp_llm.utils.model_weight import W

        executor = self.executor(prepared=False)
        executor.cfg.moe_w1_layout = "gate_up"
        weights = {
            W.moe_w1: torch.ones(2, 32, 64, dtype=torch.int8),
            W.moe_s1: torch.ones(2, 32, 4),
            W.moe_w2: torch.ones(2, 128, 8, dtype=torch.int8),
            W.moe_s2: torch.ones(2, 128, 1),
        }
        originals = {name: weakref.ref(value) for name, value in weights.items()}
        scale_calls = []

        def prepare(scale, *args):
            if scale_calls:
                self.assertIsNone(originals[W.moe_s1]())
            scale_calls.append(1)
            return scale.to(torch.int32)

        def transform(l1, l2):
            return tuple(t.clone() for t in l1), tuple(t.clone() for t in l2)

        def allocate():
            self.assertEqual(weights, {})
            self.assertTrue(all(ref() is None for ref in originals.values()))

        # Plain replacement functions avoid mocks retaining tensor call arguments.
        with patch(
            PACKAGE + ".mega_moe.prepare_fp4_weight_scale_for_deepgemm", new=prepare
        ), patch(
            "deep_gemm.transform_weights_for_mega_moe", new=transform
        ), patch.object(
            executor, "_setup_buffer_and_warmup", new=allocate
        ):
            executor.setup_weights(weights)
        self.assertEqual(len(scale_calls), 2)

    def test_native_shared_weights_keep_the_strict_ue8m0_contract(self):
        w1 = torch.empty(32, 128, dtype=torch.float8_e4m3fn)
        w2 = torch.empty(128, 16, dtype=torch.float8_e4m3fn)
        raw_scale = torch.ones(1, 1)
        packed_scale = torch.ones(1, 1, dtype=torch.int32)
        native = self.executor(MegaMoeSEExecutor, prepared=False)
        native.cfg.n_shared_experts = 1
        with self.assertRaisesRegex(TypeError, "expected shared UE8M0 scale"):
            native.setup_shared_expert_from_fp8(w1, raw_scale, w2, raw_scale)
        prepared = self.executor(MegaMoeSEExecutor)
        with patch(
            "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
            side_effect=lambda weight, scale: (weight, packed_scale),
        ) as requant, patch(
            "deep_gemm.transform_weights_for_mega_moe",
            side_effect=lambda l1, l2: (l1, l2),
        ), patch.object(
            prepared, "_setup_buffer_and_warmup"
        ):
            prepared.setup_shared_expert_from_fp8(w1, raw_scale, w2, raw_scale)
        self.assertEqual(requant.call_count, 2)
        self.assertEqual(prepared._num_shared_experts, 1)
        self.assertIs(prepared._shared_l1_sf, packed_scale)

    def test_fp4_layouts_produce_identical_kernel_weights(self):
        gate = (
            torch.arange(2 * 16 * 64, dtype=torch.int64)
            .reshape(2, 16, 64)
            .to(torch.int8)
        )
        up = (gate + 5).to(torch.int8)
        gate_sf = torch.arange(2 * 16 * 4, dtype=torch.int32).reshape(2, 16, 4)
        up_sf = gate_sf + 1000
        down = torch.ones(2, 128, 8, dtype=torch.int8)
        down_sf = torch.ones(2, 128, 1, dtype=torch.int32)

        def transform(l1, l2):
            weight, scale = l1

            def interleave(t):
                a, b = t.chunk(2, dim=1)
                return torch.stack(
                    (a.reshape(2, 2, 8, -1), b.reshape(2, 2, 8, -1)), dim=2
                ).reshape_as(t)

            return (interleave(weight), interleave(scale)), l2

        transformed = []
        for layout in ("gate_up", "up_gate"):
            executor = self.executor(prepared=layout == "up_gate")
            w = torch.cat((gate, up) if layout == "gate_up" else (up, gate), dim=1)
            sf = torch.cat(
                (gate_sf, up_sf) if layout == "gate_up" else (up_sf, gate_sf), dim=1
            )
            with patch(
                PACKAGE + ".mega_moe.prepare_fp4_weight_scale_for_deepgemm",
                side_effect=lambda t, *args: t,
            ), patch(
                "deep_gemm.transform_weights_for_mega_moe", side_effect=transform
            ), patch(
                "deep_gemm.mega._transpose_sf_for_utccp", side_effect=lambda t: t
            ), patch.object(
                executor, "_setup_buffer_and_warmup"
            ):
                executor.setup_weights_from_fp4(w, sf, down, down_sf, layout)
            transformed.append(
                (
                    executor._mega_l1_w,
                    executor._mega_l1_sf,
                    executor._mega_l2_w,
                    executor._mega_l2_sf,
                )
            )
        for left, right in zip(*transformed):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
