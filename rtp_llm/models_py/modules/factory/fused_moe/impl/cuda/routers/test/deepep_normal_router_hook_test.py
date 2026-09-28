"""DeepEP Normal router backend dispatch hook contract tests."""

from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
    DeepepNormalRouterBase,
)


class _TupleDispatchBuffer:
    def get_dispatch_layout(self, topk_ids, expert_num):
        del expert_num
        token_count = topk_ids.shape[0]
        counts = torch.tensor([token_count], dtype=torch.int32)
        return counts, None, counts, torch.ones(token_count, dtype=torch.bool), None

    def dispatch(self, expert_input, *args, **kwargs):
        del kwargs
        self.expert_input = expert_input
        topk_ids = args[5]
        topk_weights = args[6]
        token_count = (
            expert_input[0].shape[0]
            if isinstance(expert_input, tuple)
            else expert_input.shape[0]
        )
        return expert_input, topk_ids, topk_weights, [token_count], object(), None


class _TupleDispatchRouter(DeepepNormalRouterBase):
    def _prepare_dispatch_input(self, a1, slice_begin, slice_size, use_fp8):
        self.hook_called = True
        assert not use_fp8
        sliced = torch.narrow(a1, 0, slice_begin, slice_size)
        return torch.ones_like(sliced, dtype=torch.int8), torch.full(
            (slice_size, 1), 0.25, dtype=torch.float32
        )


class _InvalidFp8DispatchRouter(DeepepNormalRouterBase):
    def _prepare_dispatch_input(self, a1, slice_begin, slice_size, use_fp8):
        assert use_fp8
        return torch.narrow(a1, 0, slice_begin, slice_size)


class DeepepNormalRouterHookTest(TestCase):
    @staticmethod
    def _new_router(router_class, quant_dtype):
        router = object.__new__(router_class)
        router.config = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            model_config=SimpleNamespace(quant_config=None),
            quant_config=None,
        )
        router.quant_config = FusedMoEQuantConfig(
            quant_dtype=quant_dtype,
            per_act_token_quant=True,
            per_out_ch_quant=True,
        )
        buffer = _TupleDispatchBuffer()
        router.deepep_buffer_wrapper = SimpleNamespace(buffer=buffer)
        router.expert_num = 2
        router.rank_expert_offset = 0
        router.expert_alignment = 1
        router.handle = None
        return router, buffer

    def test_backend_hook_preserves_dispatched_tensor_metadata(self):
        router, buffer = self._new_router(_TupleDispatchRouter, torch.int8)
        router.hook_called = False

        a1 = torch.randn((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.zeros((3, 1), dtype=torch.int64)
        topk_weights = torch.ones((3, 1), dtype=torch.float32)
        payload = router.prepare(a1, None, None, topk_weights, topk_ids)

        self.assertTrue(router.hook_called)
        self.assertIsInstance(buffer.expert_input, tuple)
        self.assertEqual(payload.expert_x.dtype, torch.int8)
        self.assertIsNotNone(payload.expert_x_scale)
        self.assertEqual(payload.expert_x_scale.dtype, torch.float32)
        self.assertEqual(payload.expert_x_scale.shape, (3, 1))
        self.assertEqual(payload.expert_x_origin_dtype, torch.bfloat16)

    def test_fp8_dispatch_requires_activation_scale_tuple(self):
        router, _ = self._new_router(
            _InvalidFp8DispatchRouter, torch.float8_e4m3fn
        )
        a1 = torch.randn((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.zeros((3, 1), dtype=torch.int64)
        topk_weights = torch.ones((3, 1), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "must return"):
            router.prepare(a1, None, None, topk_weights, topk_ids)


if __name__ == "__main__":
    main()
