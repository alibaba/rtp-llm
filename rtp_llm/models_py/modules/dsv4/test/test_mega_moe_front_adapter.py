import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.moe.mega_front import (
    MegaMoeFrontAdapter,
    _decode_capture_tokens,
)


class _FakePlan:
    def __init__(self) -> None:
        self.calls = []

    def run_learned_out(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        normalized = args[8]
        post = args[14]
        comb = args[15]
        normalized[:2].fill_(3)
        post[:2].fill_(4)
        comb[:2].fill_(5)


class _FakeStrategy:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.launches = []

    def _block_m(self, tokens: int) -> int:
        self.launches.append(("block_m", tokens))
        return 16

    def forward_prepacked(self, tokens: int, device: torch.device) -> torch.Tensor:
        self.launches.append(("launch", tokens, device))
        return torch.full((tokens, self.dim), 7, dtype=torch.bfloat16)


class _FakeGate:
    hash = False
    route_scale = 2.5


class _FakeHC:
    hc_eps = 1.0e-6


class _FakeNorm:
    variance_epsilon = 1.0e-6


def _fake_adapter(dim: int = 128) -> tuple[MegaMoeFrontAdapter, _FakePlan]:
    adapter = MegaMoeFrontAdapter.__new__(MegaMoeFrontAdapter)
    adapter.layer_id = 3
    adapter.dim = dim
    adapter.strategy = _FakeStrategy(dim)
    adapter.gate = _FakeGate()
    adapter.ffn_hc = _FakeHC()
    adapter.ffn_norm = _FakeNorm()
    adapter.hidden = torch.empty((128, 4, dim), dtype=torch.bfloat16)
    adapter.collapsed = torch.empty((128, dim), dtype=torch.bfloat16)
    adapter.collapse_ssq = torch.empty((128,), dtype=torch.float32)
    adapter.normalized_mix = torch.empty((128, 24), dtype=torch.float32)
    adapter.normalized = torch.empty((128, dim), dtype=torch.bfloat16)
    adapter.x_fp8 = torch.empty((128, dim), dtype=torch.float8_e4m3fn)
    adapter.x_sf = torch.empty((128, 1), dtype=torch.int32)
    adapter.shared_l1_x_sf = torch.empty_strided((128, 1), (1, 128), dtype=torch.int32)
    adapter.router_logits = torch.empty((128, 256), dtype=torch.float32)
    adapter.topk_ids = torch.empty((128, 6), dtype=torch.int64)
    adapter.topk_weights = torch.empty((128, 6), dtype=torch.float32)
    adapter.post = torch.empty((128, 4), dtype=torch.float32)
    adapter.comb = torch.empty((128, 4, 4), dtype=torch.float32)
    adapter.hc_base = torch.empty((24,), dtype=torch.float32)
    adapter.hc_scale = torch.empty((3,), dtype=torch.float32)
    adapter.ffn_norm_weight = torch.empty((dim,), dtype=torch.bfloat16)
    adapter.router_weight = torch.empty((256, dim), dtype=torch.bfloat16)
    adapter.correction_bias = torch.empty((256,), dtype=torch.float32)
    adapter.input_ids = None
    adapter.tid2eid = None
    plan = _FakePlan()
    adapter._plans = {2: plan}
    return adapter, plan


class MegaMoeFrontAdapterTest(unittest.TestCase):
    def test_front_is_attached_when_explicitly_enabled_for_mega_se(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(_strategy=SimpleNamespace(name="mega_se")),
            ffn_hc="hc",
            ffn_norm="norm",
            _mega_front_adapter=None,
        )
        adapter = object()
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_front.MegaMoeFrontAdapter",
            return_value=adapter,
        ) as adapter_cls:
            Block.enable_mega_front(block)

        self.assertIs(block._mega_front_adapter, adapter)
        adapter_cls.assert_called_once_with(block.ffn, "hc", "norm")

    def test_front_is_not_attached_to_non_mega_strategy(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(_strategy=SimpleNamespace(name="local_loop")),
            ffn_hc="hc",
            ffn_norm="norm",
            _mega_front_adapter=None,
        )
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_front.MegaMoeFrontAdapter"
        ) as adapter_cls:
            Block.enable_mega_front(block)

        self.assertIsNone(block._mega_front_adapter)
        adapter_cls.assert_not_called()

    def test_required_front_rejects_non_mega_se_strategy(self) -> None:
        block = SimpleNamespace(
            ffn=SimpleNamespace(_strategy=SimpleNamespace(name="mega")),
            ffn_hc="hc",
            ffn_norm="norm",
            _mega_front_adapter=None,
        )

        with self.assertRaisesRegex(RuntimeError, "requires the mega_se MoE strategy"):
            Block.enable_mega_front(block, required=True)

    def test_capture_tokens_are_sorted_and_bounded(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DECODE_CAPTURE_CONFIG": "8, 1,8,32", "GEN_NUM_PER_CIRCLE": "0"},
        ):
            self.assertEqual(_decode_capture_tokens(), (1, 8, 32))
        with mock.patch.dict(
            os.environ,
            {"DECODE_CAPTURE_CONFIG": "8,256", "GEN_NUM_PER_CIRCLE": "0"},
        ):
            with self.assertRaisesRegex(RuntimeError, "supports capture token counts"):
                _decode_capture_tokens()

    def test_capture_tokens_include_dspark_and_target_verify(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DECODE_CAPTURE_CONFIG": "8,16,32", "GEN_NUM_PER_CIRCLE": "3"},
        ):
            self.assertEqual(_decode_capture_tokens(), (8, 16, 24, 32, 48, 64, 96, 128))

    def test_front_support_is_bounded_by_extension_capacity(self) -> None:
        adapter, _ = _fake_adapter()

        self.assertTrue(adapter.supports(torch.empty(64, 2, 4, adapter.dim)))
        self.assertFalse(adapter.supports(torch.empty(43, 3, 4, adapter.dim)))

    def test_learned_front_stages_and_launches_prepacked_mega(self) -> None:
        adapter, plan = _fake_adapter()
        residual = torch.arange(2 * 4 * adapter.dim, dtype=torch.float32)
        residual = residual.to(torch.bfloat16).view(2, 1, 4, adapter.dim)
        input_ids = torch.tensor([[11], [12]], dtype=torch.int64)

        y, normalized, post, comb = adapter.forward(residual, input_ids)

        self.assertTrue(torch.equal(adapter.hidden[:2], residual.view(2, 4, -1)))
        self.assertEqual(tuple(y.shape), (2, 1, adapter.dim))
        self.assertEqual(tuple(normalized.shape), (2, 1, adapter.dim))
        self.assertEqual(tuple(post.shape), (2, 1, 4, 1))
        self.assertEqual(tuple(comb.shape), (2, 1, 4, 4))
        self.assertTrue(torch.all(y == 7))
        self.assertTrue(torch.all(normalized == 3))
        self.assertTrue(torch.all(post == 4))
        self.assertTrue(torch.all(comb == 5))
        self.assertEqual(adapter.strategy.launches[0], ("block_m", 2))
        self.assertEqual(adapter.strategy.launches[1][:2], ("launch", 2))
        self.assertEqual(len(plan.calls), 1)
        args, kwargs = plan.calls[0]
        self.assertEqual(args[16], 16)
        self.assertIs(kwargs["router_logits"], adapter.router_logits)

    def test_empty_rank_skips_front_and_enters_mega_collective(self) -> None:
        adapter, plan = _fake_adapter()
        residual = torch.empty((0, 1, 4, adapter.dim), dtype=torch.bfloat16)
        input_ids = torch.empty((0, 1), dtype=torch.int64)

        y, normalized, post, comb = adapter.forward(residual, input_ids)

        self.assertEqual(tuple(y.shape), (0, 1, adapter.dim))
        self.assertEqual(tuple(normalized.shape), (0, 1, adapter.dim))
        self.assertEqual(tuple(post.shape), (0, 1, 4, 1))
        self.assertEqual(tuple(comb.shape), (0, 1, 4, 4))
        self.assertEqual(adapter.strategy.launches[0][:2], ("launch", 0))
        self.assertEqual(plan.calls, [])


if __name__ == "__main__":
    unittest.main()
