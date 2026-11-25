"""Contract checks for GC's routed weight ordering and graph shapes."""

import os
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.glm5_gc_moe.gc_moe import (
    gc_warmup_token_counts,
    transform_gc_routed_fp4,
)


def _import_gc_wheel():
    # This file's directory holds gc_moe.py, which shadows the wheel when the
    # test runs as __main__; promote the wheel's site-packages to the front.
    for entry in sys.path:
        root = entry or os.getcwd()
        if os.path.isfile(os.path.join(root, "gc_moe", "__init__.py")):
            sys.path.insert(0, root)
            import gc_moe

            return gc_moe
    raise ImportError("gc_moe wheel not found on sys.path")


class GCMoeContractTest(unittest.TestCase):
    def test_tile32_requires_adaptive_scope_and_exact_compatible_shape(self):
        from rtp_llm.models_py.modules.glm5_gc_moe.gc_moe_se import (
            _eligible_tile32,
            adaptive_gc_moe_se_tile32,
        )

        x = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            shape=(56, 6144),
        )
        cfg = SimpleNamespace(
            ep_size=4,
            n_routed_experts=256,
            n_activated_experts=8,
        )
        buf = SimpleNamespace(
            hidden=6144,
            intermediate_hidden=2048,
            group=SimpleNamespace(size=lambda: 4),
            num_experts=256,
            num_topk=8,
            mma_type="fp8xfp4",
            num_shared_experts=1,
            _for_call=lambda n: None,
        )
        self.assertFalse(_eligible_tile32(x, cfg, buf, 56, 16))
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(RuntimeError, "restore scope"):
                with adaptive_gc_moe_se_tile32():
                    self.assertTrue(_eligible_tile32(x, cfg, buf, 56, 16))
                    self.assertFalse(_eligible_tile32(x, cfg, buf, 56, 32))
                    self.assertFalse(_eligible_tile32(x, cfg, buf, 57, 16))
                    x.shape = (64, 6144)
                    self.assertTrue(_eligible_tile32(x, cfg, buf, 64, 16))
                    buf.num_shared_experts = 0
                    self.assertFalse(_eligible_tile32(x, cfg, buf, 64, 16))
                    raise RuntimeError("restore scope")
        buf.num_shared_experts = 1
        self.assertFalse(_eligible_tile32(x, cfg, buf, 64, 16))

    def test_tile32_packer_and_launch_use_same_block(self):
        from rtp_llm.models_py.modules.glm5_gc_moe import gc_moe_se

        @dataclass(frozen=True)
        class Call:
            block_m: int
            shared_experts: int

        x = torch.empty((56, 6144), dtype=torch.bfloat16)
        buf = SimpleNamespace(num_max_tokens_per_rank=84, _for_call=Mock())
        buf._for_call.return_value = buf
        subject = SimpleNamespace(
            _check_shared_expert_ready=lambda: None,
            _mega_buf=buf,
            _mega_y=x,
            cfg=SimpleNamespace(
                ep_size=4,
                n_routed_experts=256,
                n_activated_experts=8,
                layer_id=0,
            ),
            _input_packer=Mock(),
            _mega_l1_w=object(),
            _mega_l1_sf=object(),
            _mega_l2_w=object(),
            _mega_l2_sf=object(),
            _shared_l1_w=object(),
            _shared_l1_sf=object(),
            _shared_l2_w=object(),
            _shared_l2_sf=object(),
        )
        wheel = ModuleType("gc_moe")
        wheel.get_block_m_for_gc_moe = Mock(return_value=16)
        wheel.fp8_fp4_gc_moe = Mock()
        validate = Mock(return_value=Call(16, 1))
        launch = Mock()
        with (
            patch.dict(sys.modules, {"gc_moe": wheel}),
            patch.object(gc_moe_se, "_eligible_tile32", return_value=True),
            patch.object(gc_moe_se, "_tile32_api", return_value=(validate, launch)),
            patch.object(gc_moe_se, "_sync_cuda_graph_warmup_ranks"),
        ):
            weights, indices = object(), object()
            gc_moe_se.GLM5GCMoESE._forward_impl(
                subject,
                x,
                weights,
                indices,
                inputs_prepacked=False,
            )
        subject._input_packer.pack.assert_called_once_with(
            x, weights, indices, buf, 56, 32
        )
        validate.assert_called_once()
        buf._for_call.assert_called_once_with(1)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], Call(32, 1))
        wheel.fp8_fp4_gc_moe.assert_not_called()

    def test_tile32_adapter_falls_back_for_unknown_wheel(self):
        from rtp_llm.models_py.modules.glm5_gc_moe import gc_moe_se

        wheel = ModuleType("gc_moe")
        with patch.object(gc_moe_se.metadata, "version", return_value="0.7"):
            try:
                self.assertIsNone(gc_moe_se._tile32_api(wheel))
            finally:
                gc_moe_se._tile32_api.cache_clear()

    def test_up_gate_checkpoint_uses_gc_transform(self):
        gc_moe = _import_gc_wheel()

        device = "cuda"
        w1 = (
            torch.arange(2 * 256 * 128, dtype=torch.int32, device=device)
            .to(torch.int8)
            .reshape(2, 256, 128)
        )
        s1 = torch.empty_strided(
            (2, 256, 2), (512, 1, 256), dtype=torch.int32, device=device
        )
        s1.copy_(
            torch.arange(2 * 256 * 2, dtype=torch.int32, device=device).reshape(
                2, 256, 2
            )
        )
        w2 = (
            torch.arange(2 * 256 * 64, dtype=torch.int32, device=device)
            .to(torch.int8)
            .reshape(2, 256, 64)
        )
        s2 = torch.empty_strided(
            (2, 256, 1), (256, 1, 256), dtype=torch.int32, device=device
        )
        s2.copy_(
            torch.arange(2 * 256, dtype=torch.int32, device=device).reshape(2, 256, 1)
        )
        actual = transform_gc_routed_fp4((w1, s1), (w2, s2), "up_gate")
        expected_s1 = torch.empty_strided(
            s1.shape, s1.stride(), dtype=s1.dtype, device=device
        )
        expected_s1[:, :128].copy_(s1[:, 128:])
        expected_s1[:, 128:].copy_(s1[:, :128])
        expected = gc_moe.transform_weights_for_gc_moe(
            (
                torch.cat((w1[:, 128:], w1[:, :128]), dim=1),
                expected_s1,
            ),
            (w2, s2),
        )
        self.assertEqual(actual[0][1].stride(), s1.stride())
        self.assertEqual(actual[1][1].stride(), s2.stride())
        for pair_a, pair_b in zip(actual, expected):
            for a, b in zip(pair_a, pair_b):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_eager_warmup_covers_decode_and_eagle_shapes(self):
        self.assertEqual(
            gc_warmup_token_counts(64),
            [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 64],
        )
        self.assertEqual(gc_warmup_token_counts(5), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
