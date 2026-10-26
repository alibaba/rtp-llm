import os
import tempfile
import types
import unittest
from unittest import mock

import torch


class _K3CpuPacker:
    name = "reference"

    def pack(self, x, routing, ids, buf, tokens):
        buf.x[:tokens].copy_(x)
        buf.topk_weights[:tokens].copy_(routing)
        buf.topk_idx[:tokens].copy_(ids)


class K3FusedSharedJitWarmupTest(unittest.TestCase):
    def setUp(self):
        import deep_gemm

        from rtp_llm.models_py.modules.kimi_k3 import moe

        self.k3_dg, self.k3_moe = deep_gemm, moe
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patches = [
            mock.patch.dict(
                os.environ,
                {
                    "MODEL_WARM_UP": "1",
                    "MEGA_MOE_JIT_WARMUP_TOKENS": "1,7,32",
                    "MEGA_MOE_NVCC_TMPDIR": temporary.name,
                    "TMPDIR": "/before",
                },
            ),
            mock.patch.object(moe, "_FUSED_SHARED_JIT_WARMED_KEYS", set()),
            mock.patch.object(deep_gemm, "get_num_sms", return_value=148),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.distributed.barrier"),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _k3_experts(self, *, capacity=32, shared_hidden=256):
        experts = self.k3_moe._FusedSharedExperts.__new__(
            self.k3_moe._FusedSharedExperts
        )
        torch.nn.Module.__init__(experts)
        experts.cfg = types.SimpleNamespace(
            ep_size=2,
            n_routed_experts=16,
            n_local_experts=8,
            n_activated_experts=2,
            local_expert_start=8,
            dim=128,
            moe_inter_dim=256,
            warmup_include_capacity=False,
            layer_id=3,
            activation_beta=4.0,
            activation_linear_beta=25.0,
        )
        experts._max_tokens_per_rank = capacity
        experts._shared_intermediate = 256
        experts.buf = types.SimpleNamespace(
            num_max_tokens_per_rank=capacity,
            x=torch.empty(capacity, 128, dtype=torch.bfloat16),
            x_sf=torch.empty(capacity, 1),
            topk_weights=torch.empty(capacity, 2),
            topk_idx=torch.empty(capacity, 2, dtype=torch.long),
        )
        experts.routed_y = torch.empty(capacity, 128, dtype=torch.bfloat16)
        experts.shared_x = torch.empty(capacity, shared_hidden, dtype=torch.bfloat16)
        experts.shared_y = torch.empty_like(experts.shared_x)
        experts.routed_l1, experts.routed_l2 = object(), object()
        experts.shared_l1, experts.shared_l2 = object(), object()
        experts.kernel = lambda *args, **kwargs: self.k3_dg.fp8_fp4_mega_moe_k3(
            *args, **kwargs
        )
        experts.activation_kwargs = {
            "activation_beta": experts.cfg.activation_beta,
            "activation_linear_beta": experts.cfg.activation_linear_beta,
        }
        experts.packer = _K3CpuPacker()
        experts.fp4_block = 32
        return experts

    def test_k3_shared_forward_deduplicates_matching_geometry(self):
        calls = []

        def launch(y, l1, l2, buf, **kwargs):
            tokens = y.shape[0]
            self.assertEqual(kwargs["activation"], "situ")
            self.assertEqual(kwargs["activation_beta"], 4.0)
            self.assertEqual(kwargs["activation_linear_beta"], 25.0)
            self.assertEqual(kwargs["recipe"], (1, 1, 32))
            self.assertEqual(tuple(buf.x[:tokens].shape), (tokens, 128))
            self.assertTrue(
                torch.equal(
                    buf.topk_idx[:tokens], torch.tensor([8, 9]).expand(tokens, -1)
                )
            )
            self.assertEqual(torch.count_nonzero(buf.topk_weights[:tokens]).item(), 0)
            y.zero_()
            kwargs["shared_y"][:tokens].copy_(kwargs["shared_x"][:tokens])
            calls.append((tokens, kwargs["shared_y"].shape[1]))

        # Keep the K3 tensor preparation and forward real; replace the external launch.
        with mock.patch.object(self.k3_dg, "fp8_fp4_mega_moe_k3", side_effect=launch):
            self._k3_experts()._maybe_warmup_jit_once()
            count = len(calls)
            self.assertGreater(count, 0)
            self._k3_experts()._maybe_warmup_jit_once()
            self.assertEqual(len(calls), count)
            self._k3_experts(shared_hidden=512)._maybe_warmup_jit_once()
            self.assertGreater(len(calls), count)
        self.assertEqual(
            set(calls),
            {
                (1, 256),
                (7, 256),
                (32, 256),
                (1, 512),
                (7, 512),
                (32, 512),
            },
        )

    def test_k3_fused_pack_prepares_both_valid_mask_layouts(self):
        from rtp_llm.models_py.triton_kernels.moe import mega_moe_input_pack

        experts = self._k3_experts()
        experts.packer = types.SimpleNamespace(name="fused")
        masks = []

        def pack(x, routing, ids, out, scales, out_ids, out_routing, **kwargs):
            masks.append(kwargs["valid_mask"] is not None)
            out.copy_(x)
            out_ids.copy_(ids)
            out_routing.copy_(routing)
            kwargs["shared_out"].copy_(kwargs["shared_input"])

        with (
            mock.patch.object(
                mega_moe_input_pack, "fused_pack_mega_moe_inputs", side_effect=pack
            ),
            mock.patch.object(self.k3_dg, "fp8_fp4_mega_moe_k3"),
        ):
            experts.warmup_jit([1, 7])
        self.assertEqual(masks, [False, True, False, True])

    def test_k3_automatic_plan_includes_shared_pack_block_boundary(self):
        experts = self._k3_experts(capacity=2048)
        experts.packer = types.SimpleNamespace(name="fused")
        with mock.patch.dict(os.environ, clear=True):
            tokens = experts._resolve_jit_warmup_token_counts(148)
        self.assertIn(2048, tokens)

    def test_k3_padding_routes_match_shared_packer_fallback(self):
        experts = self._k3_experts(capacity=3)
        hidden = torch.randn(3, 256, dtype=torch.bfloat16)
        routed_input = torch.randn(3, 128, dtype=torch.bfloat16)
        routing = torch.tensor([[0, 0.25], [0.75, 0.25], [0.75, 0.25]])
        ids = torch.tensor([[0, 3], [2, 4], [0, 5]])
        for valid in ([True, False, True], [False] * 3, [True] * 3):
            with self.subTest(valid=valid):
                valid_mask = torch.tensor(valid)
                expected_ids = torch.where(valid_mask[:, None], ids, -1)
                expected_weights = torch.where(valid_mask[:, None], routing, 0)
                expected_shared = torch.where(valid_mask[:, None], hidden, 0)
                with mock.patch.object(self.k3_dg, "fp8_fp4_mega_moe_k3") as launch:
                    experts(hidden, routed_input, routing, ids, valid_mask)
                launch.assert_called_once()
                torch.testing.assert_close(
                    experts.buf.topk_idx, expected_ids, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    experts.buf.topk_weights, expected_weights, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    experts.shared_x, expected_shared, rtol=0, atol=0
                )
                masked_weights, masked_ids = self.k3_moe.KimiK3LatentMoE._mask_routes(
                    routing, ids, valid_mask
                )
                torch.testing.assert_close(masked_ids, expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(
                    masked_weights, expected_weights, rtol=0, atol=0
                )

    def test_k3_failed_compile_restores_tmpdir_without_marking_prepared(self):
        with mock.patch.object(
            self.k3_dg,
            "fp8_fp4_mega_moe_k3",
            side_effect=RuntimeError("compiler failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                self._k3_experts()._maybe_warmup_jit_once()
        self.assertEqual(os.environ["TMPDIR"], "/before")
        self.assertFalse(self.k3_moe._FUSED_SHARED_JIT_WARMED_KEYS)


if __name__ == "__main__":
    unittest.main()
