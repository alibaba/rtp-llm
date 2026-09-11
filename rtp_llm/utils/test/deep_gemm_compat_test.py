import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.utils import deep_gemm_compat as compat


def modern_mega(
    y,
    l1_weights,
    l2_weights,
    sym_buffer,
    shared_l1_weights=None,
    shared_l2_weights=None,
    recipe=(1, 1, 32),
    activation_clamp=None,
    round_swiglu_to_bf16=False,
):
    pass


def legacy_mega(
    y,
    l1_weights,
    l2_weights,
    sym_buffer,
    shared_l1_weights=None,
    shared_l2_weights=None,
    recipe=(1, 1, 32),
    activation_clamp=None,
    shared_recipe=(1, 128, 128),
):
    pass


def modern_buffer(group, mma_type="fp8xfp4", **kwargs):
    pass


def legacy_buffer(group, use_fp8_dispatch=True, **kwargs):
    pass


class DeepGemmCompatTest(unittest.TestCase):
    def module(self, modern=True):
        return SimpleNamespace(
            fp8_fp4_mega_moe=modern_mega if modern else legacy_mega,
            get_symm_buffer_for_mega_moe=modern_buffer if modern else legacy_buffer,
            _C=SimpleNamespace(
                get_token_alignment_for_mega_moe=lambda: 1920 if modern else 384,
                get_symm_buffer_size_for_mega_moe=mock.Mock(
                    return_value=(123456, lambda buffer: ())
                ),
            ),
        )

    def test_shared128_and_shared32_have_explicit_api_contracts(self):
        new, old = self.module(), self.module(modern=False)
        self.assertTrue(compat.mega_moe_uses_shared32(new))
        self.assertFalse(compat.mega_moe_uses_shared32(old))
        self.assertEqual(compat.mega_moe_shared_kwargs(new, 32), {})
        self.assertEqual(
            compat.mega_moe_shared_kwargs(old, 128), {"shared_recipe": (1, 128, 128)}
        )
        with self.assertRaisesRegex(RuntimeError, "independent shared128"):
            compat.mega_moe_shared_kwargs(new, 128)
        with self.assertRaises(RuntimeError):
            compat.mega_moe_shared_kwargs(old, 32)

    def test_only_shared32_explicitly_enables_bf16_activation_rounding(self):
        self.assertEqual(
            compat.mega_moe_activation_kwargs(self.module(), 32),
            {"round_swiglu_to_bf16": True},
        )
        self.assertEqual(compat.mega_moe_activation_kwargs(self.module(), 128), {})
        self.assertEqual(
            compat.mega_moe_activation_kwargs(self.module(modern=False), 128), {}
        )
        with self.assertRaisesRegex(RuntimeError, "BF16 activation-rounding patch"):
            compat.mega_moe_activation_kwargs(self.module(modern=False), 32)

    def test_legacy_routed_api_is_not_mistaken_for_new_shared_api(self):
        old = self.module(modern=False)
        old.fp8_fp4_mega_moe = lambda y, l1, l2, buf, recipe: None
        self.assertFalse(compat.mega_moe_uses_shared32(old))
        with self.assertRaisesRegex(RuntimeError, "no fused shared"):
            compat.mega_moe_shared_kwargs(old, 128)

    def test_combine_compatibility_is_explicit_and_keeps_old_modules_unchanged(self):
        def patched_mega(*arguments, torch_sum_combine=False):
            return torch_sum_combine

        patched = self.module()
        patched.fp8_fp4_mega_moe = patched_mega
        self.assertFalse(patched_mega())
        for block_size in (32, 128):
            with self.subTest(block_size=block_size):
                kwargs = compat.mega_moe_combine_kwargs(patched, block_size)
                self.assertEqual(kwargs, {"torch_sum_combine": True})
                self.assertTrue(patched_mega(**kwargs))
                self.assertEqual(
                    compat.mega_moe_combine_kwargs(self.module(), block_size), {}
                )
                self.assertEqual(
                    compat.mega_moe_combine_kwargs(self.module(False), block_size), {}
                )
        with self.assertRaisesRegex(ValueError, "unsupported shared"):
            compat.mega_moe_combine_kwargs(patched, 64)

    def test_modern_sizing_uses_string_shared_count_and_actual_alignment(self):
        module = self.module()
        for capacity, aligned in ((1, 1920), (1919, 1920), (1920, 1920), (1921, 3840)):
            for experts, topk, shared in ((384, 6, 0), (128, 3, 1)):
                with self.subTest(capacity=capacity, experts=experts, shared=shared):
                    size = compat.mega_moe_symm_buffer_bytes(
                        module,
                        8,
                        experts,
                        capacity,
                        topk,
                        5120,
                        2304,
                        num_shared_experts=shared,
                    )
                    self.assertEqual(size, 123456)
                    module._C.get_symm_buffer_size_for_mega_moe.assert_called_with(
                        8,
                        experts,
                        aligned,
                        topk,
                        5120,
                        2304,
                        "fp8xfp4",
                        "swiglu",
                        shared,
                    )

    def test_shared32_numerics_reject_rounding_only_wheel(self):
        with self.assertRaisesRegex(RuntimeError, "torch_sum_combine patch"):
            compat.mega_moe_numerics_kwargs(self.module(), 32)

    def test_numerics_keep_legacy_api_and_upstream_defaults(self):
        self.assertEqual(compat.mega_moe_numerics_kwargs(self.module(False), 128), {})
        self.assertEqual(compat.mega_moe_numerics_kwargs(self.module(), 128), {})

        def patched_mega(
            *arguments, round_swiglu_to_bf16=False, torch_sum_combine=False
        ):
            return round_swiglu_to_bf16, torch_sum_combine

        module = self.module()
        module.fp8_fp4_mega_moe = patched_mega
        self.assertEqual(patched_mega(), (False, False))
        self.assertEqual(
            patched_mega(**compat.mega_moe_numerics_kwargs(module, 32)),
            (True, True),
        )
        self.assertEqual(
            patched_mega(**compat.mega_moe_numerics_kwargs(module, 128)),
            (False, True),
        )
        self.assertEqual(patched_mega(), (False, False))

    def test_legacy_sizing_keeps_the_boolean_signature(self):
        module = self.module(modern=False)
        compat.mega_moe_symm_buffer_bytes(module, 8, 256, 1, 6, 7168, 2048)
        module._C.get_symm_buffer_size_for_mega_moe.assert_called_once_with(
            8, 256, 384, 6, 7168, 2048, True, "swiglu"
        )

    def test_sizing_and_allocation_errors_cannot_be_hidden_as_missing_estimates(self):
        module = self.module()
        module._C.get_symm_buffer_size_for_mega_moe.side_effect = RuntimeError(
            "wrong ABI"
        )
        with self.assertRaisesRegex(RuntimeError, "wrong ABI"):
            compat.mega_moe_symm_buffer_bytes(module, 8, 384, 1, 6, 5120, 2304)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            compat.validate_mega_moe_buffer_bytes(
                SimpleNamespace(buffer=torch.empty(17, dtype=torch.uint8)), 16
            )
        self.assertEqual(
            compat.validate_mega_moe_buffer_bytes(
                SimpleNamespace(buffer=torch.empty(16, dtype=torch.uint8)), 16
            ),
            16,
        )

    def test_group32_raw_exponents_are_decoded_and_rows_expand_losslessly(self):
        module = self.module()
        module.transform_sf_into_required_layout = mock.Mock(return_value="packed")
        raw = torch.tensor(
            [[126, 127, 128, 129], [130, 131, 132, 133]], dtype=torch.uint8
        )
        self.assertEqual(
            compat.prepare_mega_shared_scale(module, raw, 64, 128, 32), "packed"
        )
        values, mn, k, recipe = module.transform_sf_into_required_layout.call_args.args
        expected = torch.tensor(
            [[0.5, 1, 2, 4], [8, 16, 32, 64]], dtype=torch.float32
        ).repeat_interleave(32, dim=0)
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        self.assertEqual((mn, k, recipe), (64, 128, (1, 32)))

    def test_group32_numeric_and_expanded_scales_share_one_layout(self):
        module = self.module()
        module.transform_sf_into_required_layout = mock.Mock(return_value="packed")
        raw = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)
        compat.prepare_mega_shared_scale(module, raw, 32, 128, 32)
        compact_result = module.transform_sf_into_required_layout.call_args.args[
            0
        ].clone()
        compat.prepare_mega_shared_scale(
            module, raw.expand(32, -1).clone(), 32, 128, 32
        )
        torch.testing.assert_close(
            module.transform_sf_into_required_layout.call_args.args[0],
            compact_result,
            rtol=0,
            atol=0,
        )
        packed = torch.zeros((32, 1), dtype=torch.int32)
        self.assertIs(
            compat.prepare_mega_shared_scale(module, packed, 32, 128, 32), packed
        )
        with self.assertRaises(ValueError):
            compat.prepare_mega_shared_scale(
                module, torch.zeros((1, 1), dtype=torch.int32), 32, 128, 32
            )

    def test_legacy_shared128_scales_keep_their_recipe_and_values(self):
        module = self.module(modern=False)
        module.transform_sf_into_required_layout = mock.Mock(return_value="packed")
        raw = torch.tensor([[127, 128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        compat.prepare_mega_shared_scale(module, raw, 128, 256, 128)
        values, mn, k, recipe = module.transform_sf_into_required_layout.call_args.args
        torch.testing.assert_close(values, torch.tensor([[1.0, 2.0]]), rtol=0, atol=0)
        self.assertEqual((mn, k, recipe), (128, 256, (128, 128)))

    def test_shared32_rejects_scales_that_cannot_be_packed_losslessly(self):
        module = self.module()
        module.transform_sf_into_required_layout = mock.Mock()
        for value in (0.0, -1.0, 1.25, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "UE8M0"):
                compat.prepare_mega_shared_scale(
                    module, torch.full((1, 4), value), 32, 128, 32
                )
        module.transform_sf_into_required_layout.assert_not_called()

    def test_new_warmup_enumerates_installed_tiling_and_empty_rank(self):
        module = self.module()
        module.get_block_m_for_mega_moe = mock.Mock(
            side_effect=lambda ranks, experts, capacity, tokens, topk, mma: (
                16 if tokens < 3 else 240
            )
        )
        cfg = SimpleNamespace(
            ep_size=16,
            n_routed_experts=384,
            n_activated_experts=6,
            max_tokens_per_rank=6,
        )
        self.assertEqual(compat.mega_moe_jit_token_counts(module, cfg, 1920), [0, 3, 6])
        self.assertEqual(module.get_block_m_for_mega_moe.call_count, 7)
        module.get_block_m_for_mega_moe.assert_any_call(16, 384, 1920, 0, 6, "fp8xfp4")


if __name__ == "__main__":
    unittest.main()
