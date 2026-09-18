import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    Qwen4ExpQSAIndexer,
    build_base_rope,
    build_interleaved_mrope,
    build_qsa_rope,
    is_qsa_rope_style,
)


class IndexerRuntimeContractTest(unittest.TestCase):
    def test_interleaved_mrope_matches_independent_formula(self):
        config = SimpleNamespace(
            style="Mrope",
            index_factor=3,
            dim=8,
            mrope_dim1=2,
            mrope_dim2=1,
            mrope_dim3=1,
            mrope_interleaved=True,
            base=10000,
            scale=2.0,
        )
        position_ids = torch.tensor([2, 5, 7, 3, 11, 13], dtype=torch.int32)

        cos, sin = build_interleaved_mrope(
            position_ids,
            config,
            token_count=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        axes = torch.tensor([0, 1, 2, 0])
        positions = position_ids.view(2, 3)[:, axes].float()
        inv_freq = 10000 ** (-2 * torch.arange(4).float() / 8)
        angle = positions.div(2.0) * inv_freq
        expected_cos = torch.cat([angle.cos(), angle.cos()], dim=-1)
        expected_sin = torch.cat([angle.sin(), angle.sin()], dim=-1)
        torch.testing.assert_close(cos, expected_cos)
        torch.testing.assert_close(sin, expected_sin)

        dispatched_cos, dispatched_sin = build_qsa_rope(
            position_ids,
            config,
            token_count=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(dispatched_cos, cos)
        torch.testing.assert_close(dispatched_sin, sin)

    def test_mrope_rejects_misaligned_position_ids(self):
        config = SimpleNamespace(
            style="Mrope",
            index_factor=3,
            dim=8,
            mrope_dim1=2,
            mrope_dim2=1,
            mrope_dim3=1,
            mrope_interleaved=True,
            base=10000,
            scale=1.0,
        )
        with self.assertRaisesRegex(ValueError, "expected 6"):
            build_interleaved_mrope(
                torch.arange(5),
                config,
                token_count=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_base_rope_factor_one_and_three_match_reference(self):
        logical_positions = torch.tensor([2, 5], dtype=torch.int32)

        def _build(index_factor, position_ids):
            config = SimpleNamespace(
                style="Base",
                index_factor=index_factor,
                dim=8,
                base=10_000,
                scale=2.0,
            )
            return build_qsa_rope(
                position_ids,
                config,
                token_count=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
                logical_positions=logical_positions,
            )

        factor_one = _build(1, logical_positions)
        factor_three = _build(
            3, logical_positions.unsqueeze(1).expand(-1, 3).reshape(-1)
        )
        inv_freq = 10_000 ** (-2 * torch.arange(4).float() / 8)
        angle = logical_positions.float().div(2.0).unsqueeze(1) * inv_freq
        expected = (
            torch.cat([angle.cos(), angle.cos()], dim=-1),
            torch.cat([angle.sin(), angle.sin()], dim=-1),
        )
        for actual, reference in zip(factor_one, expected):
            torch.testing.assert_close(actual, reference)
        for actual, reference in zip(factor_three, expected):
            torch.testing.assert_close(actual, reference)

    def test_base_rope_factor_three_rejects_non_text_positions(self):
        config = SimpleNamespace(
            style="Base",
            index_factor=3,
            dim=8,
            base=10_000,
            scale=1.0,
        )
        with self.assertRaisesRegex(ValueError, "draft text-only"):
            build_base_rope(
                torch.tensor([3, 4, 3], dtype=torch.int32),
                config,
                token_count=1,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_base_rope_rejects_position_that_differs_from_logical_cache(self):
        config = SimpleNamespace(
            style="Base",
            index_factor=1,
            dim=8,
            base=10_000,
            scale=1.0,
        )
        with self.assertRaisesRegex(ValueError, "logical cache positions"):
            build_qsa_rope(
                torch.tensor([4], dtype=torch.int32),
                config,
                token_count=1,
                dtype=torch.float32,
                device=torch.device("cpu"),
                logical_positions=torch.tensor([3], dtype=torch.int32),
            )

    def test_style_predicate_normalizes_enum_like_string_and_integer(self):
        enum_like = SimpleNamespace(name="Base")
        for style in (enum_like, "Base", "RopeStyle.Base", 1):
            with self.subTest(style=style):
                self.assertTrue(is_qsa_rope_style(SimpleNamespace(style=style), "Base"))
                self.assertFalse(
                    is_qsa_rope_style(SimpleNamespace(style=style), "Mrope")
                )

    def test_mrope_rejects_non_finite_or_non_positive_scale(self):
        position_ids = torch.tensor([0, 0, 0], dtype=torch.int32)
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            config = SimpleNamespace(
                style="Mrope",
                index_factor=3,
                dim=8,
                mrope_dim1=2,
                mrope_dim2=1,
                mrope_dim3=1,
                mrope_interleaved=True,
                base=10_000,
                scale=scale,
            )
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    build_qsa_rope(
                        position_ids,
                        config,
                        token_count=1,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )

    def test_direct_builders_reject_non_neox_indexer_layout(self):
        base_config = SimpleNamespace(
            style="Base",
            index_factor=1,
            dim=8,
            base=10_000,
            scale=1.0,
            indexer_is_neox_style=False,
        )
        mrope_config = SimpleNamespace(
            style="Mrope",
            index_factor=3,
            dim=8,
            mrope_dim1=2,
            mrope_dim2=1,
            mrope_dim3=1,
            mrope_interleaved=True,
            base=10_000,
            scale=1.0,
            indexer_is_neox_style=False,
        )
        cases = (
            (build_base_rope, torch.tensor([0]), base_config),
            (build_interleaved_mrope, torch.tensor([0, 0, 0]), mrope_config),
            (build_qsa_rope, torch.tensor([0]), base_config),
        )
        for builder, position_ids, config in cases:
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(ValueError, "indexer_is_neox_style=true"):
                    builder(
                        position_ids,
                        config,
                        token_count=1,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )

    def test_project_once_selection_matches_public_forward(self):
        torch.manual_seed(5)
        indexer = Qwen4ExpQSAIndexer(
            torch.randn(40, 16),
            torch.randn(8),
            torch.randn(8),
            n_heads=4,
            kv_heads=1,
            head_dim=8,
            token_budget=8,
            compress_ratio=4,
            norm_eps=1e-6,
        )
        hidden = torch.randn(2, 9, 16)
        angles = torch.randn(2, 9, 4)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
        q, raw_keys = indexer.project(hidden)

        expected = indexer.forward_causal(hidden, cos, sin)
        got = indexer.forward_causal_from_projected(q, raw_keys, cos, sin)

        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
