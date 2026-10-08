"""Unit tests for per_channel_fp8_quant_weight helper functions.

Covers (PR #882 review-requested):
  - _identity_ensure_2d: 1D / 2D / empty scale handling
  - _ckpt_base_matches_quant_exclude: literal / templated / empty exclude paths
"""

import unittest

import torch

from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    _ckpt_base_matches_quant_exclude,
    _ckpt_base_matches_regex_exclude,
    _identity_ensure_2d,
)


class IdentityEnsure2dTest(unittest.TestCase):
    def test_1d_scale_unsqueezed_to_2d(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = _identity_ensure_2d([t])
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(torch.equal(out, t.unsqueeze(-1)))

    def test_2d_scale_passthrough(self):
        t = torch.tensor([[1.0], [2.0], [3.0]])
        out = _identity_ensure_2d([t])
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.equal(out, t))

    def test_higher_rank_passthrough(self):
        t = torch.zeros(4, 8, 2)
        out = _identity_ensure_2d([t])
        self.assertEqual(out.shape, (4, 8, 2))

    def test_empty_list_with_allow_empty_returns_none(self):
        self.assertIsNone(_identity_ensure_2d([], allow_empty=True))

    def test_empty_list_without_allow_empty_raises(self):
        with self.assertRaises(Exception):
            _identity_ensure_2d([])

    def test_takes_first_tensor_when_multiple(self):
        first = torch.tensor([1.0, 2.0])
        second = torch.tensor([99.0, 99.0])
        out = _identity_ensure_2d([first, second])
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.equal(out, first.unsqueeze(-1)))


class CkptBaseMatchesQuantExcludeTest(unittest.TestCase):
    def test_empty_exclude_returns_false(self):
        self.assertFalse(_ckpt_base_matches_quant_exclude("model.layers.0.mlp", set()))

    def test_literal_exact_match(self):
        excludes = {"lm_head", "model.embed_tokens"}
        self.assertTrue(_ckpt_base_matches_quant_exclude("lm_head", excludes))
        self.assertTrue(
            _ckpt_base_matches_quant_exclude("model.embed_tokens", excludes)
        )

    def test_literal_no_match(self):
        excludes = {"lm_head"}
        self.assertFalse(_ckpt_base_matches_quant_exclude("model.norm", excludes))

    def test_no_template_no_match_returns_false(self):
        # Without {i} placeholder, only literal equality is checked.
        excludes = {"model.layers.0.mlp"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.0.attn", excludes)
        )

    def test_templated_match_against_concrete_layers(self):
        # Template should match any layer index.
        excludes = {"model.layers.0.mlp", "model.layers.13.mlp"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )

    def test_templated_match_against_unrelated_path(self):
        excludes = {"lm_head", "model.embed_tokens"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )

    def test_templated_only_matches_digits(self):
        # The {i} placeholder is replaced by \d+, so non-digit text must not match.
        excludes = {"model.layers.foo.mlp"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )

    def test_templated_anchored_at_both_ends(self):
        # Pattern must match the whole string, not be a substring.
        excludes = {"prefix.model.layers.0.mlp"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )
        excludes_suffix = {"model.layers.0.mlp.suffix"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes_suffix)
        )

    def test_special_regex_chars_in_template_escaped(self):
        # Dots in the template should not act as regex wildcards.
        excludes = {"modelXlayersX0Xmlp"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )

    def test_template_with_multiple_placeholders(self):
        # Multiple {i} placeholders -> each becomes \d+; matches any indices.
        excludes = {"model.layers.0.experts.5.w1"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude(
                "model.layers.{i}.experts.{i}.w1", excludes
            )
        )

    def test_literal_match_takes_priority_over_template(self):
        # Even when {i} is present, exact-string match in exclude_modules wins early.
        excludes = {"model.layers.{i}.mlp"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude("model.layers.{i}.mlp", excludes)
        )


class CkptBaseMatchesRegexExcludeTest(unittest.TestCase):
    """The ``re:`` ignore entries describe which weight *templates* are excluded.

    The probe used to be the template with ``{i}`` replaced by a literal ``0``
    and the pattern was applied with ``re.search``, so a pattern naming layer 0
    matched every layer while a pattern naming layer 7 matched none - a
    layer-scoped ignore silently de-quantized the wrong layers, or silently did
    nothing, depending on which layer number was written.
    """

    TEMPLATE = "model.layers.{i}.mlp"

    def test_pattern_naming_layer_zero_does_not_match_the_template(self):
        excludes = {r"re:^model\.layers\.0\.mlp$"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )
        self.assertFalse(
            _ckpt_base_matches_regex_exclude(self.TEMPLATE, excludes)
        )

    def test_pattern_naming_another_layer_does_not_match_the_template(self):
        excludes = {r"re:^model\.layers\.7\.mlp$"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )
        self.assertFalse(
            _ckpt_base_matches_regex_exclude(self.TEMPLATE, excludes)
        )

    def test_pattern_covering_any_layer_matches_the_template(self):
        excludes = {r"re:^model\.layers\.\d+\.mlp$"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )
        self.assertTrue(
            _ckpt_base_matches_regex_exclude(self.TEMPLATE, excludes)
        )

    def test_unbounded_digit_quantifiers_cover_the_template(self):
        for pattern in (
            r"re:^model\.layers\.\d+\.mlp$",
            r"re:^model\.layers\.[0-9]+\.mlp$",
            r"re:.*mlp$",
        ):
            with self.subTest(pattern=pattern):
                self.assertTrue(
                    _ckpt_base_matches_quant_exclude(self.TEMPLATE, {pattern})
                )

    def test_digit_count_pinned_patterns_do_not_cover_the_template(self):
        r"""A pattern pinned to one digit does not stand for every layer.

        `model\.layers\.[0-9]\.mlp` matches layer 0 but not layer 12, so it
        must not be treated as covering the template - which is what the old
        single "0" probe did.
        """
        for pattern in (
            r"re:^model\.layers\.\d\.mlp$",
            r"re:^model\.layers\.[0-9]\.mlp$",
            r"re:^model\.layers\.\d{1,2}\.mlp$",
        ):
            with self.subTest(pattern=pattern):
                self.assertFalse(
                    _ckpt_base_matches_quant_exclude(self.TEMPLATE, {pattern})
                )
                self.assertFalse(
                    _ckpt_base_matches_regex_exclude(self.TEMPLATE, {pattern})
                )

    def test_pattern_without_a_trailing_anchor_still_matches(self):
        excludes = {r"re:^model\.layers\.\d+\.mlp"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )

    def test_unanchored_substring_pattern_still_matches(self):
        """The probe change must not alter matching for existing configs.

        A user pattern such as ``re:.*layernorm`` pins a substring and does not
        consume the whole template; it has to keep matching, otherwise modules
        that used to be excluded would silently start being quantized.
        """
        excludes = {r"re:.*layernorm"}
        template = "model.layers.{i}.input_layernorm.weight"
        self.assertTrue(
            _ckpt_base_matches_quant_exclude(template, excludes)
        )
        self.assertTrue(
            _ckpt_base_matches_regex_exclude(template, excludes)
        )

    def test_pattern_for_another_submodule_does_not_match(self):
        excludes = {r"re:^model\.layers\.\d+\.self_attn\.q_proj$"}
        self.assertFalse(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )

    def test_invalid_regex_still_raises(self):
        with self.assertRaises(ValueError):
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, {"re:["})

    def test_non_regex_entries_are_untouched(self):
        # A concrete path in the exclude list keeps working as before.
        excludes = {"model.layers.5.mlp"}
        self.assertTrue(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, excludes)
        )
        self.assertFalse(
            _ckpt_base_matches_quant_exclude(self.TEMPLATE, {"model.layers.5.moe"})
        )


if __name__ == "__main__":
    unittest.main()
