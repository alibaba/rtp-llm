#!/usr/bin/env python3

import re
import unittest

from rtp_llm.test.perf_test.deepseek_v4_prefill_formula_fit import (
    FEATURE_NAMES,
    Observation,
    feature_values,
    formula_text,
    split_rows,
)


class DeepseekV4PrefillFormulaFitTest(unittest.TestCase):

    def test_features_use_compute_and_hit_tokens(self) -> None:
        row = Observation(
            batch_size=1,
            input_len=4096,
            cache_len=1024,
            target_ms=10.0,
            source="synthetic",
        )
        self.assertEqual(feature_values(row), [1.0, 3.0, 1.0, 9.0, 3.0, 1.0])

    def test_exported_formula_uses_only_prefill_time_formula_names(self) -> None:
        expression = formula_text([1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
        self.assertEqual(identifiers, {"sum", "computeTokens", "hitCacheTokens"})
        self.assertNotIn("tokens", identifiers)

    def test_feature_expressions_match_flexlb_aggregate_grammar(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 6)
        for expression in FEATURE_NAMES[1:]:
            self.assertTrue(expression.startswith("sum("), expression)
            self.assertTrue(expression.endswith(")"), expression)

    def test_random_half_split_is_exact_and_reproducible(self) -> None:
        rows = [
            Observation(1, 1024 + index, index, 10.0 + index, f"row-{index}")
            for index in range(11)
        ]
        first = split_rows(rows, mode="random-50-50", seed=17)
        second = split_rows(rows, mode="random-50-50", seed=17)
        self.assertEqual(len(first["train"]), 5)
        self.assertEqual(len(first["validation"]), 0)
        self.assertEqual(len(first["test"]), 6)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["train"]) | set(first["test"]),
            set(rows),
        )
        self.assertFalse(set(first["train"]) & set(first["test"]))


if __name__ == "__main__":
    unittest.main()
