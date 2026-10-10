#!/usr/bin/env python3

import json
import math
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
    Observation,
    build_parser,
)
from rtp_llm.test.perf_test.cache_grid.formula.restricted_symbolic_fit import (
    LIBRARY_VERSION,
    PARSER_AGGREGATES,
    PARSER_BATCH_VARIABLES,
    PARSER_FUNCTIONS,
    PARSER_OPERATORS,
    PARSER_PER_REQUEST_VARIABLES,
    build_candidate_library,
    fit_restricted_symbolic,
)


class RestrictedSymbolicLibraryTest(unittest.TestCase):
    def test_unified_batch_fit_recovers_request_sum_and_batch_cost(self):
        rows = []
        for batch in (1, 2, 4):
            for index in range(1, 41):
                requests = tuple(
                    (
                        1024 * (index + j * 3 + (index * 7 + j) % 10),
                        1024 * ((index * 7 + j) % 10),
                    )
                    for j in range(batch)
                )
                total_compute = sum(i - c for i, c in requests)
                rows.append(
                    Observation(
                        batch,
                        sum(i for i, _ in requests),
                        sum(c for _, c in requests),
                        10 + 7 * batch + 0.002 * total_compute,
                        f"{batch}:{index}",
                        requests=requests,
                        geometry_key=f"{batch}:{index}",
                    )
                )
        train = [r for i, r in enumerate(rows) if i % 5 != 0]
        validation = [r for i, r in enumerate(rows) if i % 5 == 0]
        model = fit_restricted_symbolic(train, validation, rows, token_unit=1024)
        for row in rows:
            self.assertAlmostEqual(model.predict(row), row.target_ms, places=6)

    def test_library_is_versioned_and_flexlb_compatible(self):
        terms = build_candidate_library(65536)
        self.assertEqual(LIBRARY_VERSION, "prefill-request-list-v3")
        self.assertEqual(len(terms), 70)
        expressions = " ".join(term.expression for term in terms)
        self.assertNotIn("log1p", expressions)
        self.assertNotIn("**", expressions)
        self.assertIn("log(1 +", expressions)
        self.assertIn("pow(", expressions)
        self.assertIn("max(", expressions)
        self.assertIn("min(", expressions)
        self.assertIn("abs(", expressions)
        self.assertIn("exp(", expressions)
        self.assertIn(" ^ ", expressions)
        self.assertIn("hasHitCache", expressions)

    def test_library_covers_java_parser_primitives_with_finite_candidates(self):
        self.assertEqual(PARSER_OPERATORS, ("+", "-", "*", "/", "^"))
        self.assertEqual(
            PARSER_FUNCTIONS,
            ("sqrt", "log", "exp", "abs", "max", "min", "pow"),
        )
        self.assertEqual(PARSER_AGGREGATES, ("sum",))
        self.assertEqual(
            PARSER_PER_REQUEST_VARIABLES,
            ("inputTokens", "hitCacheTokens", "computeTokens", "hasHitCache"),
        )
        self.assertEqual(
            PARSER_BATCH_VARIABLES,
            (
                "batchSize",
                "totalInputTokens",
                "totalHitCacheTokens",
                "totalComputeTokens",
                "maxInputTokens",
                "maxComputeTokens",
            ),
        )
        rows = (
            Observation(1, 1, 0, 1.0, "minimum", 0),
            Observation(1, 1048575, 0, 1.0, "cold-maximum", 0),
            Observation(1, 1048575, 1040384, 1.0, "hit-maximum", 1040384),
        )
        for term in build_candidate_library(65536):
            for row in rows:
                self.assertTrue(math.isfinite(term.evaluate(row)), term.name)

    def test_forward_selection_recovers_linear_compute_term(self):
        rows = []
        for index in range(1, 61):
            input_len = index * 4096
            cache_len = input_len * (index % 4) // 4
            compute_units = (input_len - cache_len) / 65536.0
            rows.append(
                Observation(
                    batch_size=1,
                    input_len=input_len,
                    cache_len=cache_len,
                    target_ms=20.0 + 7.5 * compute_units,
                    source=f"synthetic:{index}",
                    requested_cache_len=cache_len,
                )
            )
        model = fit_restricted_symbolic(
            rows[:40],
            rows[40:50],
            rows[:50],
            token_unit=65536,
            max_terms=2,
            complexity_tolerance_pct=0.0,
        )
        self.assertEqual([term.name for term in model.terms], ["1", "u"])
        self.assertLess(
            max(abs(model.predict(row) - row.target_ms) for row in rows), 1e-8
        )


class RestrictedSymbolicCliTest(unittest.TestCase):
    def test_fit_cli_produces_search_metadata_and_formula(self):
        metrics = []
        for index in range(1, 101):
            input_len = index * 4096
            target_ms = 25.0 + 0.002 * input_len
            runs = [
                {
                    "prefill_time_ms": target_ms,
                    "success": True,
                    "input_len": input_len,
                    "output_len": 1,
                    "reuse_len": 0,
                    "ttft_ms": target_ms,
                    "ttft_source": "client_test_wall_max_new_tokens_1",
                }
                for _ in range(3)
            ]
            metrics.append(
                {
                    "case_key": f"bs1_seq{input_len}_cache0",
                    "case_id": index,
                    "batch_size": 1,
                    "input_len": input_len,
                    "cache_len_requested": 0,
                    "cache_len_observed": [0, 0, 0],
                    "success_runs": 3,
                    "measure_runs": 3,
                    "status": "ok",
                    "reuse_exact": True,
                    "runs": runs,
                }
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "complete": True,
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "formula"
            args = build_parser().parse_args(
                [
                    "fit",
                    "--inputs",
                    str(result_path),
                    "--output-dir",
                    str(output_dir),
                    "--symbolic-max-terms",
                    "3",
                ]
            )
            exit_code = args.func(args)
            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertEqual(report["model_family"], "restricted-symbolic")
            self.assertEqual(
                report["symbolic_search"]["library_version"], LIBRARY_VERSION
            )
            formula = (output_dir / "Model_prefill_formula.txt").read_text()
            self.assertNotIn("log1p", formula)
            self.assertNotIn("**", formula)


if __name__ == "__main__":
    unittest.main()
