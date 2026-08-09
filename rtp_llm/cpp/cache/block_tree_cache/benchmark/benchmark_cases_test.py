import unittest

import benchmark_cases


class BenchmarkCasesTest(unittest.TestCase):
    def test_registry_names_are_unique(self):
        names = [case.name for case in benchmark_cases.ALL_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_suite_counts_and_membership(self):
        self.assertEqual(len(benchmark_cases.get_suite_cases("smoke")), 2)
        self.assertEqual(len(benchmark_cases.get_suite_cases("profile")), 14)
        self.assertTrue(
            all(
                case.suite in {"smoke", "profile"} for case in benchmark_cases.ALL_CASES
            )
        )

    def test_copy_strategy_cases_are_explicit(self):
        for case in benchmark_cases.PROFILE_CASES:
            if case.name.endswith("_batch"):
                self.assertEqual(case.params.get("--copy-strategy"), "batch")
            if case.name.endswith("_staged_sm"):
                self.assertEqual(case.params.get("--copy-strategy"), "staged-sm")


if __name__ == "__main__":
    unittest.main()
