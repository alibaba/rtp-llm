"""Registry contracts: explicit order, pure declarations and fail-fast metadata."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_ft.registry import case, collect_cases, validate_cases


class CaseRegistryTest(unittest.TestCase):
    def declare(self, name, **metadata):
        def run(ctx):
            return True, "done"

        return case(name, category="status", **metadata)(run)

    def test_explicit_order_and_return_contract(self):
        first = self.declare("first")
        second = self.declare("second", expected_fail=True, requires=["enqueue_batch"])
        definitions = collect_cases("status", [second, first])
        self.assertEqual([d.name for d in definitions], ["second", "first"])
        self.assertIs(definitions[0].fn, second)
        self.assertEqual(first(None), (True, "done"))
        self.assertTrue(definitions[0].expected_fail)
        self.assertEqual(definitions[0].requires, ["enqueue_batch"])

    def test_repeated_collection_has_no_registration_side_effect(self):
        fn = self.declare("same")
        self.assertEqual(len(collect_cases("status", [fn])), 1)
        self.assertEqual(len(collect_cases("status", [fn])), 1)

    def test_duplicate_across_categories_rejected_before_filtering(self):
        one = self.declare("ambiguous")

        @case("ambiguous", category="kv")
        def two(ctx):
            return True, ""

        with self.assertRaisesRegex(ValueError, "duplicate case 'ambiguous'.*run.*two"):
            validate_cases(collect_cases("status", [one]) + collect_cases("kv", [two]))

    def test_duplicate_in_category_rejected(self):
        fn = self.declare("same")
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            collect_cases("status", [fn, fn])

    def test_wrong_category_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 'kv'"):
            collect_cases("kv", [self.declare("wrong")])

    def test_missing_declaration_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing @case"):
            collect_cases("status", [lambda ctx: (True, "")])

    def test_double_declaration_rejected(self):
        with self.assertRaisesRegex(ValueError, "already declared"):
            case("another", category="kv")(self.declare("one"))

    def test_invalid_metadata_rejected(self):
        for metadata in ({"profiles": ["typo"]}, {"requires": ["typo"]}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "invalid case metadata"):
                    collect_cases("status", [self.declare("invalid", **metadata)])

    def test_empty_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty case name"):
            collect_cases("status", [self.declare(" ")])


if __name__ == "__main__":
    unittest.main()
