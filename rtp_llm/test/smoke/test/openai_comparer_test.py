import unittest
from types import SimpleNamespace

from smoke.common_def import SmokeException
from smoke.openai_comparer import OpenaiComparer


class MinimumAuxInfoTest(unittest.TestCase):
    def compare(self, actual_aux, minimum=8192, grammar_only=False):
        comparer = OpenaiComparer(
            None,
            "",
            {
                "query": {},
                "grammar_constraint_only": grammar_only,
                "compare_config": {"minimum_aux_info": {"input_len": minimum}},
            },
            None,
            False,
        )
        expected = SimpleNamespace(
            choices=[], usage=None, aux_info=None, extra_outputs=None
        )
        actual = SimpleNamespace(
            choices=[], usage=None, aux_info=actual_aux, extra_outputs=None
        )
        comparer.compare_result(expected, actual)

    def test_accepts_exact_lower_bound_and_larger_values(self):
        for grammar_only in (False, True):
            for value in (8192, 8193):
                with self.subTest(grammar_only=grammar_only, value=value):
                    self.compare(
                        SimpleNamespace(input_len=value), grammar_only=grammar_only
                    )

    def test_rejects_missing_short_and_invalid_actual_values(self):
        for grammar_only in (False, True):
            for aux in (
                None,
                SimpleNamespace(),
                SimpleNamespace(input_len=8191),
                SimpleNamespace(input_len=True),
                SimpleNamespace(input_len="8192"),
            ):
                with self.subTest(grammar_only=grammar_only, aux=aux):
                    with self.assertRaisesRegex(SmokeException, "minimum.*aux_info"):
                        self.compare(aux, grammar_only=grammar_only)

    def test_rejects_invalid_fixture_thresholds(self):
        for grammar_only in (False, True):
            for minimum in (True, "8192", None):
                with self.subTest(grammar_only=grammar_only, minimum=minimum):
                    with self.assertRaisesRegex(SmokeException, "minimum.*aux_info"):
                        self.compare(
                            SimpleNamespace(input_len=8192), minimum, grammar_only
                        )


if __name__ == "__main__":
    unittest.main()
