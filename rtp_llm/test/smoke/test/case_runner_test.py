import unittest
from unittest.mock import patch

from smoke.case_runner import CaseRunner
from smoke.common_def import QueryStatus, SmokeException
from smoke.normal_comparer import NormalComparer
from smoke.openai_comparer import OpenaiComparer


class CaseRunnerComparerTest(unittest.TestCase):
    @patch("smoke.case_runner.has_internal_source", return_value=False)
    def test_pg_query_requires_internal_source(self, _has_internal_source):
        with self.assertRaises(SmokeException) as raised:
            CaseRunner._get_comparer_cls({"pg_module": True, "query": {}}, "/")
        self.assertEqual(raised.exception.error_status, QueryStatus.VALID_FAILED)
        self.assertIn("require internal_source", raised.exception.message)

    @patch("smoke.case_runner.has_internal_source", return_value=False)
    def test_ordinary_oss_queries_keep_their_comparers(self, _has_internal_source):
        self.assertIs(CaseRunner._get_comparer_cls({"query": {}}, "/"), NormalComparer)
        self.assertIs(
            CaseRunner._get_comparer_cls(
                {"query": {"messages": []}}, "/v1/chat/completions"
            ),
            OpenaiComparer,
        )
        _has_internal_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
