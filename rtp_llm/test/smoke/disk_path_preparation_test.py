import os
import tempfile
import unittest
from unittest.mock import patch

from smoke.case_runner import CaseRunner


class DiskPathPreparationTest(unittest.TestCase):
    def test_block_tree_disk_paths_are_prepared_under_test_tmpdir(self):
        runner = object.__new__(CaseRunner)
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {"TEST_TMPDIR": temp_dir}
            with patch.dict(os.environ, env, clear=False):
                result = runner.create_env_from_args(
                    [
                        "DISK_CACHE_PATHS="
                        "__TEST_TMPDIR__/disk_rank0,__TEST_TMPDIR__/disk_rank1"
                    ]
                )

            paths = result["DISK_CACHE_PATHS"].split(",")
            self.assertEqual(
                paths,
                [
                    os.path.join(temp_dir, "disk_rank0"),
                    os.path.join(temp_dir, "disk_rank1"),
                ],
            )
            self.assertTrue(all(os.path.isdir(path) for path in paths))

    def test_smoke_repeat_expands_once_and_preserves_other_messages(self):
        query_result = {
            "smoke_repeat": {
                "message_index": 0,
                "count": 3,
                "separator": "|",
            },
            "query": {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "user", "content": "tail"},
                ]
            },
        }

        CaseRunner._expand_smoke_repeat(query_result)
        self.assertEqual(
            query_result["query"]["messages"],
            [
                {"role": "user", "content": "x|x|x"},
                {"role": "user", "content": "tail"},
            ],
        )
        self.assertTrue(query_result["_smoke_repeat_expanded"])

        CaseRunner._expand_smoke_repeat(query_result)
        self.assertEqual(query_result["query"]["messages"][0]["content"], "x|x|x")

    def test_smoke_repeat_rejects_oversized_expansion(self):
        query_result = {
            "smoke_repeat": {"message_index": 0, "count": 2},
            "query": {
                "messages": [
                    {
                        "role": "user",
                        "content": "x" * CaseRunner._MAX_SMOKE_REPEAT_CHARS,
                    }
                ]
            },
        }

        with self.assertRaisesRegex(ValueError, "exceeds"):
            CaseRunner._expand_smoke_repeat(query_result)


if __name__ == "__main__":
    unittest.main()
