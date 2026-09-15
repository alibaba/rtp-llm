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


if __name__ == "__main__":
    unittest.main()
