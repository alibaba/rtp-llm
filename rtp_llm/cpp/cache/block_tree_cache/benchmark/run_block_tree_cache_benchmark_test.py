#!/usr/bin/env python3

import json
import os
import tempfile
import unittest

import run_block_tree_cache_benchmark as driver


class BenchmarkDriverProfileTest(unittest.TestCase):
    def test_loads_minimal_descriptor_size_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "descriptor_sizes.json")
            with open(path, "w") as output:
                json.dump(
                    {
                        "descriptor_size_bytes": {
                            "full_context": 2930688,
                            "swa": 1250304,
                        }
                    },
                    output,
                )

            self.assertEqual(
                driver.load_profile_group_set_payloads(path),
                {"full_context": 2930688, "swa": 1250304},
            )

    def test_selects_descriptor_profile_only_for_transfer(self):
        self.assertEqual(
            driver.profile_for_subcommand("transfer", "model.json", "sizes.json"),
            "sizes.json",
        )
        self.assertEqual(
            driver.profile_for_subcommand("tree", "model.json", "sizes.json"),
            "model.json",
        )


if __name__ == "__main__":
    unittest.main()
