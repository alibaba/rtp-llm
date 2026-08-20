#!/usr/bin/env python3

import json
import os
import unittest

import run_block_tree_cache_benchmark as driver


class BenchmarkDriverProfileTest(unittest.TestCase):
    def test_loads_dsv4_flash_full_and_swa_layout(self):
        path = driver.resolve_runfile_path(
            "profiles/deepseek_v4_flash_fp8_tp1_cp1_tpb1024.json"
        )
        self.assertIsNotNone(path)
        with open(path) as source:
            profile = json.load(source)

        self.assertEqual(profile["model"], "deepseek_v4_flash")
        self.assertEqual(profile["num_layers"], 43)
        self.assertEqual(profile["tokens_per_block"], 1024)
        self.assertEqual(profile["kernel_tokens_block"], 128)
        self.assertEqual(
            driver.load_profile_group_set_payloads(path),
            {"full_context": 4087296, "swa": 4940160},
        )

        groups = {group["tag"]: group for group in profile["groups"]}
        self.assertEqual(
            [(tag, groups[tag]["layer_count"]) for tag in ("csa_kv", "hca_kv", "indexer_kv")],
            [("csa_kv", 21), ("hca_kv", 20), ("indexer_kv", 21)],
        )
        self.assertEqual(
            [(tag, groups[tag]["layer_count"]) for tag in ("csa_state", "indexer_state", "swa_kv")],
            [("csa_state", 21), ("indexer_state", 21), ("swa_kv", 43)],
        )
        self.assertEqual(profile["device_only_groups"], ["hca_state"])
        self.assertEqual(profile["swa_config"]["gen_num_per_cycle"], 0)
        self.assertFalse(profile["swa_config"]["include_mtp"])


if __name__ == "__main__":
    unittest.main()
