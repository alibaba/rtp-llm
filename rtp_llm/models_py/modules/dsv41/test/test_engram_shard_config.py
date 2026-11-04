# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace as NS
from rtp_llm.models_py.modules.dsv41.engram_shard_config import ShardConfig, ShardConfigError


def manifest(count):
    return NS(tensors=[NS(shards=[NS(owner_id=f"host-{i}", owner_incarnation=f"boot-{i}")
                                 for i in range(count)]) for _ in range(4)])


class ConfigTest(unittest.TestCase):
    def test_p_is_independent_of_model_parallelism_and_instance_count(self):
        for count in (2, 3, 5, 18, 36, 64):
            for instances in (1, 2, 10):
                config = ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": str(count),
                    "TP_SIZE": "1", "DP_SIZE": "8", "EP_SIZE": "8", "INSTANCES": str(instances)})
                self.assertEqual(config.shard_count, count)
                config.validate_manifest(manifest(count))

    def test_inference_from_manifest_and_explicit_mismatch(self):
        self.assertEqual(ShardConfig.from_environ({}, inferred_count=3).shard_count, 3)
        with self.assertRaisesRegex(ShardConfigError, "COUNT_MISMATCH"):
            ShardConfig(3).validate_manifest(manifest(2))

    def test_single_host_and_invalid_p_are_rejected(self):
        for value in (None, True, 1, 0, -1, 65, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig(value)
        for value in ("", "-1", "2.0", "２", " 2", 2):
            with self.subTest(value=value), self.assertRaises(ShardConfigError):
                ShardConfig.from_environ({"ENGRAM_SHARD_COUNT": value})

    def test_cancelled_mode_and_rpc_options_fail_explicitly(self):
        for key in ("ENGRAM_SHARD_MODE", "ENGRAM_SHARD_INDEX", "ENGRAM_SHARD_ENDPOINTS"):
            with self.assertRaisesRegex(ShardConfigError, "REMOVED"):
                ShardConfig.from_environ({key: "x"}, inferred_count=2)

    def test_missing_duplicate_or_replaced_host_is_rejected(self):
        for case in ("missing", "duplicate", "restart"):
            doc = manifest(3)
            if case == "missing": doc.tensors[1].shards.pop()
            if case == "duplicate": doc.tensors[1].shards[1].owner_id = "host-0"
            if case == "restart": doc.tensors[1].shards[1].owner_incarnation = "new-boot"
            with self.subTest(case=case), self.assertRaises(ShardConfigError):
                ShardConfig(3).validate_manifest(doc)


if __name__ == "__main__":
    unittest.main()
