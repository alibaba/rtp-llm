import unittest
from types import SimpleNamespace

from rtp_llm.config.layer_partition import finalize_pp_partition, local_layer_range


class PPPartitionTest(unittest.TestCase):
    def test_finalization_generates_default_partition(self):
        for layers, stages, expected in (
            (65, 4, [17, 16, 16, 16]),
            (64, 4, [16, 16, 16, 16]),
            (9, 2, [5, 4]),
        ):
            with self.subTest(layers=layers, stages=stages):
                config = SimpleNamespace(pp_size=stages, pp_stage_layer_counts=[])
                finalize_pp_partition(config, layers)
                self.assertEqual(config.pp_stage_layer_counts, expected)

    def test_finalization_preserves_explicit_partition(self):
        config = SimpleNamespace(pp_size=3, pp_stage_layer_counts=[2, 3, 2])
        finalize_pp_partition(config, 7)
        self.assertEqual(config.pp_stage_layer_counts, [2, 3, 2])

    def test_finalization_rejects_invalid_partition(self):
        for counts in ([8], [4, 5], [8, 0], [-1, 9]):
            config = SimpleNamespace(pp_size=2, pp_stage_layer_counts=counts)
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                finalize_pp_partition(config, 8)

    def test_generated_partition_requires_a_layer_for_each_stage(self):
        config = SimpleNamespace(pp_size=3, pp_stage_layer_counts=[])
        with self.assertRaisesRegex(ValueError, "at least one layer"):
            finalize_pp_partition(config, 2)

    def test_single_stage_ignores_partition(self):
        for counts in ([], [2, 5]):
            with self.subTest(counts=counts):
                config = SimpleNamespace(pp_size=1, pp_stage_layer_counts=counts)
                finalize_pp_partition(config, 8)
                self.assertEqual(config.pp_stage_layer_counts, counts)


class LocalLayerRangeTest(unittest.TestCase):
    def test_target_uses_global_ids_from_configured_partition(self):
        model = SimpleNamespace(num_layers=7, is_mtp=False)
        for rank, expected in enumerate((range(2), range(2, 5), range(5, 7))):
            with self.subTest(rank=rank):
                parallel = SimpleNamespace(
                    pp_size=3, pp_rank=rank, pp_stage_layer_counts=[2, 3, 2]
                )
                self.assertEqual(local_layer_range(model, parallel), expected)

    def test_draft_keeps_all_its_layers_with_target_partition(self):
        model = SimpleNamespace(num_layers=2, is_mtp=True)
        parallel = SimpleNamespace(
            pp_size=3, pp_rank=2, pp_stage_layer_counts=[2, 3, 2]
        )
        self.assertEqual(local_layer_range(model, parallel), range(2))
        self.assertEqual(parallel.pp_stage_layer_counts, [2, 3, 2])
        self.assertEqual((parallel.pp_size, parallel.pp_rank), (3, 2))

    def test_single_stage_uses_all_layers(self):
        model = SimpleNamespace(num_layers=8, is_mtp=False)
        for counts in ([], [2, 5]):
            with self.subTest(counts=counts):
                parallel = SimpleNamespace(
                    pp_size=1, pp_rank=0, pp_stage_layer_counts=counts
                )
                self.assertEqual(local_layer_range(model, parallel), range(8))


if __name__ == "__main__":
    unittest.main()
