import unittest
from unittest.mock import Mock

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import W

_MIB = 1024**2


class FastsafetensorsMemoryTest(unittest.TestCase):
    def setUp(self):
        self.loader = object.__new__(ModelLoader)
        self.model_config = Mock()
        self.model_config.eval_model_weight_size.return_value = 8192 * _MIB
        self.model_config.moe_weight_param_count.return_value = 75
        self.model_config.layer_weight_param_count.return_value = 100
        self.loader._weights_info = Mock(model_config=self.model_config)
        self.database = Mock()
        self.database.get_max_file_size.return_value = 1024 * _MIB
        self.device = Mock()
        self.loader._load_config = Mock(
            database=self.database,
            exported_device=self.device,
            tp_size=1,
            ep_size=1,
            fastsafetensors_reserve_mb=2048,
        )
        self.loader._is_online_ptpc = Mock(return_value=False)
        self.loader._is_online_quant_without_inline = Mock(return_value=False)

    def _assert_boundary(self, threshold_mib):
        # Check both sides to catch a changed reserve, missing accounting term,
        # or an accidental >= comparison. Inputs from the device are bytes.
        for delta, expected in ((-1, False), (0, False), (1, True)):
            with self.subTest(threshold_mib=threshold_mib, delta_bytes=delta):
                self.device.get_mem_info.return_value = Mock(
                    free=threshold_mib * _MIB + delta
                )
                self.assertIs(
                    self.loader._is_memory_enough_for_fastsafetensor(), expected
                )

    def test_default_reserve_boundary(self):
        # 8192 model + 3 * 1024 largest shard + 2048 reserve.
        self._assert_boundary(13312)

    def test_nonnegative_reserve_overrides(self):
        for reserve, threshold in ((0, 11264), (512, 11776), (4096, 15360)):
            with self.subTest(reserve_mib=reserve):
                self.loader._load_config.fastsafetensors_reserve_mb = reserve
                self._assert_boundary(threshold)

    def test_tp_ep_scaling_only_divides_model_memory(self):
        for tp, ep, threshold in ((1, 1, 13312), (4, 2, 7168), (2, 4, 7168)):
            with self.subTest(tp=tp, ep=ep):
                self.loader._load_config.tp_size = tp
                self.loader._load_config.ep_size = ep
                self._assert_boundary(threshold)

    def test_largest_shard_memory_boundary(self):
        for shard_mib, threshold in ((0, 10240), (256, 11008), (2048, 16384)):
            with self.subTest(shard_mib=shard_mib):
                self.database.get_max_file_size.return_value = shard_mib * _MIB
                self._assert_boundary(threshold)

    def test_inline_ptpc_mixed_moe_dense_accounting(self):
        self.loader._load_config.tp_size = 4
        self.loader._is_online_ptpc.return_value = True
        # 75% inline MoE + 25% dense: 2048 * 1.25 + 3072 + 2048.
        self._assert_boundary(7680)
        self.loader._is_online_quant_without_inline.assert_not_called()

    def test_inline_ptpc_all_moe_needs_no_dense_multiplier(self):
        self.loader._load_config.tp_size = 4
        self.loader._is_online_ptpc.return_value = True
        self.model_config.moe_weight_param_count.return_value = 100
        self._assert_boundary(7168)

    def test_inline_ptpc_missing_moe_counts_doubles_model_memory(self):
        self.loader._load_config.tp_size = 4
        self.loader._is_online_ptpc.return_value = True
        for moe, total in ((0, 100), (75, 0)):
            with self.subTest(moe=moe, total=total):
                self.model_config.moe_weight_param_count.return_value = moe
                self.model_config.layer_weight_param_count.return_value = total
                self._assert_boundary(9216)

    def test_non_inline_online_quant_doubles_model_memory(self):
        self.loader._load_config.ep_size = 4
        self.loader._is_online_quant_without_inline.return_value = True
        self._assert_boundary(9216)
        self.model_config.moe_weight_param_count.assert_not_called()

    def test_unavailable_device_memory_rejects_fastsafetensors(self):
        self.device.get_mem_info.return_value = None
        self.assertFalse(self.loader._is_memory_enough_for_fastsafetensor())


class FastsafetensorsCopyoutWiringTest(unittest.TestCase):
    def test_production_filter_uses_rank_weight_map_and_skips_aliases(self):
        # Use the production map builder and loading method. Only descriptors,
        # checkpoint I/O and final materialization are replaced with cheap fakes.
        for rank in (0, 1):
            for language_model in (False, True):
                with self.subTest(rank=rank, language_model=language_model):
                    loader = object.__new__(ModelLoader)
                    database = Mock()
                    loader._load_config = Mock(database=database, num_layers=1)
                    loader._task_type = (
                        TaskType.LANGUAGE_MODEL if language_model else object()
                    )
                    loader._global_weight_aliases = {"tied_output": object()}
                    loader._misc_weights_info = []
                    loader._is_online_ptpc = Mock(return_value=False)
                    loader._build_stacked_key_config = Mock(return_value={})
                    model_weights = Mock()
                    loader._create_model_weights = Mock(return_value=model_weights)

                    def weight(name, keys):
                        result = Mock()
                        result.name = name
                        result.get_tensor_names.return_value = set(keys)
                        result.get_components.return_value = [result]
                        result.load.return_value = {}
                        return result

                    dense = weight("dense", ["layer.0.dense"])
                    expert = weight("moe", [f"layer.0.experts.{rank}.weight"])
                    embedding = weight(
                        "embedding", ["embedding.weight", "embedding.alias"]
                    )
                    tied = weight("tied_output", ["tied.weight"])
                    head = weight(W.lm_head, ["lm_head.weight"])
                    loader._model_weights_info = Mock(
                        layer_weights=[[dense, expert]], weights=[embedding, tied, head]
                    )
                    expected = {
                        "layer.0.dense",
                        f"layer.0.experts.{rank}.weight",
                        "embedding.weight",
                        "embedding.alias",
                    }
                    if language_model:
                        expected.add("lm_head.weight")
                    all_keys = expected | {
                        "layer.0.experts.0.weight",
                        "layer.0.experts.1.weight",
                        "tied.weight",
                        "lm_head.weight",
                        "unrelated",
                    }
                    selected = set()

                    def iterator(
                        device, use_tqdm, *, stacked_key_config, local_copyout_filter
                    ):
                        self.assertEqual(device, "cpu")
                        for key in sorted(all_keys):
                            if local_copyout_filter(key):
                                selected.add(key)
                                yield key, torch.tensor([1.0])

                    database.fastsafetensors_weights_iterator.side_effect = iterator
                    self.assertIs(
                        loader._load_from_fastsafetensor("cpu"), model_weights
                    )
                    self.assertEqual(selected, expected)
                    tied.get_tensor_names.assert_not_called()
                    tied.load.assert_not_called()
                    if not language_model:
                        head.get_tensor_names.assert_not_called()
                        head.load.assert_not_called()
                    for descriptor in (dense, expert, embedding):
                        descriptor.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
