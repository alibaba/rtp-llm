"""Real Qwen packed-expert descriptors: initial load vs streamed policy refit."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.model_loader.test.test_pure_tp_preshard import (
    _config, _database, _tensors, _weights,
)
from rtp_llm.model_loader.test.test_tensor_weight_stream import convert
from rtp_llm.ops import TaskType


class PackedMoeRlStreamTest(unittest.TestCase):
    def test_all_tp_ranks_receive_physical_packed_sources_without_stale_reads(self):
        weights = _weights(stacked=True)
        tensors = {
            name: value.bfloat16()
            for weight in weights
            for name, value in _tensors(weight).items()
        }
        for rank in (0, 1):
            with self.subTest(tp_rank=rank), _database(tensors) as database:
                config = _config(
                    rank, database=database, num_layers=1, is_ft_style_weight=False,
                )
                config.compute_dtype = torch.bfloat16
                loader = ModelLoader.__new__(ModelLoader)
                loader._load_config = config
                loader._model_weights_info = ModelWeightInfo([], [weights])
                loader._misc_weights_info = []
                loader._global_weight_aliases = {}
                loader._task_type = TaskType.LANGUAGE_MODEL
                self.assertEqual(loader.get_rl_hf_source_names(), tuple(sorted(tensors)))
                initial = {
                    (0, key): tensor
                    for weight in weights
                    for key, tensor in weight.load(
                        DatabaseTensorSource(database), 0, "cpu", config
                    ).items()
                }
                # Even with initial-load preshard enabled, refit must consume
                # only the newly pushed tensors through the actual collector.
                with patch.object(database, "load_tensor", side_effect=AssertionError("stale checkpoint read")), \
                     patch.object(database, "load_tensor_slice", side_effect=AssertionError("stale checkpoint slice")):
                    streamed = convert(loader, list(tensors.items()))
                    updated = convert(loader, [(key, value + 8) for key, value in tensors.items()])
                self.assertEqual(set(initial), set(streamed))
                for key in initial:
                    torch.testing.assert_close(streamed[key], initial[key], rtol=0, atol=0)
                    self.assertFalse(torch.equal(updated[key], initial[key]))


if __name__ == "__main__":
    unittest.main()
