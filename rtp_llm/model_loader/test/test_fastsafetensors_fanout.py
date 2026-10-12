import unittest
from types import SimpleNamespace

import torch
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.ops import TaskType


class _Weight:
    def __init__(self, name, keys):
        self.name = name
        self.keys = set(keys)
        self.loads = 0

    def get_components(self):
        return [self]

    def get_tensor_names(self, layer_id, load_config):
        return self.keys

    def load(self, tensor_source, layer_id, device, load_config):
        self.loads += 1
        return {self.name: sum(tensor_source.load_tensor(key)[0] for key in self.keys)}


class _Database:
    def __init__(self, iterated, fallback=None):
        self.iterated = iterated
        self.fallback = fallback or {}
        self.fallback_reads = []
        self.copyout_filter = None

    def fastsafetensors_weights_iterator(self, device, collective, **kwargs):
        self.copyout_filter = kwargs["local_copyout_filter"]
        yield from self.iterated

    def load_tensor(self, name, dtype):
        self.fallback_reads.append(name)
        return [self.fallback[name].to(dtype)]


class FastSafeTensorsFanoutTest(unittest.TestCase):
    def loader(self, weights, database):
        loader = ModelLoader.__new__(ModelLoader)
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._global_weight_aliases = {}
        loader._load_config = SimpleNamespace(num_layers=1, database=database)
        loader._model_weights_info = SimpleNamespace(
            layer_weights=[weights], weights=[]
        )
        loader._misc_weights_info = []
        loader._is_online_ptpc = lambda: False
        loader._build_stacked_key_config = lambda _: {}
        result = {}
        loader._create_model_weights = lambda _: SimpleNamespace(
            set_layer_weight=lambda layer, name, tensor: result.__setitem__(
                name, tensor
            )
        )
        return loader, result

    def test_shared_checkpoint_tensor_reaches_every_consumer_once(self):
        first, second = _Weight("first", {"shared"}), _Weight("second", {"shared"})
        database = _Database(
            [("shared", torch.tensor([3.0])), ("shared", torch.tensor([4.0]))]
        )
        loader, result = self.loader([first, second], database)
        loader._load_from_fastsafetensor("cpu")
        self.assertEqual((first.loads, second.loads), (1, 1))
        self.assertEqual(result["first"].item(), 3.0)
        self.assertEqual(result["second"].item(), 3.0)
        self.assertFalse(database.fallback_reads)
        self.assertTrue(database.copyout_filter("shared"))
        self.assertFalse(database.copyout_filter("unused"))

    def test_incomplete_consumer_falls_back_without_reloading_completed_units(self):
        first = _Weight("complete", {"shared"})
        second = _Weight("incomplete", {"shared", "missing"})
        database = _Database(
            [("shared", torch.tensor([3.0]))],
            {"shared": torch.tensor([5.0]), "missing": torch.tensor([7.0])},
        )
        loader, result = self.loader([first, second], database)
        loader._load_from_fastsafetensor("cpu")
        self.assertEqual((first.loads, second.loads), (1, 1))
        self.assertEqual(result["complete"].item(), 3.0)
        self.assertEqual(result["incomplete"].item(), 12.0)
        self.assertCountEqual(database.fallback_reads, ["shared", "missing"])


if __name__ == "__main__":
    unittest.main()
