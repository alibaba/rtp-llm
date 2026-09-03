import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.ops import TaskType, VitSeparation
from rtp_llm.utils.model_weight import CkptWeightInfo, identity


def transpose(tensors):
    return tensors[0].T.contiguous()


class FakeDatabase:
    def __init__(self, iterator_tensors, fallback_tensors=None):
        self.iterator_tensors = iterator_tensors
        self.fallback_tensors = fallback_tensors or {}

    def fastsafetensors_weights_iterator(self, *args, **kwargs):
        return iter(self.iterator_tensors)

    def load_tensor(self, name, data_type=torch.float16):
        return [self.fallback_tensors[name].to(data_type)]

    def has_tensor(self, name):
        return name in self.fallback_tensors


class RecordingWeight:
    def __init__(self, name, tensor_names):
        self.name = name
        self.tensor_names = set(tensor_names)
        self.load_sources = []

    def get_tensor_names(self, layer_id, load_config):
        return self.tensor_names

    def load(self, tensor_source, layer_id, device, load_config):
        self.load_sources.append(tensor_source)
        tensors = [
            tensor_source.load_tensor(name, torch.float32)[0]
            for name in sorted(self.tensor_names)
        ]
        return {self.name: sum(tensors)}


class FastSafetensorsFanoutTest(unittest.TestCase):
    def _make_loader(self, weights, database):
        loader = object.__new__(ModelLoader)
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._is_attn_model = False
        loader._misc_weights_info = []
        loader._model_weights_info = SimpleNamespace(
            layer_weights=[weights], weights=[]
        )
        exported_device = MagicMock()
        exported_device.maybe_rewrite_weight_by_key.side_effect = (
            lambda _name, tensor: tensor
        )
        loader._load_config = SimpleNamespace(
            database=database,
            vit_separation=VitSeparation.VIT_SEPARATION_LOCAL,
            num_layers=1,
            compute_dtype=torch.float32,
            merge_lora=False,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            exported_device=exported_device,
        )
        model_weights = MagicMock()
        loader._create_model_weights = MagicMock(return_value=model_weights)
        return loader, model_weights

    def test_generate_weight_info_keeps_all_consumers_of_shared_key(self):
        first = RecordingWeight("first", {"shared"})
        second = RecordingWeight("second", {"shared", "other"})
        loader, _ = self._make_loader([first, second], FakeDatabase([]))

        tensor_map, load_units = loader._generate_weight_info()

        self.assertEqual(len(load_units), 2)
        self.assertEqual(tensor_map["shared"], load_units)
        self.assertEqual(tensor_map["other"], [load_units[1]])

    def test_shared_and_duplicate_iterator_tensors_materialize_once(self):
        shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        database = FakeDatabase(
            [("shared", shared), ("shared", shared), ("unused", torch.ones(1))]
        )
        direct = AtomicWeight("direct", [CkptWeightInfo("shared", identity)], identity)
        transposed = AtomicWeight(
            "transposed", [CkptWeightInfo("shared", identity)], transpose
        )
        loader, model_weights = self._make_loader([direct, transposed], database)

        loader._load_from_fastsafetensor("cpu")

        calls = model_weights.set_layer_weight.call_args_list
        self.assertEqual(len(calls), 2)
        outputs = {call.args[1]: call.args[2] for call in calls}
        self.assertIs(outputs["direct"], shared)
        self.assertTrue(torch.equal(outputs["direct"], shared))
        self.assertTrue(torch.equal(outputs["transposed"], shared.T))

    def test_incomplete_collector_falls_back_once_after_partial_release(self):
        partial = torch.tensor([2.0])
        missing = torch.tensor([3.0])
        database = FakeDatabase(
            [("partial", partial)],
            fallback_tensors={"partial": partial, "missing": missing},
        )
        weight = RecordingWeight("combined", {"partial", "missing"})
        loader, model_weights = self._make_loader([weight], database)

        loader._load_from_fastsafetensor("cpu")

        self.assertEqual(len(weight.load_sources), 1)
        self.assertIsInstance(weight.load_sources[0], DatabaseTensorSource)
        output = model_weights.set_layer_weight.call_args.args[2]
        self.assertTrue(torch.equal(output, torch.tensor([5.0])))

    def test_multi_key_collector_loaded_from_iterator_without_fallback(self):
        database = FakeDatabase(
            [("first", torch.tensor([1.0])), ("second", torch.tensor([4.0]))]
        )
        weight = RecordingWeight("combined", {"first", "second"})
        loader, model_weights = self._make_loader([weight], database)

        loader._load_from_fastsafetensor("cpu")

        self.assertEqual(len(weight.load_sources), 1)
        self.assertNotIsInstance(weight.load_sources[0], DatabaseTensorSource)
        output = model_weights.set_layer_weight.call_args.args[2]
        self.assertTrue(torch.equal(output, torch.tensor([5.0])))


if __name__ == "__main__":
    unittest.main()
