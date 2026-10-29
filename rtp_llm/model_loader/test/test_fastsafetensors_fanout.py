import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
)
from rtp_llm.model_loader.load_config import LoadMethod
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


class FastSafeTensorsMemoryTest(unittest.TestCase):
    def choose_load_method(self, quant_config, free_gib, moe_params=0):
        gib = 1024**3
        loader = ModelLoader.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._weights_info = SimpleNamespace(
            _quant_config=quant_config,
            model_config=SimpleNamespace(
                eval_model_weight_size=lambda: 8 * gib,
                quant_algo=SimpleNamespace(getWeightBits=lambda: 8),
                moe_weight_param_count=lambda: moe_params,
                layer_weight_param_count=lambda: 100,
            ),
        )
        loader._load_config = SimpleNamespace(
            ep_size=1,
            tp_size=1,
            exported_device=SimpleNamespace(
                get_mem_info=lambda: SimpleNamespace(free=free_gib * gib)
            ),
            database=SimpleNamespace(
                is_safetensor=True,
                get_max_file_size=lambda: gib,
                get_pretrain_tensor_names=lambda: ["weight"],
            ),
        )
        loader._choose_weight_convert_device = lambda _: "cuda"
        loader._load_from_fastsafetensor = lambda _: LoadMethod.FASTSAFETENSORS
        loader._load_from_scratch = lambda _: LoadMethod.SCRATCH
        with patch("rtp_llm.model_loader.loader.has_module", return_value=True):
            return loader._load_weight("cuda")

    def test_online_block_quantization_reserves_bf16_inputs(self):
        online = Fp8BlockWiseQuantConfig(is_quanted=False)
        self.assertEqual(self.choose_load_method(online, 16), LoadMethod.SCRATCH)
        self.assertEqual(
            self.choose_load_method(online, 24), LoadMethod.FASTSAFETENSORS
        )

    def test_prequantized_checkpoint_does_not_reserve_bf16_inputs(self):
        offline = Fp8BlockWiseQuantConfig(is_quanted=True)
        self.assertEqual(
            self.choose_load_method(offline, 16), LoadMethod.FASTSAFETENSORS
        )

    def test_inline_ptpc_reserves_only_non_inline_weight_fraction(self):
        online = Fp8PerChannelCompressedQuantConfig(is_quanted=False)
        self.assertEqual(self.choose_load_method(online, 16), LoadMethod.SCRATCH)
        self.assertEqual(
            self.choose_load_method(online, 16, moe_params=50), LoadMethod.SCRATCH
        )
        self.assertEqual(
            self.choose_load_method(online, 16, moe_params=75),
            LoadMethod.FASTSAFETENSORS,
        )


if __name__ == "__main__":
    unittest.main()
