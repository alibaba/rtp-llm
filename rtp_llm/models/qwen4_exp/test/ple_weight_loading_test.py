import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import (
    Qwen4ExpPleNgramWeight,
    Qwen4ExpWeight,
)
from rtp_llm.models_py.modules.qwen4_exp.ple import Qwen4ExpNGramEmbedding
from rtp_llm.utils.model_weight import CkptWeightInfo, W

_SHARD_COUNT = 128


class _RecordingTensorSource(TensorSource):
    def __init__(self):
        self.requests = []

    def load_tensor(self, name, data_type=torch.float16):
        self.requests.append(name)
        shard_id = int(name.split("shard_")[1].split(".")[0])
        return [torch.full((2, 1), shard_id, dtype=data_type)]

    def has_tensor(self, name):
        return True

    def get_database(self):
        return None


class _HashMetadataTensorSource(TensorSource):
    _VALUES = {
        "layer_multipliers": torch.tensor([23703573157769, 20109073645365]),
        "ngram_heads_offsets": torch.tensor([0, 5]),
        "ngram_heads_vocab_sizes": torch.tensor([5, 7]),
    }

    def load_tensor(self, name, data_type=torch.float16):
        key = next(key for key in self._VALUES if name.endswith(key))
        return [self._VALUES[key].to(data_type)]

    def has_tensor(self, name):
        return True

    def get_database(self):
        return None


def _load_config(tp_size=8, tp_rank=0):
    return SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        compute_dtype=torch.float32,
        merge_lora=False,
    )


def _ngram_weight():
    return Qwen4ExpPleNgramWeight(
        W.qwen4_ple_ngram_shards,
        [
            CkptWeightInfo(
                f"model.layers.{{i}}.ple.ngram_embedding.shard_{shard_id}.weight"
            )
            for shard_id in range(_SHARD_COUNT)
        ],
    )


class Qwen4ExpPleNgramWeightTest(unittest.TestCase):
    def test_tensor_names_are_selected_before_loading(self):
        weight = _ngram_weight()

        names = weight.get_tensor_names(layer_id=1, load_config=_load_config(8, 3))

        self.assertEqual(len(names), 16)
        self.assertEqual(
            names,
            {
                f"model.layers.1.ple.ngram_embedding.shard_{i}.weight"
                for i in range(48, 64)
            },
        )

    def test_load_keeps_local_shards_separate_and_never_stacks(self):
        weight = _ngram_weight()
        source = _RecordingTensorSource()

        with mock.patch.object(
            torch, "stack", side_effect=AssertionError("must not stack PLE table")
        ):
            loaded = weight.load(source, 1, "cpu", _load_config(8, 3))

        self.assertEqual(
            source.requests,
            [
                f"model.layers.1.ple.ngram_embedding.shard_{i}.weight"
                for i in range(48, 64)
            ],
        )
        self.assertEqual(
            set(loaded),
            {f"{W.qwen4_ple_ngram_shards}.{i}" for i in range(48, 64)},
        )
        for shard_id in range(48, 64):
            self.assertEqual(
                loaded[f"{W.qwen4_ple_ngram_shards}.{shard_id}"][0, 0].item(),
                shard_id,
            )

    def test_tp_one_is_rejected_instead_of_loading_the_full_table(self):
        source = _RecordingTensorSource()

        with self.assertRaisesRegex(ValueError, "refusing to load all 128"):
            _ngram_weight().load(source, 1, "cpu", _load_config(1, 0))

        self.assertEqual(source.requests, [])

    def test_shard_count_must_be_divisible_by_tp_size(self):
        with self.assertRaisesRegex(ValueError, "not divisible by tp_size=3"):
            _ngram_weight().get_tensor_names(1, _load_config(3, 0))

    def test_rank_local_descriptor_disables_unfiltered_fast_iteration(self):
        loader = object.__new__(ModelLoader)
        loader._model_weights_info = SimpleNamespace(
            weights=[], layer_weights=[[_ngram_weight()]]
        )
        loader._misc_weights_info = []

        self.assertFalse(loader._supports_fastsafetensors_iteration())

    def test_plain_descriptors_remain_fast_iteration_compatible(self):
        loader = object.__new__(ModelLoader)
        loader._model_weights_info = SimpleNamespace(
            weights=[AtomicWeight("plain", [CkptWeightInfo("plain.weight")])],
            layer_weights=[],
        )
        loader._misc_weights_info = []

        self.assertTrue(loader._supports_fastsafetensors_iteration())

    def test_explicit_fast_iteration_is_rejected_before_loading(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._model_weights_info = SimpleNamespace(
            weights=[], layer_weights=[[_ngram_weight()]]
        )
        loader._misc_weights_info = []

        with self.assertRaisesRegex(ValueError, "use load_method=scratch"):
            loader._load_weight("cpu")


class Qwen4ExpPleWeightGateTest(unittest.TestCase):
    def _weight(self, enabled):
        weight = object.__new__(Qwen4ExpWeight)
        weight.prefix = "model."
        weight.model_config = SimpleNamespace(
            enable_qwen4_ple=enabled,
            _qwen4_ple_layer_ids=[1],
            _qwen4_split_ngram_parts=_SHARD_COUNT,
        )
        return weight

    def test_disabled_ple_does_not_request_any_ple_weight(self):
        layer_weights = [[]]

        self._weight(False)._append_ple_weights(layer_weights)

        self.assertEqual(layer_weights, [[]])

    def test_enabled_ple_uses_rank_local_ngram_descriptor(self):
        layer_weights = [[]]

        self._weight(True)._append_ple_weights(layer_weights)

        descriptors = layer_weights[0]
        self.assertEqual(len(descriptors), 10)
        self.assertIsInstance(descriptors[-1], Qwen4ExpPleNgramWeight)
        self.assertEqual(len(descriptors[-1].weights), _SHARD_COUNT)

    def test_hash_metadata_stays_int64_from_loader_to_hashed_ids(self):
        layer_weights = [[]]
        self._weight(True)._append_ple_weights(layer_weights)
        descriptors = {weight.name: weight for weight in layer_weights[0]}
        source = _HashMetadataTensorSource()
        loaded = {}
        for name in (
            W.qwen4_ple_multipliers,
            W.qwen4_ple_ngram_offsets,
            W.qwen4_ple_ngram_vocab_sizes,
        ):
            loaded.update(
                descriptors[name]._load_raw_tensor(
                    source, layer_id=0, device="cpu", load_config=_load_config()
                )
            )
            self.assertEqual(loaded[name].dtype, torch.int64)

        embedding = Qwen4ExpNGramEmbedding(
            [torch.zeros(12, 1)],
            loaded[W.qwen4_ple_ngram_vocab_sizes],
            loaded[W.qwen4_ple_ngram_offsets],
            loaded[W.qwen4_ple_multipliers],
            ngram_size=2,
            eos_token_id=7,
        )
        hashed = embedding.hashed_ids(torch.tensor([[7, 1, 2]]), seq_len=2)
        self.assertEqual(hashed.dtype, torch.int64)
        self.assertEqual(tuple(hashed.shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
