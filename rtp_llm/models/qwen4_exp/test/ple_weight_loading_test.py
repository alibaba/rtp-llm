import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import (
    Qwen4ExpPleNgramWeight,
    Qwen4ExpWeight,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding
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


def _ngram_weight(**kwargs):
    return Qwen4ExpPleNgramWeight(
        W.qwen4_ple_ngram_shards,
        [
            CkptWeightInfo(
                f"model.layers.{{i}}.ple.ngram_embedding.shard_{shard_id}.weight"
            )
            for shard_id in range(_SHARD_COUNT)
        ],
        **kwargs,
    )


class Qwen4ExpPleNgramWeightTest(unittest.TestCase):
    def test_fp8_residency_keeps_table_bytes_and_global_scale(self):
        class Fp8Source(_RecordingTensorSource):
            def load_tensor(self, name, data_type=torch.float16):
                if name == "scale":
                    self.requests.append(name)
                    return [torch.tensor([0.25], dtype=data_type)]
                return super().load_tensor(name, data_type)

        loaded = _ngram_weight(
            preserve_fp8=True, scale_weight=CkptWeightInfo("scale")
        ).load(Fp8Source(), 1, "cuda:0", _load_config(8, 3))
        self.assertEqual(loaded[W.qwen4_ple_ngram_scale].item(), 0.25)
        for index in range(48, 64):
            shard = loaded[f"{W.qwen4_ple_ngram_shards}.{index}"]
            self.assertEqual(shard.dtype, torch.float8_e4m3fn)
            self.assertEqual(shard.element_size(), 1)
            self.assertEqual(shard.device.type, "cpu")

    def test_real_scratch_preserves_host_scale_and_other_destinations(self):
        from rtp_llm.models_py.new_models.qwen4_exp.ple_residency_test import check_local_ple
        check_local_ple(self, "scratch", exported=False)

    def test_fixed_host_table_keeps_fp8_scale_on_cpu(self):
        class Source(_RecordingTensorSource):
            def load_tensor(self, name, data_type=torch.float16):
                if name == "scale":
                    return [torch.tensor([0.25], dtype=data_type)]
                return super().load_tensor(name, data_type)

        loaded = _ngram_weight(
            preserve_fp8=True,
            scale_weight=CkptWeightInfo("scale"),
        ).load(Source(), 1, "meta", _load_config())
        self.assertEqual(loaded[W.qwen4_ple_ngram_scale].device.type, "cpu")
        self.assertEqual(loaded[f"{W.qwen4_ple_ngram_shards}.0"].device.type, "cpu")

    def test_default_offload_never_transfers_table_to_execution_device(self):
        source = _RecordingTensorSource()
        # CUDA deliberately need not exist: touching the requested execution
        # device would fail, including WeightModule's final flattening step.
        loaded = _ngram_weight().load(source, 1, "cuda:0", _load_config(8, 3))

        self.assertEqual(len(source.requests), 16)
        self.assertTrue(all(t.device.type == "cpu" for t in loaded.values()))
        for shard_id in range(48, 64):
            self.assertEqual(
                loaded[f"{W.qwen4_ple_ngram_shards}.{shard_id}"][0, 0].item(),
                shard_id,
            )

    def test_table_residency_cannot_be_disabled(self):
        with self.assertRaisesRegex(TypeError, "residency is fixed to CPU"):
            _ngram_weight(cpu_offload=False)

    def test_update_preserves_cpu_residency(self):
        shards = {f"{W.qwen4_ple_ngram_shards}.0": torch.ones(2, 1)}

        updated = _ngram_weight().update(shards, "cuda:0", _load_config())

        self.assertEqual(updated[next(iter(updated))].device.type, "cpu")

    def test_rank_local_ft_preserves_host_table_and_auxiliary(self):
        from rtp_llm.models_py.new_models.qwen4_exp.ple_residency_test import check_local_ple
        for method in ("scratch", "fastsafetensors"):
            for preserved in (False, True):
                with self.subTest(method=method,preserve_fp8=preserved):
                    check_local_ple(self, method, exported=True, preserve=preserved)

    def test_fp8_global_scale_matches_lookup_cast_order(self):
        scalar = torch.tensor([0.00019931793212890625], dtype=torch.bfloat16)

        class Source(_RecordingTensorSource):
            def load_tensor(self, name, data_type=torch.float16):
                if name.endswith("weight_scale"):
                    self.requests.append(name)
                    return [scalar.to(data_type)]
                self.requests.append(name)
                shard = int(name.split("shard_")[1].split(".")[0])
                values = torch.tensor([[1.3], [float(shard)]]).to(torch.float8_e4m3fn)
                return [values.to(data_type)]

        source = Source()
        weight = _ngram_weight(
            scale_weight=CkptWeightInfo(
                "model.layers.{i}.ple.ngram_embedding.weight_scale"
            )
        )
        config = _load_config(8, 3)
        config.compute_dtype = torch.bfloat16
        names = weight.get_tensor_names(1, config)
        self.assertIn("model.layers.1.ple.ngram_embedding.weight_scale", names)
        with mock.patch.object(
            torch, "stack", side_effect=AssertionError("no table stack")
        ):
            loaded = weight.load(source, 1, "cuda:0", config)
        self.assertTrue(all(t.device.type == "cpu" for t in loaded.values()))
        self.assertEqual(len(source.requests), 17)
        for shard in range(48, 64):
            expected = (
                torch.tensor([[1.3], [float(shard)]])
                .to(torch.float8_e4m3fn)
                .to(torch.bfloat16)
                * scalar
            )
            self.assertTrue(
                torch.equal(loaded[f"{W.qwen4_ple_ngram_shards}.{shard}"], expected)
            )

    def test_fp8_invalid_global_scale_rejected_before_shard_io(self):
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                source = mock.Mock(spec=TensorSource)
                source.load_tensor.return_value = [torch.tensor([scale])]
                weight = _ngram_weight(
                    scale_weight=CkptWeightInfo(
                        "model.layers.{i}.ple.ngram_embedding.weight_scale"
                    )
                )
                with self.assertRaisesRegex(ValueError, "finite positive global scale"):
                    weight.load(source, 1, "cpu", _load_config())
                source.load_tensor.assert_called_once_with(
                    "model.layers.1.ple.ngram_embedding.weight_scale", torch.float32
                )

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

    def test_tp_one_cpu_loads_all_shards_without_gpu_staging_or_collective(self):
        source = _RecordingTensorSource()
        with mock.patch.object(torch, "stack", side_effect=AssertionError("no stack")):
            loaded = _ngram_weight().load(source, 1, "cuda:0", _load_config(1, 0))
        self.assertEqual(len(source.requests), 128)
        self.assertEqual(len(loaded), 128)
        self.assertTrue(all(t.device.type == "cpu" for t in loaded.values()))
        embedding = Qwen4ExpNGramEmbedding(
            [loaded[f"{W.qwen4_ple_ngram_shards}.{i}"] for i in range(128)],
            torch.tensor([256]),
            torch.tensor([0]),
            torch.tensor([1, 3]),
            ngram_size=2,
            eos_token_id=0,
            total_shards=128,
            distributed_reduce=False,
        )
        ids = torch.tensor([[0, 127, 128, 255]])
        self.assertTrue(
            torch.equal(embedding.gather(ids), torch.tensor([[0.0, 63.0, 64.0, 127.0]]))
        )

    def test_tp_one_gpu_table_rejected_before_io(self):
        source = _RecordingTensorSource()
        with self.assertRaisesRegex(TypeError, "residency is fixed to CPU"):
            _ngram_weight(cpu_offload=False).load(source, 1, "cpu", _load_config(1, 0))
        self.assertEqual(source.requests, [])

    def test_shard_count_must_be_divisible_by_tp_size(self):
        with self.assertRaisesRegex(ValueError, "not divisible by tp_size=3"):
            _ngram_weight().get_tensor_names(1, _load_config(3, 0))

    def test_explicit_fst_reads_only_local_ple_and_still_loads_normal_weights(self):
        from rtp_llm.models_py.new_models.qwen4_exp.ple_residency_test import check_local_ple
        check_local_ple(self, "fastsafetensors", exported=False)


class Qwen4ExpPleWeightGateTest(unittest.TestCase):
    def _weight(self, enabled, **kwargs):
        weight = object.__new__(Qwen4ExpWeight)
        weight.prefix = "model."
        weight.model_config = SimpleNamespace(
            enable_qwen4_ple=enabled,
            _qwen4_ple_layer_ids=[1],
            _qwen4_split_ngram_parts=_SHARD_COUNT,
            **kwargs,
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
        self.assertEqual(descriptors[-1].resident_device, "cpu")

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
