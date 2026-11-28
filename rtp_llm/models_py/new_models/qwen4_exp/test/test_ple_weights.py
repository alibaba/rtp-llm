"""Real CPU checkpoint tests; not a full Qwen4 model or GPU runtime test."""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.ple_weights import (
    Qwen4PLEEmbeddingWeights,
    Qwen4PLEHostTable,
)
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor, ReadSpec
from rtp_llm.models_py.registry import register_model


@register_model("qwen4_ple_storage_fixture")
class _PLECheckpointModel(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.ple_embedding = Qwen4PLEEmbeddingWeights(
            ngram_size=3,
            heads_per_ngram=2,
            shard_count=config.get("shard_count", 8),
            embedding_dim=4,
            tp_size=load_config.tp_size,
            tp_rank=load_config.tp_rank,
            dtype=load_config.compute_dtype,
            source_fp8=config.get("source_fp8", False),
            preserve_fp8=config.get("preserve_fp8", False),
        )

    def checkpoint_weight_name_filter(self):
        return self.ple_embedding.checkpoint_name_filter("ple_embedding.")


class PLECheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environ = mock.patch.dict(os.environ, {"LOAD_METHOD": ""})
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def weights(self, shard_count=8, dtype=torch.bfloat16):
        result = {
            "ple_embedding.layer_multipliers": torch.tensor(
                [2**44 + 123, 2**43 + 7, 2**42 + 13], dtype=torch.int64
            ),
            "ple_embedding.ngram_heads_offsets": torch.tensor([0, 3, 8, 15]),
            "ple_embedding.ngram_heads_vocab_sizes": torch.tensor([3, 5, 7, 11]),
        }
        for index in range(shard_count):
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4) + index / 8
            result[f"ple_embedding.ngram_embedding.shard_{index}.weight"] = values.to(
                dtype
            )
        if dtype == torch.float8_e4m3fn:
            result["ple_embedding.ngram_embedding.weight_scale"] = torch.tensor(
                0.1234567
            )
        return result

    def write(self, weights):
        # Each source shard has its own file, as large real checkpoints may.
        # An index lets the ordinary loader exclude foreign TP files up front.
        weight_map = {}
        for index, (name, tensor) in enumerate(weights.items()):
            filename = f"weights-{index:04d}.safetensors"
            save_file({name: tensor}, os.path.join(self.temp.name, filename))
            weight_map[name] = filename
        with open(
            os.path.join(self.temp.name, "model.safetensors.index.json"), "w"
        ) as stream:
            json.dump({"metadata": {}, "weight_map": weight_map}, stream)

    def loader(
        self, *, tp=1, rank=0, load_method=NewLoaderLoadMethod.SCRATCH, **config
    ):
        return NewModelLoader(
            {"model_type": "qwen4_ple_storage_fixture", **config},
            NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.bfloat16,
                load_method=load_method,
                tp_size=tp,
                tp_rank=rank,
            ),
            model_path=self.temp.name,
        )

    def test_real_checkpoint_name_partition_selects_only_local_files(self):
        weights = self.weights(shard_count=128)
        self.write(weights)
        for rank in (0, 3, 7):
            with self.subTest(rank=rank):
                observed = []
                actual = weight_mapper.get_all_weights

                def capture(files, **kwargs):
                    observed.extend(files)
                    return actual(files, **kwargs)

                with mock.patch.object(
                    weight_mapper, "get_all_weights", side_effect=capture
                ):
                    model = self.loader(tp=8, rank=rank, shard_count=128).load()
                self.assertEqual(len(observed), 3 + 16)
                embedding = model.ple_embedding
                table = embedding.ngram_embedding
                self.assertEqual(
                    table.shard_indices, tuple(range(rank * 16, (rank + 1) * 16))
                )
                self.assertEqual(len(list(table.parameters())), 16)
                for index, tensor in zip(table.shard_indices, table.runtime_shards()):
                    self.assertEqual(tensor.device.type, "cpu")
                    torch.testing.assert_close(
                        tensor,
                        weights[f"ple_embedding.ngram_embedding.shard_{index}.weight"],
                    )
                torch.testing.assert_close(
                    embedding.layer_multipliers,
                    weights["ple_embedding.layer_multipliers"],
                )
                self.assertEqual(embedding.layer_multipliers.dtype, torch.int64)
                arguments = embedding.runtime_arguments(eos_token_id=2)
                self.assertTrue(arguments["distributed_reduce"])
                self.assertEqual(arguments["total_shards"], 128)

    def test_host_table_ignores_parent_dtype_and_device_apply(self):
        weights = self.weights(dtype=torch.float8_e4m3fn)
        self.write(weights)
        model = self.loader(source_fp8=True, preserve_fp8=True).load()
        table = model.ple_embedding.ngram_embedding
        before = [(id(t), t.data_ptr(), t.dtype) for t in table.runtime_shards()]
        scale_ptr = table.weight_scale.data_ptr()
        model.to(dtype=torch.float16)
        table._apply(
            lambda tensor: (_ for _ in ()).throw(AssertionError("host tensor moved"))
        )
        self.assertEqual(
            before, [(id(t), t.data_ptr(), t.dtype) for t in table.runtime_shards()]
        )
        self.assertEqual(table.weight_scale.dtype, torch.float32)
        self.assertEqual(table.weight_scale.data_ptr(), scale_ptr)
        # The normal FST destination allocator must not reserve GPU (or meta)
        # storage for the registered host table parameters.
        NewModelLoader._prepare_model_for_fastsafetensors(table, "meta")
        self.assertEqual(
            before, [(id(t), t.data_ptr(), t.dtype) for t in table.runtime_shards()]
        )

    def test_fp8_source_preserves_bytes_scale_and_integer_hashes(self):
        weights = self.weights(dtype=torch.float8_e4m3fn)
        self.write(weights)
        model = self.loader(tp=2, rank=1, source_fp8=True, preserve_fp8=True).load()
        table = model.ple_embedding.ngram_embedding
        for index, tensor in zip(table.shard_indices, table.runtime_shards()):
            self.assertTrue(
                torch.equal(
                    tensor.view(torch.uint8),
                    weights[f"ple_embedding.ngram_embedding.shard_{index}.weight"].view(
                        torch.uint8
                    ),
                )
            )
        arguments = model.ple_embedding.runtime_arguments(eos_token_id=2)
        self.assertIs(arguments["weight_scale"], table.weight_scale)
        self.assertEqual(
            arguments["multipliers"].tolist(), [2**44 + 123, 2**43 + 7, 2**42 + 13]
        )

    def test_fp8_dequant_casts_before_multiply_once(self):
        weights = self.weights(dtype=torch.float8_e4m3fn)
        self.write(weights)
        model = self.loader(tp=2, rank=0, source_fp8=True).load()
        table = model.ple_embedding.ngram_embedding
        pointers = [tensor.data_ptr() for tensor in table.runtime_shards()]
        for index, tensor in zip(table.shard_indices, table.runtime_shards()):
            reference = weights[
                f"ple_embedding.ngram_embedding.shard_{index}.weight"
            ].to(torch.bfloat16) * weights[
                "ple_embedding.ngram_embedding.weight_scale"
            ].to(
                torch.bfloat16
            )
            self.assertTrue(torch.equal(tensor, reference))
        table.process_weights_after_loading()
        self.assertEqual(
            pointers, [tensor.data_ptr() for tensor in table.runtime_shards()]
        )
        self.assertIsNone(
            model.ple_embedding.runtime_arguments(eos_token_id=2)["weight_scale"]
        )

    def test_release_bf16_scale_normalized_after_real_scratch_and_fst(self):
        # Actual Qwen3.8-Flash-Next-FP8 release global scale: BF16 [1],
        # model-00005-of-00131.safetensors. The table remains raw E4M3.
        weights = self.weights(dtype=torch.float8_e4m3fn)
        key = "ple_embedding.ngram_embedding.weight_scale"
        weights[key] = torch.tensor([0.00019931793212890625], dtype=torch.bfloat16)
        self.write(weights)
        for method in checkpoint_readers(
            NewLoaderLoadMethod.SCRATCH,
            NewLoaderLoadMethod.FASTSAFETENSORS,
        ):
            for preserve in (False, True):
                with self.subTest(method=method, preserve=preserve):
                    loader = self.loader(
                        source_fp8=True, preserve_fp8=preserve, load_method=method
                    )
                    model = loader.load()
                    self.assertEqual(loader.loaded_method, method)
                    table = model.ple_embedding.ngram_embedding
                    self.assertEqual(table.weight_scale.dtype, torch.float32)
                    self.assertEqual(table.weight_scale.shape, torch.Size([]))
                    torch.testing.assert_close(
                        table.weight_scale,
                        weights[key].float().reshape(()),
                        rtol=0,
                        atol=0,
                    )
                    for index, tensor in zip(table.shard_indices, table.runtime_shards()):
                        raw = weights[
                            f"ple_embedding.ngram_embedding.shard_{index}.weight"
                        ]
                        if preserve:
                            torch.testing.assert_close(
                                tensor.view(torch.uint8),
                                raw.view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )
                        else:
                            # OldLoader converts the scale to FP32 on read, then
                            # casts both operands to compute dtype before mul.
                            reference = raw.bfloat16() * weights[key].float().bfloat16()
                            torch.testing.assert_close(
                                tensor, reference, rtol=0, atol=0
                            )
                    before = [(id(t), t.data_ptr()) for t in table.runtime_shards()]
                    scale_ptr = table.weight_scale.data_ptr()
                    table.process_weights_after_loading()
                    self.assertEqual(
                        before, [(id(t), t.data_ptr()) for t in table.runtime_shards()]
                    )
                    self.assertEqual(scale_ptr, table.weight_scale.data_ptr())

    def test_supported_floating_scale_normalizes_without_changing_value(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for shape in ((), (1,)):
                with self.subTest(dtype=dtype, shape=shape):
                    table = Qwen4PLEHostTable(
                        8, 4, source_fp8=True, preserve_fp8=True
                    )
                    scale = torch.full(shape, 0.25, dtype=dtype)
                    table.load_weights({"weight_scale": scale})
                    self.assertEqual(table.weight_scale.dtype, torch.float32)
                    self.assertEqual(table.weight_scale.item(), 0.25)

    def test_missing_local_shard_and_inconsistent_rows_fail(self):
        table = Qwen4PLEHostTable(8, 4, tp_size=2, tp_rank=0)
        table.load_weights({"shard_0.weight": torch.ones(4, 4, dtype=torch.bfloat16)})
        with self.assertRaisesRegex(RuntimeError, "Missing PLE"):
            table.validate_weights_loaded()
        with self.assertRaisesRegex(ValueError, "equal row"):
            table.load_weights(
                {"shard_1.weight": torch.ones(3, 4, dtype=torch.bfloat16)}
            )
        with self.assertRaisesRegex(ValueError, "Non-local"):
            table.load_weights(
                {"shard_7.weight": torch.ones(4, 4, dtype=torch.bfloat16)}
            )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            table.load_weights(
                {"shard_0.weight": torch.ones(4, 4, dtype=torch.bfloat16)}
            )

    def test_each_host_shard_owns_storage_without_concatenation(self):
        table = Qwen4PLEHostTable(8, 4, tp_size=4, tp_rank=1)
        parameter = table.shard_2
        first = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        expected = first.clone()
        with mock.patch.object(
            torch, "cat", side_effect=AssertionError("full table concat")
        ):
            table.load_weights({"shard_2.weight": first})
            table.load_weights({"shard_3.weight": first + 1})
            table.process_weights_after_loading()
        self.assertIs(table.shard_2, parameter)
        self.assertNotEqual(table.shard_2.data_ptr(), first.data_ptr())
        first.zero_()
        torch.testing.assert_close(table.shard_2, expected)
        self.assertNotEqual(table.shard_2.data_ptr(), table.shard_3.data_ptr())

    def test_runtime_host_storage_adoption_validates_all_before_publishing(self):
        weights = self.weights()
        self.write(weights)
        model = self.loader(tp=4, rank=0).load()
        table = model.ple_embedding.ngram_embedding
        original = table.runtime_shards()
        identities = [id(tensor) for tensor in original]
        pointers = [tensor.data_ptr() for tensor in original]
        # CPU clones stand in only for the allocations returned by UVA setup;
        # this test does not claim pinned memory, a CUDA kernel or GPU execution.
        returned = [tensor.clone() for tensor in original]
        with self.assertRaisesRegex(ValueError, "shape/dtype/layout"):
            table.adopt_runtime_shards([returned[0], returned[1].to(torch.float32)])
        self.assertEqual(pointers, [tensor.data_ptr() for tensor in original])
        table.adopt_runtime_shards(returned)
        self.assertEqual(identities, [id(tensor) for tensor in table.runtime_shards()])
        self.assertEqual(
            [tensor.data_ptr() for tensor in returned],
            [tensor.data_ptr() for tensor in table.runtime_shards()],
        )
        self.assertTrue(all(t._rtp_requires_restart_for_update for t in original))

    def test_invalid_scale_dtype_and_topology_fail_before_binding(self):
        for scale in (
            torch.tensor(float("nan")),
            torch.tensor(float("inf")),
            torch.tensor(0.0),
            torch.tensor(-1.0),
            torch.empty(0),
            torch.ones(2),
            torch.ones(1, 1),
            torch.tensor(1, dtype=torch.int64),
            torch.tensor(True),
            torch.tensor(1e100, dtype=torch.float64),
            torch.tensor(1e-100, dtype=torch.float64),
        ):
            with self.subTest(scale=scale):
                table = Qwen4PLEHostTable(8, 4, source_fp8=True, preserve_fp8=True)
                with self.assertRaisesRegex(ValueError, "finite positive FP32"):
                    table.load_weights({"weight_scale": scale})
        with self.assertRaisesRegex(ValueError, "divisible"):
            Qwen4PLEHostTable(8, 4, tp_size=3)
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            Qwen4PLEHostTable(8, 4, tp_size=2, tp_rank=2)
        table = Qwen4PLEHostTable(8, 4, source_fp8=True)
        with self.assertRaisesRegex(TypeError, "E4M3"):
            table.load_weights({"shard_0.weight": torch.ones(4, 4)})
        partial = ReadSpec("x", "x", (8, 4), slices=(slice(0, 4), slice(None)))
        with self.assertRaisesRegex(ValueError, "second time"):
            table.load_weights(
                {"shard_0.weight": LoadedCheckpointTensor(torch.ones(4, 4), partial)}
            )

    def test_hash_float_cast_and_out_of_capacity_fail(self):
        for bad_name, bad_value, message in (
            (
                "ple_embedding.layer_multipliers",
                torch.ones(3, dtype=torch.bfloat16),
                "Dtype mismatch",
            ),
            (
                "ple_embedding.ngram_heads_vocab_sizes",
                torch.tensor([3, 5, 7, 40]),
                "cover hashed",
            ),
        ):
            with self.subTest(name=bad_name):
                weights = self.weights()
                weights[bad_name] = bad_value
                self.write(weights)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.loader().load()


if __name__ == "__main__":
    unittest.main()
