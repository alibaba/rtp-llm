"""Actual safetensors reads, no PPU or distributed workers required."""

import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.new_models.deepseek_v4.ppu_globals import PpuHiddenShardEmbedding
from rtp_llm.models_py.new_models.deepseek_v4.ppu_mtp_weights import PpuMtpWeights


class PpuVocabTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1", "requires native FST"
    )
    def test_native_fst_reads_hidden_shards(self):
        from types import SimpleNamespace

        import fastsafetensors.cpp as native

        from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
        from rtp_llm.models_py.module_base import RtpModule

        self.assertTrue(native.__file__.endswith(".so"))
        src = torch.arange(56).reshape(7, 8).bfloat16()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.safetensors")
            save_file({"embed.weight": src}, path)
            for rank in range(2):
                root = RtpModule()
                root.embed = PpuHiddenShardEmbedding(7, 8, 2, rank)
                loader = NewModelLoader(
                    SimpleNamespace(model_type="deepseek_v4_mtp"),
                    NewLoaderConfig(device="cpu", load_method="fastsafetensors"),
                    model_path=directory,
                )
                include = lambda name: name == "embed.weight"
                resolver = loader._checkpoint_read_spec_resolver(root, include, None)
                root.load_weights(
                    loader._fastsafetensors_weights_iterator(
                        [path], "cpu", include, None, resolver
                    )
                )
                NewModelLoader._validate_loaded_weights(root)
                torch.testing.assert_close(
                    root.embed.weight, src.chunk(2, dim=1)[rank], rtol=0, atol=0
                )

    def test_replicated_mtp_fusion_weights_from_disk(self):
        from rtp_llm.models_py.model_loader import NewModelLoader

        tensors = {}
        for stem in ("enorm", "hnorm"):
            tensors[stem + ".weight"] = torch.arange(256, dtype=torch.bfloat16)
        for index, stem in enumerate(("e_proj", "h_proj")):
            tensors[stem + ".weight"] = (
                (torch.arange(256 * 256) % 127)
                .to(torch.uint8)
                .reshape(256, 256)
                .view(torch.float8_e4m3fn)
            )
            tensors[stem + ".scale"] = (
                (torch.arange(124, 128, dtype=torch.uint8) + index)
                .reshape(2, 2)
                .view(torch.float8_e8m0fnu)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.safetensors")
            save_file(tensors, path)
            owner = PpuMtpWeights(256, 1e-6, torch.bfloat16, object())
            with safe_open(path, framework="pt", device="cpu") as archive:
                owner.load_weights(
                    {key: archive.get_tensor(key) for key in archive.keys()}
                )
            NewModelLoader._validate_loaded_weights(owner)
            actual = dict(owner.named_parameters())
            self.assertEqual(set(actual), set(tensors))
            for key, tensor in tensors.items():
                self.assertTrue(
                    torch.equal(
                        tensor.view(torch.uint8), actual[key].view(torch.uint8)
                    ),
                    key,
                )

    def test_tp_rank_reads_its_hidden_columns_for_all_tokens(self):
        source = torch.arange(56, dtype=torch.bfloat16).view(7, 8)
        ids = torch.tensor([6, 0, 6, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.safetensors")
            save_file({"embed.weight": source}, path)
            outputs = []
            with safe_open(path, framework="pt", device="cpu") as archive:
                value = archive.get_slice("embed.weight")
                for rank in range(2):
                    leaf = PpuHiddenShardEmbedding(7, 8, 2, rank)
                    spec = leaf.checkpoint_read_spec("weight", value.get_shape())
                    self.assertEqual((spec.shard_dim, spec.shard_rank), (1, rank))
                    leaf.load_weights({"weight": spec.materialize(value)})
                    leaf.validate_weights_loaded()
                    outputs.append(leaf(ids))
            self.assertTrue(torch.equal(torch.cat(outputs, dim=-1), source[ids]))

    def test_full_source_and_misrouted_shard(self):
        source = torch.arange(56, dtype=torch.bfloat16).view(7, 8)
        owner = PpuHiddenShardEmbedding(7, 8, 2, 1)
        foreign = PpuHiddenShardEmbedding(7, 8, 2, 0).checkpoint_read_spec(
            "weight", source.shape
        )
        with self.assertRaisesRegex(ValueError, "different TP shard"):
            owner.load_weights({"weight": foreign.materialize(source)})
        owner.load_weights({"weight": source})
        self.assertTrue(torch.equal(owner.weight, source[:, 4:]))
        with self.assertRaises(ValueError):
            PpuHiddenShardEmbedding(7, 7, 2, 1)


if __name__ == "__main__":
    unittest.main()
