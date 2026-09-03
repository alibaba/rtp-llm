import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.utils.database import CkptDatabase


class CkptDataBaseTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "pt_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(path + "/test.pt", database.pretrain_file_list[0].file_name)
        self.assertEqual(36, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/test.safetensors", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(28, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "bin_testdata")
        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database = CkptDatabase(path)
        database.load_lora("test", lora_path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        self.assertEqual(8, list(database.lora_ckpt.LoraFileList)[0].rank)
        self.assertEqual(8, list(database.lora_ckpt.LoraFileList)[0].lora_alpha)
        self.assertEqual(0.0, list(database.lora_ckpt.LoraFileList)[0].lora_dropout)
        self.assertEqual(
            ["c_proj", "w2", "c_attn", "w1"],
            list(database.lora_ckpt.LoraFileList)[0].target_modules,
        )
        self.assertEqual(1, len(list(database.lora_ckpt.LoraFileList.values())[0]))
        self.assertEqual(
            12,
            len(
                list(database.lora_ckpt.LoraFileList.values())[0][0].get_tensor_names()
            ),
        )

    def test_mix_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "mixture_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/test.safetensors", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(28, len(database.pretrain_file_list[0].get_tensor_names()))


class LoraTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))

        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(["c_proj", "w2", "c_attn", "w1"], lora_config.target_modules)
        self.assertEqual(1, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

        self.assertTrue(database.remove_lora("test_name"))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(0, lora_config.rank)
        self.assertEqual(0, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual([], lora_config.target_modules)
        self.assertEqual(0, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(0, len(database.get_lora_tensor_names("test_name")))

        lora_path = os.path.join(self._testdata_path(), "lora_testdata_safetensor")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(["c_proj", "w2", "c_attn", "w1"], lora_config.target_modules)
        self.assertEqual(1, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))


class TensorIndexTest(unittest.TestCase):
    """Tests for the O(1) _tensor_index lookup introduced in CkptDatabase."""

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_tensor_index_lookup(self):
        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)

        # _tensor_index should contain all tensor names
        all_names = database.get_pretrain_tensor_names()
        for name in all_names:
            self.assertIn(name, database._tensor_index)

        # has_tensor should return True for known tensors, False for unknown
        self.assertTrue(database.has_tensor(all_names[0]))
        self.assertFalse(database.has_tensor("nonexistent.weight"))

        # load_tensor should return a non-empty list for known tensors
        result = database.load_tensor(all_names[0])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)

        # load_tensor for unknown tensors should return empty list
        result = database.load_tensor("nonexistent.weight")
        self.assertEqual(len(result), 0)

    def test_tensor_index_cleanup(self):
        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)

        self.assertGreater(len(database._tensor_index), 0)
        database._tensor_index.clear()
        self.assertEqual(len(database._tensor_index), 0)
        # After clearing, has_tensor should return False
        all_names = database.get_pretrain_tensor_names()
        self.assertFalse(database.has_tensor(all_names[0]))


class IndexedSafetensorManifestTest(unittest.TestCase):
    def _make_indexed_checkpoint(self, path):
        shards = {
            "model-00002-of-00002.safetensors": {"layer.1.weight": torch.ones(2)},
            "model-00001-of-00002.safetensors": {
                "layer.0.weight": torch.zeros(2, dtype=torch.int32)
            },
        }
        for shard_name, tensors in shards.items():
            save_file(tensors, os.path.join(path, shard_name))
        index = {
            "metadata": {"total_size": 16},
            "is_ft_style_weight": True,
            "__env__params__": {"inter_size": 128},
            "weight_map": {
                "layer.1.weight": "model-00002-of-00002.safetensors",
                "layer.0.weight": "model-00001-of-00002.safetensors",
            },
        }
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as writer:
            json.dump(index, writer)

    def test_indexed_init_does_not_read_shard_headers_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as path:
            self._make_indexed_checkpoint(path)
            with patch(
                "rtp_llm.utils.ckpt_file_info.CkptFileInfo._load_meta",
                autospec=True,
            ) as load_meta:
                database = CkptDatabase(path)

            load_meta.assert_not_called()
            self.assertEqual(
                [
                    os.path.basename(info.file_name)
                    for info in database.pretrain_file_list
                ],
                [
                    "model-00002-of-00002.safetensors",
                    "model-00001-of-00002.safetensors",
                ],
            )
            self.assertEqual(
                database.get_pretrain_tensor_names(),
                ["layer.1.weight", "layer.0.weight"],
            )
            self.assertTrue(database.has_tensor("layer.0.weight"))
            self.assertTrue(database.is_ft_style)
            self.assertEqual(database.ft_weight_params, {"inter_size": 128})
            self.assertIsNone(database._hf_index_data)
            self.assertEqual(database.get_tensor_type("layer.0.weight"), torch.int32)
            with self.assertRaises(KeyError):
                database.get_tensor_type("missing.weight")

    def test_read_order_loads_only_requested_header_once(self):
        with tempfile.TemporaryDirectory() as path:
            self._make_indexed_checkpoint(path)
            database = CkptDatabase(path)
            shard = database._tensor_index["layer.0.weight"]
            other_shard = database._tensor_index["layer.1.weight"]

            self.assertFalse(shard._metadata_loaded)
            self.assertFalse(other_shard._metadata_loaded)
            self.assertEqual(database.get_tensor_order("layer.0.weight")[0][1], 0)
            self.assertTrue(shard._metadata_loaded)
            self.assertFalse(other_shard._metadata_loaded)
            metadata = shard.metadata
            self.assertEqual(database.get_tensor_order("layer.0.weight")[0][1], 0)
            self.assertIs(shard.metadata, metadata)

    def test_indexed_tensor_load_does_not_require_offset_metadata(self):
        with tempfile.TemporaryDirectory() as path:
            self._make_indexed_checkpoint(path)
            database = CkptDatabase(path)

            tensor = database.load_tensor("layer.1.weight", torch.float32)[0]
            self.assertTrue(torch.equal(tensor, torch.ones(2)))
            self.assertFalse(database._tensor_index["layer.1.weight"]._metadata_loaded)

    def test_filter_limits_bulk_shards_but_preserves_targeted_fallback(self):
        with tempfile.TemporaryDirectory() as path:
            self._make_indexed_checkpoint(path)
            database = CkptDatabase(path)

            database.filter_by_tensor_name_regexes([re.compile(r"layer\.0\.weight")])

            self.assertEqual(len(database.pretrain_file_list), 1)
            self.assertTrue(database.has_tensor("layer.1.weight"))
            tensor = database.load_tensor("layer.1.weight", torch.float32)[0]
            self.assertTrue(torch.equal(tensor, torch.ones(2)))


class SafetensorHandleCacheTest(unittest.TestCase):
    """Tests for CkptFileInfo safetensor handle caching."""

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_handle_cache_returns_same_object(self):
        from rtp_llm.utils.ckpt_file_info import CkptFileInfo

        path = os.path.join(
            self._testdata_path(), "safetensor_testdata", "test.safetensors"
        )
        info = CkptFileInfo(file_name=path)

        h1 = info._get_safetensor_handle()
        h2 = info._get_safetensor_handle()
        self.assertIs(h1, h2)

    def test_close_handle_clears_cache(self):
        from rtp_llm.utils.ckpt_file_info import CkptFileInfo

        path = os.path.join(
            self._testdata_path(), "safetensor_testdata", "test.safetensors"
        )
        info = CkptFileInfo(file_name=path)

        info._get_safetensor_handle()
        self.assertIsNotNone(info._st_handle)

        info.close_safetensor_handle()
        self.assertIsNone(info._st_handle)

        # Can reopen after close
        h = info._get_safetensor_handle()
        self.assertIsNotNone(h)
        info.close_safetensor_handle()


if __name__ == "__main__":
    unittest.main()
