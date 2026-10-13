import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.utils import ckpt_file_info
from rtp_llm.utils.database import _LAYER_RE, CkptDatabase


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


class IndexedExtrasTest(unittest.TestCase):
    """Sharded releases: extras beside the index must not duplicate the shards."""

    @staticmethod
    def _write_indexed_release(tmp):
        save_file(
            {"backbone.w1": torch.zeros(1), "backbone.w2": torch.zeros(1)},
            os.path.join(tmp, "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"backbone.w3": torch.zeros(1)},
            os.path.join(tmp, "model-00002-of-00002.safetensors"),
        )
        weight_map = {
            "backbone.w1": "model-00001-of-00002.safetensors",
            "backbone.w2": "model-00001-of-00002.safetensors",
            "backbone.w3": "model-00002-of-00002.safetensors",
        }
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)

    def _file_names(self, db):
        return [os.path.basename(f.file_name) for f in db.pretrain_file_list]

    def test_head_file_beside_index_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_indexed_release(tmp)
            save_file(
                {"decision.head.weight": torch.zeros(1)},
                os.path.join(tmp, "joint_head.safetensors"),
            )
            db = CkptDatabase(tmp)
            # the shards come out of the index's weight_map set in any
            # order; only the enumerated extras are guaranteed to follow
            names = self._file_names(db)
            self.assertEqual(
                sorted(names[:-1]),
                [
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            )
            self.assertEqual(names[-1], "joint_head.safetensors")
            self.assertTrue(db.has_tensor("decision.head.weight"))

    def test_consolidated_beside_index_is_skipped(self):
        # Mistral-Nemo-Instruct-2407 ships consolidated.safetensors next to
        # the index; loading it would duplicate every backbone tensor and
        # override the shards in _tensor_index.
        with tempfile.TemporaryDirectory() as tmp:
            self._write_indexed_release(tmp)
            save_file(
                {
                    "backbone.w1": torch.ones(1),
                    "backbone.w2": torch.ones(1),
                    "backbone.w3": torch.ones(1),
                },
                os.path.join(tmp, "consolidated.safetensors"),
            )
            db = CkptDatabase(tmp)
            self.assertNotIn("consolidated.safetensors", self._file_names(db))
            all_names = db.get_pretrain_tensor_names()
            self.assertEqual(len(all_names), len(set(all_names)))
            self.assertIn(
                "model-00001", db._tensor_index["backbone.w1"].file_name
            )

    def test_extra_with_duplicate_tensors_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_indexed_release(tmp)
            # a reshard remnant: one new tensor plus one the index covers
            save_file(
                {"decision.head.weight": torch.zeros(1), "backbone.w1": torch.zeros(1)},
                os.path.join(tmp, "model.safetensors"),
            )
            db = CkptDatabase(tmp)
            self.assertNotIn("model.safetensors", self._file_names(db))
            all_names = db.get_pretrain_tensor_names()
            self.assertEqual(len(all_names), len(set(all_names)))

    def test_two_sidecars_with_same_head_key_keeps_first(self):
        # codex round1 B11: two extras both defining the same head tensor must
        # not both load (duplicate names would disable fast loaders and flip
        # the _tensor_index winner to whichever appended last).
        with tempfile.TemporaryDirectory() as tmp:
            self._write_indexed_release(tmp)
            save_file(
                {"decision.head.weight": torch.zeros(1)},
                os.path.join(tmp, "a_head.safetensors"),
            )
            save_file(
                {"decision.head.weight": torch.ones(1)},
                os.path.join(tmp, "z_head.safetensors"),
            )
            db = CkptDatabase(tmp)
            names = self._file_names(db)
            self.assertIn("a_head.safetensors", names)
            self.assertNotIn("z_head.safetensors", names)
            all_names = db.get_pretrain_tensor_names()
            self.assertEqual(len(all_names), len(set(all_names)))
            # the accepted sidecar is the authoritative source for its key
            self.assertIn("a_head", db._tensor_index["decision.head.weight"].file_name)


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


class HandleRecyclingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RTP_LLM_REQUIRE_ACCELERATOR") and torch.version.hip is None:
            raise AssertionError("ROCm target is not running on a ROCm build")

    @staticmethod
    def _write_shards(tmp):
        # float32 avoids .to() conversion, so only copy-out detaches tensors.
        for layer in (0, 1, 2):
            save_file(
                {f"model.layers.{layer}.weight": torch.tensor([float(layer)])},
                os.path.join(tmp, f"model-{layer}.safetensors"),
            )

    def test_recycling_enabled_on_real_rocm_build(self):
        # The ROCm target reaches this without patching the production gate.
        if torch.version.hip is None:
            self.skipTest("requires a ROCm build")
        with tempfile.TemporaryDirectory() as tmp:
            self._write_shards(tmp)
            db = CkptDatabase(tmp, recycle_handles=True)
            self.assertTrue(db._recycle_handles)
            self.assertIsNone(self._read_layers_0_and_2(db)._st_handle)

    def test_consumed_shard_closes_and_reopens(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_shards(tmp)
            # Copy-out keeps returned tensors valid after their handle closes.
            with patch.object(ckpt_file_info, "ROCM_COPY_OUT", True):
                db = CkptDatabase(tmp, recycle_handles=True)
                name, whole = "model.layers.0.weight", (slice(None),)
                first = db._tensor_index[name]
                tensor = db.load_tensor(name, torch.float32)[0]
                sliced = db.load_tensor_slice(name, whole, torch.float32)
                expected, expected_slice = tensor.clone(), sliced.clone()
                db.load_tensor("model.layers.1.weight", torch.float32)
                self.assertIsNotNone(first._st_handle)  # one-layer slack
                db.load_tensor_slice("model.layers.2.weight", whole, torch.float32)
                self.assertIsNone(first._st_handle)
                torch.testing.assert_close(tensor, expected)
                torch.testing.assert_close(sliced, expected_slice)
                reread = db.load_tensor_slice(name, whole, torch.float32)
                torch.testing.assert_close(reread, expected_slice)
                self.assertIsNotNone(first._st_handle)

    def _read_layers_0_and_2(self, db):
        first = db._tensor_index["model.layers.0.weight"]
        db.load_tensor("model.layers.0.weight")
        db.load_tensor("model.layers.2.weight")
        return first

    def test_switch_and_checkpoint_format_gate_recycling(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_shards(tmp)
            # Either gate off must disable recycling and keep shard handles open.
            for copy_out, asked in ((True, False), (False, True)):
                with patch.object(ckpt_file_info, "ROCM_COPY_OUT", copy_out):
                    db = CkptDatabase(tmp, recycle_handles=asked)
                    self.assertFalse(db._recycle_handles)
                    self.assertIsNotNone(self._read_layers_0_and_2(db)._st_handle)

        with tempfile.TemporaryDirectory() as tmp:
            torch.save({"w": torch.ones(1)}, os.path.join(tmp, "pytorch_model.bin"))
            with patch.object(ckpt_file_info, "ROCM_COPY_OUT", True):
                db = CkptDatabase(tmp, recycle_handles=True)
                self.assertFalse(db._recycle_handles)

    def test_layer_name_matching_is_bounded(self):
        for name in ("model.layers.3.w", "h.3.w", "model.blocks.3.w", "layer.3.w"):
            self.assertEqual(_LAYER_RE.search(name).group(1), "3", name)
        # No layer number anywhere means recycling stays off for that checkpoint.
        for name in ("model.embed_tokens.weight", "model.sublayers.3.w"):
            self.assertIsNone(_LAYER_RE.search(name), name)


if __name__ == "__main__":
    unittest.main()
