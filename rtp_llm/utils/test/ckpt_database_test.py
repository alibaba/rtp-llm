import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.utils import ckpt_file_info
from rtp_llm.utils import database as database_module
from rtp_llm.utils.ckpt_file_info import CkptMetaCache
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


class PageCacheShedTest(unittest.TestCase):
    """Scratch loads shed faulted page cache once a shard has supplied a
    threshold of reads (pod memcg protection); see CkptFileInfo._drop_page_cache."""

    @staticmethod
    def _write_shard(tmp, rows=1024):
        path = os.path.join(tmp, "test.safetensors")
        save_file({"w": torch.zeros(rows, dtype=torch.float32)}, path)
        return path

    def test_threshold_triggers_shed_and_close_sheds_finally(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = ckpt_file_info.CkptFileInfo(file_name=self._write_shard(tmp))
            with patch.object(ckpt_file_info, "_DROP_PAGE_CACHE", True), patch.object(
                ckpt_file_info, "_DROP_PAGE_CACHE_THRESHOLD", 1024
            ), patch.object(info, "_drop_page_cache") as drop:
                info.load_tensor("w", torch.float32)  # 4KiB > 1KiB threshold
                self.assertEqual(drop.call_count, 1)
                info.load_tensor_slice("w", (slice(None),), torch.float32)
                self.assertEqual(drop.call_count, 2)
                info.close_safetensor_handle()
                self.assertEqual(drop.call_count, 3)

    def test_killswitch_disables_shedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = ckpt_file_info.CkptFileInfo(file_name=self._write_shard(tmp))
            with patch.object(ckpt_file_info, "_DROP_PAGE_CACHE", False), patch.object(
                info, "_drop_page_cache"
            ) as drop:
                info.load_tensor("w", torch.float32)
                info.load_tensor_slice("w", (slice(None),), torch.float32)
                info.close_safetensor_handle()
                drop.assert_not_called()

    def test_fadvise_error_disables_without_raising(self):
        if not hasattr(os, "posix_fadvise"):
            self.skipTest("posix_fadvise unavailable on this platform")
        with tempfile.TemporaryDirectory() as tmp:
            info = ckpt_file_info.CkptFileInfo(file_name=self._write_shard(tmp))
            with patch.object(ckpt_file_info, "_DROP_PAGE_CACHE", True), patch.object(
                os, "posix_fadvise", side_effect=OSError("EINVAL")
            ) as fadvise:
                info._note_bytes_read(ckpt_file_info._DROP_PAGE_CACHE_THRESHOLD)
                self.assertTrue(info._fadvise_broken)
                # Broken stays quiet: no retry, no raise.
                info._note_bytes_read(ckpt_file_info._DROP_PAGE_CACHE_THRESHOLD)
                self.assertEqual(fadvise.call_count, 1)


class CkptMetaCacheTest(unittest.TestCase):
    """Unit tests for the CkptMetaCache class on isolated instances."""

    def test_get_put_roundtrip(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            cache.put(tmp.name, {"a": 1})
            self.assertEqual(cache.get(tmp.name), {"a": 1})

    def test_get_miss_returns_none(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            self.assertIsNone(cache.get(tmp.name))

    def test_stat_change_invalidates(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            cache.put(tmp.name, {"a": 1})
            # Bump mtime to a strictly later value.
            st = os.stat(tmp.name)
            os.utime(tmp.name, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            self.assertIsNone(cache.get(tmp.name))

    def test_size_change_invalidates(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            cache.put(tmp.name, {"a": 1})
            tmp.write(b"x")
            tmp.flush()
            self.assertIsNone(cache.get(tmp.name))

    def test_get_or_load_loads_once_and_shares_object(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            calls = []

            def loader(path):
                calls.append(path)
                return {"parsed": True}

            first = cache.get_or_load(tmp.name, loader)
            second = cache.get_or_load(tmp.name, loader)
            self.assertEqual(first, {"parsed": True})
            self.assertIs(first, second)
            self.assertEqual(calls, [tmp.name])

    def test_put_none_is_skipped(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            cache.put(tmp.name, None)
            self.assertIsNone(cache.get(tmp.name))
            # None put records no miss and caches nothing: a later real load
            # still parses once.
            value = cache.get_or_load(tmp.name, lambda p: {"v": 1})
            self.assertEqual(value, {"v": 1})

    def test_unstatable_path_is_silent(self):
        cache = CkptMetaCache()
        self.assertIsNone(cache.get("/nonexistent/path/file.safetensors"))
        cache.put("/nonexistent/path/file.safetensors", {"a": 1})  # no raise
        self.assertIsNone(cache.get("/nonexistent/path/file.safetensors"))

    def test_stats_and_clear(self):
        with tempfile.NamedTemporaryFile() as tmp:
            cache = CkptMetaCache()
            cache.put(tmp.name, {"a": 1})
            cache.get(tmp.name)
            cache.get("/nonexistent")  # miss, no stat entry
            hits, misses = cache.stats()
            self.assertEqual((hits, misses), (1, 1))
            cache.clear()
            self.assertEqual(cache.stats(), (0, 0))
            self.assertIsNone(cache.get(tmp.name))


class MetaCacheIntegrationTest(unittest.TestCase):
    """CkptDatabase-level tests for the process-wide metadata caches.

    Covers the dspark smoke scenario: target database scan -> target
    cleanup_database (metadata rebind) -> draft database construction on the
    SAME directory must hit the cache instead of re-reading shard headers.
    """

    def setUp(self):
        ckpt_file_info._FILE_META_CACHE.clear()
        database_module._INDEX_CACHE.clear()

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    @staticmethod
    def _write_two_shard_fixture(tmp):
        """A two-shard safetensors directory with model.safetensors.index.json."""
        shard_dir = os.path.join(tmp, "two_shard_ckpt")
        os.makedirs(shard_dir)
        shard1 = os.path.join(shard_dir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(shard_dir, "model-00002-of-00002.safetensors")
        save_file({"w1": torch.zeros(2), "b1": torch.zeros(2)}, shard1)
        save_file({"w2": torch.zeros(2), "b2": torch.zeros(2)}, shard2)
        index = {
            "weight_map": {
                "w1": "model-00001-of-00002.safetensors",
                "b1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
                "b2": "model-00002-of-00002.safetensors",
            }
        }
        with open(os.path.join(shard_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        return shard_dir

    @staticmethod
    def _counting_parse():
        original = ckpt_file_info._parse_file_meta
        calls = []

        def wrapper(file):
            calls.append(file)
            return original(file)

        return wrapper, calls

    def test_second_database_hits_cache(self):
        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        wrapper, calls = self._counting_parse()
        with patch.object(ckpt_file_info, "_parse_file_meta", wrapper):
            db_first = CkptDatabase(path)
            db_second = CkptDatabase(path)
        self.assertEqual(len(calls), 1)  # only the first construction parses
        self.assertEqual(
            len(db_second.pretrain_file_list), len(db_first.pretrain_file_list)
        )
        self.assertEqual(
            db_second.get_pretrain_tensor_names(),
            db_first.get_pretrain_tensor_names(),
        )

    def test_filter_independence_with_shared_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = self._write_two_shard_fixture(tmp)
            db_a = CkptDatabase(shard_dir)
            db_b = CkptDatabase(shard_dir)
            self.assertEqual(len(db_a.pretrain_file_list), 2)
            # Destructive in-place filter on one database must not affect the
            # other even though they share cached metadata dicts.
            db_a.filter_by_tensor_name_regexes([re.compile(r"w1")])
            self.assertEqual(len(db_a.pretrain_file_list), 1)
            self.assertEqual(len(db_b.pretrain_file_list), 2)

    def test_cleanup_does_not_poison_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = self._write_two_shard_fixture(tmp)
            db_target = CkptDatabase(shard_dir)
            # Simulate ModelLoader.cleanup_database (loader.py:615-656): it
            # rebinds each CkptFileInfo.metadata to None and clears the file
            # lists. The rebind must not touch the cached metadata dicts.
            for ckpt_file in db_target.pretrain_file_list:
                ckpt_file.close_safetensor_handle()
                ckpt_file.metadata = None
            db_target.pretrain_file_list.clear()
            wrapper, calls = self._counting_parse()
            with patch.object(ckpt_file_info, "_parse_file_meta", wrapper):
                db_draft = CkptDatabase(shard_dir)
            self.assertEqual(calls, [])  # draft reuses target's cached meta
            self.assertEqual(len(db_draft.pretrain_file_list), 2)
            for ckpt_file in db_draft.pretrain_file_list:
                self.assertIsNotNone(ckpt_file.metadata)
            self.assertEqual(
                sorted(db_draft.get_pretrain_tensor_names()), ["b1", "b2", "w1", "w2"]
            )

    def test_cache_invalidates_on_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = self._write_two_shard_fixture(tmp)
            db_before = CkptDatabase(shard_dir)
            self.assertIn("w1", db_before.get_pretrain_tensor_names())
            # Rewrite shard 1 with different tensors AND a later mtime.
            shard1 = os.path.join(shard_dir, "model-00001-of-00002.safetensors")
            save_file({"renamed": torch.zeros(3)}, shard1)
            st = os.stat(shard1)
            os.utime(shard1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            db_after = CkptDatabase(shard_dir)
            names = db_after.get_pretrain_tensor_names()
            self.assertIn("renamed", names)
            self.assertNotIn("w1", names)
            self.assertIn("w2", names)

    def test_index_json_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = self._write_two_shard_fixture(tmp)
            calls = []
            original = database_module._parse_index_json

            def wrapper(path):
                calls.append(path)
                return original(path)

            with patch.object(database_module, "_parse_index_json", wrapper):
                db_first = CkptDatabase(shard_dir)
                db_second = CkptDatabase(shard_dir)
            # load_hf_meta + _parse_weight_style per construction: only the
            # first parse is real, everything after hits the index cache.
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(db_second.pretrain_file_list), 2)

    def test_bin_branch_cached(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        wrapper, calls = self._counting_parse()
        with patch.object(ckpt_file_info, "_parse_file_meta", wrapper):
            CkptDatabase(path)
            CkptDatabase(path)
        self.assertEqual(len(calls), 1)  # torch.load branch also cached


if __name__ == "__main__":
    unittest.main()
