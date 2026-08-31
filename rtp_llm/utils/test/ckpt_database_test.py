import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.utils import ckpt_file_info
from rtp_llm.utils.database import _LAYER_RE, CkptDatabase


class _FakeCkptFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


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


class FastsafetensorsAutoLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_fastsafetensors = sys.modules.get("fastsafetensors")
        self._had_fastsafetensors = "fastsafetensors" in sys.modules
        self._config_env_names = (
            "FASTSAFETENSORS_CONFIG",
            "FASTSAFETENSORS_CONFIG_JSON",
            "FASTSAFETENSORS_NOGDS",
        )
        self._saved_config_env = {
            name: os.environ.get(name) for name in self._config_env_names
        }
        self._had_config_env = {
            name: name in os.environ for name in self._config_env_names
        }
        for name in self._config_env_names:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        if self._had_fastsafetensors:
            sys.modules["fastsafetensors"] = self._saved_fastsafetensors
        else:
            sys.modules.pop("fastsafetensors", None)
        for name in self._config_env_names:
            if self._had_config_env[name]:
                os.environ[name] = self._saved_config_env[name]
            else:
                os.environ.pop(name, None)

    def test_default_stacked_experts_are_split_before_delivery(self) -> None:
        closed = []
        observed_split_templates = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                dim0_split_templates=None,
            ) -> None:
                observed_split_templates.append(dim0_split_templates)

            def iterate_weights(self):
                for expert_id in range(3):
                    yield f"experts.{expert_id}.weight", f"expert-{expert_id}"
                yield "plain", "plain-tensor"

            def close(self) -> None:
                closed.append(True)

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                use_tqdm_on_load=False,
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
            )
        )

        self.assertEqual(
            [key for key, _ in result],
            [
                "experts.0.weight",
                "experts.1.weight",
                "experts.2.weight",
                "plain",
            ],
        )
        self.assertEqual(
            [tensor for _, tensor in result[:3]],
            ["expert-0", "expert-1", "expert-2"],
        )
        self.assertEqual(result[3], ("plain", "plain-tensor"))
        self.assertEqual(
            observed_split_templates,
            [{"stacked": "experts.{expert_id}.weight"}],
        )
        self.assertEqual(closed, [True])

    def test_rank_local_copyout_filter_is_forwarded_to_auto_loader(self) -> None:
        observed_filters = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                dim0_split_templates=None,
            ) -> None:
                observed_filters.append(local_copyout_filter)

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        required_keys = {"needed"}
        predicate = required_keys.__contains__

        list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                False,
                local_copyout_filter=predicate,
            )
        )

        self.assertEqual(observed_filters, [predicate])

    def test_full_stacked_mode_disables_prebroadcast_split(self) -> None:
        observed_split_templates = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                dim0_split_templates=None,
            ) -> None:
                observed_split_templates.append(dim0_split_templates)

            def iterate_weights(self):
                yield "stacked", torch.tensor([[1, 2], [3, 4]])

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                False,
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                stacked_moe_mode="full-stacked",
            )
        )

        self.assertEqual(observed_split_templates, [None])
        self.assertEqual(
            [name for name, _tensor in result],
            [
                "experts.0.weight",
                "experts.1.weight",
            ],
        )
        torch.testing.assert_close(result[0][1], torch.tensor([1, 2]))
        torch.testing.assert_close(result[1][1], torch.tensor([3, 4]))

    def test_wrapper_without_auto_loader_fails_instead_of_legacy_fallback(self) -> None:
        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(ImportError, "AutoLoader"):
            list(database.fastsafetensors_weights_iterator("cuda", False))

    def test_per_expert_mode_fails_fast_when_wrapper_lacks_split_capability(
        self,
    ) -> None:
        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(self, pg, files, device, local_copyout_filter=None) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(
            RuntimeError, "does not support bounded-memory stacked MoE delivery"
        ):
            list(
                database.fastsafetensors_weights_iterator(
                    "cuda",
                    False,
                    stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                )
            )

    def test_legacy_nogds_overrides_config_json(self) -> None:
        observed_config = []

        class FakeSingleGroup:
            def rank(self) -> int:
                return 0

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                dim0_split_templates=None,
            ) -> None:
                observed_config.append(
                    json.loads(os.environ["FASTSAFETENSORS_CONFIG_JSON"])
                )

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        fake_module = types.ModuleType("fastsafetensors")
        fake_module.SingleGroup = FakeSingleGroup
        fake_module.AutoLoader = FakeAutoLoader
        sys.modules["fastsafetensors"] = fake_module

        os.environ["FASTSAFETENSORS_CONFIG_JSON"] = json.dumps({"loader": "fuse-shm"})
        os.environ["FASTSAFETENSORS_NOGDS"] = "1"

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        list(database.fastsafetensors_weights_iterator("cuda", False))

        expected_config_json = '{"loader":"base","base":{"copier_type":"nogds"}}'
        self.assertEqual(
            os.environ["FASTSAFETENSORS_CONFIG_JSON"], expected_config_json
        )
        self.assertEqual(
            observed_config,
            [{"loader": "base", "base": {"copier_type": "nogds"}}],
        )


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


if __name__ == "__main__":
    unittest.main()
