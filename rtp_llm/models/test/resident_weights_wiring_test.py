"""BaseModel's resident-weights wiring: the switch, the role, and the fallbacks.

Every branch of ``_resident_or_loaded_weights`` and ``_finish_resident_weights`` without a checkpoint
or a GPU: what they must do is call the loader, or call the manager, and never both.
"""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch
import torch.multiprocessing as mp

from rtp_llm.config.py_config_modules import ResidentWeightsConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.resident_weights import vmm
from rtp_llm.resident_weights.store import StoreOp
from rtp_llm.resident_weights.weights import ResidentWeights, all_ranks_true
from rtp_llm.utils.model_weight import W


class _Loader:
    def __init__(self):
        self.calls = []

    def load_weights(self, device, global_weight_aliases):
        self.calls.append((device, dict(global_weight_aliases)))
        return "checkpoint weights"


class _Manager:
    """The two ResidentWeights methods BaseModel is left calling."""

    def __init__(self):
        self.load_args = None
        self.finished_with = "not called"

    def attach_or_load_weights(
        self,
        loader,
        device,
        global_weight_aliases,
        excluded_names=(),
        synchronize_ranks=False,
    ):
        self.load_args = (
            loader,
            device,
            dict(global_weight_aliases),
            tuple(excluded_names),
        )
        return "resident weights"

    def finalize_after_initialize(self, py_model):
        self.finished_with = py_model


def _model(
    config: Optional[ResidentWeightsConfig] = None,
    manager: Optional[_Manager] = None,
    loader: Optional[_Loader] = None,
) -> BaseModel:
    """A BaseModel with only the attributes the resident-weights paths touch.

    __init__ loads a tokenizer and a checkpoint config, neither of which these branches read.
    """
    model = BaseModel.__new__(BaseModel)
    model.resident_weights_config = config or ResidentWeightsConfig()
    model.parallelism_config = SimpleNamespace(local_rank=0, pp_size=1)
    model._resident_weights = None
    model.model_weights_loader = loader if loader is not None else _Loader()
    model.py_model = None
    model._created = manager
    return model


class ResidentWeightsWiringTest(unittest.TestCase):
    def test_the_default_never_touches_the_resident_machinery(self):
        model = _model()
        with patch(
            "rtp_llm.resident_weights.store.request",
            side_effect=AssertionError("default loading touched the store"),
        ), patch.object(
            BaseModel, "_create_resident_weights", side_effect=AssertionError("called")
        ):
            self.assertEqual(
                model._resident_or_loaded_weights("cuda:0", {}), "checkpoint weights"
            )
        self.assertIsNone(model._resident_weights)
        self.assertEqual(model.model_weights_loader.calls, [("cuda:0", {})])

    def test_enabled_delegates_the_whole_load_to_the_manager(self):
        manager = _Manager()
        model = _model(ResidentWeightsConfig(mode="dev"), manager)
        with patch.object(BaseModel, "_create_resident_weights", return_value=manager):
            self.assertEqual(
                model._resident_or_loaded_weights("cuda:1", {"embed": "alias"}),
                "resident weights",
            )
        self.assertIs(model._resident_weights, manager)
        self.assertEqual(model.model_weights_loader.calls, [])
        loader, device, aliases, excluded = manager.load_args
        self.assertIs(loader, model.model_weights_loader)
        self.assertEqual(device, "cuda:1")
        self.assertEqual(aliases, {"embed": "alias"})
        # The loader rebuilds the routing tables on every start.
        self.assertEqual(excluded, (W.log2phy, W.logic_expert_cnt))

    def test_an_unusable_store_falls_back_to_the_checkpoint(self):
        model = _model(ResidentWeightsConfig(mode="dev"))
        with patch.object(BaseModel, "_create_resident_weights", return_value=None):
            self.assertEqual(
                model._resident_or_loaded_weights("cuda:0", {}), "checkpoint weights"
            )
        self.assertIsNone(model._resident_weights)
        self.assertEqual(model.model_weights_loader.calls, [("cuda:0", {})])

    def test_verify_reaches_the_manager_as_a_field_not_a_branch(self):
        manager = _Manager()
        model = _model(ResidentWeightsConfig(mode="dev", verify=True), manager)
        with patch.object(BaseModel, "_create_resident_weights", return_value=manager):
            model._resident_or_loaded_weights("cuda:0", {})
        # BaseModel no longer special-cases verify; the manager owns that decision.
        self.assertIsNotNone(manager.load_args)
        self.assertTrue(model.resident_weights_config.verify)

    def test_finish_passes_the_python_model_through_once_enabled(self):
        manager = _Manager()
        model = _model(ResidentWeightsConfig(mode="dev"), manager)
        model._resident_weights = manager
        model.py_model = "py model"
        model._finish_resident_weights()
        self.assertEqual(manager.finished_with, "py model")

    def test_finish_is_a_no_op_when_the_feature_is_off(self):
        model = _model()
        model._finish_resident_weights()  # must not raise with no manager

    def test_the_draft_role_is_the_only_difference_from_the_target(self):
        target = ResidentWeightsConfig(mode="dev", verify=True)
        draft = target.with_role("draft")
        self.assertEqual(
            (draft.enable, draft.verify, draft.role), (True, True, "draft")
        )
        self.assertEqual(target.role, "target")


class ResidentWeightsRankBarrierTest(unittest.TestCase):
    """A fast rank must not enter the short-timeout C++ engine while a slow rank still loads."""

    def _load(self, config: ResidentWeightsConfig, initialized: bool, world_size: int):
        manager = _Manager()
        model = _model(config, manager)
        with patch.object(
            BaseModel, "_create_resident_weights", return_value=manager
        ), patch("torch.distributed.is_initialized", return_value=initialized), patch(
            "torch.distributed.get_world_size", return_value=world_size
        ), patch(
            "torch.distributed.barrier"
        ) as barrier, patch(
            "rtp_llm.models.base_model.all_ranks_true",
            side_effect=lambda value, device: value,
        ):
            model._resident_or_loaded_weights("cuda:0", {})
        return barrier

    def test_a_multi_rank_start_waits_for_every_rank(self):
        barrier = self._load(
            ResidentWeightsConfig(mode="dev"), initialized=True, world_size=4
        )
        barrier.assert_called_once_with()

    def test_a_process_without_a_process_group_does_not_wait(self):
        barrier = self._load(
            ResidentWeightsConfig(mode="dev"), initialized=False, world_size=4
        )
        barrier.assert_not_called()

    def test_a_single_rank_world_does_not_wait(self):
        barrier = self._load(
            ResidentWeightsConfig(mode="dev"), initialized=True, world_size=1
        )
        barrier.assert_not_called()

    def test_the_barrier_is_only_for_resident_weights(self):
        # Without resident weights every rank reads the checkpoint, so T1 is already even.
        barrier = self._load(ResidentWeightsConfig(), initialized=True, world_size=4)
        barrier.assert_not_called()


def _distributed_load(rank, world_size, rendezvous, mixed):
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        manager = ResidentWeights.__new__(ResidentWeights)
        manager.verify = False
        manager.attach = Mock(return_value=[] if not mixed or rank == 0 else None)
        manager.discard_attach = Mock()
        manager.move_to_vmm = Mock()
        manager.bind = Mock()
        loader = Mock()

        def cold_load(**kwargs):
            # Model the WORLD broadcast used inside the real checkpoint loader.
            value = torch.tensor(rank)
            torch.distributed.broadcast(value, src=0)
            assert value.item() == 0
            return "cold"

        loader.load_weights.side_effect = cold_load
        loader.assemble_weights.return_value = "warm"
        result = manager.attach_or_load_weights(
            loader, "cpu", {}, synchronize_ranks=True
        )
        assert result == ("cold" if mixed else "warm")
        assert manager.discard_attach.called == mixed
        assert all_ranks_true(rank == 0, "cpu") is False
    finally:
        torch.distributed.destroy_process_group()


class ResidentWeightsRegressionTest(unittest.TestCase):
    def test_world_hit_consensus_precedes_loader_collectives(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed), tempfile.TemporaryDirectory() as directory:
                mp.start_processes(
                    _distributed_load,
                    args=(2, "file://" + directory + "/group", mixed),
                    nprocs=2,
                    join=True,
                    start_method="fork",
                )

    def test_pp_draft_does_not_enter_world_collectives(self):
        model = _model(ResidentWeightsConfig(mode="dev", role="draft"), _Manager())
        model.parallelism_config.pp_size = 2
        with patch.object(
            BaseModel, "_create_resident_weights", return_value=model._created
        ), patch(
            "rtp_llm.models.base_model.all_ranks_true",
            side_effect=AssertionError("WORLD"),
        ), patch.object(
            BaseModel, "_wait_for_loading_ranks", side_effect=AssertionError("WORLD")
        ):
            model._resident_or_loaded_weights("cuda:0", {})

    def _consensus_failure_load(self, manager):
        """Load with manager creation agreed by no one; returns the ordered side effects."""
        events = []
        loader = Mock()
        loader.load_weights.side_effect = lambda **kwargs: (
            events.append("cold load") or "checkpoint weights"
        )
        model = _model(ResidentWeightsConfig(mode="dev"), manager, loader)

        def request(store_name, req):
            events.append((store_name, req.op, req.slot_key))
            return SimpleNamespace(record=None), []

        with patch.object(
            BaseModel, "_create_resident_weights", return_value=manager
        ), patch("rtp_llm.models.base_model.all_ranks_true", return_value=False), patch(
            "rtp_llm.resident_weights.store.request", side_effect=request
        ), patch.object(
            BaseModel, "_wait_for_loading_ranks"
        ):
            self.assertEqual(
                model._resident_or_loaded_weights("cuda:0", {}), "checkpoint weights"
            )
        self.assertIsNone(model._resident_weights)
        return events

    def test_a_peer_without_resident_weights_drops_only_this_slot_before_cold_load(
        self,
    ):
        # The real manager's own drop path, scoped to its store name and GPU/role slot.
        manager = ResidentWeights.__new__(ResidentWeights)
        manager._store_name = "rtp_llm_resident_weights.500.29940"
        manager.slot_key = "GPU-a/target"
        manager._allocation_refs = []
        manager.attach_or_load_weights = Mock(side_effect=AssertionError("attached"))
        with self.assertLogs(level="WARNING") as logs:
            events = self._consensus_failure_load(manager)
        self.assertEqual(
            events,
            [
                ("rtp_llm_resident_weights.500.29940", StoreOp.DROP, "GPU-a/target"),
                "cold load",
            ],
        )
        self.assertIn("GPU-a/target", "\n".join(logs.output))

    def test_a_rank_without_a_manager_cold_loads_without_contacting_the_store(self):
        self.assertEqual(self._consensus_failure_load(None), ["cold load"])

    def test_manager_creation_checks_capability_and_eplb(self):
        for available, eplb in ((False, False), (True, False), (True, True)):
            with self.subTest(available=available, eplb=eplb):
                model = _model(
                    ResidentWeightsConfig(mode="dev", verify=True, role="draft")
                )
                model.model_config = SimpleNamespace(
                    eplb_config=SimpleNamespace(enable_eplb=lambda: eplb)
                )
                with patch.object(vmm, "is_available", return_value=available), patch(
                    "rtp_llm.models.base_model.ResidentWeights"
                ) as create, patch.object(
                    model, "_resident_fingerprint", return_value={}
                ):
                    result = model._create_resident_weights()
                if available and not eplb:
                    self.assertIs(result, create.return_value)
                    create.assert_called_once_with("draft", 0, {}, True)
                else:
                    self.assertIsNone(result)
                    create.assert_not_called()

    def test_resident_configuration_refuses_weight_updates_on_every_rank(self):
        from rtp_llm.model_loader.weight_manager import WeightManager

        # A rank whose target fell back to a checkpoint load holds no resident weights, yet
        # must refuse exactly like the ranks that do; only the configuration decides.
        for enabled, error in ((True, RuntimeError), (False, KeyError)):
            with self.subTest(enabled=enabled):
                manager = WeightManager.__new__(WeightManager)
                manager._resident_weights_enabled = enabled
                with self.assertRaises(error) as raised:
                    manager.update({})
                if enabled:
                    self.assertIn("resident weights", str(raised.exception))

    def test_default_model_weights_use_plain_independent_dicts(self):
        from rtp_llm.model_loader.model_weight_info import ModelWeights

        weights = ModelWeights(2, "cpu", torch.float32)
        self.assertTrue(all(type(layer) is dict for layer in weights.weights))
        self.assertIs(type(weights.global_weights), dict)
        self.assertIsNot(weights.weights[0], weights.weights[1])

    def test_rlimit_errors_disable_the_store(self):
        manager = ResidentWeights.__new__(ResidentWeights)
        manager._store_name = "unavailable"
        with patch("resource.getrlimit", side_effect=OSError("denied")):
            self.assertFalse(manager.ensure_store_running())

    def test_attach_uses_the_configuration_record_without_reading_sources(self):
        from rtp_llm.resident_weights import store

        manager = ResidentWeights.__new__(ResidentWeights)
        manager._store_name = "test"
        manager.slot_key = "gpu0/target"
        manager.fingerprint = "fingerprint"
        manager._drop_quietly = Mock()
        manager._import = Mock(return_value=[])
        record = SimpleNamespace(
            total_bytes=0,
            manifest=SimpleNamespace(allocations=[]),
            eplb_placement={},
            initialize_gpu_growth_bytes=0,
        )
        result = SimpleNamespace(
            status=store.AttachStatus.HIT, record=record, allocation_fds=[]
        )
        with patch.object(store, "attach_slot", return_value=result), patch(
            "builtins.open", side_effect=AssertionError("no source scan")
        ):
            self.assertEqual(manager.attach(), [])
        manager._import.assert_called_once_with(record, [])
        manager._drop_quietly.assert_not_called()

    @staticmethod
    def _fingerprint_model(directory):
        model = _model()
        model.model_config = SimpleNamespace(
            model_type="test",
            ckpt_path=directory,
            ptuning_path="",
            lora_infos={},
            enable_fp32_lm_head=True,
            normalize_lm_head_weight=False,
            output_vocab_ids=[],
            phy2log_path="",
            quantization="",
            to_string=lambda: "config",
        )
        model._weight_alias_names = ()
        model.merge_lora = True
        model.moe_pure_tp_preshard = False
        model.model_weights_loader.weights_info = SimpleNamespace(
            resident_weights_env_vars=()
        )
        for name in ("parallelism_config", "hw_kernel_config", "moe_config"):
            setattr(model, name, SimpleNamespace(to_string=lambda: "config"))
        return model

    def test_checkpoint_guard_is_top_level_metadata_not_a_content_checksum(self):
        from rtp_llm.resident_weights import checkpoint_digest

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.bin")
            with open(path, "wb") as file:
                file.write(b"old")
            stat = os.stat(path)
            before = checkpoint_digest(directory)
            with open(path, "wb") as file:
                file.write(b"new")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(before, checkpoint_digest(directory))
            os.mkdir(os.path.join(directory, "subdir"))
            with open(os.path.join(directory, "subdir", "weights.bin"), "wb") as file:
                file.write(b"nested")
            self.assertEqual(before, checkpoint_digest(directory))
            with open(path, "ab") as file:
                file.write(b"more")
            self.assertNotEqual(before, checkpoint_digest(directory))

    def test_checkpoint_guard_covers_subdirectory_shards_of_the_index(self):
        from rtp_llm.resident_weights import checkpoint_digest

        with tempfile.TemporaryDirectory() as directory:
            os.mkdir(os.path.join(directory, "shards"))
            shard = os.path.join(directory, "shards", "model-00001.safetensors")
            with open(shard, "wb") as file:
                file.write(b"old")
            with open(
                os.path.join(directory, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"w": "shards/model-00001.safetensors"}}, f)
            before = checkpoint_digest(directory, follow_index=True)
            with open(shard, "ab") as file:
                file.write(b"replaced")
            self.assertNotEqual(before, checkpoint_digest(directory, follow_index=True))
            # p-tuning and LoRA loaders ignore the index, so a dangling one must not matter.
            os.remove(shard)
            checkpoint_digest(directory)

    def test_ignored_ptuning_paths_do_not_break_fingerprinting(self):
        from rtp_llm.utils.database import CkptDatabase

        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            not_directory = os.path.join(directory, "not-a-directory")
            with open(not_directory, "wb") as file:
                file.write(b"not a ptuning directory")
            database = CkptDatabase.__new__(CkptDatabase)
            database.pretrain_file_list = []
            database.finetune_file_list = []
            for path in (None, "", os.path.join(directory, "missing"), not_directory):
                with self.subTest(ptuning_path=path):
                    database.load_ptuning_meta(path)
                    self.assertEqual(database.finetune_file_list, [])
                    model.model_config.ptuning_path = path
                    self.assertEqual(model._resident_fingerprint()["ptuning_files"], "")

    def test_finetune_file_changes_invalidate_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            adapter = os.path.join(directory, "adapter")
            ptuning = os.path.join(directory, "ptuning")
            os.mkdir(adapter)
            os.mkdir(ptuning)
            model.model_config.ptuning_path = ptuning
            model.model_config.lora_infos = {"adapter": adapter}
            before = model._resident_fingerprint()
            with open(os.path.join(adapter, "adapter_config.json"), "w") as file:
                file.write('{"lora_alpha": 16}')
            after_lora = model._resident_fingerprint()
            self.assertNotEqual(before["lora_files"], after_lora["lora_files"])
            with open(os.path.join(ptuning, "pytorch_model.bin"), "wb") as file:
                file.write(b"weights")
            self.assertNotEqual(
                after_lora["ptuning_files"],
                model._resident_fingerprint()["ptuning_files"],
            )

    def test_registered_settings_invalidate_the_fingerprint(self):
        # to_string() stays constant here, as it does for these Python-only settings: the
        # whole fingerprint must still change, and must not change for the same inputs.
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            for field, values in (
                ("enable_fp32_lm_head", (False, True)),
                ("normalize_lm_head_weight", (False, True)),
                ("output_vocab_ids", ([1, 2], [3, 4])),
                # A target recipe: Qwen4 FP8 and source BF16 dense storage share quant_algo.
                (
                    "quantization",
                    ('{"dense_storage": "fp8"}', '{"dense_storage": "source_bf16"}'),
                ),
                # A draft recipe: its own vocabulary versus the target's.
                (
                    "quantization",
                    (
                        '{"mtp_dense_storage": "source_bf16"}',
                        '{"mtp_dense_storage": "source_bf16_shared_vocab"}',
                    ),
                ),
            ):
                with self.subTest(field=field, values=values):
                    setattr(model.model_config, field, values[0])
                    before = model._resident_fingerprint()
                    self.assertEqual(before, model._resident_fingerprint())
                    setattr(model.model_config, field, values[1])
                    self.assertNotEqual(before, model._resident_fingerprint())

    def test_borrowed_global_names_invalidate_the_fingerprint(self):
        # A Qwen4 MTP draft in source_bf16 mode borrows the target vocabulary only when the
        # target's own vocabulary is BF16: the draft's recipe is unchanged, its aliases are not.
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            own_vocabulary = model._resident_fingerprint()
            model._weight_alias_names = (W.embedding, W.lm_head)
            borrowed = model._resident_fingerprint()
            self.assertNotEqual(own_vocabulary, borrowed)
            model._weight_alias_names = (W.lm_head, W.embedding)
            self.assertEqual(borrowed, model._resident_fingerprint())

    def test_fingerprint_tracks_only_declared_weight_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            model.model_weights_loader.weights_info.resident_weights_env_vars = (
                "TEST_WEIGHT_LAYOUT",
            )
            with patch.dict(
                os.environ, {"TEST_WEIGHT_LAYOUT": "a", "UNRELATED_ENV": "a"}
            ):
                before = model._resident_fingerprint()
                os.environ["UNRELATED_ENV"] = "b"
                self.assertEqual(before, model._resident_fingerprint())
                os.environ["TEST_WEIGHT_LAYOUT"] = "b"
                self.assertNotEqual(before, model._resident_fingerprint())

    def test_default_fingerprint_tracks_fp4_gemm_backend(self):
        from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo

        self.assertIn(
            "MOE_MEGA_SHARED_EXPERT",
            ModelDeployWeightInfo.resident_weights_env_vars,
        )
        self.assertIn(
            "RTP_LLM_FP4_GEMM_BACKEND",
            ModelDeployWeightInfo.resident_weights_env_vars,
        )
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            model.model_weights_loader.weights_info.resident_weights_env_vars = (
                ModelDeployWeightInfo.resident_weights_env_vars
            )
            with patch.dict(os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}):
                cutlass_fp = model._resident_fingerprint()
                os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "trtllm"
                self.assertNotEqual(cutlass_fp, model._resident_fingerprint())

    def test_dynamic_lora_paths_are_not_probed_during_base_weight_fingerprinting(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._fingerprint_model(directory)
            model.model_config.lora_infos = {
                "a": directory + "/not-loaded-a",
                "b": directory + "/not-loaded-b",
            }
            self.assertEqual(model._resident_fingerprint()["lora_files"], "")


if __name__ == "__main__":
    unittest.main()
