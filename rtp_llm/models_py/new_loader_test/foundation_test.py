import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch
import torch.nn as nn
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import (
    get_model_class,
    register_lazy_model,
    register_model,
)
from rtp_llm.models_py.weight_mapper import discover_ckpt_files, get_all_weights


class _Block(RtpModule):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.bias = nn.Parameter(torch.empty(2))


class _FoundationModel(RtpModule):
    supports_rank_local_checkpoint = True

    def __init__(self, model_config, load_config):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])
        self.final = nn.Parameter(torch.empty(2))
        self.register_buffer("scale", torch.empty(1), persistent=True)
        self.validation_device = None
        self.post_device = None
        self.post_count = 0

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.validation_device = self.scale.device.type
        super().validate_weights_loaded(loaded_tensor_ids)

    def process_weights_after_loading(self):
        self.post_device = self.final.device.type
        self.post_count += 1


register_model("foundation_test_model")(_FoundationModel)


def _weights():
    return {
        "layers.0.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "layers.0.bias": torch.tensor([5.0, 6.0]),
        "final": torch.tensor([7.0, 8.0]),
        "scale": torch.tensor([0.5]),
    }


class FoundationLoaderTest(unittest.TestCase):
    def _loader(self, model_path, **kwargs):
        config = types.SimpleNamespace(model_type="foundation_test_model")
        load_config = NewLoaderConfig(
            device="cpu", compute_dtype=torch.float32, **kwargs
        )
        return NewModelLoader(config, load_config, model_path=model_path)

    def test_real_safetensors_stream_load_and_postprocess(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            model = self._loader(model_path).load()
        self.assertTrue(
            torch.equal(model.layers[0].weight, _weights()["layers.0.weight"])
        )
        self.assertTrue(torch.equal(model.final, _weights()["final"]))
        self.assertEqual(model.post_device, "cpu")
        self.assertEqual(model.post_count, 1)

    def test_wrapped_pytorch_state_dict(self):
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(
                {"state_dict": _weights()},
                os.path.join(model_path, "pytorch_model.bin"),
            )
            model = self._loader(model_path).load()
        self.assertTrue(torch.equal(model.layers[0].bias, _weights()["layers.0.bias"]))

    def test_missing_required_parameter_fails(self):
        checkpoint = _weights()
        checkpoint.pop("final")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "missing required.*final"):
                self._loader(model_path).load()

    def test_unknown_tensor_fails(self):
        checkpoint = _weights()
        checkpoint["unexpected.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
                self._loader(model_path).load()

    def test_unknown_inv_freq_path_is_not_silently_dropped(self):
        checkpoint = _weights()
        checkpoint["unexpected.inv_freq"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "unexpected.inv_freq"):
                self._loader(model_path).load()

    def test_checkpoint_paths_only_target_registered_module_state(self):
        class PropertyModel(RtpModule):
            def __init__(self):
                super().__init__()
                child = RtpModule()
                child.weight = nn.Parameter(torch.zeros(1))
                object.__setattr__(self, "_unregistered_child", child)

            @property
            def virtual(self):
                return self._unregistered_child

        model = PropertyModel()
        with self.assertRaisesRegex(RuntimeError, "virtual.weight"):
            model.load_weights({"virtual.weight": torch.ones(1)})

    def test_non_persistent_checkpoint_buffer_is_not_loaded(self):
        for checkpoint_value in (torch.full((2,), 9.0), torch.full((3,), 9.0)):
            with self.subTest(checkpoint_shape=tuple(checkpoint_value.shape)):
                model = RtpModule()
                model.rotary_emb = RtpModule()
                expected = torch.tensor([1.0, 2.0])
                model.rotary_emb.register_buffer(
                    "inv_freq",
                    expected.clone(),
                    persistent=False,
                )

                model.load_weights({"rotary_emb.inv_freq": checkpoint_value})

                self.assertTrue(torch.equal(model.rotary_emb.inv_freq, expected))

    def test_registered_model_without_integrity_contract_fails(self):
        class UnsafeModel(nn.Module):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1))

            def load_weights(self, weights):
                pass

        register_model("foundation_unsafe_model")(UnsafeModel)
        config = types.SimpleNamespace(model_type="foundation_unsafe_model")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"weight": torch.ones(1)}, os.path.join(model_path, "model.safetensors")
            )
            with self.assertRaisesRegex(TypeError, "must inherit RtpModule"):
                NewModelLoader(
                    config,
                    NewLoaderConfig(device="cpu"),
                    model_path=model_path,
                ).load()

    def test_custom_leaf_without_validator_fails(self):
        class UnsafeLeaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1))

            def load_weights(self, weights):
                pass

        class UnsafeTree(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.leaf = UnsafeLeaf()

        register_model("foundation_unsafe_leaf_model")(UnsafeTree)
        config = types.SimpleNamespace(model_type="foundation_unsafe_leaf_model")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"leaf.weight": torch.ones(1)},
                os.path.join(model_path, "model.safetensors"),
            )
            with self.assertRaisesRegex(
                TypeError, "must define validate_weights_loaded"
            ):
                NewModelLoader(
                    config,
                    NewLoaderConfig(device="cpu"),
                    model_path=model_path,
                ).load()

    def test_shared_parameter_accepts_one_loaded_alias(self):
        class TiedTree(RtpModule):
            def __init__(self):
                super().__init__()
                self.embed = RtpModule()
                self.head = RtpModule()
                shared = nn.Parameter(torch.empty(2, 2))
                self.embed.weight = shared
                self.head.weight = shared

        model = TiedTree()
        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        model.load_weights({"embed.weight": expected})
        NewModelLoader._validate_loaded_weights(model)
        self.assertIs(model.embed.weight, model.head.weight)
        self.assertTrue(torch.equal(model.head.weight, expected))

    def test_same_module_parameter_aliases_are_valid_checkpoint_keys(self):
        class TiedAliases(RtpModule):
            def __init__(self):
                super().__init__()
                shared = nn.Parameter(torch.empty(2, 2))
                self.canonical = shared
                self.checkpoint_alias = shared

        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        for alias in ("canonical", "checkpoint_alias"):
            with self.subTest(alias=alias):
                model = TiedAliases()
                model.load_weights({alias: expected})
                NewModelLoader._validate_loaded_weights(model)
                self.assertTrue(torch.equal(model.canonical, expected))

    def test_same_module_buffer_aliases_are_valid_checkpoint_keys(self):
        class TiedBuffers(RtpModule):
            def __init__(self):
                super().__init__()
                shared = torch.empty(2)
                self.register_buffer("canonical", shared, persistent=True)
                self.register_buffer("checkpoint_alias", shared, persistent=True)

        expected = torch.tensor([1.0, 2.0])
        for alias in ("canonical", "checkpoint_alias"):
            with self.subTest(alias=alias):
                model = TiedBuffers()
                model.load_weights({alias: expected})
                NewModelLoader._validate_loaded_weights(model)
                self.assertTrue(torch.equal(model.canonical, expected))

    def test_non_recursive_apply_preserves_shared_parameter_identity(self):
        class Parent(RtpModule):
            def __init__(self):
                super().__init__()
                self.child = RtpModule()
                shared = nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.parent_weight = shared
                self.child.child_weight = shared

        model = Parent()
        with torch.no_grad():
            model._apply(lambda tensor: tensor.to(dtype=torch.float64), recurse=False)
        self.assertEqual(model.parent_weight.dtype, torch.float64)
        self.assertEqual(model.child.child_weight.dtype, torch.float64)
        self.assertIs(model.parent_weight, model.child.child_weight)

    def test_recursive_apply_converts_shared_tensor_once(self):
        class TiedTree(RtpModule):
            def __init__(self):
                super().__init__()
                self.embed = RtpModule()
                self.head = RtpModule()
                shared = nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.embed.weight = shared
                self.head.weight = shared
                self.register_buffer("weight_alias", shared, persistent=True)

        model = TiedTree()
        source_id = id(model.embed.weight)
        calls = {}

        def convert(tensor):
            tensor_id = id(tensor)
            calls[tensor_id] = calls.get(tensor_id, 0) + 1
            return tensor.to(dtype=torch.float64)

        with torch.no_grad():
            model._apply(convert)

        self.assertEqual(calls[source_id], 1)
        self.assertIs(model.embed.weight, model.head.weight)
        self.assertIs(model.embed.weight, model.weight_alias)
        self.assertIsInstance(model.embed.weight, nn.Parameter)
        self.assertEqual(model.embed.weight.dtype, torch.float64)

    def test_shared_parameter_crosses_custom_loader_boundary_both_directions(self):
        class CustomLeaf(RtpModule):
            def load_weights(self, weights):
                super().load_weights(weights)

        class CrossBoundaryTree(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.embed = RtpModule()
                self.head = CustomLeaf()
                shared = nn.Parameter(torch.empty(2, 2))
                self.embed.weight = shared
                self.head.weight = shared

        register_model("foundation_cross_boundary_model")(CrossBoundaryTree)
        config = types.SimpleNamespace(model_type="foundation_cross_boundary_model")
        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        for alias in ("embed.weight", "head.weight"):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as model_path:
                save_file(
                    {alias: expected}, os.path.join(model_path, "model.safetensors")
                )
                model = NewModelLoader(
                    config,
                    NewLoaderConfig(device="cpu", compute_dtype=torch.float32),
                    model_path=model_path,
                ).load()
                self.assertTrue(torch.equal(model.embed.weight, expected))
                self.assertTrue(torch.equal(model.head.weight, expected))

    def test_shape_mismatch_fails(self):
        checkpoint = _weights()
        checkpoint["final"] = torch.ones(3)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                self._loader(model_path).load()

    def test_force_cpu_is_not_exposed_by_foundation(self):
        with self.assertRaisesRegex(TypeError, "force_cpu_load_weights"):
            NewLoaderConfig(force_cpu_load_weights=True)

    def test_missing_persistent_buffer_fails(self):
        checkpoint = _weights()
        checkpoint.pop("scale")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "missing required.*scale"):
                self._loader(model_path).load()

    def test_non_floating_dtype_mismatch_fails(self):
        checkpoint = _weights()
        checkpoint["scale"] = torch.ones(1, dtype=torch.int32)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
                self._loader(model_path).load()

    def test_explicit_fastsafetensors_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "not part of.*foundation"):
                self._loader(
                    model_path, load_method=NewLoaderLoadMethod.FASTSAFETENSORS
                ).load()

    def test_invalid_load_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(
                ValueError, "Unsupported newloader load method"
            ):
                self._loader(model_path, load_method="typo")

    def test_non_string_load_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), os.path.join(model_path, "model.safetensors"))
            with self.assertRaisesRegex(TypeError, "NewLoaderLoadMethod or str"):
                self._loader(model_path, load_method=object())

    def test_optimizer_bin_does_not_hide_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as model_path:
            torch.save({}, os.path.join(model_path, "optimizer.bin"))
            torch.save(_weights(), os.path.join(model_path, "model.pt"))
            self.assertEqual(
                discover_ckpt_files(model_path), [os.path.join(model_path, "model.pt")]
            )

    def test_safetensors_index_selects_only_referenced_shards(self):
        with tempfile.TemporaryDirectory() as model_path:
            shard1 = os.path.join(model_path, "model-00001-of-00002.safetensors")
            shard2 = os.path.join(model_path, "model-00002-of-00002.safetensors")
            save_file({"first": torch.ones(1)}, shard1)
            save_file({"second": torch.ones(1)}, shard2)
            save_file(
                {"duplicate": torch.ones(1)},
                os.path.join(model_path, "consolidated.safetensors"),
            )
            save_file(
                {"adapter": torch.ones(1)},
                os.path.join(model_path, "adapter_model.safetensors"),
            )
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "w"
            ) as handle:
                json.dump(
                    {
                        "weight_map": {
                            "first": os.path.basename(shard1),
                            "second": os.path.basename(shard2),
                        }
                    },
                    handle,
                )
            self.assertEqual(
                discover_ckpt_files(model_path),
                [os.path.realpath(shard1), os.path.realpath(shard2)],
            )

    def test_pytorch_index_excludes_training_and_adapter_files(self):
        with tempfile.TemporaryDirectory() as model_path:
            shard = os.path.join(model_path, "pytorch_model-00001-of-00001.bin")
            torch.save({"weight": torch.ones(1)}, shard)
            torch.save({}, os.path.join(model_path, "training_args.bin"))
            torch.save({}, os.path.join(model_path, "adapter_model.bin"))
            with open(
                os.path.join(model_path, "pytorch_model.bin.index.json"), "w"
            ) as handle:
                json.dump({"weight_map": {"weight": os.path.basename(shard)}}, handle)
            self.assertEqual(
                discover_ckpt_files(model_path),
                [os.path.realpath(shard)],
            )

    def test_index_discovery_canonicalizes_symlink_model_root(self):
        with tempfile.TemporaryDirectory() as temp_root:
            real_model_path = os.path.join(temp_root, "real_model")
            linked_model_path = os.path.join(temp_root, "linked_model")
            os.mkdir(real_model_path)
            os.symlink(real_model_path, linked_model_path)

            shard = os.path.join(
                real_model_path,
                "model-00001-of-00001.safetensors",
            )
            save_file({"weight": torch.ones(1)}, shard)
            index_path = os.path.join(
                real_model_path,
                "model.safetensors.index.json",
            )
            with open(index_path, "w") as handle:
                json.dump(
                    {"weight_map": {"weight": os.path.basename(shard)}},
                    handle,
                )

            self.assertEqual(
                discover_ckpt_files(linked_model_path),
                [os.path.realpath(shard)],
            )

    def test_index_discovery_allows_shard_symlink_to_external_blob(self):
        with tempfile.TemporaryDirectory() as temp_root:
            model_path = os.path.join(temp_root, "snapshot")
            blob_path = os.path.join(temp_root, "blobs", "checkpoint-blob")
            os.makedirs(model_path)
            os.makedirs(os.path.dirname(blob_path))
            save_file({"weight": torch.ones(1)}, blob_path)

            shard_name = "model-00001-of-00001.safetensors"
            shard_path = os.path.join(model_path, shard_name)
            os.symlink(blob_path, shard_path)
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "w"
            ) as handle:
                json.dump({"weight_map": {"weight": shard_name}}, handle)

            discovered = discover_ckpt_files(model_path)
            self.assertEqual(discovered, [shard_path])
            self.assertTrue(
                torch.equal(
                    dict(get_all_weights(discovered))["weight"],
                    torch.ones(1),
                )
            )

    def test_unindexed_discovery_excludes_non_model_files(self):
        with tempfile.TemporaryDirectory() as model_path:
            model_file = os.path.join(model_path, "model.safetensors")
            save_file({"weight": torch.ones(1)}, model_file)
            save_file(
                {"other": torch.ones(1)},
                os.path.join(model_path, "consolidated.safetensors"),
            )
            save_file(
                {"adapter": torch.ones(1)},
                os.path.join(model_path, "adapter_model.safetensors"),
            )
            torch.save({}, os.path.join(model_path, "training_args.bin"))
            self.assertEqual(discover_ckpt_files(model_path), [model_file])

    def test_standalone_consolidated_checkpoint_is_discovered(self):
        with tempfile.TemporaryDirectory() as model_path:
            consolidated = os.path.join(model_path, "consolidated.safetensors")
            save_file({"weight": torch.ones(1)}, consolidated)
            self.assertEqual(discover_ckpt_files(model_path), [consolidated])

    def test_consolidated_rank_files_follow_tp_rank(self):
        with tempfile.TemporaryDirectory() as model_path:
            for rank in range(2):
                weights = _weights()
                weights["final"] = torch.full((2,), float(rank))
                torch.save(
                    weights,
                    os.path.join(model_path, f"consolidated.{rank:02d}.pth"),
                )

            for rank in range(2):
                with self.subTest(rank=rank):
                    model = self._loader(
                        model_path,
                        tp_size=2,
                        tp_rank=rank,
                    ).load()
                    self.assertTrue(
                        torch.equal(model.final, torch.full((2,), float(rank)))
                    )

            with self.assertRaisesRegex(
                ValueError,
                "2 consolidated rank files, but tp_size=3",
            ):
                discover_ckpt_files(model_path, tp_rank=0, tp_size=3)

            with self.assertRaisesRegex(ValueError, "Invalid TP partition"):
                discover_ckpt_files(model_path, tp_rank=2, tp_size=2)
            with self.assertRaisesRegex(TypeError, "must be integers"):
                discover_ckpt_files(model_path, tp_rank=False, tp_size=2)

    def test_ranked_consolidated_files_must_be_complete(self):
        cases = (
            ((0,), 2, "1 consolidated rank files, but tp_size=2"),
            ((1,), 1, "ranks must be contiguous"),
            ((0, 2), 2, "ranks must be contiguous"),
        )
        for ranks, tp_size, error in cases:
            with self.subTest(ranks=ranks, tp_size=tp_size):
                with tempfile.TemporaryDirectory() as model_path:
                    for rank in ranks:
                        torch.save(
                            _weights(),
                            os.path.join(
                                model_path,
                                f"consolidated.{rank:02d}.pth",
                            ),
                        )
                    with self.assertRaisesRegex(ValueError, error):
                        discover_ckpt_files(
                            model_path,
                            tp_rank=0,
                            tp_size=tp_size,
                        )

    def test_unranked_consolidated_file_is_a_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as model_path:
            checkpoint = os.path.join(model_path, "consolidated.pth")
            torch.save(_weights(), checkpoint)
            self.assertEqual(
                discover_ckpt_files(model_path, tp_rank=1, tp_size=2),
                [checkpoint],
            )

    def test_pth_checkpoint_is_discovered_and_loaded(self):
        with tempfile.TemporaryDirectory() as model_path:
            checkpoint = os.path.join(model_path, "model.pth")
            torch.save({"weight": torch.ones(1)}, checkpoint)
            self.assertEqual(discover_ckpt_files(model_path), [checkpoint])
            loaded = dict(get_all_weights([checkpoint]))
            self.assertTrue(torch.equal(loaded["weight"], torch.ones(1)))

    def test_standard_checkpoint_wins_over_consolidated(self):
        with tempfile.TemporaryDirectory() as model_path:
            consolidated = os.path.join(model_path, "consolidated.safetensors")
            standard = os.path.join(model_path, "model.pt")
            save_file({"duplicate": torch.ones(1)}, consolidated)
            torch.save({"weight": torch.ones(1)}, standard)
            self.assertEqual(discover_ckpt_files(model_path), [standard])

    def test_checkpoint_index_cannot_escape_model_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            model_path = os.path.join(parent, "model")
            os.mkdir(model_path)
            outside = os.path.join(parent, "outside.safetensors")
            save_file({"weight": torch.ones(1)}, outside)
            invalid_paths = (
                "../outside.safetensors",
                "nested/../../outside.safetensors",
                outside,
                r"C:\\outside.safetensors",
            )
            for shard_name in invalid_paths:
                with self.subTest(shard_name=shard_name):
                    with open(
                        os.path.join(model_path, "model.safetensors.index.json"), "w"
                    ) as handle:
                        json.dump({"weight_map": {"weight": shard_name}}, handle)
                    with self.assertRaisesRegex(ValueError, "outside model directory"):
                        discover_ckpt_files(model_path)

    def test_checkpoint_index_rejects_non_model_file(self):
        with tempfile.TemporaryDirectory() as model_path:
            adapter = os.path.join(model_path, "adapter_model.safetensors")
            save_file({"weight": torch.ones(1)}, adapter)
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "w"
            ) as handle:
                json.dump({"weight_map": {"weight": os.path.basename(adapter)}}, handle)
            with self.assertRaisesRegex(ValueError, "non-model file"):
                discover_ckpt_files(model_path)

    def test_duplicate_tensor_across_shards_fails(self):
        with tempfile.TemporaryDirectory() as model_path:
            first = os.path.join(model_path, "model-1.safetensors")
            second = os.path.join(model_path, "model-2.safetensors")
            save_file({"duplicate": torch.ones(1)}, first)
            save_file({"duplicate": torch.zeros(1)}, second)
            with self.assertRaisesRegex(RuntimeError, "more than one shard"):
                list(get_all_weights([first, second]))

    def test_lazy_registry_loads_declared_class(self):
        module_name = "_newloader_foundation_lazy_test"
        module = types.ModuleType(module_name)

        class LazyModel(nn.Module):
            pass

        LazyModel.__module__ = module_name
        module.LazyModel = LazyModel
        sys.modules[module_name] = module
        try:
            register_lazy_model("foundation_lazy_model", module_name, "LazyModel")
            self.assertIs(get_model_class("foundation_lazy_model"), LazyModel)
        finally:
            sys.modules.pop(module_name, None)

    def test_partition_config_validation(self):
        for field, value in (
            ("tp_size", True),
            ("tp_rank", 0.0),
            ("ep_size", 1.0),
            ("ep_rank", False),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(TypeError, f"{field} must be an integer"):
                    NewLoaderConfig(**{field: value})
        with self.assertRaisesRegex(ValueError, "Invalid TP"):
            NewLoaderConfig(tp_size=0)
        with self.assertRaisesRegex(ValueError, "Invalid EP"):
            NewLoaderConfig(ep_size=2, ep_rank=2)
        with self.assertRaisesRegex(ValueError, "Invalid device"):
            NewLoaderConfig(device="not:a:device")
        with self.assertRaisesRegex(ValueError, "cannot be meta"):
            NewLoaderConfig(device="meta")

    def test_load_config_type_is_validated(self):
        config = types.SimpleNamespace(model_type="foundation_test_model")
        for invalid in (False, {}, types.SimpleNamespace()):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    TypeError, "load_config must be NewLoaderConfig"
                ):
                    NewModelLoader(config, invalid)

    def test_load_config_is_immutable(self):
        config = NewLoaderConfig(device="cpu", load_method="scratch")
        self.assertEqual(config.load_method, NewLoaderLoadMethod.SCRATCH)
        with self.assertRaises(FrozenInstanceError):
            config.device = "cuda"

    def test_model_source_introspection_cannot_block_loading(self):
        class DynamicModel(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1))

        register_model("foundation_dynamic_source_model")(DynamicModel)
        config = types.SimpleNamespace(model_type="foundation_dynamic_source_model")
        with tempfile.TemporaryDirectory() as model_path:
            torch.save({"weight": torch.ones(1)}, os.path.join(model_path, "model.pt"))
            with mock.patch(
                "rtp_llm.models_py.model_loader.inspect.getfile",
                side_effect=TypeError("no source"),
            ):
                model = NewModelLoader(
                    config,
                    NewLoaderConfig(device="cpu"),
                    model_path=model_path,
                ).load()
        self.assertEqual(model.weight.item(), 1)

    def test_device_override_is_visible_during_model_construction(self):
        class DeviceAwareModel(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.config_device = load_config.device
                self.workspace = torch.empty(1, device=load_config.device)
                self.weight = nn.Parameter(torch.empty(1))

        register_model("foundation_device_aware_model")(DeviceAwareModel)
        config = types.SimpleNamespace(model_type="foundation_device_aware_model")
        with tempfile.TemporaryDirectory() as model_path:
            torch.save({"weight": torch.ones(1)}, os.path.join(model_path, "model.pt"))
            model = NewModelLoader(
                config,
                NewLoaderConfig(device="cuda:7"),
                model_path=model_path,
                device="cpu",
            ).load()
        self.assertEqual(model.config_device, "cpu")
        self.assertEqual(model.workspace.device.type, "cpu")

    def test_invalid_device_override_fails_before_model_creation(self):
        config = types.SimpleNamespace(model_type="foundation_test_model")
        with self.assertRaisesRegex(ValueError, "device override"):
            NewModelLoader(config, NewLoaderConfig(device="cpu"), device="")
        with self.assertRaisesRegex(ValueError, "Invalid device override"):
            NewModelLoader(config, NewLoaderConfig(device="cpu"), device="not:a:device")
        with self.assertRaisesRegex(ValueError, "cannot be meta"):
            NewModelLoader(config, NewLoaderConfig(device="cpu"), device="meta")

    def test_load_and_postprocess_run_in_inference_mode(self):
        class InferenceModel(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1))
                self.loader_inference_mode = False
                self.post_inference_mode = False

            def load_weights(self, weights):
                self.loader_inference_mode = torch.is_inference_mode_enabled()
                super().load_weights(weights)
                self.weight.add_(1)

            def process_weights_after_loading(self):
                self.post_inference_mode = torch.is_inference_mode_enabled()
                self.weight.mul_(2)

        register_model("foundation_inference_model")(InferenceModel)
        config = types.SimpleNamespace(model_type="foundation_inference_model")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"weight": torch.ones(1)}, os.path.join(model_path, "model.safetensors")
            )
            model = NewModelLoader(
                config,
                NewLoaderConfig(device="cpu"),
                model_path=model_path,
            ).load()
        self.assertTrue(model.loader_inference_mode)
        self.assertTrue(model.post_inference_mode)
        self.assertEqual(model.weight.item(), 4)

    def test_loaded_model_and_children_are_in_eval_mode(self):
        class EvalModel(RtpModule):
            def __init__(self, model_config, load_config):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(1))
                self.dropout = nn.Dropout(p=0.9)

            def forward(self, inputs):
                return self.dropout(inputs) + self.weight

        register_model("foundation_eval_model")(EvalModel)
        config = types.SimpleNamespace(model_type="foundation_eval_model")
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"weight": torch.ones(1)},
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                config,
                NewLoaderConfig(device="cpu"),
                model_path=model_path,
            ).load()

        self.assertFalse(model.training)
        self.assertFalse(model.dropout.training)
        inputs = torch.ones(32)
        self.assertTrue(torch.equal(model(inputs), model(inputs)))


if __name__ == "__main__":
    unittest.main()
