"""Actual CPU checkpoint I/O at the production publication boundary."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers, requires_native_fst
from rtp_llm.models_py.downstream_heads import (
    ReplicatedHandlerHeads,
    handler_weight_mappings,
)
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_service_direct_handler")
class HandlerCheckpoint(RtpModule):
    supports_custom_weight_mappings = True
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, config, load_config):
        super().__init__()
        self.heads = ReplicatedHandlerHeads(
            load_config.custom_weight_mappings,
            self.WEIGHTS_MAPPER,
            2,
            load_config.compute_dtype,
            projection_sources=getattr(config, "projection_sources", ()),
        )

    def runtime_weight_view(self):
        return self.heads.exports()

    def load_weights(self, weights):
        for name, tensor in self.WEIGHTS_MAPPER.apply(weights):
            self.heads.capture(name, tensor)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.heads.validate()
        super().validate_weights_loaded(loaded_tensor_ids)


@register_model("_service_direct_leaf")
class ServiceLeaf(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.weight_param = torch.nn.Parameter(torch.empty(2, 2), requires_grad=False)
        self.load_config = load_config

    def runtime_weight_view(self):
        return {"lm_head": self.weight_param}


@register_model("_indexed_service_probe")
class IndexedServiceProbe(ServiceLeaf):
    def __init__(self, config, load_config):
        super().__init__(config, load_config)
        self.other = torch.nn.Parameter(torch.empty(3))

    def resolve_checkpoint_load_plan(self, name, shape):
        if name.startswith("model.layers.10."):
            raise AssertionError("unindexed expert reached ReadSpec resolution")
        return super().resolve_checkpoint_load_plan(name, shape)


def _load_indexed_rank(rank, root, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        model = service(root, "fastsafetensors", "_indexed_service_probe")
        pc = model.parallelism_config
        pc.world_size = pc.tp_size = 2
        pc.world_rank = pc.local_rank = pc.tp_rank = rank
        pc.get_attn_tp_size = pc.get_ffn_tp_size = lambda: 2
        pc.get_attn_tp_rank = pc.get_ffn_tp_rank = lambda: rank
        load_owned_model(model)
        assert model.model_weights_loader.loaded_method == NewLoaderLoadMethod.FASTSAFETENSORS
        torch.testing.assert_close(model.py_model.weight_param, torch.arange(4).reshape(2, 2).float())
        torch.testing.assert_close(model.py_model.other, torch.tensor([-1., 17., 3.]))
    finally:
        dist.destroy_process_group()


def service(path, method, model_type="_service_direct_leaf"):
    parallel = SimpleNamespace(
        world_size=1,
        world_rank=0,
        local_rank=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            ckpt_path=path,
            model_type=model_type,
            num_layers=1,
            compute_dtype=torch.float32,
            quant_config=None,
            lora_infos={},
            eplb_config=SimpleNamespace(enable_eplb=lambda: False),
        ),
        _get_device_str=lambda: "cpu",
        _weight_alias_names=(),
        _weight_alias_owner=None,
        parallelism_config=parallel,
        hw_kernel_config=SimpleNamespace(use_swizzleA=False),
        moe_config=None,
        fmha_config=None,
        max_generate_batch_size=7,
        load_method=method,
        custom_module=None,
        weight=None,
        py_model=None,
        model_weights_loader=None,
        py_eplb=None,
        weight_manager=None,
        rank_local_lora=None,
    )


class DirectServingBoundaryTest(unittest.TestCase):
    def test_unported_service_type_reports_missing_owner_before_device_or_io(self):
        tag = "example_arch_without_an_owner"
        service = SimpleNamespace(model_config=SimpleNamespace(model_type=tag))
        with self.assertRaisesRegex(ValueError, f"{tag}.*no direct checkpoint owner"):
            load_owned_model(service)

    @requires_native_fst
    def test_two_gloo_ranks_combine_collective_and_indexed_range_reads(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            save_file({"weight_param": torch.arange(4).reshape(2, 2).float(),
                       "model.layers.10.mlp.experts.0.weight": torch.ones(4)},
                      os.path.join(root, "partial.safetensors"))
            save_file({"other": torch.tensor([-1., 17., 3.])}, os.path.join(root, "clean.safetensors"))
            with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
                json.dump({"weight_map": {"weight_param": "partial.safetensors", "other": "clean.safetensors"}}, handle)
            torch.multiprocessing.spawn(_load_indexed_rank, args=(root, (Path(root)/"dist").as_uri()), nprocs=2)

    def test_hf_index_owns_physical_shard_subset(self):
        # A shortened score checkpoint keeps a physical shard from a larger
        # model. Its unrelated experts must never reach the strict dispatcher.
        formats = [("bin", "scratch")]
        formats.extend(("safetensors", method) for method in checkpoint_readers("scratch", "fastsafetensors"))
        for ext, method in formats:
            with self.subTest(ext=ext, method=method), tempfile.TemporaryDirectory() as root:
                folder = os.path.join(root, "shards")
                os.mkdir(folder)
                shard = f"shards/model-00001.{ext}"
                weight = torch.arange(4).reshape(2, 2).float()
                data = {"weight_param": weight, "model.layers.10.mlp.experts.0.scale": torch.ones(4)}
                write = torch.save if ext == "bin" else save_file
                write(data, os.path.join(root, shard))
                # A file outside the index must not be discovered as a base shard.
                write({"weight_param": -weight}, os.path.join(root, f"extra.{ext}"))
                index = "pytorch_model.bin.index.json" if ext == "bin" else "model.safetensors.index.json"
                with open(os.path.join(root, index), "w") as handle:
                    json.dump({"weight_map": {"weight_param": shard}}, handle)
                owned = service(root, method)
                load_owned_model(owned)
                self.assertEqual(owned.model_weights_loader.loaded_method.value, method)
                torch.testing.assert_close(owned.py_model.weight_param, weight)

    def test_hf_index_rejects_missing_and_misplaced_tensors(self):
        from rtp_llm.models_py.weight_mapper import checkpoint_catalog

        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                first, second = "a.safetensors", "b.safetensors"
                index = os.path.join(root, "model.safetensors.index.json")
                with open(index, "w") as handle:
                    json.dump({"weight_map": {"weight_param": first, "other": second}}, handle)
                save_file({"weight_param": torch.ones(2, 2)}, os.path.join(root, first))
                for keys, error in (({"unused": torch.ones(1)}, "missing"),
                                    ({"other": torch.ones(1), "weight_param": torch.zeros(2, 2)},
                                     "outside its indexed shard")):
                    save_file(keys, os.path.join(root, second))
                    with self.assertRaisesRegex((RuntimeError, ValueError), error):
                        load_owned_model(service(root, method))
                with open(index, "w") as handle:
                    handle.write('{"weight_map":{"weight_param":"a.safetensors","weight_param":"b.safetensors"}}')
                with self.assertRaisesRegex(ValueError, "Duplicate key"):
                    checkpoint_catalog(root)

    def test_partial_index_uses_native_ranges_only_for_dirty_shards(self):
        from rtp_llm.models_py.fst_file_plan import plan_fst_files
        from rtp_llm.models_py.weight_mapper import checkpoint_catalog

        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a.safetensors")
            b = os.path.join(root, "b.safetensors")
            save_file({"active": torch.ones(1), "inactive": torch.zeros(2)}, a)
            save_file({"other": torch.zeros(1)}, b)
            with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
                json.dump({"weight_map": {"active": "a.safetensors", "other": "b.safetensors"}}, handle)
            group = SimpleNamespace(size=lambda: 1)
            self.assertEqual(plan_fst_files([a, b], root, group, None, None, checkpoint_catalog(root)), ([b], [a]))
            self.assertEqual(plan_fst_files([a, b], root, group, None, None), ([a, b], []))

    def test_direct_loading_does_not_require_legacy_adapter_database_or_rpc(self):
        # A warm process hides accidental dependencies via sys.modules. Use
        # the same runfiles as the parent and real readers in a fresh process.
        code = """
import sys
import unittest

class ImportGuard:
    def find_spec(self, name, path=None, target=None):
        if name in {
            "rtp_llm.lora.lora_file",
            "rtp_llm.model_factory",
            "rtp_llm.server.backend_rpc_server_visitor",
        }:
            raise AssertionError(f"direct checkpoint loading imported {name}")
        return None

sys.meta_path.insert(0, ImportGuard())
from rtp_llm.models_py.serving_route_checkpoint_test import DirectServingBoundaryTest
result = unittest.TextTestRunner().run(unittest.TestSuite([
    DirectServingBoundaryTest("test_exact_backend_and_publication_preserve_identity"),
]))
sys.exit(0 if result.wasSuccessful() else 1)
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(os.path.abspath, sys.path)))
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, check=True)

    def test_extension_projection_retains_independent_dtype_and_pair(self):
        from rtp_llm.model_loader.weight_module import CustomAtomicWeight
        from rtp_llm.models.downstream_modules.custom_module import (
            CustomHandler,
            CustomModule,
        )
        from rtp_llm.utils.model_weight import CkptWeightInfo

        class Projection(CustomHandler):
            def custom_weight_info(self):
                return [
                    CustomAtomicWeight(
                        "__custom__." + key, [CkptWeightInfo(key)], data_type=dtype
                    )
                    for key, dtype in (
                        ("score.weight", torch.bfloat16),
                        ("score.bias", torch.float32),
                    )
                ]

            def init(self, params):
                self.weights = params

        weight = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
        # Explicit fp32 bias must not be rounded through the bf16 model dtype.
        bias = torch.tensor([1.001, 2.003])
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                path = os.path.join(root, "model.safetensors")

                def new_service():
                    value = service(root, method, "_service_direct_handler")
                    value.model_config.compute_dtype = torch.bfloat16
                    value.model_config.projection_sources = (
                        "score.weight",
                        "score.bias",
                    )
                    value.custom_module = CustomModule(value.model_config, None)
                    value.custom_module.handler = Projection(value.model_config)
                    return value

                save_file({"score.weight": weight, "score.bias": bias}, path)
                value = new_service()
                load_owned_model(value)
                params = value.custom_module.handler.weights
                self.assertEqual(value.model_weights_loader.loaded_method.value, method)
                torch.testing.assert_close(params["score.weight"], weight.bfloat16())
                torch.testing.assert_close(params["score.bias"], bias, rtol=0, atol=0)
                self.assertIs(
                    params["score.bias"],
                    value.weight.get_global_weight("__custom__.score.bias"),
                )
                for source, message in (
                    ({"score.weight": weight}, "Missing downstream"),
                    (
                        {"score.weight": weight, "score.bias": torch.ones(3)},
                        "label counts",
                    ),
                ):
                    save_file(source, path)
                    failed = new_service()
                    with self.assertRaisesRegex((ValueError, RuntimeError), message):
                        load_owned_model(failed)
                    self.assertIsNone(failed.weight)
                    self.assertIsNone(failed.py_model)

                invalid = new_service()
                records = invalid.custom_module.handler.custom_weight_info()
                records[0].process_fun = lambda inputs: inputs[0].T
                invalid.custom_module.handler.custom_weight_info = lambda: records
                with self.assertRaisesRegex(ValueError, "Unsupported custom"):
                    load_owned_model(invalid)
                self.assertIsNone(invalid.py_model)

    def test_offline_conversion_keeps_owner_without_publishing_runtime(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                tensor = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
                save_file(
                    {"weight_param": tensor}, os.path.join(root, "model.safetensors")
                )
                model = service(root, method)
                load_owned_model(model, publish_runtime=False)
                self.assertIsNone(model.py_model)
                self.assertIsNone(model.weight)
                self.assertIsNone(model.py_eplb)
                self.assertIsNone(model.weight_manager)
                self.assertIsNone(model.rank_local_lora)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                torch.testing.assert_close(model.checkpoint_owner.weight_param, tensor)

    @requires_native_fst
    def test_forced_cpu_stages_with_native_fst_and_never_retries(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            source = torch.tensor([[2.0, 4.0], [-1.0, 3.0]])
            save_file({"weight_param": source}, path)
            value = service(root, "fastsafetensors")
            value.force_cpu_load_weights = True
            with patch.object(
                NewModelLoader,
                "_fastsafetensors_weights_iterator",
                side_effect=AssertionError("parallel FST used"),
            ), patch(
                "rtp_llm.models_py.weight_mapper.get_all_weights",
                side_effect=AssertionError("scratch used"),
            ):
                load_owned_model(value)
                self.assertEqual(
                    value.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod.FASTSAFETENSORS,
                )
                torch.testing.assert_close(value.py_model.weight_param, source)
                failed = service(root, "fastsafetensors")
                failed.force_cpu_load_weights = True
                with patch(
                    "rtp_llm.models_py.fst_range_reader.native_cpu_range_api",
                    side_effect=ImportError("reader missing"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "reader missing"):
                        load_owned_model(failed)
                self.assertIsNone(failed.py_model)

    @requires_native_fst
    def test_native_cpu_staging_reads_pruned_vocab_rows_including_empty_rank(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            source = torch.arange(36, dtype=torch.float32).reshape(9, 4)
            save_file({"head": source}, path)
            for rows in ((0, 1, 4, 6, 7), ()):
                with self.subTest(rows=rows):
                    values = list(
                        iter_fst_range_tensors(
                            [path],
                            read_spec_resolver=lambda key, shape: ReadSpec(
                                key, key, shape, row_indices=rows
                            ),
                        )
                    )
                    self.assertEqual(len(values), 1)
                    tensor, spec = unwrap_checkpoint_tensor(values[0][1])
                    self.assertEqual(spec.row_indices, rows)
                    torch.testing.assert_close(tensor, source[list(rows)])

    def test_full_handler_sources_survive_reader_destruction(self):
        values = {
            "model.embed_tokens.weight": torch.arange(8).reshape(4, 2).float(),
            "classifier.weight": torch.arange(6).reshape(3, 2).float(),
            "classifier.bias": torch.tensor([3.0, 1.0, -1.0]),
        }
        mappings = handler_weight_mappings(tuple(values))
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                cfg = service(root, method, "_service_direct_handler").model_config
                path = os.path.join(root, "model.safetensors")
                save_file(values, path)
                load = NewLoaderConfig(
                    device="cpu",
                    load_method=method,
                    compute_dtype=torch.float32,
                    custom_weight_mappings=mappings,
                )
                loader = NewModelLoader(cfg, load, model_path=root)
                model = loader.load()
                os.remove(path)
                self.assertEqual(loader.loaded_method, NewLoaderLoadMethod(method))
                for name, value in values.items():
                    torch.testing.assert_close(
                        model.heads.exports()["__custom__." + name], value
                    )
                bad = dict(values)
                del bad["classifier.bias"]
                save_file(bad, path)
                with self.assertRaisesRegex(RuntimeError, "Missing downstream"):
                    NewModelLoader(cfg, load, model_path=root).load()

    def test_exact_backend_and_publication_preserve_identity(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                values = torch.tensor([[2.0, 5.0], [-3.0, 7.0]])
                file = os.path.join(root, "model.safetensors")
                save_file({"weight_param": values}, file)
                model = service(root, method)
                load_owned_model(model)
                os.remove(file)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                self.assertIs(
                    model.weight.get_global_weight("lm_head"),
                    model.py_model.weight_param,
                )
                self.assertIs(model.py_model.weight, model.weight)
                self.assertTrue(callable(model.weight_manager.update))
                self.assertIsNotNone(model.rank_local_lora)
                self.assertIs(
                    model.py_model.load_config.py_hw_kernel_config,
                    model.hw_kernel_config,
                )
                self.assertEqual(model.py_model.load_config.max_generate_batch_size, 7)
                self.assertTrue(torch.equal(model.py_model.weight_param, values))

    def test_explicit_fst_never_retries_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            torch.save(
                {"weight_param": torch.ones(2, 2)},
                os.path.join(root, "pytorch_model.bin"),
            )
            model = service(root, "fastsafetensors")
            with self.assertRaisesRegex((ValueError, RuntimeError), "safetensors"):
                load_owned_model(model)
            self.assertIsNone(model.py_model)
            self.assertIsNone(model.weight)
            self.assertIsNone(model.model_weights_loader)

    def test_failed_handler_is_not_published(self):
        class FailingHandler:
            def checkpoint_weight_mappings(self):
                return ()

            def init(self, weights):
                raise ValueError("invalid handler weights")

        with tempfile.TemporaryDirectory() as root:
            save_file(
                {"weight_param": torch.ones(2, 2)},
                os.path.join(root, "model.safetensors"),
            )
            model = service(root, "scratch")
            model.custom_module = FailingHandler()
            with self.assertRaisesRegex(ValueError, "invalid handler weights"):
                load_owned_model(model)
            self.assertIsNone(model.weight)
            self.assertIsNone(model.py_model)


if __name__ == "__main__":
    unittest.main()
