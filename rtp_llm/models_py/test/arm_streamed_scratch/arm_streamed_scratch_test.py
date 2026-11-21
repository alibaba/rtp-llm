"""ARM policy, distributed error and actual V4-0731 release reader tests."""

import json
import os
import platform
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod as Method,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper, get_all_weights


@register_model("_test_arm_streamed_scratch")
class _Owner(RtpModule):
    arm_cuda_streamed_scratch = True

    def __init__(self, config=None, load=None):
        super().__init__()
        self.table = torch.nn.Parameter(torch.empty(3, 4), requires_grad=False)
        self.source_and_destination = None

    def load_weights(self, weights):
        def record():
            for key, value in weights:
                self.source_and_destination = (
                    value.device.type,
                    self.table.device.type,
                )
                yield key, value

        super().load_weights(record())


@register_model("_test_release_scale_window")
class _ReleaseScaleWindow(RtpModule):
    """A bounded reader fixture, not a substitute for V4 decoder inference."""

    def __init__(self, config, load):
        super().__init__()
        self.source = config.release_key
        self.WEIGHTS_MAPPER = WeightsMapper(exact_mapping={self.source: "scale"})
        self.scale = torch.nn.Parameter(
            torch.empty(2, 16, dtype=torch.float8_e8m0fnu), requires_grad=False
        )

    def checkpoint_weight_name_filter(self):
        return lambda name: name == self.source

    def checkpoint_read_spec(self, name, shape):
        if name != "scale":
            raise KeyError(name)
        return ReadSpec(name, name, shape, slices=(slice(0, 2), slice(0, 16)))

    def load_weights(self, weights):
        super().load_weights(self.WEIGHTS_MAPPER.apply(weights))


def _loader(device="cuda:0", location=".", **kwargs):
    return NewModelLoader(
        SimpleNamespace(model_type="_test_arm_streamed_scratch"),
        NewLoaderConfig(device=device, load_method="scratch", **kwargs),
        model_path=location,
    )


def _rank_failure(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        loader = _loader()
        try:
            with loader._scratch_stage("checkpoint dispatch"):
                if rank == 1:
                    raise ValueError("bad shape")
        except RuntimeError as exc:
            assert "rank 1: ValueError: bad shape" in str(exc), str(exc)
        else:
            raise AssertionError("a peer's partial load must not be published")
        assert loader.loaded_method is None
    finally:
        dist.destroy_process_group()


def _subset_rank(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        # Only rank 0 considers these loaders. Any WORLD collective in the
        # eligibility test, or admission of a partial loader, is incorrect.
        if rank == 0:
            owner = _Owner()
            isolated = _Owner()
            isolated.fastsafetensors_load_isolated = True
            full = _loader(parallelism_config=SimpleNamespace(world_size=2, pp_size=1))
            with patch.object(
                platform, "machine", return_value="aarch64"
            ), patch.object(
                NewModelLoader, "_accelerator_backend_name", return_value="Cuda"
            ):
                assert not full._arm_streamed_scratch(isolated, Method.SCRATCH)
                for pc in (
                    None,
                    SimpleNamespace(world_size=1, pp_size=1),
                    SimpleNamespace(world_size=2, pp_size=2),
                ):
                    assert not _loader(parallelism_config=pc)._arm_streamed_scratch(
                        owner, Method.SCRATCH
                    )
                assert full._arm_streamed_scratch(owner, Method.SCRATCH)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ArmStreamedScratchTest(unittest.TestCase):
    def test_isolated_pp_and_partial_world_do_not_enter_scratch_ack(self):
        with tempfile.TemporaryDirectory() as work:
            torch.multiprocessing.spawn(
                _subset_rank, args=(Path(work, "sync").as_uri(),), nprocs=2
            )

    def test_policy_scope_and_force_cpu_override(self):
        owner = _Owner()
        loader = _loader()
        with patch.object(platform, "machine", return_value="aarch64"), patch.object(
            NewModelLoader, "_accelerator_backend_name", return_value="Cuda"
        ):
            self.assertTrue(loader._arm_streamed_scratch(owner, Method.SCRATCH))
            for candidate, obj, method in (
                (loader, torch.nn.Linear(3, 4), Method.SCRATCH),
                (loader, owner, Method.FASTSAFETENSORS),
                (_loader("cpu"), owner, Method.SCRATCH),
                (_loader(force_cpu_load_weights=True), owner, Method.SCRATCH),
            ):
                self.assertFalse(candidate._arm_streamed_scratch(obj, method))
            with patch.object(platform, "machine", return_value="x86_64"):
                self.assertFalse(loader._arm_streamed_scratch(owner, Method.SCRATCH))

    def test_reserved_budget_denies_before_destination_allocation(self):
        loader = _loader(reserve_runtime_mem_mb=1)
        owner = _Owner()
        ptr = owner.table.data_ptr()
        with patch.object(
            torch.cuda, "mem_get_info", return_value=(1048576, 2097152)
        ), patch.object(torch.cuda, "memory_reserved", return_value=0), patch.object(
            torch.cuda, "memory_allocated", return_value=0
        ), self.assertRaisesRegex(
            RuntimeError, "destination=48, runtime reserve=1048576"
        ):
            loader._prepare_streamed_scratch(owner)
        self.assertEqual(ptr, owner.table.data_ptr())
        self.assertIsNone(loader.loaded_method)

    def test_peer_read_error_reaches_both_ranks(self):
        with tempfile.TemporaryDirectory() as work:
            torch.multiprocessing.spawn(
                _rank_failure, args=(Path(work, "sync").as_uri(),), nprocs=2
            )

    def test_cpu_load_is_unchanged(self):
        with tempfile.TemporaryDirectory() as work:
            expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            save_file({"table": expected}, str(Path(work, "model.safetensors")))
            loader = _loader("cpu", work, force_cpu_load_weights=True)
            model = loader.load()
            self.assertEqual(model.source_and_destination, ("cpu", "cpu"))
            torch.testing.assert_close(model.table, expected)
            self.assertEqual(loader.loaded_method, Method.SCRATCH)

    @unittest.skipUnless(
        os.environ.get("ARM_CUDA_SCRATCH_TEST") == "1", "requires ARM CUDA"
    )
    def test_arm_cuda_load_writes_to_final_storage(self):
        self.assertIn(platform.machine().lower(), ("arm64", "aarch64"))
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory() as work:
            expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            save_file({"table": expected}, str(Path(work, "model.safetensors")))
            loader = _loader("cuda:0", work)
            model = loader.load()
            self.assertEqual(model.source_and_destination, ("cpu", "cuda"))
            torch.testing.assert_close(model.table.cpu(), expected)
            self.assertEqual(loader.loaded_method, Method.SCRATCH)

    @unittest.skipUnless(os.getenv("V4_RELEASE_CHECKPOINT"), "requires V4 Flash 0731")
    def test_0731_local_expert_slices_match_native_fst(self):
        root = Path(os.environ["V4_RELEASE_CHECKPOINT"])
        mapping = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        experts = json.loads((root / "config.json").read_text())["n_routed_experts"]
        # One real local expert from either half of the original EP=2 job.
        # Each reader touches only two 16-byte rows of its scale tensor.
        for expert in (0, experts // 2):
            key = f"layers.0.ffn.experts.{expert}.w1.scale"
            files = [str(root / mapping[key])]

            def resolve(name, shape):
                return ReadSpec(name, name, shape, slices=(slice(0, 2), slice(0, 16)))

            options = dict(
                name_filter=lambda name: name == key, read_spec_resolver=resolve
            )
            scratch = dict(get_all_weights(files, **options))[key].tensor
            native = dict(iter_fst_range_tensors(files, **options))[key].tensor
            self.assertEqual(native.dtype, torch.float8_e8m0fnu)
            self.assertEqual(scratch.dtype, native.dtype)
            self.assertEqual(scratch.shape, (2, 16))
            self.assertTrue(
                torch.equal(scratch.view(torch.uint8), native.view(torch.uint8))
            )
            for mode in (Method.SCRATCH, Method.FASTSAFETENSORS):
                with self.subTest(method=mode, expert=expert):
                    owner_loader = NewModelLoader(
                        SimpleNamespace(
                            model_type="_test_release_scale_window", release_key=key
                        ),
                        NewLoaderConfig(
                            device="cpu", load_method=mode, force_cpu_load_weights=True
                        ),
                        model_path=str(root),
                    )
                    owned = owner_loader.load()
                    self.assertEqual(owner_loader.loaded_method, mode)
                    self.assertTrue(
                        torch.equal(
                            owned.scale.view(torch.uint8), native.view(torch.uint8)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
