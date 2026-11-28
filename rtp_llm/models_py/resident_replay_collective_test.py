"""A local warm-restart error must not race a peer's executor stage barrier."""

import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model
from rtp_llm.resident_weights import GLOBAL_SCOPE, RestoredTensor


@register_model("_resident_collective")
class _Owner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.empty(2))
        if config.reject_constructor:
            raise ValueError("incompatible resident constructor")

    def prepare_resident_weights(self):
        raise AssertionError("replay must not rerun cold preparation")

    def bind_resident_runtime(self):
        if self.config.reject_binding:
            raise ValueError("incompatible local runtime binding")

    def process_weights_after_loading(self):
        # The NVFP4 DeepEP executor joins such a barrier at construction.
        self.config.hooks.append("entered")
        dist.barrier(group=self.config.stage)


@register_model("_resident_collective_isolated")
class _IsolatedOwner(_Owner):
    fastsafetensors_load_isolated = True


@register_model("_resident_collective_cold")
class _ColdOwner(_Owner):
    def prepare_resident_weights(self):
        pass

    def resident_constructor_state(self):
        return {}


class _Executor(RtpModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.packed = False

    def requires_staged_device_postprocess(self):
        return self.config.staged

    def process_weights_during_staged_load(self):
        if not self.packed:
            self.config.events.append("pack")
            self.packed = True

    def process_weights_after_loading(self):
        self.config.events.append("build")
        dist.barrier(group=self.config.stage)


class _LaterLayout(RtpModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def process_weights_after_loading(self):
        if self.config.fault == "layout":
            raise ValueError("invalid later local layout")


@register_model("_direct_collective")
class _DirectOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2))
        self.expert = _Executor(config)
        self.later = _LaterLayout(config)
        if config.fault in ("move", "staged_move"):
            # A real Tensor.to failure after valid checkpoint dispatch; no GPU
            # or unreliable OOM injection is needed to exercise this boundary.
            self.register_buffer("cache", torch.empty(1, device="meta"), persistent=False)


@register_model("_standalone_scratch")
class _StandaloneOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2))


@register_model("_direct_isolated")
class _IsolatedDirect(_StandaloneOwner):
    fastsafetensors_load_isolated = True


class _Record:
    def __init__(self, method):
        self.method = method

    def init_scalar(self, scope, key, replay_only):
        assert scope == GLOBAL_SCOPE
        return {
            "native.backend": self.method,
            "native.constructor": "{}",
            "native.format": "hf",
            "native.consumed": "[]",
            "native.loaded_fields": '{"": ["weight"]}',
        }[key]

    def bind(self, bank):
        self.bank = bank


class _ColdRecord:
    def __init__(self, reject):
        self.reject = reject

    def move_to_vmm(self, bank):
        # Exercise the real schema/record path; only CUDA VMM is replaced.
        if self.reject:
            raise RuntimeError("rank-local resident publication failed")

    def init_scalar(self, scope, key, build):
        return build()


def _check_cold(rank, stage, parallel, root, method, fault):
    config = SimpleNamespace(
        model_type="_resident_collective_cold", stage=stage, is_mtp=True,
        hooks=[], reject_binding=False, reject_constructor=False,
    )
    loader = NewModelLoader(
        config,
        NewLoaderConfig(
            device="cpu", load_method=method, parallelism_config=parallel,
            tp_size=2, tp_rank=rank - 2,
        ),
        model_path=str(root / "missing") if rank == 3 and fault == "source" else str(root),
    )
    cold = _ColdRecord(reject=rank == 3 and fault == "publish")
    with patch.object(loader, "_runtime_checkpoint_group", return_value=stage):
        if fault:
            expected = "rank-local resident publication" if fault == "publish" else "missing"
            with unittest.TestCase().assertRaisesRegex(RuntimeError, expected):
                loader.load(resident=cold, coordinated_resident_load=True)
            assert loader.loaded_method is None
            assert config.hooks == []
        else:
            model = loader.load(resident=cold, coordinated_resident_load=True)
            assert config.hooks == ["entered"]
            torch.testing.assert_close(model.weight, torch.arange(2).float())


def _check_replay(rank, stage, parallel, method, owner, fault, expected):
    config = SimpleNamespace(
        model_type=owner,
        is_mtp=True,
        stage=stage,
        hooks=[],
        reject_constructor=rank == 3 and fault == "constructor",
        reject_binding=rank == 3 and fault == "binding",
    )
    loader = NewModelLoader(
        config,
        NewLoaderConfig(
            device="cpu", load_method=method, parallelism_config=parallel,
            tp_size=2, tp_rank=rank - 2,
        ),
    )
    weight = torch.arange(1 if rank == 3 and fault == "shape" else 2).float()

    def loans():
        if rank == 3 and fault == "loan":
            raise ValueError("missing warm loan")
        return {}, None

    options = dict(
        resident=_Record(method),
        restored=[RestoredTensor(GLOBAL_SCOPE, "weight", weight)],
        prepare_loans=loans,
        coordinated_resident_load=True,
    )
    with patch.object(loader, "_runtime_checkpoint_group", return_value=stage):
        if fault:
            with unittest.TestCase().assertRaisesRegex(RuntimeError, expected):
                loader.load(**options)
            assert config.hooks == []
        else:
            model = loader.load(**options)
            assert config.hooks == ["entered"]
            torch.testing.assert_close(model.weight, weight)


def _check_direct(stage, parallel, root, method, fault):
    config = SimpleNamespace(
        model_type="_direct_collective", is_mtp=parallel.pp_size > 1, stage=stage,
        events=[], fault=fault if parallel.tp_rank == 1 else None, staged=fault == "staged_move",
    )
    loader = NewModelLoader(
        config, NewLoaderConfig(
            device="cpu", load_method=method, force_cpu_load_weights=True,
            parallelism_config=parallel, tp_size=2, tp_rank=parallel.tp_rank,
        ), model_path=str(root),
    )
    with patch.object(loader, "_runtime_checkpoint_group", return_value=stage):
        if fault:
            expected = "invalid later local layout" if fault == "layout" else "meta tensor"
            with unittest.TestCase().assertRaisesRegex(RuntimeError, expected):
                loader.load()
            assert "build" not in config.events, config.events
            assert loader.loaded_method is None
        else:
            model = loader.load()
            assert config.events == ["pack", "build"], config.events
            torch.testing.assert_close(model.weight, torch.arange(2).float())


def _direct_world(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        parallel = SimpleNamespace(
            world_size=2, world_rank=rank, pp_size=1, pp_rank=0,
            dp_size=1, dp_rank=0, tp_size=2, tp_rank=rank,
        )
        _check_direct(dist.group.WORLD, parallel, Path(rendezvous).parent, "auto", "move")
    finally:
        dist.destroy_process_group()


def _worker(rank, rendezvous):
    timeout = datetime.timedelta(seconds=10)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=4,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        stage = dist.new_group([2, 3], timeout=timeout)
        if rank in (2, 3):
            parallel = SimpleNamespace(
                world_size=4, world_rank=rank, pp_size=2, pp_rank=1,
                dp_size=1, dp_rank=0, tp_size=2, tp_rank=rank - 2,
            )
            for method, owner in (
                ("fastsafetensors", "_resident_collective"),
                ("scratch", "_resident_collective"),
                ("fastsafetensors", "_resident_collective_isolated"),
            ):
                for fault, expected in (
                    ("shape", "Resident shape or dtype mismatch"),
                    ("constructor", "incompatible resident constructor"),
                    ("binding", "incompatible local runtime binding"),
                    ("loan", "missing warm loan"),
                    (None, None),
                ):
                    _check_replay(rank, stage, parallel, method, owner, fault, expected)
            # The same barrier exists on a cold restart after local checkpoint
            # dispatch and publication. Use a real HF shard on each stage rank.
            for method in ("scratch", "auto"):
                for fault in ("publish", "source", None):
                    _check_cold(rank, stage, parallel, Path(rendezvous).parent, method, fault)
            for method in ("auto", *checkpoint_readers("scratch", "fastsafetensors")):
                for fault in ("move", "staged_move", "layout", None):
                    _check_direct(stage, parallel, Path(rendezvous).parent, method, fault)
        if rank == 0:
            # No other rank joins this auxiliary owner's standalone load.
            for owner, method in (
                ("_direct_isolated", "auto"),
                ("_direct_isolated", "scratch"),
                ("_standalone_scratch", "scratch"),
            ):
                loader = NewModelLoader(
                    SimpleNamespace(model_type=owner),
                    NewLoaderConfig(device="cpu", load_method=method),
                    model_path=str(Path(rendezvous).parent),
                )
                torch.testing.assert_close(loader.load().weight, torch.arange(2).float())
        # The first PP stage never calls the draft loader.
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ResidentReplayCollectiveTest(unittest.TestCase):
    def test_direct_move_error_before_executor_world_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file({"weight": torch.arange(2).float()}, str(Path(tmp) / "model.safetensors"))
            mp.start_processes(
                _direct_world, args=(tmp + "/gloo",), nprocs=2, start_method="fork"
            )

    def test_local_replay_agrees_before_executor_stage_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file({"weight": torch.arange(2).float()}, str(Path(tmp) / "model.safetensors"))
            mp.start_processes(
                _worker, args=(tmp + "/gloo",), nprocs=4, start_method="fork"
            )


if __name__ == "__main__":
    unittest.main()
