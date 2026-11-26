"""A local warm-restart error must not race a peer's executor stage barrier."""

import datetime
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

    def process_weights_after_loading(self):
        # The NVFP4 DeepEP executor joins such a barrier at construction.
        self.config.hooks.append("entered")
        dist.barrier(group=self.config.stage)


@register_model("_resident_collective_isolated")
class _IsolatedOwner(_Owner):
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


def _check_replay(rank, stage, parallel, method, owner, fault, expected):
    config = SimpleNamespace(
        model_type=owner,
        is_mtp=True,
        stage=stage,
        hooks=[],
        reject_constructor=rank == 3 and fault == "constructor",
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
        coordinated_resident_replay=True,
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


def _worker(rank, rendezvous):
    timeout = datetime.timedelta(seconds=10)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=4, timeout=timeout
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
                    ("loan", "missing warm loan"),
                    (None, None),
                ):
                    _check_replay(rank, stage, parallel, method, owner, fault, expected)
        # The first PP stage never calls the draft loader.
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ResidentReplayCollectiveTest(unittest.TestCase):
    def test_local_replay_agrees_before_executor_stage_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(
                _worker, args=(tmp + "/gloo",), nprocs=4, start_method="fork"
            )


if __name__ == "__main__":
    unittest.main()
