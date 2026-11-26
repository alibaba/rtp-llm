"""Resident draft loading integrates the service owner and real FST/Gloo on CPU."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.distributed.rank_layout import Group
from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.parallel_checkpoint_test_support import (
    GlooCheckpointLoader,
    runtime_config,
)
from rtp_llm.resident_weights import ResidentWeights


def _resident_draft_worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rdzv",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=40),
    )
    try:
        tail = dist.new_group([2, 3], backend="gloo", timeout=timedelta(seconds=30))
        for scenario in ("all_hit", "mixed", "no_store", "restore_error"):
            if rank >= 2:
                pc = runtime_config(rank, ep=1)
                config = SimpleNamespace(
                    model_type="_p0_main_parallel_runtime", is_mtp=True, expert_num=0
                )
                load_cfg = NewLoaderConfig.from_runtime(
                    pc,
                    device="cpu",
                    compute_dtype=torch.float32,
                    load_method="fastsafetensors",
                )
                loader = GlooCheckpointLoader(config, load_cfg, model_path=root)
                loader.groups = {Group.TP: tail, Group.STAGE: tail}
                manager = ResidentWeights.__new__(ResidentWeights)
                manager.slot_key, manager.verify = f"gpu{rank}/draft", False
                manager.discard_attach, manager.release_attach = Mock(), Mock()

                def attach():
                    if rank == 3 and scenario == "restore_error":
                        manager._restore_error = "simulated host restore failure"
                    if rank == 3 and scenario in ("mixed", "restore_error"):
                        return None
                    return []

                manager.attach = Mock(side_effect=attach)
                local = None if rank == 3 and scenario == "no_store" else manager
                service = BaseModel.__new__(BaseModel)
                service.resident_weights_config = SimpleNamespace(
                    enable=True, role="draft"
                )
                service.parallelism_config = pc
                observed = []

                def load(**kwargs):
                    warm = kwargs.get("restored") is not None
                    observed.append(warm)
                    if not warm:
                        # Real FST collective, real checkpoint bytes. Only the local
                        # store/VMM outcome is supplied by the fixture.
                        owner = loader.load()
                        expected = torch.arange(16, dtype=torch.float32).view(8, 2)
                        torch.testing.assert_close(
                            owner.attn.weight, expected.chunk(2)[pc.tp_rank]
                        )
                        assert loader.loaded_method.value == "fastsafetensors"
                    return warm

                with patch.object(service, "_create_resident_weights", return_value=local):
                    group = loader.checkpoint_group()
                    if scenario == "restore_error":
                        with unittest.TestCase().assertRaisesRegex(RuntimeError, "restore"):
                            service._load_with_residency("cpu", load, group=group)
                        manager.release_attach.assert_called_once()
                        assert not observed
                    else:
                        result = service._load_with_residency("cpu", load, group=group)
                        assert observed == [scenario == "all_hit"]
                        assert result == observed[0]
                        assert manager.discard_attach.called == (
                            scenario == "mixed" or (scenario == "no_store" and rank == 2)
                        )
                        assert manager.attach.called == (scenario != "no_store")
            # Non-draft ranks never participate in the availability vote, the
            # attach-state vote, the actual FST read, or the completion barrier.
            dist.barrier()
    finally:
        dist.destroy_process_group()


class ResidentDraftParallelRuntimeTest(unittest.TestCase):
    def test_resident_draft_mixed_hit_uses_real_tail_stage_fst(self):
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {
                    name + ".weight": torch.arange(16, dtype=torch.float32).view(8, 2)
                    for name in ("attn", "ffn", "lm_head")
                },
                os.path.join(root, "model.safetensors"),
            )
            torch.multiprocessing.start_processes(
                _resident_draft_worker, (root,), nprocs=4, start_method="fork"
            )


if __name__ == "__main__":
    unittest.main()
