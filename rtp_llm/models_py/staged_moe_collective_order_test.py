"""A rank-local expert can finish while its peer is still in reader P2P."""
import datetime
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewModelLoader, _StagedFSTLoadCoordinator
from rtp_llm.models_py.module_base import RtpModule


class _Pack:
    def process_weights_after_loading(self, layer):
        layer.packs += 1
        layer.weight.add_(100)


class _Expert(BaseMoEExperts):
    """Production expert lifecycle; stand-ins only for device pack and executor."""

    def __init__(self, builds, name):
        RtpModule.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        # The GPU-only branch is being exercised with Gloo on CPU. No native
        # GPU layout or backend is claimed by this test.
        self.w13 = torch.empty(0, device="meta")
        self.quant_method = _Pack()
        self._expert_layout_ready = self._post_load_done = False
        self.complete = False
        self.packs = 0
        self.builds = builds
        self.name = name

    def load_weights(self, weights):
        self.weight.copy_(weights["v"])
        self.complete = True

    def checkpoint_load_complete(self):
        return self.complete

    def _check_load_complete(self):
        assert self.complete

    def _maybe_build_fused_moe(self):
        # Simulate the TRTLLM executor's existing stage barrier, not a mock
        # that would silently permit mismatched collective orders.
        dist.barrier()
        self.builds.append(self.name)


def _run(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=10))
    try:
        builds = []
        model = RtpModule()
        model.first = _Expert(builds, "first")
        model.second = _Expert(builds, "second")
        coordinator = _StagedFSTLoadCoordinator(model, [model.first, model.second], "cpu")
        values = torch.tensor([7, 17], dtype=torch.int32)
        with torch.no_grad():
            if rank == 0:
                coordinator.load_child(model.first, {"v": values[:1]})
                # Rank 1 cannot finish its local expert or enter its barrier
                # until rank 0 supplies this simulated next FST transfer.
                dist.send(values, dst=1)
                coordinator.load_child(model.second, {"v": values[1:]})
            else:
                received = torch.empty_like(values)
                dist.recv(received, src=0)
                coordinator.load_child(model.second, {"v": received[1:]})
                coordinator.load_child(model.first, {"v": received[:1]})
            coordinator.finish()
            assert not builds
            # The owning loader calls this ordered pass after closing the reader.
            NewModelLoader._run_post_load_hooks(model)
            NewModelLoader._run_post_load_hooks(model)
            assert builds == ["first", "second"]
            assert (model.first.packs, model.second.packs) == (1, 1)
            result = torch.cat([model.first.weight, model.second.weight])
            peers = [torch.empty_like(result), torch.empty_like(result)]
            dist.all_gather(peers, result)
            assert all(torch.equal(peer, values + 100) for peer in peers)
    finally:
        dist.destroy_process_group()


class StagedMoECollectiveOrderTest(unittest.TestCase):
    def test_complete_p2p_before_collective_executor_builds(self):
        with tempfile.TemporaryDirectory() as work:
            mp.spawn(_run, (os.path.join(work, "store"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
