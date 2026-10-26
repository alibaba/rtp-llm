"""Control-step=1000 regression through original C++ code.

The one-GPU test observes the actual ExpertBalancer/Python bridge with a fake
runtime. The eight-GPU test calls the actual EplbController and NCCL broadcast;
it does not claim a full service, scheduler or expert-weight migration test.
"""

import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from eplb_runtime_test_utils import FakeModel, fake_runtime_module

from rtp_llm.cpp.models.eplb.test import libth_eplb_py_wrapper_test as native


class ControlStep1000Test(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        guard = torch.inference_mode()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        patcher = fake_runtime_module()
        patcher.start()
        self.addCleanup(patcher.stop)
        self.model = FakeModel()
        self.balancer = native.ExpertBalancerStepTest(
            self.model, 0, interval=0, mode=1, control_step=1000
        )
        self.addCleanup(self.balancer.destroy)

    def updates(self):
        return [call for call in self.model.state.calls if isinstance(call, tuple)]

    def run_pending_steps(self, count, updates):
        for _ in range(count):
            self.balancer.step()
        self.assertEqual(self.updates(), updates)

    def test_boundary_warmup_last_request_pause_and_resume(self):
        # The controller counts normal forwards from startup, not from request.
        self.run_pending_steps(998, [])
        self.assertTrue(self.balancer.update_config(mode=3, update_time=7))
        self.run_pending_steps(1, [])  # global ordinary step 999
        for _ in range(17):
            self.balancer.step(warmup=True)
        self.assertEqual(self.updates(), [])
        # Latest request wins; the warmups must not publish it prematurely.
        self.assertTrue(self.balancer.update_config(mode=0, update_time=5))
        self.balancer.step()  # 1000
        updates = [("update", 0, 5)]
        self.assertEqual(self.updates(), updates)
        self.assertEqual(self.model.state.calls[-2:], [updates[-1], "step"])

        # A paused mode still polls the controller. update_time=0 can be
        # published at 2000 even though the old Python mode is NONE.
        self.assertTrue(self.balancer.update_config(mode=3, update_time=0))
        self.run_pending_steps(999, updates)
        self.balancer.step()
        updates.append(("update", 3, 0))
        self.assertEqual(self.updates(), updates)

        # Mode-only requests preserve the latest requested interval, including
        # one which has not yet been published at any control boundary.
        self.assertTrue(self.balancer.update_config(mode=2, update_time=7))
        self.assertTrue(self.balancer.update_config(mode=3, update_time=-1))
        self.run_pending_steps(999, updates)
        self.balancer.step()  # 3000: recover from interval zero
        updates.append(("update", 3, 7))
        self.assertEqual(self.updates(), updates)
        self.assertEqual(self.model.state.calls[-2:], [updates[-1], "step"])

        # A redundant update reaches the next heartbeat but does not reset the
        # Python runtime. An invalid request must not replace accepted state.
        self.assertTrue(self.balancer.update_config(mode=3, update_time=7))
        self.assertFalse(self.balancer.update_config(mode=4, update_time=7))
        self.run_pending_steps(1000, updates)
        self.assertEqual(self.model.state.runtime().mode, 3)
        self.assertEqual(self.model.state.runtime().interval, 7)

    def test_initial_none_cannot_create_runtime_online(self):
        model = FakeModel()
        balancer = native.ExpertBalancerStepTest(
            model, 0, interval=0, mode=0, control_step=1000
        )
        self.addCleanup(balancer.destroy)
        self.assertFalse(balancer.update_config(mode=3, update_time=7))
        self.assertTrue(balancer.update_config(mode=0, update_time=-1))
        for _ in range(1001):
            balancer.step()
        self.assertEqual(model.state.calls, [])


def controller_worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    broadcasts = []

    def broadcast(tensors, root, mode):
        assert (root, mode) == (0, 2), "controller must broadcast across DP_AND_TP"
        assert len(tensors) == 1
        value = tensors[0]
        assert (
            value.is_cuda and value.dtype == torch.int64 and tuple(value.shape) == (2,)
        )
        dist.broadcast(value, src=root)
        broadcasts.append(value.clone())

    native._broadcast_callback = broadcast
    try:
        controller = native.EplbControllerTest(1000, world, 1, 4)
        expected_history = []

        def finish_cycle(expected):
            before = len(broadcasts)
            for _ in range(999):
                assert controller.step() is None
            assert len(broadcasts) == before, "broadcast before step 1000"
            actual = controller.step()
            assert actual == expected
            assert len(broadcasts) == before + 1
            value = torch.tensor(actual, dtype=torch.int64, device="cuda")
            peers = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(peers, value)
            assert all(torch.equal(peer, value) for peer in peers)
            assert torch.equal(broadcasts[-1], value)
            expected_history.append(expected)

        # Root-authoritative contract. A non-root-only request is accepted by
        # controller.setData but cannot affect global state. This documents the
        # API routing hazard; it does not claim HTTP rejects that request.
        if rank == 1:
            controller.request(3, 7)
        finish_cycle((1, 4))
        if rank == 0:
            controller.request(2, 8)
            controller.request(3, -1)
        elif rank % 2:
            controller.request(0, 0)
        finish_cycle((3, 8))
        if rank == 0:
            controller.request(0, -1)
        finish_cycle((0, 8))
        if rank == 0:
            controller.request(3, 0)
        finish_cycle((3, 0))
        if rank == 0:
            controller.request(3, 4)
        finish_cycle((3, 4))
        # No new request: periodic publication retains root's requested state.
        finish_cycle((3, 4))
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world": world,
                    "control_step": 1000,
                    "ordinary_steps": 6000,
                    "broadcasts": len(broadcasts),
                    "published": expected_history,
                    "nonroot_only_request_published": False,
                }
            ),
            flush=True,
        )
    finally:
        del native._broadcast_callback
        dist.destroy_process_group()


class ControlStep1000DistributedTest(unittest.TestCase):
    def test_original_controller_real_eight_rank_nccl(self):
        self.assertEqual(
            torch.cuda.device_count(), 8, "requires one outer eight-GPU lock"
        )
        # Children inherit all eight assigned GPUs; no child runs gpu_lock.
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("TEST_TMPDIR")
        ) as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(controller_worker, args=(8, rendezvous), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
