import datetime
import os
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather import (
    _PROCESS_LIFETIME_IPC,
    Qwen4HeadIpcGather,
    prepare_head_ipc_gather,
)


class HeadIpcGatherTest(unittest.TestCase):
    def test_gather_rechecks_close_after_waiting_for_launch_lock(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace.closed = False
        workspace.device = torch.device("cuda", torch.cuda.current_device())
        workspace.ops = Mock()

        @contextmanager
        def closed_while_waiting():
            workspace.closed = True
            workspace.buffers = workspace.signals = []
            yield

        workspace._lock = closed_while_waiting()
        value = torch.empty(1, 124160, device=workspace.device, dtype=torch.bfloat16)
        self.assertIsNone(workspace.try_gather(value))
        workspace.ops.head_ipc_gather.assert_not_called()

    def test_unilateral_cleanup_is_local_idempotent_and_preserves_ipc_owners(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace._lock = threading.Lock()
        workspace.closed = False
        workspace.buffers, workspace.signals = [101, 202], [303, 404]
        owners = (workspace.buffers, workspace.signals)
        workspace.ops = Mock()
        count = len(_PROCESS_LIFETIME_IPC)
        try:
            with patch(
                "torch.distributed.barrier",
                side_effect=AssertionError("collective cleanup"),
            ), patch(
                "torch.cuda.synchronize",
                side_effect=AssertionError("device wait in cleanup"),
            ):
                workspace.close()
                workspace.close()
            self.assertTrue(workspace.closed)
            self.assertEqual(workspace.buffers, [])
            self.assertEqual(workspace.signals, [])
            self.assertEqual(len(_PROCESS_LIFETIME_IPC), count + 1)
            self.assertIs(_PROCESS_LIFETIME_IPC[-1][0], owners[0])
            self.assertIs(_PROCESS_LIFETIME_IPC[-1][1], owners[1])
            workspace.ops.head_ipc_close.assert_not_called()
        finally:
            del _PROCESS_LIFETIME_IPC[count:]

    def test_unsupported_layouts_allocate_nothing(self):
        config = SimpleNamespace(_qwen4_full_fp8_nvfp4=True)
        pc = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        head = SimpleNamespace(
            weight=torch.empty(124160, 2560, device="meta", dtype=torch.float8_e4m3fn)
        )
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather.Qwen4HeadIpcGather",
            side_effect=AssertionError("unexpected allocation"),
        ):
            for key, value in [
                ("role_type", "PREFILL"),
                ("tp_size", 4),
                ("dp_size", 2),
                ("local_world_size", 4),
            ]:
                changed = SimpleNamespace(**{**vars(pc), key: value})
                self.assertIsNone(prepare_head_ipc_gather(config, changed, head))
            self.assertIsNone(prepare_head_ipc_gather(config, pc, head))

    def test_invalid_tensor_domains_do_not_touch_native_ops(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace.closed = False
        workspace.device = torch.device("cuda", torch.cuda.current_device())
        x = torch.empty(4, 124160, device=workspace.device, dtype=torch.bfloat16)
        invalid = [
            None,
            x.cpu(),
            x.float(),
            x[:0],
            x[:, :320],
            x.t(),
            x.unsqueeze(0),
            torch.empty(33, 124160, device=workspace.device, dtype=torch.bfloat16),
        ]
        for tensor in invalid:
            self.assertIsNone(workspace.try_gather(tensor))
        workspace.closed = True
        self.assertIsNone(workspace.try_gather(x))

    def test_graph_capture_defers_without_allocating_or_changing_owner(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace.closed = False
        workspace.device = torch.device("cuda", torch.cuda.current_device())
        x = torch.zeros(4, 124160, device=workspace.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertIsNone(workspace.try_gather(x))
            output = x + 1
        graph.replay()
        self.assertTrue(torch.equal(output, x + 1))
        graph.reset()

    def test_model_logits_own_transport_and_native_fallback(self):
        local = torch.ones(4, 8, dtype=torch.bfloat16)
        model = SimpleNamespace()
        self.assertIsNone(Qwen4ExpModel.compute_logits(model, local))
        model._fp8_lm_head = object()
        model._compute_local_logits = lambda _: local
        model.parallelism_config = SimpleNamespace(tp_size=2)
        expected = torch.ones(4, 16)
        seen = []
        model._head_ipc_gather = SimpleNamespace(
            try_gather=lambda x: (seen.append(x), expected)[1]
        )
        self.assertIs(Qwen4ExpModel.compute_logits(model, local), expected)
        self.assertIs(seen[0], local)


def _gather_worker():
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    workspace = Qwen4HeadIpcGather(dist.group.WORLD, torch.device("cuda", rank))
    if os.environ.get("RTP_TEST_HEAD_ASYMMETRIC_CLOSE") == "1":
        store = dist.distributed_c10d._get_default_store()
        if rank == 0:
            workspace.close()
            store.set("head_rank0_closed", "1")
            store.wait(["head_rank1_closed"], datetime.timedelta(seconds=10))
        else:
            # No collective here can accidentally match rank 0's cleanup.
            store.wait(["head_rank0_closed"], datetime.timedelta(seconds=10))
            workspace.close()
            store.set("head_rank1_closed", "1")
        dist.destroy_process_group()
        return
    try:
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        for iteration, rows in enumerate((1, 4, 32, 4)):
            with torch.cuda.stream(streams[iteration % 2]):
                base = torch.arange(rows * 124160, device="cuda").reshape(rows, 124160)
                base = base.remainder(37).float() - 18 + iteration
                local = (base + rank * 64).bfloat16()
                actual = workspace.try_gather(local)
                assert actual is not None, "supported input fell back"
                expected = torch.cat(
                    (base.bfloat16(), (base + 64).bfloat16()), dim=1
                ).float()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        workspace.close()
        workspace.close()
        assert workspace.try_gather(local) is None
        dist.destroy_process_group()


class HeadIpcGatherIntegrationTest(unittest.TestCase):
    def test_unilateral_close_does_not_wait_for_peer_cleanup(self):
        env = dict(os.environ, RTP_TEST_HEAD_ASYMMETRIC_CLOSE="1")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                __file__,
            ],
            env=env,
            check=True,
            timeout=90,
        )

    def test_rank_order_live_inputs_stream_handoff_and_close(self):
        self.assertEqual(torch.cuda.device_count(), 2)
        self.assertEqual(torch.cuda.get_device_capability(0), (12, 0))
        self.assertTrue(torch.cuda.can_device_access_peer(0, 1))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                __file__,
            ],
            check=True,
            timeout=240,
        )


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _gather_worker()
    else:
        unittest.main()
