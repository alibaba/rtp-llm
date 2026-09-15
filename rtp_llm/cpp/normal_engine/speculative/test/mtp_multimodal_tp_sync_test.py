"""Two-rank TP transfers of CUDA positions and an image-to-empty MTP update."""

import datetime
import multiprocessing as mp
import tempfile
import time
import unittest
from multiprocessing.connection import wait
from pathlib import Path

import torch
import torch.distributed as dist


def _assert_image(snapshot):
    assert snapshot["positions_on_cuda"] and snapshot["position_device_hint"]
    assert snapshot["positions"].tolist() == [1, 2, 3, 4, 5, 6]
    assert snapshot["mask"].tolist() == [0, 1]
    assert snapshot["locs"].tolist() == [0]
    assert snapshot["features_present"] and snapshot["extra_present"]
    assert len(snapshot["features"]) == len(snapshot["extra"]) == 1
    torch.testing.assert_close(snapshot["features"][0], torch.tensor([[17.0, 19.0]]))
    torch.testing.assert_close(
        snapshot["extra"][0], torch.tensor([101.0, 102.0, 201.0, 202.0])
    )


def _assert_shifted(snapshot):
    assert snapshot["positions_on_cuda"] and snapshot["position_device_hint"]
    assert snapshot["positions"].tolist() == [101, 102, 103, 104, 105, 106]
    assert snapshot["tokens"].tolist() == [5, 7]
    assert snapshot["mask"].tolist() == [1, 1]
    assert snapshot["locs"].numel() == 0
    assert snapshot["features"] == []
    assert snapshot["extra"] == []


def _rank_worker(rank, directory):
    torch.cuda.set_device(rank)
    # Load RTP type registrations before importing the test-only extension.
    import rtp_llm.ops  # noqa: F401
    from rtp_llm.cpp.normal_engine.speculative.test import (
        libmtp_multimodal_tp_sync_wrapper as wrapper,
    )

    initialized = False
    broadcaster_initialized = False
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{directory}/nccl_store",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=60),
        )
        initialized = True
        wrapper.init_exec_ctx(rank, False, False, 0)
        gpu_broadcasts = []
        cpu_callback_broadcasts = []

        def broadcast(tensors, root, mode):
            assert tensors
            for tensor in tensors:
                # The extra-input element-count shape still uses execBroadcast
                # instead of execBroadcastCpu. Match the production callback's
                # CPU -> CUDA -> CPU bridge while all other host metadata uses UDS.
                on_cpu = not tensor.is_cuda
                payload = tensor.to(device=f"cuda:{rank}") if on_cpu else tensor
                dist.broadcast(payload, src=root)
                if on_cpu:
                    tensor.copy_(payload)
                    cpu_callback_broadcasts.append((root, mode))
                else:
                    gpu_broadcasts.append((root, mode))

        def unexpected_collective(*args):
            raise AssertionError("tpSyncModelInputs must only broadcast")

        wrapper.register_comm_ops(broadcast, unexpected_collective, unexpected_collective)
        wrapper.init_cpu_tp_broadcaster(rank, 2, f"{directory}/cpu_bcast")
        broadcaster_initialized = True
        assert wrapper.cpu_broadcaster_initialized()

        fixture = wrapper.Fixture(rank)
        initial = fixture.snapshot()
        if rank == 1:
            assert initial["mask"].tolist() == [9, 9]
            assert not initial["positions_on_cuda"]
            assert not initial["position_device_hint"]
            assert initial["positions"].tolist() == [-1, -2, -3, -4, -5, -6]
        else:
            assert initial["position_device_hint"], "CUDA positions must have a TP device bit"
        fixture.sync()
        torch.cuda.synchronize()
        _assert_image(fixture.snapshot())
        first_gpu_calls = len(gpu_broadcasts)
        assert first_gpu_calls > 0, "first round must transfer the CUDA feature/extra payload"
        assert cpu_callback_broadcasts, "extra-input shape must exercise the NCCL CPU bridge"
        dist.barrier()

        # Keep the exact same input object on rank 1: it must still hold the
        # target image before the second production tpSync call clears it.
        if rank == 0:
            fixture.shift_root()
            torch.cuda.synchronize()
            _assert_shifted(fixture.snapshot())
        else:
            _assert_image(fixture.snapshot())
        dist.barrier()
        fixture.sync()
        torch.cuda.synchronize()
        final = fixture.snapshot()
        _assert_shifted(final)
        if rank == 1:
            assert not final["features_present"]
            assert not final["extra_present"]
        assert len(gpu_broadcasts) > first_gpu_calls, "second round must transfer shifted CUDA tokens/hidden states"
        dist.barrier()
    finally:
        if broadcaster_initialized:
            wrapper.destroy_cpu_tp_broadcaster()
        wrapper.clear_comm_ops()
        if initialized:
            dist.destroy_process_group()


class MtpMultimodalTpSyncTest(unittest.TestCase):
    def test_two_round_broadcast_clears_peer_visual_fields(self):
        # This target reserves two GPUs; missing devices must not produce a skip/pass.
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "two visible CUDA devices are required")
        context = mp.get_context("spawn")
        # Keep Unix-domain socket paths shorter than sockaddr_un.sun_path.
        with tempfile.TemporaryDirectory(prefix="mtp-tp-", dir="/tmp") as directory:
            processes = [
                context.Process(target=_rank_worker, args=(rank, str(Path(directory))), name=f"mtp-tp-{rank}")
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                pending = list(processes)
                deadline = time.monotonic() + 180
                while pending:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, "two-rank MTP TP regression timed out")
                    ready = wait([process.sentinel for process in pending], timeout=remaining)
                    self.assertTrue(ready, "two-rank MTP TP regression timed out")
                    for process in list(pending):
                        if process.sentinel in ready:
                            process.join()
                            self.assertEqual(process.exitcode, 0, f"{process.name} failed")
                            pending.remove(process)
            finally:
                for process in processes:
                    if process.pid is not None and process.is_alive():
                        process.terminate()
                for process in processes:
                    if process.pid is not None:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
