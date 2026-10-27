"""CPU dispatch guards; --gpu runs four-rank bitwise and timing checks.

Load the production files directly so tests do not require the model/config SO.
"""

import importlib.util
import json
import os
import statistics
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

BASE = Path(__file__).resolve().parents[1]
if not (BASE / "symm_mem.py").exists():
    BASE = Path(__file__).resolve().parent


def load(name):
    fullname = "rtp_llm.models_py.distributed." + name
    spec = importlib.util.spec_from_file_location(fullname, BASE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


symm = load("symm_mem")
cp = load("pynccl_cp")


class Guards(unittest.TestCase):
    def setUp(self):
        self.comm = symm.TorchSymmMemCommunicator.__new__(symm.TorchSymmMemCommunicator)
        self.comm.disabled = False
        self.comm.device = torch.device("cpu")
        self.comm.dtype = torch.bfloat16
        self.comm.world_size = 4
        self.comm.max_size = 64 * 1024**2
        self.comm._owner_stream = None
        self.stream = patch.object(
            torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=11)
        )
        self.stream.start()
        self.addCleanup(self.stream.stop)

    def test_size_and_row_guards(self):
        for rows in [1, 512, 544, 3114]:
            x = torch.empty(rows, 576, dtype=torch.bfloat16)
            self.assertEqual(self.comm.should_torch_symm_mem_allgather(x), rows <= 512)
            self.assertTrue(
                self.comm.should_torch_symm_mem_allgather(x, max_gathered_rows=None)
            )
        for x in [
            torch.empty(0, 576, dtype=torch.bfloat16),
            torch.empty(8, 576, dtype=torch.float32),
            torch.empty(8, 576, dtype=torch.bfloat16).t(),
            torch.empty(1, 3, dtype=torch.bfloat16),
            torch.empty(64 * 1024**2 // 8, dtype=torch.bfloat16),
            torch.empty(17, dtype=torch.bfloat16)[1:],
        ]:
            self.assertFalse(
                self.comm.should_torch_symm_mem_allgather(x, max_gathered_rows=None)
            )

    def test_shared_allreduce_gather_owner(self):
        x = torch.empty(8, 576, dtype=torch.bfloat16)
        self.comm._claim_stream()
        self.assertTrue(self.comm.should_torch_symm_mem_allgather(x))
        self.assertTrue(self.comm.should_torch_symm_mem_allreduce(x))
        with patch.object(
            torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=22)
        ):
            self.assertFalse(self.comm.should_torch_symm_mem_allgather(x))
            self.assertFalse(self.comm.should_torch_symm_mem_allreduce(x))
            with self.assertRaises(RuntimeError):
                self.comm._claim_stream()

    def test_dispatch_group_device_capture_and_switch(self):
        group = object()
        x = SimpleNamespace(
            is_cuda=True,
            device=torch.device("cuda", 0),
            numel=lambda: 8,
            size=lambda d: 2,
            shape=(2, 4),
        )
        output = torch.arange(32).view(4, 2, 4)
        comm = SimpleNamespace(
            group=group,
            device=x.device,
            world_size=4,
            should_torch_symm_mem_allgather=lambda x, **kwargs: True,
            all_gather=lambda x: output,
        )
        with patch.object(cp, "_TORCH_SYMM_GATHER", True), patch.object(
            symm, "_symm_mem_comm", comm
        ), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            result = cp._try_torch_symm_all_gather(x, group)
            self.assertEqual(tuple(result.shape), (8, 4))
            self.assertIsNone(cp._try_torch_symm_all_gather(x, object()))
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                self.assertIsNone(cp._try_torch_symm_all_gather(x, group))
            with patch.object(cp, "_TORCH_SYMM_GATHER", False):
                self.assertIsNone(cp._try_torch_symm_all_gather(x, group))
            comm.device = torch.device("cuda", 1)
            self.assertIsNone(cp._try_torch_symm_all_gather(x, group))

    def test_cpu_and_empty_rejected(self):
        self.assertIsNone(cp._try_torch_symm_all_gather(torch.empty(2, 4), object()))
        x = SimpleNamespace(is_cuda=True, numel=lambda: 0)
        self.assertIsNone(cp._try_torch_symm_all_gather(x, object()))

    def test_dispatch_preserves_fallback(self):
        x = torch.empty(2, 4, dtype=torch.bfloat16)
        group = object()
        with patch.object(cp, "_OPT_ENABLED", True), patch.object(
            cp, "_resolve_process_group", return_value=group
        ), patch.object(cp, "_world_size", return_value=4), patch.object(
            cp, "_try_torch_symm_all_gather", return_value=None
        ), patch.object(
            cp, "_symm_output", return_value=None
        ), patch.object(
            cp, "all_gather_into_tensor"
        ) as gather:
            result = cp.all_gather(x, group, role="mla_ckv")
            self.assertEqual(tuple(result.shape), (8, 4))
            gather.assert_called_once_with(result, x, group=group)

    def test_overlap_uses_separate_comm_and_retains_output(self):
        group = object()
        x = torch.empty(2, 4, dtype=torch.bfloat16)
        out = torch.empty(8, 4, dtype=torch.bfloat16)
        side = object()
        lib = SimpleNamespace(all_gather=lambda *a: self.fail("unexpected NCCL"))
        with patch.object(
            cp, "_resolve_process_group", return_value=group
        ), patch.object(cp, "_comm_key", return_value=(group, 0)), patch.dict(
            cp._OVERLAP_COMMS, {(group, 0): (lib, object())}
        ), patch.dict(
            cp._OVERLAP_SYMM_COMMS, {(group, 0): side}
        ), patch.object(
            cp, "_validate_all_gather"
        ) as validate, patch.object(
            cp, "_try_torch_symm_all_gather", return_value=out
        ) as gather:
            self.assertIs(cp.all_gather_kv_overlap(x, group, output=out), out)
            validate.assert_called_once_with(out, x, group)
            gather.assert_called_once_with(x, group, communicator=side, output=out)


def gpu_worker(rank, port, dest):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    group = dist.group.WORLD
    cp._resolve_process_group = lambda g: g
    cp._OPT_ENABLED = True
    cp._get_comm(group, device)
    comm = symm.init_symm_mem_communicator(group)
    assert comm is not None and not comm.disabled, "multicast unavailable"
    records = []
    cases = [
        (1, 576),
        (3, 576),
        (512, 576),
        (544, 576),
        (3114, 576),
        (8192, 576),
        (544, 64),
        (3114, 64),
    ]

    def gather(x, enabled):
        cp._TORCH_SYMM_GATHER = enabled
        return cp.all_gather(x, group, role="mla_ckv")

    for rows, width in cases:
        x = (
            (torch.arange(rows * width, device=device).reshape(rows, width) % 127)
            + rank
        ).to(torch.bfloat16)
        # Include zero-filled padded rows and independent request boundaries.
        if rows > 3:
            x[-1].zero_()
        cp._TORCH_SYMM_GATHER = True
        assert cp._try_torch_symm_all_gather(x, group) is not None
        expected = gather(x, False)
        actual = gather(x, True)
        saved = actual.clone()
        # Scratch reuse must not alias returned results, even with alternating shapes.
        for step in range(10):
            y = x + step
            torch.testing.assert_close(
                gather(y, True), gather(y, False), rtol=0, atol=0
            )
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        row = dict(rows=rows, width=width, total_bytes=x.nbytes * 4)
        for enabled in [False, True]:
            for _ in range(10):
                gather(x, enabled)
            torch.cuda.synchronize()
            dist.barrier()
            pairs = []
            for _ in range(40):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                a.record()
                y = gather(x, enabled)
                b.record()
                pairs.append((a, b))
            torch.cuda.synchronize()
            row["symm_us" if enabled else "nccl_us"] = statistics.median(
                a.elapsed_time(b) * 1000 for a, b in pairs
            )
        records.append(row)
    # A second stream must decline shared scratch and preserve NCCL correctness.
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        cp._TORCH_SYMM_GATHER = True
        assert cp._try_torch_symm_all_gather(x, group) is None
        other_result = gather(x, True)
    torch.cuda.current_stream().wait_stream(other)
    torch.testing.assert_close(other_result, gather(x, False), rtol=0, atol=0)
    # Byte/FP32 paths remain NCCL and bit-exact.
    for dtype in [torch.uint8, torch.float32, torch.float16]:
        y = x.to(dtype)
        assert cp._try_torch_symm_all_gather(y, group) is None
        torch.testing.assert_close(gather(y, True), gather(y, False), rtol=0, atol=0)
    # Separate symmetric buffers allow overlapping main/side collectives.
    cp._TORCH_SYMM_GATHER = True
    side_stream = cp.warmup_kv_overlap(group)
    assert cp._OVERLAP_SYMM_COMMS[(group, rank)] is not None
    torch.cuda.synchronize()
    held = []
    for step in range(20):
        cp._TORCH_SYMM_GATHER = True
        main_input = torch.full(
            (544, 576), rank + step, dtype=torch.bfloat16, device=device
        )
        side_input = torch.full(
            (544, 576), rank + step + 32, dtype=torch.bfloat16, device=device
        )
        output = torch.empty((2176, 576), dtype=torch.bfloat16, device=device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            actual_side = cp.all_gather_kv_overlap(side_input, group, output=output)
            assert actual_side is output
            side_input.record_stream(side_stream)
        actual_main = gather(main_input, True)
        torch.cuda.current_stream().wait_stream(side_stream)
        expected_main = gather(main_input, False)
        expected_side = gather(side_input, False)
        held.append((actual_main, expected_main, actual_side, expected_side))
    for a, b, c, d in held:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(c, d, rtol=0, atol=0)
    Path(dest).mkdir(exist_ok=True, parents=True)
    Path(dest, f"rank{rank}.json").write_text(json.dumps(records, indent=2))
    dist.barrier()
    cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(gpu_worker, args=(port, sys.argv[-1]), nprocs=4, join=True)
        print("PASS four-rank bitwise, retained outputs, stream fallback and timings")
    else:
        unittest.main()
