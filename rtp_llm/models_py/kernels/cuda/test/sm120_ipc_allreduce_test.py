import importlib.util
import json
import multiprocessing
import shutil
import socket
import tempfile
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _worker(rank, port, result_directory):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:" + port,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
        device_id=torch.device("cuda", rank),
    )
    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.kernels.cuda import sm120_ipc_allreduce as ipc

    ct._group_map[ct.Group.TP] = dist.group.WORLD
    ct._group_map[ct.Group.DP_AND_TP] = dist.group.WORLD
    ct._parallelism_config = SimpleNamespace(tp_size=2, dp_size=1, world_size=2)
    ct._initialized = True
    rows = []
    try:
        for cycle in range(2):
            comm = ipc.Sm120IpcAllReduce(dist.group.WORLD)
            ipc._communicator = comm
            for m in [1, 3, 8, 17, 32, 71, 128, 257, 512, 513, 1024, 4096]:
                torch.manual_seed(1548 + rank + m)
                x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
                original = x.clone()
                ref = x.clone()
                dist.all_reduce(ref)
                out = ct.all_reduce(x, ct.Group.TP)
                assert torch.equal(out, ref) and torch.equal(x, original)
                actual = ct.all_reduce(x, ct.Group.TP, inplace=True)
                assert actual.data_ptr() == x.data_ptr() and torch.equal(actual, ref)
                x.copy_(original)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = ct.all_reduce(x, ct.Group.TP)
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, ref)
                x.normal_()
                ref = x.clone()
                dist.all_reduce(ref)
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, ref)
                g.reset()
                del g
                rows.append(
                    {
                        "cycle": cycle,
                        "m": m,
                        "bit_exact": True,
                        "inplace": True,
                        "graph_update": True,
                    }
                )
            # Every finite BF16 bit pattern participates in TP2 sums, including subnormals.
            bits = (
                torch.arange(65536, device="cuda", dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            finite = bits[torch.isfinite(bits)]
            if rank:
                finite = finite.roll(1548)
            x = finite.repeat(2)[: 13 * 5120].reshape(13, 5120).contiguous()
            ref = x.clone()
            dist.all_reduce(ref)
            actual = ct.all_reduce(x, ct.Group.TP)
            assert torch.equal(actual.view(torch.uint16), ref.view(torch.uint16))
            # Unsupported matrix sizes and dtypes must continue through NCCL.
            for m, dtype in [(4097, torch.bfloat16), (3, torch.float32)]:
                x = torch.randn(m, 5120, device="cuda", dtype=dtype)
                ref = x.clone()
                dist.all_reduce(ref)
                assert not comm.is_supported(x)
                assert torch.equal(ct.all_reduce(x, ct.Group.TP), ref)
            ipc.destroy_communicator()
            assert ipc.get_communicator() is None
        result = {
            "passed": True,
            "rows": rows,
            "finite_bf16_patterns": len(finite),
            "lifecycle_cycles": 2,
            "fallbacks": True,
        }
    except Exception as error:
        traceback.print_exc()
        result = {"passed": False, "error": repr(error), "rows": rows}
    (Path(result_directory) / ("rank" + str(rank) + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    dist.destroy_process_group()
    if not result["passed"]:
        raise SystemExit(1)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.device_count() == 2
    and all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(2)),
    "requires two visible SM120 GPUs",
)
class Sm120IpcAllReduceTest(unittest.TestCase):
    def test_jit_survives_source_relocation(self):
        from rtp_llm.models_py.kernels.cuda import sm120_ipc_allreduce as ipc

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = []
            for name in ("first", "second"):
                folder = root / name
                folder.mkdir()
                for suffix in (".py", ".cu"):
                    source = Path(ipc.__file__).with_suffix(suffix)
                    shutil.copy2(source, folder / source.name)
                spec = importlib.util.spec_from_file_location(
                    "ipc_" + name, folder / Path(ipc.__file__).name
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sizes.append(module._module().signal_size())
                shutil.rmtree(folder)
            self.assertEqual(sizes[0], sizes[1])
            self.assertGreater(sizes[0], 0)

    def test_nccl_equivalence_graph_replay_and_reinitialization(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as results:
            processes = [
                context.Process(target=_worker, args=(rank, port, results))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=180)
                self.assertTrue(all(p.exitcode == 0 for p in processes))
                for rank in range(2):
                    result = json.loads(
                        (Path(results) / f"rank{rank}.json").read_text()
                    )
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(len(result["rows"]), 24)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
