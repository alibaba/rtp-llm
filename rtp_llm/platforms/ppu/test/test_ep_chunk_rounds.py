"""CPU collective control-flow regression; does not validate DeepEP kernels."""

import ast
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    # Load the production method without importing PPU-only native libraries.
    path = Path(__file__).resolve().parents[1] / "models/dsv4/ppu_ep_moe.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "PpuEPMoE")
    forward = next(n for n in cls.body
                   if isinstance(n, ast.FunctionDef) and n.name == "forward")
    namespace = {"torch": torch, "dist": dist}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    try:
        for sizes in ((16385, 1), (16385, 0), (0, 0), (16384, 1)):
            calls = []

            def run_chunk(x, ids, out, **kwargs):
                calls.append(x.size(0))
                dist.all_reduce(torch.ones(1))
                out.copy_(x)

            model = SimpleNamespace(
                dim=1, max_tokens_per_rank=16384, _is_decode_role=False,
                chunking_enabled=True, _tp_size=1, _ep_size=2,
                _observer_factory=None, _record_function_scope=nullcontext,
                _run_chunk=run_chunk,
            )
            x = torch.ones(sizes[rank], 1)
            ids = torch.zeros(sizes[rank], dtype=torch.long)
            result = namespace["forward"](model, x, ids)
            rounds = max((max(sizes) + 16383) // 16384, 1)
            assert len(calls) == rounds, (rank, sizes, calls)
            assert sum(calls) == sizes[rank]
            torch.testing.assert_close(result, x)
            if rounds > 1:
                model.chunking_enabled = False
                try:
                    namespace["forward"](model, x, ids)
                except ValueError as error:
                    assert "chunking disabled" in str(error)
                else:
                    raise AssertionError("Every rank must reject disabled chunking")
                dist.barrier()
    finally:
        dist.destroy_process_group()


class EpChunkRoundsTest(unittest.TestCase):
    def test_uneven_and_empty_ranks_participate_in_all_rounds(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_worker, args=("file://" + directory + "/rendezvous",), nprocs=2)


if __name__ == "__main__":
    unittest.main()
