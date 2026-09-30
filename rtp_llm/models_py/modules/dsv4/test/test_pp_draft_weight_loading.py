"""Real four-process loader regression: only the final PP stage loads a draft."""

import datetime
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.database import CkptDatabase


def _worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "store"),
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=15),
    )
    try:
        if rank >= 2:
            loader = ModelLoader.__new__(ModelLoader)
            loader._apply_pp_partition = False
            loader._load_config = SimpleNamespace(pp_size=2)
            # Existing reader's CPU backend, real safetensors bytes, no mocked
            # collective or loader. Earlier stages wait in WORLD meanwhile.
            database = CkptDatabase(str(root / "checkpoint"))
            tensors = dict(
                database.fastsafetensors_weights_iterator(
                    "cpu",
                    False,
                    use_distributed=loader._fastsafetensors_use_distributed(),
                )
            )
            torch.testing.assert_close(
                tensors["draft.weight"],
                torch.arange(12, dtype=torch.float32).view(3, 4),
            )
            (root / f"loaded_{rank}.json").write_text(
                json.dumps({"rank": rank, "keys": list(tensors)})
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class PPDraftWeightLoadingTest(unittest.TestCase):
    def test_cuda_local_reader_owns_values_after_close(self):
        self.assertTrue(torch.cuda.is_available(), "This case requires a GPU lease")
        # The pinned CUDA reader uses O_DIRECT. Bazel's writable test directory
        # is on the build filesystem; the host/container /tmp can be tmpfs,
        # which rejects O_DIRECT even when the file exists.
        with tempfile.TemporaryDirectory(
            prefix="rtp_pp_draft_cuda_", dir=os.environ.get("TEST_TMPDIR")
        ) as directory:
            expected = torch.arange(12, dtype=torch.float32).view(3, 4)
            save_file(
                {"draft.weight": expected}, str(Path(directory) / "model.safetensors")
            )
            database = CkptDatabase(directory)
            tensors = dict(
                database.fastsafetensors_weights_iterator(
                    "cuda:0",
                    False,
                    use_distributed=False,
                )
            )
            self.assertTrue(tensors["draft.weight"].is_cuda)
            # Exercise allocator reuse after the reader's explicit close.
            trash = [torch.full((1024,), 77.0, device="cuda") for _ in range(16)]
            torch.testing.assert_close(tensors["draft.weight"].cpu(), expected)
            self.assertEqual(len(trash), 16)

    def test_policy_preserves_target_and_single_stage_loading(self):
        loader = ModelLoader.__new__(ModelLoader)
        for pp_size, apply_partition, expected in (
            (1, True, True),
            (1, False, True),
            (2, True, True),
            (2, False, False),
            (4, False, False),
        ):
            loader._load_config = SimpleNamespace(pp_size=pp_size)
            loader._apply_pp_partition = apply_partition
            self.assertIs(loader._fastsafetensors_use_distributed(), expected)

    def test_last_stage_loads_without_entering_other_stages_world_collectives(self):
        with tempfile.TemporaryDirectory(prefix="rtp_pp_draft_") as directory:
            root = Path(directory)
            ckpt = root / "checkpoint"
            ckpt.mkdir()
            save_file(
                {"draft.weight": torch.arange(12, dtype=torch.float32).view(3, 4)},
                str(ckpt / "model.safetensors"),
            )
            # CPU fixture avoids GDS and does not initialize CUDA.
            prior = os.environ.get("FASTSAFETENSORS_NOGDS")
            os.environ["FASTSAFETENSORS_NOGDS"] = "1"
            context = None
            try:
                context = mp.spawn(_worker, args=(directory,), nprocs=4, join=False)
                deadline = time.monotonic() + 60
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("PP subset load did not complete before the deadline")
                self.assertTrue(
                    all((root / f"loaded_{rank}.json").exists() for rank in (2, 3))
                )
                self.assertTrue(all(p.exitcode == 0 for p in context.processes))
            finally:
                if context is not None:
                    for p in context.processes:
                        if p.is_alive():
                            p.terminate()
                        p.join(timeout=5)
                    for p in context.processes:
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=5)
                if prior is None:
                    os.environ.pop("FASTSAFETENSORS_NOGDS", None)
                else:
                    os.environ["FASTSAFETENSORS_NOGDS"] = prior


if __name__ == "__main__":
    unittest.main(verbosity=2)
