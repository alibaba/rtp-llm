"""Exercise the ali native error result, not just a mocked cudaMemcpy result."""

import os
import tempfile
import unittest
from pathlib import Path

import torch
from fastsafetensors import ParallelLoader, SingleGroup
from safetensors.torch import save_file

from rtp_llm.models_py.v4_fst_nogds import protect_v4_nogds


class ShortFirstRead:
    """First GPU read goes past EOF; the native worker returns nullptr."""

    def __init__(self, worker):
        self.worker = worker
        self.results = []

    def submit_read(self, fd, buf, off, size, dest):
        return self.worker.submit_read(fd, buf, os.fstat(fd).st_size, size, dest)

    def wait_read(self, token):
        result = self.worker.wait_read(token)
        self.results.append(result)
        return result


class FullTransfer(unittest.TestCase):
    def test_failed_native_extent_is_repaired_before_publication(self):
        self.assertTrue(torch.cuda.is_available(), "Requires an actual CUDA worker")
        # Cross the retry's pinned-block boundary and cover the partial tail.
        source = torch.arange((16 << 20) + 257, dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as d:
            filename = str(Path(d) / "weights.safetensors")
            save_file({"scale_bytes": source}, filename)
            loader = ParallelLoader(
                pg=SingleGroup(),
                hf_weights_files=[filename],
                use_tqdm_on_load=False,
                device="cuda:0",
                nogds=True,
                use_shm=False,
                bbuf_size_kb=1024,
            )
            make = loader.loader.copier_constructor
            workers = []

            def inject(*args):
                copier = make(*args)
                worker = ShortFirstRead(copier.reader)
                workers.append(worker)
                copier.reader = worker
                return copier

            loader.loader.copier_constructor = inject
            protect_v4_nogds(loader)
            seen = []
            try:
                for name, data in loader.iterate_weights():
                    seen.append(name)
                    self.assertTrue(data.is_cuda)
                    self.assertTrue(torch.equal(data.cpu(), source))
            finally:
                loader.loader.close()
            self.assertEqual(seen, ["scale_bytes"])
            self.assertEqual([r for w in workers for r in w.results], [0])


if __name__ == "__main__":
    unittest.main()
