"""Eight-rank post-policy agreement and mutation-safe worker recovery."""

import socket
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_collapse_dp_agreement_sm120 import (
    run_rank,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import checkpoint


class PostDPAgreementSM120Test(unittest.TestCase):
    def test_cold_policy_rejection_and_ready_worker_recovery(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            torch.manual_seed(41024)
            config = checkpoint(Path(directory))
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config, "post"), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
