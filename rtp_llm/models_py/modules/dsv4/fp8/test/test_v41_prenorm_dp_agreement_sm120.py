"""Eight-rank prenorm policy and precision agreement with composed mHC."""

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


class PrenormDPAgreementSM120Test(unittest.TestCase):
    extra_choices = {}

    def test_cold_policy_and_precision_rejection_and_recovery(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as directory, socket.socket() as sock:
            torch.manual_seed(41026)
            config = checkpoint(Path(directory))
            config.update(self.extra_choices)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config, "prenorm"), nprocs=8, join=True)


class PrenormCombinedDPAgreementSM120Test(PrenormDPAgreementSM120Test):
    extra_choices = {
        "mhc_collapse_mode": "ordered_fp32",
        "mhc_post_mode": "ordered_fp32",
    }


if __name__ == "__main__":
    unittest.main()
