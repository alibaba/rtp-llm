import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.vocab_parallel_logits import (
    gather_vocab_parallel_logits,
)

MODULE = "rtp_llm.models_py.modules.base.common.vocab_parallel_logits"


class VocabParallelLogitsTest(unittest.TestCase):
    def test_single_rank_converts_to_contiguous_sampler_dtype(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            local = torch.arange(24, dtype=dtype).reshape(4, 6).t()
            actual = gather_vocab_parallel_logits(local, 1)
            torch.testing.assert_close(actual, local.float(), rtol=0, atol=0)
            self.assertTrue(actual.is_contiguous())

    def test_rank_order_and_live_row_counts(self):
        for rows in (1, 3, 7):
            shards = [
                torch.arange(rows * 4).reshape(rows, 4) + 100 * rank
                for rank in range(3)
            ]
            shards = [x.to(torch.bfloat16) for x in shards]
            with patch(
                MODULE + ".all_gather", return_value=torch.cat(shards)
            ) as gather:
                actual = gather_vocab_parallel_logits(shards[1], 3)
            torch.testing.assert_close(
                actual, torch.cat(shards, dim=1).float(), rtol=0, atol=0
            )
            self.assertIs(gather.call_args.args[0], shards[1])
            self.assertTrue(actual.is_contiguous())

    def test_optional_transport_falls_back_only_when_declined(self):
        local = torch.ones(3, 4, dtype=torch.bfloat16)
        expected = torch.ones(3, 8)
        with patch(MODULE + ".all_gather", return_value=torch.ones(6, 4)) as gather:
            self.assertIs(
                gather_vocab_parallel_logits(local, 2, try_gather=lambda _: expected),
                expected,
            )
            gather.assert_not_called()
            torch.testing.assert_close(
                gather_vocab_parallel_logits(local, 2, try_gather=lambda _: None),
                expected,
            )
            gather.assert_called_once()

    def test_invalid_contracts_fail_explicitly(self):
        local = torch.ones(3, 4, dtype=torch.bfloat16)
        for value, tp in (
            (local, 0),
            (local, -1),
            (local.long(), 2),
            (local.unsqueeze(0), 2),
        ):
            with self.assertRaises(ValueError):
                gather_vocab_parallel_logits(value, tp)
        for invalid in (
            torch.ones(3, 8, dtype=torch.bfloat16),
            torch.ones(3, 7),
            torch.ones(8, 3).t(),
        ):
            with self.assertRaisesRegex(RuntimeError, "transport returned invalid"):
                gather_vocab_parallel_logits(local, 2, try_gather=lambda _: invalid)
        with patch(MODULE + ".all_gather", return_value=local):
            with self.assertRaisesRegex(RuntimeError, "collective returned invalid"):
                gather_vocab_parallel_logits(local, 2)


def _gather_worker():
    from rtp_llm.models_py.distributed import collective_torch

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    from rtp_llm.ops import NcclCommConfig, ParallelismConfig

    config = ParallelismConfig()
    config.tp_size = config.world_size = config.local_world_size = 2
    config.world_rank = config.local_rank = rank
    nccl_config = NcclCommConfig()
    nccl_config.nccl_ip = os.environ["MASTER_ADDR"]
    collective_torch.init_distributed_environment(
        config,
        nccl_config,
        int(os.environ["MASTER_PORT"]),
        disable_custom_all_reduce=True,
    )
    try:
        for rows in (1, 3, 8, 2):
            for offset in (0, 13):
                base = torch.arange(rows * 12, device="cuda").reshape(rows, 12)
                # Sliced input also verifies explicit-send packing without
                # altering rank order or the local transport dtype.
                local = (base + rank * 64 + offset).to(torch.bfloat16)[:, ::2]
                actual = gather_vocab_parallel_logits(local, 2)
                expected = torch.cat(
                    [
                        (base + peer * 64 + offset).to(torch.bfloat16)[:, ::2]
                        for peer in range(2)
                    ],
                    dim=1,
                ).float()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        collective_torch.destroy_distributed_environment()


class VocabParallelLogitsIntegrationTest(unittest.TestCase):
    def test_real_tp_gather_with_dynamic_rows_and_values(self):
        self.assertEqual(torch.cuda.device_count(), 2)
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
            timeout=180,
        )


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _gather_worker()
    else:
        unittest.main()
