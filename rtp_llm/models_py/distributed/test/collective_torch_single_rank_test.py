"""Real NCCL initialization and native callback registration for CP1."""

import socket
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.ops import CPRotateMethod, NcclCommConfig, ParallelismConfig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CollectiveSingleRankTest(unittest.TestCase):
    def test_real_initialization_and_registered_callback_contracts(self):
        # PyTorch and the normal RTP imports above must precede its pybind DSO.
        import librtp_compute_ops

        self.assertFalse(torch.distributed.is_initialized())
        config = ParallelismConfig()
        config.world_size = config.local_world_size = config.tp_size = (
            config.dp_size
        ) = 1
        config.world_rank = config.local_rank = config.tp_rank = config.dp_rank = 0
        config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        with socket.socket() as socket_probe:
            socket_probe.bind(("127.0.0.1", 0))
            port = socket_probe.getsockname()[1]
        comm = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=port + 1,
            dp_tp_nccl_port=port + 2,
            ffn_tp_nccl_port=port + 3,
        )

        # This observer invokes the real pybind registration. It captures the
        # actual callbacks for contract checks; it never substitutes callbacks.
        with patch.object(
            librtp_compute_ops,
            "register_comm_ops",
            wraps=librtp_compute_ops.register_comm_ops,
        ) as register:
            collective.init_distributed_environment(
                config,
                nccl_comm_config=comm,
                nccl_init_port=port,
                backend="nccl",
                timeout=30,
            )
        self.addCleanup(collective.destroy_distributed_environment)
        register.assert_called_once()
        self.assertTrue(torch.distributed.is_initialized())
        self.assertEqual(torch.distributed.get_world_size(), 1)
        broadcast, allreduce, allgather = register.call_args.args
        value = torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(3, 4)
        expected = value.clone()
        for mode in (0, 1, 2):  # Native TP, DP and DP_AND_TP conventions.
            with self.subTest(mode=mode):
                broadcast([value], 0, mode)
                self.assertTrue(torch.equal(value, expected))
                self.assertIs(allreduce(value, 0, mode, None), value)
                destination = torch.full_like(value, -1)
                self.assertIs(allreduce(value, 0, mode, destination), destination)
                self.assertTrue(torch.equal(destination, expected))
                received = torch.full_like(value, -1)
                allgather([received], mode, [value], False)
                self.assertTrue(torch.equal(received, expected))
                allgather([received], mode, [], True)
                self.assertTrue(torch.equal(received, expected))

        # Also exercise the ordinary Python calls against the real one-rank
        # process group rather than only invoking registered callback objects.
        collective.all_reduce(value, collective.Group.TP)
        gathered = collective.all_gather(value, collective.Group.TP)
        torch.distributed.barrier()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(value, expected))
        self.assertTrue(torch.equal(gathered, expected))


if __name__ == "__main__":
    unittest.main()
