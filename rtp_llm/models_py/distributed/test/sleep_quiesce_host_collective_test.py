"""The step-aligned sleep vote must never be promoted to a GPU collective."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.distributed import collective_torch as ct


class SleepQuiesceHostCollectiveTest(unittest.TestCase):
    def test_sleep_vote_and_destination_stay_on_cpu(self):
        callbacks = []
        extension = SimpleNamespace(
            register_comm_ops=lambda *args: callbacks.extend(args)
        )
        group = mock.Mock()
        group.size.return_value = 2
        with mock.patch.dict(
            sys.modules, {"librtp_compute_ops": extension}
        ), mock.patch.object(
            ct, "_group_map", {ct.Group.SLEEP_QUIESCE: group}
        ), mock.patch.object(
            ct, "_parallelism_config", None
        ):
            ct._register_process_groups_to_cpp()
        reduce = callbacks[1]

        def all_reduce(tensor, op, group):
            self.assertEqual(tensor.device.type, "cpu")
            tensor.mul_(2)

        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=all_reduce
        ), mock.patch.object(
            torch.cuda,
            "current_device",
            side_effect=AssertionError("sleep vote touched CUDA"),
        ):
            for use_dest in (False, True):
                source = torch.tensor([1, 0], dtype=torch.int64)
                dest = torch.empty_like(source) if use_dest else None
                result = reduce(source, 0, ct._CPP_PARALLEL_MODE_SLEEP_QUIESCE, dest)
                self.assertEqual(result.tolist(), [2, 0])
                self.assertIs(result, dest if use_dest else source)
                if use_dest:
                    self.assertEqual(source.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
