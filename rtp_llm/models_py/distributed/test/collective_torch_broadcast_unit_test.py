import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct


class CppBroadcastTest(unittest.TestCase):
    def callback(self, ranks, mode, tp=4, world=16):
        pg = SimpleNamespace(size=lambda: len(ranks))
        rank = ranks[0]
        key = {
            ct._CPP_PARALLEL_MODE_TP: ct.Group.TP.name + str(rank // tp),
            ct._CPP_PARALLEL_MODE_DP: ct.Group.DP.name + str(rank % tp),
            ct._CPP_PARALLEL_MODE_DP_AND_TP: ct.Group.DP_AND_TP,
        }[mode]
        extension = SimpleNamespace(register_comm_ops=Mock())
        config = SimpleNamespace(tp_size=tp, world_size=world, local_world_size=4)
        with patch.dict(sys.modules, {"librtp_compute_ops": extension}), patch.object(
            ct, "_group_map", {key: pg}
        ), patch.object(ct, "_parallelism_config", config), patch.object(
            torch.distributed, "get_rank", return_value=rank
        ):
            ct._register_process_groups_to_cpp()
        return extension.register_comm_ops.call_args.args[0], pg

    def test_subgroup_roots(self):
        cases = [
            (list(range(start, start + 4)), ct._CPP_PARALLEL_MODE_TP, 4, 16)
            for start in (0, 4, 8, 12)
        ] + [
            ([1, 5, 9, 13], ct._CPP_PARALLEL_MODE_DP, 4, 16),
            (list(range(8, 16)), ct._CPP_PARALLEL_MODE_TP, 8, 16),
            (list(range(8)), ct._CPP_PARALLEL_MODE_TP, 8, 8),
            (list(range(16)), ct._CPP_PARALLEL_MODE_DP_AND_TP, 4, 16),
        ]
        for ranks, mode, tp, world in cases:
            for root in (0, len(ranks) - 1):
                with self.subTest(ranks=ranks, root=root):
                    callback, pg = self.callback(ranks, mode, tp, world)
                    tensors = [
                        SimpleNamespace(is_cuda=True),
                        SimpleNamespace(is_cuda=True),
                    ]
                    with patch.object(
                        torch.distributed,
                        "get_global_rank",
                        side_effect=lambda group, r: ranks[r],
                    ) as convert, patch.object(
                        torch.distributed, "broadcast"
                    ) as broadcast, patch.object(
                        torch.cuda, "current_device", return_value=0
                    ):
                        callback(tensors, root, mode)
                    convert.assert_called_once_with(pg, root)
                    self.assertEqual(broadcast.call_count, 2)
                    for tensor, call in zip(tensors, broadcast.call_args_list):
                        self.assertEqual(call.args, (tensor,))
                        self.assertEqual(call.kwargs, {"src": ranks[root], "group": pg})

    def test_cpu_copyback(self):
        callback, pg = self.callback([4, 5, 6, 7], ct._CPP_PARALLEL_MODE_TP)
        gpu = object()
        cpu = SimpleNamespace(is_cuda=False, to=Mock(return_value=gpu), copy_=Mock())
        with patch.object(
            torch.distributed, "get_global_rank", return_value=4
        ), patch.object(torch.distributed, "broadcast") as broadcast, patch.object(
            torch.cuda, "current_device", return_value=0
        ):
            callback([cpu], 0, ct._CPP_PARALLEL_MODE_TP)
        broadcast.assert_called_once_with(gpu, src=4, group=pg)
        cpu.copy_.assert_called_once_with(gpu)

    def test_singleton_and_missing_group_noop(self):
        callback, _ = self.callback([4], ct._CPP_PARALLEL_MODE_TP)
        with patch.object(
            torch.distributed, "get_global_rank"
        ) as convert, patch.object(torch.distributed, "broadcast") as broadcast:
            callback([], 0, ct._CPP_PARALLEL_MODE_TP)
            callback([], 0, ct._CPP_PARALLEL_MODE_DP)
        convert.assert_not_called()
        broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
