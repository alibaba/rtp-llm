"""C++ broadcast roots are local to TP/DP, while torch src is global."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from rtp_llm.models_py.distributed import collective_torch as comm


class CppBroadcastRootTest(unittest.TestCase):
    def test_group_local_roots_resolve_inside_selected_group(self):
        for rank in range(8):
            tp_ranks = list(range(rank // 4 * 4, rank // 4 * 4 + 4))
            dp_ranks = [rank % 4, rank % 4 + 4]
            groups = [tp_ranks, dp_ranks, list(range(8))]
            pgs = [SimpleNamespace(size=lambda ranks=ranks: len(ranks)) for ranks in groups]
            group_map = {
                comm.Group.TP.name + str(rank // 4): pgs[0],
                comm.Group.DP.name + str(rank % 4): pgs[1],
                comm.Group.DP_AND_TP: pgs[2],
            }
            callbacks = Mock()
            extension = SimpleNamespace(register_comm_ops=callbacks)
            config = SimpleNamespace(tp_size=4, world_size=8, local_world_size=0)

            def global_rank(pg, root):
                index = next(i for i, candidate in enumerate(pgs) if candidate is pg)
                return groups[index][root]

            with patch.dict(sys.modules, {"librtp_compute_ops": extension}), \
                 patch.object(comm, "_parallelism_config", config), \
                 patch.object(comm, "_group_map", group_map), \
                 patch.object(torch.distributed, "get_rank", return_value=rank), \
                 patch.object(torch.distributed, "get_global_rank", side_effect=global_rank), \
                 patch.object(torch.cuda, "current_device", return_value=rank), \
                 patch.object(torch.distributed, "broadcast") as broadcast:
                comm._register_process_groups_to_cpp()
                cpp_broadcast = callbacks.call_args.args[0]
                tensor = SimpleNamespace(is_cuda=True)
                for mode, ranks in enumerate(groups):
                    for root in (0, len(ranks) - 1):
                        with self.subTest(rank=rank, mode=mode, root=root):
                            cpp_broadcast([tensor], root, mode)
                            broadcast.assert_called_with(tensor, ranks[root], group=pgs[mode])


if __name__ == "__main__":
    unittest.main()
