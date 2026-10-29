import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct


class BroadcastMetadataTest(unittest.TestCase):
    def test_pinned_tp_only_and_blocking_receive(self):
        callbacks = []
        pg = MagicMock()
        pg.size.return_value = 2
        config = SimpleNamespace(tp_size=2, world_size=2, local_world_size=2)
        groups = {
            ct.Group.TP.name + "0": pg,
            ct.Group.DP.name + "0": pg,
            ct.Group.DP_AND_TP: pg,
        }
        library = SimpleNamespace(
            register_comm_ops=lambda *args: callbacks.extend(args)
        )
        with patch.dict(sys.modules, {"librtp_compute_ops": library}), patch.object(
            ct, "_group_map", groups
        ), patch.object(ct, "_parallelism_config", config), patch.object(
            torch.distributed, "get_rank", return_value=0
        ):
            ct._register_process_groups_to_cpp()
        broadcast = callbacks[0]
        for mode in (
            ct._CPP_PARALLEL_MODE_TP,
            ct._CPP_PARALLEL_MODE_DP,
            ct._CPP_PARALLEL_MODE_DP_AND_TP,
        ):
            for pinned in (False, True):
                for cuda in (False, True):
                    events = []
                    t = MagicMock()
                    t.is_cuda = cuda
                    t.is_pinned.return_value = pinned
                    gpu = t if cuda else MagicMock()
                    t.to.side_effect = lambda *a, **kw: (
                        events.append(("upload", kw)),
                        gpu,
                    )[1]
                    t.copy_.side_effect = lambda *a, **kw: events.append(
                        ("receive", kw)
                    )
                    with patch.object(
                        torch.cuda, "current_device", return_value=0
                    ), patch.object(
                        torch.distributed,
                        "broadcast",
                        side_effect=lambda *a, **kw: events.append(("broadcast", kw)),
                    ):
                        broadcast([t], 0, mode)
                    self.assertEqual(
                        [e[0] for e in events],
                        ["broadcast"] if cuda else ["upload", "broadcast", "receive"],
                    )
                    if not cuda:
                        self.assertEqual(
                            events[0][1]["non_blocking"],
                            pinned and mode == ct._CPP_PARALLEL_MODE_TP,
                        )
                        self.assertEqual(events[-1][1], {})  # copy_ remains blocking


if __name__ == "__main__":
    unittest.main()
