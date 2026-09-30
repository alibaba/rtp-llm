"""Both sleep levels keep the existing inference process groups unchanged."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.distributed import collective_torch as ct


class SleepQuiesceHostCollectiveTest(unittest.TestCase):

    def test_sleep_does_not_add_or_warm_process_groups(self):
        for enabled in ("0", "1"):
            for level in ("1", "2"):
                for tp_size, dp_size in ((1, 2), (2, 1), (2, 2)):
                    with self.subTest(
                        enabled=enabled, level=level, tp=tp_size, dp=dp_size
                    ):
                        config = SimpleNamespace(
                            world_rank=0,
                            world_size=tp_size * dp_size,
                            tp_size=tp_size,
                            dp_size=dp_size,
                        )
                        with mock.patch.dict(
                            "os.environ",
                            {"ENABLE_SLEEP_MODE": enabled, "SLEEP_MODE_LEVEL": level},
                        ), mock.patch.object(ct, "_group_map", {}), mock.patch.object(
                            torch.distributed, "new_group"
                        ) as new_group, mock.patch.object(
                            torch.distributed, "all_reduce"
                        ) as reduce, mock.patch.object(
                            torch.distributed, "barrier"
                        ), mock.patch.object(
                            ct, "_get_symm_mem"
                        ):
                            ct._create_process_groups(config, "nccl", None)
                            mixed = tp_size > 1 and dp_size > 1
                            self.assertEqual(new_group.call_count, 4 if mixed else 0)
                            for call in new_group.call_args_list:
                                self.assertEqual(call.kwargs["backend"], "nccl")
                            self.assertEqual(
                                set(ct._group_map), {"DP0", "TP0"} if mixed else set()
                            )
                            reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
