from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ONLINE_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(ONLINE_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_EVAL_DIR))

from flexlb_smoke_base import FlexLBSmokeBase


class PriorityScheduleBuilderTest(unittest.TestCase):
    def test_priority_is_set_on_schedule_protocol(self) -> None:
        args = argparse.Namespace(
            master_ip="127.0.0.1",
            master_http_port=18080,
            mock_http_port=55150,
            flexlb_http_port=18080,
            schedule_mode="batch",
            request_id_base=20000,
        )
        smoke = FlexLBSmokeBase(args)

        low = smoke._build_schedule_request(20001, priority=30)
        high = smoke._build_schedule_request(20002, priority=70)

        self.assertEqual(30, low.priority)
        self.assertEqual(70, high.priority)


if __name__ == "__main__":
    unittest.main()
