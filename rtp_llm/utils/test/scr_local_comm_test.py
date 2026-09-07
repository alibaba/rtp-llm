import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.utils.scr_local_comm import (
    local_comm_enabled,
    validate_local_members,
    validate_local_world,
)


class ScrLocalCommTest(unittest.TestCase):
    def setUp(self):
        self.pc = NS(world_size=2, local_world_size=2)
        self.world = NS(
            num_nodes=1,
            self=NS(ip="192.0.2.1"),
            members=[NS(ip="192.0.2.2", world_rank=i, local_rank=i) for i in range(2)],
        )

    def test_requires_explicit_opt_in(self):
        for value, expected in [
            ("", False),
            ("0", False),
            ("true", False),
            ("1", True),
        ]:
            with self.subTest(value=value), patch.dict(
                os.environ, {"RTP_LLM_SCR_LOCAL_COMM": value}
            ):
                self.assertEqual(local_comm_enabled(), expected)

    def test_restored_ip_may_differ_from_seed(self):
        validate_local_members(self.world, self.pc)

    def test_member_order_does_not_change_topology(self):
        self.world.members.reverse()
        validate_local_members(self.world, self.pc)

    def test_single_rank(self):
        validate_local_world(1, 1, 1)

    def test_rejects_multinode_and_incomplete_world(self):
        for args in [(2, 1, 2), (2, 2, 2), (0, 0, 1), (2, 1, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_local_world(*args)

    def test_rejects_incomplete_members(self):
        self.world.members.pop()
        with self.assertRaises(ValueError):
            validate_local_members(self.world, self.pc)

    def test_rejects_multiple_addresses(self):
        self.world.members[1].ip = "192.0.2.3"
        with self.assertRaises(ValueError):
            validate_local_members(self.world, self.pc)

    def test_rejects_duplicate_or_nonlocal_ranks(self):
        for attr, value in [
            ("world_rank", 0),
            ("local_rank", 0),
            ("world_rank", 2),
            ("local_rank", 2),
        ]:
            with self.subTest(attr=attr, value=value):
                self.setUp()
                setattr(self.world.members[1], attr, value)
                with self.assertRaises(ValueError):
                    validate_local_members(self.world, self.pc)


if __name__ == "__main__":
    unittest.main()
