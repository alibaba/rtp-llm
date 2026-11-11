import unittest
from types import SimpleNamespace

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.session_placement import (
    DEFAULT_TTL_S,
    SessionPlacementBalancer,
    env_flag_enabled,
    env_ttl_seconds,
)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def make_hosts(*specs):
    return [SimpleNamespace(ip=ip, port=port) for ip, port in specs]


HOSTS = make_hosts(("10.0.0.1", 9000), ("10.0.0.2", 9000))


class SessionPlacementBalancerTest(unittest.TestCase):
    def test_disabled_returns_none(self):
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=FakeClock(),
            enabled=False,
        )
        self.assertIsNone(balancer.maybe_pin_prefill([11, 12, 13]))
        self.assertEqual(balancer.stats()["pin_count"], 0)

    def test_round_robin_balances_20_sessions_over_2_hosts(self):
        clock = FakeClock()
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=clock,
            enabled=True,
            ttl_s=20.0,
        )
        pins = [balancer.maybe_pin_prefill([100 + i, 7]) for i in range(20)]
        targets = [addr.ip for addr in pins if addr is not None]
        self.assertEqual(len(targets), 20)
        self.assertEqual(targets.count("10.0.0.1"), 10)
        self.assertEqual(targets.count("10.0.0.2"), 10)
        for addr in pins:
            self.assertIsNotNone(addr)
            self.assertEqual(addr.role, RoleType.PREFILL)
            # Port convention mirrors HostService.get_backend_role_addrs.
            self.assertEqual(addr.grpc_port, 9001)
            self.assertEqual(addr.http_port, 9000)
        self.assertEqual(balancer.stats()["pin_count"], 20)

    def test_pin_follow_keeps_session_on_placed_pod_within_ttl(self):
        clock = FakeClock()
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=clock,
            enabled=True,
            ttl_s=20.0,
        )
        first = balancer.maybe_pin_prefill([55, 56])
        self.assertIsNotNone(first)
        clock.advance(1.0)
        # Same session, grown key list (first key is the session anchor).
        follow = balancer.maybe_pin_prefill([55, 56, 57])
        self.assertIsNotNone(follow)
        self.assertEqual(follow.ip, first.ip)
        self.assertEqual(balancer.stats()["follow_count"], 1)
        clock.advance(5.0)
        follow2 = balancer.maybe_pin_prefill([55, 56, 57, 58])
        self.assertIsNotNone(follow2)
        self.assertEqual(follow2.ip, first.ip)

    def test_no_repin_after_ttl_expiry(self):
        clock = FakeClock()
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=clock,
            enabled=True,
            ttl_s=20.0,
        )
        first = balancer.maybe_pin_prefill([99])
        self.assertIsNotNone(first)
        clock.advance(DEFAULT_TTL_S + 0.1)
        # Window expired: master routing owns the session; never re-pin the
        # same first key within this process epoch.
        self.assertIsNone(balancer.maybe_pin_prefill([99, 100]))
        self.assertIsNone(balancer.maybe_pin_prefill([99, 101]))
        # A different first key is a new session and still gets placed.
        other = balancer.maybe_pin_prefill([7000])
        self.assertIsNotNone(other)

    def test_single_host_returns_none(self):
        balancer = SessionPlacementBalancer(
            host_provider=lambda: make_hosts(("10.0.0.1", 9000)),
            clock=FakeClock(),
            enabled=True,
        )
        self.assertIsNone(balancer.maybe_pin_prefill([42]))

    def test_empty_hosts_returns_none(self):
        balancer = SessionPlacementBalancer(
            host_provider=lambda: [],
            clock=FakeClock(),
            enabled=True,
        )
        self.assertIsNone(balancer.maybe_pin_prefill([42]))

    def test_empty_keys_returns_none(self):
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=FakeClock(),
            enabled=True,
        )
        self.assertIsNone(balancer.maybe_pin_prefill([]))

    def test_lru_capacity_bounds_entries(self):
        clock = FakeClock()
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=clock,
            enabled=True,
            ttl_s=20.0,
            max_entries=3,
        )
        for i in range(4):
            balancer.maybe_pin_prefill([i])
        self.assertEqual(balancer.stats()["entries"], 3)


class EnvFlagParsingTest(unittest.TestCase):
    def test_env_flag_disabled_by_default(self):
        import os

        saved = os.environ.pop("FE_COLD_PLACEMENT_BALANCE", None)
        try:
            self.assertFalse(env_flag_enabled("FE_COLD_PLACEMENT_BALANCE"))
        finally:
            if saved is not None:
                os.environ["FE_COLD_PLACEMENT_BALANCE"] = saved

    def test_env_flag_accepts_truthy_values(self):
        import os

        for raw in ("1", "true", "True", "YES", "on"):
            os.environ["FE_COLD_PLACEMENT_BALANCE"] = raw
            try:
                self.assertTrue(env_flag_enabled("FE_COLD_PLACEMENT_BALANCE"))
            finally:
                os.environ.pop("FE_COLD_PLACEMENT_BALANCE", None)

    def test_env_ttl_default_and_invalid(self):
        import os

        os.environ.pop("FE_SESSION_PLACEMENT_TTL_S", None)
        try:
            self.assertEqual(env_ttl_seconds(), DEFAULT_TTL_S)
            os.environ["FE_SESSION_PLACEMENT_TTL_S"] = "not-a-number"
            self.assertEqual(env_ttl_seconds(), DEFAULT_TTL_S)
            os.environ["FE_SESSION_PLACEMENT_TTL_S"] = "5.5"
            self.assertEqual(env_ttl_seconds(), 5.5)
        finally:
            os.environ.pop("FE_SESSION_PLACEMENT_TTL_S", None)


if __name__ == "__main__":
    unittest.main()
