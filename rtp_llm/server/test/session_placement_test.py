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

    def test_deterministic_placement_balances_20_sessions_over_2_hosts(self):
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

    def test_placement_is_deterministic_for_same_key(self):
        # Same session key always maps to the same host (pure function of
        # key + sorted host list), independent of pin history.
        clock = FakeClock()
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=clock,
            enabled=True,
            ttl_s=20.0,
        )
        first = balancer.maybe_pin_prefill([31337])
        self.assertIsNotNone(first)
        again = balancer.maybe_pin_prefill([31337])
        self.assertIsNotNone(again)
        self.assertEqual(again.ip, first.ip)

    def test_cross_worker_consistency_two_balancers_agree(self):
        # REGRESSION TEST for the 00035 engagement bug: the FE serves each
        # request from one of several uvicorn worker processes, each with
        # its own balancer. A per-process round-robin pinned the SAME
        # session to DIFFERENT pods (98% of sessions conflicted, splitting
        # session KV across pods). Deterministic placement must make every
        # worker agree on every session.
        balancer_a = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=FakeClock(),
            enabled=True,
            ttl_s=20.0,
        )
        balancer_b = SessionPlacementBalancer(
            host_provider=lambda: list(reversed(HOSTS)),
            clock=FakeClock(now=987654.0),
            enabled=True,
            ttl_s=20.0,
        )
        keys = [1000 + i for i in range(50)]
        for key in keys:
            pin_a = balancer_a.maybe_pin_prefill([key])
            pin_b = balancer_b.maybe_pin_prefill([key])
            self.assertIsNotNone(pin_a)
            self.assertIsNotNone(pin_b)
            self.assertEqual(
                pin_a.ip,
                pin_b.ip,
                f"workers disagree on session {key}: {pin_a.ip} vs {pin_b.ip}",
            )
            self.assertEqual(pin_a.grpc_port, pin_b.grpc_port)
        # Both pods are used across the key population (balanced).
        used = {ip for (ip, _port, _ts) in balancer_a._entries.values()}
        self.assertEqual(used, {"10.0.0.1", "10.0.0.2"})

    def test_host_order_independence(self):
        # vipserver may return hosts in a different order per worker
        # process; sorting must normalize the order so the placement is
        # identical.
        balancer = SessionPlacementBalancer(
            host_provider=lambda: HOSTS,
            clock=FakeClock(),
            enabled=True,
            ttl_s=20.0,
        )
        first = balancer.maybe_pin_prefill([4242])
        self.assertIsNotNone(first)
        reversed_balancer = SessionPlacementBalancer(
            host_provider=lambda: list(reversed(HOSTS)),
            clock=FakeClock(),
            enabled=True,
            ttl_s=20.0,
        )
        second = reversed_balancer.maybe_pin_prefill([4242])
        self.assertIsNotNone(second)
        self.assertEqual(second.ip, first.ip)

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
