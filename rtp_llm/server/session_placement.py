"""FE-side cold-session prefill placement balancing.

Background (experiment 00035, us1060_gcp_1): the flexlb master decides at
which prefill pod a session's cache tree is first computed (cold-session
placement). Under the load-resume burst after a KV-wiping rollout that
placement is winner-take-most, and the master's sticky cache affinity
(stickyPrefixHitPercent=90) then locks the resulting split indefinitely -
observed ~89/11 for two consecutive rollouts, costing -6-10% total cached
prefill TPS at otherwise identical config.

This module lets the frontend balance placement deterministically WITHOUT
touching warm routing: when the first block-cache key of a request has never
been seen in this process (a cold session, or any session after an FE restart
- which in this deployment coincides with the KV-wiping pod restart), the
caller pins the prefill role to a DETERMINISTIC choice from the VIP prefill
host list via generate_config.role_addrs (the request-routing path). The
placement is a pure function of the first key and the (sorted) host list, so
every frontend worker process pins a given session to the SAME pod: the FE
serves each request from one of several uvicorn workers, and a per-process
round-robin was observed to pin the same session to DIFFERENT pods (98% of
sessions conflicted; KV split across both pods; the aggregate split stayed
~80/20 despite perfect 50/50 pin counts). The master learns the placed KV
through its own engine cache-status poll, so all subsequent requests route
by master cache affinity exactly as before.

A bounded pin-follow window (TTL) keeps the session pinned to the same pod
right after placement, covering the master's cache-index poll lag. After the
window the entry is never re-pinned: the master owns the session again.

Env flags (read once at visitor construction by the caller):
  FE_COLD_PLACEMENT_BALANCE     enable placement pinning (default: off)
  FE_SESSION_PLACEMENT_TTL_S    pin-follow window seconds (default: 20)

Experiment 00036 (size-aware placement): session-count balancing alone
leaves the TOKEN-weighted split swinging (42/58 -> 64/36 measured at biz
61; ~2.3% TPS variance) because sessions span 52k-490k tokens. With
FE_PLACEMENT_WEIGHTED=1 the balancer additionally tracks a rolling
token-mass estimate per host (decayed so it tracks the live mix, not a
lifetime sum) and places each NEW session on the host with the lower
predicted mass, breaking ties (and cold-starting) on the deterministic
seed hash. Pins remain one-time frozen entries, so the cross-worker
conflicts=0 property of 00035 is preserved: a first_key is pinned exactly
once per process epoch and the entry is never re-decided.

Env flags (00036):
  FE_PLACEMENT_WEIGHTED         enable token-mass weighting (default: off)
  FE_PLACEMENT_MASS_HALF_LIFE_S rolling-mass half-life (default: 300)
"""

import logging
import math
import os
import time
from collections import OrderedDict
from typing import Callable, Optional, Sequence

from rtp_llm.config.generate_config import RoleAddr, RoleType

logger = logging.getLogger(__name__)

ENV_ENABLE_FLAG = "FE_COLD_PLACEMENT_BALANCE"
ENV_TTL_SECONDS = "FE_SESSION_PLACEMENT_TTL_S"
ENV_WEIGHTED_FLAG = "FE_PLACEMENT_WEIGHTED"
ENV_MASS_HALF_LIFE_S = "FE_PLACEMENT_MASS_HALF_LIFE_S"

DEFAULT_TTL_S = 20.0
DEFAULT_MAX_ENTRIES = 4096
DEFAULT_MASS_HALF_LIFE_S = 300.0
# Cold-start guard: below this many tracked placements the counters are
# considered unrepresentative and the seed hash decides (00035 behavior).
MIN_TRACKED_FOR_WEIGHTING = 4


def env_flag_enabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_ttl_seconds(default: float = DEFAULT_TTL_S) -> float:
    raw = os.environ.get(ENV_TTL_SECONDS)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def env_mass_half_life_s(default: float = DEFAULT_MASS_HALF_LIFE_S) -> float:
    raw = os.environ.get(ENV_MASS_HALF_LIFE_S)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class SessionPlacementBalancer:
    """Deterministic prefill placement for first-seen sessions.

    ``host_provider`` must return the current prefill endpoint list; each
    entry only needs ``ip`` and ``port`` attributes (duck-typed so tests and
    the VipServerWrapper Host agree). The port convention mirrors
    HostService.get_backend_role_addrs: grpc_port = port + 1, http_port =
    port.

    Placement is consistent across processes: ``_placement_host`` derives
    the target purely from the first block key and the SORTED host list
    (integer arithmetic only - immune to PYTHONHASHSEED), so every frontend
    worker pins a session to the same pod.
    """

    def __init__(
        self,
        host_provider: Callable[[], Sequence],
        clock: Optional[Callable[[], float]] = None,
        enabled: bool = False,
        ttl_s: float = DEFAULT_TTL_S,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        weighted: bool = False,
        mass_half_life_s: float = DEFAULT_MASS_HALF_LIFE_S,
    ):
        self._host_provider = host_provider
        self._clock = clock or time.monotonic
        self._enabled = bool(enabled)
        self._ttl_s = float(ttl_s) if ttl_s and ttl_s > 0 else DEFAULT_TTL_S
        self._max_entries = (
            int(max_entries) if max_entries and max_entries > 0 else DEFAULT_MAX_ENTRIES
        )
        self._weighted = bool(weighted)
        self._mass_half_life_s = (
            float(mass_half_life_s)
            if mass_half_life_s and mass_half_life_s > 0
            else DEFAULT_MASS_HALF_LIFE_S
        )
        # first block key -> (pinned ip, pinned grpc port, placed_at)
        self._entries: OrderedDict = OrderedDict()
        self._pin_count = 0
        self._follow_count = 0
        # 00036: rolling token-mass estimates per "ip:grpc_port" host key.
        # host_key -> [decayed_mass, last_decay_at, tracked_placements]
        self._mass: dict = {}

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "ttl_s": self._ttl_s,
            "entries": len(self._entries),
            "pin_count": self._pin_count,
            "follow_count": self._follow_count,
            "weighted": self._weighted,
            "mass_hosts": len(self._mass),
        }

    def maybe_pin_prefill(
        self, block_cache_keys: Sequence[int], token_count: Optional[int] = None
    ) -> Optional[RoleAddr]:
        """Return a prefill RoleAddr to pin, or None to use master routing.

        The caller passes the FULL logical block cache key list of the
        request. The first key identifies the session (its hash chain's
        anchor); it is stable across the session's requests and unique per
        session. ``token_count`` (00036) is the request's input token count;
        when weighting is enabled it biases placement toward the host with
        the lower rolling token mass. It defaults to None so the 00035 call
        signature keeps working unchanged.
        """
        if not self._enabled or not block_cache_keys:
            return None
        first_key = block_cache_keys[0]
        now = self._clock()
        entry = self._entries.get(first_key)
        if entry is not None:
            pinned_ip, pinned_port, placed_at = entry
            self._entries.move_to_end(first_key)
            if now - placed_at <= self._ttl_s:
                # Pin-follow window: keep the young session on its placed
                # pod while the master's cache index catches up.
                self._follow_count += 1
                return self._role_addr(pinned_ip, pinned_port)
            # Window expired: the session is established; master cache
            # affinity owns it from here on. Never re-pin the same first
            # key within this process epoch.
            return None
        hosts = self._host_provider()
        if len(hosts) < 2:
            # Nothing to balance (single pod or discovery unavailable).
            return None
        host = self._placement_host(first_key, hosts, token_count=token_count, now=now)
        grpc_port = int(host.port) + 1
        self._entries[first_key] = (host.ip, grpc_port, now)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        self._pin_count += 1
        logger.info(
            "session placement pin: first_key=%s target=%s:%d entries=%d "
            "tokens=%s",
            first_key,
            host.ip,
            grpc_port,
            len(self._entries),
            token_count,
        )
        return self._role_addr(host.ip, grpc_port)

    def _placement_host(self, first_key: int, hosts: Sequence, token_count=None, now=None):
        """Deterministic, process-independent host for a session.

        Integer arithmetic only (no PYTHONHASHSEED dependence): a fixed odd
        multiplier mixes the 64-bit key, then modulo the SORTED host count.
        Sorting normalizes vipserver's arbitrary per-process host order.

        00036: when weighting is enabled, a positive ``token_count`` and a
        warm mass tracker flip the choice to the host with the lower
        predicted mass (seed host breaks ties). The mass is tracked under
        a stable host key (sorted ip:port), so the decision stays a pure
        function of (first_key, token_count, tracker state) and each
        session is still pinned exactly once.
        """
        sorted_hosts = sorted(hosts, key=lambda h: (h.ip, h.port))
        mixed = (int(first_key) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        seed_host = sorted_hosts[mixed % len(sorted_hosts)]
        if not self._weighted or token_count is None or token_count <= 0:
            return seed_host
        if now is None:
            now = self._clock()
        # Cold-start guard: unrepresentative counters keep 00035 behavior.
        tracked = sum(state[2] for state in self._mass.values())
        if tracked < MIN_TRACKED_FOR_WEIGHTING * len(sorted_hosts):
            self._record_mass(seed_host, token_count, now)
            return seed_host
        candidates = []
        for host in sorted_hosts:
            mass = self._decayed_mass(host, now)
            candidates.append((mass, (host.ip, host.port), host))
        # Lowest predicted mass wins; the (ip, port) tuple breaks exact
        # ties deterministically, and the seed host wins its own ties.
        best_mass = min(c[0] for c in candidates)
        best = [c for c in candidates if c[0] <= best_mass + 1e-9]
        chosen = best[0][2] if len(best) == 1 else self._tiebreak(seed_host, best)
        self._record_mass(chosen, token_count, now)
        return chosen

    def _tiebreak(self, seed_host, best):
        for _, _, host in best:
            if (host.ip, host.port) == (seed_host.ip, seed_host.port):
                return host
        return best[0][2]

    def _host_key(self, host) -> str:
        return f"{host.ip}:{int(host.port) + 1}"

    def _decayed_mass(self, host, now: float) -> float:
        key = self._host_key(host)
        state = self._mass.get(key)
        if state is None:
            return 0.0
        mass, last_decay, tracked = state
        elapsed = now - last_decay
        if elapsed <= 0:
            return mass
        decay = math.pow(0.5, elapsed / self._mass_half_life_s)
        return mass * decay

    def _record_mass(self, host, token_count: int, now: float) -> None:
        key = self._host_key(host)
        state = self._mass.get(key)
        if state is None:
            self._mass[key] = [float(token_count), now, 1]
            return
        mass, last_decay, tracked = state
        decay = math.pow(0.5, (now - last_decay) / self._mass_half_life_s)
        self._mass[key] = [mass * decay + float(token_count), now, tracked + 1]

    @staticmethod
    def _role_addr(ip: str, grpc_port: int) -> RoleAddr:
        # Port convention mirrors HostService.get_backend_role_addrs.
        return RoleAddr(
            role=RoleType.PREFILL,
            ip=ip,
            grpc_port=grpc_port,
            http_port=grpc_port - 1,
        )
