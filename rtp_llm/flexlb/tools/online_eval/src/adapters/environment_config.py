"""Declarative environment inputs and fingerprints for process reuse."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from flexlb_cfg import OMIT, ConfigOverride


DEFAULT_MOCK_HEAP = os.environ.get("FLEXLB_FT_MOCK_HEAP", "2g")

DEFAULT_MOCK_EVENT_LOOP_THREADS = 8

DEFAULT_MOCK_COMPLETION_THREADS = 4

DEFAULT_PREFILL_CACHE_BLOCKS = 6000

DEFAULT_DECODE_CACHE_BLOCKS = 3000

DEFAULT_MASTER_HTTP_PORT = int(os.environ.get("FLEXLB_FT_MASTER_HTTP_PORT", "18080"))

DEFAULT_MASTER_MANAGEMENT_PORT = int(
    os.environ.get(
        "FLEXLB_FT_MASTER_MANAGEMENT_PORT", str(DEFAULT_MASTER_HTTP_PORT + 1)
    )
)

MASTER_PORT_STRIDE = 10

MASTER_PORT_MAX_SHIFTS = 50


def default_perf() -> dict:
    """Return the synthetic smoke preset with its named mock calibration.

    The preset loads the legacy-unverified DSv4 test scale from a file. The
    materialized decode coefficients are saved with the run performance JSON;
    master routing receives the same file's prefill expression through
    flexlb_cfg unless the scenario declares a model-specific override.
    """
    from adapters.perf_presets import load_preset
    return load_preset("default")[0]


from adapters.zookeeper_config import ZooKeeperConfig


@dataclass
class MasterSpec:
    """One Master address and port group.

    Standalone peers use different port groups on loopback. Elected peers
    use distinct loopback addresses with identical ports, advertised to ZK
    and bound by HTTP, management and gRPC. Both share one election role.
    """

    name: str  # registry key ("A" / "B" — brief p5/p6 scenario notation)
    http_port: int
    management_port: Optional[int] = None  # default http+1
    # Address shared by HTTP, management and gRPC listeners.
    bind_ip: str = "127.0.0.1"
    # Explicit IP identity consumed by election and status consistency.
    advertised_ip: Optional[str] = None
    # Default: BOTH instances share spec.label's role (mutual backup).
    hippo_role: Optional[str] = None
    log_dir_name: Optional[str] = None  # default logs_{name} under run_dir
    extra_env: dict = field(default_factory=dict)  # per-master overrides
    extra_args: list = field(default_factory=list)  # per-master CLI args

    def grpc_port(self) -> int:
        """gRPC port = HTTP + 2 (FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET)."""
        return self.http_port + 2

    def management(self) -> int:
        return (
            self.management_port
            if self.management_port is not None
            else self.http_port + 1
        )

    def fingerprint(self) -> dict:
        return {
            "name": self.name,
            "http_port": self.http_port,
            "management_port": self.management(),
            "bind_ip": self.bind_ip,
            "advertised_ip": self.advertised_ip,
            "hippo_role": self.hippo_role,
            "extra_env": self.extra_env,
            "extra_args": self.extra_args,
        }


def _override_fingerprint(overrides: Optional[ConfigOverride]) -> Optional[dict]:
    """Stable JSON-serializable snapshot of a ConfigOverride (the OMIT
    sentinel serializes as its own token) — EnvSpec.fingerprint input."""
    if overrides is None:
        return None
    snapshot: dict = {}
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        snapshot[f.name] = "OMIT" if value is OMIT else value
    return snapshot


@dataclass
class EnvSpec:
    """Declarative description of a full mock + master environment."""

    label: str = "env"
    run_dir: Optional[Path] = None
    runtime_mode: str = "functional"
    diagnostic_events: bool = True
    n_prefill: int = 2
    n_decode: int = 4
    mock_heap: str = DEFAULT_MOCK_HEAP
    mock_extra_args: list = field(default_factory=list)
    perf: dict = field(default_factory=default_perf)
    # Built-in scheduling profile (PROFILES) or "none" (master not
    # started); the FLEXLB_CONFIG document is rendered by flexlb_cfg from
    # the profile axes + config_overrides (or passthrough raw_config).
    master_profile: str = "batch-window"
    # Non-config master env overrides ONLY (e.g. FLEXLB_MONITOR_METRIC_
    # WHITELIST, FLEXLB_ADVERTISED_IP, FLEXLB_SYNC_CONSISTENCY_CONFIG).
    # FLEXLB_CONFIG must NOT ride here — the narrowed channel is
    # config_overrides (generator layering) / raw_config (negative-test
    # bypass); master_environment rejects a stray FLEXLB_CONFIG key.
    master_env: dict = field(default_factory=dict)
    # FLEXLB_CONFIG layer: flexlb_cfg.ConfigOverride applied on top of the
    # master_profile base document (base < profile < override — see
    # flexlb_cfg.render_env).  None = the profile document as-is.
    config_overrides: Optional[ConfigOverride] = None
    # Raw FLEXLB_CONFIG string — the negative-test channel (atpm
    # strict-reject variants inject deliberately-illegal documents).
    # Bypasses the generator entirely; takes precedence over
    # config_overrides.
    raw_config: Optional[str] = None
    spring_profile: str = "default"
    master_debug_log: bool = False
    # domain uses MODEL_SERVICE_CONFIG.hosts; file/discovery_file use
    # the mock-maintained MODEL_SERVICE_CONFIG.discovery_file.
    discovery: str = "file"
    domain_addrs: dict = field(default_factory=dict)  # {prefill: "a,b", decode: "a,b"}
    # Per-role KV pool size in BLOCKS — forwarded to the Java mock as
    # --prefill-kv-pool-blocks / --decode-kv-pool-blocks (NOT a key-count
    # cap: since KV v2 this is the total block count of the pool).
    prefill_cache_blocks: int = DEFAULT_PREFILL_CACHE_BLOCKS
    decode_cache_blocks: int = DEFAULT_DECODE_CACHE_BLOCKS
    master_extra_args: list = field(default_factory=list)
    master_jvm_args: list = field(default_factory=list)
    master_log_name: Optional[str] = None
    master_jvm_heap: Optional[str] = None
    master_pv_log: Optional[bool] = None
    event_loop_threads: int = DEFAULT_MOCK_EVENT_LOOP_THREADS
    completion_threads: int = DEFAULT_MOCK_COMPLETION_THREADS
    mock_auto_fetch: bool = False
    mock_fetch_attach_timeout_ms: int = 600_000
    # Seconds the freshly started master must hold "alive == discovered" for
    # every role before start_master() returns (0 disables). Skips the
    # cold-start first-connect storm during which healthy engines can be
    # 3-strike-marked dead (CONNECT_TIMEOUT 20ms intake defect).
    master_stable_window_s: float = 3.0
    # ------------------------------------------------------------------
    # HA orchestration: an empty registry starts the shared single master;
    # a non-empty registry starts the explicitly declared Master instances.
    # ------------------------------------------------------------------
    # Per-master registry (MasterSpec).  Non-empty => EnvManager starts one
    # flexlb-api JVM per entry (start_master_instance) instead of the
    # single shared master; both masters share the SAME mock cluster and
    # discovery file and poll it independently (brief p1).
    masters: list = field(default_factory=list)  # list[MasterSpec]
    # Elected groups use an owned helper or an explicitly external dependency.
    # None selects standalone scheduling without election or forwarding.
    zk_consistency: Optional[ZooKeeperConfig] = None

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "run_dir": str(self.run_dir) if self.run_dir is not None else None,
                "n_prefill": self.n_prefill,
                "runtime_mode": self.runtime_mode,
                "n_decode": self.n_decode,
                "perf": self.perf,
                "mock_extra_args": self.mock_extra_args,
                "mock_heap": self.mock_heap,
                "master_jvm_args": self.master_jvm_args,
                "master_log_name": self.master_log_name,
                "master_jvm_heap": self.master_jvm_heap,
                "master_pv_log": self.master_pv_log,
                "master_profile": self.master_profile,
                "master_env": self.master_env,
                # config axes: overrides serialized field-by-field (OMIT as
                # its own token) so distinct override specs never collide;
                # raw_config contributes its exact string.
                "config_overrides": _override_fingerprint(self.config_overrides),
                "raw_config": self.raw_config,
                "discovery": self.discovery,
                "domain_addrs": self.domain_addrs,
                "prefill_cache_blocks": self.prefill_cache_blocks,
                "decode_cache_blocks": self.decode_cache_blocks,
                "spring_profile": self.spring_profile,
                "master_stable_window_s": self.master_stable_window_s,
                # HA axes: absent-equivalent ([], None) for every legacy
                # spec — the fingerprint VALUE changes (new keys) but stays
                # stable within a process, so ensure() reuse semantics are
                # unchanged and the legacy path never takes the new branch.
                "masters": [m.fingerprint() for m in self.masters],
                "zk_consistency": self.zk_consistency.to_dict() if self.zk_consistency is not None else None,
            },
            sort_keys=True,
        )
