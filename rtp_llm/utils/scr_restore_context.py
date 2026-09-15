"""Fresh, process-local inputs shared by all fixups after one SCR barrier."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any


@dataclass(frozen=True)
class RestoreContext:
    generation: str
    pod_ip: str

    @cached_property
    def endpoint_manifest(self) -> dict[str, Any] | None:
        # Identity-only participants need not import the distributed stack.
        # A new context is created on every restore, even for the same seed.
        from rtp_llm.utils.scr_endpoint_provider import read_restore_manifest

        return read_restore_manifest(self.generation)

    def resolve_world_info(self, current, parallelism_config):
        """Apply the same endpoint/readiness policy in backend and frontend."""
        from rtp_llm.utils.scr_endpoint_provider import (
            is_restore_phase,
            resolve_world_info,
        )

        manifest = self.endpoint_manifest
        # Single-Pod control channels use loopback; CacheStore and P/D RPC
        # connections are first initialized after this barrier. Only the
        # existing multi-node topology needs the external transport gate.
        require_transport = is_restore_phase(manifest) and current.num_nodes > 1
        restored = resolve_world_info(
            current,
            generation=self.generation,
            require_manifest=require_transport,
            require_transport=require_transport,
            expected_world_size=parallelism_config.world_size,
            manifest=manifest,
        )
        # The template also captures listener addresses and local rendezvous
        # ports. Publishing a different port layout without rebuilding those
        # objects would advertise endpoints on which this worker never listens.
        previous_self = getattr(current, "self", None)
        restored_self = getattr(restored, "self", None)
        if previous_self is not None and restored_self is not None:
            for attr in (
                "server_port",
                "rpc_server_port",
                "cache_store_listen_port",
                "cache_store_rdma_listen_port",
            ):
                if getattr(previous_self, attr) != getattr(restored_self, attr):
                    raise RuntimeError(
                        "SCR restore must preserve the template's local listener port layout"
                    )
        return restored
