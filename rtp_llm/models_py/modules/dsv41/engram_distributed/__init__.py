# SPDX-License-Identifier: Apache-2.0
"""Distributed Engram storage for an immutable startup cohort.

config: storage policy, limits and gang/cohort membership.
checkpoint: ShardPublisher owns and loads local backing.
placement: pure shard geometry and NUMA/GPU placement planning.
manifest: bounded metadata parsing and layout validation.
exchange: EngramStartup stages a package for the worker.
resident: independently owned host Fabric backing and startup descriptors.
local: LocalCudaStartup transports same-host GPU file descriptors.
storage: FabricEngramStorage owns backing, CUDA mappings, views and graph leases.
runtime: EngramRuntime owns package handoff and model binding in one worker.

Importing this package performs no discovery, allocation or CUDA work.
"""

from .checkpoint import shard_publisher
from .exchange import EngramStartup
from .runtime import EngramModelAttachment, EngramRuntime, process_runtime

__all__ = [
    "EngramModelAttachment",
    "EngramRuntime",
    "EngramStartup",
    "process_runtime",
    "shard_publisher",
]
