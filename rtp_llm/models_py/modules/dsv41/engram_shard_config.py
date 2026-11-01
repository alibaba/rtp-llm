# SPDX-License-Identifier: Apache-2.0
"""Distributed storage P is independent of model instances and TP/DP/EP.

No role election, endpoints, sidecar or implicit single-owner deployment.
Startup exchange (or an explicit offline launcher) delivers one immutable bundle.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

ENV_COUNT = "ENGRAM_SHARD_COUNT"
MAX_SHARDS = 64  # manifest ABI limit, not an NVL rack size


class ShardConfigError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def validate_count(value):
    if type(value) is not int or not 2 <= value <= MAX_SHARDS:
        raise ShardConfigError("STORAGE_HOST_COUNT_MUST_BE_2_TO_64")
    return value


@dataclass(frozen=True)
class ShardConfig:
    shard_count: int

    def __post_init__(self):
        validate_count(self.shard_count)

    @classmethod
    def from_environ(cls, environ=None, *, inferred_count=None):
        env = os.environ if environ is None else environ
        if any(key in env for key in ("ENGRAM_SHARD_INDEX", "ENGRAM_SHARD_MODE", "ENGRAM_SHARD_ENDPOINTS")):
            raise ShardConfigError("REMOVED_SHARD_DEPLOYMENT_OPTIONS")
        raw = env.get(ENV_COUNT)
        if raw is None:
            return cls(inferred_count)
        if type(raw) is not str or not raw.isascii() or not raw.isdecimal():
            raise ShardConfigError("INVALID_STORAGE_HOST_COUNT")
        return cls(int(raw))

    def validate_manifest(self, manifest):
        expected_owners = None
        for tensor in manifest.tensors:
            if len(tensor.shards) != self.shard_count:
                raise ShardConfigError("STORAGE_HOST_COUNT_MISMATCH")
            owners = tuple((s.owner_id, s.owner_incarnation) for s in tensor.shards)
            if len({owner[0] for owner in owners}) != self.shard_count:
                raise ShardConfigError("STORAGE_HOSTS_MUST_DIFFER")
            if expected_owners is None:
                expected_owners = owners
            elif owners != expected_owners:
                raise ShardConfigError("INCONSISTENT_STORAGE_HOSTS")
        if expected_owners is None:
            raise ShardConfigError("EMPTY_STORAGE_GROUP")
        return self
