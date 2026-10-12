"""Strict read-only projection of the Master leadership endpoint."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LeadershipSnapshot:
    enabled: bool
    leader: bool
    local_host: str
    master_host: str | None

    @classmethod
    def read(cls, response):
        if response.get('success') is not True:
            raise ValueError('Master leadership endpoint failed')
        data = json.loads(response['lbStatus'])
        enabled, leader = data['consistency_enabled'], data['master']
        if type(enabled) is not bool or type(leader) is not bool or type(data['local_host']) is not str:
            raise ValueError('invalid leadership snapshot')
        # Java omits null fields while no leader is known.
        address = data.get('master_host')
        if address is not None and type(address) is not str:
            raise ValueError('invalid leader address')
        return cls(enabled, leader, data['local_host'], address)

    def to_dict(self):
        return asdict(self)
