"""Explicit ZooKeeper dependency for an isolated elected Master group."""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ZooKeeperConfig:
    mode: str = 'helper'
    address: str | None = None
    session_timeout_ms: int = 10000

    def __post_init__(self):
        if self.mode not in ('helper', 'external'):
            raise ValueError('zookeeper.mode must be helper or external')
        if type(self.session_timeout_ms) is not int or self.session_timeout_ms < 1000:
            raise ValueError('zookeeper.session_timeout_ms must be an integer >= 1000')
        if self.mode == 'external':
            import re
            if type(self.address) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]+:[0-9]+(,[A-Za-z0-9_.-]+:[0-9]+)*(?:/[A-Za-z0-9_/-]+)?', self.address):
                raise ValueError('external zookeeper requires host:port addresses with optional chroot')
            for endpoint in self.address.split('/')[0].split(','):
                if not 1 <= int(endpoint.rsplit(':', 1)[1]) <= 65535:
                    raise ValueError('invalid ZooKeeper port')
        elif self.address is not None:
            raise ValueError('helper zookeeper does not accept an external address')

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict) or set(value) - {'mode', 'address', 'session_timeout_ms'}:
            raise ValueError('invalid zookeeper fields')
        return cls(**value)

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}
