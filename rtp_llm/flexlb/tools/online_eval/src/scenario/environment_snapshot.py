"""Frozen render products shared by compiler and process projections."""
from dataclasses import dataclass, asdict
import hashlib
import json


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    master_json: str
    performance_json: str
    runtime_json: str
    sha256: str

    @classmethod
    def freeze(cls, master, performance, runtime):
        values = (_json(master), _json(performance), _json(runtime))
        return cls(*values, hashlib.sha256(_json(values).encode()).hexdigest())

    @classmethod
    def read(cls, data):
        if not isinstance(data, dict) or set(data) != {'master_json', 'performance_json', 'runtime_json', 'sha256'}:
            raise ValueError('invalid environment snapshot')
        if any(type(value) is not str for value in data.values()):
            raise ValueError('environment snapshot fields must be strings')
        result = cls(**data)
        documents = [json.loads(value) for value in (result.master_json, result.performance_json, result.runtime_json)]
        if any(not isinstance(value, dict) for value in documents) or cls.freeze(*documents) != result:
            raise ValueError('environment snapshot integrity mismatch')
        return result

    def to_dict(self):
        return asdict(self)

    @property
    def performance(self):
        return json.loads(self.performance_json)

    @property
    def runtime(self):
        return json.loads(self.runtime_json)
