import json
from pathlib import Path

from .errors import ConfigError


class Registry:
    def __init__(self, interfaces, configuration=None):
        self.interfaces = dict(interfaces)
        self.configuration = dict(configuration or {})
        if 'runtime/wait' in self.interfaces:
            raise ConfigError('Java catalog cannot redefine Python runtime/wait')
        self.interfaces['runtime/wait'] = {
            'kind': 'action', 'background': False, 'unit': '',
            'args': {'type': 'object', 'properties': {'duration': {'type': 'duration'}},
                     'required': ['duration'], 'additionalProperties': False},
            'returns': {'type': 'object', 'name': 'None', 'properties': {}, 'required': []},
        }

    @classmethod
    def load(cls, path):
        try:
            value = json.loads(Path(path).read_text(encoding='utf-8'))
            interfaces = value['interfaces']
            if not isinstance(interfaces, dict):
                raise ValueError('interfaces must be a mapping')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ConfigError(f'Cannot load generated Java catalog {path}: {error}') from error
        return cls(interfaces, value.get('configuration'))

    def get(self, name):
        try:
            return self.interfaces[name]
        except (KeyError, TypeError) as error:
            raise ConfigError(f'Unknown interface {name!r}') from error
