"""Read-only value trees with explicit mutable exports; no ownership of live data."""
from collections.abc import Mapping
from types import MappingProxyType
import copy


class FrozenMapping(Mapping):
    __slots__ = ('_data',)

    def __init__(self, value):
        object.__setattr__(self, '_data', MappingProxyType({key: freeze(item) for key, item in value.items()}))

    def __setattr__(self, name, value):
        raise TypeError('report value is read-only')

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        return thaw(self)


def freeze(value):
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return copy.deepcopy(value)
